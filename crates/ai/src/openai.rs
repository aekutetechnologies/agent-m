//! OpenAI-compatible chat-completions provider with SSE streaming.
//!
//! This is the generic provider behind DeepSeek (and, later, OpenAI, OpenRouter,
//! Ollama, Groq, ...): it speaks `POST /chat/completions` with `stream: true`
//! and parses `text/event-stream` deltas (content, `reasoning_content` for
//! DeepSeek reasoner-style models, and tool-call fragments).

use async_trait::async_trait;
use futures_util::stream::{self, BoxStream, Stream, StreamExt};
use serde::Deserialize;
use std::collections::{HashMap, VecDeque};

use crate::models::ModelSpec;
use crate::provider::{AiError, Provider};
use crate::types::{AssistantMessage, ChatRequest, ContentPart, StopReason, StreamEvent, Usage};
use serde_json::{Value, json};

/// Map a selected variant to the OpenAI-compatible `reasoning_effort` value
/// and inject it into the request body when the model supports it.
/// `default`/`None` omits the field; `max` maps to `high` (the widest tier
/// the OpenAI wire format has). Byte-stable: inserting into the sorted-key
/// map keeps the JSON deterministic.
fn apply_effort(body: &mut Value, supports_effort: bool, variant: Option<&str>) {
    if !supports_effort {
        return;
    }
    let effort = match variant.unwrap_or("default") {
        "low" => Some("low"),
        "high" => Some("high"),
        "max" => Some("high"),
        _ => None, // default (or unknown) → omit
    };
    match effort {
        Some(effort) => body["reasoning_effort"] = json!(effort),
        None => {
            if let Some(map) = body.as_object_mut() {
                map.remove("reasoning_effort");
            }
        }
    };
}
use crate::wire::build_chat_request_body;

/// A provider speaking the OpenAI chat-completions protocol.
#[derive(Debug)]
pub struct OpenAiCompatibleProvider {
    id: String,
    display_name: String,
    /// Base URL without the trailing `/chat/completions` path, e.g.
    /// `https://api.deepseek.com`.
    base_url: String,
    api_key: Option<String>,
    models: Vec<ModelSpec>,
    http: reqwest::Client,
}

impl OpenAiCompatibleProvider {
    pub fn new(
        id: impl Into<String>,
        display_name: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<String>,
        models: Vec<ModelSpec>,
    ) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            base_url: base_url.into(),
            api_key,
            models,
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("reqwest client build cannot fail"),
        }
    }

    /// A DeepSeek provider: OpenAI-compatible chat completions at
    /// `https://api.deepseek.com` with `deepseek-chat` (default) and
    /// `deepseek-reasoner` (emits reasoning deltas).
    pub fn deepseek(api_key: Option<String>) -> Self {
        Self::new(
            "deepseek",
            "DeepSeek",
            "https://api.deepseek.com",
            api_key,
            vec![
                ModelSpec::new("deepseek-chat")
                    .name("DeepSeek Chat")
                    // Current platform (api-docs.deepseek.com, V4 era):
                    // 1M context / 384K max output. The window is a planning
                    // budget for the context gauge + compaction threshold,
                    // not a hard cap.
                    .context_window(1_000_000)
                    // Official DeepSeek pricing, USD per 1M tokens:
                    // miss $0.27, cache-hit $0.07, output $1.10.
                    .pricing(0.27, 0.07, 1.10),
                ModelSpec::new("deepseek-reasoner")
                    .name("DeepSeek Reasoner")
                    .reasoning(true)
                    // Same 1M context as deepseek-chat (V4 era docs).
                    .context_window(1_000_000)
                    // miss $0.55, cache-hit $0.14, output $2.19.
                    .pricing(0.55, 0.14, 2.19),
                // Current platform models (api-docs.deepseek.com, V4 era):
                // thinking mode on by default, 1M context / 384K max output.
                // Pricing is an estimate converted from the CNY list
                // (v4-flash ¥1/0.02/2, v4-pro ¥3/0.025/6 per 1M tokens).
                ModelSpec::new("deepseek-v4-flash")
                    .name("DeepSeek V4 Flash")
                    .reasoning(true)
                    .context_window(1_000_000)
                    .pricing(0.14, 0.003, 0.28),
                ModelSpec::new("deepseek-v4-pro")
                    .name("DeepSeek V4 Pro")
                    .reasoning(true)
                    .context_window(1_000_000)
                    .pricing(0.42, 0.004, 0.84),
            ],
        )
    }
}

#[async_trait]
impl Provider for OpenAiCompatibleProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn display_name(&self) -> &str {
        &self.display_name
    }

    fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    fn set_api_key(&mut self, key: String) {
        self.api_key = Some(key);
    }

    fn models(&self) -> &[ModelSpec] {
        &self.models
    }

    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, StreamEvent>, AiError> {
        let key = self
            .api_key
            .as_ref()
            .ok_or_else(|| AiError::MissingApiKey {
                provider: self.id.clone(),
                env_var: format!("{}_API_KEY", self.id.to_uppercase()),
            })?;
        // Vision capability gate: image attachments require a model whose
        // spec advertises supports_images.
        let supports_images = self
            .models
            .iter()
            .find(|spec| spec.id == request.model)
            .map(|spec| spec.supports_images)
            .unwrap_or(false);
        if !supports_images
            && request.messages.iter().any(|message| match message {
                crate::types::LlmMessage::User { images, .. } => !images.is_empty(),
                _ => false,
            })
        {
            return Err(AiError::Api(format!(
                "model `{}` does not support image input; use a vision-capable model",
                request.model
            )));
        }

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let mut body = build_chat_request_body(&request);
        // Reasoning-effort variant (OpenCode-style Default/low/high/max) →
        // OpenAI-compatible `reasoning_effort`, only for models that declare
        // support. `default`/None omits the field; `max` maps to `high`
        // (no wider effort tier exists on the OpenAI wire format).
        let supports_effort = self
            .models
            .iter()
            .any(|spec| spec.id == request.model && spec.supports_effort);
        apply_effort(&mut body, supports_effort, request.variant.as_deref());

        let response = self
            .http
            .post(&url)
            .bearer_auth(key)
            .header("Accept", "text/event-stream")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            return Err(AiError::Api(format!(
                "HTTP {status} from {}: {}",
                self.display_name,
                truncate(&error_body, 500)
            )));
        }

        Ok(Box::pin(sse_events(response.bytes_stream(), request.model)))
    }
}

fn truncate(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let head: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{head}…")
    } else {
        head
    }
}

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

/// One `chat.completion.chunk` from the stream.
#[derive(Debug, Deserialize)]
struct WireChunk {
    #[serde(default)]
    choices: Vec<WireChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct WireChoice {
    #[serde(default)]
    delta: WireDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct WireDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default, rename = "reasoning_content")]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<WireToolCallDelta>,
}

#[derive(Debug, Deserialize)]
struct WireToolCallDelta {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<WireFunctionDelta>,
}

#[derive(Debug, Default, Deserialize)]
struct WireFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// OpenAI nests cache info under `prompt_tokens_details`; DeepSeek uses top-level fields.
#[derive(Debug, Default, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Debug, Default, Deserialize)]
struct WireUsage {
    #[serde(default, rename = "prompt_tokens")]
    prompt_tokens: u64,
    #[serde(default, rename = "completion_tokens")]
    completion_tokens: u64,
    /// DeepSeek field name.
    #[serde(default, rename = "prompt_cache_hit_tokens")]
    prompt_cache_hit_tokens: u64,
    /// DeepSeek field name.
    #[serde(default, rename = "prompt_cache_miss_tokens")]
    prompt_cache_miss_tokens: u64,
    #[serde(default, rename = "total_tokens")]
    total_tokens: u64,
    /// OpenAI nests cached token counts here.
    #[serde(default)]
    prompt_tokens_details: PromptTokensDetails,
}

impl WireUsage {
    fn into_usage(self) -> Usage {
        // DeepSeek: top-level prompt_cache_hit_tokens.
        // OpenAI:   prompt_tokens_details.cached_tokens.
        // Take whichever is non-zero; both zero means provider doesn't report caching.
        let cache_read = self
            .prompt_cache_hit_tokens
            .max(self.prompt_tokens_details.cached_tokens);
        Usage {
            input_tokens: self.prompt_tokens,
            output_tokens: self.completion_tokens,
            cache_read_tokens: cache_read,
            cache_creation_tokens: self.prompt_cache_miss_tokens,
            total_tokens: self.total_tokens,
            cost: 0.0,
        }
    }
}

/// Accumulated partial state for one tool call being streamed.
#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

/// Accumulates deltas across chunks until the final `Done`.
#[derive(Debug, Default)]
struct DeltaState {
    started: bool,
    text: String,
    thinking: String,
    tool_calls: HashMap<usize, ToolCallAccumulator>,
    stop_reason: Option<StopReason>,
    usage: Option<Usage>,
    done_emitted: bool,
}

/// Stream parser: consumes raw SSE bytes and yields `StreamEvent`s.
struct SseParser<S> {
    byte_stream: S,
    buffer: Vec<u8>,
    model: String,
    state: DeltaState,
    queue: VecDeque<StreamEvent>,
    finished: bool,
}

fn sse_events<S>(byte_stream: S, model: String) -> impl Stream<Item = StreamEvent> + Send
where
    S: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + Unpin + 'static,
{
    stream::unfold(
        SseParser {
            byte_stream,
            buffer: Vec::new(),
            model,
            state: DeltaState::default(),
            queue: VecDeque::new(),
            finished: false,
        },
        |mut parser| async move {
            loop {
                if let Some(event) = parser.queue.pop_front() {
                    return Some((event, parser));
                }
                if parser.finished {
                    return None;
                }
                parser.drain_buffer();
                if parser.queue.is_empty() && !parser.finished {
                    match parser.byte_stream.next().await {
                        Some(Ok(bytes)) => parser.buffer.extend_from_slice(&bytes),
                        Some(Err(error)) => {
                            parser.queue.push_back(StreamEvent::Error {
                                message: error.to_string(),
                            });
                            parser.finished = true;
                        }
                        None => {
                            // Stream ended: drain any trailing unterminated
                            // line (e.g. a final `data:` chunk without `\n`),
                            // then emit the terminal event.
                            if !parser.buffer.is_empty() {
                                let tail = std::mem::take(&mut parser.buffer);
                                parser.handle_line(&tail);
                            }
                            if !parser.state.done_emitted {
                                parser.push_done();
                            }
                            parser.finished = true;
                        }
                    }
                }
            }
        },
    )
}

impl<S> SseParser<S> {
    /// Process every complete line currently in the buffer.
    fn drain_buffer(&mut self) {
        loop {
            let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') else {
                return;
            };
            let line: Vec<u8> = self.buffer.drain(..=newline).collect();
            self.handle_line(&line[..line.len().saturating_sub(1)]);
        }
    }

    /// Process one line (without its trailing newline).
    fn handle_line(&mut self, bytes: &[u8]) {
        let line = String::from_utf8_lossy(bytes);
        let line = line.trim();
        if line.is_empty() {
            return;
        }
        let Some(data) = line.strip_prefix("data:") else {
            return;
        };
        let data = data.trim();
        if data == "[DONE]" {
            if !self.state.done_emitted {
                self.push_done();
            }
            self.finished = true;
            return;
        }
        self.handle_chunk(data);
    }

    fn handle_chunk(&mut self, data: &str) {
        let chunk: WireChunk = match serde_json::from_str(data) {
            Ok(chunk) => chunk,
            Err(error) => {
                self.queue.push_back(StreamEvent::Error {
                    message: format!("malformed chunk: {error}: {data}"),
                });
                return;
            }
        };

        if let Some(usage) = chunk.usage {
            self.state.usage = Some(usage.into_usage());
        }

        if !self.state.started {
            self.state.started = true;
            self.queue.push_back(StreamEvent::Start);
        }

        for choice in chunk.choices {
            let delta = choice.delta;
            if let Some(content) = delta.content
                && !content.is_empty()
            {
                self.state.text.push_str(&content);
                self.queue
                    .push_back(StreamEvent::TextDelta { delta: content });
            }
            if let Some(reasoning) = delta.reasoning_content
                && !reasoning.is_empty()
            {
                self.state.thinking.push_str(&reasoning);
                self.queue
                    .push_back(StreamEvent::ThinkingDelta { delta: reasoning });
            }
            for tool_delta in delta.tool_calls {
                let index = tool_delta.index.unwrap_or(0);
                let accumulator = self.state.tool_calls.entry(index).or_default();
                if let Some(id) = tool_delta.id {
                    accumulator.id = id;
                }
                if let Some(name) = tool_delta.function.as_ref().and_then(|f| f.name.clone()) {
                    accumulator.name.push_str(&name);
                }
                if let Some(arguments) = tool_delta
                    .function
                    .as_ref()
                    .and_then(|f| f.arguments.clone())
                {
                    accumulator.arguments.push_str(&arguments);
                }
                self.queue.push_back(StreamEvent::ToolCallDelta {
                    index,
                    id: accumulator.id.clone(),
                    name: accumulator.name.clone(),
                    arguments: accumulator.arguments.clone(),
                });
            }
            if let Some(reason) = choice.finish_reason {
                self.state.stop_reason = Some(stop_reason_from_wire(
                    &reason,
                    !self.state.tool_calls.is_empty(),
                ));
            }
        }
    }

    fn push_done(&mut self) {
        // Best-effort trust metadata from the reply's <trust> block (the
        // agent strips the block from the stored text before display).
        let (trust, _) = crate::trust::extract_trust_block(&self.state.text);
        let has_tool_calls = !self.state.tool_calls.is_empty();

        let mut content: Vec<ContentPart> = Vec::new();
        if !self.state.text.is_empty() {
            content.push(ContentPart::Text {
                text: std::mem::take(&mut self.state.text),
            });
        }
        if !self.state.thinking.is_empty() {
            content.push(ContentPart::Thinking {
                thinking: std::mem::take(&mut self.state.thinking),
            });
        }
        let mut tool_calls: Vec<(usize, ToolCallAccumulator)> =
            self.state.tool_calls.drain().collect();
        tool_calls.sort_by_key(|(index, _)| *index);
        for (_, accumulator) in tool_calls {
            let arguments = serde_json::from_str(&accumulator.arguments)
                .unwrap_or_else(|_| serde_json::Value::String(accumulator.arguments.clone()));
            content.push(ContentPart::ToolCall {
                id: accumulator.id,
                name: accumulator.name,
                arguments,
            });
        }

        let stop_reason = self.state.stop_reason.unwrap_or(if has_tool_calls {
            StopReason::ToolUse
        } else {
            StopReason::Stop
        });
        // Best-effort trust metadata from the reply's <trust> block (the
        // agent strips the block from the stored text before display).
        let message = AssistantMessage {
            content,
            usage: self.state.usage.take(),
            stop_reason,
            error_message: None,
            model: self.model.clone(),
            trust,
        };
        self.queue.push_back(StreamEvent::Done { message });
        self.state.done_emitted = true;
    }
}

fn stop_reason_from_wire(reason: &str, has_tool_calls: bool) -> StopReason {
    match reason {
        "stop" => StopReason::Stop,
        "length" => StopReason::Length,
        "tool_calls" | "function_call" => StopReason::ToolUse,
        _ => {
            if has_tool_calls {
                StopReason::ToolUse
            } else {
                StopReason::Stop
            }
        }
    }
}

#[cfg(test)]
mod effort_tests {
    use super::*;

    #[test]
    fn effort_maps_variants_to_reasoning_effort() {
        let mut body = serde_json::json!({ "model": "gpt-5" });
        apply_effort(&mut body, true, Some("low"));
        assert_eq!(body["reasoning_effort"], "low");
        apply_effort(&mut body, true, Some("high"));
        assert_eq!(body["reasoning_effort"], "high");
        // max has no wider tier on the OpenAI wire format → high.
        apply_effort(&mut body, true, Some("max"));
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[test]
    fn effort_default_and_unsupported_omit_the_field() {
        let mut body = serde_json::json!({ "model": "gpt-5" });
        apply_effort(&mut body, true, None);
        assert!(body.get("reasoning_effort").is_none());
        apply_effort(&mut body, true, Some("default"));
        assert!(body.get("reasoning_effort").is_none());
        // A previously-set field is removed when the variant becomes default.
        apply_effort(&mut body, true, Some("high"));
        apply_effort(&mut body, true, Some("default"));
        assert!(body.get("reasoning_effort").is_none());
        // Providers without effort support never get the field.
        let mut body = serde_json::json!({ "model": "deepseek-reasoner" });
        apply_effort(&mut body, false, Some("high"));
        assert!(body.get("reasoning_effort").is_none());
    }
}
