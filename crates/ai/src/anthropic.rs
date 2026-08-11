//! Native Anthropic Messages API provider with prefix caching and extended
//! thinking.
//!
//! Unlike the OpenAI-compatible provider, this speaks the Anthropic wire
//! format directly:
//! - `system` is a top-level field (not a message),
//! - tool definitions use `input_schema` (not `parameters`),
//! - `cache_control: {type: "ephemeral"}` markers enable prefix caching,
//! - SSE events are `content_block_start` / `content_block_delta` /
//!   `content_block_stop` / `message_delta`,
//! - thinking blocks arrive as `type: "thinking"` content blocks.

use async_trait::async_trait;
use futures_util::stream::{self, BoxStream, Stream, StreamExt};
use serde::Deserialize;
use std::collections::VecDeque;

use crate::models::ModelSpec;
use crate::provider::{AiError, Provider};
use crate::types::{
    AssistantMessage, ChatRequest, ContentPart, LlmMessage, StopReason, StreamEvent, ToolSpec,
    Usage,
};
use serde_json::{Value, json};

/// How many trailing user messages get a `cache_control` marker (Anthropic's
/// prefix-caching mechanism). The system prompt is always marked.
const CACHE_LAST_N_USER_MESSAGES: usize = 3;

/// A provider speaking the Anthropic Messages API.
#[derive(Debug)]
pub struct AnthropicProvider {
    id: String,
    display_name: String,
    /// Base URL, e.g. `https://api.anthropic.com/v1`.
    base_url: String,
    api_key: Option<String>,
    models: Vec<ModelSpec>,
    http: reqwest::Client,
}

impl AnthropicProvider {
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
}

#[async_trait]
impl Provider for AnthropicProvider {
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

        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));
        let body = build_request_body(&request);

        let response = self
            .http
            .post(&url)
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
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

/// Build the Anthropic `/messages` request body.
///
/// - `system` is top-level, marked with `cache_control` for prefix caching.
/// - Tool definitions use `input_schema`.
/// - The last few user messages get `cache_control` markers so the rolling
///   conversation prefix is cacheable.
fn build_request_body(request: &ChatRequest) -> Value {
    let mut body = json!({
        "model": request.model,
        "max_tokens": 8192,
        "stream": true,
    });
    if !request.system.is_empty() {
        body["system"] = json!([{
            "type": "text",
            "text": request.system,
            "cache_control": { "type": "ephemeral" },
        }]);
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(request.tools.iter().map(serialize_tool).collect());
    }
    body["messages"] = serialize_messages(&request.messages);
    body
}

/// Serialize a tool spec into the Anthropic `tools` entry (`input_schema`).
fn serialize_tool(tool: &ToolSpec) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    })
}

/// Serialize messages into the Anthropic `messages` array. The last few user
/// messages get `cache_control` markers.
fn serialize_messages(messages: &[LlmMessage]) -> Value {
    // Find the indices of the last N user messages to mark for caching.
    let user_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(m, LlmMessage::User { .. }))
        .map(|(i, _)| i)
        .collect();
    let cache_mark: Vec<usize> = user_indices
        .iter()
        .rev()
        .take(CACHE_LAST_N_USER_MESSAGES)
        .copied()
        .collect();

    Value::Array(
        messages
            .iter()
            .enumerate()
            .map(|(index, message)| wire_message(message, cache_mark.contains(&index)))
            .collect(),
    )
}

fn wire_message(message: &LlmMessage, cache_mark: bool) -> Value {
    match message {
        LlmMessage::System { content } => {
            // Anthropic has no system message in `messages`; the caller puts
            // it in the top-level `system` field. Defensive: emit as user.
            json!({ "role": "user", "content": content })
        }
        LlmMessage::User { content, images } => {
            let mut parts: Vec<Value> = Vec::new();
            if !content.is_empty() {
                parts.push(json!({ "type": "text", "text": content }));
            }
            for data in images {
                // Anthropic image blocks take base64 + media_type; the data
                // URI is parsed best-effort.
                if let Some((media_type, base64)) = parse_data_uri(data) {
                    parts.push(json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64,
                        }
                    }));
                }
            }
            let mut message = json!({ "role": "user", "content": parts });
            if cache_mark {
                // Mark the last *text* block for caching — Anthropic forbids
                // cache_control breakpoints on image blocks (HTTP 400), so
                // walk back to the last text block when images are present.
                if let Some(parts) = message["content"].as_array_mut() {
                    let mark_index = parts
                        .iter()
                        .rposition(|part| part.get("type") != Some(&json!("image")));
                    if let Some(index) = mark_index {
                        parts[index]["cache_control"] = json!({ "type": "ephemeral" });
                    }
                }
            }
            message
        }
        LlmMessage::Assistant { content, .. } => {
            let mut blocks: Vec<Value> = Vec::new();
            for part in content {
                match part {
                    ContentPart::Text { text } => {
                        blocks.push(json!({ "type": "text", "text": text }));
                    }
                    ContentPart::Thinking { thinking } => {
                        blocks.push(json!({
                            "type": "thinking",
                            "thinking": thinking,
                            "signature": "",
                        }));
                    }
                    ContentPart::Image { .. } => {}
                    ContentPart::ToolCall {
                        id,
                        name,
                        arguments,
                    } => {
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": arguments,
                        }));
                    }
                }
            }
            json!({ "role": "assistant", "content": blocks })
        }
        LlmMessage::Tool {
            tool_call_id,
            name: _,
            content,
        } => {
            json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }],
            })
        }
    }
}

/// Parse a `data:<media_type>;base64,<data>` URI into `(media_type, base64)`.
fn parse_data_uri(uri: &str) -> Option<(String, String)> {
    let rest = uri.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let media_type = meta.split(';').next()?.to_string();
    Some((media_type, data.to_string()))
}

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

/// One SSE event from the Anthropic stream.
#[derive(Debug, Deserialize)]
struct WireEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    content_block: Option<WireContentBlock>,
    #[serde(default)]
    delta: Option<WireDelta>,
    #[serde(default)]
    message: Option<WireMessage>,
    #[serde(default)]
    error: Option<WireError>,
}

#[derive(Debug, Deserialize)]
struct WireContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct WireDelta {
    #[serde(rename = "type")]
    delta_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WireMessage {
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Default, Deserialize)]
struct WireUsage {
    #[serde(default, rename = "input_tokens")]
    input_tokens: u64,
    #[serde(default, rename = "output_tokens")]
    output_tokens: u64,
    #[serde(default, rename = "cache_read_input_tokens")]
    cache_read_input_tokens: u64,
    #[serde(default, rename = "cache_creation_input_tokens")]
    cache_creation_input_tokens: u64,
}

impl WireUsage {
    fn into_usage(self) -> Usage {
        Usage {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            cache_read_tokens: self.cache_read_input_tokens,
            cache_creation_tokens: self.cache_creation_input_tokens,
            total_tokens: self.input_tokens + self.output_tokens,
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

/// Accumulates deltas across events until the final `Done`.
#[derive(Debug, Default)]
struct DeltaState {
    started: bool,
    text: String,
    thinking: String,
    tool_calls: Vec<ToolCallAccumulator>,
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
        self.handle_event(data);
    }

    fn handle_event(&mut self, data: &str) {
        let event: WireEvent = match serde_json::from_str(data) {
            Ok(event) => event,
            Err(error) => {
                self.queue.push_back(StreamEvent::Error {
                    message: format!("malformed event: {error}: {data}"),
                });
                return;
            }
        };

        if !self.state.started {
            self.state.started = true;
            self.queue.push_back(StreamEvent::Start);
        }

        match event.event_type.as_str() {
            "message_start" => {
                if let Some(message) = event.message
                    && let Some(usage) = message.usage
                {
                    self.state.usage = Some(usage.into_usage());
                }
            }
            "content_block_start" => {
                if let Some(block) = event.content_block
                    && block.block_type == "tool_use"
                {
                    let index = event.index.unwrap_or(self.state.tool_calls.len());
                    while self.state.tool_calls.len() <= index {
                        self.state.tool_calls.push(ToolCallAccumulator::default());
                    }
                    let acc = &mut self.state.tool_calls[index];
                    if let Some(id) = block.id {
                        acc.id = id;
                    }
                    if let Some(name) = block.name {
                        acc.name = name;
                    }
                    if let Some(input) = block.input {
                        acc.arguments = serde_json::to_string(&input).unwrap_or_default();
                    }
                    self.queue.push_back(StreamEvent::ToolCallDelta {
                        index,
                        id: acc.id.clone(),
                        name: acc.name.clone(),
                        arguments: acc.arguments.clone(),
                    });
                }
            }
            "content_block_delta" => {
                if let Some(delta) = event.delta {
                    match delta.delta_type.as_str() {
                        "text_delta" => {
                            if let Some(text) = delta.text {
                                self.state.text.push_str(&text);
                                self.queue.push_back(StreamEvent::TextDelta { delta: text });
                            }
                        }
                        "thinking_delta" => {
                            if let Some(thinking) = delta.thinking {
                                self.state.thinking.push_str(&thinking);
                                self.queue
                                    .push_back(StreamEvent::ThinkingDelta { delta: thinking });
                            }
                        }
                        "input_json_delta" => {
                            if let Some(partial) = delta.partial_json {
                                let index = event.index.unwrap_or(0);
                                while self.state.tool_calls.len() <= index {
                                    self.state.tool_calls.push(ToolCallAccumulator::default());
                                }
                                let acc = &mut self.state.tool_calls[index];
                                acc.arguments.push_str(&partial);
                                self.queue.push_back(StreamEvent::ToolCallDelta {
                                    index,
                                    id: acc.id.clone(),
                                    name: acc.name.clone(),
                                    arguments: acc.arguments.clone(),
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }
            "message_delta" => {
                if let Some(delta) = event.delta
                    && let Some(reason) = &delta.stop_reason
                {
                    self.state.stop_reason = Some(stop_reason_from_wire(
                        reason,
                        !self.state.tool_calls.is_empty(),
                    ));
                }
                // Anthropic sends input/cache tokens in `message_start` and
                // only `output_tokens` in `message_delta` — merge, never
                // overwrite, or the final usage loses the input side
                // (review: usage accounting bug).
                if let Some(message) = event.message
                    && let Some(usage) = message.usage
                {
                    let delta_usage = usage.into_usage();
                    match &mut self.state.usage {
                        Some(existing) => {
                            existing.output_tokens = delta_usage.output_tokens;
                            existing.total_tokens = existing.input_tokens
                                + existing.cache_read_tokens
                                + existing.cache_creation_tokens
                                + existing.output_tokens;
                        }
                        None => self.state.usage = Some(delta_usage),
                    }
                }
            }
            "message_stop" => {
                if !self.state.done_emitted {
                    self.push_done();
                }
                self.finished = true;
            }
            "error" => {
                let message = event
                    .error
                    .map(|error| error.message)
                    .unwrap_or_else(|| "unknown Anthropic error".to_string());
                self.queue.push_back(StreamEvent::Error { message });
                self.finished = true;
            }
            _ => {}
        }
    }

    fn push_done(&mut self) {
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
        for accumulator in std::mem::take(&mut self.state.tool_calls) {
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

#[derive(Debug, Deserialize)]
struct WireError {
    #[serde(default)]
    message: String,
}

fn stop_reason_from_wire(reason: &str, has_tool_calls: bool) -> StopReason {
    match reason {
        "end_turn" => StopReason::Stop,
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        "stop_sequence" => StopReason::Stop,
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
mod tests {
    use super::*;

    #[test]
    fn builds_request_with_system_and_tools() {
        let request = ChatRequest {
            model: "claude-sonnet-4-5".into(),
            system: "be helpful".into(),
            messages: vec![LlmMessage::User {
                content: "hi".into(),
                images: vec![],
            }],
            tools: vec![ToolSpec {
                name: "bash".into(),
                description: "run a command".into(),
                parameters: json!({ "type": "object" }),
            }],
            temperature: None,
            variant: None,
        };
        let body = build_request_body(&request);
        assert_eq!(body["model"], "claude-sonnet-4-5");
        assert_eq!(body["system"][0]["text"], "be helpful");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["tools"][0]["name"], "bash");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(body["messages"][0]["role"], "user");
    }

    #[test]
    fn marks_last_user_messages_for_cache() {
        let mut messages = Vec::new();
        for i in 0..5 {
            messages.push(LlmMessage::User {
                content: format!("msg {i}"),
                images: vec![],
            });
        }
        let body = serialize_messages(&messages);
        let arr = body.as_array().unwrap();
        // Only the last CACHE_LAST_N_USER_MESSAGES get a cache_control marker.
        let marked: Vec<usize> = arr
            .iter()
            .enumerate()
            .filter(|(_, m)| m["content"][0].get("cache_control").is_some())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(marked, vec![2, 3, 4], "last 3 user messages marked");
    }

    #[test]
    fn parses_data_uri() {
        let (media, data) = parse_data_uri("data:image/png;base64,AAAA").expect("parses");
        assert_eq!(media, "image/png");
        assert_eq!(data, "AAAA");
        assert!(parse_data_uri("not-a-uri").is_none());
    }

    #[test]
    fn maps_stop_reasons() {
        assert_eq!(stop_reason_from_wire("end_turn", false), StopReason::Stop);
        assert_eq!(
            stop_reason_from_wire("max_tokens", false),
            StopReason::Length
        );
        assert_eq!(
            stop_reason_from_wire("tool_use", false),
            StopReason::ToolUse
        );
    }
}
