//! Core message and request types shared by all providers.
//!
//! These mirror the shapes pi uses (`packages/ai/src/types.ts`): roles
//! user/assistant/tool, content parts (text/thinking/toolCall), usage with
//! cache read/write token counts, and stop reasons.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Why the assistant message finished. Mirrors pi's `StopReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StopReason {
    Pending,
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
}

/// Token usage for one assistant message, including cache read/creation counts.
///
/// Field names follow the OpenAI/DeepSeek naming (`prompt_cache_hit_tokens` /
/// `prompt_cache_miss_tokens`); Anthropic's `cache_read_input_tokens` /
/// `cache_creation_input_tokens` map onto the same fields.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default, rename = "inputTokens")]
    pub input_tokens: u64,
    #[serde(default, rename = "outputTokens")]
    pub output_tokens: u64,
    #[serde(default, rename = "cacheReadTokens")]
    pub cache_read_tokens: u64,
    #[serde(default, rename = "cacheCreationTokens")]
    pub cache_creation_tokens: u64,
    #[serde(default, rename = "totalTokens")]
    pub total_tokens: u64,
    #[serde(default)]
    pub cost: f64,
}

/// One content part of a message, mirroring pi's content-part union.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ContentPart {
    /// Assistant-visible text.
    Text { text: String },
    /// Reasoning text (DeepSeek `reasoning_content`, Anthropic thinking blocks).
    Thinking { thinking: String },
    /// A tool invocation the assistant wants executed.
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
}

/// A conversation message as the agent core stores and exchanges it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum LlmMessage {
    /// System prompt. Exactly one, always first.
    System { content: String },
    /// User turn. Plain text for now (image parts are a later extension).
    User { content: String },
    /// Assistant turn: content parts plus the final usage/stop reason.
    Assistant {
        content: Vec<ContentPart>,
        usage: Option<Usage>,
        stop_reason: Option<StopReason>,
    },
    /// Result of a tool call, tied to the call by id.
    Tool {
        tool_call_id: String,
        name: String,
        content: String,
    },
}

/// A tool the model may call, described with a JSON Schema for its arguments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    #[serde(default = "default_parameters")]
    pub parameters: Value,
}

fn default_parameters() -> Value {
    Value::Object(Default::default())
}

/// A complete request to a provider: model, system prompt, messages, tools.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub system: String,
    pub messages: Vec<LlmMessage>,
    pub tools: Vec<ToolSpec>,
    pub temperature: Option<f64>,
}

/// The finished assistant message produced by a provider stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantMessage {
    pub content: Vec<ContentPart>,
    pub usage: Option<Usage>,
    pub stop_reason: StopReason,
    pub error_message: Option<String>,
    pub model: String,
}

/// Events streamed while an assistant reply is generated.
///
/// Ordering mirrors pi's `AssistantMessageEventStream`: `Start`, then interleaved
/// text/thinking/tool-call deltas, then exactly one terminal `Done` or `Error`.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Start,
    /// A delta of visible text.
    TextDelta {
        delta: String,
    },
    /// A delta of reasoning text.
    ThinkingDelta {
        delta: String,
    },
    /// A delta of one tool call, carrying the accumulated partial state so the
    /// UI can render it progressively.
    ToolCallDelta {
        index: usize,
        id: String,
        name: String,
        arguments: String,
    },
    /// The reply is complete.
    Done {
        message: AssistantMessage,
    },
    /// The reply failed; the error is encoded as an event, never thrown.
    Error {
        message: String,
    },
}
