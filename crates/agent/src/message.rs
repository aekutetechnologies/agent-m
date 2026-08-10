//! Session-level messages and their conversion to the wire request format.

use agent_m_ai::{ContentPart, LlmMessage, StopReason, Usage};

/// A message in the session context, mirroring pi's `AgentMessage` set.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionMessage {
    User {
        content: String,
    },
    Assistant {
        content: Vec<ContentPart>,
        usage: Option<Usage>,
        stop_reason: StopReason,
        model: String,
        /// Trust metadata parsed from the reply's `<trust>` block.
        trust: agent_m_ai::TrustData,
    },
    ToolResult {
        tool_call_id: String,
        name: String,
        content: String,
        is_error: bool,
    },
    /// A compacted-history summary (memory across turns/sessions).
    Summary {
        text: String,
    },
}

/// Discriminator used by the UI and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionMessageKind {
    User,
    Assistant,
    ToolResult,
    Summary,
}

impl SessionMessage {
    pub fn kind(&self) -> SessionMessageKind {
        match self {
            SessionMessage::User { .. } => SessionMessageKind::User,
            SessionMessage::Assistant { .. } => SessionMessageKind::Assistant,
            SessionMessage::ToolResult { .. } => SessionMessageKind::ToolResult,
            SessionMessage::Summary { .. } => SessionMessageKind::Summary,
        }
    }

    /// Convert to the wire-format message. The system prompt is passed
    /// separately by the agent loop.
    pub fn to_llm_message(&self) -> LlmMessage {
        match self {
            SessionMessage::User { content } => LlmMessage::User {
                content: content.clone(),
            },
            SessionMessage::Assistant {
                content,
                usage,
                stop_reason,
                ..
            } => LlmMessage::Assistant {
                content: content.clone(),
                usage: usage.clone(),
                stop_reason: Some(*stop_reason),
            },
            SessionMessage::ToolResult {
                tool_call_id,
                name,
                content,
                ..
            } => LlmMessage::Tool {
                tool_call_id: tool_call_id.clone(),
                name: name.clone(),
                content: content.clone(),
            },
            // A summary is fed back as a user message so the model sees it as
            // conversation history, not instructions.
            SessionMessage::Summary { text } => LlmMessage::User {
                content: format!("[Session summary]\n{text}"),
            },
        }
    }
}
