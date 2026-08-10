//! agent-m-ai: model-agnostic LLM provider layer with byte-stable prefix caching.
//!
//! The layer is intentionally provider-agnostic. Providers implement the
//! [`Provider`] trait and return a stream of [`StreamEvent`]s; the first
//! concrete provider is an OpenAI-compatible client configured for DeepSeek.
//!
//! Byte-stable prefix caching: request bodies are serialized deterministically
//! (sorted JSON object keys, no volatile fields), so the conversation prefix —
//! system prompt, tool schemas, and past messages — is byte-identical across
//! turns. That lets providers serve the prefix from their context cache
//! instead of recomputing it, which is what [`CacheStats`] measures.

mod cache;
mod keys;
mod models;
mod openai;
mod provider;
mod types;
mod wire;

pub use cache::CacheStats;
pub use keys::resolve_api_key;
pub use models::ModelSpec;
pub use openai::OpenAiCompatibleProvider;
pub use provider::{AiError, Provider, ProviderRegistry};
pub use types::{
    AssistantMessage, ChatRequest, ContentPart, LlmMessage, StopReason, StreamEvent, ToolSpec,
    Usage,
};
pub use wire::{build_chat_request_body, serialize_messages};
