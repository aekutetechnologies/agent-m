//! The provider abstraction and registry.
//!
//! The shape follows warp's `crates/ai` (enum-per-provider with per-provider
//! behavior) and pi's `StreamFn` contract: a provider never throws — failures
//! come back as an error event on the stream.

use async_trait::async_trait;
use futures_util::stream::BoxStream;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::models::ModelSpec;
use crate::types::{ChatRequest, StreamEvent};

/// Errors surfaced by the AI layer. HTTP/transport errors, malformed SSE, and
/// provider-reported failures all land here.
#[derive(Debug, thiserror::Error)]
pub enum AiError {
    #[error("provider `{0}` is not registered")]
    ProviderNotFound(String),

    #[error("no API key for provider `{provider}`; set {env_var} or add it to auth.json")]
    MissingApiKey { provider: String, env_var: String },

    #[error("HTTP request to provider failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("provider returned an error: {0}")]
    Api(String),

    #[error("malformed streaming response: {0}")]
    Sse(String),
}

/// A model-agnostic chat provider.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Stable provider id, e.g. `"deepseek"`.
    fn id(&self) -> &str;

    /// Human-readable provider name.
    fn display_name(&self) -> &str;

    /// The configured API key, if any.
    fn api_key(&self) -> Option<&str>;

    /// Replace the API key (used by `--api-key` and `/login`).
    fn set_api_key(&mut self, key: String);

    /// Models this provider serves.
    fn models(&self) -> &[ModelSpec];

    /// Stream an assistant reply. The returned stream emits `StreamEvent`s and
    /// ends with `Done` or `Error` — transport and parse failures are folded
    /// into `StreamEvent::Error`, never thrown.
    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, StreamEvent>, AiError>;
}

/// A registry of providers keyed by id, plus the shared HTTP client.
#[derive(Default)]
pub struct ProviderRegistry {
    providers: BTreeMap<Arc<str>, Box<dyn Provider>>,
}

impl ProviderRegistry {
    /// Register a provider under its `id()`.
    pub fn register(&mut self, provider: Box<dyn Provider>) {
        let id: Arc<str> = provider.id().into();
        self.providers.insert(id, provider);
    }

    /// Look up a provider by id.
    pub fn get(&self, id: &str) -> Option<&dyn Provider> {
        self.providers.get(id).map(|p| p.as_ref())
    }

    /// All registered provider ids, sorted.
    pub fn ids(&self) -> Vec<String> {
        self.providers.keys().map(|id| id.to_string()).collect()
    }

    /// Iterate over all providers.
    pub fn iter(&self) -> impl Iterator<Item = &dyn Provider> {
        self.providers.values().map(|p| p.as_ref())
    }
}
