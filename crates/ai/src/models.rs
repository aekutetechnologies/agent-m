//! Per-model configuration.

use serde::{Deserialize, Serialize};

use crate::types::Usage;

/// A single model a provider serves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Model id sent on the wire, e.g. `deepseek-chat`.
    pub id: String,
    /// Optional human-friendly name.
    #[serde(default)]
    pub name: Option<String>,
    /// Whether the model emits reasoning (thinking) deltas.
    #[serde(default)]
    pub reasoning: bool,
    /// Context window in tokens, when known.
    #[serde(default)]
    pub context_window: Option<u64>,
    /// USD per 1M tokens for prompt tokens not served from the cache
    /// (cache miss + fresh tokens).
    #[serde(default)]
    pub price_in_miss: f64,
    /// USD per 1M prompt tokens served from the provider's prefix cache.
    #[serde(default)]
    pub price_in_hit: f64,
    /// USD per 1M output tokens.
    #[serde(default)]
    pub price_out: f64,
}

impl ModelSpec {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            reasoning: false,
            context_window: None,
            price_in_miss: 0.0,
            price_in_hit: 0.0,
            price_out: 0.0,
        }
    }

    pub fn reasoning(mut self, reasoning: bool) -> Self {
        self.reasoning = reasoning;
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn context_window(mut self, context_window: u64) -> Self {
        self.context_window = Some(context_window);
        self
    }

    /// Set per-1M-token pricing in USD (cache miss input, cache hit input,
    /// output).
    pub fn pricing(mut self, in_miss: f64, in_hit: f64, out: f64) -> Self {
        self.price_in_miss = in_miss;
        self.price_in_hit = in_hit;
        self.price_out = out;
        self
    }

    /// Estimated cost in USD for one usage record.
    pub fn cost_for(&self, usage: &Usage) -> f64 {
        let uncached = usage
            .input_tokens
            .saturating_sub(usage.cache_read_tokens + usage.cache_creation_tokens);
        (usage.cache_read_tokens as f64 * self.price_in_hit
            + (usage.cache_creation_tokens as f64 + uncached as f64) * self.price_in_miss
            + usage.output_tokens as f64 * self.price_out)
            / 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_uses_cache_hit_pricing() {
        let spec = ModelSpec::new("m").pricing(0.27, 0.07, 1.10);
        let usage = Usage {
            input_tokens: 1000,
            output_tokens: 1000,
            cache_read_tokens: 800,
            cache_creation_tokens: 100,
            total_tokens: 2000,
            cost: 0.0,
        };
        // 800 cached-read @ $0.07 + 200 cache-miss @ $0.27 + 1000 out @ $1.10.
        let expected = (800.0 * 0.07 + 200.0 * 0.27 + 1000.0 * 1.10) / 1_000_000.0;
        assert!((spec.cost_for(&usage) - expected).abs() < 1e-12);
    }
}
