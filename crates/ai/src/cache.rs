//! Cache observability: how many prompt tokens the provider served from its
//! prefix cache vs. recomputed. Mirrors pi's `core/cache-stats.ts`.

use crate::types::Usage;

/// Accumulated cache statistics across requests.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CacheStats {
    /// Tokens served from the provider's cache.
    pub hit_tokens: u64,
    /// Tokens the provider had to process (cache misses).
    pub miss_tokens: u64,
    /// Number of requests recorded.
    pub requests: u64,
}

impl CacheStats {
    /// Record one request's usage.
    pub fn record(&mut self, usage: &Usage) {
        self.hit_tokens += usage.cache_read_tokens;
        self.miss_tokens += usage.cache_creation_tokens;
        self.requests += 1;
    }

    /// Fraction of cached prompt tokens, or `None` when nothing was recorded.
    pub fn hit_ratio(&self) -> Option<f64> {
        let total = self.hit_tokens + self.miss_tokens;
        (total > 0).then(|| self.hit_tokens as f64 / total as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_reports_hit_ratio() {
        let mut stats = CacheStats::default();
        assert_eq!(stats.hit_ratio(), None);

        stats.record(&Usage {
            input_tokens: 1000,
            output_tokens: 50,
            cache_read_tokens: 900,
            cache_creation_tokens: 100,
            total_tokens: 1050,
            cost: 0.0,
        });

        assert_eq!(stats.requests, 1);
        assert_eq!(stats.hit_tokens, 900);
        let ratio = stats.hit_ratio().unwrap();
        assert!((ratio - 0.9).abs() < f64::EPSILON);
    }
}
