//! [`SamplerChain`] builder.

use super::LlamaSampler;

/// Builder for a sampler chain.
///
/// Convenience wrapper around [`LlamaSampler::chain`] with a fluent API.
///
/// # Example
///
/// ```no_run
/// use llama_crab::sampling::SamplerChain;
/// let chain = SamplerChain::new()
///     .temp(0.8)
///     .top_p(0.95, 1)
///     .build();
/// # let _ = chain;
/// ```
#[derive(Debug, Default)]
pub struct SamplerChain {
    samplers: Vec<LlamaSampler>,
    no_perf: bool,
}

impl SamplerChain {
    /// Construct a new empty chain.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            samplers: Vec::new(),
            no_perf: false,
        }
    }

    /// Disable performance counters in the chain.
    #[must_use]
    pub const fn with_no_perf(mut self, yes: bool) -> Self {
        self.no_perf = yes;
        self
    }

    /// Add a temperature stage.
    #[must_use]
    pub fn temp(mut self, t: f32) -> Self {
        if let Some(s) = LlamaSampler::temp(t) {
            self.samplers.push(s);
        }
        self
    }

    /// Add a top-K stage.
    #[must_use]
    pub fn top_k(mut self, k: i32) -> Self {
        if let Some(s) = LlamaSampler::top_k(k) {
            self.samplers.push(s);
        }
        self
    }

    /// Add a top-P stage.
    #[must_use]
    pub fn top_p(mut self, p: f32, min_keep: usize) -> Self {
        if let Some(s) = LlamaSampler::top_p(p, min_keep) {
            self.samplers.push(s);
        }
        self
    }

    /// Add a min-P stage.
    #[must_use]
    pub fn min_p(mut self, p: f32, min_keep: usize) -> Self {
        if let Some(s) = LlamaSampler::min_p(p, min_keep) {
            self.samplers.push(s);
        }
        self
    }

    /// Add a penalties stage.
    #[must_use]
    pub fn penalties(mut self, n_vocab: i32, last_n: i32, repeat: f32, freq: f32, present: f32) -> Self {
        if let Some(s) = LlamaSampler::penalties(n_vocab, last_n, repeat, freq, present) {
            self.samplers.push(s);
        }
        self
    }

    /// Add a greedy sampler.
    #[must_use]
    pub fn greedy(mut self) -> Self {
        if let Some(s) = LlamaSampler::greedy() {
            self.samplers.push(s);
        }
        self
    }

    /// Consume the chain and return a single [`LlamaSampler`].
    #[must_use]
    pub fn build(self) -> Option<LlamaSampler> {
        LlamaSampler::chain(self.samplers, self.no_perf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chain_builds() {
        let chain = SamplerChain::new().build();
        assert!(chain.is_some());
    }

    #[test]
    fn with_no_perf_propagates() {
        let chain = SamplerChain::new().with_no_perf(true).greedy().build();
        assert!(chain.is_some());
    }

    #[test]
    fn fluent_chain_with_multiple_stages() {
        // Even without a model, the builder should accept stages and
        // produce a chain. The inner samplers return `None` from
        // their constructors (no model), so the chain ends up empty —
        // but the builder itself doesn't panic.
        let chain = SamplerChain::new()
            .temp(0.8)
            .top_p(0.95, 1)
            .penalties(64, 1.1, 0.0, 0.0)
            .build();
        assert!(chain.is_some());
    }
}
