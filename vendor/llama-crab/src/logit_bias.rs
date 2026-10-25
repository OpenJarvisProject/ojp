//! [`LlamaLogitBias`] — a single `(token, bias)` pair, used by the
//! logit-bias sampler.
//!
//! Moved from `token_data.rs` so the public types live in a stable,
//! documented home (PLAN.md).

/// One token + the additive bias applied to its logit before sampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LlamaLogitBias {
    /// Token id whose logit should be biased.
    pub token: i32,
    /// Additive bias. Positive ⇒ more likely; negative ⇒ less likely.
    pub bias: f32,
}

impl LlamaLogitBias {
    /// Construct a new `(token, bias)` pair.
    #[must_use]
    pub const fn new(token: i32, bias: f32) -> Self {
        Self { token, bias }
    }
}

impl From<(i32, f32)> for LlamaLogitBias {
    fn from((token, bias): (i32, f32)) -> Self {
        Self::new(token, bias)
    }
}

impl From<LlamaLogitBias> for (i32, f32) {
    fn from(b: LlamaLogitBias) -> Self {
        (b.token, b.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_constructs() {
        let b = LlamaLogitBias::new(42, -1.5);
        assert_eq!(b.token, 42);
        assert!((b.bias + 1.5).abs() < 1e-6);
    }

    #[test]
    fn from_tuple() {
        let b: LlamaLogitBias = (100, 0.25).into();
        assert_eq!(b.token, 100);
        let (t, bias) = b.into();
        assert_eq!(t, 100);
        assert!((bias - 0.25).abs() < 1e-6);
    }

    #[test]
    fn copy_semantics() {
        let a = LlamaLogitBias::new(1, 2.0);
        let b = a;
        assert_eq!(a.token, b.token);
    }
}
