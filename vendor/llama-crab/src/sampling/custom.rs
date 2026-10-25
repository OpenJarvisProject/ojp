//! Custom sampler extension point.
//!
//! llama.cpp exposes a C-ABI vtable (`llama_sampler_i`) that allows
//! third-party samplers to be inserted into a chain. This module is a
//! placeholder for v0.2 — the v0.1 release ships the vtable in
//! `llama-crab-sys` (when the `llguidance` feature is enabled) but
//! does not yet expose a public Rust trait.

/// Placeholder trait reserved for a future release. Lets a downstream
/// user implement a custom sampler chain stage that runs in pure
/// Rust and interops with llama.cpp via the C-ABI vtable.
pub trait CustomSampler {
    /// Apply the sampler to the candidate set.
    fn apply(&mut self, candidates: &mut crate::token_data::LlamaTokenDataArray);
    /// Accept a sampled token.
    fn accept(&mut self, token: crate::token::LlamaToken);
    /// Reset internal state.
    fn reset(&mut self);
}
