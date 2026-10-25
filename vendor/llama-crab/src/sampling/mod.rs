//! Sampling strategies and the [`LlamaSampler`] wrapper.
//!
//! All 17 sampling strategies exposed by `llama.cpp` are available as
//! associated functions on [`LlamaSampler`]. Use [`LlamaSampler::chain`] to
//! compose them into a pipeline.
//!
//! At the top level you find the [`LlamaSampler`] wrapper itself and its
//! core methods (`sample`, `apply`, `accept`, `reset`, `get_seed`), plus
//! the [`SamplerChain`] builder.

mod chain;
mod custom;
mod grammar;
mod strategies;

use llama_crab_sys as sys;

use crate::token::LlamaToken;
use crate::token_data::LlamaTokenDataArray;

pub use chain::SamplerChain;

#[cfg(feature = "common")]
pub use grammar::GrammarError;

/// One-shot strategy. Use the associated constructors below.
///
/// `LlamaSampler` is **not** `Send`/`Sync`: llama.cpp samplers carry mutable
/// state (seeds, history) that is not safe to share across threads without
/// external synchronization.
#[derive(Debug)]
pub struct LlamaSampler {
    handle: *mut sys::llama_sampler,
}

impl LlamaSampler {
    // -- Construction ----------------------------------------------------

    /// Wrap a raw `*mut llama_sampler` produced by an `llama_sampler_init_*` call.
    ///
    /// # Safety
    /// The pointer must be a valid, non-null, non-aliased sampler returned by
    /// llama.cpp. After construction, this type takes ownership and frees the
    /// sampler in `Drop`.
    #[allow(dead_code)]
    pub(crate) unsafe fn from_raw(ptr: *mut sys::llama_sampler) -> Self {
        Self { handle: ptr }
    }

    /// Construct a [`LlamaSampler`] from an already-initialized raw pointer
    /// *without* taking ownership. The returned sampler must NOT be dropped
    /// (used internally for cloning via `llama_sampler_clone`).
    #[allow(dead_code)]
    pub(crate) unsafe fn from_raw_borrowed(ptr: *mut sys::llama_sampler) -> Self {
        Self { handle: ptr }
    }

    /// Borrow the underlying pointer (used by the C API).
    #[allow(dead_code)]
    pub(crate) fn as_ptr(&self) -> *mut sys::llama_sampler {
        self.handle
    }

    // -- Core operations -------------------------------------------------

    /// Sample a single token from the context's logits.
    ///
    /// # Safety
    /// `ctx` must point to a live, unaliased `llama_context`.
    pub unsafe fn sample(&mut self, ctx: *mut sys::llama_context, idx: i32) -> LlamaToken {
        // Safety: caller guarantees `ctx` is a valid, live, unaliased context.
        let raw = unsafe { sys::llama_sampler_sample(self.handle, ctx, idx) };
        LlamaToken(raw)
    }

    /// Apply the sampler to a [`LlamaTokenDataArray`].
    pub fn apply(&self, candidates: &mut LlamaTokenDataArray) {
        unsafe {
            sys::llama_sampler_apply(self.handle, candidates.as_mut_ptr());
        }
    }

    /// Reset internal sampler state (seeds, history, etc.).
    pub fn reset(&mut self) {
        unsafe { sys::llama_sampler_reset(self.handle) };
    }

    /// Accept a sampled token into the sampler's history.
    pub fn accept(&mut self, token: LlamaToken) {
        unsafe { sys::llama_sampler_accept(self.handle, token.0) };
    }

    /// Get the random seed used by the sampler.
    #[must_use]
    pub fn get_seed(&self) -> u32 {
        unsafe { sys::llama_sampler_get_seed(self.handle) }
    }

    // -- Composition -----------------------------------------------------

    /// Compose a chain of samplers into a single sampler. The order matters:
    /// each stage sees the candidates as transformed by the previous stage.
    ///
    /// The inner samplers are consumed (their raw pointers are moved into the
    /// chain), so calling `chain.add(...)` invalidates them.
    #[must_use]
    pub fn chain(samplers: Vec<LlamaSampler>, no_perf: bool) -> Option<Self> {
        let mut chain_params = unsafe { sys::llama_sampler_chain_default_params() };
        chain_params.no_perf = no_perf;
        let chain = unsafe { sys::llama_sampler_chain_init(chain_params) };
        if chain.is_null() {
            return None;
        }
        for mut s in samplers {
            unsafe { sys::llama_sampler_chain_add(chain, s.handle) };
            // The chain now owns the inner sampler; prevent double-free.
            s.handle = std::ptr::null_mut();
        }
        Some(unsafe { Self::from_raw(chain) })
    }
}

impl Drop for LlamaSampler {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // Safety: `self.handle` was created by llama_sampler_init_* and
            // we own it.
            unsafe { sys::llama_sampler_free(self.handle) };
        }
    }
}
