//! Sampling-state introspection.
//!
//! After a forward pass the context exposes the full logits and
//! probability distribution. These helpers let a caller inspect the
//! "what would have happened if I had sampled with a different
//! strategy" without re-running the model.

use crate::context::LlamaContext;
use crate::error::Result;
use crate::token::LlamaToken;

impl LlamaContext {
    /// Borrow the logits for the `i`-th token in the last batch.
    ///
    /// The slice is valid until the next decode/encode call or until the
    /// context is dropped.
    ///
    /// # Errors
    /// Returns an error if `i` is out of range.
    pub fn logits_ith(&self, i: i32) -> Result<&[f32]> {
        let ptr = unsafe { llama_crab_sys::llama_get_logits_ith(self.raw_handle(), i) };
        if ptr.is_null() {
            return Err(crate::error::LlamaError::Batch(format!(
                "no logits at index {i}"
            )));
        }
        // Safety: `ptr` is a `*mut f32` of length `n_vocab`.
        let n = self.model().n_vocab() as usize;
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) })
    }

    /// The token that the default sampler would have picked at position `i`.
    #[must_use]
    pub fn sampled_token_ith(&self, i: i32) -> LlamaToken {
        let raw = unsafe { llama_crab_sys::llama_get_sampled_token_ith(self.raw_handle(), i) };
        LlamaToken(raw)
    }

    /// Probability distribution that the default sampler produced at `i`.
    ///
    /// # Errors
    /// Returns an error if the index is out of range.
    pub fn sampled_probs_ith(&self, i: i32) -> Result<&[f32]> {
        let ptr = unsafe { llama_crab_sys::llama_get_sampled_probs_ith(self.raw_handle(), i) };
        if ptr.is_null() {
            return Err(crate::error::LlamaError::Batch(format!(
                "no sampled probs at index {i}"
            )));
        }
        let n = self.model().n_vocab() as usize;
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) })
    }
}
