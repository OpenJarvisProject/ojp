//! Embedding extraction.
//!
//! Use the `embeddings` method on `LlamaContext` to obtain the embedding
//! of the last decoded sequence, or `embeddings_ith` for a specific
//! sequence id.

use crate::context::LlamaContext;
use crate::error::{LlamaError, Result};

impl LlamaContext {
    /// Return the embedding vector of the last sequence in the most recent
    /// `encode` call.
    ///
    /// The returned slice is valid until the next `encode` or until the
    /// context is dropped.
    ///
    /// # Errors
    /// Returns [`LlamaError::Embedding`] if the context was not configured
    /// for embeddings (`LlamaContextParams::with_embeddings(false)`).
    pub fn embeddings(&self) -> Result<&[f32]> {
        let ptr = unsafe { llama_crab_sys::llama_get_embeddings(self.raw()) };
        if ptr.is_null() {
            return Err(LlamaError::Embedding(
                "embeddings not enabled (LlamaContextParams::with_embeddings(true))".into(),
            ));
        }
        let n = self.model().n_embd() as usize;
        // Safety: `ptr` is a `*mut f32` of length `n_embd` per the llama.cpp
        // contract. We never mutate through this slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) })
    }

    /// Return the embedding of a specific sequence in the most recent
    /// `encode` call.
    ///
    /// # Errors
    /// Returns [`LlamaError::Embedding`] on failure (not enabled or
    /// out-of-range index).
    pub fn embeddings_seq(&self, seq_id: i32) -> Result<&[f32]> {
        let ptr = unsafe { llama_crab_sys::llama_get_embeddings_seq(self.raw(), seq_id) };
        if ptr.is_null() {
            return Err(LlamaError::Embedding(format!(
                "no embedding for seq {seq_id}"
            )));
        }
        let n = self.model().n_embd() as usize;
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) })
    }

    /// Return the embedding vector of the `i`-th token in the last batch.
    pub fn embeddings_ith(&self, i: i32) -> Result<&[f32]> {
        let ptr = unsafe { llama_crab_sys::llama_get_embeddings_ith(self.raw(), i) };
        if ptr.is_null() {
            return Err(LlamaError::Embedding(format!("no embedding at index {i}")));
        }
        let n = self.model().n_embd() as usize;
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) })
    }

    /// Normalize a single embedding to unit L2 norm (in place).
    pub fn normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }
}
