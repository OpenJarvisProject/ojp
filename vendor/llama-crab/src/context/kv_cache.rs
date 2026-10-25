//! KV-cache management operations.
//!
//! The KV cache is llama.cpp's internal memory that stores key/value
//! tensors for every past token. In v0.1 we expose the four essential
//! manipulation operations: `seq_rm`, `seq_cp`, `seq_add`, `seq_div` —
//! plus the position queries `seq_pos_min` and `seq_pos_max`. All of
//! these methods are no-ops if the model was not initialised with a
//! context that has `n_seq_max > 1`.
//!
//! Each method is a member of [`LlamaContext`].

use crate::context::LlamaContext;
use crate::error::Result;

impl LlamaContext {
    /// Remove all tokens in `seq_id` between `p0` and `p1` from the KV cache.
    ///
    /// # Errors
    /// Returns an error if the context was not initialised for multi-sequence
    /// inference.
    pub fn seq_rm(&self, seq_id: i32, p0: i32, p1: i32) -> Result<()> {
        // Safety: the context handle is live and uniquely owned.
        let mem = unsafe { llama_crab_sys::llama_get_memory(self.raw_handle()) };
        if mem.is_null() {
            return Err(crate::error::LlamaError::Batch(
                "context has no memory (n_seq_max == 0?)".into(),
            ));
        }
        let _ = unsafe { llama_crab_sys::llama_memory_seq_rm(mem, seq_id, p0, p1) };
        Ok(())
    }

    /// Copy tokens in `src_seq` between `p0` and `p1` into `dst_seq`.
    pub fn seq_cp(&self, src_seq: i32, dst_seq: i32, p0: i32, p1: i32) -> Result<()> {
        let mem = unsafe { llama_crab_sys::llama_get_memory(self.raw_handle()) };
        if mem.is_null() {
            return Err(crate::error::LlamaError::Batch(
                "context has no memory (n_seq_max == 0?)".into(),
            ));
        }
        unsafe { llama_crab_sys::llama_memory_seq_cp(mem, src_seq, dst_seq, p0, p1) };
        Ok(())
    }

    /// Keep only `seq_id` in the KV cache, dropping every other sequence.
    pub fn seq_keep(&self, seq_id: i32) -> Result<()> {
        let mem = unsafe { llama_crab_sys::llama_get_memory(self.raw_handle()) };
        if mem.is_null() {
            return Err(crate::error::LlamaError::Batch(
                "context has no memory (n_seq_max == 0?)".into(),
            ));
        }
        unsafe { llama_crab_sys::llama_memory_seq_keep(mem, seq_id) };
        Ok(())
    }

    /// Shift tokens in `seq_id` between `p0` and `p1` by `shift` positions.
    pub fn seq_add(&self, seq_id: i32, p0: i32, p1: i32, shift: i32) -> Result<()> {
        let mem = unsafe { llama_crab_sys::llama_get_memory(self.raw_handle()) };
        if mem.is_null() {
            return Err(crate::error::LlamaError::Batch(
                "context has no memory (n_seq_max == 0?)".into(),
            ));
        }
        unsafe { llama_crab_sys::llama_memory_seq_add(mem, seq_id, p0, p1, shift) };
        Ok(())
    }

    /// Divide tokens in `seq_id` between `p0` and `p1` by `d` (in-place).
    pub fn seq_div(&self, seq_id: i32, p0: i32, p1: i32, d: i32) -> Result<()> {
        let mem = unsafe { llama_crab_sys::llama_get_memory(self.raw_handle()) };
        if mem.is_null() {
            return Err(crate::error::LlamaError::Batch(
                "context has no memory (n_seq_max == 0?)".into(),
            ));
        }
        unsafe { llama_crab_sys::llama_memory_seq_div(mem, seq_id, p0, p1, d) };
        Ok(())
    }

    /// Minimum position occupied by `seq_id` in the KV cache.
    #[must_use]
    pub fn seq_pos_min(&self, seq_id: i32) -> i32 {
        let mem = unsafe { llama_crab_sys::llama_get_memory(self.raw_handle()) };
        if mem.is_null() {
            return -1;
        }
        unsafe { llama_crab_sys::llama_memory_seq_pos_min(mem, seq_id) }
    }

    /// Maximum position occupied by `seq_id` in the KV cache.
    #[must_use]
    pub fn seq_pos_max(&self, seq_id: i32) -> i32 {
        let mem = unsafe { llama_crab_sys::llama_get_memory(self.raw_handle()) };
        if mem.is_null() {
            return -1;
        }
        unsafe { llama_crab_sys::llama_memory_seq_pos_max(mem, seq_id) }
    }
}
