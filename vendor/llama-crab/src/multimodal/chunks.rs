//! [`MtmdInputChunks`] and [`MtmdInputChunk`] — the list of tokenized
//! multimodal chunks produced by
//! [`MtmdContext::tokenize`](crate::multimodal::MtmdContext::tokenize).

use std::ptr::NonNull;

use llama_crab_sys as sys;

use super::context::MtmdContext;
use crate::error::{LlamaError, Result};

/// A list of tokenized chunks produced by
/// [`MtmdContext::tokenize`](crate::multimodal::MtmdContext::tokenize).
#[derive(Debug)]
pub struct MtmdInputChunks {
    pub(crate) handle: NonNull<sys::mtmd_input_chunks>,
}

impl MtmdInputChunks {
    pub(crate) fn new() -> Result<Self> {
        let handle = unsafe { sys::mtmd_input_chunks_init() };
        NonNull::new(handle)
            .map(|handle| Self { handle })
            .ok_or(LlamaError::Batch(
                "mtmd_input_chunks_init returned null".into(),
            ))
    }

    /// Number of chunks in the list.
    #[must_use]
    pub fn len(&self) -> usize {
        unsafe { sys::mtmd_input_chunks_size(self.handle.as_ptr()) }
    }

    /// True if there are no chunks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the chunk at `idx` (a borrowed view; lives as long as `self`).
    #[must_use]
    pub fn get(&self, idx: usize) -> Option<MtmdInputChunk<'_>> {
        let p = unsafe { sys::mtmd_input_chunks_get(self.handle.as_ptr(), idx) };
        NonNull::new(p.cast_mut()).map(|handle| MtmdInputChunk {
            handle,
            _owned: false,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Evaluate all chunks: encode the images, then decode the resulting
    /// tokens. The number of new positions consumed is written to
    /// `new_n_past`.
    ///
    /// # Safety
    /// `llama_ctx` must be a live, unaliased `llama_context`.
    pub unsafe fn eval(
        &self,
        mtmd_ctx: &MtmdContext,
        llama_ctx: *mut sys::llama_context,
        n_past: i32,
        seq_id: i32,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<i32> {
        let mut new_n_past: i32 = 0;
        let rc = unsafe {
            sys::mtmd_helper_eval_chunks(
                mtmd_ctx.as_ptr(),
                llama_ctx,
                self.handle.as_ptr(),
                n_past,
                seq_id,
                n_batch,
                logits_last,
                &mut new_n_past,
            )
        };
        if rc != 0 {
            return Err(LlamaError::Ffi(rc));
        }
        Ok(new_n_past)
    }
}

impl Drop for MtmdInputChunks {
    fn drop(&mut self) {
        // Safety: `handle` is exclusively owned.
        unsafe { sys::mtmd_input_chunks_free(self.handle.as_ptr()) };
    }
}

/// A single chunk (text or image embedding). Lifetime-bound to the
/// [`MtmdInputChunks`] that produced it.
#[derive(Debug)]
pub struct MtmdInputChunk<'a> {
    handle: NonNull<sys::mtmd_input_chunk>,
    _owned: bool,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> MtmdInputChunk<'a> {
    /// Number of tokens the chunk decodes into.
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe { sys::mtmd_input_chunk_get_n_tokens(self.handle.as_ptr()) }
    }

    /// Number of positions the chunk consumes in the KV cache.
    #[must_use]
    pub fn n_pos(&self) -> i32 {
        unsafe { sys::mtmd_input_chunk_get_n_pos(self.handle.as_ptr()) }
    }

    /// Internal: raw pointer (used by `MtmdContext::decode_use_non_causal`).
    pub(crate) fn as_ptr(&self) -> *mut sys::mtmd_input_chunk {
        self.handle.as_ptr()
    }
}

// Accessors for `MtmdContext` to internal handles
impl MtmdContext {
    pub(crate) fn as_ptr(&self) -> *mut sys::mtmd_context {
        self.handle.as_ptr()
    }
}
