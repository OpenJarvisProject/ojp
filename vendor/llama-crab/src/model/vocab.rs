//! Cached `*const llama_vocab` handle.
//!
//! `llama.cpp` exposes a per-vocab C struct that owns all token
//! metadata. The Rust binding obtains one via `llama_model_get_vocab`
//! and caches it on the [`crate::model::LlamaModel`] for the lifetime
//! of the model. The handle is read-only and safe to share between
//! threads.

use std::ptr::NonNull;

use llama_crab_sys as sys;

/// Thin wrapper around `*const llama_vocab`.
//
// Safety: `llama_vocab` is read-only and thread-safe per llama.cpp
// documentation.
#[derive(Debug, Clone, Copy)]
pub struct VocabPtr(pub(crate) NonNull<sys::llama_vocab>);

impl VocabPtr {
    /// Wrap a raw vocab pointer; returns `None` if null.
    pub(crate) fn from_raw(raw: *mut sys::llama_vocab) -> Option<Self> {
        NonNull::new(raw).map(Self)
    }

    /// Borrow the underlying `*const llama_vocab`.
    pub(crate) fn as_ptr(&self) -> *const sys::llama_vocab {
        self.0.as_ptr()
    }
}
