//! `LlamaModel` and its parameters.

pub mod buft_overrides;
pub mod kv_overrides;
pub mod params;
pub mod vocab;

use std::path::Path;
use std::ptr::NonNull;

use llama_crab_sys as sys;

use crate::backend::LlamaBackend;
use crate::context::{LlamaContext, LlamaContextParams};
use crate::error::{LlamaError, Result};
use crate::token::LlamaToken;

pub use buft_overrides::BufferTypeOverride;
pub use kv_overrides::ParamOverrideValue;
pub(crate) use vocab::VocabPtr;

/// A loaded GGUF model.
#[derive(Debug)]
pub struct LlamaModel {
    pub(crate) handle: NonNull<sys::llama_model>,
    pub(crate) vocab: VocabPtr,
}

impl LlamaModel {
    /// Load a GGUF file from disk.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened or if llama.cpp rejects
    /// the parameters.
    pub fn load_from_file(
        _backend: &LlamaBackend,
        path: &Path,
        params: &params::LlamaModelParams,
    ) -> Result<Self> {
        let path_cstr = std::ffi::CString::new(path.display().to_string())?;
        let raw_params = params.build();
        // Safety: caller must ensure `raw_params` is valid and `path` is a
        // valid GGUF file.
        let handle = unsafe { sys::llama_load_model_from_file(path_cstr.as_ptr(), raw_params) };
        let handle = NonNull::new(handle).ok_or_else(|| {
            LlamaError::ModelLoad(format!("llama_load_model_from_file: {}", path.display()))
        })?;
        // Fetch and cache the vocab.
        // Safety: handle was just successfully created and is non-null.
        let vocab_raw = unsafe { sys::llama_model_get_vocab(handle.as_ptr()) };
        let vocab = NonNull::new(vocab_raw.cast_mut())
            .ok_or_else(|| LlamaError::ModelLoad("llama_model_get_vocab returned null".into()))?;
        Ok(Self {
            handle,
            vocab: VocabPtr(vocab),
        })
    }

    /// Internal: borrow the underlying C handle.
    pub(crate) fn raw(&self) -> *mut sys::llama_model {
        self.handle.as_ptr()
    }

    /// Internal: borrow the underlying `*const llama_vocab`.
    pub(crate) fn vocab(&self) -> *const sys::llama_vocab {
        self.vocab.0.as_ptr()
    }

    /// Number of layers in the model.
    #[must_use]
    pub fn n_layer(&self) -> u32 {
        unsafe { sys::llama_model_n_layer(self.handle.as_ptr()) as u32 }
    }

    /// Training-time context size.
    #[must_use]
    pub fn n_ctx_train(&self) -> u32 {
        unsafe { sys::llama_model_n_ctx_train(self.handle.as_ptr()) as u32 }
    }

    /// Embedding dimension.
    #[must_use]
    pub fn n_embd(&self) -> i32 {
        unsafe { sys::llama_model_n_embd(self.handle.as_ptr()) }
    }

    /// Vocabulary size.
    #[must_use]
    pub fn n_vocab(&self) -> i32 {
        unsafe { sys::llama_n_vocab(self.vocab()) }
    }

    /// True if the model is recurrent (Mamba-style).
    #[must_use]
    pub fn is_recurrent(&self) -> bool {
        unsafe { sys::llama_model_is_recurrent(self.handle.as_ptr()) }
    }

    /// True if the model is hybrid (recurrent + attention).
    #[must_use]
    pub fn is_hybrid(&self) -> bool {
        unsafe { sys::llama_model_is_hybrid(self.handle.as_ptr()) }
    }

    /// Size of the model file in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        unsafe { sys::llama_model_size(self.handle.as_ptr()) }
    }

    /// Number of parameters.
    #[must_use]
    pub fn n_params(&self) -> u64 {
        unsafe { sys::llama_model_n_params(self.handle.as_ptr()) }
    }

    /// Begin-of-sequence token.
    #[must_use]
    pub fn token_bos(&self) -> LlamaToken {
        LlamaToken(unsafe { sys::llama_token_bos(self.vocab()) })
    }

    /// End-of-sequence token.
    #[must_use]
    pub fn token_eos(&self) -> LlamaToken {
        LlamaToken(unsafe { sys::llama_token_eos(self.vocab()) })
    }

    /// End-of-turn token.
    #[must_use]
    pub fn token_eot(&self) -> LlamaToken {
        LlamaToken(unsafe { sys::llama_token_eot(self.vocab()) })
    }

    /// Tokenize `text`. By default, BOS is added only if the model requires it.
    ///
    /// # Errors
    /// Returns an error if a null byte is present in `text`.
    pub fn tokenize(&self, text: &str, add_bos: bool, special: bool) -> Result<Vec<LlamaToken>> {
        let bytes = text.as_bytes();
        // First call: query the required size by passing a tiny buffer.
        let mut buf: Vec<i32> = vec![0; bytes.len().saturating_add(8)];
        let n = unsafe {
            sys::llama_tokenize(
                self.vocab(),
                bytes.as_ptr().cast(),
                bytes.len() as i32,
                buf.as_mut_ptr(),
                buf.len() as i32,
                add_bos,
                special,
            )
        };
        if n < 0 {
            return Err(LlamaError::Batch(format!("tokenize failed: {n}")));
        }
        if (n as usize) > buf.len() {
            buf.resize(n as usize, 0);
            let n2 = unsafe {
                sys::llama_tokenize(
                    self.vocab(),
                    bytes.as_ptr().cast(),
                    bytes.len() as i32,
                    buf.as_mut_ptr(),
                    buf.len() as i32,
                    add_bos,
                    special,
                )
            };
            if n2 < 0 {
                return Err(LlamaError::Batch(format!("tokenize retry: {n2}")));
            }
        }
        buf.truncate(n as usize);
        Ok(buf.into_iter().map(LlamaToken).collect())
    }

    /// Detokenize a slice of tokens into a `String`.
    ///
    /// UTF-8 decoding is **lossy**: bytes that do not form a valid UTF-8
    /// sequence are replaced with the Unicode replacement character
    /// (`U+FFFD`). BPE tokenizers can emit individual tokens whose raw bytes
    /// are not valid UTF-8, especially around non-Latin scripts and emoji.
    pub fn detokenize(&self, tokens: &[LlamaToken], special: bool) -> Result<String> {
        let mut raw_buf: Vec<u8> = vec![0; 64];
        let mut len = raw_buf.len() as i32;
        let n = unsafe {
            sys::llama_detokenize(
                self.vocab(),
                tokens.as_ptr().cast(),
                tokens.len() as i32,
                raw_buf.as_mut_ptr().cast(),
                len,
                special,
                special,
            )
        };
        if n < 0 {
            return Err(LlamaError::Batch(format!("detokenize: {n}")));
        }
        if (n as usize) > raw_buf.len() {
            raw_buf.resize(n as usize, 0);
            len = n;
            let n2 = unsafe {
                sys::llama_detokenize(
                    self.vocab(),
                    tokens.as_ptr().cast(),
                    tokens.len() as i32,
                    raw_buf.as_mut_ptr().cast(),
                    len,
                    special,
                    special,
                )
            };
            if n2 < 0 {
                return Err(LlamaError::Batch(format!("detokenize: {n2}")));
            }
        }
        raw_buf.truncate(n as usize);
        Ok(String::from_utf8_lossy(&raw_buf).into_owned())
    }

    /// Construct a new context from this model.
    ///
    /// # Errors
    /// Returns an error if context allocation fails (e.g. out of memory).
    pub fn new_context(
        &self,
        _backend: &LlamaBackend,
        params: LlamaContextParams,
    ) -> Result<LlamaContext> {
        let raw_params = params.build();
        let ctx = unsafe { sys::llama_new_context_with_model(self.handle.as_ptr(), raw_params) };
        // Safety: `self.handle` is a `NonNull<llama_model>` (non-null
        // by construction); cast to `NonNull<LlamaModel>` is a
        // transparent pointer cast and the resulting pointer is
        // valid as long as `&self` is alive.
        let model_ptr = NonNull::from(&*self);
        NonNull::new(ctx)
            .map(|ctx| LlamaContext::from_raw(ctx, model_ptr))
            .ok_or(LlamaError::ContextLoad(
                "llama_new_context_with_model returned null".into(),
            ))
    }
}

// Safety: `llama_model` is thread-safe in the sense documented by llama.cpp
// (read-only after initialization).
unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        // Safety: `handle` is exclusively owned and was returned by
        // `llama_load_model_from_file`.
        unsafe { sys::llama_free_model(self.handle.as_ptr()) };
    }
}
