//! Error types for `llama-crab`.
//!
//! All fallible APIs return [`Result<T, LlamaError>`].

#![allow(clippy::module_inception)]

use thiserror::Error;

/// Result alias used throughout `llama-crab`.
pub type Result<T> = std::result::Result<T, LlamaError>;

/// Top-level error type for all `llama-crab` operations.
///
/// This enum is `#[non_exhaustive]` — new variants will be added in minor
/// releases as more features are exposed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LlamaError {
    /// Generic IO error.
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    /// Null byte inside a string passed to the C API.
    #[error("interior nul byte in string: {0}")]
    Nul(#[from] std::ffi::NulError),

    /// UTF-8 decoding failed.
    #[error("invalid utf-8: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    /// UTF-8 conversion failed.
    #[error("invalid utf-8: {0}")]
    Utf8Lossy(#[from] std::str::Utf8Error),

    /// Failed to load a GGUF model.
    #[error("failed to load model: {0}")]
    ModelLoad(String),

    /// Failed to download a model from Hugging Face.
    #[error("huggingface download: {0}")]
    ModelDownload(String),

    /// Failed to create a context.
    #[error("failed to create context: {0}")]
    ContextLoad(String),

    /// Decode failure (returned by `llama_decode`).
    #[error("decode failed (code {0})")]
    Decode(i32),

    /// Encode failure (returned by `llama_encode`).
    #[error("encode failed (code {0})")]
    Encode(i32),

    /// A batch operation overflowed or was invalid.
    #[error("batch error: {0}")]
    Batch(String),

    /// Embedding extraction failed.
    #[error("embedding error: {0}")]
    Embedding(String),

    /// Operation attempted without an initialized backend.
    #[error("backend not initialized")]
    BackendNotInitialized,

    /// JSON-schema → GBNF conversion failed.
    #[error("json schema to grammar: {0}")]
    JsonSchemaToGrammar(String),

    /// Chat template processing failed.
    #[error("chat template: {0}")]
    ChatTemplate(String),

    /// Generic catch-all for C-ABI FFI errors.
    #[error("C-ABI error (code {0})")]
    Ffi(i32),
}

impl LlamaError {
    /// Construct from a null pointer returned by a C function.
    #[must_use]
    pub fn null_return() -> Self {
        Self::Ffi(-1)
    }
}

impl From<std::num::TryFromIntError> for LlamaError {
    fn from(e: std::num::TryFromIntError) -> Self {
        Self::Batch(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaError;

    #[test]
    fn model_download_display() {
        assert_eq!(
            format!("{}", crate::LlamaError::ModelDownload("404".into())),
            "huggingface download: 404"
        );
    }
}
