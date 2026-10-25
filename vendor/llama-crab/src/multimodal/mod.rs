//! Multimodal (vision + audio) support via `mtmd`.
//!
//! This module wraps the `mtmd` C API exposed by llama.cpp's
//! `tools/mtmd/` directory. To use it, enable the `mtmd` cargo feature
//! and load a GGUF model that has a paired `mmproj` (multimodal projector)
//! file.
//!
//! Top-level types: [`MtmdContext`], [`MtmdBitmap`], [`MtmdInputText`],
//! [`MtmdInputChunks`], [`MtmdInputChunk`], [`MtmdContextParams`].
//! Use [`default_media_marker`] to discover the placeholder string that
//! must appear in the prompt.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "mtmd")] {
//! use llama_crab::multimodal::{MtmdContext, MtmdBitmap, MtmdInputText};
//! use llama_crab::{Llama, LlamaParams};
//!
//! let mut llama = Llama::load(
//!     LlamaParams::new("gemma-4-E4B-it-Q4_K_M.gguf").with_n_ctx(4096)
//! )?;
//! let mtmd = MtmdContext::init_from_file("gemma-4-E4B-it-mmproj.gguf", llama.model())?;
//! let bitmap = MtmdBitmap::from_file("image.png")?;
//! let chunks = mtmd.tokenize(MtmdInputText::new("Describe this image"), &[&bitmap])?;
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod bitmap;
mod chunks;
mod context;

pub use bitmap::{MtmdBitmap, MtmdInputText};
pub use chunks::{MtmdInputChunk, MtmdInputChunks};
pub use context::{default_media_marker, MtmdContext, MtmdContextParams};
