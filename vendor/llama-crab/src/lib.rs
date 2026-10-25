//! `llama-crab` — safe, ergonomic Rust bindings to `llama.cpp`.
//!
//! ## Quickstart
//!
//! ```no_run
//! use llama_crab::{Llama, LlamaParams};
//!
//! let mut llama = Llama::load(LlamaParams::new("model.gguf").with_n_ctx(2048))?;
//! let resp = llama.create_completion("Hello, world!", 64)?;
//! println!("{}", resp.text);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Loading from Hugging Face
//!
//! Pass a Hugging Face repository id (e.g. `TheBloke/Llama-2-7B-Chat-GGUF`)
//! directly to `LlamaParams::new`; the library will download the GGUF to
//! the official HF cache and load it. For repos with multiple `.gguf`
//! files, specify the filename via `with_hf_filename`:
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! #[cfg(feature = "hf-hub")]
//! {
//! use llama_crab::{Llama, LlamaParams};
//!
//! let mut llama = Llama::load(
//!     LlamaParams::new("TheBloke/Llama-2-7B-Chat-GGUF")
//!         .with_hf_filename("llama-2-7b-chat.Q4_K_M.gguf")
//!         .with_n_ctx(2048),
//! )?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! The model is cached at `~/.cache/huggingface/hub` (or `$HF_HOME/hub` if
//! set). Set `HF_TOKEN` for gated repos. Requires the `hf-hub` cargo feature.

#![doc(
    html_logo_url = "https://gist.githubusercontent.com/DominguesM/127b9e5614e0e2da6b896fb3da3c8f2d/raw/a42e6daa629eb6f0254de026c47da9aff3c1f0e2/canarim-crab.webp"
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::needless_doctest_main)]
// A binding crate has a large public API surface; pedantic lints add
// little value. The CI enforces *correctness* (compilation, tests,
// docs) via `-D warnings` and the workspace's curated lint set, but we
// don't promote every individual pedantic warning to an error.
#![allow(
    dead_code,
    unused_imports,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo
)]

pub mod backend;
pub mod batch;
pub mod cache;
pub mod chat;
pub mod context;
pub mod error;
pub mod hf;
pub mod high_level;
pub mod json_schema;
pub mod log;
pub mod logit_bias;
pub mod model;
pub mod sampling;
pub mod speculative;
pub mod token;
pub mod token_data;
pub mod util;

#[cfg(feature = "mtmd")]
#[cfg_attr(docsrs, doc(cfg(feature = "mtmd")))]
pub mod multimodal;

pub use crate::backend::{LlamaBackend, NumaStrategy};
pub use crate::batch::{BatchAddError, LlamaBatch};
pub use crate::chat::Role;
pub use crate::context::{LlamaContext, LlamaContextParams};
pub use crate::error::{LlamaError, Result};
pub use crate::hf::{HfDownloader, HfRepo, MockHfDownloader};
pub use crate::high_level::chat_completion::ChatMessage;
pub use crate::high_level::completion::{
    Completion, CompletionChunk, CompletionLogprobs, CompletionOptions, StopReason, StreamControl,
    TokenLogprob,
};
pub use crate::high_level::tokenizer::{FimTokens, LlamaTokenizer, Tokenizer};
pub use crate::high_level::{Llama, LlamaParams, MobilePreset};
pub use crate::log::{send_logs_to_tracing, LogOptions};
pub use crate::logit_bias::LlamaLogitBias;
pub use crate::model::{params::LlamaModelParams, LlamaModel};
pub use crate::sampling::{LlamaSampler, SamplerChain};
pub use crate::token::LlamaToken;
pub use crate::token_data::{LlamaTokenData, LlamaTokenDataArray};
