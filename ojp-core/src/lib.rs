//! ojp-core: everything the OJP apps share, with no UI.
//!
//! Every piece is optional and swappable (config.toml): a router model, a
//! main model (in-process GGUF or any OpenAI-compatible server), tools, and
//! the persona (name, wake word, voice). A piece that is missing or fails
//! degrades gracefully instead of taking the rest down; see engine.rs.
pub mod config;
pub mod download;
pub mod engine;
pub mod format;
pub mod model;
pub mod router;
pub mod storage;
pub mod tools;

pub use config::Config;
pub use engine::{Engine, Event};
pub use router::Route;
