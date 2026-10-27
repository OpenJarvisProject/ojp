//! ojp-core: everything the OJP apps share, with no UI.
//!
//! Every piece is optional and swappable (config.toml): a router model, a
//! main model (in-process GGUF or any OpenAI-compatible server), tools, and
//! the persona (name, wake word, voice). A piece that is missing or fails
//! degrades gracefully instead of taking the rest down; see engine.rs.
pub mod chats;
pub mod config;
pub mod dsp;
pub mod download;
pub mod engine;
pub mod format;
pub mod link;
pub mod math;
pub mod memory;
pub mod model;
pub mod platform;
pub mod reminders;
pub mod router;
pub mod storage;
pub mod template;
pub mod tools;
pub mod tools_android;
pub mod tools_ext;
pub mod tts;

pub use config::Config;
pub use engine::{Engine, Event};
pub use router::Route;

/// One lock for every test that changes process-wide environment variables (HOME, ...).
#[cfg(test)]
pub(crate) static TEST_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());
