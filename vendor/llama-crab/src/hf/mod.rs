//! Hugging Face integration. Real network access requires the `hf-hub` feature.
//!
//! Submodules:
//! - [`repo`]: validated `HfRepo` newtype (`org/name` or bare `name`).
//! - [`downloader`]: `HfDownloader` trait with `MockHfDownloader` (always
//!   available) and the feature-gated `RealHfDownloader` (requires `hf-hub`).

pub mod downloader;
pub mod repo;
pub mod source;

pub use self::downloader::{HfDownloader, MockHfDownloader};
pub use self::repo::HfRepo;
