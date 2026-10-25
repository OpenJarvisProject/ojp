//! End-to-end test: download a small HF model and verify cache hit.
//!
//! Skips cleanly unless `LLAMA_CRAB_RUN_HF_INTEGRATION=1` is set. Requires
//! the `hf-hub` cargo feature (gated by `required-features = ["hf-hub"]` in
//! `Cargo.toml`, so the test does not even compile under `--no-default-features`).
//!
//! When enabled, this test exercises the real downloader against a public,
//! ungated HF repo. The download populates the standard HF cache; the second
//! call must hit that cache and return the same on-disk path. A successful
//! `Llama::load` at the end is the smoke test of the full pipeline.

use llama_crab::hf::downloader::RealHfDownloader;
use llama_crab::HfDownloader;
use llama_crab::{HfRepo, Llama, LlamaParams};
use std::path::PathBuf;

mod common;

const REPO: &str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
const FILENAME: &str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

#[test]
fn hf_download_and_cache_hit() {
    let Some(_) = common::resolve_hf_integration(REPO, FILENAME) else {
        return;
    };
    let repo = HfRepo::new(REPO).expect("valid repo id");
    let dl = RealHfDownloader::new().expect("downloader init");

    // First download (or cache hit if the model is already on disk).
    let p1 = dl.get(&repo, FILENAME).expect("first download");
    assert!(p1.exists(), "downloaded file must exist on disk");
    let size1 = std::fs::metadata(&p1).map(|m| m.len()).unwrap_or(0);
    assert!(size1 > 0, "downloaded file must be non-empty");

    // Second call: cache hit must return the same path and size.
    let p2 = dl.get(&repo, FILENAME).expect("cache hit");
    assert_eq!(p1, p2, "second get must return the same path (cache hit)");
    assert_eq!(
        std::fs::metadata(&p2).map(|m| m.len()).unwrap_or(0),
        size1,
        "cached size must match"
    );

    eprintln!("hf_download: cache hit verified at {}", p1.display());

    // Smoke: actually try to load the model through Llama::load. The path
    // is on disk so the resolver takes the Local branch.
    let path_buf = PathBuf::from(&p1);
    let _ = Llama::load(LlamaParams::new(path_buf).with_n_ctx(512))
        .expect("Llama::load should succeed with cached .gguf path");
}
