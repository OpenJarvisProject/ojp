//! Helpers shared by the integration tests in this folder.
//!
//! All tests that exercise real GGUF models **skip cleanly** when the
//! model file is not present, so `cargo test` never fails on a fresh
//! checkout. To run them, set one of the env vars below or place the
//! model in the conventional path.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// Default location for the text-only Gemma 4 GGUF.
pub const GEMMA4_DEFAULT_PATH: &str = "models/gemma-4-E4B-it-Q4_K_M.gguf";

/// Default location for the Gemma 4 multimodal projector.
pub const GEMMA4_MMPROJ_DEFAULT_PATH: &str = "models/mmproj-gemma-4-E4B-it-BF16.gguf";

/// Default location for the LFM2.5-VL GGUF.
pub const LFM_VL_DEFAULT_PATH: &str = "models/LFM2.5-VL-1.6B-Q4_K_M.gguf";

/// Default location for the LFM2.5-VL multimodal projector.
pub const LFM_VL_MMPROJ_DEFAULT_PATH: &str = "models/LFM2.5-VL-1.6B-mmproj-BF16.gguf";

/// Default location for a small test image (256×256 PNG).
pub const TEST_IMAGE_DEFAULT_PATH: &str = "tests/fixtures/test_image.png";

/// Default location for Qwen2.5 0.5B Instruct (used by infill / streaming tests).
pub const QWEN_DEFAULT_PATH: &str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

/// Default location for bge-reranker-base (used by the rerank API test).
pub const RERANK_DEFAULT_PATH: &str = "models/bge-reranker-base-q4_k_m.gguf";

/// Resolve a model path from `LLAMA_CRAB_<NAME>` env var, falling back to
/// the conventional default.
pub fn resolve_path(env_var: &str, default: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env_var) {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    let path = PathBuf::from(default);
    if path.exists() {
        return Some(path);
    }
    let workspace_path = PathBuf::from("..").join(default);
    if workspace_path.exists() {
        return Some(workspace_path);
    }
    None
}

/// Skip a test if the given path does not exist; return a guard value
/// otherwise.
#[macro_export]
macro_rules! require_path {
    ($path:expr) => {{
        match $path {
            Some(p) => p,
            None => {
                eprintln!(
                    "skipping test: model not found (set the corresponding env var \
                     or place the file in the conventional path)"
                );
                return;
            }
        }
    }};
}

/// Common skip message used by every multimodal test.
pub fn skip_unless(cond: bool, what: &str) -> Option<()> {
    if cond {
        Some(())
    } else {
        eprintln!("skipping test: {what} not available");
        None
    }
}

/// Print a banner so the test output is easy to grep.
pub fn banner(test: &str, model: &Path) {
    eprintln!();
    eprintln!("================================================================");
    eprintln!("  {test}");
    eprintln!("  model : {}", model.display());
    eprintln!("================================================================");
}

/// Skip the current test if `LLAMA_CRAB_RUN_HF_INTEGRATION` is not set.
/// Returns `Some((repo, filename))` on success, `None` on skip.
pub fn resolve_hf_integration(
    repo: &'static str,
    filename: &'static str,
) -> Option<(&'static str, &'static str)> {
    if std::env::var("LLAMA_CRAB_RUN_HF_INTEGRATION").is_ok() {
        Some((repo, filename))
    } else {
        eprintln!("skipping HF integration test: set LLAMA_CRAB_RUN_HF_INTEGRATION=1 to enable");
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires manual env manipulation; run with --ignored"]
    fn resolve_hf_integration_returns_none_when_env_unset() {
        // SAFETY: this test is `#[ignore]` so it does not run in parallel with other env-using tests.
        unsafe {
            std::env::remove_var("LLAMA_CRAB_RUN_HF_INTEGRATION");
        }
        assert!(super::resolve_hf_integration("foo/bar", "x.gguf").is_none());
    }
}
