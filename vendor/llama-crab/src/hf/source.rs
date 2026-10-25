//! `ModelSource` discriminator and the `resolve` precedence function.
//!
//! Three input sources collapse to one of two load strategies:
//!
//! 1. **Local**: the path exists on disk (or is not a valid HF repo id).
//! 2. **Hf**: the path looks like `"org/name"` and does NOT exist on disk,
//!    OR the builder set an explicit `hf_repo_override`.
//!
//! Auto-pick logic (Hf branch with no explicit filename) is the only piece
//! that talks to the [`HfDownloader`]: it asks for the file list, filters
//! to `.gguf`, and either picks the sole candidate or errors out.
//!
//! This module is pure resolver logic — no FFI, no `LlamaBackend`, no
//! `LlamaParams` coupling. The integration in `Llama::load` is the one that
//! extracts the public input fields and calls `resolve`.

use std::path::{Path, PathBuf};

use crate::error::LlamaError;
use crate::hf::downloader::HfDownloader;
use crate::hf::repo::HfRepo;

/// How to load a model: from the local filesystem, or from a HF repo.
///
/// The enum is the *output* of a successful dispatch; the resolver returns
/// a [`PathBuf`] because callers want to hand it to `LlamaModel::load_from_file`
/// unconditionally — the Hf branch caches the downloaded blob on disk and
/// returns the local path to it.
pub(crate) enum ModelSource {
    /// Path on the local filesystem.
    Local(PathBuf),
    /// A HF repo and (optionally) a specific file inside it.
    Hf {
        /// Validated HF repo id (`"org/name"` or bare `"name"`).
        repo: HfRepo,
        /// Explicit filename inside the repo. `None` triggers auto-pick.
        filename: Option<String>,
    },
}

/// Resolve a `(model_path, hf_filename, hf_repo_override)` triple to a
/// concrete on-disk path, dispatching to HF if the input is a repo id
/// (or if the caller set an override).
///
/// # Precedence (locked — see plan + Task 6 spec)
/// 1. `hf_repo_override.is_some()` → Hf branch with that repo. The
///    explicit `hf_filename` is used; if `None`, the resolver asks
///    `downloader.list_repo_files` to auto-pick a single `.gguf`.
/// 2. `HfRepo::looks_like_repo_id(model_path_str) && !model_path.exists()`
///    → Hf branch, no explicit filename, auto-pick.
/// 3. Otherwise → Local branch, return `model_path.to_path_buf()`.
///
/// # Why primitives, not `&LlamaParams`?
/// Keeping the function free of `LlamaParams` decoupling means it can be
/// tested in isolation (Task 6) without spinning up the high-level
/// builder, and the integration site (Task 10) controls the mapping from
/// the builder's public fields to the resolver's arguments.
pub(crate) fn resolve(
    model_path: &Path,
    hf_filename: Option<&str>,
    hf_repo_override: Option<&HfRepo>,
    downloader: &dyn HfDownloader,
) -> Result<PathBuf, LlamaError> {
    // Precedence 1: explicit override from the builder wins.
    if let Some(repo) = hf_repo_override {
        return resolve_hf(repo, hf_filename, downloader);
    }

    // Precedence 2: auto-detect when the path string looks like a HF
    // repo id AND the file is not on disk. Non-UTF-8 paths fall through
    // to Local (a non-UTF-8 string cannot be a valid HF repo id).
    let is_hf_candidate = model_path.to_str().is_some_and(HfRepo::looks_like_repo_id);
    if is_hf_candidate && !model_path.exists() {
        // Safe to unwrap: `is_some_and` proved the path is UTF-8 above.
        let path_str = model_path.to_str().expect("UTF-8 checked above");
        let repo =
            HfRepo::new(path_str).expect("looks_like_repo_id returned true but new() failed");
        return resolve_hf(&repo, hf_filename, downloader);
    }

    // Precedence 3: local.
    Ok(model_path.to_path_buf())
}

/// Hf branch: pick a filename (explicit or auto) and download it.
fn resolve_hf(
    repo: &HfRepo,
    filename: Option<&str>,
    downloader: &dyn HfDownloader,
) -> Result<PathBuf, LlamaError> {
    let file = match filename {
        Some(f) => f.to_string(),
        None => auto_pick(repo, downloader)?,
    };
    downloader.get(repo, &file)
}

/// Auto-pick a single `.gguf` from the repo's file list.
fn auto_pick(repo: &HfRepo, downloader: &dyn HfDownloader) -> Result<String, LlamaError> {
    let files = downloader.list_repo_files(repo)?;
    let gguf: Vec<String> = files.into_iter().filter(|f| f.ends_with(".gguf")).collect();
    match gguf.len() {
        0 => Err(LlamaError::ModelDownload(format!(
            "no .gguf files in repo {}",
            repo.as_str()
        ))),
        1 => {
            let file = gguf.into_iter().next().expect("len == 1");
            tracing::info!(
                repo = repo.as_str(),
                file = %file,
                "auto-picked single .gguf"
            );
            Ok(file)
        }
        n => Err(LlamaError::ModelDownload(format!(
            "ambiguous: {n} gguf files in repo {}, use with_hf_filename",
            repo.as_str()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf::downloader::MockHfDownloader;
    use std::path::PathBuf;

    // ----- Local branch -----

    /// An existing file on disk is always Local, even if the path string
    /// would also satisfy `looks_like_repo_id`. (Non-regression for the
    /// most common case: a local `./models/foo.gguf` path.)
    #[test]
    fn resolve_local_when_path_exists() {
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        let path = tmp.path().to_path_buf();
        let dl = MockHfDownloader::default();

        let result = resolve(&path, None, None, &dl).expect("local resolves");

        assert_eq!(result, path, "local path returned unchanged");
    }

    /// `TheBloke/Llama` is a valid `HfRepo` (`looks_like_repo_id` == true),
    /// but if a file at that exact path exists, the resolver must still
    /// treat it as Local. This is the guardrail the
    /// `FORBIDDEN_FIRST_SEGMENT` denylist cannot save us from on its own
    /// (the denylist only catches `"models/..."` and `"model/..."`).
    #[test]
    fn resolve_local_when_path_looks_like_repo_id_but_file_exists() {
        let tmp_dir = tempfile::tempdir().expect("create tempdir");
        let path = tmp_dir.path().join("TheBloke").join("Llama");
        std::fs::create_dir_all(path.parent().unwrap()).expect("mkdir");
        std::fs::write(&path, b"local model bytes").expect("write");
        let dl = MockHfDownloader::default();

        let result = resolve(&path, None, None, &dl).expect("local resolves");

        assert_eq!(result, path, "existing file wins over looks_like_repo_id");
    }

    // ----- Hf branch: explicit override -----

    /// `hf_repo_override` set + explicit `hf_filename` → Hf branch with
    /// that exact filename, regardless of `model_path`. The pre-populated
    /// cache makes the mock return a known path so the test can assert
    /// the resolver threaded the arguments through to `downloader.get`.
    #[test]
    fn resolve_hf_when_repo_override_and_filename() {
        let repo = HfRepo::new("TheBloke/Foo").expect("valid repo id");
        let expected = PathBuf::from("/tmp/expected-explicit.gguf");
        let dl = MockHfDownloader::default().with_paths(
            "TheBloke/Foo",
            "explicit.gguf",
            expected.clone(),
        );
        let model_path = PathBuf::from("/nonexistent/local/model.gguf");

        let result = resolve(&model_path, Some("explicit.gguf"), Some(&repo), &dl)
            .expect("hf override resolves");

        assert_eq!(
            result, expected,
            "downloader.get called with explicit filename, returned cached path"
        );
    }

    // ----- Hf branch: auto-pick -----

    /// `model_path = "TheBloke/Foo"` and the file does not exist → Hf
    /// branch with auto-pick. The mock is configured with one `.gguf` so
    /// the resolver picks it.
    #[test]
    fn resolve_hf_when_path_looks_like_repo_id() {
        let expected = PathBuf::from("/tmp/only.gguf");
        let dl = MockHfDownloader::default()
            .with_files("TheBloke/Foo", vec!["only.gguf".to_string()])
            .with_paths("TheBloke/Foo", "only.gguf", expected.clone());
        let model_path = PathBuf::from("TheBloke/Foo");
        // Sanity guard: the test fixture must NOT exist on the cwd.
        assert!(
            !model_path.exists(),
            "test fixture leaked: {} exists on disk",
            model_path.display()
        );

        let result = resolve(&model_path, None, None, &dl).expect("hf auto-detect");

        assert_eq!(result, expected, "auto-picked the sole .gguf");
    }

    /// Same shape as the previous test, but spelled out as a dedicated
    /// "auto-pick 1 file" case so the failure mode is unambiguous if a
    /// regression breaks the 1-file branch specifically.
    #[test]
    fn resolve_hf_auto_pick_1_gguf() {
        let expected = PathBuf::from("/tmp/only.gguf");
        let dl = MockHfDownloader::default()
            .with_files("TheBloke/Foo", vec!["only.gguf".to_string()])
            .with_paths("TheBloke/Foo", "only.gguf", expected.clone());
        let model_path = PathBuf::from("TheBloke/Foo");

        let result = resolve(&model_path, None, None, &dl).expect("hf auto-pick 1");

        assert_eq!(result, expected, "1 .gguf -> picked exactly that file");
    }

    /// Zero `.gguf` files in the repo is a hard error — the auto-picker
    /// has nothing to pick. Message must call out the repo id so the
    /// user knows which repo failed.
    #[test]
    fn resolve_hf_error_0_gguf() {
        // Default mock returns empty for list_repo_files.
        let dl = MockHfDownloader::default();
        let model_path = PathBuf::from("TheBloke/Empty");

        let err = resolve(&model_path, None, None, &dl).expect_err("0 gguf files must error");

        match err {
            LlamaError::ModelDownload(msg) => {
                assert!(
                    msg.contains("no .gguf files in repo"),
                    "msg must say 'no .gguf files in repo', got: {msg}"
                );
                assert!(
                    msg.contains("TheBloke/Empty"),
                    "msg must include the repo id, got: {msg}"
                );
            }
            other => panic!("expected ModelDownload, got {other:?}"),
        }
    }

    /// More than one `.gguf` file is ambiguous: the user must call
    /// `with_hf_filename` to disambiguate. The error message must
    /// include the count and the suggested fix.
    #[test]
    fn resolve_hf_error_many_gguf() {
        let dl = MockHfDownloader::default().with_files(
            "TheBloke/Ambiguous",
            vec!["a.gguf".to_string(), "b.gguf".to_string()],
        );
        let model_path = PathBuf::from("TheBloke/Ambiguous");

        let err = resolve(&model_path, None, None, &dl).expect_err(">1 gguf files must error");

        match err {
            LlamaError::ModelDownload(msg) => {
                assert!(
                    msg.contains("ambiguous"),
                    "msg must say 'ambiguous', got: {msg}"
                );
                assert!(
                    msg.contains("2 gguf files in repo"),
                    "msg must include the count '2 gguf files in repo', got: {msg}"
                );
                assert!(
                    msg.contains("use with_hf_filename"),
                    "msg must point at the fix, got: {msg}"
                );
            }
            other => panic!("expected ModelDownload, got {other:?}"),
        }
    }

    /// Non-`.gguf` files in the repo must be filtered out of the
    /// auto-pick candidate set, and the first remaining `.gguf`
    /// (in the order returned by the downloader) wins. `README.md`
    /// and `config.json` flank the candidate to prove the filter
    /// actually runs (if we forgot to filter, `README.md` would have
    /// been a top-of-list candidate and the `a.gguf` cache key would
    /// never have been hit).
    #[test]
    fn resolve_hf_auto_pick_filters_to_gguf() {
        let expected = PathBuf::from("/tmp/a.gguf");
        let dl = MockHfDownloader::default()
            .with_files(
                "TheBloke/Foo",
                vec![
                    "a.gguf".to_string(),
                    "README.md".to_string(),
                    "config.json".to_string(),
                ],
            )
            .with_paths("TheBloke/Foo", "a.gguf", expected.clone());
        let model_path = PathBuf::from("TheBloke/Foo");

        let result = resolve(&model_path, None, None, &dl).expect("hf auto-pick filters");

        assert_eq!(
            result, expected,
            "first .gguf wins, non-.gguf files are ignored"
        );
    }
}
