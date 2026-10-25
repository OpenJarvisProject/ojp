//! [`HfDownloader`] trait and [`MockHfDownloader`] test double.
//!
//! The trait is `Send + Sync` so it can be shared across the multi-threaded
//! server worker. [`MockHfDownloader`] is a thread-safe in-memory stub used
//! by the resolver tests (Task 6+) and the load-time integration tests
//! (Task 11) without requiring a network round-trip.
//!
//! The `HfDownloader` impl for `MockHfDownloader` is `#[cfg(test)]`-gated
//! because the mock creates on-disk `.gguf` blobs via the `tempfile` crate,
//! which is currently a `dev-dependency` (kept out of the production lib to
//! avoid pulling in a binary-format helper that is never used at runtime).

use crate::error::LlamaError;
use std::path::PathBuf;
use std::sync::Arc;

use super::repo::HfRepo;

/// Abstraction for fetching a single file from a Hugging Face repo.
///
/// Implementations must be safe to share across threads (`Send + Sync`) —
/// the HTTP server worker may resolve several model lookups concurrently.
pub trait HfDownloader: Send + Sync {
    /// Download (or return from cache) `filename` from `repo`.
    ///
    /// # Errors
    /// Returns [`LlamaError::ModelDownload`] (or another variant) on network
    /// failure, missing file, or any other I/O problem.
    fn get(&self, repo: &HfRepo, filename: &str) -> Result<PathBuf, LlamaError>;

    /// Enumerate filenames available in `repo`.
    ///
    /// The default impl returns `Ok(vec![])` so simple stubs / stubs that
    /// only need `get` do not have to implement listing. The real downloader
    /// (Task 8) will use `Api::model(...).info()` to surface `.gguf` files
    /// for the auto-pick path in the resolver.
    fn list_repo_files(&self, _repo: &HfRepo) -> Result<Vec<String>, LlamaError> {
        Ok(vec![])
    }
}

/// In-memory [`HfDownloader`] double used by unit tests.
///
/// Holds:
/// - a `(repo_id, filename) -> path` cache (serves `get` cache hits),
/// - a single-shot error injection slot (one pre-armed error is consumed
///   on the next `get` call),
/// - a per-repo `list_repo_files` override map (default: empty).
///
/// The fields are `std::sync::Mutex<...>` to keep the type `Send + Sync`
/// without pulling in `parking_lot` for this single use site.
#[derive(Debug)]
pub struct MockHfDownloader {
    cache: std::sync::Mutex<std::collections::HashMap<(String, String), PathBuf>>,
    next_error: std::sync::Mutex<Option<LlamaError>>,
    list_files: std::sync::Mutex<std::collections::HashMap<String, Vec<String>>>,
}

impl Default for MockHfDownloader {
    fn default() -> Self {
        Self {
            cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            next_error: std::sync::Mutex::new(None),
            list_files: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }
}

#[cfg(test)]
impl MockHfDownloader {
    /// Pre-populate the cache: subsequent `get(repo, filename)` returns `path`
    /// without going through the on-disk blob-creation path.
    #[must_use]
    pub fn with_paths(self, repo: &str, filename: &str, path: PathBuf) -> Self {
        self.cache
            .lock()
            .expect("mock cache mutex poisoned")
            .insert((repo.to_string(), filename.to_string()), path);
        self
    }

    /// Configure `list_repo_files(repo)` to return `files`. Replaces any
    /// previously-configured list for the same repo.
    #[must_use]
    pub fn with_files(self, repo: &str, files: Vec<String>) -> Self {
        self.list_files
            .lock()
            .expect("mock list_files mutex poisoned")
            .insert(repo.to_string(), files);
        self
    }

    /// Arm a one-shot error: the next call to `get` will return `err` and
    /// clear the slot. Subsequent `get` calls behave normally.
    #[must_use]
    pub fn with_next_error(self, err: LlamaError) -> Self {
        *self
            .next_error
            .lock()
            .expect("mock next_error mutex poisoned") = Some(err);
        self
    }
}

#[cfg(test)]
impl HfDownloader for MockHfDownloader {
    fn get(&self, repo: &HfRepo, filename: &str) -> Result<PathBuf, LlamaError> {
        // 1. If an error is armed, take it and clear the slot in a single
        //    critical section so a concurrent caller does not see it twice.
        let armed = std::mem::replace(
            &mut *self
                .next_error
                .lock()
                .expect("mock next_error mutex poisoned"),
            None,
        );
        if let Some(err) = armed {
            return Err(err);
        }

        // 2. Cache hit short-circuits before any I/O.
        //    Uses `as_str()` because the more specific `repo_id()` accessor
        //    is added by Task 7; both return the same inner `&str`.
        let key = (repo.as_str().to_string(), filename.to_string());
        if let Some(cached) = self
            .cache
            .lock()
            .expect("mock cache mutex poisoned")
            .get(&key)
            .cloned()
        {
            return Ok(cached);
        }

        // 3. Cold path: create a temp .gguf file with a valid GGUF v3 magic
        //    header so downstream code that does a defensive magic-check
        //    (and tests that do `std::fs::read`) observe the expected bytes.
        //    `keep()` returns `(File, PathBuf)`; the File is bound to a name
        //    we never close so the path survives.
        use std::io::Write as _;
        use tempfile::NamedTempFile;

        let mut tmp = NamedTempFile::new()
            .map_err(|e| LlamaError::ModelDownload(format!("tempfile create: {e}")))?;
        tmp.write_all(b"GGUF\x00\x00\x00\x03")
            .map_err(|e| LlamaError::ModelDownload(format!("tempfile write: {e}")))?;
        let (_file, path) = tmp
            .keep()
            .map_err(|e| LlamaError::ModelDownload(format!("tempfile keep: {e}")))?;

        // 4. Store and return.
        self.cache
            .lock()
            .expect("mock cache mutex poisoned")
            .insert(key, path.clone());
        Ok(path)
    }

    fn list_repo_files(&self, repo: &HfRepo) -> Result<Vec<String>, LlamaError> {
        let configured = self
            .list_files
            .lock()
            .expect("mock list_files mutex poisoned")
            .get(repo.as_str())
            .cloned();
        Ok(configured.unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LlamaError;
    use std::path::PathBuf;

    /// Second call with the same `(repo, filename)` returns the SAME path.
    /// (The first call still has to produce a usable on-disk file so the
    /// resolver can hand it to `LlamaModel::load_from_file`.)
    #[test]
    fn mock_caches_path_per_repo_filename() {
        let mock = MockHfDownloader::default();
        let repo = HfRepo::new("TheBloke/Foo").expect("valid repo id");

        let first = mock.get(&repo, "model.Q4_K_M.gguf").expect("first get");
        let second = mock.get(&repo, "model.Q4_K_M.gguf").expect("second get");

        assert_eq!(first, second, "same (repo, filename) must return same path");
        assert!(
            first.exists(),
            "cached path must still exist on disk: {}",
            first.display()
        );
        assert_eq!(
            std::fs::read(&first).expect("read back blob"),
            b"GGUF\x00\x00\x00\x03",
            "blob must contain valid GGUF v3 magic"
        );
    }

    /// When `next_error` is armed, the next `get` returns that error and
    /// clears the slot (so the call after that succeeds).
    #[test]
    fn mock_returns_injected_error() {
        let mock = MockHfDownloader::default()
            .with_next_error(LlamaError::ModelDownload("injected 404".into()));
        let repo = HfRepo::new("TheBloke/Foo").expect("valid repo id");

        let err = mock
            .get(&repo, "model.gguf")
            .expect_err("injected error must propagate");

        match err {
            LlamaError::ModelDownload(msg) => {
                assert_eq!(msg, "injected 404", "error message preserved");
            }
            other => panic!("expected ModelDownload, got {other:?}"),
        }

        // After consuming, the mock is healthy again.
        let recovered = mock
            .get(&repo, "model.gguf")
            .expect("post-injection get must succeed");
        assert!(recovered.exists(), "recovered path must exist on disk");
    }

    /// `list_repo_files` returns whatever was configured via `with_files`,
    /// falling back to the default empty `Vec` for repos that were not
    /// configured.
    #[test]
    fn mock_list_files_returns_configured() {
        let mock = MockHfDownloader::default()
            .with_files("TheBloke/Foo", vec!["a.gguf".into(), "b.gguf".into()])
            // A second repo is configured as empty to cover the "explicit
            // empty" branch separately from "never configured".
            .with_files("TheBloke/Empty", vec![]);

        let configured = mock
            .list_repo_files(&HfRepo::new("TheBloke/Foo").expect("valid"))
            .expect("list configured");
        assert_eq!(configured, vec!["a.gguf".to_string(), "b.gguf".to_string()]);

        let empty = mock
            .list_repo_files(&HfRepo::new("TheBloke/Empty").expect("valid"))
            .expect("list empty");
        assert!(empty.is_empty(), "explicit empty must be empty");

        let never = mock
            .list_repo_files(&HfRepo::new("TheBloke/Other").expect("valid"))
            .expect("list never-configured");
        assert!(
            never.is_empty(),
            "never-configured repo must fall through to default empty"
        );

        // Suppress the unused-import warning for PathBuf when nothing else
        // references it on a given toolchain.
        let _ = PathBuf::new();
    }

    /// Network integration test: assert that `RealHfDownloader::get` emits
    /// the expected start + success `tracing::info!` events, and that the
    /// captured output does NOT contain the token, endpoint, or a
    /// fully-qualified URL. Gated by three layers:
    ///
    /// 1. `#[cfg(feature = "hf-hub")]` — the downloader is conditionally
    ///    compiled so the test only exists when the impl exists.
    /// 2. `#[ignore]` — won't run with plain `cargo test`; opt in with
    ///    `cargo test -p llama-crab --features hf-hub -- --ignored`.
    /// 3. `LLAMA_CRAB_RUN_HF_INTEGRATION=1` env var — without it the test
    ///    silently `return`s so accidental `--ignored` runs without the
    ///    env var don't hit the network.
    ///
    /// The subscriber is scoped to the test body via
    /// `tracing::subscriber::with_default` so it does NOT install a
    /// process-global subscriber and cannot interfere with other tests.
    #[cfg(feature = "hf-hub")]
    #[test]
    #[ignore]
    fn real_downloader_logs_start_and_end() {
        if std::env::var("LLAMA_CRAB_RUN_HF_INTEGRATION").is_err() {
            eprintln!("skipping: set LLAMA_CRAB_RUN_HF_INTEGRATION=1 to enable");
            return;
        }
        // Thread-safe buffer that backs a custom `MakeWriter` so the
        // subscriber's output ends up in a `String` we can assert on.
        let buf: Arc<std::sync::Mutex<Vec<u8>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let writer = TracingBufWriter(buf.clone());
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_max_level(tracing::Level::INFO)
            .with_target(false)
            .with_ansi(false)
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            let dl = RealHfDownloader::new().expect("downloader init");
            let repo =
                HfRepo::new("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF").expect("valid repo id");
            dl.get(&repo, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
                .expect("download must succeed");
        });

        let captured = {
            let guard = buf.lock().expect("buf lock");
            String::from_utf8(guard.clone()).expect("captured bytes are utf-8")
        };

        // The two new `tracing::info!` sites (Task 11) must both appear.
        assert!(
            captured.contains("downloading from Hugging Face"),
            "start log missing in captured output: {captured}"
        );
        assert!(
            captured.contains("downloaded from Hugging Face"),
            "end log missing in captured output: {captured}"
        );
        // Field names per the plan: `repo=`, `filename=`, and (on the
        // success branch) `size_bytes=`, `elapsed_ms=`.
        assert!(
            captured.contains("repo=") && captured.contains("filename="),
            "expected fields 'repo=' and 'filename=' in start log: {captured}"
        );
        assert!(
            captured.contains("size_bytes=") && captured.contains("elapsed_ms="),
            "expected fields 'size_bytes=' and 'elapsed_ms=' in end log: {captured}"
        );
        // SEC-1: must NOT log token / endpoint / URL.
        assert!(
            !captured.contains("HF_TOKEN") && !captured.contains("token="),
            "token leaked into logs: {captured}"
        );
        assert!(
            !captured.contains("HF_ENDPOINT") && !captured.contains("endpoint="),
            "endpoint leaked into logs: {captured}"
        );
        assert!(
            !captured.contains("https://"),
            "URL leaked into logs: {captured}"
        );
    }

    /// `MakeWriter` impl that funnels every emitted line into a shared
    /// `Arc<Mutex<Vec<u8>>>`. Used by `real_downloader_logs_start_and_end`
    /// to capture `tracing` output without installing a global subscriber.
    #[cfg(feature = "hf-hub")]
    struct TracingBufWriter(Arc<std::sync::Mutex<Vec<u8>>>);

    #[cfg(feature = "hf-hub")]
    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for TracingBufWriter {
        type Writer = TracingBufGuard<'a>;
        fn make_writer(&'a self) -> Self::Writer {
            TracingBufGuard(self.0.lock().expect("buf lock"))
        }
    }

    #[cfg(feature = "hf-hub")]
    struct TracingBufGuard<'a>(std::sync::MutexGuard<'a, Vec<u8>>);

    #[cfg(feature = "hf-hub")]
    impl std::io::Write for TracingBufGuard<'_> {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
}

// =====================================================================
// `DisabledHfDownloader` (always compiled)
//
// Used as the production default when the `hf-hub` cargo feature is off.
// Every dispatch returns a clear runtime error that points the user at
// the build flag they need to enable. Always-compiled so `Llama::load`
// can resolve its default downloader regardless of feature state.
// =====================================================================

/// Fallback downloader used when the `hf-hub` feature is disabled.
/// Any HF dispatch returns a clear runtime error pointing the user at
/// the build flag they need to enable.
#[derive(Debug)]
struct DisabledHfDownloader;

impl HfDownloader for DisabledHfDownloader {
    fn get(&self, _repo: &HfRepo, _filename: &str) -> Result<PathBuf, LlamaError> {
        Err(LlamaError::ModelDownload(
            "hf-hub feature is disabled \u{2014} rebuild with --features hf-hub".into(),
        ))
    }
}

/// Construct the production-default [`HfDownloader`] for `Llama::load`.
///
/// When the `hf-hub` feature is enabled this returns a [`RealHfDownloader`]
/// configured from the current process environment (`HF_TOKEN`,
/// `HF_ENDPOINT`). Otherwise it returns a [`DisabledHfDownloader`] that
/// surfaces a clear "feature disabled" error at every dispatch site, so
/// `Llama::load` always compiles regardless of feature state.
pub(crate) fn default_downloader() -> Result<Arc<dyn HfDownloader>, LlamaError> {
    #[cfg(feature = "hf-hub")]
    {
        Ok(Arc::new(RealHfDownloader::new()?))
    }
    #[cfg(not(feature = "hf-hub"))]
    {
        Ok(Arc::new(DisabledHfDownloader))
    }
}

// =====================================================================
// `RealHfDownloader` (gated behind the `hf-hub` feature)
//
// This impl is compiled only when the `hf-hub` feature is enabled,
// so the rest of the crate does not pay the cost of pulling in the
// `ureq` / `http` stack. It is the only place in `llama-crab` that
// uses `hf-hub` types.
//
// Env-var handling: `RealHfDownloader::new()` reads `HF_TOKEN` and
// `HF_ENDPOINT` exactly once, at construction time. Builder methods
// (`with_*`) accept explicit values from the caller and must NOT
// re-read the environment — this keeps env-reads observable in a
// single grep and prevents accidental logging of the token via
// builder-call chain dumps.
//
// Security note: never log `self.token`, `self.endpoint`, or any
// fully-qualified URL. The Bearer header (`hf-hub` build_headers())
// would otherwise be one accidental `{:?}` away from being
// exfiltrated.
// =====================================================================

#[cfg(feature = "hf-hub")]
#[derive(Debug, Clone, Default)]
/// Production [`HfDownloader`] backed by the `hf-hub` crate.
///
/// Only available when the `hf-hub` cargo feature is enabled. Reads
/// `HF_TOKEN` and `HF_ENDPOINT` from the environment at construction
/// time (see [`RealHfDownloader::new`]); the `with_*` builders override
/// those env-derived defaults on a per-instance basis.
pub struct RealHfDownloader {
    /// Override for the on-disk cache directory. `None` -> use `hf-hub`'s
    /// default (`~/.cache/huggingface/hub` or `$HF_HOME/hub`).
    cache_dir: Option<PathBuf>,
    /// Hugging Face access token (for gated / private repos). `None` ->
    /// anonymous public access.
    token: Option<String>,
    /// Override for the HF endpoint (e.g. a mirror URL). `None` ->
    /// `https://huggingface.co`.
    endpoint: Option<String>,
    /// Pin the repo to a specific revision (branch, tag, or commit SHA).
    /// `None` -> `"main"`.
    revision: Option<String>,
}

#[cfg(feature = "hf-hub")]
impl RealHfDownloader {
    /// Construct a `RealHfDownloader` from the current process environment.
    ///
    /// Reads `HF_TOKEN` and `HF_ENDPOINT` via `std::env::var().ok()`. Missing
    /// env vars are NOT an error — the downloader falls back to anonymous
    /// public access (works for ungated models on `huggingface.co`).
    ///
    /// The signature returns `Result<Self, LlamaError>` to mirror the
    /// `new() -> Result` + `.with_*` chain that callers expect. The
    /// `Err` arm is reserved for genuine construction failures (none
    /// in v1; env-var reads are infallible).
    ///
    /// # Errors
    /// Currently never returns `Err`. Reserved for future construction
    /// failures (e.g. invalid `HF_HOME`).
    pub fn new() -> Result<Self, LlamaError> {
        Ok(Self {
            cache_dir: None,
            token: std::env::var("HF_TOKEN").ok(),
            endpoint: std::env::var("HF_ENDPOINT").ok(),
            revision: None,
        })
    }

    /// Override the cache directory. Defaults to `~/.cache/huggingface/hub`
    /// (or `$HF_HOME/hub` if `HF_HOME` is set) — controlled by `hf-hub`.
    #[must_use]
    pub fn with_cache_dir(mut self, dir: PathBuf) -> Self {
        self.cache_dir = Some(dir);
        self
    }

    /// Override the Hugging Face endpoint (e.g. `https://hf-mirror.com`).
    /// Equivalent to setting the `HF_ENDPOINT` env var at process level,
    /// but scoped to this downloader instance.
    #[must_use]
    pub fn with_endpoint(mut self, ep: String) -> Self {
        self.endpoint = Some(ep);
        self
    }

    /// Pin the repo to a specific revision (branch, tag, or commit SHA).
    /// Defaults to `"main"` if unset.
    #[must_use]
    pub fn with_revision(mut self, rev: String) -> Self {
        self.revision = Some(rev);
        self
    }

    /// Build an `hf_hub::Repo` from the configured `HfRepo` + revision.
    /// `hf-hub` 0.5 only exposes `Api::model(repo_id: String)` (no
    /// revision parameter), so we use `Api::repo(Repo::with_revision(...))`
    /// to pass a non-default revision.
    fn build_repo(&self, repo: &HfRepo) -> hf_hub::Repo {
        let repo_id = repo.repo_id().to_string();
        match &self.revision {
            Some(rev) => hf_hub::Repo::with_revision(repo_id, hf_hub::RepoType::Model, rev.clone()),
            None => hf_hub::Repo::new(repo_id, hf_hub::RepoType::Model),
        }
    }

    /// Build a configured `hf_hub::api::sync::Api` from this struct's
    /// fields. Reads `HF_HOME` / `HF_ENDPOINT` / token-file via
    /// `ApiBuilder::from_env()`, then layers our overrides on top.
    fn build_api(&self) -> Result<hf_hub::api::sync::Api, LlamaError> {
        let mut builder = hf_hub::api::sync::ApiBuilder::from_env();
        if let Some(dir) = &self.cache_dir {
            builder = builder.with_cache_dir(dir.clone());
        }
        if let Some(tok) = &self.token {
            builder = builder.with_token(Some(tok.clone()));
        }
        if let Some(ep) = &self.endpoint {
            builder = builder.with_endpoint(ep.clone());
        }
        builder
            .build()
            .map_err(|e| LlamaError::ModelDownload(format!("api build: {e}")))
    }
}

#[cfg(feature = "hf-hub")]
impl HfDownloader for RealHfDownloader {
    fn get(&self, repo: &HfRepo, filename: &str) -> Result<PathBuf, LlamaError> {
        // SEC-1: never log `self.token` / `self.endpoint` / any URL. The
        // only fields emitted here are the public `repo_id`, the public
        // `filename`, the on-disk size, and the elapsed wall time.
        let started = std::time::Instant::now();
        tracing::info!(
            repo = repo.repo_id(),
            filename,
            "downloading from Hugging Face"
        );
        let result: Result<PathBuf, LlamaError> = (|| {
            let api = self.build_api()?;
            let api_repo = api.repo(self.build_repo(repo));
            api_repo
                .get(filename)
                .map_err(|e| LlamaError::ModelDownload(format!("download: {e}")))
        })();
        match &result {
            Ok(path) => {
                let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                tracing::info!(
                    repo = repo.repo_id(),
                    filename,
                    size_bytes,
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    "downloaded from Hugging Face"
                );
            }
            Err(_) => {
                tracing::warn!(
                    repo = repo.repo_id(),
                    filename,
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    "Hugging Face download failed"
                );
            }
        }
        result
    }

    fn list_repo_files(&self, repo: &HfRepo) -> Result<Vec<String>, LlamaError> {
        let api = self.build_api()?;
        let api_repo = api.repo(self.build_repo(repo));
        let info = api_repo
            .info()
            .map_err(|e| LlamaError::ModelDownload(format!("repo info: {e}")))?;
        Ok(info
            .siblings
            .iter()
            .map(|s| s.rfilename.clone())
            .filter(|n| n.ends_with(".gguf"))
            .collect())
    }
}

#[cfg(all(test, feature = "hf-hub"))]
mod real_tests {
    use super::*;

    /// Smoke test: construction must succeed with no env vars set. This
    /// is not a network test — it only verifies the env-var reads in
    /// `new()` are infallible and the struct can be built.
    ///
    /// The implementation reads `HF_TOKEN` / `HF_ENDPOINT` via `.ok()`,
    /// so this test cannot and does NOT set/unset env vars (env-var
    /// manipulation in tests is racy under multi-threaded test
    /// runners). It would only fail if `std::env::var` itself panicked
    /// or if the struct construction was otherwise impossible.
    #[test]
    fn real_downloader_constructs_with_no_env() {
        let result = RealHfDownloader::new();
        assert!(
            result.is_ok(),
            "RealHfDownloader::new must succeed (env reads are .ok())"
        );
    }

    /// Network integration test: fetch a real `.gguf` from HF Hub.
    ///
    /// Gated by three layers:
    /// 1. `#[cfg(feature = "hf-hub")]` — won't compile when the feature
    ///    is off (real impl doesn't exist).
    /// 2. `#[ignore]` — won't run with plain `cargo test`; must use
    ///    `cargo test -- --ignored`.
    /// 3. `LLAMA_CRAB_RUN_HF_INTEGRATION=1` env var — checked at test
    ///    runtime. If unset, the test panics with a clear message so
    ///    accidental `--ignored` runs in CI are visible.
    ///
    /// Run with:
    /// ```text
    /// LLAMA_CRAB_RUN_HF_INTEGRATION=1 \
    ///   cargo test -p llama-crab --features hf-hub \
    ///   --lib hf::downloader::real_tests::real_downloader_fetches_tinyllama -- --ignored
    /// ```
    #[test]
    #[ignore]
    fn real_downloader_fetches_tinyllama() {
        if std::env::var("LLAMA_CRAB_RUN_HF_INTEGRATION")
            .ok()
            .as_deref()
            != Some("1")
        {
            panic!("LLAMA_CRAB_RUN_HF_INTEGRATION must be set to 1 to run this network test");
        }
        let dl = RealHfDownloader::new().expect("downloader init");
        let repo = HfRepo::new("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF").expect("valid repo id");
        let path = dl
            .get(&repo, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
            .expect("download must succeed");
        assert!(
            path.exists(),
            "downloaded file must exist on disk: {}",
            path.display()
        );
        let meta = std::fs::metadata(&path).expect("stat downloaded file");
        assert!(meta.len() > 0, "downloaded file must be > 0 bytes");
    }
}
