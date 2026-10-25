//! High-level orchestrator: load model, create context, generate tokens.
//!
//! The API keeps model loading, context ownership and common generation flows
//! behind one safe Rust type.

pub mod chat_completion;
pub mod completion;
pub mod embedding;
pub mod hf_tokenizer;
pub mod infill;
pub mod openai_compat;
pub mod rerank;
pub mod tokenizer;

#[cfg(feature = "hf-tokenizer")]
#[cfg_attr(docsrs, doc(cfg(feature = "hf-tokenizer")))]
pub use self::hf_tokenizer::HfTokenizer;

pub use self::tokenizer::{FimTokens, LlamaTokenizer, Tokenizer};

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use crate::backend::LlamaBackend;
use crate::context::{LlamaContext, LlamaContextParams};
use crate::error::Result;
use crate::hf::downloader::HfDownloader;
use crate::hf::repo::HfRepo;
use crate::model::params::LlamaModelParams;
use crate::model::LlamaModel;

pub use self::chat_completion::{
    create_chat_completion, create_chat_completion_stream, create_chat_completion_stream_with,
    ChatMessage,
};
pub use self::completion::{
    create_completion, create_completion_stream, create_completion_stream_with_sampler,
    create_completion_with_options, create_completion_with_sampler, Completion, CompletionChunk,
    CompletionLogprobs, CompletionOptions, SamplingOptions, StopReason, StreamControl,
    TokenLogprob,
};

/// Top-level orchestrator. Owns the backend, the model and the context.
///
/// # Drop order
/// Fields are declared in **reverse drop order** so that Rust's
/// declaration-order dropping produces the correct teardown:
///
/// 1. `context` (first declared, dropped first) — `LlamaContext::Drop`
///    calls `llama_free`; the boxed `model` is still alive.
/// 2. `model` — the `Box<LlamaModel>` is freed. The `NonNull<LlamaModel>`
///    inside the context is no longer accessed past this point.
/// 3. `_backend` — the backend is unloaded after both the context and
///    the model have been released.
/// 4. `_not_send_sync` — zero-sized marker.
#[derive(Debug)]
pub struct Llama {
    context: LlamaContext,
    // Boxed: `LlamaContext` stores a raw `NonNull<LlamaModel>`
    // pointer that must point at a stable address across any move
    // of the outer `Llama` value. A heap allocation satisfies that
    // requirement; a stack slot in `Llama::load`'s frame does not
    // (that was the use-after-move, masked by a `mem::transmute`
    // that extended the context's lifetime to `'static`).
    model: Box<LlamaModel>,
    _backend: LlamaBackend,
    _not_send_sync: std::marker::PhantomData<*mut ()>,
}

impl Llama {
    /// Load a GGUF model with the given parameters.
    ///
    /// # Errors
    /// Returns an error if the file cannot be loaded, the model is
    /// rejected by llama.cpp, or context creation fails.
    pub fn load(params: LlamaParams) -> Result<Self> {
        let backend = LlamaBackend::init()?;

        // Resolve the model path: if the user passed a HF repo id and/or
        // filename, this dispatches to the configured `HfDownloader` and
        // returns a local on-disk path to the downloaded GGUF. The
        // resolver's precedence is:
        //   1. `params.hf_repo_override` (auto-set by `with_hf_filename`
        //      when `model_path` parses as a `HfRepo`) wins outright.
        //   2. `model_path` looks like a repo id AND does not exist on
        //      disk -> Hf branch with auto-pick.
        //   3. Otherwise -> Local branch (return `model_path` as-is).
        //
        // The test-injection slot `params.hf_downloader` lets unit tests
        // pass a `MockHfDownloader` and assert the dispatch path without
        // touching the env-reading `RealHfDownloader::new`.
        let downloader: Arc<dyn HfDownloader> = match params.hf_downloader.clone() {
            Some(d) => d,
            None => crate::hf::downloader::default_downloader()?,
        };
        let resolved_path = crate::hf::source::resolve(
            &params.model_path,
            params.hf_filename.as_deref(),
            params.hf_repo_override.as_ref(),
            downloader.as_ref(),
        )?;

        // Box the model before borrowing it: the `&LlamaModel`
        // captured by `LlamaContext::model` must point at a
        // heap-stable address that survives the move of `Llama`
        // into the caller's frame.
        let model = Box::new(LlamaModel::load_from_file(
            &backend,
            &resolved_path,
            &params.model,
        )?);
        // `ctx` now legitimately borrows from the boxed model;
        // its lifetime is tied to the borrow of `model` here,
        // which the borrow checker extends through the move
        // into `Self` because both fields are co-owned by
        // `Llama`. The previous `mem::transmute` to `'static`
        // was unsound and is gone.
        let ctx = model.new_context(&backend, params.context.clone())?;
        Ok(Self {
            context: ctx,
            model,
            _backend: backend,
            _not_send_sync: std::marker::PhantomData,
        })
    }

    /// Borrow the inner model.
    #[must_use]
    pub fn model(&self) -> &LlamaModel {
        // Deref through the Box; the return type is unchanged.
        &self.model
    }

    /// Borrow the inner context.
    #[must_use]
    pub fn context(&mut self) -> &mut LlamaContext {
        &mut self.context
    }

    /// Synchronous text completion. Generates up to `max_tokens` tokens.
    pub fn create_completion(&mut self, prompt: &str, max_tokens: usize) -> Result<Completion> {
        create_completion(self, prompt, max_tokens)
    }

    /// Synchronous text completion with high-level options.
    pub fn create_completion_with_options(
        &mut self,
        prompt: &str,
        options: CompletionOptions,
    ) -> Result<Completion> {
        create_completion_with_options(self, prompt, options)
    }

    /// Synchronous text completion using a caller-provided sampler.
    pub fn create_completion_with_sampler(
        &mut self,
        prompt: &str,
        options: CompletionOptions,
        sampler: &mut crate::sampling::LlamaSampler,
    ) -> Result<Completion> {
        create_completion_with_sampler(self, prompt, options, sampler)
    }

    /// Synchronous streaming text completion. The callback is invoked as text
    /// becomes available and can return [`StreamControl::Stop`] to end
    /// generation.
    pub fn create_completion_stream<F>(
        &mut self,
        prompt: &str,
        options: CompletionOptions,
        on_chunk: F,
    ) -> Result<Completion>
    where
        F: FnMut(CompletionChunk) -> StreamControl,
    {
        create_completion_stream(self, prompt, options, on_chunk)
    }

    /// Synchronous streaming text completion using a caller-provided sampler.
    pub fn create_completion_stream_with_sampler<F>(
        &mut self,
        prompt: &str,
        options: CompletionOptions,
        sampler: &mut crate::sampling::LlamaSampler,
        on_chunk: F,
    ) -> Result<Completion>
    where
        F: FnMut(CompletionChunk) -> StreamControl,
    {
        create_completion_stream_with_sampler(self, prompt, options, sampler, on_chunk)
    }

    /// Synchronous chat completion. The messages are rendered with the Plain
    /// built-in template and the response is decoded token-by-token.
    pub fn create_chat_completion(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
    ) -> Result<ChatMessage> {
        create_chat_completion(self, messages, max_tokens)
    }

    /// Synchronous streaming chat completion using the Plain template.
    pub fn create_chat_completion_stream<F>(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        on_chunk: F,
    ) -> Result<ChatMessage>
    where
        F: FnMut(CompletionChunk) -> StreamControl,
    {
        create_chat_completion_stream(self, messages, max_tokens, on_chunk)
    }

    /// Synchronous streaming chat completion with a chosen built-in template,
    /// optional tools, and completion options.
    pub fn create_chat_completion_stream_with<F>(
        &mut self,
        messages: &[ChatMessage],
        template: crate::chat::BuiltinTemplate,
        tools: &[crate::chat::ToolDefinition],
        options: CompletionOptions,
        on_chunk: F,
    ) -> Result<ChatMessage>
    where
        F: FnMut(CompletionChunk) -> StreamControl,
    {
        create_chat_completion_stream_with(self, messages, template, tools, options, on_chunk)
    }
}

/// All parameters accepted by [`Llama::load`].
#[derive(Clone)]
pub struct LlamaParams {
    /// Path to the GGUF file.
    pub model_path: PathBuf,
    /// Model-side params (GPU offload, mmap, etc.).
    pub model: LlamaModelParams,
    /// Context-side params (n_ctx, embeddings, etc.).
    pub context: LlamaContextParams,
    /// Filename within a Hugging Face repo (set via `with_hf_filename`).
    hf_filename: Option<String>,
    /// Revision (branch, tag, or commit) of a Hugging Face repo.
    hf_revision: Option<String>,
    /// Hugging Face access token for gated/private repos.
    hf_token: Option<String>,
    /// Override for the Hugging Face cache directory.
    hf_cache_dir: Option<PathBuf>,
    /// Override for the Hugging Face endpoint (e.g. a mirror).
    hf_endpoint: Option<String>,
    /// Auto-populated by `with_hf_filename` when `model_path` parses as a
    /// valid [`HfRepo`]. Forces the resolver's precedence #1 (override
    /// wins) so the Hf branch fires even when `model_path` is also a valid
    /// local file path.
    hf_repo_override: Option<HfRepo>,
    /// Test-injection slot for the [`HfDownloader`] implementation.
    /// `None` in production -> `Llama::load` calls
    /// `crate::hf::downloader::default_downloader()` to obtain the
    /// feature-appropriate default. `Some(d)` in tests -> `d` is used
    /// as-is, bypassing the env-reading `RealHfDownloader` constructor.
    hf_downloader: Option<Arc<dyn HfDownloader>>,
}

// Manual `Debug` impl: `Arc<dyn HfDownloader>` does not implement `Debug`
// (the trait is intentionally not required to). The `hf_downloader` field
// is rendered as the literal string `"<HfDownloader>"` to keep the trait
// object opaque in logs and to avoid accidentally invoking any `Debug`
// impl on the inner concrete type (which could leak implementation
// details in user code).
impl std::fmt::Debug for LlamaParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaParams")
            .field("model_path", &self.model_path)
            .field("model", &self.model)
            .field("context", &self.context)
            .field("hf_filename", &self.hf_filename)
            .field("hf_revision", &self.hf_revision)
            .field("hf_token", &self.hf_token)
            .field("hf_cache_dir", &self.hf_cache_dir)
            .field("hf_endpoint", &self.hf_endpoint)
            .field("hf_repo_override", &self.hf_repo_override)
            .field(
                "hf_downloader",
                &self.hf_downloader.as_ref().map(|_| "<HfDownloader>"),
            )
            .finish()
    }
}

/// High-level mobile-oriented parameter presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MobilePreset {
    /// Small batches and CPU-only execution for memory-constrained devices.
    LowRam,
    /// Balanced defaults for interactive mobile chat.
    Balanced,
    /// Prefer GPU offload and larger batches for capable devices.
    GpuMax,
}

impl FromStr for MobilePreset {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "low-ram" | "low_ram" | "lowram" => Ok(Self::LowRam),
            "balanced" => Ok(Self::Balanced),
            "gpu-max" | "gpu_max" | "gpumax" => Ok(Self::GpuMax),
            other => Err(format!(
                "unknown mobile preset: {other} (expected low-ram, balanced, or gpu-max)"
            )),
        }
    }
}

impl LlamaParams {
    /// Construct parameters targeting the given model.
    #[must_use]
    pub fn new(model_path: impl AsRef<Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            model: LlamaModelParams::default(),
            context: LlamaContextParams::default(),
            hf_filename: None,
            hf_revision: None,
            hf_token: None,
            hf_cache_dir: None,
            hf_endpoint: None,
            hf_repo_override: None,
            hf_downloader: None,
        }
    }

    /// Set the path to the GGUF file.
    #[must_use]
    pub fn with_model_path(mut self, p: impl AsRef<Path>) -> Self {
        self.model_path = p.as_ref().to_path_buf();
        self
    }

    /// Number of layers to offload to the GPU.
    #[must_use]
    pub fn with_n_gpu_layers(mut self, n: i32) -> Self {
        self.model = self.model.with_n_gpu_layers(n);
        self
    }

    /// Memory-map the model file.
    #[must_use]
    pub fn with_use_mmap(mut self, yes: bool) -> Self {
        self.model = self.model.with_use_mmap(yes);
        self
    }

    /// Set the context size.
    #[must_use]
    pub fn with_n_ctx(mut self, n: u32) -> Self {
        self.context = self.context.with_n_ctx(n);
        self
    }

    /// Configure the logical maximum batch size.
    #[must_use]
    pub fn with_n_batch(mut self, n: u32) -> Self {
        self.context = self.context.with_n_batch(n);
        self
    }

    /// Configure the physical batch size used by a forward pass.
    #[must_use]
    pub fn with_n_ubatch(mut self, n: u32) -> Self {
        self.context = self.context.with_n_ubatch(n);
        self
    }

    /// Enable embeddings collection.
    #[must_use]
    pub fn with_embeddings(mut self, yes: bool) -> Self {
        self.context = self.context.with_embeddings(yes);
        self
    }

    /// Configure the number of CPU threads.
    #[must_use]
    pub fn with_n_threads(mut self, n: i32) -> Self {
        self.context = self.context.with_n_threads(n);
        self
    }

    /// Configure the number of batch CPU threads.
    #[must_use]
    pub fn with_n_threads_batch(mut self, n: i32) -> Self {
        self.context = self.context.with_n_threads_batch(n);
        self
    }

    /// Enable or disable KQV cache offload to the active GPU backend.
    #[must_use]
    pub fn with_offload_kqv(mut self, yes: bool) -> Self {
        self.context = self.context.with_offload_kqv(yes);
        self
    }

    /// Enable or disable flash attention.
    #[must_use]
    pub fn with_flash_attn(mut self, yes: bool) -> Self {
        self.context = self.context.with_flash_attn(yes);
        self
    }

    /// Configure the pooling type (used by embedding models).
    #[must_use]
    pub fn with_pooling_type(mut self, p: crate::context::params::PoolingType) -> Self {
        self.context = self.context.with_pooling_type(p);
        self
    }

    /// Set the filename within a Hugging Face repo. Required when the repo has multiple .gguf files.
    ///
    /// As a convenience, if the current `model_path` already parses as a
    /// valid HF repo id (e.g. `"TheBloke/Llama-2-7B-Chat-GGUF"`), this
    /// builder also populates the internal `hf_repo_override` field so the
    /// resolver's precedence #1 (override wins) catches the Hf branch
    /// even when `model_path` is also a valid local file path. If the path
    /// is not a valid repo id, the override is left unset and the
    /// resolver's auto-detect branch decides.
    #[must_use]
    pub fn with_hf_filename(mut self, filename: impl Into<String>) -> Self {
        self.hf_filename = Some(filename.into());
        if let Some(s) = self.model_path.to_str() {
            if HfRepo::looks_like_repo_id(s) {
                // `looks_like_repo_id` and `new` share the same validator;
                // a successful `looks_like_repo_id` guarantees `new` is Ok.
                if let Ok(repo) = HfRepo::new(s) {
                    self.hf_repo_override = Some(repo);
                }
            }
        }
        self
    }
    /// Set the revision (branch, tag, or commit) of the Hugging Face repo. Defaults to "main".
    #[must_use]
    pub fn with_hf_revision(mut self, revision: impl Into<String>) -> Self {
        self.hf_revision = Some(revision.into());
        self
    }
    /// Set the Hugging Face access token (for gated/private repos). Equivalent to setting HF_TOKEN.
    #[must_use]
    pub fn with_hf_token(mut self, token: impl Into<String>) -> Self {
        self.hf_token = Some(token.into());
        self
    }
    /// Override the cache directory. Defaults to `~/.cache/huggingface/hub` (or $HF_HOME/hub).
    #[must_use]
    pub fn with_hf_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.hf_cache_dir = Some(dir.into());
        self
    }
    /// Set the Hugging Face endpoint (for mirrors like hf-mirror.com). Equivalent to setting HF_ENDPOINT.
    #[must_use]
    pub fn with_hf_endpoint(mut self, ep: impl Into<String>) -> Self {
        self.hf_endpoint = Some(ep.into());
        self
    }

    /// Apply a mobile-oriented preset. Call explicit setters after this method
    /// to override individual values.
    #[must_use]
    pub fn with_mobile_preset(self, preset: MobilePreset) -> Self {
        match preset {
            MobilePreset::LowRam => self
                .with_n_ctx(2048)
                .with_n_batch(128)
                .with_n_ubatch(128)
                .with_n_threads(4)
                .with_n_threads_batch(4)
                .with_n_gpu_layers(0)
                .with_flash_attn(false)
                .with_use_mmap(true),
            MobilePreset::Balanced => self
                .with_n_ctx(4096)
                .with_n_batch(512)
                .with_n_ubatch(256)
                .with_n_threads(4)
                .with_n_threads_batch(4)
                .with_n_gpu_layers(32)
                .with_flash_attn(true)
                .with_use_mmap(true),
            MobilePreset::GpuMax => self
                .with_n_ctx(4096)
                .with_n_batch(1024)
                .with_n_ubatch(512)
                .with_n_gpu_layers(99)
                .with_flash_attn(true)
                .with_offload_kqv(true)
                .with_use_mmap(true),
        }
    }
}

impl Default for LlamaParams {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            model: LlamaModelParams::default(),
            context: LlamaContextParams::default(),
            hf_filename: None,
            hf_revision: None,
            hf_token: None,
            hf_cache_dir: None,
            hf_endpoint: None,
            hf_repo_override: None,
            hf_downloader: None,
        }
    }
}

// StopReason is re-exported above for downstream users.
#[doc(inline)]
pub use StopReason as _StopReasonShim;

#[cfg(test)]
mod tests {
    use super::{HfDownloader, Llama, LlamaParams, MobilePreset};
    use crate::error::LlamaError;
    use crate::hf::downloader::MockHfDownloader;
    use std::path::PathBuf;
    use std::sync::Arc;

    #[test]
    fn mobile_preset_can_be_overridden() {
        let params = LlamaParams::new("model.gguf")
            .with_mobile_preset(MobilePreset::Balanced)
            .with_n_ctx(1024)
            .with_n_gpu_layers(0);

        assert_eq!(params.context.build().n_ctx, 1024);
        assert_eq!(params.model.n_gpu_layers(), 0);
    }

    #[test]
    fn mobile_preset_parse_accepts_cli_names() {
        assert_eq!("low-ram".parse(), Ok(MobilePreset::LowRam));
        assert_eq!("balanced".parse(), Ok(MobilePreset::Balanced));
        assert_eq!("gpu-max".parse(), Ok(MobilePreset::GpuMax));
        assert!("fast".parse::<MobilePreset>().is_err());
    }

    #[test]
    fn with_hf_filename_sets_field() {
        let p = LlamaParams::new("foo.gguf").with_hf_filename("model.Q4_K_M.gguf");
        assert_eq!(p.hf_filename.as_deref(), Some("model.Q4_K_M.gguf"));
    }

    #[test]
    fn with_hf_revision_sets_field() {
        let p = LlamaParams::new("foo.gguf").with_hf_revision("refs/pr/42");
        assert_eq!(p.hf_revision.as_deref(), Some("refs/pr/42"));
    }

    #[test]
    fn with_hf_token_sets_field() {
        let p = LlamaParams::new("foo.gguf").with_hf_token("hf_secret");
        assert_eq!(p.hf_token.as_deref(), Some("hf_secret"));
    }

    #[test]
    fn with_hf_cache_dir_sets_field() {
        let p =
            LlamaParams::new("foo.gguf").with_hf_cache_dir(std::path::PathBuf::from("/tmp/cache"));
        assert_eq!(
            p.hf_cache_dir.as_deref(),
            Some(std::path::Path::new("/tmp/cache"))
        );
    }

    #[test]
    fn with_hf_endpoint_sets_field() {
        let p = LlamaParams::new("foo.gguf").with_hf_endpoint("https://hf-mirror.com");
        assert_eq!(p.hf_endpoint.as_deref(), Some("https://hf-mirror.com"));
    }

    // ============================================================
    // Task 10: Llama::load HF resolution integration tests.
    //
    // All three tests use a `MockHfDownloader` injected via the
    // `hf_downloader` test slot on `LlamaParams`. The mock's
    // `with_next_error` arms a one-shot error so we can prove the
    // downloader was (or was not) invoked by inspecting the message of
    // the error that `Llama::load` returns.
    // ============================================================

    /// Sanity: writing 8 bytes of GGUF magic + length creates a file
    /// that the resolver can hand to `LlamaModel::load_from_file`.
    /// `load_from_file` will reject it (not a real GGUF), but the
    /// downloader is NOT called because the resolver short-circuits
    /// to the Local branch when the file exists on disk.
    #[test]
    fn load_with_existing_local_path_does_not_invoke_downloader() {
        // Pre-create a real file on disk; resolver must take Local branch.
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        std::fs::write(tmp.path(), b"GGUF\x00\x00\x00\x03not-a-real-gguf").expect("write blob");
        let local_path = tmp.path().to_path_buf();

        // Arm the downloader with a recognizable error. If the resolver
        // takes the Local branch, the downloader is never called and
        // the error from `Llama::load` comes from `load_from_file`
        // (or from context creation), NOT from the mock.
        let mock = MockHfDownloader::default()
            .with_next_error(LlamaError::ModelDownload("DOWNLOADER_INVOKED".into()));

        let mut params = LlamaParams::new(&local_path);
        params.hf_downloader = Some(Arc::new(mock));

        let err = Llama::load(params)
            .err()
            .expect("load must fail (blob is not a real GGUF)");

        let msg = format!("{err}");
        assert!(
            !msg.contains("DOWNLOADER_INVOKED"),
            "downloader must NOT have been invoked for an existing local file, got: {msg}"
        );
    }

    /// When the model_path looks like a HF repo id and does NOT exist
    /// on disk, the resolver takes the Hf branch and the injected
    /// mock downloader IS invoked. We arm the mock with
    /// `with_next_error` so the test fails loudly if the resolver
    /// fails to dispatch to the downloader.
    #[test]
    fn load_with_hf_repo_invokes_mock_downloader() {
        // Path that looks like a HF repo id and (we will assert) does
        // not exist on disk; otherwise the resolver would silently
        // take the Local branch.
        let model_path = PathBuf::from("TheBloke/LoadTestRepo");
        assert!(
            !model_path.exists(),
            "test fixture leaked: {} exists on disk",
            model_path.display()
        );

        let mock = MockHfDownloader::default()
            .with_next_error(LlamaError::ModelDownload("HF_DISPATCH_PROOF".into()));

        let mut params = LlamaParams::new(&model_path).with_hf_filename("foo.gguf");
        params.hf_downloader = Some(Arc::new(mock));

        let err = Llama::load(params)
            .err()
            .expect("load must surface the downloader error");

        let msg = format!("{err}");
        assert!(
            msg.contains("HF_DISPATCH_PROOF"),
            "resolver must have dispatched to the mock downloader, got: {msg}"
        );
    }

    /// Under `--no-default-features`, the default downloader is
    /// `DisabledHfDownloader`. Every dispatch returns a clear
    /// runtime error pointing the user at the `--features hf-hub`
    /// build flag. The `hf_repo_override` slot is auto-populated
    /// by `with_hf_filename` because the path parses as a valid
    /// `HfRepo` (`org/repo`).
    #[cfg(not(feature = "hf-hub"))]
    #[test]
    fn load_with_hf_repo_and_feature_off_returns_runtime_error() {
        let model_path = PathBuf::from("org/repo");
        assert!(
            !model_path.exists(),
            "test fixture leaked: {} exists on disk",
            model_path.display()
        );

        // No `hf_downloader` injection: `Llama::load` falls through
        // to `default_downloader()` -> `DisabledHfDownloader`.
        let params = LlamaParams::new(&model_path).with_hf_filename("foo.gguf");

        let err = Llama::load(params)
            .err()
            .expect("load must fail under --no-default-features");

        let msg = format!("{err}");
        assert!(
            msg.contains("hf-hub feature is disabled"),
            "error must point at the build flag, got: {msg}"
        );
    }
}
