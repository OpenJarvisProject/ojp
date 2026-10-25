//! `LlamaModelParams` builder.

use llama_crab_sys as sys;

/// Strategy for splitting a model across multiple GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(dead_code)]
pub enum SplitMode {
    /// Single-GPU / no splitting.
    None,
    /// Split by layer (default).
    #[default]
    Layer,
    /// Split by row.
    Row,
    /// Split by tensor.
    Tensor,
}

impl From<SplitMode> for sys::llama_split_mode {
    fn from(s: SplitMode) -> Self {
        match s {
            SplitMode::None => Self::LLAMA_SPLIT_MODE_NONE,
            SplitMode::Layer => Self::LLAMA_SPLIT_MODE_LAYER,
            SplitMode::Row => Self::LLAMA_SPLIT_MODE_ROW,
            SplitMode::Tensor => Self::LLAMA_SPLIT_MODE_TENSOR,
        }
    }
}

/// Builder for `llama_model_params`.
///
/// `Default` produces the same parameters as `llama_model_default_params()`.
#[derive(Debug, Clone)]
pub struct LlamaModelParams {
    n_gpu_layers: i32,
    split_mode: SplitMode,
    main_gpu: i32,
    vocab_only: bool,
    use_mmap: bool,
    use_mlock: bool,
    /// OJP patch: let llama.cpp copy weights into CPU-friendly interleaved
    /// layouts (faster, but a second copy of those weights in memory).
    use_extra_bufts: bool,
}

impl Default for LlamaModelParams {
    fn default() -> Self {
        let raw = unsafe { sys::llama_model_default_params() };
        Self {
            n_gpu_layers: raw.n_gpu_layers,
            split_mode: match raw.split_mode {
                sys::llama_split_mode::LLAMA_SPLIT_MODE_NONE => SplitMode::None,
                sys::llama_split_mode::LLAMA_SPLIT_MODE_LAYER => SplitMode::Layer,
                sys::llama_split_mode::LLAMA_SPLIT_MODE_ROW => SplitMode::Row,
                sys::llama_split_mode::LLAMA_SPLIT_MODE_TENSOR => SplitMode::Tensor,
                #[allow(unreachable_patterns)]
                _ => SplitMode::Layer,
            },
            main_gpu: raw.main_gpu,
            vocab_only: raw.vocab_only,
            // OJP patch (newer llama.cpp): mmap/mlock became one `load_mode`.
            use_extra_bufts: raw.use_extra_bufts,
            use_mmap: !matches!(
                raw.load_mode,
                sys::llama_load_mode::LLAMA_LOAD_MODE_NONE | sys::llama_load_mode::LLAMA_LOAD_MODE_MLOCK
            ),
            use_mlock: matches!(
                raw.load_mode,
                sys::llama_load_mode::LLAMA_LOAD_MODE_MLOCK | sys::llama_load_mode::LLAMA_LOAD_MODE_MMAP_MLOCK
            ),
        }
    }
}

impl LlamaModelParams {
    /// Number of layers configured for GPU offload.
    #[must_use]
    pub const fn n_gpu_layers(&self) -> i32 {
        self.n_gpu_layers
    }

    /// Number of layers to offload to the GPU. `-1` offloads every layer.
    #[must_use]
    pub fn with_n_gpu_layers(mut self, n: i32) -> Self {
        self.n_gpu_layers = n;
        self
    }

    /// How to split a model across multiple GPUs.
    #[must_use]
    pub fn with_split_mode(mut self, mode: SplitMode) -> Self {
        self.split_mode = mode;
        self
    }

    /// Index of the main GPU.
    #[must_use]
    pub fn with_main_gpu(mut self, idx: i32) -> Self {
        self.main_gpu = idx;
        self
    }

    /// Load only the vocabulary (no model weights).
    #[must_use]
    pub fn with_vocab_only(mut self, yes: bool) -> Self {
        self.vocab_only = yes;
        self
    }

    /// OJP patch: allow repacked CPU weight copies (default on).
    #[must_use]
    pub fn with_use_extra_bufts(mut self, yes: bool) -> Self {
        self.use_extra_bufts = yes;
        self
    }

    /// Memory-map the model file.
    #[must_use]
    pub fn with_use_mmap(mut self, yes: bool) -> Self {
        self.use_mmap = yes;
        self
    }

    /// Lock the model into physical memory.
    #[must_use]
    pub fn with_use_mlock(mut self, yes: bool) -> Self {
        self.use_mlock = yes;
        self
    }

    /// Convert to the underlying C struct.
    #[allow(clippy::missing_const_for_fn)]
    pub(crate) fn build(&self) -> sys::llama_model_params {
        let mut raw = unsafe { sys::llama_model_default_params() };
        raw.n_gpu_layers = self.n_gpu_layers;
        raw.split_mode = self.split_mode.into();
        raw.main_gpu = self.main_gpu;
        raw.vocab_only = self.vocab_only;
        raw.use_extra_bufts = self.use_extra_bufts;
        raw.load_mode = match (self.use_mmap, self.use_mlock) {
            (false, false) => sys::llama_load_mode::LLAMA_LOAD_MODE_NONE,
            (true, false) => sys::llama_load_mode::LLAMA_LOAD_MODE_MMAP,
            (false, true) => sys::llama_load_mode::LLAMA_LOAD_MODE_MLOCK,
            (true, true) => sys::llama_load_mode::LLAMA_LOAD_MODE_MMAP_MLOCK,
        };
        raw
    }
}
