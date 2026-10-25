//! `LlamaContextParams` builder.

use llama_crab_sys as sys;

/// Pooling type used when the context is initialized for embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PoolingType {
    /// Unspecified — use the model's default.
    #[default]
    Unspecified,
    /// No pooling.
    None,
    /// Mean pooling.
    Mean,
    /// CLS pooling.
    Cls,
    /// Last token.
    Last,
    /// Rank pooling (reranking models).
    Rank,
}

impl From<PoolingType> for sys::llama_pooling_type {
    fn from(p: PoolingType) -> Self {
        match p {
            PoolingType::Unspecified => Self::LLAMA_POOLING_TYPE_UNSPECIFIED,
            PoolingType::None => Self::LLAMA_POOLING_TYPE_NONE,
            PoolingType::Mean => Self::LLAMA_POOLING_TYPE_MEAN,
            PoolingType::Cls => Self::LLAMA_POOLING_TYPE_CLS,
            PoolingType::Last => Self::LLAMA_POOLING_TYPE_LAST,
            PoolingType::Rank => Self::LLAMA_POOLING_TYPE_RANK,
        }
    }
}

/// Attention type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AttentionType {
    /// Unspecified — use the model's default.
    #[default]
    Unspecified,
    /// Standard causal attention.
    Causal,
    /// Non-causal (used for encoder/embedding models).
    NonCausal,
}

impl From<AttentionType> for sys::llama_attention_type {
    fn from(a: AttentionType) -> Self {
        match a {
            AttentionType::Unspecified => Self::LLAMA_ATTENTION_TYPE_UNSPECIFIED,
            AttentionType::Causal => Self::LLAMA_ATTENTION_TYPE_CAUSAL,
            AttentionType::NonCausal => Self::LLAMA_ATTENTION_TYPE_NON_CAUSAL,
        }
    }
}

/// Builder for `llama_context_params`.
#[derive(Debug, Clone)]
pub struct LlamaContextParams {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_threads: i32,
    n_threads_batch: i32,
    pooling_type: PoolingType,
    attention_type: AttentionType,
    embeddings: bool,
    offload_kqv: bool,
    flash_attn: bool,
    no_perf: bool,
}

impl Default for LlamaContextParams {
    fn default() -> Self {
        let raw = unsafe { sys::llama_context_default_params() };
        Self {
            n_ctx: raw.n_ctx,
            n_batch: raw.n_batch,
            n_ubatch: raw.n_ubatch,
            n_seq_max: raw.n_seq_max,
            n_threads: raw.n_threads,
            n_threads_batch: raw.n_threads_batch,
            pooling_type: PoolingType::Unspecified,
            attention_type: AttentionType::Unspecified,
            embeddings: raw.embeddings,
            offload_kqv: raw.offload_kqv,
            flash_attn: matches!(
                raw.flash_attn_type,
                sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED
            ),
            no_perf: raw.no_perf,
        }
    }
}

impl LlamaContextParams {
    /// Context size (number of tokens).
    #[must_use]
    pub fn with_n_ctx(mut self, n: u32) -> Self {
        self.n_ctx = n;
        self
    }

    /// Logical maximum batch size.
    #[must_use]
    pub fn with_n_batch(mut self, n: u32) -> Self {
        self.n_batch = n;
        self
    }

    /// Physical batch size (forward pass).
    #[must_use]
    pub fn with_n_ubatch(mut self, n: u32) -> Self {
        self.n_ubatch = n;
        self
    }

    /// Maximum number of parallel sequences.
    #[must_use]
    pub fn with_n_seq_max(mut self, n: u32) -> Self {
        self.n_seq_max = n;
        self
    }

    /// Number of threads used for evaluation.
    #[must_use]
    pub fn with_n_threads(mut self, n: i32) -> Self {
        self.n_threads = n;
        self
    }

    /// Number of threads used for batch evaluation.
    #[must_use]
    pub fn with_n_threads_batch(mut self, n: i32) -> Self {
        self.n_threads_batch = n;
        self
    }

    /// Enable embeddings collection.
    #[must_use]
    pub fn with_embeddings(mut self, yes: bool) -> Self {
        self.embeddings = yes;
        self
    }

    /// Offload the KQV cache to GPU.
    #[must_use]
    pub fn with_offload_kqv(mut self, yes: bool) -> Self {
        self.offload_kqv = yes;
        self
    }

    /// Enable flash attention.
    #[must_use]
    pub fn with_flash_attn(mut self, yes: bool) -> Self {
        self.flash_attn = yes;
        self
    }

    /// Disable internal perf counters.
    #[must_use]
    pub fn with_no_perf(mut self, yes: bool) -> Self {
        self.no_perf = yes;
        self
    }

    /// Get the current pooling type.
    #[must_use]
    pub const fn pooling_type(&self) -> PoolingType {
        self.pooling_type
    }

    /// Set the pooling type.
    #[must_use]
    pub const fn with_pooling_type(mut self, p: PoolingType) -> Self {
        self.pooling_type = p;
        self
    }

    /// Get the current attention type.
    #[must_use]
    pub const fn attention_type(&self) -> AttentionType {
        self.attention_type
    }

    /// Set the attention type.
    #[must_use]
    pub const fn with_attention_type(mut self, a: AttentionType) -> Self {
        self.attention_type = a;
        self
    }

    #[allow(clippy::missing_const_for_fn)]
    pub(crate) fn build(&self) -> sys::llama_context_params {
        let mut raw = unsafe { sys::llama_context_default_params() };
        raw.n_ctx = self.n_ctx;
        raw.n_batch = self.n_batch;
        raw.n_ubatch = self.n_ubatch;
        raw.n_seq_max = self.n_seq_max;
        raw.n_threads = self.n_threads;
        raw.n_threads_batch = self.n_threads_batch;
        raw.pooling_type = self.pooling_type.into();
        raw.attention_type = self.attention_type.into();
        raw.embeddings = self.embeddings;
        raw.offload_kqv = self.offload_kqv;
        raw.flash_attn_type = if self.flash_attn {
            sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED
        } else {
            sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_DISABLED
        };
        raw.no_perf = self.no_perf;
        raw
    }
}
