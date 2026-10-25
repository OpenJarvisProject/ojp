//! Global backend initialization, NUMA strategy, device enumeration.

use std::sync::atomic::{AtomicBool, Ordering};

use llama_crab_sys as sys;

use crate::error::{LlamaError, Result};

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Opaque token proving that the llama.cpp backend was initialized.
#[derive(Debug)]
pub struct LlamaBackend {
    _private: (),
}

impl LlamaBackend {
    /// Initialize the global backend. Calling this more than once is a no-op
    /// (the second call still returns `Ok`).
    ///
    /// # Errors
    /// Returns an error only if llama.cpp rejected the initialization.
    pub fn init() -> Result<Self> {
        // Safety: `llama_backend_init` is safe to call once globally.
        unsafe {
            sys::llama_backend_init();
        }
        INITIALIZED.store(true, Ordering::SeqCst);
        Ok(Self { _private: () })
    }

    /// Initialize the backend with a specific NUMA strategy.
    pub fn init_numa(strategy: NumaStrategy) -> Result<Self> {
        let raw: sys::ggml_numa_strategy = strategy.into();
        // Safety: ggml_numa_init is safe to call once globally.
        unsafe {
            sys::llama_numa_init(raw);
        }
        INITIALIZED.store(true, Ordering::SeqCst);
        Ok(Self { _private: () })
    }

    /// Returns `true` if the GPU can offload work.
    #[must_use]
    pub fn supports_gpu_offload() -> bool {
        unsafe { sys::llama_supports_gpu_offload() }
    }

    /// Returns `true` if the platform supports `mmap` of model files.
    #[must_use]
    pub fn supports_mmap() -> bool {
        unsafe { sys::llama_supports_mmap() }
    }

    /// Returns `true` if the platform supports `mlock` of model files.
    #[must_use]
    pub fn supports_mlock() -> bool {
        unsafe { sys::llama_supports_mlock() }
    }

    /// Returns `true` if llama.cpp supports RPC (distributed inference).
    #[must_use]
    pub fn supports_rpc() -> bool {
        unsafe { sys::llama_supports_rpc() }
    }
}

impl Drop for LlamaBackend {
    fn drop(&mut self) {
        // Safety: `llama_backend_free` is safe to call once.
        unsafe {
            sys::llama_backend_free();
        }
        INITIALIZED.store(false, Ordering::SeqCst);
    }
}

/// NUMA allocation strategy, passed to [`LlamaBackend::init_numa`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumaStrategy {
    /// Disable NUMA optimizations.
    Disabled,
    /// Distribute model layers across NUMA nodes.
    Distribute,
    /// Run on a specific NUMA node.
    Isolate,
    /// Use the operating system's NUMA policy.
    NumaCtl,
    /// Mirror buffers across nodes.
    Mirror,
    /// Count (internal).
    Count,
}

impl From<NumaStrategy> for sys::ggml_numa_strategy {
    fn from(s: NumaStrategy) -> Self {
        match s {
            NumaStrategy::Disabled => Self::GGML_NUMA_STRATEGY_DISABLED,
            NumaStrategy::Distribute => Self::GGML_NUMA_STRATEGY_DISTRIBUTE,
            NumaStrategy::Isolate => Self::GGML_NUMA_STRATEGY_ISOLATE,
            NumaStrategy::NumaCtl => Self::GGML_NUMA_STRATEGY_NUMACTL,
            NumaStrategy::Mirror => Self::GGML_NUMA_STRATEGY_MIRROR,
            NumaStrategy::Count => Self::GGML_NUMA_STRATEGY_COUNT,
        }
    }
}

impl TryFrom<sys::ggml_numa_strategy> for NumaStrategy {
    type Error = LlamaError;
    fn try_from(v: sys::ggml_numa_strategy) -> Result<Self> {
        Ok(match v {
            sys::ggml_numa_strategy::GGML_NUMA_STRATEGY_DISABLED => Self::Disabled,
            sys::ggml_numa_strategy::GGML_NUMA_STRATEGY_DISTRIBUTE => Self::Distribute,
            sys::ggml_numa_strategy::GGML_NUMA_STRATEGY_ISOLATE => Self::Isolate,
            sys::ggml_numa_strategy::GGML_NUMA_STRATEGY_NUMACTL => Self::NumaCtl,
            sys::ggml_numa_strategy::GGML_NUMA_STRATEGY_MIRROR => Self::Mirror,
            sys::ggml_numa_strategy::GGML_NUMA_STRATEGY_COUNT => Self::Count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numa_roundtrip() {
        for s in [
            NumaStrategy::Disabled,
            NumaStrategy::Distribute,
            NumaStrategy::Isolate,
            NumaStrategy::NumaCtl,
            NumaStrategy::Mirror,
            NumaStrategy::Count,
        ] {
            let raw: sys::ggml_numa_strategy = s.into();
            let back = NumaStrategy::try_from(raw).unwrap();
            assert_eq!(s, back);
        }
    }
}
