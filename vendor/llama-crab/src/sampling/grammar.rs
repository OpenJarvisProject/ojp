//! GBNF-constrained sampling and error type.
//!
//! Only available with the `common` cargo feature because the upstream
//! `common::json_schema_to_grammar` library must be linked.

use llama_crab_sys as sys;

use crate::model::LlamaModel;

use super::LlamaSampler;
#[allow(unused_imports)]
use crate::token::LlamaToken;

/// Errors that can arise while building a GBNF grammar sampler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarError {
    /// A grammar trigger word contained an interior NUL byte.
    TriggerWordNullBytes,
    /// The grammar string itself contained a NUL byte.
    GrammarNullBytes,
    /// The model produced a null sampler (out of memory or invalid grammar).
    NullGrammar,
    /// llama.cpp rejected the grammar (parse error, conflicting rules, …).
    RootNotFound,
    /// The FFI call failed with a non-zero status code.
    Ffi(i32),
}

impl std::fmt::Display for GrammarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TriggerWordNullBytes => f.write_str("trigger word contains a NUL byte"),
            Self::GrammarNullBytes => f.write_str("grammar string contains a NUL byte"),
            Self::NullGrammar => f.write_str("llama_sampler_init_grammar returned null"),
            Self::RootNotFound => f.write_str("grammar root rule not found"),
            Self::Ffi(c) => write!(f, "grammar FFI failed (code {c})"),
        }
    }
}

impl std::error::Error for GrammarError {}

impl From<std::ffi::NulError> for GrammarError {
    fn from(_: std::ffi::NulError) -> Self {
        Self::TriggerWordNullBytes
    }
}

impl LlamaSampler {
    /// GBNF grammar constrained sampler.
    ///
    /// # Safety
    /// The `grammar` and `grammar_root` C strings must outlive the sampler.
    #[cfg(feature = "common")]
    pub unsafe fn grammar(
        model: &LlamaModel,
        grammar: &str,
        grammar_root: &str,
    ) -> Result<Self, GrammarError> {
        let grammar =
            std::ffi::CString::new(grammar).map_err(|_| GrammarError::GrammarNullBytes)?;
        let root =
            std::ffi::CString::new(grammar_root).map_err(|_| GrammarError::GrammarNullBytes)?;
        let p = unsafe {
            sys::llama_sampler_init_grammar(model.vocab(), grammar.as_ptr(), root.as_ptr())
        };
        if p.is_null() {
            return Err(GrammarError::NullGrammar);
        }
        Ok(unsafe { Self::from_raw(p) })
    }
}
