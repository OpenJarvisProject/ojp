//! Tokenizer abstractions and FIM (Fill-in-Middle) support.
//!
//! The native llama.cpp tokenizer is always available. Hugging Face tokenizer
//! support is available through the `hf-tokenizer` cargo feature.
//!
//! The [`Tokenizer`] trait gives downstream code a uniform API; the
//! provided [`LlamaTokenizer`] delegates to the model that is loaded.

use crate::error::{LlamaError, Result};
use crate::model::LlamaModel;
use crate::token::LlamaToken;

/// A pluggable tokenizer front-end.
pub trait Tokenizer {
    /// Encode `text` into a sequence of token ids.
    ///
    /// `add_bos` controls whether the BOS token is prepended (the
    /// model's tokenizer may add it automatically even when `false`).
    fn encode(&self, text: &str, add_bos: bool, special: bool) -> Result<Vec<LlamaToken>>;

    /// Decode token ids back into a `String`.
    fn decode(&self, tokens: &[LlamaToken], special: bool) -> Result<String>;
}

/// Tokenizer that delegates to a loaded [`LlamaModel`].
#[derive(Debug)]
pub struct LlamaTokenizer<'a> {
    model: &'a LlamaModel,
}

impl<'a> LlamaTokenizer<'a> {
    /// Construct a new tokenizer bound to a model.
    #[must_use]
    pub const fn new(model: &'a LlamaModel) -> Self {
        Self { model }
    }
}

impl<'a> Tokenizer for LlamaTokenizer<'a> {
    fn encode(&self, text: &str, add_bos: bool, special: bool) -> Result<Vec<LlamaToken>> {
        self.model.tokenize(text, add_bos, special)
    }

    fn decode(&self, tokens: &[LlamaToken], special: bool) -> Result<String> {
        self.model.detokenize(tokens, special)
    }
}

/// FIM (Fill-in-Middle) special tokens returned by
/// [`LlamaModel::fim_tokens`].
///
/// Many code-completion models (Code Llama, DeepSeek Coder, Qwen2.5-Coder,
/// StarCoder) use a 3-segment FIM format:
///
/// ```text
/// <PRE> {prefix} <SUF> {suffix} <MID> {completion}
/// ```
///
/// The model exposes the special token ids for those markers through
/// llama.cpp's vocabulary API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FimTokens {
    /// Token used to introduce the prefix.
    pub prefix: LlamaToken,
    /// Token used to introduce the suffix.
    pub suffix: LlamaToken,
    /// Token used to introduce the middle (completion) segment.
    pub middle: LlamaToken,
    /// Optional EOT token (some models emit it instead of EOS).
    pub eot: Option<LlamaToken>,
}

impl FimTokens {
    /// True if the prefix token is non-sentinel (i.e. this model
    /// supports FIM).
    #[must_use]
    pub fn is_supported(&self) -> bool {
        self.prefix.0 >= 0 && self.suffix.0 >= 0 && self.middle.0 >= 0
    }

    /// Build a FIM prompt `prefix <SUF> suffix <MID>` ready to be
    /// fed into a completion.
    ///
    /// Returns the prompt **string**; use
    /// [`LlamaTokenizer::encode`] to tokenize.
    pub fn build_prompt(&self, prefix: &str, suffix: &str) -> Result<String> {
        if !self.is_supported() {
            return Err(LlamaError::Batch("model does not support FIM".into()));
        }
        // We use a simple text representation; downstream callers should
        // tokenize with `parse_special=true` so llama.cpp recognises the
        // FIM markers.
        Ok(format!("{prefix} <FIM_SUF> {suffix} <FIM_MID>"))
    }
}

impl LlamaModel {
    /// Return the FIM special tokens for this model.
    ///
    /// Returns `None` if the model does not expose FIM tokens
    /// (the IDs will be -1).
    #[must_use]
    pub fn fim_tokens(&self) -> Option<FimTokens> {
        use llama_crab_sys as sys;
        let pre = unsafe { sys::llama_token_fim_pre(self.vocab()) };
        let suf = unsafe { sys::llama_token_fim_suf(self.vocab()) };
        let mid = unsafe { sys::llama_token_fim_mid(self.vocab()) };
        let eot_raw = unsafe { sys::llama_token_eot(self.vocab()) };
        if pre < 0 || suf < 0 || mid < 0 {
            return None;
        }
        Some(FimTokens {
            prefix: LlamaToken(pre),
            suffix: LlamaToken(suf),
            middle: LlamaToken(mid),
            eot: if eot_raw >= 0 {
                Some(LlamaToken(eot_raw))
            } else {
                None
            },
        })
    }

    /// Convenience: build a FIM prompt for code infill.
    ///
    /// # Example
    /// ```no_run
    /// # use llama_crab::Llama;
    /// # let llama = Llama::load(Default::default()).unwrap();
    /// if let Some(fim) = llama.model().fim_tokens() {
    ///     let p = fim.build_prompt("fn main() {", "}").unwrap();
    ///     // tokenize with `parse_special = true`
    /// }
    /// ```
    pub fn fim_prompt(&self, prefix: &str, suffix: &str) -> Result<String> {
        match self.fim_tokens() {
            Some(t) => t.build_prompt(prefix, suffix),
            None => Err(LlamaError::Batch("model does not support FIM".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fim_tokens_unsupported() {
        // When the binding is not available, fim_tokens() returns None.
        // We only test the contract of the helper.
        assert!(!FimTokens {
            prefix: LlamaToken(-1),
            suffix: LlamaToken(-1),
            middle: LlamaToken(-1),
            eot: None,
        }
        .is_supported());
        assert!(FimTokens {
            prefix: LlamaToken(100),
            suffix: LlamaToken(101),
            middle: LlamaToken(102),
            eot: None,
        }
        .is_supported());
    }

    #[test]
    fn fim_build_prompt_unsupported() {
        let t = FimTokens {
            prefix: LlamaToken(-1),
            suffix: LlamaToken(-1),
            middle: LlamaToken(-1),
            eot: None,
        };
        assert!(t.build_prompt("a", "b").is_err());
    }
}
