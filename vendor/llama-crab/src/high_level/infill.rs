//! Code infill (Fill-in-Middle) helper.
//!
//! Takes a prefix and a suffix, asks the model to fill in the middle,
//! and returns the generated completion (without the surrounding
//! prefix/suffix markers).
//!
//! Requires the model to support FIM special tokens — see
//! [`crate::FimTokens`].

use crate::batch::LlamaBatch;
use crate::error::Result;
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;
use crate::Llama;

impl Llama {
    /// Perform FIM-style code infill.
    ///
    /// # Example
    /// ```no_run
    /// # use llama_crab::Llama;
    /// # let mut llama = Llama::load(Default::default()).unwrap();
    /// let fill = llama
    ///     .complete_infill("fn main() {", "}")
    ///     .unwrap();
    /// # let _ = fill;
    /// ```
    ///
    /// # Errors
    /// Returns an error if the model does not support FIM, the
    /// construction of the prompt fails, or the decode loop hits an
    /// unrecoverable sampler error.
    pub fn complete_infill(&mut self, prefix: &str, suffix: &str) -> Result<String> {
        let fim = self
            .model()
            .fim_tokens()
            .ok_or_else(|| crate::error::LlamaError::Batch("model does not support FIM".into()))?;
        // Clear the KV cache for sequence 0 so we start from position 0
        // regardless of any previous decode (mirrors `create_completion`).
        let _ = self.context_mut().seq_rm(0, -1, -1);
        let prompt = fim.build_prompt(prefix, suffix)?;
        let tokens = self.model().tokenize(&prompt, true, false)?;
        if tokens.is_empty() {
            return Ok(String::new());
        }
        // Decode the full prompt in one shot; only the last token has
        // `logits = true`.
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, &t) in tokens.iter().enumerate() {
            let logits = i + 1 == tokens.len();
            batch
                .add(t, i as i32, &[0], logits)
                .map_err(crate::error::LlamaError::from)?;
        }
        self.context_mut().decode(&batch)?;
        // Greedy sample, up to 256 tokens or until EOS/EOT.
        let mut sampler = LlamaSampler::greedy()
            .ok_or_else(|| crate::error::LlamaError::Batch("greedy sampler init failed".into()))?;
        let ctx_ptr = self.context().raw_handle();
        let eos = self.model().token_eos();
        let eot = fim.eot.unwrap_or(eos);
        let mut out = String::new();
        let mut n_generated = 0_usize;
        for _ in 0..256 {
            // Same convention as `create_completion`: the initial batch has
            // logits at index `n_tokens - 1`; every subsequent 1-token batch
            // has logits at index 0.
            let idx = if n_generated == 0 {
                (tokens.len() as i32) - 1
            } else {
                0
            };
            let tok: LlamaToken = unsafe { sampler.sample(ctx_ptr, idx) };
            sampler.accept(tok);
            if tok == eos || tok == eot {
                break;
            }
            if let Ok(piece) = self.model().detokenize(&[tok], false) {
                out.push_str(&piece);
            }
            n_generated += 1;
            // Feed back the new token.
            let mut single = LlamaBatch::new(1, 1);
            single
                .add(
                    tok,
                    tokens.len() as i32 + n_generated as i32 - 1,
                    &[0],
                    true,
                )
                .map_err(crate::error::LlamaError::from)?;
            self.context_mut().decode(&single)?;
        }
        Ok(out.trim().to_string())
    }
}

// Small shim to keep `self.context()` available.
impl Llama {
    /// Borrow the context mutably (used internally by helpers).
    pub(crate) fn context_mut(&mut self) -> &mut crate::context::LlamaContext {
        &mut self.context
    }
}
