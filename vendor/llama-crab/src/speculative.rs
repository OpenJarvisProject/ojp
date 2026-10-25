//! Speculative decoding: a draft model proposes candidate tokens, the
//! main model validates them in a single forward pass.
//!
//! Two flavours are exposed:
//!
//! * [`PromptLookupDecoding`] — pure prompt-lookup n-gram search, no
//!   extra model required.
//! * The trait [`DraftModel`] — implement for any custom draft strategy
//!   (a smaller model, a hand-crafted finite-state generator, etc.).
//!
//! Use [`speculative_decode`] to drive a single speculative step.

/// A draft model that proposes candidate tokens for speculative decoding.
pub trait DraftModel {
    /// Produce up to `n` candidate tokens for the current input.
    fn draft(&self, input: &[crate::token::LlamaToken], n: usize) -> Vec<crate::token::LlamaToken>;
}

/// N-gram prompt-lookup speculative decoding.
///
/// Repeatedly scans the prompt for the longest matching n-gram and emits
/// the tokens that followed it as the draft.
#[derive(Debug, Clone)]
pub struct PromptLookupDecoding {
    max_ngram_size: usize,
    num_pred_tokens: usize,
}

impl PromptLookupDecoding {
    /// Construct a new prompt-lookup draft model.
    ///
    /// * `max_ngram_size` — maximum length of the n-gram to look up.
    /// * `num_pred_tokens` — how many tokens to emit per draft.
    #[must_use]
    pub const fn new(max_ngram_size: usize, num_pred_tokens: usize) -> Self {
        Self {
            max_ngram_size,
            num_pred_tokens,
        }
    }
}

impl Default for PromptLookupDecoding {
    fn default() -> Self {
        Self::new(2, 10)
    }
}

impl DraftModel for PromptLookupDecoding {
    fn draft(&self, input: &[crate::token::LlamaToken], n: usize) -> Vec<crate::token::LlamaToken> {
        if input.len() < self.max_ngram_size + 1 || n == 0 {
            return Vec::new();
        }
        for ngram in (1..=self.max_ngram_size.min(input.len() - 1)).rev() {
            let start = input.len() - ngram;
            let pat = &input[start..];
            for candidate_start in 0..=input.len().saturating_sub(ngram + 1) {
                if &input[candidate_start..candidate_start + ngram] == pat {
                    let mut out = Vec::with_capacity(n.min(self.num_pred_tokens));
                    for j in 0..self.num_pred_tokens.min(n) {
                        if let Some(&t) = input.get(candidate_start + ngram + j) {
                            out.push(t);
                        } else {
                            break;
                        }
                    }
                    if !out.is_empty() {
                        return out;
                    }
                }
            }
        }
        Vec::new()
    }
}

/// Run one step of speculative decoding using a draft model and a main
/// sampler chain.
///
/// Returns the number of accepted tokens (and appends them to `output`).
///
/// # Algorithm
///
/// 1. Draft `n_draft` tokens with `draft`.
/// 2. Feed the draft through the main model's context (one decode pass).
/// 3. Sample each draft position with `main_sampler`; accept a prefix
///    of the draft matching the sampled tokens.
/// 4. The first *rejected* draft token is replaced by the sampled one,
///    and decoding continues from there.
///
/// # Safety
/// `main_ctx` must point to a live, unaliased `llama_context` owned
/// exclusively by the caller for the duration of this call.
pub unsafe fn speculative_decode<M: DraftModel>(
    main_ctx: *mut llama_crab_sys::llama_context,
    main_sampler: &mut crate::sampling::LlamaSampler,
    draft_model: &M,
    history: &[crate::token::LlamaToken],
    n_draft: usize,
) -> Vec<crate::token::LlamaToken> {
    use crate::token::LlamaToken;
    if n_draft == 0 {
        return Vec::new();
    }
    let draft_tokens = draft_model.draft(history, n_draft);
    if draft_tokens.is_empty() {
        return Vec::new();
    }

    // Build a batch with the draft tokens (logits on the last one).
    let mut batch = crate::batch::LlamaBatch::new(draft_tokens.len(), 1);
    for (i, &t) in draft_tokens.iter().enumerate() {
        let logits = i + 1 == draft_tokens.len();
        let _ = batch.add(t, (history.len() + i) as i32, &[0], logits);
    }
    // Safety: caller guarantees the context is live and unaliased.
    if unsafe { llama_crab_sys::llama_decode(main_ctx, *batch.raw()) } != 0 {
        return Vec::new();
    }

    // Sample and accept prefix.
    let mut accepted = Vec::with_capacity(draft_tokens.len());
    for (i, &draft_tok) in draft_tokens.iter().enumerate() {
        let sampled = unsafe { main_sampler.sample(main_ctx, i as i32) };
        main_sampler.accept(sampled);
        if sampled == draft_tok {
            accepted.push(draft_tok);
        } else {
            accepted.push(sampled);
            break;
        }
    }
    accepted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::LlamaToken;

    #[test]
    fn prompt_lookup_finds_match() {
        let d = PromptLookupDecoding::new(2, 5);
        // The last 2 tokens are (3, 4). An earlier occurrence is at
        // index 0; the tokens that follow are 1, 2, 3, 4.
        let input: Vec<LlamaToken> = vec![
            LlamaToken(3),
            LlamaToken(4),
            LlamaToken(5),
            LlamaToken(6),
            LlamaToken(3),
            LlamaToken(4),
        ];
        let out = d.draft(&input, 4);
        assert_eq!(
            out,
            vec![LlamaToken(5), LlamaToken(6), LlamaToken(3), LlamaToken(4)]
        );
    }

    #[test]
    fn prompt_lookup_no_match() {
        let d = PromptLookupDecoding::new(2, 4);
        let input: Vec<LlamaToken> = vec![
            LlamaToken(0),
            LlamaToken(1),
            LlamaToken(2),
            LlamaToken(3),
            LlamaToken(4),
        ];
        let out = d.draft(&input, 4);
        assert!(out.is_empty());
    }
}
