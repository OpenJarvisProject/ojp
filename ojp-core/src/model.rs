//! A loaded model plus prompt caching.
//!
//! llama-crab's high-level completion clears the KV cache and re-reads the
//! whole prompt on every call. For the router that meant re-processing the
//! same system prompt and few-shot examples for every message, and for chat
//! the whole history every turn. Here the tokens already in the KV cache
//! are remembered, and each call only evaluates the part of the new prompt
//! that differs from them (longest common prefix).
use anyhow::{Context, Result, bail};
use llama_crab::{Llama, LlamaBatch, LlamaParams, LlamaSampler, LlamaToken};
use std::path::Path;

pub struct Model {
    llama: Llama,
    /// Tokens currently in the KV cache (sequence 0), in position order.
    cached: Vec<LlamaToken>,
    /// End-of-generation tokens: eos, eot, and the end markers of common
    /// chat formats that exist in this vocabulary as single tokens.
    eog: Vec<LlamaToken>,
    n_ctx: usize,
}

const END_MARKERS: [&str; 6] = ["<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "</s>"];

impl Model {
    pub fn load(path: &Path, n_ctx: u32, gpu_layers: i32) -> Result<Model> {
        if !path.exists() {
            bail!("{} not found", path.display());
        }
        let params = LlamaParams::new(path).with_n_ctx(n_ctx).with_n_gpu_layers(gpu_layers);
        let llama = Llama::load(params).with_context(|| format!("loading {}", path.display()))?;
        let vocab = llama.model();
        let mut eog = vec![vocab.token_eos(), vocab.token_eot()];
        for marker in END_MARKERS {
            if let Ok(tokens) = vocab.tokenize(marker, false, true)
                && let [token] = tokens[..]
                && vocab.detokenize(&[token], true).is_ok_and(|s| s == marker)
            {
                eog.push(token);
            }
        }
        Ok(Model { llama, cached: Vec::new(), eog, n_ctx: n_ctx as usize })
    }

    pub fn llama(&self) -> &Llama {
        &self.llama
    }

    /// Evaluate `prompt` (reusing the cached prefix), then sample up to
    /// `max_tokens` with `sampler`, streaming text through `on_text`.
    /// Returns the generated text.
    pub fn complete(
        &mut self,
        prompt: &str,
        sampler: &mut LlamaSampler,
        max_tokens: usize,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let tokens = self.llama.model().tokenize(prompt, true, true)?;
        if tokens.is_empty() {
            bail!("empty prompt");
        }
        if tokens.len() + max_tokens > self.n_ctx {
            bail!("prompt of {} tokens + {max_tokens} to generate exceeds the {}-token context", tokens.len(), self.n_ctx);
        }
        let result = self.generate(&tokens, sampler, max_tokens, on_text);
        if result.is_err() {
            // Unknown KV state after a failed decode: start clean next time.
            self.reset();
        }
        result
    }

    fn reset(&mut self) {
        let _ = self.llama.context().seq_rm(0, -1, -1);
        self.cached.clear();
    }

    fn generate(
        &mut self,
        tokens: &[LlamaToken],
        sampler: &mut LlamaSampler,
        max_tokens: usize,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<String> {
        // Reuse the common prefix, but always re-evaluate at least the last
        // prompt token: its logits are what the first sample is drawn from.
        let common = self.cached.iter().zip(tokens).take_while(|(a, b)| a == b).count();
        let keep = common.min(tokens.len() - 1);
        if self.llama.context().seq_rm(0, keep as i32, -1).is_err() {
            self.reset();
        }
        let keep = keep.min(self.cached.len());
        self.cached.truncate(keep);

        let n_batch = self.llama.context().n_batch() as usize;
        let mut logits_idx = 0;
        for start in (keep..tokens.len()).step_by(n_batch.max(1)) {
            let chunk = &tokens[start..(start + n_batch).min(tokens.len())];
            let mut batch = LlamaBatch::new(chunk.len(), 1);
            for (i, &token) in chunk.iter().enumerate() {
                let pos = start + i;
                batch.add(token, pos as i32, &[0], pos + 1 == tokens.len())?;
            }
            self.llama.context().decode(&batch)?;
            self.cached.extend_from_slice(chunk);
            logits_idx = chunk.len() as i32 - 1;
        }

        let mut text = String::new();
        // Tokens sampled but not yet turned into text. A multi-byte UTF-8
        // character can be split across tokens, and decoding those alone
        // gives replacement characters, so they wait here until the
        // character is complete. Decoding only this short run also stays
        // inside llama-crab's detokenize, which fails on output over 64
        // bytes (it treats llama.cpp's "buffer too small" as an error).
        let mut pending: Vec<LlamaToken> = Vec::new();
        for _ in 0..max_tokens {
            // SAFETY: the context handle is valid for the lifetime of
            // self.llama, and logits_idx indexes the last decoded batch.
            let token = unsafe { sampler.sample(self.llama.context().raw_handle(), logits_idx) };
            if self.eog.contains(&token) {
                break;
            }
            pending.push(token);
            let piece = self.llama.model().detokenize(&pending, false)?;
            if !piece.ends_with('\u{FFFD}') || pending.len() >= 8 {
                on_text(&piece);
                text.push_str(&piece);
                pending.clear();
            }

            let mut batch = LlamaBatch::new(1, 1);
            batch.add(token, self.cached.len() as i32, &[0], true)?;
            self.llama.context().decode(&batch)?;
            self.cached.push(token);
            logits_idx = 0;
        }
        if !pending.is_empty() {
            let piece = self.llama.model().detokenize(&pending, false)?;
            on_text(&piece);
            text.push_str(&piece);
        }
        Ok(text)
    }
}
