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

/// Independent KV-cache sequences in one model. Routing and chat prompts
/// differ from their first tokens, so on a router-only device (where the
/// router both classifies and answers) one shared cache was evicted on
/// every message; each kind of prompt now keeps its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seq {
    /// Routing (a short, mostly fixed prompt).
    Route = 0,
    /// Chat answers (system prompt + history).
    Chat = 1,
    /// Tool calls: a long fixed prompt (tool list and examples) that must not
    /// be evicted by the router's different prompt on every message.
    Tool = 2,
}
const SEQS: usize = 3;
const ALL_SEQS: [Seq; SEQS] = [Seq::Route, Seq::Chat, Seq::Tool];

pub struct Model {
    llama: Llama,
    /// Tokens currently in the KV cache per sequence, in position order.
    cached: [Vec<LlamaToken>; SEQS],
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
        // Each sequence gets its own n_ctx: llama.cpp splits a non-unified KV
        // cache evenly between the n_seq_max sequences.
        let mut params = LlamaParams::new(path).with_n_ctx(n_ctx * SEQS as u32).with_n_gpu_layers(gpu_layers);
        params.context = params.context.with_n_seq_max(SEQS as u32);
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
        Ok(Model { llama, cached: Default::default(), eog, n_ctx: n_ctx as usize })
    }

    pub fn llama(&self) -> &Llama {
        &self.llama
    }

    /// Save the KV cache (all sequences) and the token lists describing it,
    /// so a later start can skip re-evaluating the same warm-up prompts.
    pub fn save_cache(&mut self, path: &Path) -> Result<()> {
        // One state blob per sequence (llama_state_seq_*): the whole-context
        // state restores the two sequences' cells wrongly, which made the
        // first request after a restart misroute when it matched the warm-up.
        let tokens: Vec<Vec<i32>> = self.cached.iter().map(|seq| seq.iter().map(|t| t.raw()).collect()).collect();
        let header = serde_json::to_vec(&tokens)?;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        let ctx = self.llama.context().raw_handle();
        for seq in ALL_SEQS {
            // SAFETY: ctx is valid for the call; the buffer has the size llama.cpp asked for.
            let blob = unsafe {
                let size = llama_crab_sys::llama_state_seq_get_size(ctx, seq as i32);
                let mut buf = vec![0_u8; size];
                let written = llama_crab_sys::llama_state_seq_get_data(ctx, buf.as_mut_ptr(), size, seq as i32);
                buf.truncate(written);
                buf
            };
            out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
            out.extend_from_slice(&blob);
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, out)?;
        std::fs::rename(tmp, path)?;
        Ok(())
    }

    /// Restore a cache written by `save_cache` for this same model/prompts
    /// (the caller keys the file name). On any error the model is left with
    /// an empty cache, which is always correct, just slower.
    pub fn load_cache(&mut self, path: &Path) -> Result<()> {
        let result = (|| -> Result<()> {
            let bytes = std::fs::read(path)?;
            let read_len = |at: usize| -> Result<usize> {
                Ok(u64::from_le_bytes(bytes.get(at..at + 8).context("short cache file")?.try_into()?) as usize)
            };
            let len = read_len(0)?;
            let header = bytes.get(8..8 + len).context("short cache file")?;
            let tokens: Vec<Vec<i32>> = serde_json::from_slice(header)?;
            if tokens.len() != SEQS {
                bail!("cache has {} sequences, expected {SEQS}", tokens.len());
            }
            let ctx = self.llama.context().raw_handle();
            let mut at = 8 + len;
            for seq in ALL_SEQS {
                let size = read_len(at)?;
                let blob = bytes.get(at + 8..at + 8 + size).context("short cache file")?;
                at += 8 + size;
                let _ = self.llama.context().seq_rm(seq as i32, -1, -1);
                // SAFETY: ctx is valid and blob is a slice of the stated length.
                let loaded = unsafe { llama_crab_sys::llama_state_seq_set_data(ctx, blob.as_ptr(), blob.len(), seq as i32) };
                if loaded == 0 && !tokens[seq as usize].is_empty() {
                    bail!("llama.cpp rejected the saved state");
                }
            }
            for (seq, toks) in self.cached.iter_mut().zip(tokens) {
                *seq = toks.into_iter().map(LlamaToken::from).collect();
            }
            Ok(())
        })();
        if result.is_err() {
            for seq in ALL_SEQS {
                self.reset(seq);
            }
        }
        result
    }

    /// Evaluate `prompt` (reusing the cached prefix), then sample up to
    /// `max_tokens` with `sampler`, streaming text through `on_text`.
    /// Returns the generated text.
    pub fn complete(
        &mut self,
        seq: Seq,
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
        let result = self.generate(seq, &tokens, sampler, max_tokens, on_text);
        if result.is_err() {
            // Unknown KV state after a failed decode: start clean next time.
            self.reset(seq);
        }
        result
    }

    fn reset(&mut self, seq: Seq) {
        let _ = self.llama.context().seq_rm(seq as i32, -1, -1);
        self.cached[seq as usize].clear();
    }

    fn generate(
        &mut self,
        seq: Seq,
        tokens: &[LlamaToken],
        sampler: &mut LlamaSampler,
        max_tokens: usize,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<String> {
        // Reuse the common prefix, but always re-evaluate at least the last
        // prompt token: its logits are what the first sample is drawn from.
        let id = seq as i32;
        let common = self.cached[seq as usize].iter().zip(tokens).take_while(|(a, b)| a == b).count();
        let keep = common.min(tokens.len() - 1);
        if self.llama.context().seq_rm(id, keep as i32, -1).is_err() {
            self.reset(seq);
        }
        let keep = keep.min(self.cached[seq as usize].len());
        self.cached[seq as usize].truncate(keep);

        let n_batch = self.llama.context().n_batch() as usize;
        let mut logits_idx = 0;
        for start in (keep..tokens.len()).step_by(n_batch.max(1)) {
            let chunk = &tokens[start..(start + n_batch).min(tokens.len())];
            // Capacity 1 sequence id per token: llama-crab marks every token as
            // belonging to that many sequences, so a larger capacity would tie
            // each token to sequence 0 too ("coupled sequences" in llama.cpp).
            let mut batch = LlamaBatch::new(chunk.len(), 1);
            for (i, &token) in chunk.iter().enumerate() {
                let pos = start + i;
                batch.add(token, pos as i32, &[id], pos + 1 == tokens.len())?;
            }
            self.llama.context().decode(&batch)?;
            self.cached[seq as usize].extend_from_slice(chunk);
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
            batch.add(token, self.cached[seq as usize].len() as i32, &[id], true)?;
            self.llama.context().decode(&batch)?;
            self.cached[seq as usize].push(token);
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
