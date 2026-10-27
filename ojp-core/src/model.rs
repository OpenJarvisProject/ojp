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

/// CPU threads for evaluation; 0 = pick automatically (see `auto_threads`).
static THREADS: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

pub fn set_threads(n: i32) {
    THREADS.store(n, std::sync::atomic::Ordering::Relaxed);
}

/// Physical cores on x86 (hyper-threads share a core's memory bandwidth, and 12 threads on 6 cores
/// measured 15x slower than 6 when anything else was running), at most 4 on Arm (efficiency cores
/// slow the fast ones down).
pub fn auto_threads() -> i32 {
    let logical = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
    if cfg!(any(target_arch = "aarch64", target_arch = "arm")) { logical.clamp(2, 4) } else { (logical / 2).clamp(2, 8) }
}

pub struct Model {
    llama: Llama,
    /// Tokens currently in the KV cache per sequence, in position order.
    cached: [Vec<LlamaToken>; SEQS],
    /// End-of-generation tokens: eos, eot, and the end markers of common
    /// chat formats that exist in this vocabulary as single tokens.
    eog: Vec<LlamaToken>,
    n_ctx: usize,
    /// Recurrent or hybrid architecture (Qwen3.5, Mamba, ...): its memory
    /// can't be trimmed back to an earlier position, only restored whole.
    recurrent: bool,
    /// Per sequence, for recurrent models: the last prompt evaluated, and a
    /// snapshot of the state at the longest prefix it shared with the one before.
    last_prompt: [Vec<LlamaToken>; SEQS],
    checkpoint: [Vec<Checkpoint>; SEQS],
    /// Counts restores, to find the least recently used snapshot.
    tick: u64,
    /// Index (in the last decoded batch) of the logits of the last prompt token.
    logits_idx: i32,
    /// One sequence and a small context: a probe router that sits beside a main model.
    light: bool,
    /// Whether the context is currently set to output hidden states (see `last_hidden`).
    embeddings_on: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CacheHeader {
    cached: Vec<Vec<i32>>,
    last_prompt: Vec<Vec<i32>>,
    checkpoint: Vec<Vec<Vec<i32>>>,
}

struct Checkpoint {
    tokens: Vec<LlamaToken>,
    state: Vec<u8>,
    used: u64,
}

/// Snapshots kept per sequence: the shortest one (the fixed start of the prompt, which every request
/// shares) is never evicted; of the rest the least recently used goes first.
const MAX_CHECKPOINTS: usize = 3;

const END_MARKERS: [&str; 6] = ["<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "</s>"];

impl Model {
    /// `embeddings` keeps the last token's hidden state available (`last_hidden`).
    /// `light` makes a router-only model: one sequence and `n_ctx` tokens of context
    /// (every `Seq` then shares that one), which saves about 100 MB.
    /// `low_memory` trades about 10% speed for a few hundred MB less RAM: no repacked
    /// weight copies and small scratch buffers.
    pub fn load(path: &Path, n_ctx: u32, gpu_layers: i32, low_memory: bool, embeddings: bool, light: bool) -> Result<Model> {
        if !path.exists() {
            bail!("{} not found", path.display());
        }
        // Each sequence gets its own n_ctx: llama.cpp splits a non-unified KV
        // cache evenly between the n_seq_max sequences.
        let n_seq = if light { 1 } else { SEQS as u32 };
        let mut params = LlamaParams::new(path).with_n_ctx(n_ctx * n_seq).with_n_gpu_layers(gpu_layers);
        params.context = params.context.with_n_seq_max(n_seq);
        let threads = match std::env::var("OJP_THREADS").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or_else(|| THREADS.load(std::sync::atomic::Ordering::Relaxed)) {
            0 => auto_threads(),
            n => n,
        };
        params.context = params.context.with_n_threads(threads).with_n_threads_batch(threads);
        let ubatch = std::env::var("OJP_N_UBATCH").ok().and_then(|v| v.parse::<u32>().ok());
        // A light (classify-only) router reads about 60 tokens per request, so its scratch buffers can be tiny.
        if let Some(ub) = ubatch.or(if light { Some(32) } else { low_memory.then_some(128) }) {
            params.context = params.context.with_n_ubatch(ub);
        }
        // Hidden states are switched on only around `last_hidden`: with them on, llama.cpp
        // outputs every prompt token, which for a long prompt costs hundreds of MB.
        let _ = embeddings;
        if low_memory || std::env::var_os("OJP_NO_REPACK").is_some() {
            params.model = params.model.with_use_extra_bufts(false);
        }
        let mut llama = Llama::load(params).with_context(|| format!("loading {}", path.display()))?;
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
        let mut recurrent = vocab.is_recurrent() || vocab.is_hybrid();
        if !recurrent {
            // Not every delta-net/hybrid architecture reports itself as one (Qwen3.5
            // doesn't), so probe: can the last token of a short prompt be dropped?
            recurrent = !Model::can_rewind(&mut llama);
        }
        Ok(Model {
            llama,
            cached: Default::default(),
            eog,
            n_ctx: n_ctx as usize,
            recurrent,
            last_prompt: Default::default(),
            checkpoint: Default::default(),
            tick: 0,
            logits_idx: 0,
            light,
            embeddings_on: false,
        })
    }

    /// Whether the memory can drop a sequence's newest tokens.
    fn can_rewind(llama: &mut Llama) -> bool {
        let Ok(tokens) = llama.model().tokenize("hello there, how are you", false, false) else { return true };
        if tokens.len() < 2 {
            return true;
        }
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, &t) in tokens.iter().enumerate() {
            if batch.add(t, i as i32, &[0], i + 1 == tokens.len()).is_err() {
                return true;
            }
        }
        if llama.context().decode(&batch).is_err() {
            return true;
        }
        let ctx = llama.context().raw_handle();
        // SAFETY: ctx is valid; the memory handle is only used within this call.
        let ok = unsafe {
            let mem = llama_crab_sys::llama_get_memory(ctx);
            !mem.is_null() && llama_crab_sys::llama_memory_seq_rm(mem, 0, tokens.len() as i32 - 1, -1)
        };
        let _ = llama.context().seq_rm(0, -1, -1);
        ok
    }

    pub fn llama(&self) -> &Llama {
        &self.llama
    }

    /// The sequences this model actually has.
    fn active(&self) -> &'static [Seq] {
        if self.light { &ALL_SEQS[..1] } else { &ALL_SEQS }
    }

    fn norm(&self, seq: Seq) -> Seq {
        if self.light { Seq::Route } else { seq }
    }

    /// True for models whose memory can't be rewound (see `recurrent`).
    pub fn is_recurrent(&self) -> bool {
        self.recurrent
    }

    fn snapshot(&mut self, seq: Seq) -> Vec<u8> {
        let ctx = self.llama.context().raw_handle();
        // SAFETY: ctx is valid for the call; the buffer has the size llama.cpp asked for.
        unsafe {
            let size = llama_crab_sys::llama_state_seq_get_size(ctx, seq as i32);
            let mut buf = vec![0_u8; size];
            let written = llama_crab_sys::llama_state_seq_get_data(ctx, buf.as_mut_ptr(), size, seq as i32);
            buf.truncate(written);
            buf
        }
    }

    fn restore(&mut self, seq: Seq, idx: usize) -> bool {
        self.tick += 1;
        let tick = self.tick;
        let cp = &mut self.checkpoint[seq as usize][idx];
        cp.used = tick;
        if std::env::var_os("OJP_DEBUG_CACHE").is_some() {
            eprintln!("cache: restoring checkpoint of {} tokens ({} bytes)", cp.tokens.len(), cp.state.len());
        }
        let ctx = self.llama.context().raw_handle();
        let t = std::time::Instant::now();
        let _ = self.llama.context().seq_rm(seq as i32, -1, -1);
        let cp = &self.checkpoint[seq as usize][idx];
        // SAFETY: ctx is valid and the blob is a slice of the stated length.
        let loaded = unsafe { llama_crab_sys::llama_state_seq_set_data(ctx, cp.state.as_ptr(), cp.state.len(), seq as i32) };
        if std::env::var_os("OJP_DEBUG_CACHE").is_some() {
            eprintln!("cache: restore took {} ms", t.elapsed().as_millis());
        }
        if loaded == 0 {
            self.checkpoint[seq as usize].remove(idx);
            self.reset(seq);
            return false;
        }
        self.cached[seq as usize] = self.checkpoint[seq as usize][idx].tokens.clone();
        true
    }

    /// Keep `cp`, evicting the least recently used snapshot (never the shortest) when over the limit.
    fn add_checkpoint(&mut self, seq: Seq, mut cp: Checkpoint) {
        self.tick += 1;
        cp.used = self.tick;
        let list = &mut self.checkpoint[seq as usize];
        list.push(cp);
        if list.len() > MAX_CHECKPOINTS {
            let shortest = (0..list.len()).min_by_key(|&i| list[i].tokens.len()).unwrap_or(0);
            if let Some(victim) = (0..list.len()).filter(|&i| i != shortest).min_by_key(|&i| list[i].used) {
                list.remove(victim);
            }
        }
    }

    /// Save the KV cache (all sequences) and the token lists describing it,
    /// so a later start can skip re-evaluating the same warm-up prompts.
    pub fn save_cache(&mut self, path: &Path) -> Result<()> {
        // One state blob per sequence (llama_state_seq_*): the whole-context
        // state restores the two sequences' cells wrongly, which made the
        // first request after a restart misroute when it matched the warm-up.
        let header = CacheHeader {
            cached: self.cached.iter().map(|seq| seq.iter().map(|t| t.raw()).collect()).collect(),
            last_prompt: self.last_prompt.iter().map(|seq| seq.iter().map(|t| t.raw()).collect()).collect(),
            checkpoint: self
                .checkpoint
                .iter()
                .map(|list| list.iter().map(|cp| cp.tokens.iter().map(|t| t.raw()).collect()).collect())
                .collect(),
        };
        let header = serde_json::to_vec(&header)?;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        for seq in ALL_SEQS {
            let blob = if self.active().contains(&seq) { self.snapshot(seq) } else { Vec::new() };
            out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
            out.extend_from_slice(&blob);
        }
        // Recurrent models' snapshots, in the order the header lists them.
        for list in &self.checkpoint {
            for cp in list {
                out.extend_from_slice(&(cp.state.len() as u64).to_le_bytes());
                out.extend_from_slice(&cp.state);
            }
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
            let header: CacheHeader = serde_json::from_slice(header)?;
            if header.cached.len() != SEQS || header.last_prompt.len() != SEQS || header.checkpoint.len() != SEQS {
                bail!("cache has the wrong number of sequences, expected {SEQS}");
            }
            let ctx = self.llama.context().raw_handle();
            let mut at = 8 + len;
            for seq in ALL_SEQS {
                let size = read_len(at)?;
                let blob = bytes.get(at + 8..at + 8 + size).context("short cache file")?;
                at += 8 + size;
                if !self.active().contains(&seq) {
                    continue;
                }
                let _ = self.llama.context().seq_rm(seq as i32, -1, -1);
                // SAFETY: ctx is valid and blob is a slice of the stated length.
                let loaded = unsafe { llama_crab_sys::llama_state_seq_set_data(ctx, blob.as_ptr(), blob.len(), seq as i32) };
                if loaded == 0 && !header.cached[seq as usize].is_empty() {
                    bail!("llama.cpp rejected the saved state");
                }
            }
            let tokens = |raw: &[i32]| raw.iter().map(|&t| LlamaToken::from(t)).collect::<Vec<_>>();
            for (i, list) in header.checkpoint.iter().enumerate() {
                self.checkpoint[i].clear();
                for cp_tokens in list {
                    let size = read_len(at)?;
                    let blob = bytes.get(at + 8..at + 8 + size).context("short cache file")?;
                    at += 8 + size;
                    self.checkpoint[i].push(Checkpoint { tokens: tokens(cp_tokens), state: blob.to_vec(), used: 0 });
                }
            }
            for i in 0..SEQS {
                self.cached[i] = tokens(&header.cached[i]);
                self.last_prompt[i] = tokens(&header.last_prompt[i]);
            }
            Ok(())
        })();
        if result.is_err() {
            for &seq in self.active() {
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
        let seq = self.norm(seq);
        // `last_hidden` turns this on itself, just before calling here.
        let hidden_call = max_tokens == 0 && self.embeddings_on;
        if !hidden_call {
            self.set_embeddings(false);
        }
        let tokens = self.llama.model().tokenize(prompt, true, true)?;
        if tokens.is_empty() {
            bail!("empty prompt");
        }
        if tokens.len() + max_tokens > self.n_ctx {
            bail!("prompt of {} tokens + {max_tokens} to generate exceeds the {}-token context", tokens.len(), self.n_ctx);
        }
        let started = std::time::Instant::now();
        let result = self.generate(seq, &tokens, sampler, max_tokens, on_text);
        if std::env::var_os("OJP_DEBUG_CACHE").is_some() {
            eprintln!("cache: seq {} total {} ms", seq as i32, started.elapsed().as_millis());
        }
        if result.is_err() {
            // Unknown KV state after a failed decode: start clean next time.
            self.reset(seq);
        }
        result
    }

    /// Evaluate `prompt` (reusing the cache) and return the model's raw logits
    /// for the first token of each of `candidates`, without generating anything.
    pub fn first_token_logits(&mut self, seq: Seq, prompt: &str, candidates: &[&str]) -> Result<Vec<f32>> {
        let ids: Vec<usize> = candidates
            .iter()
            .map(|c| {
                let toks = self.llama.model().tokenize(c, false, false)?;
                toks.first().map(|t| t.raw() as usize).context("candidate tokenized to nothing")
            })
            .collect::<Result<_>>()?;
        let mut sampler = LlamaSampler::greedy().context("greedy sampler")?;
        self.complete(seq, prompt, &mut sampler, 0, &mut |_| {})?;
        let logits = self.llama.context().logits_ith(self.logits_idx)?;
        Ok(ids.iter().map(|&i| logits[i]).collect())
    }

    /// Evaluate `prompt` and return the final hidden state of its last token
    /// (needs the model loaded with OJP_EMBED set).
    pub fn last_hidden(&mut self, seq: Seq, prompt: &str) -> Result<Vec<f32>> {
        self.set_embeddings(true);
        let mut sampler = LlamaSampler::greedy().context("greedy sampler")?;
        self.complete(seq, prompt, &mut sampler, 0, &mut |_| {})?;
        Ok(self.llama.context().embeddings_ith(self.logits_idx)?.to_vec())
    }

    fn set_embeddings(&mut self, on: bool) {
        if self.embeddings_on != on {
            // SAFETY: the context handle is valid for the lifetime of self.llama.
            unsafe { llama_crab_sys::llama_set_embeddings(self.llama.context().raw_handle(), on) };
            self.embeddings_on = on;
        }
    }

    fn reset(&mut self, seq: Seq) {
        let _ = self.llama.context().seq_rm(seq as i32, -1, -1);
        self.cached[seq as usize].clear();
        if self.recurrent {
            self.last_prompt[seq as usize].clear();
            self.checkpoint[seq as usize].clear();
        }
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
        let mut checkpoint_at = None;
        let keep = if self.recurrent {
            // The memory can't be trimmed, so continue from what is cached
            // only when it is entirely a prefix of the new prompt; otherwise
            // go back to the snapshot, or to nothing.
            let cached_len = self.cached[seq as usize].len();
            if common == cached_len && cached_len < tokens.len() {
                cached_len
            } else {
                let best = self.checkpoint[seq as usize]
                    .iter()
                    .enumerate()
                    .filter(|(_, cp)| cp.tokens.len() < tokens.len() && tokens.starts_with(&cp.tokens))
                    .max_by_key(|(_, cp)| cp.tokens.len())
                    .map(|(i, _)| i);
                if !best.is_some_and(|i| self.restore(seq, i)) {
                    let _ = self.llama.context().seq_rm(id, -1, -1);
                    self.cached[seq as usize].clear();
                }
                self.cached[seq as usize].len()
            }
        } else {
            let keep = common.min(tokens.len() - 1);
            if self.llama.context().seq_rm(id, keep as i32, -1).is_err() {
                self.reset(seq);
            }
            let keep = keep.min(self.cached[seq as usize].len());
            self.cached[seq as usize].truncate(keep);
            keep
        };
        if self.recurrent {
            // Where this prompt departs from the previous one is where the
            // next one will probably depart too: snapshot the state there.
            let shared = self.last_prompt[seq as usize].iter().zip(tokens).take_while(|(a, b)| a == b).count();
            let at = shared.min(tokens.len() - 1);
            if at > keep && self.checkpoint[seq as usize].iter().all(|cp| cp.tokens.len() != at) {
                checkpoint_at = Some(at);
            }
            self.last_prompt[seq as usize] = tokens.to_vec();
        }

        if std::env::var_os("OJP_DEBUG_CACHE").is_some() {
            eprintln!("cache: seq {id} prompt {} tokens, reusing {keep}, checkpoint_at {checkpoint_at:?}", tokens.len());
        }
        let n_batch = self.llama.context().n_batch() as usize;
        let mut logits_idx = 0;
        let mut start = keep;
        while start < tokens.len() {
            let mut end = (start + n_batch).min(tokens.len());
            if let Some(at) = checkpoint_at.filter(|&at| start < at && at < end) {
                end = at;
            }
            let chunk = &tokens[start..end];
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
            start = end;
            if checkpoint_at == Some(start) {
                let state = self.snapshot(seq);
                self.add_checkpoint(seq, Checkpoint { tokens: tokens[..start].to_vec(), state, used: 0 });
            }
        }

        self.logits_idx = logits_idx;
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
