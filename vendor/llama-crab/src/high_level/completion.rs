//! Text completion driver.

use serde_json::Value;

use crate::error::{LlamaError, Result};
use crate::logit_bias::LlamaLogitBias;
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;
use llama_crab_sys as sys;

use super::Llama;

/// The result of a text completion.
#[derive(Debug, Clone, PartialEq)]
pub struct Completion {
    /// Concatenated generated text.
    pub text: String,
    /// Number of tokens generated.
    pub n_tokens: usize,
    /// Reason generation stopped (`"length"`, `"eos"`, or `"stop"`).
    pub stop_reason: StopReason,
    /// Optional per-token logprob information.
    pub logprobs: Option<CompletionLogprobs>,
}

/// Per-token logprob data for a completion.
#[derive(Debug, Clone, PartialEq)]
pub struct CompletionLogprobs {
    /// Token text fragments.
    pub tokens: Vec<String>,
    /// Byte offsets into the logical completion text.
    pub text_offset: Vec<usize>,
    /// Logprob of each selected token. Prompt tokens can be represented as `None`.
    pub token_logprobs: Vec<Option<f32>>,
    /// Top candidate logprobs for each selected token.
    pub top_logprobs: Vec<Option<Vec<TokenLogprob>>>,
}

impl CompletionLogprobs {
    fn new() -> Self {
        Self {
            tokens: Vec::new(),
            text_offset: Vec::new(),
            token_logprobs: Vec::new(),
            top_logprobs: Vec::new(),
        }
    }

    fn from_record(record: TokenLogprobRecord) -> Self {
        let mut logprobs = Self::new();
        logprobs.push(record);
        logprobs
    }

    fn push(&mut self, record: TokenLogprobRecord) {
        self.tokens.push(record.token);
        self.text_offset.push(record.text_offset);
        self.token_logprobs.push(Some(record.token_logprob));
        self.top_logprobs.push(Some(record.top_logprobs));
    }
}

/// One top-logprob candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    /// Token id.
    pub token: i32,
    /// Token text.
    pub text: String,
    /// Log probability.
    pub logprob: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct TokenLogprobRecord {
    token: String,
    text_offset: usize,
    token_logprob: f32,
    top_logprobs: Vec<TokenLogprob>,
}

/// Options for high-level text completion.
#[derive(Debug, Clone, PartialEq)]
pub struct CompletionOptions {
    /// Maximum number of generated tokens.
    pub max_tokens: usize,
    /// Stop strings that terminate generation without being emitted.
    pub stop_sequences: Vec<String>,
    /// Sampling pipeline options.
    pub sampling: SamplingOptions,
    /// Include the prompt at the beginning of the returned text.
    pub echo_prompt: bool,
    /// Text appended to the final completion.
    pub suffix: Option<String>,
    /// Additive token-logit biases applied before sampling.
    pub logit_bias: Vec<LlamaLogitBias>,
    /// Minimum generated tokens before EOS/EOT can terminate generation.
    pub min_tokens: usize,
    /// Number of top logprobs to retain per generated token.
    pub logprobs: Option<usize>,
}

/// Sampling options used by high-level generation helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingOptions {
    /// Temperature. `0.0` selects greedy decoding; negative values select
    /// distribution sampling directly.
    pub temperature: f32,
    /// Top-K cutoff.
    pub top_k: i32,
    /// Top-P nucleus cutoff.
    pub top_p: f32,
    /// Tail-free sampling cutoff. `1.0` disables it.
    pub tfs_z: f32,
    /// Min-P cutoff.
    pub min_p: f32,
    /// Locally-typical cutoff.
    pub typical_p: f32,
    /// Minimum candidates retained by probability filters.
    pub min_keep: usize,
    /// Repeat penalty lookback window.
    pub penalty_last_n: i32,
    /// Repeat penalty multiplier.
    pub repeat_penalty: f32,
    /// Frequency penalty.
    pub frequency_penalty: f32,
    /// Presence penalty.
    pub presence_penalty: f32,
    /// Mirostat mode: `0` disabled, `1` v1, `2` v2.
    pub mirostat_mode: i32,
    /// Mirostat target entropy.
    pub mirostat_tau: f32,
    /// Mirostat learning rate.
    pub mirostat_eta: f32,
    /// Sampler seed. `None` delegates random seeding to llama.cpp.
    pub seed: Option<u32>,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            tfs_z: 1.0,
            min_p: 0.05,
            typical_p: 1.0,
            min_keep: 1,
            penalty_last_n: 64,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            mirostat_mode: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            seed: None,
        }
    }
}

impl SamplingOptions {
    /// Defaults commonly used for chat generation.
    #[must_use]
    pub fn chat() -> Self {
        Self {
            temperature: 0.2,
            ..Self::default()
        }
    }

    /// Set the temperature.
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u32) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build a sampler chain from these options for the given model/context.
    pub fn build_sampler(self, llama: &Llama) -> Result<LlamaSampler> {
        let mut samplers = Vec::new();
        let seed = self.seed.unwrap_or(u32::MAX);

        if self.repeat_penalty != 1.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
        {
            samplers.push(
                LlamaSampler::penalties(
                    llama.model().n_vocab(),
                    self.penalty_last_n,
                    self.repeat_penalty,
                    self.frequency_penalty,
                    self.presence_penalty,
                )
                .ok_or_else(|| LlamaError::Batch("sampler_init_penalties returned null".into()))?,
            );
        }

        if self.temperature < 0.0 {
            samplers.push(
                LlamaSampler::dist(seed)
                    .ok_or_else(|| LlamaError::Batch("sampler_init_dist returned null".into()))?,
            );
        } else if self.temperature == 0.0 {
            samplers
                .push(LlamaSampler::greedy().ok_or_else(|| {
                    LlamaError::Batch("sampler_init_greedy returned null".into())
                })?);
        } else if self.mirostat_mode == 1 {
            samplers.push(
                LlamaSampler::mirostat(
                    llama.model().n_vocab(),
                    seed,
                    self.mirostat_tau,
                    self.mirostat_eta,
                    100,
                )
                .ok_or_else(|| LlamaError::Batch("sampler_init_mirostat returned null".into()))?,
            );
        } else if self.mirostat_mode == 2 {
            samplers.push(
                LlamaSampler::mirostat_v2(seed, self.mirostat_tau, self.mirostat_eta).ok_or_else(
                    || LlamaError::Batch("sampler_init_mirostat_v2 returned null".into()),
                )?,
            );
        } else {
            samplers.push(
                LlamaSampler::top_k(self.top_k)
                    .ok_or_else(|| LlamaError::Batch("sampler_init_top_k returned null".into()))?,
            );
            if self.tfs_z != 1.0 {
                samplers.push(
                    LlamaSampler::tail_free(self.tfs_z, self.min_keep).ok_or_else(|| {
                        LlamaError::Batch("sampler_init_tail_free returned null".into())
                    })?,
                );
            }
            samplers.push(
                LlamaSampler::typical(self.typical_p, self.min_keep).ok_or_else(|| {
                    LlamaError::Batch("sampler_init_typical returned null".into())
                })?,
            );
            samplers.push(
                LlamaSampler::top_p(self.top_p, self.min_keep)
                    .ok_or_else(|| LlamaError::Batch("sampler_init_top_p returned null".into()))?,
            );
            samplers.push(
                LlamaSampler::min_p(self.min_p, self.min_keep)
                    .ok_or_else(|| LlamaError::Batch("sampler_init_min_p returned null".into()))?,
            );
            samplers.push(
                LlamaSampler::temp(self.temperature)
                    .ok_or_else(|| LlamaError::Batch("sampler_init_temp returned null".into()))?,
            );
            samplers.push(
                LlamaSampler::dist(seed)
                    .ok_or_else(|| LlamaError::Batch("sampler_init_dist returned null".into()))?,
            );
        }

        LlamaSampler::chain(samplers, false)
            .ok_or_else(|| LlamaError::Batch("sampler_chain_init returned null".into()))
    }
}

impl CompletionOptions {
    /// Create completion options with no stop sequences.
    #[must_use]
    pub const fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            stop_sequences: Vec::new(),
            sampling: SamplingOptions {
                temperature: 0.0,
                top_k: 40,
                top_p: 0.95,
                tfs_z: 1.0,
                min_p: 0.05,
                typical_p: 1.0,
                min_keep: 1,
                penalty_last_n: 64,
                repeat_penalty: 1.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                mirostat_mode: 0,
                mirostat_tau: 5.0,
                mirostat_eta: 0.1,
                seed: None,
            },
            echo_prompt: false,
            suffix: None,
            logit_bias: Vec::new(),
            min_tokens: 0,
            logprobs: None,
        }
    }

    /// Create completion options with the default probabilistic sampler.
    #[must_use]
    pub fn sampled(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            stop_sequences: Vec::new(),
            sampling: SamplingOptions::default(),
            echo_prompt: false,
            suffix: None,
            logit_bias: Vec::new(),
            min_tokens: 0,
            logprobs: None,
        }
    }

    /// Replace sampling options.
    #[must_use]
    pub const fn with_sampling(mut self, sampling: SamplingOptions) -> Self {
        self.sampling = sampling;
        self
    }

    /// Include or suppress the prompt in the returned text.
    #[must_use]
    pub const fn with_echo_prompt(mut self, echo_prompt: bool) -> Self {
        self.echo_prompt = echo_prompt;
        self
    }

    /// Append a suffix to the final completion text.
    #[must_use]
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        let suffix = suffix.into();
        self.suffix = (!suffix.is_empty()).then_some(suffix);
        self
    }

    /// Replace all token-logit biases.
    #[must_use]
    pub fn with_logit_biases<I>(mut self, biases: I) -> Self
    where
        I: IntoIterator<Item = LlamaLogitBias>,
    {
        self.logit_bias = biases.into_iter().collect();
        self
    }

    /// Require at least this many generated tokens before EOS/EOT can stop generation.
    #[must_use]
    pub const fn with_min_tokens(mut self, min_tokens: usize) -> Self {
        self.min_tokens = min_tokens;
        self
    }

    /// Retain per-token logprobs and this many top candidates.
    #[must_use]
    pub const fn with_logprobs(mut self, logprobs: usize) -> Self {
        self.logprobs = Some(logprobs);
        self
    }

    /// Build the sampler chain represented by these options.
    pub fn build_sampler(&self, llama: &Llama) -> Result<LlamaSampler> {
        build_completion_sampler(llama, self)
    }

    /// Add one stop sequence.
    #[must_use]
    pub fn with_stop_sequence(mut self, stop_sequence: impl Into<String>) -> Self {
        let stop_sequence = stop_sequence.into();
        if !stop_sequence.is_empty() {
            self.stop_sequences.push(stop_sequence);
        }
        self
    }

    /// Add multiple stop sequences.
    #[must_use]
    pub fn with_stop_sequences<I, S>(mut self, stop_sequences: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stop_sequences.extend(
            stop_sequences
                .into_iter()
                .map(Into::into)
                .filter(|s: &String| !s.is_empty()),
        );
        self
    }
}

/// A synchronous streaming completion callback chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct CompletionChunk {
    /// Text made available by this callback invocation.
    pub text: String,
    /// Token that produced this chunk, if the chunk corresponds to a sampled token.
    ///
    /// Terminal flush chunks can carry no token when they only report a final
    /// [`StopReason`].
    pub token: Option<LlamaToken>,
    /// Cumulative number of generated non-EOS tokens.
    pub n_tokens: usize,
    /// Terminal reason when this is the last chunk.
    pub stop_reason: Option<StopReason>,
    /// Optional per-token logprob information for this chunk.
    pub logprobs: Option<CompletionLogprobs>,
}

/// Why a completion ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StopReason {
    /// Reached the requested `max_tokens`.
    Length,
    /// Model emitted an end-of-sequence token.
    Eos,
    /// Custom stop string matched.
    Stop,
    /// Model emitted one or more tool calls (streaming consumers only).
    ToolCalls,
}

/// Return value from synchronous streaming callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamControl {
    /// Continue generation.
    Continue,
    /// Stop generation after the current callback.
    Stop,
}

/// Generate a single completion for `prompt`.
///
/// This is the simplest possible inference loop: tokenize → decode the prompt
/// → sample → decode one token at a time. For more control, use the lower
/// level [`crate::LlamaContext`] / [`crate::sampling::LlamaSampler`] APIs.
///
/// The KV cache for sequence 0 is cleared before each call, so each call is
/// independent. For multi-turn conversations, build the full history into the
/// prompt and call again.
pub fn create_completion(llama: &mut Llama, prompt: &str, max_tokens: usize) -> Result<Completion> {
    create_completion_with_options(llama, prompt, CompletionOptions::new(max_tokens))
}

/// Generate a single completion for `prompt` using high-level options.
pub fn create_completion_with_options(
    llama: &mut Llama,
    prompt: &str,
    options: CompletionOptions,
) -> Result<Completion> {
    create_completion_stream(llama, prompt, options, |_| StreamControl::Continue)
}

/// Generate a single completion using a caller-provided sampler.
pub fn create_completion_with_sampler(
    llama: &mut Llama,
    prompt: &str,
    options: CompletionOptions,
    sampler: &mut LlamaSampler,
) -> Result<Completion> {
    create_completion_stream_with_sampler(llama, prompt, options, sampler, |_| {
        StreamControl::Continue
    })
}

/// Generate a completion and synchronously call `on_chunk` as text becomes
/// available.
pub fn create_completion_stream<F>(
    llama: &mut Llama,
    prompt: &str,
    options: CompletionOptions,
    on_chunk: F,
) -> Result<Completion>
where
    F: FnMut(CompletionChunk) -> StreamControl,
{
    let mut sampler = options.build_sampler(llama)?;
    create_completion_stream_with_sampler(llama, prompt, options, &mut sampler, on_chunk)
}

fn build_completion_sampler(llama: &Llama, options: &CompletionOptions) -> Result<LlamaSampler> {
    let base_sampler = options.sampling.build_sampler(llama)?;
    if options.logit_bias.is_empty() {
        return Ok(base_sampler);
    }

    let raw_biases: Vec<sys::llama_logit_bias> = options
        .logit_bias
        .iter()
        .map(|bias| sys::llama_logit_bias {
            token: bias.token,
            bias: bias.bias,
        })
        .collect();
    let bias_sampler = unsafe { LlamaSampler::logit_bias(llama.model().n_vocab(), &raw_biases) }
        .ok_or_else(|| LlamaError::Batch("sampler_init_logit_bias returned null".into()))?;
    LlamaSampler::chain(vec![bias_sampler, base_sampler], false)
        .ok_or_else(|| LlamaError::Batch("sampler_chain_init returned null".into()))
}

/// Generate a streaming completion using a caller-provided sampler.
pub fn create_completion_stream_with_sampler<F>(
    llama: &mut Llama,
    prompt: &str,
    options: CompletionOptions,
    sampler: &mut LlamaSampler,
    mut on_chunk: F,
) -> Result<Completion>
where
    F: FnMut(CompletionChunk) -> StreamControl,
{
    // Clear sequence 0 so the new batch can start at position 0 regardless
    // of any previous decode. p0 = p1 = -1 means "the entire range".
    let _ = llama.context().seq_rm(0, -1, -1);

    let tokens = llama.model().tokenize(prompt, true, true)?;

    // Build a batch with the prompt; only the last token produces logits.
    let mut batch = crate::batch::LlamaBatch::new(tokens.len(), 1);
    for (i, &t) in tokens.iter().enumerate() {
        batch
            .add(t, i as i32, &[0], i + 1 == tokens.len())
            .map_err(LlamaError::from)?;
    }
    llama.context().decode(&batch)?;

    let eos = llama.model().token_eos();
    let eot = llama.model().token_eot();
    let mut generated = String::new();
    let mut stop_buffer = StopBuffer::new(options.stop_sequences);
    let mut last_pos = tokens.len() as i32;
    let mut n_generated = 0_usize;
    let mut stop_reason = StopReason::Length;

    if options.echo_prompt
        && emit_chunk(
            &mut on_chunk,
            &mut generated,
            prompt.to_string(),
            None,
            0,
            None,
            None,
        ) == StreamControl::Stop
    {
        return Ok(Completion {
            text: generated,
            n_tokens: 0,
            stop_reason: StopReason::Stop,
            logprobs: None,
        });
    }

    let mut logprobs = options.logprobs.map(|_| CompletionLogprobs::new());
    for _ in 0..options.max_tokens {
        // `idx` is the index in the *current* batch whose logits we sample
        // from. For the initial prompt (all tokens in one batch) the logits
        // are at the last position. For every subsequent single-token batch
        // the logits are at index 0.
        let idx = if n_generated == 0 {
            (tokens.len() as i32) - 1
        } else {
            0
        };
        let suppress_eog = n_generated < options.min_tokens;
        let next = sample_next_token(llama, sampler, idx, suppress_eog)?;
        let logits_for_logprobs = options
            .logprobs
            .map(|_| logits_for_logprobs(llama, suppress_eog))
            .transpose()?;
        if next == eos || next == eot {
            stop_reason = StopReason::Eos;
            break;
        }
        let piece = llama.model().detokenize(&[next], false)?;
        let mut chunk_logprobs = None;
        if let (Some(logprobs), Some(top_n), Some(logits)) =
            (&mut logprobs, options.logprobs, logits_for_logprobs)
        {
            let text_offset = if options.echo_prompt {
                generated.len()
            } else {
                prompt.len() + generated.len()
            };
            let mut record = token_logprob_record(&logits, next, piece.clone(), text_offset, top_n);
            for candidate in &mut record.top_logprobs {
                if candidate.text == candidate.token.to_string() {
                    candidate.text = llama
                        .model()
                        .detokenize(&[LlamaToken::from(candidate.token)], false)?;
                }
            }
            chunk_logprobs = Some(CompletionLogprobs::from_record(record.clone()));
            logprobs.push(record);
        }
        n_generated += 1;
        let step = stop_buffer.push(&piece);
        if step.stopped {
            stop_reason = StopReason::Stop;
            if emit_chunk(
                &mut on_chunk,
                &mut generated,
                step.text,
                Some(next),
                n_generated,
                None,
                chunk_logprobs,
            ) == StreamControl::Stop
            {
                stop_reason = StopReason::Stop;
            }
            break;
        }
        if emit_chunk(
            &mut on_chunk,
            &mut generated,
            step.text,
            Some(next),
            n_generated,
            None,
            chunk_logprobs,
        ) == StreamControl::Stop
        {
            stop_reason = StopReason::Stop;
            break;
        }
        // Feed back the new token.
        let mut single = crate::batch::LlamaBatch::new(1, 1);
        single
            .add(next, last_pos, &[0], true)
            .map_err(LlamaError::from)?;
        llama.context().decode(&single)?;
        last_pos += 1;
    }

    let final_text = format!(
        "{}{}",
        stop_buffer.flush(),
        options.suffix.as_deref().unwrap_or("")
    );
    if emit_chunk(
        &mut on_chunk,
        &mut generated,
        final_text,
        None,
        n_generated,
        Some(stop_reason),
        None,
    ) == StreamControl::Stop
    {
        stop_reason = StopReason::Stop;
    }

    Ok(Completion {
        text: generated,
        n_tokens: n_generated,
        stop_reason,
        logprobs,
    })
}

fn logits_for_logprobs(llama: &mut Llama, suppress_eog: bool) -> Result<Vec<f32>> {
    let ctx = llama.context().raw_handle();
    let logits = unsafe { sys::llama_get_logits(ctx) };
    if logits.is_null() {
        return Err(LlamaError::Batch("no logits".into()));
    }
    let n_vocab = llama.model().n_vocab() as usize;
    let mut logits = unsafe { std::slice::from_raw_parts(logits, n_vocab) }.to_vec();
    if suppress_eog {
        for token in [llama.model().token_eos(), llama.model().token_eot()] {
            let raw = token.raw();
            if raw >= 0 && (raw as usize) < logits.len() {
                logits[raw as usize] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(logits)
}

fn token_logprob_record(
    logits: &[f32],
    selected: LlamaToken,
    selected_text: String,
    text_offset: usize,
    top_n: usize,
) -> TokenLogprobRecord {
    let logprobs = logits_to_logprobs(logits);
    let selected_id = selected.raw();
    let selected_logprob = selected_logprob(&logprobs, selected_id);
    let mut candidates: Vec<(i32, f32)> = logprobs
        .iter()
        .enumerate()
        .map(|(token, &logprob)| (token as i32, logprob))
        .collect();
    candidates.sort_by(|(_, lhs), (_, rhs)| rhs.total_cmp(lhs));
    candidates.truncate(top_n);
    if !candidates.iter().any(|(token, _)| *token == selected_id) {
        candidates.push((selected_id, selected_logprob));
    }

    let top_logprobs = candidates
        .into_iter()
        .map(|(token, logprob)| TokenLogprob {
            token,
            text: if token == selected_id {
                selected_text.clone()
            } else {
                token.to_string()
            },
            logprob,
        })
        .collect();

    TokenLogprobRecord {
        token: selected_text,
        text_offset,
        token_logprob: selected_logprob,
        top_logprobs,
    }
}

fn selected_logprob(logprobs: &[f32], selected: i32) -> f32 {
    if selected < 0 {
        return f32::NEG_INFINITY;
    }
    logprobs
        .get(selected as usize)
        .copied()
        .unwrap_or(f32::NEG_INFINITY)
}

fn logits_to_logprobs(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![f32::NEG_INFINITY; logits.len()];
    }
    let sum_exp: f32 = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| (value - max).exp())
        .sum();
    let log_sum_exp = max + sum_exp.ln();
    logits
        .iter()
        .map(|&value| {
            if value.is_finite() {
                value - log_sum_exp
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

fn sample_next_token(
    llama: &mut Llama,
    sampler: &mut LlamaSampler,
    idx: i32,
    suppress_eog: bool,
) -> Result<LlamaToken> {
    if !suppress_eog {
        return Ok(unsafe { sampler.sample(llama.context().raw_handle(), idx) });
    }

    let eos = llama.model().token_eos();
    let eot = llama.model().token_eot();
    let ctx = llama.context().raw_handle();
    let logits = unsafe { sys::llama_get_logits_ith(ctx, idx) };
    if logits.is_null() {
        return Err(LlamaError::Batch(format!("no logits at index {idx}")));
    }

    let mut restore = Vec::new();
    for token in [eos, eot] {
        let raw = token.raw();
        if raw >= 0 && raw < llama.model().n_vocab() {
            let slot = unsafe { logits.add(raw as usize) };
            let previous = unsafe { *slot };
            unsafe {
                *slot = f32::NEG_INFINITY;
            }
            restore.push((slot, previous));
        }
    }

    let sampled = unsafe { sampler.sample(ctx, idx) };
    for (slot, previous) in restore {
        unsafe {
            *slot = previous;
        }
    }
    Ok(sampled)
}

#[cfg(test)]
fn format_completion_text(prompt: &str, generated: &str, options: &CompletionOptions) -> String {
    let mut text = String::new();
    if options.echo_prompt {
        text.push_str(prompt);
    }
    text.push_str(generated);
    if let Some(suffix) = &options.suffix {
        text.push_str(suffix);
    }
    text
}

/// Create a GBNF grammar from a JSON Schema (used with `LlamaSampler::grammar`).
///
/// # Example
///
/// ```no_run
/// use llama_crab::high_level::completion::json_schema_grammar;
/// use serde_json::json;
/// let grammar = json_schema_grammar(&json!({
///     "type": "object",
///     "properties": {
///         "answer": { "type": "string" }
///     },
///     "required": ["answer"]
/// })).unwrap();
/// assert!(grammar.contains("root"));
/// ```
pub fn json_schema_grammar(schema: &Value) -> Result<String> {
    crate::json_schema::schema_to_grammar(schema, "root")
        .map_err(|e| LlamaError::JsonSchemaToGrammar(e.to_string()))
}

fn emit_chunk<F>(
    on_chunk: &mut F,
    generated: &mut String,
    text: String,
    token: Option<LlamaToken>,
    n_tokens: usize,
    stop_reason: Option<StopReason>,
    logprobs: Option<CompletionLogprobs>,
) -> StreamControl
where
    F: FnMut(CompletionChunk) -> StreamControl,
{
    if text.is_empty() && stop_reason.is_none() {
        return StreamControl::Continue;
    }

    generated.push_str(&text);
    on_chunk(CompletionChunk {
        text,
        token,
        n_tokens,
        stop_reason,
        logprobs,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StopBuffer {
    pending: String,
    stop_sequences: Vec<String>,
    stopped: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StopBufferStep {
    text: String,
    stopped: bool,
}

impl StopBuffer {
    fn new(stop_sequences: Vec<String>) -> Self {
        Self {
            pending: String::new(),
            stop_sequences: stop_sequences
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect(),
            stopped: false,
        }
    }

    fn push(&mut self, text: &str) -> StopBufferStep {
        if self.stopped {
            return StopBufferStep {
                text: String::new(),
                stopped: true,
            };
        }
        if self.stop_sequences.is_empty() {
            return StopBufferStep {
                text: text.to_string(),
                stopped: false,
            };
        }

        self.pending.push_str(text);
        if let Some(stop_start) = self.find_stop_start() {
            self.stopped = true;
            let text = self.pending[..stop_start].to_string();
            self.pending.clear();
            return StopBufferStep {
                text,
                stopped: true,
            };
        }

        let hold_start = self.longest_stop_prefix_suffix_start();
        let text = self.pending[..hold_start].to_string();
        self.pending = self.pending[hold_start..].to_string();
        StopBufferStep {
            text,
            stopped: false,
        }
    }

    fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    fn find_stop_start(&self) -> Option<usize> {
        self.stop_sequences
            .iter()
            .filter_map(|stop| self.pending.find(stop))
            .min()
    }

    fn longest_stop_prefix_suffix_start(&self) -> usize {
        let mut hold_start = self.pending.len();
        for (start, _) in self.pending.char_indices() {
            let suffix = &self.pending[start..];
            if self
                .stop_sequences
                .iter()
                .any(|stop| stop.starts_with(suffix))
            {
                hold_start = start;
                break;
            }
        }
        hold_start
    }
}

#[cfg(test)]
mod tests {
    use super::{format_completion_text, token_logprob_record, CompletionOptions, StopBuffer};
    use crate::LlamaToken;

    #[test]
    fn stop_buffer_holds_stop_prefix_across_token_boundaries() {
        let mut buffer = StopBuffer::new(vec!["</stop>".to_string()]);

        let first = buffer.push("hello </");
        assert_eq!(first.text, "hello ");
        assert!(!first.stopped);

        let second = buffer.push("stop> ignored");
        assert_eq!(second.text, "");
        assert!(second.stopped);
    }

    #[test]
    fn stop_buffer_removes_stop_sequence_inside_chunk() {
        let mut buffer = StopBuffer::new(vec!["END".to_string()]);

        let step = buffer.push("answerEND trailing");

        assert_eq!(step.text, "answer");
        assert!(step.stopped);
    }

    #[test]
    fn completion_options_apply_echo_and_suffix_to_final_text() {
        let options = CompletionOptions::new(16)
            .with_echo_prompt(true)
            .with_suffix(" done");

        let text = format_completion_text("prompt: ", "answer", &options);

        assert_eq!(text, "prompt: answer done");
    }

    #[test]
    fn completion_options_apply_min_tokens() {
        let options = CompletionOptions::new(8).with_min_tokens(3);

        assert_eq!(options.min_tokens, 3);
    }

    #[test]
    fn completion_options_apply_logprobs() {
        let options = CompletionOptions::new(8).with_logprobs(3);

        assert_eq!(options.logprobs, Some(3));
    }

    #[test]
    fn token_logprobs_include_selected_token_and_top_candidates() {
        let record = token_logprob_record(
            &[0.0, 2.0, 1.0],
            LlamaToken::from(0),
            "zero".to_string(),
            4,
            1,
        );

        assert_eq!(record.token, "zero");
        assert_eq!(record.text_offset, 4);
        assert_eq!(record.top_logprobs.len(), 2);
        assert!(record
            .top_logprobs
            .iter()
            .any(|candidate| candidate.token == 0 && candidate.text == "zero"));
    }

    #[test]
    fn stop_buffer_flushes_pending_prefix_when_generation_finishes_without_stop() {
        let mut buffer = StopBuffer::new(vec!["foobar".to_string()]);

        let step = buffer.push("hello foo");
        assert_eq!(step.text, "hello ");
        assert!(!step.stopped);

        assert_eq!(buffer.flush(), "foo");
    }
}
