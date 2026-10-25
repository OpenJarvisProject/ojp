//! Sampling strategies exposed as associated constructors on [`LlamaSampler`].

use llama_crab_sys as sys;
use std::os::raw::c_char;

#[allow(unused_imports)]
use super::LlamaSampler;
#[allow(unused_imports)]
use crate::model::LlamaModel;
#[allow(unused_imports)]
use crate::token::LlamaToken;

impl LlamaSampler {
    /// Always pick the highest-probability token.
    #[must_use]
    pub fn greedy() -> Option<Self> {
        let p = unsafe { sys::llama_sampler_init_greedy() };
        (!p.is_null()).then(|| unsafe { Self::from_raw(p) })
    }

    /// Uniform random sampling with a fixed seed.
    #[must_use]
    pub fn dist(seed: u32) -> Option<Self> {
        let p = unsafe { sys::llama_sampler_init_dist(seed) };
        (!p.is_null()).then(|| unsafe { Self::from_raw(p) })
    }

    /// Top-K sampling.
    #[must_use]
    pub fn top_k(k: i32) -> Option<Self> {
        let p = unsafe { sys::llama_sampler_init_top_k(k) };
        (!p.is_null()).then(|| unsafe { Self::from_raw(p) })
    }

    /// Nucleus (top-P) sampling.
    #[must_use]
    pub fn top_p(p: f32, min_keep: usize) -> Option<Self> {
        let ptr = unsafe { sys::llama_sampler_init_top_p(p, min_keep) };
        (!ptr.is_null()).then(|| unsafe { Self::from_raw(ptr) })
    }

    /// Tail-free sampling.
    #[must_use]
    pub fn tail_free(z: f32, min_keep: usize) -> Option<Self> {
        let ptr = tail_free_raw(z, min_keep);
        (!ptr.is_null()).then(|| unsafe { Self::from_raw(ptr) })
    }

    /// Min-P sampling.
    #[must_use]
    pub fn min_p(p: f32, min_keep: usize) -> Option<Self> {
        let ptr = unsafe { sys::llama_sampler_init_min_p(p, min_keep) };
        (!ptr.is_null()).then(|| unsafe { Self::from_raw(ptr) })
    }

    /// Locally-typical sampling.
    #[must_use]
    pub fn typical(p: f32, min_keep: usize) -> Option<Self> {
        let ptr = unsafe { sys::llama_sampler_init_typical(p, min_keep) };
        (!ptr.is_null()).then(|| unsafe { Self::from_raw(ptr) })
    }

    /// Temperature scaling.
    #[must_use]
    pub fn temp(t: f32) -> Option<Self> {
        let p = unsafe { sys::llama_sampler_init_temp(t) };
        (!p.is_null()).then(|| unsafe { Self::from_raw(p) })
    }

    /// Dynamic temperature (scaled by entropy).
    #[must_use]
    pub fn temp_ext(t: f32, delta: f32, exponent: f32) -> Option<Self> {
        let p = unsafe { sys::llama_sampler_init_temp_ext(t, delta, exponent) };
        (!p.is_null()).then(|| unsafe { Self::from_raw(p) })
    }

    /// Exclude Top Choices (XTC).
    #[must_use]
    pub fn xtc(p: f32, t: f32, min_keep: usize, seed: u32) -> Option<Self> {
        let ptr = unsafe { sys::llama_sampler_init_xtc(p, t, min_keep, seed) };
        (!ptr.is_null()).then(|| unsafe { Self::from_raw(ptr) })
    }

    /// Top-N-Sigma sampling.
    #[must_use]
    pub fn top_n_sigma(n: f32) -> Option<Self> {
        let p = unsafe { sys::llama_sampler_init_top_n_sigma(n) };
        (!p.is_null()).then(|| unsafe { Self::from_raw(p) })
    }

    /// Mirostat v1 sampling. Requires the model's vocab size.
    #[must_use]
    pub fn mirostat(n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> Option<Self> {
        let p = unsafe { sys::llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m) };
        (!p.is_null()).then(|| unsafe { Self::from_raw(p) })
    }

    /// Mirostat v2 sampling.
    #[must_use]
    pub fn mirostat_v2(seed: u32, tau: f32, eta: f32) -> Option<Self> {
        let p = unsafe { sys::llama_sampler_init_mirostat_v2(seed, tau, eta) };
        (!p.is_null()).then(|| unsafe { Self::from_raw(p) })
    }

    /// Repetition / frequency / presence penalties.
    #[must_use]
    pub fn penalties(
        n_vocab: i32,
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> Option<Self> {
        let p = unsafe {
            sys::llama_sampler_init_penalties(
                n_vocab,
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_present,
            )
        };
        (!p.is_null()).then(|| unsafe { Self::from_raw(p) })
    }

    /// Don't Repeat Yourself (DRY) sampler.
    ///
    /// # Safety
    /// `seq_breakers` must outlive the sampler and remain valid.
    pub unsafe fn dry(
        model: &LlamaModel,
        multiplier: f32,
        base: f32,
        allowed_length: i32,
        penalty_last_n: i32,
        seq_breakers: &[&str],
    ) -> Option<Self> {
        let breakers: Vec<std::ffi::CString> = seq_breakers
            .iter()
            .map(|s| std::ffi::CString::new(*s).unwrap())
            .collect();
        let mut ptrs: Vec<*const std::os::raw::c_char> =
            breakers.iter().map(|c| c.as_ptr()).collect();
        let p = unsafe {
            sys::llama_sampler_init_dry(
                model.vocab(),
                multiplier,
                base,
                allowed_length,
                penalty_last_n,
                ptrs.as_mut_ptr(),
                ptrs.len(),
            )
        };
        (!p.is_null()).then(|| unsafe { Self::from_raw(p) })
    }

    /// Adaptive probabilistic sampling.
    #[must_use]
    pub fn adaptive_p(target: f32, decay: f32, seed: u32) -> Option<Self> {
        let p = unsafe { sys::llama_sampler_init_adaptive_p(target, decay, seed) };
        (!p.is_null()).then(|| unsafe { Self::from_raw(p) })
    }

    /// Logit-bias sampler.
    ///
    /// # Safety
    /// The `biases` slice must outlive the sampler.
    pub unsafe fn logit_bias(n_vocab: i32, biases: &[sys::llama_logit_bias]) -> Option<Self> {
        let p = unsafe {
            sys::llama_sampler_init_logit_bias(n_vocab, biases.len() as i32, biases.as_ptr())
        };
        (!p.is_null()).then(|| unsafe { Self::from_raw(p) })
    }

    /// Code-infill sampler (requires the model vocab).
    ///
    /// # Safety
    /// `model` must be a valid, live, unaliased `LlamaModel`.
    pub unsafe fn infill(model: &LlamaModel) -> Option<Self> {
        let p = unsafe { sys::llama_sampler_init_infill(model.vocab()) };
        (!p.is_null()).then(|| unsafe { Self::from_raw(p) })
    }
}

/// Required to expose `LlamaModel` to `dry` and `infill` via the
/// `super::LlamaSampler` type — they call `model.vocab()` which is
/// `pub(crate)` on `LlamaModel`.
#[doc(hidden)]
pub struct _ModelTokenBridge(LlamaToken);

#[derive(Debug, Clone, Copy)]
struct TailFreeSampler {
    z: f32,
    min_keep: usize,
}

static TAIL_FREE_NAME: &[u8] = b"tail-free\0";

static mut TAIL_FREE_IFACE: sys::llama_sampler_i = sys::llama_sampler_i {
    name: Some(tail_free_name),
    accept: None,
    apply: Some(tail_free_apply),
    reset: None,
    clone: Some(tail_free_clone),
    free: Some(tail_free_free),
    backend_init: None,
    backend_accept: None,
    backend_apply: None,
    backend_set_input: None,
    backend_reset: None,
    copy_state: None,
};

fn tail_free_raw(z: f32, min_keep: usize) -> *mut sys::llama_sampler {
    let ctx = Box::into_raw(Box::new(TailFreeSampler { z, min_keep }));
    let sampler = unsafe { sys::llama_sampler_init(&raw mut TAIL_FREE_IFACE, ctx.cast()) };
    if sampler.is_null() {
        unsafe {
            drop(Box::from_raw(ctx));
        }
    }
    sampler
}

unsafe extern "C" fn tail_free_name(_smpl: *const sys::llama_sampler) -> *const c_char {
    TAIL_FREE_NAME.as_ptr().cast()
}

unsafe extern "C" fn tail_free_apply(
    smpl: *mut sys::llama_sampler,
    cur_p: *mut sys::llama_token_data_array,
) {
    if smpl.is_null() || cur_p.is_null() {
        return;
    }
    let ctx = unsafe { &*((*smpl).ctx.cast::<TailFreeSampler>()) };
    if ctx.z >= 1.0 || ctx.z <= 0.0 {
        return;
    }
    let cur = unsafe { &mut *cur_p };
    if cur.size <= ctx.min_keep || cur.data.is_null() {
        return;
    }
    let candidates = unsafe { std::slice::from_raw_parts_mut(cur.data, cur.size) };
    apply_tail_free(candidates, ctx.z, ctx.min_keep);
}

unsafe extern "C" fn tail_free_clone(smpl: *const sys::llama_sampler) -> *mut sys::llama_sampler {
    if smpl.is_null() {
        return std::ptr::null_mut();
    }
    let ctx = unsafe { &*((*smpl).ctx.cast::<TailFreeSampler>()) };
    tail_free_raw(ctx.z, ctx.min_keep)
}

unsafe extern "C" fn tail_free_free(smpl: *mut sys::llama_sampler) {
    if smpl.is_null() {
        return;
    }
    let ctx = unsafe { (*smpl).ctx.cast::<TailFreeSampler>() };
    if !ctx.is_null() {
        unsafe {
            drop(Box::from_raw(ctx));
        }
    }
}

fn apply_tail_free(candidates: &mut [sys::llama_token_data], z: f32, min_keep: usize) {
    let max_logit = candidates
        .iter()
        .map(|candidate| candidate.logit)
        .filter(|logit| logit.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return;
    }

    let mut ranked: Vec<(usize, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let prob = if candidate.logit.is_finite() {
                (candidate.logit - max_logit).exp()
            } else {
                0.0
            };
            (index, prob)
        })
        .collect();
    let sum: f32 = ranked.iter().map(|(_, prob)| *prob).sum();
    if sum <= 0.0 || !sum.is_finite() {
        return;
    }
    for (_, prob) in &mut ranked {
        *prob /= sum;
    }
    ranked.sort_by(|(_, left), (_, right)| right.total_cmp(left));

    let mut probs: Vec<f32> = ranked.iter().map(|(_, prob)| *prob).collect();
    tail_free_filter_probs(&mut probs, z, min_keep);

    for ((index, _), prob) in ranked.into_iter().zip(probs) {
        candidates[index].p = prob;
        if prob == 0.0 {
            candidates[index].logit = f32::NEG_INFINITY;
        }
    }
}

fn tail_free_filter_probs(probs: &mut [f32], z: f32, min_keep: usize) {
    if z >= 1.0 || z <= 0.0 || probs.len() <= min_keep || probs.len() < 3 {
        return;
    }

    let first_derivatives: Vec<f32> = probs
        .windows(2)
        .map(|window| (window[0] - window[1]).abs())
        .collect();
    let second_derivatives: Vec<f32> = first_derivatives
        .windows(2)
        .map(|window| (window[0] - window[1]).abs())
        .collect();
    let derivative_sum: f32 = second_derivatives.iter().sum();
    if derivative_sum <= 0.0 || !derivative_sum.is_finite() {
        return;
    }

    let mut keep = probs.len();
    let min_keep = min_keep.max(1).min(probs.len());
    let mut cumulative = 0.0_f32;
    for (index, second_derivative) in second_derivatives.iter().enumerate() {
        cumulative += second_derivative / derivative_sum;
        let candidate_keep = index + 2;
        if cumulative > z && candidate_keep >= min_keep {
            keep = candidate_keep;
            break;
        }
    }

    for prob in probs.iter_mut().skip(keep) {
        *prob = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::tail_free_filter_probs;

    #[test]
    fn tail_free_filter_truncates_probability_tail() {
        let mut probs = vec![0.55, 0.25, 0.12, 0.05, 0.03];

        tail_free_filter_probs(&mut probs, 0.5, 1);

        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
        assert_eq!(probs[2], 0.0);
        assert_eq!(probs[3], 0.0);
        assert_eq!(probs[4], 0.0);
    }

    #[test]
    fn tail_free_filter_respects_min_keep() {
        let mut probs = vec![0.55, 0.25, 0.12, 0.05, 0.03];

        tail_free_filter_probs(&mut probs, 0.1, 3);

        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
        assert!(probs[2] > 0.0);
        assert_eq!(probs[3], 0.0);
        assert_eq!(probs[4], 0.0);
    }
}
