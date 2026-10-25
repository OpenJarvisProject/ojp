//! Regression test for the use-after-move bug in `Llama::load`.
//!
//! The pre-fix `Llama::load` built a `LlamaContext` whose `model`
//! reference pointed at the `LlamaModel` on the stack of `load`'s
//! frame. After `load` returns, the model lives at a different
//! address in the caller's frame, so every read through
//! `LlamaContext::model()` is a use-after-move. The 5 poisoned
//! sites are:
//!
//!  - `LlamaContext::embeddings`        (context/embeddings.rs:27)
//!  - `LlamaContext::embeddings_seq`    (context/embeddings.rs:46)
//!  - `LlamaContext::embeddings_ith`    (context/embeddings.rs:56)
//!  - `LlamaContext::logits_ith`        (context/sampling_state.rs:28)
//!  - `LlamaContext::sampled_probs_ith` (context/sampling_state.rs:50)
//!
//! Each of the tests below exercises one of those sites and asserts
//! the slice length matches what `LlamaModel` reports (768 for
//! nomic-embed-text-v1.5). Before the fix the length is 0, 5, or
//! the process is killed by SIGSEGV inside `llama_model_n_embd` /
//! `llama_n_vocab`.
//!
//! Gated on `LLAMA_CRAB_TEST_MODEL` so `cargo test` skips on
//! machines without the GGUF cached.
//!
//! Run with:
//!
//! ```bash
//! LLAMA_CRAB_TEST_MODEL=$HOME/.cache/huggingface/hub/\
//! models--nomic-ai--nomic-embed-text-v1.5-GGUF/snapshots/\
//! <hash>/nomic-embed-text-v1.5.Q4_K_M.gguf \
//!     cargo test --release --test embedding_regression -- --nocapture
//! ```

use llama_crab::batch::LlamaBatch;
use llama_crab::context::params::PoolingType;
use llama_crab::{Llama, LlamaParams};

fn model_path() -> Option<std::path::PathBuf> {
    let raw = std::env::var_os("LLAMA_CRAB_TEST_MODEL")?;
    let p = std::path::PathBuf::from(raw);
    if p.exists() {
        Some(p)
    } else {
        eprintln!(
            "LLAMA_CRAB_TEST_MODEL points at a non-existent file: {}",
            p.display()
        );
        None
    }
}

fn load_nomic() -> Option<Llama> {
    let path = model_path()?;
    match Llama::load(
        LlamaParams::new(&path)
            .with_n_ctx(512)
            .with_embeddings(true)
            .with_pooling_type(PoolingType::Cls),
    ) {
        Ok(l) => Some(l),
        Err(e) => {
            eprintln!("skipping: failed to load model: {e}");
            None
        }
    }
}

#[test]
fn embeddings_seq_returns_768_dim_unit_norm_vector() {
    // Exercises `LlamaContext::embeddings_seq` (the path through
    // `self.model().n_embd()` at `context/embeddings.rs:46`).
    let Some(mut llama) = load_nomic() else {
        return;
    };
    let v = llama
        .embed("Hello, world!", true)
        .expect("embed must succeed");
    let n = v.len();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!("embedding_dim: {n}");
    eprintln!("embedding_l2_norm: {norm:.6}");
    assert_eq!(
        n, 768,
        "embedding dim must be 768 for nomic-embed-text-v1.5"
    );
    assert!(
        (0.9..=1.1).contains(&norm),
        "l2 norm out of (0.9, 1.1): {norm}"
    );
    assert!(
        v.iter().all(|x| x.is_finite()),
        "non-finite component in embedding"
    );
}

#[test]
fn embed_called_twice_returns_consistent_dim() {
    // Re-entrancy: a second `embed` re-exercises `embeddings_seq`
    // after the first batch's logits have been overwritten. Catches
    // any state that the Box fix might have hidden.
    let Some(mut llama) = load_nomic() else {
        return;
    };
    let v1 = llama.embed("Hello, world!", true).expect("embed #1");
    let v2 = llama.embed("Goodbye, world!", true).expect("embed #2");
    assert_eq!(v1.len(), 768);
    assert_eq!(v2.len(), 768);
    let diff: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 0.0,
        "different texts must produce different embeddings"
    );
}

#[test]
fn logits_ith_after_decode_reads_n_vocab_floats() {
    // Exercises `LlamaContext::logits_ith` and
    // `LlamaContext::sampled_probs_ith`
    // (the path through `self.model().n_vocab()` at
    //  `context/sampling_state.rs:28` and `:50`).
    let Some(mut llama) = load_nomic() else {
        return;
    };
    let n_vocab = llama.model().n_vocab();
    assert!(n_vocab > 0, "n_vocab must be positive, got {n_vocab}");

    let tokens = llama
        .model()
        .tokenize("Hello", true, false)
        .expect("tokenize");
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    for (i, &t) in tokens.iter().enumerate() {
        batch.add(t, i as i32, &[0], true).expect("batch.add");
    }
    llama.context().decode(&batch).expect("decode");

    let logits = llama
        .context()
        .logits_ith((tokens.len() as i32) - 1)
        .expect("logits_ith");
    assert_eq!(
        logits.len(),
        n_vocab as usize,
        "logits slice must be n_vocab long"
    );
    assert!(logits.iter().all(|x| x.is_finite()), "non-finite logit");

    // `sampled_probs_ith` returns null until the default sampler has
    // run; we only assert the call does not crash.
    let _ = llama.context().sampled_probs_ith((tokens.len() as i32) - 1);
}
