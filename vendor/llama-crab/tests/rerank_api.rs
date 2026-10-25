//! Cross-encoder rerank smoke test.
//!
//! Exercises `Llama::rerank` from the high-level API and verifies the
//! `encode()` path works correctly after the Box+NonNull refactor.
//!
//! Skip if the model is not found. Set `LLAMA_CRAB_RERANK_PATH` env var
//! or place the GGUF at `models/bge-reranker-base-q4_k_m.gguf`.

use llama_crab::context::params::PoolingType;
use llama_crab::{Llama, LlamaParams};

mod common;

const QUERY: &str = "safe systems programming language";
const DOCUMENTS: &[&str] = &[
    "Rust is a memory-safe systems programming language.",
    "Paris is the capital city of France.",
    "Bananas are yellow fruit rich in potassium.",
];

#[test]
fn rerank_scores_documents_and_top_match_is_rust() {
    let Some(model_path) =
        common::resolve_path("LLAMA_CRAB_RERANK_PATH", common::RERANK_DEFAULT_PATH)
    else {
        eprintln!(
            "skipping rerank_api: model not found. \
             Set LLAMA_CRAB_RERANK_PATH or place the GGUF at {}",
            common::RERANK_DEFAULT_PATH
        );
        return;
    };
    common::banner("rerank_scores_documents_and_top_match_is_rust", &model_path);

    let mut llama = Llama::load(
        LlamaParams::new(&model_path)
            .with_n_ctx(512)
            .with_embeddings(true)
            .with_pooling_type(PoolingType::Rank),
    )
    .expect("failed to load reranker model");

    let scores = llama.rerank(QUERY, DOCUMENTS).expect("rerank call failed");

    assert_eq!(
        scores.len(),
        DOCUMENTS.len(),
        "must return one score per document"
    );

    // All scores should be finite.
    for (i, &s) in scores.iter().enumerate() {
        assert!(
            s.is_finite(),
            "score[{i}]={s} is not finite (doc: {:?})",
            DOCUMENTS[i]
        );
    }

    // Rust document should score highest for "safe systems programming language".
    let rust_idx = 0; // index of the Rust document
    let paris_idx = 1;
    let bananas_idx = 2;
    assert!(
        scores[rust_idx] > scores[paris_idx],
        "Rust doc (score={}) should rank above Paris doc (score={})",
        scores[rust_idx],
        scores[paris_idx]
    );
    assert!(
        scores[rust_idx] > scores[bananas_idx],
        "Rust doc (score={}) should rank above Bananas doc (score={})",
        scores[rust_idx],
        scores[bananas_idx]
    );
}

#[test]
fn rerank_empty_documents_returns_empty_vec() {
    let Some(model_path) =
        common::resolve_path("LLAMA_CRAB_RERANK_PATH", common::RERANK_DEFAULT_PATH)
    else {
        eprintln!("skipping rerank_api (empty): model not found");
        return;
    };
    let mut llama = Llama::load(
        LlamaParams::new(&model_path)
            .with_n_ctx(512)
            .with_embeddings(true)
            .with_pooling_type(PoolingType::Rank),
    )
    .expect("failed to load reranker model");

    let scores: &[&str] = &[];
    let result = llama
        .rerank("any query", scores)
        .expect("empty rerank should succeed");
    assert!(
        result.is_empty(),
        "expected empty vec, got {}",
        result.len()
    );
}
