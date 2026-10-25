//! Streaming completion smoke test.
//!
//! Exercises `Llama::create_completion_stream` which drives the decode
//! loop through a callback. The context decode path exercises the
//! kv_cache and sampling_state modules — both of which dereference
//! the raw model pointer changed in the Box+NonNull refactor.
//!
//! Skip if the model is not found. Set `LLAMA_CRAB_QWEN_PATH` env var
//! or place the GGUF at `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`.

use llama_crab::{CompletionOptions, Llama, LlamaParams, StreamControl};

mod common;

#[test]
fn streaming_completion_collects_tokens() {
    let Some(model_path) = common::resolve_path("LLAMA_CRAB_QWEN_PATH", common::QWEN_DEFAULT_PATH)
    else {
        eprintln!(
            "skipping streaming_api: model not found. \
             Set LLAMA_CRAB_QWEN_PATH or place the GGUF at {}",
            common::QWEN_DEFAULT_PATH
        );
        return;
    };
    common::banner("streaming_completion_collects_tokens", &model_path);

    let mut llama = Llama::load(LlamaParams::new(&model_path).with_n_ctx(512))
        .expect("failed to load Qwen model");

    let mut collected = String::new();
    let completion = llama
        .create_completion_stream(
            "The capital of France is",
            CompletionOptions::new(16),
            |chunk| {
                collected.push_str(&chunk.text);
                StreamControl::Continue
            },
        )
        .expect("streaming completion failed");

    assert!(
        !collected.is_empty(),
        "streaming completion must produce at least one token of text"
    );
    assert!(
        completion.n_tokens > 0,
        "completion metadata must report n_tokens > 0"
    );
    assert!(
        completion.text.contains("Paris") || completion.text.len() > 5,
        "completion should contain a recognizable answer or at least some text: {:?}",
        completion.text
    );
}

#[test]
fn streaming_completion_can_stop_early() {
    let Some(model_path) = common::resolve_path("LLAMA_CRAB_QWEN_PATH", common::QWEN_DEFAULT_PATH)
    else {
        eprintln!("skipping streaming_api (stop): model not found");
        return;
    };
    let mut llama = Llama::load(LlamaParams::new(&model_path).with_n_ctx(512))
        .expect("failed to load Qwen model");

    let mut chunks_seen = 0_u32;
    let completion = llama
        .create_completion_stream(
            "Write a very long essay about cats.",
            CompletionOptions::new(64),
            |_chunk| {
                chunks_seen += 1;
                // Stop after the first chunk.
                StreamControl::Stop
            },
        )
        .expect("streaming completion failed");

    assert!(
        (1..=2).contains(&chunks_seen),
        "should have seen 1-2 chunks before stopping (llama.cpp may fire \
         one extra token before acknowledging StreamControl::Stop); saw {chunks_seen}"
    );
    // The returned completion should reflect the stopped state.
    assert!(
        completion.n_tokens > 0,
        "at least 1 token was generated before stop"
    );
    assert!(
        !completion.text.is_empty(),
        "stopped completion should still contain the chunk text"
    );
}
