//! End-to-end text-completion test using
//! [`lmstudio-community/gemma-4-E4B-it-GGUF`].
//!
//! Skips if the model file is not present. Set `LLAMA_CRAB_GEMMA4_PATH`
//! or place the GGUF in `models/gemma-4-E4B-it-Q4_K_M.gguf` to enable it.
//!
//! [`lmstudio-community/gemma-4-E4B-it-GGUF`]: https://huggingface.co/lmstudio-community/gemma-4-E4B-it-GGUF

use llama_crab::{Llama, LlamaParams};
use std::time::Instant;

mod common;

#[test]
fn gemma4_text_completion_smoke() {
    let Some(model_path) =
        common::resolve_path("LLAMA_CRAB_GEMMA4_PATH", common::GEMMA4_DEFAULT_PATH)
    else {
        eprintln!(
            "skipping gemma4_text_completion_smoke: model not found. \
             Set LLAMA_CRAB_GEMMA4_PATH or place the GGUF at {}",
            common::GEMMA4_DEFAULT_PATH
        );
        return;
    };
    common::banner("gemma4_text_completion_smoke", &model_path);

    let mut llama = Llama::load(
        LlamaParams::new(&model_path)
            .with_n_ctx(2048)
            .with_n_threads(4),
    )
    .expect("failed to load Gemma 4");

    let n_vocab = llama.model().n_vocab();
    let n_ctx_train = llama.model().n_ctx_train();
    let n_layer = llama.model().n_layer();
    let n_embd = llama.model().n_embd();
    eprintln!("model: vocab={n_vocab} ctx_train={n_ctx_train} layer={n_layer} embd={n_embd}");
    assert!(n_vocab > 0);
    assert!(n_layer > 0);
    assert!(n_embd > 0);

    // Tokenize a simple prompt and assert we get a non-empty sequence.
    let tokens = llama
        .model()
        .tokenize("Hello, world", true, false)
        .expect("tokenize");
    eprintln!("token count for 'Hello, world': {}", tokens.len());
    assert!(!tokens.is_empty());

    // Greedy single completion.
    let start = Instant::now();
    let resp = llama
        .create_completion("The capital of France is", 24)
        .expect("create_completion");
    let elapsed = start.elapsed();
    eprintln!(
        "completion ({} tokens, {:?}): {:?}",
        resp.n_tokens, elapsed, resp.text
    );
    assert!(resp.n_tokens > 0, "should produce at least one token");
    assert!(!resp.text.is_empty(), "completion should have text");
}

#[test]
fn gemma4_round_trip_tokenize_detokenize() {
    let Some(model_path) =
        common::resolve_path("LLAMA_CRAB_GEMMA4_PATH", common::GEMMA4_DEFAULT_PATH)
    else {
        return;
    };
    common::banner("gemma4_round_trip_tokenize_detokenize", &model_path);

    let llama =
        Llama::load(LlamaParams::new(&model_path).with_n_ctx(512)).expect("failed to load Gemma 4");

    let prompt = "Llama-crab is a Rust binding for llama.cpp";
    let tokens = llama
        .model()
        .tokenize(prompt, true, false)
        .expect("tokenize");
    let reconstructed = llama
        .model()
        .detokenize(&tokens, false)
        .expect("detokenize");
    eprintln!("tokens  : {tokens:?}");
    eprintln!("reconst : {reconstructed:?}");
    assert!(!tokens.is_empty());
    // The detokenized text should contain the prompt's words (it may have
    // spaces and BOS markers, so we only check the alphanumeric content).
    for word in prompt.split_whitespace() {
        let needle: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
        if needle.is_empty() {
            continue;
        }
        let haystack: String = reconstructed
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect();
        assert!(
            haystack.to_lowercase().contains(&needle.to_lowercase()),
            "round-trip lost the word {word:?} (have {reconstructed:?})"
        );
    }
}

#[test]
fn gemma4_chat_completion_returns_assistant_message() {
    use llama_crab::high_level::chat_completion::ChatMessage;
    use llama_crab::Role;

    let Some(model_path) =
        common::resolve_path("LLAMA_CRAB_GEMMA4_PATH", common::GEMMA4_DEFAULT_PATH)
    else {
        return;
    };
    common::banner(
        "gemma4_chat_completion_returns_assistant_message",
        &model_path,
    );

    let mut llama = Llama::load(LlamaParams::new(&model_path).with_n_ctx(2048)).expect("load");
    let history = vec![
        ChatMessage::new(Role::System, "You are a concise assistant."),
        ChatMessage::new(Role::User, "Say 'OK' and nothing else."),
    ];
    let resp = llama
        .create_chat_completion(&history, 8)
        .expect("create_chat_completion");
    eprintln!("assistant> {}", resp.content);
    assert!(!resp.content.is_empty());
}
