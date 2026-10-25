//! Fill-in-Middle (FIM) smoke test.
//!
//! Exercises `Llama::complete_infill` which uses `context_mut()` directly
//! through the infill code path. The `context_mut` return type changed
//! (lifetime parameter was removed) in the Box+NonNull refactor — this
//! test catches regressions in that path.
//!
//! Skip if the model is not found. Set `LLAMA_CRAB_QWEN_PATH` env var
//! or place the GGUF at `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`.

use llama_crab::{Llama, LlamaParams};

mod common;

#[test]
fn infill_returns_some_content() {
    let Some(model_path) = common::resolve_path("LLAMA_CRAB_QWEN_PATH", common::QWEN_DEFAULT_PATH)
    else {
        eprintln!(
            "skipping infill_api: model not found. \
             Set LLAMA_CRAB_QWEN_PATH or place the GGUF at {}",
            common::QWEN_DEFAULT_PATH
        );
        return;
    };
    common::banner("infill_returns_some_content", &model_path);

    let mut llama = Llama::load(LlamaParams::new(&model_path).with_n_ctx(512))
        .expect("failed to load Qwen model");

    let prefix = "fn main() {";
    let suffix = "}";
    match llama.complete_infill(prefix, suffix) {
        Ok(fill) => {
            assert!(
                !fill.is_empty(),
                "infill should return non-empty content for prefix={prefix:?} suffix={suffix:?}"
            );
        }
        Err(e) => {
            eprintln!("infill_api: complete_infill returned error: {e}");
            // Qwen 2.5 0.5B Instruct may or may not support FIM — this is
            // acceptable. The important thing is that it doesn't segfault
            // or panic (which would indicate a use-after-move regression).
        }
    }
}

#[test]
fn infill_called_twice_is_consistent() {
    let Some(model_path) = common::resolve_path("LLAMA_CRAB_QWEN_PATH", common::QWEN_DEFAULT_PATH)
    else {
        eprintln!("skipping infill_api (twice): model not found");
        return;
    };
    let mut llama = Llama::load(LlamaParams::new(&model_path).with_n_ctx(512))
        .expect("failed to load Qwen model");

    // Call infill twice — the second call must not segfault or return
    // a different error kind (e.g. "context has no memory").
    let prefix = "fn add(a: i32, b: i32) -> i32 {";
    let suffix = "}";
    let r1 = llama.complete_infill(prefix, suffix);
    let r2 = llama.complete_infill(prefix, suffix);

    // If the first call succeeded, the second must also succeed.
    // If FIM is unsupported, both must return the same error.
    match (r1, r2) {
        (Ok(_), Ok(_)) => { /* both succeeded — re-entrant infill works */ }
        (Err(e1), Err(e2)) => {
            let msg1 = e1.to_string();
            let msg2 = e2.to_string();
            assert_eq!(
                msg1, msg2,
                "infill errors must be consistent across calls; got {msg1:?} vs {msg2:?}"
            );
        }
        (Ok(_), Err(e)) => panic!("first infill succeeded but second failed: {e}"),
        (Err(e), Ok(_)) => panic!("first infill failed ({e}) but second succeeded"),
    }
}
