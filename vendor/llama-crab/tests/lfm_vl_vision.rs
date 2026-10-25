//! End-to-end vision test using
//! [`unsloth/LFM2.5-VL-1.6B-GGUF`].
//!
//! Skips if the model or a test image is not present. Set
//! `LLAMA_CRAB_LFM_VL_PATH` (and `LLAMA_CRAB_LFM_VL_MMPROJ_PATH`) or
//! place the files in the conventional locations:
//!
//!   * `models/LFM2.5-VL-1.6B-Q4_K_M.gguf`
//!   * `models/LFM2.5-VL-1.6B-mmproj.gguf`
//!   * `tests/fixtures/test_image.png` (any 256×256+ RGB image works)
//!
//! [`unsloth/LFM2.5-VL-1.6B-GGUF`]: https://huggingface.co/unsloth/LFM2.5-VL-1.6B-GGUF

#![cfg(feature = "mtmd")]

use llama_crab::batch::LlamaBatch;
use llama_crab::chat::{render_builtin, BuiltinTemplate, ChatMessage};
use llama_crab::multimodal::{MtmdBitmap, MtmdContext, MtmdInputText};
use llama_crab::sampling::LlamaSampler;
use llama_crab::token::LlamaToken;
use llama_crab::{Llama, LlamaParams, Role};
use std::time::Instant;

mod common;

#[test]
fn lfm_vl_vision_question_answering() {
    let Some(model_path) =
        common::resolve_path("LLAMA_CRAB_LFM_VL_PATH", common::LFM_VL_DEFAULT_PATH)
    else {
        eprintln!(
            "skipping lfm_vl_vision_question_answering: model not found. \
             Set LLAMA_CRAB_LFM_VL_PATH or place the GGUF at {}",
            common::LFM_VL_DEFAULT_PATH
        );
        return;
    };
    let Some(mmproj_path) = common::resolve_path(
        "LLAMA_CRAB_LFM_VL_MMPROJ_PATH",
        common::LFM_VL_MMPROJ_DEFAULT_PATH,
    ) else {
        eprintln!(
            "skipping lfm_vl_vision_question_answering: mmproj not found. \
             Set LLAMA_CRAB_LFM_VL_MMPROJ_PATH or place it at {}",
            common::LFM_VL_MMPROJ_DEFAULT_PATH
        );
        return;
    };
    let Some(image_path) =
        common::resolve_path("LLAMA_CRAB_TEST_IMAGE", common::TEST_IMAGE_DEFAULT_PATH)
    else {
        eprintln!(
            "skipping lfm_vl_vision_question_answering: test image not found. \
             Set LLAMA_CRAB_TEST_IMAGE or place a PNG at {}",
            common::TEST_IMAGE_DEFAULT_PATH
        );
        return;
    };
    common::banner("lfm_vl_vision_question_answering", &model_path);

    // 1. Load the text model.
    let mut llama =
        Llama::load(LlamaParams::new(&model_path).with_n_ctx(4096)).expect("load LFM2.5-VL");
    eprintln!("LFM2.5-VL loaded: {} layers", llama.model().n_layer());

    // 2. Initialize the multimodal context.
    let mtmd = MtmdContext::init_from_file(&mmproj_path, llama.model()).expect("mtmd init");
    assert!(mtmd.support_vision(), "LFM2.5-VL should support vision");
    eprintln!("mmproj supports vision: OK");

    // 3. Decode the image.
    let bitmap = MtmdBitmap::from_file(&image_path).expect("decode image");
    eprintln!(
        "decoded image: {}x{} ({} bytes, audio={})",
        bitmap.nx(),
        bitmap.ny(),
        bitmap.n_bytes(),
        bitmap.is_audio()
    );
    assert!(bitmap.nx() > 0 && bitmap.ny() > 0);
    assert!(!bitmap.is_audio());

    // 4. Tokenize a prompt + image together.
    let marker = llama_crab::multimodal::default_media_marker();
    let prompt = render_builtin(
        BuiltinTemplate::ChatMl,
        &[ChatMessage::new(
            Role::User,
            format!("{marker}\nWhat do you see in this image? Answer briefly."),
        )],
        &[],
        true,
    );
    let chunks = mtmd
        .tokenize(MtmdInputText::new(&prompt), &[&bitmap])
        .expect("tokenize");
    eprintln!("tokenized into {} chunks", chunks.len());
    assert!(!chunks.is_empty(), "should produce at least one chunk");

    // 5. Evaluate the chunks on the model context.
    let ctx_ptr = llama.context().raw_handle();
    let new_n_past = unsafe {
        chunks
            .eval(&mtmd, ctx_ptr, 0, 0, llama.context().n_batch() as i32, true)
            .expect("eval chunks")
    };
    eprintln!("eval consumed {new_n_past} positions");

    // 6. Sample a response (greedy).
    let start = Instant::now();
    let mut sampler = LlamaSampler::greedy().expect("greedy");
    let mut out = String::new();
    let eos = llama.model().token_eos();
    for n_generated in 0..64 {
        let tok: LlamaToken = unsafe { sampler.sample(ctx_ptr, -1) };
        sampler.accept(tok);
        if tok == eos {
            break;
        }
        if let Ok(piece) = llama.model().detokenize(&[tok], false) {
            out.push_str(&piece);
        }
        let single = LlamaBatch::one(tok, new_n_past + n_generated, 0, true);
        llama
            .context()
            .decode(&single)
            .expect("decode generated token");
    }
    let elapsed = start.elapsed();
    eprintln!("vision answer ({:?}): {:?}", elapsed, out);
    assert!(
        !out.trim().is_empty(),
        "vision model should produce non-whitespace text"
    );
}
