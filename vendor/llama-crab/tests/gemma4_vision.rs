//! End-to-end vision test with the Gemma 4 multimodal model.
//!
//! Gemma 4 is a **text+vision** model from Google. Its GGUF (from
//! `lmstudio-community/gemma-4-E4B-it-GGUF`) ships a paired `mmproj`
//! projector file. This test loads both, attaches an image, and verifies
//! the multimodal pipeline runs end-to-end.

#![cfg(feature = "mtmd")]

use llama_crab::batch::LlamaBatch;
use llama_crab::multimodal::{default_media_marker, MtmdBitmap, MtmdContext, MtmdInputText};
use llama_crab::sampling::LlamaSampler;
use llama_crab::token::LlamaToken;
use llama_crab::{Llama, LlamaParams};
use std::time::Instant;

mod common;

#[test]
fn gemma4_vision_question_answering() {
    let Some(model_path) =
        common::resolve_path("LLAMA_CRAB_GEMMA4_PATH", common::GEMMA4_DEFAULT_PATH)
    else {
        eprintln!(
            "skipping gemma4_vision_question_answering: model not found at {}",
            common::GEMMA4_DEFAULT_PATH
        );
        return;
    };
    let Some(mmproj_path) = common::resolve_path(
        "LLAMA_CRAB_GEMMA4_MMPROJ_PATH",
        common::GEMMA4_MMPROJ_DEFAULT_PATH,
    ) else {
        eprintln!(
            "skipping gemma4_vision_question_answering: mmproj not found at {}",
            common::GEMMA4_MMPROJ_DEFAULT_PATH
        );
        return;
    };
    let Some(image_path) =
        common::resolve_path("LLAMA_CRAB_TEST_IMAGE", common::TEST_IMAGE_DEFAULT_PATH)
    else {
        eprintln!("skipping gemma4_vision_question_answering: test image missing");
        return;
    };
    common::banner("gemma4_vision_question_answering", &model_path);

    let mut llama =
        Llama::load(LlamaParams::new(&model_path).with_n_ctx(4096)).expect("load Gemma 4");
    eprintln!("Gemma 4: {} layers", llama.model().n_layer());

    let mtmd = MtmdContext::init_from_file(&mmproj_path, llama.model()).expect("mtmd init");
    assert!(mtmd.support_vision());

    let bitmap = MtmdBitmap::from_file(&image_path).expect("decode image");
    eprintln!("image: {}x{}", bitmap.nx(), bitmap.ny());

    let prompt = format!(
        "{}\nDescribe this image in one sentence.",
        default_media_marker()
    );
    let chunks = mtmd
        .tokenize(MtmdInputText::new(&prompt), &[&bitmap])
        .expect("tokenize");
    assert!(!chunks.is_empty());

    let ctx_ptr = llama.context().raw_handle();
    let new_n_past = unsafe {
        chunks
            .eval(&mtmd, ctx_ptr, 0, 0, llama.context().n_batch() as i32, true)
            .expect("eval")
    };
    eprintln!("eval consumed {new_n_past} positions");

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
    eprintln!("Gemma 4 vision answer ({:?}): {:?}", start.elapsed(), out);
    assert!(!out.trim().is_empty());
}
