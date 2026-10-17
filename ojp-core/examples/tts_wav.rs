//! Synthesize one sentence to a WAV file without playing it:
//! cargo run --release -p ojp-core --features tts --example tts_wav -- MODEL VOICES OUT.wav "text"
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let engine = rt.block_on(kokoro_micro::TtsEngine::with_paths(&a[1], &a[2])).unwrap();
    let t = std::time::Instant::now();
    let audio = engine.synthesize_with_options(&a[4], Some(ojp_core::tts::DEFAULT_VOICE), 1.0, 1.0, None).unwrap();
    println!("{} samples ({:.1} s of audio) in {:.1} s", audio.len(), audio.len() as f32 / 24000.0, t.elapsed().as_secs_f32());
    engine.save_wav(&a[3], &audio).unwrap();
}
