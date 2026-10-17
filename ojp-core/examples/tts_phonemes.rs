//! Print the phonemes the voice would say: ... --example tts_phonemes -- MODEL VOICES voice "text"
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let engine = rt.block_on(kokoro_micro::TtsEngine::with_paths(&a[1], &a[2])).unwrap();
    println!("{}", engine.phonemize(&a[4], Some(&a[3])).unwrap());
}
