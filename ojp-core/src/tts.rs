//! Text-to-speech: Kokoro-82M (via the kokoro-micro crate, ONNX on the CPU)
//! played through the default audio output. Entirely optional: without the
//! `tts` feature [`AVAILABLE`] is false and [`Tts::start`] always fails, and
//! the apps simply never speak.
//!
//! Speaking runs on its own thread, a sentence at a time, so a reply starts
//! playing after the first sentence is synthesized and [`Tts::stop`] takes
//! effect between sentences.
use std::path::PathBuf;

/// Whether this build can speak at all.
pub const AVAILABLE: bool = cfg!(feature = "tts");

/// The voice used when the persona's voice is "default": calm British male.
pub const DEFAULT_VOICE: &str = "bm_george";

/// The English voices in Kokoro's voice file, best-rated first within each
/// group (the grades are the model authors' own, from their VOICES.md).
/// `id` is what goes in config.toml's `[tts] voice`.
pub const VOICES: &[(&str, &str)] = &[
    ("af_heart", "Heart: American female (best)"),
    ("af_bella", "Bella: American female"),
    ("af_nicole", "Nicole: American female"),
    ("af_aoede", "Aoede: American female"),
    ("af_kore", "Kore: American female"),
    ("af_sarah", "Sarah: American female"),
    ("af_nova", "Nova: American female"),
    ("af_sky", "Sky: American female"),
    ("af_alloy", "Alloy: American female"),
    ("af_jessica", "Jessica: American female"),
    ("af_river", "River: American female"),
    ("bf_emma", "Emma: British female"),
    ("bf_isabella", "Isabella: British female"),
    ("bf_alice", "Alice: British female"),
    ("bf_lily", "Lily: British female"),
    ("am_michael", "Michael: American male"),
    ("am_fenrir", "Fenrir: American male"),
    ("am_puck", "Puck: American male"),
    ("am_echo", "Echo: American male"),
    ("am_eric", "Eric: American male"),
    ("am_liam", "Liam: American male"),
    ("am_onyx", "Onyx: American male"),
    ("am_adam", "Adam: American male"),
    ("bm_fable", "Fable: British male"),
    ("bm_george", "George: British male (default)"),
    ("bm_lewis", "Lewis: British male"),
    ("bm_daniel", "Daniel: British male"),
];

/// The speeds offered in settings, slowest first.
pub const SPEEDS: &[(f32, &str)] = &[(0.8, "Slower"), (0.9, "Slow"), (1.0, "Normal"), (1.1, "Fast")];

/// Index into [`SPEEDS`] closest to `speed`.
pub fn speed_index(speed: f32) -> usize {
    (0..SPEEDS.len()).min_by(|&a, &b| (SPEEDS[a].0 - speed).abs().total_cmp(&(SPEEDS[b].0 - speed).abs())).unwrap_or(2)
}

/// Index into [`VOICES`] of `voice` ("default" and unknown ids give the default's).
pub fn voice_index(voice: &str) -> usize {
    let id = if voice.is_empty() || voice == "default" { DEFAULT_VOICE } else { voice };
    VOICES.iter().position(|(v, _)| *v == id).or_else(|| VOICES.iter().position(|(v, _)| *v == DEFAULT_VOICE)).unwrap_or(0)
}

/// Number of voices that can be blended.
pub const MIX_SLOTS: usize = 3;

/// A voice blend as the settings screen shows it: for each slot a choice
/// (0 = none, n = `VOICES[n - 1]`) and a strength 0..=10.
pub type MixSlots = [(usize, u32); MIX_SLOTS];

/// Read a `[tts] voice` value ("default", "af_heart", or a blend written
/// `af_heart.6+af_bella.4`, strengths in tenths) into settings slots.
pub fn parse_mix(voice: &str) -> MixSlots {
    let mut slots = [(0, 0); MIX_SLOTS];
    let voice = if voice.is_empty() || voice == "default" { DEFAULT_VOICE } else { voice };
    for (slot, part) in slots.iter_mut().zip(voice.split('+')) {
        let (id, tenths) = match part.split_once('.') {
            Some((id, n)) => (id, n.parse::<u32>().unwrap_or(10).min(10)),
            None => (part, 10),
        };
        if let Some(i) = VOICES.iter().position(|(v, _)| *v == id) {
            *slot = (i + 1, tenths);
        }
    }
    if slots.iter().all(|(v, _)| *v == 0) {
        slots[0] = (voice_index(DEFAULT_VOICE) + 1, 10);
    }
    slots
}

/// The `[tts] voice` value for the settings slots: voices with strength 0
/// or no choice are dropped and the rest are scaled to add up to ten
/// tenths, the unit Kokoro's blend syntax uses. Nothing chosen gives the
/// default voice.
pub fn mix_string(slots: &MixSlots) -> String {
    let chosen: Vec<(&str, u32)> = slots
        .iter()
        .filter(|(v, a)| *v > 0 && *a > 0 && *v <= VOICES.len())
        .map(|&(v, a)| (VOICES[v - 1].0, a))
        .collect();
    let total: u32 = chosen.iter().map(|c| c.1).sum();
    if chosen.is_empty() {
        return DEFAULT_VOICE.to_string();
    }
    if chosen.len() == 1 {
        return chosen[0].0.to_string();
    }
    // Largest-remainder rounding so the tenths add up to exactly 10.
    let mut tenths: Vec<(usize, u32, u32)> = chosen
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.1 * 10 / total, c.1 * 10 % total))
        .collect();
    let mut left = 10 - tenths.iter().map(|t| t.1).sum::<u32>();
    let mut order: Vec<usize> = (0..tenths.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(tenths[i].2));
    for i in order {
        if left == 0 {
            break;
        }
        tenths[i].1 += 1;
        left -= 1;
    }
    let parts: Vec<String> = tenths.iter().filter(|t| t.1 > 0).map(|t| format!("{}.{}", chosen[t.0].0, t.1)).collect();
    if parts.len() == 1 { chosen[tenths.iter().find(|t| t.1 > 0).map_or(0, |t| t.0)].0.to_string() } else { parts.join("+") }
}

/// What to load and how to speak.
#[derive(Debug, Clone)]
pub struct Options {
    pub engine: Engine,
    pub model: PathBuf,
    pub voices: PathBuf,
    /// Kokoro: a voice id or blend (e.g. `bm_george`, `af_heart.6+af_bella.4`),
    /// or "default". MOSS: one of [`MOSS_VOICES`].
    pub voice: String,
    pub speed: f32,
    /// Where `scripts/setup-moss.sh` put the MOSS environment.
    pub moss_dir: PathBuf,
    /// MOSS: the recording the "Custom" voice imitates, if not `reference.wav`.
    pub moss_reference: Option<PathBuf>,
}

/// Which speech engine to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Kokoro,
    /// MOSS-TTS-Nano through a Python sidecar (experimental).
    Moss,
}

impl Engine {
    pub const ALL: [(Engine, &'static str, &'static str); 2] =
        [(Engine::Kokoro, "kokoro", "Kokoro (built in)"), (Engine::Moss, "moss", "MOSS (experimental)")];

    pub fn from_id(id: &str) -> Engine {
        Engine::ALL.iter().find(|(_, i, _)| *i == id).map_or(Engine::Kokoro, |(e, _, _)| *e)
    }

    pub fn id(self) -> &'static str {
        Engine::ALL.iter().find(|(e, _, _)| *e == self).map_or("kokoro", |(_, i, _)| *i)
    }

    pub fn index(self) -> usize {
        Engine::ALL.iter().position(|(e, _, _)| *e == self).unwrap_or(0)
    }
}

/// MOSS-TTS-Nano's bundled English voices (short reference recordings the
/// model imitates; its sample of a real public figure is left out).
pub const MOSS_VOICES: &[(&str, &str)] = &[("Ava", "Ava"), ("Bella", "Bella"), ("Adam", "Adam"), ("Nathan", "Nathan")];

/// The MOSS voice id that means "use `reference.wav`".
pub const CUSTOM_VOICE: &str = "Custom";

/// Prefix of the voice ids for clips in `<moss dir>/references/`.
pub const REFERENCE_PREFIX: &str = "ref:";

/// Every MOSS voice the user can pick as (id, label): the bundled ones, then
/// "My own reference clip" (`<moss dir>/reference.wav`, made by
/// scripts/make-reference.py), then each `<moss dir>/references/<name>.wav`.
pub fn moss_voice_list(moss_dir: &std::path::Path) -> Vec<(String, String)> {
    let mut voices: Vec<(String, String)> = MOSS_VOICES.iter().map(|(i, l)| (i.to_string(), l.to_string())).collect();
    if moss_dir.join("reference.wav").exists() {
        voices.push((CUSTOM_VOICE.into(), "My own reference clip".into()));
    }
    let mut names: Vec<String> = std::fs::read_dir(moss_dir.join("references"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            (path.extension()? == "wav").then(|| path.file_stem().map(|s| s.to_string_lossy().into_owned()))?
        })
        .collect();
    names.sort();
    voices.extend(names.into_iter().map(|n| (format!("{REFERENCE_PREFIX}{n}"), format!("Mine: {n}"))));
    voices
}

/// Index of `voice` in `voices` (the Bella entry if it isn't there).
pub fn moss_voice_index(voices: &[(String, String)], voice: &str) -> usize {
    voices.iter().position(|(v, _)| v == voice).unwrap_or(1)
}

/// The recording to imitate for a MOSS voice id, if it names one.
pub fn moss_reference_path(moss_dir: &std::path::Path, voice: &str) -> Option<PathBuf> {
    if voice == CUSTOM_VOICE {
        Some(moss_dir.join("reference.wav"))
    } else {
        voice.strip_prefix(REFERENCE_PREFIX).map(|name| moss_dir.join("references").join(format!("{name}.wav")))
    }
}

impl Options {
    /// The recording a MOSS `voice` imitates, if it names one.
    pub fn reference_for(&self, voice: &str) -> Option<PathBuf> {
        if voice == CUSTOM_VOICE
            && let Some(path) = &self.moss_reference
        {
            return Some(path.clone());
        }
        moss_reference_path(&self.moss_dir, voice)
    }

    pub fn files_present(&self) -> bool {
        match self.engine {
            Engine::Kokoro => self.model.exists() && self.voices.exists(),
            Engine::Moss => self.moss_dir.join("venv/bin/python").exists() && self.moss_dir.join("moss_server.py").exists(),
        }
    }

    #[cfg_attr(not(feature = "tts"), allow(dead_code))]
    fn voice_id(&self) -> &str {
        match self.engine {
            Engine::Moss => &self.voice,
            Engine::Kokoro if self.voice.is_empty() || self.voice == "default" => DEFAULT_VOICE,
            Engine::Kokoro => &self.voice,
        }
    }
}

#[cfg_attr(not(feature = "tts"), allow(dead_code))]
/// Plain text fit for speaking: markdown, code and links removed.
pub fn speakable(text: &str) -> String {
    // LaTeX becomes Unicode math, then math symbols become words, so a
    // formula is read ("x squared plus 1 equals 5") rather than spelled out.
    let text = spoken_math(&crate::format::latex_to_unicode(text));
    let mut out = String::new();
    let mut in_code = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if in_code {
            continue;
        }
        let line = line.trim().trim_start_matches(['#', '>', '-', '*', ' ']);
        let mut chars = line.chars().peekable();
        let mut cleaned = String::new();
        while let Some(c) = chars.next() {
            match c {
                '*' | '_' | '`' | '~' | '|' => {}
                // [label](url) -> label
                ']' if chars.peek() == Some(&'(') => {
                    for c in chars.by_ref() {
                        if c == ')' {
                            break;
                        }
                    }
                }
                '[' => {}
                c => cleaned.push(c),
            }
        }
        let cleaned = cleaned.trim();
        if !cleaned.is_empty() {
            out.push_str(cleaned);
            // A line end is a pause.
            if !cleaned.ends_with(['.', '!', '?', ':', ';']) {
                out.push('.');
            }
            out.push(' ');
        }
    }
    out.trim().to_string()
}

/// Math symbols as words.
fn spoken_math(text: &str) -> String {
    const WORDS: &[(char, &str)] = &[
        ('×', " times "), ('·', " times "), ('÷', " divided by "), ('=', " equals "), ('≠', " is not equal to "),
        ('≤', " is less than or equal to "), ('≥', " is greater than or equal to "), ('±', " plus or minus "),
        ('+', " plus "), ('−', " minus "), ('²', " squared "), ('³', " cubed "), ('√', " square root of "),
        ('π', " pi "), ('∞', " infinity "), ('∑', " the sum of "), ('∫', " the integral of "),
        ('→', " goes to "), ('%', " percent "), ('^', " to the power of "),
    ];
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + 16);
    for (i, &c) in chars.iter().enumerate() {
        if let Some((_, word)) = WORDS.iter().find(|(w, _)| *w == c) {
            out.push_str(word);
        } else if c == '/'
            && i > 0
            && chars[i - 1].is_alphanumeric()
            && chars.get(i + 1).is_some_and(|n| n.is_alphanumeric())
        {
            out.push_str(" over ");
        } else {
            out.push(c);
        }
    }
    // Collapse the doubled spaces the padding makes.
    let mut collapsed = String::with_capacity(out.len());
    for c in out.chars() {
        if c != ' ' || !collapsed.ends_with(' ') {
            collapsed.push(c);
        }
    }
    collapsed.replace(" \n", "\n")
}

/// Split into chunks to speak, each ending at a sentence end and at least
/// ~60 characters long where possible (at most ~220): Kokoro slurs and
/// rushes very short inputs like "Hello." or "I am Jarvis.", so short
/// sentences are spoken together with the next one.
fn sentences(text: &str) -> Vec<String> {
    const MIN: usize = 60;
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        cur.push(c);
        let end = matches!(c, '.' | '!' | '?' | ';') || (c == ',' && cur.len() > 160);
        if (end && cur.trim().len() >= MIN) || cur.len() > 220 {
            out.push(std::mem::take(&mut cur).trim().to_string());
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

#[cfg(feature = "tts")]
mod imp {
    use std::sync::mpsc::{self, Receiver, Sender};

    use anyhow::{Context, Result, anyhow};

    use super::{Options, sentences, speakable};

    enum Cmd {
        Speak(String),
        Stop,
        Voice(String),
        Speed(f32),
    }

    pub struct Tts {
        tx: Sender<Cmd>,
    }

    impl Tts {
        /// Load the model and open the audio output; fails (with a message
        /// for the user) if either isn't possible. Blocks while loading.
        pub fn start(options: Options) -> Result<Tts> {
            if options.engine == super::Engine::Moss
                && let Some(reference) = options.reference_for(&options.voice)
                && !reference.exists()
            {
                anyhow::bail!(
                    "the reference clip {} doesn't exist yet; make it with scripts/make-reference.py",
                    reference.display()
                );
            }
            if !options.files_present() {
                match options.engine {
                    super::Engine::Kokoro => anyhow::bail!("the voice files aren't downloaded yet"),
                    super::Engine::Moss => anyhow::bail!(
                        "MOSS isn't set up in {}: run scripts/setup-moss.sh {}",
                        options.moss_dir.display(),
                        options.moss_dir.display()
                    ),
                }
            }
            let (tx, rx) = mpsc::channel();
            let (ready_tx, ready_rx) = mpsc::channel();
            std::thread::Builder::new()
                .name("ojp-tts".into())
                .spawn(move || run(options, rx, ready_tx))
                .context("starting the speech thread")?;
            ready_rx.recv().map_err(|_| anyhow!("the speech thread died while starting"))??;
            Ok(Tts { tx })
        }

        /// Queue `text` (markdown is stripped) after what is already playing.
        pub fn speak(&self, text: &str) {
            let text = speakable(text);
            if !text.is_empty() {
                let _ = self.tx.send(Cmd::Speak(text));
            }
        }

        /// Use another voice from now on.
        pub fn set_voice(&self, voice: &str) {
            let _ = self.tx.send(Cmd::Voice(voice.to_string()));
        }

        /// Speak faster or slower from now on (1.0 = normal).
        pub fn set_speed(&self, speed: f32) {
            let _ = self.tx.send(Cmd::Speed(speed));
        }

        /// Cut off whatever is playing or queued.
        pub fn stop(&self) {
            let _ = self.tx.send(Cmd::Stop);
        }
    }

    /// A speech engine: turns one chunk of text into mono samples and their
    /// sample rate. Anything that can do that can be the voice; see
    /// docs/tts-engines.md. `voice` and `speed` are whatever the engine
    /// understands (an engine may ignore either).
    trait SpeechEngine {
        fn synthesize(&mut self, text: &str, voice: &str, speed: f32) -> Result<(Vec<f32>, u32)>;
    }

    struct Kokoro(kokoro_micro::TtsEngine);

    impl SpeechEngine for Kokoro {
        fn synthesize(&mut self, text: &str, voice: &str, speed: f32) -> Result<(Vec<f32>, u32)> {
            self.0
                .synthesize_with_options(text, Some(voice), speed, 1.0, None)
                .map(|samples| (samples, 24_000))
                .map_err(|e| anyhow!(e))
        }
    }

    /// The MOSS Python process (scripts/moss_server.py), talking over pipes.
    struct Sidecar {
        options: Options,
        child: std::process::Child,
        stdin: std::process::ChildStdin,
        stdout: std::io::BufReader<std::process::ChildStdout>,
    }

    impl Sidecar {
        fn start(options: &Options) -> Result<Sidecar> {
            use std::io::BufRead;
            let dir = &options.moss_dir;
            let mut child = std::process::Command::new(dir.join("venv/bin/python"))
                .arg(dir.join("moss_server.py"))
                .arg(dir.join("repo"))
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn()
                .context("starting the MOSS process")?;
            let stdin = child.stdin.take().context("no stdin")?;
            let mut stdout = std::io::BufReader::new(child.stdout.take().context("no stdout")?);
            let mut line = String::new();
            stdout.read_line(&mut line)?;
            if line.trim() != "READY" {
                let _ = child.kill();
                anyhow::bail!("MOSS didn't start (run scripts/setup-moss.sh again?)");
            }
            Ok(Sidecar { options: options.clone(), child, stdin, stdout })
        }

    }

    impl SpeechEngine for Sidecar {
        fn synthesize(&mut self, text: &str, voice: &str, _speed: f32) -> Result<(Vec<f32>, u32)> {
            use std::io::{BufRead, Read, Write};
            let reference = self.options.reference_for(voice).map(|p| p.to_string_lossy().into_owned());
            let request = format!("{}\n", serde_json::json!({ "text": text, "voice": voice, "reference": reference }));
            self.stdin.write_all(request.as_bytes())?;
            self.stdin.flush()?;
            let mut header = String::new();
            self.stdout.read_line(&mut header)?;
            let mut parts = header.trim().splitn(3, ' ');
            match parts.next() {
                Some("OK") => {
                    let rate: u32 = parts.next().context("no sample rate")?.parse()?;
                    let len: usize = parts.next().context("no length")?.parse()?;
                    let mut bytes = vec![0_u8; len];
                    self.stdout.read_exact(&mut bytes)?;
                    let samples = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                    Ok((samples, rate))
                }
                Some("ERR") => Err(anyhow!("MOSS: {}", header.trim().trim_start_matches("ERR "))),
                _ => Err(anyhow!("MOSS stopped answering")),
            }
        }
    }

    impl Drop for Sidecar {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    fn run(options: Options, rx: Receiver<Cmd>, ready: Sender<Result<()>>) {
        let setup = || -> Result<_> {
            let backend: Box<dyn SpeechEngine> = match options.engine {
                super::Engine::Kokoro => {
                    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
                    let engine = rt
                        .block_on(kokoro_micro::TtsEngine::with_paths(
                            &options.model.to_string_lossy(),
                            &options.voices.to_string_lossy(),
                        ))
                        .map_err(|e| anyhow!("loading the voice model: {e}"))?;
                    Box::new(Kokoro(engine))
                }
                super::Engine::Moss => Box::new(Sidecar::start(&options)?),
            };
            let (stream, handle) = rodio::OutputStream::try_default().context("opening the audio output")?;
            let sink = rodio::Sink::try_new(&handle).context("opening the audio output")?;
            Ok((backend, stream, sink))
        };
        let (mut backend, _stream, sink) = match setup() {
            Ok(parts) => {
                let _ = ready.send(Ok(()));
                parts
            }
            Err(e) => {
                let _ = ready.send(Err(e));
                return;
            }
        };
        let mut voice = options.voice_id().to_string();
        let mut speed = options.speed;
        let mut pending: std::collections::VecDeque<String> = Default::default();
        loop {
            // Take commands: block when idle, otherwise just poll.
            let cmd = if pending.is_empty() && sink.empty() { rx.recv().ok() } else { rx.try_recv().ok() };
            match cmd {
                Some(Cmd::Speak(text)) => pending.extend(sentences(&text)),
                Some(Cmd::Voice(v)) => voice = v,
                Some(Cmd::Speed(v)) => speed = v,
                Some(Cmd::Stop) => {
                    pending.clear();
                    sink.stop();
                    continue;
                }
                // recv() returned None: every sender is gone, so the app is closing.
                None if pending.is_empty() && sink.empty() => break,
                None => {}
            }
            // Keep about one sentence queued ahead of the speaker.
            if sink.len() < 2
                && let Some(sentence) = pending.pop_front()
            {
                if let Ok((samples, rate)) = backend.synthesize(&sentence, &voice, speed)
                    && samples.len() > 100
                {
                    sink.append(rodio::buffer::SamplesBuffer::new(1, rate, samples));
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
        }
    }
}

#[cfg(not(feature = "tts"))]
mod imp {
    use anyhow::Result;

    use super::Options;

    pub struct Tts;

    impl Tts {
        pub fn start(_: Options) -> Result<Tts> {
            anyhow::bail!("this build has no text-to-speech")
        }
        pub fn speak(&self, _: &str) {}
        pub fn set_voice(&self, _: &str) {}
        pub fn set_speed(&self, _: f32) {}
        pub fn stop(&self) {}
    }
}

pub use imp::Tts;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_is_removed() {
        let t = speakable("# Title\n**Bold** and `code` see [docs](http://x.y).\n```\nlet x = 1;\n```\n- item one\n- item two");
        assert_eq!(t, "Title. Bold and code see docs. item one. item two.");
    }

    #[test]
    fn math_is_spoken_as_words() {
        assert_eq!(speakable(r"So \( 12 \times 13 = 156 \)."), "So 12 times 13 equals 156.");
        assert_eq!(speakable(r"$$x^2 + 1 = 5$$"), "x squared plus 1 equals 5.");
        assert_eq!(speakable("Half is 1/2."), "Half is 1 over 2.");
    }

    #[test]
    fn mixes_round_trip() {
        let heart = voice_index("af_heart") + 1;
        let bella = voice_index("af_bella") + 1;
        assert_eq!(mix_string(&[(heart, 6), (bella, 4), (0, 0)]), "af_heart.6+af_bella.4");
        assert_eq!(mix_string(&[(heart, 3), (bella, 3), (0, 9)]), "af_heart.5+af_bella.5");
        assert_eq!(mix_string(&[(heart, 10), (0, 5), (0, 0)]), "af_heart");
        assert_eq!(mix_string(&[(0, 5), (0, 5), (0, 5)]), DEFAULT_VOICE);
        assert_eq!(mix_string(&[(heart, 1), (bella, 1), (heart, 1)]), "af_heart.4+af_bella.3+af_heart.3");
        assert_eq!(parse_mix("af_heart.6+af_bella.4"), [(heart, 6), (bella, 4), (0, 0)]);
        assert_eq!(parse_mix("af_heart"), [(heart, 10), (0, 0), (0, 0)]);
        assert_eq!(parse_mix("default")[0].0, voice_index(DEFAULT_VOICE) + 1);
    }

    #[test]
    fn voice_lookup() {
        assert_eq!(VOICES[voice_index("default")].0, DEFAULT_VOICE);
        assert_eq!(VOICES[voice_index("af_heart")].0, "af_heart");
        assert_eq!(VOICES[voice_index("nonsense")].0, DEFAULT_VOICE);
    }

    #[test]
    fn long_text_is_split_at_sentences() {
        assert_eq!(sentences("One. Two! Three?"), vec!["One. Two! Three?"]);
        let long = "This sentence is comfortably longer than sixty characters in total. Short one.";
        assert_eq!(sentences(long), vec!["This sentence is comfortably longer than sixty characters in total.", "Short one."]);
    }
}
