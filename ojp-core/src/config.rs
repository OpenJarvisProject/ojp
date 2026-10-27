//! User configuration (config.toml). Every field has a default, so a
//! missing or partial file is fine; on first run the commented default
//! below is written out for the user to edit.
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use crate::template::Template;
use serde::Deserialize;

pub const DEFAULT_CONFIG: &str = r#"# OpenJarvisProject (ojp) configuration.
# Relative paths are resolved against the app's data directory
# (Linux: ~/.local/share/ojp, Android: the app's external files dir).

[persona]
# What the assistant calls itself.
name = "Jarvis"
# Spoken wake phrase, and how its name is pronounced (for the voice
# pipeline: wake-word training and text-to-speech).
wake_word = "hey jarvis"
pronunciation = "jarvis"
# Text-to-speech voice id. "default" is a calm, neutral male voice.
voice = "default"

[models]
# Developer switch (also in the app's settings): false = never load the
# main model, so the router answers everything.
main_enabled = true
# Small, always-loaded model: classifies each request and runs tools.
router = "models/router.gguf"
# Bigger model for the actual answers. Either an in-process GGUF...
main = "models/main.gguf"
# ...or any OpenAI-compatible server (Ollama, llama-server, another
# device). When main_url is set it is used instead of `main`.
# main_url = "http://127.0.0.1:11434"
# main_model = "qwen2.5:3b"

# Layers to offload to the GPU (0 = CPU only, 99 = everything).
gpu_layers = 0
n_ctx = 4096
# The router's accuracy comes from a tiny classifier fitted on the router model's
# hidden state (scripts/make-router-probe.sh). By default it is looked up next to
# the router as <router file>.probe.json; set this to use another file.
# router_probe = "models/router.gguf.probe.json"
# CPU threads for generation (0 = automatic: the physical cores on a PC, at most 4 on a phone).
threads = 0
# Use less RAM for the main model (about 250 MB less for a 0.8B) at about 10%
# slower answers: no repacked weight copies, smaller scratch buffers. The small
# router always runs this way, since it is meant to stay light.
low_memory = false
# Where models/ lives. On Android this is set when you pick phone storage
# or SD card at download time; unset means the app's data directory.
# location = "/path/to/dir"
# Prompt format of the models: chatml (Qwen and most others), chatml-nothink
# (Qwen3.5: thinking off), llama3,
# mistral, gemma, phi3.
chat_template = "chatml"

[link]
# Paired devices work together (pair them in Settings, or with `ojp-cli link-pair-host` and
# `ojp-cli link-pair-join`). Only devices you paired by hand can connect.
enabled = true
# Who answers chat. "auto": this device's own main model if it has one, otherwise a
# paired device's. "local": never a paired device. "remote": a paired device's whenever one is reachable.
brain = "auto"

[tts]
# Speak replies aloud. Also a switch in the app's settings; the voice files
# (about 340 MB) are downloaded the first time you turn it on.
enabled = false
# Overrides the persona's voice (a Kokoro id such as "bm_george", "bm_lewis",
# "af_heart" or "bf_emma"); empty = use [persona] voice.
voice = ""
# 1.0 = normal; lower is slower.
speed = 1.0
# How much to sharpen consonants, 0.0 (off) to 1.0 (strong). 0.65 is a good
# default; lower it if speech sounds too crisp.
clarity = 0.65
# "kokoro" (built in) or "moss" (experimental: MOSS-TTS-Nano through a
# Python sidecar; set it up with scripts/setup-moss.sh).
engine = "kokoro"
# MOSS voice: Ava, Bella, Adam, Nathan, Custom (your own reference clip, see
# moss_reference below), or ref:<name> for <moss_dir>/references/<name>.wav.
# Settings lists whichever of these exist.
moss_voice = "Bella"
# The recording that moss_voice = "Custom" imitates (WAV, a few seconds to
# half a minute of clean speech; use only voices you have the right to use).
# Empty = <moss_dir>/reference.wav, which scripts/make-reference.py writes.
moss_reference = ""
# Where setup-moss.sh installed MOSS; empty = <data dir>/moss.
moss_dir = ""
model = "models/tts/kokoro.onnx"
voices = "models/tts/voices.bin"

[tools]
# Let the assistant search the web and read web pages. Off by default: it
# means pages you never asked for can reach the model. Addresses on your own
# machine or network are always refused.
internet = false

[ui]
# "system", "light" or "dark".
theme = "system"
# Android: tint the app with your wallpaper's colours (Material You).
dynamic_colors = true

[download]
# Where the app fetches a model whose file above is missing (first run:
# the "Download" button). Each is verified against its SHA-256 before use.
# The router is OJP's 347 MB build of Qwen2.5-0.5B-Instruct (Q4_K_M +
# imatrix + q5_0 embedding): same quality as the original, 1/3 the size.
# Bigger main model for desktops with a GPU (2.1 GB):
#   main = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
#   main_sha256 = "626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d"
# Set by unticking "also download the main model" at first run; set back
# to false to be offered the main model again.
skip_main = false
router = "https://huggingface.co/OpenJarvisProject/Qwen2.5-0.5B-Instruct-OJP-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m-imat-embq5_0.gguf"
router_sha256 = "225dd294b01dab1cd24b5d404e12bc591dc9335d99f8a1f584e7b4c5bca86186"
main = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
main_sha256 = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
# Kokoro-82M voice (the files the kokoro-micro crate ships with).
tts_model = "https://github.com/8b-is/kokoro-tiny/raw/main/models/0.onnx"
tts_model_sha256 = "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"
tts_voices = "https://github.com/8b-is/kokoro-tiny/raw/main/models/0.bin"
tts_voices_sha256 = "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"
"#;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub persona: Persona,
    pub models: Models,
    pub download: Download,
    pub ui: Ui,
    pub tts: Tts,
    pub tools: ToolsConfig,
    pub link: Link,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Link {
    pub enabled: bool,
    /// "auto", "local" or "remote" (see DEFAULT_CONFIG).
    pub brain: String,
}

impl Default for Link {
    fn default() -> Self {
        Link { enabled: true, brain: "auto".into() }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct ToolsConfig {
    pub internet: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Tts {
    pub enabled: bool,
    pub voice: String,
    pub speed: f32,
    pub model: PathBuf,
    pub voices: PathBuf,
    pub clarity: f32,
    pub engine: String,
    pub moss_voice: String,
    pub moss_dir: PathBuf,
    pub moss_reference: String,
}

impl Default for Tts {
    fn default() -> Self {
        Tts {
            enabled: false,
            voice: String::new(),
            speed: 1.0,
            model: "models/tts/kokoro.onnx".into(),
            voices: "models/tts/voices.bin".into(),
            clarity: 0.65,
            engine: "kokoro".into(),
            moss_voice: "Bella".into(),
            moss_dir: PathBuf::new(),
            moss_reference: String::new(),
        }
    }
}

impl Config {
    /// What the speech engine needs; the voice falls back to the persona's.
    pub fn tts_options(&self) -> crate::tts::Options {
        let engine = crate::tts::Engine::from_id(&self.tts.engine);
        let voice = match engine {
            crate::tts::Engine::Moss => self.tts.moss_voice.clone(),
            // Always a concrete id or blend, never "default".
            crate::tts::Engine::Kokoro => {
                let voice = if self.tts.voice.is_empty() { &self.persona.voice } else { &self.tts.voice };
                crate::tts::mix_string(&crate::tts::parse_mix(voice))
            }
        };
        crate::tts::Options {
            engine,
            model: self.tts.model.clone(),
            voices: self.tts.voices.clone(),
            voice,
            speed: self.tts.speed,
            clarity: self.tts.clarity,
            moss_dir: self.tts.moss_dir.clone(),
            moss_reference: (!self.tts.moss_reference.is_empty()).then(|| PathBuf::from(&self.tts.moss_reference)),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Ui {
    /// "system", "light" or "dark".
    pub theme: String,
    /// Android: tint the UI with the wallpaper-based system accent colour.
    pub dynamic_colors: bool,
}

impl Default for Ui {
    fn default() -> Self {
        Ui { theme: "system".into(), dynamic_colors: true }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Persona {
    pub name: String,
    pub wake_word: String,
    pub pronunciation: String,
    pub voice: String,
}

/// The one-model profile: huihui-ai's uncensored Qwen3.5-0.8B, trimmed and
/// quantized by OJP (Q5_K_M, 401 MB) with the router and tool probes fitted
/// to this exact file. File, URL, SHA-256.
pub const UNCENSORED_FILES: [(&str, &str, &str); 3] = [
    (
        "unc-0.8b-q5km.gguf",
        "https://huggingface.co/OpenJarvisProject/Qwen3.5-0.8B-Uncensored-OJP-GGUF/resolve/main/unc-0.8b-q5km.gguf",
        "976f632982a1c51eded0422d965841e3484dafd16b864b108b28dae38870b6f7",
    ),
    (
        "unc-0.8b-q5km.gguf.probe.json",
        "https://huggingface.co/OpenJarvisProject/Qwen3.5-0.8B-Uncensored-OJP-GGUF/resolve/main/unc-0.8b-q5km.gguf.probe.json",
        "976f942fd1d0f78bf51c5ccbba77af56968389f4a2a95f8abd3bddd9d298c2cd",
    ),
    (
        "unc-0.8b-q5km.gguf.tools.json",
        "https://huggingface.co/OpenJarvisProject/Qwen3.5-0.8B-Uncensored-OJP-GGUF/resolve/main/unc-0.8b-q5km.gguf.tools.json",
        "cab507cf7e20c8395539721ad088e7e839009c1560f84244947190b6aa79964b",
    ),
];

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Models {
    /// Developer switch: false = don't load the main model (router only).
    pub main_enabled: bool,
    /// Base directory for relative model paths (default: the data dir).
    pub location: Option<PathBuf>,
    pub router: Option<PathBuf>,
    pub router_probe: Option<PathBuf>,
    pub main: Option<PathBuf>,
    pub main_url: Option<String>,
    pub main_model: Option<String>,
    pub gpu_layers: i32,
    pub n_ctx: u32,
    pub low_memory: bool,
    pub threads: i32,
    pub chat_template: String,
    /// "classic" (the Qwen2.5 pair above) or "uncensored" (one Qwen3.5-0.8B
    /// model for everything; see [`UNCENSORED`]). Picked in Settings.
    pub profile: String,
}

// All three defaults are written out by hand, never by parsing
// DEFAULT_CONFIG: serde's container-level default calls Default::default()
// while deserializing, so parsing inside default() recurses forever. A
// test keeps them in sync with DEFAULT_CONFIG.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Download {
    /// The user chose router-only at first run: don't offer the main model.
    pub skip_main: bool,
    pub router: Option<String>,
    pub router_sha256: Option<String>,
    pub main: Option<String>,
    pub main_sha256: Option<String>,
    pub tts_model: Option<String>,
    pub tts_model_sha256: Option<String>,
    pub tts_voices: Option<String>,
    pub tts_voices_sha256: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config { persona: Persona::default(), models: Models::default(), download: Download::default(), ui: Ui::default(), tts: Tts::default(), tools: ToolsConfig::default(), link: Link::default() }
    }
}

impl Default for Download {
    fn default() -> Self {
        Download {
            skip_main: false,
            router: Some("https://huggingface.co/OpenJarvisProject/Qwen2.5-0.5B-Instruct-OJP-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m-imat-embq5_0.gguf".into()),
            router_sha256: Some("225dd294b01dab1cd24b5d404e12bc591dc9335d99f8a1f584e7b4c5bca86186".into()),
            main: Some("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf".into()),
            main_sha256: Some("6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e".into()),
            tts_model: Some("https://github.com/8b-is/kokoro-tiny/raw/main/models/0.onnx".into()),
            tts_model_sha256: Some("7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5".into()),
            tts_voices: Some("https://github.com/8b-is/kokoro-tiny/raw/main/models/0.bin".into()),
            tts_voices_sha256: Some("bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d".into()),
        }
    }
}

impl Default for Persona {
    fn default() -> Self {
        Persona {
            name: "Jarvis".into(),
            wake_word: "hey jarvis".into(),
            pronunciation: "jarvis".into(),
            voice: "default".into(),
        }
    }
}

impl Default for Models {
    fn default() -> Self {
        Models {
            main_enabled: true,
            location: None,
            router: Some("models/router.gguf".into()),
            router_probe: None,
            main: Some("models/main.gguf".into()),
            main_url: None,
            main_model: None,
            gpu_layers: 0,
            n_ctx: 4096,
            low_memory: false,
            threads: 0,
            chat_template: "chatml".into(),
            profile: "classic".into(),
        }
    }
}

/// Linux desktop locations: ($XDG_CONFIG_HOME/ojp/config.toml,
/// $XDG_DATA_HOME/ojp), defaulting to ~/.config and ~/.local/share.
pub fn desktop_paths() -> (PathBuf, PathBuf) {
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    let xdg = |var: &str, fallback: &str| std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| home.join(fallback));
    (xdg("XDG_CONFIG_HOME", ".config").join("ojp/config.toml"), xdg("XDG_DATA_HOME", ".local/share").join("ojp"))
}

impl Config {
    /// Read `path`, writing the commented default there first if it
    /// doesn't exist yet. Relative model paths resolve against `data_dir`.
    pub fn load_or_init(path: &Path, data_dir: &Path) -> Result<Config> {
        if !path.exists() {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, DEFAULT_CONFIG).with_context(|| format!("writing {}", path.display()))?;
        }
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut config: Config = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        config.resolve_paths(data_dir);
        Ok(config)
    }

    /// Like `load_or_init`, but never fails: if the file can't be read or
    /// parsed, use the defaults (still rooted in `data_dir`) and return a
    /// warning for the UI to show instead of silently ignoring it.
    pub fn load_or_default(path: &Path, data_dir: &Path) -> (Config, Option<String>) {
        match Config::load_or_init(path, data_dir) {
            Ok(config) => (config, None),
            Err(e) => {
                let mut config = Config::default();
                config.resolve_paths(data_dir);
                (config, Some(format!("Couldn't use the settings file ({e:#}), so default settings are in use.")))
            }
        }
    }

    fn resolve_paths(&mut self, data_dir: &Path) {
        let base = self.models.location.clone().unwrap_or_else(|| data_dir.to_path_buf());
        for model in [&mut self.models.router, &mut self.models.main, &mut self.models.router_probe].into_iter().flatten() {
            if model.is_relative() {
                *model = base.join(&*model);
            }
        }
        if self.models.profile == "uncensored" {
            let (file, url, sha) = UNCENSORED_FILES[0];
            self.models.router = Some(base.join("models").join(file));
            self.models.router_probe = None; // found next to the model
            self.models.main_enabled = false;
            self.models.chat_template = "chatml-nothink".into();
            self.download.router = Some(url.into());
            self.download.router_sha256 = Some(sha.into());
        }
        if self.tts.moss_dir.as_os_str().is_empty() {
            self.tts.moss_dir = data_dir.join("moss");
        }
        for file in [&mut self.tts.model, &mut self.tts.voices] {
            if file.is_relative() {
                *file = base.join(&*file);
            }
        }
    }

    pub fn template(&self) -> Result<Template> {
        Template::named(&self.models.chat_template)
            .ok_or_else(|| anyhow::anyhow!("unknown chat_template {:?}", self.models.chat_template))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_parses_with_expected_values() {
        let c: Config = toml::from_str(DEFAULT_CONFIG).unwrap();
        assert_eq!(c.persona.name, "Jarvis");
        assert_eq!(c.models.router.as_deref(), Some(Path::new("models/router.gguf")));
        assert_eq!(c.models.main_url, None);
        assert!(c.template().unwrap() == Template::CHATML);
    }

    #[test]
    fn unreadable_config_falls_back_to_defaults_under_data_dir() {
        let dir = std::env::temp_dir().join("ojp-config-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "this is [not toml").unwrap();
        let (config, warning) = Config::load_or_default(&path, Path::new("/data/ojp"));
        assert!(warning.is_some());
        assert_eq!(config.models.router.as_deref(), Some(Path::new("/data/ojp/models/router.gguf")));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn location_overrides_data_dir_for_relative_models() {
        let dir = std::env::temp_dir().join("ojp-location-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "[models]\nlocation = \"/sd/ojp\"\nmain = \"/abs/main.gguf\"\n").unwrap();
        let config = Config::load_or_init(&path, Path::new("/data/ojp")).unwrap();
        assert_eq!(config.models.router.as_deref(), Some(Path::new("/sd/ojp/models/router.gguf")));
        assert_eq!(config.models.main.as_deref(), Some(Path::new("/abs/main.gguf")));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn partial_file_keeps_other_defaults() {
        let c: Config = toml::from_str("[persona]\nname = \"Zelia\"\n").unwrap();
        assert_eq!(c.persona.name, "Zelia");
        assert_eq!(c.persona.voice, "default");
        assert_eq!(c.models.n_ctx, 4096);
        assert_eq!(c.models.router.as_deref(), Some(Path::new("models/router.gguf")));
    }

    #[test]
    fn hand_written_defaults_match_default_config() {
        let parsed: Config = toml::from_str(DEFAULT_CONFIG).unwrap();
        let (p, m) = (Persona::default(), Models::default());
        assert_eq!((parsed.persona.name, parsed.persona.wake_word), (p.name, p.wake_word));
        assert_eq!((parsed.persona.pronunciation, parsed.persona.voice), (p.pronunciation, p.voice));
        assert_eq!((parsed.models.router, parsed.models.main), (m.router, m.main));
        assert_eq!((parsed.models.gpu_layers, parsed.models.n_ctx), (m.gpu_layers, m.n_ctx));
        assert_eq!(parsed.models.chat_template, m.chat_template);
        assert_eq!(parsed.models.main_enabled, m.main_enabled);
        let u = Ui::default();
        assert_eq!((parsed.ui.theme, parsed.ui.dynamic_colors), (u.theme, u.dynamic_colors));
        let d = Download::default();
        assert_eq!(parsed.download.skip_main, d.skip_main);
        assert_eq!((parsed.download.router, parsed.download.router_sha256), (d.router, d.router_sha256));
        assert_eq!((parsed.download.main, parsed.download.main_sha256), (d.main, d.main_sha256));
    }
}

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn uncensored_profile_points_at_the_single_model() {
        let dir = std::env::temp_dir().join("ojp-profile-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "[models]\nprofile = \"uncensored\"\n").unwrap();
        let c = Config::load_or_init(&path, Path::new("/data/ojp")).unwrap();
        assert_eq!(c.models.router.as_deref(), Some(Path::new("/data/ojp/models/unc-0.8b-q5km.gguf")));
        assert!(!c.models.main_enabled);
        assert_eq!(c.models.chat_template, "chatml-nothink");
        assert!(c.template().is_ok());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
