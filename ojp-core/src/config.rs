//! User configuration (config.toml). Every field has a default, so a
//! missing or partial file is fine; on first run the commented default
//! below is written out for the user to edit.
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use llama_crab::chat::BuiltinTemplate;
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
# Prompt format of the models: chatml (Qwen and most others), llama3,
# mistral, gemma, phi3.
chat_template = "chatml"
"#;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub persona: Persona,
    pub models: Models,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Persona {
    pub name: String,
    pub wake_word: String,
    pub pronunciation: String,
    pub voice: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Models {
    pub router: Option<PathBuf>,
    pub main: Option<PathBuf>,
    pub main_url: Option<String>,
    pub main_model: Option<String>,
    pub gpu_layers: i32,
    pub n_ctx: u32,
    pub chat_template: String,
}

// All three defaults are written out by hand, never by parsing
// DEFAULT_CONFIG: serde's container-level default calls Default::default()
// while deserializing, so parsing inside default() recurses forever. A
// test keeps them in sync with DEFAULT_CONFIG.
impl Default for Config {
    fn default() -> Self {
        Config { persona: Persona::default(), models: Models::default() }
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
            router: Some("models/router.gguf".into()),
            main: Some("models/main.gguf".into()),
            main_url: None,
            main_model: None,
            gpu_layers: 0,
            n_ctx: 4096,
            chat_template: "chatml".into(),
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
        for model in [&mut config.models.router, &mut config.models.main].into_iter().flatten() {
            if model.is_relative() {
                *model = data_dir.join(&*model);
            }
        }
        Ok(config)
    }

    pub fn template(&self) -> Result<BuiltinTemplate> {
        Ok(match self.models.chat_template.to_lowercase().as_str() {
            "chatml" => BuiltinTemplate::ChatMl,
            "llama3" => BuiltinTemplate::Llama3,
            "mistral" => BuiltinTemplate::MistralInstruct,
            "gemma" => BuiltinTemplate::Gemma,
            "phi3" => BuiltinTemplate::Phi3,
            other => anyhow::bail!("unknown chat_template {other:?}"),
        })
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
        assert!(matches!(c.template().unwrap(), BuiltinTemplate::ChatMl));
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
    }
}
