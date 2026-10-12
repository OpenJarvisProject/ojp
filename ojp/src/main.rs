//! ojp desktop app: the shared chat UI plus the engine, configured from
//! ~/.config/ojp/config.toml with models under ~/.local/share/ojp.
slint::include_modules!();

#[path = "../../ui/glue.rs"]
mod glue;

fn main() -> anyhow::Result<()> {
    let (config_path, data_dir) = ojp_core::config::desktop_paths();
    let (config, warning) = ojp_core::Config::load_or_default(&config_path, &data_dir);
    let hint = format!(
        "To get started, put a GGUF model at {}/models/main.gguf (and optionally a small one at \
         models/router.gguf), or point main_url at an Ollama/llama-server instance in {}.",
        data_dir.display(),
        config_path.display()
    );
    glue::run(glue::Setup { config, config_path, data_dir, locations: Vec::new(), setup_hint: hint, warning, set_dynamic_colors: None })
}
