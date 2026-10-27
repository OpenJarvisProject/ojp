//! Headless check of the app's real logic (ui/glue.rs) with Slint's testing backend: flips Settings to the
//! uncensored model on a data folder that has no such model and reports what the chat page then shows.
//!   cargo run -p ojp --no-default-features --features uitest --bin ui-test
slint::include_modules!();

#[path = "../../../ui/glue.rs"]
mod glue;

use slint::Model;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    i_slint_backend_testing::init_integration_test_with_system_time();
    let root = std::env::temp_dir().join(format!("ojp-uitest-{}", std::process::id()));
    let data_dir = root.join("data");
    std::fs::create_dir_all(data_dir.join("models"))?;
    // The user's state: classic models present (so nothing is missing), the uncensored one absent.
    for f in ["router.gguf", "main.gguf"] {
        std::fs::write(data_dir.join("models").join(f), b"")?;
    }
    let config_path = root.join("config.toml");
    unsafe {
        std::env::set_var("OJP_LINK_DIR", root.join("link"));
        std::env::set_var("XDG_DATA_HOME", &root);
    }
    let (config, warning) = ojp_core::Config::load_or_default(&config_path, &data_dir);
    glue::run(glue::Setup {
        config,
        config_path: config_path.clone(),
        data_dir,
        locations: Vec::new(),
        setup_hint: String::new(),
        warning,
        set_dynamic_colors: None,
        set_light_status_bar: None,
        test_hook: Some(Box::new(|ui| {
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(3));
                let _ = ui.upgrade_in_event_loop(|ui| {
                    println!("before: download-needed={}", ui.global::<Chat>().get_download_needed());
                    let s = ui.global::<Settings>();
                    s.set_profile(1);
                    s.invoke_changed();
                });
                std::thread::sleep(Duration::from_secs(20));
                let _ = ui.upgrade_in_event_loop(|ui| {
                    let c = ui.global::<Chat>();
                    println!("after:  download-needed={} progress={} targets={}", c.get_download_needed(), c.get_download_progress(), c.get_download_targets().row_count());
                    println!("        text={:?}", c.get_download_text().to_string());
                    println!("        status={:?}", c.get_status().to_string());
                    let _ = slint::quit_event_loop();
                });
            });
        })),
    })?;
    println!("config after:\n{}", std::fs::read_to_string(&config_path).unwrap_or_default().lines().filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty()).collect::<Vec<_>>().join("\n"));
    Ok(())
}
