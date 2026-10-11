//! ojp Android app. Built as libojp_android.so and launched by the
//! platform's NativeActivity (android/AndroidManifest.xml); the same
//! shared UI and engine as the desktop app. Config and models live in the
//! app's external files dir, reachable over USB/adb without root:
//! Android/data/org.openjarvisproject.ojp/files/{config.toml,models/}.
#![cfg(target_os = "android")]

slint::include_modules!();

#[path = "../../ui/glue.rs"]
mod glue;

#[unsafe(no_mangle)]
fn android_main(app: slint::android::AndroidApp) {
    let data_dir = app
        .external_data_path()
        .or_else(|| app.internal_data_path())
        .expect("Android provides an app data directory");
    slint::android::init(app).expect("slint android backend");

    let config_path = data_dir.join("config.toml");
    let (config, warning) = ojp_core::Config::load_or_default(&config_path, &data_dir);
    let hint = format!(
        "To get started, copy a GGUF model to {}/models/main.gguf (and optionally a small one to \
         models/router.gguf), e.g. with `adb push`, then restart the app. Settings are in {}.",
        data_dir.display(),
        config_path.display()
    );
    glue::run(config, hint, warning).expect("ui event loop");
}
