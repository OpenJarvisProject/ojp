//! ojp Android app. Built as libojp_android.so and launched by the
//! platform's NativeActivity (android/AndroidManifest.xml); the same
//! shared UI and engine as the desktop app. Config and models live in the
//! app's external files dir, reachable over USB/adb without root:
//! Android/data/org.openjarvisproject.ojp/files/{config.toml,models/}.
#![cfg(target_os = "android")]

slint::include_modules!();

#[path = "../../ui/glue.rs"]
mod glue;

use std::path::PathBuf;

use ojp_core::storage::Location;

/// The app's private folders on each storage volume, via the Java
/// `Context.getExternalFilesDirs(null)`: the first is phone storage, any
/// others are removable (SD card). Unmounted volumes come back as null
/// and are skipped. No storage permission is needed for these.
fn external_files_dirs(app: &slint::android::AndroidApp) -> jni::errors::Result<Vec<PathBuf>> {
    use jni::objects::{JObject, JObjectArray, JString, JValue};
    // SAFETY: both pointers come from the running activity and stay valid
    // for its lifetime; the activity reference is not ours to delete.
    let vm = unsafe { jni::JavaVM::from_raw(app.vm_as_ptr().cast()) }?;
    let mut env = vm.attach_current_thread()?;
    let activity = unsafe { JObject::from_raw(app.activity_as_ptr().cast()) };
    let dirs = env
        .call_method(&activity, "getExternalFilesDirs", "(Ljava/lang/String;)[Ljava/io/File;", &[JValue::Object(
            &JObject::null(),
        )])?
        .l()?;
    let dirs = JObjectArray::from(dirs);
    let mut paths = Vec::new();
    for i in 0..env.get_array_length(&dirs)? {
        let dir = env.get_object_array_element(&dirs, i)?;
        if dir.is_null() {
            continue;
        }
        let path = env.call_method(&dir, "getAbsolutePath", "()Ljava/lang/String;", &[])?.l()?;
        let path: String = env.get_string(&JString::from(path))?.into();
        paths.push(PathBuf::from(path));
    }
    Ok(paths)
}

#[unsafe(no_mangle)]
fn android_main(app: slint::android::AndroidApp) {
    let data_dir = app
        .external_data_path()
        .or_else(|| app.internal_data_path())
        .expect("Android provides an app data directory");
    let locations: Vec<Location> = external_files_dirs(&app)
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(i, path)| Location {
            label: match i {
                0 => "Phone storage".into(),
                1 => "SD card".into(),
                n => format!("SD card {n}"),
            },
            path,
        })
        .collect();
    slint::android::init(app).expect("slint android backend");

    let config_path = data_dir.join("config.toml");
    let (config, warning) = ojp_core::Config::load_or_default(&config_path, &data_dir);
    let hint = format!(
        "To get started, copy a GGUF model to {}/models/main.gguf (and optionally a small one to \
         models/router.gguf), e.g. with `adb push`, then restart the app. Settings are in {}.",
        data_dir.display(),
        config_path.display()
    );
    glue::run(glue::Setup {
        config,
        config_path,
        data_dir,
        locations,
        setup_hint: hint,
        warning,
        set_dynamic_colors: Some(i_slint_backend_android_activity::set_dynamic_color),
    }).expect("ui event loop");
}
