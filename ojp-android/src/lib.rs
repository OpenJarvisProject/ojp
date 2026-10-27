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

/// The Java class `PhoneActions` and the VM, kept for calls from any thread.
static BRIDGE: std::sync::OnceLock<(jni::JavaVM, jni::objects::GlobalRef)> = std::sync::OnceLock::new();

/// `ojp_core::platform`'s one call into Java: `PhoneActions.invoke(method, arg)` answers "ok:<text>" or "err:<message>".
fn native_call(method: &str, arg: &str) -> anyhow::Result<String> {
    use jni::objects::{JClass, JString, JValue};
    let (vm, class) = BRIDGE.get().ok_or_else(|| anyhow::anyhow!("the phone layer isn't ready"))?;
    let mut env = vm.attach_current_thread()?;
    let (m, a) = (env.new_string(method)?, env.new_string(arg)?);
    // SAFETY: the global reference is a java.lang.Class for the lifetime of the process.
    let class = unsafe { JClass::from_raw(class.as_obj().as_raw()) };
    let out = env
        .call_static_method(&class, "invoke", "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;", &[JValue::Object(&m), JValue::Object(&a)])?
        .l()?;
    let out: String = env.get_string(&JString::from(out))?.into();
    match out.split_once(':') {
        Some(("ok", text)) => Ok(text.to_string()),
        Some(("err", message)) => anyhow::bail!("{message}"),
        _ => anyhow::bail!("unexpected answer from the phone layer"),
    }
}

/// Load `PhoneActions` through the activity's class loader (a native thread can't find app classes by name),
/// hand it the activity, and install [`native_call`] for ojp-core.
fn install_phone_layer(app: &slint::android::AndroidApp) -> jni::errors::Result<()> {
    use jni::objects::{JClass, JObject, JValue};
    // SAFETY: as in `external_files_dirs`.
    let vm = unsafe { jni::JavaVM::from_raw(app.vm_as_ptr().cast()) }?;
    let mut env = vm.attach_current_thread()?;
    let activity = unsafe { JObject::from_raw(app.activity_as_ptr().cast()) };
    let loader = env.call_method(&activity, "getClassLoader", "()Ljava/lang/ClassLoader;", &[])?.l()?;
    let name = env.new_string("org.openjarvisproject.ojp.PhoneActions")?;
    let class = env.call_method(&loader, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;", &[JValue::Object(&name)])?.l()?;
    let global = env.new_global_ref(&class)?;
    // SAFETY: `class` is a java.lang.Class.
    let jc = unsafe { JClass::from_raw(class.as_raw()) };
    env.call_static_method(&jc, "init", "(Landroid/app/Activity;)V", &[JValue::Object(&activity)])?;
    drop(env);
    let _ = BRIDGE.set((vm, global));
    ojp_core::platform::install(native_call);
    Ok(())
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
    if let Err(e) = install_phone_layer(&app) {
        eprintln!("phone layer unavailable: {e}");
    }
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
        set_light_status_bar: Some(i_slint_backend_android_activity::set_light_status_bar),
        test_hook: None,
    }).expect("ui event loop");
}
