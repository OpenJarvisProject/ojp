fn main() {
    // Only the Android build has a UI to compile; on other targets this
    // crate is empty so `cargo build` of the whole workspace still works.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android") {
        slint_build::compile("../ui/app.slint").expect("ui/app.slint failed to compile");
    }
}
