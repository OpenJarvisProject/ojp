//! Renders the app's pages to PNG files with Slint's software renderer, at a phone-sized window, so layout
//! can be checked without a device:
//!   cargo run -p ojp --no-default-features --features shots --bin ui-shot -- out-dir
slint::include_modules!();

use std::rc::Rc;

use slint::platform::software_renderer::MinimalSoftwareWindow;
use slint::{ModelRc, SharedString, VecModel};

struct Headless(Rc<MinimalSoftwareWindow>);

impl slint::platform::Platform for Headless {
    fn create_window_adapter(&self) -> Result<Rc<dyn slint::platform::WindowAdapter>, slint::PlatformError> {
        Ok(self.0.clone())
    }
}

fn shot(window: &MinimalSoftwareWindow, name: &str, dir: &std::path::Path, w: u32, h: u32) {
    window.set_size(slint::PhysicalSize::new(w, h));
    slint::platform::update_timers_and_animations();
    let mut buf = vec![slint::Rgb8Pixel { r: 0, g: 0, b: 0 }; (w * h) as usize];
    // Twice: the first pass lays out, the second draws the settled result.
    for _ in 0..2 {
        window.draw_if_needed(|r| {
            r.render(&mut buf, w as usize);
        });
        window.request_redraw();
    }
    window.draw_if_needed(|r| {
        r.render(&mut buf, w as usize);
    });
    let file = std::fs::File::create(dir.join(format!("{name}.png"))).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let bytes: Vec<u8> = buf.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    enc.write_header().unwrap().write_image_data(&bytes).unwrap();
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| ".".into()));
    std::fs::create_dir_all(&dir).unwrap();
    let window = MinimalSoftwareWindow::new(slint::platform::software_renderer::RepaintBufferType::NewBuffer);
    slint::platform::set_platform(Box::new(Headless(window.clone()))).unwrap();
    let ui = AppWindow::new().unwrap();
    ui.show().unwrap();

    let chat = ui.global::<Chat>();
    chat.set_status("No model loaded yet".into());
    chat.set_download_needed(true);
    chat.set_download_text("OpenJarvis needs to download its AI models once: router model (401 MB) and router probe model (0 MB) and tool probe model (0 MB). They stay on this device, and everything runs offline after that.".into());
    shot(&window, "chat-download", &dir, 412, 900);
    chat.set_download_targets(ModelRc::new(VecModel::from(vec![SharedString::from("Phone storage (23.4 GB free)"), SharedString::from("SD card (51.0 GB free)")])));
    shot(&window, "chat-download-storage", &dir, 412, 900);
    chat.set_download_needed(false);

    let settings = ui.global::<Settings>();
    settings.set_shizuku_supported(true);
    settings.set_shizuku_text("Shizuku isn't running (optional). Install it and start it to let the assistant change Wi-Fi, Bluetooth, brightness and run commands.".into());
    settings.set_devices(ModelRc::new(VecModel::from(vec![
        Device { name: "alpha".into(), status: "desktop · online · has a big model".into(), chat: true, ask: 1, notify: true, sync: true },
        Device { name: "pixel".into(), status: "phone · not reachable right now".into(), chat: true, ask: 3, notify: true, sync: false },
    ])));
    settings.set_info("Models: /storage/emulated/0/Android/data/org.openjarvisproject.ojp/files/models\nSettings file: /storage/emulated/0/Android/data/org.openjarvisproject.ojp/files/config.toml".into());
    settings.set_profile(1);
    settings.set_brain(1);
    settings.set_model_needs_download(true);
    settings.set_model_note("Not on this device yet (router, router probe, tool probe). Tap Download; it resumes if interrupted, and everything works offline afterwards.".into());
    ui.set_show_settings(true);
    shot(&window, "settings-phone", &dir, 412, 900);
    shot(&window, "settings-tall", &dir, 412, 3200);
}
