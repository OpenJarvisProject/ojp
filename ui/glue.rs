// UI <-> engine glue shared by the desktop and Android apps (each includes
// this file with #[path] and provides AppWindow/Chat/Message through
// slint::include_modules!). The engine lives on its own thread so model
// loading and generation never block the UI; results come back through
// slint's event loop.
use std::rc::Rc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ojp_core::download::{self, Item};
use ojp_core::{Config, Engine, Event};
use slint::{ComponentHandle, Model, VecModel};

use crate::{AppWindow, Chat, Message};

enum Request {
    Ask(String),
    Clear,
    Download,
}

/// Show the window and run until it closes. `setup_hint` is shown if no
/// model could be loaded, telling the user where to put one; `warning`
/// (e.g. an unreadable settings file) is shown as the first message.
pub fn run(config: Config, setup_hint: String, warning: Option<String>) -> anyhow::Result<()> {
    let ui = AppWindow::new()?;
    let chat = ui.global::<Chat>();
    chat.set_assistant_name(config.persona.name.clone().into());
    let messages = Rc::new(VecModel::<Message>::default());
    if let Some(warning) = warning {
        messages.push(Message { from_user: false, text: warning.into(), meta: "settings".into() });
    }
    chat.set_messages(messages.clone().into());
    chat.set_busy(true);
    chat.set_status("Loading models…".into());

    let (tx, rx) = mpsc::channel();
    let weak = ui.as_weak();
    std::thread::spawn(move || worker(config, setup_hint, rx, weak));

    chat.on_send({
        let (tx, messages, weak) = (tx.clone(), messages.clone(), ui.as_weak());
        move |text| {
            messages.push(Message { from_user: true, text: text.clone(), meta: "".into() });
            messages.push(Message { from_user: false, text: "".into(), meta: "thinking…".into() });
            if let Some(ui) = weak.upgrade() {
                ui.global::<Chat>().set_busy(true);
            }
            let _ = tx.send(Request::Ask(text.into()));
        }
    });
    chat.on_download({
        let tx = tx.clone();
        move || {
            let _ = tx.send(Request::Download);
        }
    });
    chat.on_clear(move || {
        messages.set_vec(Vec::new());
        let _ = tx.send(Request::Clear);
    });
    ui.run()?;
    Ok(())
}

/// Apply `edit` to the newest message (the reply being streamed).
fn edit_last(ui: &slint::Weak<AppWindow>, edit: impl FnOnce(&mut Message) + Send + 'static) {
    let _ = ui.upgrade_in_event_loop(move |ui| {
        let messages = ui.global::<Chat>().get_messages();
        if let Some(last) = messages.row_count().checked_sub(1)
            && let Some(mut message) = messages.row_data(last)
        {
            edit(&mut message);
            messages.set_row_data(last, message);
        }
    });
}

fn set_state(ui: &slint::Weak<AppWindow>, busy: bool, status: Option<String>) {
    let _ = ui.upgrade_in_event_loop(move |ui| {
        let chat = ui.global::<Chat>();
        chat.set_busy(busy);
        if let Some(status) = status {
            chat.set_status(status.into());
        }
    });
}

fn set_download(ui: &slint::Weak<AppWindow>, needed: bool, text: String, progress: f32) {
    let _ = ui.upgrade_in_event_loop(move |ui| {
        let chat = ui.global::<Chat>();
        chat.set_download_needed(needed);
        chat.set_download_text(text.into());
        chat.set_download_progress(progress);
    });
}

fn push_message(ui: &slint::Weak<AppWindow>, text: String, meta: &'static str) {
    let _ = ui.upgrade_in_event_loop(move |ui| {
        let messages = ui.global::<Chat>().get_messages();
        let model = messages.as_any().downcast_ref::<VecModel<Message>>().expect("VecModel");
        model.push(Message { from_user: false, text: text.into(), meta: meta.into() });
    });
}

fn mb(bytes: u64) -> u64 {
    bytes / 1_000_000
}

/// Load whatever models exist. Missing downloadable models are offered
/// through the download panel instead of an error message.
fn load_engine(config: &Config, setup_hint: &str, ui: &slint::Weak<AppWindow>) -> Option<Engine> {
    set_state(ui, true, Some("Loading models…".into()));
    let missing = download::missing(config);
    if !missing.is_empty() {
        let sizes: Vec<Option<u64>> = missing.iter().map(|item| download::size(&item.url).ok().flatten()).collect();
        let parts: Vec<String> = missing
            .iter()
            .zip(&sizes)
            .map(|(item, size)| match size {
                Some(size) => format!("{} model ({} MB)", item.role, mb(*size)),
                None => format!("{} model", item.role),
            })
            .collect();
        set_download(
            ui,
            true,
            format!(
                "OpenJarvis needs to download its AI models once: {}. They stay on this device, and \
                 everything runs offline after that.",
                parts.join(" and ")
            ),
            -1.0,
        );
    }
    match Engine::new(config) {
        Ok(engine) => {
            set_state(ui, false, Some(engine.status.join(" · ")));
            Some(engine)
        }
        Err(e) => {
            if missing.is_empty() {
                push_message(ui, format!("{e:#}\n\n{setup_hint}"), "setup");
            }
            set_state(ui, false, Some("No model loaded yet".into()));
            None
        }
    }
}

/// Fetch every missing model, showing overall progress. Returns false
/// (with the error shown in the panel) if anything failed.
fn download_all(items: &[Item], ui: &slint::Weak<AppWindow>) -> bool {
    let sizes: Vec<u64> = items.iter().map(|item| download::size(&item.url).ok().flatten().unwrap_or(0)).collect();
    let total: u64 = sizes.iter().sum();
    let mut finished = 0;
    for (item, size) in items.iter().zip(&sizes) {
        let mut last_update = Instant::now() - Duration::from_secs(1);
        let result = download::fetch(item, &mut |done, item_total| {
            if last_update.elapsed() < Duration::from_millis(200) {
                return;
            }
            last_update = Instant::now();
            let item_total = item_total.unwrap_or(*size);
            let progress = if total > 0 { (finished + done) as f32 / total as f32 } else { 0.0 };
            let text = format!("Downloading the {} model… {} of {} MB", item.role, mb(done), mb(item_total));
            set_download(ui, true, text, progress);
        });
        if let Err(e) = result {
            set_download(ui, true, format!("Download failed: {e:#}. Tap Download to retry (it resumes)."), -1.0);
            return false;
        }
        finished += size;
    }
    set_download(ui, false, String::new(), -1.0);
    true
}

fn worker(config: Config, setup_hint: String, rx: mpsc::Receiver<Request>, ui: slint::Weak<AppWindow>) {
    let mut engine = load_engine(&config, &setup_hint, &ui);

    for request in rx {
        match (request, &mut engine) {
            (Request::Download, _) => {
                set_state(&ui, true, Some("Downloading models…".into()));
                let items = download::missing(&config);
                if download_all(&items, &ui) {
                    engine = load_engine(&config, &setup_hint, &ui);
                } else {
                    set_state(&ui, false, Some("Download failed".into()));
                }
            }
            (Request::Clear, Some(engine)) => engine.clear(),
            (Request::Clear, None) => {}
            (Request::Ask(_), None) => {
                edit_last(&ui, |m| {
                    m.text = "No model is loaded yet. Use the Download button below first.".into();
                    m.meta = "".into();
                });
                set_state(&ui, false, None);
            }
            (Request::Ask(text), Some(engine)) => {
                let start = Instant::now();
                let mut meta = Vec::new();
                let result = engine.ask(&text, &mut |event| match event {
                    Event::Routed { route, ms } => meta.push(format!("{} ({ms} ms)", route.label())),
                    Event::Token(token) => edit_last(&ui, move |m| m.text = format!("{}{token}", m.text).into()),
                    Event::Notice(notice) => meta.push(notice),
                });
                if let Err(e) = result {
                    meta.push(format!("error: {e:#}"));
                }
                meta.push(format!("{:.1} s", start.elapsed().as_secs_f32()));
                let meta = meta.join(" · ");
                edit_last(&ui, move |m| m.meta = meta.into());
                set_state(&ui, false, None);
            }
        }
    }
}
