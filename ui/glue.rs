// UI <-> engine glue shared by the desktop and Android apps (each includes
// this file with #[path] and provides AppWindow/Chat/Message through
// slint::include_modules!). The engine lives on its own thread so model
// loading and generation never block the UI; results come back through
// slint's event loop.
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ojp_core::download::{self, Item};
use ojp_core::storage::{self, Location};
use ojp_core::{Config, Engine, Event};
use slint::{ComponentHandle, Model, StyledText, VecModel};

use crate::{AppWindow, Chat, Message};

enum Request {
    Ask(String),
    Clear,
    /// Download missing models into `Setup::locations[i]`, or into the
    /// current location for a negative index; the bool is "include the
    /// main model".
    Download(i32, bool),
}

/// Everything the platform-specific `main` provides.
pub struct Setup {
    pub config: Config,
    pub config_path: PathBuf,
    pub data_dir: PathBuf,
    /// Places the user may keep models (phone storage, SD card). With
    /// fewer than two there is nothing to choose.
    pub locations: Vec<Location>,
    /// Shown if no model can be loaded and none can be downloaded.
    pub setup_hint: String,
    /// Shown as the first message (e.g. an unreadable settings file).
    pub warning: Option<String>,
}

/// Show the window and run until it closes.
pub fn run(mut setup: Setup) -> anyhow::Result<()> {
    let ui = AppWindow::new()?;
    let chat = ui.global::<Chat>();
    chat.set_assistant_name(setup.config.persona.name.clone().into());
    let messages = Rc::new(VecModel::<Message>::default());
    if let Some(warning) = setup.warning.take() {
        messages.push(reply(&warning, "settings"));
    }
    chat.set_messages(messages.clone().into());
    chat.set_busy(true);
    chat.set_status("Loading models…".into());

    let (tx, rx) = mpsc::channel();
    let weak = ui.as_weak();
    std::thread::spawn(move || worker(setup, rx, weak));

    chat.on_send({
        let (tx, messages, weak) = (tx.clone(), messages.clone(), ui.as_weak());
        move |text| {
            messages.push(Message { from_user: true, text: text.clone(), ..Default::default() });
            messages.push(Message { meta: "thinking…".into(), ..Default::default() });
            if let Some(ui) = weak.upgrade() {
                ui.global::<Chat>().set_busy(true);
            }
            let _ = tx.send(Request::Ask(text.into()));
        }
    });
    chat.on_download({
        let (tx, weak) = (tx.clone(), ui.as_weak());
        move |index| {
            let include_main = weak.upgrade().is_none_or(|ui| ui.global::<Chat>().get_download_main());
            let _ = tx.send(Request::Download(index, include_main));
        }
    });
    chat.on_clear(move || {
        messages.set_vec(Vec::new());
        let _ = tx.send(Request::Clear);
    });
    ui.run()?;
    Ok(())
}

/// Markdown for display, with LaTeX math turned into Unicode first.
fn render(text: &str) -> StyledText {
    let text = ojp_core::format::latex_to_unicode(text);
    StyledText::from_markdown(&text).unwrap_or_else(|_| StyledText::from_plain_text(&text))
}

fn reply(text: &str, meta: &str) -> Message {
    Message { text: text.into(), rendered: render(text), meta: meta.into(), ..Default::default() }
}

/// Apply `edit` to the newest message (the reply being streamed), then
/// re-render it.
fn edit_last(ui: &slint::Weak<AppWindow>, edit: impl FnOnce(&mut Message) + Send + 'static) {
    let _ = ui.upgrade_in_event_loop(move |ui| {
        let messages = ui.global::<Chat>().get_messages();
        if let Some(last) = messages.row_count().checked_sub(1)
            && let Some(mut message) = messages.row_data(last)
        {
            edit(&mut message);
            message.rendered = render(&message.text);
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

/// One button per storage location; an empty list shows a single
/// "Download" button that keeps the current location.
fn set_download_targets(ui: &slint::Weak<AppWindow>, targets: Vec<String>) {
    let _ = ui.upgrade_in_event_loop(move |ui| {
        let targets: Vec<slint::SharedString> = targets.into_iter().map(Into::into).collect();
        ui.global::<Chat>().set_download_targets(Rc::new(VecModel::from(targets)).into());
    });
}

fn push_message(ui: &slint::Weak<AppWindow>, text: String, meta: &'static str) {
    let _ = ui.upgrade_in_event_loop(move |ui| {
        let messages = ui.global::<Chat>().get_messages();
        let model = messages.as_any().downcast_ref::<VecModel<Message>>().expect("VecModel");
        model.push(reply(&text, meta));
    });
}

fn mb(bytes: u64) -> u64 {
    bytes / 1_000_000
}

/// Load whatever models exist. Missing downloadable models are offered
/// through the download panel instead of an error message.
fn load_engine(setup: &Setup, ui: &slint::Weak<AppWindow>) -> Option<Engine> {
    let config = &setup.config;
    set_state(ui, true, Some("Loading models…".into()));
    let missing = download::missing(config);
    // Offer a choice of storage only before one has been made.
    let choose = setup.locations.len() > 1 && config.models.location.is_none();
    let targets = if choose {
        setup
            .locations
            .iter()
            .map(|l| match l.free_bytes() {
                Some(free) => format!("{} ({:.1} GB free)", l.label, free as f64 / 1e9),
                None => l.label.clone(),
            })
            .collect()
    } else {
        Vec::new()
    };
    set_download_targets(ui, targets);
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
        let main_size = missing.iter().zip(&sizes).find(|(item, _)| item.role == "main").map(|(_, size)| *size);
        let main_label = match main_size {
            Some(Some(size)) => format!("Also download the main model ({} MB), needed for good answers", mb(size)),
            _ => "Also download the main model, needed for good answers".into(),
        };
        let offer_main = main_size.is_some();
        let _ = ui.upgrade_in_event_loop(move |ui| {
            let chat = ui.global::<Chat>();
            chat.set_offer_main(offer_main);
            chat.set_main_label(main_label.into());
        });
        set_download(
            ui,
            true,
            format!(
                "OpenJarvis needs to download its AI models once: {}. They stay on this device, and \
                 everything runs offline after that.{}",
                parts.join(" and "),
                if choose { " Where should they be stored?" } else { "" }
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
                push_message(ui, format!("{e:#}\n\n{}", setup.setup_hint), "setup");
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

/// Save the chosen storage location and reload the config so model paths
/// point there.
fn choose_location(setup: &mut Setup, location: &Location) -> anyhow::Result<()> {
    std::fs::create_dir_all(&location.path)?;
    storage::save_location(&setup.config_path, &location.path)?;
    let (config, warning) = Config::load_or_default(&setup.config_path, &setup.data_dir);
    if let Some(warning) = warning {
        anyhow::bail!(warning);
    }
    setup.config = config;
    Ok(())
}

/// Remember "router only" so the main model isn't offered again.
fn skip_main(setup: &mut Setup) -> anyhow::Result<()> {
    storage::save_skip_main(&setup.config_path)?;
    let (config, warning) = Config::load_or_default(&setup.config_path, &setup.data_dir);
    if let Some(warning) = warning {
        anyhow::bail!(warning);
    }
    setup.config = config;
    Ok(())
}

fn worker(mut setup: Setup, rx: mpsc::Receiver<Request>, ui: slint::Weak<AppWindow>) {
    let mut engine = load_engine(&setup, &ui);

    for request in rx {
        match (request, &mut engine) {
            (Request::Download(index, include_main), _) => {
                if !include_main
                    && let Err(e) = skip_main(&mut setup)
                {
                    set_download(&ui, true, format!("Couldn't save the router-only choice: {e:#}"), -1.0);
                    continue;
                }
                if let Some(location) = usize::try_from(index).ok().and_then(|i| setup.locations.get(i)).cloned()
                    && let Err(e) = choose_location(&mut setup, &location)
                {
                    set_download(&ui, true, format!("Couldn't use {}: {e:#}", location.label), -1.0);
                    continue;
                }
                set_download_targets(&ui, Vec::new());
                set_state(&ui, true, Some("Downloading models…".into()));
                let items = download::missing(&setup.config);
                if download_all(&items, &ui) {
                    engine = load_engine(&setup, &ui);
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
