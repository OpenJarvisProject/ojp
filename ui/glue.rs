// UI <-> engine glue shared by the desktop and Android apps (each includes
// this file with #[path] and provides AppWindow/Chat/Settings/Message
// through slint::include_modules!). A worker thread owns the engine, the
// saved chats and the settings file, so model loading and generation never
// block the UI; it reports back through slint's event loop.
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ojp_core::chats::{ChatStore, StoredChat, StoredMessage};
use ojp_core::download::{self, Item};
use ojp_core::storage::{self, Location};
use ojp_core::{Config, Engine, Event};
use slint::{ComponentHandle, Model, ModelRc, StyledText, VecModel};

use crate::{AppWindow, Block, Chat, ChatSummary, Message, Settings};

enum Request {
    Ask(String),
    /// Start a new, empty chat.
    NewChat,
    OpenChat(String),
    DeleteChat(String),
    /// Download missing models into `Setup::locations[i]`, or into the
    /// current location for a negative index; the bool is "include the
    /// main model".
    Download(i32, bool),
    Settings { theme: i32, dynamic_colors: bool, main_enabled: bool, tts_enabled: bool, tts_mix: ojp_core::tts::MixSlots, tts_speed: i32, tts_engine: i32, tts_moss_voice: i32, tts_clarity: i32 },
    /// Play a sample in the current settings.
    Preview,
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
    /// Applies the "dynamic colours" setting; None where unsupported.
    pub set_dynamic_colors: Option<fn(bool)>,
    /// Picks light or dark system status-bar icons; None where unsupported.
    pub set_light_status_bar: Option<fn(bool)>,
}

fn theme_index(theme: &str) -> i32 {
    match theme {
        "light" => 1,
        "dark" => 2,
        _ => 0,
    }
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

    let settings = ui.global::<Settings>();
    settings.set_theme(theme_index(&setup.config.ui.theme));
    settings.set_dynamic_colors(setup.config.ui.dynamic_colors);
    settings.set_dynamic_colors_supported(setup.set_dynamic_colors.is_some());
    settings.set_main_enabled(setup.config.models.main_enabled);
    settings.set_tts_supported(ojp_core::tts::AVAILABLE);
    settings.set_tts_enabled(ojp_core::tts::AVAILABLE && setup.config.tts.enabled);
    let choices: Vec<slint::SharedString> = std::iter::once("None")
        .chain(ojp_core::tts::VOICES.iter().map(|(_, label)| *label))
        .map(slint::SharedString::from)
        .collect();
    settings.set_tts_voice_choices(ModelRc::new(VecModel::from(choices)));
    let mix = ojp_core::tts::parse_mix(&setup.config.tts_options().voice);
    settings.set_mix_v0(mix[0].0 as i32);
    settings.set_mix_a0(mix[0].1 as f32);
    settings.set_mix_v1(mix[1].0 as i32);
    settings.set_mix_a1(mix[1].1 as f32);
    settings.set_mix_v2(mix[2].0 as i32);
    settings.set_mix_a2(mix[2].1 as f32);
    settings.set_tts_speeds(ModelRc::new(VecModel::from(
        ojp_core::tts::SPEEDS.iter().map(|(_, label)| slint::SharedString::from(*label)).collect::<Vec<_>>(),
    )));
    settings.set_tts_speed(ojp_core::tts::speed_index(setup.config.tts.speed) as i32);
    settings.set_tts_clarities(ModelRc::new(VecModel::from(
        ojp_core::tts::CLARITIES.iter().map(|(_, label)| slint::SharedString::from(*label)).collect::<Vec<_>>(),
    )));
    settings.set_tts_clarity(ojp_core::tts::clarity_index(setup.config.tts.clarity) as i32);
    settings.set_tts_engines(ModelRc::new(VecModel::from(
        ojp_core::tts::Engine::ALL.iter().map(|(_, _, label)| slint::SharedString::from(*label)).collect::<Vec<_>>(),
    )));
    settings.set_tts_engine(ojp_core::tts::Engine::from_id(&setup.config.tts.engine).index() as i32);
    settings.set_tts_moss_voices(ModelRc::new(VecModel::from(
        ojp_core::tts::moss_voice_list(&setup.config.tts.moss_dir)
            .into_iter()
            .map(|(_, label)| slint::SharedString::from(label))
            .collect::<Vec<_>>(),
    )));
    settings.set_tts_moss_voice(
        ojp_core::tts::moss_voice_index(&ojp_core::tts::moss_voice_list(&setup.config.tts.moss_dir), &setup.config.tts.moss_voice) as i32,
    );
    if let Some(apply) = setup.set_light_status_bar {
        // Dark icons on a light theme.
        settings.on_dark_changed(move |dark| apply(!dark));
    }
    if let Some(apply) = setup.set_dynamic_colors {
        apply(setup.config.ui.dynamic_colors);
    }

    let (tx, rx) = mpsc::channel();
    let weak = ui.as_weak();
    std::thread::spawn(move || worker(setup, rx, weak));

    let send = move |request| {
        let _ = tx.send(request);
    };
    chat.on_send({
        let (send, messages, weak) = (send.clone(), messages.clone(), ui.as_weak());
        move |text| {
            messages.push(Message { from_user: true, text: text.clone(), ..Default::default() });
            messages.push(Message { meta: "thinking…".into(), ..Default::default() });
            if let Some(ui) = weak.upgrade() {
                ui.global::<Chat>().set_busy(true);
            }
            send(Request::Ask(text.into()));
        }
    });
    chat.on_download({
        let (send, weak) = (send.clone(), ui.as_weak());
        move |index| {
            let include_main = weak.upgrade().is_none_or(|ui| ui.global::<Chat>().get_download_main());
            send(Request::Download(index, include_main));
        }
    });
    chat.on_clear({
        let send = send.clone();
        move || send(Request::NewChat)
    });
    chat.on_open_chat({
        let send = send.clone();
        move |id| send(Request::OpenChat(id.into()))
    });
    chat.on_delete_chat({
        let send = send.clone();
        move |id| send(Request::DeleteChat(id.into()))
    });
    settings.on_preview({
        let send = send.clone();
        move || send(Request::Preview)
    });
    settings.on_changed({
        let weak = ui.as_weak();
        move || {
            let Some(ui) = weak.upgrade() else { return };
            let s = ui.global::<Settings>();
            send(Request::Settings {
                theme: s.get_theme(),
                dynamic_colors: s.get_dynamic_colors(),
                main_enabled: s.get_main_enabled(),
                tts_enabled: s.get_tts_enabled(),
                tts_mix: [
                    (s.get_mix_v0() as usize, s.get_mix_a0().round() as u32),
                    (s.get_mix_v1() as usize, s.get_mix_a1().round() as u32),
                    (s.get_mix_v2() as usize, s.get_mix_a2().round() as u32),
                ],
                tts_speed: s.get_tts_speed(),
                tts_engine: s.get_tts_engine(),
                tts_moss_voice: s.get_tts_moss_voice(),
                tts_clarity: s.get_tts_clarity(),
            });
        }
    });
    ui.run()?;
    Ok(())
}

/// Markdown for display, with LaTeX math turned into Unicode first.
fn render(text: &str) -> StyledText {
    let text = ojp_core::format::latex_to_unicode(text);
    StyledText::from_markdown(&text).unwrap_or_else(|_| StyledText::from_plain_text(&text))
}

thread_local! {
    /// Typeset formulas by LaTeX source (None = it didn't typeset), so
    /// re-rendering a streaming reply doesn't redo finished formulas.
    static FORMULAS: std::cell::RefCell<std::collections::HashMap<String, Option<slint::Image>>> = Default::default();
}

fn formula_image(latex: &str) -> Option<slint::Image> {
    FORMULAS.with_borrow_mut(|cache| {
        cache
            .entry(latex.to_string())
            .or_insert_with(|| {
                let r = ojp_core::math::render(latex, 18.0)?;
                let pixels = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(&r.rgba, r.width, r.height);
                Some(slint::Image::from_rgba8(pixels))
            })
            .clone()
    })
}

/// The pieces of a reply with typeset display math, or none if it has no
/// (complete, valid) display math and `rendered` is all there is.
fn blocks(text: &str) -> ModelRc<Block> {
    use ojp_core::math::Segment;
    let segments = ojp_core::math::split(text);
    let mut blocks = Vec::new();
    let mut any_image = false;
    for segment in segments {
        match segment {
            Segment::Text(t) if t.trim().is_empty() => {}
            Segment::Text(t) => blocks.push(Block { text: render(&t), image: Default::default(), is_math: false }),
            Segment::Display(latex) => match formula_image(&latex) {
                Some(image) => {
                    any_image = true;
                    blocks.push(Block { text: Default::default(), image, is_math: true });
                }
                // Didn't typeset: show the Unicode approximation instead.
                None => blocks.push(Block {
                    text: render(&format!("$${latex}$$")),
                    image: Default::default(),
                    is_math: false,
                }),
            },
        }
    }
    if !any_image {
        blocks.clear();
    }
    ModelRc::new(VecModel::from(blocks))
}

fn reply(text: &str, meta: &str) -> Message {
    Message { text: text.into(), rendered: render(text), blocks: blocks(text), meta: meta.into(), ..Default::default() }
}

fn to_message(m: &StoredMessage) -> Message {
    if m.from_user {
        Message { from_user: true, text: m.text.as_str().into(), meta: m.meta.as_str().into(), ..Default::default() }
    } else {
        reply(&m.text, &m.meta)
    }
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
            message.blocks = blocks(&message.text);
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

/// Show `chat` on screen and refresh the sidebar list.
fn show_chat(ui: &slint::Weak<AppWindow>, store: &ChatStore, chat: &StoredChat) {
    let messages: Vec<StoredMessage> = chat.messages.clone();
    let id = if chat.messages.is_empty() { String::new() } else { chat.id.clone() };
    let list: Vec<(String, String)> = store.list().into_iter().map(|c| (c.id, c.title)).collect();
    let _ = ui.upgrade_in_event_loop(move |ui| {
        let chat = ui.global::<Chat>();
        let model = chat.get_messages();
        let model = model.as_any().downcast_ref::<VecModel<Message>>().expect("VecModel");
        model.set_vec(messages.iter().map(to_message).collect::<Vec<_>>());
        chat.set_current_chat(id.into());
        let list: Vec<ChatSummary> =
            list.into_iter().map(|(id, title)| ChatSummary { id: id.into(), title: title.into() }).collect();
        chat.set_chats(Rc::new(VecModel::from(list)).into());
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
    } else {
        set_download(ui, false, String::new(), -1.0);
    }
    let engine = match Engine::new(config) {
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
    };
    let base = config.models.location.clone().unwrap_or_else(|| setup.data_dir.clone());
    let info = format!(
        "{}\nModels: {}\nSettings file: {}",
        engine.as_ref().map_or("No model loaded".into(), |e| e.status.join(" · ")),
        base.join("models").display(),
        setup.config_path.display()
    );
    let _ = ui.upgrade_in_event_loop(move |ui| ui.global::<Settings>().set_info(info.into()));
    engine
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

/// Re-read config.toml after the host changed it.
fn reload_config(setup: &mut Setup) -> anyhow::Result<()> {
    let (config, warning) = Config::load_or_default(&setup.config_path, &setup.data_dir);
    if let Some(warning) = warning {
        anyhow::bail!(warning);
    }
    setup.config = config;
    Ok(())
}

/// Save the chosen storage location and reload the config so model paths
/// point there.
fn choose_location(setup: &mut Setup, location: &Location) -> anyhow::Result<()> {
    std::fs::create_dir_all(&location.path)?;
    storage::save_location(&setup.config_path, &location.path)?;
    reload_config(setup)
}

fn worker(mut setup: Setup, rx: mpsc::Receiver<Request>, ui: slint::Weak<AppWindow>) {
    let store = ChatStore::new(&setup.data_dir);
    let mut chat = StoredChat::new();
    show_chat(&ui, &store, &chat);
    let mut engine = load_engine(&setup, &ui);
    // Loaded the first time a reply is spoken, dropped when speech is turned off.
    let mut tts: Option<ojp_core::tts::Tts> = None;

    for request in rx {
        // Whatever is being said stops when the user does anything.
        if let Some(tts) = &tts {
            tts.stop();
        }
        match request {
            Request::Download(index, include_main) => {
                if !include_main
                    && let Err(e) = storage::save_skip_main(&setup.config_path).and_then(|_| reload_config(&mut setup))
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
                    if let Some(engine) = engine.as_mut() {
                        engine.set_history(chat.messages.iter().map(|m| (m.from_user, m.text.clone())));
                    }
                } else {
                    set_state(&ui, false, Some("Download failed".into()));
                }
            }
            Request::NewChat => {
                chat = StoredChat::new();
                if let Some(engine) = engine.as_mut() {
                    engine.clear();
                }
                show_chat(&ui, &store, &chat);
            }
            Request::OpenChat(id) => match store.load(&id) {
                Ok(loaded) => {
                    chat = loaded;
                    if let Some(engine) = engine.as_mut() {
                        engine.set_history(chat.messages.iter().map(|m| (m.from_user, m.text.clone())));
                    }
                    show_chat(&ui, &store, &chat);
                }
                Err(e) => push_message(&ui, format!("Couldn't open that chat: {e:#}"), "error"),
            },
            Request::DeleteChat(id) => {
                let _ = store.delete(&id);
                if id == chat.id {
                    chat = StoredChat::new();
                    if let Some(engine) = engine.as_mut() {
                        engine.clear();
                    }
                }
                show_chat(&ui, &store, &chat);
            }
            Request::Settings { theme, dynamic_colors, main_enabled, tts_enabled, tts_mix, tts_speed, tts_engine, tts_moss_voice, tts_clarity } => {
                let voice_id = ojp_core::tts::mix_string(&tts_mix);
                let voice_id = voice_id.as_str();
                let speed = ojp_core::tts::SPEEDS.get(tts_speed as usize).map_or(1.0, |(v, _)| *v);
                let clarity = ojp_core::tts::CLARITIES.get(tts_clarity as usize).map_or(0.65, |(v, _)| *v);
                let engine_id = ojp_core::tts::Engine::ALL.get(tts_engine as usize).map_or("kokoro", |(_, id, _)| *id);
                let moss_voices = ojp_core::tts::moss_voice_list(&setup.config.tts.moss_dir);
                let moss_voice = moss_voices.get(tts_moss_voice as usize).map_or("Bella", |(id, _)| id.as_str());
                let theme = ["system", "light", "dark"].get(theme as usize).copied().unwrap_or("system");
                let path = &setup.config_path;
                let saved = storage::set_str(path, "ui", "theme", theme)
                    .and_then(|_| storage::set_bool(path, "ui", "dynamic_colors", dynamic_colors))
                    .and_then(|_| storage::set_bool(path, "models", "main_enabled", main_enabled))
                    .and_then(|_| storage::set_bool(path, "tts", "enabled", tts_enabled))
                    .and_then(|_| storage::set_str(path, "tts", "voice", voice_id))
                    .and_then(|_| storage::set_f64(path, "tts", "speed", speed as f64))
                    .and_then(|_| storage::set_f64(path, "tts", "clarity", clarity as f64))
                    .and_then(|_| storage::set_str(path, "tts", "engine", engine_id))
                    .and_then(|_| storage::set_str(path, "tts", "moss_voice", moss_voice));
                if let Err(e) = saved {
                    push_message(&ui, format!("Couldn't save settings: {e:#}"), "settings");
                }
                if let Some(apply) = setup.set_dynamic_colors {
                    apply(dynamic_colors);
                }
                let main_changed = main_enabled != setup.config.models.main_enabled;
                let tts_changed = tts_enabled != setup.config.tts.enabled || engine_id != setup.config.tts.engine;
                let _ = reload_config(&mut setup);
                if let Some(tts) = &tts
                    && !tts_changed
                {
                    tts.set_voice(&setup.config.tts_options().voice);
                    tts.set_speed(speed);
                    tts.set_clarity(clarity);
                }
                if tts_changed {
                    tts = None;
                    if tts_enabled && !enable_tts(&mut setup, &ui, &mut tts) {
                        let _ = storage::set_bool(&setup.config_path, "tts", "enabled", false);
                        let _ = reload_config(&mut setup);
                    }
                }
                if main_changed {
                    // Free the old models before loading the new set.
                    drop(engine.take());
                    engine = load_engine(&setup, &ui);
                    if let Some(engine) = engine.as_mut() {
                        engine.set_history(chat.messages.iter().map(|m| (m.from_user, m.text.clone())));
                    }
                }
            }
            Request::Preview => {
                if setup.config.tts.enabled {
                    if tts.is_none() {
                        enable_tts(&mut setup, &ui, &mut tts);
                    }
                    if let Some(tts) = &tts {
                        // Not the persona's name: Kokoro slurs "Jarvis" into the words around it.
                        tts.set_voice(&setup.config.tts_options().voice);
                        tts.set_speed(setup.config.tts.speed);
                        tts.set_clarity(setup.config.tts.clarity);
                        tts.speak("Hello. I am your AI assistant. This is how I sound.");
                    }
                }
            }
            Request::Ask(_) if engine.is_none() => {
                edit_last(&ui, |m| {
                    m.text = "No model is loaded yet. Use the Download button below first.".into();
                    m.meta = "".into();
                });
                set_state(&ui, false, None);
            }
            Request::Ask(text) => {
                let engine = engine.as_mut().expect("checked above");
                let start = Instant::now();
                let mut meta = Vec::new();
                let result = engine.ask(&text, &mut |event| match event {
                    Event::Routed { route, ms } => meta.push(format!("{} ({ms} ms)", route.label())),
                    Event::Token(token) => edit_last(&ui, move |m| m.text = format!("{}{token}", m.text).into()),
                    Event::Notice(notice) => meta.push(notice),
                });
                if let Err(e) = &result {
                    meta.push(format!("error: {e:#}"));
                }
                meta.push(format!("{:.1} s", start.elapsed().as_secs_f32()));
                let meta = meta.join(" · ");

                chat.messages.push(StoredMessage { from_user: true, text, meta: String::new() });
                chat.messages.push(StoredMessage {
                    from_user: false,
                    text: result.unwrap_or_default(),
                    meta: meta.clone(),
                });
                chat.refresh_title();
                chat.touch();
                if let Err(e) = store.save(&chat) {
                    meta_note(&ui, format!("{meta} · not saved: {e:#}"));
                } else {
                    meta_note(&ui, meta);
                }
                refresh_list(&ui, &store, &chat);
                set_state(&ui, false, None);
                if setup.config.tts.enabled {
                    if tts.is_none() {
                        // Loaded at startup was skipped; do it now.
                        enable_tts(&mut setup, &ui, &mut tts);
                    }
                    if let (Some(tts), Some(reply)) = (&tts, chat.messages.last()) {
                        tts.speak(&reply.text);
                    }
                }
            }
        }
    }
}

/// Turn speech on: download the voice if needed, then load it into `tts`.
/// Reports problems in the chat and returns false if it couldn't.
fn enable_tts(setup: &mut Setup, ui: &slint::Weak<AppWindow>, tts: &mut Option<ojp_core::tts::Tts>) -> bool {
    let fail = |ui: &slint::Weak<AppWindow>, why: String| {
        push_message(ui, format!("Couldn't turn on speech: {why}"), "voice");
        let weak = ui.clone();
        let _ = weak.upgrade_in_event_loop(|ui| ui.global::<Settings>().set_tts_enabled(false));
        false
    };
    let items = if setup.config.tts_options().engine == ojp_core::tts::Engine::Kokoro {
        download::tts_missing(&setup.config)
    } else {
        Vec::new()
    };
    if !items.is_empty() {
        set_state(ui, true, Some("Downloading the voice…".into()));
        let ok = download_all(&items, ui);
        set_state(ui, false, Some(if ok { "Ready".into() } else { "Download failed".into() }));
        if !ok {
            return fail(ui, "the voice download failed (try again from Settings)".into());
        }
    }
    set_state(ui, true, Some("Loading the voice…".into()));
    let result = ojp_core::tts::Tts::start(setup.config.tts_options());
    set_state(ui, false, Some("Ready".into()));
    match result {
        Ok(started) => {
            *tts = Some(started);
            true
        }
        Err(e) => fail(ui, format!("{e:#}")),
    }
}

fn meta_note(ui: &slint::Weak<AppWindow>, meta: String) {
    edit_last(ui, move |m| m.meta = meta.into());
}

fn refresh_list(ui: &slint::Weak<AppWindow>, store: &ChatStore, chat: &StoredChat) {
    let list: Vec<(String, String)> = store.list().into_iter().map(|c| (c.id, c.title)).collect();
    let id = chat.id.clone();
    let _ = ui.upgrade_in_event_loop(move |ui| {
        let c = ui.global::<Chat>();
        let list: Vec<ChatSummary> =
            list.into_iter().map(|(id, title)| ChatSummary { id: id.into(), title: title.into() }).collect();
        c.set_chats(Rc::new(VecModel::from(list)).into());
        c.set_current_chat(id.into());
    });
}
