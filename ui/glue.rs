// UI <-> engine glue shared by the desktop and Android apps (each includes
// this file with #[path] and provides AppWindow/Chat/Message through
// slint::include_modules!). The engine lives on its own thread so model
// loading and generation never block the UI; results come back through
// slint's event loop.
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Instant;

use ojp_core::{Config, Engine, Event};
use slint::{ComponentHandle, Model, VecModel};

use crate::{AppWindow, Chat, Message};

enum Request {
    Ask(String),
    Clear,
}

/// Show the window and run until it closes. `setup_hint` is shown if no
/// model could be loaded, telling the user where to put one.
pub fn run(config: Config, setup_hint: String) -> anyhow::Result<()> {
    let ui = AppWindow::new()?;
    let chat = ui.global::<Chat>();
    chat.set_assistant_name(config.persona.name.clone().into());
    let messages = Rc::new(VecModel::<Message>::default());
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

fn worker(config: Config, setup_hint: String, rx: mpsc::Receiver<Request>, ui: slint::Weak<AppWindow>) {
    let mut engine = match Engine::new(&config) {
        Ok(engine) => {
            set_state(&ui, false, Some(engine.status.join(" · ")));
            Some(engine)
        }
        Err(e) => {
            set_state(&ui, false, Some("No model loaded".into()));
            let text = format!("{e:#}\n\n{setup_hint}");
            let _ = ui.upgrade_in_event_loop(move |ui| {
                let messages = ui.global::<Chat>().get_messages();
                let model = messages.as_any().downcast_ref::<VecModel<Message>>().expect("VecModel");
                model.push(Message { from_user: false, text: text.into(), meta: "setup".into() });
            });
            None
        }
    };

    for request in rx {
        match (request, &mut engine) {
            (Request::Clear, Some(engine)) => engine.clear(),
            (Request::Clear, None) => {}
            (Request::Ask(_), None) => {
                edit_last(&ui, |m| {
                    m.text = "No model is loaded yet; see the setup message.".into();
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
