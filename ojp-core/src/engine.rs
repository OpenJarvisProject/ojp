//! The assistant engine: route each request, then dispatch it (OJP-9).
//!
//! Pieces are independent and optional, and each failure degrades instead
//! of taking the others down:
//!   - routing: the router model, else the in-process main model; if
//!     neither exists, or classification fails, the request is answered
//!     directly
//!   - answering: the main backend (in-process GGUF or OpenAI-compatible
//!     server); if it is missing or fails, the router model answers
//!   - device_control: a grammar-constrained tool call (tools.rs)
//!   - a model file that is missing or fails to load is skipped with a
//!     notice; only having no model at all is an error
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use llama_crab::chat::BuiltinTemplate;
use llama_crab::high_level::completion::SamplingOptions;
use llama_crab::{ChatMessage, CompletionOptions, Llama, LlamaParams, Role, StreamControl};
use serde_json::{Value, json};

use crate::config::Config;
use crate::router::{Route, Router};

const MAX_TOKENS: usize = 512;
/// System prompt + this many most recent messages are sent each turn, so
/// a long chat never overflows the context window.
const HISTORY_MESSAGES: usize = 12;

/// Progress reported while a request is handled, for the UI to render.
pub enum Event {
    /// Which route was chosen, and how long routing took.
    Routed { route: Route, ms: u128 },
    /// Streamed reply text.
    Token(String),
    /// Something degraded (a fallback happened); shown, not fatal.
    Notice(String),
}

enum Main {
    Local(Llama),
    Remote { base_url: String, model: String },
}

pub struct Engine {
    router_llm: Option<Llama>,
    main: Option<Main>,
    router: Router,
    template: BuiltinTemplate,
    system_prompt: String,
    history: Vec<ChatMessage>,
    /// What loaded, and what was skipped and why, for display.
    pub status: Vec<String>,
}

fn load_model(path: &Path, config: &Config) -> Result<Llama> {
    if !path.exists() {
        bail!("{} not found", path.display());
    }
    let params = LlamaParams::new(path)
        .with_n_ctx(config.models.n_ctx)
        .with_n_gpu_layers(config.models.gpu_layers);
    Llama::load(params).with_context(|| format!("loading {}", path.display()))
}

impl Engine {
    pub fn new(config: &Config) -> Result<Engine> {
        let mut status = Vec::new();
        let mut load = |role: &str, path: &Option<std::path::PathBuf>| -> Option<Llama> {
            let path = path.as_ref()?;
            match load_model(path, config) {
                Ok(llama) => {
                    let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into());
                    status.push(format!("{role}: {name}"));
                    Some(llama)
                }
                Err(e) => {
                    status.push(format!("{role}: skipped ({e:#})"));
                    None
                }
            }
        };
        let router_llm = load("router", &config.models.router);
        let main = match (&config.models.main_url, &config.models.main_model) {
            (Some(url), model) => {
                let model = model.clone().unwrap_or_default();
                status.push(format!("main: {model} @ {url}"));
                Some(Main::Remote { base_url: url.clone(), model })
            }
            (None, _) => load("main", &config.models.main).map(Main::Local),
        };
        if router_llm.is_none() && main.is_none() {
            bail!("no model available:\n  {}", status.join("\n  "));
        }

        let template = config.template()?;
        let name = &config.persona.name;
        let system_prompt = format!(
            "You are {name}, a helpful personal assistant running locally on the user's own device. \
             Answer accurately and concisely. If you don't know something, say so."
        );
        Ok(Engine {
            router_llm,
            main,
            router: Router::new(template),
            template,
            history: Vec::new(),
            system_prompt,
            status,
        })
    }

    /// Forget the conversation so far.
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Handle one request; returns the full reply text.
    pub fn ask(&mut self, request: &str, on_event: &mut dyn FnMut(Event)) -> Result<String> {
        let route = self.route(request, on_event);
        self.history.push(ChatMessage::new(Role::User, request));

        let reply = if route == Route::DeviceControl {
            let text = self.device_control(request);
            on_event(Event::Token(text.clone()));
            text
        } else {
            self.answer(on_event)?
        };
        self.history.push(ChatMessage::new(Role::Assistant, reply.clone()));
        Ok(reply)
    }

    fn route(&mut self, request: &str, on_event: &mut dyn FnMut(Event)) -> Route {
        let start = Instant::now();
        let classifier = match (&mut self.router_llm, &mut self.main) {
            (Some(llama), _) | (None, Some(Main::Local(llama))) => llama,
            (None, _) => return Route::AnswerDirectly,
        };
        let route = match self.router.classify(classifier, request) {
            Ok(decision) => decision.route,
            Err(e) => {
                on_event(Event::Notice(format!("routing failed ({e:#}), answering directly")));
                Route::AnswerDirectly
            }
        };
        on_event(Event::Routed { route, ms: start.elapsed().as_millis() });
        route
    }

    fn device_control(&mut self, request: &str) -> String {
        if cfg!(target_os = "android") {
            return "Device control isn't available on Android yet.".into();
        }
        let llama = match (&mut self.router_llm, &mut self.main) {
            (Some(llama), _) | (None, Some(Main::Local(llama))) => llama,
            (None, _) => return "Device control needs a local model.".into(),
        };
        crate::tools::handle(llama, self.template, request).unwrap_or_else(|e| format!("Couldn't do that: {e:#}"))
    }

    fn messages(&self) -> Vec<ChatMessage> {
        let recent = self.history.len().saturating_sub(HISTORY_MESSAGES);
        let mut messages = vec![ChatMessage::new(Role::System, self.system_prompt.clone())];
        messages.extend(self.history[recent..].iter().cloned());
        messages
    }

    fn answer(&mut self, on_event: &mut dyn FnMut(Event)) -> Result<String> {
        let messages = self.messages();
        let main_result = match &mut self.main {
            Some(Main::Local(llama)) => Some(generate_local(llama, self.template, &messages, on_event)),
            Some(Main::Remote { base_url, model }) => Some(generate_remote(base_url, model, &messages, on_event)),
            None => None,
        };
        match (main_result, &mut self.router_llm) {
            (Some(Ok(text)), _) => Ok(text),
            (Some(Err(e)), Some(llama)) => {
                on_event(Event::Notice(format!("main model failed ({e:#}), using the router model")));
                generate_local(llama, self.template, &messages, on_event)
            }
            (Some(Err(e)), None) => Err(e),
            (None, Some(llama)) => generate_local(llama, self.template, &messages, on_event),
            (None, None) => unreachable!("Engine::new requires at least one model"),
        }
    }
}

fn generate_local(
    llama: &mut Llama,
    template: BuiltinTemplate,
    messages: &[ChatMessage],
    on_event: &mut dyn FnMut(Event),
) -> Result<String> {
    let options = CompletionOptions { sampling: SamplingOptions::chat(), ..CompletionOptions::new(MAX_TOKENS) };
    let reply = llama.create_chat_completion_stream_with(messages, template, &[], options, |chunk| {
        if !chunk.text.is_empty() {
            on_event(Event::Token(chunk.text));
        }
        StreamControl::Continue
    })?;
    Ok(reply.content)
}

/// Non-streaming OpenAI-compatible chat completion.
fn generate_remote(
    base_url: &str,
    model: &str,
    messages: &[ChatMessage],
    on_event: &mut dyn FnMut(Event),
) -> Result<String> {
    let body = json!({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": messages.iter().map(|m| json!({"role": m.role.as_str(), "content": m.content})).collect::<Vec<_>>(),
    });
    let reply: Value = ureq::post(format!("{}/v1/chat/completions", base_url.trim_end_matches('/')))
        .send_json(&body)?
        .body_mut()
        .read_json()?;
    let text = reply["choices"][0]["message"]["content"]
        .as_str()
        .context("no choices[0].message.content in reply")?
        .to_string();
    on_event(Event::Token(text.clone()));
    Ok(text)
}
