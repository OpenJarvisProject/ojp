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
use std::time::Instant;

use anyhow::{Context, Result, bail};
use llama_crab::chat::BuiltinTemplate;
use llama_crab::chat::template::render_builtin;
use llama_crab::high_level::completion::SamplingOptions;
use llama_crab::{ChatMessage, Role};
use serde_json::{Value, json};

use crate::config::Config;
use crate::model::{Model, Seq};
use crate::router::{Route, Router};
use crate::tools;

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
    Local(Model),
    Remote { base_url: String, model: String },
}

pub struct Engine {
    router_llm: Option<Model>,
    main: Option<Main>,
    router: Router,
    template: BuiltinTemplate,
    system_prompt: String,
    history: Vec<ChatMessage>,
    /// What loaded, and what was skipped and why, for display.
    pub status: Vec<String>,
}

impl Engine {
    pub fn new(config: &Config) -> Result<Engine> {
        let mut status = Vec::new();
        let mut load = |role: &str, path: &Option<std::path::PathBuf>| -> Option<Model> {
            let path = path.as_ref()?;
            match Model::load(path, config.models.n_ctx, config.models.gpu_layers) {
                Ok(model) => {
                    let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into());
                    status.push(format!("{role}: {name}"));
                    Some(model)
                }
                Err(_) if !path.exists() => {
                    status.push(format!("{role}: not downloaded"));
                    None
                }
                Err(e) => {
                    status.push(format!("{role}: failed to load ({e:#})"));
                    None
                }
            }
        };
        let mut router_llm = load("router", &config.models.router);
        let mut main = match (&config.models.main_url, &config.models.main_model) {
            (Some(url), model) => {
                let model = model.clone().unwrap_or_default();
                status.push(format!("main: {model} @ {url}"));
                Some(Main::Remote { base_url: url.clone(), model })
            }
            (None, _) if !config.models.main_enabled => {
                status.push("main: off".into());
                None
            }
            (None, _) => load("main", &config.models.main).map(Main::Local),
        };
        if router_llm.is_none() && main.is_none() {
            bail!("no model available:\n  {}", status.join("\n  "));
        }

        let template = config.template()?;
        // Announce timers and reminders that come due (and any missed while closed).
        if !cfg!(target_os = "android") {
            crate::reminders::start_watcher();
        }
        let router = Router::new(template);
        let name = &config.persona.name;
        let system_prompt = format!(
            "You are {name}, a helpful personal assistant running locally on the user's own device. \
             Answer accurately and concisely. If you don't know something, say so. \
             Write math in LaTeX: put standalone equations on their own line between $$ and $$, \
             and inline math between \\( and \\)."
        );
        // Warm-up, done while loading so the first message is fast: the
        // router evaluates its few-shot prompt, and whichever model answers
        // evaluates the system prompt (which also pages its memory-mapped
        // weights in). The resulting KV cache is saved next to the models
        // and restored on later starts instead of being recomputed.
        let chat_prompt = render_builtin(template, &[ChatMessage::new(Role::System, system_prompt.clone())], &[], false);
        let router_only = main.is_none();
        let warm_chat = |model: &mut Model| {
            if let Ok(mut sampler) = llama_crab::LlamaSampler::greedy().context("greedy") {
                let _ = model.complete(Seq::Chat, &chat_prompt, &mut sampler, 1, &mut |_| {});
            }
        };
        if let Some(model) = router_llm.as_mut() {
            let tool_prompt = tools::system_prompt();
            let path = cache_path(config, config.models.router.as_deref(), &[&chat_prompt, &format!("{router_only}"), &tool_prompt]);
            if path.as_ref().is_none_or(|p| model.load_cache(p).is_err()) {
                let _ = router.classify(model, "hello");
                // Evaluates the tool prompt once so the first tool call is fast.
                let _ = tools::ask(model, template, "what time is it");
                if router_only {
                    warm_chat(model);
                }
                if let Some(p) = &path {
                    let _ = model.save_cache(p);
                }
            }
        }
        if let Some(Main::Local(model)) = main.as_mut() {
            // The main model also routes and picks tools when it is loaded (it is
            // much better at both than the small router), so warm those prompts too.
            let tool_prompt = tools::system_prompt();
            let path = cache_path(config, config.models.main.as_deref(), &[&chat_prompt, &tool_prompt, "routes+tools"]);
            if path.as_ref().is_none_or(|p| model.load_cache(p).is_err()) {
                warm_chat(model);
                let _ = router.classify(model, "hello");
                let _ = tools::ask(model, template, "what time is it");
                if let Some(p) = &path {
                    let _ = model.save_cache(p);
                }
            }
        }
        Ok(Engine {
            router_llm,
            main,
            router,
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

    /// Continue an earlier conversation: (from_user, text) pairs, oldest first.
    pub fn set_history(&mut self, messages: impl IntoIterator<Item = (bool, String)>) {
        self.history = messages
            .into_iter()
            .map(|(from_user, text)| ChatMessage::new(if from_user { Role::User } else { Role::Assistant }, text))
            .collect();
    }

    /// Handle one request; returns the full reply text.
    pub fn ask(&mut self, request: &str, on_event: &mut dyn FnMut(Event)) -> Result<String> {
        // Typed commands ("/volume 40", "/calc 17*23", "/help") skip the models
        // entirely, so they are instant and always do exactly what they say.
        if let Some(parsed) = tools::command(request) {
            let reply = self.run_command(parsed, on_event);
            self.history.push(ChatMessage::new(Role::User, request));
            self.history.push(ChatMessage::new(Role::Assistant, reply.clone()));
            return Ok(reply);
        }
        let route = self.route(request, on_event);
        self.history.push(ChatMessage::new(Role::User, request));

        let reply = if route == Route::DeviceControl {
            self.use_tool(request, on_event)?
        } else {
            self.answer(on_event)?
        };
        self.history.push(ChatMessage::new(Role::Assistant, reply.clone()));
        Ok(reply)
    }

    fn route(&mut self, request: &str, on_event: &mut dyn FnMut(Event)) -> Route {
        let start = Instant::now();
        // The in-process main model, when loaded, is far more accurate than the
        // 0.5B router (measured with `ojp-cli route-eval` and `tool-eval`).
        let classifier = match (&mut self.router_llm, &mut self.main) {
            (_, Some(Main::Local(model))) | (Some(model), _) => model,
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

    /// Run a typed command and show its result as-is.
    fn run_command(&mut self, parsed: Result<Vec<serde_json::Value>, String>, on_event: &mut dyn FnMut(Event)) -> String {
        let text = match parsed {
            Err(message) => message,
            Ok(calls) => {
                let run = tools::run_calls(&calls);
                for o in &run.outcomes {
                    on_event(Event::Notice(format!("command: {}", tools::describe_call(&o.call))));
                }
                let mut lines: Vec<String> = run.outcomes.iter().map(|o| o.text.clone()).collect();
                if let Some(e) = run.error {
                    lines.push(format!("Couldn't do that: {e:#}"));
                }
                lines.join("\n")
            }
        };
        on_event(Event::Token(text.clone()));
        text
    }

    /// Run the tools for `request` and produce the reply. Actions answer with
    /// their own confirmations; lookups are phrased by the answering model
    /// from the looked-up facts (shown as-is if that fails).
    fn use_tool(&mut self, request: &str, on_event: &mut dyn FnMut(Event)) -> Result<String> {
        let run = match self.device_control(request) {
            Ok(run) => run,
            Err(e) => {
                let text = format!("Couldn't do that: {e:#}");
                on_event(Event::Token(text.clone()));
                return Ok(text);
            }
        };
        for o in &run.outcomes {
            on_event(Event::Notice(format!("tool: {}", tools::describe_call(&o.call))));
        }
        let acts: Vec<&str> = run.outcomes.iter().filter(|o| o.kind == tools::Kind::Act).map(|o| o.text.as_str()).collect();
        let facts: Vec<&str> = run.outcomes.iter().filter(|o| o.kind == tools::Kind::Read).map(|o| o.text.as_str()).collect();
        let failure = run.error.as_ref().map(|e| format!("Couldn't do the rest: {e:#}"));

        let mut text = if facts.is_empty() {
            acts.join(" ")
        } else {
            let last = self.history.len() - 1;
            let done = if acts.is_empty() { String::new() } else { format!("Already done: {}\n", acts.join(" ")) };
            self.history[last] = ChatMessage::new(
                Role::User,
                format!(
                    "{request}\n\n{done}[Looked up on this computer: {}]\nAnswer in one or two short sentences using only that.",
                    facts.join("\n")
                ),
            );
            let phrased = self.answer(on_event);
            self.history[last] = ChatMessage::new(Role::User, request);
            match phrased {
                Ok(t) if !t.trim().is_empty() => return Ok(self.with_failure(t, failure, on_event)),
                _ => {
                    let shown = format!("{}{}", acts.join(" "), if acts.is_empty() { facts.join("\n") } else { format!(" {}", facts.join("\n")) });
                    on_event(Event::Token(shown.clone()));
                    shown
                }
            }
        };
        if text.is_empty() && failure.is_none() {
            text = "Done.".into();
        }
        if !text.is_empty() {
            on_event(Event::Token(text.clone()));
        }
        Ok(self.with_failure(text, failure, on_event))
    }

    /// Append the "couldn't do the rest" line (if any) to a reply.
    fn with_failure(&self, text: String, failure: Option<String>, on_event: &mut dyn FnMut(Event)) -> String {
        match failure {
            None => text,
            Some(f) => {
                let line = if text.is_empty() { f } else { format!("\n{f}") };
                on_event(Event::Token(line.clone()));
                format!("{text}{line}")
            }
        }
    }

    fn device_control(&mut self, request: &str) -> Result<tools::Run> {
        if cfg!(target_os = "android") {
            bail!("device control isn't available on Android yet");
        }
        let model = match (&mut self.router_llm, &mut self.main) {
            (_, Some(Main::Local(model))) | (Some(model), _) => model,
            (None, _) => bail!("device control needs a local model"),
        };
        tools::handle(model, self.template, request)
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
            Some(Main::Local(model)) => Some(generate_local(model, self.template, &messages, on_event)),
            Some(Main::Remote { base_url, model }) => Some(generate_remote(base_url, model, &messages, on_event)),
            None => None,
        };
        match (main_result, &mut self.router_llm) {
            (Some(Ok(text)), _) => Ok(text),
            (Some(Err(e)), Some(model)) => {
                on_event(Event::Notice(format!("main model failed ({e:#}), using the router model")));
                generate_local(model, self.template, &messages, on_event)
            }
            (Some(Err(e)), None) => Err(e),
            (None, Some(model)) => generate_local(model, self.template, &messages, on_event),
            (None, None) => unreachable!("Engine::new requires at least one model"),
        }
    }
}

fn generate_local(
    model: &mut Model,
    template: BuiltinTemplate,
    messages: &[ChatMessage],
    on_event: &mut dyn FnMut(Event),
) -> Result<String> {
    // The rendered history only grows at the end between turns, so the
    // prompt cache makes each turn evaluate just the newest messages.
    let prompt = render_builtin(template, messages, &[], true);
    let mut sampler = SamplingOptions::chat().build_sampler(model.llama())?;
    let text = model.complete(Seq::Chat, &prompt, &mut sampler, MAX_TOKENS, &mut |t| on_event(Event::Token(t.to_string())))?;
    Ok(text.trim().to_string())
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

/// Where a model's warm-up cache lives: `<models dir>/.cache/<name>-<key>.kv`,
/// keyed on everything that would change its contents (the model file's
/// path, size and modification time, the prompts, this build).
fn cache_path(config: &Config, model: Option<&std::path::Path>, prompts: &[&str]) -> Option<std::path::PathBuf> {
    use std::hash::{Hash, Hasher};
    let model = model?;
    let meta = std::fs::metadata(model).ok()?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    model.hash(&mut h);
    meta.len().hash(&mut h);
    meta.modified().ok().hash(&mut h);
    prompts.hash(&mut h);
    crate::router::PROMPT_VERSION.hash(&mut h);
    config.models.n_ctx.hash(&mut h);
    config.models.gpu_layers.hash(&mut h);
    env!("CARGO_PKG_VERSION").hash(&mut h);
    let name = model.file_stem()?.to_string_lossy();
    Some(model.parent()?.join(".cache").join(format!("{name}-{:016x}.kv", h.finish())))
}
