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
use crate::template::Template;
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

/// Something the assistant wants to do but is waiting for permission to do.
struct Pending {
    call: serde_json::Value,
}

/// How a reply to "shall I go ahead?" reads.
#[derive(Debug, PartialEq, Eq)]
enum Answer {
    Yes,
    No,
    Other,
}

fn interpret(reply: &str) -> Answer {
    let r = reply.trim().trim_end_matches(['.', '!']).to_lowercase();
    match r.as_str() {
        "yes" | "y" | "yeah" | "yep" | "yup" | "ok" | "okay" | "sure" | "do it" | "go ahead" | "go" | "confirm" | "allow" | "please do" | "yes please" | "proceed" => Answer::Yes,
        "no" | "n" | "nope" | "nah" | "cancel" | "stop" | "don't" | "dont" | "no thanks" | "no thank you" | "abort" => Answer::No,
        _ => Answer::Other,
    }
}

enum Main {
    Local(Model),
    Remote { base_url: String, model: String },
}

pub struct Engine {
    router_llm: Option<Model>,
    main: Option<Main>,
    router: Router,
    template: Template,
    system_prompt: String,
    history: Vec<ChatMessage>,
    /// An action waiting for the user's yes or no.
    pending: Option<Pending>,
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
        crate::tools_ext::set_policy(crate::tools_ext::Policy { internet: config.tools.internet });
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
             and inline math between \\( and \\). You have a memory: things the user told you in \
             earlier chats appear in square brackets before their message. Use them naturally, and \
             never claim you cannot remember or lack access to personal information when they are there."
        );
        // Warm-up, done while loading so the first message is fast: the
        // router evaluates its few-shot prompt, and whichever model answers
        // evaluates the system prompt (which also pages its memory-mapped
        // weights in). The resulting KV cache is saved next to the models
        // and restored on later starts instead of being recomputed.
        let chat_prompt = template.render(&[ChatMessage::new(Role::System, system_prompt.clone())], false);
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
                if model.is_recurrent() {
                    // A recurrent model can't rewind: it snapshots where two prompts diverge.
                    let _ = router.classify(model, "what time is it");
                    let _ = tools::ask(model, template, "hello");
                }
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
                if model.is_recurrent() {
                    let _ = router.classify(model, "what time is it");
                    let _ = tools::ask(model, template, "hello");
                }
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
            pending: None,
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
        // The user is answering "shall I go ahead?"
        if let Some(pending) = self.pending.take() {
            match interpret(request) {
                Answer::Yes => {
                    let text = match tools::run_confirmed(&pending.call) {
                        Ok(o) => o.text,
                        Err(e) => format!("Couldn't do that: {e:#}"),
                    };
                    on_event(Event::Notice(format!("tool: {}", tools::describe_call(&pending.call))));
                    on_event(Event::Token(text.clone()));
                    self.history.push(ChatMessage::new(Role::User, request));
                    self.history.push(ChatMessage::new(Role::Assistant, text.clone()));
                    return Ok(text);
                }
                Answer::No => {
                    let text = "Okay, I won't do that.".to_string();
                    on_event(Event::Token(text.clone()));
                    self.history.push(ChatMessage::new(Role::User, request));
                    self.history.push(ChatMessage::new(Role::Assistant, text.clone()));
                    return Ok(text);
                }
                // Something else: the earlier question is dropped and this is a new request.
                Answer::Other => on_event(Event::Notice("dropped the earlier action you hadn't confirmed".into())),
            }
        }
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
            self.answer_with_memory(request, on_event)?
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
                let run = tools::run_typed(&calls);
                if run.outcomes.iter().any(|o| tools::writes_memory(&o.call)) {
                    self.refine_memory();
                }
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

    /// Work out and run what `request` needs, step by step: the model picks
    /// calls, sees what they found, and may pick more (up to five rounds), or
    /// stop. Anything that changes something that matters waits for a yes.
    /// Actions answer with their own confirmations; lookups are phrased by the
    /// answering model from the findings (shown as-is if that fails).
    fn use_tool(&mut self, request: &str, on_event: &mut dyn FnMut(Event)) -> Result<String> {
        const MAX_STEPS: usize = 5;
        let mut steps: Vec<tools::Step> = Vec::new();
        let mut acts: Vec<String> = Vec::new();
        let mut facts: Vec<String> = Vec::new();
        let mut failure: Option<String> = None;
        let mut ask_first: Option<String> = None;

        // Becomes true once the request has read text that someone else wrote.
        let mut tainted = false;
        'steps: for _ in 0..MAX_STEPS {
            let calls = match self.plan_step(request, &steps) {
                Ok(c) => c,
                Err(e) if steps.is_empty() => {
                    let text = format!("Couldn't do that: {e:#}");
                    on_event(Event::Token(text.clone()));
                    return Ok(text);
                }
                Err(_) => break,
            };
            let mut results = Vec::new();
            let mut explored = false;
            for call in calls.iter().filter(|c| !tools::is_done(c)) {
                // Text from a web page or file must not be able to plant "memories".
                if tainted && tools::writes_memory(call) {
                    let what = call["args"]["text"].as_str().unwrap_or("");
                    self.pending = Some(Pending { call: call.clone() });
                    ask_first = Some(format!("I'd like to save this to memory: \"{what}\". It came from something I just read, so I'm checking with you first. Shall I? (yes / no)"));
                    break 'steps;
                }
                if let Some(plan) = tools::needs_confirmation(call) {
                    match plan {
                        Ok(what) => {
                            self.pending = Some(Pending { call: call.clone() });
                            ask_first = Some(format!("I'd like to {what}. Shall I go ahead? (yes / no)"));
                        }
                        Err(e) => failure = Some(format!("I can't do that: {e:#}")),
                    }
                    break 'steps;
                }
                match tools::run_call(call) {
                    Ok(o) => {
                        on_event(Event::Notice(format!("tool: {}", tools::describe_call(&o.call))));
                        if tools::writes_memory(&o.call) && std::env::var("OJP_TOOLS_DRY_RUN").is_err() {
                            self.refine_memory();
                        }
                        explored |= tools::is_explore(&o.call);
                        tainted |= tools::reads_untrusted(&o.call);
                        results.push(o.text.clone());
                        match o.kind {
                            tools::Kind::Act => acts.push(o.text),
                            tools::Kind::Read => facts.push(o.text),
                        }
                    }
                    Err(e) => {
                        failure = Some(format!("Couldn't do that: {e:#}"));
                        break 'steps;
                    }
                }
            }
            let finished = calls.iter().any(tools::is_done);
            steps.push(tools::Step { calls, results });
            if finished || !explored {
                break;
            }
        }

        let mut text = if facts.is_empty() {
            acts.join(" ")
        } else {
            let last = self.history.len() - 1;
            let done = if acts.is_empty() { String::new() } else { format!("Already done: {}\n", acts.join(" ")) };
            self.history[last] = ChatMessage::new(
                Role::User,
                format!("{request}\n\n{done}[Looked up on this computer: {}]\nAnswer in one or two short sentences using only that.", facts.join("\n")),
            );
            let phrased = self.answer(on_event);
            self.history[last] = ChatMessage::new(Role::User, request);
            match phrased {
                Ok(t) if !t.trim().is_empty() => return Ok(self.finish(t, failure, ask_first, on_event)),
                _ => {
                    let shown = if acts.is_empty() { facts.join("\n") } else { format!("{} {}", acts.join(" "), facts.join("\n")) };
                    on_event(Event::Token(shown.clone()));
                    shown
                }
            }
        };
        if text.is_empty() && failure.is_none() && ask_first.is_none() {
            text = "Done.".into();
        } else if !text.is_empty() {
            on_event(Event::Token(text.clone()));
        }
        Ok(self.finish(text, failure, ask_first, on_event))
    }

    /// Add the failure and question lines (if any) after a reply.
    fn finish(&self, text: String, failure: Option<String>, question: Option<String>, on_event: &mut dyn FnMut(Event)) -> String {
        let mut text = text;
        for extra in [failure, question].into_iter().flatten() {
            let line = if text.is_empty() { extra } else { format!("\n{extra}") };
            on_event(Event::Token(line.clone()));
            text.push_str(&line);
        }
        text
    }

    /// Answer, giving the model what it remembers about the user. The memory goes
    /// into the system prompt for this one answer only (a small model follows facts
    /// there, but ignores them in the user's message); the stored chat stays as typed.
    fn answer_with_memory(&mut self, request: &str, on_event: &mut dyn FnMut(Event)) -> Result<String> {
        let Some(context) = crate::memory::context_for(request) else {
            return self.answer(on_event);
        };
        let base = self.system_prompt.clone();
        self.system_prompt = format!(
            "{base}\n\n{context}\nIf the user's question relates to these, answer using them; don't say you lack personal information."
        );
        let reply = self.answer(on_event);
        self.system_prompt = base;
        reply
    }

    /// After something is saved to memory: if it is long, or no tag fit by its words,
    /// ask the model for tags (from the fixed vocabulary only) and a short summary.
    /// Never fails the request: the word-based filing stays if this doesn't work.
    fn refine_memory(&mut self) {
        let Some(m) = crate::memory::list().into_iter().max_by_key(|m| m.id) else { return };
        if m.text.chars().count() <= 90 && m.tags != [crate::memory::OTHER] {
            return;
        }
        let model = match (&mut self.router_llm, &mut self.main) {
            (_, Some(Main::Local(model))) | (Some(model), _) => model,
            _ => return,
        };
        if let Ok((tags, summary)) = tools::file_memory(model, self.template, &m.text) {
            let summary = if m.text.chars().count() <= 90 { m.summary.clone() } else { summary };
            // The model's choices first, then the word-based ones; "other" only if nothing else fits.
            let mut merged: Vec<String> = Vec::new();
            for t in tags.into_iter().chain(m.tags.iter().cloned()) {
                if t != crate::memory::OTHER && !merged.contains(&t) {
                    merged.push(t);
                }
            }
            merged.truncate(3);
            if merged.is_empty() {
                merged.push(crate::memory::OTHER.into());
            }
            let _ = crate::memory::set_meta(m.id, merged, summary);
        }
    }

    fn plan_step(&mut self, request: &str, steps: &[tools::Step]) -> Result<Vec<serde_json::Value>> {
        if cfg!(target_os = "android") {
            bail!("device control isn't available on Android yet");
        }
        let model = match (&mut self.router_llm, &mut self.main) {
            (_, Some(Main::Local(model))) | (Some(model), _) => model,
            (None, _) => bail!("device control needs a local model"),
        };
        tools::plan(model, self.template, request, steps)
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
    template: Template,
    messages: &[ChatMessage],
    on_event: &mut dyn FnMut(Event),
) -> Result<String> {
    // The rendered history only grows at the end between turns, so the
    // prompt cache makes each turn evaluate just the newest messages.
    let prompt = template.render(messages, true);
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
