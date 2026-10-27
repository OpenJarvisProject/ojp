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
use crate::link::{self, AskLevel, Client, Identity, Incoming, Msg, Peer, Peers, Presence, Request, Server};
use crate::link::net::Caps;
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

/// Everything the device link needs, cheap to clone into a background thread.
#[derive(Clone)]
pub struct LinkHandle {
    pub me: Identity,
    pub client: Client,
    pub presence: std::sync::Arc<Presence>,
    pub caps: std::sync::Arc<Caps>,
}

impl LinkHandle {
    /// Listen for paired devices; their requests arrive on `incoming` (see [`Engine::handle_incoming`]).
    pub fn start_server(&self, incoming: std::sync::mpsc::Sender<Incoming>) -> Result<Server> {
        Server::start(self.me.clone(), self.caps.clone(), incoming)
    }

    /// Like [`start_server`](Self::start_server), with state the caller keeps across engine reloads.
    pub fn start_server_with(&self, caps: std::sync::Arc<Caps>, incoming: std::sync::mpsc::Sender<Incoming>) -> Result<Server> {
        Server::start(self.me.clone(), caps, incoming)
    }

    /// Share memories with every paired device that is reachable and allows it.
    /// Returns how many memories changed here.
    pub fn sync_all(&self) -> usize {
        let mut changed = 0;
        for peer in Peers::load().peers.iter().filter(|p| p.perms.sync) {
            let mut theirs = None;
            let sent = Msg::Sync { state: crate::memory::export_sync() };
            let ok = self.client.request(peer, sent, &mut |m| {
                if let Msg::SyncReply { state } = m {
                    theirs = Some(state.clone());
                }
            });
            if ok.is_ok()
                && let Some(state) = theirs
            {
                changed += crate::memory::merge(state).unwrap_or(0);
            }
        }
        changed
    }

    /// Sync in the background (after a memory was saved or forgotten here).
    pub fn sync_soon(&self) {
        let me = self.clone();
        std::thread::spawn(move || {
            me.sync_all();
        });
    }

    /// Every paired device with what its last check found (None = not reachable).
    pub fn status(&self) -> Vec<(Peer, Option<link::net::PeerState>)> {
        Peers::load().peers.into_iter().map(|p| { let s = self.presence.state(&p); (p, s) }).collect()
    }
}

/// A request from another device that this engine is carrying out.
struct Delegation {
    level: AskLevel,
    from: String,
}

/// Something another device asked for that waits for its user's yes (kept here, on the device that
/// would do it, until the answer arrives).
struct RemotePending {
    /// Public key (hex) of the device that asked; only it can answer.
    peer: String,
    call: serde_json::Value,
    at: Instant,
}

/// How long an approval question stays open.
const APPROVAL_WINDOW: std::time::Duration = std::time::Duration::from_secs(600);

pub struct Engine {
    link: Option<LinkHandle>,
    brain: String,
    delegated: Option<Delegation>,
    /// Here, as the device being asked: an action waiting for the asking device's user to approve.
    remote_pending: Option<RemotePending>,
    /// Here, as the device that asked: the other device is waiting for our user's yes or no.
    awaiting_remote: Option<Peer>,
    /// Messages other devices asked to show, for the UI to pick up.
    inbox: Vec<(String, String)>,
    router_llm: Option<Model>,
    /// Picks the tool for a single-tool request from the router's hidden state (see tools::ToolProbe).
    tool_probe: Option<tools::ToolProbe>,
    /// The router model's hidden state for the request being handled, when it routed with a probe.
    route_hidden: Option<Vec<f32>>,
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
        crate::model::set_threads(config.models.threads);
        let mut status = Vec::new();
        // A probe router next to a main model only classifies, so it needs one short sequence.
        // (Tool calls run on the main model when it is local; with a remote main they run on the router.)
        let main_planned = config.models.main_url.is_none()
            && config.models.main_enabled
            && config.models.main.as_ref().is_some_and(|m| m.exists());
        let probe = config.models.router.as_ref().and_then(|r| crate::router::Probe::find(r, config.models.router_probe.as_deref()));
        let light_router = probe.is_some() && main_planned;
        let mut load = |role: &str, path: &Option<std::path::PathBuf>| -> Option<Model> {
            let path = path.as_ref()?;
            match Model::load(path, if role == "router" && light_router { 1024 } else { config.models.n_ctx }, config.models.gpu_layers, role == "router" || config.models.low_memory, role == "router" && probe.is_some(), role == "router" && light_router) {
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
        let router = Router::new(template).with_probe(probe);
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
            let path = cache_path(config, config.models.router.as_deref(), &[&chat_prompt, &format!("{router_only}{light_router}{}", router.has_probe()), &tool_prompt]);
            if path.as_ref().is_none_or(|p| model.load_cache(p).is_err()) {
                let _ = router.classify(model, "hello");
                // Evaluates the tool prompt once so the first tool call is fast.
                if !light_router {
                    let _ = tools::ask(model, template, "what time is it");
                }
                if model.is_recurrent() {
                    // A recurrent model can't rewind: it snapshots where two prompts diverge.
                    let _ = router.classify(model, "what time is it");
                    if !light_router {
                        let _ = tools::ask(model, template, "hello");
                    }
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
                let routes_here = !(router.has_probe() && router_llm.is_some());
                if routes_here {
                    let _ = router.classify(model, "hello");
                }
                let _ = tools::ask(model, template, "what time is it");
                if model.is_recurrent() {
                    if routes_here {
                        let _ = router.classify(model, "what time is it");
                    }
                    let _ = tools::ask(model, template, "hello");
                }
                if let Some(p) = &path {
                    let _ = model.save_cache(p);
                }
            }
        }
        let tool_probe = config.models.router.as_ref().and_then(|r| tools::ToolProbe::find(r));
        let link = if config.link.enabled { Identity::load_or_create().ok() } else { None }.map(|me| {
            let caps = std::sync::Arc::new(Caps::default());
            let has = |b: bool| b;
            caps.chat.store(has(router_llm.is_some() || main.is_some()), std::sync::atomic::Ordering::Relaxed);
            caps.main.store(matches!(main, Some(Main::Local(_) | Main::Remote { .. })), std::sync::atomic::Ordering::Relaxed);
            caps.ask.store(true, std::sync::atomic::Ordering::Relaxed);
            LinkHandle { client: Client::new(me.clone()), presence: Presence::start(me.clone(), std::time::Duration::from_secs(15)), caps, me }
        });
        Ok(Engine {
            link,
            brain: config.link.brain.clone(),
            delegated: None,
            remote_pending: None,
            awaiting_remote: None,
            inbox: Vec::new(),
            router_llm,
            tool_probe,
            route_hidden: None,
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
        // The user is answering a question another device asked on their behalf ("allow it to ...?").
        if self.delegated.is_none()
            && let Some(peer) = self.awaiting_remote.take()
        {
            match interpret(request) {
                Answer::Other => on_event(Event::Notice(format!("dropped the question from {}", peer.name))),
                answer => {
                    let reply = self.answer_remote(&peer, answer == Answer::Yes, on_event);
                    self.history.push(ChatMessage::new(Role::User, request));
                    self.history.push(ChatMessage::new(Role::Assistant, reply.clone()));
                    return Ok(reply);
                }
            }
        }
        // The user is answering "shall I go ahead?"
        if let Some(pending) = self.pending.take().filter(|_| self.delegated.is_none()) {
            match interpret(request) {
                Answer::Yes => {
                    let text = match tools::run_confirmed(&pending.call) {
                        Ok(o) => o.text,
                        Err(e) => format!("Couldn't do that: {e:#}"),
                    };
                    if pending.call["tool"] == "forget" {
                        self.memory_changed();
                    }
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
        // (Not for requests that came from another device: they get no typed-command powers.)
        if self.delegated.is_none()
            && let Some(parsed) = tools::command(request)
        {
            let reply = self.run_command(parsed, on_event);
            self.history.push(ChatMessage::new(Role::User, request));
            self.history.push(ChatMessage::new(Role::Assistant, reply.clone()));
            return Ok(reply);
        }
        // "remind me on my phone ...": that device's own assistant does it.
        if self.delegated.is_none()
            && let Some(link) = self.link.clone()
        {
            let peers = Peers::load().peers;
            if let Some((peer, rest)) = link::addressed(request, &peers, &link.me) {
                let (reply, question) = delegate(&link.client, peer, &rest, on_event);
                if question {
                    self.awaiting_remote = Some(peer.clone());
                }
                self.history.push(ChatMessage::new(Role::User, request));
                self.history.push(ChatMessage::new(Role::Assistant, reply.clone()));
                return Ok(reply);
            }
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
        // A probe-equipped router is the most accurate and the fastest classifier
        // (98% on held-out requests). Without one, the in-process main model, when
        // loaded, is far more accurate than the small router answering for itself
        // (measured with `ojp-cli route-eval` and `tool-eval`).
        let classifier = match (&mut self.router_llm, &mut self.main) {
            (Some(model), _) if self.router.has_probe() => model,
            (_, Some(Main::Local(model))) | (Some(model), _) => model,
            (None, _) => return Route::AnswerDirectly,
        };
        self.route_hidden = None;
        let route = match self.router.classify(classifier, request) {
            Ok(decision) => {
                self.route_hidden = decision.hidden;
                decision.route
            }
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
                if run.outcomes.iter().any(|o| matches!(o.call["tool"].as_str(), Some("remember" | "forget"))) {
                    self.memory_changed();
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
            // A small model sometimes repeats a call it already made instead of saying it is done;
            // running it again would only waste rounds (and time), so a round of repeats ends the task.
            let earlier: Vec<&serde_json::Value> = steps.iter().flat_map(|s| s.calls.iter()).collect();
            let fresh: Vec<&serde_json::Value> = calls.iter().filter(|c| !tools::is_done(c) && !earlier.contains(c)).collect();
            if fresh.is_empty() && !steps.is_empty() && std::env::var("OJP_LOOP_GUARD").map_or(true, |v| v != "0") {
                break;
            }
            let mut results = Vec::new();
            let mut explored = false;
            for call in fresh {
                // For another device: only what that device was allowed to ask for. Anything that needs a
                // yes is put to the user of the device that asked (see `ask_for`), never done unasked.
                if let Some(d) = &self.delegated
                    && !link::remote_allowed(call["tool"].as_str().unwrap_or(""), d.level)
                {
                    let here = self.link.as_ref().map_or("this device", |l| l.me.name.as_str());
                    failure = Some(format!("{here} doesn't let {} do that remotely; do it on {here} itself.", d.from));
                    break 'steps;
                }
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
                            let place = match (&self.delegated, &self.link) {
                                (Some(_), Some(l)) => format!("On {}, ", l.me.name),
                                _ => String::new(),
                            };
                            ask_first = Some(format!("{place}I'd like to {what}. Shall I go ahead? (yes / no)"));
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
                            self.memory_changed();
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
                format!(
                    "{request}\n\n{done}[Looked up on this computer: {}]\nAnswer in one or two short sentences using only that.{}",
                    facts.join("\n"),
                    // Qwen3.5-based models paraphrased and invented numbers; this keeps them to the text.
                    if self.template.qwen35_tuned() { " Copy names and numbers exactly as written and add nothing." } else { "" }
                ),
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
        let model = match (&mut self.router_llm, &mut self.main) {
            (_, Some(Main::Local(model))) | (Some(model), _) => model,
            (None, _) => bail!("device control needs a local model"),
        };
        if steps.is_empty() {
            return tools::ask_fast(model, self.template, self.tool_probe.as_ref(), self.route_hidden.as_deref(), request);
        }
        tools::plan(model, self.template, request, steps)
    }

    fn messages(&self) -> Vec<ChatMessage> {
        let recent = self.history.len().saturating_sub(HISTORY_MESSAGES);
        let mut messages = vec![ChatMessage::new(Role::System, self.system_prompt.clone())];
        messages.extend(self.history[recent..].iter().cloned());
        messages
    }

    /// A paired device whose model should answer instead of this one's: when this device has
    /// no main model (a phone with only the small router), or the user chose "remote".
    fn chat_peer(&self) -> Option<Peer> {
        let link = self.link.as_ref()?;
        if self.delegated.is_some() || self.brain == "local" {
            return None;
        }
        let have_main = matches!(self.main, Some(Main::Local(_) | Main::Remote { .. }));
        if self.brain != "remote" && have_main {
            return None;
        }
        let mut best: Option<(bool, Peer)> = None;
        for (peer, state) in link.status() {
            let Some(state) = state.filter(|s| s.chat) else { continue };
            // Automatic mode only borrows a bigger model, not another small router.
            if self.brain != "remote" && !state.main {
                continue;
            }
            if best.as_ref().is_none_or(|(main, _)| state.main && !main) {
                best = Some((state.main, peer));
            }
        }
        best.map(|(_, p)| p)
    }

    /// Something that other devices should know changed (a memory was saved or forgotten).
    fn memory_changed(&self) {
        if self.delegated.is_none()
            && let Some(link) = &self.link
        {
            link.sync_soon();
        }
    }

    /// Report what this engine can do into `caps` (kept by the app across reloads) and use it from now on.
    pub fn share_caps(&mut self, caps: &std::sync::Arc<Caps>) {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(link) = self.link.as_mut() else { return };
        caps.chat.store(self.router_llm.is_some() || self.main.is_some(), Relaxed);
        caps.main.store(matches!(self.main, Some(Main::Local(_) | Main::Remote { .. })), Relaxed);
        caps.ask.store(true, Relaxed);
        link.caps = caps.clone();
    }

    pub fn link(&self) -> Option<&LinkHandle> {
        self.link.as_ref()
    }

    /// Messages other devices sent to show the user, oldest first (title, text).
    pub fn take_inbox(&mut self) -> Vec<(String, String)> {
        std::mem::take(&mut self.inbox)
    }

    /// Carry out a request from a paired device. Run this on the thread that owns the engine.
    pub fn handle_incoming(&mut self, incoming: Incoming) {
        let Incoming { peer, request, reply } = incoming;
        match request {
            Request::Chat(messages) => {
                let messages: Vec<ChatMessage> =
                    messages.into_iter().filter_map(|(role, text)| role.parse::<Role>().ok().map(|r| ChatMessage::new(r, text))).collect();
                let template = self.template;
                let mut on = |event: Event| {
                    if let Event::Token(t) = event {
                        reply.tok(&t);
                    }
                };
                let result = match &mut self.main {
                    Some(Main::Local(model)) => generate_local(model, template, &messages, &mut on),
                    Some(Main::Remote { base_url, model }) => generate_remote(base_url, model, &messages, &mut on),
                    None => match &mut self.router_llm {
                        Some(model) => generate_local(model, template, &messages, &mut on),
                        None => Err(anyhow::anyhow!("no model is loaded")),
                    },
                };
                match result {
                    Ok(text) => reply.end(&text),
                    Err(e) => reply.err(&format!("{e:#}")),
                }
            }
            Request::Ask(text) => {
                let mut on = |event: Event| match event {
                    Event::Token(t) => reply.tok(&t),
                    Event::Notice(n) => reply.note(&n),
                    Event::Routed { .. } => {}
                };
                match self.ask_for(&peer, &text, &mut on) {
                    Ok((text, true)) => reply.question(&text),
                    Ok((text, false)) => reply.end(&text),
                    Err(e) => reply.err(&format!("{e:#}")),
                }
            }
            Request::Answer(yes) => {
                let waiting = self.remote_pending.take().filter(|p| p.peer == peer.public && p.at.elapsed() < APPROVAL_WINDOW);
                match waiting {
                    None => reply.err("nothing is waiting for approval here (it may have timed out)"),
                    Some(_) if !yes => reply.end("Okay, I won't do that."),
                    // Re-checked: the permission may have been lowered since the question was asked.
                    Some(_) if peer.perms.ask != AskLevel::Everything => reply.err("that device isn't allowed to do that any more"),
                    Some(p) => match tools::run_confirmed(&p.call) {
                        Ok(o) => {
                            if tools::writes_memory(&o.call) || o.call["tool"] == "forget" {
                                self.memory_changed();
                            }
                            reply.end(&o.text);
                        }
                        Err(e) => reply.end(&format!("Couldn't do that: {e:#}")),
                    },
                }
            }
            Request::Notify { title, text } => {
                show_notification(&format!("{} ({})", title, peer.name), &text);
                self.inbox.push((format!("{} ({})", title, peer.name), text));
                reply.ok();
            }
            Request::Sync(state) => {
                let _ = crate::memory::merge(state);
                reply.sync(crate::memory::export_sync());
            }
        }
    }

    /// Do what another device asked, within what it is allowed to ask. The conversation here
    /// is left as it was.
    fn ask_for(&mut self, peer: &Peer, request: &str, on_event: &mut dyn FnMut(Event)) -> Result<(String, bool)> {
        let history = std::mem::take(&mut self.history);
        let pending = self.pending.take();
        self.remote_pending = None;
        self.delegated = Some(Delegation { level: peer.perms.ask, from: peer.name.clone() });
        let result = self.ask(request, on_event);
        self.delegated = None;
        // An action that needs a yes is kept for the asking device's user to approve, not this one's.
        let waiting = self.pending.take();
        self.history = history;
        self.pending = pending;
        let question = waiting.is_some();
        if let Some(p) = waiting {
            self.remote_pending = Some(RemotePending { peer: peer.public.clone(), call: p.call, at: Instant::now() });
        }
        result.map(|text| (text, question))
    }

    /// Relay our user's yes or no for a question another device asked, and show what it says.
    fn answer_remote(&mut self, peer: &Peer, yes: bool, on_event: &mut dyn FnMut(Event)) -> String {
        let Some(client) = self.link.as_ref().map(|l| l.client.clone()) else { return "The device link is off.".into() };
        let text = match client.request(peer, Msg::Answer { yes }, &mut |_| {}) {
            Ok(text) => text,
            Err(e) => format!("Couldn't tell {}: {e:#}", peer.name),
        };
        on_event(Event::Token(text.clone()));
        text
    }

    fn answer(&mut self, on_event: &mut dyn FnMut(Event)) -> Result<String> {
        let messages = self.messages();
        if let Some(peer) = self.chat_peer() {
            let client = self.link.as_ref().map(|l| l.client.clone()).expect("chat_peer needs the link");
            match generate_peer(&client, &peer, &messages, on_event) {
                Ok(text) if !text.is_empty() => return Ok(text),
                Ok(_) => {}
                Err(e) => on_event(Event::Notice(format!("{} couldn't answer ({e:#}), using this device's model", peer.name))),
            }
        }
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

/// Have a paired device's model write the answer, passing its words on as they arrive.
fn generate_peer(client: &Client, peer: &Peer, messages: &[ChatMessage], on_event: &mut dyn FnMut(Event)) -> Result<String> {
    let wire: Vec<(String, String)> = messages.iter().map(|m| (m.role.as_str().to_string(), m.content.clone())).collect();
    on_event(Event::Notice(format!("answering with {}'s model", peer.name)));
    let mut streamed = false;
    let text = client.request(peer, Msg::Chat { messages: wire }, &mut |m| {
        if let Msg::Tok { text } = m {
            streamed = true;
            on_event(Event::Token(text.clone()));
        }
    })?;
    if !streamed && !text.is_empty() {
        on_event(Event::Token(text.clone()));
    }
    Ok(text.trim().to_string())
}

/// Send a request to another device's assistant and show what it says. If it can't be
/// reached the user is told so; the request is not run here instead, since it was meant for there.
fn delegate(client: &Client, peer: &Peer, request: &str, on_event: &mut dyn FnMut(Event)) -> (String, bool) {
    on_event(Event::Notice(format!("asking {}", peer.name)));
    let mut streamed = false;
    let result = client.request_ex(peer, Msg::Ask { text: request.to_string() }, &mut |m| match m {
        Msg::Tok { text } => {
            streamed = true;
            on_event(Event::Token(text.clone()));
        }
        Msg::Note { text } => on_event(Event::Notice(format!("{}: {text}", peer.name))),
        _ => {}
    });
    match result {
        Ok((text, question)) => {
            if !streamed && !text.is_empty() {
                on_event(Event::Token(text.clone()));
            }
            (text, question)
        }
        Err(e) => {
            let text = format!("I couldn't do that on {}: {e:#}", peer.name);
            on_event(Event::Token(text.clone()));
            (text, false)
        }
    }
}

/// Show a message from another device as a desktop notification (best effort).
fn show_notification(title: &str, text: &str) {
    if cfg!(target_os = "linux") && !cfg!(target_os = "android") {
        let _ = std::process::Command::new("notify-send").args(["--app-name=OpenJarvis", title, text]).spawn();
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
