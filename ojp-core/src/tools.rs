//! Tools the assistant can use on this computer. The model never writes free
//! text here: a GBNF grammar built from the registry below forces its output
//! into one of the call shapes, so the JSON always parses and only these
//! tools and argument types can appear. Tools act through standard desktop
//! commands (wpctl, playerctl, xdg-open, notify-send, ...), never through a
//! shell, and every text argument is validated again before use.
//!
//! Two kinds: *Read* tools look something up and their result goes back to
//! the model to phrase the answer; *Act* tools do something and answer with
//! their own short confirmation.
//!
//! OJP_TOOLS_DRY_RUN=1 prints each action instead of running it (read-only
//! lookups still run).
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use llama_crab::chat::BuiltinTemplate;
use llama_crab::chat::template::render_builtin;
use llama_crab::{ChatMessage, LlamaSampler, Role};
use serde_json::Value;

use crate::model::{Model, Seq};
use crate::tools_ext as ext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Looks something up; the result is phrased into an answer by the model.
    Read,
    /// Does something; the tool's own confirmation is the answer.
    Act,
}

/// Whether a tool may run on the model's say-so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Risk {
    /// Fine to run directly.
    Safe,
    /// Changes something that matters: the user is asked first.
    Confirm,
}

enum ArgKind {
    Int { max: u32 },
    Text { max: u32 },
    Choice(&'static [&'static str]),
}

struct Arg {
    name: &'static str,
    kind: ArgKind,
}

struct Tool {
    name: &'static str,
    doc: &'static str,
    args: &'static [Arg],
    kind: Kind,
    risk: Risk,
    /// Explores (lists, searches, reads): what it finds may call for another step.
    explore: bool,
    run: fn(&Value) -> Result<String>,
}

const REGISTRY: &[Tool] = &[
    Tool { name: "set_timer", doc: "start a countdown timer that rings after some minutes", args: &[Arg { name: "minutes", kind: ArgKind::Int { max: 999 } }], kind: Kind::Act, risk: Risk::Safe, explore: false, run: |a| set_timer(int(a, "minutes")?) },
    Tool { name: "remind", doc: "remind the user about something after some minutes", args: &[Arg { name: "minutes", kind: ArgKind::Int { max: 999 } }, Arg { name: "text", kind: ArgKind::Text { max: 100 } }], kind: Kind::Act, risk: Risk::Safe, explore: false, run: |a| remind(int(a, "minutes")?, text(a, "text")?) },
    Tool { name: "volume", doc: "set the speaker volume, 0-100", args: &[Arg { name: "level", kind: ArgKind::Int { max: 100 } }], kind: Kind::Act, risk: Risk::Safe, explore: false, run: |a| volume(int(a, "level")?) },
    Tool { name: "mute", doc: "turn the speaker sound off or on (muting)", args: &[Arg { name: "state", kind: ArgKind::Choice(&["on", "off"]) }], kind: Kind::Act, risk: Risk::Safe, explore: false, run: |a| mute(text(a, "state")?) },
    Tool { name: "media", doc: "play, pause or skip the music or video player (not the volume)", args: &[Arg { name: "action", kind: ArgKind::Choice(&["play_pause", "next", "previous"]) }], kind: Kind::Act, risk: Risk::Safe, explore: false, run: |a| media(text(a, "action")?) },
    Tool { name: "open_app", doc: "launch an installed application by name", args: &[Arg { name: "name", kind: ArgKind::Text { max: 40 } }], kind: Kind::Act, risk: Risk::Safe, explore: false, run: |a| open_app(text(a, "name")?) },
    Tool { name: "open_url", doc: "show the user a web page in their browser (to read a page yourself, use web_fetch)", args: &[Arg { name: "url", kind: ArgKind::Text { max: 200 } }], kind: Kind::Act, risk: Risk::Safe, explore: false, run: |a| open_url(text(a, "url")?) },
    Tool { name: "note_add", doc: "save a short note", args: &[Arg { name: "text", kind: ArgKind::Text { max: 200 } }], kind: Kind::Act, risk: Risk::Safe, explore: false, run: |a| note_add(text(a, "text")?) },
    Tool { name: "calculate", doc: "exact arithmetic, e.g. 7^3+4*7*2-3 (use * / ^ sqrt() sin() ln() pi)", args: &[Arg { name: "expression", kind: ArgKind::Text { max: 120 } }], kind: Kind::Read, risk: Risk::Safe, explore: false, run: |a| calculate(text(a, "expression")?) },
    Tool { name: "get_time", doc: "the current time, date or day of the week", args: &[], kind: Kind::Read, risk: Risk::Safe, explore: false, run: |_| get_time() },
    Tool { name: "system_status", doc: "battery, memory, disk space, load and uptime", args: &[], kind: Kind::Read, risk: Risk::Safe, explore: false, run: |_| system_status() },
    Tool { name: "now_playing", doc: "which song or video is playing (not the time)", args: &[], kind: Kind::Read, risk: Risk::Safe, explore: false, run: |_| now_playing() },
    Tool { name: "clipboard", doc: "read the text on the clipboard", args: &[], kind: Kind::Read, risk: Risk::Safe, explore: false, run: |_| clipboard() },
    Tool { name: "find_files", doc: "find files by part of their NAME (not what is inside them)", args: &[Arg { name: "query", kind: ArgKind::Text { max: 60 } }], kind: Kind::Read, risk: Risk::Safe, explore: true, run: |a| find_files(text(a, "query")?) },
    Tool { name: "reminders_list", doc: "show the pending timers and reminders", args: &[], kind: Kind::Read, risk: Risk::Safe, explore: false, run: |_| reminders_list() },
    Tool { name: "list_dir", doc: "list the files in a folder (~ is the home folder)", args: &[Arg { name: "path", kind: ArgKind::Text { max: 150 } }], kind: Kind::Read, risk: Risk::Safe, explore: true, run: |a| ext::list_dir(text(a, "path")?) },
    Tool { name: "search_text", doc: "search INSIDE the user's files for words, when they mention their files, notes or documents (money, plans, passwords, dates); for what the user told you, use recall", args: &[Arg { name: "query", kind: ArgKind::Text { max: 80 } }], kind: Kind::Read, risk: Risk::Safe, explore: true, run: |a| ext::search_text(text(a, "query")?) },
    Tool { name: "read_file", doc: "read a text file", args: &[Arg { name: "path", kind: ArgKind::Text { max: 150 } }], kind: Kind::Read, risk: Risk::Safe, explore: true, run: |a| ext::read_file(text(a, "path")?) },
    Tool { name: "write_file", doc: "create or replace a text file (asks the user first)", args: &[Arg { name: "path", kind: ArgKind::Text { max: 150 } }, Arg { name: "text", kind: ArgKind::Text { max: 1500 } }], kind: Kind::Act, risk: Risk::Confirm, explore: false, run: |a| ext::write_file(text(a, "path")?, text(a, "text")?) },
    Tool { name: "run_command", doc: "run a terminal command (asks the user first)", args: &[Arg { name: "command", kind: ArgKind::Text { max: 250 } }], kind: Kind::Act, risk: Risk::Confirm, explore: false, run: |a| ext::run_command(text(a, "command")?) },
    Tool { name: "web_search", doc: "search the web (needs internet access to be on)", args: &[Arg { name: "query", kind: ArgKind::Text { max: 120 } }], kind: Kind::Read, risk: Risk::Safe, explore: true, run: |a| ext::web_search(text(a, "query")?) },
    Tool { name: "web_fetch", doc: "read the text of a web page so you can tell the user what it says (needs internet access)", args: &[Arg { name: "url", kind: ArgKind::Text { max: 300 } }], kind: Kind::Read, risk: Risk::Safe, explore: true, run: |a| ext::web_fetch(text(a, "url")?) },
    Tool { name: "remember", doc: "save a lasting fact about the user so it is remembered in later chats", args: &[Arg { name: "text", kind: ArgKind::Text { max: 250 } }], kind: Kind::Act, risk: Risk::Safe, explore: false, run: |a| remember(text(a, "text")?) },
    Tool { name: "recall", doc: "look up what is remembered about the user on a topic", args: &[Arg { name: "query", kind: ArgKind::Text { max: 80 } }], kind: Kind::Read, risk: Risk::Safe, explore: false, run: |a| recall(text(a, "query")?) },
    Tool { name: "memories_list", doc: "show everything that is remembered about the user", args: &[], kind: Kind::Read, risk: Risk::Safe, explore: false, run: |_| memories_list() },
    Tool { name: "tags_list", doc: "show the tags that memories are filed under", args: &[], kind: Kind::Read, risk: Risk::Safe, explore: false, run: |_| tags_list() },
    Tool { name: "tag_add", doc: "add a memory tag, or more words for one; words are separated by commas", args: &[Arg { name: "name", kind: ArgKind::Text { max: 24 } }, Arg { name: "words", kind: ArgKind::Text { max: 120 } }], kind: Kind::Act, risk: Risk::Safe, explore: false, run: |a| tag_add(text(a, "name")?, text(a, "words")?) },
    Tool { name: "forget", doc: "remove something that is remembered about the user (asks first)", args: &[Arg { name: "query", kind: ArgKind::Text { max: 80 } }], kind: Kind::Act, risk: Risk::Confirm, explore: false, run: |a| forget(text(a, "query")?) },
    Tool { name: "notes_list", doc: "show the saved notes", args: &[], kind: Kind::Read, risk: Risk::Safe, explore: false, run: |_| notes_list() },
];

/// The tool-use instructions given to the model, generated from the registry.
pub fn system_prompt() -> String {
    system_prompt_for(false)
}

fn system_prompt_for(with_done: bool) -> String {
    let mut s = String::from("You control this computer through tools. Available tools:\n");
    for t in REGISTRY {
        let args: Vec<String> = t
            .args
            .iter()
            .map(|a| match &a.kind {
                ArgKind::Int { max } => format!("{}: int 0-{max}", a.name),
                ArgKind::Text { .. } => format!("{}: str", a.name),
                ArgKind::Choice(c) => format!("{}: {}", a.name, c.join("|")),
            })
            .collect();
        s.push_str(&format!("- {}({})  // {}\n", t.name, args.join(", "), t.doc));
    }
    if with_done {
        s.push_str("- done()  // use this when nothing more is needed\n");
    }
    s.push_str(
        "Respond with ONLY a JSON array of 1 to 3 tool calls, in the order to do them, like \
         [{\"tool\": \"<name>\", \"args\": {...}}], and nothing else. Use more than one call only when the \
         request asks for more than one thing.",
    );
    s
}

/// The GBNF grammar accepting exactly the registry's call shapes.
fn grammar(with_done: bool) -> String {
    let mut alternatives: Vec<String> = REGISTRY
        .iter()
        .map(|t| {
            let args: Vec<String> = t
                .args
                .iter()
                .map(|a| {
                    let value = match &a.kind {
                        ArgKind::Int { .. } => "int".to_string(),
                        ArgKind::Text { max } => format!("\"\\\"\" [^\"\\\\\\n]{{1,{max}}} \"\\\"\""),
                        ArgKind::Choice(c) => {
                            let options: Vec<String> = c.iter().map(|o| format!("\"{o}\"")).collect();
                            format!("\"\\\"\" ( {} ) \"\\\"\"", options.join(" | "))
                        }
                    };
                    format!("\"\\\"{}\\\": \" {value}", a.name)
                })
                .collect();
            let body = if args.is_empty() { String::new() } else { args.join(" \", \" ") };
            format!("\"{}\\\", \\\"args\\\": {{\" {}", t.name, body)
        })
        .collect();
    if with_done {
        alternatives.push("\"done\\\", \\\"args\\\": {\"".to_string());
    }
    format!(
        "root ::= \"[\" one ( \", \" one ){{0,2}} \"]\"\none ::= \"{{\\\"tool\\\": \\\"\" call \"}}}}\"\ncall ::= ( {} )\nint ::= [0-9] | [1-9] [0-9] | [1-9] [0-9] [0-9]\n",
        alternatives.join("\n         | ")
    )
}

/// Worked examples shown to the model as earlier turns, one or two per tool.
/// Kept apart from the phrasings in `ojp-cli tool-eval`, which only reports.
const EXAMPLES: &[(&str, &str)] = &[
    ("Set a timer for 20 minutes.", r#"[{"tool": "set_timer", "args": {"minutes": 20}}]"#),
    ("Turn the sound up to 65.", r#"[{"tool": "volume", "args": {"level": 65}}]"#),
    ("Silence the speakers.", r#"[{"tool": "mute", "args": {"state": "on"}}]"#),
    ("Bring the sound back.", r#"[{"tool": "mute", "args": {"state": "off"}}]"#),
    ("Play the next song.", r#"[{"tool": "media", "args": {"action": "next"}}]"#),
    ("Hit pause on the video.", r#"[{"tool": "media", "args": {"action": "play_pause"}}]"#),
    ("Open the file manager.", r#"[{"tool": "open_app", "args": {"name": "file manager"}}]"#),
    ("Take me to wikipedia.org.", r#"[{"tool": "open_url", "args": {"url": "https://wikipedia.org"}}]"#),
    ("Note to self: call mom on Sunday.", r#"[{"tool": "note_add", "args": {"text": "call mom on Sunday"}}]"#),
    ("What is 45 plus 38?", r#"[{"tool": "calculate", "args": {"expression": "45+38"}}]"#),
    ("What's the current time?", r#"[{"tool": "get_time", "args": {}}]"#),
    ("Is my laptop charged?", r#"[{"tool": "system_status", "args": {}}]"#),
    ("Which track is on?", r#"[{"tool": "now_playing", "args": {}}]"#),
    ("What did I copy?", r#"[{"tool": "clipboard", "args": {}}]"#),
    ("Locate my vacation photos.", r#"[{"tool": "find_files", "args": {"query": "vacation"}}]"#),
    ("Remind me to stretch in 30 minutes.", r#"[{"tool": "remind", "args": {"minutes": 30, "text": "stretch"}}]"#),
    ("What reminders do I have?", r#"[{"tool": "reminders_list", "args": {}}]"#),
    ("Open the browser and set a timer for 5 minutes.", r#"[{"tool": "open_app", "args": {"name": "browser"}}, {"tool": "set_timer", "args": {"minutes": 5}}]"#),
    ("Mute the sound and tell me the time.", r#"[{"tool": "mute", "args": {"state": "on"}}, {"tool": "get_time", "args": {}}]"#),
    ("What files are in my Documents folder?", r#"[{"tool": "list_dir", "args": {"path": "~/Documents"}}]"#),
    ("Show me what's inside notes.txt in my Documents.", r#"[{"tool": "read_file", "args": {"path": "~/Documents/notes.txt"}}]"#),
    ("Save the text hello into hi.txt in my Documents folder.", r#"[{"tool": "write_file", "args": {"path": "~/Documents/hi.txt", "text": "hello"}}]"#),
    ("Run ls in my Downloads folder.", r#"[{"tool": "run_command", "args": {"command": "ls ~/Downloads"}}]"#),
    ("Check my files and tell me what my wifi password is.", r#"[{"tool": "search_text", "args": {"query": "wifi password"}}]"#),
    ("Which of my files mentions the passport number?", r#"[{"tool": "search_text", "args": {"query": "passport number"}}]"#),
    ("Search the web for rust tutorials.", r#"[{"tool": "web_search", "args": {"query": "rust tutorials"}}]"#),
    ("Read the page at https://example.org.", r#"[{"tool": "web_fetch", "args": {"url": "https://example.org"}}]"#),
    ("Remember that my sister is called Mia.", r#"[{"tool": "remember", "args": {"text": "my sister is called Mia"}}]"#),
    ("Look up what I told you about my allergies.", r#"[{"tool": "recall", "args": {"query": "allergies"}}]"#),
    ("What do I like to drink?", r#"[{"tool": "recall", "args": {"query": "drink"}}]"#),
    ("What is my dog called?", r#"[{"tool": "recall", "args": {"query": "dog"}}]"#),
    ("What do you remember about me?", r#"[{"tool": "memories_list", "args": {}}]"#),
    ("Which tags do you use for my memories?", r#"[{"tool": "tags_list", "args": {}}]"#),
    ("Add a tag called gardening for plants and flowers.", r#"[{"tool": "tag_add", "args": {"name": "gardening", "words": "plants, flowers"}}]"#),
    ("Forget that I work on Tuesdays.", r#"[{"tool": "forget", "args": {"query": "work on Tuesdays"}}]"#),
    ("Show me what I wrote down.", r#"[{"tool": "notes_list", "args": {}}]"#),
];

/// What a tool call did.
pub struct Outcome {
    pub tool: &'static str,
    pub kind: Kind,
    /// The parsed call, for display and tests.
    pub call: Value,
    /// The tool's output: a confirmation (Act) or the looked-up facts (Read).
    pub text: String,
}

/// The result of running the calls for one request.
pub struct Run {
    /// Calls that ran, in order.
    pub outcomes: Vec<Outcome>,
    /// The first call that failed, after which nothing more was run.
    pub error: Option<anyhow::Error>,
}

/// One round of an agent task: the calls the model made and what came back.
pub struct Step {
    pub calls: Vec<Value>,
    pub results: Vec<String>,
}

/// Ask the model for the first one to three tool calls (grammar-constrained).
pub fn ask(model: &mut Model, template: BuiltinTemplate, request: &str) -> Result<Vec<Value>> {
    plan(model, template, request, &[])
}

/// Ask the model what to do next, given what earlier steps found. After the
/// first step it may also answer `done`.
pub fn plan(model: &mut Model, template: BuiltinTemplate, request: &str, steps: &[Step]) -> Result<Vec<Value>> {
    let with_done = !steps.is_empty();
    let mut messages = vec![ChatMessage::new(Role::System, system_prompt_for(with_done))];
    for (user, calls) in EXAMPLES {
        messages.push(ChatMessage::new(Role::User, *user));
        messages.push(ChatMessage::new(Role::Assistant, *calls));
    }
    // One worked example of a task that takes several rounds.
    const NEXT: &str = "If the request needs another step, give the next call; otherwise answer [{\"tool\": \"done\", \"args\": {}}].";
    for (role, text) in [
        (Role::User, "Find my groceries list and read it to me.".to_string()),
        (Role::Assistant, r#"[{"tool": "find_files", "args": {"query": "groceries"}}]"#.to_string()),
        (Role::User, format!("Results:\n1 match(es):\n~/Documents/groceries.txt\n{NEXT}")),
        (Role::Assistant, r#"[{"tool": "read_file", "args": {"path": "~/Documents/groceries.txt"}}]"#.to_string()),
        (Role::User, format!("Results:\n~/Documents/groceries.txt:\nmilk, eggs, bread\n{NEXT}")),
        (Role::Assistant, r#"[{"tool": "done", "args": {}}]"#.to_string()),
    ] {
        messages.push(ChatMessage::new(role, text));
    }
    messages.push(ChatMessage::new(Role::User, request));
    for step in steps {
        messages.push(ChatMessage::new(Role::Assistant, Value::Array(step.calls.clone()).to_string()));
        let results: Vec<String> = step.results.iter().map(|r| clip_result(r)).collect();
        messages.push(ChatMessage::new(
            Role::User,
            format!("Results:\n{}\n{NEXT}", results.join("\n")),
        ));
    }
    let prompt = render_builtin(template, &messages, &[], true);
    // SAFETY: llama_sampler_init_grammar parses the grammar during the call.
    let grammar = unsafe { LlamaSampler::grammar(model.llama().model(), &grammar(with_done), "root") }.context("tool grammar rejected")?;
    let greedy = LlamaSampler::greedy().context("greedy sampler")?;
    let mut sampler = LlamaSampler::chain(vec![grammar, greedy], true).context("sampler chain")?;
    let text = model.complete(Seq::Tool, &prompt, &mut sampler, 200, &mut |_| {})?;
    match serde_json::from_str::<Value>(text.trim()) {
        Ok(Value::Array(calls)) if !calls.is_empty() => Ok(calls),
        _ => bail!("tool calls didn't parse: {text:?}"),
    }
}

/// Ask the model to file a memory: up to three tags from the vocabulary and a
/// summary of at most 90 characters (grammar-constrained, so it can only pick
/// existing tags and never invents new ones).
pub fn file_memory(model: &mut Model, template: BuiltinTemplate, text: &str) -> Result<(Vec<String>, String)> {
    let mut tags: Vec<String> = crate::memory::vocabulary().into_iter().map(|(t, _)| t).collect();
    tags.push(crate::memory::OTHER.into());
    let options: Vec<String> = tags.iter().map(|t| format!("\"\\\"{t}\\\"\"")).collect();
    let grammar = format!(
        "root ::= \"{{\\\"tags\\\": [\" tag ( \", \" tag ){{0,2}} \"], \\\"summary\\\": \\\"\" [^\"\\\\\\n]{{1,90}} \"\\\"}}\"\ntag ::= {}\n",
        options.join(" | ")
    );
    let messages = [
        ChatMessage::new(
            Role::System,
            format!(
                "You file notes about a user. Pick the 1 to 3 tags that fit best from: {}. Then write a summary of at most 90 characters that keeps the key facts (names, dates, numbers). Reply with JSON only.",
                tags.join(", ")
            ),
        ),
        ChatMessage::new(Role::User, text),
    ];
    let prompt = render_builtin(template, &messages, &[], true);
    // SAFETY: llama_sampler_init_grammar parses the grammar during the call.
    let g = unsafe { LlamaSampler::grammar(model.llama().model(), &grammar, "root") }.context("filing grammar rejected")?;
    let greedy = LlamaSampler::greedy().context("greedy sampler")?;
    let mut sampler = LlamaSampler::chain(vec![g, greedy], true).context("sampler chain")?;
    let out = model.complete(Seq::Route, &prompt, &mut sampler, 120, &mut |_| {})?;
    let v: Value = serde_json::from_str(out.trim()).with_context(|| format!("filing didn't parse: {out:?}"))?;
    let picked: Vec<String> = v["tags"].as_array().map(|a| a.iter().filter_map(|t| t.as_str()).map(String::from).collect()).unwrap_or_default();
    let mut unique: Vec<String> = Vec::new();
    for t in picked {
        if tags.contains(&t) && !unique.contains(&t) {
            unique.push(t);
        }
    }
    let summary = v["summary"].as_str().unwrap_or("").trim().to_string();
    if unique.is_empty() || summary.is_empty() {
        bail!("the filing was empty");
    }
    Ok((unique, summary))
}

/// A result trimmed to what fits in the next prompt.
fn clip_result(text: &str) -> String {
    let t: String = text.chars().take(1200).collect();
    if text.chars().count() > 1200 { format!("{t}…") } else { t }
}

/// Is this the "nothing more to do" call?
pub fn is_done(call: &Value) -> bool {
    call["tool"] == "done"
}

fn find(call: &Value) -> Option<&'static Tool> {
    let name = call["tool"].as_str()?;
    REGISTRY.iter().find(|t| t.name == name)
}

/// Does this call bring in text that someone else wrote (a web page, a file, the
/// clipboard, command output)? Such text can contain instructions meant to trick the assistant.
pub fn reads_untrusted(call: &Value) -> bool {
    matches!(call["tool"].as_str(), Some("web_search" | "web_fetch" | "read_file" | "search_text" | "clipboard" | "run_command" | "find_files"))
}

/// Is this a call that saves something to long-term memory?
pub fn writes_memory(call: &Value) -> bool {
    call["tool"] == "remember"
}

/// Does this call explore (list, search, read), so its result may call for another step?
pub fn is_explore(call: &Value) -> bool {
    find(call).is_some_and(|t| t.explore)
}

/// If this call changes something that matters, a plain-words description of
/// what it would do, to show the user before asking. `Err` if it can't be done at all.
pub fn needs_confirmation(call: &Value) -> Option<Result<String>> {
    let tool = find(call)?;
    if tool.risk != Risk::Confirm {
        return None;
    }
    let args = &call["args"];
    Some(match tool.name {
        "write_file" => ext::write_plan(args["path"].as_str().unwrap_or(""), args["text"].as_str().unwrap_or("")),
        "run_command" => Ok(format!("run this command in your home folder: {}", args["command"].as_str().unwrap_or(""))),
        "forget" => crate::memory::forget_matches(args["query"].as_str().unwrap_or("")).and_then(|m| {
            if m.len() > 5 {
                anyhow::bail!("that matches {} memories; be more specific", m.len());
            }
            Ok(format!("forget: {}", m.iter().map(|x| format!("\"{}\"", x.text)).collect::<Vec<_>>().join(", ")))
        }),
        other => Ok(format!("use {other}")),
    })
}

/// Run calls one after another; stops at the first that fails. Calls that
/// need the user's OK are refused here (see [`run_confirmed`]).
pub fn run_calls(calls: &[Value]) -> Run {
    let mut outcomes = Vec::new();
    for call in calls {
        match run_call(call) {
            Ok(o) => outcomes.push(o),
            Err(e) => return Run { outcomes, error: Some(e) },
        }
    }
    Run { outcomes, error: None }
}

/// Run a parsed call `{"tool": ..., "args": {...}}`. Calls that change
/// something that matters are refused: they go through the user first.
pub fn run_call(call: &Value) -> Result<Outcome> {
    if find(call).is_some_and(|t| t.risk == Risk::Confirm) {
        bail!("that needs your OK first");
    }
    run_trusted(call)
}

/// Run a call the user has agreed to (or typed themselves).
pub fn run_confirmed(call: &Value) -> Result<Outcome> {
    run_trusted(call)
}

/// Run calls the user typed as commands: they asked for exactly that, so no
/// second question is needed. Stops at the first failure.
pub fn run_typed(calls: &[Value]) -> Run {
    let mut outcomes = Vec::new();
    for call in calls {
        match run_trusted(call) {
            Ok(o) => outcomes.push(o),
            Err(e) => return Run { outcomes, error: Some(e) },
        }
    }
    Run { outcomes, error: None }
}

fn run_trusted(call: &Value) -> Result<Outcome> {
    eprintln!("[tool call: {call}]");
    let name = call["tool"].as_str().context("no tool name")?;
    let tool = REGISTRY.iter().find(|t| t.name == name).with_context(|| format!("unknown tool {name:?}"))?;
    let text = (tool.run)(&call["args"])?;
    Ok(Outcome { tool: tool.name, kind: tool.kind, call: call.clone(), text })
}

/// A short readable form of a call, e.g. `volume(level=40)`.
pub fn describe_call(call: &Value) -> String {
    let name = call["tool"].as_str().unwrap_or("?");
    let args: Vec<String> = call["args"]
        .as_object()
        .map(|o| o.iter().map(|(k, v)| format!("{k}={}", v.as_str().map_or_else(|| v.to_string(), str::to_string))).collect())
        .unwrap_or_default();
    format!("{name}({})", args.join(", "))
}

// --- typed commands -------------------------------------------------------

/// Typed shortcuts: (command, tool, fixed argument value if any).
const COMMANDS: &[(&str, &str, Option<&str>)] = &[
    ("timer", "set_timer", None),
    ("remind", "remind", None),
    ("reminders", "reminders_list", None),
    ("volume", "volume", None),
    ("vol", "volume", None),
    ("mute", "mute", Some("on")),
    ("unmute", "mute", Some("off")),
    ("pause", "media", Some("play_pause")),
    ("play", "media", Some("play_pause")),
    ("next", "media", Some("next")),
    ("prev", "media", Some("previous")),
    ("previous", "media", Some("previous")),
    ("open", "open_app", None),
    ("url", "open_url", None),
    ("note", "note_add", None),
    ("notes", "notes_list", None),
    ("calc", "calculate", None),
    ("time", "get_time", None),
    ("date", "get_time", None),
    ("status", "system_status", None),
    ("playing", "now_playing", None),
    ("clipboard", "clipboard", None),
    ("find", "find_files", None),
    ("ls", "list_dir", None),
    ("read", "read_file", None),
    ("write", "write_file", None),
    ("run", "run_command", None),
    ("search", "web_search", None),
    ("grep", "search_text", None),
    ("remember", "remember", None),
    ("recall", "recall", None),
    ("memories", "memories_list", None),
    ("forget", "forget", None),
    ("tags", "tags_list", None),
    ("tag", "tag_add", None),
    ("fetch", "web_fetch", None),
];

/// The list of typed commands, for `/help`.
pub fn help() -> String {
    let mut s = String::from("Type a request in plain words, or use a command:\n");
    for t in REGISTRY {
        let usage: Vec<String> = t.args.iter().map(|a| format!("<{}>", a.name)).collect();
        let commands: Vec<&str> = COMMANDS.iter().filter(|c| c.1 == t.name).map(|c| c.0).collect();
        let shown = commands.iter().map(|c| format!("/{c}")).collect::<Vec<_>>().join(" ");
        let fixed = COMMANDS.iter().any(|c| c.1 == t.name && c.2.is_some());
        let args = if fixed || usage.is_empty() { String::new() } else { format!(" {}", usage.join(" ")) };
        s.push_str(&format!("{shown}{args}: {}\n", t.doc));
    }
    s.push_str("/help: this list");
    s
}

/// Parse a typed command such as `/volume 40` into tool calls. `None` means
/// the line is not a command. `Some(Err(text))` is a message to show: a usage
/// hint, or the help text.
pub fn command(line: &str) -> Option<Result<Vec<Value>, String>> {
    let line = line.trim();
    let rest = line.strip_prefix('/')?;
    let (word, arg) = rest.split_once(char::is_whitespace).map_or((rest, ""), |(w, a)| (w, a.trim()));
    let word = word.to_lowercase();
    if word == "help" || word == "?" || word.is_empty() {
        return Some(Err(help()));
    }
    let Some(&(_, tool_name, fixed)) = COMMANDS.iter().find(|c| c.0 == word) else {
        return Some(Err(format!("I don't know the command /{word}. Type /help to see them.")));
    };
    let tool = REGISTRY.iter().find(|t| t.name == tool_name)?;
    let arg = if tool_name == "list_dir" && arg.is_empty() { "~" } else { arg };
    // "/tag add gardening plants, flowers" reads naturally; the word "add" isn't part of the tag.
    let arg = if tool_name == "tag_add" { arg.strip_prefix("add ").map_or(arg, str::trim_start) } else { arg };
    let mut args = serde_json::Map::new();
    if let Some(value) = fixed {
        args.insert(tool.args[0].name.into(), Value::String(value.into()));
    } else if !tool.args.is_empty() {
        // The last argument takes the rest of the line; earlier integers come first.
        let mut remaining = arg;
        for (i, a) in tool.args.iter().enumerate() {
            let last = i + 1 == tool.args.len();
            let (piece, after) = if last { (remaining, "") } else { remaining.split_once(char::is_whitespace).unwrap_or((remaining, "")) };
            let piece = piece.trim();
            let usage = || format!("Usage: /{word} {}", tool.args.iter().map(|a| format!("<{}>", a.name)).collect::<Vec<_>>().join(" "));
            if piece.is_empty() {
                return Some(Err(usage()));
            }
            let value = match &a.kind {
                ArgKind::Int { max } => match piece.trim_end_matches(['m', '%']).parse::<u64>() {
                    Ok(n) if n <= u64::from(*max) => Value::from(n),
                    Ok(_) => return Some(Err(format!("{} must be at most {max}.", a.name))),
                    Err(_) => return Some(Err(usage())),
                },
                ArgKind::Text { .. } => Value::String(piece.to_string()),
                ArgKind::Choice(choices) => match choices.iter().find(|c| c.eq_ignore_ascii_case(piece)) {
                    Some(c) => Value::String((*c).into()),
                    None => return Some(Err(format!("{} must be one of: {}.", a.name, choices.join(", ")))),
                },
            };
            args.insert(a.name.into(), value);
            remaining = after;
        }
    }
    // "/open https://..." means the web address, not an app.
    let tool_name = if tool_name == "open_app" && (arg.starts_with("http://") || arg.starts_with("https://")) {
        args = serde_json::Map::from_iter([("url".to_string(), Value::String(arg.to_string()))]);
        "open_url"
    } else {
        tool_name
    };
    Some(Ok(vec![serde_json::json!({ "tool": tool_name, "args": args })]))
}

/// Names of all tools, for tests and `ojp-cli`.
pub fn names() -> Vec<&'static str> {
    REGISTRY.iter().map(|t| t.name).collect()
}

fn int(args: &Value, key: &str) -> Result<u64> {
    args[key].as_u64().with_context(|| format!("missing integer {key}"))
}

fn text<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args[key].as_str().with_context(|| format!("missing {key}"))
}

fn dry_run() -> bool {
    std::env::var("OJP_TOOLS_DRY_RUN").is_ok_and(|v| v == "1")
}

/// Run an action command (skipped under dry run).
fn run(program: &str, args: &[&str]) -> Result<()> {
    if dry_run() {
        eprintln!("[dry run: {program} {}]", args.join(" "));
        return Ok(());
    }
    let status = Command::new(program).args(args).status().with_context(|| format!("couldn't run {program}"))?;
    if !status.success() {
        bail!("{program} exited with {status}");
    }
    Ok(())
}

/// Run a read-only command and return its trimmed output.
fn capture(program: &str, args: &[&str]) -> Result<String> {
    let out = Command::new(program).args(args).output().with_context(|| format!("couldn't run {program}"))?;
    if !out.status.success() {
        bail!("{program} exited with {}", out.status);
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn volume(level: u64) -> Result<String> {
    let level = level.min(100);
    run("wpctl", &["set-volume", "@DEFAULT_AUDIO_SINK@", &format!("{level}%")])?;
    Ok(format!("Volume set to {level}%."))
}

fn mute(state: &str) -> Result<String> {
    let on = state == "on";
    run("wpctl", &["set-mute", "@DEFAULT_AUDIO_SINK@", if on { "1" } else { "0" }])?;
    Ok(if on { "Muted.".into() } else { "Unmuted.".into() })
}

fn media(action: &str) -> Result<String> {
    let (command, said) = match action {
        "play_pause" => ("play-pause", "Toggled play/pause."),
        "next" => ("next", "Skipped to the next track."),
        "previous" => ("previous", "Back to the previous track."),
        other => bail!("unknown media action {other:?}"),
    };
    run("playerctl", &[command])?;
    Ok(said.into())
}

fn set_timer(minutes: u64) -> Result<String> {
    if minutes == 0 {
        bail!("a timer needs at least one minute");
    }
    schedule(minutes, &format!("Your {minutes}-minute timer is done."))?;
    Ok(format!("Timer set for {minutes} minute{}.", if minutes == 1 { "" } else { "s" }))
}

fn remind(minutes: u64, what: &str) -> Result<String> {
    if minutes == 0 {
        bail!("a reminder needs at least one minute");
    }
    let what = what.trim();
    if what.is_empty() {
        bail!("remind you about what?");
    }
    schedule(minutes, what)?;
    Ok(format!("I'll remind you to {what} in {minutes} minute{}.", if minutes == 1 { "" } else { "s" }))
}

/// Store a timer or reminder (kept even if the app is closed) and make sure
/// the thread that announces them is running.
fn schedule(minutes: u64, text: &str) -> Result<()> {
    if dry_run() {
        eprintln!("[dry run: reminder in {minutes} min: {text}]");
        return Ok(());
    }
    crate::reminders::add(minutes, text)?;
    crate::reminders::start_watcher();
    Ok(())
}

fn reminders_list() -> Result<String> {
    let list = crate::reminders::list();
    if list.is_empty() {
        return Ok("There are no pending timers or reminders.".into());
    }
    let now = crate::reminders::now();
    let lines: Vec<String> = list.iter().map(|r| format!("{}: {}", crate::reminders::describe(r.due, now), r.text)).collect();
    Ok(format!("{} pending:\n{}", list.len(), lines.join("\n")))
}

/// Launch an installed app by matching its .desktop entry's Name= (or file
/// name) case-insensitively, so only real installed applications can start.
fn open_app(name: &str) -> Result<String> {
    let wanted = name.trim().to_lowercase();
    let entries = desktop_entries();
    let exact = entries.iter().find(|(path, display)| {
        display.to_lowercase() == wanted
            || path.file_stem().is_some_and(|s| s.to_string_lossy().to_lowercase().ends_with(&wanted))
    });
    let entry = match exact {
        Some(e) => e,
        // A looser match: the name appears inside the app's name ("code" -> "Visual Studio Code").
        None => match entries.iter().filter(|(_, d)| d.to_lowercase().contains(&wanted)).collect::<Vec<_>>().as_slice() {
            [only] => only,
            [] => {
                let words: Vec<&str> = wanted.split_whitespace().collect();
                let mut near: Vec<&str> = entries
                    .iter()
                    .filter(|(_, d)| words.iter().any(|w| w.len() > 2 && d.to_lowercase().contains(w)))
                    .map(|(_, d)| d.as_str())
                    .collect();
                // Typos: app names within a couple of letters of what was typed.
                let slack = if wanted.len() > 7 { 3 } else { 2 };
                near.extend(entries.iter().map(|(_, d)| d.as_str()).filter(|d| edit_distance(&d.to_lowercase(), &wanted) <= slack));
                near.sort();
                near.dedup();
                near.truncate(5);
                if near.is_empty() {
                    bail!("I couldn't find an app called {name:?}. Use the name shown in your app menu.");
                }
                bail!("I couldn't find an app called {name:?}. Did you mean: {}?", near.join(", "));
            }
            many => {
                let mut names: Vec<&str> = many.iter().map(|(_, d)| d.as_str()).collect();
                names.sort();
                names.dedup();
                names.truncate(5);
                bail!("{name:?} matches several apps: {}. Which one?", names.join(", "));
            }
        },
    };
    run("gio", &["launch", &entry.0.to_string_lossy()])?;
    Ok(format!("Opening {}.", entry.1))
}

/// Levenshtein distance between two short strings.
fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            cur.push((prev[j] + usize::from(ca != cb)).min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Open a web address; only plain http(s) links without spaces.
fn open_url(url: &str) -> Result<String> {
    let ok = (url.starts_with("https://") || url.starts_with("http://"))
        && !url.contains(char::is_whitespace)
        && !url.contains(['"', '\'', '`', '<', '>', '\\']);
    if !ok {
        bail!("{url:?} isn't a plain http(s) address");
    }
    run("xdg-open", &[url])?;
    Ok(format!("Opening {url}."))
}

fn data_dir() -> PathBuf {
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).unwrap_or_else(|| home.join(".local/share")).join("ojp")
}

fn notes_path() -> PathBuf {
    std::env::var_os("OJP_NOTES_FILE").map(PathBuf::from).unwrap_or_else(|| data_dir().join("notes.md"))
}

fn note_add(note: &str) -> Result<String> {
    let note = note.trim();
    if note.is_empty() {
        bail!("nothing to note");
    }
    if dry_run() {
        eprintln!("[dry run: note {note:?}]");
        return Ok("Saved your note.".into());
    }
    let stamp = capture("date", &["+%F %R"]).unwrap_or_default();
    let path = notes_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
    writeln!(file, "- {stamp} {}", note.replace('\n', " "))?;
    Ok("Saved your note.".into())
}

fn notes_list() -> Result<String> {
    let text = std::fs::read_to_string(notes_path()).unwrap_or_default();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return Ok("There are no saved notes.".into());
    }
    let recent = &lines[lines.len().saturating_sub(10)..];
    Ok(format!("{} note(s), newest last:\n{}", lines.len(), recent.join("\n")))
}

fn remember(what: &str) -> Result<String> {
    if dry_run() {
        eprintln!("[dry run: remember {what:?}]");
        return Ok("I'll remember that.".into());
    }
    let m = crate::memory::add(what)?;
    Ok(format!("I'll remember that: {} (filed under {}; /forget #{} removes it).", m.text, m.tags.join(", "), m.id))
}

fn recall(query: &str) -> Result<String> {
    let query = query.trim();
    // "#3": the full text of one memory.
    if let Some(id) = query.strip_prefix('#').and_then(|n| n.trim().parse::<u64>().ok()) {
        let m = crate::memory::list().into_iter().find(|m| m.id == id).with_context(|| format!("I have no memory #{id}"))?;
        return Ok(format!("#{} [{}]: {}", m.id, m.tags.join(", "), m.text));
    }
    let mut found: Vec<crate::memory::Memory> = Vec::new();
    // A tag (or a word meaning one): everything filed under it.
    if let Some(tag) = query.split_whitespace().find_map(crate::memory::tag_for_word).filter(|_| query.split_whitespace().count() <= 2) {
        found.extend(crate::memory::with_tag(&tag));
    }
    for (_, m) in crate::memory::search(query, 6) {
        if !found.iter().any(|f| f.id == m.id) {
            found.push(m);
        }
    }
    found.truncate(8);
    if found.is_empty() {
        return Ok(format!("I don't remember anything about {query:?}."));
    }
    let lines: Vec<String> = found.iter().map(|m| format!("#{} [{}]: {}", m.id, m.tags.join(", "), m.gist())).collect();
    Ok(format!("What I remember:\n{}\n(Ask for #number to see the full text.)", lines.join("\n")))
}

fn tags_list() -> Result<String> {
    let counts = crate::memory::tag_counts();
    let used: Vec<String> = counts.iter().filter(|(_, n)| *n > 0).map(|(t, n)| format!("{t} ({n})")).collect();
    let unused: Vec<&str> = counts.iter().filter(|(_, n)| *n == 0).map(|(t, _)| t.as_str()).collect();
    Ok(format!("Memory tags in use: {}.\nAvailable: {}.", if used.is_empty() { "none yet".to_string() } else { used.join(", ") }, unused.join(", ")))
}

fn tag_add(name: &str, words: &str) -> Result<String> {
    let words: Vec<String> = words.split([',', ' ']).filter(|w| !w.is_empty()).map(String::from).collect();
    crate::memory::add_vocabulary(name, &words)?;
    Ok(format!("Tag \"{}\" now covers: {}.", name.trim().to_lowercase(), if words.is_empty() { "(just its own name)".to_string() } else { words.join(", ") }))
}

fn memories_list() -> Result<String> {
    let all = crate::memory::list();
    if all.is_empty() {
        return Ok("I don't have anything saved about you yet.".into());
    }
    let recent = &all[all.len().saturating_sub(25)..];
    let lines: Vec<String> = recent.iter().map(|m| format!("#{} [{}]: {}", m.id, m.tags.join(", "), m.gist())).collect();
    Ok(format!("{} thing(s) remembered{}:\n{}", all.len(), if all.len() > 25 { ", newest 25" } else { "" }, lines.join("\n")))
}

fn forget(query: &str) -> Result<String> {
    let gone = crate::memory::forget(query)?;
    let lines: Vec<String> = gone.iter().map(|m| m.text.clone()).collect();
    Ok(format!("Forgotten: {}.", lines.join("; ")))
}

fn get_time() -> Result<String> {
    capture("date", &["+%A, %B %-d, %Y, %-I:%M %p"])
}

fn system_status() -> Result<String> {
    let mut parts = Vec::new();
    if let Ok(mem) = std::fs::read_to_string("/proc/meminfo") {
        let kb = |key: &str| -> Option<f64> {
            mem.lines().find(|l| l.starts_with(key))?.split_whitespace().nth(1)?.parse().ok()
        };
        if let (Some(total), Some(avail)) = (kb("MemTotal:"), kb("MemAvailable:")) {
            parts.push(format!("memory: {:.1} GB free out of {:.1} GB total", avail / 1e6, total / 1e6));
        }
    }
    if let Ok(load) = std::fs::read_to_string("/proc/loadavg")
        && let Some(one) = load.split_whitespace().next()
    {
        parts.push(format!("CPU load average {one}"));
    }
    if let Ok(df) = capture("df", &["-h", "--output=avail,pcent", "/home"])
        && let Some(line) = df.lines().nth(1)
    {
        let mut f = line.split_whitespace();
        if let (Some(avail), Some(used)) = (f.next(), f.next()) {
            parts.push(format!("home disk: {avail} free, {used} full"));
        }
    }
    if let Ok(read) = std::fs::read_dir("/sys/class/power_supply") {
        for entry in read.flatten() {
            let p = entry.path();
            if p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("BAT"))
                && let (Ok(cap), Ok(status)) =
                    (std::fs::read_to_string(p.join("capacity")), std::fs::read_to_string(p.join("status")))
            {
                parts.push(format!("battery {}% ({})", cap.trim(), status.trim().to_lowercase()));
            }
        }
    }
    if let Ok(up) = std::fs::read_to_string("/proc/uptime")
        && let Some(secs) = up.split_whitespace().next().and_then(|s| s.parse::<f64>().ok())
    {
        parts.push(format!("running for {:.1} hours since the last restart", secs / 3600.0));
    }
    if parts.is_empty() {
        bail!("couldn't read the system status");
    }
    Ok(parts.join("; "))
}

fn now_playing() -> Result<String> {
    let status = capture("playerctl", &["status"]).context("no media player is running")?;
    let what = capture("playerctl", &["metadata", "--format", "{{artist}} - {{title}}"]).unwrap_or_default();
    Ok(format!("{status}: {what}"))
}

fn clipboard() -> Result<String> {
    let text = capture("wl-paste", &["-n"])
        .or_else(|_| capture("xclip", &["-selection", "clipboard", "-o"]))
        .context("couldn't read the clipboard (needs wl-paste or xclip)")?;
    if text.is_empty() {
        return Ok("The clipboard is empty.".into());
    }
    let shown: String = text.chars().take(500).collect();
    Ok(if text.chars().count() > 500 { format!("{shown}… (cut off)") } else { shown })
}

fn find_files(query: &str) -> Result<String> {
    let query = query.trim();
    if query.is_empty() || !query.chars().all(|c| c.is_alphanumeric() || " ._-".contains(c)) {
        bail!("{query:?} isn't a simple file name to search for");
    }
    let home = std::env::var("HOME").context("no home folder")?;
    let pattern = format!("*{query}*");
    let out = capture(
        "find",
        &[
            &home, "-maxdepth", "6", "(", "-name", ".*", "-o", "-name", "node_modules", "-o", "-name", "target", ")", "-prune",
            "-o", "-iname", &pattern, "-print",
        ],
    )?;
    let hits: Vec<String> = out.lines().take(8).map(|l| ext::shorten(Path::new(l))).collect();
    if hits.is_empty() {
        return Ok(format!("No files matching {query:?} in your home folder."));
    }
    let total = out.lines().count();
    Ok(format!("{total} match(es){}:\n{}", if total > 8 { ", first 8" } else { "" }, hits.join("\n")))
}

// --- calculator -----------------------------------------------------------

/// Evaluate an arithmetic expression exactly as a person would write it.
fn calculate(expression: &str) -> Result<String> {
    let value = Calc::new(expression)?.run()?;
    if !value.is_finite() {
        bail!("that isn't a finite number");
    }
    Ok(format!("{} = {}", expression.trim(), format_number(value)))
}

fn format_number(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    let s = format!("{v:.10}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

struct Calc {
    tokens: Vec<Token>,
    pos: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
    Op(char),
    Ident(String),
}

impl Calc {
    fn new(src: &str) -> Result<Calc> {
        let chars: Vec<char> = src.chars().collect();
        let mut tokens = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c.is_whitespace() || c == ',' {
                i += 1;
            } else if c.is_ascii_digit() || c == '.' {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                tokens.push(Token::Num(s.parse().with_context(|| format!("bad number {s:?}"))?));
            } else if c.is_ascii_alphabetic() {
                let start = i;
                while i < chars.len() && chars[i].is_ascii_alphabetic() {
                    i += 1;
                }
                tokens.push(Token::Ident(chars[start..i].iter().collect::<String>().to_lowercase()));
            } else if "+-*/^()%".contains(c) {
                tokens.push(Token::Op(c));
                i += 1;
            } else if c == '×' || c == '·' {
                tokens.push(Token::Op('*'));
                i += 1;
            } else if c == '÷' {
                tokens.push(Token::Op('/'));
                i += 1;
            } else if c == '−' {
                tokens.push(Token::Op('-'));
                i += 1;
            } else {
                bail!("can't calculate with {c:?}");
            }
        }
        if tokens.is_empty() {
            bail!("nothing to calculate");
        }
        Ok(Calc { tokens, pos: 0 })
    }

    fn run(mut self) -> Result<f64> {
        let v = self.expr()?;
        if self.pos != self.tokens.len() {
            bail!("unexpected {:?}", self.tokens[self.pos]);
        }
        Ok(v)
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn eat(&mut self, op: char) -> bool {
        if self.peek() == Some(&Token::Op(op)) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    // expr := term (("+"|"-") term)*
    fn expr(&mut self) -> Result<f64> {
        let mut v = self.term()?;
        loop {
            if self.eat('+') {
                v += self.term()?;
            } else if self.eat('-') {
                v -= self.term()?;
            } else {
                return Ok(v);
            }
        }
    }

    // term := power (("*"|"/"|"%") power | implicit multiplication)*
    fn term(&mut self) -> Result<f64> {
        let mut v = self.power()?;
        loop {
            if self.eat('*') {
                v *= self.power()?;
            } else if self.eat('/') {
                let d = self.power()?;
                if d == 0.0 {
                    bail!("division by zero");
                }
                v /= d;
            } else if self.eat('%') {
                // "15% of 240" and "50%" are percentages; "17 % 5" is a remainder.
                if self.peek() == Some(&Token::Ident("of".into())) {
                    self.pos += 1;
                    v = v / 100.0 * self.power()?;
                } else if matches!(self.peek(), Some(Token::Num(_)) | Some(Token::Op('(')) | Some(Token::Ident(_))) {
                    let d = self.power()?;
                    if d == 0.0 {
                        bail!("division by zero");
                    }
                    v %= d;
                } else {
                    v /= 100.0;
                }
            } else if matches!(self.peek(), Some(Token::Num(_)) | Some(Token::Ident(_)) | Some(Token::Op('('))) {
                v *= self.power()?; // "2(3+4)", "3 pi"
            } else {
                return Ok(v);
            }
        }
    }

    // power := unary ("^" power)?   (right associative)
    fn power(&mut self) -> Result<f64> {
        let base = self.unary()?;
        if self.eat('^') {
            let exp = self.power()?;
            return Ok(base.powf(exp));
        }
        Ok(base)
    }

    fn unary(&mut self) -> Result<f64> {
        if self.eat('-') {
            return Ok(-self.unary()?);
        }
        if self.eat('+') {
            return self.unary();
        }
        self.atom()
    }

    fn atom(&mut self) -> Result<f64> {
        match self.tokens.get(self.pos).cloned() {
            Some(Token::Num(n)) => {
                self.pos += 1;
                Ok(n)
            }
            Some(Token::Op('(')) => {
                self.pos += 1;
                let v = self.expr()?;
                if !self.eat(')') {
                    bail!("missing )");
                }
                Ok(v)
            }
            Some(Token::Ident(name)) => {
                self.pos += 1;
                match name.as_str() {
                    "pi" => Ok(std::f64::consts::PI),
                    "e" => Ok(std::f64::consts::E),
                    f @ ("sqrt" | "sin" | "cos" | "tan" | "ln" | "log" | "abs" | "exp" | "floor" | "ceil" | "round") => {
                        if !self.eat('(') {
                            bail!("{f} needs ( )");
                        }
                        let x = self.expr()?;
                        if !self.eat(')') {
                            bail!("missing )");
                        }
                        Ok(match f {
                            "sqrt" => x.sqrt(),
                            "sin" => x.sin(),
                            "cos" => x.cos(),
                            "tan" => x.tan(),
                            "ln" => x.ln(),
                            "log" => x.log10(),
                            "abs" => x.abs(),
                            "exp" => x.exp(),
                            "floor" => x.floor(),
                            "ceil" => x.ceil(),
                            _ => x.round(),
                        })
                    }
                    other => bail!("don't know {other:?}"),
                }
            }
            Some(t) => bail!("unexpected {t:?}"),
            None => bail!("the expression ends too early"),
        }
    }
}

fn desktop_entries() -> Vec<(PathBuf, String)> {
    let mut dirs: Vec<PathBuf> = std::env::var("XDG_DATA_DIRS")
        .unwrap_or_else(|_| "/usr/local/share:/usr/share".into())
        .split(':')
        .map(PathBuf::from)
        .collect();
    if let Some(home) = std::env::var_os("HOME") {
        dirs.insert(0, PathBuf::from(home).join(".local/share"));
    }
    let mut entries = Vec::new();
    for dir in dirs {
        let Ok(read) = std::fs::read_dir(dir.join("applications")) else { continue };
        for path in read.flatten().map(|e| e.path()) {
            if path.extension().is_none_or(|e| e != "desktop") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            if let Some(display) = text.lines().find_map(|l| l.strip_prefix("Name=")) {
                entries.push((path.clone(), display.to_string()));
            }
        }
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    /// Tests that change environment variables run one at a time.
    static ENV: Mutex<()> = Mutex::new(());

    fn calc(s: &str) -> String {
        calculate(s).unwrap()
    }

    #[test]
    fn arithmetic() {
        assert_eq!(calc("7^3+4*7*2-3"), "7^3+4*7*2-3 = 396");
        assert_eq!(calc("2+3*4"), "2+3*4 = 14");
        assert_eq!(calc("(2+3)*4"), "(2+3)*4 = 20");
        assert_eq!(calc("2^3^2"), "2^3^2 = 512");
        assert_eq!(calc("10/4"), "10/4 = 2.5");
        assert_eq!(calc("0.15*240"), "0.15*240 = 36");
        assert_eq!(calc("sqrt(144)+abs(-3)"), "sqrt(144)+abs(-3) = 15");
        assert_eq!(calc("2(3+4)"), "2(3+4) = 14");
        assert_eq!(calc("12 × 13"), "12 × 13 = 156");
        assert_eq!(calc("1/3"), "1/3 = 0.3333333333");
        assert_eq!(calc("17 % 5"), "17 % 5 = 2");
        assert_eq!(calc("15% of 240"), "15% of 240 = 36");
        assert_eq!(calc("15 % of 240 + 4"), "15 % of 240 + 4 = 40");
        assert_eq!(calc("50%"), "50% = 0.5");
        assert_eq!(calc("200 * 15%"), "200 * 15% = 30");
    }

    #[test]
    fn calculator_rejects_nonsense() {
        assert!(calculate("1/0").is_err());
        assert!(calculate("2+").is_err());
        assert!(calculate("foo(2)").is_err());
        assert!(calculate("2; rm -rf /").is_err());
        assert!(calculate("").is_err());
        assert!(calculate("(1+2").is_err());
    }

    #[test]
    fn every_tool_is_in_the_prompt_and_grammar() {
        let (prompt, grammar) = (system_prompt(), grammar(false));
        for name in names() {
            assert!(prompt.contains(&format!("- {name}(")), "{name} missing from the prompt");
            assert!(grammar.contains(&format!("\"{name}\\\", \\\"args\\\"")), "{name} missing from the grammar");
        }
        assert!(grammar.starts_with("root ::="));
    }

    #[test]
    fn examples_are_valid_calls() {
        for (request, calls) in EXAMPLES {
            let calls: Value = serde_json::from_str(calls).unwrap_or_else(|e| panic!("{request}: {e}"));
            let calls = calls.as_array().unwrap_or_else(|| panic!("{request}: not an array"));
            assert!(!calls.is_empty() && calls.len() <= 3, "{request}");
            for call in calls {
                let tool = REGISTRY.iter().find(|t| t.name == call["tool"]).unwrap_or_else(|| panic!("{request}: unknown tool"));
                assert_eq!(call["args"].as_object().unwrap().len(), tool.args.len(), "{request}");
            }
        }
        for t in REGISTRY {
            assert!(EXAMPLES.iter().any(|(_, c)| c.contains(&format!("\"tool\": \"{}\"", t.name))), "{} needs an example", t.name);
        }
    }

    #[test]
    fn urls_are_checked() {
        let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
        assert!(open_url("file:///etc/passwd").is_err());
        assert!(open_url("https://example.com/a b").is_err());
        assert!(open_url("javascript:alert(1)").is_err());
        unsafe { std::env::set_var("OJP_TOOLS_DRY_RUN", "1") };
        assert!(open_url("https://example.com/path?x=1").is_ok());
    }

    #[test]
    fn find_files_rejects_odd_queries() {
        assert!(find_files("a; ls").is_err());
        assert!(find_files("").is_err());
        assert!(find_files("*").is_err());
    }

    #[test]
    fn calls_dispatch_and_unknown_tools_fail() {
        let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("OJP_TOOLS_DRY_RUN", "1") };
        let out = run_call(&json!({"tool": "calculate", "args": {"expression": "6*7"}})).unwrap();
        assert_eq!((out.tool, out.kind, out.text.as_str()), ("calculate", Kind::Read, "6*7 = 42"));
        let out = run_call(&json!({"tool": "volume", "args": {"level": 30}})).unwrap();
        assert_eq!((out.kind, out.text.as_str()), (Kind::Act, "Volume set to 30%."));
        let run = run_calls(&[json!({"tool": "calculate", "args": {"expression": "2+2"}}), json!({"tool": "nope", "args": {}}), json!({"tool": "get_time", "args": {}})]);
        assert_eq!(run.outcomes.len(), 1, "stops at the first failure");
        assert!(run.error.is_some());
        assert!(run_call(&json!({"tool": "format_disk", "args": {}})).is_err());
    }

    #[test]
    fn notes_round_trip() {
        let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("ojp-notes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        unsafe {
            std::env::set_var("OJP_NOTES_FILE", dir.join("notes.md"));
            std::env::remove_var("OJP_TOOLS_DRY_RUN");
        }
        assert_eq!(notes_list().unwrap(), "There are no saved notes.");
        note_add("buy oat milk").unwrap();
        note_add("call the dentist").unwrap();
        let list = notes_list().unwrap();
        assert!(list.starts_with("2 note(s)") && list.contains("buy oat milk") && list.contains("call the dentist"), "{list}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn typed_commands() {
        let call = |line: &str| command(line).unwrap().unwrap()[0].clone();
        assert_eq!(call("/volume 40"), json!({"tool": "volume", "args": {"level": 40}}));
        assert_eq!(call("/vol 15%"), json!({"tool": "volume", "args": {"level": 15}}));
        assert_eq!(call("/timer 10"), json!({"tool": "set_timer", "args": {"minutes": 10}}));
        assert_eq!(call("/mute"), json!({"tool": "mute", "args": {"state": "on"}}));
        assert_eq!(call("/unmute"), json!({"tool": "mute", "args": {"state": "off"}}));
        assert_eq!(call("/pause"), json!({"tool": "media", "args": {"action": "play_pause"}}));
        assert_eq!(call("/next"), json!({"tool": "media", "args": {"action": "next"}}));
        assert_eq!(call("/calc 17 * 23"), json!({"tool": "calculate", "args": {"expression": "17 * 23"}}));
        assert_eq!(call("/remind 30 stretch your legs"), json!({"tool": "remind", "args": {"minutes": 30, "text": "stretch your legs"}}));
        assert_eq!(call("/open Visual Studio Code"), json!({"tool": "open_app", "args": {"name": "Visual Studio Code"}}));
        assert_eq!(call("/open https://example.com"), json!({"tool": "open_url", "args": {"url": "https://example.com"}}));
        assert_eq!(call("/time"), json!({"tool": "get_time", "args": {}}));
        assert_eq!(call("/tag add gardening plants, flowers"), json!({"tool": "tag_add", "args": {"name": "gardening", "words": "plants, flowers"}}));
        assert_eq!(call("/note buy oat milk"), json!({"tool": "note_add", "args": {"text": "buy oat milk"}}));
        assert!(command("hello").is_none());
        assert!(command("/volume").unwrap().unwrap_err().starts_with("Usage: /volume"));
        assert!(command("/volume 500").unwrap().unwrap_err().contains("at most 100"));
        assert!(command("/volume loud").unwrap().unwrap_err().starts_with("Usage"));
        assert!(command("/frobnicate").unwrap().unwrap_err().contains("/help"));
        assert!(command("/help").unwrap().unwrap_err().contains("/volume"));
    }

    #[test]
    fn typos_are_measured() {
        assert_eq!(edit_distance("firefox", "fierfox"), 2);
        assert_eq!(edit_distance("brave", "brave"), 0);
        assert_eq!(edit_distance("", "abc"), 3);
    }

    #[test]
    fn every_tool_has_a_typed_command() {
        for name in names() {
            assert!(COMMANDS.iter().any(|c| c.1 == name), "{name} has no /command");
        }
    }
}
