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
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use llama_crab::chat::BuiltinTemplate;
use llama_crab::chat::template::render_builtin;
use llama_crab::{ChatMessage, LlamaSampler, Role};
use serde_json::Value;

use crate::model::{Model, Seq};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Looks something up; the result is phrased into an answer by the model.
    Read,
    /// Does something; the tool's own confirmation is the answer.
    Act,
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
    run: fn(&Value) -> Result<String>,
}

const REGISTRY: &[Tool] = &[
    Tool { name: "set_timer", doc: "start a countdown timer", args: &[Arg { name: "minutes", kind: ArgKind::Int { max: 999 } }], kind: Kind::Act, run: |a| set_timer(int(a, "minutes")?) },
    Tool { name: "volume", doc: "set the speaker volume, 0-100", args: &[Arg { name: "level", kind: ArgKind::Int { max: 100 } }], kind: Kind::Act, run: |a| volume(int(a, "level")?) },
    Tool { name: "mute", doc: "turn the speaker sound off or on (muting)", args: &[Arg { name: "state", kind: ArgKind::Choice(&["on", "off"]) }], kind: Kind::Act, run: |a| mute(text(a, "state")?) },
    Tool { name: "media", doc: "play, pause or skip the music or video player (not the volume)", args: &[Arg { name: "action", kind: ArgKind::Choice(&["play_pause", "next", "previous"]) }], kind: Kind::Act, run: |a| media(text(a, "action")?) },
    Tool { name: "open_app", doc: "launch an installed application by name", args: &[Arg { name: "name", kind: ArgKind::Text { max: 40 } }], kind: Kind::Act, run: |a| open_app(text(a, "name")?) },
    Tool { name: "open_url", doc: "open a web address in the browser", args: &[Arg { name: "url", kind: ArgKind::Text { max: 200 } }], kind: Kind::Act, run: |a| open_url(text(a, "url")?) },
    Tool { name: "note_add", doc: "save a short note", args: &[Arg { name: "text", kind: ArgKind::Text { max: 200 } }], kind: Kind::Act, run: |a| note_add(text(a, "text")?) },
    Tool { name: "calculate", doc: "exact arithmetic, e.g. 7^3+4*7*2-3 (use * / ^ sqrt() sin() ln() pi)", args: &[Arg { name: "expression", kind: ArgKind::Text { max: 120 } }], kind: Kind::Read, run: |a| calculate(text(a, "expression")?) },
    Tool { name: "get_time", doc: "the current time, date or day of the week", args: &[], kind: Kind::Read, run: |_| get_time() },
    Tool { name: "system_status", doc: "battery, memory, disk space, load and uptime", args: &[], kind: Kind::Read, run: |_| system_status() },
    Tool { name: "now_playing", doc: "which song or video is playing (not the time)", args: &[], kind: Kind::Read, run: |_| now_playing() },
    Tool { name: "clipboard", doc: "read the text on the clipboard", args: &[], kind: Kind::Read, run: |_| clipboard() },
    Tool { name: "find_files", doc: "find files in the home folder by part of the name", args: &[Arg { name: "query", kind: ArgKind::Text { max: 60 } }], kind: Kind::Read, run: |a| find_files(text(a, "query")?) },
    Tool { name: "notes_list", doc: "show the saved notes", args: &[], kind: Kind::Read, run: |_| notes_list() },
];

/// The tool-use instructions given to the model, generated from the registry.
pub fn system_prompt() -> String {
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
    s.push_str("Respond with ONLY a JSON object like {\"tool\": \"<name>\", \"args\": {...}} and nothing else.");
    s
}

/// The GBNF grammar accepting exactly the registry's call shapes.
fn grammar() -> String {
    let alternatives: Vec<String> = REGISTRY
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
    format!(
        "root ::= \"{{\\\"tool\\\": \\\"\" call \"}}}}\"\ncall ::= ( {} )\nint ::= [0-9] | [1-9] [0-9] | [1-9] [0-9] [0-9]\n",
        alternatives.join("\n         | ")
    )
}

/// Worked examples shown to the model as earlier turns, one or two per tool.
/// Kept apart from the phrasings in `ojp-cli tool-eval`, which only reports.
const EXAMPLES: &[(&str, &str)] = &[
    ("Set a timer for 20 minutes.", r#"{"tool": "set_timer", "args": {"minutes": 20}}"#),
    ("Turn the sound up to 65.", r#"{"tool": "volume", "args": {"level": 65}}"#),
    ("Silence the speakers.", r#"{"tool": "mute", "args": {"state": "on"}}"#),
    ("Bring the sound back.", r#"{"tool": "mute", "args": {"state": "off"}}"#),
    ("Play the next song.", r#"{"tool": "media", "args": {"action": "next"}}"#),
    ("Hit pause on the video.", r#"{"tool": "media", "args": {"action": "play_pause"}}"#),
    ("Open the file manager.", r#"{"tool": "open_app", "args": {"name": "file manager"}}"#),
    ("Take me to wikipedia.org.", r#"{"tool": "open_url", "args": {"url": "https://wikipedia.org"}}"#),
    ("Note to self: call mom on Sunday.", r#"{"tool": "note_add", "args": {"text": "call mom on Sunday"}}"#),
    ("What is 45 plus 38?", r#"{"tool": "calculate", "args": {"expression": "45+38"}}"#),
    ("What's the current time?", r#"{"tool": "get_time", "args": {}}"#),
    ("Is my laptop charged?", r#"{"tool": "system_status", "args": {}}"#),
    ("Which track is on?", r#"{"tool": "now_playing", "args": {}}"#),
    ("What did I copy?", r#"{"tool": "clipboard", "args": {}}"#),
    ("Locate my vacation photos.", r#"{"tool": "find_files", "args": {"query": "vacation"}}"#),
    ("Show me what I wrote down.", r#"{"tool": "notes_list", "args": {}}"#),
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

/// Ask the model for a tool call, then run it.
pub fn handle(model: &mut Model, template: BuiltinTemplate, request: &str) -> Result<Outcome> {
    let call = ask(model, template, request)?;
    run_call(&call)
}

/// Ask the model for one tool call (grammar-constrained) and parse it.
pub fn ask(model: &mut Model, template: BuiltinTemplate, request: &str) -> Result<Value> {
    let mut messages = vec![ChatMessage::new(Role::System, system_prompt())];
    for (user, call) in EXAMPLES {
        messages.push(ChatMessage::new(Role::User, *user));
        messages.push(ChatMessage::new(Role::Assistant, *call));
    }
    messages.push(ChatMessage::new(Role::User, request));
    let prompt = render_builtin(template, &messages, &[], true);
    // SAFETY: llama_sampler_init_grammar parses the grammar during the call.
    let grammar = unsafe { LlamaSampler::grammar(model.llama().model(), &grammar(), "root") }.context("tool grammar rejected")?;
    let greedy = LlamaSampler::greedy().context("greedy sampler")?;
    let mut sampler = LlamaSampler::chain(vec![grammar, greedy], true).context("sampler chain")?;
    let text = model.complete(Seq::Tool, &prompt, &mut sampler, 96, &mut |_| {})?;
    serde_json::from_str(text.trim()).with_context(|| format!("tool call didn't parse: {text:?}"))
}

/// Run a parsed call `{"tool": ..., "args": {...}}`.
pub fn run_call(call: &Value) -> Result<Outcome> {
    eprintln!("[tool call: {call}]");
    let name = call["tool"].as_str().context("no tool name")?;
    let tool = REGISTRY.iter().find(|t| t.name == name).with_context(|| format!("unknown tool {name:?}"))?;
    let text = (tool.run)(&call["args"])?;
    Ok(Outcome { tool: tool.name, kind: tool.kind, call: call.clone(), text })
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
    // Lives only as long as this process; persistent timers belong in the
    // daemon once there is one.
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(minutes * 60));
        let _ = run("notify-send", &["Timer", &format!("Your {minutes}-minute timer is done.")]);
    });
    Ok(format!("Timer set for {minutes} minute{}.", if minutes == 1 { "" } else { "s" }))
}

/// Launch an installed app by matching its .desktop entry's Name= (or file
/// name) case-insensitively, so only real installed applications can start.
fn open_app(name: &str) -> Result<String> {
    let wanted = name.trim().to_lowercase();
    let entry = desktop_entries()
        .into_iter()
        .find(|(path, display)| {
            display.to_lowercase() == wanted
                || path.file_stem().is_some_and(|s| s.to_string_lossy().to_lowercase().ends_with(&wanted))
        })
        .with_context(|| format!("no installed app called {name:?}"))?;
    run("gio", &["launch", &entry.0.to_string_lossy()])?;
    Ok(format!("Opening {}.", entry.1))
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
    let hits: Vec<&str> = out.lines().take(8).collect();
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
                let d = self.power()?;
                if d == 0.0 {
                    bail!("division by zero");
                }
                v %= d;
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
        let (prompt, grammar) = (system_prompt(), grammar());
        for name in names() {
            assert!(prompt.contains(&format!("- {name}(")), "{name} missing from the prompt");
            assert!(grammar.contains(&format!("\"{name}\\\", \\\"args\\\"")), "{name} missing from the grammar");
        }
        assert!(grammar.starts_with("root ::="));
    }

    #[test]
    fn examples_are_valid_calls() {
        let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("OJP_TOOLS_DRY_RUN", "1") };
        for (request, call) in EXAMPLES {
            let call: Value = serde_json::from_str(call).unwrap_or_else(|e| panic!("{request}: {e}"));
            let tool = REGISTRY.iter().find(|t| t.name == call["tool"]).unwrap_or_else(|| panic!("{request}: unknown tool"));
            assert_eq!(call["args"].as_object().unwrap().len(), tool.args.len(), "{request}");
        }
        assert!(EXAMPLES.len() >= REGISTRY.len(), "every tool needs an example");
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
}
