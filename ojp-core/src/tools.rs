//! Device-control tools. The model never writes free text here: a GBNF
//! grammar forces its output into one of the call shapes below, so the
//! JSON always parses and only these tools/argument types can appear.
//! Tools act through standard desktop commands (PipeWire's wpctl, the
//! .desktop launcher, notify-send), never through a shell.
//!
//! OJP_TOOLS_DRY_RUN=1 prints each command instead of running it.
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use llama_crab::chat::BuiltinTemplate;
use llama_crab::chat::template::render_builtin;
use llama_crab::{ChatMessage, CompletionOptions, Llama, LlamaSampler, Role};
use serde_json::Value;

const SYSTEM_PROMPT: &str = "You control this computer through tools. Available tools:\n\
- set_timer(minutes: int)\n- open_app(name: str)\n- volume(level: int)  // 0-100\n\
Respond with ONLY a JSON object like {\"tool\": \"<name>\", \"args\": {...}} and nothing else.";

const GRAMMAR: &str = r#"root ::= "{\"tool\": \"" call "}}"
call ::= ( "set_timer\", \"args\": {\"minutes\": " int
         | "open_app\", \"args\": {\"name\": \"" name "\""
         | "volume\", \"args\": {\"level\": " int )
int  ::= [0-9] | [1-9] [0-9] | [1-9] [0-9] [0-9]
name ::= [a-zA-Z0-9 ._-]+
"#;

/// Ask the model for a tool call, then run it. Returns a short
/// user-facing description of what happened.
pub fn handle(llama: &mut Llama, template: BuiltinTemplate, request: &str) -> Result<String> {
    let messages = [ChatMessage::new(Role::System, SYSTEM_PROMPT), ChatMessage::new(Role::User, request)];
    let prompt = render_builtin(template, &messages, &[], true);
    // SAFETY: llama_sampler_init_grammar parses the grammar during the call.
    let grammar = unsafe { LlamaSampler::grammar(llama.model(), GRAMMAR, "root") }.context("tool grammar rejected")?;
    let greedy = LlamaSampler::greedy().context("greedy sampler")?;
    let mut sampler = LlamaSampler::chain(vec![grammar, greedy], true).context("sampler chain")?;
    let completion = llama.create_completion_with_sampler(&prompt, CompletionOptions::new(48), &mut sampler)?;

    let call: Value = serde_json::from_str(completion.text.trim())
        .with_context(|| format!("tool call didn't parse: {:?}", completion.text))?;
    eprintln!("[tool call: {call}]");
    let args = &call["args"];
    match call["tool"].as_str() {
        Some("volume") => volume(int_arg(args, "level")?),
        Some("set_timer") => set_timer(int_arg(args, "minutes")?),
        Some("open_app") => open_app(args["name"].as_str().context("missing name")?),
        other => bail!("unknown tool {other:?}"),
    }
}

fn int_arg(args: &Value, key: &str) -> Result<u64> {
    args[key].as_u64().with_context(|| format!("missing integer {key}"))
}

fn dry_run() -> bool {
    std::env::var("OJP_TOOLS_DRY_RUN").is_ok_and(|v| v == "1")
}

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

fn volume(level: u64) -> Result<String> {
    let level = level.min(100);
    run("wpctl", &["set-volume", "@DEFAULT_AUDIO_SINK@", &format!("{level}%")])?;
    Ok(format!("Volume set to {level}%."))
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
