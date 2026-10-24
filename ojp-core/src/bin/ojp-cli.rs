//! ojp-cli: terminal front end and test harness for ojp-core.
//!   ojp-cli [model.gguf]                     Stage 1/2 smoke test (default)
//!   ojp-cli route <model.gguf> "<request>"   classify one request
//!   ojp-cli route-eval <model.gguf> [--dev]  score the router on fixed cases
//!   ojp-cli tool-eval <model.gguf>           score routing and tool choice on fixed requests
//!   ojp-cli chat [config.toml]               assistant REPL (default: the
//!                                            desktop app's config)
//!   ojp-cli download [config.toml]           fetch missing models
use std::io::Write;

use llama_crab::chat::BuiltinTemplate;
use llama_crab::{Llama, LlamaParams};
use ojp_core::router::Router;
use ojp_core::{Config, Engine, Event, Route};

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn load(model_path: &str) -> anyhow::Result<Llama> {
    // Stage 1 default is 0 (CPU-only, no GPU variable). Stage 2 sets
    // OJP_GPU_LAYERS=99 to offload everything to the GPU backend,
    // same binary, one variable changed, per the validation plan.
    let gpu_layers: i32 = env_or("OJP_GPU_LAYERS", 0);

    // For investigating the fixed compute-buffer VRAM cost (doesn't
    // shrink with pruning - see model-tooling's README). n_ctx default
    // unchanged from the original hardcoded value; n_ubatch, when set,
    // overrides llama.cpp's own default (observed as 512 from
    // sched_reserve's "worst-case: n_tokens = 512" log line).
    let n_ctx: u32 = env_or("OJP_N_CTX", 2048);
    let n_ubatch: Option<u32> = std::env::var("OJP_N_UBATCH").ok().and_then(|v| v.parse().ok());

    println!(
        "Loading model: {model_path} (n_gpu_layers={gpu_layers}, n_ctx={n_ctx}, n_ubatch={n_ubatch:?})"
    );
    let mut params = LlamaParams::new(model_path)
        .with_n_ctx(n_ctx)
        .with_n_gpu_layers(gpu_layers);
    if let Some(n) = n_ubatch {
        params = params.with_n_ubatch(n);
    }
    Ok(Llama::load(params)?)
}

/// Same env knobs as `load`, through ojp-core's prompt-caching Model.
fn load_cached(model_path: &str) -> anyhow::Result<ojp_core::model::Model> {
    ojp_core::model::Model::load(model_path.as_ref(), env_or("OJP_N_CTX", 2048), env_or("OJP_GPU_LAYERS", 0))
}


/// What a tool-eval case checks beyond the tool name.
enum Check {
    None,
    /// The argument `key` contains this text (case-insensitive).
    Arg(&'static str, &'static str),
    /// Running the call (read-only tools only) gives output containing this.
    Output(&'static str),
}

/// Requests that need a tool, with the tool a person would expect. Written
/// once and not tuned against; the numbers it reports are the model's.
const TOOL_CASES: &[(&str, &str, Check)] = &[
    ("Set a timer for 10 minutes.", "set_timer", Check::Arg("minutes", "10")),
    ("Start a 5 minute timer.", "set_timer", Check::Arg("minutes", "5")),
    ("Wake me up in 45 minutes.", "set_timer", Check::Arg("minutes", "45")),
    ("Set the volume to 40.", "volume", Check::Arg("level", "40")),
    ("Turn the volume down to 15 percent.", "volume", Check::Arg("level", "15")),
    ("Make it louder, 80 please.", "volume", Check::Arg("level", "80")),
    ("Mute the sound.", "mute", Check::Arg("state", "on")),
    ("Unmute my speakers.", "mute", Check::Arg("state", "off")),
    ("Pause the music.", "media", Check::Arg("action", "play_pause")),
    ("Skip this song.", "media", Check::Arg("action", "next")),
    ("Go back to the previous track.", "media", Check::Arg("action", "previous")),
    ("Open Firefox.", "open_app", Check::Arg("name", "firefox")),
    ("Launch the terminal.", "open_app", Check::Arg("name", "terminal")),
    ("Start Spotify.", "open_app", Check::Arg("name", "spotify")),
    ("Open github.com in the browser.", "open_url", Check::Arg("url", "github.com")),
    ("Go to https://example.com", "open_url", Check::Arg("url", "example.com")),
    ("Take a note: buy oat milk.", "note_add", Check::Arg("text", "oat milk")),
    ("Remember that the wifi password is on the fridge.", "note_add", Check::Arg("text", "fridge")),
    ("What's 17 times 23?", "calculate", Check::Output("391")),
    ("Calculate 15% of 240.", "calculate", Check::Output("36")),
    ("What is the square root of 144 plus 5?", "calculate", Check::Output("17")),
    ("How much is 1234 divided by 8?", "calculate", Check::Output("154.25")),
    ("What's 2 to the power of 10?", "calculate", Check::Output("1024")),
    ("What time is it?", "get_time", Check::None),
    ("What's today's date?", "get_time", Check::None),
    ("What day is it?", "get_time", Check::None),
    ("How much battery do I have left?", "system_status", Check::None),
    ("How much free disk space do I have?", "system_status", Check::None),
    ("How's my computer doing?", "system_status", Check::None),
    ("How much RAM is free?", "system_status", Check::None),
    ("What song is playing?", "now_playing", Check::None),
    ("What am I listening to right now?", "now_playing", Check::None),
    ("What's on my clipboard?", "clipboard", Check::None),
    ("Read my clipboard.", "clipboard", Check::None),
    ("Find my tax return files.", "find_files", Check::Arg("query", "tax")),
    ("Where is my resume?", "find_files", Check::Arg("query", "resume")),
    ("Search for files named budget.", "find_files", Check::Arg("query", "budget")),
    ("What notes do I have?", "notes_list", Check::None),
    ("Read my notes back to me.", "notes_list", Check::None),
];

/// Written after the prompts above were tuned against `TOOL_CASES`, with
/// different wording, and never used to tune anything: the honest score.
const TOOL_CASES_UNSEEN: &[(&str, &str, Check)] = &[
    ("Give me a 15 minute countdown.", "set_timer", Check::Arg("minutes", "15")),
    ("Put the speaker volume at 55.", "volume", Check::Arg("level", "55")),
    ("Shut the speakers up for now.", "mute", Check::Arg("state", "on")),
    ("Turn the sound back on.", "mute", Check::Arg("state", "off")),
    ("Stop the song for a moment.", "media", Check::Arg("action", "play_pause")),
    ("Next track, please.", "media", Check::Arg("action", "next")),
    ("Bring up the calculator app.", "open_app", Check::Arg("name", "calc")),
    ("Open wikipedia.org for me.", "open_url", Check::Arg("url", "wikipedia")),
    ("Write down: dentist on Friday at 9.", "note_add", Check::Arg("text", "dentist")),
    ("How much is 144 divided by 12?", "calculate", Check::Output("12")),
    ("What's 9 times 9 times 9?", "calculate", Check::Output("729")),
    ("Tell me what time it is now.", "get_time", Check::None),
    ("What's the date today?", "get_time", Check::None),
    ("Is my laptop plugged in and how full is the battery?", "system_status", Check::None),
    ("How much memory is being used?", "system_status", Check::None),
    ("What's playing at the moment?", "now_playing", Check::None),
    ("Show me the text I copied last.", "clipboard", Check::None),
    ("Look for my passport scan.", "find_files", Check::Arg("query", "passport")),
    ("List my saved notes.", "notes_list", Check::None),
    ("Mute it.", "mute", Check::Arg("state", "on")),
];

/// Dev set: for iterating on the router prompt. Disjoint from the test
/// set below, which is only used to report results, never to tune.
const DEV_CASES: [(&str, Route); 24] = [
    ("Good morning!", Route::AnswerDirectly),
    ("Who wrote Romeo and Juliet?", Route::AnswerDirectly),
    ("Thanks, that's all.", Route::AnswerDirectly),
    ("How many legs does a spider have?", Route::AnswerDirectly),
    ("What does 'ephemeral' mean?", Route::AnswerDirectly),
    ("Tell me a joke.", Route::AnswerDirectly),
    ("How do I read a file line by line in Go?", Route::Coding),
    ("What's the difference between a list and a tuple in Python?", Route::Coding),
    ("My C++ program segfaults when I access a vector, why?", Route::Coding),
    ("Write a bash script that renames all .txt files to .md.", Route::Coding),
    ("Explain what this regex does: ^\\d{3}-\\d{4}$", Route::Coding),
    ("Convert this SQL query to use a JOIN instead of a subquery.", Route::Coding),
    ("I have $500 a month for groceries for a family of four; help me budget it.", Route::Reasoning),
    ("Should I learn Spanish or Japanese first if I want to travel a lot?", Route::Reasoning),
    ("A bat and a ball cost $1.10 in total and the bat costs $1 more than the ball. How much is the ball?", Route::Reasoning),
    ("Outline a study plan for passing a calculus exam in two weeks.", Route::Reasoning),
    ("My houseplant's leaves are turning yellow; what are the likely causes and how do I narrow it down?", Route::Reasoning),
    ("Estimate how many piano tuners work in Chicago.", Route::Reasoning),
    ("Set an alarm for 7 AM.", Route::DeviceControl),
    ("Turn on dark mode.", Route::DeviceControl),
    ("Mute my microphone.", Route::DeviceControl),
    ("Lower the screen brightness to 30 percent.", Route::DeviceControl),
    ("Close all Firefox windows.", Route::DeviceControl),
    ("Connect to my Bluetooth headphones.", Route::DeviceControl),
];

/// Test set: same 10 cases as model-tooling's agent_tests.py "router" category.
const ROUTER_CASES: [(&str, Route); 10] = [
    ("Hey, how's it going?", Route::AnswerDirectly),
    ("What's the capital of Canada?", Route::AnswerDirectly),
    ("Write a Python function that reverses a string.", Route::Coding),
    ("Why does my Rust code say 'borrowed value does not live long enough'?", Route::Coding),
    ("Fix the bug in this JavaScript: for (i = 0; i < arr.length; i--) {}", Route::Coding),
    ("Plan a three-day trip to Rome on a tight budget.", Route::Reasoning),
    ("If a shirt costs $20 after a 20% discount, what was the original price? Explain.", Route::Reasoning),
    ("Compare the pros and cons of renting versus buying a house.", Route::Reasoning),
    ("Turn the volume up.", Route::DeviceControl),
    ("Open Spotify and play my liked songs.", Route::DeviceControl),
];

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("route") => {
            let [_, model, request] = args.as_slice() else {
                anyhow::bail!("usage: ojp-core route <model.gguf> \"<request>\"");
            };
            let mut model = load_cached(model)?;
            let d = Router::new(BuiltinTemplate::ChatMl).classify(&mut model, request)?;
            println!("{} ({} ms)", d.route.label(), d.elapsed.as_millis());
        }
        Some("chat") => {
            let (default_config, data_dir) = ojp_core::config::desktop_paths();
            let config_path = args.get(1).map_or(default_config, Into::into);
            let config = Config::load_or_init(&config_path, &data_dir)?;
            chat(&config)?;
        }
        Some("download") => {
            let (default_config, data_dir) = ojp_core::config::desktop_paths();
            let config_path = args.get(1).map_or(default_config, Into::into);
            let config = Config::load_or_init(&config_path, &data_dir)?;
            for item in ojp_core::download::missing(&config) {
                let mut last = 0;
                ojp_core::download::fetch(&item, &mut |done, total| {
                    if done - last >= 50_000_000 || Some(done) == total {
                        last = done;
                        eprintln!("{}: {} / {:?} MB", item.role, done / 1_000_000, total.map(|t| t / 1_000_000));
                    }
                })?;
                println!("{}: {}", item.role, item.dest.display());
            }
        }
        Some("tool-eval") => {
            let [_, model_path] = args.as_slice() else {
                anyhow::bail!("usage: ojp-core tool-eval <model.gguf>");
            };
            // Actions are only printed, never run.
            unsafe { std::env::set_var("OJP_TOOLS_DRY_RUN", "1") };
            let mut model = load_cached(model_path)?;
            let router = Router::new(BuiltinTemplate::ChatMl);
            for (set_name, cases) in [("SEEN (tuned against)", TOOL_CASES), ("UNSEEN (written after tuning)", TOOL_CASES_UNSEEN)] {
            println!("\n=== {set_name} ===");
            let (mut routed, mut chose, mut args_ok, mut full, mut total_ms) = (0, 0, 0, 0, 0u128);
            let mut by_tool: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
            for (request, tool, check) in cases {
                let start = std::time::Instant::now();
                let route_ok = router.classify(&mut model, request).map(|d| d.route == Route::DeviceControl).unwrap_or(false);
                let call = ojp_core::tools::ask(&mut model, BuiltinTemplate::ChatMl, request);
                total_ms += start.elapsed().as_millis();
                let (tool_ok, detail) = match &call {
                    Err(e) => (false, format!("{e:#}")),
                    Ok(call) => {
                        let name_ok = call["tool"] == *tool;
                        let check_ok = name_ok
                            && match check {
                                Check::None => true,
                                Check::Arg(key, want) => {
                                    call["args"][key].to_string().to_lowercase().contains(&want.to_lowercase())
                                }
                                Check::Output(want) => ojp_core::tools::run_call(call).is_ok_and(|o| o.text.contains(want)),
                            };
                        (check_ok, call.to_string())
                    }
                };
                routed += route_ok as usize;
                chose += call.as_ref().is_ok_and(|c| c["tool"] == *tool) as usize;
                args_ok += tool_ok as usize;
                full += (route_ok && tool_ok) as usize;
                let e = by_tool.entry(tool).or_default();
                e.0 += (route_ok && tool_ok) as usize;
                e.1 += 1;
                println!("{} route:{} {request}  ->  {detail}", if route_ok && tool_ok { "PASS" } else { "FAIL" }, if route_ok { "ok " } else { "NO " });
            }
            let n = cases.len();
            println!("\nrouted to device_control: {routed}/{n}   right tool: {chose}/{n}   right tool and arguments: {args_ok}/{n}");
            println!("full pipeline (routed AND right tool/args): {full}/{n}, mean {} ms per request", total_ms / n as u128);
            for (tool, (ok, of)) in by_tool {
                println!("  {tool:<14} {ok}/{of}");
            }
            }
        }
        Some("route-eval") => {
            let (model, cases): (&String, &[(&str, Route)]) = match args.as_slice() {
                [_, model] => (model, &ROUTER_CASES),
                [_, model, flag] if flag == "--dev" => (model, &DEV_CASES),
                _ => anyhow::bail!("usage: ojp-core route-eval <model.gguf> [--dev]"),
            };
            let mut model = load_cached(model)?;
            let router = Router::new(BuiltinTemplate::ChatMl);
            let (mut passed, mut total_ms) = (0, 0);
            for &(request, expected) in cases {
                let d = match router.classify(&mut model, request) {
                    Ok(d) => d,
                    Err(e) => {
                        println!("FAIL  {e:#}  (want {}) {request}", expected.label());
                        continue;
                    }
                };
                let ok = d.route == expected;
                passed += ok as usize;
                total_ms += d.elapsed.as_millis();
                println!(
                    "{} {:>5} ms  {:<15} (want {:<15}) {request}",
                    if ok { "PASS" } else { "FAIL" },
                    d.elapsed.as_millis(),
                    d.route.label(),
                    expected.label()
                );
            }
            println!("router: {passed}/{} correct, mean {} ms", cases.len(), total_ms / cases.len() as u128);
        }
        model => {
            let model_path = model.unwrap_or("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
            let mut llama = load(model_path)?;
            println!("Model loaded. Running completion...");
            let response = llama.create_completion("The capital of France is", 32)?;
            println!("--- completion ---\n{}\n------------------", response.text);
        }
    }
    Ok(())
}

fn chat(config: &Config) -> anyhow::Result<()> {
    let mut engine = Engine::new(config)?;
    for line in &engine.status {
        eprintln!("{line}");
    }
    eprintln!("{} is ready. Empty line or Ctrl-D quits.", config.persona.name);
    let stdin = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 || line.trim().is_empty() {
            return Ok(());
        }
        engine.ask(line.trim(), &mut |event| match event {
            Event::Routed { route, ms } => eprintln!("[route: {} in {ms} ms]", route.label()),
            Event::Token(text) => {
                print!("{text}");
                let _ = std::io::stdout().flush();
            }
            Event::Notice(text) => eprintln!("[{text}]"),
        })?;
        println!();
    }
}
