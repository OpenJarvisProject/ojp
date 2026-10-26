//! Router: a small always-loaded model classifies each request into one
//! route; the caller then dispatches it (OJP-9). The model can only ever
//! emit one of the route labels: a GBNF grammar restricts sampling to
//! them, so there's no free text to parse and no way to get an invalid
//! label, however small/quantized the router model is.
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use crate::template::Template;
use llama_crab::{ChatMessage, LlamaSampler, Role};

use crate::model::{Model, Seq};

/// Kept identical to model-tooling's agent_tests.py ROUTER_SYSTEM so the
/// constrained in-process router and the unconstrained llama-server
/// baseline are scored on the same prompt (one variable: the grammar).
/// Part of the warm-up cache key (engine.rs): bump when SYSTEM_PROMPT or
/// FEW_SHOT change so stale caches are never reused.
pub const PROMPT_VERSION: u32 = 5;

const SYSTEM_PROMPT: &str = "Classify the user's request into exactly one category and reply with ONLY the category name.\n\
answer_directly: casual chat or a simple fact.\n\
coding: writing, fixing, or explaining code.\n\
reasoning: multi-step problems, planning, analysis, or math word problems.\n\
device_control: doing things on this computer: apps, settings, timers, reminders, volume, music; looking up live facts (time, battery, files, clipboard, notes) or the web; reading, listing or creating files; running commands; exact arithmetic; saving notes; remembering or forgetting facts about the user.";

/// Worked examples shown to the model as prior turns. Disjoint from both
/// the dev and test case sets in main.rs.
const FEW_SHOT: [(&str, Route); 27] = [
    ("Hi there!", Route::AnswerDirectly),
    ("What is the boiling point of water?", Route::AnswerDirectly),
    ("Write a function to check if a number is prime in JavaScript.", Route::Coding),
    ("Why am I getting a NullPointerException in Java?", Route::Coding),
    ("Help me decide between two job offers based on salary, commute, and growth.", Route::Reasoning),
    ("If 3 workers build a wall in 6 hours, how long do 2 workers take?", Route::Reasoning),
    ("Pause the music.", Route::DeviceControl),
    ("Turn on Wi-Fi.", Route::DeviceControl),
    ("Is my battery low?", Route::DeviceControl),
    ("What's 84 divided by 7?", Route::DeviceControl),
    ("Locate my invoices folder.", Route::DeviceControl),
    ("Jot down that the meeting moved to 3pm.", Route::DeviceControl),
    ("What is 56 times 12?", Route::DeviceControl),
    ("Add 1,250 and 3,475.", Route::DeviceControl),
    ("Tell me the time.", Route::DeviceControl),
    ("Which song is on?", Route::DeviceControl),
    ("What's the capital of Canada?", Route::AnswerDirectly),
    ("Who painted the Mona Lisa?", Route::AnswerDirectly),
    ("How do I reverse a string in Python?", Route::Coding),
    ("Why is my JavaScript promise returning undefined?", Route::Coding),
    ("Find my groceries list and read it to me.", Route::DeviceControl),
    ("Search the web for good pasta recipes.", Route::DeviceControl),
    ("Create a file called todo.txt with my tasks in it.", Route::DeviceControl),
    ("List the files in my Downloads folder.", Route::DeviceControl),
    ("Remember that I'm vegetarian.", Route::DeviceControl),
    ("What do you remember about me?", Route::DeviceControl),
    ("Forget what I told you about my old address.", Route::DeviceControl),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    AnswerDirectly,
    Coding,
    Reasoning,
    DeviceControl,
}

impl Route {
    pub const ALL: [Route; 4] = [
        Route::AnswerDirectly,
        Route::Coding,
        Route::Reasoning,
        Route::DeviceControl,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Route::AnswerDirectly => "answer_directly",
            Route::Coding => "coding",
            Route::Reasoning => "reasoning",
            Route::DeviceControl => "device_control",
        }
    }

    fn from_label(label: &str) -> Option<Route> {
        Route::ALL.into_iter().find(|r| r.label() == label)
    }
}

pub struct Decision {
    pub route: Route,
    pub elapsed: Duration,
    /// The router model's hidden state for this request, when a probe was used
    /// (the tool probe reads the same state).
    pub hidden: Option<Vec<f32>>,
}

/// Stateless: classifies with whichever model it is handed, so any loaded
/// model (a dedicated small router, or the main model on a one-model
/// device) can do the routing.
pub struct Router {
    grammar: String,
    template: Template,
    probe: Option<Probe>,
}

/// A linear classifier on the router model's hidden state at the answer
/// position. Fitted offline (scripts/fit-router-probe.py) for one exact model
/// file; on held-out requests it routes far better than the small model's own
/// answer (about 98% against 76-83%) and needs no worked examples in the
/// prompt, so a request costs one short prompt instead of generating a label.
pub struct Probe {
    weights: Vec<Vec<f32>>, // [route][dim], in Route::ALL order
    bias: Vec<f32>,
}

impl Probe {
    pub fn load(path: &std::path::Path) -> Result<Probe> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let labels: Vec<&str> = v["labels"].as_array().context("probe has no labels")?.iter().filter_map(|l| l.as_str()).collect();
        let want: Vec<&str> = Route::ALL.iter().map(|r| r.label()).collect();
        if labels != want {
            bail!("probe labels {labels:?} don't match the routes");
        }
        let weights: Vec<Vec<f32>> = serde_json::from_value(v["weights"].clone())?;
        let bias: Vec<f32> = serde_json::from_value(v["bias"].clone())?;
        if weights.len() != 4 || bias.len() != 4 || weights.iter().any(|w| w.len() != weights[0].len()) {
            bail!("probe has the wrong shape");
        }
        Ok(Probe { weights, bias })
    }

    /// The explicit path if given, else `<router file>.probe.json` when it exists.
    pub fn find(router: &std::path::Path, explicit: Option<&std::path::Path>) -> Option<Probe> {
        let path = match explicit {
            Some(p) => p.to_path_buf(),
            None => {
                let mut name = router.as_os_str().to_owned();
                name.push(".probe.json");
                name.into()
            }
        };
        Probe::load(&path).ok()
    }

    fn classify(&self, hidden: &[f32]) -> Option<Route> {
        if hidden.len() != self.weights[0].len() {
            return None;
        }
        let score = |i: usize| self.bias[i] + self.weights[i].iter().zip(hidden).map(|(w, h)| w * h).sum::<f32>();
        (0..4).max_by(|&a, &b| score(a).total_cmp(&score(b))).map(|i| Route::ALL[i])
    }
}

impl Router {
    pub fn new(template: Template) -> Self {
        let alternatives: Vec<String> = Route::ALL.iter().map(|r| format!("\"{}\"", r.label())).collect();
        let grammar = format!("root ::= {}\n", alternatives.join(" | "));
        Router { grammar, template, probe: None }
    }

    /// The hidden state at the answer position for `request` (probe routers' input).
    pub fn hidden(&self, model: &mut Model, request: &str) -> Result<Vec<f32>> {
        model.last_hidden(Seq::Route, &self.short_prompt(request))
    }

    pub fn has_probe(&self) -> bool {
        self.probe.is_some()
    }

    pub fn with_probe(mut self, probe: Option<Probe>) -> Self {
        self.probe = probe;
        self
    }

    fn short_prompt(&self, request: &str) -> String {
        let messages = [ChatMessage::new(Role::System, SYSTEM_PROMPT), ChatMessage::new(Role::User, request)];
        self.template.render(&messages, true)
    }

    fn build_prompt(&self, request: &str) -> String {
        let mut messages = vec![ChatMessage::new(Role::System, SYSTEM_PROMPT)];
        // Few-shot is the default: on the dev set it lifted 0.5B from 6/24
        // to 16/24 and 1.5B from 17/24 to 19/24. OJP_ROUTER_FEWSHOT=0 turns
        // it off for comparison.
        if !std::env::var("OJP_ROUTER_FEWSHOT").is_ok_and(|v| v == "0") {
            for (example, route) in FEW_SHOT {
                messages.push(ChatMessage::new(Role::User, example));
                messages.push(ChatMessage::new(Role::Assistant, route.label()));
            }
        }
        messages.push(ChatMessage::new(Role::User, request));
        self.template.render(&messages, true)
    }

    /// Raw first-token logits of the four labels, in `Route::ALL` order
    /// (for calibration experiments).
    pub fn scores(&self, model: &mut Model, request: &str) -> Result<Vec<f32>> {
        let labels: Vec<&str> = Route::ALL.iter().map(|r| r.label()).collect();
        model.first_token_logits(Seq::Route, &self.build_prompt(request), &labels)
    }

    /// Hidden state at the answer position, for probe experiments. `short` drops
    /// the few-shot examples.
    pub fn features(&self, model: &mut Model, request: &str, short: bool) -> Result<Vec<f32>> {
        if short {
            return model.last_hidden(Seq::Route, &self.short_prompt(request));
        }
        model.last_hidden(Seq::Route, &self.build_prompt(request))
    }

    pub fn classify(&self, model: &mut Model, request: &str) -> Result<Decision> {
        let start = Instant::now();
        if let Some(probe) = &self.probe
            && let Ok(hidden) = model.last_hidden(Seq::Route, &self.short_prompt(request))
            && let Some(route) = probe.classify(&hidden)
        {
            return Ok(Decision { route, elapsed: start.elapsed(), hidden: Some(hidden) });
        }
        let prompt = self.build_prompt(request);

        // Grammar first, then greedy: greedy picks the most likely token
        // among those the grammar still allows.
        // SAFETY: llama_sampler_init_grammar parses the grammar into its
        // own structures during the call; nothing borrows the strings after.
        let mut stages = Vec::new();
        // OJP_ROUTER_NO_GRAMMAR=1 is a diagnostic: same prompt and greedy
        // decoding, grammar removed, so the model's raw answer is visible.
        let unconstrained = std::env::var("OJP_ROUTER_NO_GRAMMAR").is_ok_and(|v| v == "1");
        if !unconstrained {
            stages.push(
                unsafe { LlamaSampler::grammar(model.llama().model(), &self.grammar, "root") }
                    .context("router grammar rejected")?,
            );
        }
        stages.push(LlamaSampler::greedy().context("greedy sampler")?);
        let mut sampler = LlamaSampler::chain(stages, true).context("sampler chain")?;

        // The system prompt and few-shot turns are identical every time, so
        // after the first request only the user's message is evaluated.
        let text = model.complete(Seq::Route, &prompt, &mut sampler, if unconstrained { 32 } else { 8 }, &mut |_| {})?;
        let label = text.trim();
        if unconstrained {
            eprintln!("raw: {label:?}");
        }
        let Some(route) = Route::from_label(label) else {
            bail!("router produced {label:?}, which the grammar should have made impossible");
        };
        Ok(Decision { route, elapsed: start.elapsed(), hidden: None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_picks_the_highest_scoring_route() {
        let probe = Probe {
            weights: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0], vec![0.0, -1.0]],
            bias: vec![0.0; 4],
        };
        assert_eq!(probe.classify(&[2.0, 0.5]), Some(Route::AnswerDirectly));
        assert_eq!(probe.classify(&[0.1, 3.0]), Some(Route::Coding));
        assert_eq!(probe.classify(&[-4.0, 0.0]), Some(Route::Reasoning));
        assert_eq!(probe.classify(&[0.0, -2.0]), Some(Route::DeviceControl));
        assert_eq!(probe.classify(&[1.0]), None, "wrong hidden size falls back");
    }

    #[test]
    fn probe_file_must_use_the_route_labels() {
        let dir = std::env::temp_dir().join(format!("ojp-probe-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ok = dir.join("ok.json");
        std::fs::write(&ok, r#"{"labels":["answer_directly","coding","reasoning","device_control"],"weights":[[1],[2],[3],[4]],"bias":[0,0,0,0]}"#).unwrap();
        assert!(Probe::load(&ok).is_ok());
        let bad = dir.join("bad.json");
        std::fs::write(&bad, r#"{"labels":["a","b","c","d"],"weights":[[1],[2],[3],[4]],"bias":[0,0,0,0]}"#).unwrap();
        assert!(Probe::load(&bad).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
}
