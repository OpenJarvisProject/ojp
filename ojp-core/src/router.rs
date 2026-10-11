//! Router: a small always-loaded model classifies each request into one
//! route; the caller then dispatches it (OJP-9). The model can only ever
//! emit one of the route labels: a GBNF grammar restricts sampling to
//! them, so there's no free text to parse and no way to get an invalid
//! label, however small/quantized the router model is.
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use llama_crab::chat::template::{BuiltinTemplate, render_builtin};
use llama_crab::{ChatMessage, CompletionOptions, Llama, LlamaSampler, Role};

/// Kept identical to model-tooling's agent_tests.py ROUTER_SYSTEM so the
/// constrained in-process router and the unconstrained llama-server
/// baseline are scored on the same prompt (one variable: the grammar).
const SYSTEM_PROMPT: &str = "Classify the user's request into exactly one category and reply with ONLY the category name.\n\
answer_directly: casual chat or a simple fact.\n\
coding: writing, fixing, or explaining code.\n\
reasoning: multi-step problems, planning, analysis, or math word problems.\n\
device_control: controlling apps, settings, timers, volume, or hardware on this computer.";

/// Worked examples shown to the model as prior turns. Disjoint from both
/// the dev and test case sets in main.rs.
const FEW_SHOT: [(&str, Route); 8] = [
    ("Hi there!", Route::AnswerDirectly),
    ("What is the boiling point of water?", Route::AnswerDirectly),
    ("Write a function to check if a number is prime in JavaScript.", Route::Coding),
    ("Why am I getting a NullPointerException in Java?", Route::Coding),
    ("Help me decide between two job offers based on salary, commute, and growth.", Route::Reasoning),
    ("If 3 workers build a wall in 6 hours, how long do 2 workers take?", Route::Reasoning),
    ("Pause the music.", Route::DeviceControl),
    ("Turn on Wi-Fi.", Route::DeviceControl),
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
}

/// Stateless: classifies with whichever model it is handed, so any loaded
/// model (a dedicated small router, or the main model on a one-model
/// device) can do the routing.
pub struct Router {
    grammar: String,
    template: BuiltinTemplate,
}

impl Router {
    pub fn new(template: BuiltinTemplate) -> Self {
        let alternatives: Vec<String> = Route::ALL.iter().map(|r| format!("\"{}\"", r.label())).collect();
        let grammar = format!("root ::= {}\n", alternatives.join(" | "));
        Router { grammar, template }
    }

    pub fn classify(&self, llama: &mut Llama, request: &str) -> Result<Decision> {
        let start = Instant::now();
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
        let prompt = render_builtin(self.template, &messages, &[], true);

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
                unsafe { LlamaSampler::grammar(llama.model(), &self.grammar, "root") }
                    .context("router grammar rejected")?,
            );
        }
        stages.push(LlamaSampler::greedy().context("greedy sampler")?);
        let mut sampler = LlamaSampler::chain(stages, true).context("sampler chain")?;

        let options = CompletionOptions::new(if unconstrained { 32 } else { 8 });
        let completion = llama.create_completion_with_sampler(&prompt, options, &mut sampler)?;
        let label = completion.text.trim();
        if unconstrained {
            eprintln!("raw: {label:?}");
        }
        let Some(route) = Route::from_label(label) else {
            bail!("router produced {label:?}, which the grammar should have made impossible");
        };
        Ok(Decision { route, elapsed: start.elapsed() })
    }
}
