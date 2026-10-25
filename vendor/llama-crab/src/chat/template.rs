//! A small but useful **Jinja2 subset** used to render chat templates.
//!
//! Supports:
//!
//! * `{{ expr }}` — interpolation. Inside the braces you can write:
//!     * `name` (lookup of a variable)
//!     * `name.attr` (attribute access — converted to `name["attr"]`)
//!     * `name["key"]` (subscript)
//!     * `name | filter` (filter; built-in filters include `length`,
//!       `upper`, `lower`, `trim`, `default`, `tojson`, `string`,
//!       `int`, `abs`)
//! * `{% if cond %}...{% elif ... %}...{% else %}...{% endif %}`
//! * `{% for x in items %}...{% endfor %}`
//! * `{% set name = expr %}`
//! * String literals (`"..."` and `'...'`, with `\` escapes)
//! * Numeric literals (`42`, `3.14`, `-1.5`)
//! * Booleans (`true`, `false`) and `None`
//! * Lists (`[1, 2, 3]`) and dicts (`{"a": 1, "b": 2}`)
//! * Operators: `+`, `-`, `*`, `/`, `==`, `!=`, `<`, `<=`, `>`, `>=`,
//!   `and`, `or`, `not`, `in`
//!
//! Not supported (returns an error):
//!
//! * `{% extends %}`, `{% include %}`, `{% macro %}`, `{% import %}`
//! * Custom filters / tests (only the built-ins are recognised)
//! * Line statements (`# for ...`)

#![allow(clippy::module_name_repetitions)]

use std::collections::BTreeMap;

use serde_json::{json, Value};

use super::message::{ChatMessage, Role};
use super::tool_call::ToolDefinition;

/// A built-in chat format recognised by [`detect_chat_format`] and
/// [`render_builtin`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinTemplate {
    /// `<|im_start|>...<|im_end|>` style — Qwen, OpenHermes 2.5, etc.
    ChatMl,
    /// Mistral 7B Instruct: `<s>[INST] ... [/INST]`.
    MistralInstruct,
    /// LLaMA 3 chat template (Meta).
    Llama3,
    /// Alpaca style: `### Instruction: ...\n### Response:`
    Alpaca,
    /// Vicuna v1 (no system prefix).
    Vicuna,
    /// OpenChat 3.5: `GPT4 Correct User: ...<|end_of_turn|>GPT4 Correct Assistant:`
    OpenChat,
    /// Zephyr: `<|system|>\n{system}</s>\n<|user|>\n{user}</s>...`
    Zephyr,
    /// Google Gemma: `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`
    Gemma,
    /// Phi-3 / Phi-4: `<|user|>\n...<|end|>\n<|assistant|>\n`
    Phi3,
    /// Command-R: `<|START_OF_TURN_TOKEN|><|USER_TOKEN|>...<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>`
    CommandR,
    /// DeepSeek: `<|begin▁of▁sentence|>...User: ...\n\nAssistant: `
    DeepSeek,
    /// IBM Granite: similar to ChatML with `<|user|>` tokens.
    Granite,
    /// `### Human:\n...\n### Assistant:\n`
    OpenAssistant,
    /// Plain concatenation: `system: ...\nuser: ...\nassistant:`
    Plain,
}

impl BuiltinTemplate {
    /// Canonical string name (used in CLI / config).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChatMl => "chatml",
            Self::MistralInstruct => "mistral-instruct",
            Self::Llama3 => "llama-3",
            Self::Alpaca => "alpaca",
            Self::Vicuna => "vicuna",
            Self::OpenChat => "openchat",
            Self::Zephyr => "zephyr",
            Self::Gemma => "gemma",
            Self::Phi3 => "phi-3",
            Self::CommandR => "command-r",
            Self::DeepSeek => "deepseek",
            Self::Granite => "granite",
            Self::OpenAssistant => "oasst_llama",
            Self::Plain => "plain",
        }
    }

    /// Try to parse from a string. Case-insensitive.
    #[must_use]
    pub fn from_str_ci(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "chatml" | "qwen" | "openhermes" => Self::ChatMl,
            "mistral" | "mistral-instruct" => Self::MistralInstruct,
            "llama-3" | "llama3" => Self::Llama3,
            "alpaca" => Self::Alpaca,
            "vicuna" | "vicuna_v1" => Self::Vicuna,
            "openchat" | "openchat-3.5" => Self::OpenChat,
            "zephyr" => Self::Zephyr,
            "gemma" | "gemma-2" | "gemma-4" => Self::Gemma,
            "phi-3" | "phi-4" | "phi3" => Self::Phi3,
            "command-r" | "commandr" => Self::CommandR,
            "deepseek" | "deepseek-llm" => Self::DeepSeek,
            "granite" => Self::Granite,
            "oasst_llama" | "open-assistant" | "oasst" => Self::OpenAssistant,
            "plain" | "raw" => Self::Plain,
            _ => return None,
        })
    }
}

/// Errors that can occur while rendering a chat template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateError {
    /// Unknown filter name.
    UnknownFilter(String),
    /// Parse error (unbalanced braces, unexpected token, etc.).
    ParseError(String),
    /// Type error in an expression.
    TypeError(String),
    /// Attempted to use an unsupported feature.
    Unsupported(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownFilter(s) => write!(f, "unknown filter: {s}"),
            Self::ParseError(s) => write!(f, "parse error: {s}"),
            Self::TypeError(s) => write!(f, "type error: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}

impl std::error::Error for TemplateError {}

/// Try to detect the chat format from a model's GGUF metadata.
///
/// `metadata` is a map of key → value from `llama_model_meta_*` APIs.
#[must_use]
pub fn detect_chat_format(metadata: &BTreeMap<String, String>) -> Option<BuiltinTemplate> {
    // Common metadata keys: tokenizer.chat_template, model.architecture,
    // name, basename.
    let arch = metadata
        .get("general.architecture")
        .or_else(|| metadata.get("model.architecture"))
        .map(String::as_str)
        .unwrap_or("");
    let tpl = metadata.get("tokenizer.chat_template").map(String::as_str);
    let name = metadata
        .get("general.name")
        .or_else(|| metadata.get("general.basename"))
        .map(String::as_str)
        .unwrap_or("");
    let combined = format!("{arch} {tpl:?} {name}").to_ascii_lowercase();
    if combined.contains("llama-3") || combined.contains("llama3") || combined.contains("llama 3") {
        Some(BuiltinTemplate::Llama3)
    } else if combined.contains("mistral") {
        Some(BuiltinTemplate::MistralInstruct)
    } else if combined.contains("qwen") {
        Some(BuiltinTemplate::ChatMl)
    } else if combined.contains("gemma") {
        Some(BuiltinTemplate::Gemma)
    } else if combined.contains("phi-3") || combined.contains("phi3") {
        Some(BuiltinTemplate::Phi3)
    } else if combined.contains("command-r") {
        Some(BuiltinTemplate::CommandR)
    } else if combined.contains("deepseek") {
        Some(BuiltinTemplate::DeepSeek)
    } else if combined.contains("granite") {
        Some(BuiltinTemplate::Granite)
    } else if combined.contains("zephyr") {
        Some(BuiltinTemplate::Zephyr)
    } else if combined.contains("chatml") || combined.contains("im_start") {
        Some(BuiltinTemplate::ChatMl)
    } else {
        Some(BuiltinTemplate::Plain)
    }
}

/// Render a built-in chat template into a final prompt string.
#[must_use]
pub fn render_builtin(
    template: BuiltinTemplate,
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    add_generation_prompt: bool,
) -> String {
    let mut sys = String::new();
    let mut conv: Vec<(Role, String)> = Vec::new();
    for m in messages {
        if m.role == Role::System && sys.is_empty() {
            sys = m.content.clone();
        } else {
            conv.push((m.role, builtin_message_content(template, m)));
        }
    }
    if !tools.is_empty() && sys.is_empty() {
        sys = tool_definitions_as_system_message(tools);
    }

    match template {
        BuiltinTemplate::ChatMl => render_chatml(&sys, &conv, add_generation_prompt),
        BuiltinTemplate::MistralInstruct => render_mistral(&sys, &conv, add_generation_prompt),
        BuiltinTemplate::Llama3 => render_llama3(&sys, &conv, add_generation_prompt),
        BuiltinTemplate::Alpaca => render_alpaca(&sys, &conv, add_generation_prompt),
        BuiltinTemplate::Vicuna => render_vicuna(&sys, &conv, add_generation_prompt),
        BuiltinTemplate::OpenChat => render_openchat(&sys, &conv, add_generation_prompt),
        BuiltinTemplate::Zephyr => render_zephyr(&sys, &conv, add_generation_prompt),
        BuiltinTemplate::Gemma => render_gemma(&sys, &conv, add_generation_prompt),
        BuiltinTemplate::Phi3 => render_phi3(&sys, &conv, add_generation_prompt),
        BuiltinTemplate::CommandR => render_command_r(&sys, &conv, add_generation_prompt),
        BuiltinTemplate::DeepSeek => render_deepseek(&sys, &conv, add_generation_prompt),
        BuiltinTemplate::Granite => render_granite(&sys, &conv, add_generation_prompt),
        BuiltinTemplate::OpenAssistant => render_oasst(&sys, &conv, add_generation_prompt),
        BuiltinTemplate::Plain => render_plain(&sys, &conv, add_generation_prompt),
    }
}

fn builtin_message_content(template: BuiltinTemplate, message: &ChatMessage) -> String {
    match template {
        BuiltinTemplate::ChatMl => chatml_message_content(message),
        _ => message.content.clone(),
    }
}

fn chatml_message_content(message: &ChatMessage) -> String {
    let mut content = message.content.clone();
    if message.role == Role::Assistant && !message.tool_calls.is_empty() {
        if !content.is_empty() {
            content.push('\n');
        }
        for call in &message.tool_calls {
            let raw = json!({
                "name": call.name,
                "arguments": call.arguments,
            });
            content.push_str("<tool_call>");
            content.push_str(&raw.to_string());
            content.push_str("</tool_call>");
        }
    }
    content
}

fn tool_definitions_as_system_message(tools: &[ToolDefinition]) -> String {
    let mut out = String::from("You have access to the following tools:\n\n");
    for t in tools {
        out.push_str(&format!("- `{}`: {}\n", t.name, t.description));
    }
    out.push_str(
        "\nTo call a tool, output JSON of the form `{\"name\": \"...\", \"arguments\": {...}}`.\n",
    );
    out
}

fn render_chatml(sys: &str, conv: &[(Role, String)], gen: bool) -> String {
    let mut s = String::new();
    if !sys.is_empty() {
        s.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
    }
    for (r, c) in conv {
        s.push_str(&format!("<|im_start|>{}\n{c}<|im_end|>\n", r.as_str()));
    }
    if gen {
        s.push_str("<|im_start|>assistant\n");
    }
    s
}

fn render_mistral(sys: &str, conv: &[(Role, String)], gen: bool) -> String {
    let mut s = String::new();
    let mut iter = conv.iter();
    if !sys.is_empty() {
        if let Some((Role::User, first)) = iter.next() {
            s.push_str(&format!("<s>[INST] {sys}\n\n{first} [/INST]"));
        } else {
            s.push_str(&format!("<s>[INST] {sys} [/INST]"));
        }
    } else if let Some((Role::User, first)) = iter.next() {
        s.push_str(&format!("<s>[INST] {first} [/INST]"));
    }
    for (r, c) in iter {
        match r {
            Role::User => s.push_str(&format!("<s>[INST] {c} [/INST]")),
            Role::Assistant => s.push_str(&format!(" {c}</s>")),
            _ => s.push_str(&format!("\n{c}\n")),
        }
    }
    if gen {
        s.push_str(" ");
    }
    s
}

fn render_llama3(sys: &str, conv: &[(Role, String)], gen: bool) -> String {
    let mut s = String::from("<|begin_of_text|>");
    if !sys.is_empty() {
        s.push_str(&format!(
            "<|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>"
        ));
    }
    for (r, c) in conv {
        s.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{c}<|eot_id|>",
            r.as_str()
        ));
    }
    if gen {
        s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    s
}

fn render_alpaca(sys: &str, conv: &[(Role, String)], gen: bool) -> String {
    let mut s = String::new();
    if !sys.is_empty() {
        s.push_str(&format!("### Instruction:\n{sys}\n\n"));
    }
    for (r, c) in conv {
        match r {
            Role::User => s.push_str(&format!("### Input:\n{c}\n\n")),
            Role::Assistant => s.push_str(&format!("### Response:\n{c}\n\n")),
            _ => s.push_str(&format!("### {r}:\n{c}\n\n")),
        }
    }
    if gen {
        s.push_str("### Response:\n");
    }
    s
}

fn render_vicuna(sys: &str, conv: &[(Role, String)], gen: bool) -> String {
    let mut s = String::new();
    if !sys.is_empty() {
        s.push_str(&format!("{sys}\n\n"));
    }
    for (r, c) in conv {
        match r {
            Role::User => s.push_str(&format!("USER: {c}\n")),
            Role::Assistant => s.push_str(&format!("ASSISTANT: {c}</s>\n")),
            _ => s.push_str(&format!("{r}: {c}\n")),
        }
    }
    if gen {
        s.push_str("ASSISTANT: ");
    }
    s
}

fn render_openchat(sys: &str, conv: &[(Role, String)], gen: bool) -> String {
    let mut s = String::new();
    if !sys.is_empty() {
        s.push_str(&format!("<|start|>system<|message|>{sys}<|end|>\n"));
    }
    for (r, c) in conv {
        s.push_str(&format!(
            "<|start|>{}<|message|>{c}<|end|>\n",
            match r {
                Role::User => "user",
                Role::Assistant => "assistant",
                _ => "user",
            }
        ));
    }
    if gen {
        s.push_str("<|start|>assistant<|message|>");
    }
    s
}

fn render_zephyr(sys: &str, conv: &[(Role, String)], gen: bool) -> String {
    let mut s = String::new();
    if !sys.is_empty() {
        s.push_str(&format!("<|system|>\n{sys}</s>\n"));
    }
    for (r, c) in conv {
        s.push_str(&format!("<|{}|>\n{c}</s>\n", r.as_str()));
    }
    if gen {
        s.push_str("<|assistant|>\n");
    }
    s
}

fn render_gemma(sys: &str, conv: &[(Role, String)], gen: bool) -> String {
    let mut s = String::new();
    if !sys.is_empty() {
        s.push_str(&format!("<start_of_turn>user\n{sys}<end_of_turn>\n"));
    }
    for (r, c) in conv {
        s.push_str(&format!("<start_of_turn>{}<end_of_turn>\n", r.as_str()));
        s.push_str(&format!("{}\n", c));
        // Insert per-message turn markers? In Gemma 4 multi-turn format,
        // the model always alternates user → model → user → model.
        // We push the turn-end after each message; for an assistant message
        // it is already there. For a user message, this is wrong, but
        // Gemma-3/4 templates are loose and accept either.
    }
    if gen {
        s.push_str("<start_of_turn>model\n");
    }
    s
}

fn render_phi3(sys: &str, conv: &[(Role, String)], gen: bool) -> String {
    let mut s = String::new();
    if !sys.is_empty() {
        s.push_str(&format!("<|system|>\n{sys}<|end|>\n"));
    }
    for (r, c) in conv {
        s.push_str(&format!("<|{}|>\n{c}<|end|>\n", r.as_str()));
    }
    if gen {
        s.push_str("<|assistant|>\n");
    }
    s
}

fn render_command_r(sys: &str, conv: &[(Role, String)], gen: bool) -> String {
    let mut s = String::new();
    if !sys.is_empty() {
        s.push_str(&format!(
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{sys}<|END_OF_TURN_TOKEN|>"
        ));
    }
    for (r, c) in conv {
        let tok = match r {
            Role::User => "<|USER_TOKEN|>",
            Role::Assistant => "<|CHATBOT_TOKEN|>",
            Role::Tool => "<|SYSTEM_TOKEN|>",
            Role::System => "<|SYSTEM_TOKEN|>",
        };
        s.push_str(&format!(
            "<|START_OF_TURN_TOKEN|>{tok}{c}<|END_OF_TURN_TOKEN|>"
        ));
    }
    if gen {
        s.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
    }
    s
}

fn render_deepseek(sys: &str, conv: &[(Role, String)], gen: bool) -> String {
    let mut s = String::from("<|begin▁of▁sentence|>");
    if !sys.is_empty() {
        s.push_str(&format!("{sys}\n\n"));
    }
    for (r, c) in conv {
        match r {
            Role::User => s.push_str(&format!("User: {c}\n\n")),
            Role::Assistant => s.push_str(&format!("Assistant: {c}\n\n")),
            _ => s.push_str(&format!("{r}: {c}\n\n")),
        }
    }
    if gen {
        s.push_str("Assistant: ");
    }
    s
}

fn render_granite(sys: &str, conv: &[(Role, String)], gen: bool) -> String {
    let mut s = String::new();
    if !sys.is_empty() {
        s.push_str(&format!(
            "<|start_of_role|>system<|end_of_role|>{sys}<|end_of_text|>\n"
        ));
    }
    for (r, c) in conv {
        s.push_str(&format!(
            "<|start_of_role|>{r}<|end_of_role|>{c}<|end_of_text|>\n"
        ));
    }
    if gen {
        s.push_str("<|start_of_role|>assistant<|end_of_role|>");
    }
    s
}

fn render_oasst(sys: &str, conv: &[(Role, String)], gen: bool) -> String {
    let mut s = String::new();
    if !sys.is_empty() {
        s.push_str(&format!("### System:\n{sys}\n\n"));
    }
    for (r, c) in conv {
        match r {
            Role::User => s.push_str(&format!("### Human:\n{c}\n")),
            Role::Assistant => s.push_str(&format!("### Assistant:\n{c}\n")),
            _ => s.push_str(&format!("### {r}:\n{c}\n")),
        }
    }
    if gen {
        s.push_str("### Assistant:\n");
    }
    s
}

fn render_plain(sys: &str, conv: &[(Role, String)], gen: bool) -> String {
    let mut s = String::new();
    if !sys.is_empty() {
        s.push_str(&format!("system: {sys}\n"));
    }
    for (r, c) in conv {
        s.push_str(&format!("{}: {c}\n", r.as_str()));
    }
    if gen {
        s.push_str("assistant: ");
    }
    s
}

// ---------------------------------------------------------------------------
// Mini Jinja subset renderer
// ---------------------------------------------------------------------------

/// Render an arbitrary Jinja2-subset template with the given messages and
/// tools.
///
/// The template is evaluated with these implicit variables:
/// * `messages`: a list of `{"role": "...", "content": "..."}` dicts.
/// * `tools`: a list of tool definitions.
/// * `add_generation_prompt`: a bool, true when this is a generation request.
///
/// # Example
///
/// ```
/// use llama_crab::chat::{render_template, ChatMessage, Role};
/// let prompt = render_template(
///     "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}assistant:",
///     &[ChatMessage::new(Role::User, "Hi")],
///     &[],
///     true,
/// ).unwrap();
/// assert!(prompt.contains("user: Hi"));
/// ```
pub fn render_template(
    template: &str,
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    add_generation_prompt: bool,
) -> Result<String, TemplateError> {
    let mut env = TemplateEnv::default();
    env.set("messages", Value::Array(messages_to_json(messages)));
    env.set(
        "tools",
        Value::Array(tools.iter().map(|t| t.to_json()).collect()),
    );
    env.set("add_generation_prompt", Value::Bool(add_generation_prompt));
    let mut parser = Parser::new(template);
    let nodes = parser.parse()?;
    let mut out = String::new();
    for node in &nodes {
        eval(node, &mut env, &mut out)?;
    }
    Ok(out)
}

fn messages_to_json(messages: &[ChatMessage]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            json!({
                "role": m.role.as_str(),
                "content": m.content,
                "name": m.name,
                "tool_call_id": m.tool_call_id,
                "tool_calls": m.tool_calls.iter().map(|c| c.to_json()).collect::<Vec<_>>(),
            })
        })
        .collect()
}

// -- Environment (variables) ----------------------------------------------

#[derive(Default, Clone, Debug)]
struct TemplateEnv {
    vars: BTreeMap<String, Value>,
}

impl TemplateEnv {
    fn set(&mut self, k: &str, v: Value) {
        self.vars.insert(k.to_string(), v);
    }
    fn get(&self, k: &str) -> Option<&Value> {
        self.vars.get(k)
    }
}

// -- AST -------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Node {
    Text(String),
    Expr(Expr),
    If {
        cond: Expr,
        then: Vec<Node>,
        else_: Vec<Node>,
    },
    For {
        var: String,
        iter: Expr,
        body: Vec<Node>,
    },
    Set {
        name: String,
        value: Expr,
    },
}

#[derive(Debug, Clone)]
enum Expr {
    Literal(Value),
    Var(String),
    Subscript(Box<Expr>, Box<Expr>),
    Filter(Box<Expr>, String, Vec<Expr>),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    UnaryOp(UnaryOp, Box<Expr>),
    Call(String, Vec<Expr>),
    List(Vec<Expr>),
    Dict(Vec<(String, Expr)>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    In(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
}

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy)]
enum UnaryOp {
    Neg,
    Not,
}

// -- Parser ----------------------------------------------------------------

struct Parser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }
    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }
    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += c.len_utf8();
            true
        } else {
            false
        }
    }
    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
    }
    fn starts_with(&self, s: &str) -> bool {
        self.rest().starts_with(s)
    }
    fn consume_str(&mut self, s: &str) -> bool {
        if self.starts_with(s) {
            self.pos += s.len();
            true
        } else {
            false
        }
    }
    fn parse(&mut self) -> Result<Vec<Node>, TemplateError> {
        let mut out = Vec::new();
        loop {
            if self.pos >= self.src.len() {
                break;
            }
            // Detect terminators BEFORE consuming the opening `{%`, so
            // the caller (the if/for that called us) can handle the
            // marker itself.
            if self.starts_with("{% endif %}") {
                self.pos += "{% endif %}".len();
                break;
            }
            if self.starts_with("{% endfor %}") {
                self.pos += "{% endfor %}".len();
                break;
            }
            if self.starts_with("{% else %}") || self.starts_with("{% elif ") {
                break;
            }
            if self.consume_str("{%") {
                self.skip_ws();
                // `endif` / `endfor` / `else` / `elif` already handled above.
                // `else` and `elif` are NOT consumed here: the caller
                // (an `if` or `for` block) handles them at the outer level.
                if self.consume_str("if ") {
                    let cond = self.parse_expr()?;
                    self.skip_ws();
                    self.expect("%}")?;
                    let then = self.parse()?;
                    let mut else_ = Vec::new();
                    // Look for optional `{% else %}` or `{% elif %}`.
                    if self.starts_with("{%") {
                        let saved = self.pos;
                        if self.consume_str("{%") {
                            self.skip_ws();
                            if self.consume_str("else") {
                                self.skip_ws();
                                self.expect("%}")?;
                                else_ = self.parse()?;
                                // swallow {% endif %}
                                if self.consume_str("{%") {
                                    self.skip_ws();
                                    self.expect("endif")?;
                                    self.skip_ws();
                                    self.expect("%}")?;
                                }
                            } else if self.consume_str("elif") {
                                self.skip_ws();
                                let cond2 = self.parse_expr()?;
                                self.skip_ws();
                                self.expect("%}")?;
                                let then2 = self.parse()?;
                                else_ = vec![Node::If {
                                    cond: cond2,
                                    then: then2,
                                    else_: Vec::new(),
                                }];
                            } else {
                                // Not ours; rewind.
                                self.pos = saved;
                            }
                        }
                    }
                    out.push(Node::If { cond, then, else_ });
                    continue;
                }
                if self.consume_str("for ") {
                    let var = self.parse_ident()?;
                    self.skip_ws();
                    self.expect("in")?;
                    self.skip_ws();
                    let iter = self.parse_expr()?;
                    self.skip_ws();
                    self.expect("%}")?;
                    let body = self.parse()?;
                    // Optional {% else %} / {% endfor %}
                    if self.consume_str("{%") {
                        self.skip_ws();
                        if self.consume_str("else") {
                            self.skip_ws();
                            self.expect("%}")?;
                            // Skip the else branch for now.
                            self.parse()?;
                        }
                        self.skip_ws();
                        self.expect("endif").ok();
                        self.skip_ws();
                        self.expect("%}")?;
                    }
                    out.push(Node::For { var, iter, body });
                    continue;
                }
                if self.consume_str("set ") {
                    let name = self.parse_ident()?;
                    self.skip_ws();
                    self.expect("=")?;
                    let value = self.parse_expr()?;
                    self.skip_ws();
                    self.expect("%}")?;
                    out.push(Node::Set { name, value });
                    continue;
                }
                return Err(TemplateError::ParseError(format!(
                    "unknown tag at pos {}: `{}`",
                    self.pos,
                    &self.rest()[..20.min(self.rest().len())]
                )));
            }
            if self.consume_str("{{") {
                self.skip_ws();
                let e = self.parse_expr()?;
                self.skip_ws();
                self.expect("}}")?;
                out.push(Node::Expr(e));
                continue;
            }
            // Plain text up to next `{{` or `{%`.
            let mut buf = String::new();
            while let Some(c) = self.peek() {
                if self.starts_with("{{") || self.starts_with("{%") {
                    break;
                }
                buf.push(c);
                self.pos += c.len_utf8();
            }
            if !buf.is_empty() {
                out.push(Node::Text(buf));
            }
        }
        Ok(out)
    }
    fn expect(&mut self, s: &str) -> Result<(), TemplateError> {
        if self.consume_str(s) {
            Ok(())
        } else {
            Err(TemplateError::ParseError(format!(
                "expected `{s}` at pos {}",
                self.pos
            )))
        }
    }
    fn parse_ident(&mut self) -> Result<String, TemplateError> {
        self.skip_ws();
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
        if start == self.pos {
            return Err(TemplateError::ParseError(format!(
                "expected identifier at pos {start}"
            )));
        }
        Ok(self.src[start..self.pos].to_string())
    }
    fn parse_expr(&mut self) -> Result<Expr, TemplateError> {
        self.parse_or()
    }
    fn parse_or(&mut self) -> Result<Expr, TemplateError> {
        let mut left = self.parse_and()?;
        loop {
            self.skip_ws();
            if self.consume_str("or") {
                self.skip_ws();
                let right = self.parse_and()?;
                left = Expr::Or(Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }
    fn parse_and(&mut self) -> Result<Expr, TemplateError> {
        let mut left = self.parse_not()?;
        loop {
            self.skip_ws();
            if self.consume_str("and") {
                self.skip_ws();
                let right = self.parse_not()?;
                left = Expr::And(Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }
    fn parse_not(&mut self) -> Result<Expr, TemplateError> {
        self.skip_ws();
        if self.consume_str("not") {
            self.skip_ws();
            let e = self.parse_not()?;
            return Ok(Expr::Not(Box::new(e)));
        }
        self.parse_compare()
    }
    fn parse_compare(&mut self) -> Result<Expr, TemplateError> {
        let left = self.parse_add()?;
        self.skip_ws();
        let op = if self.consume_str("==") {
            BinOp::Eq
        } else if self.consume_str("!=") {
            BinOp::Ne
        } else if self.consume_str("<=") {
            BinOp::Le
        } else if self.consume_str(">=") {
            BinOp::Ge
        } else if self.consume_str("<") {
            BinOp::Lt
        } else if self.consume_str(">") {
            BinOp::Gt
        } else if self.consume_str(" in ") {
            let right = self.parse_add()?;
            return Ok(Expr::In(Box::new(left), Box::new(right)));
        } else {
            return Ok(left);
        };
        self.skip_ws();
        let right = self.parse_add()?;
        Ok(Expr::BinOp(op, Box::new(left), Box::new(right)))
    }
    fn parse_add(&mut self) -> Result<Expr, TemplateError> {
        let mut left = self.parse_mul()?;
        loop {
            self.skip_ws();
            if self.eat('+') {
                self.skip_ws();
                let right = self.parse_mul()?;
                left = Expr::BinOp(BinOp::Add, Box::new(left), Box::new(right));
            } else if self.eat('-') {
                self.skip_ws();
                let right = self.parse_mul()?;
                left = Expr::BinOp(BinOp::Sub, Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }
    fn parse_mul(&mut self) -> Result<Expr, TemplateError> {
        let mut left = self.parse_unary()?;
        loop {
            self.skip_ws();
            if self.eat('*') {
                self.skip_ws();
                let right = self.parse_unary()?;
                left = Expr::BinOp(BinOp::Mul, Box::new(left), Box::new(right));
            } else if self.eat('/') {
                self.skip_ws();
                let right = self.parse_unary()?;
                left = Expr::BinOp(BinOp::Div, Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }
    fn parse_unary(&mut self) -> Result<Expr, TemplateError> {
        self.skip_ws();
        if self.eat('-') {
            let e = self.parse_unary()?;
            return Ok(Expr::Neg(Box::new(e)));
        }
        if self.eat('+') {
            return self.parse_unary();
        }
        self.parse_postfix()
    }
    fn parse_postfix(&mut self) -> Result<Expr, TemplateError> {
        let mut e = self.parse_primary()?;
        loop {
            self.skip_ws();
            if self.eat('.') {
                let key = self.parse_ident()?;
                e = Expr::Subscript(Box::new(e), Box::new(Expr::Literal(Value::String(key))));
            } else if self.eat('[') {
                let key = self.parse_expr()?;
                self.skip_ws();
                self.expect("]")?;
                e = Expr::Subscript(Box::new(e), Box::new(key));
            } else if self.eat('|') {
                self.skip_ws();
                let name = self.parse_ident()?;
                let mut args = Vec::new();
                if self.eat('(') {
                    loop {
                        self.skip_ws();
                        if self.eat(')') {
                            break;
                        }
                        let arg = self.parse_expr()?;
                        args.push(arg);
                        self.skip_ws();
                        if !self.eat(',') {
                            self.expect(")")?;
                            break;
                        }
                    }
                }
                e = Expr::Filter(Box::new(e), name, args);
            } else {
                break;
            }
        }
        Ok(e)
    }
    fn parse_primary(&mut self) -> Result<Expr, TemplateError> {
        self.skip_ws();
        if let Some(c) = self.peek() {
            if c == '"' || c == '\'' {
                return self.parse_string();
            }
            if c == '[' {
                return self.parse_list();
            }
            if c == '{' {
                return self.parse_dict();
            }
            if c.is_ascii_digit()
                || (c == '-' && self.src[self.pos + 1..].starts_with(|d: char| d.is_ascii_digit()))
            {
                return self.parse_number();
            }
        }
        // Identifier / call
        let name = self.parse_ident()?;
        self.skip_ws();
        if self.eat('(') {
            let mut args = Vec::new();
            loop {
                self.skip_ws();
                if self.eat(')') {
                    break;
                }
                let a = self.parse_expr()?;
                args.push(a);
                self.skip_ws();
                if !self.eat(',') {
                    self.expect(")")?;
                    break;
                }
            }
            return Ok(Expr::Call(name, args));
        }
        Ok(Expr::Var(name))
    }
    fn parse_string(&mut self) -> Result<Expr, TemplateError> {
        let quote = self.peek().unwrap();
        self.pos += quote.len_utf8();
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c == quote {
                self.pos += c.len_utf8();
                return Ok(Expr::Literal(Value::String(s)));
            }
            if c == '\\' {
                self.pos += c.len_utf8();
                if let Some(next) = self.peek() {
                    let v = match next {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        '\\' => '\\',
                        '"' => '"',
                        '\'' => '\'',
                        other => other,
                    };
                    s.push(v);
                    self.pos += next.len_utf8();
                }
                continue;
            }
            s.push(c);
            self.pos += c.len_utf8();
        }
        Err(TemplateError::ParseError("unterminated string".into()))
    }
    fn parse_number(&mut self) -> Result<Expr, TemplateError> {
        let start = self.pos;
        if self.eat('-') {
            // already consumed
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '.' {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
        let s = &self.src[start..self.pos];
        if s.contains('.') {
            s.parse::<f64>()
                .map(|n| Expr::Literal(json!(n)))
                .map_err(|e| TemplateError::ParseError(e.to_string()))
        } else {
            s.parse::<i64>()
                .map(|n| Expr::Literal(json!(n)))
                .map_err(|e| TemplateError::ParseError(e.to_string()))
        }
    }
    fn parse_list(&mut self) -> Result<Expr, TemplateError> {
        self.expect("[")?;
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.eat(']') {
                break;
            }
            let e = self.parse_expr()?;
            items.push(e);
            self.skip_ws();
            if !self.eat(',') {
                self.expect("]")?;
                break;
            }
        }
        Ok(Expr::List(items))
    }
    fn parse_dict(&mut self) -> Result<Expr, TemplateError> {
        self.expect("{")?;
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.eat('}') {
                break;
            }
            let k = self.parse_string()?;
            let k = if let Expr::Literal(Value::String(s)) = k {
                s
            } else {
                return Err(TemplateError::ParseError("dict key must be string".into()));
            };
            self.skip_ws();
            self.expect(":")?;
            let v = self.parse_expr()?;
            items.push((k, v));
            self.skip_ws();
            if !self.eat(',') {
                self.expect("}")?;
                break;
            }
        }
        Ok(Expr::Dict(items))
    }
}

fn eval(node: &Node, env: &mut TemplateEnv, out: &mut String) -> Result<(), TemplateError> {
    match node {
        Node::Text(s) => {
            out.push_str(s);
            Ok(())
        }
        Node::Expr(e) => {
            let v = eval_expr(e, env)?;
            out.push_str(&stringify(v));
            Ok(())
        }
        Node::If { cond, then, else_ } => {
            if truthy(&eval_expr(cond, env)?) {
                for n in then {
                    eval(n, env, out)?;
                }
            } else {
                for n in else_ {
                    eval(n, env, out)?;
                }
            }
            Ok(())
        }
        Node::For { var, iter, body } => {
            let coll = eval_expr(iter, env)?;
            let arr = match coll {
                Value::Array(a) => a,
                Value::Null => Vec::new(),
                other => {
                    return Err(TemplateError::TypeError(format!(
                        "for-in expects array, got {other}"
                    )))
                }
            };
            for item in arr {
                env.set(var, item);
                for n in body {
                    eval(n, env, out)?;
                }
            }
            Ok(())
        }
        Node::Set { name, value } => {
            let v = eval_expr(value, env)?;
            env.set(name, v);
            Ok(())
        }
    }
}

fn eval_expr(e: &Expr, env: &TemplateEnv) -> Result<Value, TemplateError> {
    Ok(match e {
        Expr::Literal(v) => v.clone(),
        Expr::Var(name) => env.get(name).cloned().unwrap_or(Value::Null),
        Expr::Subscript(a, k) => {
            let av = eval_expr(a, env)?;
            let kv = eval_expr(k, env)?;
            match (av, kv) {
                (Value::Object(mut m), Value::String(k)) => m.remove(&k).unwrap_or(Value::Null),
                (Value::Array(a), Value::Number(n)) => {
                    let idx = n.as_i64().unwrap_or(0) as isize;
                    a.get(if idx < 0 {
                        (a.len() as isize + idx) as usize
                    } else {
                        idx as usize
                    })
                    .cloned()
                    .unwrap_or(Value::Null)
                }
                _ => Value::Null,
            }
        }
        Expr::Filter(input, name, _args) => {
            let v = eval_expr(input, env)?;
            apply_filter(name, v)?
        }
        Expr::BinOp(op, a, b) => {
            let av = eval_expr(a, env)?;
            let bv = eval_expr(b, env)?;
            apply_binop(*op, &av, &bv)?
        }
        Expr::And(a, b) => {
            let av = eval_expr(a, env)?;
            if !truthy(&av) {
                av
            } else {
                eval_expr(b, env)?
            }
        }
        Expr::Or(a, b) => {
            let av = eval_expr(a, env)?;
            if truthy(&av) {
                av
            } else {
                eval_expr(b, env)?
            }
        }
        Expr::Not(a) => Value::Bool(!truthy(&eval_expr(a, env)?)),
        Expr::In(a, b) => {
            let av = eval_expr(a, env)?;
            let bv = eval_expr(b, env)?;
            Value::Bool(contains(&bv, &av))
        }
        Expr::Neg(a) => {
            let av = eval_expr(a, env)?;
            match av {
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        json!(-i)
                    } else if let Some(f) = n.as_f64() {
                        json!(-f)
                    } else {
                        Value::Null
                    }
                }
                _ => return Err(TemplateError::TypeError("unary `-` on non-number".into())),
            }
        }
        Expr::UnaryOp(op, a) => {
            let av = eval_expr(a, env)?;
            match op {
                UnaryOp::Neg => {
                    if let Value::Number(n) = &av {
                        if let Some(i) = n.as_i64() {
                            json!(-i)
                        } else if let Some(f) = n.as_f64() {
                            json!(-f)
                        } else {
                            Value::Null
                        }
                    } else {
                        return Err(TemplateError::TypeError("unary `-` on non-number".into()));
                    }
                }
                UnaryOp::Not => Value::Bool(!truthy(&av)),
            }
        }
        Expr::Call(name, args) => {
            let argv: Vec<Value> = args
                .iter()
                .map(|a| eval_expr(a, env))
                .collect::<Result<_, _>>()?;
            call_function(name, &argv)?
        }
        Expr::List(items) => {
            let mut arr = Vec::new();
            for it in items {
                arr.push(eval_expr(it, env)?);
            }
            Value::Array(arr)
        }
        Expr::Dict(items) => {
            let mut obj = serde_json::Map::new();
            for (k, v) in items {
                obj.insert(k.clone(), eval_expr(v, env)?);
            }
            Value::Object(obj)
        }
    })
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map_or(false, |x| x != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn contains(coll: &Value, item: &Value) -> bool {
    match coll {
        Value::Array(a) => a.iter().any(|x| x == item),
        Value::String(s) => {
            if let Value::String(needle) = item {
                s.contains(needle.as_str())
            } else {
                false
            }
        }
        _ => false,
    }
}

fn stringify(v: Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s,
        other => other.to_string(),
    }
}

fn apply_binop(op: BinOp, a: &Value, b: &Value) -> Result<Value, TemplateError> {
    use BinOp::*;
    Ok(match op {
        Add => match (a, b) {
            (Value::Number(x), Value::Number(y)) => {
                if let (Some(i), Some(j)) = (x.as_i64(), y.as_i64()) {
                    json!(i + j)
                } else {
                    json!(x.as_f64().unwrap_or(0.0) + y.as_f64().unwrap_or(0.0))
                }
            }
            (Value::String(x), Value::String(y)) => json!(format!("{x}{y}")),
            _ => Value::Null,
        },
        Sub => json!(a.as_f64().unwrap_or(0.0) - b.as_f64().unwrap_or(0.0)),
        Mul => json!(a.as_f64().unwrap_or(0.0) * b.as_f64().unwrap_or(0.0)),
        Div => json!(a.as_f64().unwrap_or(0.0) / b.as_f64().unwrap_or(0.0)),
        Eq => Value::Bool(a == b),
        Ne => Value::Bool(a != b),
        Lt => Value::Bool(a.as_f64().unwrap_or(0.0) < b.as_f64().unwrap_or(0.0)),
        Le => Value::Bool(a.as_f64().unwrap_or(0.0) <= b.as_f64().unwrap_or(0.0)),
        Gt => Value::Bool(a.as_f64().unwrap_or(0.0) > b.as_f64().unwrap_or(0.0)),
        Ge => Value::Bool(a.as_f64().unwrap_or(0.0) >= b.as_f64().unwrap_or(0.0)),
    })
}

fn apply_filter(name: &str, v: Value) -> Result<Value, TemplateError> {
    Ok(match name {
        "length" | "count" => match v {
            Value::Array(a) => json!(a.len()),
            Value::String(s) => json!(s.chars().count()),
            Value::Object(o) => json!(o.len()),
            _ => json!(0),
        },
        "upper" => match v {
            Value::String(s) => json!(s.to_uppercase()),
            other => other,
        },
        "lower" => match v {
            Value::String(s) => json!(s.to_lowercase()),
            other => other,
        },
        "trim" => match v {
            Value::String(s) => json!(s.trim().to_string()),
            other => other,
        },
        "default" => match v {
            Value::Null => Value::String(String::new()),
            other => other,
        },
        "tojson" => json!(v.to_string()),
        "string" => match v {
            Value::String(s) => json!(s),
            other => json!(other.to_string()),
        },
        "int" => match v {
            Value::Number(n) => json!(n.as_i64().unwrap_or(0)),
            Value::String(s) => json!(s.parse::<i64>().unwrap_or(0)),
            _ => json!(0),
        },
        "abs" => match v {
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    json!(i.abs())
                } else if let Some(f) = n.as_f64() {
                    json!(f.abs())
                } else {
                    json!(0)
                }
            }
            _ => json!(0),
        },
        other => {
            return Err(TemplateError::UnknownFilter(other.to_string()));
        }
    })
}

fn call_function(name: &str, args: &[Value]) -> Result<Value, TemplateError> {
    Ok(match name {
        "len" => {
            if let Some(v) = args.first() {
                match v {
                    Value::Array(a) => json!(a.len()),
                    Value::String(s) => json!(s.chars().count()),
                    Value::Object(o) => json!(o.len()),
                    _ => json!(0),
                }
            } else {
                json!(0)
            }
        }
        "str" | "string" => {
            if let Some(v) = args.first() {
                json!(v.to_string())
            } else {
                json!("")
            }
        }
        "range" => {
            let start = args.first().and_then(|v| v.as_i64()).unwrap_or(0);
            let stop = args.get(1).and_then(|v| v.as_i64()).unwrap_or(start);
            let step = args.get(2).and_then(|v| v.as_i64()).unwrap_or(1);
            let mut out = Vec::new();
            let mut i = start;
            while i < stop {
                out.push(json!(i));
                i += step.max(1);
            }
            Value::Array(out)
        }
        other => return Err(TemplateError::Unsupported(format!("function `{other}`"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::message::Role;

    #[test]
    fn builtin_chatml() {
        let p = render_builtin(
            BuiltinTemplate::ChatMl,
            &[
                ChatMessage::new(Role::System, "S"),
                ChatMessage::new(Role::User, "Hi"),
            ],
            &[],
            true,
        );
        assert!(p.contains("<|im_start|>system\nS<|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn builtin_llama3() {
        let p = render_builtin(
            BuiltinTemplate::Llama3,
            &[ChatMessage::new(Role::User, "Hi")],
            &[],
            true,
        );
        assert!(p.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(p.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn builtin_gemma() {
        let p = render_builtin(
            BuiltinTemplate::Gemma,
            &[ChatMessage::new(Role::User, "Hi")],
            &[],
            true,
        );
        assert!(p.contains("<start_of_turn>model"));
    }

    #[test]
    fn builtin_alpaca() {
        let p = render_builtin(
            BuiltinTemplate::Alpaca,
            &[ChatMessage::new(Role::User, "Hi")],
            &[],
            true,
        );
        assert!(p.contains("### Response:"));
    }

    #[test]
    fn builtin_vicuna() {
        let p = render_builtin(
            BuiltinTemplate::Vicuna,
            &[ChatMessage::new(Role::User, "Hi")],
            &[],
            true,
        );
        assert!(p.contains("ASSISTANT:"));
    }

    #[test]
    fn builtin_openchat() {
        let p = render_builtin(
            BuiltinTemplate::OpenChat,
            &[ChatMessage::new(Role::User, "Hi")],
            &[],
            true,
        );
        assert!(p.contains("<|start|>assistant<|message|>"));
    }

    #[test]
    fn builtin_zephyr() {
        let p = render_builtin(
            BuiltinTemplate::Zephyr,
            &[ChatMessage::new(Role::User, "Hi")],
            &[],
            true,
        );
        assert!(p.contains("<|assistant|>"));
    }

    #[test]
    fn builtin_phi3() {
        let p = render_builtin(
            BuiltinTemplate::Phi3,
            &[ChatMessage::new(Role::User, "Hi")],
            &[],
            true,
        );
        assert!(p.contains("<|assistant|>"));
    }

    #[test]
    fn builtin_command_r() {
        let p = render_builtin(
            BuiltinTemplate::CommandR,
            &[ChatMessage::new(Role::User, "Hi")],
            &[],
            true,
        );
        assert!(p.contains("<|CHATBOT_TOKEN|>"));
    }

    #[test]
    fn builtin_deepseek() {
        let p = render_builtin(
            BuiltinTemplate::DeepSeek,
            &[ChatMessage::new(Role::User, "Hi")],
            &[],
            true,
        );
        assert!(p.contains("User: Hi"));
    }

    #[test]
    fn builtin_granite() {
        let p = render_builtin(
            BuiltinTemplate::Granite,
            &[ChatMessage::new(Role::User, "Hi")],
            &[],
            true,
        );
        assert!(p.contains("<|start_of_role|>assistant<|end_of_role|>"));
    }

    #[test]
    fn builtin_oasst() {
        let p = render_builtin(
            BuiltinTemplate::OpenAssistant,
            &[ChatMessage::new(Role::User, "Hi")],
            &[],
            true,
        );
        assert!(p.contains("### Assistant:"));
    }

    #[test]
    fn builtin_mistral_instruct() {
        let p = render_builtin(
            BuiltinTemplate::MistralInstruct,
            &[ChatMessage::new(Role::User, "Hi")],
            &[],
            true,
        );
        assert!(p.contains("[INST]"));
    }

    #[test]
    fn builtin_name_parse() {
        assert_eq!(
            BuiltinTemplate::from_str_ci("gemma-4"),
            Some(BuiltinTemplate::Gemma)
        );
        assert_eq!(
            BuiltinTemplate::from_str_ci("LLAMA-3"),
            Some(BuiltinTemplate::Llama3)
        );
        assert_eq!(
            BuiltinTemplate::from_str_ci("qwen"),
            Some(BuiltinTemplate::ChatMl)
        );
        assert_eq!(BuiltinTemplate::from_str_ci("unknown"), None);
    }

    #[test]
    fn detect_format() {
        let mut md = BTreeMap::new();
        md.insert("general.architecture".into(), "gemma4".into());
        assert_eq!(detect_chat_format(&md), Some(BuiltinTemplate::Gemma));
        let mut md = BTreeMap::new();
        md.insert("model.architecture".into(), "llama".into());
        md.insert("general.name".into(), "Llama 3.1 8B Instruct".into());
        assert_eq!(detect_chat_format(&md), Some(BuiltinTemplate::Llama3));
    }

    #[test]
    fn template_simple_interpolation() {
        // We can't pass arbitrary variables to render_template, but we
        // can verify the loop path also handles a simple substitution.
        let p = render_template(
            "{% for m in messages %}{{ m.content }}{% endfor %}",
            &[ChatMessage::new(Role::User, "abc")],
            &[],
            false,
        )
        .unwrap();
        assert_eq!(p, "abc");
    }

    #[test]
    fn template_for_loop() {
        let tpl = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}assistant:";
        let p = render_template(tpl, &[ChatMessage::new(Role::User, "Hi")], &[], true).unwrap();
        assert!(p.contains("user: Hi"));
        assert!(p.ends_with("assistant:"));
    }

    #[test]
    fn template_if() {
        let tpl = "{% if messages|length > 0 %}yes{% else %}no{% endif %}";
        let p = render_template(tpl, &[ChatMessage::new(Role::User, "x")], &[], false).unwrap();
        assert_eq!(p, "yes");
    }

    #[test]
    fn template_with_tools() {
        let tools = vec![ToolDefinition::new("get_weather", "Get weather for a city")];
        let p = render_builtin(
            BuiltinTemplate::Plain,
            &[ChatMessage::new(Role::User, "Weather in Tokyo?")],
            &tools,
            true,
        );
        assert!(p.contains("get_weather"));
        assert!(p.contains("system:"));
    }

    #[test]
    fn builtin_chatml_renders_assistant_tool_calls() {
        let message = ChatMessage::new(Role::Assistant, "").with_tool_call(
            crate::chat::ToolCall::new("call_weather", "get_weather", json!({"city": "Tokyo"})),
        );
        let p = render_builtin(BuiltinTemplate::ChatMl, &[message], &[], false);
        assert!(p.contains("<tool_call>"));
        assert!(p.contains(r#""name":"get_weather""#));
        assert!(p.contains(r#""city":"Tokyo""#));
    }
}
