//! Tool / function-calling types and parsers.
//!
//! [`ToolDefinition`] describes a callable tool (JSON-Schema shape).
//! [`ToolCall`] represents the model's request to invoke one.
//! [`ToolParser`] is a stateful parser that extracts tool calls from a
//! streaming model response.

use serde_json::{json, Value};

use super::message::ChatMessage;

/// A tool the model is allowed to call.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolDefinition {
    /// Unique name (e.g. `get_weather`).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema describing the arguments shape.
    pub parameters: Value,
}

impl ToolDefinition {
    /// Construct a new tool definition.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    /// Set the JSON Schema for the tool's arguments.
    #[must_use]
    pub fn with_parameters(mut self, params: Value) -> Self {
        self.parameters = params;
        self
    }

    /// Serialize to a function-tool JSON shape.
    #[must_use]
    pub fn to_openai_function(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }

    /// Internal: convert to a JSON object for the template renderer.
    pub(crate) fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        })
    }
}

/// A single tool invocation request emitted by the model.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    /// Unique id (e.g. `call_abc123`).
    pub id: String,
    /// Name of the tool.
    pub name: String,
    /// Arguments (must conform to the tool's schema).
    pub arguments: Value,
}

impl ToolCall {
    /// Construct a new tool call.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    /// JSON shape used by the template renderer.
    pub(crate) fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        })
    }
}

/// Errors produced while parsing a model response for tool calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolParseError {
    /// Model produced a call but its JSON was malformed.
    InvalidJson(String),
    /// Call had a missing `name` field.
    MissingName,
    /// Call had a missing `arguments` field.
    MissingArguments,
}

impl std::fmt::Display for ToolParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidJson(s) => write!(f, "invalid JSON: {s}"),
            Self::MissingName => write!(f, "missing `name` field"),
            Self::MissingArguments => write!(f, "missing `arguments` field"),
        }
    }
}

impl std::error::Error for ToolParseError {}

/// Format the model was trained to use for tool calling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ToolFormat {
    /// ChatML + JSON function calls (Qwen 2.5, Hermes, etc.).
    /// Format: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
    #[default]
    ChatMl,
    /// Mistral / Mixtral `[]` syntax.
    /// Format: `[TOOL_CALLS][{"name": "...", "arguments": {...}}]`
    Mistral,
    /// Llama-3 / Firefunction-v2: `<|python_tag|>` prefix.
    /// Format: `<|python_tag|>{"name": "...", "arguments": {...}}`
    Llama3,
    /// Plain JSON object (no wrapping tokens).
    Plain,
    /// Functionary: `<|start|>function<|message|>...<|call|>`
    Functionary,
}

impl ToolFormat {
    /// Try to auto-detect from the chat template name.
    #[must_use]
    pub fn from_chat_format(name: &str) -> Self {
        let n = name.to_ascii_lowercase();
        if n.contains("qwen") || n.contains("hermes") || n.contains("chatml") {
            Self::ChatMl
        } else if n.contains("mistral") || n.contains("mixtral") {
            Self::Mistral
        } else if n.contains("llama-3") || n.contains("llama3") || n.contains("firefunction") {
            Self::Llama3
        } else if n.contains("functionary") {
            Self::Functionary
        } else {
            Self::Plain
        }
    }
}

/// Stateful tool-call parser.
///
/// ```rust
/// use llama_crab::chat::tool_call::ToolParser;
/// let mut p = ToolParser::new(Default::default());
/// let calls = p.feed("<tool_call>{\"name\": \"foo\", \"arguments\": {}}</tool_call>");
/// assert_eq!(calls.len(), 1);
/// ```
#[derive(Debug)]
pub struct ToolParser {
    format: ToolFormat,
    /// Buffered partial JSON.
    buffer: String,
    /// Number of opening braces seen minus closing braces.
    brace_depth: i32,
    /// True while inside a `<tool_call>` block.
    in_call: bool,
    next_id: u32,
}

impl ToolParser {
    /// Construct a parser for the given format.
    #[must_use]
    pub fn new(format: ToolFormat) -> Self {
        Self {
            format,
            buffer: String::new(),
            brace_depth: 0,
            in_call: false,
            next_id: 0,
        }
    }

    /// Construct from a chat-format name.
    #[must_use]
    pub fn for_chat_format(name: &str) -> Self {
        Self::new(ToolFormat::from_chat_format(name))
    }

    /// Feed a chunk of model output; return the list of tool calls
    /// completed during this chunk.
    pub fn feed(&mut self, chunk: &str) -> Vec<Result<ToolCall, ToolParseError>> {
        let mut out = Vec::new();
        for c in chunk.chars() {
            self.feed_char(c, &mut out);
        }
        out
    }

    /// True if a tool call body is currently being assembled.
    ///
    /// Streaming consumers can use this together with [`Self::current_partial`]
    /// to emit argument deltas as the model produces them.
    #[must_use]
    pub fn in_call(&self) -> bool {
        self.in_call
    }

    /// Current partial JSON of the in-progress tool call.
    ///
    /// Only meaningful when [`Self::in_call`] is `true`. Returns `None`
    /// outside a call. The returned string grows monotonically until the
    /// call completes (or the parser transitions out of `in_call`).
    #[must_use]
    pub fn current_partial(&self) -> Option<&str> {
        if self.in_call {
            Some(self.buffer.as_str())
        } else {
            None
        }
    }

    fn feed_char(&mut self, c: char, out: &mut Vec<Result<ToolCall, ToolParseError>>) {
        let s = c.to_string();
        match self.format {
            ToolFormat::ChatMl => {
                if !self.in_call {
                    // Looking for `<tool_call>` token.
                    if self.buffer.len() < 12 {
                        self.buffer.push(c);
                        if self.buffer.ends_with("<tool_call>") {
                            self.buffer.clear();
                            self.in_call = true;
                        }
                    } else {
                        // Discard stale buffer.
                        self.buffer.clear();
                        self.buffer.push(c);
                    }
                } else {
                    // Inside call: collect JSON until matching brace.
                    self.buffer.push(c);
                    match c {
                        '{' => self.brace_depth += 1,
                        '}' => {
                            self.brace_depth -= 1;
                            if self.brace_depth == 0 && self.buffer.trim().ends_with('}') {
                                let raw = std::mem::take(&mut self.buffer);
                                if let Some(call) = self.parse_json_call(&raw) {
                                    out.push(Ok(call));
                                }
                                self.in_call = false;
                            }
                        }
                        _ => {}
                    }
                }
            }
            ToolFormat::Mistral => {
                if !self.in_call {
                    self.buffer.push(c);
                    if self.buffer.contains("[TOOL_CALLS]") {
                        // Start scanning for the `[`.
                        self.buffer.clear();
                        self.in_call = true;
                        self.brace_depth = 0;
                    }
                    if self.buffer.len() > 64 {
                        // Truncate to avoid runaway.
                        self.buffer.drain(..self.buffer.len() - 32);
                    }
                } else {
                    self.buffer.push(c);
                    match c {
                        '[' => self.brace_depth += 1,
                        ']' => {
                            self.brace_depth -= 1;
                            if self.brace_depth <= 0 {
                                let raw = std::mem::take(&mut self.buffer);
                                // Try to parse as `[ {...} ]` or `[ {...}, {...} ]`.
                                if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(&raw)
                                {
                                    for item in items {
                                        if let Some(call) = self.parse_call_obj(&item) {
                                            out.push(Ok(call));
                                        }
                                    }
                                }
                                self.in_call = false;
                                self.brace_depth = 0;
                            }
                        }
                        _ => {}
                    }
                }
            }
            ToolFormat::Llama3 => {
                if !self.in_call {
                    self.buffer.push(c);
                    if self.buffer.ends_with("<|python_tag|>") {
                        self.buffer.clear();
                        self.in_call = true;
                        self.brace_depth = 0;
                    }
                } else {
                    self.buffer.push(c);
                    match c {
                        '{' => self.brace_depth += 1,
                        '}' => {
                            self.brace_depth -= 1;
                            if self.brace_depth == 0 {
                                let raw = std::mem::take(&mut self.buffer);
                                if let Some(call) = self.parse_json_call(&raw) {
                                    out.push(Ok(call));
                                }
                                self.in_call = false;
                            }
                        }
                        _ => {}
                    }
                }
            }
            ToolFormat::Plain => {
                self.buffer.push(c);
                match c {
                    '{' => self.brace_depth += 1,
                    '}' => {
                        self.brace_depth -= 1;
                        if self.brace_depth == 0 && self.buffer.trim().starts_with('{') {
                            let raw = std::mem::take(&mut self.buffer);
                            if let Some(call) = self.parse_json_call(&raw) {
                                out.push(Ok(call));
                            }
                        }
                    }
                    _ => {}
                }
            }
            ToolFormat::Functionary => {
                if !self.in_call {
                    self.buffer.push(c);
                    if self.buffer.contains("<|call|>") {
                        // The payload is the JSON before this tag.
                        let raw = self
                            .buffer
                            .replace("<|call|>", "")
                            .replace("<|start|>function<|message|>", "")
                            .trim()
                            .to_string();
                        self.buffer.clear();
                        if let Some(call) = self.parse_json_call(&raw) {
                            out.push(Ok(call));
                        }
                    }
                    if self.buffer.len() > 1024 {
                        self.buffer.clear();
                    }
                }
                let _ = s;
            }
        }
    }

    /// Flush remaining buffered content as a final call (used at end-of-stream).
    pub fn finish(&mut self) -> Vec<Result<ToolCall, ToolParseError>> {
        let mut out = Vec::new();
        let buf = std::mem::take(&mut self.buffer);
        if !buf.is_empty() && buf.trim().starts_with('{') && buf.trim().ends_with('}') {
            if let Some(call) = self.parse_json_call(&buf) {
                out.push(Ok(call));
            }
        }
        out
    }

    fn parse_json_call(&mut self, raw: &str) -> Option<ToolCall> {
        let v: Value = serde_json::from_str(raw).ok()?;
        self.parse_call_obj(&v)
    }

    fn parse_call_obj(&mut self, v: &Value) -> Option<ToolCall> {
        let name = v.get("name")?.as_str()?.to_string();
        let arguments = v.get("arguments")?.clone();
        self.next_id += 1;
        let id = format!("call_{}", self.next_id);
        Some(ToolCall::new(id, name, arguments))
    }
}

/// Helper: extract tool calls from a complete (non-streaming) response.
pub fn extract_tool_calls(format: ToolFormat, text: &str) -> Vec<Result<ToolCall, ToolParseError>> {
    let mut p = ToolParser::new(format);
    p.feed(text)
}

/// One per-tool-call delta emitted while streaming model output.
///
/// Each `ToolCallDelta` carries a single `index` (assigned in the order
/// the calls start) and at most one of: a fresh `id`, a `name`, an
/// `arguments` diff, or a `completed` final value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallDelta {
    /// Index of the call this delta refers to.
    pub index: usize,
    /// Fresh id; set only on the first delta for a call.
    pub id: Option<String>,
    /// Tool name; emitted once it is observable in the partial JSON.
    pub name: Option<String>,
    /// Incremental argument JSON; only the freshly seen suffix.
    pub arguments: Option<String>,
    /// Set on the final delta for a call (carries the full parsed result).
    pub completed: Option<ToolCall>,
}

/// Streaming wrapper around [`ToolParser`] that emits OpenAI-style
/// `tool_calls` deltas.
#[derive(Debug)]
pub struct ToolCallStream {
    parser: ToolParser,
    next_index: usize,
    /// State for the call currently being built (if any).
    active: Option<ActiveStreamCall>,
    /// Whether the most recent transition was into a new call.
    just_started: bool,
}

#[derive(Debug)]
struct ActiveStreamCall {
    index: usize,
    id: String,
    name: Option<String>,
    name_emitted: bool,
    /// The arguments string already emitted as a delta.
    args_emitted: String,
    /// Byte offset of the `arguments` value in `partial_seen`, once
    /// the `"arguments"` key has been seen in the partial JSON.
    args_value_start: Option<usize>,
}

impl ToolCallStream {
    /// Construct a stream for the given format.
    #[must_use]
    pub fn new(format: ToolFormat) -> Self {
        Self {
            parser: ToolParser::new(format),
            next_index: 0,
            active: None,
            just_started: false,
        }
    }

    /// Construct from a chat-format name.
    #[must_use]
    pub fn for_chat_format(name: &str) -> Self {
        Self::new(ToolFormat::from_chat_format(name))
    }

    /// Number of completed calls so far.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.next_index
            .saturating_sub(usize::from(self.active.is_some()))
    }

    /// True if a tool-call body is currently being assembled.
    ///
    /// Consumers can use this to suppress content chunks while a
    /// tool call's arguments are being parsed out of the model output.
    #[must_use]
    pub fn in_call(&self) -> bool {
        self.parser.in_call()
    }

    /// Feed a chunk of model output; return the deltas to emit.
    pub fn feed(&mut self, chunk: &str) -> Vec<ToolCallDelta> {
        let was_in_call = self.parser.in_call();
        let completed = self.parser.feed(chunk);
        let now_in_call = self.parser.in_call();
        let mut out = Vec::new();

        // 1. Handle calls completed during this feed.
        for call in completed {
            let call = match call {
                Ok(c) => c,
                Err(err) => {
                    // Surface as a delta with a synthetic id so the client
                    // sees the error attached to a specific index.
                    let index = if let Some(a) = self.active.take() {
                        a.index
                    } else {
                        let idx = self.next_index;
                        self.next_index += 1;
                        idx
                    };
                    out.push(ToolCallDelta {
                        index,
                        id: None,
                        name: None,
                        arguments: None,
                        completed: Some(ToolCall::new(
                            format!("call_err_{index}"),
                            String::new(),
                            serde_json::Value::String(err.to_string()),
                        )),
                    });
                    continue;
                }
            };

            // Promote the active call to completed.
            let index = if let Some(a) = self.active.take() {
                a.index
            } else {
                let idx = self.next_index;
                self.next_index += 1;
                idx
            };
            out.push(ToolCallDelta {
                index,
                id: None,
                name: None,
                arguments: None,
                completed: Some(call),
            });
        }

        // 2. Detect a fresh call start (transition !in_call -> in_call).
        if !was_in_call && now_in_call {
            self.just_started = true;
        }

        // 3. If we just started a call, emit the id delta.
        if self.just_started {
            self.just_started = false;
            let index = self.next_index;
            self.next_index += 1;
            let id = format!("call_{index}");
            self.active = Some(ActiveStreamCall {
                index,
                id: id.clone(),
                name: None,
                name_emitted: false,
                args_emitted: String::new(),
                args_value_start: None,
            });
            out.push(ToolCallDelta {
                index,
                id: Some(id),
                name: None,
                arguments: None,
                completed: None,
            });
        }

        // 4. Stream the current partial JSON.
        if let Some(active) = self.active.as_mut() {
            if let Some(partial) = self.parser.current_partial() {
                // Detect `name` field in the partial (top-level only).
                if !active.name_emitted {
                    if let Some(name) = extract_top_level_name(partial) {
                        active.name = Some(name.clone());
                        active.name_emitted = true;
                        out.push(ToolCallDelta {
                            index: active.index,
                            id: None,
                            name: Some(name),
                            arguments: None,
                            completed: None,
                        });
                    }
                }
                // Detect `arguments` value range.
                if active.args_value_start.is_none() {
                    if let Some(start) = extract_top_level_value_start(partial, "arguments") {
                        active.args_value_start = Some(start);
                    }
                }
                // Emit the arguments diff.
                if let Some(start) = active.args_value_start {
                    let value_end = value_end_offset(partial, start);
                    if value_end > active.args_emitted.len() {
                        let diff =
                            partial[start + active.args_emitted.len()..value_end].to_string();
                        active.args_emitted.push_str(&diff);
                        out.push(ToolCallDelta {
                            index: active.index,
                            id: None,
                            name: None,
                            arguments: Some(diff),
                            completed: None,
                        });
                    }
                }
            }
        }

        out
    }

    /// Flush any remaining buffered content (call at end-of-stream).
    pub fn finish(&mut self) -> Vec<ToolCallDelta> {
        let mut out = Vec::new();
        let final_completed = self.parser.finish();
        for call in final_completed {
            let call = match call {
                Ok(c) => c,
                Err(_) => continue,
            };
            let index = if let Some(a) = self.active.take() {
                a.index
            } else {
                let idx = self.next_index;
                self.next_index += 1;
                idx
            };
            out.push(ToolCallDelta {
                index,
                id: None,
                name: None,
                arguments: None,
                completed: Some(call),
            });
        }
        out
    }
}

/// Extract a top-level `"name": "..."` value from a partial JSON object
/// being built up character-by-character.
///
/// Returns `None` while the value is still being built (incomplete
/// string, missing colon, missing opening quote, etc.).
fn extract_top_level_name(partial: &str) -> Option<String> {
    // Find the first `"name"` key at the top level.
    let bytes = partial.as_bytes();
    let mut i = 0;
    // Skip leading whitespace.
    while i < bytes.len() && (bytes[i] as char).is_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'{' {
        return None;
    }
    i += 1;
    while i < bytes.len() {
        // Skip whitespace.
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            return None;
        }
        if bytes[i] == b'}' {
            return None;
        }
        // Expect `"`.
        if bytes[i] != b'"' {
            return None;
        }
        i += 1;
        // Read key.
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'"' {
            if bytes[i] == b'\\' && i + 1 < bytes.len() {
                i += 2;
            } else {
                i += 1;
            }
        }
        if i >= bytes.len() {
            return None;
        }
        let key = &partial[key_start..i];
        i += 1; // skip closing `"`
                // Skip whitespace.
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b':' {
            return None;
        }
        i += 1;
        // Skip whitespace.
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if key == "name" {
            if i >= bytes.len() || bytes[i] != b'"' {
                return None;
            }
            i += 1;
            let mut val = String::new();
            while i < bytes.len() {
                let c = bytes[i];
                if c == b'"' {
                    return Some(val);
                }
                if c == b'\\' && i + 1 < bytes.len() {
                    val.push(bytes[i + 1] as char);
                    i += 2;
                } else {
                    val.push(c as char);
                    i += 1;
                }
            }
            return None;
        } else {
            if i >= bytes.len() {
                return None;
            }
            match bytes[i] {
                b'{' | b'[' => {
                    let open = bytes[i];
                    let close = if open == b'{' { b'}' } else { b']' };
                    let mut depth = 1_i32;
                    i += 1;
                    while i < bytes.len() && depth > 0 {
                        if bytes[i] == b'"' {
                            i += 1;
                            while i < bytes.len() && bytes[i] != b'"' {
                                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                                    i += 2;
                                } else {
                                    i += 1;
                                }
                            }
                            if i < bytes.len() {
                                i += 1;
                            }
                        } else if bytes[i] == open {
                            depth += 1;
                            i += 1;
                        } else if bytes[i] == close {
                            depth -= 1;
                            i += 1;
                        } else {
                            i += 1;
                        }
                    }
                    if depth > 0 {
                        return None;
                    }
                }
                b'"' => {
                    i += 1;
                    while i < bytes.len() && bytes[i] != b'"' {
                        if bytes[i] == b'\\' && i + 1 < bytes.len() {
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                    if i < bytes.len() {
                        i += 1;
                    }
                }
                _ => {
                    while i < bytes.len() && bytes[i] != b',' && bytes[i] != b'}' {
                        i += 1;
                    }
                }
            }
            if i < bytes.len() && bytes[i] == b',' {
                i += 1;
            }
        }
    }
    None
}

/// Find the byte offset in `partial` where the top-level value of the
/// key `target_key` starts (i.e. just after the colon and any whitespace).
fn extract_top_level_value_start(partial: &str, target_key: &str) -> Option<usize> {
    let bytes = partial.as_bytes();
    let mut i = 0;
    while i < bytes.len() && (bytes[i] as char).is_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'{' {
        return None;
    }
    i += 1;
    while i < bytes.len() {
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] == b'}' {
            return None;
        }
        if bytes[i] != b'"' {
            return None;
        }
        i += 1;
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'"' {
            if bytes[i] == b'\\' && i + 1 < bytes.len() {
                i += 2;
            } else {
                i += 1;
            }
        }
        if i >= bytes.len() {
            return None;
        }
        let key = &partial[key_start..i];
        i += 1;
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b':' {
            return None;
        }
        i += 1;
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if key == target_key {
            return Some(i);
        }
        if i >= bytes.len() {
            return None;
        }
        match bytes[i] {
            b'{' | b'[' => {
                let open = bytes[i];
                let close = if open == b'{' { b'}' } else { b']' };
                let mut depth = 1_i32;
                i += 1;
                while i < bytes.len() && depth > 0 {
                    if bytes[i] == b'"' {
                        i += 1;
                        while i < bytes.len() && bytes[i] != b'"' {
                            if bytes[i] == b'\\' && i + 1 < bytes.len() {
                                i += 2;
                            } else {
                                i += 1;
                            }
                        }
                        if i < bytes.len() {
                            i += 1;
                        }
                    } else if bytes[i] == open {
                        depth += 1;
                        i += 1;
                    } else if bytes[i] == close {
                        depth -= 1;
                        i += 1;
                    } else {
                        i += 1;
                    }
                }
                if depth > 0 {
                    return None;
                }
            }
            b'"' => {
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                if i < bytes.len() {
                    i += 1;
                }
            }
            _ => {
                while i < bytes.len() && bytes[i] != b',' && bytes[i] != b'}' {
                    i += 1;
                }
            }
        }
        if i < bytes.len() && bytes[i] == b',' {
            i += 1;
        }
    }
    None
}

/// Given a `partial` JSON and the byte offset where a top-level value
/// starts, return the byte offset where the value ends (exclusive).
fn value_end_offset(partial: &str, start: usize) -> usize {
    let bytes = partial.as_bytes();
    if start >= bytes.len() {
        return start;
    }
    match bytes[start] {
        b'{' | b'[' => {
            let open = bytes[start];
            let close = if open == b'{' { b'}' } else { b']' };
            let mut depth = 1_i32;
            let mut i = start + 1;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == b'"' {
                    i += 1;
                    while i < bytes.len() && bytes[i] != b'"' {
                        if bytes[i] == b'\\' && i + 1 < bytes.len() {
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                    if i < bytes.len() {
                        i += 1;
                    }
                } else if bytes[i] == open {
                    depth += 1;
                    i += 1;
                } else if bytes[i] == close {
                    depth -= 1;
                    i += 1;
                } else {
                    i += 1;
                }
            }
            if depth == 0 {
                i
            } else {
                bytes.len()
            }
        }
        b'"' => {
            let mut i = start + 1;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if i < bytes.len() {
                i + 1
            } else {
                bytes.len()
            }
        }
        _ => {
            let mut i = start;
            while i < bytes.len() && bytes[i] != b',' && bytes[i] != b'}' {
                i += 1;
            }
            i
        }
    }
}

/// Convenience: turn tool calls into a synthetic assistant message.
///
/// Useful when you want to feed tool calls back into a chat history.
pub fn tool_calls_to_message(calls: &[ToolCall]) -> ChatMessage {
    use super::message::Role;
    let mut m = ChatMessage::new(Role::Assistant, String::new());
    for c in calls {
        m = m.with_tool_call(c.clone());
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_chatml() {
        let s = r#"<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>"#;
        let mut p = ToolParser::new(ToolFormat::ChatMl);
        let calls: Vec<_> = p.feed(s).into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn parse_mistral() {
        let s = r#"[TOOL_CALLS][{"name": "x", "arguments": {}}]"#;
        let mut p = ToolParser::new(ToolFormat::Mistral);
        let calls: Vec<_> = p.feed(s).into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn parse_llama3() {
        let s = r#"<|python_tag|>{"name": "x", "arguments": {}}"#;
        let mut p = ToolParser::new(ToolFormat::Llama3);
        let calls: Vec<_> = p.feed(s).into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn parse_plain() {
        let s = r#"{"name": "x", "arguments": {}}"#;
        let mut p = ToolParser::new(ToolFormat::Plain);
        let calls: Vec<_> = p.feed(s).into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn auto_detect_format() {
        assert_eq!(ToolFormat::from_chat_format("qwen"), ToolFormat::ChatMl);
        assert_eq!(ToolFormat::from_chat_format("llama-3"), ToolFormat::Llama3);
        assert_eq!(ToolFormat::from_chat_format("mistral"), ToolFormat::Mistral);
        assert_eq!(ToolFormat::from_chat_format("plain"), ToolFormat::Plain);
    }

    #[test]
    fn stream_emits_id_then_name_then_arguments_for_chatml() {
        let mut s = ToolCallStream::new(ToolFormat::ChatMl);
        let mut all = Vec::new();
        all.extend(s.feed("<tool_call>"));
        all.extend(s.feed(r#"{"name":"#));
        all.extend(s.feed(r#""get_weather""#));
        all.extend(s.feed(r#","arguments":{"city":"Tokyo"}}"#));
        all.extend(s.feed("</tool_call>"));

        // First delta: id only, index 0
        assert_eq!(all[0].index, 0);
        assert_eq!(all[0].id.as_deref(), Some("call_0"));
        assert!(all[0].name.is_none());
        assert!(all[0].arguments.is_none());

        // Find a delta carrying the name.
        let name_delta = all
            .iter()
            .find(|d| d.name.is_some())
            .expect("expected a name delta");
        assert_eq!(name_delta.index, 0);
        assert_eq!(name_delta.name.as_deref(), Some("get_weather"));

        // The final delta carries the completed call.
        let completed = all
            .iter()
            .rev()
            .find(|d| d.completed.is_some())
            .expect("expected a completed delta");
        let call = completed.completed.as_ref().unwrap();
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments["city"], "Tokyo");
    }

    #[test]
    fn stream_emits_arguments_growth() {
        let mut s = ToolCallStream::new(ToolFormat::ChatMl);
        let mut all = Vec::new();
        for chunk in [
            "<tool_call>",
            r#"{"name":"f","arguments":{"#,
            r#""a":1}"#,
            "}",
            "</tool_call>",
        ] {
            all.extend(s.feed(chunk));
        }
        let arg_diffs: Vec<String> = all.iter().filter_map(|d| d.arguments.clone()).collect();
        let joined: String = arg_diffs.iter().map(String::as_str).collect();
        assert_eq!(joined, r#"{"a":1}"#);
    }

    #[test]
    fn stream_mistral_array_emits_two_calls() {
        let mut s = ToolCallStream::new(ToolFormat::Mistral);
        let mut all = Vec::new();
        all.extend(s.feed("[TOOL_CALLS]"));
        all.extend(s.feed(r#"[{"name":"a","arguments":{}},{"name":"b","arguments":{"x":1}}]"#));
        let completed: Vec<_> = all.iter().filter_map(|d| d.completed.as_ref()).collect();
        assert_eq!(completed.len(), 2);
        assert_eq!(completed[0].name, "a");
        assert_eq!(completed[1].name, "b");
        assert_eq!(completed[1].arguments["x"], 1);
    }
}
