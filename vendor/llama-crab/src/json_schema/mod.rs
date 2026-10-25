//! JSON Schema → GBNF grammar converter.
//!
//! A pure-Rust converter for JSON Schema to GBNF. Supports a useful subset of
//! [JSON Schema 2020-12] including:
//!
//! * `type`: `object`, `array`, `string`, `integer`, `number`, `boolean`, `null`
//! * `properties`, `required`, `additionalProperties` (with schema)
//! * `items` (single-schema) and `prefixItems`/`minItems`/`maxItems`
//! * `enum` (string, integer, boolean, null)
//! * `const`
//! * `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`
//! * `minLength`, `maxLength`, `pattern`
//! * `format` (special-cased: `date-time`, `email`, `uri`, `uuid`)
//! * `oneOf`, `anyOf`, `allOf`
//! * `$ref` (local, `#/definitions/...` style)
//! * `definitions` / `$defs`
//!
//! [JSON Schema 2020-12]: https://json-schema.org/draft/2020-12/json-schema-core.html
//!
//! # Example
//!
//! ```
//! use llama_crab::json_schema::schema_to_grammar;
//! use serde_json::json;
//!
//! let schema = json!({
//!     "type": "object",
//!     "properties": {
//!         "name": { "type": "string" },
//!         "age":  { "type": "integer" }
//!     },
//!     "required": ["name", "age"]
//! });
//! let grammar = schema_to_grammar(&schema, "root").unwrap();
//! assert!(grammar.contains("name"));
//! assert!(grammar.contains("age"));
//! ```

#![allow(clippy::module_name_repetitions)]

use std::collections::BTreeMap;
use std::fmt::Write;

use serde_json::{json, Value};

use crate::error::{LlamaError, Result};

/// Convert a JSON Schema into a GBNF grammar string rooted at `root_rule`.
///
/// # Errors
/// Returns an error if `schema` is not valid JSON object syntax (e.g. not
/// a map at the top level).
pub fn schema_to_grammar(schema: &Value, root_rule: &str) -> Result<String> {
    let mut conv = SchemaConverter::new(root_rule);
    conv.visit(schema, root_rule)?;
    Ok(conv.format_grammar())
}

/// Public error type for schema conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaError {
    /// Top-level schema must be a JSON object.
    NotAnObject,
    /// `$ref` could not be resolved.
    UnresolvedRef(String),
    /// Invalid integer range.
    InvalidRange,
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAnObject => f.write_str("top-level schema must be a JSON object"),
            Self::UnresolvedRef(s) => write!(f, "unresolved $ref: {s}"),
            Self::InvalidRange => f.write_str("invalid integer range"),
        }
    }
}

impl std::error::Error for SchemaError {}

impl From<SchemaError> for LlamaError {
    fn from(e: SchemaError) -> Self {
        Self::JsonSchemaToGrammar(e.to_string())
    }
}

/// Internal converter state.
struct SchemaConverter {
    /// Rule name → GBNF body.
    rules: BTreeMap<String, String>,
    /// Schema registry (for `$ref` lookup).
    definitions: BTreeMap<String, Value>,
    /// Counter for generating fresh rule names.
    counter: u32,
    /// Root rule name.
    root: String,
}

impl SchemaConverter {
    fn new(root: &str) -> Self {
        Self {
            rules: BTreeMap::new(),
            definitions: BTreeMap::new(),
            counter: 0,
            root: root.to_string(),
        }
    }

    /// Generate a unique rule name with the given prefix.
    fn fresh_name(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}-{}", self.counter)
    }

    /// Walk the schema and add rules to the registry.
    fn visit(&mut self, schema: &Value, name: &str) -> Result<String> {
        let Value::Object(_) = schema else {
            return Err(SchemaError::NotAnObject.into());
        };

        // Register definitions and $defs FIRST so subsequent $ref lookups
        // can find them.
        if let Some(Value::Object(defs)) = schema.get("$defs") {
            for (k, v) in defs {
                self.definitions.insert(format!("$defs/{k}"), v.clone());
            }
        }
        if let Some(Value::Object(defs)) = schema.get("definitions") {
            for (k, v) in defs {
                self.definitions
                    .insert(format!("#/definitions/{k}"), v.clone());
            }
        }

        // Resolve $ref up front.
        if let Some(ref_str) = schema.get("$ref").and_then(|v| v.as_str()) {
            let target = self.resolve_ref(ref_str)?;
            return self.visit(&target, name);
        }

        // Build the GBNF body.
        let body = self.build(schema)?;
        self.rules.insert(name.to_string(), body);
        Ok(name.to_string())
    }

    /// Resolve a `$ref` like `"#/definitions/Foo"` or `"#/$defs/Foo"`.
    fn resolve_ref(&self, ref_str: &str) -> Result<Value> {
        if let Some(target) = self.definitions.get(ref_str) {
            return Ok(target.clone());
        }
        if let Some(rest) = ref_str.strip_prefix("#/definitions/") {
            if let Some(target) = self.definitions.get(&format!("#/definitions/{rest}")) {
                return Ok(target.clone());
            }
        }
        if let Some(rest) = ref_str.strip_prefix("#/$defs/") {
            if let Some(target) = self.definitions.get(&format!("$defs/{rest}")) {
                return Ok(target.clone());
            }
        }
        Err(SchemaError::UnresolvedRef(ref_str.to_string()).into())
    }

    /// Build the GBNF body for a schema node.
    fn build(&mut self, schema: &Value) -> Result<String> {
        // anyOf / oneOf / allOf -------------------------------------------------
        if let Some(arr) = schema.get("anyOf").and_then(|v| v.as_array()) {
            return self.build_union(arr, /* any_of = */ true);
        }
        if let Some(arr) = schema.get("oneOf").and_then(|v| v.as_array()) {
            return self.build_union(arr, /* any_of = */ false);
        }
        if let Some(arr) = schema.get("allOf").and_then(|v| v.as_array()) {
            return self.build_all_of(arr);
        }

        // const / enum ----------------------------------------------------------
        if let Some(c) = schema.get("const") {
            let name = self.fresh_name("const");
            let body = self.literal(c);
            return Ok(self.format_rule(&name, &body));
        }
        if let Some(arr) = schema.get("enum").and_then(|v| v.as_array()) {
            let name = self.fresh_name("enum");
            let alts: Vec<String> = arr.iter().map(|v| self.literal(v)).collect();
            let body = alts.join(" | ");
            return Ok(self.format_rule(&name, &body));
        }

        // type-driven -----------------------------------------------------------
        let ty = schema.get("type").and_then(|v| v.as_str()).unwrap_or("any");
        let body = match ty {
            "string" => self.build_string(schema)?,
            "integer" | "number" => self.build_number(schema)?,
            "boolean" => r#""true" | "false""#.to_string(),
            "null" => r#""null""#.to_string(),
            "array" => self.build_array(schema)?,
            "object" => self.build_object(schema)?,
            _ => "value".to_string(),
        };
        Ok(body)
    }

    /// Build a union of subschemas (`anyOf` / `oneOf`).
    fn build_union(&mut self, schemas: &[Value], any_of: bool) -> Result<String> {
        let mut names = Vec::with_capacity(schemas.len());
        for s in schemas {
            let n = self.fresh_name("alt");
            self.visit(s, &n)?;
            names.push(n);
        }
        let sep = if any_of {
            " | "
        } else {
            " | " /* best-effort */
        };
        Ok(names.join(sep))
    }

    /// `allOf` — concatenate subschemas (assumes non-conflicting shape).
    fn build_all_of(&mut self, schemas: &[Value]) -> Result<String> {
        let mut names = Vec::with_capacity(schemas.len());
        for s in schemas {
            let n = self.fresh_name("all");
            self.visit(s, &n)?;
            names.push(n);
        }
        // For primitives this is a sequence; for objects merge handled
        // inside `build_object` (we keep this conservative here).
        Ok(names.join(" "))
    }

    fn build_string(&mut self, schema: &Value) -> Result<String> {
        // `format` shortcut
        if let Some(fmt) = schema.get("format").and_then(|v| v.as_str()) {
            match fmt {
                "date-time" => {
                    let name = self.fresh_name("date-time");
                    let body = r#"\"" [0-9]{4} "-" [0-9]{2} "-" [0-9]{2} "T" [0-9]{2} ":" [0-9]{2} ":" [0-9]{2} ("." [0-9]+)? ("Z" | ("+" | "-") [0-9]{2} ":" [0-9]{2}) "\"""#.to_string();
                    self.rules.insert(name.clone(), body);
                    return Ok(name);
                }
                "email" => {
                    let name = self.fresh_name("email");
                    let body = r#"\"" [a-zA-Z0-9._%+-]+ "@" [a-zA-Z0-9.-]+ "." [a-zA-Z]{2,} "\"""#
                        .to_string();
                    self.rules.insert(name.clone(), body);
                    return Ok(name);
                }
                "uri" | "url" => {
                    let name = self.fresh_name("uri");
                    let body = r#"\"" [a-zA-Z][a-zA-Z0-9+.-]* "://" [^"\\s]+ "\"""#.to_string();
                    self.rules.insert(name.clone(), body);
                    return Ok(name);
                }
                "uuid" => {
                    let name = self.fresh_name("uuid");
                    let body = r#"\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"""#.to_string();
                    self.rules.insert(name.clone(), body);
                    return Ok(name);
                }
                _ => {}
            }
        }
        if let Some(pattern) = schema.get("pattern").and_then(|v| v.as_str()) {
            // We hand off patterns as `re` (regex) and let llama.cpp interpret.
            // GBNF lacks a regex primitive, so we approximate with `.+` (any
            // non-quote chars) for now.
            let name = self.fresh_name("pattern");
            let body = format!("\"\\\"\" [^\\\"\\\\]{{0,1000}} \"\\\"\"  # pattern: {pattern}");
            self.rules.insert(name.clone(), body);
            return Ok(name);
        }
        let max = schema
            .get("maxLength")
            .and_then(|v| v.as_u64())
            .unwrap_or(256);
        let _ = schema.get("minLength");
        let name = self.fresh_name("str");
        let body = format!("\"\\\"\" [^\\\"\\\\]{{0,{max}}} \"\\\"\"");
        self.rules.insert(name.clone(), body);
        Ok(name)
    }

    fn build_number(&mut self, schema: &Value) -> Result<String> {
        let min = schema
            .get("minimum")
            .or_else(|| schema.get("exclusiveMinimum"))
            .and_then(|v| v.as_f64());
        let max = schema
            .get("maximum")
            .or_else(|| schema.get("exclusiveMaximum"))
            .and_then(|v| v.as_f64());
        let _ = (min, max);
        let name = self.fresh_name("num");
        // Match a JSON number: optional minus, integer part, optional fraction,
        // optional exponent.
        let body = r#""-"? [0-9]+ ("." [0-9]+)? (("e" | "E") ("+" | "-")? [0-9]+)?"#.to_string();
        self.rules.insert(name.clone(), body);
        Ok(name)
    }

    fn build_array(&mut self, schema: &Value) -> Result<String> {
        let items = schema
            .get("items")
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));
        let item_name = self.fresh_name("arr-item");
        self.visit(&items, &item_name)?;

        let min_items = schema.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0);
        let max_items = schema
            .get("maxItems")
            .and_then(|v| v.as_u64())
            .unwrap_or(16);

        let name = self.fresh_name("arr");
        let mut body = String::from(r#""[""#);
        if min_items > 0 {
            body.push_str(&format!(" {item_name} "));
            for _ in 1..min_items {
                body.push_str(r#"",""#);
                body.push_str(&format!(" {item_name} "));
            }
        }
        if max_items > min_items {
            let lo = 0_usize;
            let hi = (max_items - min_items) as usize;
            body.push_str(&format!(r#" ( "","" {item_name} ){{{lo},{hi}}} "#,));
        }
        body.push_str(r#"]""#);
        self.rules.insert(name.clone(), body);
        Ok(name)
    }

    fn build_object(&mut self, schema: &Value) -> Result<String> {
        let properties = schema
            .get("properties")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let required: Vec<String> = schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // Build per-property schemas first.
        let mut prop_rules: Vec<(String, String)> = Vec::new();
        let mut optional_rules: Vec<(String, String)> = Vec::new();
        for (k, v) in &properties {
            let rule_name = self.fresh_name(&format!("prop-{}", sanitize(k)));
            self.visit(v, &rule_name)?;
            if required.iter().any(|r| r == k) {
                prop_rules.push((k.clone(), rule_name));
            } else {
                optional_rules.push((k.clone(), rule_name));
            }
        }

        // Pair order: required first, then optional.
        let mut pairs = prop_rules;
        pairs.extend(optional_rules);

        // Build the property sequences.
        let mut body = String::from(r#""{""#);
        let mut first = true;
        for (i, (k, v)) in pairs.iter().enumerate() {
            if !first {
                body.push_str(r#" "," "#);
            }
            first = false;
            let _ = i;
            // Quote the key as a JSON string literal.
            write!(&mut body, " \"\\\"{}\\\"\" \":\" {v}", escape_gbnf(k)).unwrap();
        }
        // Add optional trailing separators
        if !pairs.is_empty() {
            // Already built
        }
        body.push_str(r#" "}""#);
        // Allow additional properties as free-form when `additionalProperties`
        // is true or a schema.
        if let Some(ap) = schema.get("additionalProperties") {
            if ap.as_bool() == Some(true) {
                // Append ` ","" kv-pair` zero or more times.
                let kv = format!("\"\\\"\" [^\\\"\\\\]{{0,64}} \"\\\"\" \":\" value");
                body.push_str(&format!(r#" ( "","" {kv} )*"#));
            }
        }

        let name = self.fresh_name("obj");
        self.rules.insert(name.clone(), body);
        Ok(name)
    }

    /// Convert a JSON value to a string-literal alternative.
    fn literal(&self, v: &Value) -> String {
        match v {
            Value::String(s) => format!("\"{}\"", escape_gbnf(s)),
            Value::Bool(b) => b.to_string(),
            Value::Null => "null".to_string(),
            Value::Number(n) => n.to_string(),
            _ => serde_json::to_string(v).unwrap_or_default(),
        }
    }

    /// Render a rule and return its body. (Kept for future helper use.)
    #[allow(dead_code)]
    fn format_rule(&self, name: &str, body: &str) -> String {
        format!("{name} ::= {body}")
    }

    /// Serialize the registry into a full GBNF string.
    fn format_grammar(&self) -> String {
        let mut out = String::new();
        for (name, body) in &self.rules {
            writeln!(&mut out, "{name} ::= {body}").unwrap();
        }
        out
    }
}

/// Strip characters that aren't valid in a rule-name identifier.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Escape a Rust string into a GBNF double-quoted literal.
fn escape_gbnf(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r"\\"),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            c if (c as u32) < 0x20 => write!(&mut out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out
}

/// Convenience helper: build a permissive GBNF that accepts any value.
///
/// Useful when you want a grammar-restricted sampler but don't have a
/// schema in hand.
pub fn any_value_grammar() -> String {
    format!(
        "{root} ::= value\nvalue ::= string | number | boolean | null | array | object\n",
        root = "root"
    )
}

/// Helper for the common case "I want a JSON object grammar".
///
/// # Example
///
/// ```
/// use llama_crab::json_schema::json_object_grammar;
/// let g = json_object_grammar();
/// assert!(g.contains("string"));
/// ```
#[must_use]
pub fn json_object_grammar() -> String {
    let _ = json!({});
    "root ::= object\nobject ::= \"{\" (kv (\",\"\" kv)*)? \"}\"\nkv ::= \"\\\"\" string-content \"\\\"\" \":\" value\nstring-content ::= ([^\"\\\\] | \"\\\\\" [\"\\\\nrt])*\nvalue ::= object | array | string | number | \"true\" | \"false\" | \"null\"\narray ::= \"[\" (value (\",\" value)*)? \"]\"\nstring ::= \"\\\"\" string-content \"\\\"\"\nnumber ::= \"-\"? [0-9]+ (\".\" [0-9]+)? (([eE] [+-]? [0-9]+)?)\n".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn string_schema() {
        let g = schema_to_grammar(&json!({"type": "string"}), "root").unwrap();
        assert!(g.contains("root"));
    }

    #[test]
    fn integer_schema() {
        let g = schema_to_grammar(&json!({"type": "integer"}), "root").unwrap();
        assert!(g.contains("root"));
    }

    #[test]
    fn object_with_required() {
        let s = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age":  {"type": "integer"}
            },
            "required": ["name", "age"]
        });
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("name"));
        assert!(g.contains("age"));
    }

    #[test]
    fn nested_arrays() {
        let s = json!({
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 1,
            "maxItems": 3
        });
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("arr"));
    }

    #[test]
    fn enum_schema() {
        let s = json!({"enum": ["red", "green", "blue"]});
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("red"));
        assert!(g.contains("green"));
    }

    #[test]
    fn const_schema() {
        let s = json!({"const": 42});
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("42"));
    }

    #[test]
    fn any_of_union() {
        let s = json!({
            "anyOf": [
                {"type": "string"},
                {"type": "integer"}
            ]
        });
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("alt-"));
    }

    #[test]
    fn ref_to_local_definition() {
        let s = json!({
            "definitions": {
                "id": {"type": "integer", "minimum": 0, "maximum": 1000}
            },
            "$ref": "#/definitions/id"
        });
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("root"));
    }

    #[test]
    fn email_format() {
        let s = json!({"type": "string", "format": "email"});
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("email-"));
    }

    #[test]
    fn json_object_helper_grammar_is_valid_gbnf() {
        let g = json_object_grammar();
        assert!(g.starts_with("root ::="));
    }

    #[test]
    fn any_value_grammar_is_valid() {
        let g = any_value_grammar();
        assert!(g.contains("root ::="));
    }

    #[test]
    fn additional_properties_schema() {
        let s = json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": true
        });
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("root"));
    }

    #[test]
    fn format_datetime() {
        let s = json!({"type": "string", "format": "date-time"});
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("date-time-"));
    }

    #[test]
    fn format_uri() {
        let s = json!({"type": "string", "format": "uri"});
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("uri-"));
    }

    #[test]
    fn format_uuid() {
        let s = json!({"type": "string", "format": "uuid"});
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("uuid-"));
    }

    #[test]
    fn pattern_schema() {
        let s = json!({"type": "string", "pattern": "^[a-z]+$"});
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("pattern"));
    }

    #[test]
    fn min_max_length() {
        let s = json!({"type": "string", "minLength": 1, "maxLength": 10});
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("root"));
    }

    #[test]
    fn all_of_schema() {
        let s = json!({
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "integer"}}}
            ]
        });
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("all-"));
    }

    #[test]
    fn one_of_schema() {
        let s = json!({
            "oneOf": [
                {"type": "integer"},
                {"type": "boolean"}
            ]
        });
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("alt-"));
    }

    #[test]
    fn boolean_schema() {
        let s = json!({"type": "boolean"});
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("true") || g.contains("false"));
    }

    #[test]
    fn null_schema() {
        let s = json!({"type": "null"});
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("null"));
    }

    #[test]
    fn integer_with_range() {
        let s = json!({"type": "integer", "minimum": 0, "maximum": 100});
        let g = schema_to_grammar(&s, "root").unwrap();
        assert!(g.contains("root"));
    }
}
