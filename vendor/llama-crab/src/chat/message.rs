//! [`ChatMessage`] and [`Role`] types.

/// Role of a chat message in a multi-turn conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Role {
    /// System prompt.
    #[serde(rename = "system")]
    System,
    /// User turn.
    #[serde(rename = "user")]
    User,
    /// Assistant turn.
    #[serde(rename = "assistant")]
    Assistant,
    /// Tool/function result.
    #[serde(rename = "tool")]
    Tool,
}

impl Role {
    /// Conventional wire name (`system`, `user`, `assistant`, `tool`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Role {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "system" => Self::System,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "tool" => Self::Tool,
            other => return Err(format!("unknown chat role: {other}")),
        })
    }
}

/// A single chat message.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    /// Role (system/user/assistant/tool).
    pub role: Role,
    /// Textual content. Multimodal content is supported by
    /// `crate::multimodal` (feature `mtmd`).
    pub content: String,
    /// Optional tool call id (only meaningful when `role == Tool`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional tool calls emitted by the assistant in this message.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Optional name (for function-style messages).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Create a new message.
    #[must_use]
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_call_id: None,
            tool_calls: Vec::new(),
            name: None,
        }
    }

    /// Mark this message as a tool result (sets `role = Tool` and the id).
    #[must_use]
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: Vec::new(),
            name: None,
        }
    }

    /// Builder-style: add a tool call.
    #[must_use]
    pub fn with_tool_call(mut self, call: ToolCall) -> Self {
        self.tool_calls.push(call);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::tool_call::ToolCall;
    use serde_json::json;
    use std::str::FromStr;

    #[test]
    fn new_constructs() {
        let m = ChatMessage::new(Role::User, "hi");
        assert_eq!(m.role, Role::User);
        assert_eq!(m.content, "hi");
        assert!(m.tool_call_id.is_none());
        assert!(m.tool_calls.is_empty());
        assert!(m.name.is_none());
    }

    #[test]
    fn tool_result_constructs() {
        let m = ChatMessage::tool_result("call_1", "ok");
        assert_eq!(m.role, Role::Tool);
        assert_eq!(m.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(m.content, "ok");
    }

    #[test]
    fn with_tool_call_appends() {
        let c = ToolCall::new("id_1", "f", json!({}));
        let m = ChatMessage::new(Role::Assistant, "x").with_tool_call(c.clone());
        assert_eq!(m.tool_calls.len(), 1);
        assert_eq!(m.tool_calls[0].id, "id_1");
    }

    #[test]
    fn role_from_str() {
        assert_eq!(Role::from_str("system").unwrap(), Role::System);
        assert_eq!(Role::from_str("user").unwrap(), Role::User);
        assert_eq!(Role::from_str("assistant").unwrap(), Role::Assistant);
        assert_eq!(Role::from_str("tool").unwrap(), Role::Tool);
        assert!(Role::from_str("nope").is_err());
    }

    #[test]
    fn role_serialize_round_trip() {
        let json = serde_json::to_string(&Role::User).unwrap();
        assert_eq!(json, "\"user\"");
        let r: Role = serde_json::from_str("\"assistant\"").unwrap();
        assert_eq!(r, Role::Assistant);
    }
}

// Avoid orphan-rule violations; we need `ToolCall` in scope.
use crate::chat::tool_call::ToolCall;
