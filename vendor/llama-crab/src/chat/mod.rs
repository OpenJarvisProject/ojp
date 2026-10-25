//! Chat message types, templates and tool calling.
//!
//! This module is the public entry-point for everything related to
//! multi-turn chat: messages, templates (a Jinja2 subset), tool/function
//! calling, and the streaming state needed to parse the model's output.
//!
//! # Quick tour
//!
//! ```
//! use llama_crab::chat::{ChatMessage, Role, render_template};
//!
//! let prompt = render_template(
//!     "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}assistant:",
//!     &[ChatMessage::new(Role::User, "Hi")],
//!     &[], // no tools
//!     true, // add generation prompt
//! ).unwrap();
//! assert!(prompt.contains("user: Hi"));
//! assert!(prompt.ends_with("assistant:"));
//! ```

pub mod message;
pub mod parser;
pub mod template;
pub mod tool_call;

pub use message::{ChatMessage, Role};
pub use parser::ChatParseState;
pub use template::{
    detect_chat_format, render_builtin, render_template, BuiltinTemplate, TemplateError,
};
pub use tool_call::{ToolCall, ToolDefinition, ToolParseError, ToolParser};
