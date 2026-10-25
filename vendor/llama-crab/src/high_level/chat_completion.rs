//! Chat completion driver — renders messages into a prompt via a
//! [`BuiltinTemplate`] and runs [`create_completion`].

use crate::error::Result;

use super::completion::{
    create_completion, create_completion_stream, CompletionChunk, CompletionOptions, StreamControl,
};
use super::Llama;
pub use crate::chat::ChatMessage;
use crate::chat::{render_builtin, BuiltinTemplate, ToolDefinition};

/// Run a single-turn chat completion with the **Plain** template.
pub fn create_chat_completion(
    llama: &mut Llama,
    messages: &[ChatMessage],
    max_tokens: usize,
) -> Result<ChatMessage> {
    create_chat_completion_with(llama, messages, BuiltinTemplate::Plain, &[], max_tokens)
}

/// Run a single-turn chat completion with a chosen built-in template and
/// optional tool definitions.
pub fn create_chat_completion_with(
    llama: &mut Llama,
    messages: &[ChatMessage],
    template: BuiltinTemplate,
    tools: &[ToolDefinition],
    max_tokens: usize,
) -> Result<ChatMessage> {
    let prompt = render_builtin(template, messages, tools, true);
    let resp = create_completion(llama, &prompt, max_tokens)?;
    Ok(ChatMessage::new(crate::chat::Role::Assistant, resp.text))
}

/// Stream a single-turn chat completion with the **Plain** template.
pub fn create_chat_completion_stream<F>(
    llama: &mut Llama,
    messages: &[ChatMessage],
    max_tokens: usize,
    on_chunk: F,
) -> Result<ChatMessage>
where
    F: FnMut(CompletionChunk) -> StreamControl,
{
    create_chat_completion_stream_with(
        llama,
        messages,
        BuiltinTemplate::Plain,
        &[],
        CompletionOptions::new(max_tokens),
        on_chunk,
    )
}

/// Stream a single-turn chat completion with a chosen built-in template,
/// optional tool definitions, and completion options.
pub fn create_chat_completion_stream_with<F>(
    llama: &mut Llama,
    messages: &[ChatMessage],
    template: BuiltinTemplate,
    tools: &[ToolDefinition],
    options: CompletionOptions,
    on_chunk: F,
) -> Result<ChatMessage>
where
    F: FnMut(CompletionChunk) -> StreamControl,
{
    let prompt = render_builtin(template, messages, tools, true);
    let resp = create_completion_stream(llama, &prompt, options, on_chunk)?;
    Ok(ChatMessage::new(crate::chat::Role::Assistant, resp.text))
}
