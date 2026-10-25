//! Prompt formats. Wraps llama-crab's built-in templates and adds ChatML for
//! models that think before answering (Qwen3.5): the assistant turn starts
//! with an already-closed empty think block, which turns the thinking off.
use llama_crab::chat::BuiltinTemplate;
use llama_crab::chat::template::render_builtin;
use llama_crab::chat::ChatMessage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    Builtin(BuiltinTemplate),
    /// ChatML with thinking switched off.
    ChatMlNoThink,
}

impl Template {
    pub const CHATML: Template = Template::Builtin(BuiltinTemplate::ChatMl);

    /// The format named by the OJP_TEMPLATE environment variable (test tools).
    pub fn from_env() -> Template {
        match std::env::var("OJP_TEMPLATE").ok().as_deref().and_then(Template::named) {
            Some(t) => t,
            None => Template::CHATML,
        }
    }

    pub fn named(name: &str) -> Option<Template> {
        Some(match name.to_lowercase().as_str() {
            "chatml" => Template::CHATML,
            "chatml-nothink" | "qwen3.5" | "qwen35" => Template::ChatMlNoThink,
            "llama3" => Template::Builtin(BuiltinTemplate::Llama3),
            "mistral" => Template::Builtin(BuiltinTemplate::MistralInstruct),
            "gemma" => Template::Builtin(BuiltinTemplate::Gemma),
            "phi3" => Template::Builtin(BuiltinTemplate::Phi3),
            _ => return None,
        })
    }

    pub fn render(self, messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        match self {
            Template::Builtin(t) => render_builtin(t, messages, &[], add_generation_prompt),
            Template::ChatMlNoThink => {
                let mut prompt = render_builtin(BuiltinTemplate::ChatMl, messages, &[], add_generation_prompt);
                if add_generation_prompt {
                    prompt.push_str("<think>\n\n</think>\n\n");
                }
                prompt
            }
        }
    }
}
