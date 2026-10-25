//! Optional HuggingFace `tokenizers` integration.

#[cfg(feature = "hf-tokenizer")]
mod inner {
    use crate::error::{LlamaError, Result};
    use crate::token::LlamaToken;

    /// Bridge between the HuggingFace `tokenizers` crate and the
    /// llama.cpp token vocabulary of a loaded [`crate::model::LlamaModel`].
    #[derive(Debug)]
    pub struct HfTokenizer {
        inner: tokenizers::Tokenizer,
    }

    impl HfTokenizer {
        /// Load a tokenizer from a local `tokenizer.json` file.
        ///
        /// # Errors
        /// Returns an error if the file cannot be read or is not a valid
        /// `tokenizers` JSON.
        pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
            let inner = tokenizers::Tokenizer::from_file(path.as_ref())
                .map_err(|e| LlamaError::Batch(format!("hf tokenizer: {e}")))?;
            Ok(Self { inner })
        }

        /// Tokenize `text` into token ids for a loaded llama.cpp model.
        pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<LlamaToken>> {
            let enc = self
                .inner
                .encode(text, add_bos)
                .map_err(|e| LlamaError::Batch(format!("hf encode: {e}")))?;
            Ok(enc
                .get_ids()
                .iter()
                .map(|&i| LlamaToken(i as i32))
                .collect())
        }

        /// Decode token ids back into a `String`.
        pub fn decode(&self, tokens: &[LlamaToken]) -> Result<String> {
            let ids: Vec<u32> = tokens.iter().map(|t| t.0 as u32).collect();
            self.inner
                .decode(&ids, false)
                .map_err(|e| LlamaError::Batch(format!("hf decode: {e}")))
        }
    }
}

#[cfg(feature = "hf-tokenizer")]
pub use inner::HfTokenizer;
