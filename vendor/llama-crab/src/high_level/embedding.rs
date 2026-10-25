//! High-level embedding helpers.

use crate::batch::LlamaBatch;
use crate::context::LlamaContextParams;
use crate::error::{LlamaError, Result};
use crate::Llama;

impl Llama {
    /// Tokenize + encode `text` and return its L2-normalized embedding.
    ///
    /// The context must have been created with
    /// `LlamaContextParams::with_embeddings(true)`. If you want raw
    /// (un-normalized) embeddings, use the `embeddings` method on
    /// `LlamaContext` directly.
    ///
    /// # Errors
    /// Returns an error if embeddings are not enabled, the model fails
    /// to encode, or the embedding slice cannot be read.
    pub fn embed(&mut self, text: &str, normalize: bool) -> Result<Vec<f32>> {
        let _ = self.context().seq_rm(0, -1, -1);
        let tokens = self.model().tokenize(text, true, false)?;
        if tokens.is_empty() {
            return Err(LlamaError::Batch("empty tokenization".into()));
        }
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, &t) in tokens.iter().enumerate() {
            batch
                .add(t, i as i32, &[0], true)
                .map_err(LlamaError::from)?;
        }
        self.context_mut().encode(&batch)?;
        let mut v = self.context().embeddings_seq(0)?.to_vec();
        if normalize {
            LlamaContextParams::l2_normalize(&mut v);
        }
        Ok(v)
    }
}

impl LlamaContextParams {
    /// In-place L2 normalization of an embedding vector.
    pub fn l2_normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }
}
