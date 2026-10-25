//! Cross-encoder re-ranking helper.
//!
//! A re-ranker model scores a `(query, document)` pair; higher score ⇒
//! more relevant. The model must have been trained with
//! `pooling_type = Rank` (the GGUF will declare it).
//!
//! This implementation encodes each pair in a separate forward pass —
//! simple and correct, even if not the most efficient. For a batched
//! implementation use the `seq_*` methods on `LlamaContext` directly.

use crate::batch::LlamaBatch;
use crate::error::Result;
use crate::Llama;

impl Llama {
    /// Score each `(query, document)` pair and return a `Vec<f32>` of
    /// scores in the same order.
    ///
    /// # Errors
    /// Returns an error if encoding fails or the context was not
    /// configured with `pooling_type = Rank`.
    pub fn rerank(&mut self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(documents.len());
        for doc in documents {
            let _ = self.context().seq_rm(0, -1, -1);
            let q = self.model().tokenize(query, true, false)?;
            let d = self.model().tokenize(doc, false, false)?;
            let mut batch = LlamaBatch::new(q.len() + d.len(), 1);
            for (j, &t) in q.iter().chain(d.iter()).enumerate() {
                let logits = j + 1 == q.len() + d.len();
                let _ = batch
                    .add(t, j as i32, &[0], logits)
                    .map_err(crate::error::LlamaError::from)?;
            }
            self.context_mut().encode(&batch)?;
            let emb = self.context().embeddings_seq(0)?;
            scores.push(emb.first().copied().unwrap_or(0.0));
        }
        Ok(scores)
    }
}
