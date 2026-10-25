//! Reusable batching primitive.

use llama_crab_sys as sys;

use crate::error::LlamaError;
use crate::token::LlamaToken;

/// A `llama_batch` wrapper. Owns the underlying C struct and its memory.
#[derive(Debug)]
pub struct LlamaBatch {
    raw: sys::llama_batch,
    // The C struct borrows from these vectors; we keep them alive.
    tokens: Vec<sys::llama_token>,
    positions: Vec<sys::llama_pos>,
    n_seq_id: Vec<i32>,
    seq_ids: Vec<Vec<sys::llama_seq_id>>,
    seq_ids_ptrs: Vec<*mut sys::llama_seq_id>,
    logits: Vec<i8>,
    allocated: bool,
}

/// Reason a token could not be added to the batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchAddError {
    /// The batch is full.
    InsufficientSpace(usize),
    /// Attempted to add nothing.
    Empty,
}

impl std::fmt::Display for BatchAddError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientSpace(n) => write!(f, "batch only has space for {n} tokens"),
            Self::Empty => write!(f, "no token to add"),
        }
    }
}

impl std::error::Error for BatchAddError {}

impl LlamaBatch {
    /// Allocate a batch that can hold up to `n_tokens`.
    ///
    /// `n_seq_max` is the maximum number of sequences any single token can
    /// belong to (typically 1 for single-stream inference).
    #[must_use]
    pub fn new(n_tokens: usize, n_seq_max: i32) -> Self {
        let tokens = vec![0_i32; n_tokens];
        let positions = vec![0_i32; n_tokens];
        let n_seq_id = vec![n_seq_max; n_tokens];
        let mut seq_ids = Vec::with_capacity(n_tokens);
        let mut seq_ids_ptrs: Vec<*mut sys::llama_seq_id> = Vec::with_capacity(n_tokens);
        for _ in 0..n_tokens {
            let mut v: Vec<i32> = vec![0; n_seq_max as usize];
            seq_ids_ptrs.push(v.as_mut_ptr());
            seq_ids.push(v);
        }
        let logits = vec![0_i8; n_tokens];
        let raw = sys::llama_batch {
            n_tokens: 0,
            token: tokens.as_ptr().cast_mut(),
            embd: std::ptr::null_mut(),
            pos: positions.as_ptr().cast_mut(),
            n_seq_id: n_seq_id.as_ptr().cast_mut(),
            seq_id: seq_ids_ptrs.as_ptr().cast_mut(),
            logits: logits.as_ptr().cast_mut(),
        };
        Self {
            raw,
            tokens,
            positions,
            n_seq_id,
            seq_ids,
            seq_ids_ptrs,
            logits,
            allocated: true,
        }
    }

    /// Construct a single-sequence batch of one token. Convenience for the
    /// most common decode step.
    #[must_use]
    pub fn one(token: LlamaToken, pos: i32, seq_id: i32, logits: bool) -> Self {
        let mut b = Self::new(1, 1);
        b.add(token, pos, &[seq_id], logits).expect("capacity 1");
        b
    }

    /// Number of tokens currently in the batch.
    #[must_use]
    pub fn n_tokens(&self) -> i32 {
        self.raw.n_tokens
    }

    /// Reset the batch so it can be reused without reallocating.
    pub fn clear(&mut self) {
        self.raw.n_tokens = 0;
    }

    /// Append a single token to the batch.
    ///
    /// # Errors
    /// Returns [`BatchAddError::InsufficientSpace`] if the batch is full.
    pub fn add(
        &mut self,
        token: LlamaToken,
        pos: i32,
        seq_ids: &[i32],
        logits: bool,
    ) -> std::result::Result<(), BatchAddError> {
        let idx = self.raw.n_tokens as usize;
        if idx >= self.tokens.len() {
            return Err(BatchAddError::InsufficientSpace(self.tokens.len()));
        }
        if seq_ids.is_empty() {
            return Err(BatchAddError::Empty);
        }
        // Storage vectors are immutable for borrow-checker; we go through raw
        // pointers for the mutation because the C batch only reads from them.
        // Safety: idx < capacity and the vectors outlive the batch.
        unsafe {
            let mut_ptr = self.tokens.as_ptr().cast_mut();
            std::ptr::write(mut_ptr.add(idx), token.0);
            let pos_ptr = self.positions.as_ptr().cast_mut();
            std::ptr::write(pos_ptr.add(idx), pos);
            let logits_ptr = self.logits.as_ptr().cast_mut();
            std::ptr::write(logits_ptr.add(idx), i8::from(logits));
        }
        for (i, &sid) in seq_ids.iter().enumerate() {
            if i < self.seq_ids[idx].len() {
                self.seq_ids[idx][i] = sid;
            }
        }
        self.raw.n_tokens += 1;
        Ok(())
    }

    /// Borrow the underlying C struct (read-only).
    pub(crate) fn raw(&self) -> &sys::llama_batch {
        &self.raw
    }
}

/// Convert a `BatchAddError` into the crate-wide [`LlamaError`].
impl From<BatchAddError> for LlamaError {
    fn from(e: BatchAddError) -> Self {
        Self::Batch(e.to_string())
    }
}
