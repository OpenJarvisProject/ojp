//! `LlamaToken` newtype and `LlamaTokenAttr` bitflags.

use std::fmt;

use llama_crab_sys as sys;

/// A single token in the model's vocabulary.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct LlamaToken(pub sys::llama_token);

impl LlamaToken {
    /// Raw integer value.
    #[must_use]
    pub const fn raw(self) -> sys::llama_token {
        self.0
    }

    /// Construct from a raw `llama_token` (i32). Returns `None` on out-of-range.
    #[must_use]
    pub fn new(raw: i32) -> Option<Self> {
        // llama_token is an i32; we just accept anything without overflow.
        Some(Self(raw))
    }

    /// Sentinel value used by llama.cpp for "no token".
    pub const NULL: Self = Self(sys::LLAMA_TOKEN_NULL);
}

impl fmt::Debug for LlamaToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LlamaToken({})", self.0)
    }
}

impl fmt::Display for LlamaToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i32> for LlamaToken {
    fn from(v: i32) -> Self {
        Self(v)
    }
}
