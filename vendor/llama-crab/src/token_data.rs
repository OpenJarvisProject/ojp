//! [`LlamaTokenData`] and [`LlamaTokenDataArray`].

use llama_crab_sys as sys;

use crate::token::LlamaToken;

/// A `(id, logit, p)` triple representing a token in the candidate set.
#[derive(Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct LlamaTokenData(pub sys::llama_token_data);

impl LlamaTokenData {
    /// Construct from id, logit and probability.
    #[must_use]
    pub fn new(id: LlamaToken, logit: f32, p: f32) -> Self {
        Self(sys::llama_token_data { id: id.0, logit, p })
    }

    /// Token id.
    #[must_use]
    pub fn id(&self) -> LlamaToken {
        LlamaToken(self.0.id)
    }

    /// Set the token id.
    pub fn set_id(&mut self, id: LlamaToken) {
        self.0.id = id.0;
    }

    /// Logit value.
    #[must_use]
    pub fn logit(&self) -> f32 {
        self.0.logit
    }

    /// Set the logit value.
    pub fn set_logit(&mut self, logit: f32) {
        self.0.logit = logit;
    }

    /// Probability in `[0, 1]` (filled in by `apply_sampler`).
    #[must_use]
    pub fn p(&self) -> f32 {
        self.0.p
    }

    /// Set the probability.
    pub fn set_p(&mut self, p: f32) {
        self.0.p = p;
    }
}

impl std::fmt::Debug for LlamaTokenData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaTokenData")
            .field("id", &self.id())
            .field("logit", &self.logit())
            .field("p", &self.p())
            .finish()
    }
}

/// A mutable array of [`LlamaTokenData`] with selection/sort metadata.
#[derive(Debug)]
pub struct LlamaTokenDataArray {
    inner: sys::llama_token_data_array,
    // The boxed slice keeps the backing storage alive for the C struct.
    _data: Box<[LlamaTokenData]>,
}

impl LlamaTokenDataArray {
    /// Construct a new `LlamaTokenDataArray` with `sorted = false` and
    /// `selected = -1`.
    #[must_use]
    pub fn new(data: Vec<LlamaTokenData>) -> Self {
        let mut data: Box<[LlamaTokenData]> = data.into_boxed_slice();
        let ptr = data.as_mut_ptr().cast::<sys::llama_token_data>();
        let inner = sys::llama_token_data_array {
            data: ptr,
            size: data.len(),
            selected: -1,
            sorted: false,
        };
        Self { inner, _data: data }
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.size
    }

    /// True if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.size == 0
    }

    /// Index of the selected token (-1 if none).
    #[must_use]
    pub fn selected(&self) -> i64 {
        self.inner.selected
    }

    /// Borrow the underlying `&[sys::llama_token_data]`.
    #[must_use]
    pub fn as_raw(&self) -> &[sys::llama_token_data] {
        // Safety: the boxed slice has the same length as the C array.
        unsafe { std::slice::from_raw_parts(self.inner.data, self.inner.size) }
    }

    /// Borrow the `&[LlamaTokenData]` wrapper.
    #[must_use]
    pub fn data(&self) -> &[LlamaTokenData] {
        &self._data
    }

    /// Mutable borrow of the inner C struct (private — use higher-level
    /// wrappers when adding new operations).
    #[allow(dead_code)]
    pub(crate) fn inner_mut(&mut self) -> &mut sys::llama_token_data_array {
        &mut self.inner
    }

    /// Borrow a raw `*mut` for the C API.
    pub(crate) fn as_mut_ptr(&mut self) -> *mut sys::llama_token_data_array {
        &mut self.inner
    }
}
