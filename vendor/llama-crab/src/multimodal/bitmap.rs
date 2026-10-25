//! [`MtmdBitmap`] — a single image or audio waveform to feed into an
//! [`MtmdContext`](crate::multimodal::MtmdContext).
//!
//! Plus [`MtmdInputText`] — the textual prompt accompanying the bitmaps.

use std::ptr::NonNull;

use llama_crab_sys as sys;

use crate::error::{LlamaError, Result};

/// A single bitmap (image or audio waveform) to feed into an [`MtmdContext`](crate::multimodal::MtmdContext).
#[derive(Debug)]
pub struct MtmdBitmap {
    handle: NonNull<sys::mtmd_bitmap>,
}

impl MtmdBitmap {
    /// Construct a bitmap from raw pixel data (`nx × ny × 3` bytes, RGB).
    ///
    /// # Errors
    /// Returns an error if the dimensions are invalid.
    pub fn from_image_data(nx: u32, ny: u32, data: &[u8]) -> Result<Self> {
        if data.len() < (nx as usize) * (ny as usize) * 3 {
            return Err(LlamaError::Batch("bitmap data too small".into()));
        }
        let handle = unsafe { sys::mtmd_bitmap_init(nx, ny, data.as_ptr()) };
        NonNull::new(handle)
            .map(|handle| Self { handle })
            .ok_or(LlamaError::Batch("mtmd_bitmap_init returned null".into()))
    }

    /// Construct a bitmap from a float audio buffer.
    pub fn from_audio_data(data: &[f32]) -> Result<Self> {
        let handle = unsafe { sys::mtmd_bitmap_init_from_audio(data.len(), data.as_ptr()) };
        NonNull::new(handle)
            .map(|handle| Self { handle })
            .ok_or(LlamaError::Batch(
                "mtmd_bitmap_init_from_audio returned null".into(),
            ))
    }

    /// Construct a bitmap from an image file on disk.
    ///
    /// Requires the `image` cargo feature. The image is decoded to RGB8
    /// before being passed to mtmd.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let img = image::open(path.as_ref())
            .map_err(|e| LlamaError::Batch(format!("image decode: {e}")))?;
        let rgb8 = img.to_rgb8();
        let (nx, ny) = (rgb8.width(), rgb8.height());
        let bytes = rgb8.into_raw();
        let handle = unsafe { sys::mtmd_bitmap_init(nx, ny, bytes.as_ptr()) };
        NonNull::new(handle)
            .map(|handle| Self { handle })
            .ok_or(LlamaError::Batch("mtmd_bitmap_init returned null".into()))
    }

    /// Width (px) of the image bitmap.
    #[must_use]
    pub fn nx(&self) -> u32 {
        unsafe { sys::mtmd_bitmap_get_nx(self.handle.as_ptr()) }
    }

    /// Height (px) of the image bitmap.
    #[must_use]
    pub fn ny(&self) -> u32 {
        unsafe { sys::mtmd_bitmap_get_ny(self.handle.as_ptr()) }
    }

    /// Number of bytes backing the bitmap.
    #[must_use]
    pub fn n_bytes(&self) -> usize {
        unsafe { sys::mtmd_bitmap_get_n_bytes(self.handle.as_ptr()) }
    }

    /// Whether the bitmap carries audio (as opposed to image) data.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        unsafe { sys::mtmd_bitmap_is_audio(self.handle.as_ptr()) }
    }

    /// Internal: borrow a `*const mtmd_bitmap` (used by
    /// [`MtmdContext::tokenize`](crate::multimodal::MtmdContext::tokenize)).
    pub(crate) fn as_ptr_const(&self) -> *const sys::mtmd_bitmap {
        self.handle.as_ptr().cast_const()
    }
}

impl Drop for MtmdBitmap {
    fn drop(&mut self) {
        // Safety: `handle` is exclusively owned.
        unsafe { sys::mtmd_bitmap_free(self.handle.as_ptr()) };
    }
}

/// Text input paired with tokenization options for
/// [`MtmdContext::tokenize`](crate::multimodal::MtmdContext::tokenize).
#[derive(Debug, Clone)]
pub struct MtmdInputText<'a> {
    /// Raw text (may contain `<image>`/`<audio>` markers).
    pub text: &'a str,
    /// Add special tokens (BOS, chat template tokens).
    pub add_special: bool,
    /// Parse special tokens like `<|image|>`.
    pub parse_special: bool,
}

impl<'a> MtmdInputText<'a> {
    /// Construct a simple text input.
    #[must_use]
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            add_special: true,
            parse_special: true,
        }
    }

    /// Build the C struct.
    pub(crate) fn into_c(self) -> sys::mtmd_input_text {
        // mtmd_input_text does NOT copy the text — the C struct holds a
        // pointer. We leak the CString to give it `'static` lifetime; the
        // caller will typically re-tokenize for each prompt, and the leak
        // is bounded.
        let cstr = std::ffi::CString::new(self.text).expect("text with no nul bytes");
        sys::mtmd_input_text {
            text: cstr.into_raw().cast_const(),
            add_special: self.add_special,
            parse_special: self.parse_special,
        }
    }
}
