//! [`MtmdContext`] — the high-level entry point for multimodal
//! inference, plus its parameters struct.

use std::path::Path;
use std::ptr::NonNull;

use llama_crab_sys as sys;

use super::bitmap::{MtmdBitmap, MtmdInputText};
use super::chunks::{MtmdInputChunk, MtmdInputChunks};
use crate::error::{LlamaError, Result};
use crate::model::LlamaModel;

/// Return the default media marker string used by `mtmd` — currently
/// `"<__media__>"`. The prompt passed to
/// [`MtmdContext::tokenize`] must contain one occurrence of this
/// string per bitmap supplied.
///
/// # Example
/// ```no_run
/// use llama_crab::multimodal::default_media_marker;
/// let marker = default_media_marker();
/// assert!(marker.starts_with('<'));
/// ```
#[must_use]
pub fn default_media_marker() -> &'static str {
    // Safety: `mtmd_default_marker` returns a pointer to a static
    // string literal owned by libmtmd; it is valid for the lifetime
    // of the process.
    let ptr = unsafe { sys::mtmd_default_marker() };
    if ptr.is_null() {
        "<__media__>"
    } else {
        // Safety: ptr is a valid C string with static lifetime.
        unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_str()
            .unwrap_or("<__media__>")
    }
}

/// Initialization parameters for an [`MtmdContext`].
#[derive(Debug, Clone)]
pub struct MtmdContextParams {
    /// Whether to use GPU for the projector model.
    pub use_gpu: bool,
    /// Print timing information on shutdown.
    pub print_timings: bool,
    /// Number of threads to use for the projector.
    pub n_threads: i32,
}

impl Default for MtmdContextParams {
    fn default() -> Self {
        Self {
            use_gpu: true,
            print_timings: false,
            n_threads: 1,
        }
    }
}

impl MtmdContextParams {
    /// Convert to the C struct.
    fn to_c(&self) -> sys::mtmd_context_params {
        let mut p = unsafe { sys::mtmd_context_params_default() };
        p.use_gpu = self.use_gpu;
        p.print_timings = self.print_timings;
        p.n_threads = self.n_threads;
        p
    }
}

/// An initialized multimodal context, bound to a text [`LlamaModel`].
#[derive(Debug)]
pub struct MtmdContext {
    pub(crate) handle: NonNull<sys::mtmd_context>,
}

impl MtmdContext {
    /// Initialize the multimodal context from an `mmproj` GGUF file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or the projector does not
    /// match the supplied text model.
    pub fn init_from_file(mmproj_path: impl AsRef<Path>, text_model: &LlamaModel) -> Result<Self> {
        Self::init_from_file_with(mmproj_path, text_model, MtmdContextParams::default())
    }

    /// Initialize with custom parameters.
    pub fn init_from_file_with(
        mmproj_path: impl AsRef<Path>,
        text_model: &LlamaModel,
        params: MtmdContextParams,
    ) -> Result<Self> {
        let cpath = std::ffi::CString::new(mmproj_path.as_ref().display().to_string())?;
        let handle =
            unsafe { sys::mtmd_init_from_file(cpath.as_ptr(), text_model.raw(), params.to_c()) };
        NonNull::new(handle)
            .map(|handle| Self { handle })
            .ok_or_else(|| {
                LlamaError::ModelLoad(format!(
                    "mtmd_init_from_file({}) failed",
                    mmproj_path.as_ref().display()
                ))
            })
    }

    /// Whether the projector requires non-causal attention.
    #[must_use]
    pub fn decode_use_non_causal(&self, chunk: &MtmdInputChunk<'_>) -> bool {
        unsafe { sys::mtmd_decode_use_non_causal(self.handle.as_ptr(), chunk.as_ptr()) }
    }

    /// Whether the projector uses M-RoPE.
    #[must_use]
    pub fn decode_use_mrope(&self) -> bool {
        unsafe { sys::mtmd_decode_use_mrope(self.handle.as_ptr()) }
    }

    /// Whether the projector supports image inputs.
    #[must_use]
    pub fn support_vision(&self) -> bool {
        unsafe { sys::mtmd_support_vision(self.handle.as_ptr()) }
    }

    /// Whether the projector supports audio inputs.
    #[must_use]
    pub fn support_audio(&self) -> bool {
        unsafe { sys::mtmd_support_audio(self.handle.as_ptr()) }
    }

    /// Audio sample rate expected by the projector.
    #[must_use]
    pub fn audio_sample_rate(&self) -> i32 {
        unsafe { sys::mtmd_get_audio_sample_rate(self.handle.as_ptr()) }
    }

    /// Tokenize `text` along with the supplied bitmaps into a list of chunks.
    ///
    /// # Errors
    /// Returns an error on tokenization failure.
    pub fn tokenize(
        &self,
        text: MtmdInputText<'_>,
        bitmaps: &[&MtmdBitmap],
    ) -> Result<MtmdInputChunks> {
        let c_text = text.into_c();
        let mut bitmap_ptrs: Vec<*const sys::mtmd_bitmap> =
            bitmaps.iter().map(|b| b.as_ptr_const()).collect();
        let chunks = MtmdInputChunks::new()?;
        let rc = unsafe {
            sys::mtmd_tokenize(
                self.handle.as_ptr(),
                chunks.handle.as_ptr(),
                &c_text,
                bitmap_ptrs.as_mut_ptr(),
                bitmap_ptrs.len(),
            )
        };
        if rc != 0 {
            return Err(LlamaError::Batch(format!("mtmd_tokenize: {rc}")));
        }
        Ok(chunks)
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        // Safety: `handle` is exclusively owned and was returned by
        // `mtmd_init_from_file`.
        unsafe { sys::mtmd_free(self.handle.as_ptr()) };
    }
}

// Safety: the mtmd context is not thread-safe per llama.cpp docs.
unsafe impl Send for MtmdContext {}
