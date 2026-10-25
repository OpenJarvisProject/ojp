//! State (KV cache) save and load.
//!
//! Two flavors are provided:
//!
//! * Whole-context state via [`LlamaContext::state_size`],
//!   [`LlamaContext::state_to_bytes`] and [`LlamaContext::load_state`].
//! * File-based via [`LlamaContext::state_save_file`] and
//!   [`LlamaContext::state_load_file`].

use std::path::Path;

use crate::context::LlamaContext;
use crate::error::{LlamaError, Result};

impl LlamaContext {
    /// Return the number of bytes required to serialize the full state.
    #[must_use]
    pub fn state_size(&self) -> usize {
        unsafe { llama_crab_sys::llama_state_get_size(self.raw()) }
    }

    /// Serialize the full state to a freshly-allocated `Vec<u8>`.
    ///
    /// # Errors
    /// Returns an error if llama.cpp refuses to write the data.
    pub fn state_to_bytes(&self) -> Result<Vec<u8>> {
        let size = self.state_size();
        let mut buf = vec![0_u8; size];
        let written =
            unsafe { llama_crab_sys::llama_state_get_data(self.raw(), buf.as_mut_ptr(), size) };
        if written != size {
            return Err(LlamaError::Ffi(-1));
        }
        Ok(buf)
    }

    /// Load state from a byte buffer.
    ///
    /// # Errors
    /// Returns an error if llama.cpp rejects the data (corrupted, wrong
    /// format, etc.).
    pub fn load_state(&mut self, bytes: &[u8]) -> Result<()> {
        let n = unsafe {
            llama_crab_sys::llama_state_set_data(self.raw(), bytes.as_ptr(), bytes.len())
        };
        if n != bytes.len() {
            return Err(LlamaError::Ffi(-1));
        }
        Ok(())
    }

    /// Save the full state to a file at `path`.
    ///
    /// # Errors
    /// Returns an error if the file cannot be created or the write fails.
    pub fn state_save_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let cstr = std::ffi::CString::new(path.as_ref().display().to_string())?;
        // Safety: `cstr` is a valid null-terminated C string.
        let tokens_to_copy = self.n_ctx() as usize;
        let ok = unsafe {
            llama_crab_sys::llama_state_save_file(
                self.raw(),
                cstr.as_ptr(),
                std::ptr::null(), // no token filter
                tokens_to_copy,
            )
        };
        if !ok {
            return Err(LlamaError::Io(std::io::Error::other("state_save_file")));
        }
        Ok(())
    }

    /// Load the full state from a file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or its contents are
    /// rejected by llama.cpp.
    pub fn state_load_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let cstr = std::ffi::CString::new(path.as_ref().display().to_string())?;
        let mut token_count: usize = 0;
        let ok = unsafe {
            llama_crab_sys::llama_state_load_file(
                self.raw(),
                cstr.as_ptr(),
                std::ptr::null_mut(),
                0,
                &mut token_count,
            )
        };
        if !ok {
            return Err(LlamaError::Io(std::io::Error::other("state_load_file")));
        }
        Ok(())
    }
}
