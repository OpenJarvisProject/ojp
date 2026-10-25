//! Small utilities used across the crate.

use crate::error::{LlamaError, Result};

/// Convert a `&str` to a `CString`, erroring on interior nul bytes.
pub fn to_cstr(s: &str) -> Result<std::ffi::CString> {
    std::ffi::CString::new(s).map_err(LlamaError::from)
}
