//! KV overrides for model loading.
//!
//! Allow overriding the value of specific GGUF metadata keys before the
//! model is loaded. Useful for forcing an architecture or disabling a
//! capability without editing the GGUF file.

use llama_crab_sys as sys;

/// A value override for a GGUF metadata key.
#[derive(Clone, Debug)]
pub enum ParamOverrideValue {
    /// Boolean override.
    Bool(bool),
    /// 64-bit floating-point override.
    Float(f64),
    /// 64-bit signed integer override.
    Int(i64),
    /// String override (truncated to 127 bytes + NUL).
    Str(String),
}

impl ParamOverrideValue {
    /// Construct the C-ABI `llama_model_kv_override` struct from this value.
    pub(crate) fn as_c(&self) -> sys::llama_model_kv_override {
        let mut out: sys::llama_model_kv_override = unsafe { std::mem::zeroed() };
        match self {
            Self::Bool(b) => {
                out.tag = sys::llama_model_kv_override_type::LLAMA_KV_OVERRIDE_TYPE_BOOL;
                out.__bindgen_anon_1.val_bool = *b;
            }
            Self::Float(f) => {
                out.tag = sys::llama_model_kv_override_type::LLAMA_KV_OVERRIDE_TYPE_FLOAT;
                out.__bindgen_anon_1.val_f64 = *f;
            }
            Self::Int(i) => {
                out.tag = sys::llama_model_kv_override_type::LLAMA_KV_OVERRIDE_TYPE_INT;
                out.__bindgen_anon_1.val_i64 = *i;
            }
            Self::Str(s) => {
                out.tag = sys::llama_model_kv_override_type::LLAMA_KV_OVERRIDE_TYPE_STR;
                let bytes = s.as_bytes();
                let n = bytes.len().min(127);
                let dst = unsafe { &mut out.__bindgen_anon_1.val_str };
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.as_mut_ptr().cast(), n);
                    dst[n] = 0;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_override() {
        let c = ParamOverrideValue::Bool(true).as_c();
        assert_eq!(
            c.tag,
            sys::llama_model_kv_override_type::LLAMA_KV_OVERRIDE_TYPE_BOOL
        );
    }

    #[test]
    fn string_override_truncates() {
        let big = "x".repeat(200);
        let c = ParamOverrideValue::Str(big).as_c();
        assert_eq!(
            c.tag,
            sys::llama_model_kv_override_type::LLAMA_KV_OVERRIDE_TYPE_STR
        );
    }
}
