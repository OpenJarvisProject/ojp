//! Buffer-type overrides for specific tensor names.
//!
//! Force a tensor to a particular ggml type by matching on its name.

use llama_crab_sys as sys;

/// A buffer-type override for a single tensor.
#[derive(Clone, Debug)]
pub struct BufferTypeOverride {
    /// Tensor name to match (exact).
    pub tensor_name: String,
    /// Target ggml type.
    pub ggml_type: GgmlType,
}

/// The subset of `ggml_type` that we expose at the binding level. The
/// raw enum is also available via [`llama_crab_sys::ggml_type`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgmlType {
    /// 32-bit float.
    F32,
    /// 16-bit float.
    F16,
    /// 4-bit quantization.
    Q4_0,
    /// 4-bit quantization (variant 1).
    Q4_1,
    /// 5-bit quantization (variant 0).
    Q5_0,
    /// 5-bit quantization (variant 1).
    Q5_1,
    /// 8-bit quantization.
    Q8_0,
    /// K-quantization 2_0.
    Q2K,
    /// K-quantization 3_0.
    Q3K,
    /// K-quantization 4_0.
    Q4K,
    /// K-quantization 5_0.
    Q5K,
    /// K-quantization 6_0.
    Q6K,
}

impl GgmlType {
    /// Convert to the raw `ggml_type` enum value.
    #[must_use]
    pub const fn as_raw(self) -> sys::ggml_type {
        match self {
            Self::F32 => sys::ggml_type::GGML_TYPE_F32,
            Self::F16 => sys::ggml_type::GGML_TYPE_F16,
            Self::Q4_0 => sys::ggml_type::GGML_TYPE_Q4_0,
            Self::Q4_1 => sys::ggml_type::GGML_TYPE_Q4_1,
            Self::Q5_0 => sys::ggml_type::GGML_TYPE_Q5_0,
            Self::Q5_1 => sys::ggml_type::GGML_TYPE_Q5_1,
            Self::Q8_0 => sys::ggml_type::GGML_TYPE_Q8_0,
            Self::Q2K => sys::ggml_type::GGML_TYPE_Q2_K,
            Self::Q3K => sys::ggml_type::GGML_TYPE_Q3_K,
            Self::Q4K => sys::ggml_type::GGML_TYPE_Q4_K,
            Self::Q5K => sys::ggml_type::GGML_TYPE_Q5_K,
            Self::Q6K => sys::ggml_type::GGML_TYPE_Q6_K,
        }
    }
}

impl BufferTypeOverride {
    /// Construct a new override.
    #[must_use]
    pub fn new(tensor_name: impl Into<String>, ggml_type: GgmlType) -> Self {
        Self {
            tensor_name: tensor_name.into(),
            ggml_type,
        }
    }
}

/// Internal: serialize to the C struct.
///
/// Note: the C struct's `pattern` field is a `*const c_char`. The
/// backing `CString` is leaked (bounded by the lifetime of the
/// override list).
pub(crate) fn to_c_array(items: &[BufferTypeOverride]) -> Vec<sys::llama_model_tensor_override> {
    items
        .iter()
        .map(|o| {
            let leaked = std::ffi::CString::new(o.tensor_name.as_str())
                .unwrap()
                .into_raw();
            sys::llama_model_tensor_override {
                pattern: leaked.cast_const(),
                type_: o.ggml_type.as_raw(),
            }
        })
        .collect()
}
