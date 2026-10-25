//! Forwarding llama.cpp/ggml logs into the `tracing` ecosystem.

use llama_crab_sys as sys;

/// Options controlling the log forwarder.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogOptions {
    /// Forward logs from the GGML library.
    pub forward_ggml: bool,
    /// Forward logs from llama.cpp itself.
    pub forward_llama: bool,
}

/// Install a callback that forwards llama.cpp and ggml log messages to the
/// [`tracing`] macros (`trace!`, `debug!`, `info!`, `warn!`, `error!`).
///
/// This must be called after a [`crate::LlamaBackend`] has been initialized.
/// Calling it multiple times replaces the previous callback.
///
/// v0.1 stub: the callback is installed but currently drops messages. The
/// real implementation will route through a stateful C function exported by
/// the FFI crate.
pub fn send_logs_to_tracing(opts: LogOptions) {
    let _ = opts;
    let cb: sys::ggml_log_callback = Some(noop_log);
    unsafe { sys::llama_log_set(cb, std::ptr::null_mut()) };
}

unsafe extern "C" fn noop_log(
    _level: sys::ggml_log_level,
    _text: *const std::os::raw::c_char,
    _ud: *mut std::os::raw::c_void,
) {
}
