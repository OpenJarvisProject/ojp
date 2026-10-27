//! What only the host platform can do. On Android the app installs one function
//! that calls into Java (`PhoneActions.invoke`); everywhere else nothing is installed
//! and calls fail with a plain message. Keeping it to one `(method, argument) -> text`
//! function means ojp-core never touches JNI.
//!
//! Optional Shizuku support lives behind the same function: Shizuku is a helper app that
//! lets other apps run commands with the same rights as `adb shell`, after the user
//! starts it once (over wireless debugging) and allows the app.
use std::sync::Mutex;

use anyhow::{Result, bail};

pub type Native = fn(&str, &str) -> Result<String>;

static NATIVE: Mutex<Option<Native>> = Mutex::new(None);

/// Install the platform's call-in function (the Android app does this at start-up).
pub fn install(f: Native) {
    *NATIVE.lock().unwrap_or_else(|e| e.into_inner()) = Some(f);
}

/// Is there a platform layer (Android)?
pub fn available() -> bool {
    NATIVE.lock().unwrap_or_else(|e| e.into_inner()).is_some()
}

/// Run a platform action: `volume`, `mute`, `media`, `open_app`, `open_url`, `remind`, `battery`,
/// `shizuku_state`, `shizuku_request`, `shell` ... (see `PhoneActions.java`).
pub fn call(method: &str, arg: &str) -> Result<String> {
    let f = *NATIVE.lock().unwrap_or_else(|e| e.into_inner());
    match f {
        Some(f) => f(method, arg),
        None => bail!("that isn't available on this device"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shizuku {
    /// Not installed, or installed but not running.
    Unavailable,
    /// Running, but this app hasn't been allowed yet.
    NeedsPermission,
    Ready,
    /// A Shizuku too old to use.
    TooOld,
}

pub fn shizuku() -> Shizuku {
    match call("shizuku_state", "").as_deref() {
        Ok("ready") => Shizuku::Ready,
        Ok("permission") => Shizuku::NeedsPermission,
        Ok("old") => Shizuku::TooOld,
        _ => Shizuku::Unavailable,
    }
}

/// Run a shell command with adb-level rights through Shizuku. Fails with a message saying what to
/// do when Shizuku isn't ready, or with the command's own output when it exits non-zero.
pub fn shell(command: &str) -> Result<String> {
    match shizuku() {
        Shizuku::Ready => {}
        Shizuku::NeedsPermission => bail!("Shizuku is running but hasn't allowed Open Jarvis yet: open Settings and tap the Shizuku button"),
        Shizuku::TooOld => bail!("the installed Shizuku is too old; update it"),
        Shizuku::Unavailable => bail!("that needs Shizuku: install it, start it (it can start itself over wireless debugging), then allow Open Jarvis in Settings"),
    }
    let out = call("shell", command)?;
    let (status, body) = out.split_once('\n').unwrap_or((&out, ""));
    let body = body.trim();
    match status.strip_prefix("exit=").and_then(|n| n.trim().parse::<i32>().ok()) {
        Some(0) | None => Ok(body.to_string()),
        Some(code) => bail!("the command failed (exit {code}){}", if body.is_empty() { String::new() } else { format!(": {body}") }),
    }
}

/// Single-quote `s` for `sh -c`.
pub fn quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    pub static CALLS: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
    pub static REPLY: Mutex<Option<String>> = Mutex::new(None);

    /// Install a recording fake; replies come from `REPLY` (or "ok"). Hold `crate::TEST_ENV`.
    pub fn fake() {
        CALLS.lock().unwrap().clear();
        *REPLY.lock().unwrap() = None;
        install(|m, a| {
            CALLS.lock().unwrap().push((m.into(), a.into()));
            Ok(match m {
                "shizuku_state" => "ready".into(),
                "shell" => REPLY.lock().unwrap().clone().unwrap_or_else(|| "exit=0\n".into()),
                _ => REPLY.lock().unwrap().clone().unwrap_or_else(|| "ok".into()),
            })
        });
    }

    pub fn calls() -> Vec<(String, String)> {
        CALLS.lock().unwrap().clone()
    }

    pub fn uninstall() {
        *NATIVE.lock().unwrap() = None;
    }
}
