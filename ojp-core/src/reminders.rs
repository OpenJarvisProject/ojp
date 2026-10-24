//! Timers and reminders that survive closing the app: they are stored in a
//! small file and a background thread (started by the engine) notifies when
//! one is due. Reminders that came due while the app was closed are announced
//! the next time it starts.
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Reminder {
    /// When it is due, in seconds since 1970.
    pub due: u64,
    pub text: String,
}

/// Serializes read-modify-write cycles on the file.
static LOCK: Mutex<()> = Mutex::new(());

fn path() -> PathBuf {
    if let Some(p) = std::env::var_os("OJP_REMINDERS_FILE") {
        return PathBuf::from(p);
    }
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).unwrap_or_else(|| home.join(".local/share")).join("ojp/reminders.json")
}

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn read() -> Vec<Reminder> {
    std::fs::read_to_string(path()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

fn write(list: &[Reminder]) -> Result<()> {
    let path = path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(list)?)?;
    std::fs::rename(tmp, path).context("saving reminders")
}

/// Add a reminder due in `minutes`.
pub fn add(minutes: u64, text: &str) -> Result<()> {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut list = read();
    list.push(Reminder { due: now() + minutes * 60, text: text.to_string() });
    list.sort_by_key(|r| r.due);
    write(&list)
}

/// All pending reminders, soonest first.
pub fn list() -> Vec<Reminder> {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    read()
}

/// Remove and return the reminders due at `now`.
pub fn take_due(now: u64) -> Vec<Reminder> {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (due, pending): (Vec<Reminder>, Vec<Reminder>) = read().into_iter().partition(|r| r.due <= now);
    if !due.is_empty() {
        let _ = write(&pending);
    }
    due
}

/// How long ago / from now, in words: "in 5 minutes", "2 hours ago".
pub fn describe(due: u64, now: u64) -> String {
    let (secs, past) = if due >= now { (due - now, false) } else { (now - due, true) };
    let (n, unit) = match secs {
        0..=59 => (secs.max(1), "second"),
        60..=5399 => ((secs + 30) / 60, "minute"),
        _ => ((secs + 1800) / 3600, "hour"),
    };
    let plural = if n == 1 { "" } else { "s" };
    if past { format!("{n} {unit}{plural} ago") } else { format!("in {n} {unit}{plural}") }
}

/// Start the thread that announces due reminders (once per process).
pub fn start_watcher() {
    static STARTED: OnceLock<()> = OnceLock::new();
    if STARTED.set(()).is_err() {
        return;
    }
    std::thread::Builder::new()
        .name("ojp-reminders".into())
        .spawn(|| loop {
            let now = now();
            for r in take_due(now) {
                let late = now.saturating_sub(r.due) > 120;
                let body = if late { format!("{} (was due {})", r.text, describe(r.due, now)) } else { r.text.clone() };
                let _ = std::process::Command::new("notify-send").args(["-a", "OpenJarvis", "Reminder", &body]).status();
            }
            std::thread::sleep(Duration::from_secs(10));
        })
        .ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_in_words() {
        assert_eq!(describe(1000 + 300, 1000), "in 5 minutes");
        assert_eq!(describe(1000 + 60, 1000), "in 1 minute");
        assert_eq!(describe(1000, 1000 + 7200), "2 hours ago");
        assert_eq!(describe(1000 + 20, 1000), "in 20 seconds");
    }

    #[test]
    fn reminders_persist_and_come_due_in_order() {
        let dir = std::env::temp_dir().join(format!("ojp-rem-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        unsafe { std::env::set_var("OJP_REMINDERS_FILE", dir.join("r.json")) };
        add(30, "stretch").unwrap();
        add(5, "tea").unwrap();
        let listed = list();
        assert_eq!(listed.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(), ["tea", "stretch"]);
        let t = now();
        assert!(take_due(t).is_empty());
        let due = take_due(t + 10 * 60);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].text, "tea");
        assert_eq!(list().len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }
}
