//! The phone's versions of the tools, and the tools only a phone has. Plain things (volume, media
//! keys, opening apps and links, reminders, battery) go through Android's own APIs and need nothing
//! extra; the rest (Wi-Fi, Bluetooth, airplane mode, brightness, screenshots, any shell command)
//! need Shizuku (see `platform`). Everything is a short string in, a short sentence out.
use anyhow::{Result, bail};

use crate::platform::{self, call, quote, shell};

pub fn volume(level: u64) -> Result<String> {
    let level = level.min(100);
    call("volume", &level.to_string())?;
    Ok(format!("Volume set to {level}%."))
}

pub fn mute(on: bool) -> Result<String> {
    call("mute", if on { "on" } else { "off" })?;
    Ok(if on { "Muted.".into() } else { "Unmuted.".into() })
}

pub fn media(action: &str) -> Result<String> {
    let said = match action {
        "play_pause" => "Toggled play/pause.",
        "next" => "Skipped to the next track.",
        "previous" => "Back to the previous track.",
        other => bail!("unknown media action {other:?}"),
    };
    call("media", action)?;
    Ok(said.into())
}

pub fn open_app(name: &str) -> Result<String> {
    let opened = call("open_app", name)?;
    Ok(format!("Opened {opened}."))
}

pub fn open_url(url: &str) -> Result<String> {
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        bail!("only web links (http or https) can be opened");
    }
    call("open_url", url)?;
    Ok(format!("Opened {url}."))
}

/// Make the phone notify at `due` (seconds since 1970), even if the app is closed.
pub fn schedule(due: u64, text: &str) -> Result<()> {
    let text = text.replace(['\t', '\n'], " ");
    call("remind", &format!("{}\t{text}", due * 1000))?;
    Ok(())
}

pub fn battery() -> Result<String> {
    call("battery", "")
}

pub fn brightness(level: u64) -> Result<String> {
    let level = level.min(100);
    // 0 = manual brightness; the value is 0-255.
    shell(&format!("settings put system screen_brightness_mode 0 && settings put system screen_brightness {}", (level * 255).div_ceil(100)))?;
    Ok(format!("Brightness set to {level}%."))
}

pub fn wifi(on: bool) -> Result<String> {
    shell(&format!("svc wifi {}", if on { "enable" } else { "disable" }))?;
    Ok(format!("Wi-Fi turned {}.", if on { "on" } else { "off" }))
}

pub fn bluetooth(on: bool) -> Result<String> {
    shell(&format!("svc bluetooth {}", if on { "enable" } else { "disable" }))?;
    Ok(format!("Bluetooth turned {}.", if on { "on" } else { "off" }))
}

pub fn airplane(on: bool) -> Result<String> {
    shell(&format!("cmd connectivity airplane-mode {}", if on { "enable" } else { "disable" }))?;
    Ok(format!("Airplane mode turned {}.", if on { "on" } else { "off" }))
}

pub fn screenshot() -> Result<String> {
    let path = format!("/sdcard/Pictures/jarvis-{}.png", crate::reminders::now());
    shell(&format!("screencap -p {}", quote(&path)))?;
    Ok(format!("Saved a screenshot to {path}."))
}

/// Any shell command with adb-level rights (the caller has already asked the user).
pub fn adb_shell(command: &str) -> Result<String> {
    let out = shell(command)?;
    let out: String = out.chars().take(1500).collect();
    Ok(if out.is_empty() { "Done (no output).".into() } else { out })
}

/// Commands the assistant may run on its own, because they only look at things and change nothing.
const LOOKUPS: &[&str] = &[
    "ip", "ifconfig", "netstat", "ss", "getprop", "uptime", "date", "id", "whoami", "uname", "df", "du", "free", "ls", "cat", "head", "tail", "wc",
    "grep", "sort", "uniq", "cut", "tr", "ps", "pidof", "which", "getenforce", "hostname", "dumpsys", "settings", "pm", "cmd", "wm", "find", "stat",
    "service", "lsmod", "mount", "printenv", "nslookup", "ping", "arp",
];

/// Words that turn a lookup into a change, wherever they appear in the command.
const CHANGERS: &[&str] = &[
    " add ", " del ", " delete ", " flush ", " set ", " put ", " change ", " replace ", " reset ", " unplug ", " enable ", " disable ", " install ",
    " uninstall ", " clear ", " grant ", " revoke ", " kill ", " start ", " stop ", " remove ", " rm ", " mv ", " cp ", " write ", " reboot ", " -exec", " -delete",
    " -ok", " -fprint", "svc ",
];

/// Whether `command` only looks at things (see [`LOOKUPS`]): no redirects, no chaining, no substitutions, every
/// piece of a pipeline a lookup, and `settings`/`pm`/`cmd`/`wm` used only in their reading forms.
pub fn is_lookup(command: &str) -> bool {
    let c = command.trim();
    if c.is_empty() || c.len() > 250 || c.contains(['>', '<', '`', ';', '\n', '&', '\\']) || c.contains("$(") || c.contains("${") {
        return false;
    }
    let padded = format!(" {} ", c.to_lowercase());
    if CHANGERS.iter().any(|w| padded.contains(w)) {
        return false;
    }
    c.split('|').all(|piece| {
        let mut words = piece.split_whitespace();
        let Some(first) = words.next() else { return false };
        let rest: Vec<&str> = words.collect();
        if !LOOKUPS.contains(&first) {
            return false;
        }
        match first {
            // Only the reading forms of these.
            "settings" => rest.first().is_some_and(|w| matches!(*w, "get" | "list")),
            "pm" => rest.first().is_some_and(|w| matches!(*w, "list" | "path" | "dump" | "resolve-activity" | "get-install-location")),
            "cmd" => matches!(rest.as_slice(), ["wifi", "status" | "list-scan-results" | "list-networks"] | ["package", "list", ..]),
            "wm" => matches!(rest.as_slice(), ["size"] | ["density"]),
            "service" => rest.first() == Some(&"list"),
            "mount" => rest.is_empty(),
            "ping" => rest.iter().any(|w| *w == "-c"),
            _ => true,
        }
    })
}

/// Look something up on the phone by running a read-only shell command through Shizuku.
pub fn phone_read(command: &str) -> Result<String> {
    if !is_lookup(command) {
        bail!("that isn't a read-only lookup, so I won't run it on my own; adb_shell can run it after asking you");
    }
    let out = shell(command.trim())?;
    let out: String = out.chars().take(2500).collect();
    Ok(if out.trim().is_empty() { "(the command printed nothing)".into() } else { out })
}

/// Words for the Shizuku state, shown in Settings.
pub fn shizuku_text() -> String {
    let manager = platform::shizuku_manager();
    let name = manager.as_ref().map_or("Shizuku".to_string(), |(label, _)| label.clone());
    match platform::shizuku() {
        platform::Shizuku::Ready => format!("{name} is on: Open Jarvis can change Wi-Fi, Bluetooth, brightness and run commands for you."),
        platform::Shizuku::NeedsPermission => format!("{name} is running. Tap the button to allow Open Jarvis."),
        platform::Shizuku::TooOld => format!("The installed {name} is too old; update it."),
        platform::Shizuku::Unavailable if manager.is_some() => format!("Found {name}, but it isn't running. Tap the button to open it and start it."),
        platform::Shizuku::Unavailable => "No Shizuku app found (optional; the original and its forks both work). Install one and start it to let the assistant change Wi-Fi, Bluetooth, brightness and run commands.".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::testing::{calls, fake, uninstall, REPLY};

    #[test]
    fn only_lookups_run_without_asking() {
        for ok in [
            "ip -f inet addr show wlan0",
            "cmd wifi status",
            "dumpsys battery",
            "dumpsys wifi | grep SSID",
            "getprop ro.build.version.release",
            "settings get system screen_brightness",
            "pm list packages -3",
            "df -h /storage/emulated/0",
            "ls /sdcard/Download",
            "wm size",
        ] {
            assert!(is_lookup(ok), "{ok}");
        }
        for bad in [
            "rm -rf /sdcard/x",
            "settings put system screen_brightness 10",
            "svc wifi disable",
            "cmd wifi set-wifi-enabled disabled",
            "ip link set wlan0 down",
            "dumpsys battery set level 5",
            "cat /sdcard/a > /sdcard/b",
            "ls; reboot",
            "ls && rm x",
            "echo $(reboot)",
            "pm uninstall com.x",
            "pm clear com.x",
            "find / -delete",
            "cmd connectivity airplane-mode enable",
            "wm size 100x100",
            "",
        ] {
            assert!(!is_lookup(bad), "{bad}");
        }
    }

    #[test]
    fn plain_actions_use_the_platform_layer() {
        let _g = crate::TEST_ENV.lock().unwrap_or_else(|e| e.into_inner());
        fake();
        assert_eq!(volume(250).unwrap(), "Volume set to 100%.");
        assert_eq!(media("next").unwrap(), "Skipped to the next track.");
        assert!(media("explode").is_err());
        *REPLY.lock().unwrap() = Some("Spotify".into());
        assert_eq!(open_app("spotify").unwrap(), "Opened Spotify.");
        assert!(open_url("file:///etc/passwd").is_err(), "only web links");
        schedule(1_700_000_000, "call\tmom\nnow").unwrap();
        let c = calls();
        assert_eq!(c[0], ("volume".into(), "100".into()));
        assert_eq!(c.last().unwrap(), &("remind".to_string(), "1700000000000\tcall mom now".to_string()));
        uninstall();
    }

    #[test]
    fn shizuku_actions_build_the_right_commands_and_report_failures() {
        let _g = crate::TEST_ENV.lock().unwrap_or_else(|e| e.into_inner());
        fake();
        assert_eq!(wifi(false).unwrap(), "Wi-Fi turned off.");
        brightness(50).unwrap();
        airplane(true).unwrap();
        let shells: Vec<String> = calls().into_iter().filter(|c| c.0 == "shell").map(|c| c.1).collect();
        assert_eq!(shells[0], "svc wifi disable");
        assert!(shells[1].ends_with("screen_brightness 128"), "{}", shells[1]);
        assert_eq!(shells[2], "cmd connectivity airplane-mode enable");
        *REPLY.lock().unwrap() = Some("exit=1\nSecurity exception".into());
        let err = adb_shell("settings put global x 1").unwrap_err();
        assert!(format!("{err}").contains("exit 1") && format!("{err}").contains("Security exception"));
        *REPLY.lock().unwrap() = Some("exit=0\nline one\n".into());
        assert_eq!(adb_shell("echo hi").unwrap(), "line one");
        uninstall();
    }

    #[test]
    fn without_a_platform_layer_nothing_is_available() {
        let _g = crate::TEST_ENV.lock().unwrap_or_else(|e| e.into_inner());
        uninstall();
        assert!(volume(10).is_err());
        let err = wifi(true).unwrap_err();
        assert!(format!("{err}").contains("Shizuku"), "{err}");
        assert_eq!(quote("it's"), "'it'\\''s'");
    }
}
