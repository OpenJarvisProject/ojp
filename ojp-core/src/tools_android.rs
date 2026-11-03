//! The phone's versions of the tools, and the tools only a phone has. Plain things (volume, media
//! keys, opening apps and links and settings pages, reminders, alarms, the flashlight, calls and texts,
//! battery, Wi-Fi/IP/storage/version lookups, brightness after one permission) go through Android's own APIs and need
//! nothing extra. With Shizuku (see `platform`) the assistant can also flip Wi-Fi, Bluetooth and airplane mode itself,
//! take screenshots and run shell commands; without it those open the right settings page instead.
//! Everything is a short string in, a short sentence out.
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

fn has_shizuku() -> bool {
    platform::shizuku() == platform::Shizuku::Ready
}

pub fn brightness(level: u64) -> Result<String> {
    let level = level.min(100);
    if has_shizuku() {
        // 0 = manual brightness; the value is 0-255.
        shell(&format!("settings put system screen_brightness_mode 0 && settings put system screen_brightness {}", (level * 255).div_ceil(100)))?;
    } else {
        call("brightness", &level.to_string())?;
    }
    Ok(format!("Brightness set to {level}%."))
}

/// Android lets only system apps flip these switches; with Shizuku we do, otherwise open the page with the switch.
fn switch(on: bool, what: &str, command: &str, page: &str) -> Result<String> {
    let state = if on { "on" } else { "off" };
    if has_shizuku() {
        shell(command)?;
        return Ok(format!("{what} turned {state}."));
    }
    call("settings_page", page)?;
    Ok(format!("Android doesn't let apps flip {what} by themselves, so I opened its settings: tap the switch to turn it {state}. (With Shizuku I could do it for you.)"))
}

pub fn wifi(on: bool) -> Result<String> {
    switch(on, "Wi-Fi", &format!("svc wifi {}", if on { "enable" } else { "disable" }), "wifi")
}

pub fn bluetooth(on: bool) -> Result<String> {
    switch(on, "Bluetooth", &format!("svc bluetooth {}", if on { "enable" } else { "disable" }), "bluetooth")
}

pub fn airplane(on: bool) -> Result<String> {
    switch(on, "airplane mode", &format!("cmd connectivity airplane-mode {}", if on { "enable" } else { "disable" }), "airplane")
}

pub fn flashlight(on: bool) -> Result<String> {
    call("flashlight", if on { "on" } else { "off" })?;
    Ok(format!("Flashlight turned {}.", if on { "on" } else { "off" }))
}

pub fn alarm(hour: u64, minute: u64) -> Result<String> {
    if hour > 23 || minute > 59 {
        bail!("that isn't a time of day");
    }
    call("alarm", &format!("{hour}\t{minute}\t"))?;
    Ok(format!("Alarm set for {hour:02}:{minute:02}."))
}

/// A number, or a contact's name (the phone asks once for permission to read contacts).
fn who(text: &str) -> Result<String> {
    let text = text.trim().replace(['\t', '\n'], " ");
    if text.is_empty() {
        bail!("who should I contact?");
    }
    Ok(text)
}

pub fn dial(person: &str) -> Result<String> {
    let number = call("dial", &who(person)?)?;
    Ok(format!("Opened the dialer with {number}; press call to ring."))
}

pub fn text_message(person: &str, message: &str) -> Result<String> {
    let number = call("sms", &format!("{}\t{}", who(person)?, message.replace(['\t', '\n'], " ")))?;
    Ok(format!("Opened a text to {number} with your message written; press send to send it."))
}

pub fn open_settings(page: &str) -> Result<String> {
    let page = call("settings_page", page)?;
    Ok(format!("Opened the {page} settings."))
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

/// Look something up on the phone with a read-only command: through Shizuku when it's on, otherwise by
/// answering the common ones from what Android tells any app.
pub fn phone_read(command: &str) -> Result<String> {
    if !is_lookup(command) {
        bail!("that isn't a read-only lookup, so I won't run it on my own; adb_shell can run it after asking you");
    }
    let command = command.trim();
    let out = if has_shizuku() {
        shell(command)?
    } else {
        match without_shizuku(command) {
            Some(topic) => call("info", &topic)?,
            None => bail!(
                "that lookup needs Shizuku. Without it I can look up Wi-Fi, IP address, battery, storage, Android version, phone model, settings values and installed apps"
            ),
        }
    };
    let out: String = out.chars().take(2500).collect();
    Ok(if out.trim().is_empty() { "(the command printed nothing)".into() } else { out })
}

/// Which `PhoneActions.info` topic answers a lookup command, for when Shizuku isn't there.
fn without_shizuku(command: &str) -> Option<String> {
    let words: Vec<&str> = command.split_whitespace().collect();
    let first = *words.first()?;
    let has = |w: &str| words.iter().any(|x| x.contains(w));
    Some(match first {
        "cmd" if has("wifi") => "wifi".into(),
        "dumpsys" if has("wifi") => "wifi".into(),
        "dumpsys" if has("battery") => "battery".into(),
        "ip" | "ifconfig" => "ip".into(),
        "df" | "du" => "storage".into(),
        "uptime" => "uptime".into(),
        "pm" if has("packages") => "packages".into(),
        "getprop" => match words.get(1).copied().unwrap_or("") {
            "ro.build.version.release" | "ro.build.version.sdk" | "ro.build.version.security_patch" => "version".into(),
            "ro.product.model" | "ro.product.manufacturer" | "ro.product.brand" | "ro.product.name" | "ro.product.device" => "model".into(),
            _ => return None,
        },
        "settings" if words.len() >= 4 && words[1] == "get" && matches!(words[2], "system" | "global" | "secure") => {
            format!("setting\t{}\t{}", words[2], words[3])
        }
        _ => return None,
    })
}

/// Words for the Shizuku state, shown in Settings.
pub fn shizuku_text() -> String {
    let manager = platform::shizuku_manager();
    let name = manager.as_ref().map_or("Shizuku".to_string(), |(label, _)| label.clone());
    match platform::shizuku() {
        platform::Shizuku::Ready => format!("{name} is on: Open Jarvis can change Wi-Fi, Bluetooth, brightness and run commands for you."),
        platform::Shizuku::NeedsPermission => format!("{name} is running. Tap the button to allow Open Jarvis."),
        platform::Shizuku::Denied => format!("{name} has Open Jarvis switched off, and it won't ask again. Tap the button to open it, then switch Open Jarvis on in its list of authorized apps."),
        platform::Shizuku::TooOld => format!("The installed {name} is too old; update it."),
        platform::Shizuku::Unavailable if manager.is_some() => format!("Found {name}, but it isn't running. Tap the button to open it and start it."),
        platform::Shizuku::Unavailable => "No Shizuku app found (optional; the original and its forks both work). Install one and start it to let the assistant change Wi-Fi, Bluetooth, brightness and run commands.".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::testing::{calls, fake, uninstall, REPLY, SHIZUKU};

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
    fn without_shizuku_the_common_things_still_work() {
        let _g = crate::TEST_ENV.lock().unwrap_or_else(|e| e.into_inner());
        fake();
        *SHIZUKU.lock().unwrap() = "none";
        *REPLY.lock().unwrap() = Some("Wi-Fi network X, 5 GHz".into());
        assert_eq!(phone_read("cmd wifi status").unwrap(), "Wi-Fi network X, 5 GHz");
        phone_read("ip -f inet addr show wlan0").unwrap();
        phone_read("dumpsys battery").unwrap();
        phone_read("getprop ro.product.model").unwrap();
        phone_read("settings get global bluetooth_on").unwrap();
        phone_read("pm list packages -3").unwrap();
        phone_read("df -h /sdcard").unwrap();
        let err = phone_read("getprop ro.some.thing").unwrap_err();
        assert!(format!("{err}").contains("needs Shizuku"), "{err}");
        let sent: Vec<(String, String)> = calls().into_iter().filter(|c| c.0 == "info").collect();
        let topics: Vec<&str> = sent.iter().map(|c| c.1.as_str()).collect();
        assert_eq!(topics, ["wifi", "ip", "battery", "model", "setting\tglobal\tbluetooth_on", "packages", "storage"]);
        // Switches open their page instead; brightness uses the system-settings permission.
        assert!(wifi(true).unwrap().contains("opened its settings"));
        brightness(40).unwrap();
        assert!(airplane(false).unwrap().contains("tap the switch"));
        let c = calls();
        assert!(c.contains(&("settings_page".into(), "wifi".into())) && c.contains(&("brightness".into(), "40".into())));
        assert!(!c.iter().any(|x| x.0 == "shell"), "no shell without Shizuku");
        // Calls, texts, alarms, flashlight.
        flashlight(true).unwrap();
        assert_eq!(alarm(7, 5).unwrap(), "Alarm set for 07:05.");
        assert!(alarm(25, 0).is_err());
        *REPLY.lock().unwrap() = Some("+441234".into());
        assert!(dial("Mum").unwrap().contains("+441234"));
        text_message("Sam", "late\nsorry").unwrap();
        assert!(dial("  ").is_err());
        let c = calls();
        assert!(c.contains(&("alarm".into(), "7\t5\t".into())) && c.contains(&("sms".into(), "Sam\tlate sorry".into())));
        uninstall();
    }

    #[test]
    fn without_a_platform_layer_nothing_is_available() {
        let _g = crate::TEST_ENV.lock().unwrap_or_else(|e| e.into_inner());
        uninstall();
        assert!(volume(10).is_err());
        assert!(wifi(true).is_err());
        let err = adb_shell("ls").unwrap_err();
        assert!(format!("{err}").contains("Shizuku"), "{err}");
        assert_eq!(quote("it's"), "'it'\\''s'");
    }
}
