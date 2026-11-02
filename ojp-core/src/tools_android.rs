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

/// Which Wi-Fi network the phone is on, and how good the link is. With Shizuku the network's name and signal come
/// from the system; without it Android only tells apps whether the phone is on Wi-Fi at all.
pub fn wifi_info() -> Result<String> {
    if platform::shizuku() == platform::Shizuku::Ready {
        let out = shell("cmd wifi status")?;
        return Ok(describe_wifi(&out));
    }
    Ok(match call("wifi_link", "").unwrap_or_default().as_str() {
        "wifi" => "The phone is on Wi-Fi, but Android hides the network's name from apps. Allow Open Jarvis in Shizuku (Settings) and ask again to see it.".into(),
        "cellular" => "The phone is on mobile data, not Wi-Fi.".into(),
        "ethernet" => "The phone is on a wired connection, not Wi-Fi.".into(),
        "none" => "The phone has no network connection right now.".into(),
        _ => "The phone is connected through something other than Wi-Fi.".into(),
    })
}

/// One sentence from the output of `cmd wifi status`.
pub fn describe_wifi(out: &str) -> String {
    let lower = out.to_lowercase();
    if lower.contains("wifi is disabled") {
        return "Wi-Fi is turned off.".into();
    }
    let quoted = |after: &str| -> Option<String> {
        let at = out.find(after)? + after.len();
        let rest = &out[at..];
        Some(rest[..rest.find('"')?].to_string())
    };
    let Some(ssid) = quoted("Wifi is connected to \"").or_else(|| quoted("SSID: \"")).filter(|s| !s.is_empty() && s != "<unknown ssid>") else {
        return "Wi-Fi is on, but the phone isn't connected to a network.".into();
    };
    let number = |key: &str| -> Option<i64> {
        let at = out.find(key)? + key.len();
        let rest = &out[at..];
        let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '-')).unwrap_or(rest.len());
        rest[..end].parse().ok()
    };
    let mut details = Vec::new();
    if let Some(freq) = number("Frequency: ") {
        details.push(if freq < 3000 { "2.4 GHz" } else if freq < 5900 { "5 GHz" } else { "6 GHz" }.to_string());
    }
    if let Some(rssi) = number("RSSI: ") {
        details.push(format!("signal {rssi} dBm"));
    }
    if let Some(speed) = number("Link speed: ") {
        details.push(format!("{speed} Mbps"));
    }
    if details.is_empty() { format!("Connected to Wi-Fi \"{ssid}\".") } else { format!("Connected to Wi-Fi \"{ssid}\" ({}).", details.join(", ")) }
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
    fn wifi_status_output_becomes_one_sentence() {
        let on = "Wifi is enabled\n==== Primary ClientModeManager instance ====\nWifi is connected to \"HomeNet\"\nWifiInfo: SSID: \"HomeNet\", BSSID: aa:bb, Supplicant state: COMPLETED, RSSI: -48, Link speed: 866Mbps, Frequency: 5180MHz, Net ID: 0";
        assert_eq!(describe_wifi(on), "Connected to Wi-Fi \"HomeNet\" (5 GHz, signal -48 dBm, 866 Mbps).");
        assert_eq!(describe_wifi("Wifi is disabled"), "Wi-Fi is turned off.");
        assert_eq!(describe_wifi("Wifi is enabled\nWifi is not connected"), "Wi-Fi is on, but the phone isn't connected to a network.");
        assert_eq!(describe_wifi("Wifi is connected to \"Cafe\""), "Connected to Wi-Fi \"Cafe\".");
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
