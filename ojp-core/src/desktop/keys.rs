//! US keyboard: which evdev key (and whether Shift) types a character, and names of keys for shortcuts.
use anyhow::{Result, bail};

pub const KEY_LEFTSHIFT: u32 = 42;

/// A key to press, and whether Shift must be held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Key {
    pub code: u32,
    pub shift: bool,
}

const ROW_TOP: &str = "qwertyuiop"; // evdev 16..=25
const ROW_HOME: &str = "asdfghjkl"; // 30..=38
const ROW_BOTTOM: &str = "zxcvbnm"; // 44..=50
const DIGITS: &str = "1234567890"; // 2..=11
const DIGITS_SHIFTED: &str = "!@#$%^&*()";

pub fn key_for_char(c: char) -> Option<Key> {
    let plain = |code| Some(Key { code, shift: false });
    let shifted = |code| Some(Key { code, shift: true });
    let lower = c.to_ascii_lowercase();
    let letter = |code_of: Option<u32>| code_of.map(|code| Key { code, shift: c.is_ascii_uppercase() });
    if let Some(i) = ROW_TOP.find(lower) {
        return letter(Some(16 + i as u32));
    }
    if let Some(i) = ROW_HOME.find(lower) {
        return letter(Some(30 + i as u32));
    }
    if let Some(i) = ROW_BOTTOM.find(lower) {
        return letter(Some(44 + i as u32));
    }
    if let Some(i) = DIGITS.find(c) {
        return plain(2 + i as u32);
    }
    if let Some(i) = DIGITS_SHIFTED.find(c) {
        return shifted(2 + i as u32);
    }
    match c {
        ' ' => plain(57),
        '\n' => plain(28),
        '\t' => plain(15),
        '-' => plain(12),
        '_' => shifted(12),
        '=' => plain(13),
        '+' => shifted(13),
        '[' => plain(26),
        '{' => shifted(26),
        ']' => plain(27),
        '}' => shifted(27),
        ';' => plain(39),
        ':' => shifted(39),
        '\'' => plain(40),
        '"' => shifted(40),
        '`' => plain(41),
        '~' => shifted(41),
        '\\' => plain(43),
        '|' => shifted(43),
        ',' => plain(51),
        '<' => shifted(51),
        '.' => plain(52),
        '>' => shifted(52),
        '/' => plain(53),
        '?' => shifted(53),
        _ => None,
    }
}

/// The code of a key by name ("ctrl", "return", "f5", "a").
pub fn key_by_name(name: &str) -> Option<u32> {
    let n = name.trim().to_lowercase();
    let code = match n.as_str() {
        "ctrl" | "control" => 29,
        "shift" => 42,
        "alt" => 56,
        "super" | "meta" | "win" | "logo" => 125,
        "return" | "enter" => 28,
        "escape" | "esc" => 1,
        "tab" => 15,
        "backspace" => 14,
        "space" => 57,
        "delete" | "del" => 111,
        "insert" => 110,
        "home" => 102,
        "end" => 107,
        "pageup" | "pgup" => 104,
        "pagedown" | "pgdn" => 109,
        "left" => 105,
        "right" => 106,
        "up" => 103,
        "down" => 108,
        _ => {
            if let Some(num) = n.strip_prefix('f').and_then(|d| d.parse::<u32>().ok()) {
                return match num {
                    1..=10 => Some(58 + num),
                    11 => Some(87),
                    12 => Some(88),
                    _ => None,
                };
            }
            let mut chars = n.chars();
            return match (chars.next(), chars.next()) {
                (Some(c), None) => key_for_char(c).map(|k| k.code),
                _ => None,
            };
        }
    };
    Some(code)
}

/// "ctrl+shift+t" -> the key codes to hold, in order (modifiers first, the last key is the one pressed).
pub fn parse_combo(combo: &str) -> Result<Vec<u32>> {
    let parts: Vec<&str> = combo.split('+').map(str::trim).filter(|p| !p.is_empty()).collect();
    if parts.is_empty() {
        bail!("no keys given");
    }
    parts.iter().map(|p| key_by_name(p).ok_or_else(|| anyhow::anyhow!("unknown key {p:?}"))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn characters_map_to_us_keys() {
        assert_eq!(key_for_char('a'), Some(Key { code: 30, shift: false }));
        assert_eq!(key_for_char('A'), Some(Key { code: 30, shift: true }));
        assert_eq!(key_for_char('1'), Some(Key { code: 2, shift: false }));
        assert_eq!(key_for_char('!'), Some(Key { code: 2, shift: true }));
        assert_eq!(key_for_char('/'), Some(Key { code: 53, shift: false }));
        assert_eq!(key_for_char('?'), Some(Key { code: 53, shift: true }));
        assert_eq!(key_for_char(' '), Some(Key { code: 57, shift: false }));
        assert_eq!(key_for_char('é'), None);
        // Every printable ASCII character can be typed.
        for c in (' '..='~').chain(['\n', '\t']) {
            assert!(key_for_char(c).is_some(), "{c:?}");
        }
    }

    #[test]
    fn combos_parse() {
        assert_eq!(parse_combo("ctrl+s").unwrap(), vec![29, 31]);
        assert_eq!(parse_combo("Ctrl + Shift + T").unwrap(), vec![29, 42, 20]);
        assert_eq!(parse_combo("alt+f4").unwrap(), vec![56, 62]);
        assert_eq!(parse_combo("return").unwrap(), vec![28]);
        assert!(parse_combo("ctrl+nonsense").is_err());
        assert!(parse_combo("").is_err());
    }
}
