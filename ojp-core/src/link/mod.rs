//! Devices that work together. Each device has a key pair that never leaves
//! it. Two devices become peers only when the user pairs them by hand (one
//! shows a one-time code, the other types it in), after which they talk over
//! an encrypted, mutually authenticated channel and nothing else can connect.
//!
//! What peers do for each other (each can be switched off per device):
//! - **chat**: a device without a big model answers with a peer's;
//! - **ask**: "remind me on my phone ..." runs on the phone, by its own assistant;
//! - **notify**: show a message on the other device;
//! - **sync**: memories are shared and merged.
//!
//! Files: `<data>/ojp/link/identity.json` (this device) and `peers.json`
//! (who is trusted), both private to the user.
pub mod net;
pub mod pair;
pub mod wire;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use net::{Client, Incoming, Presence, Reply, Request, Server};
pub use wire::{Channel, Msg};

/// Where the link files live: `$OJP_LINK_DIR`, else `<data dir>/ojp/link`.
pub fn dir() -> PathBuf {
    if let Some(p) = std::env::var_os("OJP_LINK_DIR") {
        return PathBuf::from(p);
    }
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).unwrap_or_else(|| home.join(".local/share")).join("ojp/link")
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok()).collect()
}

/// A short name for a public key, shown to people and used for discovery.
pub fn fingerprint(public: &[u8]) -> String {
    hex(&Sha256::digest(public)[..8])
}

fn write_private(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(tmp, path).with_context(|| format!("saving {}", path.display()))
}

/// This device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    pub name: String,
    /// "phone", "laptop", "desktop", "server" or "tablet": lets "my phone" mean the right device.
    pub kind: String,
    private: String,
    public: String,
}

impl Identity {
    fn path() -> PathBuf {
        dir().join("identity.json")
    }

    pub fn load_or_create() -> Result<Identity> {
        if let Ok(text) = std::fs::read_to_string(Self::path()) {
            return serde_json::from_str(&text).context("reading identity.json");
        }
        let keys = snow::Builder::new(wire::PARAMS_XX.parse()?).generate_keypair()?;
        let id = Identity { name: default_name(), kind: default_kind(), private: hex(&keys.private), public: hex(&keys.public) };
        id.save()?;
        Ok(id)
    }

    /// A fresh identity that is not saved (for tests and for pairing helpers).
    pub fn generate(name: &str, kind: &str) -> Result<Identity> {
        let keys = snow::Builder::new(wire::PARAMS_XX.parse()?).generate_keypair()?;
        Ok(Identity { name: name.into(), kind: kind.into(), private: hex(&keys.private), public: hex(&keys.public) })
    }

    pub fn save(&self) -> Result<()> {
        write_private(&Self::path(), &serde_json::to_vec_pretty(self)?)
    }

    pub fn private_key(&self) -> Vec<u8> {
        unhex(&self.private).unwrap_or_default()
    }

    pub fn public_key(&self) -> Vec<u8> {
        unhex(&self.public).unwrap_or_default()
    }

    pub fn fingerprint(&self) -> String {
        fingerprint(&self.public_key())
    }
}

fn default_name() -> String {
    if cfg!(target_os = "android") {
        return "phone".into();
    }
    let host = std::fs::read_to_string("/etc/hostname").ok().or_else(|| std::env::var("HOSTNAME").ok()).unwrap_or_default();
    let host = host.trim();
    if host.is_empty() { "device".into() } else { host.to_string() }
}

fn default_kind() -> String {
    if cfg!(target_os = "android") {
        return "phone".into();
    }
    let has_battery = std::fs::read_dir("/sys/class/power_supply")
        .map(|d| d.flatten().any(|e| e.file_name().to_string_lossy().starts_with("BAT")))
        .unwrap_or(false);
    if has_battery { "laptop" } else { "desktop" }.into()
}

/// How much a peer may ask this device to do for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AskLevel {
    Off,
    /// Timers, reminders, notes, memory, volume, media, time, status, calculator.
    Basic,
    /// Also files, the clipboard and the web. Never anything that asks the user first.
    Full,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct Perms {
    /// May use this device's model to answer.
    pub chat: bool,
    pub ask: AskLevel,
    /// May show a message here.
    pub notify: bool,
    /// May exchange memories.
    pub sync: bool,
}

impl Default for Perms {
    fn default() -> Self {
        Perms { chat: true, ask: AskLevel::Basic, notify: true, sync: true }
    }
}

/// A paired device.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Peer {
    pub name: String,
    pub kind: String,
    /// Its public key, hex.
    pub public: String,
    /// Where it was last reached ("ip:port").
    #[serde(default)]
    pub addrs: Vec<String>,
    #[serde(default)]
    pub perms: Perms,
}

impl Peer {
    pub fn fingerprint(&self) -> String {
        fingerprint(&unhex(&self.public).unwrap_or_default())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Peers {
    pub peers: Vec<Peer>,
}

impl Peers {
    fn path() -> PathBuf {
        dir().join("peers.json")
    }

    pub fn load() -> Peers {
        std::fs::read_to_string(Self::path()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    pub fn save(&self) -> Result<()> {
        write_private(&Self::path(), &serde_json::to_vec_pretty(self)?)
    }

    pub fn by_key(&self, public: &[u8]) -> Option<&Peer> {
        let h = hex(public);
        self.peers.iter().find(|p| p.public == h)
    }

    pub fn by_name(&self, name: &str) -> Option<&Peer> {
        self.peers.iter().find(|p| p.name.eq_ignore_ascii_case(name))
    }

    /// Add a peer, or update the one with the same key (keeping its permissions).
    pub fn upsert(&mut self, mut peer: Peer) {
        if let Some(existing) = self.peers.iter_mut().find(|p| p.public == peer.public) {
            peer.perms = existing.perms;
            *existing = peer;
        } else {
            // Names must be unique so "my laptop" means one device.
            let base = peer.name.clone();
            let mut n = 2;
            while self.peers.iter().any(|p| p.name.eq_ignore_ascii_case(&peer.name)) {
                peer.name = format!("{base}-{n}");
                n += 1;
            }
            self.peers.push(peer);
        }
    }

    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.peers.len();
        self.peers.retain(|p| !p.name.eq_ignore_ascii_case(name));
        self.peers.len() != before
    }
}

/// If the request names one of the paired devices ("remind me on my phone ...",
/// "ask the laptop what time it is"), which one. `None` when it names no device,
/// names this one, or the name fits more than one device.
pub fn addressed<'a>(request: &str, peers: &'a [Peer], me: &Identity) -> Option<&'a Peer> {
    let lower = request.to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric() && c != '-').filter(|w| !w.is_empty()).collect();
    const LEAD: &[&str] = &["on", "to", "ask", "tell", "using", "from", "at", "in"];
    const ARTICLE: &[&str] = &["my", "the"];
    let kind_words = |kind: &str| -> &'static [&'static str] {
        match kind {
            "phone" => &["phone", "mobile", "cellphone"],
            "laptop" => &["laptop", "notebook"],
            "desktop" => &["desktop", "pc"],
            "server" => &["server"],
            "tablet" => &["tablet"],
            _ => &[],
        }
    };
    let mut hits: Vec<&Peer> = Vec::new();
    for (i, w) in words.iter().enumerate() {
        if !LEAD.contains(w) {
            continue;
        }
        let mut j = i + 1;
        let article = words.get(j).is_some_and(|a| ARTICLE.contains(a));
        if article {
            j += 1;
        }
        let Some(target) = words.get(j) else { continue };
        // "on my laptop" / "tell my laptop"; a bare name needs no article, a bare kind word does.
        for peer in peers {
            let by_name = peer.name.to_lowercase() == *target;
            let by_kind = article && kind_words(&peer.kind).contains(target);
            if (by_name || by_kind) && !hits.iter().any(|h| h.public == peer.public) {
                hits.push(peer);
            }
        }
    }
    // "tell my phone to ..." with "my phone" matching several phones is ambiguous.
    if hits.len() != 1 {
        return None;
    }
    let peer = hits[0];
    // Naming this very device just means "here".
    let me_name = me.name.to_lowercase();
    if peer.name.to_lowercase() == me_name {
        return None;
    }
    // A kind word shared with this device ("my laptop" said on the laptop) means this one,
    // unless it was the peer's own name that was used.
    let by_name = words.contains(&peer.name.to_lowercase().as_str());
    if !by_name && me.kind == peer.kind {
        return None;
    }
    Some(peer)
}

/// Tools another device may use on this one at each level.
pub fn remote_allowed(tool: &str, level: AskLevel) -> bool {
    const BASIC: &[&str] = &[
        "set_timer", "remind", "volume", "mute", "media", "note_add", "calculate", "get_time", "system_status", "now_playing",
        "reminders_list", "remember", "recall", "memories_list", "tags_list", "notes_list", "done",
    ];
    match level {
        AskLevel::Off => false,
        AskLevel::Basic => BASIC.contains(&tool),
        AskLevel::Full => !matches!(tool, "write_file" | "run_command" | "forget" | "tag_add"),
    }
}

pub fn ensure_name(name: &str) -> Result<()> {
    if name.trim().is_empty() || name.len() > 32 || name.chars().any(|c| !(c.is_alphanumeric() || c == '-' || c == '_')) {
        bail!("a device name is 1-32 letters, digits, - or _");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(name: &str, kind: &str) -> Peer {
        Peer { name: name.into(), kind: kind.into(), public: hex(name.as_bytes()), addrs: vec![], perms: Perms::default() }
    }

    fn me(name: &str, kind: &str) -> Identity {
        Identity { name: name.into(), kind: kind.into(), private: String::new(), public: String::new() }
    }

    #[test]
    fn hex_roundtrip() {
        assert_eq!(unhex(&hex(&[0, 1, 254, 255])).unwrap(), vec![0, 1, 254, 255]);
        assert!(unhex("abc").is_none());
    }

    #[test]
    fn names_a_device_by_kind_or_name() {
        let peers = [peer("pixel", "phone"), peer("zex-server", "server")];
        let desk = me("zex-pc", "desktop");
        assert_eq!(addressed("remind me on my phone in 10 minutes to call mom", &peers, &desk).unwrap().name, "pixel");
        assert_eq!(addressed("ask my server what the disk space is", &peers, &desk).unwrap().name, "zex-server");
        assert_eq!(addressed("tell pixel to turn the volume down", &peers, &desk).unwrap().name, "pixel");
        assert!(addressed("what time is it", &peers, &desk).is_none());
        // "phone" as a topic, not a place.
        assert!(addressed("how do I reset a phone", &peers, &desk).is_none());
    }

    #[test]
    fn ambiguous_or_self_is_not_forwarded() {
        let two = [peer("pixel", "phone"), peer("galaxy", "phone")];
        let desk = me("zex-pc", "desktop");
        assert!(addressed("remind me on my phone to sleep", &two, &desk).is_none());
        assert_eq!(addressed("remind me on galaxy to sleep", &two, &desk).unwrap().name, "galaxy");
        let one = [peer("pixel", "phone")];
        // Said on the phone itself.
        assert!(addressed("remind me on my phone to sleep", &one, &me("pixel2", "phone")).is_none());
        assert!(addressed("remind me on pixel to sleep", &one, &me("pixel", "phone")).is_none());
    }

    #[test]
    fn permission_levels() {
        assert!(remote_allowed("remind", AskLevel::Basic));
        assert!(!remote_allowed("read_file", AskLevel::Basic));
        assert!(remote_allowed("read_file", AskLevel::Full));
        assert!(!remote_allowed("run_command", AskLevel::Full));
        assert!(!remote_allowed("remind", AskLevel::Off));
    }

    #[test]
    fn upsert_keeps_permissions_and_makes_names_unique() {
        let mut peers = Peers::default();
        let mut a = peer("pixel", "phone");
        a.perms.notify = false;
        peers.upsert(a.clone());
        let mut again = a.clone();
        again.perms = Perms::default();
        again.addrs = vec!["1.2.3.4:5".into()];
        peers.upsert(again);
        assert_eq!(peers.peers.len(), 1);
        assert!(!peers.peers[0].perms.notify, "re-pairing doesn't reset permissions");
        let mut other = peer("pixel", "phone");
        other.public = "ff".into();
        peers.upsert(other);
        assert_eq!(peers.peers[1].name, "pixel-2");
    }
}
