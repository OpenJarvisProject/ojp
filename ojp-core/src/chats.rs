//! Saved chats: one JSON file per chat in `<data dir>/chats/`, kept only on
//! this device (no logs or copies anywhere else).
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredMessage {
    pub from_user: bool,
    pub text: String,
    /// The small grey line under a reply (route, timing).
    #[serde(default)]
    pub meta: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredChat {
    pub id: String,
    pub title: String,
    /// Unix seconds of the last change; the chat list is sorted by this.
    pub updated: u64,
    pub messages: Vec<StoredMessage>,
    /// The user chose the title, so it isn't rebuilt from the first message.
    #[serde(default)]
    pub renamed: bool,
}

impl StoredChat {
    pub fn new() -> StoredChat {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        StoredChat { id: format!("{}", now.as_millis()), title: String::new(), updated: now.as_secs(), messages: Vec::new(), renamed: false }
    }

    /// Title from the first user message, shortened at a word boundary.
    pub fn refresh_title(&mut self) {
        if self.renamed {
            return;
        }
        let Some(first) = self.messages.iter().find(|m| m.from_user) else { return };
        let line = first.text.lines().next().unwrap_or_default().trim();
        self.title = if line.chars().count() <= 40 {
            line.to_string()
        } else {
            let cut: String = line.chars().take(40).collect();
            let cut = cut.rsplit_once(' ').map_or(cut.as_str(), |(head, _)| head).to_string();
            format!("{cut}…")
        };
    }

    /// The whole chat as plain text (with the grey line under each reply: tools used, timing), for pasting elsewhere.
    pub fn transcript(&self, assistant: &str) -> String {
        let mut out = format!("{}\n", if self.title.is_empty() { "Chat" } else { &self.title });
        for m in &self.messages {
            out.push_str(&format!("\n{}: {}\n", if m.from_user { "You" } else { assistant }, m.text.trim()));
            if !m.meta.is_empty() {
                out.push_str(&format!("({})\n", m.meta));
            }
        }
        out
    }

    pub fn touch(&mut self) {
        self.updated = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    }
}

impl Default for StoredChat {
    fn default() -> Self {
        StoredChat::new()
    }
}

#[derive(Clone)]
pub struct ChatStore {
    dir: PathBuf,
    /// Unsent messages, one per chat (`new` for a chat that has no messages yet): `<data dir>/drafts.json`.
    drafts: PathBuf,
}

impl ChatStore {
    pub fn new(data_dir: &std::path::Path) -> ChatStore {
        ChatStore { dir: data_dir.join("chats"), drafts: data_dir.join("drafts.json") }
    }

    fn path(&self, id: &str) -> PathBuf {
        // Ids are generated digits; strip anything else so a crafted id can
        // never point outside the chats directory.
        let safe: String = id.chars().filter(char::is_ascii_alphanumeric).collect();
        self.dir.join(format!("{safe}.json"))
    }

    /// All saved chats, most recently updated first. Unreadable files are
    /// skipped rather than failing the whole list.
    pub fn list(&self) -> Vec<StoredChat> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else { return Vec::new() };
        let mut chats: Vec<StoredChat> = entries
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|e| serde_json::from_slice(&std::fs::read(e.path()).ok()?).ok())
            .collect();
        chats.sort_by(|a, b| b.updated.cmp(&a.updated).then(b.id.cmp(&a.id)));
        chats
    }

    pub fn load(&self, id: &str) -> Result<StoredChat> {
        let path = self.path(id);
        serde_json::from_slice(&std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?)
            .with_context(|| format!("parsing {}", path.display()))
    }

    /// Save atomically (write a temp file, then rename), so a crash mid-write
    /// never leaves a corrupt chat. Chats with no messages aren't saved.
    pub fn save(&self, chat: &StoredChat) -> Result<()> {
        if chat.messages.is_empty() {
            return Ok(());
        }
        std::fs::create_dir_all(&self.dir)?;
        let path = self.path(&chat.id);
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(chat)?)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    fn read_drafts(&self) -> std::collections::BTreeMap<String, String> {
        std::fs::read(&self.drafts).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    /// The unsent message typed in chat `key` (`new` for a chat with no messages), or "".
    pub fn draft(&self, key: &str) -> String {
        self.read_drafts().remove(key).unwrap_or_default()
    }

    /// Remember (or, when empty, forget) the unsent message of chat `key`.
    pub fn set_draft(&self, key: &str, text: &str) -> Result<()> {
        let mut drafts = self.read_drafts();
        if text.is_empty() {
            if drafts.remove(key).is_none() {
                return Ok(());
            }
        } else {
            drafts.insert(key.to_string(), text.chars().take(20_000).collect());
        }
        if let Some(parent) = self.drafts.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = self.drafts.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&drafts)?)?;
        std::fs::rename(&tmp, &self.drafts)?;
        Ok(())
    }

    /// Give a saved chat a title of the user's choosing (an empty one goes back to the automatic title).
    pub fn rename(&self, id: &str, title: &str) -> Result<()> {
        let mut chat = self.load(id)?;
        let title = title.trim();
        if title.is_empty() {
            chat.renamed = false;
            chat.refresh_title();
        } else {
            chat.renamed = true;
            chat.title = title.chars().take(80).collect();
        }
        self.save(&chat)
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        let path = self.path(id);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn transcript_has_every_message_and_its_grey_line() {
        let mut c = StoredChat::new();
        c.title = "Wi-Fi".into();
        c.messages.push(StoredMessage { from_user: true, text: "hi".into(), meta: String::new() });
        c.messages.push(StoredMessage { from_user: false, text: "hello".into(), meta: "tool: x · 2 s".into() });
        assert_eq!(c.transcript("Jarvis"), "Wi-Fi\n\nYou: hi\n\nJarvis: hello\n(tool: x · 2 s)\n");
    }

    use super::*;

    fn msg(from_user: bool, text: &str) -> StoredMessage {
        StoredMessage { from_user, text: text.into(), meta: String::new() }
    }

    #[test]
    fn save_list_load_delete_roundtrip() {
        let dir = std::env::temp_dir().join("ojp-chats-test");
        let _ = std::fs::remove_dir_all(&dir);
        let store = ChatStore::new(&dir);
        let mut a = StoredChat { id: "1".into(), updated: 10, ..StoredChat::new() };
        a.messages = vec![msg(true, "When was the Roman Empire founded?"), msg(false, "27 BC.")];
        a.refresh_title();
        let mut b = StoredChat { id: "2".into(), updated: 20, ..StoredChat::new() };
        b.messages = vec![msg(true, "hi")];
        store.save(&a).unwrap();
        store.save(&b).unwrap();
        store.save(&StoredChat { id: "3".into(), ..StoredChat::new() }).unwrap(); // empty: not saved
        let ids: Vec<_> = store.list().into_iter().map(|c| c.id).collect();
        assert_eq!(ids, ["2", "1"]);
        assert_eq!(store.load("1").unwrap(), a);
        store.delete("1").unwrap();
        assert_eq!(store.list().len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn title_is_shortened_at_a_word() {
        let mut c = StoredChat::new();
        c.messages = vec![msg(true, "Please explain in detail how photosynthesis works in desert plants")];
        c.refresh_title();
        assert_eq!(c.title, "Please explain in detail how…");
    }

    #[test]
    fn ids_cannot_escape_the_directory() {
        let store = ChatStore::new(std::path::Path::new("/data"));
        assert_eq!(store.path("../../etc/passwd"), std::path::Path::new("/data/chats/etcpasswd.json"));
    }

    #[test]
    fn a_renamed_chat_keeps_its_title() {
        let dir = std::env::temp_dir().join("ojp-chats-rename-test");
        let _ = std::fs::remove_dir_all(&dir);
        let store = ChatStore::new(&dir);
        let mut chat = StoredChat { id: "7".into(), ..StoredChat::new() };
        chat.messages.push(msg(true, "first words"));
        chat.refresh_title();
        store.save(&chat).unwrap();
        assert_eq!(store.load("7").unwrap().title, "first words");
        store.rename("7", "  Holiday plans ").unwrap();
        let mut loaded = store.load("7").unwrap();
        assert_eq!(loaded.title, "Holiday plans");
        loaded.messages.push(msg(true, "another"));
        loaded.refresh_title();
        assert_eq!(loaded.title, "Holiday plans", "a later message must not overwrite a chosen title");
        store.rename("7", "").unwrap();
        assert_eq!(store.load("7").unwrap().title, "first words");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unsent_messages_are_kept_per_chat() {
        let dir = std::env::temp_dir().join("ojp-drafts-test");
        let _ = std::fs::remove_dir_all(&dir);
        let store = ChatStore::new(&dir);
        assert_eq!(store.draft("new"), "");
        store.set_draft("new", "half a thought").unwrap();
        store.set_draft("123", "another one").unwrap();
        assert_eq!(store.draft("new"), "half a thought");
        assert_eq!(ChatStore::new(&dir).draft("123"), "another one", "survives a restart");
        store.set_draft("new", "").unwrap();
        assert_eq!(store.draft("new"), "");
        assert_eq!(store.draft("123"), "another one");
        assert!(store.list().is_empty(), "drafts are not chats");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
