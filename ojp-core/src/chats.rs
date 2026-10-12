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
}

impl StoredChat {
    pub fn new() -> StoredChat {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        StoredChat { id: format!("{}", now.as_millis()), title: String::new(), updated: now.as_secs(), messages: Vec::new() }
    }

    /// Title from the first user message, shortened at a word boundary.
    pub fn refresh_title(&mut self) {
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

    pub fn touch(&mut self) {
        self.updated = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    }
}

impl Default for StoredChat {
    fn default() -> Self {
        StoredChat::new()
    }
}

pub struct ChatStore {
    dir: PathBuf,
}

impl ChatStore {
    pub fn new(data_dir: &std::path::Path) -> ChatStore {
        ChatStore { dir: data_dir.join("chats") }
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
}
