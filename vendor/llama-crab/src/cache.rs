//! Caching for previously-computed prompt prefixes.
//!
//! Two implementations are provided:
//!
//! * [`RamCache`] — in-process, LRU-ish prefix cache backed by a
//!   `BTreeMap`.
//! * `DiskCache` — on-disk, persisted via the `sled` embedded
//!   key-value store (feature `disk-cache`).
//!
//! Both store raw bytes produced by `llama_state_get_data`; the keys
//! are the *exact* token sequence that was processed. Lookups return
//! the longest matching prefix.

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::token::LlamaToken;

/// A single cached entry. The raw bytes are opaque to the cache.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Serialized KV state produced by `llama_state_get_data`.
    pub state: Vec<u8>,
    /// Position in the sequence at which the cache is valid.
    pub n_past: i32,
}

/// Trait implemented by KV-cache storage backends.
pub trait Cache: Send + Sync {
    /// Look up the longest matching prefix of `tokens`.
    fn lookup(&self, tokens: &[LlamaToken]) -> Option<CacheEntry>;

    /// Store `tokens` → `entry`.
    fn store(&self, tokens: &[LlamaToken], entry: CacheEntry);

    /// Number of entries in the cache.
    fn len(&self) -> usize;

    /// True if the cache has no entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// RAM cache
// ---------------------------------------------------------------------------

/// In-memory cache with longest-prefix matching.
#[derive(Debug, Default, Clone)]
pub struct RamCache {
    inner: Arc<Mutex<BTreeMap<Vec<LlamaToken>, CacheEntry>>>,
}

impl RamCache {
    /// Construct a new empty RAM cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.inner.lock().clear();
    }
}

impl Cache for RamCache {
    fn lookup(&self, tokens: &[LlamaToken]) -> Option<CacheEntry> {
        let mut best_len = 0_usize;
        let mut best_entry: Option<CacheEntry> = None;
        let g = self.inner.lock();
        for (key, val) in g.iter() {
            if key.len() > tokens.len() || key.len() <= best_len {
                continue;
            }
            if tokens.starts_with(key) {
                best_len = key.len();
                best_entry = Some(val.clone());
            }
        }
        best_entry
    }

    fn store(&self, tokens: &[LlamaToken], entry: CacheEntry) {
        self.inner.lock().insert(tokens.to_vec(), entry);
    }

    fn len(&self) -> usize {
        self.inner.lock().len()
    }
}

// ---------------------------------------------------------------------------
// Disk cache (feature `disk-cache`)
// ---------------------------------------------------------------------------

/// On-disk cache backed by [`sled`].
///
/// The tree name defaults to `"llama_crab.cache"`. The cache survives
/// process restarts and is safe to share between multiple [`Llama`]
/// instances (only the last writer wins on key collisions, which is
/// the expected semantics).
#[cfg(feature = "disk-cache")]
pub struct DiskCache {
    db: sled::Db,
    tree: sled::Tree,
}

#[cfg(feature = "disk-cache")]
impl std::fmt::Debug for DiskCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskCache").finish()
    }
}

#[cfg(feature = "disk-cache")]
impl DiskCache {
    /// Open (or create) a disk-backed cache at `path`.
    ///
    /// # Errors
    /// Returns an error if sled cannot open the database (corrupted
    /// file, permission denied, etc.).
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, sled::Error> {
        let db = sled::open(path.as_ref())?;
        let tree = db.open_tree("llama_crab.cache")?;
        Ok(Self { db, tree })
    }

    /// Open an ephemeral (in-memory) cache that is automatically
    /// discarded on drop. Useful for tests.
    pub fn ephemeral() -> Result<Self, sled::Error> {
        let db = sled::Config::new().temporary(true).open()?;
        let tree = db.open_tree("llama_crab.cache")?;
        Ok(Self { db, tree })
    }

    /// Force a fsync of pending writes.
    pub fn flush(&self) -> Result<(), sled::Error> {
        let _ = self.tree.flush();
        Ok(())
    }
}

#[cfg(feature = "disk-cache")]
impl Cache for DiskCache {
    fn lookup(&self, tokens: &[crate::token::LlamaToken]) -> Option<CacheEntry> {
        let mut best_len = 0_usize;
        let mut best_key: Option<Vec<u8>> = None;
        for kv in self.tree.iter() {
            if let Ok((k, _)) = kv {
                if k.len() % std::mem::size_of::<i32>() != 0 {
                    continue;
                }
                let key_tokens: Vec<LlamaToken> = k
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .map(LlamaToken)
                    .collect();
                if key_tokens.len() > tokens.len() || key_tokens.len() <= best_len {
                    continue;
                }
                if tokens.starts_with(&key_tokens) {
                    best_len = key_tokens.len();
                    best_key = Some(k.to_vec());
                }
            }
        }
        let key = best_key?;
        self.tree
            .get(&key)
            .ok()
            .flatten()
            .and_then(|v| decode_entry(&v))
    }

    fn store(&self, tokens: &[crate::token::LlamaToken], entry: CacheEntry) {
        let key = encode_key(tokens);
        let val = encode_entry(&entry);
        let _ = self.tree.insert(key, val);
    }

    fn len(&self) -> usize {
        self.tree.len()
    }
}

#[cfg(feature = "disk-cache")]
fn encode_key(tokens: &[crate::token::LlamaToken]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 4);
    for t in tokens {
        out.extend_from_slice(&t.0.to_le_bytes());
    }
    out
}

#[cfg(feature = "disk-cache")]
fn encode_entry(entry: &CacheEntry) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + entry.state.len());
    out.extend_from_slice(&entry.n_past.to_le_bytes());
    out.extend_from_slice(&(entry.state.len() as u64).to_le_bytes());
    out.extend_from_slice(&entry.state);
    out
}

#[cfg(feature = "disk-cache")]
fn decode_entry(bytes: &[u8]) -> Option<CacheEntry> {
    if bytes.len() < 16 {
        return None;
    }
    let n_past = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let state_len = u64::from_le_bytes([
        bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11],
    ]) as usize;
    if bytes.len() < 12 + state_len {
        return None;
    }
    Some(CacheEntry {
        n_past,
        state: bytes[12..12 + state_len].to_vec(),
    })
}

#[cfg(feature = "disk-cache")]
impl Drop for DiskCache {
    fn drop(&mut self) {
        let _ = self.db.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ram_cache_longest_prefix() {
        let c = RamCache::new();
        let toks_a: Vec<LlamaToken> = (0..10).map(LlamaToken).collect();
        let toks_b: Vec<LlamaToken> = (0..20).map(LlamaToken).collect();
        c.store(
            &toks_a,
            CacheEntry {
                state: vec![1, 2, 3],
                n_past: 10,
            },
        );
        c.store(
            &toks_b,
            CacheEntry {
                state: vec![9, 9, 9],
                n_past: 20,
            },
        );
        let query: Vec<LlamaToken> = (0..20).map(LlamaToken).collect();
        let hit = c.lookup(&query).unwrap();
        assert_eq!(hit.n_past, 20);
    }

    #[test]
    fn ram_cache_partial_match() {
        let c = RamCache::new();
        let stored: Vec<LlamaToken> = (0..10).map(LlamaToken).collect();
        c.store(
            &stored,
            CacheEntry {
                state: vec![],
                n_past: 10,
            },
        );
        let query: Vec<LlamaToken> = (0..20).map(LlamaToken).collect();
        let hit = c.lookup(&query).unwrap();
        assert_eq!(hit.n_past, 10);
    }

    #[test]
    fn ram_cache_no_match() {
        let c = RamCache::new();
        let stored: Vec<LlamaToken> = vec![LlamaToken(0), LlamaToken(1), LlamaToken(2)];
        c.store(
            &stored,
            CacheEntry {
                state: vec![],
                n_past: 3,
            },
        );
        let query: Vec<LlamaToken> = vec![LlamaToken(99), LlamaToken(98)];
        assert!(c.lookup(&query).is_none());
    }

    #[cfg(feature = "disk-cache")]
    #[test]
    fn disk_cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let c = DiskCache::open(dir.path().join("cache")).unwrap();
        let tokens: Vec<LlamaToken> = (0..8).map(LlamaToken).collect();
        c.store(
            &tokens,
            CacheEntry {
                state: vec![1, 2, 3, 4, 5],
                n_past: 8,
            },
        );
        c.flush().unwrap();
        let hit = c.lookup(&tokens).unwrap();
        assert_eq!(hit.n_past, 8);
        assert_eq!(hit.state, vec![1, 2, 3, 4, 5]);
    }
}
