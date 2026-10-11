//! Where model files live. On Android the user picks phone storage or an
//! SD card (each app gets a private folder on both, no permission needed);
//! on desktop there is a single location. The choice is saved as
//! `[models] location` in config.toml, preserving the file's comments.
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct Location {
    /// Shown to the user, e.g. "Phone storage" or "SD card".
    pub label: String,
    /// Base directory; models go in `<path>/models/`.
    pub path: PathBuf,
}

impl Location {
    /// Free bytes on the filesystem holding this location.
    pub fn free_bytes(&self) -> Option<u64> {
        free_bytes(&self.path)
    }
}

#[cfg(unix)]
fn free_bytes(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: c_path is a valid NUL-terminated string and stat is a
    // properly sized out-parameter.
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    Some(stat.f_bavail as u64 * stat.f_frsize as u64)
}

#[cfg(not(unix))]
fn free_bytes(_path: &Path) -> Option<u64> {
    None
}

/// Persist the chosen location in config.toml, keeping comments and
/// everything else in the file unchanged.
pub fn save_location(config_path: &Path, location: &Path) -> Result<()> {
    let text = std::fs::read_to_string(config_path).unwrap_or_default();
    let mut doc: toml_edit::DocumentMut = text.parse().with_context(|| format!("parsing {}", config_path.display()))?;
    if !doc.contains_table("models") {
        doc["models"] = toml_edit::table();
    }
    doc["models"]["location"] = toml_edit::value(location.to_string_lossy().as_ref());
    std::fs::write(config_path, doc.to_string()).with_context(|| format!("writing {}", config_path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_location_keeps_comments_and_other_keys() {
        let dir = std::env::temp_dir().join("ojp-storage-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "# keep me\n[models]\n# and me\ngpu_layers = 5\n").unwrap();
        save_location(&path, Path::new("/sd/ojp")).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("# keep me") && text.contains("# and me") && text.contains("gpu_layers = 5"));
        let config: crate::Config = toml::from_str(&text).unwrap();
        assert_eq!(config.models.location.as_deref(), Some(Path::new("/sd/ojp")));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn free_bytes_of_temp_dir_is_known() {
        assert!(free_bytes(&std::env::temp_dir()).is_some());
    }
}
