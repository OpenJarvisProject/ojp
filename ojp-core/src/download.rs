//! First-run model download. Each model in config `[download]` whose file
//! is missing can be fetched in-app: streamed to `<file>.part` (resumed
//! with an HTTP Range request if interrupted), checked against its pinned
//! SHA-256, and only then renamed into place, so a half-downloaded or
//! corrupted file is never loaded.
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::config::Config;

#[derive(Debug, Clone)]
pub struct Item {
    /// "router" or "main".
    pub role: &'static str,
    pub url: String,
    pub sha256: Option<String>,
    pub dest: PathBuf,
}

/// Configured downloads whose destination file doesn't exist yet. A main
/// model served remotely (main_url) is never downloaded.
pub fn missing(config: &Config) -> Vec<Item> {
    let d = &config.download;
    let mut items = Vec::new();
    let mut add = |role, url: &Option<String>, sha256: &Option<String>, dest: &Option<PathBuf>| {
        if let (Some(url), Some(dest)) = (url, dest)
            && !dest.exists()
        {
            items.push(Item { role, url: url.clone(), sha256: sha256.clone(), dest: dest.clone() });
        }
    };
    add("router", &d.router, &d.router_sha256, &config.models.router);
    if config.models.main_url.is_none() && !d.skip_main && config.models.main_enabled {
        add("main", &d.main, &d.main_sha256, &config.models.main);
    }
    items
}

/// The voice files, if not downloaded yet. Separate from [`missing`]:
/// they are only fetched once the user turns speech on.
pub fn tts_missing(config: &Config) -> Vec<Item> {
    let d = &config.download;
    let mut items = Vec::new();
    for (url, sha256, dest) in [
        (&d.tts_model, &d.tts_model_sha256, &config.tts.model),
        (&d.tts_voices, &d.tts_voices_sha256, &config.tts.voices),
    ] {
        if let Some(url) = url
            && !dest.exists()
        {
            items.push(Item { role: "voice", url: url.clone(), sha256: sha256.clone(), dest: dest.clone() });
        }
    }
    items
}

/// Size in bytes, from a HEAD request (None if the server doesn't say).
pub fn size(url: &str) -> Result<Option<u64>> {
    let response = ureq::head(url).call()?;
    Ok(response.headers().get("content-length").and_then(|v| v.to_str().ok()?.parse().ok()))
}

/// Download one item, reporting (bytes done, total bytes if known).
pub fn fetch(item: &Item, progress: &mut dyn FnMut(u64, Option<u64>)) -> Result<()> {
    if let Some(parent) = item.dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = part_path(&item.dest);
    let have = std::fs::metadata(&part).map_or(0, |m| m.len());

    let mut request = ureq::get(&item.url);
    if have > 0 {
        request = request.header("Range", format!("bytes={have}-"));
    }
    let mut response = request.call().with_context(|| format!("downloading {}", item.url))?;
    // 206 = the server honoured the Range header; anything else restarts.
    let resumed = have > 0 && response.status() == 206;
    let start = if resumed { have } else { 0 };
    let remaining: Option<u64> =
        response.headers().get("content-length").and_then(|v| v.to_str().ok()?.parse().ok());
    let total = remaining.map(|n| n + start);

    let mut file = OpenOptions::new().create(true).write(true).append(resumed).truncate(!resumed).open(&part)?;
    let mut reader = response.body_mut().as_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = start;
    progress(done, total);
    loop {
        let n = reader.read(&mut buf).context("download interrupted")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        progress(done, total);
    }
    file.sync_all()?;
    drop(file);
    if let Some(total) = total
        && done != total
    {
        bail!("download ended early ({done} of {total} bytes); try again to resume");
    }

    if let Some(expected) = &item.sha256 {
        let actual = sha256_file(&part)?;
        if !actual.eq_ignore_ascii_case(expected) {
            std::fs::remove_file(&part)?;
            bail!("{} failed its checksum (got {actual}, expected {expected}); deleted, try again", item.role);
        }
    }
    std::fs::rename(&part, &item.dest)?;
    Ok(())
}

fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    dest.with_file_name(name)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn part_path_appends_suffix() {
        assert_eq!(part_path(Path::new("/m/main.gguf")), Path::new("/m/main.gguf.part"));
    }

    #[test]
    fn sha256_of_known_content() {
        let path = std::env::temp_dir().join("ojp-sha-test");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(sha256_file(&path).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        std::fs::remove_file(path).unwrap();
    }
}
