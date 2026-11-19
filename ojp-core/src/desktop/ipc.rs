//! fht-compositor's IPC (JSON lines over a Unix socket): which windows exist and where they are.
//! Written against the compositor's own protocol (kebab-case names), without depending on its crate.
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct Window {
    pub id: usize,
    pub title: Option<String>,
    pub app_id: Option<String>,
    pub workspace_id: usize,
    pub size: (u32, u32),
    pub location: (i32, i32),
    pub focused: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Output {
    pub name: String,
    pub position: (i32, i32),
    pub size: (u32, u32),
}

#[derive(Debug, Clone, Deserialize)]
struct Workspace {
    output: String,
}

/// A window with its place on the whole desktop (logical pixels).
#[derive(Debug, Clone, PartialEq)]
pub struct Placed {
    pub window: Window,
    /// Top-left corner of the window in desktop coordinates.
    pub origin: (i32, i32),
    pub output: String,
}

/// The socket to use: `FHTC_SOCKET_PATH`, else the newest `fhtc-*.socket` in the runtime directory.
pub fn socket_path() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("FHTC_SOCKET_PATH") {
        return Ok(PathBuf::from(p));
    }
    let dir = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).context("no XDG_RUNTIME_DIR")?;
    let mut found: Vec<_> = std::fs::read_dir(&dir)?
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("fhtc-") && e.file_name().to_string_lossy().ends_with(".socket"))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    found.sort();
    found.pop().map(|f| f.1).context("fht-compositor's IPC socket wasn't found (is it running? set FHTC_SOCKET_PATH)")
}

/// Send one request and read the one-line answer.
pub fn request(req: &Value) -> Result<Value> {
    let mut stream = UnixStream::connect(socket_path()?).context("couldn't connect to the compositor's IPC socket")?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
    writeln!(stream, "{req}")?;
    let mut line = String::new();
    BufReader::new(&stream).read_line(&mut line)?;
    let answer: Value = serde_json::from_str(line.trim()).with_context(|| format!("unreadable IPC answer {line:?}"))?;
    if let Some(e) = answer.get("error").and_then(Value::as_str) {
        bail!("the compositor said: {e}");
    }
    Ok(answer)
}

pub fn outputs() -> Result<Vec<Output>> {
    let answer = request(&json!("outputs"))?;
    let map: HashMap<String, Output> = serde_json::from_value(answer["outputs"].clone()).context("unexpected Outputs answer")?;
    let mut list: Vec<Output> = map.into_values().collect();
    list.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(list)
}

pub fn windows() -> Result<Vec<Window>> {
    let answer = request(&json!("windows"))?;
    let map: HashMap<usize, Window> = serde_json::from_value(answer["windows"].clone()).context("unexpected Windows answer")?;
    let mut list: Vec<Window> = map.into_values().collect();
    list.sort_by_key(|w| w.id);
    Ok(list)
}

/// Every window with its place on the desktop.
pub fn placed_windows() -> Result<Vec<Placed>> {
    let outputs = outputs()?;
    let mut out = Vec::new();
    let mut workspace_output: HashMap<usize, String> = HashMap::new();
    for window in windows()? {
        let output_name = match workspace_output.get(&window.workspace_id) {
            Some(n) => n.clone(),
            None => {
                let answer = request(&json!({ "workspace": window.workspace_id }))?;
                let ws: Option<Workspace> = serde_json::from_value(answer["workspace"].clone()).unwrap_or(None);
                let name = ws.map(|w| w.output).unwrap_or_default();
                workspace_output.insert(window.workspace_id, name.clone());
                name
            }
        };
        let base = outputs.iter().find(|o| o.name == output_name).or(outputs.first()).map_or((0, 0), |o| o.position);
        out.push(Placed { origin: (base.0 + window.location.0, base.1 + window.location.1), output: output_name, window });
    }
    Ok(out)
}

/// The rectangle that covers every output: (x, y, width, height).
pub fn desktop_bounds(outputs: &[Output]) -> Option<(i32, i32, u32, u32)> {
    let min_x = outputs.iter().map(|o| o.position.0).min()?;
    let min_y = outputs.iter().map(|o| o.position.1).min()?;
    let max_x = outputs.iter().map(|o| o.position.0 + o.size.0 as i32).max()?;
    let max_y = outputs.iter().map(|o| o.position.1 + o.size.1 as i32).max()?;
    Some((min_x, min_y, (max_x - min_x) as u32, (max_y - min_y) as u32))
}

/// The windows matching a name (title or app id, any case), or the one with this numeric id.
pub fn find<'a>(windows: &'a [Placed], name: &str) -> Vec<&'a Placed> {
    let wanted = name.trim().to_lowercase();
    if let Ok(id) = wanted.parse::<usize>()
        && let Some(w) = windows.iter().find(|w| w.window.id == id)
    {
        return vec![w];
    }
    let hits = |exact: bool| -> Vec<&'a Placed> {
        windows
            .iter()
            .filter(|w| {
                [w.window.title.as_deref(), w.window.app_id.as_deref()].into_iter().flatten().any(|t| {
                    let t = t.to_lowercase();
                    if exact { t == wanted } else { t.contains(&wanted) }
                })
            })
            .collect()
    };
    let exact = hits(true);
    if exact.is_empty() { hits(false) } else { exact }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placed(id: usize, title: &str, app: &str) -> Placed {
        Placed {
            window: Window { id, title: Some(title.into()), app_id: Some(app.into()), workspace_id: 0, size: (100, 100), location: (0, 0), focused: false },
            origin: (0, 0),
            output: "o".into(),
        }
    }

    #[test]
    fn windows_are_found_by_id_title_or_app() {
        let all = vec![placed(1, "Notes", "org.gnome.TextEditor"), placed(2, "main.rs - Editor", "code"), placed(3, "Terminal", "foot")];
        assert_eq!(find(&all, "2")[0].window.id, 2);
        assert_eq!(find(&all, "foot")[0].window.id, 3);
        assert_eq!(find(&all, "notes")[0].window.id, 1);
        assert_eq!(find(&all, "editor").len(), 2, "both contain it, so the caller must say which");
        assert!(find(&all, "browser").is_empty());
    }

    #[test]
    fn bounds_cover_all_outputs() {
        let o = |x, y, w, h| Output { name: "x".into(), position: (x, y), size: (w, h) };
        assert_eq!(desktop_bounds(&[o(0, 0, 1920, 1080)]), Some((0, 0, 1920, 1080)));
        assert_eq!(desktop_bounds(&[o(0, 0, 1920, 1080), o(1920, -200, 1280, 1024)]), Some((0, -200, 3200, 1280)));
        assert_eq!(desktop_bounds(&[]), None);
    }
}
