//! The assistant's hands on a Linux desktop (fht-compositor): see which windows exist, and click and type in them
//! on its own transient seat, so the user's own mouse and keyboard are never touched. See `docs/desktop-control-plan.md`.
pub mod ipc;
pub mod keys;
pub mod seat;

use anyhow::{Result, bail};

use ipc::Placed;
use seat::Button;

/// One line per window: id, app, title, size.
pub fn list_windows() -> Result<String> {
    let windows = ipc::placed_windows()?;
    if windows.is_empty() {
        return Ok("There are no windows open.".into());
    }
    Ok(windows
        .iter()
        .map(|p| {
            let w = &p.window;
            format!(
                "{}: {} — {} ({}x{}){}",
                w.id,
                w.app_id.as_deref().unwrap_or("?"),
                w.title.as_deref().unwrap_or("(no title)"),
                w.size.0,
                w.size.1,
                if w.focused { " [your focused window]" } else { "" }
            )
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

fn pick<'a>(windows: &'a [Placed], name: &str) -> Result<&'a Placed> {
    let hits = ipc::find(windows, name);
    match hits.as_slice() {
        [one] => Ok(one),
        [] => bail!("no window matches {name:?}; list_windows shows what is open"),
        many => bail!(
            "{} windows match {name:?}: {}. Use the number.",
            many.len(),
            many.iter().map(|p| format!("{} ({})", p.window.id, p.window.title.as_deref().unwrap_or("?"))).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// Click inside a window at (x, y) measured from its top-left corner.
pub fn click(window: &str, x: i32, y: i32, button: Button, times: u32) -> Result<String> {
    let windows = ipc::placed_windows()?;
    let w = pick(&windows, window)?;
    if x < 0 || y < 0 || x >= w.window.size.0 as i32 || y >= w.window.size.1 as i32 {
        bail!("({x}, {y}) is outside the window, which is {}x{}", w.window.size.0, w.window.size.1);
    }
    let bounds = ipc::desktop_bounds(&ipc::outputs()?).ok_or_else(|| anyhow::anyhow!("no screen"))?;
    let (gx, gy) = (w.origin.0 + x, w.origin.1 + y);
    seat::with_session(|s| {
        s.move_to(gx, gy, bounds)?;
        s.click(button, times)
    })?;
    Ok(format!("Clicked at ({x}, {y}) in {}.", w.window.title.as_deref().unwrap_or("the window")))
}

/// Type text where the assistant's own keyboard focus is (the window it last clicked).
pub fn type_text(text: &str) -> Result<String> {
    let typed = seat::with_session(|s| s.type_text(text))?;
    Ok(format!("Typed {typed} characters."))
}

/// Press a key combination such as "ctrl+s" or "Return" in the window the assistant last clicked.
pub fn press_keys(combo: &str) -> Result<String> {
    let codes = keys::parse_combo(combo)?;
    seat::with_session(|s| s.press(&codes))?;
    Ok(format!("Pressed {combo}."))
}

/// Scroll the window under the assistant's pointer (positive scrolls down).
pub fn scroll(window: &str, amount: i32) -> Result<String> {
    let windows = ipc::placed_windows()?;
    let w = pick(&windows, window)?;
    let bounds = ipc::desktop_bounds(&ipc::outputs()?).ok_or_else(|| anyhow::anyhow!("no screen"))?;
    let (gx, gy) = (w.origin.0 + w.window.size.0 as i32 / 2, w.origin.1 + w.window.size.1 as i32 / 2);
    seat::with_session(|s| {
        s.move_to(gx, gy, bounds)?;
        s.scroll(f64::from(amount))
    })?;
    Ok(format!("Scrolled {amount} in {}.", w.window.title.as_deref().unwrap_or("the window")))
}
