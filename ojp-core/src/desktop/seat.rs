//! The assistant's own seat: a transient `wl_seat` (ext-transient-seat-v1) with a virtual keyboard and a virtual
//! pointer, so it can click and type in windows without touching the user's mouse and keyboard.
//! One session lives for as long as the program does (the compositor removes the seat when the connection closes).
use std::os::fd::AsFd;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use wayland_client::protocol::{wl_pointer, wl_registry, wl_seat};
use wayland_client::{Connection, Dispatch, EventQueue, QueueHandle, delegate_noop};
use wayland_protocols::ext::transient_seat::v1::client::{ext_transient_seat_manager_v1, ext_transient_seat_v1};
use wayland_protocols_misc::zwp_virtual_keyboard_v1::client::{zwp_virtual_keyboard_manager_v1, zwp_virtual_keyboard_v1};
use wayland_protocols_wlr::virtual_pointer::v1::client::{zwlr_virtual_pointer_manager_v1, zwlr_virtual_pointer_v1};

use super::keys::{self, KEY_LEFTSHIFT};

const BTN_LEFT: u32 = 0x110;
const BTN_RIGHT: u32 = 0x111;
const BTN_MIDDLE: u32 = 0x112;
const EXTENT: u32 = 100_000;

/// A plain keymap that names the system's own pieces: whoever reads it (the compositor) uses its own xkeyboard-config.
const KEYMAP: &str = "xkb_keymap {\n xkb_keycodes { include \"evdev+aliases(qwerty)\" };\n xkb_types { include \"complete\" };\n xkb_compat { include \"complete\" };\n xkb_symbols { include \"pc+us+inet(evdev)\" };\n};\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Button {
    Left,
    Right,
    Middle,
}

#[derive(Default)]
struct State {
    globals: Vec<(u32, String, u32)>,
    seat_manager: Option<ext_transient_seat_manager_v1::ExtTransientSeatManagerV1>,
    keyboard_manager: Option<zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1>,
    pointer_manager: Option<zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1>,
    ready: Option<u32>,
    denied: bool,
}

impl Dispatch<wl_registry::WlRegistry, ()> for State {
    fn event(s: &mut Self, registry: &wl_registry::WlRegistry, event: wl_registry::Event, _: &(), _: &Connection, qh: &QueueHandle<Self>) {
        if let wl_registry::Event::Global { name, interface, version } = event {
            match interface.as_str() {
                "ext_transient_seat_manager_v1" => s.seat_manager = Some(registry.bind(name, 1, qh, ())),
                "zwp_virtual_keyboard_manager_v1" => s.keyboard_manager = Some(registry.bind(name, 1, qh, ())),
                "zwlr_virtual_pointer_manager_v1" => s.pointer_manager = Some(registry.bind(name, version.min(2), qh, ())),
                _ => {}
            }
            s.globals.push((name, interface, version));
        }
    }
}

impl Dispatch<ext_transient_seat_v1::ExtTransientSeatV1, ()> for State {
    fn event(s: &mut Self, _: &ext_transient_seat_v1::ExtTransientSeatV1, event: ext_transient_seat_v1::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        match event {
            ext_transient_seat_v1::Event::Ready { global_name } => s.ready = Some(global_name),
            ext_transient_seat_v1::Event::Denied => s.denied = true,
            _ => {}
        }
    }
}

delegate_noop!(State: ignore ext_transient_seat_manager_v1::ExtTransientSeatManagerV1);
delegate_noop!(State: ignore zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1);
delegate_noop!(State: ignore zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1);
delegate_noop!(State: ignore zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1);
delegate_noop!(State: ignore zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1);
delegate_noop!(State: ignore wl_seat::WlSeat);

/// A live transient seat with its virtual devices.
pub struct Session {
    queue: EventQueue<State>,
    state: State,
    keyboard: zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1,
    pointer: zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1,
    started: Instant,
    /// Kept so the seat lives as long as the session.
    _seat: ext_transient_seat_v1::ExtTransientSeatV1,
    _wl_seat: wl_seat::WlSeat,
    _conn: Connection,
}

impl Session {
    /// Connect to the compositor and create the seat with its keyboard and pointer.
    pub fn open() -> Result<Session> {
        let conn = Connection::connect_to_env().context("couldn't connect to the Wayland compositor (is WAYLAND_DISPLAY set?)")?;
        let mut queue = conn.new_event_queue();
        let qh = queue.handle();
        let registry = conn.display().get_registry(&qh, ());
        let mut state = State::default();
        queue.roundtrip(&mut state)?;
        queue.roundtrip(&mut state)?;
        let manager = state.seat_manager.clone().context("this compositor doesn't offer transient seats (ext_transient_seat_manager_v1)")?;
        let kb_manager = state.keyboard_manager.clone().context("no virtual keyboard support (zwp_virtual_keyboard_manager_v1)")?;
        let ptr_manager = state.pointer_manager.clone().context("no virtual pointer support (zwlr_virtual_pointer_manager_v1)")?;
        let seat = manager.create(&qh, ());
        for _ in 0..4 {
            queue.roundtrip(&mut state)?;
            if state.ready.is_some() || state.denied {
                break;
            }
        }
        if state.denied {
            bail!("the compositor refused another transient seat (a client may hold 8)");
        }
        let global = state.ready.context("the compositor never answered the transient seat request")?;
        let mut tries = 0;
        let (name, _, version) = loop {
            if let Some(g) = state.globals.iter().find(|g| g.0 == global && g.1 == "wl_seat") {
                break g.clone();
            }
            tries += 1;
            if tries > 6 {
                bail!("the new seat's wl_seat global ({global}) never appeared");
            }
            queue.roundtrip(&mut state)?;
        };
        let wl_seat: wl_seat::WlSeat = registry.bind(name, version.min(7), &qh, ());
        let keyboard = kb_manager.create_virtual_keyboard(&wl_seat, &qh, ());
        let mut file = tempfile::tempfile()?;
        std::io::Write::write_all(&mut file, KEYMAP.as_bytes())?;
        std::io::Write::write_all(&mut file, &[0])?;
        keyboard.keymap(1, file.as_fd(), KEYMAP.len() as u32 + 1);
        let pointer = ptr_manager.create_virtual_pointer(Some(&wl_seat), &qh, ());
        queue.roundtrip(&mut state)?;
        Ok(Session { queue, state, keyboard, pointer, started: Instant::now(), _seat: seat, _wl_seat: wl_seat, _conn: conn })
    }

    fn time(&self) -> u32 {
        self.started.elapsed().as_millis() as u32
    }

    fn flush(&mut self) -> Result<()> {
        self.queue.roundtrip(&mut self.state)?;
        Ok(())
    }

    /// Put the pointer at a point on the desktop (logical pixels); `bounds` is the rectangle all outputs cover.
    pub fn move_to(&mut self, x: i32, y: i32, bounds: (i32, i32, u32, u32)) -> Result<()> {
        let (bx, by, bw, bh) = bounds;
        if bw == 0 || bh == 0 {
            bail!("the desktop has no size");
        }
        let fx = f64::from(x - bx) / f64::from(bw);
        let fy = f64::from(y - by) / f64::from(bh);
        if !(0.0..=1.0).contains(&fx) || !(0.0..=1.0).contains(&fy) {
            bail!("({x}, {y}) is outside the screen");
        }
        self.pointer.motion_absolute(self.time(), (fx * f64::from(EXTENT)) as u32, (fy * f64::from(EXTENT)) as u32, EXTENT, EXTENT);
        self.pointer.frame();
        self.flush()
    }

    pub fn click(&mut self, button: Button, times: u32) -> Result<()> {
        let code = match button {
            Button::Left => BTN_LEFT,
            Button::Right => BTN_RIGHT,
            Button::Middle => BTN_MIDDLE,
        };
        for _ in 0..times.max(1) {
            self.pointer.button(self.time(), code, wl_pointer::ButtonState::Pressed);
            self.pointer.frame();
            self.pointer.button(self.time(), code, wl_pointer::ButtonState::Released);
            self.pointer.frame();
            self.flush()?;
            if times > 1 {
                std::thread::sleep(Duration::from_millis(60));
            }
        }
        Ok(())
    }

    /// Scroll vertically: positive scrolls down.
    pub fn scroll(&mut self, amount: f64) -> Result<()> {
        self.pointer.axis(self.time(), wl_pointer::Axis::VerticalScroll, amount);
        self.pointer.frame();
        self.flush()
    }

    fn key(&mut self, code: u32, down: bool) {
        self.keyboard.key(self.time(), code, u32::from(down));
    }

    /// Type text on a US keyboard; returns how many characters were typed (others have no key and are skipped).
    pub fn type_text(&mut self, text: &str) -> Result<usize> {
        let mut typed = 0;
        for c in text.chars() {
            let Some(k) = keys::key_for_char(c) else { continue };
            if k.shift {
                self.key(KEY_LEFTSHIFT, true);
                self.keyboard.modifiers(1, 0, 0, 0);
            }
            self.key(k.code, true);
            self.key(k.code, false);
            if k.shift {
                self.keyboard.modifiers(0, 0, 0, 0);
                self.key(KEY_LEFTSHIFT, false);
            }
            typed += 1;
            self.flush()?;
            std::thread::sleep(Duration::from_millis(6));
        }
        Ok(typed)
    }

    /// Press a key combination: hold all but the last, press the last, release in reverse.
    pub fn press(&mut self, codes: &[u32]) -> Result<()> {
        let Some((last, held)) = codes.split_last() else { bail!("no keys") };
        let mut mods = 0u32;
        for &m in held {
            self.key(m, true);
            mods |= match m {
                42 | 54 => 1,
                29 | 97 => 4,
                56 | 100 => 8,
                125 | 126 => 64,
                _ => 0,
            };
        }
        self.keyboard.modifiers(mods, 0, 0, 0);
        self.key(*last, true);
        self.key(*last, false);
        for &m in held.iter().rev() {
            self.key(m, false);
        }
        self.keyboard.modifiers(0, 0, 0, 0);
        self.flush()
    }
}

static SESSION: Mutex<Option<Session>> = Mutex::new(None);

/// Run something with the assistant's seat, opening it the first time. If the compositor connection broke the seat is
/// opened again once.
pub fn with_session<T>(mut f: impl FnMut(&mut Session) -> Result<T>) -> Result<T> {
    let mut guard = SESSION.lock().unwrap_or_else(|e| e.into_inner());
    for attempt in 0..2 {
        if guard.is_none() {
            *guard = Some(Session::open()?);
        }
        match f(guard.as_mut().expect("just opened")) {
            Ok(v) => return Ok(v),
            Err(e) if attempt == 0 && format!("{e:#}").contains("Broken pipe") || format!("{e:#}").contains("protocol") => {
                *guard = None;
            }
            Err(e) => return Err(e),
        }
    }
    bail!("the compositor connection keeps failing")
}
