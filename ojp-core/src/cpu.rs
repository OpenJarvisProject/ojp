//! Being a good neighbour on a phone: which cores the model may use, depending on whether anyone is using the phone.
//!
//! Phones have fast and slow cores. When the user is in another app, background work (reading the tool list ahead of
//! time, later: tasks the assistant runs on its own) is kept on the slow cores and a few threads, so the foreground app
//! keeps the fast ones and still feels normal. When the user is waiting for the assistant (they asked something in the
//! app), or the screen is off, it gets the fast cores. llama.cpp makes its worker threads from the calling thread, and
//! they inherit its core mask, so setting the mask of the calling thread before each batch is enough.
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The user is waiting, or isn't using the phone: the fast cores.
    Fast,
    /// The user is busy in something else: the slow cores, fewer threads.
    Polite,
}

/// Which cores are fast and which are slow (empty `little` when all cores are alike).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Topology {
    pub big: Vec<usize>,
    pub little: Vec<usize>,
}

/// Split cores by their capacity: those within 70% of the strongest are fast.
pub fn classify(capacities: &[(usize, u64)]) -> Topology {
    let Some(max) = capacities.iter().map(|c| c.1).max() else { return Topology::default() };
    let (mut big, mut little) = (Vec::new(), Vec::new());
    for &(cpu, cap) in capacities {
        if cap * 10 >= max * 7 { big.push(cpu) } else { little.push(cpu) }
    }
    Topology { big, little }
}

/// The phone's cores, from the kernel (`cpu_capacity`, else the highest frequency).
pub fn topology() -> &'static Topology {
    static TOPOLOGY: OnceLock<Topology> = OnceLock::new();
    TOPOLOGY.get_or_init(|| {
        let mut caps = Vec::new();
        for cpu in 0..64usize {
            let base = format!("/sys/devices/system/cpu/cpu{cpu}");
            if !std::path::Path::new(&base).exists() {
                break;
            }
            let read = |file: &str| std::fs::read_to_string(format!("{base}/{file}")).ok().and_then(|t| t.trim().parse::<u64>().ok());
            if let Some(cap) = read("cpu_capacity").or_else(|| read("cpufreq/cpuinfo_max_freq")) {
                caps.push((cpu, cap));
            }
        }
        classify(&caps)
    })
}

/// Whether the core and thread management is on: always on a phone, on a computer only with `OJP_GOVERNOR=1`.
pub fn enabled() -> bool {
    match std::env::var("OJP_GOVERNOR").as_deref() {
        Ok("0") => false,
        Ok(_) => true,
        Err(_) => cfg!(target_os = "android"),
    }
}

/// Threads to use in a mode (`None`: leave the configured number alone).
pub fn threads_for(mode: Mode, topology: &Topology) -> Option<i32> {
    if topology.big.is_empty() {
        return None;
    }
    Some(match mode {
        Mode::Fast => topology.big.len().clamp(2, 8) as i32,
        Mode::Polite => (if topology.little.is_empty() { (topology.big.len() / 2).max(1) } else { topology.little.len().clamp(1, 4) }) as i32,
    })
}

/// The cores a mode may run on.
pub fn cores_for(mode: Mode, topology: &Topology) -> Vec<usize> {
    match (mode, topology.little.is_empty()) {
        (Mode::Fast, _) => topology.big.clone(),
        (Mode::Polite, false) => topology.little.clone(),
        // All cores alike: the upper half in Polite mode would be arbitrary, so keep every core and just use fewer threads.
        (Mode::Polite, true) => topology.big.clone(),
    }
}

/// Limit the calling thread (and so the threads it makes next) to `cores`. Returns whether the kernel accepted it.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn pin_current_thread(cores: &[usize]) -> bool {
    if cores.is_empty() {
        return false;
    }
    // SAFETY: a zeroed cpu_set_t is the empty set; the libc macros only touch that struct.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cores {
            libc::CPU_SET(c, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub fn pin_current_thread(_cores: &[usize]) -> bool {
    false
}

// --- what the work is, and what the phone is doing ---------------------------------------------------------------

static BACKGROUND: AtomicU8 = AtomicU8::new(0);
static BACKGROUND_MODE: AtomicU8 = AtomicU8::new(0);

/// Mark the work being done now as background work (not something the user is waiting for).
pub fn set_background(on: bool) {
    BACKGROUND.store(u8::from(on), Ordering::Relaxed);
}

/// How polite background work should be right now (set by [`governor`] from what the phone is doing).
pub fn set_background_mode(mode: Mode) {
    BACKGROUND_MODE.store(u8::from(mode == Mode::Polite), Ordering::Relaxed);
}

/// The mode for the work being done now: the user's own requests always get the fast cores.
pub fn current_mode() -> Mode {
    if BACKGROUND.load(Ordering::Relaxed) == 1 && BACKGROUND_MODE.load(Ordering::Relaxed) == 1 { Mode::Polite } else { Mode::Fast }
}

/// What the phone is doing, from the platform layer ("app", "phone" or "idle", then flags like "hot" and "saver").
pub fn mode_for_activity(activity: &str) -> Mode {
    let mut words = activity.split_whitespace();
    let main = words.next().unwrap_or("idle");
    if main == "phone" || words.any(|w| w == "hot" || w == "saver") { Mode::Polite } else { Mode::Fast }
}

/// Start watching what the phone is doing (every few seconds) so background work can step aside. Does nothing
/// without a platform layer.
pub fn start_governor() {
    if !enabled() || !crate::platform::available() {
        return;
    }
    std::thread::Builder::new()
        .name("ojp-governor".into())
        .spawn(|| loop {
            if let Ok(activity) = crate::platform::call("activity", "") {
                set_background_mode(mode_for_activity(&activity));
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
        })
        .ok();
}

thread_local! {
    static APPLIED: std::cell::Cell<Option<Mode>> = const { std::cell::Cell::new(None) };
}

/// Move the calling thread to the cores of `mode` if it isn't there already; returns the thread count to use.
pub fn tune_current_thread(mode: Mode) -> Option<i32> {
    if !enabled() {
        return None;
    }
    let topology = topology();
    if APPLIED.with(|a| a.get()) != Some(mode) {
        let cores = cores_for(mode, topology);
        let pinned = pin_current_thread(&cores);
        APPLIED.with(|a| a.set(Some(mode)));
        eprintln!("[cpu] {mode:?}: cores {cores:?} (pinned: {pinned}), threads {:?}", threads_for(mode, topology));
    }
    threads_for(mode, topology)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cores_split_into_fast_and_slow() {
        // 4 slow (cap 400), 3 medium (800), 1 prime (1024): the medium and prime ones count as fast.
        let caps: Vec<(usize, u64)> = vec![(0, 400), (1, 400), (2, 400), (3, 400), (4, 800), (5, 800), (6, 800), (7, 1024)];
        let t = classify(&caps);
        assert_eq!(t.little, vec![0, 1, 2, 3]);
        assert_eq!(t.big, vec![4, 5, 6, 7]);
        assert_eq!(threads_for(Mode::Fast, &t), Some(4));
        assert_eq!(threads_for(Mode::Polite, &t), Some(4));
        assert_eq!(cores_for(Mode::Polite, &t), vec![0, 1, 2, 3]);
        // All alike: nothing to separate; politeness is fewer threads.
        let even = classify(&[(0, 1000), (1, 1000), (2, 1000), (3, 1000)]);
        assert!(even.little.is_empty());
        assert_eq!(threads_for(Mode::Polite, &even), Some(2));
        assert_eq!(classify(&[]), Topology::default());
        assert_eq!(threads_for(Mode::Fast, &Topology::default()), None);
    }

    #[test]
    fn phone_activity_picks_the_mode() {
        assert_eq!(mode_for_activity("app"), Mode::Fast);
        assert_eq!(mode_for_activity("idle"), Mode::Fast);
        assert_eq!(mode_for_activity("phone"), Mode::Polite);
        assert_eq!(mode_for_activity("app hot"), Mode::Polite);
        assert_eq!(mode_for_activity("idle saver"), Mode::Polite);
        assert_eq!(mode_for_activity(""), Mode::Fast);
    }

    #[test]
    fn only_background_work_is_ever_polite() {
        set_background_mode(Mode::Polite);
        set_background(false);
        assert_eq!(current_mode(), Mode::Fast, "the user's own request is never slowed");
        set_background(true);
        assert_eq!(current_mode(), Mode::Polite);
        set_background(false);
        set_background_mode(Mode::Fast);
    }
}
