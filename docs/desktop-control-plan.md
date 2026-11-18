# Next version: desktop hands (plan)

Decisions (user, 2026-11-17, refined 2026-11-18): the brain stays **local only** (nothing is exposed to outside agents);
the AI **sees the desktop as text**, no vision model yet; the first demo is a **dev loop in the background**. The
**transient seat is the main way of acting** (it works in every app, which the accessibility tree does not): accessibility
is only an optional helper for finding things. Target: the user's fht-compositor (branch `feat/touch-transient`, transient seats), on an aarch64 laptop
that is short of RAM. Voice, morning brief and root support are not part of this version.

## The idea

A small model can't read a screen and aim a mouse. So the code does the aiming and the model only picks names:

- **High-level tools** that find things themselves (`click_control(window, "Save")`, `type_text(window, "...")`).
- **Recipes**: saved routines (open the project, run the tests, report) that the model triggers by name.
- Input goes through the AI's own **transient seat**, so the user's mouse and keyboard are never touched.

No MCP server for now (nothing outside should drive it); the tools are plain Rust tools in the same registry as the
phone's. An MCP *client* for light offline servers stays on the list for later.

## Milestones

**M0, needed from the compositor session** (not OJP work): transient seat stable (the crash from seat-test fixed); IPC
that lists windows (id, app id, title, geometry, workspace, focused) and can focus/raise a window and switch workspace
(for later "show me" tasks); a way to give the AI windows that are visible to it but not in the user's way, because a
transient pointer only reaches windows on an *active* workspace (ideal: an "AI workspace" or output the user doesn't
look at; second best: capture of windows that aren't visible).

**M1, seat session** (`ojp-desktop`, Linux, pure Rust like seat-test): create a transient seat with virtual keyboard
and pointer, `click(x, y)`, `type(text)`, `key(combo)`, `scroll`, and clean shutdown (seats vanish with the process).
Reuse the verified code of `/mnt/a/OJP/seat-test`. Tool: `open_app` already exists.

**M2, seeing** (all apps, text only): the window list from the compositor IPC, then a screenshot of the window
(screencopy) read by a small **OCR** that returns the words with their boxes (about 50-100 MB, no vision model). The
model picks a label ("Save"); the code finds its box and clicks its centre through the seat. That covers every app,
including Electron, games and terminals. Icons without text come later (template matching, or a small vision model).
AT-SPI (the accessibility tree) stays an optional accelerator where an app exposes it: exact names, and button presses
that need no pointer.

**M3, tools for the planner** (six or so, the tool list costs prompt time): `list_windows`, `read_window`,
`click_control`, `type_text`, `press_keys`, `scroll`, plus `run_recipe`. Each result is short text.

**M4, recipes**: a small file format (name, parameters, steps, which steps ask first), a runner that logs every step,
and a first recipe: *dev loop* (open editor and terminal on the AI seat, open a file, run the tests, report the result).

**M5, safety and log**: an activity log of every action (app, control, text), confirmation for anything that sends,
deletes or installs, rate limits, and a quick stop that destroys the seats (the compositor drops them with the client).

**M6, polite background tasks**: the task queue with priority and an explicit "ASAP" flag, on top of `cpu.rs`, so a
recipe running while the user works never slows them down.

## Order and checks

M1, M2 and M3 can be built and tested without the compositor changes except a stable seat; M4 on top; M0's IPC extras
and the AI workspace unlock "show me" tasks and hidden windows later. Test with the `seat-test` harness extended to
drive a real app, and a script that runs the dev-loop recipe and checks its log.

RAM: nothing here loads a second model; AT-SPI reading is small. The laptop keeps the 0.8B model.

## Risks

- OCR misses small, stylised or icon-only controls, and costs a second or so per window read; the tools say plainly when
  a label isn't found rather than clicking blindly.
- The model's reliability: recipes carry the weight, free-form requests will fail more often.
- The transient seat's rules (only visible windows) shape what "in the background" can mean until M0's workspace
  exists.
