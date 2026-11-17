# Next version: desktop hands (plan)

Decisions (user, 2026-11-17): the brain stays **local only** (nothing is exposed to outside agents); the AI **sees the
desktop as text** (window list + accessibility tree), no vision model yet; the first demo is a **dev loop in the
background**. Target: the user's fht-compositor (branch `feat/touch-transient`, transient seats), on an aarch64 laptop
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

**M2, seeing**: window list from the compositor IPC; the accessibility tree of a window over AT-SPI (role, name, state,
bounds, children), found by role and name; the tree as short text for the model. AT-SPI's own actions (press a button,
set a text field) are used first because they need no pointer at all; clicks by coordinates are the fallback.
Known gap: apps that don't expose AT-SPI (many Electron apps, games, terminals like foot) give little; for the dev loop
the shell tools cover the terminal anyway.

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

- AT-SPI coverage differs per app, so the tools say plainly when a window exposes nothing.
- The model's reliability: recipes carry the weight, free-form requests will fail more often.
- The transient seat's rules (only visible windows) shape what "in the background" can mean until M0's workspace
  exists.
