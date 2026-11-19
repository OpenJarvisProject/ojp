# Desktop control (Linux, fht-compositor)

The desktop app can see and use the windows of a running fht-compositor on the assistant's **own transient seat**: its
virtual keyboard and mouse. Your own mouse and keyboard are never used or moved. Needs a compositor with
`ext-transient-seat-v1`, `zwp_virtual_keyboard_v1` and `zwlr_virtual_pointer_v1` (the fork's `feat/touch-transient`)
and its IPC socket (found through `FHTC_SOCKET_PATH`, else `$XDG_RUNTIME_DIR/fhtc-*.socket`). Code: `ojp-core/src/desktop/`.

## Tools (desktop only)

| tool | what it does | asks first |
|---|---|---|
| `list_windows` | number, app and title of every window | no |
| `click_window(window, x, y)` | click at x,y pixels from the window's top-left corner | yes |
| `type_text(text)` | type into the window the assistant clicked last (US keyboard) | yes |
| `press_keys(keys)` | a shortcut such as `ctrl+s`, `Return`, `alt+tab` | yes |
| `scroll_window(window, direction)` | scroll a window under the assistant's pointer | no |
| `open_app(name)` | start an installed app | no |

A window is named by its number from `list_windows`, or by (part of) its title or app id. If a name matches several
windows the tool says so and asks for the number. Typed shortcuts exist too: `/windows`, `/click`, `/typetext`,
`/keys`, `/scroll`.

## Try it without a model

    ojp-cli desk            # commands on standard input, one per line, all in one seat session
    windows
    click 3 120 80
    type hello world
    key ctrl+s
    scroll 3 5

Verified against a headless fht-compositor with the `seat-test` windows: a click at (200, 150) in a window arrived at
(199.99, 149.99) in that window, typing and shortcuts arrived on the transient seat only.

## Limits

- The transient pointer only reaches windows on a workspace that is on screen.
- Seeing is the window list for now; reading what a window shows (OCR with boxes) is next.
- Characters without a US key (accents, emoji) are skipped.
- Window positions come from the compositor's IPC and assume the window's location is measured from its output's corner.
