# Vendored, patched Slint 1.18.1 crates

Wired in through `[patch.crates-io]` in the workspace Cargo.toml. Each crate
is an unmodified copy of its crates.io release except for the lines marked
`OJP patch`. Remove these directories and the patch entries once upstream
Slint fixes the issues. License: as upstream (GPL-3.0-only OR
LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0).

## Android: the word being typed showed as selected

Seen with Gboard on a real phone. Two parts:

- `i-slint-backend-android-activity/javahelper.rs`, `callback_update_text`:
  the IME's composing range was forwarded with a preedit selection spanning
  the whole word. It now carries the cursor position inside the word.
- `i-slint-core/textlayout/sharedparley.rs`: the preedit range was painted
  as a selection. Now only the real selection is.

An earlier attempt committed the composing text as plain text instead.
That broke the IME round-trip: holding backspace re-inserted the word
being deleted. This approach keeps Slint's composition handling intact.

## Android: optional dynamic (wallpaper) colours

`i-slint-backend-android-activity/lib.rs` and `javahelper.rs`:
`set_dynamic_color(bool)` chooses whether the system accent colour is
passed to Slint, so the app can offer a "dynamic colours" switch.

## Quick flicks didn't scroll inside a SwipeGestureHandler

`i-slint-core/items/input_items.rs`, `SwipeGestureHandler`: on press it
used `DelayForwarding`, and the held-back press is only replayed when the
delay timer fires or on release. A fast vertical flick on the chat's
ScrollView, which sits inside the swipe-to-open-sidebar handler, reached
the ScrollView without its press and didn't scroll; slow drags worked.
The press is now forwarded at once (`ForwardAndInterceptGrab`): children
get the whole gesture, and the handler still intercepts once a horizontal
swipe passes its threshold.
