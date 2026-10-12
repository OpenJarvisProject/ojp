# Vendored i-slint-backend-android-activity 1.18.1 (patched)

Unmodified copy of the crates.io release except `javahelper.rs`,
`callback_update_text`: the IME composing range is committed as plain text
instead of being forwarded as a preedit whose selection spans the whole
word. Upstream renders that as a highlighted selection while typing (seen
with Gboard on a real phone). Remove this directory and the
`[patch.crates-io]` entry once upstream Slint fixes it.

License: as upstream (GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR
LicenseRef-Slint-Software-3.0); see the crate's own headers.
