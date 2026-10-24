# ojp

The OpenJarvisProject assistant: a local, offline-first "Jarvis" for Linux
and Android. You can type to it, and on the desktop it can speak its answers. Every device runs its
own models, and nothing leaves the device unless you configure a remote
model server yourself.

| Crate | What |
|---|---|
| `ojp-core` | Engine shared by everything: config, model loading (llama.cpp via `llama-crab`), router, tools. Also `ojp-cli`, a terminal front end and test harness. |
| `ojp` | Linux desktop app (Slint). |
| `ojp-android` | Android app: the same UI and engine, plus Java services that make it selectable as the **Digital assistant app** (`ojp-android/android/`). |
| `ui/` | Shared Slint UI (`app.slint`) and its Rust glue (`glue.rs`). |

History: this repo grew out of the `core` repo, whose commits it keeps.

## How it works

Each message is classified by a small **router** model into
`answer_directly`, `coding`, `reasoning`, or `device_control`. A grammar
restricts it to exactly one of those labels. Device control becomes a
grammar-constrained tool call (volume, timers, launching apps). Everything
else goes to the **main** model, which is either an in-process GGUF or any
OpenAI-compatible server (Ollama, llama-server, another of your devices).

Each piece is optional and swappable, and failures degrade instead of
breaking:

- **No router:** the main model routes.
- **No main model, or it fails:** the router model answers.
- **Routing fails:** the message is answered directly.

The persona (name, wake word, pronunciation, TTS voice) is plain config.

## Setup

The installed app is called **OpenJarvis**. On first run it offers a
**Download** button that fetches its models: a 0.5B router and a 1.5B
main model, about 1.5 GB: OJP's own optimized router build and the official
Qwen 1.5B GGUF. Each file
is verified against a pinned SHA-256, and interrupted downloads resume.
After that, everything runs offline.

- **Linux:** `scripts/install-linux.sh` installs the `openjarvis` command
  and an "OpenJarvis" app-menu entry, both user-local.
- **Android:** `scripts/build-android.sh` builds `ojp.apk`.

Config is created on first run with comments explaining every option,
including which models to download:

- **Linux:** `~/.config/ojp/config.toml`, models in `~/.local/share/ojp/models/`
- **Android:** `Android/data/org.openjarvisproject.ojp/files/` (`config.toml`, `models/`)

Models the Download button fetches, Qwen2.5-Instruct GGUF (swap them in
`[download]`):

| Role | File | Size |
|---|---|---|
| `models/router.gguf` | 0.5B Q4_K_M + imatrix + q5_0 embedding ([OpenJarvisProject/Qwen2.5-0.5B-Instruct-OJP-GGUF](https://huggingface.co/OpenJarvisProject/Qwen2.5-0.5B-Instruct-OJP-GGUF)) | 347 MB |
| `models/main.gguf` | 1.5B Q4_K_M (3B is in the config comments, for GPU desktops) | 1.1 GB |

Measurements behind these picks are in
[model-tooling/RESULTS.md](https://github.com/OpenJarvisProject/model-tooling/blob/main/RESULTS.md).
On its own, the 0.5B is too weak for facts; it is meant only as the router.

Example persona:

```toml
[persona]
name = "Zelia"
wake_word = "hey zelia"
pronunciation = "zelya"
voice = "female"
```

## What it can do (typed)

Type in plain words, or use commands (`/help` lists them). It sets timers and
reminders that survive closing the app, controls volume and music, opens apps
and sites, keeps notes, does exact arithmetic, looks up the time, battery,
disk, clipboard and files, reads and searches your files, and can do several
steps in a row. Anything that changes something (writing a file, running a
terminal command) is described first and waits for your "yes". Reading is
limited to your home folder and never touches keys, passwords or browser data.
Web search and page reading are off until you set `internet = true` under
`[tools]`; addresses on your own machine or network are always refused.

## Speech (desktop)

Settings has **Speak replies**, off by default. Turning it on downloads the
Kokoro-82M voice (about 340 MB, SHA-256 checked), then lets you blend up to
three voices, set the speed, and play a sample. It runs on the CPU. Speech
engines are swappable (`docs/tts-engines.md`); the experimental
MOSS-TTS-Nano engine needs `scripts/setup-moss.sh` (about 1.5 GB, Python).
To speak in a voice of your own: record 10-20 seconds of clean speech, run
`scripts/make-reference.py recording.wav` (add `--feminize` to shift a male
voice with Praat), and pick "MOSS voice: My own reference clip". Use only
voices you have the right to use.

## Building

```bash
cargo build --release                 # desktop (x86_64, Vulkan GPU offload)
cargo build --release -p ojp --no-default-features --features tts   # CPU-only (drop `--features tts` for no speech)
./target/release/openjarvis           # the app (scripts/install-linux.sh installs it)
./target/release/ojp-cli chat         # same engine in the terminal
```

Vulkan build gotchas (Manjaro): llama.cpp's Vulkan backend needs
`spirv/unified1/spirv.hpp`. If your distro doesn't ship SPIRV-Headers,
build them to a local prefix and set
`CMAKE_PREFIX_PATH=$PWD/.deps CPLUS_INCLUDE_PATH=$PWD/.deps/include`.
With both AMDVLK and Mesa installed, select RADV:
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json`.

aarch64 Linux and the Android APK: see `scripts/` and `ojp-android/android/`.

## Status

| Feature | State |
|---|---|
| Typed chat, routing, device tools, fallbacks | done |
| In-app model download (resume, SHA-256 check) | done |
| Prompt caching (router examples, chat history) | done |
| Android app, Digital assistant role | done (emulator-verified) |
| Rename the app to the assistant's name (e.g. Zelia) | planned |
| Text-to-speech, desktop: Kokoro voices with a three-voice mixer, speed, sample button (off by default) | done |
| Typeset display math ($$ ... $$) | done |
| Experimental MOSS voice and your own reference clip (`docs/tts-engines.md`) | done, needs `scripts/setup-moss.sh` |
| Text-to-speech on Android; speech-to-text; wake word ("hey zelia") | planned next |

The old `ojp-cli route-eval` / benchmark harness is still in `ojp-core`.
