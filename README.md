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
`answer_directly`, `coding`, `reasoning`, or `device_control`. With a **probe**
file next to the router (`router.gguf.probe.json`, made by
`scripts/make-router-probe.sh`) the router reads the model's internal state at
the answer position and a tiny fitted classifier picks the label: about 98% on
held-out requests, against 76-83% when the small model answers for itself, and
faster because no worked examples or generated label are needed. Without a
probe, a grammar restricts the model's own answer to exactly one of those labels. Device control becomes a
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
- **Several devices:** pair them by hand and they share a model, memories and requests ("remind me on my phone ..."): see [docs/devices.md](docs/devices.md).

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

## Memory

It remembers what you tell it ("remember that I like green tea") in
`memory.jsonl` next to the chats (owner-only permissions). Each memory is one
JSON line: `{id, created, text, summary, tags}`. Tags come from a small
canonical vocabulary, and near-synonyms are folded together (baking and
cooking both become `cooking`) so tags don't multiply and old memories stay
findable. Long memories get a one-line summary; only the summary is shown to
the model unless the full text is needed, which saves tokens. Recall is BM25
text search plus a tag boost. With 15 memories or fewer all are given to the
model; beyond that it gets the best 6 and the newest 3. The memories go into
the system prompt for that answer only: small models ignore them in the user
message. Memories written after reading untrusted text (web pages, files)
ask for confirmation first.

## Model families

Models are plain GGUF files, so any family llama.cpp supports works. Set
`chat_template` to match: `chatml` (Qwen2.5), `chatml-nothink` (Qwen3.5,
which thinks by default; the template pre-closes the think block), `llama3`,
`mistral`, `gemma`, `phi3`. Qwen3.5 is a hybrid recurrent model whose memory
can't be rewound, so the engine keeps a snapshot of the state where
consecutive prompts diverge and restores it instead of trimming
(`ojp-core/src/model.rs`); this is automatic. The bundled llama.cpp is
vendored under `vendor/llama-crab-sys` (see `vendor/OJP-PATCHES.md`).

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

"Verified" means I ran it and looked at the result; "built" means it compiles and has unit tests but nobody has run it on the
real target yet. Last checked at v0.0.6.

**Working and verified (Linux desktop)**

| Feature | How it was checked |
|---|---|
| Typed chat with a router, a main model, and fallbacks between them | daily use; `ojp-cli chat` |
| Tools: timers, reminders, volume, media, open apps and links, files (read/find/search), notes, calculator, web, memory | agent test, 46/48 scenarios with the Qwen3.5 0.8B and 32/36 with the Qwen2.5 pair; the 36 original scenarios were used for tuning, so treat those numbers as optimistic |
| Router and tool probes (read the answer from the model's hidden state: ~98% routing, 93% tool choice on held-out requests) | `ojp-cli route-eval`, `tool-eval`; model-tooling `RESULTS.md` |
| Long-term memory with tags and summaries; typed `/commands` | unit tests, daily use |
| In-app model download (resume, SHA-256 check), including the uncensored 0.8B bundle on Hugging Face | download and checksum verified against the public files |
| Uncensored Qwen3.5-0.8B as the single model (Settings > AI model) | agent test, refusal test (0/18 refused), 70-request soak test |
| Several devices as one assistant: pairing by code, encrypted link, "on my phone ..." requests, approvals on the asking device, shared memories | two real processes on one PC (pairing, remote chat, delegated reminders, refusals, memory and forget sync, VPN-only address) |
| Text-to-speech (Kokoro voices, mixer, speed); typeset math | works (desktop only, off by default) |

**Built, not yet verified on a device**

| Feature | Notes |
|---|---|
| Android app, Digital assistant role | worked on an emulator and phone before v0.0.5 |
| Android: phone tools (volume, media, apps, links, reminders as alarms, battery) | compiles; never run on a phone |
| Android: optional Shizuku (Wi-Fi, Bluetooth, airplane mode, brightness, screenshots, adb commands that ask first) | compiles; never run on a phone |
| Settings > Devices on the phone, pairing a phone with a computer | the pairing and link code is tested between two desktop processes only |
| Settings > Model download button for the uncensored model | the logic ran headlessly through the real app code and the screens were rendered, but nothing was tapped on a phone |

**Known gaps**

- Android has no file tools, clipboard, speech, or background service; a phone only serves other devices while the app is open.
- The tool probe only knows the desktop tool names; the new phone tools are chosen by the slower planner.
- Speech-to-text, wake word ("hey zelia"), and text-to-speech on Android: planned.
- Renaming the app to the assistant's name (e.g. Zelia): planned.
- Not started: desktop control of other applications, a homelab agent (`docs/homelab.md`).

Developer tools: `cargo run -p ojp --no-default-features --features uitest --bin ui-test` drives the real app logic
headlessly; `--features shots --bin ui-shot -- <dir>` renders the pages to PNG at phone size.

The old `ojp-cli route-eval` / benchmark harness is still in `ojp-core`.

## Router probe

`scripts/gen-router-data.py` writes `ojp-core/data/router-train.tsv` (hand-written
seeds plus templated variety) and the held-out `router-xl.tsv`;
`scripts/make-router-probe.sh <router.gguf> <template>` extracts the model's hidden
state for each request (`ojp-cli route-feats`), fits a linear classifier
(`scripts/fit-router-probe.py`, L2 chosen by 5-fold cross-validation), reports
held-out accuracy and writes `<router.gguf>.probe.json`. The probe belongs to that
exact file: refit it after any change to the model or its quantization. Router
quantization barely matters for the probe (Q8_0 down to Q2_K all score 97-100%
held-out), so the router file can be chosen for size.
