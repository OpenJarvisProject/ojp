#!/usr/bin/env bash
# Turn a recording of your voice into a "Mine: NAME" voice for MOSS, with the
# timbre of a Kokoro voice or blend. Uses Seed-VC (GPL-3.0, zero-shot voice
# conversion) as an offline tool; nothing from it ships with the app.
#   scripts/make-voice-variant.sh RECORDING NAME "af_heart.6+af_bella.4"
# Needs: the MOSS environment (scripts/setup-moss.sh), ffmpeg, a built
# `cargo build --release -p ojp-core --features tts --example tts_wav`, and
# Seed-VC in $SEEDVC (default /media/ojp-seedvc: repo/ + a Python 3.10 venv/
# with its requirements; see docs/voice-conversion.md). Takes a few minutes
# on the CPU. The result goes in <moss dir>/references/NAME.wav and shows up
# in Settings after a restart.
set -euo pipefail
cd "$(dirname "$0")/.."
REC=$1 NAME=$2 BLEND=$3
SEEDVC=${SEEDVC:-/media/ojp-seedvc}
MOSS=${MOSS_DIR:-/media/ojp-moss}
SECONDS_OF_SPEECH=${SECONDS_OF_SPEECH:-14}
DATA=${XDG_DATA_HOME:-$HOME/.local/share}/ojp
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# 1. A short clip of the target timbre, spoken by Kokoro.
./target/release/examples/tts_wav "$DATA/models/tts/kokoro.onnx" "$DATA/models/tts/voices.bin" "$WORK/target.wav" \
    "This is a calm, clear sample of a woman speaking in a natural voice. She takes her time, and she speaks gently, with a warm tone. The sentence is long enough to be a good reference for the conversion." "$BLEND" >/dev/null
# 2. Your speech, mono, silence trimmed, a few seconds of it.
ffmpeg -v error -y -i "$REC" -ac 1 -ar 22050 \
    -af "silenceremove=start_periods=1:start_threshold=-45dB,atrim=0:$SECONDS_OF_SPEECH" "$WORK/source.wav"
# 3. Convert (no f0 conditioning: that is for singing).
(cd "$SEEDVC/repo" && HF_HOME="$SEEDVC/hf" ../venv/bin/python inference.py --source "$WORK/source.wav" \
    --target "$WORK/target.wav" --output "$WORK/out" --diffusion-steps 25 --fp16 False --f0-condition False >/dev/null 2>&1)
# 4. Level it and store it as a reference clip.
mkdir -p "$MOSS/references"
"$MOSS/venv/bin/python" scripts/make-reference.py "$WORK"/out/*.wav --out "$MOSS/references/$NAME.wav"
