#!/usr/bin/env bash
# Design a voice from a text description with Qwen3-TTS VoiceDesign (offline,
# slow: about 3x slower than real time on a CPU), and store the result as a
# "Mine: NAME" reference clip for the MOSS engine.
#   scripts/design-voice.sh NAME "female, young adult, gentle and caring" ["text to say"]
# Needs qwentts.cpp built in $QWENTTS (default /media/ojp-qwentts: repo/build/
# plus models/qwen-talker-1.7b-voicedesign-Q4_K_M.gguf and
# models/qwen-tokenizer-12hz-Q8_0.gguf from Serveurperso/Qwen3-TTS-GGUF) and the
# MOSS environment (scripts/setup-moss.sh). The raw clip is kept in
# <moss dir>/designs/NAME.wav. Set SEED to repeat or vary a result.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1 DESC=$2
TEXT=${3:-"Hello, it's nice to meet you. I'm here to help with whatever you need today. Take your time, and tell me what's on your mind. I'll listen carefully, and we'll figure it out together."}
QWENTTS=${QWENTTS:-/media/ojp-qwentts}
MOSS=${MOSS_DIR:-/media/ojp-moss}
mkdir -p "$MOSS/designs" "$MOSS/references"
"$QWENTTS/repo/build/qwen-tts" \
    --model "$QWENTTS/models/qwen-talker-1.7b-voicedesign-Q4_K_M.gguf" \
    --codec "$QWENTTS/models/qwen-tokenizer-12hz-Q8_0.gguf" \
    --device CPU --lang English --seed "${SEED:--1}" --instruct "$DESC" \
    -o "$MOSS/designs/$NAME.wav" <<< "$TEXT" >/dev/null 2>&1
"$MOSS/venv/bin/python" scripts/make-reference.py "$MOSS/designs/$NAME.wav" --out "$MOSS/references/$NAME.wav"
