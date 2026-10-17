#!/usr/bin/env bash
# Set up the experimental MOSS-TTS-Nano voice (Apache-2.0, OpenMOSS) in DIR:
#   scripts/setup-moss.sh [DIR]      # default DIR: ~/.local/share/ojp/moss
# Creates a Python venv with onnxruntime and CPU PyTorch (the reference code
# imports it), clones the pinned MOSS-TTS-Nano revision, installs the sidecar
# script, and downloads the ONNX models (~760 MB). About 1.5 GB in total.
# TEMPORARY: a native Rust port would remove all of this.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=${1:-${XDG_DATA_HOME:-$HOME/.local/share}/ojp/moss}
REV=8b7bcc9341b3b4ef3a3a58ba1338a7d85ff133eb
mkdir -p "$DIR"
[[ -d "$DIR/venv" ]] || python3 -m venv "$DIR/venv"
"$DIR/venv/bin/pip" -q install onnxruntime numpy sentencepiece soundfile huggingface_hub praat-parselmouth
"$DIR/venv/bin/pip" -q install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
if [[ ! -d "$DIR/repo" ]]; then
    git clone -q https://github.com/OpenMOSS/MOSS-TTS-Nano "$DIR/repo"
fi
git -C "$DIR/repo" checkout -q "$REV"
install -m755 scripts/moss_server.py "$DIR/moss_server.py"
# Loading the runtime downloads the model files on first use.
"$DIR/venv/bin/python" "$DIR/moss_server.py" "$DIR/repo" --check
echo "MOSS ready in $DIR"
