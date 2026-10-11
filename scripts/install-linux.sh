#!/usr/bin/env bash
# Install OpenJarvis for the current user (no root): the `openjarvis`
# command in ~/.local/bin and an "OpenJarvis" entry in the app menu.
#   scripts/install-linux.sh [path/to/openjarvis]
# Defaults to this checkout's release build.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${1:-target/release/openjarvis}
[[ -x "$BIN" ]] || { echo "no binary at $BIN (cargo build --release first)"; exit 1; }

install -Dm755 "$BIN" "$HOME/.local/bin/openjarvis"
install -Dm644 /dev/stdin "$HOME/.local/share/applications/openjarvis.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=OpenJarvis
Comment=Local AI assistant
Exec=$HOME/.local/bin/openjarvis
Terminal=false
Categories=Utility;
DESKTOP
echo "Installed: run 'openjarvis' or pick OpenJarvis from your app menu."
