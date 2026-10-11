#!/usr/bin/env bash
# Cross-compile the desktop app and CLI for aarch64 Linux (e.g. Debian on
# an ARM laptop) from an x86_64 machine, so the target never has to
# compile llama.cpp itself. CPU-only inference.
#
# Targets glibc 2.36 (Debian 12+) via cargo-zigbuild:
#   cargo install cargo-zigbuild && pip install ziglang   (or install zig)
# Output: target/aarch64-unknown-linux-gnu/release/{ojp,ojp-cli}
set -euo pipefail
cd "$(dirname "$0")/.."
cargo zigbuild --release -p ojp -p ojp-core --no-default-features \
    --target aarch64-unknown-linux-gnu.2.36
ls -lh target/aarch64-unknown-linux-gnu/release/{ojp,ojp-cli}
