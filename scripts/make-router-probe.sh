#!/usr/bin/env bash
# Build the probe file for one router model.
#   scripts/make-router-probe.sh <model.gguf> <template> [out.json]
# Writes <model.gguf>.probe.json next to the model by default. Needs ojp-cli built.
set -euo pipefail
cd "$(dirname "$0")/.."
model=$1; template=$2; out=${3:-$model.probe.json}
tmp=$(mktemp -d)
for set in router-train router-xl; do
    OJP_EMBED=1 OJP_LOW_MEMORY=1 OJP_TEMPLATE=$template OJP_N_CTX=2048 \
        "${OJP_CLI:-target/release/ojp-cli}" route-feats "$model" ojp-core/data/$set.tsv short 2>/dev/null \
        | grep '^FEAT ' | sed 's/^FEAT //' > "$tmp/$set.jsonl"
done
python3 -I scripts/fit-router-probe.py "$tmp/router-train.jsonl" "$tmp/router-xl.jsonl" "$out"
rm -rf "$tmp"
