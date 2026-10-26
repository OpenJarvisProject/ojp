#!/usr/bin/env bash
# Build <router.gguf>.tools.json: which single tool a request needs, from the router's hidden state.
#   scripts/make-tool-probe.sh <router.gguf> <template>
# Run it after make-router-probe.sh, for the same router file. Needs ojp-cli built.
set -euo pipefail
cd "$(dirname "$0")/.."
model=$1; template=$2; out=$model.tools.json
tmp=$(mktemp -d)
for set in train test; do
    OJP_LOW_MEMORY=1 OJP_TEMPLATE=$template OJP_N_CTX=2048 \
        "${OJP_CLI:-target/release/ojp-cli}" tool-feats "$model" ojp-core/data/tool-$set.tsv 2>/dev/null \
        | grep '^FEAT ' | sed 's/^FEAT //' > "$tmp/$set.jsonl"
done
python3 -I scripts/fit-tool-probe.py "$tmp/train.jsonl" "$tmp/test.jsonl" "$out"
rm -rf "$tmp"
