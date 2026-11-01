#!/bin/sh
# Upload a model bundle (unc-0.8b by default, or e.g. unc-2b) to Hugging Face. Needs a write token:
#   hf auth login   (once)  or  HF_TOKEN=hf_xxx;  then  scripts/upload-hf.sh [org/repo] [bundle-dir]
#   scripts/upload-hf.sh OpenJarvisProject/Qwen3.5-2B-Uncensored-OJP-GGUF /mnt/a/OJP/ojp-models/unc-2b
set -e
REPO=${1:-OpenJarvisProject/Qwen3.5-0.8B-Uncensored-OJP-GGUF}
DIR=${2:-/media/ojp-models/unc-0.8b}
export HF_HOME=${HF_HOME:-/media/hf-home}
VENV=${HF_VENV:-/media/hf-venv}
[ -x "$VENV/bin/python" ] || { python3 -m venv "$VENV"; "$VENV/bin/pip" install -q huggingface_hub; }
"$VENV/bin/python" - "$REPO" "$DIR" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, d = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj=f"{d}/MODELCARD.md", path_in_repo="README.md", repo_id=repo)
names = sorted(os.listdir(d))
ggufs = [f for f in names if f.endswith(".gguf")]
small = [f for f in names if f.endswith((".probe.json", ".tools.json")) or f == "SHA256SUMS"]
for f in small + ggufs:
    print("uploading", f, flush=True)
    api.upload_file(path_or_fileobj=f"{d}/{f}", path_in_repo=f, repo_id=repo)
print("done: https://huggingface.co/" + repo)
PY
