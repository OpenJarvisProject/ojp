#!/bin/sh
# Upload the unc-0.8b bundle to Hugging Face. Needs a write token:
#   hf auth login   (once)  or  HF_TOKEN=hf_xxx;  then  scripts/upload-hf.sh [org/repo] [bundle-dir]
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
for f in ("unc-0.8b-q5km.gguf.probe.json", "unc-0.8b-q5km.gguf.tools.json", "SHA256SUMS", "unc-0.8b-q5km.gguf"):
    print("uploading", f, flush=True)
    api.upload_file(path_or_fileobj=f"{d}/{f}", path_in_repo=f, repo_id=repo)
print("done: https://huggingface.co/" + repo)
PY
