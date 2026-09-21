#!/usr/bin/env bash
# Pull model weights into $HF_HUB_CACHE (under $WORK_DIR, NOT home -- ~15 GB each).
#
# Run on a node with outbound network (the login node). The SLURM stage scripts set
# HF_HUB_OFFLINE=1 because compute nodes may not have any, so anything not fetched
# here makes the job fail at load time.
#
#   env/fetch_models.sh                                    # Qwen-7B only (all that has been run)
#   env/fetch_models.sh deepseek-ai/DeepSeek-R1-Distill-Llama-8B   # in scope, never run
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source env.sh

MODELS=("$@")
[ "${#MODELS[@]}" -gt 0 ] || MODELS=(deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)

echo "cache: $HF_HUB_CACHE"
for m in "${MODELS[@]}"; do
  echo "== $m"
  HF_HUB_OFFLINE=0 "$PY" - "$m" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("  ->", snapshot_download(sys.argv[1]))
PY
done
df -h "$HF_HUB_CACHE" | tail -1
du -sh "$HF_HUB_CACHE"
