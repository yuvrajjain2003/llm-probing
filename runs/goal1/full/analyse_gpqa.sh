#!/usr/bin/env bash
set -uo pipefail
cd /playpen-ssd/shared/llm-probing-project
source env.sh
export OUTLINES_CACHE_DIR="${REPO_DIR}/.outlines_cache_yuvraj"
R=runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B
echo "[merge]";      "$PY" src/generation/merge_shards.py --run-dir "$R"
echo "[labels]";     "$PY" src/analysis/labels.py --continuations "$R/raw/continuations.jsonl"
echo "[robustness]"; "$PY" src/analysis/robustness.py --run-dir "$R"
echo "[ksuff]";      "$PY" src/analysis/k_sufficiency.py --run-dir "$R" --draws 200
echo "[DONE]"
