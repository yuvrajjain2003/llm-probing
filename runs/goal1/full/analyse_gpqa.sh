#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source env.sh
R=runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B
echo "[merge]";      "$PY" src/generation/merge_shards.py --run-dir "$R"
echo "[labels]";     "$PY" src/analysis/labels.py --continuations "$R/raw/continuations.jsonl"
echo "[robustness]"; "$PY" src/analysis/robustness.py --run-dir "$R"
echo "[ksuff]";      "$PY" src/analysis/k_sufficiency.py --run-dir "$R" --draws 200
echo "[DONE]"
