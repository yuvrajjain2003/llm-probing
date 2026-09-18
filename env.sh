#!/usr/bin/env bash
# Shared environment for this repo.  Usage:  source env.sh
#
# There is no venv in this repo; the only interpreter on this box with vLLM is
# /playpen-ssd/yuvraj/.venv (vllm 0.5.5, torch 2.4.0+cu121).  Model weights and
# the HF hub cache also live under that user's tree and are READ-ONLY to us, so
# HF_HOME (which HF and outlines both write to) is pointed somewhere writable
# and only HF_HUB_CACHE points at the shared read-only weights.
#
# Datasets are NOT read from the HF cache -- run_eval.load_examples reads the
# local save_to_disk copies in ./data (written by src/setup/download_dataset.py).

export REPO_DIR="/playpen-ssd/shared/llm-probing-project"
export PY="/playpen-ssd/yuvraj/.venv/bin/python"

export HF_HUB_CACHE="/playpen-ssd/yuvraj/.cache/huggingface/hub"   # read-only: 7B + 8B weights
export HF_HOME="${REPO_DIR}/.hf_home"
export OUTLINES_CACHE_DIR="${REPO_DIR}/.outlines_cache"
export HF_HUB_OFFLINE=1

mkdir -p "$HF_HOME" "$OUTLINES_CACHE_DIR"
