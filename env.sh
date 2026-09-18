#!/usr/bin/env bash
# Shared environment for this repo.  Usage:  source env.sh
#
# Every value below is overridable from the caller's environment, so a clone on
# a new machine needs no edits to this file -- export what differs and source it:
#
#     export PY=/path/to/goal1-venv/bin/python
#     export HF_HUB_CACHE=/scratch/$USER/hf/hub
#     source env.sh
#
# Two SEPARATE interpreters are in play; see .claude/SETUP.md:
#   $PY        Goal 1 generation + all CPU analysis (vLLM 0.5.5, torch 2.4.0)
#   $PY_ES     Goal 2 activation capture only (EasySteer fork, vLLM 0.17.1)
# They cannot be merged -- the two vLLM versions are mutually incompatible.
#
# Datasets are NOT read from the HF cache: run_eval.load_examples reads the
# local save_to_disk copies in ./data (written by src/setup/download_dataset.py).

# Repo root, derived from this file's location so a clone anywhere works.
_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="${REPO_DIR:-$_ENV_SH_DIR}"

# Goal 1 / analysis interpreter.  Prefers an in-repo venv, then the original
# UNC-box path, then whatever `python3` is on PATH.
if [ -z "${PY:-}" ]; then
  if   [ -x "${REPO_DIR}/.venv-goal1/bin/python" ]; then PY="${REPO_DIR}/.venv-goal1/bin/python"
  elif [ -x "/playpen-ssd/yuvraj/.venv/bin/python" ]; then PY="/playpen-ssd/yuvraj/.venv/bin/python"
  else PY="$(command -v python3)"
  fi
fi
export PY

# Goal 2 activation-capture interpreter, built by runs/goal2/setup_easysteer.sh.
export PY_ES="${PY_ES:-${REPO_DIR}/.venv-easysteer/bin/python}"

# HF_HOME and the outlines cache must be WRITABLE; HF_HUB_CACHE may point at a
# read-only shared weights tree (that was the arrangement on the original box).
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_DIR}/.hf_home/hub}"
export HF_HOME="${HF_HOME:-${REPO_DIR}/.hf_home}"
# Per-USER by default.  outlines' cache is a sqlite DB that the importing
# process must be able to WRITE; on a shared repo a cache dir created by another
# user makes `import vllm` die with "attempt to write a readonly database".
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-${REPO_DIR}/.outlines_cache_${USER:-$(id -un)}}"

# 1 on the original box because the weights were already cached locally.
# Leave 0 on a fresh cluster until the models have been pulled once.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

mkdir -p "$HF_HOME" "$OUTLINES_CACHE_DIR" "$HF_HUB_CACHE"
