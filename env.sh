#!/usr/bin/env bash
# Shared environment for this repo.  Usage:  source env.sh
#
# Every value below is overridable from the caller's environment, so a clone on
# a new machine needs no edits to this file -- export what differs and source it:
#
#     export PY=/path/to/goal1-venv/bin/python
#     export WORK_DIR=/scratch/$USER
#     source env.sh
#
# Two SEPARATE interpreters are in play; see .claude/SETUP.md:
#   $PY        Goal 1 generation + all CPU analysis (vLLM 0.5.5, torch 2.4.0)
#   $PY_ES     Goal 2 activation capture only (EasySteer fork, vLLM 0.17.1)
# They cannot be merged -- the two vLLM versions are mutually incompatible.
#
# STORAGE.  Bulk data does not belong in the repo directory: on Sycamore $HOME
# has a ~71 GB / 1M-file quota, and the two model checkpoints alone are ~28 GB.
# Everything large is rooted at $WORK_DIR (the user's /work directory there):
#
#   $ENVS_DIR    python environments        $WORK_DIR/envs
#   $CACHE_ROOT  HF weights + vLLM/Triton   $WORK_DIR/cache
#   $RUNS_ROOT   run outputs                $WORK_DIR/llm-probing/runs
#
# Code, .claude/ and small results stay in the repo. Datasets stay in ./data
# (~89 MB): run_eval.load_examples reads local save_to_disk copies there, NOT the
# HF cache (written by src/setup/download_dataset.py).

# Repo root, derived from this file's location so a clone anywhere works.
_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="${REPO_DIR:-$_ENV_SH_DIR}"

# Bulk-storage root. Sycamore's per-user directory is /work/users/<u>/<s>/<user>
# (first and second letter of the username). Elsewhere fall back to a gitignored
# directory inside the repo so the same scripts still run.
if [ -z "${WORK_DIR:-}" ]; then
  _u="${USER:-$(id -un)}"
  _w="/work/users/${_u:0:1}/${_u:1:1}/${_u}"
  if [ -d "$_w" ]; then WORK_DIR="$_w"; else WORK_DIR="${REPO_DIR}/.work"; fi
fi
export WORK_DIR
export ENVS_DIR="${ENVS_DIR:-${WORK_DIR}/envs}"
export CACHE_ROOT="${CACHE_ROOT:-${WORK_DIR}/cache}"
export RUNS_ROOT="${RUNS_ROOT:-${WORK_DIR}/llm-probing/runs}"

# Goal 1 / analysis interpreter. Order: exported > $ENVS_DIR/goal1 (built by
# env/setup_goal1.sh) > in-repo .venv-goal1 > the original UNC box's venv > python3.
if [ -z "${PY:-}" ]; then
  if   [ -x "${ENVS_DIR}/goal1/bin/python" ];        then PY="${ENVS_DIR}/goal1/bin/python"
  elif [ -x "${REPO_DIR}/.venv-goal1/bin/python" ];  then PY="${REPO_DIR}/.venv-goal1/bin/python"
  elif [ -x "/playpen-ssd/yuvraj/.venv/bin/python" ]; then PY="/playpen-ssd/yuvraj/.venv/bin/python"
  else PY="$(command -v python3)"
  fi
fi
export PY

# Goal 2 activation-capture interpreter, built by runs/goal2/setup_easysteer.sh.
if [ -z "${PY_ES:-}" ]; then
  if [ -x "${ENVS_DIR}/easysteer/bin/python" ]; then PY_ES="${ENVS_DIR}/easysteer/bin/python"
  else PY_ES="${REPO_DIR}/.venv-easysteer/bin/python"
  fi
fi
export PY_ES

# Every cache below must be WRITABLE by you. run_eval.py used to default these to
# /playpen-ssd/yuvraj/... at import time (setdefault, so exporting them here wins);
# on any other machine that path does not exist.
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${CACHE_ROOT}/config}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
# Per-USER by default.  outlines' cache is a sqlite DB that the importing
# process must be able to WRITE; a cache dir left by another user makes
# `import vllm` die with "attempt to write a readonly database". The SLURM stage
# scripts (runs/gpqa25/common.sh) override this with a node-local per-task dir,
# because many array tasks sharing one sqlite file on NFS is a locking hazard.
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-${CACHE_ROOT}/outlines_${USER:-$(id -un)}}"

# 0 by default so the login node can fetch weights. The SLURM stage scripts set 1
# (compute nodes may have no outbound network; pull weights first with
# env/fetch_models.sh).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

mkdir -p "$ENVS_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CONFIG_HOME" "$OUTLINES_CACHE_DIR" \
         "$RUNS_ROOT" 2>/dev/null || true

# Sycamore ships Python only as modules (system python is 3.12; both envs here
# need 3.10), so environments are built with conda. Call this, do not source it
# at top level -- it changes PATH.
load_conda() {
  type module >/dev/null 2>&1 || source /usr/share/lmod/lmod/init/bash
  module load "${CONDA_MODULE:-anaconda/2025.12-2}"
}
