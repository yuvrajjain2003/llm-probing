#!/usr/bin/env bash
# Goal 1 env -> $ENVS_DIR/goal1  (conda, python 3.10, env/requirements-goal1.in + freeze constraints)
#
# Used for base CoTs, continuations and ALL CPU analysis/probing. Needs python 3.10
# (the pins are vllm==0.5.5 / torch==2.4.0) but Sycamore's system python is 3.12,
# hence conda. The Goal 2 capture env is separate and cannot be merged with this
# one -- see runs/goal2/setup_easysteer.sh and .claude/SETUP.md.
#
# Run on a node with outbound network (the login node), not inside a job. It needs
# no GPU. torch 2.4.0 ships CUDA 12.1 wheels that bundle their own runtime, so no
# `module load cuda` is needed -- only a driver new enough for CUDA 12.1.
#
#   env/setup_goal1.sh            # refuses if the env already exists
#   FORCE=1 env/setup_goal1.sh    # delete and rebuild
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source env.sh

ENV="${GOAL1_ENV:-$ENVS_DIR/goal1}"
if [ -e "$ENV" ]; then
  if [ "${FORCE:-0}" != 1 ]; then
    echo "refusing: $ENV already exists (FORCE=1 to delete and rebuild)" >&2
    exit 1
  fi
  echo "FORCE=1: removing $ENV"; rm -rf "$ENV"
fi

echo "[1/3] conda env, python 3.10 -> $ENV"
load_conda
conda create -y -q -p "$ENV" python=3.10

echo "[2/3] installing env/requirements-goal1.in (constrained by the original freeze)"
# requirements-goal1.txt is a raw freeze that cannot be installed as-is (orphaned
# packages from a newer vLLM contradict its own pins); see the header of the .in file.
"$ENV/bin/pip" install -q --upgrade pip wheel setuptools
# FIRST, a local stand-in for `pyairports` (outlines 0.0.46 imports it). The only
# project of that name on PyPI is a placeholder sample package with no airports data
# and a stray `byted-wandb` console script, and `import vllm` fails with it installed.
# Installed before the requirements so pip sees the dependency as already satisfied.
# See env/shims/pyairports/pyairports/airports.py.
"$ENV/bin/pip" install -q --no-build-isolation --no-deps ./env/shims/pyairports
"$ENV/bin/pip" install -r env/requirements-goal1.in -c env/requirements-goal1.txt
"$ENV/bin/pip" check || echo "WARNING: pip check reports inconsistencies (listed above)"

echo "[3/3] verifying (imports only; vllm is not imported because that can need a GPU driver)"
"$ENV/bin/python" - <<'PY'
import sys, importlib.metadata as md
print("python", sys.version.split()[0])
for pkg in ("torch", "vllm", "transformers", "scikit-learn", "pandas", "datasets", "math-verify"):
    try: print(f"  {pkg:14s} {md.version(pkg)}")
    except md.PackageNotFoundError: print(f"  {pkg:14s} MISSING")
PY
echo "SETUP COMPLETE  ->  PY=$ENV/bin/python  (env.sh picks it up automatically)"
