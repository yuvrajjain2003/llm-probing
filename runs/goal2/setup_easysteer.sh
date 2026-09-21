#!/usr/bin/env bash
# Goal 2 env: EasySteer v0.17.1-Preview  ->  $ENVS_DIR/easysteer  (conda, python 3.10)
#
# Why this tag and not main: main pins vLLM 0.26.0, whose wheels are compiled
# against CUDA 13. The ORIGINAL box's driver (555.42.06) capped at CUDA 12.5, so
# those kernels could not launch -- "libcudart.so.13: cannot open shared object
# file". vLLM only moved to CUDA 13 at 0.22, so the v0.17.1 tag (vLLM 0.17.1,
# torch 2.10.0 -> nvidia-cuda-runtime-cu12==12.8.90) was the newest that ran.
#
# SYCAMORE: that constraint may not apply -- cuda/13.x modules exist -- but the
# driver version is only visible on a GPU node (run runs/gpqa25/00_gpu_check.sh).
# Until that is known this stays on the tag that is known to work. If you move
# tags, re-run all three `verify_capture.py` stages: the layer/position
# conventions are version-specific and silently wrong when they change.
#
# Run this on a node with outbound network (the login node), not inside a job.
# It does not need a GPU; the CUDA check at the end is skipped when none is visible.
#
#   runs/goal2/setup_easysteer.sh            # refuses if the env or source exist
#   FORCE=1 runs/goal2/setup_easysteer.sh    # delete and rebuild both
#   PYTHON310=/path/to/python3.10 ...        # use this interpreter instead of conda
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
source env.sh

VENV="${ES_ENV:-$ENVS_DIR/easysteer}"
SRC="${ES_SRC:-$WORK_DIR/src/easysteer}"
TAG=v0.17.1-Preview
# the vLLM commit the tag's precompiled wheel corresponds to (v0.17.1 release)
export VLLM_PRECOMPILED_WHEEL_COMMIT=95c0f928cdeeaa21c4906e73cee6a156e1b3b995

if [ -e "$VENV" ] || [ -e "$SRC" ]; then
  if [ "${FORCE:-0}" != 1 ]; then
    echo "refusing: $VENV or $SRC already exists (FORCE=1 to delete and rebuild)" >&2
    exit 1
  fi
  echo "FORCE=1: removing $VENV and $SRC"
  rm -rf "$VENV" "$SRC"
fi
mkdir -p "$SRC"

echo "[1/5] fresh env (python 3.10, as the tag requires) -> $VENV"
if [ -n "${PYTHON310:-}" ]; then
  "$PYTHON310" -m venv "$VENV"
else
  set +u; load_conda; set -u
  conda create -y -q -p "$VENV" python=3.10
fi
"$VENV/bin/pip" install -q --upgrade pip wheel setuptools

echo "[2/5] cloning EasySteer @ $TAG with submodules -> $SRC"
git clone --recurse-submodules -q https://github.com/ZJU-REAL/EasySteer.git "$SRC/EasySteer"
git -C "$SRC/EasySteer" checkout -q "$TAG"
git -C "$SRC/EasySteer" submodule update --init --recursive -q

echo "[3/5] installing the vllm-steer fork (precompiled -- no CUDA compilation)"
( cd "$SRC/EasySteer/vllm-steer" && VLLM_USE_PRECOMPILED=1 "$VENV/bin/pip" install --editable . )

echo "[4/5] installing easysteer"
"$VENV/bin/pip" install --editable "$SRC/EasySteer"
"$VENV/bin/pip" install -q gguf math_verify datasets

echo "[5/5] verifying"
"$VENV/bin/python" - <<'PY'
import sys, torch, vllm, inspect
print("python", sys.version.split()[0], "| torch", torch.__version__, "| vllm", vllm.__version__)
if torch.cuda.is_available():
    x = torch.randn(512, 512, device="cuda"); print("CUDA matmul OK:", float((x @ x).sum()) != 0.0)
else:
    print("no GPU visible here -- CUDA check skipped (runs/gpqa25/00_gpu_check.sh does it on a GPU node)")
from easysteer.hidden_states import get_all_hidden_states_generate as f
print("capture API:", inspect.signature(f))
PY
echo "SETUP COMPLETE  ->  PY_ES=$VENV/bin/python"
