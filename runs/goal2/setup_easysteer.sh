#!/usr/bin/env bash
# Goal 2 env: EasySteer v0.17.1-Preview.
#
# Why this tag and not main: main pins vLLM 0.26.0, whose wheels are compiled
# against CUDA 13. This box's driver (555.42.06) caps at CUDA 12.5, so those
# kernels cannot launch -- confirmed by "libcudart.so.13: cannot open shared
# object file". vLLM only moved to CUDA 13 at 0.22, so the v0.17.1 tag (vLLM
# 0.17.1, torch 2.10.0 -> nvidia-cuda-runtime-cu12==12.8.90) runs here.
set -euo pipefail
cd /playpen-ssd/shared/llm-probing-project
VENV=.venv-easysteer
SRC=runs/goal2/_src
TAG=v0.17.1-Preview
# the vLLM commit the tag's precompiled wheel corresponds to (v0.17.1 release)
export VLLM_PRECOMPILED_WHEEL_COMMIT=95c0f928cdeeaa21c4906e73cee6a156e1b3b995

echo "[1/5] fresh venv (python3.10, as the tag specifies)"
rm -rf "$VENV" "$SRC"; mkdir -p "$SRC"
/usr/bin/python3.10 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip wheel setuptools

echo "[2/5] cloning EasySteer @ $TAG with submodules"
git clone --recurse-submodules -q https://github.com/ZJU-REAL/EasySteer.git "$SRC/EasySteer"
git -C "$SRC/EasySteer" checkout -q "$TAG"
git -C "$SRC/EasySteer" submodule update --init --recursive -q

echo "[3/5] installing the vllm-steer fork (precompiled -- no CUDA compilation)"
( cd "$SRC/EasySteer/vllm-steer" && VLLM_USE_PRECOMPILED=1 "$OLDPWD/$VENV/bin/pip" install --editable . )

echo "[4/5] installing easysteer"
"$VENV/bin/pip" install --editable "$SRC/EasySteer"
"$VENV/bin/pip" install -q gguf math_verify datasets

echo "[5/5] verifying"
"$VENV/bin/python" - <<'PY'
import sys, torch, vllm, inspect
print("python", sys.version.split()[0], "| torch", torch.__version__, "| vllm", vllm.__version__)
x = torch.randn(512,512, device="cuda"); print("CUDA matmul OK:", float((x@x).sum()) != 0.0)
from easysteer.hidden_states import get_all_hidden_states_generate as f
print("capture API:", inspect.signature(f))
PY
echo "SETUP COMPLETE"
