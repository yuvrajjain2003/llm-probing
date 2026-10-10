#!/usr/bin/env bash
# ICL env -> $ENVS_DIR/icl (conda, python 3.12, current vLLM: Qwen3.5 + Gemma 4 need it).
# Login node, no GPU. vLLM 0.31.0 wheel = CUDA 13.0 build (driver on h100_all nodes: 595.71, CUDA 13.2).
#   env/setup_icl.sh            # refuses if the env exists;  FORCE=1 rebuilds
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
ENV="${ICL_ENV:-$ENVS_DIR/icl}"
if [ -e "$ENV" ]; then
  [ "${FORCE:-0}" = 1 ] || { echo "refusing: $ENV exists (FORCE=1 to rebuild)" >&2; exit 1; }
  rm -rf "$ENV"
fi
load_conda
conda create -y -q -p "$ENV" python=3.12
"$ENV/bin/pip" install -q --upgrade pip wheel
"$ENV/bin/pip" install "vllm==0.31.0" pandas numpy scikit-learn pyyaml
"$ENV/bin/pip" check || echo "WARNING: pip check reports inconsistencies"
