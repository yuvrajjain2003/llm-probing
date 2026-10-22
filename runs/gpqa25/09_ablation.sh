#!/bin/bash
# Probe-family ablation: linear vs RBF kernel vs MLP, on identical features/folds.
# CPU only.  One job per checkpoint, so the five run concurrently:
#
#   cd <repo> && source env.sh
#   for f in 0.1 0.3 0.5 0.7 0.9; do ABL_FRACTION=$f sbatch runs/gpqa25/09_ablation.sh; done
#
# Why one job per checkpoint: the MLP's L2 grid needs several fits per fold, and the
# checkpoints are completely independent, so splitting them is free parallelism.
#SBATCH --job-name=gpqa25-ablation
#SBATCH --partition=small    # the batch QoS demands >=48 CPUs per job; small allows 2-192
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=runs/logs/slurm/slurm-%j.out
set -uo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"
need_py "$PY"
SHARD=0; log_to "ablation_${ABL_FRACTION:-all}"

OUT="$RUNS_ROOT/goal2/$RUN_TAG/ablation"
mkdir -p "$OUT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "checkpoint=${ABL_FRACTION:-all}  layers=${ABL_LAYERS:-1,7,14,21,28}  mlp=h${ABL_MLP_HIDDEN:-256}/i${ABL_MLP_ITERS:-50}/L2[${ABL_MLP_L2:-1e-4,1e-2,1,100}]"
"$PY" src/probing/probe_ablation.py \
  --act-dir "$ACT_DIR" --run-dir "$RUN_DIR" --out-dir "$OUT" \
  --layers "${ABL_LAYERS:-1,7,14,21,28}" --folds 5 \
  --families "${ABL_FAMILIES:-linear,kernel,mlp}" \
  --mlp-hidden "${ABL_MLP_HIDDEN:-256}" --mlp-iters "${ABL_MLP_ITERS:-50}" \
  --mlp-l2 "${ABL_MLP_L2:-1e-4,1e-2,1,100}" --mlp-inner "${ABL_MLP_INNER:-2}" \
  ${ABL_FRACTION:+--fractions "$ABL_FRACTION"}
echo "exit=$?"
