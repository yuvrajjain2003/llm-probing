#!/bin/bash
# Goal 2: probe the checkpoint activations against the Goal 1 labels.  CPU only.
#
#   cd <repo> && source env.sh
#   PROBE_LAYERS=14,21,28 PROBE_EXTRA=--no-combo sbatch runs/gpqa25/08_probe.sh   # fast
#   sbatch runs/gpqa25/08_probe.sh                                                # all 28 layers
#
# Cost scales badly with this dataset, and it is worth knowing why before waiting:
# RidgePath takes ONE economy SVD per (layer, fold) and reads every alpha off it, which
# is O(n_train^2 * d).  The old design had ~575 rows per checkpoint (3 CoTs/question);
# this one has ~4,900 (25 CoTs/question), so each SVD is ~70x more expensive even though
# the feature count is unchanged.  Hence: run a layer subset first, then the full sweep.
#SBATCH --job-name=gpqa25-probe
#SBATCH --partition=small    # the batch QoS demands >=48 CPUs per job; small allows 2-192
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=runs/logs/slurm/slurm-%j.out
set -uo pipefail
source "${REPO_DIR:-${SLURM_SUBMIT_DIR:?run 'source env.sh' in the repo, then sbatch}}/runs/gpqa25/common.sh"
need_py "$PY"
SHARD=0; log_to "probe_${PROBE_TAG:-all}"

PROBE_OUT="$RUNS_ROOT/goal2/$RUN_TAG/probes_ckpt/${PROBE_TAG:-all}"
mkdir -p "$PROBE_OUT"
echo "layers=${PROBE_LAYERS:-all}  extra=${PROBE_EXTRA:-none}  out=$PROBE_OUT"

# BLAS threads: the SVDs are the whole cost, so let LAPACK use the allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"$PY" src/probing/probe_checkpoints.py \
  --act-dir "$ACT_DIR" --run-dir "$RUN_DIR" --out-dir "$PROBE_OUT" \
  --folds 5 --layers "${PROBE_LAYERS:-all}" ${PROBE_EXTRA:-}
rc=$?
echo "probe_checkpoints exit=$rc"

# The three stages are separate scripts: probe_checkpoints scores models,
# raw_units_table refits the chosen model to store OUT-OF-FOLD predictions (and
# expresses scores in raw units, e.g. tokens rather than R^2), and bootstrap_ci
# resamples those predictions over QUESTIONS -- 198 of them, not the 4,900 rows,
# because the 25 CoTs of one question are not independent observations.
# Report the CIs, not raw R^2 deltas.
if [ $rc -eq 0 ]; then
  echo; echo "=== raw units + out-of-fold predictions ==="
  # raw_units_table takes ONE layer. Use the layer the probe actually scored best on
  # (averaged over checkpoints and regression targets) rather than a guess.
  BEST=$("$PY" - "$PROBE_OUT/probe_checkpoints.csv" <<'PY'
import sys, pandas as pd
try:
    d = pd.read_csv(sys.argv[1])
    c = [x for x in ("act_best_layer", "best_layer", "layer") if x in d.columns]
    print(int(d[c[0]].dropna().mode().iloc[0]) if c else 20)
except Exception:
    print(20)
PY
)
  echo "  using layer $BEST for raw units / predictions"
  "$PY" src/probing/raw_units_table.py \
    --act-dir "$ACT_DIR" --run-dir "$RUN_DIR" --out-dir "$PROBE_OUT" \
    --layer "$BEST" 2>&1 | tail -25
  if [ -f "$PROBE_OUT/predictions.csv" ]; then
    for vs in text prefixlen; do
      echo; echo "=== bootstrap CI: activations vs $vs ==="
      "$PY" src/probing/bootstrap_ci.py --preds "$PROBE_OUT/predictions.csv" \
        --out-dir "$PROBE_OUT" --vs "$vs" 2>&1 | tail -30
    done
  else
    echo "raw_units_table produced no predictions.csv -- skipping the CIs"
  fi
fi
exit $rc
