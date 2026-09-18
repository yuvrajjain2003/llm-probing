#!/usr/bin/env bash
# Chained AIME work, unattended:
#   A) wait for the running K=100 AIME-2025 job
#   B) push a slice to K=300 so K* stops being censored at 100
#   C) measure K* on the extended grid
#   D) base CoTs for the combined 2022-2025 set (120 questions)
#   E) continuations at min(K*, KCAP) -- see the cost note below
#   F) labels, robustness, k-sufficiency, and the year/contamination probe
#
# KCAP exists because AIME is the most expensive dataset we have: at 120 questions
# a continuation costs ~3000 tokens, so K=100 is ~18h and K=300 is ~54h on 8 GPUs.
# If K* exceeds KCAP the run is knowingly under-powered and says so in the log.
set -uo pipefail
cd /playpen-ssd/shared/llm-probing-project
source env.sh
export OUTLINES_CACHE_DIR="${REPO_DIR}/.outlines_cache_yuvraj"
KCAP=${KCAP:-100}

A25=runs/goal1/full/aime2025/DeepSeek-R1-Distill-Qwen-7B
ALL=runs/goal1/full/aime_2022_2025/DeepSeek-R1-Distill-Qwen-7B
LOG=runs/goal1/full/pipeline_aime_next.log
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "A) waiting for the running AIME-2025 K=100 job"
while pgrep -f "continuations.py .*aime2025" >/dev/null; do sleep 60; done
"$PY" src/generation/merge_shards.py --run-dir "$A25" --no-metrics >>"$LOG" 2>&1
say "A) done"

say "B) pushing a slice (id%%8==0) to K=300 to uncensor K*"
"$PY" src/generation/gpu_pool.py --run-dir "$A25" --dataset aime2025 --k 300 \
  --num-shards 64 --indices 0,8,16,24,32,40,48,56 --gpus 0,1,2,3,4,5,6,7 \
  --max-tokens 16384 --max-model-len 24576 --max-n 4 --max-num-seqs 64 \
  >>"$LOG" 2>&1
"$PY" src/generation/merge_shards.py --run-dir "$A25" --no-metrics >>"$LOG" 2>&1

say "C) measuring K* on the extended grid (125-300)"
"$PY" src/analysis/k_sufficiency.py --run-dir "$A25" --kgrid extended --draws 200 >>"$LOG" 2>&1
KSTAR=$("$PY" - <<'PY'
import pandas as pd, math
try:
    s=pd.read_csv("runs/goal1/full/aime2025/DeepSeek-R1-Distill-Qwen-7B/k_sufficiency/summary.csv")
    v=s.loc[s.scope=="ALL","k_bias_mm"].iloc[0]
    print(int(v) if v==v and not math.isnan(float(v)) else 100)
except Exception: print(100)
PY
)
KUSE=$(( KSTAR < KCAP ? KSTAR : KCAP ))
say "C) K* = ${KSTAR}; running the 120-question set at K=${KUSE} (cap ${KCAP})"
[ "$KSTAR" -gt "$KCAP" ] && say "    WARNING: K* exceeds the cap -- combined AIME is under-powered by design"

say "D) base CoTs for aime_2022_2025 (120 questions x 3, cap 16384)"
./runs/goal1/full/launch_base_sharded.sh aime_2022_2025 120 16384 24576 8 >>"$LOG" 2>&1
"$PY" src/generation/merge_shards.py --run-dir "$ALL" >>"$LOG" 2>&1

say "E) continuations at K=${KUSE}"
"$PY" src/generation/gpu_pool.py --run-dir "$ALL" --dataset aime_2022_2025 --k "$KUSE" \
  --num-shards 16 --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-tokens 16384 --max-model-len 24576 --max-n 4 --max-num-seqs 64 \
  >>"$LOG" 2>&1

say "F) labels / robustness / k-sufficiency / year effect"
"$PY" src/generation/merge_shards.py --run-dir "$ALL" >>"$LOG" 2>&1
"$PY" src/analysis/labels.py --continuations "$ALL/raw/continuations.jsonl" >>"$LOG" 2>&1
"$PY" src/analysis/robustness.py --run-dir "$ALL" >>"$LOG" 2>&1
"$PY" src/analysis/k_sufficiency.py --run-dir "$ALL" --kgrid extended --draws 200 >>"$LOG" 2>&1
"$PY" src/analysis/year_effect.py --run-dir "$ALL" >>"$LOG" 2>&1
say "PIPELINE COMPLETE (K* = ${KSTAR}, ran at K = ${KUSE})"
