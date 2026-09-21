#!/usr/bin/env bash
# GPQA-Diamond, unattended. The point of the phase split: MEASURE the required K
# on a 20% slice, then run the remaining 80% at that K instead of guessing 100.
#
#   phase A  20% of questions at K=100  (headroom to actually locate K*)
#   phase B  k-sufficiency on phase A   -> K*
#   phase C  remaining 80% at K*
#
# All stages resumable (append-only JSONL + resume-by-count); re-running continues.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source env.sh

RUN=runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B
LOG=runs/goal1/full/pipeline_gpqa.log
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "STAGE 1/7  base CoTs, all 198 questions x 3 (cap 16384)"
./runs/goal1/full/launch_base_sharded.sh gpqa_diamond 198 16384 24576 8 >>"$LOG" 2>&1
"$PY" src/generation/merge_shards.py --run-dir "$RUN" >>"$LOG" 2>&1

# MAXN=8 not 20: at a 16384-token budget a 20-wide request group needs ~1.17x the
# KV cache and vLLM dies on CPU swap (that is what killed AIME). 8 keeps it at 0.47x.
say "STAGE 2/7  phase A: 20% slice (id%%5==0) at K=100"
MAXN=8 SUBSET_MOD=5 SUBSET_KEEP=0 LOGTAG=_pA \
  ./runs/goal1/full/launch_cont_sharded.sh gpqa_diamond 100 16384 24576 8 0 >>"$LOG" 2>&1
"$PY" src/generation/merge_shards.py --run-dir "$RUN" --no-metrics >>"$LOG" 2>&1

say "STAGE 3/7  phase B: k-sufficiency on the 20% slice"
"$PY" src/analysis/k_sufficiency.py --run-dir "$RUN" --draws 200 >>"$LOG" 2>&1

# Read K* back out. k_bias_mm = smallest K whose Miller-Madow entropy is within
# tolerance of the K->infinity estimate; that is the criterion math500 settled on.
KSTAR=$("$PY" - <<'PY'
import pandas as pd, math
try:
    s = pd.read_csv("runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B/k_sufficiency/summary.csv")
    v = s.loc[s.scope == "ALL", "k_bias_mm"].iloc[0]
    print(int(v) if v == v and not math.isnan(float(v)) else 100)
except Exception:
    print(100)
PY
)
say "STAGE 4/7  K* measured on GPQA = ${KSTAR}"

say "STAGE 5/7  phase C: remaining 80% (id%%5!=0) at K=${KSTAR}"
MAXN=8 SUBSET_MOD=5 SUBSET_KEEP=1,2,3,4 LOGTAG=_pC \
  ./runs/goal1/full/launch_cont_sharded.sh gpqa_diamond "$KSTAR" 16384 24576 8 0 >>"$LOG" 2>&1

say "STAGE 6/7  merge + labels + robustness"
"$PY" src/generation/merge_shards.py --run-dir "$RUN" >>"$LOG" 2>&1
"$PY" src/analysis/labels.py --continuations "$RUN/raw/continuations.jsonl" >>"$LOG" 2>&1
"$PY" src/analysis/robustness.py --run-dir "$RUN" >>"$LOG" 2>&1

say "STAGE 7/7  final k-sufficiency over everything"
"$PY" src/analysis/k_sufficiency.py --run-dir "$RUN" --draws 200 >>"$LOG" 2>&1
say "PIPELINE COMPLETE (K* = ${KSTAR})"
