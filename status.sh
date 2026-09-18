#!/usr/bin/env bash
# Overnight status:  ./status.sh
cd /playpen-ssd/shared/llm-probing-project
R=runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B
echo "=================== GOAL 1 / GPQA PIPELINE ==================="
if tmux has-session -t goal1 2>/dev/null; then echo "tmux 'goal1' : ALIVE"
else echo "tmux 'goal1' : GONE (finished, or died)"; fi
echo
echo "--- stages ---"
grep -E "STAGE|COMPLETE|K\* " runs/goal1/full/pipeline_gpqa.log 2>/dev/null | tail -8
echo
b=$(cat "$R"/raw/sample_records.shard*.jsonl 2>/dev/null | wc -l)
echo "--- base CoTs:      $b / 594"
c=$(cat "$R"/raw/continuations.shard*.jsonl 2>/dev/null | wc -l)
t=$(grep -h "prefixes need work" runs/goal1/full/logs/cont_gpqa_diamond*_shard*.log 2>/dev/null | awk '{s+=$5} END {print s+0}')
echo "--- continuations:  $c / ${t:-?}"
if [ -f "$R/k_sufficiency/summary.csv" ]; then
  echo
  echo "--- K-sufficiency (GPQA) ---"; cat "$R/k_sufficiency/summary.csv"
fi
echo
echo "--- GPUs ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' '; echo
echo "--- real errors ---"
hits=$(grep -lE "No space left on device|lack of CPU swap space|CUDA out of memory" runs/goal1/full/logs/*gpqa*.log 2>/dev/null | head -5)
if [ -n "$hits" ]; then echo "$hits"; else echo "  none"; fi
echo "--- disk ---"; df -h /playpen-ssd | tail -1
echo "=============================================================="
