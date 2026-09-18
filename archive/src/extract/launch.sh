#!/bin/bash
# launch_v3_full.sh
# GSM8K:  GPUs 0-3, 5 workers each = 20 workers total, ~374 examples each
# LogiQA: GPUs 4-7, 5 workers each = 20 workers total, ~370 examples each

SESSION="extraction_v3"
VENV="source /playpen-ssd/yuvraj/.venv/bin/activate"
GSM_CMD="python3 /playpen-ssd/yuvraj/llm-probing-project/src/extract/extract_activations_gsm8k_v3.py"
LOG_CMD="python3 /playpen-ssd/yuvraj/llm-probing-project/src/extract/extract_activations_logiqa_v3.py"

tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -d -s $SESSION

# ── GSM8K: GPUs 0-3, 5 workers each, 20 workers total ───────────────────────

# GPU 0 — workers 0-4
tmux new-window -t $SESSION -n "gsm_g0_w0"
tmux send-keys -t $SESSION:gsm_g0_w0 "$VENV && CUDA_VISIBLE_DEVICES=0 $GSM_CMD --worker_id 0 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g0_w1"
tmux send-keys -t $SESSION:gsm_g0_w1 "$VENV && CUDA_VISIBLE_DEVICES=0 $GSM_CMD --worker_id 1 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g0_w2"
tmux send-keys -t $SESSION:gsm_g0_w2 "$VENV && CUDA_VISIBLE_DEVICES=0 $GSM_CMD --worker_id 2 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g0_w3"
tmux send-keys -t $SESSION:gsm_g0_w3 "$VENV && CUDA_VISIBLE_DEVICES=0 $GSM_CMD --worker_id 3 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g0_w4"
tmux send-keys -t $SESSION:gsm_g0_w4 "$VENV && CUDA_VISIBLE_DEVICES=0 $GSM_CMD --worker_id 4 --num_workers 20 --gpu_id 0" Enter

# GPU 1 — workers 5-9
tmux new-window -t $SESSION -n "gsm_g1_w5"
tmux send-keys -t $SESSION:gsm_g1_w5 "$VENV && CUDA_VISIBLE_DEVICES=1 $GSM_CMD --worker_id 5 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g1_w6"
tmux send-keys -t $SESSION:gsm_g1_w6 "$VENV && CUDA_VISIBLE_DEVICES=1 $GSM_CMD --worker_id 6 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g1_w7"
tmux send-keys -t $SESSION:gsm_g1_w7 "$VENV && CUDA_VISIBLE_DEVICES=1 $GSM_CMD --worker_id 7 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g1_w8"
tmux send-keys -t $SESSION:gsm_g1_w8 "$VENV && CUDA_VISIBLE_DEVICES=1 $GSM_CMD --worker_id 8 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g1_w9"
tmux send-keys -t $SESSION:gsm_g1_w9 "$VENV && CUDA_VISIBLE_DEVICES=1 $GSM_CMD --worker_id 9 --num_workers 20 --gpu_id 0" Enter

# GPU 2 — workers 10-14
tmux new-window -t $SESSION -n "gsm_g2_w10"
tmux send-keys -t $SESSION:gsm_g2_w10 "$VENV && CUDA_VISIBLE_DEVICES=2 $GSM_CMD --worker_id 10 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g2_w11"
tmux send-keys -t $SESSION:gsm_g2_w11 "$VENV && CUDA_VISIBLE_DEVICES=2 $GSM_CMD --worker_id 11 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g2_w12"
tmux send-keys -t $SESSION:gsm_g2_w12 "$VENV && CUDA_VISIBLE_DEVICES=2 $GSM_CMD --worker_id 12 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g2_w13"
tmux send-keys -t $SESSION:gsm_g2_w13 "$VENV && CUDA_VISIBLE_DEVICES=2 $GSM_CMD --worker_id 13 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g2_w14"
tmux send-keys -t $SESSION:gsm_g2_w14 "$VENV && CUDA_VISIBLE_DEVICES=2 $GSM_CMD --worker_id 14 --num_workers 20 --gpu_id 0" Enter

# GPU 3 — workers 15-19
tmux new-window -t $SESSION -n "gsm_g3_w15"
tmux send-keys -t $SESSION:gsm_g3_w15 "$VENV && CUDA_VISIBLE_DEVICES=3 $GSM_CMD --worker_id 15 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g3_w16"
tmux send-keys -t $SESSION:gsm_g3_w16 "$VENV && CUDA_VISIBLE_DEVICES=3 $GSM_CMD --worker_id 16 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g3_w17"
tmux send-keys -t $SESSION:gsm_g3_w17 "$VENV && CUDA_VISIBLE_DEVICES=3 $GSM_CMD --worker_id 17 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g3_w18"
tmux send-keys -t $SESSION:gsm_g3_w18 "$VENV && CUDA_VISIBLE_DEVICES=3 $GSM_CMD --worker_id 18 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "gsm_g3_w19"
tmux send-keys -t $SESSION:gsm_g3_w19 "$VENV && CUDA_VISIBLE_DEVICES=3 $GSM_CMD --worker_id 19 --num_workers 20 --gpu_id 0" Enter

# ── LogiQA: GPUs 4-7, 5 workers each, 20 workers total ──────────────────────

# GPU 4 — workers 0-4
tmux new-window -t $SESSION -n "log_g4_w0"
tmux send-keys -t $SESSION:log_g4_w0 "$VENV && CUDA_VISIBLE_DEVICES=4 $LOG_CMD --worker_id 0 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g4_w1"
tmux send-keys -t $SESSION:log_g4_w1 "$VENV && CUDA_VISIBLE_DEVICES=4 $LOG_CMD --worker_id 1 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g4_w2"
tmux send-keys -t $SESSION:log_g4_w2 "$VENV && CUDA_VISIBLE_DEVICES=4 $LOG_CMD --worker_id 2 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g4_w3"
tmux send-keys -t $SESSION:log_g4_w3 "$VENV && CUDA_VISIBLE_DEVICES=4 $LOG_CMD --worker_id 3 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g4_w4"
tmux send-keys -t $SESSION:log_g4_w4 "$VENV && CUDA_VISIBLE_DEVICES=4 $LOG_CMD --worker_id 4 --num_workers 20 --gpu_id 0" Enter

# GPU 5 — workers 5-9
tmux new-window -t $SESSION -n "log_g5_w5"
tmux send-keys -t $SESSION:log_g5_w5 "$VENV && CUDA_VISIBLE_DEVICES=5 $LOG_CMD --worker_id 5 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g5_w6"
tmux send-keys -t $SESSION:log_g5_w6 "$VENV && CUDA_VISIBLE_DEVICES=5 $LOG_CMD --worker_id 6 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g5_w7"
tmux send-keys -t $SESSION:log_g5_w7 "$VENV && CUDA_VISIBLE_DEVICES=5 $LOG_CMD --worker_id 7 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g5_w8"
tmux send-keys -t $SESSION:log_g5_w8 "$VENV && CUDA_VISIBLE_DEVICES=5 $LOG_CMD --worker_id 8 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g5_w9"
tmux send-keys -t $SESSION:log_g5_w9 "$VENV && CUDA_VISIBLE_DEVICES=5 $LOG_CMD --worker_id 9 --num_workers 20 --gpu_id 0" Enter

# GPU 6 — workers 10-14
tmux new-window -t $SESSION -n "log_g6_w10"
tmux send-keys -t $SESSION:log_g6_w10 "$VENV && CUDA_VISIBLE_DEVICES=6 $LOG_CMD --worker_id 10 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g6_w11"
tmux send-keys -t $SESSION:log_g6_w11 "$VENV && CUDA_VISIBLE_DEVICES=6 $LOG_CMD --worker_id 11 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g6_w12"
tmux send-keys -t $SESSION:log_g6_w12 "$VENV && CUDA_VISIBLE_DEVICES=6 $LOG_CMD --worker_id 12 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g6_w13"
tmux send-keys -t $SESSION:log_g6_w13 "$VENV && CUDA_VISIBLE_DEVICES=6 $LOG_CMD --worker_id 13 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g6_w14"
tmux send-keys -t $SESSION:log_g6_w14 "$VENV && CUDA_VISIBLE_DEVICES=6 $LOG_CMD --worker_id 14 --num_workers 20 --gpu_id 0" Enter

# GPU 7 — workers 15-19
tmux new-window -t $SESSION -n "log_g7_w15"
tmux send-keys -t $SESSION:log_g7_w15 "$VENV && CUDA_VISIBLE_DEVICES=7 $LOG_CMD --worker_id 15 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g7_w16"
tmux send-keys -t $SESSION:log_g7_w16 "$VENV && CUDA_VISIBLE_DEVICES=7 $LOG_CMD --worker_id 16 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g7_w17"
tmux send-keys -t $SESSION:log_g7_w17 "$VENV && CUDA_VISIBLE_DEVICES=7 $LOG_CMD --worker_id 17 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g7_w18"
tmux send-keys -t $SESSION:log_g7_w18 "$VENV && CUDA_VISIBLE_DEVICES=7 $LOG_CMD --worker_id 18 --num_workers 20 --gpu_id 0" Enter

tmux new-window -t $SESSION -n "log_g7_w19"
tmux send-keys -t $SESSION:log_g7_w19 "$VENV && CUDA_VISIBLE_DEVICES=7 $LOG_CMD --worker_id 19 --num_workers 20 --gpu_id 0" Enter

echo "All 40 workers launched in tmux session '$SESSION'"
echo "  GSM8K:  20 workers on GPUs 0-3 (~374 examples each)"
echo "  LogiQA: 20 workers on GPUs 4-7 (~370 examples each)"
echo ""
echo "Attach with: tmux attach -t $SESSION"
echo "Switch windows with: Ctrl+b then w"
echo ""
echo "Monitor GSM8K:  watch -n 60 'ls /playpen-ssd/yuvraj/llm-probing-project/activations/gsm8k_v3/sampled/*.pt | wc -l'"
echo "Monitor LogiQA: watch -n 60 'ls /playpen-ssd/yuvraj/llm-probing-project/activations/logiqa_v3/sampled/*.pt | wc -l'"