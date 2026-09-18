#!/bin/bash

SESSION="logiqa"
VENV="source /playpen-ssd/yuvraj/.venv/bin/activate"
CMD="python3 /playpen-ssd/yuvraj/llm-probing-project/src/extract/extract_activations_logiqa.py"

tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -d -s $SESSION

# GPU 5 — workers 0-4
tmux new-window -t $SESSION -n "gpu5_w0"
tmux send-keys -t $SESSION:gpu5_w0 "$VENV && CUDA_VISIBLE_DEVICES=5 $CMD --worker_id 0 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu5_w1"
tmux send-keys -t $SESSION:gpu5_w1 "$VENV && CUDA_VISIBLE_DEVICES=5 $CMD --worker_id 1 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu5_w2"
tmux send-keys -t $SESSION:gpu5_w2 "$VENV && CUDA_VISIBLE_DEVICES=5 $CMD --worker_id 2 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu5_w3"
tmux send-keys -t $SESSION:gpu5_w3 "$VENV && CUDA_VISIBLE_DEVICES=5 $CMD --worker_id 3 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu5_w4"
tmux send-keys -t $SESSION:gpu5_w4 "$VENV && CUDA_VISIBLE_DEVICES=5 $CMD --worker_id 4 --num_workers 15" Enter

# GPU 6 — workers 5-9
tmux new-window -t $SESSION -n "gpu6_w5"
tmux send-keys -t $SESSION:gpu6_w5 "$VENV && CUDA_VISIBLE_DEVICES=6 $CMD --worker_id 5 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu6_w6"
tmux send-keys -t $SESSION:gpu6_w6 "$VENV && CUDA_VISIBLE_DEVICES=6 $CMD --worker_id 6 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu6_w7"
tmux send-keys -t $SESSION:gpu6_w7 "$VENV && CUDA_VISIBLE_DEVICES=6 $CMD --worker_id 7 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu6_w8"
tmux send-keys -t $SESSION:gpu6_w8 "$VENV && CUDA_VISIBLE_DEVICES=6 $CMD --worker_id 8 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu6_w9"
tmux send-keys -t $SESSION:gpu6_w9 "$VENV && CUDA_VISIBLE_DEVICES=6 $CMD --worker_id 9 --num_workers 15" Enter

# GPU 7 — workers 10-14
tmux new-window -t $SESSION -n "gpu7_w10"
tmux send-keys -t $SESSION:gpu7_w10 "$VENV && CUDA_VISIBLE_DEVICES=7 $CMD --worker_id 10 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu7_w11"
tmux send-keys -t $SESSION:gpu7_w11 "$VENV && CUDA_VISIBLE_DEVICES=7 $CMD --worker_id 11 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu7_w12"
tmux send-keys -t $SESSION:gpu7_w12 "$VENV && CUDA_VISIBLE_DEVICES=7 $CMD --worker_id 12 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu7_w13"
tmux send-keys -t $SESSION:gpu7_w13 "$VENV && CUDA_VISIBLE_DEVICES=7 $CMD --worker_id 13 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu7_w14"
tmux send-keys -t $SESSION:gpu7_w14 "$VENV && CUDA_VISIBLE_DEVICES=7 $CMD --worker_id 14 --num_workers 15" Enter

echo "All LogiQA workers launched in tmux session '$SESSION'"
echo "Attach with: tmux attach -t $SESSION"
echo "Switch windows with: Ctrl+b then w"