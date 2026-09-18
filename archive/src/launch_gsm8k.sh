#!/bin/bash

SESSION="gsm8k"
VENV="source /playpen-ssd/yuvraj/.venv/bin/activate"
CMD="python3 /playpen-ssd/yuvraj/llm-probing-project/src/extract_activations.py"

tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -d -s $SESSION

# GPU 2 — workers 0-4
tmux new-window -t $SESSION -n "gpu2_w0"
tmux send-keys -t $SESSION:gpu2_w0 "$VENV && CUDA_VISIBLE_DEVICES=2 $CMD --worker_id 0 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu2_w1"
tmux send-keys -t $SESSION:gpu2_w1 "$VENV && CUDA_VISIBLE_DEVICES=2 $CMD --worker_id 1 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu2_w2"
tmux send-keys -t $SESSION:gpu2_w2 "$VENV && CUDA_VISIBLE_DEVICES=2 $CMD --worker_id 2 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu2_w3"
tmux send-keys -t $SESSION:gpu2_w3 "$VENV && CUDA_VISIBLE_DEVICES=2 $CMD --worker_id 3 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu2_w4"
tmux send-keys -t $SESSION:gpu2_w4 "$VENV && CUDA_VISIBLE_DEVICES=2 $CMD --worker_id 4 --num_workers 15" Enter

# GPU 3 — workers 5-9
tmux new-window -t $SESSION -n "gpu3_w5"
tmux send-keys -t $SESSION:gpu3_w5 "$VENV && CUDA_VISIBLE_DEVICES=3 $CMD --worker_id 5 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu3_w6"
tmux send-keys -t $SESSION:gpu3_w6 "$VENV && CUDA_VISIBLE_DEVICES=3 $CMD --worker_id 6 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu3_w7"
tmux send-keys -t $SESSION:gpu3_w7 "$VENV && CUDA_VISIBLE_DEVICES=3 $CMD --worker_id 7 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu3_w8"
tmux send-keys -t $SESSION:gpu3_w8 "$VENV && CUDA_VISIBLE_DEVICES=3 $CMD --worker_id 8 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu3_w9"
tmux send-keys -t $SESSION:gpu3_w9 "$VENV && CUDA_VISIBLE_DEVICES=3 $CMD --worker_id 9 --num_workers 15" Enter

# GPU 4 — workers 10-14
tmux new-window -t $SESSION -n "gpu4_w10"
tmux send-keys -t $SESSION:gpu4_w10 "$VENV && CUDA_VISIBLE_DEVICES=4 $CMD --worker_id 10 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu4_w11"
tmux send-keys -t $SESSION:gpu4_w11 "$VENV && CUDA_VISIBLE_DEVICES=4 $CMD --worker_id 11 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu4_w12"
tmux send-keys -t $SESSION:gpu4_w12 "$VENV && CUDA_VISIBLE_DEVICES=4 $CMD --worker_id 12 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu4_w13"
tmux send-keys -t $SESSION:gpu4_w13 "$VENV && CUDA_VISIBLE_DEVICES=4 $CMD --worker_id 13 --num_workers 15" Enter

tmux new-window -t $SESSION -n "gpu4_w14"
tmux send-keys -t $SESSION:gpu4_w14 "$VENV && CUDA_VISIBLE_DEVICES=4 $CMD --worker_id 14 --num_workers 15" Enter

echo "All workers launched in tmux session '$SESSION'"
echo "Attach with: tmux attach -t $SESSION"
echo "Switch windows with: Ctrl+b then w"