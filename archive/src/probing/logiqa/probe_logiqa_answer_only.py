"""probe_logiqa_answer_only.py — Ablation: ANSWER-PHASE POSITIONS ONLY"""
from probe_utils_logiqa import run_ablation
import pandas as pd

N_SAMPLES = 100  # Set to None for full run

def slice_fn(row):
    if pd.isna(row.get('think_end_pos')) or pd.isna(row.get('sequence_length')):
        return [99]
    seq_len   = int(row['sequence_length'])
    think_end = int(row['think_end_pos'])
    if seq_len <= 0:
        return [99]
    think_frac   = min(think_end / seq_len, 1.0)
    n_think      = int(think_frac * 100)
    answer_start = min(n_think, 99)
    indices      = list(range(answer_start, 100))
    return indices if indices else [99]

if __name__ == "__main__":
    run_ablation("answer_only", slice_fn, n_samples=N_SAMPLES)