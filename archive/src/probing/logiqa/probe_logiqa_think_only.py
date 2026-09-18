"""probe_logiqa_think_only.py — Ablation: THINK-PHASE POSITIONS ONLY"""
from probe_utils_logiqa import run_ablation
import pandas as pd

N_SAMPLES = None  # Set to None for full run

def slice_fn(row):
    if pd.isna(row.get('think_end_pos')) or pd.isna(row.get('sequence_length')):
        return list(range(100))
    seq_len   = int(row['sequence_length'])
    think_end = int(row['think_end_pos'])
    if seq_len <= 0:
        return list(range(100))
    think_frac = min(think_end / seq_len, 1.0)
    n_think    = max(1, int(think_frac * 100))
    return list(range(n_think))

if __name__ == "__main__":
    run_ablation("think_only", slice_fn, n_samples=N_SAMPLES)