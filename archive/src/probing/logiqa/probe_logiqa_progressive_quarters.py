"""probe_logiqa_progressive_quarters.py — Ablation: PROGRESSIVE QUARTERS OF THINKING"""
from probe_utils_logiqa import run_ablation
import pandas as pd

N_SAMPLES = 100  # Set to None for full run

def slice_fn(row):
    if pd.isna(row.get('think_end_pos')) or pd.isna(row.get('sequence_length')):
        return [24, 49, 74, 99]
    seq_len   = int(row['sequence_length'])
    think_end = int(row['think_end_pos'])
    if seq_len <= 0:
        return [24, 49, 74, 99]
    think_frac = min(think_end / seq_len, 1.0)
    n_think    = max(4, int(think_frac * 100))
    q1 = max(0, int(n_think * 0.25) - 1)
    q2 = max(0, int(n_think * 0.50) - 1)
    q3 = max(0, int(n_think * 0.75) - 1)
    q4 = max(0, n_think - 1)
    seen, indices = set(), []
    for i in [q1, q2, q3, q4]:
        if i not in seen:
            seen.add(i)
            indices.append(i)
    return indices

if __name__ == "__main__":
    run_ablation("progressive_quarters", slice_fn, n_samples=N_SAMPLES)