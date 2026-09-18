"""
probe_gsm8k_progressive_quarters.py
────────────────────────────────────
Ablation: PROGRESSIVE QUARTERS OF THINKING
Splits the thinking-phase positions into 4 equal quarters and
takes the LAST position of each quarter. This gives 4 positions
that trace how the answer crystallizes through reasoning.

Positions used: 25%, 50%, 75%, 100% through the thinking phase.
"""
from probe_utils_gsm8k import run_ablation
import pandas as pd

N_SAMPLES = 100  # Set to None for full run

def slice_fn(row):
    """
    Find the 4 quarter-boundary positions within the thinking phase.
    Returns 4 position indices from [0..99].
    Falls back gracefully if think_end_pos is missing.
    """
    if pd.isna(row.get('think_end_pos')) or pd.isna(row.get('sequence_length')):
        # No think boundary — use evenly spaced positions across full sequence
        return [24, 49, 74, 99]

    seq_len   = int(row['sequence_length'])
    think_end = int(row['think_end_pos'])
    if seq_len <= 0:
        return [24, 49, 74, 99]

    think_frac = min(think_end / seq_len, 1.0)
    n_think    = max(4, int(think_frac * 100))  # at least 4 positions

    # Last position of each quarter of the thinking phase
    q1 = max(0, int(n_think * 0.25) - 1)
    q2 = max(0, int(n_think * 0.50) - 1)
    q3 = max(0, int(n_think * 0.75) - 1)
    q4 = max(0, n_think - 1)

    # Deduplicate while preserving order
    seen = set()
    indices = []
    for i in [q1, q2, q3, q4]:
        if i not in seen:
            seen.add(i)
            indices.append(i)

    return indices

if __name__ == "__main__":
    run_ablation("progressive_quarters", slice_fn, n_samples=N_SAMPLES)