"""
probe_gsm8k_think_only.py
─────────────────────────
Ablation: THINK-PHASE POSITIONS ONLY
Uses only the subset of the 100 positions that fall within
the <think>...</think> block, based on think_end_pos and sequence_length.

think_fraction = think_end_pos / sequence_length
think_positions = positions 0 .. int(think_fraction * 100)
"""
from probe_utils_gsm8k import run_ablation
import numpy as np
import pandas as pd

N_SAMPLES = 100  # Set to None for full run

def slice_fn(row):
    """
    Return the subset of [0..99] that corresponds to thinking tokens.
    Falls back to all 100 if think_end_pos is missing.
    """
    if pd.isna(row.get('think_end_pos')) or pd.isna(row.get('sequence_length')):
        return list(range(100))
    seq_len   = int(row['sequence_length'])
    think_end = int(row['think_end_pos'])
    if seq_len <= 0:
        return list(range(100))
    # Fraction of sequence that is thinking
    think_frac = min(think_end / seq_len, 1.0)
    # How many of the 100 positions fall in the thinking phase
    n_think = max(1, int(think_frac * 100))
    return list(range(n_think))

if __name__ == "__main__":
    run_ablation("think_only", slice_fn, n_samples=N_SAMPLES)