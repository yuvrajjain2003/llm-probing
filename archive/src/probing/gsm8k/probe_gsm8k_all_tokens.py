"""
probe_gsm8k_all_tokens.py
─────────────────────────
Ablation: ALL 100 POSITIONS
Uses all 100 evenly-spaced positions stored in v3 activations.
"""
from probe_utils_gsm8k import run_ablation
import numpy as np

N_SAMPLES = 100  # Set to None for full run

def slice_fn(row):
    """Use all 100 positions."""
    return list(range(100))

if __name__ == "__main__":
    run_ablation("all_tokens", slice_fn, n_samples=N_SAMPLES)