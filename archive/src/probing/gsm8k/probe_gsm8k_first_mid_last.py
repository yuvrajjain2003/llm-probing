"""
probe_gsm8k_first_mid_last.py
─────────────────────────────
Ablation: FIRST / MID / LAST
Uses position indices 0, 49, 99 — beginning, middle, and end
of the 100 stored positions. Gives a coarse trajectory.
"""
from probe_utils_gsm8k import run_ablation

N_SAMPLES = 100  # Set to None for full run

def slice_fn(row):
    """Use positions 0, 49, 99."""
    return [0, 49, 99]

if __name__ == "__main__":
    run_ablation("first_mid_last", slice_fn, n_samples=N_SAMPLES)