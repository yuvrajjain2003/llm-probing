"""
probe_gsm8k_last_token.py
─────────────────────────
Ablation: LAST TOKEN ONLY
Uses only position index 99 — the final generated token.
This has attended to the entire sequence via attention.
"""
from probe_utils_gsm8k import run_ablation

N_SAMPLES = None # Set to None for full run

def slice_fn(row):
    """Use only the last of the 100 positions."""
    return [99]

if __name__ == "__main__":
    run_ablation("last_token", slice_fn, n_samples=N_SAMPLES)