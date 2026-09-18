"""probe_logiqa_last_token.py — Ablation: LAST TOKEN ONLY"""
from probe_utils_logiqa import run_ablation

N_SAMPLES = None  # Set to None for full run

def slice_fn(row):
    return [99]

if __name__ == "__main__":
    run_ablation("last_token", slice_fn, n_samples=N_SAMPLES)