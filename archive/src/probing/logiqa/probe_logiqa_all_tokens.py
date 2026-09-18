"""probe_logiqa_all_tokens.py — Ablation: ALL 100 POSITIONS"""
from probe_utils_logiqa import run_ablation

N_SAMPLES = 100  # Set to None for full run

def slice_fn(row):
    return list(range(100))

if __name__ == "__main__":
    run_ablation("all_tokens", slice_fn, n_samples=N_SAMPLES)