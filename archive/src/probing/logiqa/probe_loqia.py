"""
probe_logiqa_all_ablations.py
─────────────────────────────
Runs all 4 position ablations for LogiQA v3 classification probing.
Calls probe_utils_logiqa.run_ablation() for each, which internally
loops over both sampled and averaged activation types.

Ablations:
  1. last_think_token  — last position in thinking phase
  2. first_think_token — position 1 (first generated token)
  3. pos98             — second to last position window
  4. pos99             — last position window

Task: 4-class classification (A/B/C/D)
Probes: LogisticRegression + MLP
"""

import sys
import os
import pandas as pd

# Make sure probe_utils_logiqa is importable
sys.path.insert(0, "/playpen-ssd/yuvraj/llm-probing-project")
from probe_utils_logiqa import run_ablation


# ==========================================
# SLICE FUNCTIONS
# one per ablation — each returns a list of
# position indices into the 100-position axis
# ==========================================

def slice_last_think_token(row):
    try:
        think_frac = min(float(row['think_end_pos']) / float(row['sequence_length']), 1.0)
        n_think = max(1, int(think_frac * 100))
        pos = min(n_think - 1, 99)
    except (TypeError, ValueError):
        pos = 74  # fallback
    return [pos]


def slice_first_think_token(row):
    return [1]


def slice_pos98(row):
    return [98]


def slice_pos99(row):
    return [99]


# ==========================================
# MAIN
# ==========================================
ABLATIONS = [
    ("last_think_token",  slice_last_think_token),
    ("first_think_token", slice_first_think_token),
    ("pos98",             slice_pos98),
    ("pos99",             slice_pos99),
]

if __name__ == "__main__":
    for ablation_name, slice_fn in ABLATIONS:
        run_ablation(ablation_name, slice_fn)

    print("\nALL ABLATIONS COMPLETE")