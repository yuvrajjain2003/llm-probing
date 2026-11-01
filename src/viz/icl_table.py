#!/usr/bin/env python3
"""Black-box in-context LLMs vs our probe, scored exactly as the master table scores everything else.

Separate from master_table.py because the ICL runs cover only the 1,178-row subsample (1 CoT per
question): folding them into the master table would shrink all 22 approaches to those rows. Here the
ICL row set defines the common set instead, and the probes are re-scored on it.

    srun -p small -c 4 --mem=32G -t 30 ./envs/goal1/bin/python src/viz/icl_table.py
Outputs: runs/goal2/gpqa25/sweep/_icl/{ci_scores.csv, ci_paired.csv, table.md}
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import master_table as mt

OUT = Path("runs/goal2/gpqa25/sweep/_icl"); OUT.mkdir(exist_ok=True)
ICL = Path("runs/goal2/gpqa25/icl")

# in-context runs to show: label -> (raw tag, how many examples)
RUNS = [("GPT-6 Luna, k=0", "gpt6_luna_high_k0"),
        ("GPT-6 Luna, k=10", "gpt6_luna_high_k10"),
        ("GPT-6 Luna, k=50", "gpt6_luna_high_k50"),
        ("GPT-6 Luna, k=10, examples hide depth", "gpt6_luna_high_nockpt_k10"),
        ("GPT-6 Luna, k=50, examples hide depth", "gpt6_luna_high_nockpt_k50"),
        ("GPT-6 Luna, k=0, depth hidden everywhere", "gpt6_luna_high_blind_k0"),
        ("GPT-6 Luna, k=10, depth hidden everywhere", "gpt6_luna_high_blind_k10"),
        ("GPT-6 Luna, k=50, depth hidden everywhere", "gpt6_luna_high_blind_k50"),
        ("Gemma-4-31B, k=0 (best local ICL)", "gemma4_31b_off_k0"),
        ("Gemma, k=10", "gemma4_31b_off_k10"),
        ("Gemma, k=30", "gemma4_31b_off_k30"),
        ("Gemma, k=93", "gemma4_31b_off_k93"),
        ("Gemma, k=10, examples shortened", "gemma4_31b_off_s2_k10"),
        ("Gemma, k=30, examples shortened", "gemma4_31b_off_s2_k30"),
        ("Gemma, k=93, examples shortened", "gemma4_31b_off_s2_k93"),
        ("Gemma, k=400, examples shortened", "gemma4_31b_off_s2_k400"),
        ("Qwen3.5-9B, k=0", "qwen35_9b_off_k0"),
        ("Our own model, k=0", "r1qwen7b_off_k0")]

SPEC = {"Our model L21 (ridge)": ("ref_fixed", "act_L21", None, "probe"),
        "TF-IDF on prefix":      ("ref_fixed", "tfidf", None, "text baseline"),
        "Prefix length":         ("ref_fixed", "prefix_len", None, "text baseline")}

base_label, N = None, 0
for lab, tag in RUNS:
    f = ICL / f"preds_{tag}.csv"
    if not f.exists():
        print(f"skip {lab}: no predictions yet"); continue
    d = pd.read_csv(f)
    n = len(d[(d.probe == "icl_cal") & (d.target == mt.TARGETS[0]) & (d.fraction > 0)])
    if n < 900:          # 196 rows x 5 checkpoints = 980 when complete
        print(f"skip {lab}: only {n} of 980 scored rows"); continue
    SPEC[lab] = (f"icl:{tag}", tag, "icl_cal", "in-context LLM")
    if n > N:            # the biggest complete ICL run defines the common rows
        base_label, N = lab, n

assert base_label, "no complete ICL run found"
print(f"{len(SPEC) - 3} in-context runs; common rows from {base_label} ({N})")
# also pair each "depth hidden" run against the matching run that shows depth (the professor's ask)
REFS = ["Our model L21 (ridge)"] + [l for l in ("GPT-6 Luna, k=0", "GPT-6 Luna, k=10",
                                                "GPT-6 Luna, k=50", "Gemma, k=10", "Gemma, k=30",
                                                "Gemma, k=93") if l in SPEC]
mt.main(SPEC=SPEC, REFS=REFS, OUT=OUT, base_label=base_label,
        title="In-context LLMs vs the probe (calibrated predictions, same rows)")
