#!/usr/bin/env python3
"""Contamination probe: do the labels differ between AIME years?

This model is distilled from R1 (trained 2024), so AIME 2022-2024 problems were
plausibly in its training data while 2025 was not. If memorisation drives the
labels, the earlier years should show LOWER entropy, EARLIER commitment and
HIGHER accuracy. That is testable rather than assumed, because examples.csv
carries `year`.

A null result is just as useful: it justifies pooling all 120 questions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()
    run = Path(args.run_dir)

    ex = pd.read_csv(run / "raw" / "examples.csv")[["example_id", "year"]]
    ck = pd.read_csv(run / "labels" / "checkpoint_labels.csv").merge(ex, on="example_id")
    cm = pd.read_csv(run / "labels" / "commitment.csv").merge(ex, on="example_id")

    by = ck.groupby("year").agg(
        questions=("example_id", "nunique"),
        entropy=("answer_entropy_bits", "mean"),
        p_mode=("p_mode", "mean"),
        p_gold=("p_gold", "mean"),
        uniq_answers=("n_unique_answers", "mean"),
        rem_tokens=("rem_len_mean", "mean"),
        acc=("acc_single", "mean"),
    ).round(3)
    cmt = cm.groupby("year").agg(committed=("committed", "mean"),
                                 commit_frac=("commitment_fraction", "mean")).round(3)
    out = by.join(cmt)
    print("\n--- labels by AIME year (2022-24 plausibly in training data; 2025 not) ---")
    print(out.to_string())

    # Test at QUESTION level, not checkpoint level. The ~1,458 checkpoint rows come
    # from only ~120 questions (5 checkpoints x 3 CoTs each), and rows within a
    # question are strongly correlated. Feeding them to the test as if independent
    # inflates the effective n by ~12x and would report significance for
    # differences that are not there. One mean per question is the honest unit.
    perq = (ck.groupby(["example_id", "year"], as_index=False)
              .agg(entropy=("answer_entropy_bits", "mean"),
                   acc=("acc_single", "mean")))
    pre = perq[perq.year < 2025]
    post = perq[perq.year == 2025]
    print(f"\n--- contamination test (unit = question, not checkpoint) ---")
    if len(pre) and len(post):
        d = pre.entropy.mean() - post.entropy.mean()
        # Cliff's delta: how often a 2022-24 question exceeds a 2025 one. Reported
        # because a p-value says nothing about how BIG the difference is.
        import numpy as np
        a, b = pre.entropy.to_numpy(), post.entropy.to_numpy()
        delta = float(np.mean(np.sign(a[:, None] - b[None, :])))
        print(f"  entropy  2022-24: n={len(pre):3d} mean {pre.entropy.mean():.3f}")
        print(f"           2025   : n={len(post):3d} mean {post.entropy.mean():.3f}")
        print(f"  difference {d:+.3f} bits   Cliff's delta {delta:+.3f} "
              f"(0 = no effect, +-1 = total separation)")
        try:
            from scipy import stats
            _, pv = stats.mannwhitneyu(a, b, alternative="two-sided")
            print(f"  Mann-Whitney U p = {pv:.4f}"
                  f"{'  <- significant' if pv < 0.05 else '  <- not significant'}")
        except ImportError:
            print("  (scipy unavailable -- effect size reported without a p-value)")
        print("  NOTE: 30 questions per year is a small sample; treat a null result")
        print("        as 'no detectable effect', not as 'no contamination'.")
    outdir = run / "labels"; outdir.mkdir(exist_ok=True, parents=True)
    out.to_csv(outdir / "year_effect.csv")
    print(f"\nwrote {outdir/'year_effect.csv'}\n")


if __name__ == "__main__":
    main()
