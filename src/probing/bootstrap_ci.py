#!/usr/bin/env python3
"""Confidence intervals on the activation-vs-baseline gaps, by bootstrapping questions.

Every score in ``probe_checkpoints.csv`` and ``raw_units.csv`` is a point estimate
from ~195 questions, and the gaps we care about are 2-16 R^2 points.  Without an
interval there is no way to tell which of those gaps are real, and the fold-to-fold
spread (0.169 to 0.347 on one probe at one checkpoint) says the noise is the same
size as many of them.

Resampling is over QUESTIONS, not rows: the 3 CoTs of a question are 82-91%
correlated, so resampling rows would treat correlated observations as independent
and produce intervals that are far too narrow.  All of a question's rows move
together.

No refitting happens here.  It reads the stored out-of-fold predictions from
``predictions.csv``, so the intervals describe sampling variability in WHICH
questions we drew, which is the relevant uncertainty for "would this replicate on
a different sample of questions".

Also reports, for each gap, the number of questions that would be needed to make
it significant, on the 1/sqrt(n) scaling of the standard error.  That is the
number to use when deciding whether a bigger dataset is worth the GPU time.

Usage
-----
    python src/probing/bootstrap_ci.py
    python src/probing/bootstrap_ci.py --vs prefixlen    # compare against prefix length
"""
from __future__ import annotations

import argparse, builtins, functools
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(builtins.print, flush=True)

TARGETS = ["answer_entropy_bits_mm", "rem_len_mean", "rem_len_std", "p_mode"]


def _r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((p - y) ** 2).sum() / ss) if ss > 0 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds", default="runs/goal2/probes_ckpt/gpqa_diamond/predictions.csv")
    p.add_argument("--out-dir", default="runs/goal2/probes_ckpt/gpqa_diamond")
    p.add_argument("--vs", default="text", choices=["text", "prefixlen"],
                   help="baseline the activation probe is compared against")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    d = pd.read_csv(a.preds)
    rng = np.random.default_rng(a.seed)
    rows = []
    for tgt in TARGETS:
        base_col = f"{tgt}__pred_{a.vs}"
        if base_col not in d.columns:
            print(f"  {base_col} missing; skipping {tgt}")
            continue
        for f in sorted(d.fraction.unique()):
            s = d[(d.fraction == f) & d[f"{tgt}__true"].notna()
                  & d[base_col].notna() & d[f"{tgt}__pred"].notna()]
            if len(s) < 50:
                continue
            y = s[f"{tgt}__true"].to_numpy()
            pa = s[f"{tgt}__pred"].to_numpy()
            pb = s[base_col].to_numpy()
            q = s.example_id.to_numpy()
            uq = np.unique(q)
            where = {u: np.where(q == u)[0] for u in uq}

            diffs = np.empty(a.draws)
            for i in range(a.draws):
                samp = np.concatenate([where[u] for u in rng.choice(uq, len(uq), replace=True)])
                diffs[i] = _r2(y[samp], pa[samp]) - _r2(y[samp], pb[samp])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            obs = _r2(y, pa) - _r2(y, pb)
            se = float(diffs.std())
            # questions needed for the observed gap to clear zero at 95%
            need = len(uq) * (1.96 * se / abs(obs)) ** 2 if obs != 0 else np.inf
            rows.append(dict(target=tgt, checkpoint=f, n_questions=len(uq), n_rows=len(s),
                             r2_act=_r2(y, pa), r2_baseline=_r2(y, pb), gap=obs,
                             ci_lo=lo, ci_hi=hi, se=se,
                             significant=bool(lo > 0 or hi < 0),
                             favours=("activations" if obs > 0 else a.vs),
                             questions_needed=min(need, 1e9)))
            print(f"  {tgt:<24}{f:>5.1f}  gap {obs:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  "
                  f"{'SIGNIFICANT (' + rows[-1]['favours'] + ')' if rows[-1]['significant'] else 'n.s.'}")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    dest = out / f"bootstrap_ci_vs_{a.vs}.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"\n  wrote {dest}  ({len(rows)} comparisons, {a.draws} bootstrap draws)")


if __name__ == "__main__":
    main()
