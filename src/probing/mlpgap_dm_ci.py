#!/usr/bin/env python3
"""Paired question-bootstrap CI for NONLINEAR minus LINEAR kernel ridge on question-demeaned
targets -- the sharp form of "does a nonlinear readout see within-question structure that a linear
one cannot?".

mlpgap_demeaned.py already CIs each kernel against the STANDARD probe. That is the wrong pairing
for this question: both the linear and the nonlinear demeaned fits differ from the standard probe
by more than they differ from each other, so those CIs cannot resolve a ~0.005 gap. Here the
reference is the LINEAR kernel on the same rows, folds and demeaned target, with shared bootstrap
draws, so the difference is paired.

Metric: pooled out-of-fold R2 against the demeaned target ydm = y - (question mean of y), rebuilt
from the stored y_true exactly as mlpgap_demeaned.py built it (oracle question means).

USAGE  ./envs/goal1/bin/python src/probing/mlpgap_dm_ci.py [--ref linear]
OUT    runs/goal2/gpqa25/sweep/_mlpgap/demeaned_krr_paired.csv
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.mlpgap_core import OUT, print  # noqa: E402

B, SEED = 2000, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="linear")
    ap.add_argument("--glob", default="demeaned_krr*.preds.csv")
    ap.add_argument("--exclude", default="smoke",
                    help="skip preds files whose name contains this (the 60-question smoke\n                          shares (example_id, cot_id) keys with the real layer-21 file, and\n                          pivot_table would silently AVERAGE the duplicates)")
    a = ap.parse_args()
    fs = [f for f in sorted(OUT.glob(a.glob)) if not (a.exclude and a.exclude in f.name)]
    assert fs, f"no {a.glob} under {OUT}"
    P = pd.concat([pd.read_csv(f).assign(src=f.name) for f in fs], ignore_index=True)
    P["layers"] = P.src.str.extract(r"_L(\d+)\.preds")[0].fillna("21")
    print(f"  {len(fs)} files {[f.name for f in fs]}\n  {len(P)} rows, "
          f"layers {sorted(P.layers.unique())}, kernels {sorted(P.kernel.unique())}")
    assert not P.duplicated(["layers", "kernel", "target", "fraction",
                             "example_id", "cot_id"]).any(), "duplicate prediction keys"

    rows = []
    for (lay, t, f), x in P.groupby(["layers", "target", "fraction"]):
        ks = sorted(x.kernel.unique())
        if a.ref not in ks:
            print(f"  skip {lay} {t} f{f}: no {a.ref}")
            continue
        w = x.pivot_table(index=["example_id", "cot_id"], columns="kernel",
                          values="y_pred")[ks].dropna()
        yt = (x.drop_duplicates(["example_id", "cot_id"])
               .set_index(["example_id", "cot_id"]).y_true.loc[w.index].to_numpy())
        q = w.index.get_level_values(0).to_numpy()
        ydm = yt - pd.Series(yt).groupby(q).transform("mean").to_numpy()
        qs = np.unique(q); Q = len(qs); qi = {int(v): i for i, v in enumerate(qs)}
        qq = np.array([qi[int(e)] for e in q])
        W = np.random.default_rng(SEED).multinomial(Q, np.ones(Q) / Q, size=B).astype(float)
        agg = lambda v: np.bincount(qq, weights=v, minlength=Q)
        n1, s1, s2 = agg(np.ones(len(ydm))), agg(ydm), agg(ydm * ydm)
        draws, pt = {}, {}
        for k in ks:
            se = agg((w[k].to_numpy() - ydm) ** 2)
            draws[k] = 1 - (W @ se) / ((W @ s2) - (W @ s1) ** 2 / (W @ n1))
            pt[k] = 1 - se.sum() / (s2.sum() - s1.sum() ** 2 / n1.sum())
        for k in ks:
            if k == a.ref:
                continue
            d = draws[k] - draws[a.ref]
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append(dict(layers=lay, target=t, fraction=f, kernel=k, ref=a.ref, n=len(ydm),
                             r2dm=pt[k], r2dm_ref=pt[a.ref], diff=pt[k] - pt[a.ref],
                             ci_lo=lo, ci_hi=hi, significant=bool(lo > 0 or hi < 0)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "demeaned_krr_paired.csv", index=False)
    pd.set_option("display.width", 250)
    R["cell"] = [f"{d:+.4f} [{l:+.4f},{h:+.4f}]{'*' if s else ''}"
                 for d, l, h, s in zip(R["diff"], R.ci_lo, R.ci_hi, R.significant)]
    print(f"\n== R2 on the DEMEANED target: kernel minus {a.ref} (* = 95% paired CI excludes 0)")
    print(R.pivot_table(index=["layers", "kernel"], columns=["target", "fraction"],
                        values="cell", aggfunc="first").to_string())
    print("\n== absolute R2 on the demeaned target (last row per layer = the linear reference)")
    print(R.pivot_table(index=["layers", "kernel"], columns=["target", "fraction"],
                        values="r2dm").round(4).to_string())
    print(R.groupby(["layers", "target", "fraction"]).r2dm_ref.first().round(4).to_string())
    print(f"\n  wrote {OUT/'demeaned_krr_paired.csv'}")


if __name__ == "__main__":
    main()
