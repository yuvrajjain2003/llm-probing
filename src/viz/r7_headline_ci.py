#!/usr/bin/env python3
"""The headline comparison for round 7 item 7, for whatever cuts have landed.

attention pooling minus MEAN pooling, and minus the last token, on the two
headline targets, with a paired question bootstrap.  The full version over all
five cuts and five targets is `src/viz/r7_pooling_table.py`, which calls the
published `master_table.main()`; this one exists so the comparison can be read
off as soon as the first cut's attention states have been read out, and so that
the number quoted in the report is reproducible by one command.

    $PY src/viz/r7_headline_ci.py --tag gpqa25 --attn-preds DIR --ridge DIR
"""
import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd

B, SEED = 2000, 0
K = ["example_id", "cot_id"]
TARGETS = ["answer_entropy_bits_mm", "rem_len_mean"]
FEATS = {"act_L21": "last", "mean_L21": "mean", "max_L21": "max"}


def r2(y, p):
    return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn-preds", required=True, help="dir holding preds_attn_ridge.csv")
    ap.add_argument("--ridge", required=True, help="dir holding the closed-form preds*.csv")
    a = ap.parse_args()
    A = pd.read_csv(Path(a.attn_preds) / "preds_attn_ridge.csv")
    R = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(str(Path(a.ridge) / "preds*.csv")))])
    A["fraction"] = A.fraction.round(3); R["fraction"] = R.fraction.round(3)
    for frac in sorted(A.fraction.unique()):
        Af, Rf = A[np.isclose(A.fraction, frac)], R[np.isclose(R.fraction, frac)]
        for t in TARGETS:
            w = {"attn": Af[Af.target == t].set_index(K).y_pred}
            for f, lab in FEATS.items():
                w[lab] = Rf[(Rf.target == t) & (Rf.feature == f)].set_index(K).y_pred
            yt = Rf[(Rf.target == t) & (Rf.feature == "act_L21")].set_index(K).y_true
            W = pd.DataFrame(w).dropna()
            if not len(W):
                continue
            yt = yt.loc[W.index].to_numpy()
            qs = np.sort(W.index.get_level_values(0).unique()); Q = len(qs)
            qi = {int(v): i for i, v in enumerate(qs)}
            M = np.random.default_rng(SEED).multinomial(Q, np.ones(Q) / Q, size=B).astype(float)
            idx = np.array([qi[int(e)] for e in W.index.get_level_values(0)])
            agg = lambda v: np.bincount(idx, weights=v, minlength=Q)
            n, s1, s2 = agg(np.ones(len(yt))), agg(yt), agg(yt ** 2)
            dr = {k: 1 - (M @ agg((W[k].to_numpy() - yt) ** 2)) /
                  ((M @ s2) - (M @ s1) ** 2 / (M @ n)) for k in W.columns}
            print(f"=== cut {frac}  {t}  n={len(W)} rows, {Q} questions ===")
            for k in W.columns:
                print(f"    {k:5s} {r2(yt, W[k].to_numpy()):+.4f}")
            for ref in ("mean", "last"):
                d = dr["attn"] - dr[ref]
                lo, hi = np.percentile(d, [2.5, 97.5])
                pt = r2(yt, W["attn"].to_numpy()) - r2(yt, W[ref].to_numpy())
                print(f"    attn minus {ref}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}] "
                      f"{'SIGNIFICANT' if lo > 0 or hi < 0 else 'n.s.'}")


if __name__ == "__main__":
    main()
