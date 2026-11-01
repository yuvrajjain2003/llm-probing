#!/usr/bin/env python3
"""Rung 7: effective degrees of freedom of the chosen ridge fit, df(alpha) = sum s_i^2/(s_i^2+alpha).

Explains the regime in one number: how many of the 3,584 feature directions ridge is actually
using at the alpha its own inner-fold CV picks, out of min(n_train, d) available. Also records the
spectrum shape (share of Gram energy in the top 1/10/100 directions) and, for context, df at the
grid's neighbouring alphas.

USAGE  ./envs/goal1/bin/python src/probing/mlpgap_dof.py [--fractions 0.1,0.3,0.5,0.7,0.9]
OUT    runs/goal2/gpqa25/sweep/_mlpgap/dof.csv
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.mlpgap_core import Data, OUT, ALPHAS, REG, score, print  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--targets", default=",".join(REG))
    ap.add_argument("--max-questions", type=int)
    a = ap.parse_args()
    D = Data(a.max_questions)
    rows, curve = [], []
    for frac in [float(x) for x in a.fractions.split(",")]:
        for t in a.targets.split(","):
            sps = D.splits(frac, t)
            # pooled out-of-fold R2 at each FIXED alpha: how much of any "nonlinear" gain could
            # be alpha-selection slack instead? best-on-test, so labelled as such.
            y = D.M[t].to_numpy(float)[D.last_rows]
            for al_ in ALPHAS:
                o = np.full(len(y), np.nan)
                for sp in sps:
                    w, b = sp.ridge_coef(al_)
                    o[sp.te] = sp.Xte @ w + b
                curve.append(dict(fraction=round(frac, 3), target=t, alpha=al_,
                                  test_pooled=score("reg", y, o)))
            pd.DataFrame(curve).to_csv(OUT / "alpha_curve.csv", index=False)
            for fi, sp in enumerate(sps):
                al = sp.choose_alpha()
                s2 = sp.svd[1] ** 2
                cs = np.cumsum(s2) / s2.sum()
                rows.append(dict(
                    fraction=round(frac, 3), target=t, fold=fi, alpha=al,
                    n_train=len(sp.tr), d=sp.Xtr.shape[1], rank=int(len(s2)),
                    df=sp.eff_dof(al), df_frac=sp.eff_dof(al) / len(s2),
                    df_alpha_lo=sp.eff_dof(al / 10), df_alpha_hi=sp.eff_dof(al * 10),
                    mean_eig=float(s2.mean()), top_eig=float(s2[0]),
                    energy_top1=float(cs[0]), energy_top10=float(cs[9]),
                    energy_top100=float(cs[99])))
                print(f"  {frac} {t:<24}f{fi} alpha {al:.3g} df {rows[-1]['df']:.1f} "
                      f"({100*rows[-1]['df_frac']:.1f}% of {rows[-1]['rank']}) "
                      f"mean_eig {rows[-1]['mean_eig']:.0f}")
            pd.DataFrame(rows).to_csv(OUT / "dof.csv", index=False)
    d = pd.DataFrame(rows)
    print("\n== mean over folds")
    print(d.groupby(["fraction", "target"])[["alpha", "df", "df_frac"]].mean().round(3).to_string())
    c = pd.DataFrame(curve)
    print("\n== pooled test R2 vs FIXED alpha (best-on-test; the CV choice is in dof.csv)")
    print(c.pivot_table(index="alpha", columns=["target", "fraction"], values="test_pooled")
          .round(4).to_string())
    print(f"\n  wrote {OUT/'dof.csv'} and {OUT/'alpha_curve.csv'}")


if __name__ == "__main__":
    main()
