#!/usr/bin/env python3
"""Re-baseline: is the MLP/kernel gain beating ridge, or beating a MIS-TUNED ridge?

Per-fold inner-CV alpha selection is unstable: in 6 of 20 (target, checkpoint) cells the five
outer folds disagree about alpha, and in the worst cell (rem_len_mean at 50%) that costs 0.0071
pooled R2 against the best single fixed alpha. Since the MLP's gain at that same cell is +0.0045,
the pooled MLP result has to be re-measured against a ridge that is not mistuned.

Baselines produced here (all on the SAME rows and fixed folds as ref_fixed):
  ridge_perfold   alpha by inner 3-fold grouped CV per outer fold -- the project's reference
  ridge_1se       same inner curves, but the 1-SE rule: the LARGEST alpha whose mean inner score
                  is within 1 standard error of the best. Honest (training rows only) and much
                  more stable, because it stops chasing noise in the inner-CV curve.
  ridge_oracle    one fixed alpha per cell, chosen to maximise POOLED TEST R2.
                  *** ORACLE / BEST-ON-TEST -- an upper bound on what any alpha tuning could do,
                  NOT an honest estimator. Labelled as such everywhere. ***

USAGE  ./envs/goal1/bin/python src/probing/mlpgap_rebase.py --fractions 0.1,0.3,0.5,0.7,0.9 \
           --targets rem_len_mean
OUT    runs/goal2/gpqa25/sweep/_mlpgap/{results,preds}.rebase.csv
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.mlpgap_core import (Data, OUT, ALPHAS, REG, score, write, pred_frame,  # noqa: E402
                                 print)
from probing.probe_checkpoints import _r2  # noqa: E402


def inner_curve(sp):
    """(n_alpha,) mean and standard error of inner-fold R2 over sp.inners."""
    S = []
    for inn in sp.inners:
        U, s, Vt = inn.svd
        yb = float(inn.ytr.mean())
        Uty = U.T @ (inn.ytr - yb)
        S.append([_r2(inn.yte, inn.Xte @ (Vt.T @ (s / (s ** 2 + a) * Uty)) + yb) for a in ALPHAS])
    S = np.array(S)
    return S.mean(0), S.std(0, ddof=1) / np.sqrt(len(S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--targets", default="rem_len_mean")
    ap.add_argument("--max-questions", type=int)
    ap.add_argument("--tag", default="rebase")
    a = ap.parse_args()

    D = Data(a.max_questions)
    rows, preds, diag = [], [], []
    for frac in [float(x) for x in a.fractions.split(",")]:
        for t in a.targets.split(","):
            sps = D.splits(frac, t)
            rws = D.last_rows
            y = D.M[t].to_numpy(float)[rws]
            t0 = time.time()
            # per-fold inner curves (one set of SVDs serves every selection rule)
            curves = [inner_curve(sp) for sp in sps]
            oof = {k: np.full(len(y), np.nan) for k in ["ridge_perfold", "ridge_1se"]}
            chosen = {k: [] for k in oof}
            # whole alpha path of out-of-fold predictions, so the oracle costs nothing extra
            path = np.full((len(ALPHAS), len(y)), np.nan)
            for sp, (m, se) in zip(sps, curves):
                for j, al in enumerate(ALPHAS):
                    w, b = sp.ridge_coef(al)
                    path[j, sp.te] = sp.Xte @ w + b
                j0 = int(np.argmax(m))
                ok = np.where(m >= m[j0] - se[j0])[0]
                j1 = int(ok.max())                      # 1-SE rule: strongest admissible shrinkage
                oof["ridge_perfold"][sp.te] = path[j0, sp.te]
                oof["ridge_1se"][sp.te] = path[j1, sp.te]
                chosen["ridge_perfold"].append(float(ALPHAS[j0]))
                chosen["ridge_1se"].append(float(ALPHAS[j1]))
            pooled = np.array([score("reg", y, path[j]) for j in range(len(ALPHAS))])
            jo = int(np.argmax(pooled))
            oof["ridge_oracle"] = path[jo].copy()
            chosen["ridge_oracle"] = [float(ALPHAS[jo])] * 5
            for k in ["ridge_perfold", "ridge_1se", "ridge_oracle"]:
                rows.append(dict(variant=k, fraction=round(frac, 3), target=t, n=len(y),
                                 metric="R2", test=score("reg", y, oof[k]), train=np.nan,
                                 hp=json.dumps(chosen[k]),
                                 oracle=(k == "ridge_oracle"),
                                 runtime_s=round(time.time() - t0, 1)))
                preds.append(pred_frame(D, rws, k, t, frac, y, oof[k]))
                print(f"   {t:<14}f{frac} {k:<16}R2 {rows[-1]['test']:.5f} "
                      f"alphas {['%.3g' % x for x in chosen[k]]}")
            diag.append(dict(target=t, fraction=round(frac, 3),
                             perfold=score("reg", y, oof["ridge_perfold"]),
                             one_se=score("reg", y, oof["ridge_1se"]),
                             oracle=pooled[jo], oracle_alpha=float(ALPHAS[jo]),
                             gain_1se=score("reg", y, oof["ridge_1se"])
                             - score("reg", y, oof["ridge_perfold"]),
                             gain_oracle=pooled[jo] - score("reg", y, oof["ridge_perfold"]),
                             n_distinct_perfold=len(set(chosen["ridge_perfold"]))))
            pd.DataFrame(diag).to_csv(OUT / f"{a.tag}_alpha_rules.csv", index=False)
            write(rows, preds, a.tag)
    d = pd.DataFrame(diag)
    print("\n== alpha-selection rules (pooled out-of-fold R2)")
    print(d.round(5).to_string(index=False))
    print(f"\n  mean gain of the 1-SE rule over per-fold CV: {d.gain_1se.mean():+.5f}")
    print(f"  mean gain of the ORACLE fixed alpha:          {d.gain_oracle.mean():+.5f}")


if __name__ == "__main__":
    main()
