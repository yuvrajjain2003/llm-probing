#!/usr/bin/env python3
"""Explicit low-rank QUADRATIC probe: ridge on [Z, s * pairwise products of the top-K PCA scores].

Why this exists. B2 showed the only real nonlinear signal is on rem_len_mean and that a degree-2
polynomial kernel captures it (+0.011 R2, paired CI [+0.007,+0.015]). A degree-2 kernel's RKHS is
"linear + all pairwise products", which in an MLP is a SQUARE activation,
g(x) = sum_k c_k (v_k.x)^2 -- a quadratic form of rank = hidden size. This script is the
closed-form, rank-K version of exactly that function class, so it answers "how many hidden units
does the square-activation net need?" without any optimisation confound, and it is itself a
candidate winner.

Features per outer fold (everything fitted on the training rows only):
    Z            3,584 standardised activations (the linear part, as ridge sees it)
    P            top-K PCA scores of Z, standardised
    Q            s * {P_i P_j, i <= j}, standardised then scaled -- K(K+1)/2 columns
One alpha for the concatenation, so the block scale `s` is what sets the quadratic block's
effective penalty (alpha/s^2). (alpha, s) are chosen together on the inner 3 grouped folds;
`quad_joint` additionally selects K there. Nothing is selected on test.

USAGE  ./envs/goal1/bin/python src/probing/mlpgap_quad.py --fractions 0.5 --targets rem_len_mean
OUT    runs/goal2/gpqa25/sweep/_mlpgap/{results,preds}.quad<tag>.csv
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.mlpgap_core import (Data, TARGETS, score, task_of, write, pred_frame,  # noqa: E402
                                 print)
from probing.mlpgap_b2 import Basis  # noqa: E402

ALPHAS = np.logspace(1, 7, 13)


def quad_block(Ztr, Zte, K, s):
    """-> augmented (train, test) matrices [Z, s*Q]; PCA and all scalings fitted on train only."""
    Vt = np.linalg.svd(Ztr, full_matrices=False)[2][:K]
    Ptr, Pte = Ztr @ Vt.T, Zte @ Vt.T
    sd = Ptr.std(0); sd[sd < 1e-8] = 1.0
    Ptr, Pte = Ptr / sd, Pte / sd
    i, j = np.triu_indices(K)
    Qtr, Qte = Ptr[:, i] * Ptr[:, j], Pte[:, i] * Pte[:, j]
    mu = Qtr.mean(0); sq = Qtr.std(0); sq[sq < 1e-8] = 1.0
    Qtr, Qte = (Qtr - mu) / sq * s, (Qte - mu) / sq * s
    return (np.concatenate([Ztr, Qtr], 1).astype(np.float32),
            np.concatenate([Zte, Qte], 1).astype(np.float32))


def basis(Xtr, Xte):
    """Ridge over the whole alpha grid via the n x n GRAM, not an SVD of the (n, 11840) augmented
    matrix: a linear-kernel KRR gives identical predictions to primal ridge at the same alpha
    (B2 verified krr_linear == ridge exactly), and costs ~5x less once d_aug > n."""
    A, Bm = Xtr.astype(np.float64), Xte.astype(np.float64)
    return Basis(A @ A.T, Bm @ A.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.5")
    ap.add_argument("--targets", default="rem_len_mean")
    ap.add_argument("--ks", default="32,64,128")
    ap.add_argument("--scales", default="0.1,0.3,1.0")
    ap.add_argument("--max-questions", type=int)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    Ks = [int(x) for x in a.ks.split(",")]
    SS = [float(x) for x in a.scales.split(",")]

    D = Data(a.max_questions)
    rows, preds = [], []
    for frac in [float(x) for x in a.fractions.split(",")]:
        for t in a.targets.split(","):
            task = task_of(t)
            sps = D.splits(frac, t)
            rws = D.last_rows
            y = D.M[t].to_numpy(float)[rws]
            yi = y.astype(int) if task == "bin" else y
            names = ["ridge_qrun"] + [f"quad_K{K}" for K in Ks] + ["quad_joint"]
            oof = {n: np.full(len(y), np.nan) for n in names}
            trs = {n: [] for n in names}
            ch = {n: [] for n in names}
            t0 = time.time()
            for fi, sp in enumerate(sps):
                al = sp.choose_alpha()
                w0, b0 = sp.ridge_coef(al)
                oof["ridge_qrun"][sp.te] = sp.Xte @ w0 + b0
                trs["ridge_qrun"].append(score(task, sp.ytr, sp.Xtr @ w0 + b0))
                ch["ridge_qrun"].append(al)
                # inner scores over (K, s, alpha)
                S = np.zeros((len(Ks), len(SS), len(ALPHAS)))
                for ki, K in enumerate(Ks):
                    for si, s in enumerate(SS):
                        for inn in sp.inners:
                            Atr, Ate = quad_block(inn.Xtr, inn.Xte, K, s)
                            P = basis(Atr, Ate).predict(inn.ytr, ALPHAS)
                            S[ki, si] += [score(task, inn.yte, p) for p in P]
                for ki, K in enumerate(Ks):
                    si, ai = np.unravel_index(int(np.argmax(S[ki])), S[ki].shape)
                    Atr, Ate = quad_block(sp.Xtr, sp.Xte, K, SS[si])
                    bs = basis(Atr, Ate)
                    oof[f"quad_K{K}"][sp.te] = bs.predict(sp.ytr, ALPHAS[ai:ai + 1])[0]
                    trs[f"quad_K{K}"].append(score(task, sp.ytr,
                                                   bs.train_predict(sp.ytr, ALPHAS[ai:ai + 1])[0]))
                    ch[f"quad_K{K}"].append(f"s={SS[si]} a={ALPHAS[ai]:.3g}")
                    del bs
                ki, si, ai = np.unravel_index(int(np.argmax(S)), S.shape)
                Atr, Ate = quad_block(sp.Xtr, sp.Xte, Ks[ki], SS[si])
                oof["quad_joint"][sp.te] = basis(Atr, Ate).predict(sp.ytr, ALPHAS[ai:ai + 1])[0]
                ch["quad_joint"].append(f"K={Ks[ki]} s={SS[si]} a={ALPHAS[ai]:.3g}")
                print(f"   {t} f{fi} joint K={Ks[ki]} s={SS[si]} a={ALPHAS[ai]:.3g} "
                      f"{time.time()-t0:.0f}s")
            for n in names:
                rows.append(dict(variant=n, fraction=round(frac, 3), target=t, n=len(y),
                                 metric="acc" if task == "bin" else "R2",
                                 test=score(task, yi, oof[n]),
                                 train=float(np.mean(trs[n])) if trs[n] else np.nan,
                                 hp=json.dumps(ch[n], default=str),
                                 runtime_s=round(time.time() - t0, 1)))
                preds.append(pred_frame(D, rws, n, t, frac, y, oof[n]))
                print(f"  == {t:<16}{n:<14}{rows[-1]['metric']} test {rows[-1]['test']:.4f} "
                      f"train {rows[-1]['train']:.4f}")
            write(rows, preds, "quad" + a.tag)


if __name__ == "__main__":
    main()
