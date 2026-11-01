#!/usr/bin/env python3
"""B2 -- the CEILING. Closed-form kernel ridge (no optimisation confound) vs linear ridge.

If RBF / poly-2 never beat the LINEAR kernel (which reproduces ridge exactly) there is no
nonlinear signal in these features, and "improve the MLP" tops out at "tie ridge".

Kernels, all on train-standardised features (one squared-distance matrix serves all of them):
  linear    K = <x,x'>                              -> identical predictions to ridge(alpha)
  poly2_c   K = (<x,x'>/d + c)^2,  c in {1,3,10}    -> RKHS strictly contains the linear one
  rbf_s     K = exp(-||x-x'||^2 / (s * med)),  s in {0.25,0.5,1,2,4}, med = median sq. distance
alpha: logspace(-6, 6, 25), chosen on the inner 3 grouped folds (never on test).
Per target we report each kernel's honestly-selected score, the jointly-selected
(kernel, alpha) score ("krr_honest_joint"), and -- in the analysis, labelled -- best-on-test.

Eigendecompositions depend on X only, so they are computed once per (outer fold, kernel) and
reused for every target that shares the row mask.

USAGE  $PY src/probing/mlpgap_b2.py --fractions 0.5 [--targets a,b] [--max-questions N] [--tag T]
OUT    runs/goal2/gpqa25/sweep/_mlpgap/{results,preds}.b2<tag>.csv
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.mlpgap_core import (Data, TARGETS, score, task_of, write, pred_frame,  # noqa: E402
                                 print)

ALPHAS = np.logspace(-6, 6, 25)
POLY_C = (1.0, 3.0, 10.0)
RBF_S = (0.25, 0.5, 1.0, 2.0, 4.0)
KNAMES = ["linear"] + [f"poly2_c{c:g}" for c in POLY_C] + [f"rbf_s{s:g}" for s in RBF_S]


def _gram_d2(Xa, Xb):
    A, B = Xa.astype(np.float64), Xb.astype(np.float64)
    G = A @ B.T
    D2 = np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * G, 0.0)
    return G, D2


def _kern(kn, G, D2, d, med):
    if kn == "linear":
        return G
    if kn.startswith("poly2_c"):
        return (G / d + float(kn[7:])) ** 2
    return np.exp(-D2 / (float(kn[5:]) * med))


class Basis:
    """Eigendecomposition of one training kernel + the test kernel. y-independent, so shared
    by every target. predict(y, alphas) -> (n_alpha, n_te) and train_pred(y, alpha)."""

    def __init__(self, Ktr, Kte):
        lam, V = np.linalg.eigh(Ktr)
        self.lam, self.V, self.Ktr, self.Kte = np.maximum(lam, 0.0), V, Ktr, Kte

    def _co(self, y, alphas):
        yb = float(y.mean())
        Vty = self.V.T @ (y - yb)
        return yb, np.stack([self.V @ (Vty / (self.lam + a)) for a in alphas])

    def predict(self, y, alphas):
        yb, C = self._co(y, alphas)
        return (self.Kte @ C.T).T + yb

    def train_predict(self, y, alphas):
        yb, C = self._co(y, alphas)
        return (self.Ktr @ C.T).T + yb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.5")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--max-questions", type=int)
    ap.add_argument("--kernels", default=",".join(KNAMES))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    fracs = [float(x) for x in a.fractions.split(",")]
    tgts = a.targets.split(",")
    knames = a.kernels.split(",")

    D = Data(a.max_questions)
    rows, preds = [], []
    for frac in fracs:
        gro = {}                                  # identical usable-row masks share the bases
        for t in tgts:
            gro.setdefault(D.rows_for(frac, t).tobytes(), []).append(t)
        for _, ts in gro.items():
            sps = D.splits(frac, ts[0])
            rws = D.last_rows
            Y = {t: D.M[t].to_numpy(float)[rws] for t in ts}
            print(f"== frac {frac} n={len(rws)} targets {ts}")
            oof = {(t, k): np.full(len(rws), np.nan) for t in ts for k in knames + ["honest_joint"]}
            trs = {(t, k): [] for t in ts for k in knames}
            chosen = {(t, k): [] for t in ts for k in knames + ["honest_joint"]}
            for fi, sp in enumerate(sps):
                t0 = time.time()
                d = sp.Xtr.shape[1]
                Gt, D2t = _gram_d2(sp.Xtr, sp.Xtr)
                med = float(np.median(D2t[np.triu_indices_from(D2t, 1)]))
                Ge, D2e = _gram_d2(sp.Xte, sp.Xtr)
                IG = [(_gram_d2(ii.Xtr, ii.Xtr), _gram_d2(ii.Xte, ii.Xtr)) for ii in sp.inners]
                imed = [float(np.median(g[0][1][np.triu_indices_from(g[0][1], 1)])) for g in IG]
                print(f"   fold {fi} grams {time.time()-t0:.0f}s med {med:.0f}")
                for kn in knames:
                    t1 = time.time()
                    B = Basis(_kern(kn, Gt, D2t, d, med), _kern(kn, Ge, D2e, d, med))
                    IB = [Basis(_kern(kn, g[0][0], g[0][1], d, m), _kern(kn, g[1][0], g[1][1], d, m))
                          for g, m in zip(IG, imed)]
                    for t in ts:
                        task = task_of(t)
                        y = Y[t]
                        sc = np.zeros(len(ALPHAS))
                        for ib, ii in zip(IB, sp.inners):
                            yv = y[sp.tr][ii.te]
                            P = ib.predict(y[sp.tr][ii.tr], ALPHAS)
                            sc += [score(task, yv, p) for p in P]
                        j = int(np.argmax(sc))
                        oof[(t, kn)][sp.te] = B.predict(y[sp.tr], ALPHAS[j:j + 1])[0]
                        trs[(t, kn)].append(score(task, y[sp.tr],
                                                  B.train_predict(y[sp.tr], ALPHAS[j:j + 1])[0]))
                        chosen[(t, kn)].append(float(ALPHAS[j]))
                        D.__dict__.setdefault("_sc", {})[(t, fi, kn)] = sc
                    del B, IB
                    print(f"      {kn:<12}{time.time()-t1:.0f}s")
                # jointly selected (kernel, alpha) from the stored inner score curves
                for t in ts:
                    S = np.stack([D._sc[(t, fi, kn)] for kn in knames])
                    ki, ai = np.unravel_index(int(np.argmax(S)), S.shape)
                    kn = knames[ki]
                    B = Basis(_kern(kn, Gt, D2t, d, med), _kern(kn, Ge, D2e, d, med))
                    oof[(t, "honest_joint")][sp.te] = B.predict(Y[t][sp.tr], ALPHAS[ai:ai + 1])[0]
                    chosen[(t, "honest_joint")].append(f"{kn}@{ALPHAS[ai]:.3g}")
                    del B
                del Gt, D2t, Ge, D2e, IG
            for t in ts:
                task = task_of(t)
                yi = Y[t].astype(int) if task == "bin" else Y[t]
                for kn in knames + ["honest_joint"]:
                    v = f"krr_{kn}"
                    rows.append(dict(variant=v, fraction=round(frac, 3), target=t, n=len(rws),
                                     metric="acc" if task == "bin" else "R2",
                                     test=score(task, yi, oof[(t, kn)]),
                                     train=float(np.mean(trs[(t, kn)])) if trs.get((t, kn)) else np.nan,
                                     hp=str(chosen[(t, kn)])))
                    preds.append(pred_frame(D, rws, v, t, frac, Y[t], oof[(t, kn)]))
                    print(f"   {t:<24}{v:<18}{rows[-1]['metric']} test {rows[-1]['test']:.4f} "
                          f"train {rows[-1]['train']:.4f} {chosen[(t, kn)]}")
            write(rows, preds, "b2" + a.tag)


if __name__ == "__main__":
    main()
