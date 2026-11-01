#!/usr/bin/env python3
"""Shared harness for idea 2 (close the MLP/ridge gap). NO new shared code: imports sweep.py.

Everything here is read-only w.r.t. the rest of the repo. Outputs go to
runs/goal2/gpqa25/sweep/_mlpgap/.

Protocol copied exactly from sweep.py so results are comparable with ref_fixed:
  * rows        = load_meta(act_dir, run_dir) inner joins, one fraction at a time, NaN target dropped
  * outer folds = fixed_fold_of(run_dir, 5, 0) by question (40/40/40/39/39)
  * inner folds = _group_folds(n_tr, g_tr, 3) on the training part only
  * features    = Act(M, cfg, [21]).raw  (1-indexed -> stored index 20), standardised on train rows
  * score       = pooled out-of-fold R2 (_r2) for reg targets, accuracy for binary
  * ridge grid  = np.logspace(0, 6, 13)

API
  D = Data()                       # loads M once
  F = D.folds(fraction, target)    # list[Fold]; .Xtr/.Xte standardised, .inners prepared
  Fold.ridge_path(alpha)           # coef, intercept from one cached SVD
  write(rows, preds, tag)          # -> _mlpgap/results.csv, _mlpgap/preds.csv (append, dedup on merge)
"""
from __future__ import annotations

import builtins, functools, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.sweep import load_meta, Act, fixed_fold_of, _fixed_folds, _group_folds, KEYS  # noqa: E402
from probing.probe_checkpoints import _r2  # noqa: E402

print = functools.partial(builtins.print, flush=True)

REPO = Path(__file__).resolve().parents[2]
ACT_DIR = REPO / "runs/goal2/gpqa25/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B"
RUN_DIR = REPO / "runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B"
SWEEP = REPO / "runs/goal2/gpqa25/sweep"
OUT = SWEEP / "_mlpgap"
ALPHAS = np.logspace(0, 6, 13)
REG = ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std"]
BIN = ["committed_by_here"]
TARGETS = REG + BIN
FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]


def _std(A_tr, A_te):
    mu, sd = A_tr.mean(0), A_tr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (A_tr - mu) / sd, (A_te - mu) / sd


class Split:
    """One train/test split with standardised inputs and a cached SVD for the ridge path."""

    def __init__(self, tr, te, A, y, g):
        self.tr, self.te = tr, te
        self.Xtr, self.Xte = _std(A[tr], A[te])
        self.ytr, self.yte = y[tr], y[te]
        self.gtr, self.gte = g[tr], g[te]
        self._svd = None

    @property
    def svd(self):
        if self._svd is None:
            t0 = time.time()
            self._svd = np.linalg.svd(self.Xtr.astype(np.float64), full_matrices=False)
            print(f"      svd {self.Xtr.shape} {time.time()-t0:.0f}s")
        return self._svd

    def ridge_coef(self, alpha, y=None):
        """coef, intercept of ridge(alpha) on (Xtr, y or ytr). Matches RidgePath / sklearn Ridge."""
        U, s, Vt = self.svd
        yy = self.ytr if y is None else y
        yb = float(yy.mean())
        w = Vt.T @ (s / (s ** 2 + alpha) * (U.T @ (yy - yb)))
        return w, yb

    def ridge_pred(self, alpha, y=None):
        w, b = self.ridge_coef(alpha, y)
        return self.Xtr @ w + b, self.Xte @ w + b

    def eff_dof(self, alpha):
        s = self.svd[1]
        return float(np.sum(s ** 2 / (s ** 2 + alpha)))

    def choose_alpha(self, alphas=ALPHAS, task="reg"):
        """alpha picked on the inner 3 grouped folds of the training part (sum of inner scores)."""
        sc = np.zeros(len(alphas))
        for inn in self.inners:
            U, s, Vt = inn.svd
            yb = float(inn.ytr.mean())
            Uty = U.T @ (inn.ytr - yb)
            for j, a in enumerate(alphas):
                p = inn.Xte @ (Vt.T @ (s / (s ** 2 + a) * Uty)) + yb
                sc[j] += (_r2(inn.yte, p) if task == "reg"
                          else float(((p >= 0.5).astype(int) == inn.yte.astype(int)).mean()))
        return float(alphas[int(np.argmax(sc))])


class Data:
    def __init__(self, max_questions=None, layers=(21,)):
        """layers: 1-indexed decoder layers, as sweep.py's Act means them (21 -> stored index 20)."""
        self.M = load_meta(ACT_DIR, RUN_DIR)
        if max_questions:
            keep = sorted(self.M.example_id.unique())[:max_questions]
            self.M = self.M[self.M.example_id.isin(keep)].reset_index(drop=True)
        self.fold_of = fixed_fold_of(RUN_DIR, 5, 0)
        self.layers = tuple(layers)
        self.feat = Act(self.M, {"act_dir": str(ACT_DIR)}, list(layers))
        self.feat.pca = None
        print(f"  {len(self.M)} rows, {self.M.example_id.nunique()} questions, "
              f"layers {self.layers}, act {self.feat.raw.shape}")

    def rows_for(self, fraction, target):
        Mf = np.where(np.isclose(self.M.fraction.to_numpy(), fraction))[0]
        ya = self.M[target].to_numpy(float)[Mf]
        return Mf[~np.isnan(ya)]

    def splits(self, fraction, target, n_inner=3, inner=True, train_questions=0, seed=0):
        """train_questions > 0: keep only that many TRAINING questions per outer fold (the test
        fold is untouched), for the gap-vs-training-size curve."""
        rows = self.rows_for(fraction, target)
        y = self.M[target].to_numpy(float)[rows]
        g = self.M.example_id.to_numpy()[rows]
        A = self.feat.raw[rows]
        out = []
        for tr, te in _fixed_folds(g, self.fold_of, 5):
            if train_questions:
                u = np.unique(g[tr])
                keep = np.random.default_rng(seed).choice(u, min(train_questions, len(u)),
                                                          replace=False)
                tr = tr[np.isin(g[tr], keep)]
            sp = Split(tr, te, A, y, g)
            sp.inners = []
            if inner:
                for itr, ite in _group_folds(len(tr), g[tr], n_inner):
                    ii = Split(itr, ite, A[tr], y[tr], g[tr])
                    ii.inners = []
                    sp.inners.append(ii)
            out.append(sp)
        self.last_rows, self.last_y, self.last_g = rows, y, g
        return out

    def key_frame(self, rows):
        return self.M.iloc[rows][KEYS].reset_index(drop=True)


def score(task, y, p):
    return _r2(y, p) if task == "reg" else float(((p >= 0.5).astype(int) == y.astype(int)).mean())


def task_of(target):
    return "reg" if target in REG else "bin"


# ---- output -------------------------------------------------------------------

def write(rows, preds, tag):
    OUT.mkdir(parents=True, exist_ok=True)
    rp, pp = OUT / f"results.{tag}.csv", OUT / f"preds.{tag}.csv"
    pd.DataFrame(rows).to_csv(rp, index=False)
    if preds is not None and len(preds):
        pd.concat(preds, ignore_index=True).to_csv(pp, index=False)
    print(f"  wrote {rp}" + (f" and {pp}" if preds is not None and len(preds) else ""))


def pred_frame(D, rows, variant, target, fraction, y, oof):
    f = D.key_frame(rows)
    f.insert(0, "target", target)
    f.insert(0, "variant", variant)
    f["fraction"] = np.round(fraction, 3)
    f["y_true"], f["y_pred"] = y, oof
    return f


# ---- reference ridge ----------------------------------------------------------

def ref_preds():
    """ref_fixed out-of-fold predictions for act_L21 (ridge for reg, logistic for binary)."""
    P = pd.concat([pd.read_csv(f) for f in sorted((SWEEP / "ref_fixed").glob("preds*.csv"))])
    P = P[P.feature == "act_L21"].copy()
    P["fraction"] = P.fraction.round(3)
    return P.drop_duplicates(["probe", "target"] + KEYS)
