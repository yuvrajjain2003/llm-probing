#!/usr/bin/env python3
"""Idea 1, part D: does the redundancy matter? Refits ridge/logistic on act_L21 with fewer training rows.
USAGE  $PY src/analysis/redundancy_curves.py --part d1|d2 --fraction 0.5 [--max-questions N] [--seeds 3]
Same rows, fixed folds, alpha grid logspace(0,6,13), inner 3-fold CV as sweep.py (run_combo / Views).
The TEST rows of each outer fold are always ALL of its rows; only the training rows are subsampled.
 d1  m in {1,2,3,5,10,25} CoTs per question (all train questions kept), seeds 0..S-1 (m=25: once)
 d2  matched total rows, split differently: (questions q, CoTs m per question) with q*m ~ N
Writes runs/goal2/gpqa25/sweep/_redundancy/<part>_f<fraction>.csv (one row per config x seed x target; pooled test score
over the 5 outer folds, plus n_train_rows). ridge for the 4 regression targets, logistic (C in 1e-5..1e-3) for committed_by_here.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.sweep import (load_meta, Act, fixed_fold_of, _fixed_folds, Views, run_combo, score, PROBES, expand)

ACT = Path("runs/goal2/gpqa25/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
RUN = Path("runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
OUT = Path("runs/goal2/gpqa25/sweep/_redundancy")
REG = ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std"]
D1_M = [1, 2, 3, 5, 10, 25]
D2 = [(750, [(30, 25), (75, 10), (150, 5)]), (300, [(12, 25), (30, 10), (60, 5), (150, 2)])]   # (N, [(questions, CoTs)])

ap = argparse.ArgumentParser()
ap.add_argument("--part", required=True, choices=["d1", "d2"]); ap.add_argument("--fraction", type=float, required=True)
ap.add_argument("--max-questions", type=int); ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--tag", default="")
a = ap.parse_args()

M = load_meta(ACT, RUN)
if a.max_questions:
    M = M[M.example_id.isin(sorted(M.example_id.unique())[:a.max_questions])].reset_index(drop=True)
fold_of = fixed_fold_of(RUN, 5, 0)
Mf = np.where(np.isclose(M.fraction.to_numpy(), a.fraction))[0]
ok = ~np.isnan(M.answer_entropy_bits_mm.to_numpy(float)[Mf])        # same mask as ref_fixed (4 reg targets share it)
rows_reg = Mf[ok]; rows_bin = Mf                                    # committed_by_here has no NaN
feat = Act(M, {"act_dir": str(ACT)}, [21]); feat.pca = None
ridge, logi = PROBES["ridge"], PROBES["logistic"]
ridge_hps = expand({"alpha": {"logspace": [0, 6, 13]}}, None)
logi_hps = expand({"C": {"logspace": [-5, -3, 3]}}, None)


def subsample(g, outer, rng, q, m):
    """outer: list of (tr, te). Keep q random train questions (None = all) and m random CoTs of each."""
    out = []
    for tr, te in outer:
        gt = g[tr]; qs = np.unique(gt)
        if q is not None and q < len(qs):
            qs = rng.choice(qs, q, replace=False)
        keep = []
        for u in qs:
            ix = tr[gt == u]
            keep.append(ix if m >= len(ix) else rng.choice(ix, m, replace=False))
        out.append((np.concatenate(keep), te))
    return out


def run(cfg_name, q, m, seed, rows, targets, cls, hps, task):
    g = M.example_id.to_numpy()[rows]
    outer = subsample(g, _fixed_folds(g, fold_of, 5), np.random.default_rng(1000 * seed + 7), q, m)
    ntr = int(np.mean([len(t) for t, _ in outer]))
    views = Views(feat, feat.raw[rows], g, outer)
    res = []
    for t in targets:
        y = M[t].to_numpy(float)[rows]; yi = y.astype(int) if task == "bin" else y
        t0 = time.time()
        # run_combo scores only on outer test rows, which we want to be the FULL held-out fold
        oof, tr_s, chosen = run_combo(views, cls, hps, yi, g, task)
        res.append(dict(part=a.part, fraction=a.fraction, config=cfg_name, q_train=q if q else -1, m_per_q=m, seed=seed,
                        target=t, n_train_rows=ntr, metric="acc" if task == "bin" else "R2",
                        test=score(task, yi, oof), train=tr_s, hp=str(chosen), runtime_s=round(time.time() - t0, 1)))
        print(f"  {cfg_name} seed{seed} {t:<24}{res[-1]['metric']} {res[-1]['test']:.4f}  ntrain~{ntr}", flush=True)
    return res


def both(cfg_name, q, m, seed):
    return (run(cfg_name, q, m, seed, rows_reg, REG, ridge, ridge_hps, "reg") +
            run(cfg_name, q, m, seed, rows_bin, ["committed_by_here"], logi, logi_hps, "bin"))


out = OUT / f"{a.part}_f{a.fraction}{a.tag}.csv"; allr = []
if a.part == "d1":
    for m in D1_M:
        for s in ([0] if m == 25 else range(a.seeds)):
            allr += both(f"m{m}", None, m, s)
            pd.DataFrame(allr).to_csv(out, index=False)
else:
    for N, cfgs in D2:
        for q, m in cfgs:
            for s in range(5):
                allr += both(f"N{N}_q{q}_m{m}", q, m, s)
                pd.DataFrame(allr).to_csv(out, index=False)
print("done ->", out)
