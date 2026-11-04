#!/usr/bin/env python3
"""Raw and log1p (and winsorised) length targets for the four poolings (round 7, item 7).

RESEARCH.md's convention is to quote BOTH scales for `rem_len_mean` and
`rem_len_std`, and the `_loglen` work of 2026-11-02 showed the raw scale is tail
dominated on GlobalOpinionQA and that the "largely token counting" caveat is a
raw-scale artefact.  So the pooling comparison has to be quoted on both scales
too.  Same probes, same alpha grid, same fixed folds, target transformed only.

Modelled on `src/probing/loglen_refit.py` and reusing sweep.py's own helpers, so
sweep.py is untouched.  The `raw` rows are a control: they must reproduce the
`ridge_<variant>` sweep for the same feature.

    $PY src/probing/r7_loglen.py --tag gpqa25 --tok-sub activations_tokens_pilot5 \
        --out runs/goal2/gpqa25/sweep/r7_pooling/loglen_pilot5 [--states DIR]
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.sweep import (Feature, Ridge, Views, run_combo, fixed_fold_of,      # noqa: E402
                           _fixed_folds)
from probing.attn_probe import rows_frame, paths, KEYS                           # noqa: E402

TARGETS = ["rem_len_mean", "rem_len_std"]
TRANSFORMS = ["raw", "log1p", "winsor99"]
FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]
HPS = [dict(alpha=a) for a in np.logspace(0, 6, 13)]
B, SEED = 2000, 0


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def boot(y, p, eid, W, qi):
    i = np.array([qi[int(e)] for e in eid])
    a = lambda v: np.bincount(i, weights=v, minlength=W.shape[1])
    n, s1, s2, se = a(np.ones(len(y))), a(y), a(y * y), a((p - y) ** 2)
    return 1 - (W @ se) / ((W @ s2) - (W @ s1) ** 2 / (W @ n))


def transform(y0, tr, train_mask=None):
    if tr == "raw":
        return y0
    if tr == "log1p":
        return np.log1p(y0)
    cut = np.percentile(y0[train_mask], 99)        # clip point from TRAINING rows only
    return np.minimum(y0, cut)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tok-sub", default="activations_tokens")
    ap.add_argument("--pooled-dir", default=None)
    ap.add_argument("--states", default=None, help="attention-pooled states dir (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fracs", nargs="+", type=float, default=FRACS)
    a = ap.parse_args()

    P = paths(a.tag, a.tok_sub)
    pooled = Path(a.pooled_dir or P["tok_dir"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    M = rows_frame(a.tag, a.tok_sub)
    fold_of = fixed_fold_of(P["run_dir"], 5, 0)
    print(f"{a.tag}: {len(M):,} rows, {M.example_id.nunique()} questions")

    K = pd.read_parquet(pooled / "pooled_keys.parquet")
    K["fraction"] = K.fraction.round(6)
    order = (K.reset_index().set_index(KEYS)
             .loc[list(zip(M.example_id, M.cot_id, M.fraction)), "index"].to_numpy())
    X = {p: np.load(pooled / f"pooled_{p}.npy", mmap_mode="r")[order]
         for p in ("last", "mean", "max")}

    # how heavy is the tail on each scale (the reason this script exists)
    tail = []
    for frac in a.fracs:
        m = np.isclose(M.fraction.to_numpy(), frac)
        for t in TARGETS:
            y = M[t].to_numpy(float)[m]; y = y[~np.isnan(y)]
            for tr in TRANSFORMS:
                z = transform(y, tr, np.ones(len(y), bool))
                ss = (z - z.mean()) ** 2; o = np.sort(ss)[::-1]
                tail.append(dict(fraction=frac, target=t, transform=tr, n=len(z),
                                 top10_share_of_ss=o[:10].sum() / ss.sum(),
                                 top50_share_of_ss=o[:50].sum() / ss.sum()))
    pd.DataFrame(tail).to_csv(out / "tail_mass.csv", index=False)

    res, preds = [], []
    for frac in a.fracs:
        rows = np.where(np.isclose(M.fraction.to_numpy(), frac)
                        & M[TARGETS].notna().all(axis=1).to_numpy())[0]
        g = M.example_id.to_numpy()[rows]
        outer = _fixed_folds(g, fold_of, 5)
        jobs = [(p, None) for p in ("last", "mean", "max")]
        if a.states:
            jobs.append(("attn", Path(a.states)))
        for pool, sdir in jobs:
            if sdir is None:
                views = Views(Feature(), np.asarray(X[pool][rows], np.float32), g, outer)
            for t in TARGETS:
                y0 = M[t].to_numpy(float)[rows]
                for tr in TRANSFORMS:
                    t0 = time.time()
                    y = y0.copy(); oof = np.full(len(rows), np.nan)
                    if sdir is None and tr != "winsor99":
                        y = transform(y0, tr)
                        oof, _, ch = run_combo(views, Ridge, HPS, y, g, "reg")
                    elif sdir is None:
                        for tri, te in outer:
                            m = np.zeros(len(rows), bool); m[tri] = True
                            yy = transform(y0, tr, m)
                            o, _, _ = run_combo(views, Ridge, HPS, yy, g, "reg")
                            oof[te] = o[te]; y[te] = yy[te]
                        ch = "per-fold clip"
                    else:
                        # attention pooling: fold-specific features, seeds averaged
                        acc = np.zeros(len(rows)); cnt = np.zeros(len(rows))
                        fs = sorted(sdir.glob(f"states.frac{frac}.fold*.{t}.npz")) or \
                             sorted(sdir.glob(f"states.frac{frac}.fold*.rem_len_mean.npz"))
                        if not fs:
                            print(f"  {frac} attn {t}: no states, skipped"); continue
                        for f in fs:
                            z = np.load(f, allow_pickle=True)
                            srow, Xa = z["rows"], z["X"].astype(np.float32)
                            fold = int(z["fold"])
                            pos = {int(v): i for i, v in enumerate(srow)}
                            sel = np.array([pos[int(v)] for v in rows])
                            te = np.where(np.array([fold_of[int(q)] == fold for q in g]))[0]
                            tri = np.setdiff1d(np.arange(len(rows)), te)
                            m = np.zeros(len(rows), bool); m[tri] = True
                            yy = transform(y0, tr, m)
                            v = Views(Feature(), Xa[sel], g, [(tri, te)])
                            o, _, ch = run_combo(v, Ridge, HPS, yy, g, "reg")
                            acc[te] += o[te]; cnt[te] += 1; y[te] = yy[te]
                        oof = acc / np.maximum(cnt, 1)
                        oof[cnt == 0] = np.nan
                    ok = ~np.isnan(oof)
                    res.append(dict(feature=f"tok_L21_{pool}", probe="ridge", target=t,
                                    transform=tr, fraction=frac, n=int(ok.sum()),
                                    test=r2(y[ok], oof[ok]), hp=str(ch),
                                    runtime_s=round(time.time() - t0, 1)))
                    q = M.iloc[rows[ok]][KEYS].copy()
                    q.insert(0, "transform", tr); q.insert(0, "target", t)
                    q.insert(0, "probe", "ridge"); q.insert(0, "feature", f"tok_L21_{pool}")
                    q["y_true"], q["y_pred"] = y[ok], oof[ok]
                    preds.append(q)
                    print(f"  {frac} {pool:<5}{t:<14}{tr:<9} R2 {res[-1]['test']:+.4f} "
                          f"({res[-1]['runtime_s']}s)", flush=True)
            if sdir is None:
                del views
    R = pd.DataFrame(res); PR = pd.concat(preds, ignore_index=True)
    R.to_csv(out / "results.csv", index=False); PR.to_csv(out / "preds.csv", index=False)

    # paired question bootstrap, mean over cuts, against mean pool and last token
    qs = np.sort(PR.example_id.unique()); Q = len(qs)
    qi = {int(v): i for i, v in enumerate(qs)}
    W = np.random.default_rng(SEED).multinomial(Q, np.ones(Q) / Q, size=B).astype(float)
    feats = sorted(PR.feature.unique())
    ci = []
    for t in TARGETS:
        for tr in TRANSFORMS:
            dr, pt = {}, {}
            for f in feats:
                ds, ps = [], []
                for frac in a.fracs:
                    x = PR[(PR.target == t) & (PR["transform"] == tr)
                           & (PR.feature == f) & np.isclose(PR.fraction, frac)]
                    if not len(x):
                        break
                    ds.append(boot(x.y_true.to_numpy(), x.y_pred.to_numpy(),
                                   x.example_id.to_numpy(), W, qi))
                    ps.append(r2(x.y_true.to_numpy(), x.y_pred.to_numpy()))
                if len(ds) == len(a.fracs):
                    dr[f], pt[f] = np.mean(ds, 0), float(np.mean(ps))
            for f in dr:
                for ref in ("tok_L21_mean", "tok_L21_last"):
                    if ref not in dr or ref == f:
                        continue
                    d = dr[f] - dr[ref]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    ci.append(dict(target=t, transform=tr, feature=f, minus=ref,
                                   score=pt[f], ref_score=pt[ref], diff=pt[f] - pt[ref],
                                   ci_lo=lo, ci_hi=hi,
                                   significant=bool(lo > 0 or hi < 0)))
                lo, hi = np.percentile(dr[f], [2.5, 97.5])
                ci.append(dict(target=t, transform=tr, feature=f, minus="",
                               score=pt[f], ref_score=np.nan, diff=np.nan,
                               ci_lo=lo, ci_hi=hi, significant=False))
    pd.DataFrame(ci).to_csv(out / "ci.csv", index=False)
    print(R.pivot_table(index=["target", "transform"], columns="feature",
                        values="test").round(4).to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
