#!/usr/bin/env python3
"""Log and winsorised versions of the two length targets, for GlobalOpinionQA (P1's blocking caveat).

P1 measured that on GlobalOpinionQA 10 rows of 12,736 carry 55.6% of `rem_len_mean`'s total sum of
squares (MMLU-Pro's comparable figure is 2.8%), so a raw-scale R2 there is largely a measure of how
well a handful of extreme rows are fitted. RESEARCH.md's own convention is to report raw AND log
scales for length. This refits the SAME probes with the SAME grids and the SAME fixed folds on
transformed targets, so the log-scale numbers become quotable. The raw run is untouched and stays the
comparable-with-GPQA number: this is an addition, not a replacement.

Transforms (applied to the target only, never to the features):
    raw        the published target, refitted here as a control that must reproduce ref_fixed
    log1p      log(1 + y); y is a token count, so y >= 0 and log1p is defined everywhere
    winsor99   y clipped at its 99th percentile WITHIN the cut, computed on the TRAINING rows of
               each outer fold only, so no test-row information leaks into the clip point

Reuses sweep.py's own helpers (load_meta / Act / PrefixLen / Tfidf / fixed_fold_of / Views /
run_combo), the pattern src/probing/newds_ktier.py and r3_rf_diagnosis.py use, so sweep.py itself is
untouched and the GPQA reference cannot move.

    ./envs/goal1/bin/python src/probing/loglen_refit.py --tag globalop_v1 --ds global_opinions
Outputs: runs/goal2/<tag>/sweep/_loglen/{results.csv, preds.csv, tail_mass.csv, ci.csv}
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.sweep import (load_meta, Act, PrefixLen, Tfidf, fixed_fold_of, _fixed_folds,  # noqa
                           Views, run_combo, expand, PROBES)

KEYS = ["example_id", "cot_id", "fraction"]
TARGETS = ["rem_len_mean", "rem_len_std"]
TRANSFORMS = ["raw", "log1p", "winsor99"]
FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]
B, SEED = 2000, 0
# exactly the configs/sweep_ref_<tag>.yaml feature/probe pairs for the regression targets
SPEC = [("act_L21", "ridge_act", ("ridge", {"alpha": {"logspace": [0, 6, 13]}})),
        ("tfidf", "ridge_text", ("ridge", {"alpha": {"logspace": [-3, 3, 7]}})),
        ("prefix_len", "ridge_len", ("ridge", {"alpha": {"logspace": [-6, 3, 10]}}))]


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def boot(y, p, eid, W, qi):
    """Question-bootstrap pooled R2 draws (same algebra as newds_ktier.pooled_draws)."""
    idx = np.array([qi[int(e)] for e in eid])
    a = lambda v: np.bincount(idx, weights=v, minlength=W.shape[1])
    n, s1, s2, se = a(np.ones(len(y))), a(y), a(y * y), a((p - y) ** 2)
    return 1 - (W @ se) / ((W @ s2) - (W @ s1) ** 2 / (W @ n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="globalop_v1")
    ap.add_argument("--ds", default="global_opinions")
    ap.add_argument("--model", default="DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fracs", nargs="+", type=float, default=FRACS)
    a = ap.parse_args()
    S = Path(f"runs/goal2/{a.tag}/sweep")
    out = Path(a.out) if a.out else S / "_loglen"
    out.mkdir(parents=True, exist_ok=True)
    act = Path(f"runs/goal2/{a.tag}/activations_ckpt/{a.ds}/{a.model}")
    run = Path(f"runs/goal1/{a.tag}/{a.ds}/{a.model}")

    M = load_meta(act, run)
    M["fraction"] = M.fraction.round(3)
    fold_of = fixed_fold_of(run, 5, 0)
    print(f"{a.tag}: {len(M)} rows, {M.example_id.nunique()} questions")

    # ---- how heavy is the tail, per cut and target, on each scale
    tail = []
    for frac in a.fracs:
        for t in TARGETS:
            y = M[t].to_numpy(float)[np.isclose(M.fraction.to_numpy(), frac)]
            y = y[~np.isnan(y)]
            for tr in TRANSFORMS:
                z = (y if tr == "raw" else np.log1p(y) if tr == "log1p"
                     else np.minimum(y, np.percentile(y, 99)))
                ss = (z - z.mean()) ** 2
                o = np.sort(ss)[::-1]
                tail.append(dict(fraction=frac, target=t, transform=tr, n=len(z),
                                 mean=z.mean(), sd=z.std(), max=z.max(),
                                 top10_share_of_ss=o[:10].sum() / ss.sum(),
                                 top50_share_of_ss=o[:50].sum() / ss.sum()))
    T = pd.DataFrame(tail)
    T.to_csv(out / "tail_mass.csv", index=False)
    print("\n== share of the target's total sum of squares carried by its 10 most extreme rows ==")
    print(T.pivot_table(index=["target", "fraction"], columns="transform",
                        values="top10_share_of_ss").round(4).to_string())

    # ---- refit
    res, preds = [], []
    for frac in a.fracs:
        rows = np.where(np.isclose(M.fraction.to_numpy(), frac)
                        & M[TARGETS].notna().all(axis=1).to_numpy())[0]
        g = M.example_id.to_numpy()[rows]
        outer = _fixed_folds(g, fold_of, 5)
        for fname, pname, (cls_name, grid) in SPEC:
            feat = {"act_L21": lambda: Act(M, {"act_dir": str(act)}, layers=[21]),
                    "prefix_len": lambda: PrefixLen(M, {}),
                    "tfidf": lambda: Tfidf(M, {})}[fname]()
            views = Views(feat, feat.raw[rows], g, outer)
            for t in TARGETS:
                y0 = M[t].to_numpy(float)[rows]
                for tr in TRANSFORMS:
                    t0 = time.time()
                    if tr == "winsor99":
                        # clip point from TRAINING rows only, per outer fold
                        y = y0.copy()
                        oof = np.full(len(rows), np.nan)
                        for tr_i, te in outer:          # outer is [(train_idx, test_idx), ...]
                            m = np.zeros(len(rows), bool); m[tr_i] = True
                            cut = np.percentile(y0[m], 99)
                            yy = np.minimum(y0, cut)
                            o, _, ch = run_combo(views, PROBES[cls_name],
                                                 expand(grid, None), yy, g, "reg")
                            oof[te] = o[te]
                            y[te] = yy[te]
                        tr_s, chosen = np.nan, "per-fold clip"
                    else:
                        y = y0 if tr == "raw" else np.log1p(y0)
                        oof, tr_s, ch = run_combo(views, PROBES[cls_name], expand(grid, None),
                                                  y, g, "reg")
                        chosen = str(ch)
                    res.append(dict(feature=fname, probe=pname, target=t, transform=tr,
                                    fraction=frac, n=len(rows), test=r2(y, oof), train=tr_s,
                                    hp=chosen, runtime_s=round(time.time() - t0, 1)))
                    p = M.iloc[rows][KEYS].copy()
                    p.insert(0, "transform", tr); p.insert(0, "target", t)
                    p.insert(0, "probe", pname); p.insert(0, "feature", fname)
                    p["y_true"], p["y_pred"] = y, oof
                    preds.append(p)
                    print(f"  {frac} {fname:<11}{t:<14}{tr:<9} R2 {res[-1]['test']:+.4f} "
                          f"({res[-1]['runtime_s']}s)")
            del views, feat
    R = pd.DataFrame(res); P = pd.concat(preds, ignore_index=True)
    R.to_csv(out / "results.csv", index=False); P.to_csv(out / "preds.csv", index=False)

    # ---- CIs, paired against our probe, mean over cuts
    ci = []
    draws, point = {}, {}
    for t in TARGETS:
        for tr in TRANSFORMS:
            for frac in a.fracs:
                x = P[(P["target"] == t) & (P["transform"] == tr)
                      & np.isclose(P["fraction"], frac)]   # P.transform is DataFrame.transform
                eid = x[x.feature == "act_L21"].example_id.to_numpy()
                qs = np.sort(np.unique(eid)); qi = {int(v): i for i, v in enumerate(qs)}
                W = np.random.default_rng(SEED).multinomial(
                    len(qs), np.ones(len(qs)) / len(qs), size=B).astype(float)
                for fname, _, _ in SPEC:
                    d = x[x.feature == fname]
                    assert (d.example_id.to_numpy() == eid).all()
                    draws[(fname, t, tr, frac)] = boot(d.y_true.to_numpy(float),
                                                       d.y_pred.to_numpy(float), eid, W, qi)
                    point[(fname, t, tr, frac)] = r2(d.y_true.to_numpy(float),
                                                     d.y_pred.to_numpy(float))
            for fname, _, _ in SPEC:
                d = np.mean([draws[(fname, t, tr, f)] for f in a.fracs], 0)
                v = float(np.mean([point[(fname, t, tr, f)] for f in a.fracs]))
                row = dict(feature=fname, target=t, transform=tr, score=v,
                           ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5))
                if fname != "act_L21":
                    b = np.mean([draws[("act_L21", t, tr, f)] for f in a.fracs], 0)
                    lo, hi = np.percentile(d - b, [2.5, 97.5])
                    row.update(diff_vs_ours=v - float(np.mean(
                        [point[("act_L21", t, tr, f)] for f in a.fracs])),
                        diff_lo=lo, diff_hi=hi, significant=bool(lo > 0 or hi < 0))
                ci.append(row)
    C = pd.DataFrame(ci)
    C.to_csv(out / "ci.csv", index=False)
    print("\n== mean over cuts, R2 by scale (95% question-bootstrap CI; diff is vs our probe) ==")
    print(C.round(4).to_string(index=False))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
