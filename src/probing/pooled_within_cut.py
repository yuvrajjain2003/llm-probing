#!/usr/bin/env python3
"""Round 7 item 5: score a POOLED sweep (`pooled: true`) honestly, three ways at once.

A pooled fit is the only way to hand `fraction` to a probe (sweep.py fits one probe per cut, where
it is constant), but a pooled R2 is not comparable to anything in our write-ups: pooled across
depth the trivial prefix_len baseline reaches 0.718 against our probe's 0.716, because depth alone
carries most of the pooled variance. So for every pooled model this reports

  1. the pooled score over all cuts at once (the `fraction`-only model is its floor),
  2. the WITHIN-CUT score recovered from the same model's out-of-fold predictions: split them by
     cut, score inside each cut, average over cuts. That is the project's standard convention and
     is directly comparable to the per-cut reference (ref_fixed),
  3. two paired differences with 95% question bootstraps on identical rows:
     `diff_vs_pooled_base` = this feature minus the same pooled feature without the cut indicator
     (act_L21_frac minus act_L21: the effect of GIVING the probe the checkpoint), and
     `diff_vs_ref` = this feature minus ref_fixed's per-cut fit of the same base feature
     (the cost of pooling at all).

Reg targets are scored by R2, binary ones by accuracy (pooled AUC is reported beside it; it comes
from the shallow `ref_fixed` logistic C grid here). The bootstrap algebra is imported from
loglen_refit, so it is the same code the published CIs used.

    ./envs/goal1/bin/python src/probing/pooled_within_cut.py \
        --dir runs/goal2/gpqa25/sweep/r7_pooled_ckpt \
        --ref runs/goal2/gpqa25/sweep/ref_fixed
Output: <dir>/within_cut.csv
"""
from __future__ import annotations

import argparse, glob, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.loglen_refit import boot, r2                      # noqa: E402
from probing.probe_checkpoints import BIN_TARGETS              # noqa: E402

KEYS = ["example_id", "cot_id", "fraction"]
B, SEED = 2000, 0


def load_preds(d: Path) -> pd.DataFrame:
    fs = sorted(glob.glob(str(d / "preds*.csv")))
    assert fs, f"no preds*.csv under {d}"
    P = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return P.drop_duplicates(["feature", "probe", "target"] + KEYS, keep="last")


def acc(y, p):
    return float(((p >= 0.5).astype(int) == y).mean())


def acc_boot(y, p, eid, W, qi):
    """Question-bootstrap accuracy draws (same resampling weights as `boot`)."""
    idx = np.array([qi[int(e)] for e in eid])
    a = lambda v: np.bincount(idx, weights=v, minlength=W.shape[1])    # noqa: E731
    return (W @ a(((p >= 0.5).astype(int) == y).astype(float))) / (W @ a(np.ones(len(y))))


def per_cut(d, fracs, sc, bf, W, qi, eids):
    """-> (per-cut point scores, per-cut bootstrap draws); checks rows against `eids` if given."""
    pts, drs, keys = [], [], {}
    for fr in fracs:
        x = d[np.isclose(d.fraction, fr)].sort_values(["example_id", "cot_id"])
        y, p, eid = x.y_true.to_numpy(float), x.y_pred.to_numpy(float), x.example_id.to_numpy()
        cot = x.cot_id.to_numpy()
        if eids is not None:
            e0, c0, y0 = eids[fr]
            assert len(eid) == len(e0) and (eid == e0).all() and (cot == c0).all(), \
                f"row mismatch at cut {fr}"
            assert np.allclose(y, y0), f"different y_true at cut {fr}"
        keys[fr] = (eid, cot, y)
        pts.append(sc(y, p)); drs.append(bf(y, p, eid, W, qi))
    return pts, drs, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="pooled sweep out_dir")
    ap.add_argument("--ref", required=True, help="per-cut reference sweep out_dir (ref_fixed)")
    a = ap.parse_args()
    out = Path(a.dir)
    P, R = load_preds(out), load_preds(Path(a.ref))
    fracs = sorted(P.fraction.unique())
    print(f"pooled: {len(P):,} pred rows, {P.example_id.nunique()} questions, cuts {fracs}")

    qs = np.sort(P.example_id.unique()); qi = {int(v): i for i, v in enumerate(qs)}
    W = np.random.default_rng(SEED).multinomial(len(qs), np.ones(len(qs)) / len(qs),
                                                size=B).astype(float)
    pooled_draws, pooled_pts, rows = {}, {}, []
    for (f, p, t), d in P.groupby(["feature", "probe", "target"]):
        bin_t = t in BIN_TARGETS
        sc, bf = (acc, acc_boot) if bin_t else (r2, boot)
        pts, drs, keys = per_cut(d, fracs, sc, bf, W, qi, None)
        pooled_pts[(f, p, t)], pooled_draws[(f, p, t)] = pts, drs
        row = dict(feature=f, probe=p, target=t, metric="acc" if bin_t else "R2", n=len(d),
                   pooled=sc(d.y_true.to_numpy(float), d.y_pred.to_numpy(float)),
                   within_cut_mean=float(np.mean(pts)))
        if bin_t:
            from sklearn.metrics import roc_auc_score
            row["pooled_auc"] = float(roc_auc_score(d.y_true.to_numpy(int),
                                                    d.y_pred.to_numpy(float)))
        for fr, v in zip(fracs, pts):
            row[f"within_{fr}"] = v
        lo, hi = np.percentile(np.mean(drs, 0), [2.5, 97.5])
        row["within_ci_lo"], row["within_ci_hi"] = lo, hi
        base = f[:-5] if f.endswith("_frac") else f
        ref = R[(R.feature == base) & (R.probe == p) & (R.target == t)]
        if len(ref):                       # per-cut reference, same rows (asserted in per_cut)
            rpts, rdrs, _ = per_cut(ref, fracs, sc, bf, W, qi, keys)
            dl, dh = np.percentile(np.mean(drs, 0) - np.mean(rdrs, 0), [2.5, 97.5])
            row.update(ref_base=base, ref_percut_mean=float(np.mean(rpts)),
                       diff_vs_ref=float(np.mean(pts) - np.mean(rpts)),
                       diff_ref_lo=dl, diff_ref_hi=dh, sig_vs_ref=bool(dl > 0 or dh < 0))
        rows.append(row)
    C = pd.DataFrame(rows)
    for i, r in C.iterrows():              # pooled with the cut indicator vs pooled without it
        k, kb = (r.feature, r.probe, r.target), (r.feature[:-5], r.probe, r.target)
        if not r.feature.endswith("_frac") or kb not in pooled_draws:
            continue
        dl, dh = np.percentile(np.mean(pooled_draws[k], 0) - np.mean(pooled_draws[kb], 0),
                               [2.5, 97.5])
        C.loc[i, ["pooled_base", "diff_vs_pooled_base", "diff_pool_lo", "diff_pool_hi",
                  "sig_vs_pooled_base"]] = [
            kb[0], float(np.mean(pooled_pts[k]) - np.mean(pooled_pts[kb])), dl, dh,
            bool(dl > 0 or dh < 0)]
    C = C.sort_values(["target", "feature"])
    C.to_csv(out / "within_cut.csv", index=False)
    cols = ["target", "feature", "metric", "n", "pooled", "pooled_auc", "within_cut_mean",
            "within_ci_lo", "within_ci_hi", "ref_percut_mean", "diff_vs_ref", "diff_ref_lo",
            "diff_ref_hi", "diff_vs_pooled_base", "diff_pool_lo", "diff_pool_hi"]
    print("\n== pooled vs within-cut (mean over cuts), 95% question bootstrap ==")
    print(C[[c for c in cols if c in C]].round(4).to_string(index=False))
    print("\nwrote", out / "within_cut.csv")


if __name__ == "__main__":
    main()
