#!/usr/bin/env python3
"""Probe accuracy in the target's OWN units, before it is compressed into R^2.

``probe_checkpoints.py`` reports R^2, which is a ratio of squared errors and
therefore flatters: R^2 = 0.28 on entropy sounds respectable, but it is only a
19% reduction in typical error (0.46 bits instead of 0.57, on a scale running
0 to 2.25).  This table reports both, side by side, so the R^2 column can be
read for what it is.

It also records the RANGE of the probe's predictions next to the range of the
truth.  A heavily-regularised probe on a weak signal hedges toward the mean and
never makes a confident call -- at the 90% checkpoint the true entropy spans
0 to 2.07 bits while the probe only ever outputs -0.08 to 0.97.  That behaviour
is invisible in any single accuracy number.

One fixed layer (default 20) is used for every row, deliberately.  Taking the
best of 28 layers per row -- as ``probe_checkpoints.py``'s ``act_best`` column
does -- is optimistically biased, because the maximum of 28 noisy estimates
exceeds their typical value even when no layer is better than another.  A fixed
layer is the honest single-shot number.

Two outputs:
  raw_units.csv    one row per (checkpoint, target) -- the summary above.
  predictions.csv  one row per (example_id, cot_id, checkpoint) carrying the
                   MEASURED ground truth next to the probe's held-out prediction
                   for every target.  This is the rawest form of the result: the
                   summary statistics are just reductions over these columns, and
                   any disagreement between them is a bug in the summary.

Usage
-----
    python src/probing/raw_units_table.py
    python src/probing/raw_units_table.py --layer 14
"""
from __future__ import annotations

import argparse, builtins, functools
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probing.probe_checkpoints import (  # noqa: E402
    ALPHAS, FoldViews, RidgePath, _r2, load, text_features,
)

print = functools.partial(builtins.print, flush=True)

SPECS = [("answer_entropy_bits_mm", "bits"), ("rem_len_mean", "tokens"),
         ("rem_len_std", "tokens"), ("p_mode", "fraction")]


def oof_predictions(fv: FoldViews, y: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Out-of-fold ridge predictions, alpha chosen on inner folds only.

    ``ok`` masks rows whose label is NaN.  The fold views were built on ALL rows
    at this checkpoint, so each fold's row indices are re-masked here rather than
    rebuilt -- the projection does not depend on y, so it stays valid.
    """
    oof = np.full(len(y), np.nan)
    for tr, te, Ztr, Zte, inners in fv.folds:
        tr_ok, te_ok = ok[tr], ok[te]
        if te_ok.sum() == 0 or tr_ok.sum() < 10:
            continue
        sc = np.zeros(len(ALPHAS))
        for itr, ite, Itr, Ite in inners:
            i_tr, i_te = ok[tr][itr], ok[tr][ite]
            if i_te.sum() < 3 or i_tr.sum() < 10:
                continue
            rp = RidgePath(Itr[i_tr], y[tr][itr][i_tr], standardize=False)
            for j, a in enumerate(ALPHAS):
                sc[j] += _r2(y[tr][ite][i_te], rp.predict(Ite[i_te], a))
        best = float(ALPHAS[int(np.argmax(sc))])
        rp = RidgePath(Ztr[tr_ok], y[tr][tr_ok], standardize=False)
        oof[te[te_ok]] = rp.predict(Zte[te_ok], best)
    return oof


def baseline_oof(kind: str, Mi, y, ok, g, o):
    """Out-of-fold predictions for a competitor, on the same folds as the probe.

    Without these the raw-units table only compares against "guess the average",
    which is the weakest baseline there is.  The research question is whether the
    activation beats the PREFIX TEXT, so the text model has to be here too, in
    the same units -- an R^2 gap of 0.01 and a gap of 300 tokens read very
    differently.  Each competitor gets its own swept regularisation, like the
    probe, so the comparison is about features and not tuning effort.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    oof = np.full(len(y), np.nan)
    grid = np.logspace(-6, 3, 10) if kind == "len" else np.logspace(-3, 3, 7)
    feats = (Mi.prefix_tokens.to_numpy(dtype=float).reshape(-1, 1) if kind == "len"
             else Mi.prefix_text.fillna("").to_numpy())
    for tr, te in o:
        tr2, te2 = tr[ok[tr]], te[ok[te]]
        if len(te2) == 0 or len(tr2) < 10:
            continue
        gi = g[tr2]
        best = grid[len(grid) // 2]
        k = min(3, len(np.unique(gi)))
        if k >= 2:
            sc = np.zeros(len(grid))
            for itr, ite in GroupKFold(n_splits=k).split(tr2, gi, gi):
                a_tr, a_te = tr2[itr], tr2[ite]
                if kind == "len":
                    Xt, Xe = feats[a_tr], feats[a_te]
                else:
                    Xt, Xe = text_features(feats[a_tr], feats[a_te])
                for j, al in enumerate(grid):
                    sc[j] += _r2(y[a_te], Ridge(alpha=al).fit(Xt, y[a_tr]).predict(Xe))
            best = float(grid[int(np.argmax(sc))])
        if kind == "len":
            Xt, Xe = feats[tr2], feats[te2]
        else:
            Xt, Xe = text_features(feats[tr2], feats[te2])
        oof[te2] = Ridge(alpha=best).fit(Xt, y[tr2]).predict(Xe)
    return oof


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--act-dir", default="runs/goal2/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--run-dir", default="runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--out-dir", default="runs/goal2/probes_ckpt/gpqa_diamond")
    p.add_argument("--layer", type=int, default=20, help="1-indexed decoder layer")
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()

    from sklearn.model_selection import GroupKFold

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    X, M = load(Path(a.act_dir), Path(a.run_dir))

    rows, per_row = [], []
    for frac in sorted(M.fraction.unique()):
        m = (M.fraction == frac).to_numpy()
        Mi, Xi = M[m].reset_index(drop=True), X[m]
        g = Mi.example_id.to_numpy()
        A = Xi[:, a.layer - 1, :].astype(np.float32)
        o = list(GroupKFold(n_splits=a.folds).split(A, g, g))
        fv = FoldViews(A, g, o)
        print(f"  checkpoint {frac:.0%}: {len(Mi)} CoTs")

        block = Mi[["example_id", "cot_id", "fraction", "n_continuations"]].copy()
        for tgt, unit in SPECS:
            y = Mi[tgt].to_numpy(dtype=float)
            ok = ~np.isnan(y)
            pred = oof_predictions(fv, y, ok)
            pred_len = baseline_oof("len", Mi, y, ok, g, o)
            pred_txt = baseline_oof("text", Mi, y, ok, g, o)
            block[f"{tgt}__true"] = y
            block[f"{tgt}__pred"] = pred
            block[f"{tgt}__pred_prefixlen"] = pred_len
            block[f"{tgt}__pred_text"] = pred_txt
            keep = ok & ~np.isnan(pred)
            yy, pp = y[keep], pred[keep]

            mae_probe = float(np.abs(pp - yy).mean())
            mae_mean = float(np.abs(yy.mean() - yy).mean())
            rows.append(dict(
                checkpoint=frac, target=tgt, unit=unit, layer=a.layer, n=int(keep.sum()),
                true_min=yy.min(), true_max=yy.max(),
                true_mean=yy.mean(), true_sd=yy.std(),
                pred_min=pp.min(), pred_max=pp.max(),
                mae_probe=mae_probe, mae_guess_mean=mae_mean,
                mae_prefixlen=float(np.abs(pred_len[keep] - yy).mean()),
                mae_text=float(np.abs(pred_txt[keep] - yy).mean()),
                pct_error_reduction=100.0 * (1.0 - mae_probe / mae_mean),
                r2=_r2(yy, pp),
                r2_prefixlen=_r2(yy, pred_len[keep]),
                r2_text=_r2(yy, pred_txt[keep]),
            ))
        per_row.append(block)

    df = pd.DataFrame(rows)
    pr = pd.concat(per_row, ignore_index=True).sort_values(
        ["example_id", "cot_id", "fraction"])
    pr.insert(4, "layer", a.layer)
    pr.to_csv(out / "predictions.csv", index=False)
    print(f"  wrote {out / 'predictions.csv'}  ({len(pr)} rows: "
          f"measured truth + held-out prediction per CoT per checkpoint)")
    dest = out / "raw_units.csv"
    df.to_csv(dest, index=False)
    print(f"\n  wrote {dest}  ({len(df)} rows, layer {a.layer})")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df[["checkpoint", "target", "n", "mae_guess_mean", "mae_prefixlen",
                  "mae_text", "mae_probe", "r2_prefixlen", "r2_text", "r2"
                  ]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
