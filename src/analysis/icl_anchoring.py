#!/usr/bin/env python3
"""Round 6, concern 1: why do in-context examples HURT? (k=0 beats k=10 and k=50, in two model families)

Free tests on predictions already on disk. Two hypotheses from NEXT_STEPS.md:
  H1 anchoring   - the model copies / is pulled toward the example values instead of reasoning.
  H2 noisy labels - each example label is a mean over ~25 rollouts, so it carries sampling error.

Uses the RAW (uncalibrated) predictions: calibration rescales per fold and would hide any shrinkage.
The example set is `icl_predict.example_order(d, fold, k)`, which depends only on (fold, k), so every
test row in a fold saw the SAME k examples. That bounds what H2 can be tested at: 5 example sets per k.

    ./envs/goal1/bin/python src/analysis/icl_anchoring.py
Outputs under runs/goal2/gpqa25/sweep/_iclwhy/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baselines.icl_predict as ip

OUT = Path("runs/goal2/gpqa25/sweep/_iclwhy"); OUT.mkdir(exist_ok=True)
ICL = Path("runs/goal2/gpqa25/icl")
REG = ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std"]
FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]
# the value the model actually SAW for each target, with label_json's rounding
ROUND = {"answer_entropy_bits_mm": lambda v: np.round(v, 2), "p_mode": lambda v: np.round(v, 2),
         "rem_len_mean": lambda v: np.round(v).astype(float), "rem_len_std": lambda v: np.round(v).astype(float)}

RUNS = [("Luna", "ckpt", 0, "gpt6_luna_high_k0"), ("Luna", "ckpt", 10, "gpt6_luna_high_k10"),
        ("Luna", "ckpt", 50, "gpt6_luna_high_k50"),
        ("Luna", "blind", 0, "gpt6_luna_high_blind_k0"), ("Luna", "blind", 10, "gpt6_luna_high_blind_k10"),
        ("Luna", "blind", 50, "gpt6_luna_high_blind_k50"),
        ("Luna", "nockpt", 10, "gpt6_luna_high_nockpt_k10"), ("Luna", "nockpt", 50, "gpt6_luna_high_nockpt_k50"),
        ("Gemma", "ckpt", 0, "gemma4_31b_off_k0"), ("Gemma", "ckpt", 10, "gemma4_31b_off_k10"),
        ("Gemma", "ckpt", 30, "gemma4_31b_off_k30"), ("Gemma", "ckpt", 93, "gemma4_31b_off_k93"),
        ("Gemma", "short", 10, "gemma4_31b_off_s2_k10"), ("Gemma", "short", 30, "gemma4_31b_off_s2_k30"),
        ("Gemma", "short", 93, "gemma4_31b_off_s2_k93"), ("Gemma", "short", 400, "gemma4_31b_off_s2_k400")]


def example_tables(d, ks):
    """(fold, k) -> the example rows the model saw, with their values rounded as shown."""
    ex = {}
    for k in sorted(ks):
        if k == 0:
            continue
        for fold in range(5):
            e = d.iloc[ip.example_order(d, fold, k)].copy()
            for t in REG:
                e[t] = ROUND[t](e[t].to_numpy(float))
            ex[(fold, k)] = e
    return ex


def main():
    d = ip.load_data()
    fold_of = d.drop_duplicates("example_id").set_index("example_id").fold
    ex = example_tables(d, {k for *_, k, _ in RUNS})

    # chance baseline for the verbatim-copy rate: the k=0 predictions of the same model, matched
    # against the example values that fold WOULD have been shown. Needed because entropy and p_mode
    # are shown to 2 decimals on a short range, so with 50 examples a coincidental match is likely.
    zero = {}
    for model, version, k, tag in RUNS:
        if k == 0:
            zero[model] = pd.read_csv(ICL / f"preds_{tag}.csv") if (ICL / f"preds_{tag}.csv").exists() else None
            if version == "ckpt":
                zero[model + "_ckpt0"] = zero[model]

    def copy_rate(pred_df, t, k, fold_of_):
        """mean over folds of the share of predictions that exactly equal one of that fold's k shown values."""
        x = pred_df[(pred_df.probe == "icl_raw") & (pred_df.fraction > 0) & (pred_df.target == t)].dropna(subset=["y_pred"])
        x = x.assign(fold=fold_of_.reindex(x.example_id).to_numpy())
        return float(np.mean([np.isin(ROUND[t](g.y_pred.to_numpy()), ex[(int(fo), k)][t].to_numpy(float)).mean()
                              for fo, g in x.groupby("fold")]))

    rows, folds = [], []
    for model, version, k, tag in RUNS:
        f = ICL / f"preds_{tag}.csv"
        if not f.exists():
            print("missing", tag); continue
        P = pd.read_csv(f)
        P = P[(P.probe == "icl_raw") & (P.fraction > 0)].copy()
        P["fraction"] = P.fraction.round(3)
        P["fold"] = fold_of.reindex(P.example_id).to_numpy()
        for t in REG:
            x = P[P.target == t].dropna(subset=["y_pred"])
            if not len(x):
                continue
            # within-cut spread of predictions and of the truth (averaged over cuts), on the same rows
            sd_p = x.groupby("fraction").y_pred.std().mean()
            sd_y = x.groupby("fraction").y_true.std().mean()
            # pull toward the example mean, and verbatim copying, per fold
            dist, copy, emeans, pmeans = [], [], [], []
            for fo, g in x.groupby("fold"):
                if k == 0:
                    continue
                vals = ex[(int(fo), k)][t].to_numpy(float)
                em = vals.mean()
                dist.append(np.abs(g.y_pred.to_numpy() - em).mean() / sd_y)
                copy.append(np.isin(ROUND[t](g.y_pred.to_numpy()), vals).mean())
                emeans.append(em); pmeans.append(g.y_pred.mean())
            r = dict(model=model, version=version, k=k, target=t, n=len(x),
                     sd_pred=sd_p, sd_true=sd_y, spread_ratio=sd_p / sd_y,
                     pooled_r2_raw=1 - ((x.y_true - x.y_pred) ** 2).sum() / ((x.y_true - x.y_true.mean()) ** 2).sum(),
                     mean_within_cut_corr=x.groupby("fraction").apply(
                         lambda g: np.corrcoef(g.y_true, g.y_pred)[0, 1]).mean())
            if k:
                chance = (copy_rate(zero[model + "_ckpt0"], t, k, fold_of)
                          if zero.get(model + "_ckpt0") is not None else np.nan)
                r.update(dist_to_example_mean_over_sd=float(np.mean(dist)),
                         verbatim_copy_rate=float(np.mean(copy)),
                         copy_rate_chance=chance, copy_rate_excess=float(np.mean(copy)) - chance,
                         fold_anchor_slope=float(np.polyfit(emeans, pmeans, 1)[0]),
                         fold_anchor_corr=float(np.corrcoef(emeans, pmeans)[0, 1]))
                # H2: example-label noise vs this cell's error, one point per fold
                for fo, g in x.groupby("fold"):
                    e = ex[(int(fo), k)]
                    se = (e.rem_len_std / np.sqrt(e.n_continuations)).mean()
                    folds.append(dict(model=model, version=version, k=k, target=t, fold=int(fo),
                                      mean_example_se_rem_len=float(se),
                                      rmse_over_sd=float(np.sqrt(((g.y_true - g.y_pred) ** 2).mean()) / sd_y)))
            rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(OUT / "anchoring.csv", index=False)
    F = pd.DataFrame(folds); F.to_csv(OUT / "h2_example_noise_by_fold.csv", index=False)

    print("== H1 anchoring: prediction spread and pull toward the example values ==")
    print("spread_ratio = sd(prediction)/sd(truth) within cut; dist = |pred - example mean| / sd(truth)")
    for t in REG:
        print(f"\n-- {t}")
        c = ["model", "version", "k", "pooled_r2_raw", "mean_within_cut_corr", "spread_ratio",
             "dist_to_example_mean_over_sd", "verbatim_copy_rate", "copy_rate_chance", "copy_rate_excess"]
        print(R[R.target == t][[x for x in c if x in R]].round(3).to_string(index=False))

    print("\n== H2: example-label noise vs error, across the 5 folds (one example set per fold) ==")
    if len(F):
        g = (F.groupby(["model", "version", "k", "target"])
             .apply(lambda x: pd.Series(dict(n_folds=len(x),
                                             corr=np.corrcoef(x.mean_example_se_rem_len, x.rmse_over_sd)[0, 1],
                                             se_spread=x.mean_example_se_rem_len.std())))
             .reset_index())
        print(g.round(3).to_string(index=False))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
