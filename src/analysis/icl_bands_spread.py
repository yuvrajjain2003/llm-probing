#!/usr/bin/env python3
"""Are the probe and the frontier model just predicting the mean inside a difficulty band?

A predictor that has learned nothing within a group emits a near-constant value there: its spread
collapses to zero and its correlation with the truth is zero. This measures both, per difficulty band,
for answer entropy and remaining length, alongside the truth's own mean and spread.

Also shows how each predictor's MEAN moves across bands. If it tracks the truth's mean across bands
while being flat inside them, the predictor has learned question difficulty and nothing finer.

Luna is shown both calibrated (`icl_cal`, what fig9 scores) and raw, because the per-fold calibration
rescales the predictions and so partly sets their spread.

    ./envs/goal1/bin/python src/analysis/icl_bands_spread.py
Output: runs/goal2/gpqa25/sweep/_iclsplit/p7_band_spread.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import icl_decompose as dc
from round2_metrics import B, SEED

ORDER = ["all questions", "v_hard", "hard", "medium", "easy", "v_easy"]
ICL = Path("runs/goal2/gpqa25/icl")
T = {"answer_entropy_bits_mm": "entropy", "rem_len_mean": "remaining length"}


def series():
    """label -> dataframe of (example_id, cot_id, fraction, target, y_true, y_pred)."""
    P = dc.load()
    out = {"Best probe": P[P.label == dc.OURS], "Luna k=0 (calibrated)": P[P.label == "Luna k=0"]}
    raw = pd.read_csv(ICL / "preds_gpt6_luna_high_k0.csv")
    raw = raw[(raw.probe == "icl_raw") & (raw.fraction > 0)].copy()
    raw["fraction"] = raw.fraction.round(3)
    out["Luna k=0 (raw)"] = raw
    return out


def main():
    dif = pd.read_csv(dc.DIFF).set_index("example_id")
    S = series()
    rows = []
    for lab, D in S.items():
        for t in T:
            for b in ORDER:
                keep = set(dif.index) if b == "all questions" else set(dif.index[dif.band_quintile == b])
                st = {k: [] for k in ("mean_true", "sd_true", "mean_pred", "sd_pred", "corr")}
                bs = {k: [] for k in ("mean_true", "sd_true", "mean_pred", "sd_pred")}
                for f in dc.FRACS:
                    x = D[(D.target == t) & (D.fraction.round(3) == f) & (D.example_id.isin(keep))]
                    x = x.dropna(subset=["y_pred"])
                    y, p = x.y_true.to_numpy(float), x.y_pred.to_numpy(float)
                    st["mean_true"].append(y.mean()); st["sd_true"].append(y.std())
                    st["mean_pred"].append(p.mean()); st["sd_pred"].append(p.std())
                    st["corr"].append(np.corrcoef(y, p)[0, 1] if y.std() > 0 and p.std() > 0 else np.nan)
                    # question bootstrap for the mean and the spread
                    e = x.example_id.to_numpy()
                    qs = np.sort(np.unique(e)); qi = {int(v): i for i, v in enumerate(qs)}
                    W = np.random.default_rng(SEED).multinomial(
                        len(qs), np.ones(len(qs)) / len(qs), size=B).astype(float)[:, [qi[int(v)] for v in e]]
                    sw = W.sum(1)
                    for nm, v in (("true", y), ("pred", p)):
                        m = (W * v).sum(1) / sw
                        bs[f"mean_{nm}"].append(m)
                        bs[f"sd_{nm}"].append(np.sqrt((W * (v[None, :] - m[:, None]) ** 2).sum(1) / sw))
                r = {k: float(np.mean(v)) for k, v in st.items()}
                r["sd_ratio"] = r["sd_pred"] / r["sd_true"]
                for k, v in bs.items():                      # mean over cuts, then 95% interval
                    d = np.mean(v, 0)
                    r[k + "_lo"], r[k + "_hi"] = np.percentile(d, [2.5, 97.5])
                rows.append(dict(label=lab, target=t, band=b, **r))
    R = pd.DataFrame(rows)
    R.to_csv(dc.OUT / "p7_band_spread.csv", index=False)
    for t in T:
        print(f"\n===== {T[t].upper()} (mean over the 5 cuts) =====")
        for lab in S:
            x = R[(R.label == lab) & (R.target == t)].set_index("band").reindex(ORDER)
            print(f"\n-- {lab}")
            print(x[["mean_true", "mean_pred", "sd_true", "sd_pred", "sd_ratio", "corr"]]
                  .round(3 if t.startswith("answer") else 1).to_string())
    print("\nwrote", dc.OUT / "p7_band_spread.csv")


if __name__ == "__main__":
    main()
