#!/usr/bin/env python3
"""Round 7, items 2 and 3: prediction ERROR per question against a continuous question measure.

The professor asked to move away from the difficulty buckets and plot the relationship continuously.
Item 2 does that with question accuracy (`p_gold`) on GPQA and MMLU-Pro; item 3 does it on
GlobalOpinionQA, which has no gold answer, using the model's self-consistency (`p_mode0`), the HUMAN
survey disagreement that ships with the dataset (`human_disagree`, never used before here) and the
non-circular out-of-fold question-text measure (`qtext_ent_norm_mean`).

Nothing is refitted. Out-of-fold predictions are read from:
  probe / TF-IDF / prefix length, raw targets : <tag>/sweep/ref_fixed/preds*.csv
  the same three on log1p length targets      : <tag>/sweep/_loglen/preds.csv
  the black-box monitor (GPT-6 Luna, k=0)     : <tag>/icl/preds_gpt6_luna_high_k0.csv
Luna is reported BOTH calibrated (`icl_cal`, as published) and raw (`icl_raw`), because the probe is
not calibrated and squared error is scale sensitive in a way R2 is not.

TWO ROW SETS, both emitted and both named on every figure (the project's most repeated trap):
  full : every sweep row. The probe and the text baselines only; the monitor never ran on these.
  icl  : inner-joined to the rows the monitor was actually called on (1 chain per question per cut,
         500 questions on the new datasets, 196 on GPQA). Every label on identical rows.

ERROR MEASURE. Squared error is computed inside each cut and then averaged over cuts, never pooled
across depth. Two columns per question:
  mse      mean over cuts of the squared error, in the target's own units
  nmse     the same with each cut's squared error divided by that cut's target variance over all
           questions of the row set, so deep cuts are not drowned by shallow ones (learnings.md,
           2026-10-09: a mean MSE over checkpoints is dominated by the early cuts). nmse = 1.0 is
           "no better than predicting that cut's average".

    ./envs/goal1/bin/python src/analysis/r7_err_vs_x.py
Outputs: runs/goal2/<tag>/sweep/_r7err/{per_question.csv, binned.csv, x_measures.csv}
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R = Path("runs/goal2")
K = ["example_id", "cot_id", "fraction"]
FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]
B, SEED = 2000, 0

OURS, TFIDF, PLEN = "Our probe", "TF-IDF on prefix", "Prefix length"
LUNA_C, LUNA_R = "Luna k=0 (calibrated)", "Luna k=0 (raw)"
SWEEP = {OURS: ("act_L21", "ridge_act"), TFIDF: ("tfidf", "ridge_text"),
         PLEN: ("prefix_len", "ridge_len")}

RAW_TARGETS = ["answer_entropy_bits_mm", "answer_entropy_bits_mm_norm", "p_mode",
               "rem_len_mean", "rem_len_std"]
LOG_TARGETS = ["rem_len_mean", "rem_len_std"]

# tag -> list of (column, source, human-readable name, which way "harder" points)
XMEAS = {
    "gpqa25": [("p_gold", "difficulty", "share of the 25 attempts that were correct", "accuracy"),
               ("qtext_pgold", "qtext", "accuracy predicted from the question words alone",
                "accuracy")],
    "mmlupro_v1": [("p_gold", "difficulty", "share of the attempts that were correct", "accuracy"),
                   ("qtext_pgold", "qtext", "accuracy predicted from the question words alone",
                    "accuracy")],
    "globalop_v1": [("p_mode0", "difficulty", "model self-consistency at the prompt", "consistency"),
                    ("human_disagree_multi", "human",
                     "disagreement among the human survey answers", "disagreement"),
                    ("qtext_ent_norm_mean", "qtext",
                     "uncertainty predicted from the question words alone", "uncertainty"),
                    ("ent0_norm", "difficulty", "model uncertainty at the prompt", "uncertainty")],
}


def add_human_multi(dif):
    """`human_disagree` is identically 0 whenever only ONE country was surveyed, which is 849 of the
    2,556 released GlobalOpinionQA rows: that zero means "nobody to disagree with", not "nobody
    disagreed". `human_disagree_multi` is the same column restricted to the multi-country questions,
    which is the only version the figure should use. n_countries is recomputed by importing
    newds_difficulty.human_measures, the function that built the published column."""
    import importlib
    nd = importlib.import_module("analysis.newds_difficulty")
    ex = pd.read_csv("runs/goal1/globalop_v1/global_opinions/"
                     "DeepSeek-R1-Distill-Qwen-7B/raw/examples.csv")
    h = nd.human_measures(ex)[["example_id", "n_countries"]]
    d = dif.merge(h, on="example_id", how="left")
    d["human_disagree_multi"] = d.human_disagree.where(d.n_countries >= 2)
    print(f"  human_disagree: {int((d.n_countries == 1).sum())} single-country questions excluded, "
          f"{int(d.human_disagree_multi.notna().sum())} kept")
    return d


def read_sweep(tag):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / tag / "sweep/ref_fixed/preds*.csv")))],
                  ignore_index=True)
    d["fraction"] = d.fraction.round(3)
    return d[d.fraction > 0]


def read_loglen(tag):
    d = pd.read_csv(R / tag / "sweep/_loglen/preds.csv")
    d["fraction"] = d.fraction.round(3)
    return d[(d["transform"] == "log1p") & (d.fraction > 0)]


def read_icl(tag):
    d = pd.read_csv(R / tag / "icl/preds_gpt6_luna_high_k0.csv")
    d["fraction"] = d.fraction.round(3)
    return d[d.fraction > 0]


def collect(tag):
    """One long frame: label, target, transform, row_set, keys, y_true, y_pred."""
    sw, lg, icl = read_sweep(tag), read_loglen(tag), read_icl(tag)
    keys = icl[K].drop_duplicates()
    print(f"{tag}: sweep rows {len(sw):,}, ICL rows {len(keys):,} "
          f"({keys.example_id.nunique()} questions)")
    out = []
    for lab, (ft, pr) in SWEEP.items():
        for src, tr, tgts in [(sw, "raw", RAW_TARGETS), (lg, "log1p", LOG_TARGETS)]:
            x = src[(src.feature == ft) & (src.probe == pr) & (src.target.isin(tgts))]
            x = x.drop_duplicates(["target"] + K)
            out.append(x.assign(label=lab, transform=tr, row_set="full")
                       [["label", "target", "transform", "row_set"] + K + ["y_true", "y_pred"]])
            j = x.merge(keys, on=K)
            out.append(j.assign(label=lab, transform=tr, row_set="icl")
                       [["label", "target", "transform", "row_set"] + K + ["y_true", "y_pred"]])
    for lab, pr in [(LUNA_C, "icl_cal"), (LUNA_R, "icl_raw")]:
        x = icl[icl.probe == pr].drop_duplicates(["target"] + K)
        out.append(x.assign(label=lab, transform="raw", row_set="icl")
                   [["label", "target", "transform", "row_set"] + K + ["y_true", "y_pred"]])
    P = pd.concat(out, ignore_index=True)
    return P.dropna(subset=["y_true", "y_pred"])


def per_question(P):
    """Squared error inside each cut, averaged over cuts, per question."""
    P = P.copy()
    P["se"] = (P.y_true - P.y_pred) ** 2
    # each cut's target variance over all questions of that row_set, for the normalised column
    var = (P.drop_duplicates(["row_set", "target", "transform", "fraction"] + K)
           .groupby(["row_set", "target", "transform", "fraction"]).y_true.var(ddof=0)
           .rename("cut_var").reset_index())
    P = P.merge(var, on=["row_set", "target", "transform", "fraction"], how="left")
    P["nse"] = P.se / P.cut_var
    g = P.groupby(["row_set", "label", "target", "transform", "fraction", "example_id"])
    per_cut = g.agg(se=("se", "mean"), nse=("nse", "mean"), n_rows=("se", "size")).reset_index()
    q = (per_cut.groupby(["row_set", "label", "target", "transform", "example_id"])
         .agg(mse=("se", "mean"), nmse=("nse", "mean"), n_cuts=("fraction", "nunique"),
              n_rows=("n_rows", "sum")).reset_index())
    return q


def bins_of(x, max_distinct=12, nbins=10):
    """Each distinct value if the measure is coarse, else equal-count bins."""
    u = np.unique(x)
    if len(u) <= max_distinct:
        return np.searchsorted(u, x), [f"{v:.3g}" for v in u], len(u)
    qs = np.quantile(x, np.linspace(0, 1, nbins + 1))
    qs[0] -= 1e-9
    qs = np.unique(qs)
    idx = np.clip(np.searchsorted(qs, x, side="left") - 1, 0, len(qs) - 2)
    return idx, [f"{qs[i]:.3g}-{qs[i + 1]:.3g}" for i in range(len(qs) - 1)], len(qs) - 1


def binned(Q, P, dif, tag):
    """Bin means with a question-bootstrap interval, per x measure.

    Two reference labels are emitted beside the predictors, the same two item 1 puts on the band
    figures: `that bin's own average` (the MSE of predicting the mean of the truth inside the bin,
    which is where R2 would be 0) and `the full set's average`. Without them an MSE curve cannot be
    read, because a bin where the truth barely varies also has a small achievable error."""
    rows = []
    rng = np.random.default_rng(SEED)
    # truth per row, once: identical for every label at the same (row_set, target, transform, key)
    Y = (P[P.label == OURS][["row_set", "target", "transform", "fraction", "example_id", "y_true"]]
         .copy())
    for col, _, nice, _ in XMEAS[tag]:
        if col not in dif.columns:
            print(f"  {col} missing, skipped")
            continue
        xmap = dif[["example_id", col]].dropna()
        # one binning per x measure, shared by every predictor and by the reference lines
        idx, names, _ = bins_of(xmap[col].to_numpy(float))
        binmap = xmap.assign(bin=idx)[["example_id", "bin"]]
        for (rs, lab, t, tr), g in Q.groupby(["row_set", "label", "target", "transform"]):
            m = g.merge(xmap, on="example_id").merge(binmap, on="example_id")
            if len(m) < 30:
                continue
            for b in sorted(m.bin.unique()):
                s = m[m.bin == b]
                if len(s) < 5:
                    continue
                for vcol in ["mse", "nmse"]:
                    v = s[vcol].to_numpy(float)
                    draws = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
                    rows.append(dict(tag=tag, x_measure=col, x_name=nice, row_set=rs, label=lab,
                                     target=t, transform=tr, metric=vcol, bin=int(b),
                                     bin_label=names[int(b)], x=float(s[col].mean()),
                                     n_questions=len(s), value=float(v.mean()),
                                     ci_lo=float(np.percentile(draws, 2.5)),
                                     ci_hi=float(np.percentile(draws, 97.5))))
        # the two reference lines, from the truth alone
        y = Y.merge(xmap, on="example_id").merge(binmap, on="example_id")
        for (rs, t, tr), g in y.groupby(["row_set", "target", "transform"]):
            # per cut: the full-set mean of the truth, and that cut's variance (the nmse divisor)
            g = g.assign(se_global=(g.y_true - g.groupby("fraction").y_true.transform("mean")) ** 2)
            cutvar = g.groupby("fraction").y_true.var(ddof=0)
            for b in sorted(g.bin.unique()):
                s = g[g.bin == b]
                if s.example_id.nunique() < 5:
                    continue
                bm = s.groupby("fraction").y_true.transform("mean")   # this bin's mean at that cut
                per_cut_bin = ((s.y_true - bm) ** 2).groupby(s.fraction).mean()
                per_cut_glob = s.se_global.groupby(s.fraction).mean()
                cv = cutvar.reindex(per_cut_bin.index)
                for vcol, vb, vg in [
                        ("mse", float(per_cut_bin.mean()), float(per_cut_glob.mean())),
                        ("nmse", float((per_cut_bin / cv).mean()),
                         float((per_cut_glob / cv).mean()))]:
                    for nm, val in [("that bin's own average", vb),
                                    ("the full set's average", vg)]:
                        rows.append(dict(tag=tag, x_measure=col, x_name=nice, row_set=rs, label=nm,
                                         target=t, transform=tr, metric=vcol, bin=int(b),
                                         bin_label=names[int(b)], x=float(s[col].mean()),
                                         n_questions=int(s.example_id.nunique()), value=val,
                                         ci_lo=np.nan, ci_hi=np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=list(XMEAS))
    a = ap.parse_args()
    for tag in a.tags:
        S = R / tag / "sweep"
        out = S / "_r7err"
        out.mkdir(parents=True, exist_ok=True)
        dif = pd.read_csv(S / "_difficulty/question_difficulty.csv")
        qt = pd.read_csv(S / "_qtext/question_difficulty.csv")
        extra = [c for c in qt.columns if c.startswith("qtext_") and c not in dif.columns]
        dif = dif.merge(qt[["example_id"] + extra], on="example_id", how="left")
        if tag == "globalop_v1":
            dif = add_human_multi(dif)
        P = collect(tag)
        Q = per_question(P)
        Q.insert(0, "tag", tag)
        Q.to_csv(out / "per_question.csv", index=False)
        BN = binned(Q, P, dif, tag)
        BN.to_csv(out / "binned.csv", index=False)
        # what the x measures look like, so a figure can state n per bin honestly
        xm = []
        for col, _, nice, direction in XMEAS[tag]:
            if col not in dif.columns:
                continue
            v = dif[col].dropna().to_numpy(float)
            xm.append(dict(tag=tag, x_measure=col, x_name=nice, direction=direction, n=len(v),
                           n_distinct=len(np.unique(v)), vmin=v.min(), vmax=v.max(),
                           mean=v.mean(), share_at_extremes=float(
                               ((v == v.min()) | (v == v.max())).mean())))
        pd.DataFrame(xm).to_csv(out / "x_measures.csv", index=False)
        print(f"  per_question {len(Q):,} rows, binned {len(BN):,} rows -> {out}")
        print(pd.DataFrame(xm).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
