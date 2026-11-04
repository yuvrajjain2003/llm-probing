#!/usr/bin/env python3
"""P2.3 / P2.4: score the probe INSIDE difficulty bands, and with difficulty partialled out.

This is round 6's `_iclsplit` analysis (src/analysis/icl_decompose.py part 2, src/analysis/icl_bands.py)
rerun on a new dataset, for the PROBE and the two text baselines. There are no in-context runs on the
new datasets, so the Luna / Gemma rows of the round-6 SPEC are absent and `Our model L21 (ridge)`,
`TF-IDF on prefix` and `Prefix length` take their place.

Nothing in icl_decompose.py / icl_bands.py is edited. They are imported and their module-level
constants are repointed at runtime (output dir, difficulty file, SPEC, target list, loader), so the
published GPQA path stays byte-identical; that is proved separately by re-running them unchanged with
only the output directory redirected and diffing against runs/goal2/gpqa25/sweep/_iclsplit/.

Two difficulty measures are meant to be run and reported as a bracket:
  --diff rollout : _difficulty/question_difficulty.csv, label-derived (p_gold), so it OVER-removes
                   -> the partialled score is a LOWER bound on what the probe retains
  --diff qtext   : _qtext/question_difficulty.csv, an out-of-fold question-text prediction, so it is
                   non-circular but attenuated and UNDER-removes -> an UPPER bound

    ./envs/goal1/bin/python src/analysis/newds_difficulty_scoring.py --tag mmlupro_v1 \
        --diff rollout --band-cols band_int band_quintile
Outputs: runs/goal2/<tag>/sweep/_iclsplit/*.<diff>[.<band-col>].csv
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from round2_metrics import BIN, SEED, B
import icl_decompose as dc
import icl_bands as ib

OURS = "Our model L21 (ridge)"
SPEC = {OURS: (None, "act_L21", None, "ours"),
        "TF-IDF on prefix": (None, "tfidf", None, "text baseline"),
        "Prefix length": (None, "prefix_len", None, "text baseline")}
REFS = [OURS, "TF-IDF on prefix"]
REG_BASE = ["answer_entropy_bits_mm", "answer_entropy_bits_mm_norm", "p_mode",
            "rem_len_mean", "rem_len_std"]
ORDER = ib.ORDER


def load_preds(tag, sweep="ref_fixed"):
    """Long frame of out-of-fold predictions, one row per (label, target, example_id, cot_id, cut)."""
    S = Path("runs/goal2") / tag / "sweep"
    d = pd.concat([pd.read_csv(x) for x in sorted((S / sweep).glob("preds*.csv"))], ignore_index=True)
    d["fraction"] = d.fraction.round(3)
    d = d[d.fraction > 0]
    reg = [t for t in REG_BASE if t in set(d.target)]
    rows = []
    for lab, (_, ft, pr, _) in SPEC.items():
        x = d[d.feature == ft]
        assert len(x), f"{tag}: no rows for feature {ft}"
        for t in reg + [BIN]:
            y = x[x.target == t]
            if pr is not None:
                y = y[y.probe == pr]
            assert y.probe.nunique() == 1, f"{tag}/{ft}/{t}: probes {y.probe.unique()}"
            rows.append(y.drop_duplicates(["example_id", "cot_id", "fraction"])
                        .assign(label=lab)[["label", "target", "example_id", "cot_id", "fraction",
                                            "y_true", "y_pred"]])
    P = pd.concat(rows, ignore_index=True)
    return P, reg


def refit_and_spread(P, dif, band_col, reg):
    """Per band: does refitting the output map inside the band rescue it, and is the predictor just
    emitting a constant there?  The three refit columns use the same formulas as
    icl_bands_refit.py (as stored / intercept only / intercept and slope = squared correlation);
    the spread columns the same as icl_bands_spread.py.  In sample, so generous to the predictor."""
    rows = []
    groups = [("all questions", set(dif.index))] + \
             [(b, set(dif.index[dif[band_col] == b])) for b in ORDER]
    for lab in SPEC:
        for t in reg:
            for b, keep in groups:
                acc = {k: [] for k in ("as_stored", "refit_intercept", "refit_intercept_and_slope",
                                       "mean_true", "sd_true", "mean_pred", "sd_pred", "corr")}
                n = 0
                for f in dc.FRACS:
                    x = P[(P.target == t) & (P.fraction == f) & (P.label == lab)
                          & (P.example_id.isin(keep))]
                    y, p = x.y_true.to_numpy(float), x.y_pred.to_numpy(float)
                    sst = ((y - y.mean()) ** 2).sum(); n += len(y)
                    acc["as_stored"].append(1 - ((y - p) ** 2).sum() / sst)
                    acc["refit_intercept"].append(1 - ((y - (p - p.mean() + y.mean())) ** 2).sum() / sst)
                    acc["refit_intercept_and_slope"].append(np.corrcoef(y, p)[0, 1] ** 2)
                    acc["mean_true"].append(y.mean()); acc["sd_true"].append(y.std())
                    acc["mean_pred"].append(p.mean()); acc["sd_pred"].append(p.std())
                    acc["corr"].append(np.corrcoef(y, p)[0, 1])
                r = {k: float(np.mean(v)) for k, v in acc.items()}
                r["sd_ratio"] = r["sd_pred"] / r["sd_true"]
                rows.append(dict(label=lab, target=t, band=b, band_col=band_col,
                                 n_questions=len(keep), rows_per_cut=n // len(dc.FRACS), **r))
    return pd.DataFrame(rows)


def band_decomposition(P, dif, band_col, reg, group=None):
    """Split the truth's spread into a BETWEEN-band and a WITHIN-band part and score each.

    Exactly round 6's learnings.md decomposition (2026-10-29), with "difficulty band" as the group.
    The cross term is identically zero, so pooled R2 = w_between * R2_between + w_within * R2_within;
    the identity is asserted. This is the offset-free way to read the band result: a per-band R2 is
    scored against that band's own small variance with predictions that were never recentred there,
    while the within-band part uses one common recentring per band for truth and prediction alike."""
    b = dif[band_col] if group is None else group
    rows = []
    for lab in SPEC:
        for t in reg:
            acc = []
            for f in dc.FRACS:
                x = P[(P.target == t) & (P.fraction == f) & (P.label == lab)].copy()
                x["band"] = b.reindex(x.example_id).to_numpy()
                x = x[x.band.notna()]
                y, p = x.y_true.to_numpy(float), x.y_pred.to_numpy(float)
                gb = x.groupby("band", observed=True)
                yb = gb.y_true.transform("mean").to_numpy(float)
                pb = gb.y_pred.transform("mean").to_numpy(float)
                ssb = ((yb - y.mean()) ** 2).sum(); ssw = ((y - yb) ** 2).sum()
                sst = ((y - y.mean()) ** 2).sum()
                r2b = 1 - ((yb - pb) ** 2).sum() / ssb
                r2w = 1 - (((y - yb) - (p - pb)) ** 2).sum() / ssw
                pooled = 1 - ((y - p) ** 2).sum() / sst
                assert abs(pooled - (ssb * r2b + ssw * r2w) / sst) < 1e-8
                acc.append((ssb / sst, ssw / sst, r2b, r2w, pooled))
            m = np.mean(acc, 0)
            rows.append(dict(label=lab, target=t, band_col=band_col,
                             share_between_bands=m[0], share_within_bands=m[1],
                             r2_between_bands=m[2], r2_within_bands=m[3], r2_pooled=m[4]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--diff", default="rollout", choices=["rollout", "qtext"])
    ap.add_argument("--band-cols", nargs="+", default=["band_int"])
    ap.add_argument("--out", default=None,
                    help="override the output directory (used for the GPQA comparison runs, so that "
                         "nothing is written under runs/goal2/gpqa25/)")
    a = ap.parse_args()

    S = Path("runs/goal2") / a.tag / "sweep"
    out = Path(a.out) if a.out else S / "_iclsplit"
    out.mkdir(parents=True, exist_ok=True)
    diff_path = S / ("_difficulty/question_difficulty.csv" if a.diff == "rollout"
                     else "_qtext/question_difficulty.csv")
    P, reg = load_preds(a.tag)
    nq = P.example_id.nunique()
    print(f"{a.tag}: {nq} questions, targets {reg + [BIN]}, difficulty = {a.diff} ({diff_path})")

    dif = pd.read_csv(diff_path).set_index("example_id")
    if "difficulty" not in dif.columns:          # GPQA's published round-6 file predates the column
        dif = dif.assign(difficulty=dif.p_gold, difficulty_measure="p_gold")
    dif = dif[dif.difficulty.notna()]
    print("difficulty measure:", dif.difficulty_measure.unique(),
          "| questions with a difficulty:", len(dif),
          "| covered by preds:", dif.index.isin(P.example_id.unique()).sum())
    # dc.part2 / part3 read the column named p_gold; feed the chosen measure through it.
    dif2 = dif.assign(p_gold=dif.difficulty.astype(float))
    # drop questions with no difficulty from the row set so every part uses one row set
    P = P[P.example_id.isin(dif2.index)]

    # ---- repoint the round-6 modules (runtime only; their files are untouched)
    dc.OUT, dc.DIFF, dc.SPEC, dc.REFS, dc.OURS, dc.REG = out, diff_path, SPEC, REFS, OURS, reg
    dc.SHORT = {"answer_entropy_bits_mm": "entropy", "answer_entropy_bits_mm_norm": "entropy_norm",
                "p_mode": "p_mode", "rem_len_mean": "rem_len_mean", "rem_len_std": "rem_len_std",
                BIN: "committed"}
    dc.load = lambda: P
    ib.OUT, ib.LABELS, ib.TARGETS, ib.REG = out, list(SPEC), reg + [BIN], reg

    # ---- P2.4 partialling
    point, draws, labels = dc.part2(P, dif2)
    point.to_csv(out / f"p2_per_checkpoint.{a.diff}.csv", index=False)
    SC, PR = dc.summarise(point, draws, labels)
    SC.to_csv(out / f"p2_ci_scores.{a.diff}.csv", index=False)
    PR.to_csv(out / f"p2_ci_paired.{a.diff}.csv", index=False)
    print("\n== P2.4: R2 as published and after partialling difficulty (mean over cuts) ==")
    print(SC[SC.metric != "r2_partialled_rescaled"]
          .pivot_table(index="label", columns=["metric", "target"], values="value").round(3).to_string())

    # ---- commitment: accuracy, AUC, majority baseline
    CM, AG, AC = dc.part3(P, dif2)
    CM.to_csv(out / f"p3_committed.{a.diff}.csv", index=False)
    AG.to_csv(out / f"p3_agreement.{a.diff}.csv", index=False)
    AC.to_csv(out / f"p3_committed_auc_ci.{a.diff}.csv", index=False)
    print("\n== commitment (mean over cuts): accuracy, AUC, majority ==")
    print(CM.groupby("label")[["accuracy", "auc", "majority_baseline"]].mean().round(3)
          .join(AC.set_index("label")[["diff_vs_ours", "diff_lo", "diff_hi", "significant"]].round(3))
          .to_string())

    # ---- P2.3 bands, once per banding scheme
    for bc in a.band_cols:
        sys.argv = ["icl_bands", "--band-col", bc]
        ib.main()
        for nm in ["p5_bands_scores", "p5_bands_paired"]:
            shutil.move(out / f"{nm}.csv", out / f"{nm}.{a.diff}.{bc}.csv")
        R = refit_and_spread(P, dif, bc, reg)
        R.to_csv(out / f"p6_band_refit_spread.{a.diff}.{bc}.csv", index=False)
        D = band_decomposition(P, dif, bc, reg)
        D.to_csv(out / f"p8_band_decomposition.{a.diff}.{bc}.csv", index=False)
        print(f"\n== between-band vs within-band decomposition ({bc}) ==")
        print(D.set_index(["target", "label"]).drop(columns="band_col").round(3).to_string())
        print(f"\n== band refit and spread ({bc}), entropy and rem_len_mean, our probe ==")
        print(R[(R.label == OURS)
                & R.target.isin(["answer_entropy_bits_mm", "rem_len_mean"])]
              .set_index(["target", "band"])[["rows_per_cut", "as_stored", "refit_intercept",
                                              "refit_intercept_and_slope", "sd_ratio", "corr"]]
              .round(3).to_string())
    # ---- the same decomposition with QUESTION as the group: the no-difficulty-measure version of
    # the same question. Computable here (about 5 chains per question) where the GPQA ICL rows made
    # it degenerate (1 chain per question, so within-question SS is identically 0).
    qgrp = pd.Series(dif.index.to_numpy(), index=dif.index, name="example_id")
    Q = band_decomposition(P, dif, "example_id", reg, group=qgrp)
    Q.to_csv(out / f"p9_question_decomposition.{a.diff}.csv", index=False)
    print("\n== between-question vs within-question decomposition (no difficulty measure used) ==")
    print(Q.set_index(["target", "label"]).drop(columns="band_col").round(3).to_string())

    print("\nwrote", out)


if __name__ == "__main__":
    main()
