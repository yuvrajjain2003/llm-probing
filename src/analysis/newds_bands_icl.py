#!/usr/bin/env python3
"""Score the probe, the frontier model and prefix length INSIDE difficulty bands, on a NEW dataset.

Round 6 did this on GPQA (`gpqa25/sweep/_iclsplit/p5_bands_*.csv`, via src/analysis/icl_bands.py).
`newds_difficulty_scoring.py` repeated it on the new datasets but WITHOUT the frontier model, because
when it was written there were no in-context runs there. P3 has since run GPT-6 Luna on both
(2026-11-02), so this adds that row back.

Rows: the ICL row set, i.e. the 500 questions x 1 chain x 5 cuts that Luna was actually called on, so
every approach is scored on IDENTICAL rows. That is the same choice round 6 made on GPQA and it is
why the probe reads slightly differently here than on the full sweep.

Nothing in icl_bands.py / icl_decompose.py is edited: their module constants are repointed and
`ib.main()` is called, exactly the pattern newds_difficulty_scoring.py uses.

    ./envs/goal1/bin/python src/analysis/newds_bands_icl.py --tag mmlupro_v1 --diff rollout
Outputs: runs/goal2/<tag>/sweep/_iclbands/{p5_bands_scores,p5_bands_paired,p7_band_spread}.<diff>.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from round2_metrics import BIN
import icl_decompose as dc
import icl_bands as ib

OURS = "Our model L21 (ridge)"
LUNA = "Luna k=0"
PLEN = "Prefix length"
TFIDF = "TF-IDF on prefix"
LABELS = [OURS, LUNA, PLEN, TFIDF]
SWEEP_FEAT = {OURS: "act_L21", PLEN: "prefix_len", TFIDF: "tfidf"}
REG_BASE = ["answer_entropy_bits_mm", "answer_entropy_bits_mm_norm", "p_mode",
            "rem_len_mean", "rem_len_std"]
K = ["example_id", "cot_id", "fraction"]


def build(tag):
    """Predictions for every label on the COMMON ICL rows."""
    S = Path("runs/goal2") / tag / "sweep"
    sw = pd.concat([pd.read_csv(x) for x in sorted((S / "ref_fixed").glob("preds*.csv"))],
                   ignore_index=True)
    sw["fraction"] = sw.fraction.round(3)
    sw = sw[sw.fraction > 0]
    reg = [t for t in REG_BASE if t in set(sw.target)]

    icl = pd.read_csv(Path("runs/goal2") / tag / "icl/preds_gpt6_luna_high_k0.csv")
    icl["fraction"] = icl.fraction.round(3)
    icl = icl[(icl.fraction > 0) & (icl.probe == "icl_cal")]          # calibrated, as published
    keys = icl[K].drop_duplicates()
    print(f"{tag}: ICL rows {len(keys)}, questions {keys.example_id.nunique()}")

    rows = []
    for lab in LABELS:
        if lab == LUNA:
            x = icl
        else:
            x = sw[sw.feature == SWEEP_FEAT[lab]]
        for t in reg + [BIN]:
            y = x[x.target == t]
            if not len(y):
                continue
            y = y.drop_duplicates(K).merge(keys, on=K)                 # inner join -> common rows
            rows.append(y.assign(label=lab)[["label", "target"] + K + ["y_true", "y_pred"]])
    P = pd.concat(rows, ignore_index=True)
    # keep only targets every label has, so the pivot in score_rows never drops rows silently
    have = P.groupby("target").label.nunique()
    keep = set(have.index[have == len(LABELS)])
    dropped = sorted(set(P.target) - keep)
    if dropped:
        print(f"  targets without all {len(LABELS)} labels, dropped: {dropped}")
    P = P[P.target.isin(keep)]
    return P, [t for t in reg if t in keep], (BIN in keep)


def spread(P, dif, band_col, reg):
    """Per band: mean and spread of the truth against each predictor (the 'are they just predicting
    the mean?' diagnostic). Same formulas as icl_bands_spread.py."""
    rows = []
    groups = [("all questions", set(dif.index))] + \
             [(b, set(dif.index[dif[band_col] == b])) for b in ib.ORDER]
    for lab in LABELS:
        for t in reg:
            for b, keep in groups:
                acc = {k: [] for k in ("mean_true", "sd_true", "mean_pred", "sd_pred", "corr")}
                for f in dc.FRACS:
                    x = P[(P.target == t) & (P.fraction == f) & (P.label == lab)
                          & (P.example_id.isin(keep))]
                    if len(x) < 5:
                        continue
                    y, p = x.y_true.to_numpy(float), x.y_pred.to_numpy(float)
                    acc["mean_true"].append(y.mean()); acc["sd_true"].append(y.std())
                    acc["mean_pred"].append(p.mean()); acc["sd_pred"].append(p.std())
                    acc["corr"].append(np.corrcoef(y, p)[0, 1] if y.std() > 0 and p.std() > 0 else np.nan)
                if not acc["mean_true"]:
                    continue
                r = {k: float(np.nanmean(v)) for k, v in acc.items()}
                r["sd_ratio"] = r["sd_pred"] / r["sd_true"] if r["sd_true"] else np.nan
                rows.append(dict(label=lab, target=t, band=b, band_col=band_col,
                                 n_questions=len(keep), **r))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--diff", default="rollout", choices=["rollout", "qtext"])
    ap.add_argument("--band-col", default="band_quintile")
    a = ap.parse_args()
    S = Path("runs/goal2") / a.tag / "sweep"
    out = S / "_iclbands"; out.mkdir(parents=True, exist_ok=True)
    diff_path = (S / "_difficulty/question_difficulty.csv") if a.diff == "rollout" \
        else (S / "_qtext/question_difficulty.csv")
    dif = pd.read_csv(diff_path).set_index("example_id")
    print(f"difficulty: {diff_path}  measure={dif.difficulty_measure.iloc[0]}")

    P, reg, has_bin = build(a.tag)
    targets = reg + ([BIN] if has_bin else [])
    dc.OUT, dc.DIFF = out, diff_path
    dc.load = lambda: P
    ib.OUT, ib.LABELS, ib.TARGETS, ib.REG = out, LABELS, targets, reg
    sys.argv = ["icl_bands.py", "--band-col", a.band_col]
    ib.main()

    sp = spread(P, dif, a.band_col, reg)
    sp.to_csv(out / f"p7_band_spread.{a.diff}.csv", index=False)
    for src in ["p5_bands_scores.csv", "p5_bands_paired.csv"]:
        f = out / src
        if f.exists():
            f.rename(out / f"{f.stem}.{a.diff}.csv")
    print("wrote", out)


if __name__ == "__main__":
    main()
