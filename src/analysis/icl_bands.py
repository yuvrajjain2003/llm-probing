#!/usr/bin/env python3
"""Round 6, action 2, at zero cost: score Luna and our probe INSIDE each difficulty band.

No new API calls are needed. A query prompt in icl_openai.py does not depend on the difficulty band
(the examples come from example_order over all training-fold questions), so restricting the row set
to one band is exactly the same measurement as rerunning Luna on that band's rows. The stored
predictions already cover all 196 questions.

That also removes the calibration worry in NEXT_STEPS.md: `icl_cal` was fitted per fold on the FULL
other-fold rows, so nothing is refitted on a 3-question cell here. Raw is reported alongside.

    ./envs/goal1/bin/python src/analysis/icl_bands.py [--band-col band_quintile]
Outputs: runs/goal2/gpqa25/sweep/_iclsplit/p5_bands_{scores,paired}.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
from round2_metrics import REG, BIN, TARGETS, KEYS, B, SEED
import icl_decompose as dc

OUT = dc.OUT
LABELS = ["Our model L21 (ridge)", "Prefix length", "Luna k=0", "Luna k=50", "Gemma-4-31B k=0"]
ORDER = ["v_hard", "hard", "medium", "easy", "v_easy"]


def wcorr(y, p, W):
    """Weighted Pearson r of y and p under each row-weight vector in W (B, n). -> (B,)"""
    sw = W.sum(1)
    a = y[None, :] - (W * y).sum(1, keepdims=True) / sw[:, None]
    b = p[None, :] - (W * p).sum(1, keepdims=True) / sw[:, None]
    return (W * a * b).sum(1) / np.sqrt((W * a * a).sum(1) * (W * b * b).sum(1))


def wr2_recentred(y, p, W):
    """Weighted R2 after shifting the prediction so its mean matches the truth's, INSIDE the subset.

    The stored predictions come from a fit on all 198 questions and are never recentred per band, so a
    band score is penalised for an offset the full-set fit never had to carry. Removing one degree of
    freedom (the intercept) is the honest band score; the slope is left alone, so under-confident
    predictions are still penalised, which is the real effect we want to see."""
    sw = W.sum(1)
    my = (W * y).sum(1) / sw
    mp = (W * p).sum(1) / sw
    r = y[None, :] - (p[None, :] - mp[:, None] + my[:, None])
    return 1 - (W * r ** 2).sum(1) / (W * (y[None, :] - my[:, None]) ** 2).sum(1)


def score_rows(P, keep_q, labels):
    """Mean over the five cuts, with question-bootstrap CIs, on the questions in keep_q."""
    draws, cdraws, rdraws, point = {}, {}, {}, []
    for t in TARGETS:
        for f in dc.FRACS:
            x = P[(P.target == t) & (P.fraction.round(3) == f) & (P.example_id.isin(keep_q))]
            wide = x.pivot_table(index=["example_id", "cot_id"], columns="label", values="y_pred").dropna()
            y = (x.drop_duplicates(["example_id", "cot_id"]).set_index(["example_id", "cot_id"])
                 .y_true.loc[wide.index].to_numpy(float))
            eid = wide.index.get_level_values(0).to_numpy()
            qs = np.sort(np.unique(eid)); qi = {int(v): i for i, v in enumerate(qs)}
            Wq = np.random.default_rng(SEED).multinomial(len(qs), np.ones(len(qs)) / len(qs), size=B).astype(float)
            W = Wq[:, [qi[int(e)] for e in eid]]
            for lab in labels:
                p = wide[lab].to_numpy(float)
                if t in REG:
                    draws[(lab, t, f)] = dc.wr2(y, p, W)
                    rdraws[(lab, t, f)] = wr2_recentred(y, p, W)
                    pt = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                    pr = 1 - ((y - (p - p.mean() + y.mean())) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                else:
                    c = ((p >= .5) == y).astype(float)
                    draws[(lab, t, f)] = (W * c).sum(1) / W.sum(1)
                    rdraws[(lab, t, f)] = draws[(lab, t, f)]
                    pt = pr = c.mean()
                r = (np.corrcoef(y, p)[0, 1] if y.std() > 0 and p.std() > 0 else np.nan)
                cdraws[(lab, t, f)] = wcorr(y, p, W)
                point.append(dict(label=lab, target=t, fraction=f, score=pt, score_recentred=pr, n=len(y),
                                  n_questions=len(qs), target_sd=float(y.std()), wcorr=float(r)))
    return pd.DataFrame(point), draws, cdraws, rdraws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band-col", default="band_quintile", choices=["band_quintile", "band_int"])
    a = ap.parse_args()
    dif = pd.read_csv(dc.DIFF).set_index("example_id")
    P = dc.load()
    P = P[P.label.isin(LABELS)]
    sc_all, pr_all = [], []
    groups = [("all questions", dif.index)] + [(b, dif.index[dif[a.band_col] == b]) for b in ORDER]
    sd_all = None
    for band, qs in groups:
        pt, draws, cdraws, rdraws = score_rows(P, set(qs), LABELS)
        m = pt.groupby(["label", "target"]).score.mean()
        mr = pt.groupby(["label", "target"]).score_recentred.mean()
        mc = pt.groupby(["label", "target"]).wcorr.mean()
        sd = pt.groupby("target").target_sd.mean()
        nq = pt.n_questions.min()
        if sd_all is None:
            sd_all = sd
        for lab in LABELS:
            for t in TARGETS:
                d = np.mean([draws[(lab, t, f)] for f in dc.FRACS], 0)
                cd = np.mean([cdraws[(lab, t, f)] for f in dc.FRACS], 0)
                rd = np.mean([rdraws[(lab, t, f)] for f in dc.FRACS], 0)
                sc_all.append(dict(band=band, band_col=a.band_col, n_questions=nq, label=lab, target=t,
                                   score=m.loc[(lab, t)], ci_lo=np.percentile(d, 2.5),
                                   ci_hi=np.percentile(d, 97.5), target_sd=sd.loc[t],
                                   sd_vs_all=sd.loc[t] / sd_all.loc[t], within_cut_corr=mc.loc[(lab, t)],
                                   corr_ci_lo=np.nanpercentile(cd, 2.5),
                                   corr_ci_hi=np.nanpercentile(cd, 97.5),
                                   score_recentred=mr.loc[(lab, t)],
                                   recentred_ci_lo=np.nanpercentile(rd, 2.5),
                                   recentred_ci_hi=np.nanpercentile(rd, 97.5)))
                if lab == dc.OURS:
                    continue
                dd = d - np.mean([draws[(dc.OURS, t, f)] for f in dc.FRACS], 0)
                lo, hi = np.nanpercentile(dd, [2.5, 97.5])
                cdd = cd - np.mean([cdraws[(dc.OURS, t, f)] for f in dc.FRACS], 0)
                clo, chi = np.nanpercentile(cdd, [2.5, 97.5])
                pr_all.append(dict(band=band, n_questions=nq, label=lab, target=t,
                                   diff=m.loc[(lab, t)] - m.loc[(dc.OURS, t)], ci_lo=lo, ci_hi=hi,
                                   significant=bool(lo > 0 or hi < 0),
                                   corr_diff=mc.loc[(lab, t)] - mc.loc[(dc.OURS, t)],
                                   corr_ci_lo=clo, corr_ci_hi=chi,
                                   corr_significant=bool(clo > 0 or chi < 0)))
        print(f"{band:14s} ({nq:3d} questions, {nq*5:4d} rows) done")
    SC = pd.DataFrame(sc_all); PR = pd.DataFrame(pr_all)
    SC.to_csv(OUT / "p5_bands_scores.csv", index=False); PR.to_csv(OUT / "p5_bands_paired.csv", index=False)

    sh = dc.SHORT
    for t in ["answer_entropy_bits_mm", "rem_len_mean"]:
        print(f"\n== {sh[t]}: OFFSET-CORRECTED R2 by difficulty band (mean over cuts; CI half-width) ==")
        x = SC[SC.target == t].copy()
        x["cell"] = x.apply(lambda r: f"{r.score_recentred:+.3f} (+-{(r.recentred_ci_hi - r.recentred_ci_lo) / 2:.2f})", axis=1)
        print(x.pivot_table(index="label", columns="band", values="cell", aggfunc="first")
              .reindex(columns=["all questions"] + ORDER).reindex(LABELS).to_string())
        print(f"-- within-cut correlation (offset-free; R2 above punishes an uncorrected offset), {sh[t]}:")
        print(x.pivot_table(index="label", columns="band", values="within_cut_corr")
              .reindex(columns=["all questions"] + ORDER).reindex(LABELS).round(3).to_string())
        print(f"-- Luna k=0 minus our probe, {sh[t]}:")
        for _, r in PR[(PR.target == t) & (PR.label == "Luna k=0")].iterrows():
            print(f"   {r.band:14s} {r['diff']:+.3f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}] sig={r.significant}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
