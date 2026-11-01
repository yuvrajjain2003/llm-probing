#!/usr/bin/env python3
"""Round 6, concern 2: is the frontier model's tie with our probe also a question-difficulty read?

The ask was to apply round 4's between-question / within-question decomposition
(src/analysis/redundancy.py part C) to GPT-6 Luna's stored predictions. That decomposition cannot be
computed on the ICL row set: it uses 1 CoT per question per cut, so within-question sums of squares
are identically zero and every cut's R2 is 100% between-question by construction. Part 1 below
documents that instead of pretending otherwise.

The free substitute, on all 196 questions rather than a 40-question band: partial question difficulty
out of both the target and each prediction, and see what survives. Difficulty is the 0% cut's p_gold
(fraction of the 25 base CoTs that were correct), quadratic to allow its non-monotonic shape.

    ./envs/goal1/bin/python src/analysis/icl_decompose.py
Outputs under runs/goal2/gpqa25/sweep/_iclsplit/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
from round2_metrics import REG, BIN, TARGETS, KEYS, B, SEED
import master_table as mt

S = Path("runs/goal2/gpqa25/sweep"); OUT = S / "_iclsplit"; OUT.mkdir(exist_ok=True)
DIFF = S / "_difficulty/question_difficulty.csv"
FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]

# label -> (sweep, feature, probe, family).  "icl:<tag>" reads runs/goal2/gpqa25/icl/preds_<tag>.csv
SPEC = {
    "Our model L21 (ridge)":  ("ref_fixed", "act_L21", None, "probe"),
    "Prefix length":          ("ref_fixed", "prefix_len", None, "text baseline"),
    "Luna k=0":               ("icl:gpt6_luna_high_k0", "gpt6_luna_high_k0", "icl_cal", "frontier ICL"),
    "Luna k=10":              ("icl:gpt6_luna_high_k10", "gpt6_luna_high_k10", "icl_cal", "frontier ICL"),
    "Luna k=50":              ("icl:gpt6_luna_high_k50", "gpt6_luna_high_k50", "icl_cal", "frontier ICL"),
    "Luna k=0 blind":         ("icl:gpt6_luna_high_blind_k0", "gpt6_luna_high_blind_k0", "icl_cal", "frontier ICL"),
    "Luna k=10 blind":        ("icl:gpt6_luna_high_blind_k10", "gpt6_luna_high_blind_k10", "icl_cal", "frontier ICL"),
    "Luna k=50 blind":        ("icl:gpt6_luna_high_blind_k50", "gpt6_luna_high_blind_k50", "icl_cal", "frontier ICL"),
    "Luna k=10 nockpt":       ("icl:gpt6_luna_high_nockpt_k10", "gpt6_luna_high_nockpt_k10", "icl_cal", "frontier ICL"),
    "Luna k=50 nockpt":       ("icl:gpt6_luna_high_nockpt_k50", "gpt6_luna_high_nockpt_k50", "icl_cal", "frontier ICL"),
    "Gemma-4-31B k=0":        ("icl:gemma4_31b_off_k0", "gemma4_31b_off_k0", "icl_cal", "local ICL"),
}
OURS, REFS = "Our model L21 (ridge)", ["Our model L21 (ridge)", "Luna k=0"]
SHORT = {"answer_entropy_bits_mm": "entropy", "p_mode": "p_mode", "rem_len_mean": "rem_len_mean",
         "rem_len_std": "rem_len_std", BIN: "committed"}


def load():
    """Predictions for every label on the rows the ICL runs cover, wide by label."""
    rows = []
    for lab, (sw, ft, pr, _) in SPEC.items():
        for t in TARGETS:
            rows.append(mt.preds_for(sw, ft, pr, t).assign(label=lab)[["label", "target"] + KEYS
                                                                      + ["y_true", "y_pred"]])
    P = pd.concat(rows, ignore_index=True)
    P = P[P.fraction > 0]
    base = P[(P.label == "Luna k=0") & (P.target == TARGETS[0])].set_index(KEYS).index
    for lab in SPEC:                                  # coverage, as master_table does it
        cov = min(P[(P.label == lab) & (P.target == t)].set_index(KEYS).index.isin(base).sum()
                  / len(base) for t in TARGETS)
        assert cov >= 0.99, f"{lab} covers only {cov:.1%} of the ICL rows"
    return P


# --------------------------------------------------------------- Part 1: the structural fact
def part1(P):
    """Why round 4's between/within decomposition is degenerate here."""
    full = mt.preds_for("ref_fixed", "act_L21", None, TARGETS[0])
    full = full[full.fraction > 0]
    icl = P[(P.label == "Luna k=0") & (P.target == TARGETS[0])]
    rows = []
    for name, d in [("ICL row set (1 CoT/question)", icl), ("full ref_fixed row set", full)]:
        for f in FRACS:
            x = d[d.fraction.round(3) == f]
            g = x.groupby("example_id").y_true
            n = g.size().to_numpy(float)
            ssw = float(g.apply(lambda v: ((v - v.mean()) ** 2).sum()).sum())
            sst = float(((x.y_true - x.y_true.mean()) ** 2).sum())
            rows.append(dict(row_set=name, fraction=f, n_rows=len(x), n_questions=len(n),
                             rows_per_question_min=n.min(), rows_per_question_max=n.max(),
                             within_question_ss=ssw, total_ss=sst, within_share_of_ss=ssw / sst))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "p1_row_structure.csv", index=False)
    return R


# --------------------------------------------------------------- weighted helpers
def wls_resid(X, y, W):
    """Residuals of y on X under each row-weight vector in W (B, n).  -> (B, n) residuals."""
    XtWX = np.einsum("bn,ni,nj->bij", W, X, X)
    XtWy = np.einsum("bn,ni,n->bi", W, X, y)
    beta = np.linalg.solve(XtWX + 1e-12 * np.eye(X.shape[1]), XtWy)      # (B, p)
    return y[None, :] - beta @ X.T


def wr2(y, p, W):
    """Weighted pooled R2 of prediction p for target y, per weight vector. -> (B,)"""
    sw = W.sum(1)
    my = (W * y).sum(1) / sw
    return 1 - (W * (y - p) ** 2).sum(1) / (W * (y[None, :] - my[:, None]) ** 2).sum(1)


def wr2_resid(ry, rp, W):
    """Same, on already-residualised y and p, which are (B, n)."""
    return 1 - (W * (ry - rp) ** 2).sum(1) / (W * ry ** 2).sum(1)


def wcorr2(ry, rp, W):
    """Squared weighted correlation of two (B, n) residual sets: the best affine rescaling's R2."""
    sw = W.sum(1)
    a, b = ry - (W * ry).sum(1, keepdims=True) / sw[:, None], rp - (W * rp).sum(1, keepdims=True) / sw[:, None]
    return (W * a * b).sum(1) ** 2 / ((W * a * a).sum(1) * (W * b * b).sum(1))


# --------------------------------------------------------------- Part 2: partial difficulty out
def part2(P, dif):
    """Per cut and target: R2 as published, and R2 after removing a quadratic in p_gold from both
    the target and the prediction.  Bootstrap over questions, refitting the quadratic each draw."""
    labels = list(SPEC)
    point, draws = [], {}
    for t in REG:
        for f in FRACS:
            x = P[(P.target == t) & (P.fraction.round(3) == f)]
            wide = x.pivot_table(index=["example_id", "cot_id"], columns="label", values="y_pred").dropna()
            y = (x.drop_duplicates(["example_id", "cot_id"]).set_index(["example_id", "cot_id"])
                 .y_true.loc[wide.index].to_numpy(float))
            eid = wide.index.get_level_values(0).to_numpy()
            pg = dif.p_gold.reindex(eid).to_numpy(float)
            assert np.isfinite(pg).all()
            X = np.column_stack([np.ones_like(pg), pg, pg ** 2])
            qs = np.sort(np.unique(eid)); qi = {int(v): i for i, v in enumerate(qs)}
            Wq = np.random.default_rng(SEED).multinomial(len(qs), np.ones(len(qs)) / len(qs), size=B).astype(float)
            W = np.vstack([np.ones(len(y)), Wq[:, [qi[int(e)] for e in eid]]])   # row 0 = the point estimate
            ry = wls_resid(X, y, W)
            r2_dif = wr2(y, ry[0] * 0 + (y - ry[0]), W)          # R2 of the quadratic itself (in-sample)
            for lab in labels:
                p = wide[lab].to_numpy(float)
                rp = wls_resid(X, p, W)
                full = wr2(y, p, W)
                strict = wr2_resid(ry, rp, W)
                resc = wcorr2(ry, rp, W)
                draws[(lab, t, f, "r2")] = full[1:]
                draws[(lab, t, f, "r2_partialled")] = strict[1:]
                draws[(lab, t, f, "r2_partialled_rescaled")] = resc[1:]
                point.append(dict(label=lab, family=SPEC[lab][3], target=t, fraction=f, n=len(y),
                                  r2=full[0], r2_partialled=strict[0], r2_partialled_rescaled=resc[0],
                                  r2_of_difficulty_alone=r2_dif[0],
                                  corr_pred_pgold=float(np.corrcoef(p, pg)[0, 1])))
    return pd.DataFrame(point), draws, labels


def summarise(point, draws, labels):
    """Mean over the five cuts, with question-bootstrap CIs and paired differences vs the references."""
    m = point.groupby(["label", "target"])[["r2", "r2_partialled", "r2_partialled_rescaled"]].mean()
    sc, pr = [], []
    for lab in labels:
        for t in REG:
            for met in ["r2", "r2_partialled", "r2_partialled_rescaled"]:
                d = np.mean([draws[(lab, t, f, met)] for f in FRACS], 0)
                sc.append(dict(label=lab, family=SPEC[lab][3], target=t, metric=met,
                               value=m.loc[(lab, t), met], ci_lo=np.percentile(d, 2.5),
                               ci_hi=np.percentile(d, 97.5)))
                for ref in REFS:
                    if ref == lab:
                        continue
                    dd = d - np.mean([draws[(ref, t, f, met)] for f in FRACS], 0)
                    lo, hi = np.percentile(dd, [2.5, 97.5])
                    pr.append(dict(label=lab, minus=ref, target=t, metric=met,
                                   diff=m.loc[(lab, t), met] - m.loc[(ref, t), met],
                                   ci_lo=lo, ci_hi=hi, significant=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(sc), pd.DataFrame(pr)


# --------------------------------------------------------------- Part 3: commitment, and agreement
def part3(P, dif, nb=500):
    """committed_by_here: accuracy AND AUC with the majority baseline, plus paired AUC CIs vs ours.
    Accuracy ties across methods here while AUC does not, so both are needed."""
    rows, agree, dr, ptv = [], [], {}, {}
    for f in FRACS:
        x = P[(P.target == BIN) & (P.fraction.round(3) == f)]
        wide = x.pivot_table(index=["example_id", "cot_id"], columns="label", values="y_pred").dropna()
        y = (x.drop_duplicates(["example_id", "cot_id"]).set_index(["example_id", "cot_id"])
             .y_true.loc[wide.index].to_numpy(float))
        maj = max(y.mean(), 1 - y.mean())
        eid = wide.index.get_level_values(0).to_numpy()
        qs = np.sort(np.unique(eid)); qi = {int(v): i for i, v in enumerate(qs)}
        cnt = np.random.default_rng(SEED).multinomial(len(qs), np.ones(len(qs)) / len(qs),
                                                      size=nb)[:, [qi[int(e)] for e in eid]]
        for lab in SPEC:
            p = wide[lab].to_numpy(float)
            ptv[(lab, f)] = roc_auc_score(y, p)
            dr[(lab, f)] = np.array([roc_auc_score(y[i], p[i]) if len(np.unique(y[i])) > 1 else np.nan
                                     for i in (np.repeat(np.arange(len(y)), cnt[b]) for b in range(nb))])
            rows.append(dict(label=lab, fraction=f, n=len(y), accuracy=((p >= .5) == y).mean(),
                             auc=ptv[(lab, f)], majority_baseline=maj))
    auc_ci, base = [], np.mean([dr[(OURS, f)] for f in FRACS], 0)
    for lab in SPEC:
        v = np.mean([ptv[(lab, f)] for f in FRACS]); d = np.mean([dr[(lab, f)] for f in FRACS], 0)
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        dd = np.nanpercentile(d - base, [2.5, 97.5])
        auc_ci.append(dict(label=lab, mean_auc=v, ci_lo=lo, ci_hi=hi,
                           diff_vs_ours=v - np.mean([ptv[(OURS, f)] for f in FRACS]),
                           diff_lo=dd[0], diff_hi=dd[1],
                           significant=bool(lab != OURS and (dd[0] > 0 or dd[1] < 0))))
    for t in REG:
        for f in FRACS:
            x = P[(P.target == t) & (P.fraction.round(3) == f)]
            wide = x.pivot_table(index=["example_id", "cot_id"], columns="label", values="y_pred").dropna()
            pg = dif.p_gold.reindex(wide.index.get_level_values(0)).to_numpy(float)
            for lab in SPEC:
                agree.append(dict(target=t, fraction=f, label=lab,
                                  corr_with_ours=float(np.corrcoef(wide[lab], wide[OURS])[0, 1]),
                                  corr_with_pgold=float(np.corrcoef(wide[lab], pg)[0, 1])))
    return pd.DataFrame(rows), pd.DataFrame(agree), pd.DataFrame(auc_ci)


# --------------------------------------------------------------- Part 4: between vs across-depth
def part4(P):
    """The only 'within-question' dimension the ICL rows have: variation ACROSS DEPTH for one chain.
    Pooled over the five cuts, groups = question.  A depth-only predictor (leave-one-fold-out mean
    target per cut) is included because most across-depth variation is monotone in depth."""
    fold = pd.read_csv(DIFF).set_index("example_id").fold
    rows = []
    for t in REG:
        x = P[(P.target == t)].copy()
        x["fraction"] = x.fraction.round(3)
        wide = x.pivot_table(index=["example_id", "cot_id", "fraction"], columns="label", values="y_pred").dropna()
        y = (x.drop_duplicates(KEYS).set_index(KEYS).y_true.loc[wide.index].to_numpy(float))
        eid = wide.index.get_level_values(0).to_numpy()
        fr = wide.index.get_level_values(2).to_numpy()
        fo = fold.reindex(eid).to_numpy()
        depth = np.empty_like(y)                      # leave-one-fold-out mean per cut
        for g in np.unique(fo):
            for f in FRACS:
                sel, oth = (fo == g) & (fr == f), (fo != g) & (fr == f)
                depth[sel] = y[oth].mean()
        preds = {lab: wide[lab].to_numpy(float) for lab in SPEC}
        preds["depth only (reference)"] = depth
        qy = pd.Series(y).groupby(eid).transform("mean").to_numpy()
        for lab, p in preds.items():
            qp = pd.Series(p).groupby(eid).transform("mean").to_numpy()
            bq = pd.DataFrame(dict(y=qy, p=qp, e=eid)).drop_duplicates("e")
            between = 1 - ((bq.y - bq.p) ** 2).sum() / ((bq.y - bq.y.mean()) ** 2).sum()
            dy, dp = y - qy, p - qp
            rows.append(dict(target=t, label=lab, n=len(y), n_questions=bq.e.nunique(),
                             pooled_r2=1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                             between_q_r2=between, within_q_across_depth_r2=1 - ((dy - dp) ** 2).sum() / (dy ** 2).sum(),
                             within_q_across_depth_pearson=float(np.corrcoef(dy, dp)[0, 1])))
    return pd.DataFrame(rows)


def main():
    dif = pd.read_csv(DIFF).set_index("example_id")
    P = load()
    print("labels:", len(SPEC), " rows per label per target:",
          len(P[(P.label == OURS) & (P.target == TARGETS[0])]))

    R1 = part1(P)
    print("\n== Part 1: row structure ==\n", R1.round(4).to_string(index=False))

    point, draws, labels = part2(P, dif)
    point.to_csv(OUT / "p2_per_checkpoint.csv", index=False)
    SC, PR = summarise(point, draws, labels)
    SC.to_csv(OUT / "p2_ci_scores.csv", index=False); PR.to_csv(OUT / "p2_ci_paired.csv", index=False)
    print("\n== Part 2: R2 and R2 after partialling difficulty (mean over cuts) ==")
    print(SC[SC.metric != "r2_partialled_rescaled"]
          .pivot_table(index="label", columns=["metric", "target"], values="value")
          .round(3).to_string())

    CM, AG, AC = part3(P, dif)
    CM.to_csv(OUT / "p3_committed.csv", index=False); AG.to_csv(OUT / "p3_agreement.csv", index=False)
    AC.to_csv(OUT / "p3_committed_auc_ci.csv", index=False)
    print("\n== Part 3: committed (mean over cuts). Accuracy ties, AUC does not ==")
    print(CM.groupby("label")[["accuracy", "auc", "majority_baseline"]].mean().round(3)
          .join(AC.set_index("label")[["diff_vs_ours", "diff_lo", "diff_hi", "significant"]].round(3))
          .to_string())

    R4 = part4(P)
    R4.to_csv(OUT / "p4_between_across_depth.csv", index=False)
    print("\n== Part 4: between-question vs within-question-across-depth (pooled over cuts) ==")
    print(R4.pivot_table(index="label", columns="target",
                         values=["between_q_r2", "within_q_across_depth_r2"]).round(3).to_string())
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
