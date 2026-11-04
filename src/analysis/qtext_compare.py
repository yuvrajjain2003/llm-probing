#!/usr/bin/env python3
"""Score the question-text-only baseline against the probe and the prefix-text baselines.

Reads stored out-of-fold predictions only; nothing is refitted.  Rows common to every label, scored
WITHIN each cut point and then averaged (never pooled across depth), with question-bootstrap CIs and
paired differences against our probe and against the prefix-TF-IDF baseline.  `committed_by_here`
gets AUC next to accuracy plus the majority-class baseline.

The CI machinery is imported from src/analysis/icl_decompose.py (`wr2`, and the same multinomial
question-weight draws every round-6 number used); the AUC bootstrap follows that file's part 3.
A wide predictions file is also written so src/probing/bootstrap_ci.py can be run unchanged as an
independent check of the per-cut paired intervals.

    ./envs/goal1/bin/python src/analysis/qtext_compare.py --tag mmlupro_v1
Outputs: runs/goal2/<tag>/sweep/_qtext/{cmp_scores,cmp_paired,cmp_per_checkpoint}.csv, wide_preds.csv
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))
from analysis.icl_decompose import wr2            # the project's question-bootstrap R2
from analysis.qtext_baseline import BIN, FRACS, NORM, REG

B, SEED, NB_AUC = 2000, 0, 500
KEYS = ["example_id", "cot_id", "fraction"]
OURS, TEXT = "Our model L21", "TF-IDF on prefix"
SHORT = {"answer_entropy_bits_mm": "entropy", NORM: "entropy_norm", "p_mode": "p_mode",
         "rem_len_mean": "rem_len_mean", "rem_len_std": "rem_len_std", BIN: "committed"}


def load(tag):
    """-> long predictions, one row per (label, target, key)."""
    S = Path(f"runs/goal2/{tag}/sweep")
    spec = {                                        # label -> (dir, feature, reg probe, bin probe)
        OURS:                      ("ref_fixed", "act_L21", "ridge_act", "logistic_deep"),
        TEXT:                      ("ref_fixed", "tfidf", "ridge_text", "logistic_deep"),
        "Prefix length":           ("ref_fixed", "prefix_len", "ridge_len", "logistic_deep"),
        "Question text only":      ("_qtext", "qtext_q", "ridge_qtext", "logistic_qtext"),
        "Question text + choices": ("_qtext", "qtext_qc", "ridge_qtext", "logistic_qtext"),
    }
    cache, rows, missing = {}, [], []
    for lab, (sub, ft, rp, bp) in spec.items():
        for sd in [sub] + (["ref_logistic_deep"] if bp == "logistic_deep" else []):
            if sd not in cache:
                fs = sorted((S / sd).glob("preds*.csv"))
                cache[sd] = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None
        for t in REG + [NORM, BIN]:
            sd = "ref_logistic_deep" if (t == BIN and bp == "logistic_deep") else sub
            d, pr = cache[sd], (bp if t == BIN else rp)
            if d is None:
                missing.append((lab, t, f"{sd} has no preds"))
                continue
            x = d[(d.feature == ft) & (d.probe == pr) & (d.target == t)]
            if not len(x):
                missing.append((lab, t, f"{sd}/{ft}/{pr}"))
                continue
            x = x.drop_duplicates(KEYS).copy()
            x["fraction"] = x.fraction.round(3)
            rows.append(x.assign(label=lab)[["label", "target"] + KEYS + ["y_true", "y_pred"]])
    P = pd.concat(rows, ignore_index=True)
    return P[P.fraction.isin(FRACS)], missing, list(spec)


def f0_compare(tag, out):
    """The 0% cut: both predictors see only the question, one row per question.

    The activation probe reads the model's internal state on the rendered prompt; this baseline
    reads the question's words.  The gap is what the representation knows about a question beyond
    what its statement says.  prefix_len is kept as the degenerate control (it must score ~0)."""
    S = Path(f"runs/goal2/{tag}/sweep")
    fp = S / "f0_probe"
    if not (fp / "results.csv").exists():
        print("  no f0_probe results; 0% comparison skipped")
        return None
    pf = pd.concat([pd.read_csv(x) for x in sorted(fp.glob("preds*.csv"))], ignore_index=True)
    pq = pd.read_csv(S / "_qtext/preds.csv")
    spec = {"Our model L21 @0%": (pf, "act_L21", "ridge_act"),
            "Prefix length @0% (control)": (pf, "prefix_len", "ridge_len"),
            "Question text only @0%": (pq, "qtext_q", "ridge_qtext"),
            "Question text + choices @0%": (pq, "qtext_qc", "ridge_qtext")}
    rows = []
    for t in REG + [NORM]:
        wide, y, eid = {}, None, None
        for lab, (d, ft, pr) in spec.items():
            x = d[(d.feature == ft) & (d.probe == pr) & (d.target == t)
                  & (d.fraction.round(3) == 0.0)].drop_duplicates("example_id")
            if not len(x):
                wide = {}
                break
            x = x.sort_values("example_id")
            wide[lab] = x.y_pred.to_numpy(float)
            y, eid = x.y_true.to_numpy(float), x.example_id.to_numpy()
        if not wide:
            continue
        n = len(y)
        W = np.random.default_rng(SEED).multinomial(n, np.ones(n) / n, size=B).astype(float)
        dr = {lab: wr2(y, p, W) for lab, p in wide.items()}
        pt = {lab: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum() for lab, p in wide.items()}
        for lab in spec:
            lo, hi = np.percentile(dr[lab], [2.5, 97.5])
            r = dict(target=t, label=lab, n_questions=n, r2=pt[lab], ci_lo=lo, ci_hi=hi)
            if lab != "Our model L21 @0%":
                dd = dr[lab] - dr["Our model L21 @0%"]
                dlo, dhi = np.percentile(dd, [2.5, 97.5])
                r.update(diff_vs_probe=pt[lab] - pt["Our model L21 @0%"], diff_lo=dlo, diff_hi=dhi,
                         significant=bool(dlo > 0 or dhi < 0),
                         share_of_probe=pt[lab] / pt["Our model L21 @0%"]
                         if pt["Our model L21 @0%"] > 0 else np.nan)
            rows.append(r)
    F = pd.DataFrame(rows)
    F.to_csv(out / "cmp_f0.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n== 0% cut: question text vs the probe on the SAME single row per question ==")
    print(F.pivot_table(index="label", columns="target", values="r2")
          .rename(columns=SHORT).reindex(list(spec)).round(3).to_string())
    print("-- minus the 0% probe (* = 95% CI excludes 0)")
    x = F[F.label != "Our model L21 @0%"].copy()
    x["cell"] = [f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if s else ''}"
                 for d, lo, hi, s in zip(x.diff_vs_probe, x.diff_lo, x.diff_hi, x.significant)]
    print(x.pivot_table(index="label", columns="target", values="cell", aggfunc="first")
          .rename(columns=SHORT).to_string())
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out = Path(f"runs/goal2/{a.tag}/sweep/_qtext")
    P, missing, labels = load(a.tag)
    targets = [t for t in REG + [NORM, BIN]
               if set(labels) <= set(P[P.target == t].label.unique())]
    for lab, t, why in missing:
        print(f"  MISSING {lab} / {t}: {why}")
    print(f"  targets scored: {[SHORT[t] for t in targets]}")

    point, draws, auc_pt, auc_dr = [], {}, {}, {}
    for t in targets:
        for f in FRACS:
            x = P[(P.target == t) & (P.fraction == f)]
            wide = x.pivot_table(index=["example_id", "cot_id"], columns="label",
                                 values="y_pred").dropna()
            if not len(wide):
                print(f"  no common rows for {SHORT[t]} at {f}")
                continue
            y = (x.drop_duplicates(["example_id", "cot_id"]).set_index(["example_id", "cot_id"])
                 .y_true.loc[wide.index].to_numpy(float))
            n_ours = (x.label == OURS).sum()
            assert len(wide) >= 0.98 * n_ours, (t, f, len(wide), n_ours)
            eid = wide.index.get_level_values(0).to_numpy()
            qs = np.sort(np.unique(eid)); qi = {int(v): i for i, v in enumerate(qs)}
            col = [qi[int(e)] for e in eid]
            rng = np.random.default_rng(SEED)
            W = rng.multinomial(len(qs), np.ones(len(qs)) / len(qs), size=B).astype(float)[:, col]
            cnt = np.random.default_rng(SEED).multinomial(
                len(qs), np.ones(len(qs)) / len(qs), size=NB_AUC)[:, col]
            maj = float(max(y.mean(), 1 - y.mean())) if t == BIN else np.nan
            for lab in labels:
                p = wide[lab].to_numpy(float)
                if t == BIN:
                    c = ((p >= .5) == y).astype(float)
                    draws[(lab, t, f)] = (W * c).sum(1) / W.sum(1)
                    pt = float(c.mean())
                    auc_pt[(lab, f)] = (float(roc_auc_score(y, p))
                                        if len(np.unique(y)) > 1 else np.nan)
                    auc_dr[(lab, f)] = np.array(
                        [roc_auc_score(y[i], p[i]) if len(np.unique(y[i])) > 1 else np.nan
                         for i in (np.repeat(np.arange(len(y)), cnt[b]) for b in range(NB_AUC))])
                else:
                    draws[(lab, t, f)] = wr2(y, p, W)
                    pt = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                point.append(dict(label=lab, target=t, fraction=f, n=len(y), n_questions=len(qs),
                                  score=pt, auc=auc_pt.get((lab, f), np.nan),
                                  majority_baseline=maj))
    PT = pd.DataFrame(point)
    PT.to_csv(out / "cmp_per_checkpoint.csv", index=False)

    m = PT.groupby(["label", "target"]).score.mean()
    sc, pr = [], []
    for lab in labels:
        for t in targets:
            fr = sorted(PT[(PT.label == lab) & (PT.target == t)].fraction.unique())
            if not fr:
                continue
            d = np.mean([draws[(lab, t, f)] for f in fr], 0)
            r = dict(label=lab, target=t, metric="acc" if t == BIN else "R2",
                     score=m.loc[(lab, t)], ci_lo=np.percentile(d, 2.5),
                     ci_hi=np.percentile(d, 97.5))
            if t == BIN:
                ad = np.mean([auc_dr[(lab, f)] for f in fr], 0)
                r.update(auc=float(np.mean([auc_pt[(lab, f)] for f in fr])),
                         auc_ci_lo=np.nanpercentile(ad, 2.5), auc_ci_hi=np.nanpercentile(ad, 97.5),
                         majority_baseline=float(PT[(PT.target == t)].majority_baseline.mean()))
            sc.append(r)
            for ref in [OURS, TEXT]:
                if ref == lab:
                    continue
                dd = d - np.mean([draws[(ref, t, f)] for f in fr], 0)
                lo, hi = np.nanpercentile(dd, [2.5, 97.5])
                q = dict(label=lab, minus=ref, target=t,
                         diff=m.loc[(lab, t)] - m.loc[(ref, t)], ci_lo=lo, ci_hi=hi,
                         significant=bool(lo > 0 or hi < 0))
                if t == BIN:
                    add = np.mean([auc_dr[(lab, f)] for f in fr], 0) - np.mean(
                        [auc_dr[(ref, f)] for f in fr], 0)
                    alo, ahi = np.nanpercentile(add, [2.5, 97.5])
                    q.update(auc_diff=float(np.mean([auc_pt[(lab, f)] for f in fr])
                                            - np.mean([auc_pt[(ref, f)] for f in fr])),
                             auc_ci_lo=alo, auc_ci_hi=ahi,
                             auc_significant=bool(alo > 0 or ahi < 0))
                pr.append(q)
    SC, PR = pd.DataFrame(sc), pd.DataFrame(pr)
    SC.to_csv(out / "cmp_scores.csv", index=False)
    PR.to_csv(out / "cmp_paired.csv", index=False)

    # wide file in src/probing/bootstrap_ci.py's format, for an independent check
    w = None
    for t in [x for x in targets if x != BIN]:
        x = P[P.target == t]
        base = x[x.label == "Question text + choices"].set_index(KEYS)
        part = pd.DataFrame({f"{t}__true": base.y_true, f"{t}__pred": base.y_pred})
        for nm, lab in [("probe", OURS), ("text", TEXT), ("prefixlen", "Prefix length")]:
            o = x[x.label == lab].set_index(KEYS)
            part[f"{t}__pred_{nm}"] = o.y_pred
        w = part if w is None else w.join(part, how="outer")
    w.reset_index().to_csv(out / "wide_preds.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n== mean over 10-90%, within-cut scores ==")
    print(SC.pivot_table(index="label", columns="target", values="score")
          .rename(columns=SHORT).reindex(labels).round(3).to_string())
    if BIN in targets:
        print("\n== committed: accuracy, AUC, majority baseline ==")
        print(SC[SC.target == BIN].set_index("label")[
            ["score", "auc", "majority_baseline"]].reindex(labels).round(3).to_string())
    for ref in [OURS, TEXT]:
        x = PR[PR.minus == ref].copy()
        x["cell"] = [f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if s else ''}"
                     for d, lo, hi, s in zip(x["diff"], x.ci_lo, x.ci_hi, x.significant)]
        print(f"\n== label minus {ref} (mean over 10-90%; * = 95% CI excludes 0) ==")
        print(x.pivot_table(index="label", columns="target", values="cell", aggfunc="first")
              .rename(columns=SHORT).round(3).to_string())
    f0_compare(a.tag, out)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
