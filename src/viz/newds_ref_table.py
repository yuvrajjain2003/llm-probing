#!/usr/bin/env python3
"""P1.4: question-bootstrap CIs for the ref_fixed probe reference on a NEW dataset.

Reuses src/viz/master_table.py's `main()` unchanged (same 2000 shared question-resampling draws,
pooled R2 / accuracy per cut, mean over the five cuts, paired differences) by pointing its sweep
root at the new dataset's folder. Nothing is refitted.

Two blocks are added on top, because master_table does not cover them and the project's reporting
conventions require them:
  * `committed_by_here` AUC with the majority-class baseline, with paired CIs, using exactly the
    repeat-the-rows bootstrap of src/analysis/icl_decompose.py part 3.
  * `answer_entropy_bits_mm_norm` pooled R2 (n_options varies on these datasets), same weights.

    ./envs/goal1/bin/python src/viz/newds_ref_table.py --tag mmlupro_v1
Outputs: runs/goal2/<tag>/sweep/_ref_ci/{table.md, ci_scores.csv, ci_paired.csv,
         per_checkpoint.csv, committed_auc.csv, entropy_norm.csv}
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import master_table as MT
from round2_metrics import KEYS, B, SEED

OURS = "Our model L21 (ridge)"
SPEC = {
    OURS:               ("ref_fixed", "act_L21", None, "ours"),
    "TF-IDF on prefix": ("ref_fixed", "tfidf", None, "text baseline"),
    "Prefix length":    ("ref_fixed", "prefix_len", None, "text baseline"),
}
REFS = [OURS, "TF-IDF on prefix", "Prefix length"]
NORM = "answer_entropy_bits_mm_norm"
BIN = "committed_by_here"


def long_preds(S, target):
    """label -> stored out-of-fold predictions for one target, as one long frame."""
    rows = []
    for lab, (sw, ft, pr, _) in SPEC.items():
        d = MT.preds_for(sw, ft, pr, target)
        rows.append(d.assign(label=lab)[["label"] + KEYS + ["y_true", "y_pred"]])
    P = pd.concat(rows, ignore_index=True)
    return P[P.fraction > 0]


def weights(eid):
    qs = np.sort(np.unique(eid))
    qi = {int(v): i for i, v in enumerate(qs)}
    idx = np.array([qi[int(e)] for e in eid])
    return qs, idx


def committed_auc(S, out, nb=500):
    """Accuracy, AUC and the majority-class baseline per cut, plus paired AUC CIs (nb draws).
    Same estimator as icl_decompose.part3: resample questions, repeat their rows, score."""
    P = long_preds(S, BIN)
    fracs = sorted(P.fraction.unique())
    rows, dr, ptv = [], {}, {}
    for f in fracs:
        x = P[P.fraction == f]
        wide = x.pivot_table(index=["example_id", "cot_id"], columns="label", values="y_pred").dropna()
        y = (x.drop_duplicates(["example_id", "cot_id"]).set_index(["example_id", "cot_id"])
             .y_true.loc[wide.index].to_numpy(float))
        maj = max(y.mean(), 1 - y.mean())
        eid = wide.index.get_level_values(0).to_numpy()
        qs, idx = weights(eid)
        cnt = np.random.default_rng(SEED).multinomial(len(qs), np.ones(len(qs)) / len(qs), size=nb)[:, idx]
        boot = [np.repeat(np.arange(len(y)), cnt[b]) for b in range(nb)]
        for lab in SPEC:
            p = wide[lab].to_numpy(float)
            ptv[(lab, f)] = roc_auc_score(y, p)
            dr[(lab, f)] = np.array([roc_auc_score(y[i], p[i]) if len(np.unique(y[i])) > 1 else np.nan
                                     for i in boot])
            rows.append(dict(label=lab, fraction=f, n=len(y), accuracy=float(((p >= .5) == y).mean()),
                             auc=ptv[(lab, f)], majority_baseline=maj))
    CM = pd.DataFrame(rows)
    ci = []
    for lab in SPEC:
        v = float(np.mean([ptv[(lab, f)] for f in fracs]))
        d = np.mean([dr[(lab, f)] for f in fracs], 0)
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        rec = dict(label=lab, mean_auc=v, ci_lo=lo, ci_hi=hi,
                   mean_accuracy=float(CM[CM.label == lab].accuracy.mean()),
                   mean_majority=float(CM.majority_baseline.mean()))
        for ref in REFS:
            if ref == lab:
                continue
            dd = np.nanpercentile(d - np.mean([dr[(ref, f)] for f in fracs], 0), [2.5, 97.5])
            rec[f"auc_minus_{ref}"] = v - float(np.mean([ptv[(ref, f)] for f in fracs]))
            rec[f"lo_minus_{ref}"], rec[f"hi_minus_{ref}"] = dd[0], dd[1]
            rec[f"sig_vs_{ref}"] = bool(dd[0] > 0 or dd[1] < 0)
        ci.append(rec)
    AC = pd.DataFrame(ci)
    CM.to_csv(out / "committed_auc_per_checkpoint.csv", index=False)
    AC.to_csv(out / "committed_auc.csv", index=False)
    return CM, AC


def norm_entropy(S, out):
    """Pooled R2 on the n_options-normalised entropy, same question bootstrap as master_table."""
    P = long_preds(S, NORM)
    fracs = sorted(P.fraction.unique())
    draws, pt = {}, {}
    for f in fracs:
        x = P[P.fraction == f]
        wide = x.pivot_table(index=["example_id", "cot_id"], columns="label", values="y_pred").dropna()
        yt = (x.drop_duplicates(["example_id", "cot_id"]).set_index(["example_id", "cot_id"])
              .y_true.loc[wide.index].to_numpy(float))
        eid = wide.index.get_level_values(0).to_numpy()
        qs, idx = weights(eid)
        W = np.random.default_rng(SEED).multinomial(len(qs), np.ones(len(qs)) / len(qs),
                                                    size=B).astype(float)
        a = lambda v: np.bincount(idx, weights=v, minlength=len(qs))
        n, s1, s2 = a(np.ones(len(yt))), a(yt), a(yt * yt)
        for lab in SPEC:
            se = a((wide[lab].to_numpy(float) - yt) ** 2)
            draws[(lab, f)] = 1 - (W @ se) / ((W @ s2) - (W @ s1) ** 2 / (W @ n))
            pt[(lab, f)] = 1 - se.sum() / (s2.sum() - s1.sum() ** 2 / n.sum())
    rows = []
    for lab in SPEC:
        v = float(np.mean([pt[(lab, f)] for f in fracs]))
        d = np.mean([draws[(lab, f)] for f in fracs], 0)
        rec = dict(label=lab, target=NORM, mean_score=v,
                   ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5))
        for ref in REFS:
            if ref == lab:
                continue
            dd = np.percentile(d - np.mean([draws[(ref, f)] for f in fracs], 0), [2.5, 97.5])
            rec[f"minus_{ref}"] = v - float(np.mean([pt[(ref, f)] for f in fracs]))
            rec[f"lo_minus_{ref}"], rec[f"hi_minus_{ref}"] = dd[0], dd[1]
            rec[f"sig_vs_{ref}"] = bool(dd[0] > 0 or dd[1] < 0)
        rows.append(rec)
    NE = pd.DataFrame(rows)
    NE.to_csv(out / "entropy_norm.csv", index=False)
    pc = pd.DataFrame([dict(label=l, fraction=f, score=pt[(l, f)]) for l in SPEC for f in fracs])
    pc.to_csv(out / "entropy_norm_per_checkpoint.csv", index=False)
    return NE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", help="override the output dir (used to validate against GPQA)")
    a = ap.parse_args()
    S = Path(f"runs/goal2/{a.tag}/sweep")
    MT.S = S                                   # master_table.preds_for reads this module global
    out = Path(a.out) if a.out else S / "_ref_ci"
    out.mkdir(parents=True, exist_ok=True)
    MT.main(SPEC=SPEC, REFS=REFS, OUT=out, base_label=OURS,
            title=f"Probe reference, {a.tag} (mean over the 10-90% cut points)")
    CM, AC = committed_auc(S, out)
    print("\ncommitted_by_here, mean over cuts:")
    print(AC[["label", "mean_accuracy", "mean_auc", "ci_lo", "ci_hi", "mean_majority"]].round(4).to_string(index=False))
    if any(NORM in pd.read_csv(f).target.unique()
           for f in sorted((S / "ref_fixed").glob("results*.csv"))):
        NE = norm_entropy(S, out)
        print(f"\n{NORM}:")
        print(NE[["label", "mean_score", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
    else:
        print(f"\n{NORM} not in this sweep; skipped")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
