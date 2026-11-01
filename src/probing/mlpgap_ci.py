#!/usr/bin/env python3
"""Paired question-bootstrap CIs for the _mlpgap variants, against ridge on the same rows/folds.

Method identical to R5 / round3_ci.py: stored out-of-fold predictions, rows common to every
variant compared, 2,000 SHARED question-resampling draws (so differences are paired), pooled
R2 (reg) / accuracy (bin). No refitting.

The reference is the `ridge` variant refit inside _mlpgap when it is present (identical rows,
folds and alpha selection); otherwise ref_fixed's act_L21 ridge/logistic predictions.

USAGE  $PY src/probing/mlpgap_ci.py [--tags b2,ladder] [--ref ridge] [--fractions 0.5]
OUT    runs/goal2/gpqa25/sweep/_mlpgap/{ci_scores,ci_paired}<suffix>.csv
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.mlpgap_core import OUT, REG, TARGETS, ref_preds, print  # noqa: E402

B, SEED = 2000, 0
KEYS = ["example_id", "cot_id", "fraction"]


def load(tags):
    fs = []
    for t in tags:
        p = OUT / f"preds.{t}.csv"
        if p.exists():
            fs.append(pd.read_csv(p))
        else:
            print(f"  (missing {p})")
    assert fs, "no preds files"
    P = pd.concat(fs, ignore_index=True)
    P["fraction"] = P.fraction.round(3)
    return P.drop_duplicates(["variant", "target"] + KEYS, keep="last")


def add_ref(P):
    """ref_fixed ridge/logistic as the variant `ref_ridge`, restricted to what P covers."""
    R = ref_preds()
    R = R[R.probe.isin(["ridge_act", "logistic"])].copy()
    R["variant"] = "ref_ridge"
    keep = R.target.isin(P.target.unique()) & R.fraction.isin(P.fraction.unique())
    return pd.concat([P, R[keep][["variant", "target"] + KEYS + ["y_true", "y_pred"]]],
                     ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="b2,ladder")
    ap.add_argument("--ref", default="ridge")
    ap.add_argument("--fractions", default="")
    ap.add_argument("--targets", default="")
    ap.add_argument("--variants", default="", help="restrict to these (comma list)")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--per-fraction", action="store_true",
                    help="also write a paired CI for EACH checkpoint, not just the pooled mean")
    a = ap.parse_args()

    P = add_ref(load(a.tags.split(",")))
    if a.fractions:
        P = P[P.fraction.isin([round(float(x), 3) for x in a.fractions.split(",")])]
    if a.targets:
        P = P[P.target.isin(a.targets.split(","))]
    if a.variants:
        keep = set(a.variants.split(",")) | {a.ref}
        P = P[P.variant.isin(keep)]
    variants = [v for v in P.variant.unique()]
    assert a.ref in variants, f"reference {a.ref} not in {variants}"
    fracs = sorted(P.fraction.unique())
    tgts = [t for t in TARGETS if t in set(P.target)]
    print(f"  variants {variants}\n  fractions {fracs}\n  targets {tgts}")

    qs = np.sort(P.example_id.unique()); Q = len(qs); qi = {int(v): i for i, v in enumerate(qs)}
    W = np.random.default_rng(SEED).multinomial(Q, np.ones(Q) / Q, size=B).astype(float)
    draws, sc = {}, []
    for t in tgts:
        for f in fracs:
            x = P[(P.target == t) & (P.fraction == f)]
            if not len(x):
                continue
            have = [v for v in variants if (x.variant == v).any()]
            wide = x.pivot_table(index=["example_id", "cot_id"], columns="variant",
                                 values="y_pred")[have].dropna()
            if not len(wide):
                continue
            yt = (x.drop_duplicates(["example_id", "cot_id"])
                   .set_index(["example_id", "cot_id"]).y_true.loc[wide.index].to_numpy())
            qq = np.array([qi[int(e)] for e in wide.index.get_level_values(0)])
            agg = lambda v: np.bincount(qq, weights=v, minlength=Q)
            n1, s1, s2 = agg(np.ones(len(yt))), agg(yt), agg(yt * yt)
            for v in have:
                p = wide[v].to_numpy()
                if t in REG:
                    se = agg((p - yt) ** 2)
                    draws[(v, t, f)] = 1 - (W @ se) / ((W @ s2) - (W @ s1) ** 2 / (W @ n1))
                    pt = 1 - se.sum() / (s2.sum() - s1.sum() ** 2 / n1.sum())
                else:
                    c = agg(((p >= .5) == yt).astype(float))
                    draws[(v, t, f)] = (W @ c) / (W @ n1)
                    pt = c.sum() / n1.sum()
                sc.append(dict(variant=v, target=t, fraction=f, score=pt, n=len(yt)))
    S = pd.DataFrame(sc)
    out_s, out_p = [], []
    for t in tgts:
        # Only average over checkpoints where the REFERENCE exists, and only compare variants
        # present at every one of those. Without this a partially-written results file raises a
        # KeyError on the paired lookup instead of skipping the incomplete cell.
        ff = sorted(f for f in S[S.target == t].fraction.unique() if (a.ref, t, f) in draws)
        if not ff:
            print(f"  skip {t}: reference {a.ref} has no checkpoints")
            continue
        for v in variants:
            have = [f for f in ff if (v, t, f) in draws]
            if len(have) != len(ff):
                print(f"  skip {v} on {t}: {len(have)}/{len(ff)} checkpoints")
                continue
            d = np.mean([draws[(v, t, f)] for f in have], 0)
            # the point estimate must average over EXACTLY the checkpoints the CI averages over,
            # or diff and its interval describe different quantities
            m = S[(S.variant == v) & (S.target == t) & S.fraction.isin(have)].score.mean()
            out_s.append(dict(variant=v, target=t, n_fractions=len(have),
                              fractions="/".join(str(f) for f in have), mean_score=m,
                              ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5)))
            if v == a.ref:
                continue
            dd = d - np.mean([draws[(a.ref, t, f)] for f in have], 0)
            mr = S[(S.variant == a.ref) & (S.target == t)
                   & S.fraction.isin(have)].score.mean()
            lo, hi = np.percentile(dd, [2.5, 97.5])
            out_p.append(dict(variant=v, minus=a.ref, target=t, n_fractions=len(have),
                              score=m, ref_score=mr, diff=m - mr, ci_lo=lo, ci_hi=hi,
                              significant=bool(lo > 0 or hi < 0)))
    sfx = a.suffix
    if a.per_fraction:
        pf = []
        for t in tgts:
            for f in sorted(S[S.target == t].fraction.unique()):
                if (a.ref, t, f) not in draws:
                    continue
                for v in variants:
                    if v == a.ref or (v, t, f) not in draws:
                        continue
                    dd = draws[(v, t, f)] - draws[(a.ref, t, f)]
                    g = lambda vv: S[(S.variant == vv) & (S.target == t)
                                     & (S.fraction == f)].score.iloc[0]
                    lo, hi = np.percentile(dd, [2.5, 97.5])
                    pf.append(dict(variant=v, minus=a.ref, target=t, fraction=f,
                                   score=g(v), ref_score=g(a.ref), diff=g(v) - g(a.ref),
                                   ci_lo=lo, ci_hi=hi, significant=bool(lo > 0 or hi < 0)))
        P2 = pd.DataFrame(pf)
        P2.to_csv(OUT / f"ci_perfraction{sfx}.csv", index=False)
        P2["cell"] = [f"{d:+.4f} [{l:+.4f},{h:+.4f}]{'*' if sg else ''}"
                      for d, l, h, sg in zip(P2["diff"], P2.ci_lo, P2.ci_hi, P2.significant)]
        print(f"\n== PER CHECKPOINT: variant minus {a.ref}")
        for t in tgts:
            x = P2[P2.target == t]
            if len(x):
                print(f"-- {t}")
                print(x.pivot(index="variant", columns="fraction", values="cell").to_string())
    pd.DataFrame(out_s).to_csv(OUT / f"ci_scores{sfx}.csv", index=False)
    pr = pd.DataFrame(out_p); pr.to_csv(OUT / f"ci_paired{sfx}.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"\n== scores (per variant, averaged over the checkpoints it shares with {a.ref}; "
          f"see the 'fractions' column of ci_scores{sfx}.csv)")
    print(pd.DataFrame(out_s).pivot(index="variant", columns="target", values="mean_score")
          .reindex(columns=tgts).round(4).to_string())
    pr["cell"] = [f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if s else ''}"
                  for d, lo, hi, s in zip(pr["diff"], pr.ci_lo, pr.ci_hi, pr.significant)]
    print(f"\n== variant minus {a.ref}  (* = 95% paired CI excludes 0)")
    print(pr.pivot(index="variant", columns="target", values="cell").reindex(columns=tgts).to_string())
    print(f"\n  wrote {OUT/f'ci_paired{sfx}.csv'}")


if __name__ == "__main__":
    main()
