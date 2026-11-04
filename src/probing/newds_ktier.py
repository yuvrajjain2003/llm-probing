#!/usr/bin/env python3
"""P1.3: does the two-tier K (20% of rows at the pilot's K=100, the rest at K=40/30) matter?

One cut point only. Two separate questions, both answered without touching the main run:

  A. SCORE side, no refitting. Split the stored out-of-fold predictions of `ref_fixed` at this cut
     into the K=100 rows and the K=low rows and score each, with its own question-bootstrap CI.
     A gap here is label noise (K=low labels are noisier estimates of the same quantity).

  B. TRAIN side, one refit. Refit the same probes with the same grids and the same fixed folds on
     the K=low rows ONLY, then compare to the main run's predictions RESTRICTED TO THOSE SAME ROWS,
     paired over the same bootstrap draws. That is the "go uniform" option.

Reuses sweep.py's own helpers (load_meta / Act / fixed_fold_of / Views / run_combo), the pattern
src/probing/r3_rf_diagnosis.py uses, so sweep.py itself is untouched.

    ./envs/goal1/bin/python src/probing/newds_ktier.py --tag mmlupro_v1 --ds mmlu_pro --cut 0.5
Outputs: runs/goal2/<tag>/sweep/_ktier/{a_score_split.csv, b_refit.csv, b_paired.csv}
"""
from __future__ import annotations

import argparse, glob, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.sweep import (load_meta, Act, PrefixLen, fixed_fold_of, _fixed_folds,  # noqa: E402
                           Views, run_combo, expand, score, PROBES)

KEYS = ["example_id", "cot_id", "fraction"]
REG = ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std"]
BIN = "committed_by_here"
B, SEED = 2000, 0
# same grids as configs/sweep_ref_<tag>.yaml
GRIDS = {"ridge_act": ("ridge", {"alpha": {"logspace": [0, 6, 13]}}),
         "logistic": ("logistic", {"C": {"logspace": [-4, 1, 6]}})}


def pooled_draws(y, p, eid, W, qi, binary):
    idx = np.array([qi[int(e)] for e in eid])
    a = lambda v: np.bincount(idx, weights=v, minlength=W.shape[1])
    if binary:
        n, c = a(np.ones(len(y))), a(((p >= .5) == y).astype(float))
        return (W @ c) / (W @ n), float(c.sum() / n.sum())
    n, s1, s2, se = a(np.ones(len(y))), a(y), a(y * y), a((p - y) ** 2)
    d = 1 - (W @ se) / ((W @ s2) - (W @ s1) ** 2 / (W @ n))
    return d, float(1 - se.sum() / (s2.sum() - s1.sum() ** 2 / n.sum()))


def make_W(eid, seed=SEED, draws=B):
    qs = np.sort(np.unique(eid))
    qi = {int(v): i for i, v in enumerate(qs)}
    W = np.random.default_rng(seed).multinomial(len(qs), np.ones(len(qs)) / len(qs),
                                                size=draws).astype(float)
    return W, qi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--cut", type=float, default=0.5)
    a = ap.parse_args()
    S = Path(f"runs/goal2/{a.tag}/sweep")
    out = S / "_ktier"; out.mkdir(parents=True, exist_ok=True)
    act = Path(f"runs/goal2/{a.tag}/activations_ckpt/{a.ds}/DeepSeek-R1-Distill-Qwen-7B")
    run = Path(f"runs/goal1/{a.tag}/{a.ds}/DeepSeek-R1-Distill-Qwen-7B")

    M = load_meta(act, run)
    M["fraction"] = M.fraction.round(3)
    Mc = M[M.fraction == a.cut].reset_index(drop=True)
    kmax = int(Mc.n_continuations.max())
    low = Mc.n_continuations != kmax
    print(f"cut {a.cut}: {len(Mc)} rows, K={kmax} on {int((~low).sum())}, "
          f"K={sorted(Mc.n_continuations[low].unique())} on {int(low.sum())}")

    # ---- A: score-side split of the stored predictions -------------------------------------
    P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(S / "ref_fixed" / "preds*.csv")))])
    P["fraction"] = P.fraction.round(3)
    P = P[(P.fraction == a.cut) & (P.feature == "act_L21")].drop_duplicates(["target"] + KEYS)
    nk = Mc.set_index(["example_id", "cot_id"]).n_continuations
    rows = []
    for t in REG + [BIN]:
        d = P[P.target == t].copy()
        d["K"] = nk.reindex(pd.MultiIndex.from_frame(d[["example_id", "cot_id"]])).to_numpy()
        d = d[d.y_pred.notna() & d.y_true.notna() & d.K.notna()]
        for name, s in [("all", d), (f"K={kmax}", d[d.K == kmax]), ("K=low", d[d.K != kmax])]:
            W, qi = make_W(s.example_id.to_numpy())
            dr, pt = pooled_draws(s.y_true.to_numpy(float), s.y_pred.to_numpy(float),
                                  s.example_id.to_numpy(), W, qi, t == BIN)
            rows.append(dict(target=t, subset=name, n=len(s), n_questions=s.example_id.nunique(),
                             metric="acc" if t == BIN else "R2", score=pt,
                             ci_lo=np.percentile(dr, 2.5), ci_hi=np.percentile(dr, 97.5)))
    A = pd.DataFrame(rows)
    A.to_csv(out / "a_score_split.csv", index=False)
    print("\nA) stored predictions scored on each K tier (act_L21):")
    print(A.round(4).to_string(index=False))

    # ---- B: refit on the K=low rows only, compare on those rows ----------------------------
    # one Views for all five targets, so part B uses the K=low rows where EVERY target is present
    okall = Mc[REG + [BIN]].notna().all(axis=1)
    if int((low & ~okall).sum()):
        print(f"  part B drops {int((low & ~okall).sum())} of {int(low.sum())} K=low rows "
              f"with a NaN target (one shared row set for all five targets)")
    rows_lo = np.where((low & okall).to_numpy())[0]
    fold_of = fixed_fold_of(run, 5, 0)
    g = Mc.example_id.to_numpy()[rows_lo]
    outer = _fixed_folds(g, fold_of, 5)
    feat = Act(Mc, {"act_dir": str(act)}, layers=[21])
    views = Views(feat, feat.raw[rows_lo], g, outer)
    refit = {}
    for t in REG + [BIN]:
        pname = "logistic" if t == BIN else "ridge_act"
        cls_name, grid = GRIDS[pname]
        cls = PROBES[cls_name]
        y = Mc[t].to_numpy(float)[rows_lo]
        okm = ~np.isnan(y)
        assert okm.all(), f"{t}: {int((~okm).sum())} NaN rows in the K=low subset at this cut"
        yi = y.astype(int) if t == BIN else y
        oof, tr, chosen = run_combo(views, cls, expand(grid, None), yi, g, "bin" if t == BIN else "reg")
        refit[t] = (yi, oof)
        print(f"  refit {t:<26}{score('bin' if t == BIN else 'reg', yi, oof):.4f} (train {tr:.4f})")

    keys_lo = Mc.iloc[rows_lo][["example_id", "cot_id"]].reset_index(drop=True)
    recs, paired = [], []
    for t in REG + [BIN]:
        yi, oof = refit[t]
        d = P[P.target == t].set_index(["example_id", "cot_id"])
        main_pred = d.y_pred.reindex(pd.MultiIndex.from_frame(keys_lo)).to_numpy(float)
        ok = ~np.isnan(main_pred)
        eid = keys_lo.example_id.to_numpy()[ok]
        W, qi = make_W(eid)
        binary = t == BIN
        d1, p1 = pooled_draws(yi[ok], oof[ok], eid, W, qi, binary)
        d0, p0 = pooled_draws(yi[ok], main_pred[ok], eid, W, qi, binary)
        lo, hi = np.percentile(d1 - d0, [2.5, 97.5])
        recs.append(dict(target=t, metric="acc" if binary else "R2", n=int(ok.sum()),
                         trained_on_K_low=p1, trained_on_all=p0))
        paired.append(dict(target=t, diff=p1 - p0, ci_lo=lo, ci_hi=hi,
                           significant=bool(lo > 0 or hi < 0)))
    Bdf = pd.DataFrame(recs); Pd = pd.DataFrame(paired)
    Bdf.to_csv(out / "b_refit.csv", index=False)
    Pd.to_csv(out / "b_paired.csv", index=False)
    print("\nB) same rows (K=low only), trained on K=low only vs trained on all rows:")
    print(Bdf.merge(Pd, on="target").round(4).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
