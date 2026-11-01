#!/usr/bin/env python3
"""Closed-form KERNEL ridge on question-DEMEANED targets: can a NONLINEAR readout at layer 21 see
within-question structure that linear ridge cannot?

Agent 1 established that (a) the probe captures only 0.3-12% of the available within-question
signal and (b) training linear ridge directly on question-demeaned targets does not recover it
(the chosen alpha runs to the grid's upper edge, i.e. the fit shrinks to nothing). Its stated
caveat is that this covers LINEAR probes at ONE layer. This script closes the nonlinear door with
closed-form kernel ridge, so there is no optimisation confound, and (with --layers) the
other-layer door for linear ridge.

ORACLE DIAGNOSTIC, and labelled as such: each question's mean is subtracted using ALL of that
question's rows at that checkpoint, so for TEST questions the subtracted mean uses held-out
labels. Exactly agent 1's construction in src/analysis/redundancy_demeaned.py; scoring reuses that
module's `stats_per_q` / `metrics`, so `within_*` numbers are directly comparable to
_redundancy/decomposition.csv and to demeaned_probe_act_L21.csv.

USAGE
  ./envs/goal1/bin/python src/probing/mlpgap_demeaned.py --fractions 0.5,0.9 \
      --targets rem_len_mean,answer_entropy_bits_mm
  ./envs/goal1/bin/python src/probing/mlpgap_demeaned.py --layers 14 --kernels linear   # layer door
OUT runs/goal2/gpqa25/sweep/_mlpgap/demeaned_krr<tag>.csv  (+ .preds.csv)
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.mlpgap_core import Data, OUT, score, print  # noqa: E402
from probing.mlpgap_b2 import Basis, _gram_d2, _kern, KNAMES  # noqa: E402
from analysis.redundancy import stats_per_q, metrics, load_preds, pick  # noqa: E402

ALPHAS = np.logspace(-6, 8, 29)      # wider than B2: agent 1 saw linear ridge pin at 1e6
NB = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.5,0.9")
    ap.add_argument("--targets", default="rem_len_mean,answer_entropy_bits_mm")
    ap.add_argument("--kernels", default="linear,poly2_c1,poly2_c3,poly2_c10,rbf_s1,rbf_s2,rbf_s4")
    ap.add_argument("--layers", default="21", help="comma list of 1-indexed decoder layers")
    ap.add_argument("--max-questions", type=int)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    knames = a.kernels.split(",")
    assert all(k in KNAMES for k in knames), knames
    layers = tuple(int(x) for x in a.layers.split(","))
    tag = a.tag or ("" if layers == (21,) else "_L" + "_".join(map(str, layers)))

    D = Data(a.max_questions, layers=layers)
    std = load_preds()
    rng = np.random.default_rng(0)
    rows, preds = [], []
    for frac in [float(x) for x in a.fractions.split(",")]:
        for t in a.targets.split(","):
            sps = D.splits(frac, t)
            rws, g = D.last_rows, D.last_g
            y = D.M[t].to_numpy(float)[rws]
            # ORACLE: question means use every row of the question, test questions included
            ydm = y - pd.Series(y).groupby(g).transform("mean").to_numpy()
            print(f"== frac {frac} {t} n={len(rws)} var(y) {y.var():.4g} "
                  f"var(ydm) {ydm.var():.4g} ({100*ydm.var()/y.var():.1f}% within)")
            oof = {k: np.full(len(rws), np.nan) for k in knames + ["honest_joint"]}
            chosen = {k: [] for k in knames + ["honest_joint"]}
            for fi, sp in enumerate(sps):
                t0 = time.time()
                d = sp.Xtr.shape[1]
                Gt, D2t = _gram_d2(sp.Xtr, sp.Xtr)
                med = float(np.median(D2t[np.triu_indices_from(D2t, 1)]))
                Ge, D2e = _gram_d2(sp.Xte, sp.Xtr)
                IG = [(_gram_d2(ii.Xtr, ii.Xtr), _gram_d2(ii.Xte, ii.Xtr)) for ii in sp.inners]
                imed = [float(np.median(x[0][1][np.triu_indices_from(x[0][1], 1)])) for x in IG]
                S = np.zeros((len(knames), len(ALPHAS)))
                for ki, kn in enumerate(knames):
                    B = Basis(_kern(kn, Gt, D2t, d, med), _kern(kn, Ge, D2e, d, med))
                    for x, m, ii in zip(IG, imed, sp.inners):
                        ib = Basis(_kern(kn, x[0][0], x[0][1], d, m),
                                   _kern(kn, x[1][0], x[1][1], d, m))
                        P = ib.predict(ydm[sp.tr][ii.tr], ALPHAS)
                        S[ki] += [score("reg", ydm[sp.tr][ii.te], p) for p in P]
                        del ib
                    j = int(np.argmax(S[ki]))
                    oof[kn][sp.te] = B.predict(ydm[sp.tr], ALPHAS[j:j + 1])[0]
                    chosen[kn].append(float(ALPHAS[j]))
                    del B
                ki, ai = np.unravel_index(int(np.argmax(S)), S.shape)
                kn = knames[ki]
                B = Basis(_kern(kn, Gt, D2t, d, med), _kern(kn, Ge, D2e, d, med))
                oof["honest_joint"][sp.te] = B.predict(ydm[sp.tr], ALPHAS[ai:ai + 1])[0]
                chosen["honest_joint"].append(f"{kn}@{ALPHAS[ai]:.3g}")
                del B, Gt, D2t, Ge, D2e, IG
                print(f"   fold {fi} {time.time()-t0:.0f}s joint {chosen['honest_joint'][-1]}")
            # within-question metrics, exactly as analysis/redundancy scores the standard probe
            s0 = pick(std, "act_L21", t)
            s0 = s0[(s0.fraction == round(frac, 3)) & s0.example_id.isin(set(g))]
            ss = stats_per_q(s0.reset_index(drop=True))
            m0 = metrics(ss, np.ones(len(ss)))
            for kn in knames + ["honest_joint"]:
                dd = pd.DataFrame(dict(example_id=g, cot_id=D.M.cot_id.to_numpy()[rws],
                                       y_true=y, y_pred=oof[kn]))
                sd = stats_per_q(dd)
                assert list(sd.index) == list(ss.index), "question sets differ"
                m = metrics(sd, np.ones(len(sd)))
                diffs = []
                for _ in range(NB):
                    w = np.bincount(rng.integers(0, len(sd), len(sd)),
                                    minlength=len(sd)).astype(float)
                    b, b0 = metrics(sd, w), metrics(ss, w)
                    diffs.append(b["within_r2_rescaled"] - b0["within_r2_rescaled"])
                al = [x for x in chosen[kn] if isinstance(x, float)]
                rows.append(dict(
                    layers="+".join(map(str, layers)), kernel=kn, target=t,
                    fraction=round(frac, 3), n=len(rws),
                    dm_r2_on_demeaned=score("reg", ydm, oof[kn]),
                    dm_within_r2=m["within_r2"], dm_within_r2_rescaled=m["within_r2_rescaled"],
                    dm_within_pearson=m["within_pearson"],
                    std_within_r2=m0["within_r2"],
                    std_within_r2_rescaled=m0["within_r2_rescaled"],
                    std_within_pearson=m0["within_pearson"],
                    diff_rescaled=m["within_r2_rescaled"] - m0["within_r2_rescaled"],
                    diff_rescaled_lo=float(np.percentile(diffs, 2.5)),
                    diff_rescaled_hi=float(np.percentile(diffs, 97.5)),
                    alpha=json.dumps(chosen[kn], default=str),
                    alpha_median=float(np.median(al)) if al else np.nan,
                    alpha_at_grid_edge=bool(al and (min(al) <= ALPHAS[0] * 1.0001
                                                    or max(al) >= ALPHAS[-1] * 0.9999))))
                dd["fraction"], dd["target"], dd["kernel"] = round(frac, 3), t, kn
                preds.append(dd)
                print(f"   {t:<24}{kn:<16}R2(dm) {rows[-1]['dm_r2_on_demeaned']:+.4f}  "
                      f"within_resc {m['within_r2_rescaled']:.4f} vs std "
                      f"{m0['within_r2_rescaled']:.4f}  diff {rows[-1]['diff_rescaled']:+.4f} "
                      f"[{rows[-1]['diff_rescaled_lo']:+.4f},{rows[-1]['diff_rescaled_hi']:+.4f}]"
                      f"{' EDGE' if rows[-1]['alpha_at_grid_edge'] else ''}")
                pd.DataFrame(rows).to_csv(OUT / f"demeaned_krr{tag}.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUT / f"demeaned_krr{tag}.preds.csv", index=False)
    print(f"\n  wrote {OUT/f'demeaned_krr{tag}.csv'}")


if __name__ == "__main__":
    main()
