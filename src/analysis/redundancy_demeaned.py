#!/usr/bin/env python3
"""Idea 1 follow-up: ridge trained on QUESTION-DEMEANED targets (diagnostic / oracle evaluation).
USAGE  envs/goal1/bin/python src/analysis/redundancy_demeaned.py --feature act_L21|prefix_len|tfidf [--max-questions N]
Each question's mean (over its rows at that checkpoint) is subtracted from y. Questions never cross folds, so for TRAIN
questions this is the train-row mean; for TEST questions it is the TRUE mean (oracle, uses held-out labels: diagnostic only).
Ridge alpha picked by inner CV on the demeaned targets (sweep.run_combo). Fixed folds unchanged.
Within metrics: the prediction is re-centred per test question (as in decomposition.csv), so they are directly comparable
to the standard probe's within_r2 / within_r2_rescaled. Paired question-bootstrap (2000) of demeaned - standard.
Writes _redundancy/demeaned_probe_<feature>.csv (and .preds.csv)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.sweep import load_meta, FEATURES, fixed_fold_of, _fixed_folds, Views, run_combo, PROBES, expand
from analysis.redundancy import load_preds, pick, stats_per_q, metrics, FRACS, REG, OUT, ACT, RUN, FEATS, NB

GRIDS = {"act_L21": {"alpha": {"logspace": [-2, 6, 17]}},    # standard grid was logspace(0,6,13); lower end extended
         "prefix_len": {"alpha": {"logspace": [-6, 3, 10]}},  # unchanged
         "tfidf": {"alpha": {"logspace": [-4, 4, 9]}}}        # standard was logspace(-3,3,7); both ends extended
ap = argparse.ArgumentParser(); ap.add_argument("--feature", required=True, choices=list(GRIDS))
ap.add_argument("--max-questions", type=int); a = ap.parse_args()

M = load_meta(ACT, RUN); M["fraction"] = M.fraction.round(3)
if a.max_questions:
    M = M[M.example_id.isin(sorted(M.example_id.unique())[:a.max_questions])].reset_index(drop=True)
fold_of = fixed_fold_of(RUN, 5, 0)
if a.feature == "act_L21":
    feat = FEATURES["act"](M, {"act_dir": str(ACT)}, [21])
else:
    feat = FEATURES[a.feature](M, {})
feat.pca = None
hps = expand(GRIDS[a.feature], None); ridge = PROBES["ridge"]
std = load_preds(); rng = np.random.default_rng(0); rows, preds = [], []
for f in FRACS:
    Mf = np.where(M.fraction.to_numpy() == f)[0]
    rws = Mf[M.answer_entropy_bits_mm.to_numpy(float)[Mf] == M.answer_entropy_bits_mm.to_numpy(float)[Mf]]
    g = M.example_id.to_numpy()[rws]
    views = Views(feat, feat.raw[rws], g, _fixed_folds(g, fold_of, 5))
    for t in REG:
        y = M[t].to_numpy(float)[rws]
        ydm = y - pd.Series(y).groupby(g).transform("mean").to_numpy()
        oof, tr_s, chosen = run_combo(views, ridge, hps, ydm, g, "reg")
        d = pd.DataFrame(dict(example_id=g, cot_id=M.cot_id.to_numpy()[rws], y_true=y, y_pred=oof))
        sd = stats_per_q(d); m = metrics(sd, np.ones(len(sd)))
        s0 = pick(std, a.feature, t); s0 = s0[(s0.fraction == f) & s0.example_id.isin(set(g))]
        ss = stats_per_q(s0.reset_index(drop=True)); m0 = metrics(ss, np.ones(len(ss)))
        assert list(sd.index) == list(ss.index)
        diffs = []
        for _ in range(NB):
            w = np.bincount(rng.integers(0, len(sd), len(sd)), minlength=len(sd)).astype(float)
            b, b0 = metrics(sd, w), metrics(ss, w)
            diffs.append({k: b[k] - b0[k] for k in ("within_r2_rescaled", "within_pearson", "within_r2")})
        df = pd.DataFrame(diffs)
        al = [c["alpha"] for c in chosen]
        rows.append(dict(feature=a.feature, target=t, fraction=f, n=len(d),
                         dm_within_r2=m["within_r2"], dm_within_r2_rescaled=m["within_r2_rescaled"], dm_within_pearson=m["within_pearson"],
                         std_within_r2=m0["within_r2"], std_within_r2_rescaled=m0["within_r2_rescaled"], std_within_pearson=m0["within_pearson"],
                         diff_rescaled=m["within_r2_rescaled"] - m0["within_r2_rescaled"],
                         diff_rescaled_lo=df.within_r2_rescaled.quantile(.025), diff_rescaled_hi=df.within_r2_rescaled.quantile(.975),
                         diff_pearson=m["within_pearson"] - m0["within_pearson"],
                         diff_pearson_lo=df.within_pearson.quantile(.025), diff_pearson_hi=df.within_pearson.quantile(.975),
                         alpha_dm=json.dumps(al), alpha_dm_median=float(np.median(al)),
                         alpha_at_grid_edge=bool(min(al) <= hps[0]["alpha"] * 1.0001 or max(al) >= hps[-1]["alpha"] * 0.9999)))
        d["fraction"] = f; d["target"] = t; preds.append(d)
        print(f, t, f"dm rescaled {m['within_r2_rescaled']:.4f} std {m0['within_r2_rescaled']:.4f} alpha {np.median(al):.3g}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"demeaned_probe_{a.feature}.csv", index=False)
pd.concat(preds).to_csv(OUT / f"demeaned_probe_{a.feature}.preds.csv", index=False)
