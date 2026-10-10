#!/usr/bin/env python3
"""R3: why is random forest below ridge? One checkpoint per call, fixed folds, layer-21 activations.
USAGE  $PY src/probing/r3_rf_diagnosis.py --fraction 0.5 [--max-questions N] [--trees 500]
Writes runs/goal2/gpqa25/sweep/_round2/r3_rf/r3_f<frac>.csv (variant rows) and w_f<frac>.csv (weight spread).
Variants per target (all scored as pooled out-of-fold R2 over the 5 fixed folds, 'train' = in-sample mean):
  ridge            ridge, alpha = the one chosen in ref_fixed for that fold
  rf_raw           RF (leaf 5, max_features 0.1) on 3584 standardised dims          [b/baseline]
  rf_mf0.3/1.0     same, more features per split (100 trees)
  syn_*            target replaced by X w (+ noise); ridge and RF on it           [b]
  rf_on_ridgeoof   target = ridge's out-of-fold prediction from ref_fixed          [b]
  rf_ridgepred     RF on ONE feature: cross-fitted ridge prediction               [c]
  pls{k}_ols / pls{k}_rf   PLS-k scores (fit on train fold) -> OLS / RF           [c]
  top{k}_ridge / top{k}_rf  top-k |w| dims (w from ridge on the train fold)       [c, d]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.sweep import (load_meta, Act, fixed_fold_of, _fixed_folds, _group_folds, KEYS)
from probing.probe_checkpoints import _r2
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression

ap = argparse.ArgumentParser()
ap.add_argument("--fraction", type=float, required=True)
ap.add_argument("--targets", default="answer_entropy_bits_mm,rem_len_mean")
ap.add_argument("--max-questions", type=int); ap.add_argument("--trees", type=int, default=500)
ap.add_argument("--jobs", type=int, default=32); ap.add_argument("--tag", default="")
a = ap.parse_args()
S = Path("runs/goal2/gpqa25/sweep"); OUT = S / "_round2/r3_rf"
act = "runs/goal2/gpqa25/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B"
run = "runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B"
M = load_meta(Path(act), Path(run))
if a.max_questions:
    M = M[M.example_id.isin(sorted(M.example_id.unique())[:a.max_questions])].reset_index(drop=True)
ref = pd.read_csv(S / "ref_fixed/results.csv")
refp = pd.concat([pd.read_csv(f) for f in sorted((S / "ref_fixed").glob("preds.shard*.csv"))])
refp = refp[(refp.feature == "act_L21") & (refp.probe == "ridge_act")]
fold_of = fixed_fold_of(Path(run), 5, 0)
Mf = np.where(np.isclose(M.fraction.to_numpy(), a.fraction))[0]
feat = Act(M, {"act_dir": act}, [21]); feat.pca = None
NJ = a.jobs


def ridge_fit(X, y, alpha, svd=None):
    U, s, Vt = svd if svd is not None else np.linalg.svd(X, full_matrices=False)
    yb = y.mean()
    w = Vt.T @ (s / (s**2 + alpha) * (U.T @ (y - yb)))
    return w, yb


def rf(X, y, mf, leaf=5, trees=None):
    return RandomForestRegressor(n_estimators=trees or a.trees, min_samples_leaf=leaf, max_features=mf,
                                 n_jobs=NJ, random_state=0).fit(X, y)


rows, wrows = [], []
def rec(t, var, oof, y, tr, extra=""):
    rows.append(dict(fraction=a.fraction, target=t, variant=var, n=len(y), test=_r2(y, oof),
                     train=float(np.mean(tr)) if len(tr) else np.nan, note=extra))
    print(f"  {t:<24}{var:<18}test {rows[-1]['test']:.4f} train {rows[-1]['train']:.4f} {extra}", flush=True)


for t in a.targets.split(","):
    ya = M[t].to_numpy(float)[Mf]; rws = Mf[~np.isnan(ya)]
    y = M[t].to_numpy(float)[rws]; g = M.example_id.to_numpy()[rws]
    outer = _fixed_folds(g, fold_of, 5)
    rr = ref[(ref.feature == "act_L21") & (ref.probe == "ridge_act") & (ref.target == t) &
             np.isclose(ref.fraction, a.fraction)].iloc[0]
    alphas = [h["alpha"] for h in json.loads(rr.hp)]
    key = M.iloc[rws][KEYS].reset_index(drop=True)
    rp = key.merge(refp[(refp.target == t) & np.isclose(refp.fraction, a.fraction)],
                   on=KEYS, how="left")
    yoof_ref = rp.y_pred.to_numpy()
    print(f"=== {t} n={len(rws)} ref ridge test {rr.test:.4f} alphas {alphas}", flush=True)
    A = feat.raw[rws]
    # synthetic targets: w from ridge on ALL rows, signal scaled to variance var(y)*R2_ridge, plus noise
    Zall = (A - A.mean(0)) / np.where(A.std(0) < 1e-8, 1, A.std(0))
    wall, _ = ridge_fit(Zall, y, float(np.median(alphas)))
    sig = Zall @ wall; sig = (sig - sig.mean()) / sig.std()
    rng = np.random.default_rng(0)
    r2ref = rr.test
    syn = {"syn_matched": np.sqrt(r2ref * y.var()) * sig + np.sqrt((1 - r2ref) * y.var()) * rng.standard_normal(len(y)),
           "syn_noiseless": np.sqrt(y.var()) * sig}
    ytargets = {"": y, "rf_on_ridgeoof": yoof_ref, **syn}
    K_TOP, K_PLS = [16, 64, 256, 1024], [1, 5, 16]
    oof = {}
    def put(name, idx, val):
        oof.setdefault(name, np.full(len(y), np.nan))[idx] = val
    trs = {}
    def put_tr(name, v): trs.setdefault(name, []).append(v)
    t0 = time.time()
    for fi, (tr, te) in enumerate(outer):
        mu, sd = A[tr].mean(0), A[tr].std(0); sd[sd < 1e-8] = 1
        Xtr, Xte = (A[tr] - mu) / sd, (A[te] - mu) / sd
        svd = np.linalg.svd(Xtr, full_matrices=False)
        al = alphas[fi]
        w, yb = ridge_fit(Xtr, y[tr], al, svd)
        put("ridge", te, Xte @ w + yb); put_tr("ridge", _r2(y[tr], Xtr @ w + yb))
        # (d) weight spread (standardised inputs, so |w| comparable)
        w2 = np.sort(w**2)[::-1]; cs = np.cumsum(w2) / w2.sum()
        wrows.append(dict(fraction=a.fraction, target=t, fold=fi, alpha=al, d=len(w),
                          participation_ratio=float(w2.sum()**2 / (w2**2).sum()),
                          dims_50pct_energy=int(np.searchsorted(cs, .5) + 1),
                          dims_90pct_energy=int(np.searchsorted(cs, .9) + 1),
                          top_share_1pct=float(cs[len(w) // 100 - 1])))
        # (a)/(b) RF on raw, on the real target and on derived targets
        for name, yt in ytargets.items():
            m = rf(Xtr, yt[tr], 0.1); n_ = name or "rf_raw"
            put(n_, te, m.predict(Xte)); put_tr(n_, _r2(yt[tr], m.predict(Xtr)))
            if name.startswith("syn"):
                ws, yb2 = ridge_fit(Xtr, yt[tr], al, svd)
                put("ridge_" + name, te, Xte @ ws + yb2); put_tr("ridge_" + name, _r2(yt[tr], Xtr @ ws + yb2))
        for mf in (0.3, 1.0):
            m = rf(Xtr, y[tr], mf, trees=100); put(f"rf_mf{mf}", te, m.predict(Xte)); put_tr(f"rf_mf{mf}", _r2(y[tr], m.predict(Xtr)))
        # (c) cross-fitted ridge prediction as ONE feature (inner group folds, same alpha)
        cf = np.zeros(len(tr))
        for itr, ite in _group_folds(len(tr), g[tr], 3):
            wi, ybi = ridge_fit(Xtr[itr], y[tr][itr], al)
            cf[ite] = Xtr[ite] @ wi + ybi
        m = rf(cf[:, None], y[tr], 1.0, leaf=20); pt = Xte @ w + yb
        put("rf_ridgepred", te, m.predict(pt[:, None])); put_tr("rf_ridgepred", _r2(y[tr], m.predict(cf[:, None])))
        # (c) PLS-k
        pls = PLSRegression(n_components=max(K_PLS), scale=False).fit(Xtr, y[tr])
        Ptr, Pte = pls.transform(Xtr), pls.transform(Xte)
        for k in K_PLS:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression().fit(Ptr[:, :k], y[tr])
            put(f"pls{k}_ols", te, lr.predict(Pte[:, :k])); put_tr(f"pls{k}_ols", _r2(y[tr], lr.predict(Ptr[:, :k])))
            m = rf(Ptr[:, :k], y[tr], 1.0 if k == 1 else 0.5, leaf=5)
            put(f"pls{k}_rf", te, m.predict(Pte[:, :k])); put_tr(f"pls{k}_rf", _r2(y[tr], m.predict(Ptr[:, :k])))
        # (c)/(d) top-k |w| dims
        order = np.argsort(-np.abs(w))
        for k in K_TOP:
            d_ = order[:k]; Xa, Xb = Xtr[:, d_], Xte[:, d_]
            wk, ybk = ridge_fit(Xa, y[tr], max(al * k / len(w), 1.0))  # alpha scaled down with k, floor 1
            put(f"top{k}_ridge", te, Xb @ wk + ybk); put_tr(f"top{k}_ridge", _r2(y[tr], Xa @ wk + ybk))
            if k <= 256:
                m = rf(Xa, y[tr], 0.3 if k > 16 else 0.5, leaf=5)
                put(f"top{k}_rf", te, m.predict(Xb)); put_tr(f"top{k}_rf", _r2(y[tr], m.predict(Xa)))
        print(f"  fold {fi} done {time.time()-t0:.0f}s", flush=True)
    for name, o in oof.items():
        base = name.replace("ridge_", "") if name.startswith("ridge_syn") else name
        yt = ytargets.get(base, ytargets.get("rf_on_ridgeoof" if name == "rf_on_ridgeoof" else "", y))
        if name in ("rf_on_ridgeoof",): yt = yoof_ref
        elif name.startswith("syn"): yt = syn[name]
        elif name.startswith("ridge_syn"): yt = syn[name[6:]]
        else: yt = y
        rec(t, name, o, yt, trs[name])
    pd.DataFrame(rows).to_csv(OUT / f"r3_f{a.fraction}{a.tag}.csv", index=False)
    pd.DataFrame(wrows).to_csv(OUT / f"w_f{a.fraction}{a.tag}.csv", index=False)
