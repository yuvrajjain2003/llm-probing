#!/usr/bin/env python3
"""Idea 1: how redundant is the dataset (25 CoTs per question), and does it taint the probes?
USAGE  $PY src/analysis/redundancy.py {b|c|a} [--layers 21,14,28]
  b  label redundancy: eta^2, ICC(1), ceiling, all-0/all-1 share   -> icc.csv
  c  taint test on stored ref_fixed predictions (CPU, seconds)      -> decomposition.csv, decomposition_ci.csv
  a  cosine similarity within/across questions, text side            -> cos_summary.csv, cos_auc.csv, text_*.csv
All outputs under runs/goal2/gpqa25/sweep/_redundancy/. Checkpoints 10-90% only (0% has 1 row/question).
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.sweep import load_meta, Act, KEYS
from probing.probe_checkpoints import _r2

S = Path("runs/goal2/gpqa25/sweep"); OUT = S / "_redundancy"
ACT = Path("runs/goal2/gpqa25/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
RUN = Path("runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
REG = ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std"]
TARGETS = REG + ["committed_by_here"]
FEATS = {"act_L21": "ridge_act", "prefix_len": "ridge_len", "tfidf": "ridge_text"}   # + logistic for committed
FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]
NB = 2000


def load_preds():
    p = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(S / "ref_fixed/preds.shard*.csv")))], ignore_index=True)
    p["fraction"] = p.fraction.round(3)
    return p


def pick(p, feat, target):
    probe = "logistic" if target == "committed_by_here" else FEATS[feat]
    return p[(p.feature == feat) & (p.probe == probe) & (p.target == target)]


# ------------------------------------------------------------------ Part B
def anova(n, mean, ssw):
    """one-way ANOVA from per-group (n, mean, within-SS). -> eta2, icc1, msb, msw, mbar"""
    N, k = n.sum(), len(n)
    gm = (n * mean).sum() / N
    ssb = (n * (mean - gm) ** 2).sum(); sst = ssb + ssw.sum()
    msb, msw = ssb / (k - 1), ssw.sum() / (N - k)
    mbar = (N - (n ** 2).sum() / N) / (k - 1)
    return ssb / sst, (msb - msw) / (msb + (mbar - 1) * msw), msb, msw, mbar


def part_b():
    p = load_preds(); rng = np.random.default_rng(0); rows = []
    for t in TARGETS:
        for f in FRACS:
            d = pick(p, "act_L21", t); d = d[d.fraction == f]
            g = d.groupby("example_id").y_true
            n = g.size().to_numpy(float); mean = g.mean().to_numpy()
            ssw = g.apply(lambda v: ((v - v.mean()) ** 2).sum()).to_numpy()
            eta, icc, msb, msw, mbar = anova(n, mean, ssw)
            bs = []
            for _ in range(NB):
                i = rng.integers(0, len(n), len(n)); bs.append(anova(n[i], mean[i], ssw[i])[:2])
            bs = np.array(bs)
            r = dict(target=t, fraction=f, n_rows=len(d), n_questions=len(n), mean_group_size=mbar,
                     eta2=eta, eta2_lo=np.percentile(bs[:, 0], 2.5), eta2_hi=np.percentile(bs[:, 0], 97.5),
                     icc1=icc, icc1_lo=np.percentile(bs[:, 1], 2.5), icc1_hi=np.percentile(bs[:, 1], 97.5),
                     ceiling_icc1=icc, ceiling_eta2=eta)
            if t == "committed_by_here":
                m = d.groupby("example_id").y_true.mean()
                r["share_all0"], r["share_all1"] = float((m == 0).mean()), float((m == 1).mean())
                r["share_constant"] = r["share_all0"] + r["share_all1"]
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "icc.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


# ------------------------------------------------------------------ Part C
def stats_per_q(d):
    """per-question sufficient statistics for within / between scores."""
    d = d.copy()
    d["my"] = d.groupby("example_id").y_true.transform("mean")
    d["mh"] = d.groupby("example_id").y_pred.transform("mean")
    d["ydm"], d["hdm"] = d.y_true - d.my, d.y_pred - d.mh
    g = d.groupby("example_id")
    st = pd.DataFrame(dict(n=g.size(), my=g.my.first(), mh=g.mh.first(),
                           syy=g.ydm.apply(lambda v: (v ** 2).sum()), shh=g.hdm.apply(lambda v: (v ** 2).sum()),
                           syh=(d.ydm * d.hdm).groupby(d.example_id).sum()))
    st["sse"] = st.syy + st.shh - 2 * st.syh
    rho = {}
    for q, x in g:
        rho[q] = spearmanr(x.y_true, x.y_pred)[0] if x.y_true.nunique() > 1 and x.y_pred.nunique() > 1 else np.nan
    st["rho"] = pd.Series(rho)
    return st


def metrics(st, w):
    """w = question weights (bootstrap counts). -> dict of between R2, within R2/Pearson/rescaled/mean Spearman."""
    n = st.n.to_numpy() * w
    my, mh = st.my.to_numpy(), st.mh.to_numpy()
    gm = (w * my).sum() / w.sum()                         # between: question means, unweighted over questions
    between = 1 - (w * (my - mh) ** 2).sum() / (w * (my - gm) ** 2).sum()
    syy, shh, syh = (w * st.syy).sum(), (w * st.shh).sum(), (w * st.syh).sum()
    sse = (w * st.sse).sum()
    ok = ~np.isnan(st.rho.to_numpy())
    return dict(between_r2=between, within_r2=1 - sse / syy, within_pearson=syh / np.sqrt(syy * shh),
                within_r2_rescaled=syh ** 2 / (syy * shh),
                within_spearman=(w[ok] * st.rho.to_numpy()[ok]).sum() / w[ok].sum())


def part_c():
    p = load_preds(); ref = pd.read_csv(S / "ref_fixed/results.csv"); ref["fraction"] = ref.fraction.round(3)
    rng = np.random.default_rng(0); rows, cis, bad = [], [], 0
    for feat in FEATS:
        for t in TARGETS:
            for f in FRACS:
                d0 = pick(p, feat, t); d = d0[d0.fraction == f]
                probe = "logistic" if t == "committed_by_here" else FEATS[feat]
                r0 = ref[(ref.feature == feat) & (ref.probe == probe) & (ref.target == t) & (ref.fraction == f)].iloc[0]
                y, h = d.y_true.to_numpy(float), d.y_pred.to_numpy(float)
                pooled = float((((h >= 0.5).astype(int) == y).mean()) if t == "committed_by_here" else _r2(y, h))
                if abs(pooled - r0.test) > 1e-6 or len(d) != r0.n:
                    bad += 1; print("MISMATCH", feat, t, f, pooled, r0.test, len(d), r0.n)
                st = stats_per_q(d); w1 = np.ones(len(st))
                m = metrics(st, w1)
                rows.append(dict(feature=feat, target=t, fraction=f, n=len(d), n_questions=len(st),
                                 pooled=pooled, pooled_ref=r0.test, pooled_r2_of_probs=_r2(y, h), **m))
                bs = []
                for _ in range(NB):
                    w = np.bincount(rng.integers(0, len(st), len(st)), minlength=len(st)).astype(float)
                    bs.append(metrics(st, w))
                bs = pd.DataFrame(bs)
                for k in m:
                    cis.append(dict(feature=feat, target=t, fraction=f, metric=k, value=m[k],
                                    lo=bs[k].quantile(.025), hi=bs[k].quantile(.975),
                                    frac_boot_le0=float((bs[k] <= 0).mean())))
    pd.DataFrame(rows).to_csv(OUT / "decomposition.csv", index=False)
    pd.DataFrame(cis).to_csv(OUT / "decomposition_ci.csv", index=False)
    print("pooled-vs-ref mismatches:", bad, "of", len(rows))
    if bad:
        raise SystemExit("CORRECTNESS CHECK FAILED")
    print("CHECK PASSED: max |pooled - ref| <=1e-6 for all", len(rows), "cells")


# ------------------------------------------------------------------ Part A
def auc_d(w, a):
    """P(within > across) (ties half) and Cohen's d."""
    r = rankdata(np.concatenate([w, a])); n1, n2 = len(w), len(a)
    auc = (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
    sp = np.sqrt(((n1 - 1) * w.var(ddof=1) + (n2 - 1) * a.var(ddof=1)) / (n1 + n2 - 2))
    return auc, (w.mean() - a.mean()) / sp


def pair_cos(Z, i, j):
    """cosine of row pairs, float64, chunked."""
    out = np.empty(len(i))
    for s in range(0, len(i), 20000):
        a, b = Z[i[s:s + 20000]].astype(np.float64), Z[j[s:s + 20000]].astype(np.float64)
        out[s:s + 20000] = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    return out


def pairs_within(q, rng=None):
    i, j = [], []
    for u in np.unique(q):
        ix = np.where(q == u)[0]
        a, b = np.triu_indices(len(ix), 1); i.append(ix[a]); j.append(ix[b])
    return np.concatenate(i), np.concatenate(j)


def pairs_across(q, n, rng):
    i = rng.integers(0, len(q), 3 * n); j = rng.integers(0, len(q), 3 * n)
    k = q[i] != q[j]
    return i[k][:n], j[k][:n]


def qstats(x):
    return dict(mean=x.mean(), sd=x.std(), q05=np.quantile(x, .05), q25=np.quantile(x, .25),
                median=np.median(x), q75=np.quantile(x, .75), q95=np.quantile(x, .95))


def part_a(layers):
    rng = np.random.default_rng(0)
    M = load_meta(ACT, RUN)
    M["fraction"] = M.fraction.round(3)
    keep = M.answer_entropy_bits_mm.notna() & M.fraction.isin(FRACS)
    M = M[keep].reset_index(drop=True)
    print(len(M), "rows after dropping NaN-label rows;", M.example_id.nunique(), "questions", flush=True)
    summ, aucs, hist = [], [], {}
    # ---- text side (once)
    from sklearn.feature_extraction.text import TfidfVectorizer
    trows, tauc = [], []
    for f in FRACS:
        m = np.where(M.fraction == f)[0]; q = M.example_id.to_numpy()[m]
        X = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True).fit_transform(
            M.prefix_text.fillna("").to_numpy()[m]).tocsr()
        wi, wj = pairs_within(q); ai, aj = pairs_across(q, len(wi), rng)
        cs = lambda i, j: np.asarray(X[i].multiply(X[j]).sum(1)).ravel()    # rows are L2-normalised
        cw, ca = cs(wi, wj), cs(ai, aj); auc, d = auc_d(cw, ca)
        tauc.append(dict(fraction=f, n_within=len(cw), n_across=len(ca), auc=auc, cohens_d=d,
                         within_mean=cw.mean(), across_mean=ca.mean()))
        hist[("tfidf", f)] = (cw, ca)
        tk = M.prefix_tokens.to_numpy(float)[m]
        df = pd.DataFrame(dict(q=q, t=tk)).groupby("q").t.agg(["mean", "std", "size"])
        trows.append(dict(fraction=f, mean_within_q_cv=float((df["std"] / df["mean"]).mean()),
                          overall_cv=float(tk.std() / tk.mean()),
                          between_q_share_of_var=float(1 - (df["std"] ** 2 * (df["size"] - 1)).sum() / (tk.var() * (len(tk) - 1)))))
    pd.DataFrame(tauc).to_csv(OUT / "text_tfidf_cos_auc.csv", index=False)
    pd.DataFrame(trows).to_csv(OUT / "text_prefix_tokens_cv.csv", index=False)
    print(pd.DataFrame(tauc).round(3).to_string(), "\n", pd.DataFrame(trows).round(3).to_string(), flush=True)
    # ---- activations
    for L in layers:
        feat = Act(M, {"act_dir": str(ACT)}, [L]); X = feat.raw
        pp = {}
        for f in FRACS:
            m = np.where(M.fraction == f)[0]; q = M.example_id.to_numpy()[m]; Xf = X[m].astype(np.float64)
            mu, sd = Xf.mean(0), Xf.std(0); sd[sd < 1e-8] = 1.0
            Zs = {"raw": Xf, "centered": Xf - mu, "standardised": (Xf - mu) / sd}
            pp[f] = (m, Zs)
            wi, wj = pairs_within(q); ai, aj = pairs_across(q, len(wi), rng)
            for name, Z in Zs.items():
                cw, ca = pair_cos(Z, wi, wj), pair_cos(Z, ai, aj); auc, d = auc_d(cw, ca)
                for pop, c in (("within_question", cw), ("across_question", ca)):
                    summ.append(dict(layer=L, fraction=f, preproc=name, population=pop, n_pairs=len(c), **qstats(c)))
                aucs.append(dict(layer=L, fraction=f, preproc=name, n_within=len(cw), n_across=len(ca), auc=auc, cohens_d=d))
                hist[(L, name, f)] = (cw, ca)
            print(f"L{L} f{f}", [round(a["auc"], 3) for a in aucs[-3:]], flush=True)
        # ---- same CoT across depth (bonus)
        key = M[["example_id", "cot_id", "fraction"]].copy(); key["row"] = np.arange(len(M))
        for fa, fb in [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.5, 0.9), (0.1, 0.9)]:
            ka = key[key.fraction == fa].set_index(["example_id", "cot_id"]).row
            kb = key[key.fraction == fb].set_index(["example_id", "cot_id"]).row
            com = ka.index.intersection(kb.index)
            ia, ib = ka.loc[com].to_numpy(), kb.loc[com].to_numpy()
            q = np.array([c[0] for c in com]); c_ = np.array([c[1] for c in com])
            # other chain of the SAME question across the two depths: random partner with same q, different cot
            si, sj = [], []
            for u in np.unique(q):
                ix = np.where(q == u)[0]
                if len(ix) < 2: continue
                for a_ in ix:
                    b_ = rng.choice(ix[ix != a_]); si.append(a_); sj.append(b_)
            si, sj = np.array(si), np.array(sj)
            ri = rng.integers(0, len(q), 3 * len(si)); rj = rng.integers(0, len(q), 3 * len(si))
            k = q[ri] != q[rj]; ri, rj = ri[k][:len(si)], rj[k][:len(si)]
            for name in ("raw", "centered", "standardised"):
                Za, Zb = pp[fa][1][name], pp[fb][1][name]
                ra = {r: i for i, r in enumerate(M.index[pp[fa][0]])}; rb = {r: i for i, r in enumerate(M.index[pp[fb][0]])}
                A = Za[[ra[r] for r in ia]]; B = Zb[[rb[r] for r in ib]]
                nrm = lambda U: U / np.linalg.norm(U, axis=1, keepdims=True)
                A, B = nrm(A), nrm(B)
                same = (A * B).sum(1)
                oth_q = (A[si] * B[sj]).sum(1)          # other chain, same question, other depth
                oth_x = (A[ri] * B[rj]).sum(1)          # other question, other depth
                for pop, c in (("same_cot", same), ("same_question_other_cot", oth_q), ("other_question", oth_x)):
                    summ.append(dict(layer=L, fraction=f"{fa}->{fb}", preproc=name, population=pop, n_pairs=len(c), **qstats(c)))
                a1, d1 = auc_d(same, oth_q); a2, d2 = auc_d(oth_q, oth_x)
                aucs.append(dict(layer=L, fraction=f"{fa}->{fb}", preproc=name, n_within=len(same), n_across=len(oth_q),
                                 auc=a1, cohens_d=d1, note="depth: same-CoT vs same-question-other-CoT (auc2 below)",
                                 auc2_sameq_vs_otherq=a2))
        del X, pp
    pd.DataFrame(summ).to_csv(OUT / "cos_summary.csv", index=False)
    pd.DataFrame(aucs).to_csv(OUT / "cos_auc.csv", index=False)
    rng = np.random.default_rng(1)
    np.savez_compressed(OUT / "cos_hist_samples.npz", **{"_".join(map(str, k)) + "_w": rng.permutation(v[0])[:5000] for k, v in hist.items()},
                        **{"_".join(map(str, k)) + "_a": rng.permutation(v[1])[:5000] for k, v in hist.items()})


def figures():
    """Figures for A (histograms per layer) and C (between/within), from the saved CSV/npz."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FD = Path("docs/figures/redundancy"); FD.mkdir(parents=True, exist_ok=True)
    ci = pd.read_csv(OUT / "decomposition_ci.csv"); ci = ci[ci.feature == "act_L21"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for a, (m, ttl) in zip(ax, [("between_r2", "between-question R2"), ("within_r2", "within-question R2"),
                                ("within_pearson", "within-question Pearson r")]):
        for t in TARGETS:
            x = ci[(ci.metric == m) & (ci.target == t)].sort_values("fraction")
            a.errorbar(x.fraction, x.value, [x.value - x.lo, x.hi - x.value], marker="o", capsize=2, label=t)
        a.axhline(0, color="k", lw=.5); a.set_title(ttl); a.set_xlabel("checkpoint")
    ax[0].legend(fontsize=7); fig.suptitle("act_L21 ridge/logistic, stored out-of-fold predictions (95% question bootstrap)")
    fig.tight_layout(); fig.savefig(FD / "c_between_within.png", dpi=130); plt.close(fig)
    z = np.load(OUT / "cos_hist_samples.npz")
    for L in sorted({k.split("_")[0] for k in z.files if k[0].isdigit()}):
        fig, ax = plt.subplots(3, 5, figsize=(16, 7))
        for r, pre in enumerate(["raw", "centered", "standardised"]):
            for c, f in enumerate(FRACS):
                k = f"{L}_{pre}_{f}"; a = ax[r, c]
                a.hist(z[k + "_a"], 60, alpha=.5, label="across", density=True); a.hist(z[k + "_w"], 60, alpha=.5, label="within", density=True)
                a.set_title(f"L{L} {pre} {int(f*100)}%", fontsize=8)
        ax[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FD / f"a_cos_hist_L{L}.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("part", choices=["a", "b", "c", "fig"])
    ap.add_argument("--layers", default="21,14,28"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"fig": figures, "b": part_b, "c": part_c, "a": lambda: part_a([int(x) for x in a.layers.split(",")])}[a.part]()
