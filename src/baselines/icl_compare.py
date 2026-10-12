# R1 (thinking off + on): ICL vs reference probes on the SAME rows (10-90%), mean over checkpoints
import glob, pandas as pd, numpy as np
K = ["example_id", "cot_id", "fraction"]
icl = pd.concat([pd.read_csv(f) for f in glob.glob("runs/goal2/gpqa25/icl/preds_*_k*.csv")])
ref = pd.concat([pd.read_csv(f) for f in glob.glob("runs/goal2/gpqa25/sweep/ref_fixed/preds.shard*.csv")])
for d in (icl, ref): d["fraction"] = d.fraction.round(6)
rows = icl.loc[icl.fraction > 0, K].drop_duplicates()
ref = ref.merge(rows, on=K)
d = pd.concat([icl, ref])
def score(g):
    y, p = g.y_true.to_numpy(), g.y_pred.to_numpy()
    if g.name[2] == "committed_by_here": return ((p > .5) == (y > .5)).mean()
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
s = d.groupby(["feature", "probe", "target", "fraction"]).apply(score).rename("s").reset_index()
for name, sub in (("10-90% (mean over 5 ckpts)", s[s.fraction > 0]), ("0% only", s[s.fraction == 0])):
    t = sub.groupby(["feature", "probe", "target"]).s.mean().unstack().round(3)
    t = t[[c for c in ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std", "committed_by_here"] if c in t]]
    print("\n==", name, " n rows/ckpt:", sub.shape[0] and len(rows) // 5); print(t.to_string())

# ---- error bars: question bootstrap (2000 draws, same draws for all models -> paired differences)
B = 2000
T5 = ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std", "committed_by_here"]
s10 = d[(d.fraction > 0) & d.target.isin(T5)].copy()
qs = np.sort(s10.example_id.unique()); qi = {q: i for i, q in enumerate(qs)}
W = np.random.default_rng(0).multinomial(len(qs), np.full(len(qs), 1 / len(qs)), size=B).T   # (questions, draws)
def boot(g):   # per-draw score for one (feature, probe, target, fraction)
    q = g.example_id.map(qi).to_numpy(); y, p = g.y_true.to_numpy(), g.y_pred.to_numpy()
    agg = lambda v: np.bincount(q, v, len(qs)) @ W
    if g.name[2] == "committed_by_here":
        return agg(((p > .5) == (y > .5)).astype(float)) / agg(np.ones_like(y))
    n, sy, sy2, sse = agg(np.ones_like(y)), agg(y), agg(y ** 2), agg((y - p) ** 2)
    return 1 - sse / (sy2 - sy ** 2 / n)
bs = s10.groupby(["feature", "probe", "target", "fraction"]).apply(boot)
M = {k: np.mean(np.stack(v.tolist()), 0) for k, v in bs.groupby(level=[0, 1, 2])}          # mean over ckpts, per draw
pt = s[s.fraction > 0].groupby(["feature", "probe", "target"]).s.mean()
ref = {t: ("act_L21", "logistic" if t == "committed_by_here" else "ridge_act") for t in pt.index.get_level_values(2).unique()}
out = []
for (f, p, t), v in M.items():
    r = M[(*ref[t], t)]; diff = pt[(f, p, t)] - pt[(*ref[t], t)]
    out.append(dict(feature=f, probe=p, target=t, score=pt[(f, p, t)], lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5),
                    diff_vs_ridge=diff, diff_lo=np.percentile(v - r, 2.5), diff_hi=np.percentile(v - r, 97.5)))
out = pd.DataFrame(out).round(3); out.to_csv("runs/goal2/gpqa25/icl/ci_vs_probes.csv", index=False)
keep = out[(out.probe != "icl_raw") & ~out.feature.str.startswith("r1qwen7b")]
print(keep.to_string(index=False))
