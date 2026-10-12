# R8: LoRA heads vs ridge/logistic on the SAME rows at one checkpoint; per-fold scores + paired question bootstrap
import sys, glob, numpy as np, pandas as pd
sys.path.insert(0, "src")
from probing.sweep import fixed_fold_of
from pathlib import Path
FR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
K = ["example_id", "cot_id", "fraction"]; T5 = ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std", "committed_by_here"]
L = pd.concat([pd.read_csv(f).assign(probe=f.split("/")[-2]) for f in glob.glob(f"runs/goal2/gpqa25/lora/f{round(FR * 10):02d}_*/preds.csv")])   # label = run dir
R = pd.concat([pd.read_csv(f) for f in glob.glob("runs/goal2/gpqa25/sweep/ref_fixed/preds.shard*.csv")])
R = R[(R.feature == "act_L21") & R.probe.isin(["ridge_act", "logistic"])]
d = pd.concat([L, R]); d["fraction"] = d.fraction.round(6); d = d[(d.fraction == FR) & d.target.isin(T5)]
rows = L[K].drop_duplicates().assign(fraction=lambda x: x.fraction.round(6)); d = d.merge(rows, on=K)
d["fold"] = d.example_id.map(fixed_fold_of(Path("runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B"), 5, 0))
def sc(g):
    y, p = g.y_true.to_numpy(), g.y_pred.to_numpy()
    return ((p > .5) == (y > .5)).mean() if g.name[-1] == "committed_by_here" else 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
d["m"] = d.probe.replace({"logistic": "ridge_act"})
print("pooled:\n", d.groupby(["m", "target"]).apply(sc).unstack()[T5].round(3).to_string())
print("per fold:\n", d.groupby(["m", "fold", "target"]).apply(sc).unstack()[T5].round(3).to_string())
qs = np.sort(d.example_id.unique()); qi = {q: i for i, q in enumerate(qs)}
W = np.random.default_rng(0).multinomial(len(qs), np.full(len(qs), 1 / len(qs)), size=2000).T
def boot(g):
    q = g.example_id.map(qi).to_numpy(); y, p = g.y_true.to_numpy(), g.y_pred.to_numpy(); agg = lambda v: np.bincount(q, v, len(qs)) @ W
    if g.name[-1] == "committed_by_here": return agg(((p > .5) == (y > .5)).astype(float)) / agg(np.ones_like(y))
    n, sy, sy2 = agg(np.ones_like(y)), agg(y), agg(y ** 2); return 1 - agg((y - p) ** 2) / (sy2 - sy ** 2 / n)
B = d.groupby(["m", "target"]).apply(boot)
out = []
for m in sorted(L.probe.unique()):
    for t in T5:
        v = B[(m, t)] - B[("ridge_act", t)]
        out.append(dict(model=m, target=t, diff=round(float(np.mean(v)), 3), lo=round(np.percentile(v, 2.5), 3), hi=round(np.percentile(v, 97.5), 3)))
out = pd.DataFrame(out); out.to_csv(f"runs/goal2/gpqa25/lora/ci_f{FR}.csv", index=False); print("vs ridge (paired bootstrap):\n", out.to_string(index=False))
