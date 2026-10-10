#!/usr/bin/env python3
"""R2: are the model bases what we claim? Compares configs, tokenizers and weights tensor by tensor.

    $PY src/analysis/compare_model_weights.py     # CPU job; ~15 GB read per model, one tensor at a time

Outputs: runs/goal2/gpqa25/sweep/_round2/r2_model_bases/{config_tokenizer.txt,tensor_similarity.csv,summary.csv}
and docs/figures/round2/r2_weight_similarity.png.
"""
import itertools, json, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from safetensors import safe_open

HUB = Path("cache/huggingface/hub")
OUT = Path("runs/goal2/gpqa25/sweep/_round2/r2_model_bases")
FIG = Path("docs/figures/round2"); FIG.mkdir(parents=True, exist_ok=True)
M = {"R1-Distill": "deepseek-ai--DeepSeek-R1-Distill-Qwen-7B", "Math": "Qwen--Qwen2.5-Math-7B",
     "Instruct": "Qwen--Qwen2.5-7B-Instruct", "Base": "Qwen--Qwen2.5-7B"}
D = {k: next((HUB / f"models--{v}" / "snapshots").glob("*")) for k, v in M.items()}

# ---- configs + tokenizers
keys = ["num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size",
        "vocab_size", "rope_theta", "max_position_embeddings", "sliding_window", "tie_word_embeddings",
        "bos_token_id", "eos_token_id", "rms_norm_eps"]
cfg = pd.DataFrame({k: {x: json.load(open(D[k] / "config.json")).get(x) for x in keys} for k in D})
tok = {k: json.load(open(D[k] / "tokenizer.json")) for k in D}
lines = [cfg.to_string(), ""]
for a, b in itertools.combinations(D, 2):
    va, vb = tok[a]["model"]["vocab"], tok[b]["model"]["vocab"]
    ta = {t["content"]: t["id"] for t in tok[a]["added_tokens"]}
    tb = {t["content"]: t["id"] for t in tok[b]["added_tokens"]}
    lines.append(f"{a} vs {b}: base vocab identical={va == vb} (n={len(va)},{len(vb)}); merges identical="
                 f"{tok[a]['model']['merges'] == tok[b]['model']['merges']}; added tokens {len(ta)},{len(tb)}; "
                 f"same content->id for shared added tokens: {all(ta[c] == tb[c] for c in ta.keys() & tb.keys())}; "
                 f"only in {a}: {sorted(ta.keys() - tb.keys())[:8]}; only in {b}: {sorted(tb.keys() - ta.keys())[:8]}")
(OUT / "config_tokenizer.txt").write_text("\n".join(lines)); print("\n".join(lines))

# ---- weights
wm = {k: json.load(open(D[k] / "model.safetensors.index.json"))["weight_map"] for k in D}
assert all(wm[k].keys() == wm["Math"].keys() for k in D), "tensor names differ"
files = {k: {f: safe_open(D[k] / f, "pt") for f in set(wm[k].values())} for k in D}
get = lambda m, n: files[m][wm[m][n]].get_tensor(n).float()
rows = []
for n in sorted(wm["Math"]):
    T = {m: get(m, n) for m in D}
    lay = re.search(r"layers\.(\d+)\.", n)
    typ = re.sub(r"^model\.layers\.\d+\.", "", n)
    for a, b in itertools.combinations(D, 2):
        x, y = T[a], T[b]
        r = min(x.shape[0], y.shape[0]); x, y = x[:r].flatten().double(), y[:r].flatten().double()   # shared rows (embed/lm_head); float64 sums (float32 cosine gave values >1)
        rows.append(dict(tensor=n, layer=int(lay[1]) if lay else -1, type=typ, pair=f"{a} vs {b}", n=x.numel(),
                         shape_a=tuple(T[a].shape), shape_b=tuple(T[b].shape),
                         d2=(x - y).pow(2).sum().item(), a2=x.pow(2).sum().item(),
                         rel_diff=((x - y).norm() / x.norm()).item(), cos=torch.cosine_similarity(x, y, 0).item()))
    print(n, flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "tensor_similarity.csv", index=False)

# ---- summary: all params pooled, plus by depth third, for the 2D weight matrices of the decoder layers
W = df[(df.layer >= 0) & df.type.str.endswith("weight") & df.type.str.contains("proj")].copy()
W["depth"] = pd.cut(W.layer, [-1, 8, 18, 27], labels=["layers 0-8", "layers 9-18", "layers 19-27"])
pool = lambda g: pd.Series({"rel_diff": (g.d2.sum() / g.a2.sum()) ** 0.5, "mean_cos": g.cos.mean()})
S = pd.concat([df.groupby("pair").apply(pool).assign(group="all tensors"),
               df[df.layer < 0].groupby(["pair", "type"]).apply(pool).reset_index("type").assign(group=lambda d: d.type).drop(columns="type"),
               W.groupby(["pair", "depth"], observed=True).apply(pool).reset_index("depth").assign(group=lambda d: "proj " + d.depth.astype(str)).drop(columns="depth")]).reset_index()
S.to_csv(OUT / "summary.csv", index=False); print(S.pivot(index="group", columns="pair", values="rel_diff").round(4).to_string())

# ---- figure (palette from src/viz/update_figures.py)
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#a8a7a0"]
INK2, GRID, SURF = "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for c, (p, g) in zip(COL, W.groupby("pair")):
    L = g.groupby("layer").apply(pool)
    ax[0].plot(L.index, L.rel_diff, color=c, label=p); ax[1].plot(L.index, 1 - L.mean_cos, color=c)
ax[0].set(xlabel="layer", ylabel="relative difference ||A-B||/||A||", yscale="log", title="Weight difference per layer")
ax[1].set(xlabel="layer", ylabel="1 - cosine similarity", yscale="log", title="Cosine distance per layer")
ax[0].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.33)); fig.tight_layout(); fig.savefig(FIG / "r2_weight_similarity.png", dpi=150)
