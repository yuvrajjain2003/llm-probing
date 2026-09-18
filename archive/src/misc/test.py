"""
tsne_logiqa.py
──────────────
t-SNE visualization of LogiQA v3 hidden states colored by ground truth label (A/B/C/D).
Compares early layer (layer 1) vs late layer (layer 36) vs last think token layer.
Uses last_think_token position and pos99 position for comparison.
"""

import os, glob, torch
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

BASE_DIR    = "/playpen-ssd/yuvraj/llm-probing-project"
SAMPLED_DIR = os.path.join(BASE_DIR, "activations/logiqa_v3/sampled")
RESULTS_DIR = os.path.join(BASE_DIR, "results/probing/logiqa_v3/tsne")
os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL_MAP   = {"A": 0, "B": 1, "C": 2, "D": 3}
LABEL_NAMES = ["A", "B", "C", "D"]
COLORS      = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]
N_SAMPLES   = 400   # keep small for t-SNE speed
RANDOM_STATE = 42

# ==========================================
# LOAD DATA
# ==========================================
dfs = [pd.read_csv(f) for f in glob.glob(os.path.join(BASE_DIR, "activations/logiqa_v3/master_log_worker*.csv"))]
df = pd.concat(dfs, ignore_index=True).drop_duplicates('example_id')
df = df[df['resumed'] == False]
df = df[df['ground_truth'].isin(['A','B','C','D'])]
df['label'] = df['ground_truth'].map(LABEL_MAP)
df = df[df['think_end_pos'].notna() & df['sequence_length'].notna()]

def pt_valid(row):
    sp = os.path.join(SAMPLED_DIR, row['sampled_tensor_file'])
    return os.path.exists(sp) and os.path.getsize(sp) > 1024
df = df[df.apply(pt_valid, axis=1)].reset_index(drop=True)

# Sample N_SAMPLES examples stratified by label
df = df.groupby('label').apply(lambda x: x.sample(min(len(x), N_SAMPLES//4), random_state=RANDOM_STATE)).reset_index(drop=True)
print(f"Using {len(df)} examples for t-SNE")
print(f"Label dist: {df['label'].value_counts().sort_index().to_dict()}")

# ==========================================
# LOAD CACHE
# ==========================================
print("Loading cache...")
cache = {}
for _, row in df.iterrows():
    fpath = os.path.join(SAMPLED_DIR, row['sampled_tensor_file'])
    if os.path.exists(fpath):
        cache[row['sampled_tensor_file']] = torch.load(fpath, map_location='cpu')
print(f"Loaded {len(cache)} files")

def get_last_think_pos(row):
    think_frac = min(float(row['think_end_pos']) / float(row['sequence_length']), 1.0)
    n_think = max(1, int(think_frac * 100))
    return min(n_think - 1, 99)

# ==========================================
# EXTRACT HIDDEN STATES
# ==========================================
def extract_X(layer, pos_fn):
    X, y = [], []
    for _, row in df.iterrows():
        act = cache.get(row['sampled_tensor_file'])
        if act is None: continue
        pos = pos_fn(row)
        X.append(act[layer, pos, :].float().numpy())
        y.append(int(row['label']))
    return np.array(X), np.array(y)

# ==========================================
# PLOT t-SNE
# ==========================================
def plot_tsne(X, y, title, fname):
    print(f"  Running t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8, 7))
    for label_idx in range(4):
        mask = y == label_idx
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=COLORS[label_idx], label=LABEL_NAMES[label_idx],
            alpha=0.7, s=40, edgecolors='none'
        )
    plt.title(title, fontsize=13)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(title="Ground Truth", markerscale=1.5)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=300)
    plt.close()
    print(f"  Saved → {fname}")

# ==========================================
# CONFIGURATIONS TO PLOT
# ==========================================
configs = [
    # (layer, pos_fn, title, fname)
    (0,  lambda row: 1,                  "Layer 0 — First Think Token\n(Qwen3-4B on LogiQA v3)",          "tsne_layer0_first_think.png"),
    (1,  lambda row: 1,                  "Layer 1 — First Think Token\n(Qwen3-4B on LogiQA v3)",          "tsne_layer1_first_think.png"),
    (18, get_last_think_pos,             "Layer 18 — Last Think Token\n(Qwen3-4B on LogiQA v3)",          "tsne_layer18_last_think.png"),
    (36, get_last_think_pos,             "Layer 36 — Last Think Token\n(Qwen3-4B on LogiQA v3)",          "tsne_layer36_last_think.png"),
    (1,  lambda row: 99,                 "Layer 1 — Last Token (pos99)\n(Qwen3-4B on LogiQA v3)",         "tsne_layer1_pos99.png"),
    (36, lambda row: 99,                 "Layer 36 — Last Token (pos99)\n(Qwen3-4B on LogiQA v3)",        "tsne_layer36_pos99.png"),
]

for layer, pos_fn, title, fname in configs:
    print(f"\nProcessing: {title}")
    X, y = extract_X(layer, pos_fn)
    plot_tsne(X, y, title, fname)

# ==========================================
# COMBINED 2x3 FIGURE
# ==========================================
print("\nCreating combined figure...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for ax_idx, (layer, pos_fn, title, fname) in enumerate(configs):
    X, y = extract_X(layer, pos_fn)
    print(f"  t-SNE for subplot {ax_idx+1}/6...")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    ax = axes[ax_idx]
    for label_idx in range(4):
        mask = y == label_idx
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=COLORS[label_idx], label=LABEL_NAMES[label_idx],
                   alpha=0.7, s=30, edgecolors='none')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t-SNE dim 1", fontsize=8)
    ax.set_ylabel("t-SNE dim 2", fontsize=8)
    ax.grid(True, alpha=0.2)
    if ax_idx == 0:
        ax.legend(title="Ground Truth", markerscale=1.5, fontsize=8)

plt.suptitle("t-SNE of Qwen3-4B Hidden States — LogiQA v3\nColored by Ground Truth Answer (A/B/C/D)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "tsne_combined.png"), dpi=300)
plt.close()
print(f"\nCombined figure saved → {RESULTS_DIR}/tsne_combined.png")
print("Done!")