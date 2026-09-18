"""
probe_thinking_length_logiqa_v2.py
────────────────────────────────────
Option 3: Does thinking length correlate with hidden state signal?

We bin examples into 3 groups by think_end_pos:
  - short  : bottom tercile of think_end_pos
  - medium : middle tercile
  - long   : top tercile

For each bin, at each layer, we train a 4-class probe (LogReg) to predict
the correct answer (A/B/C/D) from the hidden state at pre_answer position
(seq_len - 9), the most informative position we identified.

Key question: do examples where the model thinks longer show stronger
hidden state signal (higher probe accuracy)?

  long >> short → more thinking → better encoded answer
  short >= long → thinking length doesn't help; answer is determined early
"""

import os, glob, torch
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# ==========================================
# CONFIG
# ==========================================
BASE_DIR    = "/playpen-ssd/yuvraj/llm-probing-project"
ACT_DIR     = os.path.join(BASE_DIR, "activations/logiqa_v2")
RESULTS_DIR = os.path.join(BASE_DIR, "results/thinking_length_probe/logiqa_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_LAYERS    = 37
HIDDEN_DIM    = 2560
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
ANSWER_OFFSET = 9

LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
COLORS    = {"short": "#E69F00", "medium": "#56B4E9", "long": "#009E73"}

# ==========================================
# LOAD DATA
# ==========================================
dfs = [pd.read_csv(f) for f in glob.glob(os.path.join(ACT_DIR, "master_log_worker*.csv"))]
df  = pd.concat(dfs, ignore_index=True).drop_duplicates("example_id")
if "resumed" in df.columns:
    df = df[df["resumed"] == False].reset_index(drop=True)

df = df[df["model_generation"].str.contains("boxed", na=False)]
df = df[df["think_end_pos"].notna()]
df = df[df["ground_truth"].isin(["A", "B", "C", "D"])]
df["label"] = df["ground_truth"].map(LABEL_MAP)

def tensor_valid(row):
    p = os.path.join(ACT_DIR, row["tensor_file"])
    return os.path.exists(p) and os.path.getsize(p) > 1024
df = df[df.apply(tensor_valid, axis=1)].reset_index(drop=True)

# Bin by thinking length into terciles
terciles = df["think_end_pos"].quantile([1/3, 2/3]).values
df["think_bin"] = pd.cut(
    df["think_end_pos"],
    bins=[-np.inf, terciles[0], terciles[1], np.inf],
    labels=["short", "medium", "long"]
)

print(f"Valid examples: {len(df)}")
print(f"\nThinking length tercile boundaries: {terciles.astype(int)}")
print(f"\nBin distribution:")
print(df["think_bin"].value_counts().sort_index())
print(f"\nMean think_end_pos per bin:")
print(df.groupby("think_bin")["think_end_pos"].mean().astype(int))
print(f"\nAccuracy per bin:")
print(df.groupby("think_bin")["is_correct"].mean().round(3))

# ==========================================
# LOAD ACTIVATION CACHE
# ==========================================
print("\nLoading activation cache...")
cache = {}
for _, row in df.iterrows():
    fpath = os.path.join(ACT_DIR, row["tensor_file"])
    cache[row["tensor_file"]] = torch.load(fpath, map_location="cpu")
print(f"Cache loaded: {len(cache)} tensors")

# ==========================================
# MAIN LOOP — per bin, per layer
# ==========================================
all_records = []

for layer in range(NUM_LAYERS):
    print(f"\n── Layer {layer}/{NUM_LAYERS-1} ──", flush=True)
    rec = {"layer": layer}

    for bin_name in ["short", "medium", "long"]:
        bin_df = df[df["think_bin"] == bin_name].reset_index(drop=True)

        train_df, test_df = train_test_split(
            bin_df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=bin_df["label"]
        )

        # Build arrays at pre_answer position
        X_tr, y_tr = [], []
        for _, row in train_df.iterrows():
            act = cache[row["tensor_file"]]
            pos = max(0, int(row["sequence_length"]) - ANSWER_OFFSET)
            X_tr.append(act[layer, pos, :].float().numpy())
            y_tr.append(int(row["label"]))

        X_te, y_te = [], []
        for _, row in test_df.iterrows():
            act = cache[row["tensor_file"]]
            pos = max(0, int(row["sequence_length"]) - ANSWER_OFFSET)
            X_te.append(act[layer, pos, :].float().numpy())
            y_te.append(int(row["label"]))

        X_tr, y_tr = np.array(X_tr), np.array(y_tr)
        X_te, y_te = np.array(X_te), np.array(y_te)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = LogisticRegression(
            max_iter=200, class_weight='balanced',
            random_state=RANDOM_STATE, solver='saga', C=1.0,
            multi_class='multinomial'
        )
        clf.fit(X_tr_s, y_tr)
        P = clf.predict(X_te_s)

        rec[f"{bin_name}_acc"]     = accuracy_score(y_te, P)
        rec[f"{bin_name}_bal_acc"] = balanced_accuracy_score(y_te, P)
        rec[f"{bin_name}_n"]       = len(bin_df)

        print(f"  [{bin_name:6s} n={len(bin_df)}] "
              f"acc={rec[f'{bin_name}_acc']:.4f} "
              f"bal_acc={rec[f'{bin_name}_bal_acc']:.4f}", flush=True)

    all_records.append(rec)
    pd.DataFrame(all_records).to_csv(
        os.path.join(RESULTS_DIR, "thinking_length_probe_results.csv"), index=False
    )

# ==========================================
# PLOTS
# ==========================================
results_df = pd.DataFrame(all_records)
layers     = results_df["layer"]

# Get bin sizes and mean think lengths for labels
bin_sizes   = df.groupby("think_bin")["think_end_pos"].agg(["count", "mean"]).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Does Thinking Length Affect Answer Encoding?\n"
             "(Qwen3-4B on LogiQA v2, probe at pre-answer position)", fontsize=13)

for ax, metric, ylabel in zip(axes,
                               ["acc",     "bal_acc"],
                               ["Accuracy", "Balanced Accuracy"]):
    for bin_name in ["short", "medium", "long"]:
        n    = bin_sizes.loc[bin_name, "count"]
        mean = bin_sizes.loc[bin_name, "mean"]
        ax.plot(layers, results_df[f"{bin_name}_{metric}"],
                marker='o', markersize=3, color=COLORS[bin_name],
                label=f"{bin_name} (n={n}, mean_think={mean})")
    ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (25%)')
    ax.set_title(f"LogReg {ylabel} by Thinking Length Bin")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.15, 1.0)
    ax.set_xticks(range(0, NUM_LAYERS, 2))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "thinking_length_probe.png"), dpi=300)
plt.close()

# Also plot: mean accuracy per bin as a bar chart at the best layer
best_layer = results_df.loc[results_df["short_bal_acc"].idxmax(), "layer"]
print(f"\nBest layer (short bal_acc): {best_layer}")

fig, ax = plt.subplots(figsize=(7, 5))
bins  = ["short", "medium", "long"]
accs  = [results_df.loc[results_df["layer"] == best_layer, f"{b}_bal_acc"].values[0] for b in bins]
means = [bin_sizes.loc[b, "mean"] for b in bins]
bars  = ax.bar(bins, accs, color=[COLORS[b] for b in bins], alpha=0.85)
ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Chance')
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'mean think\n={mean} tok', ha='center', va='bottom', fontsize=9)
ax.set_title(f"Balanced Accuracy at Best Layer (Layer {best_layer})\nby Thinking Length Bin")
ax.set_ylabel("Balanced Accuracy")
ax.set_ylim(0, 0.8)
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "thinking_length_bar.png"), dpi=300)
plt.close()

print(f"\nAll plots saved → {RESULTS_DIR}/")
print("DONE")