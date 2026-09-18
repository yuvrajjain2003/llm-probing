"""
probe_correctness_logiqa_v2.py
───────────────────────────────
Option 2: Can we predict whether the model will get the answer right or wrong
from its hidden states alone?

Balanced dataset: 200 correct + 200 incorrect examples (randomly sampled).

At each layer, train a binary probe (LogReg + MLP) to predict is_correct (0/1)
from the hidden state at 3 positions:
  - first_think : position ~5 (before any reasoning)
  - last_think  : think_end_pos (end of reasoning)
  - pre_answer  : seq_len - 9 (just before committing to answer)

Key question: at which layer AND position does the model's hidden state
best predict whether it will answer correctly?

  first_think >> chance early  → model "knows" before reasoning begins
  last_think is best           → reasoning genuinely determines the outcome
  pre_answer is best           → decision is made at the last moment

Metrics: accuracy, balanced accuracy (should be same given balanced data)
"""

import os, glob, torch, torch.nn as nn, torch.optim as optim
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# ==========================================
# CONFIG
# ==========================================
BASE_DIR    = "/playpen-ssd/yuvraj/llm-probing-project"
ACT_DIR     = os.path.join(BASE_DIR, "activations/logiqa_v2")
RESULTS_DIR = os.path.join(BASE_DIR, "results/correctness_probe/logiqa_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_LAYERS    = 37
HIDDEN_DIM    = 2560
BATCH_SIZE    = 64
EPOCHS        = 15
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
ANSWER_OFFSET = 9
N_PER_CLASS   = 200   # 200 correct + 200 incorrect = 400 total

POSITIONS = ["first_think", "last_think", "pre_answer"]
COLORS    = {"first_think": "#CC79A7", "last_think": "#56B4E9", "pre_answer": "#E69F00"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==========================================
# MLP PROBE (binary)
# ==========================================
class MLPProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def train_mlp(model, X_tr, y_tr):
    ds     = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                           torch.tensor(y_tr, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    crit   = nn.BCEWithLogitsLoss()
    opt    = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(EPOCHS):
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx.to(device)), by.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


@torch.no_grad()
def eval_mlp(model, X_te, y_te):
    model.eval()
    logits = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
    preds  = (logits > 0).astype(int)
    return {
        "acc":          accuracy_score(y_te, preds),
        "balanced_acc": balanced_accuracy_score(y_te, preds),
    }

# ==========================================
# LOAD & BALANCE DATA
# ==========================================
dfs = [pd.read_csv(f) for f in glob.glob(os.path.join(ACT_DIR, "master_log_worker*.csv"))]
df  = pd.concat(dfs, ignore_index=True).drop_duplicates("example_id")
if "resumed" in df.columns:
    df = df[df["resumed"] == False].reset_index(drop=True)

df = df[df["model_generation"].str.contains("boxed", na=False)]
df = df[df["think_end_pos"].notna()]
df = df[df["is_correct"].isin([0, 1])]

def tensor_valid(row):
    p = os.path.join(ACT_DIR, row["tensor_file"])
    return os.path.exists(p) and os.path.getsize(p) > 1024
df = df[df.apply(tensor_valid, axis=1)].reset_index(drop=True)

# Balance: sample N_PER_CLASS from each class
correct_df   = df[df["is_correct"] == 1].sample(n=N_PER_CLASS, random_state=RANDOM_STATE)
incorrect_df = df[df["is_correct"] == 0].sample(n=N_PER_CLASS, random_state=RANDOM_STATE)
df_balanced  = pd.concat([correct_df, incorrect_df]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"Balanced dataset: {len(df_balanced)} examples")
print(f"Correct: {df_balanced['is_correct'].sum()} | Incorrect: {(df_balanced['is_correct']==0).sum()}")

train_df, test_df = train_test_split(
    df_balanced, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df_balanced["is_correct"]
)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ==========================================
# POSITION FUNCTIONS
# ==========================================
def get_pos(row, position):
    seq_len   = int(row["sequence_length"])
    think_end = int(row["think_end_pos"])
    if position == "first_think":
        return min(5, think_end - 1)
    elif position == "last_think":
        return min(think_end, seq_len - 1)
    elif position == "pre_answer":
        return max(0, seq_len - ANSWER_OFFSET)
    return 0

# ==========================================
# LOAD ACTIVATION CACHE
# ==========================================
print("\nLoading activation cache...")
cache = {}
for _, row in df_balanced.iterrows():
    fpath = os.path.join(ACT_DIR, row["tensor_file"])
    cache[row["tensor_file"]] = torch.load(fpath, map_location="cpu")
print(f"Cache loaded: {len(cache)} tensors")

# ==========================================
# MAIN LOOP
# ==========================================
all_records = []

for layer in range(NUM_LAYERS):
    print(f"\n── Layer {layer}/{NUM_LAYERS-1} ──", flush=True)
    rec = {"layer": layer}

    for position in POSITIONS:
        # Build train/test arrays
        X_tr, y_tr = [], []
        for _, row in train_df.iterrows():
            act = cache[row["tensor_file"]]
            pos = get_pos(row, position)
            X_tr.append(act[layer, pos, :].float().numpy())
            y_tr.append(int(row["is_correct"]))

        X_te, y_te = [], []
        for _, row in test_df.iterrows():
            act = cache[row["tensor_file"]]
            pos = get_pos(row, position)
            X_te.append(act[layer, pos, :].float().numpy())
            y_te.append(int(row["is_correct"]))

        X_tr, y_tr = np.array(X_tr), np.array(y_tr)
        X_te, y_te = np.array(X_te), np.array(y_te)

        # LogReg
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        clf = LogisticRegression(
            max_iter=200, random_state=RANDOM_STATE, solver='saga', C=1.0
        )
        clf.fit(X_tr_s, y_tr)
        P = clf.predict(X_te_s)
        rec[f"{position}_logreg_acc"] = accuracy_score(y_te, P)
        rec[f"{position}_logreg_bal_acc"] = balanced_accuracy_score(y_te, P)
        print(f"  [{position}] LogReg acc={rec[f'{position}_logreg_acc']:.4f} "
              f"bal_acc={rec[f'{position}_logreg_bal_acc']:.4f}", flush=True)

        # MLP
        mlp = MLPProbe().to(device)
        train_mlp(mlp, X_tr, y_tr)
        m = eval_mlp(mlp, X_te, y_te)
        rec[f"{position}_mlp_acc"]     = m["acc"]
        rec[f"{position}_mlp_bal_acc"] = m["balanced_acc"]
        print(f"  [{position}] MLP     acc={m['acc']:.4f} "
              f"bal_acc={m['balanced_acc']:.4f}", flush=True)

    all_records.append(rec)
    pd.DataFrame(all_records).to_csv(
        os.path.join(RESULTS_DIR, "correctness_probe_results.csv"), index=False
    )

# ==========================================
# PLOTS
# ==========================================
results_df = pd.DataFrame(all_records)
layers = results_df["layer"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Correctness Probe — Can Hidden States Predict Model Accuracy?\n"
             "(Qwen3-4B on LogiQA v2, 200 correct + 200 incorrect)", fontsize=13)

for ax, probe, probe_label in zip(axes, ["logreg", "mlp"], ["LogReg", "MLP"]):
    for position in POSITIONS:
        ax.plot(layers, results_df[f"{position}_{probe}_acc"],
                marker='o', markersize=3, color=COLORS[position], label=position)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance (50%)')
    ax.set_title(f"{probe_label} Probe")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.set_xticks(range(0, NUM_LAYERS, 2))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "correctness_probe.png"), dpi=300)
plt.close()
print(f"\nPlot saved → {RESULTS_DIR}/correctness_probe.png")
print("DONE")