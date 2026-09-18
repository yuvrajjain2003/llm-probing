"""
probe_gsm8k_regression_filtered.py
────────────────────────────────────
Regression probing on GSM8K v3 activations, filtered to answers in [0, 500].
Runs 4 ablations:
  1. last_think_token  — last position in thinking phase
  2. first_think_token — position 1 (first generated token)
  3. pos98             — second to last position
  4. pos99             — last position (<|im_end|>)

Probe: sklearn Ridge (scale-invariant) + PyTorch MLP with LayerNorm
Target: standardized reg_target (z-score from train set)
Metrics: R², MSE (standardized), MSE (original scale)
Sampled activations only.
"""

import os, glob, torch, torch.nn as nn, torch.optim as optim
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ==========================================
# PATHS & CONFIG
# ==========================================
BASE_DIR    = "/playpen-ssd/yuvraj/llm-probing-project"
SAMPLED_DIR = os.path.join(BASE_DIR, "activations/gsm8k_v3/sampled")
RESULTS_BASE = os.path.join(BASE_DIR, "results/probing/gsm8k_v3_filtered")
os.makedirs(RESULTS_BASE, exist_ok=True)

ANSWER_CAP   = 500
HIDDEN_DIM   = 2560
NUM_LAYERS   = 37
BATCH_SIZE   = 64
EPOCHS       = 10
TEST_SIZE    = 0.2
RANDOM_STATE = 42
COLORS       = ["#E69F00", "#56B4E9"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==========================================
# DATA LOADING
# ==========================================
def load_df():
    dfs = [pd.read_csv(f) for f in glob.glob(os.path.join(BASE_DIR, "activations/gsm8k_v3/master_log_worker*.csv"))]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates('example_id')
    df = df[df['resumed'] == False].reset_index(drop=True)
    df['reg_target'] = pd.to_numeric(df['extracted_answer'], errors='coerce')
    df = df.dropna(subset=['reg_target'])
    df = df[df['model_generation'].str.contains('boxed', na=False)]
    df = df[df['reg_target'] >= 0]
    df = df[df['reg_target'] <= ANSWER_CAP]

    def pt_valid(row):
        sp = os.path.join(SAMPLED_DIR, row['sampled_tensor_file'])
        return os.path.exists(sp) and os.path.getsize(sp) > 1024
    df = df[df.apply(pt_valid, axis=1)].reset_index(drop=True)

    print(f"Valid examples (answer in [0, {ANSWER_CAP}]): {len(df)}")
    print(f"Answer range: {df.reg_target.min():.0f} to {df.reg_target.max():.0f}")
    print(f"Median: {df.reg_target.median():.0f} | Mean: {df.reg_target.mean():.1f} | Std: {df.reg_target.std():.1f}")
    return df


# ==========================================
# MLP PROBE
# ==========================================
class MLPProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def train_mlp(model, X_tr, y_tr):
    ds     = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                           torch.tensor(y_tr, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    crit   = nn.MSELoss()
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
def eval_mlp(model, X_te, y_te, target_mean, target_std):
    model.eval()
    P = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
    return compute_metrics(y_te, P, target_mean, target_std)


def compute_metrics(T, P, target_mean, target_std):
    r2      = r2_score(T, P)
    mse_std = float(np.mean((T - P) ** 2))
    P_orig  = P * target_std + target_mean
    T_orig  = T * target_std + target_mean
    mse_orig = float(np.mean((T_orig - P_orig) ** 2))
    rmse_orig = float(np.sqrt(mse_orig))
    return {"r2": r2, "mse_std": mse_std, "mse_orig": mse_orig, "rmse_orig": rmse_orig}


# ==========================================
# ABLATION DEFINITIONS
# ==========================================
def get_position(row, ablation):
    if ablation == "last_think_token":
        if pd.isna(row.get('think_end_pos')) or pd.isna(row.get('sequence_length')):
            return 74
        think_frac = min(float(row['think_end_pos']) / float(row['sequence_length']), 1.0)
        n_think = max(1, int(think_frac * 100))
        return min(n_think - 1, 99)
    elif ablation == "first_think_token":
        return 1
    elif ablation == "pos98":
        return 98
    elif ablation == "pos99":
        return 99
    return 99


ABLATIONS = ["last_think_token", "first_think_token", "pos98", "pos99"]


# ==========================================
# RUN ONE ABLATION
# ==========================================
def run_ablation(ablation, train_df, test_df, train_cache, test_cache, target_mean, target_std):
    print(f"\n{'='*60}")
    print(f"ABLATION: {ablation.upper()}")
    print(f"{'='*60}")

    results_dir = os.path.join(RESULTS_BASE, ablation)
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    for layer in range(NUM_LAYERS):
        print(f"\n── Layer {layer}/{NUM_LAYERS-1} ──", flush=True)

        X_tr, y_tr = [], []
        for _, row in train_df.iterrows():
            act = train_cache.get(row['sampled_tensor_file'])
            if act is None: continue
            pos = get_position(row, ablation)
            std_target = (float(row['reg_target']) - target_mean) / (target_std + 1e-8)
            X_tr.append(act[layer, pos, :].float().numpy())
            y_tr.append(std_target)

        X_te, y_te = [], []
        for _, row in test_df.iterrows():
            act = test_cache.get(row['sampled_tensor_file'])
            if act is None: continue
            pos = get_position(row, ablation)
            std_target = (float(row['reg_target']) - target_mean) / (target_std + 1e-8)
            X_te.append(act[layer, pos, :].float().numpy())
            y_te.append(std_target)

        X_tr, y_tr = np.array(X_tr), np.array(y_tr)
        X_te, y_te = np.array(X_te), np.array(y_te)

        rr = {"layer": layer}

        # Ridge (with StandardScaler to handle scale differences across layers)
        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)
        ridge   = Ridge(alpha=1.0)
        ridge.fit(X_tr_s, y_tr)
        m = compute_metrics(y_te, ridge.predict(X_te_s), target_mean, target_std)
        rr["ridge_r2"] = m["r2"]; rr["ridge_mse_std"] = m["mse_std"]
        rr["ridge_mse_orig"] = m["mse_orig"]; rr["ridge_rmse_orig"] = m["rmse_orig"]
        print(f"  Ridge → R²: {m['r2']:.4f} | RMSE(orig): {m['rmse_orig']:.1f}", flush=True)

        # MLP
        mlp = MLPProbe().to(device)
        train_mlp(mlp, X_tr, y_tr)
        m = eval_mlp(mlp, X_te, y_te, target_mean, target_std)
        rr["mlp_r2"] = m["r2"]; rr["mlp_mse_std"] = m["mse_std"]
        rr["mlp_mse_orig"] = m["mse_orig"]; rr["mlp_rmse_orig"] = m["rmse_orig"]
        print(f"  MLP   → R²: {m['r2']:.4f} | RMSE(orig): {m['rmse_orig']:.1f}", flush=True)

        all_results.append(rr)
        pd.DataFrame(all_results).to_csv(os.path.join(results_dir, "results_sampled.csv"), index=False)

    # Plots
    results_df = pd.DataFrame(all_results)
    layers = results_df['layer']

    # R² plot
    plt.figure(figsize=(12, 5))
    plt.plot(layers, results_df['ridge_r2'], marker='o', color=COLORS[0], label='Ridge')
    plt.plot(layers, results_df['mlp_r2'],   marker='s', color=COLORS[1], label='MLP')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No-skill (R²=0)')
    plt.title(f"Layer-wise R² — {ablation} | sampled\n(Qwen3-4B on GSM8K v3, answers 0-{ANSWER_CAP})")
    plt.xlabel("Layer"); plt.ylabel("R²")
    plt.xticks(range(0, NUM_LAYERS, 2)); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "r2_sampled.png"), dpi=300); plt.close()

    # RMSE plot
    plt.figure(figsize=(12, 5))
    plt.plot(layers, results_df['ridge_rmse_orig'], marker='o', color=COLORS[0], label='Ridge')
    plt.plot(layers, results_df['mlp_rmse_orig'],   marker='s', color=COLORS[1], label='MLP')
    plt.title(f"Layer-wise RMSE (original scale) — {ablation} | sampled\n(Qwen3-4B on GSM8K v3, answers 0-{ANSWER_CAP})")
    plt.xlabel("Layer"); plt.ylabel("RMSE (answer units)")
    plt.xticks(range(0, NUM_LAYERS, 2)); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "rmse_sampled.png"), dpi=300); plt.close()

    best_layer = results_df.loc[results_df['ridge_r2'].idxmax(), 'layer']
    best_r2    = results_df['ridge_r2'].max()
    print(f"\n  Best layer (Ridge R²): Layer {best_layer} ({best_r2:.4f})")
    print(f"  Results → {results_dir}/")


# ==========================================
# MAIN
# ==========================================
df = load_df()
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

target_mean = train_df['reg_target'].mean()
target_std  = train_df['reg_target'].std()
print(f"Target mean: {target_mean:.2f} | std: {target_std:.2f}")

print("\nLoading train cache...")
train_cache = {}
for _, row in train_df.iterrows():
    fpath = os.path.join(SAMPLED_DIR, row['sampled_tensor_file'])
    if os.path.exists(fpath):
        train_cache[row['sampled_tensor_file']] = torch.load(fpath, map_location='cpu')
print(f"Train cache: {len(train_cache)} files")

print("Loading test cache...")
test_cache = {}
for _, row in test_df.iterrows():
    fpath = os.path.join(SAMPLED_DIR, row['sampled_tensor_file'])
    if os.path.exists(fpath):
        test_cache[row['sampled_tensor_file']] = torch.load(fpath, map_location='cpu')
print(f"Test cache: {len(test_cache)} files")

# Run all 4 ablations
for ablation in ABLATIONS:
    run_ablation(ablation, train_df, test_df, train_cache, test_cache, target_mean, target_std)

print(f"\n{'='*60}")
print("ALL ABLATIONS COMPLETE")
print(f"Results → {RESULTS_BASE}/")
print(f"{'='*60}")