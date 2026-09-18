"""
probe_utils_gsm8k.py
────────────────────
Shared utilities for all GSM8K v3 probing ablation scripts.

Task: Regression — predict standardized numeric answer from hidden states.
Probes:
  - sklearn Ridge: closed-form, scale-invariant, robust across all layers
  - PyTorch MLP:  nonlinear probe with LayerNorm input to handle scale differences

Activations: gsm8k_v3 format, shape (37, 100, 2560), both sampled and averaged.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# ==========================================
# PATHS
# ==========================================
BASE_DIR     = "/playpen-ssd/yuvraj/llm-probing-project"
SAMPLED_DIR  = os.path.join(BASE_DIR, "activations/gsm8k_v3/sampled")
AVERAGED_DIR = os.path.join(BASE_DIR, "activations/gsm8k_v3/averaged")
RESULTS_BASE = os.path.join(BASE_DIR, "results/probing/gsm8k_v3")
MODELS_BASE  = os.path.join(BASE_DIR, "saved_models/gsm8k_v3")

# ==========================================
# CONSTANTS
# ==========================================
N_POSITIONS  = 100
HIDDEN_DIM   = 2560
NUM_LAYERS   = 37
BATCH_SIZE   = 64
EPOCHS       = 10
TEST_SIZE    = 0.2
RANDOM_STATE = 42
COLORS       = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]


# ==========================================
# DATA LOADING
# ==========================================
def load_gsm8k_v3_df(n_samples=None):
    """
    Merges all worker CSVs from gsm8k_v3, filters for valid examples,
    and returns a clean DataFrame ready for probing.
    """
    csv_dir = os.path.join(BASE_DIR, "activations/gsm8k_v3")
    dfs = []
    for f in glob.glob(os.path.join(csv_dir, "master_log_worker*.csv")):
        dfs.append(pd.read_csv(f))

    if not dfs:
        raise FileNotFoundError(f"No worker CSVs found in {csv_dir}")

    df = pd.concat(dfs, ignore_index=True).drop_duplicates('example_id')
    df = df.sort_values("example_id").reset_index(drop=True)

    # Drop resumed placeholders
    if 'resumed' in df.columns:
        df = df[df['resumed'] == False].reset_index(drop=True)

    # Parse numeric answer
    df['reg_target'] = pd.to_numeric(df['extracted_answer'], errors='coerce')
    df = df.dropna(subset=['reg_target'])

    # Keep only examples with \boxed{} completed
    df = df[df['model_generation'].str.contains('boxed', na=False)]

    # Keep only examples where both .pt files exist and are valid
    def pt_valid(row):
        sp = os.path.join(SAMPLED_DIR,  row['sampled_tensor_file'])
        ap = os.path.join(AVERAGED_DIR, row['averaged_tensor_file'])
        return (os.path.exists(sp) and os.path.getsize(sp) > 1024 and
                os.path.exists(ap) and os.path.getsize(ap) > 1024)

    df = df[df.apply(pt_valid, axis=1)].reset_index(drop=True)
    print(f"Valid examples after filtering: {len(df)}")
    print(f"Answer range: {df['reg_target'].min():.0f} to {df['reg_target'].max():.0f}")
    print(f"Answer mean:  {df['reg_target'].mean():.1f} | std: {df['reg_target'].std():.1f}")

    if n_samples is not None:
        df = df.head(n_samples)
        print(f"[N_SAMPLES={n_samples}] Using first {n_samples} examples.")

    return df


class ProbeDataModule:
    def __init__(self, df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        self.df = df
        self.train_df, self.test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        print(f"Train: {len(self.train_df)} | Test: {len(self.test_df)}")

        # Standardization from train set only — prevents leakage
        self.target_mean = self.train_df['reg_target'].mean()
        self.target_std  = self.train_df['reg_target'].std()
        print(f"Target mean: {self.target_mean:.2f} | std: {self.target_std:.2f}")


# ==========================================
# ACTIVATION CACHE
# ==========================================
class ActivationCache:
    """
    Loads all v3 .pt files for a split into RAM.
    Each file: (37, 100, 2560), 18MB — total ~30GB for 1500 examples.
    """
    def __init__(self, df, activation_type="sampled"):
        assert activation_type in ("sampled", "averaged")
        act_dir  = SAMPLED_DIR  if activation_type == "sampled"  else AVERAGED_DIR
        file_col = "sampled_tensor_file" if activation_type == "sampled" else "averaged_tensor_file"

        print(f"Loading {len(df)} files into RAM ({activation_type})...")
        self.cache    = {}
        self.df       = df.reset_index(drop=True)
        self.file_col = file_col
        self.act_dir  = act_dir

        for _, row in df.iterrows():
            fpath = os.path.join(act_dir, row[file_col])
            if os.path.exists(fpath) and os.path.getsize(fpath) > 1024:
                self.cache[row[file_col]] = torch.load(fpath, map_location='cpu')

        print(f"Cache built. {len(self.cache)} files loaded.")

    def get_tensor(self, row):
        return self.cache.get(row[self.file_col], None)


# ==========================================
# DATASET BUILDER
# ==========================================
def build_numpy_dataset(cache, target_mean, target_std, layer, slice_fn):
    """
    Builds numpy arrays (X, y) for a given layer using slice_fn.
    Used by Ridge (sklearn).
    Each example contributes len(slice_fn(row)) rows to X.
    y is the standardized numeric answer, replicated per token.
    """
    xs, ys = [], []
    for _, row in cache.df.iterrows():
        act = cache.get_tensor(row)
        if act is None:
            continue
        indices = slice_fn(row)
        if len(indices) == 0:
            continue
        std_target = (float(row['reg_target']) - target_mean) / (target_std + 1e-8)
        xs.append(act[layer, indices, :].float().numpy())
        ys.append(np.full(len(indices), std_target, dtype=np.float32))

    if not xs:
        return np.zeros((0, HIDDEN_DIM)), np.zeros(0)
    return np.vstack(xs), np.concatenate(ys)


def build_torch_dataset(cache, target_mean, target_std, layer, slice_fn):
    """
    Builds TensorDataset for a given layer using slice_fn.
    Used by PyTorch MLP.
    """
    xs, ys = [], []
    for _, row in cache.df.iterrows():
        act = cache.get_tensor(row)
        if act is None:
            continue
        indices = slice_fn(row)
        if len(indices) == 0:
            continue
        std_target = (float(row['reg_target']) - target_mean) / (target_std + 1e-8)
        xs.append(act[layer, indices, :].float())
        ys.append(torch.full((len(indices),), std_target, dtype=torch.float32))

    if not xs:
        return TensorDataset(torch.zeros(0, HIDDEN_DIM), torch.zeros(0))
    return TensorDataset(torch.cat(xs), torch.cat(ys))


# ==========================================
# PROBE MODELS
# ==========================================
class MLPProbe(nn.Module):
    """
    MLP with LayerNorm as first layer to handle the 1000x scale
    difference across transformer layers (layer 0 std=0.008 vs
    layer 35 std=8.84). Without this, Adam behaves very differently
    per layer and results are not comparable.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, bottleneck_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),   # normalize inputs — critical for cross-layer comparability
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ==========================================
# TRAINING & EVALUATION
# ==========================================
def train_mlp(model, train_loader, optimizer, epochs=EPOCHS):
    criterion = nn.MSELoss()
    model.train()
    device = next(model.parameters()).device
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
    return model


@torch.no_grad()
def evaluate_mlp(model, test_loader, target_mean, target_std):
    model.eval()
    device = next(model.parameters()).device
    preds, targets = [], []
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        preds.append(model(batch_x).cpu().numpy())
        targets.append(batch_y.numpy())
    P = np.concatenate(preds)
    T = np.concatenate(targets)
    r2      = r2_score(T, P)
    mse_std = float(np.mean((T - P) ** 2))
    P_orig  = P * target_std + target_mean
    T_orig  = T * target_std + target_mean
    mse_orig = float(np.mean((T_orig - P_orig) ** 2))
    return {"r2": r2, "mse_std": mse_std, "mse_orig": mse_orig}


def train_eval_ridge(X_train, y_train, X_test, y_test, target_mean, target_std):
    """
    Ridge regression — closed-form solution, scale-invariant.
    alpha=1.0 provides mild regularization.
    """
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    P       = ridge.predict(X_test)
    T       = y_test
    r2      = r2_score(T, P)
    mse_std = float(np.mean((T - P) ** 2))
    P_orig  = P * target_std + target_mean
    T_orig  = T * target_std + target_mean
    mse_orig = float(np.mean((T_orig - P_orig) ** 2))
    return {"r2": r2, "mse_std": mse_std, "mse_orig": mse_orig}, ridge


# ==========================================
# PLOTTING
# ==========================================
def save_plots(results_df, num_layers, results_dir, ablation_name, activation_type):
    layers = results_df['layer']
    title_suffix = f"{ablation_name} | {activation_type}\n(Qwen3-4B on GSM8K v3)"

    # R² — primary metric
    plt.figure(figsize=(12, 5))
    plt.plot(layers, results_df['ridge_r2'], marker='o', color=COLORS[0], label='Ridge')
    plt.plot(layers, results_df['mlp_r2'],   marker='s', color=COLORS[1], label='MLP')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No-skill (R²=0)')
    plt.title(f"Layer-wise R² — {title_suffix}")
    plt.xlabel("Layer")
    plt.ylabel("R²")
    plt.xticks(range(0, num_layers, 2))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"r2_{activation_type}.png"), dpi=300)
    plt.close()

    # MSE (std)
    plt.figure(figsize=(12, 5))
    plt.plot(layers, results_df['ridge_mse_std'], marker='o', color=COLORS[0], label='Ridge')
    plt.plot(layers, results_df['mlp_mse_std'],   marker='s', color=COLORS[1], label='MLP')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No-skill (MSE=1.0)')
    plt.title(f"Layer-wise MSE (std) — {title_suffix}")
    plt.xlabel("Layer")
    plt.ylabel("MSE (std)")
    plt.xticks(range(0, num_layers, 2))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"mse_std_{activation_type}.png"), dpi=300)
    plt.close()

    print(f"  Plots saved → {results_dir}/")


# ==========================================
# MAIN ABLATION RUNNER
# ==========================================
def run_ablation(ablation_name, slice_fn, n_samples=None):
    """
    Main entry point for each ablation script.

    ablation_name : str   — used for results/models directory naming
    slice_fn      : callable(row) -> list of position indices [0..99]
    n_samples     : int or None — for quick tests with fewer examples
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"ABLATION: {ablation_name.upper()}")
    print(f"Device: {device} | N_SAMPLES: {n_samples}")
    print(f"{'='*60}")

    df          = load_gsm8k_v3_df(n_samples=n_samples)
    data_module = ProbeDataModule(df)

    results_dir = os.path.join(RESULTS_BASE, ablation_name)
    models_dir  = os.path.join(MODELS_BASE,  ablation_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, "mlp"),   exist_ok=True)

    for activation_type in ("sampled", "averaged"):
        print(f"\n{'─'*50}")
        print(f"Activation type: {activation_type}")
        print(f"{'─'*50}")

        train_cache = ActivationCache(data_module.train_df, activation_type)
        test_cache  = ActivationCache(data_module.test_df,  activation_type)

        all_results = []

        for layer in range(NUM_LAYERS):
            print(f"\n── Layer {layer}/{NUM_LAYERS-1} ──")

            # Build numpy datasets for Ridge
            X_tr, y_tr = build_numpy_dataset(
                train_cache, data_module.target_mean, data_module.target_std,
                layer, slice_fn
            )
            X_te, y_te = build_numpy_dataset(
                test_cache, data_module.target_mean, data_module.target_std,
                layer, slice_fn
            )

            # Build torch datasets for MLP
            train_ds = build_torch_dataset(
                train_cache, data_module.target_mean, data_module.target_std,
                layer, slice_fn
            )
            test_ds = build_torch_dataset(
                test_cache, data_module.target_mean, data_module.target_std,
                layer, slice_fn
            )

            if len(X_tr) == 0 or len(X_te) == 0:
                print("  WARNING: empty dataset, skipping.")
                continue

            row = {"layer": layer}

            # --- Ridge ---
            m, ridge_model = train_eval_ridge(
                X_tr, y_tr, X_te, y_te,
                data_module.target_mean, data_module.target_std
            )
            row["ridge_r2"] = m["r2"]; row["ridge_mse_std"] = m["mse_std"]; row["ridge_mse_orig"] = m["mse_orig"]
            print(f"  Ridge → R²: {m['r2']:.4f} | MSE(std): {m['mse_std']:.4f}")

            # --- MLP ---
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
            mlp          = MLPProbe().to(device)
            optimizer    = optim.Adam(mlp.parameters(), lr=1e-3)
            train_mlp(mlp, train_loader, optimizer, epochs=EPOCHS)
            m = evaluate_mlp(mlp, test_loader, data_module.target_mean, data_module.target_std)
            row["mlp_r2"] = m["r2"]; row["mlp_mse_std"] = m["mse_std"]; row["mlp_mse_orig"] = m["mse_orig"]
            print(f"  MLP   → R²: {m['r2']:.4f} | MSE(std): {m['mse_std']:.4f}")
            torch.save(mlp.state_dict(), os.path.join(models_dir, "mlp", f"layer_{layer}_{activation_type}.pt"))

            all_results.append(row)

            # Incremental checkpoint
            pd.DataFrame(all_results).to_csv(
                os.path.join(results_dir, f"results_{activation_type}.csv"), index=False
            )

        # Final save + plots
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(results_dir, f"results_{activation_type}.csv"), index=False)
        save_plots(results_df, NUM_LAYERS, results_dir, ablation_name, activation_type)

        best_ridge = results_df.loc[results_df['ridge_r2'].idxmax(), 'layer']
        best_mlp   = results_df.loc[results_df['mlp_r2'].idxmax(),   'layer']
        print(f"\n  Best layer (Ridge R²): Layer {best_ridge} ({results_df['ridge_r2'].max():.4f})")
        print(f"  Best layer (MLP R²):   Layer {best_mlp}   ({results_df['mlp_r2'].max():.4f})")

    print(f"\n{'='*60}")
    print(f"ABLATION COMPLETE: {ablation_name}")
    print(f"Results → {results_dir}/")
    print(f"{'='*60}")