"""
probe_utils_logiqa.py
─────────────────────
Shared utilities for all LogiQA v3 probing ablation scripts.

Task: 4-class classification — predict ground truth answer (A/B/C/D)
      from intermediate hidden states.

Probes:
  - sklearn LogisticRegression: scale-invariant, class_weight='balanced'
  - PyTorch MLP with LayerNorm: nonlinear probe, handles scale differences

Metrics: accuracy, balanced accuracy, macro F1
Activations: logiqa_v3 format, shape (37, 100, 2560), sampled and averaged.
"""

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# ==========================================
# PATHS
# ==========================================
BASE_DIR     = "/playpen-ssd/yuvraj/llm-probing-project"
SAMPLED_DIR  = os.path.join(BASE_DIR, "activations/logiqa_v3/sampled")
AVERAGED_DIR = os.path.join(BASE_DIR, "activations/logiqa_v3/averaged")
RESULTS_BASE = os.path.join(BASE_DIR, "results/probing/logiqa_v3")
MODELS_BASE  = os.path.join(BASE_DIR, "saved_models/logiqa_v3")

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
NUM_CLASSES  = 4
LABEL_MAP    = {"A": 0, "B": 1, "C": 2, "D": 3}
COLORS       = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]


# ==========================================
# DATA LOADING
# ==========================================
def load_logiqa_v3_df(n_samples=None):
    """
    Merges all worker CSVs from logiqa_v3, filters for valid examples,
    encodes ground truth A/B/C/D as 0/1/2/3.
    """
    csv_dir = os.path.join(BASE_DIR, "activations/logiqa_v3")
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

    # Keep only examples with valid ground truth letter
    df = df[df['ground_truth'].isin(['A', 'B', 'C', 'D'])]

    # Encode ground truth as integer label
    df['label'] = df['ground_truth'].map(LABEL_MAP)

    # Keep only examples where both .pt files exist and are valid
    def pt_valid(row):
        sp = os.path.join(SAMPLED_DIR,  row['sampled_tensor_file'])
        ap = os.path.join(AVERAGED_DIR, row['averaged_tensor_file'])
        return (os.path.exists(sp) and os.path.getsize(sp) > 1024 and
                os.path.exists(ap) and os.path.getsize(ap) > 1024)

    df = df[df.apply(pt_valid, axis=1)].reset_index(drop=True)
    print(f"Valid examples after filtering: {len(df)}")
    print(f"Ground truth distribution: {df['ground_truth'].value_counts().to_dict()}")
    print(f"Label distribution: {df['label'].value_counts().sort_index().to_dict()}")

    if n_samples is not None:
        df = df.head(n_samples)
        print(f"[N_SAMPLES={n_samples}] Using first {n_samples} examples.")

    return df


class ProbeDataModule:
    def __init__(self, df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        self.df = df
        self.train_df, self.test_df = train_test_split(
            df, test_size=test_size, random_state=random_state,
            stratify=df['label']  # stratify to preserve class balance
        )
        print(f"Train: {len(self.train_df)} | Test: {len(self.test_df)}")
        print(f"Train label dist: {self.train_df['label'].value_counts().sort_index().to_dict()}")
        print(f"Test  label dist: {self.test_df['label'].value_counts().sort_index().to_dict()}")


# ==========================================
# ACTIVATION CACHE
# ==========================================
class ActivationCache:
    """
    Loads all v3 .pt files for a split into RAM.
    Each file: (37, 100, 2560), 18MB.
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
# DATASET BUILDERS
# ==========================================
def build_numpy_dataset(cache, layer, slice_fn):
    """
    Builds numpy arrays (X, y) for sklearn LogisticRegression.
    y = integer label 0/1/2/3 for A/B/C/D, replicated per token.
    """
    xs, ys = [], []
    for _, row in cache.df.iterrows():
        act = cache.get_tensor(row)
        if act is None:
            continue
        indices = slice_fn(row)
        if len(indices) == 0:
            continue
        label = int(row['label'])
        xs.append(act[layer, indices, :].float().numpy())
        ys.extend([label] * len(indices))

    if not xs:
        return np.zeros((0, HIDDEN_DIM)), np.zeros(0, dtype=int)
    return np.vstack(xs), np.array(ys, dtype=int)


def build_torch_dataset(cache, layer, slice_fn):
    """
    Builds TensorDataset for PyTorch MLP.
    y = long tensor of integer labels 0/1/2/3.
    """
    xs, ys = [], []
    for _, row in cache.df.iterrows():
        act = cache.get_tensor(row)
        if act is None:
            continue
        indices = slice_fn(row)
        if len(indices) == 0:
            continue
        label = int(row['label'])
        xs.append(act[layer, indices, :].float())
        ys.append(torch.full((len(indices),), label, dtype=torch.long))

    if not xs:
        return TensorDataset(torch.zeros(0, HIDDEN_DIM), torch.zeros(0, dtype=torch.long))
    return TensorDataset(torch.cat(xs), torch.cat(ys))


# ==========================================
# PROBE MODELS
# ==========================================
class MLPProbe(nn.Module):
    """
    4-class MLP with LayerNorm input to handle scale differences
    across transformer layers (layer 0 std=0.008 vs layer 35 std=8.84).
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, bottleneck_dim=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # (batch, num_classes)


# ==========================================
# TRAINING & EVALUATION
# ==========================================
def train_mlp(model, train_loader, optimizer, class_weights, epochs=EPOCHS):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    model.train()
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
def evaluate_mlp(model, test_loader):
    model.eval()
    device = next(model.parameters()).device
    preds, targets = [], []
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x).cpu()
        preds.append(logits.argmax(dim=-1).numpy())
        targets.append(batch_y.numpy())
    P = np.concatenate(preds)
    T = np.concatenate(targets)
    return compute_metrics(T, P)


def train_eval_logreg(X_train, y_train, X_test, y_test):
    """
    sklearn LogisticRegression — scale-invariant, class_weight='balanced'
    handles the slight A/B/C/D imbalance automatically.
    """
    clf = LogisticRegression(
        max_iter=100,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        solver='saga',
        multi_class='multinomial',
        C=1.0,
    )
    clf.fit(X_train, y_train)
    P = clf.predict(X_test)
    return compute_metrics(y_test, P), clf


def compute_metrics(T, P):
    return {
        "acc":          accuracy_score(T, P),
        "balanced_acc": balanced_accuracy_score(T, P),
        "f1_macro":     f1_score(T, P, average='macro',    zero_division=0),
    }


def compute_class_weights(train_df):
    """Compute class weights for CrossEntropyLoss from training labels."""
    counts = train_df['label'].value_counts().sort_index()
    total  = len(train_df)
    weights = torch.tensor(
        [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float32
    )
    return weights


# ==========================================
# PLOTTING
# ==========================================
def save_plots(results_df, num_layers, results_dir, ablation_name, activation_type):
    layers = results_df['layer']
    title_suffix = f"{ablation_name} | {activation_type}\n(Qwen3-4B on LogiQA v3)"

    for metric, ylabel, fname in [
        ("balanced_acc", "Balanced Accuracy", "balanced_acc"),
        ("f1_macro",     "Macro F1",          "f1_macro"),
        ("acc",          "Accuracy",           "accuracy"),
    ]:
        plt.figure(figsize=(12, 5))
        plt.plot(layers, results_df[f'logreg_{metric}'], marker='o', color=COLORS[0], label='LogReg')
        plt.plot(layers, results_df[f'mlp_{metric}'],    marker='s', color=COLORS[1], label='MLP')

        # Baselines
        if metric == "balanced_acc":
            plt.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (25%)')
        elif metric == "acc":
            majority = results_df.get('majority_baseline', pd.Series([0.30]*len(results_df)))
            plt.axhline(y=majority.iloc[0], color='gray', linestyle='--',
                        alpha=0.5, label=f'Majority baseline')
        elif metric == "f1_macro":
            plt.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (25%)')

        plt.title(f"Layer-wise {ylabel} — {title_suffix}")
        plt.xlabel("Layer")
        plt.ylabel(ylabel)
        plt.ylim(0, 1.05)
        plt.xticks(range(0, num_layers, 2))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{fname}_{activation_type}.png"), dpi=300)
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
    n_samples     : int or None — for quick tests
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"ABLATION: {ablation_name.upper()}")
    print(f"Device: {device} | N_SAMPLES: {n_samples}")
    print(f"{'='*60}")

    df          = load_logiqa_v3_df(n_samples=n_samples)
    data_module = ProbeDataModule(df)

    # Compute class weights from training set
    class_weights = compute_class_weights(data_module.train_df)
    print(f"Class weights: {class_weights.tolist()}")

    results_dir = os.path.join(RESULTS_BASE, ablation_name)
    models_dir  = os.path.join(MODELS_BASE,  ablation_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, "mlp"), exist_ok=True)

    for activation_type in ("sampled", "averaged"):
        print(f"\n{'─'*50}")
        print(f"Activation type: {activation_type}")
        print(f"{'─'*50}")

        train_cache = ActivationCache(data_module.train_df, activation_type)
        test_cache  = ActivationCache(data_module.test_df,  activation_type)

        all_results = []

        for layer in range(NUM_LAYERS):
            print(f"\n── Layer {layer}/{NUM_LAYERS-1} ──")

            # Build numpy datasets for LogReg
            X_tr, y_tr = build_numpy_dataset(train_cache, layer, slice_fn)
            X_te, y_te = build_numpy_dataset(test_cache,  layer, slice_fn)

            # Build torch datasets for MLP
            train_ds = build_torch_dataset(train_cache, layer, slice_fn)
            test_ds  = build_torch_dataset(test_cache,  layer, slice_fn)

            if len(X_tr) == 0 or len(X_te) == 0:
                print("  WARNING: empty dataset, skipping.")
                continue

            row = {"layer": layer}

            # --- LogReg ---
            m, clf = train_eval_logreg(X_tr, y_tr, X_te, y_te)
            row["logreg_acc"]          = m["acc"]
            row["logreg_balanced_acc"] = m["balanced_acc"]
            row["logreg_f1_macro"]     = m["f1_macro"]
            print(f"  LogReg → Acc: {m['acc']:.4f} | Bal Acc: {m['balanced_acc']:.4f} | F1: {m['f1_macro']:.4f}")

            # --- MLP ---
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
            mlp          = MLPProbe().to(device)
            optimizer    = optim.Adam(mlp.parameters(), lr=1e-3)
            train_mlp(mlp, train_loader, optimizer, class_weights, epochs=EPOCHS)
            m = evaluate_mlp(mlp, test_loader)
            row["mlp_acc"]          = m["acc"]
            row["mlp_balanced_acc"] = m["balanced_acc"]
            row["mlp_f1_macro"]     = m["f1_macro"]
            print(f"  MLP    → Acc: {m['acc']:.4f} | Bal Acc: {m['balanced_acc']:.4f} | F1: {m['f1_macro']:.4f}")
            torch.save(mlp.state_dict(),
                       os.path.join(models_dir, "mlp", f"layer_{layer}_{activation_type}.pt"))

            all_results.append(row)

            # Incremental checkpoint
            pd.DataFrame(all_results).to_csv(
                os.path.join(results_dir, f"results_{activation_type}.csv"), index=False
            )

        # Final save + plots
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(results_dir, f"results_{activation_type}.csv"), index=False)
        save_plots(results_df, NUM_LAYERS, results_dir, ablation_name, activation_type)

        best_logreg = results_df.loc[results_df['logreg_balanced_acc'].idxmax(), 'layer']
        best_mlp    = results_df.loc[results_df['mlp_balanced_acc'].idxmax(),    'layer']
        print(f"\n  Best layer (LogReg Bal Acc): Layer {best_logreg} "
              f"({results_df['logreg_balanced_acc'].max():.4f})")
        print(f"  Best layer (MLP Bal Acc):    Layer {best_mlp} "
              f"({results_df['mlp_balanced_acc'].max():.4f})")

    print(f"\n{'='*60}")
    print(f"ABLATION COMPLETE: {ablation_name}")
    print(f"Results → {results_dir}/")
    print(f"{'='*60}")