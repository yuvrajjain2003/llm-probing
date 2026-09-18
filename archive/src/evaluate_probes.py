import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Colorblind-friendly palette (Wong 2011)
COLORS = ["#E69F00", "#56B4E9", "#009E73"]

# ==========================================
# CONFIG — must match training script
# ==========================================
CSV_PATH     = "/playpen-ssd/yuvraj/llm-probing-project/activations/gsm8k/master_log.csv"
DATA_DIR     = "/playpen-ssd/yuvraj/llm-probing-project/activations/gsm8k"
SAVE_DIR     = "/playpen-ssd/yuvraj/llm-probing-project/saved_models/gsm8k"
RESULTS_DIR  = "/playpen-ssd/yuvraj/llm-probing-project/results/gsm8k"
BATCH_SIZE   = 64
TEST_SIZE    = 0.2
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# PROBE MODELS — must match training script
# ==========================================
class LinearProbe(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class SimpleMLPProbe(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ==========================================
# RAM CACHE — same as training script
# ==========================================
class ActivationCache:
    def __init__(self, df, data_dir):
        print(f"Loading {len(df)} files into RAM (bfloat16)...")
        self.cache = {}
        self.df    = df.reset_index(drop=True)
        for _, row in df.iterrows():
            fpath = os.path.join(data_dir, row['tensor_file'])
            self.cache[row['tensor_file']] = torch.load(fpath, map_location='cpu')
        print(f"Cache built. {len(self.cache)} files loaded.")

    def get_dataset(self, layer):
        xs, ycs, yrs = [], [], []
        for _, row in self.df.iterrows():
            act     = self.cache[row['tensor_file']]
            seq_len = int(row['sequence_length'])
            xs.append(act[layer, :seq_len, :])
            ycs.append(torch.full((seq_len,), float(row['is_correct']), dtype=torch.float32))
            yrs.append(torch.full((seq_len,), float(row['reg_target']), dtype=torch.float32))
        return TensorDataset(torch.cat(xs), torch.cat(ycs), torch.cat(yrs))


# ==========================================
# EVALUATION HELPERS
# ==========================================
@torch.no_grad()
def evaluate_pytorch_probe(model, test_loader, task_type="classification"):
    model.eval()
    device = next(model.parameters()).device
    preds_all, targets_all = [], []

    for batch_x, batch_yc, batch_yr in test_loader:
        targets = batch_yc if task_type == "classification" else batch_yr
        batch_x = batch_x.to(device, dtype=torch.float32)
        outputs = model(batch_x).cpu()
        if task_type == "classification":
            preds_all.append((torch.sigmoid(outputs) >= 0.5).float().numpy())
        else:
            preds_all.append(outputs.numpy())
        targets_all.append(targets.numpy())

    P = np.concatenate(preds_all)
    T = np.concatenate(targets_all)

    if task_type == "classification":
        return accuracy_score(T, P), f1_score(T, P, zero_division=0)
    else:
        ss_res = np.sum((T - P) ** 2)
        ss_tot = np.sum((T - T.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return float(np.mean((T - P) ** 2)), r2  # MSE, R2


def dataloader_to_numpy(dataloader):
    x_all, y_class_all, y_reg_all = [], [], []
    for batch_x, batch_yc, batch_yr in dataloader:
        x_all.append(batch_x.float().numpy())
        y_class_all.append(batch_yc.numpy())
        y_reg_all.append(batch_yr.numpy())
    return np.vstack(x_all), np.concatenate(y_class_all), np.concatenate(y_reg_all)


# ==========================================
# MAIN
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reconstruct the same train/test split as training
df = pd.read_csv(CSV_PATH)
df['reg_target'] = pd.to_numeric(df['extracted_answer'], errors='coerce')
df = df.dropna(subset=['reg_target']).reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# Detect num_layers and hidden_dim
sample_act = torch.load(
    os.path.join(DATA_DIR, df.iloc[0]['tensor_file']), map_location='cpu'
)
num_layers = sample_act.shape[0]
hidden_dim = sample_act.shape[2]
del sample_act
print(f"Detected {num_layers} layers | hidden_dim: {hidden_dim}")

# Build test cache only — we don't need train data for evaluation
test_cache = ActivationCache(test_df, DATA_DIR)

# ==========================================
# LOOP OVER ALL LAYERS
# ==========================================
results = []

for layer in range(num_layers):
    print(f"\n{'='*50}")
    print(f"LAYER {layer} / {num_layers - 1}")
    print(f"{'='*50}")

    test_loader = DataLoader(test_cache.get_dataset(layer), batch_size=BATCH_SIZE, shuffle=False)
    X_test_np, Y_cls_test, Y_reg_test = dataloader_to_numpy(test_loader)

    row = {"layer": layer}

    # --- Logistic Regression ---
    print("  Evaluating LogReg...")
    model = LinearProbe(hidden_dim).to(device)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/logreg/layer_{layer}.pt", map_location=device))
    acc, f1 = evaluate_pytorch_probe(model, test_loader, task_type="classification")
    row["logreg_acc"] = acc
    row["logreg_f1"]  = f1
    print(f"  LogReg     → Acc: {acc:.4f} | F1: {f1:.4f}")

    # --- MLP Classification ---
    print("  Evaluating MLP Cls...")
    model = SimpleMLPProbe(hidden_dim).to(device)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/mlp_cls/layer_{layer}.pt", map_location=device))
    acc, f1 = evaluate_pytorch_probe(model, test_loader, task_type="classification")
    row["mlp_cls_acc"] = acc
    row["mlp_cls_f1"]  = f1
    print(f"  MLP Cls    → Acc: {acc:.4f} | F1: {f1:.4f}")

    # --- Linear Regression ---
    print("  Evaluating LinReg...")
    model = LinearProbe(hidden_dim).to(device)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/linreg/layer_{layer}.pt", map_location=device))
    mse, r2 = evaluate_pytorch_probe(model, test_loader, task_type="regression")
    row["linreg_mse"] = mse
    row["linreg_r2"]  = r2
    print(f"  LinReg     → MSE: {mse:.4f} | R²: {r2:.4f}")

    # --- MLP Regression ---
    print("  Evaluating MLP Reg...")
    model = SimpleMLPProbe(hidden_dim).to(device)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/mlp_reg/layer_{layer}.pt", map_location=device))
    mse, r2 = evaluate_pytorch_probe(model, test_loader, task_type="regression")
    row["mlp_reg_mse"] = mse
    row["mlp_reg_r2"]  = r2
    print(f"  MLP Reg    → MSE: {mse:.4f} | R²: {r2:.4f}")

    # --- XGBoost Classifier ---
    print("  Evaluating XGB Clf...")
    xgb_clf = joblib.load(f"{SAVE_DIR}/xgb_clf/layer_{layer}.pkl")
    xgb_preds = xgb_clf.predict(X_test_np)
    row["xgb_clf_acc"] = accuracy_score(Y_cls_test, xgb_preds)
    row["xgb_clf_f1"]  = f1_score(Y_cls_test, xgb_preds, zero_division=0)
    print(f"  XGB Clf    → Acc: {row['xgb_clf_acc']:.4f} | F1: {row['xgb_clf_f1']:.4f}")

    # --- XGBoost Regressor ---
    print("  Evaluating XGB Reg...")
    xgb_reg = joblib.load(f"{SAVE_DIR}/xgb_reg/layer_{layer}.pkl")
    xgb_reg_preds = xgb_reg.predict(X_test_np)
    ss_res = np.sum((Y_reg_test - xgb_reg_preds) ** 2)
    ss_tot = np.sum((Y_reg_test - Y_reg_test.mean()) ** 2)
    row["xgb_reg_mse"] = float(np.mean((Y_reg_test - xgb_reg_preds) ** 2))
    row["xgb_reg_r2"]  = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    print(f"  XGB Reg    → MSE: {row['xgb_reg_mse']:.4f} | R²: {row['xgb_reg_r2']:.4f}")

    results.append(row)

# ==========================================
# SAVE RESULTS CSV
# ==========================================
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, "probe_results_by_layer.csv"), index=False)
print(f"\nResults saved to {RESULTS_DIR}/probe_results_by_layer.csv")

# ==========================================
# PLOTS
# ==========================================
layers = results_df['layer']

# --- Classification Accuracy ---
plt.figure(figsize=(12, 5))
plt.plot(layers, results_df['logreg_acc'],   marker='o', label='LogReg', color=COLORS[0])
plt.plot(layers, results_df['mlp_cls_acc'],  marker='s', label='MLP', color=COLORS[1])
plt.plot(layers, results_df['xgb_clf_acc'],  marker='^', label='XGBoost', color=COLORS[2])
plt.title("Layer-wise Classification Accuracy\n(Qwen3-4B on GSM8K)")
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.xticks(range(0, num_layers, 2))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "classification_accuracy_by_layer.png"), dpi=300)
plt.close()

# --- Classification F1 ---
plt.figure(figsize=(12, 5))
plt.plot(layers, results_df['logreg_f1'],   marker='o', label='LogReg', color=COLORS[0])
plt.plot(layers, results_df['mlp_cls_f1'],  marker='s', label='MLP', color=COLORS[1])
plt.plot(layers, results_df['xgb_clf_f1'],  marker='^', label='XGBoost', color=COLORS[2])
plt.title("Layer-wise Classification F1 Score\n(Qwen3-4B on GSM8K)")
plt.xlabel("Layer")
plt.ylabel("F1 Score")
plt.ylim(0, 1.05)
plt.xticks(range(0, num_layers, 2))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "classification_f1_by_layer.png"), dpi=300)
plt.close()

# --- Regression R² ---
plt.figure(figsize=(12, 5))
plt.plot(layers, results_df['linreg_r2'],   marker='o', label='LinReg', color=COLORS[0])
plt.plot(layers, results_df['mlp_reg_r2'],  marker='s', label='MLP', color=COLORS[1])
plt.plot(layers, results_df['xgb_reg_r2'],  marker='^', label='XGBoost', color=COLORS[2])
plt.title("Layer-wise Regression R²\n(Qwen3-4B on GSM8K)")
plt.xlabel("Layer")
plt.ylabel("R²")
plt.xticks(range(0, num_layers, 2))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "regression_r2_by_layer.png"), dpi=300)
plt.close()

print(f"\nPlots saved to {RESULTS_DIR}/")
print("\nBest layers:")
print(f"  Classification Acc (LogReg): Layer {results_df.loc[results_df['logreg_acc'].idxmax(), 'layer']}")
print(f"  Classification Acc (MLP):    Layer {results_df.loc[results_df['mlp_cls_acc'].idxmax(), 'layer']}")
print(f"  Classification Acc (XGB):    Layer {results_df.loc[results_df['xgb_clf_acc'].idxmax(), 'layer']}")