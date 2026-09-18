import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ==========================================
# CONFIG
# ==========================================
CSV_PATH     = "/playpen-ssd/yuvraj/llm-probing-project/activations/gsm8k/master_log.csv"
DATA_DIR     = "/playpen-ssd/yuvraj/llm-probing-project/activations/gsm8k"
SAVE_DIR     = "/playpen-ssd/yuvraj/llm-probing-project/saved_models/gsm8k"
BATCH_SIZE   = 64
EPOCHS       = 5
TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_SAMPLES    = None  # Set to an int like 10 for smoke testing, None for full run

os.makedirs(SAVE_DIR, exist_ok=True)
for subdir in ["logreg", "mlp_cls", "linreg", "mlp_reg", "xgb_clf", "xgb_reg"]:
    os.makedirs(os.path.join(SAVE_DIR, subdir), exist_ok=True)


# ==========================================
# DATA MODULE
# ==========================================
class ProbeDataModule:
    def __init__(self, csv_path, data_dir, test_size=0.2, random_state=42, n_samples=None):
        self.data_dir = data_dir

        df = pd.read_csv(csv_path)
        df['reg_target'] = pd.to_numeric(df['extracted_answer'], errors='coerce')

        initial_len = len(df)
        df = df.dropna(subset=['reg_target']).reset_index(drop=True)
        print(f"Dropped {initial_len - len(df)} samples that failed numeric casting.")

        # Only keep examples where \boxed{} was found
        df = df[df['boxed_token_pos'] >= 0].reset_index(drop=True)
        print(f"Kept {len(df)} examples with valid \\boxed{{}} position.")

        if n_samples is not None:
            df = df.head(n_samples)
            print(f"[SMOKE TEST] Using first {n_samples} samples only.")

        self.df          = df
        self.max_seq_len = df['sequence_length'].max()

        self.train_df, self.test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        print(f"Train sequences: {len(self.train_df)} | Test sequences: {len(self.test_df)}")

        # Compute class imbalance ratio for weighted loss
        n_pos = (self.train_df['is_correct'] == 1).sum()
        n_neg = (self.train_df['is_correct'] == 0).sum()
        self.pos_weight = n_pos / n_neg
        print(f"Class balance — correct: {n_pos} | incorrect: {n_neg} | pos_weight: {self.pos_weight:.2f}")

    def get_single_step_dataloaders(self, target_layer, batch_size=32):
        train_dataset = SingleStepDataset(self.train_df, self.data_dir, target_layer)
        test_dataset  = SingleStepDataset(self.test_df,  self.data_dir, target_layer)
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def get_sequential_dataloaders(self, target_layer, batch_size=32):
        train_dataset = SequentialDataset(self.train_df, self.data_dir, target_layer, self.max_seq_len)
        test_dataset  = SequentialDataset(self.test_df,  self.data_dir, target_layer, self.max_seq_len)
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
        return train_loader, test_loader


# ==========================================
# RAM CACHE
# Truncates each sequence at boxed_token_pos
# so probes only train on real reasoning tokens.
# boxed_token_pos is the generation token index,
# +1 gives the activation index (offset by prompt).
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
            act = self.cache[row['tensor_file']]

            # Truncate at boxed position — +1 for prompt offset
            # This excludes \boxed{} and everything after it
            boxed_act_pos = int(row['boxed_token_pos']) + 1
            # Make sure we have at least 1 token
            boxed_act_pos = max(1, min(boxed_act_pos, int(row['sequence_length'])))

            xs.append(act[layer, :boxed_act_pos, :])
            ycs.append(torch.full((boxed_act_pos,), float(row['is_correct']), dtype=torch.float32))
            yrs.append(torch.full((boxed_act_pos,), float(row['reg_target']), dtype=torch.float32))
        return TensorDataset(torch.cat(xs), torch.cat(ycs), torch.cat(yrs))


# ==========================================
# TYPE 1: Single-Step Dataset
# ==========================================
class SingleStepDataset(Dataset):
    def __init__(self, df, data_dir, target_layer):
        self.data_dir     = data_dir
        self.target_layer = target_layer
        self.samples      = []

        for _, row in df.iterrows():
            filepath      = os.path.join(self.data_dir, row['tensor_file'])
            boxed_act_pos = max(1, min(int(row['boxed_token_pos']) + 1, int(row['sequence_length'])))
            for step_idx in range(boxed_act_pos):
                self.samples.append({
                    'filepath': filepath,
                    'step_idx': step_idx,
                    'y_class':  torch.tensor(row['is_correct'], dtype=torch.float32),
                    'y_reg':    torch.tensor(row['reg_target'], dtype=torch.float32),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample      = self.samples[idx]
        activations = torch.load(sample['filepath'], map_location='cpu')
        x           = activations[self.target_layer, sample['step_idx'], :].float()
        return x, sample['y_class'], sample['y_reg']


# ==========================================
# TYPE 2: Sequential (Padded/Masked) Dataset
# ==========================================
class SequentialDataset(Dataset):
    def __init__(self, df, data_dir, target_layer, max_seq_len):
        self.data_dir     = data_dir
        self.target_layer = target_layer
        self.max_seq_len  = max_seq_len
        self.df           = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row           = self.df.iloc[idx]
        activations   = torch.load(os.path.join(self.data_dir, row['tensor_file']), map_location='cpu')
        boxed_act_pos = max(1, min(int(row['boxed_token_pos']) + 1, int(row['sequence_length'])))
        x_seq         = activations[self.target_layer, :boxed_act_pos, :]
        seq_len, hidden_dim = x_seq.shape

        padded_x = torch.zeros((self.max_seq_len, hidden_dim))
        padded_x[:seq_len, :] = x_seq

        pad_mask    = torch.zeros(self.max_seq_len)
        pad_mask[:seq_len] = 1
        causal_mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len))

        y_class = torch.tensor(row['is_correct'], dtype=torch.float32)
        y_reg   = torch.tensor(row['reg_target'], dtype=torch.float32)
        return padded_x, pad_mask, causal_mask, y_class, y_reg


# ==========================================
# PROBE MODELS
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
# TRAINING
# ==========================================
def train_pytorch_probe(model, train_loader, criterion, optimizer,
                        task_type="classification", epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y_class, batch_y_reg in train_loader:
            optimizer.zero_grad()
            targets = batch_y_class if task_type == "classification" else batch_y_reg
            device  = next(model.parameters()).device
            batch_x, targets = batch_x.to(device, dtype=torch.float32), targets.to(device)
            outputs = model(batch_x)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
    return model


def dataloader_to_numpy(dataloader):
    x_all, y_class_all, y_reg_all = [], [], []
    for batch_x, batch_y_class, batch_y_reg in dataloader:
        x_all.append(batch_x.float().numpy())
        y_class_all.append(batch_y_class.numpy())
        y_reg_all.append(batch_y_reg.numpy())
    return np.vstack(x_all), np.concatenate(y_class_all), np.concatenate(y_reg_all)


def train_xgboost_probes(X_train, Y_class_train, Y_reg_train, pos_weight):
    print("  Training XGBoost Classifier...")
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        objective='binary:logistic', verbosity=0,
        scale_pos_weight=pos_weight,
    )
    xgb_classifier.fit(X_train, Y_class_train)

    print("  Training XGBoost Regressor...")
    xgb_regressor = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        objective='reg:squarederror', verbosity=0,
    )
    xgb_regressor.fit(X_train, Y_reg_train)
    return xgb_classifier, xgb_regressor


# ==========================================
# MAIN
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_module = ProbeDataModule(
    csv_path=CSV_PATH,
    data_dir=DATA_DIR,
    n_samples=N_SAMPLES,
)

pos_weight    = torch.tensor([data_module.pos_weight]).to(device)
criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion_mse = nn.MSELoss()

sample_act = torch.load(
    os.path.join(DATA_DIR, data_module.df.iloc[0]['tensor_file']), map_location='cpu'
)
num_layers = sample_act.shape[0]
hidden_dim = sample_act.shape[2]
del sample_act
print(f"Detected {num_layers} layers | hidden_dim: {hidden_dim}")

train_cache = ActivationCache(data_module.train_df, DATA_DIR)
test_cache  = ActivationCache(data_module.test_df,  DATA_DIR)

# ==========================================
# LOOP OVER ALL LAYERS
# ==========================================
for layer in range(num_layers):
    print(f"\n{'='*50}")
    print(f"LAYER {layer} / {num_layers - 1}")
    print(f"{'='*50}")

    train_loader = DataLoader(train_cache.get_dataset(layer), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_cache.get_dataset(layer),  batch_size=BATCH_SIZE, shuffle=False)

    # --- Logistic Regression (Classification) ---
    print("\n--- Training Logistic Regression (Classification) ---")
    log_reg      = LinearProbe(hidden_dim).to(device)
    optimizer_lr = optim.Adam(log_reg.parameters(), lr=1e-3)
    train_pytorch_probe(log_reg, train_loader, criterion_bce, optimizer_lr, task_type="classification")
    torch.save(log_reg.state_dict(), f"{SAVE_DIR}/logreg/layer_{layer}.pt")

    # --- MLP (Classification) ---
    print("\n--- Training MLP (Classification) ---")
    mlp_class       = SimpleMLPProbe(hidden_dim).to(device)
    optimizer_mlp_c = optim.Adam(mlp_class.parameters(), lr=1e-3)
    train_pytorch_probe(mlp_class, train_loader, criterion_bce, optimizer_mlp_c, task_type="classification")
    torch.save(mlp_class.state_dict(), f"{SAVE_DIR}/mlp_cls/layer_{layer}.pt")

    # --- Linear Regression ---
    print("\n--- Training Linear Regression (Regression) ---")
    lin_reg       = LinearProbe(hidden_dim).to(device)
    optimizer_lin = optim.Adam(lin_reg.parameters(), lr=1e-3)
    train_pytorch_probe(lin_reg, train_loader, criterion_mse, optimizer_lin, task_type="regression")
    torch.save(lin_reg.state_dict(), f"{SAVE_DIR}/linreg/layer_{layer}.pt")

    # --- MLP (Regression) ---
    print("\n--- Training MLP (Regression) ---")
    mlp_reg         = SimpleMLPProbe(hidden_dim).to(device)
    optimizer_mlp_r = optim.Adam(mlp_reg.parameters(), lr=1e-3)
    train_pytorch_probe(mlp_reg, train_loader, criterion_mse, optimizer_mlp_r, task_type="regression")
    torch.save(mlp_reg.state_dict(), f"{SAVE_DIR}/mlp_reg/layer_{layer}.pt")

    # --- XGBoost ---
    print("\n--- Extracting NumPy arrays for XGBoost ---")
    X_train_np, Y_class_train_np, Y_reg_train_np = dataloader_to_numpy(train_loader)
    X_test_np,  Y_class_test_np,  Y_reg_test_np  = dataloader_to_numpy(test_loader)
    xgb_clf, xgb_reg = train_xgboost_probes(X_train_np, Y_class_train_np, Y_reg_train_np,
                                             pos_weight=data_module.pos_weight)
    joblib.dump(xgb_clf, f"{SAVE_DIR}/xgb_clf/layer_{layer}.pkl")
    joblib.dump(xgb_reg, f"{SAVE_DIR}/xgb_reg/layer_{layer}.pkl")
    print("XGBoost training complete.")