"""
diagnose.py
───────────
Full diagnostic — traces one example through the entire pipeline
to find exactly where MSE is computed incorrectly.
"""
import torch, numpy as np, glob, pandas as pd, os

BASE_DIR     = "/playpen-ssd/yuvraj/llm-probing-project"
SAMPLED_DIR  = os.path.join(BASE_DIR, "activations/gsm8k_v3/sampled")
AVERAGED_DIR = os.path.join(BASE_DIR, "activations/gsm8k_v3/averaged")

# ── 1. Load data ──────────────────────────────────────────────────────────────
dfs = []
for f in glob.glob(os.path.join(BASE_DIR, "activations/gsm8k_v3/master_log_worker*.csv")):
    dfs.append(pd.read_csv(f))
df = pd.concat(dfs).drop_duplicates('example_id')
df = df[df['resumed'] == False]
df['reg_target'] = pd.to_numeric(df['extracted_answer'], errors='coerce')
df = df.dropna(subset=['reg_target'])
df = df[df['model_generation'].str.contains('boxed', na=False)]

def pt_valid(row):
    sp = os.path.join(SAMPLED_DIR, row['sampled_tensor_file'])
    return os.path.exists(sp) and os.path.getsize(sp) > 1024
df = df[df.apply(pt_valid, axis=1)].reset_index(drop=True)
print(f"Using {len(df)} examples")

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

target_mean = train_df['reg_target'].mean()
target_std  = train_df['reg_target'].std()
print(f"target_mean={target_mean:.2f} | target_std={target_std:.2f}")

# ── 2. Build X, y for layer 18, last position ─────────────────────────────────
def build_Xy(split_df, layer, pos_idx):
    xs, ys = [], []
    for _, row in split_df.iterrows():
        path = os.path.join(SAMPLED_DIR, row['sampled_tensor_file'])
        act  = torch.load(path, map_location='cpu')
        x    = act[layer, pos_idx, :].float().numpy()
        y    = (float(row['reg_target']) - target_mean) / (target_std + 1e-8)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_tr, y_tr = build_Xy(train_df, layer=18, pos_idx=99)
X_te, y_te = build_Xy(test_df,  layer=18, pos_idx=99)

print(f"\nX_tr shape: {X_tr.shape} | y_tr shape: {y_tr.shape}")
print(f"y_tr mean: {y_tr.mean():.4f} | y_tr std: {y_tr.std():.4f}")
print(f"y_te mean: {y_te.mean():.4f} | y_te std: {y_te.std():.4f}")
print(f"y_tr sample: {y_tr[:5]}")
print(f"y_te sample: {y_te[:5]}")

# ── 3. Fit Ridge ──────────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ridge = Ridge(alpha=1.0)
ridge.fit(X_tr, y_tr)
preds = ridge.predict(X_te)

print(f"\nRidge preds sample: {preds[:5]}")
print(f"y_te sample:        {y_te[:5]}")
print(f"MSE (std):          {np.mean((y_te - preds)**2):.4f}")
print(f"R²:                 {r2_score(y_te, preds):.4f}")

# ── 4. Check MLP predictions ──────────────────────────────────────────────────
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLPProbe(nn.Module):
    def __init__(self, hidden_dim=2560):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
X_te_t = torch.tensor(X_te, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.float32)

print(f"\ny_tr_t sample: {y_tr_t[:5]}")
print(f"y_te_t sample: {y_te_t[:5]}")

train_ds = TensorDataset(X_tr_t, y_tr_t)
test_ds  = TensorDataset(X_te_t, y_te_t)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False)

mlp = MLPProbe().to(device)
opt = optim.Adam(mlp.parameters(), lr=1e-3)
crit = nn.MSELoss()

for epoch in range(5):
    mlp.train()
    for bx, by in train_loader:
        opt.zero_grad()
        loss = crit(mlp(bx.to(device)), by.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        opt.step()

mlp.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for bx, by in test_loader:
        all_preds.append(mlp(bx.to(device)).cpu().numpy())
        all_targets.append(by.numpy())

P = np.concatenate(all_preds)
T = np.concatenate(all_targets)
print(f"\nMLP preds sample:   {P[:5]}")
print(f"y_te sample:        {T[:5]}")
print(f"MLP MSE (std):      {np.mean((T - P)**2):.4f}")
print(f"MLP R²:             {r2_score(T, P):.4f}")

# ── 5. Sanity check — what does mse_orig look like? ───────────────────────────
P_orig = P * target_std + target_mean
T_orig = T * target_std + target_mean
print(f"\nMLP MSE (orig):     {np.mean((T_orig - P_orig)**2):.1f}")
print(f"P_orig sample:      {P_orig[:5]}")
print(f"T_orig sample:      {T_orig[:5]}")