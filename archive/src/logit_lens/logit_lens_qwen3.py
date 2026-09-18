"""
logit_lens_logiqa_v2.py
────────────────────────
Proper logit lens analysis on LogiQA v2 activations.

Key insight from data exploration:
  - Tensors are shape [37, seq_len, 2560] — every token position saved
  - Answer letter is ALWAYS at position seq_len - 8
  - Therefore logit lens target position is seq_len - 9 (the '{' token
    just before the answer letter, where the model commits to its answer)

Ablations (4 positions, compared on the same plot):
  1. pre_answer   : seq_len - 9  ← the critical one
  2. last_think   : think_end_pos (last token of <think> phase)
  3. mid_think    : think_end_pos // 2 (halfway through reasoning)
  4. first_think  : fixed offset ~5 tokens into generation

Research question:
  "Does the model know the answer before it finishes reasoning?"
  If pre_answer >> last_think >> mid_think >> first_think in ABCD accuracy,
  reasoning is doing real work. If first_think already shows signal,
  the answer may be encoded before reasoning begins.

Logit lens formula (Belrose et al. 2023):
  LogitLens(h_ℓ) = lm_head( final_RMSNorm(h_ℓ) )
"""

import os, glob, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# CONFIG
# ==========================================
BASE_DIR     = "/playpen-ssd/yuvraj/llm-probing-project"
ACT_DIR      = os.path.join(BASE_DIR, "activations/logiqa_v2")
RESULTS_DIR  = os.path.join(BASE_DIR, "results/logit_lens/logiqa_v2")
MODEL_NAME   = "Qwen/Qwen3-4B"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_LAYERS   = 37
HIDDEN_DIM   = 2560
ANSWER_OFFSET = 9   # answer letter is at seq_len-8, so we look at seq_len-9

LABEL_MAP    = {"A": 0, "B": 1, "C": 2, "D": 3}
COLORS       = {
    "pre_answer":  "#E69F00",
    "last_think":  "#56B4E9",
    "mid_think":   "#009E73",
    "first_think": "#CC79A7",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==========================================
# STEP 1: Load model norm + lm_head only
# ==========================================
print(f"\nLoading norm + lm_head from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
final_norm = model.model.norm.to(device).eval()
lm_head    = model.lm_head.to(device).eval()
print(f"lm_head weight shape: {lm_head.weight.shape}")  # [151936, 2560]
del model
torch.cuda.empty_cache()
print("Model freed — only norm + lm_head kept\n")

# Resolve A/B/C/D token IDs
answer_token_ids = {}
for letter in ["A", "B", "C", "D"]:
    ids = tokenizer.encode(letter, add_special_tokens=False)
    assert len(ids) == 1, f"Expected single token for '{letter}', got {ids}"
    answer_token_ids[letter] = ids[0]
print(f"Answer token IDs: {answer_token_ids}")
abcd_ids = torch.tensor(list(answer_token_ids.values()), dtype=torch.long)  # [4]

# ==========================================
# STEP 2: Load CSV
# ==========================================
dfs = [pd.read_csv(f) for f in glob.glob(os.path.join(ACT_DIR, "master_log_worker*.csv"))]
df  = pd.concat(dfs, ignore_index=True).drop_duplicates("example_id")
if "resumed" in df.columns:
    df = df[df["resumed"] == False].reset_index(drop=True)

# Filter: valid ground truth + boxed answer + valid tensor
df = df[df["ground_truth"].isin(["A", "B", "C", "D"])]
df = df[df["model_generation"].str.contains("boxed", na=False)]
df = df[df["think_end_pos"].notna()]   # need think_end_pos for think ablations

def tensor_valid(row):
    p = os.path.join(ACT_DIR, row["tensor_file"])
    return os.path.exists(p) and os.path.getsize(p) > 1024
df = df[df.apply(tensor_valid, axis=1)].reset_index(drop=True)

df["label"]            = df["ground_truth"].map(LABEL_MAP)
df["correct_token_id"] = df["ground_truth"].map(answer_token_ids)

print(f"\nValid examples: {len(df)}")
print(f"Ground truth distribution:\n{df['ground_truth'].value_counts().sort_index()}\n")

# ==========================================
# STEP 3: Position functions
# ==========================================
def get_positions(row):
    seq_len      = int(row["sequence_length"])
    think_end    = int(row["think_end_pos"])

    # Clamp all positions to valid range [0, seq_len-1]
    pre_answer  = max(0, seq_len - ANSWER_OFFSET)
    last_think  = min(think_end, seq_len - 1)
    mid_think   = max(1, think_end // 2)
    first_think = min(5, think_end - 1)   # ~5 tokens into generation

    return {
        "pre_answer":  pre_answer,
        "last_think":  last_think,
        "mid_think":   mid_think,
        "first_think": first_think,
    }

ABLATIONS = ["pre_answer", "last_think", "mid_think", "first_think"]

# ==========================================
# STEP 4: Logit lens
# ==========================================
@torch.no_grad()
def logit_lens_batch(hidden: torch.Tensor) -> torch.Tensor:
    """hidden: [N, 2560] bfloat16 → logits: [N, vocab] float32"""
    return lm_head(final_norm(hidden.to(device))).float()

# ==========================================
# STEP 5: Main loop — layer by layer
# ==========================================
all_records = []

for layer in range(NUM_LAYERS):
    print(f"── Layer {layer}/{NUM_LAYERS-1} ──", flush=True)

    # Collect hidden states and labels per ablation
    hiddens = {abl: [] for abl in ABLATIONS}
    correct_ids_list = []
    labels_list      = []

    for _, row in df.iterrows():
        fpath = os.path.join(ACT_DIR, row["tensor_file"])
        act   = torch.load(fpath, map_location="cpu")  # [37, seq_len, 2560]
        positions = get_positions(row)

        for abl in ABLATIONS:
            pos = positions[abl]
            hiddens[abl].append(act[layer, pos, :])    # [2560]

        correct_ids_list.append(int(row["correct_token_id"]))
        labels_list.append(int(row["label"]))

    correct_ids = torch.tensor(correct_ids_list, dtype=torch.long)
    true_labels = torch.tensor(labels_list, dtype=torch.long)

    rec = {"layer": layer}

    for abl in ABLATIONS:
        hs     = torch.stack(hiddens[abl])              # [N, 2560]

        # Run logit lens in batches
        BATCH  = 64
        logits_list = []
        for i in range(0, len(hs), BATCH):
            logits_list.append(logit_lens_batch(hs[i:i+BATCH]).cpu())
        logits = torch.cat(logits_list, dim=0)          # [N, vocab]
        probs  = torch.softmax(logits, dim=-1)

        # Top-1 accuracy (any vocab token)
        top1 = logits.argmax(dim=-1)
        rec[f"{abl}_top1_acc"] = (top1 == correct_ids).float().mean().item()

        # ABCD-restricted accuracy
        abcd_logits   = logits[:, abcd_ids]             # [N, 4]
        abcd_pred_idx = abcd_logits.argmax(dim=-1)      # [N]
        rec[f"{abl}_abcd_acc"] = (abcd_pred_idx == true_labels).float().mean().item()

        # Probability of correct answer token
        correct_probs = probs[torch.arange(len(probs)), correct_ids]
        rec[f"{abl}_correct_prob"] = correct_probs.mean().item()

        # Rank of correct answer token
        correct_logit_vals = logits[torch.arange(len(logits)), correct_ids].unsqueeze(1)
        ranks = (logits > correct_logit_vals).sum(dim=-1).float() + 1
        rec[f"{abl}_rank"] = ranks.mean().item()

        # Entropy
        entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
        rec[f"{abl}_entropy"] = entropy.mean().item()

        print(f"  [{abl}] abcd_acc={rec[f'{abl}_abcd_acc']:.3f} | "
              f"correct_prob={rec[f'{abl}_correct_prob']:.4f} | "
              f"rank={rec[f'{abl}_rank']:.0f}", flush=True)

    all_records.append(rec)

    # Incremental save
    pd.DataFrame(all_records).to_csv(
        os.path.join(RESULTS_DIR, "logit_lens_v2.csv"), index=False
    )

results_df = pd.DataFrame(all_records)
results_df.to_csv(os.path.join(RESULTS_DIR, "logit_lens_v2.csv"), index=False)
print(f"\nResults saved → {RESULTS_DIR}/logit_lens_v2.csv")

# ==========================================
# STEP 6: Plots — all ablations on one figure
# ==========================================
layers = results_df["layer"]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Logit Lens — All Positions Compared\n(Qwen3-4B on LogiQA v2)", fontsize=14)

# ABCD accuracy
ax = axes[0, 0]
for abl in ABLATIONS:
    ax.plot(layers, results_df[f"{abl}_abcd_acc"], marker='o', markersize=3,
            color=COLORS[abl], label=abl)
ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (25%)')
ax.set_title("ABCD-restricted Accuracy")
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Correct token probability
ax = axes[0, 1]
for abl in ABLATIONS:
    ax.plot(layers, results_df[f"{abl}_correct_prob"], marker='o', markersize=3,
            color=COLORS[abl], label=abl)
ax.set_title("Mean Prob of Correct Answer Token")
ax.set_ylabel("Probability"); ax.set_ylim(0, 1)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Rank (log scale for readability)
ax = axes[1, 0]
for abl in ABLATIONS:
    ax.plot(layers, results_df[f"{abl}_rank"], marker='o', markersize=3,
            color=COLORS[abl], label=abl)
ax.set_title("Mean Rank of Correct Answer Token (lower=better)")
ax.set_ylabel("Rank"); ax.set_yscale("log")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Entropy
ax = axes[1, 1]
for abl in ABLATIONS:
    ax.plot(layers, results_df[f"{abl}_entropy"], marker='o', markersize=3,
            color=COLORS[abl], label=abl)
ax.set_title("Mean Entropy (lower = more confident)")
ax.set_ylabel("Entropy (nats)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel("Layer")
    ax.set_xticks(range(0, NUM_LAYERS, 2))

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "logit_lens_v2_all_positions.png"), dpi=300)
plt.close()
print(f"Plot saved → {RESULTS_DIR}/logit_lens_v2_all_positions.png")