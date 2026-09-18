"""
logit_lens.py
─────────────
True logit lens: for each layer, apply the unembedding matrix (lm_head)
directly to the residual stream at the token position JUST BEFORE \boxed{}.

This position has attended to the full reasoning chain, making it a genuine
prediction — the model hasn't seen the answer yet. We ask: at each layer,
does the model already "know" the answer token before writing it?
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================
CSV_PATH     = "/playpen-ssd/yuvraj/llm-probing-project/activations/gsm8k/master_log.csv"
DATA_DIR     = "/playpen-ssd/yuvraj/llm-probing-project/activations/gsm8k"
MODEL_PATH   = "/playpen-ssd/shared/pretrained_models/models--Qwen--Qwen3-4B"
RESULTS_DIR  = "/playpen-ssd/yuvraj/llm-probing-project/results/gsm8k/logit_lens"
DEVICE       = "cpu"
TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_SAMPLES    = None  # Set to 10 for smoke test
N_HEATMAP_EXAMPLES = 30

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# LOAD MODEL
# ==========================================
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
)
model.eval()

lm_head = model.lm_head
norm    = model.model.norm
print(f"Vocab size: {lm_head.weight.shape[0]} | Hidden dim: {lm_head.weight.shape[1]}")


# ==========================================
# LOGIT LENS
# ==========================================
@torch.no_grad()
def logit_lens(hidden_state):
    """
    hidden_state: (1, hidden_dim) bfloat16
    Returns top-5 token ids and their probabilities.
    """
    hidden_state = hidden_state.to(DEVICE)
    normed       = norm(hidden_state)
    logits       = lm_head(normed)
    probs        = torch.softmax(logits, dim=-1)
    top5_probs, top5_ids = probs.topk(5, dim=-1)
    return top5_ids[0].cpu(), top5_probs[0].float().cpu()


# ==========================================
# FIND POSITION JUST BEFORE \boxed{}
# Returns:
#   act_pos  — index into saved activation tensor
#   gt_tok_id — token id of first answer digit
# ==========================================
def find_pre_boxed_position(generation, seq_len):
    """
    Tokenize the generation, find first 'boxed' token.
    Target position = boxed_pos - 1 (token just before \boxed{}).
    act_pos = target_pos + 1 (offset by 1 because activation[0] = last prompt token).
    GT token = first token inside \boxed{} (boxed_pos + 2).
    """
    tokens  = tokenizer.encode(generation, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]

    # Find first 'boxed' token
    boxed_pos = None
    for i, t in enumerate(decoded):
        if 'boxed' in t:
            boxed_pos = i
            break

    if boxed_pos is None or boxed_pos < 1:
        return None, None

    # Token just before \boxed{}
    target_pos = boxed_pos - 1
    # +1 offset because activation[0] is last prompt token
    act_pos    = target_pos + 1

    # Ground truth: first answer token inside \boxed{} (boxed_pos + 2)
    ans_pos = boxed_pos + 2
    if ans_pos >= len(tokens) or act_pos >= seq_len:
        return None, None

    gt_tok_id = tokens[ans_pos]
    return act_pos, gt_tok_id


# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv(CSV_PATH)
df['reg_target'] = pd.to_numeric(df['extracted_answer'], errors='coerce')
df = df.dropna(subset=['reg_target']).reset_index(drop=True)

# Keep only completed examples
df = df[df['model_generation'].str.contains(r'\\boxed\{', regex=True)].reset_index(drop=True)
print(f"Completed examples: {len(df)}")

if N_SAMPLES is not None:
    df = df.head(N_SAMPLES)
    print(f"[SMOKE TEST] Using first {N_SAMPLES} samples only.")

_, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
test_df = test_df.reset_index(drop=True)
print(f"Test sequences: {len(test_df)}")

# Detect num_layers (skip layer 0 — it's just raw embeddings)
sample_act = torch.load(os.path.join(DATA_DIR, df.iloc[0]['tensor_file']), map_location='cpu')
num_layers  = sample_act.shape[0]
del sample_act
print(f"Detected {num_layers} layers (will skip layer 0)")

# Pre-compute positions
print("Finding pre-boxed token positions...")
act_positions = []
gt_token_ids  = []
valid_mask    = []

for _, row in test_df.iterrows():
    act_pos, gt_tok_id = find_pre_boxed_position(
        row['model_generation'], int(row['sequence_length'])
    )
    act_positions.append(act_pos)
    gt_token_ids.append(gt_tok_id)
    valid_mask.append(act_pos is not None and gt_tok_id is not None)

n_valid = sum(valid_mask)
print(f"Valid examples: {n_valid} / {len(test_df)}")


# ==========================================
# MAIN LOOP — skip layer 0
# ==========================================
layers_to_eval = list(range(num_layers))

layer_top1_acc_correct   = np.zeros(num_layers)
layer_top1_acc_incorrect = np.zeros(num_layers)
layer_top5_acc_correct   = np.zeros(num_layers)
layer_top5_acc_incorrect = np.zeros(num_layers)

n_correct   = sum(1 for i, (_, row) in enumerate(test_df.iterrows()) if valid_mask[i] and row['is_correct'] == 1)
n_incorrect = sum(1 for i, (_, row) in enumerate(test_df.iterrows()) if valid_mask[i] and row['is_correct'] == 0)

# For heatmap
all_top1_tokens    = []   # (n_valid, n_layers) decoded token strings
all_top1_probs     = []
heatmap_is_correct = []
heatmap_gt_strs    = []

print("Running logit lens (layers 0 to {})...".format(num_layers - 1))
for i, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df), desc="Examples")):
    if not valid_mask[i]:
        continue

    act     = torch.load(os.path.join(DATA_DIR, row['tensor_file']), map_location='cpu')
    act_pos = act_positions[i]
    gt_id   = gt_token_ids[i]
    gt_str  = tokenizer.decode([gt_id]).strip()

    example_top1_tokens = []
    example_top1_probs  = []

    for layer in layers_to_eval:
        hidden = act[layer, act_pos, :].unsqueeze(0)  # (1, hidden_dim)
        top5_ids, top5_probs = logit_lens(hidden)

        top1_id   = top5_ids[0].item()
        top1_prob = top5_probs[0].item()
        top1_str  = tokenizer.decode([top1_id]).strip()

        example_top1_tokens.append(top1_str)
        example_top1_probs.append(top1_prob)

        # Top-1 accuracy
        if top1_id == gt_id:
            if row['is_correct'] == 1:
                layer_top1_acc_correct[layer]   += 1
            else:
                layer_top1_acc_incorrect[layer] += 1

        # Top-5 accuracy
        if gt_id in top5_ids.tolist():
            if row['is_correct'] == 1:
                layer_top5_acc_correct[layer]   += 1
            else:
                layer_top5_acc_incorrect[layer] += 1

    all_top1_tokens.append(example_top1_tokens)
    all_top1_probs.append(example_top1_probs)
    heatmap_is_correct.append(row['is_correct'])
    heatmap_gt_strs.append(gt_str)
    del act

# Normalize
for layer in layers_to_eval:
    layer_top1_acc_correct[layer]   /= n_correct   if n_correct   > 0 else 1
    layer_top1_acc_incorrect[layer] /= n_incorrect if n_incorrect > 0 else 1
    layer_top5_acc_correct[layer]   /= n_correct   if n_correct   > 0 else 1
    layer_top5_acc_incorrect[layer] /= n_incorrect if n_incorrect > 0 else 1


# ==========================================
# SAVE RESULTS
# ==========================================
summary_df = pd.DataFrame({
    "layer":                  layers_to_eval,
    "top1_acc_correct":       layer_top1_acc_correct[1:],
    "top1_acc_incorrect":     layer_top1_acc_incorrect[1:],
    "top5_acc_correct":       layer_top5_acc_correct[1:],
    "top5_acc_incorrect":     layer_top5_acc_incorrect[1:],
})
summary_df.to_csv(os.path.join(RESULTS_DIR, "logit_lens_summary.csv"), index=False)
print(f"Results saved to {RESULTS_DIR}/")


# ==========================================
# PLOT 1: Top-1 accuracy curve
# ==========================================
COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

# Top-1
ax1.plot(layers_to_eval, layer_top1_acc_correct[1:],   marker='o', color=COLORS[0], label=f'Correct (n={n_correct})')
ax1.plot(layers_to_eval, layer_top1_acc_incorrect[1:], marker='s', color=COLORS[1], label=f'Incorrect (n={n_incorrect})')
ax1.set_title("Logit Lens: Top-1 Accuracy\nat Token Before \\boxed{}")
ax1.set_xlabel("Layer")
ax1.set_ylabel("Fraction predicting correct answer token")
ax1.set_ylim(0, 1.05)
ax1.set_xticks(range(0, num_layers, 2))
ax1.legend()
ax1.grid(True, alpha=0.3)

# Top-5
ax2.plot(layers_to_eval, layer_top5_acc_correct[1:],   marker='o', color=COLORS[0], label=f'Correct (n={n_correct})')
ax2.plot(layers_to_eval, layer_top5_acc_incorrect[1:], marker='s', color=COLORS[1], label=f'Incorrect (n={n_incorrect})')
ax2.set_title("Logit Lens: Top-5 Accuracy\nat Token Before \\boxed{}")
ax2.set_xlabel("Layer")
ax2.set_ylabel("Fraction where answer token in top-5")
ax2.set_ylim(0, 1.05)
ax2.set_xticks(range(0, num_layers, 2))
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("Qwen3-4B on GSM8K — Does the Model Know the Answer Before Writing It?", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "logit_lens_accuracy_curve.png"), dpi=300)
plt.close()
print("Saved: logit_lens_accuracy_curve.png")


# ==========================================
# PLOT 2: Heatmap
# ==========================================
n_show     = min(N_HEATMAP_EXAMPLES, len(all_top1_tokens))
sort_order = np.argsort([-c for c in heatmap_is_correct[:n_show]])

token_matrix  = [all_top1_tokens[j] for j in sort_order]
is_cor_sorted = [heatmap_is_correct[j] for j in sort_order]
gt_sorted     = [heatmap_gt_strs[j]   for j in sort_order]

# Match matrix: 1 = top1 prediction matches GT answer token
match_matrix = np.zeros((n_show, len(layers_to_eval)))
for i in range(n_show):
    for j, layer in enumerate(layers_to_eval):
        match_matrix[i, j] = 1 if token_matrix[i][j] == gt_sorted[i] else 0

fig, ax = plt.subplots(figsize=(max(16, len(layers_to_eval) * 0.5), max(8, n_show * 0.35)))
cmap = mcolors.ListedColormap(["#D55E00", "#009E73"])
ax.imshow(match_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none')

for i in range(n_show):
    for j in range(len(layers_to_eval)):
        tok = token_matrix[i][j][:5]
        ax.text(j, i, tok, ha='center', va='center', fontsize=5,
                color='white', fontweight='bold')

ylabels = [f"{'✓' if c == 1 else '✗'} gt={gt_sorted[i]}" for i, c in enumerate(is_cor_sorted)]
ax.set_yticks(range(n_show))
ax.set_yticklabels(ylabels, fontsize=7)
ax.set_xticks(range(0, len(layers_to_eval), 2))
ax.set_xticklabels(layers_to_eval[::2])
ax.set_xlabel("Layer")
ax.set_ylabel("Example (✓=correct, ✗=incorrect)")
ax.set_title("Logit Lens: Top-1 Predicted Token at Position Before \\boxed{}\n(Green = matches GT answer token, Qwen3-4B on GSM8K)")

legend_elements = [
    Patch(facecolor="#009E73", label="Matches GT answer token"),
    Patch(facecolor="#D55E00", label="Does not match"),
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "logit_lens_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: logit_lens_heatmap.png")

print(f"\nSummary:")
print(f"  Best layer top-1 (correct):   Layer {layers_to_eval[np.argmax(layer_top1_acc_correct[1:])]}")
print(f"  Best layer top-1 (incorrect): Layer {layers_to_eval[np.argmax(layer_top1_acc_incorrect[1:])]}")
print(f"  Best layer top-5 (correct):   Layer {layers_to_eval[np.argmax(layer_top5_acc_correct[1:])]}")
print(f"  Best layer top-5 (incorrect): Layer {layers_to_eval[np.argmax(layer_top5_acc_incorrect[1:])]}")