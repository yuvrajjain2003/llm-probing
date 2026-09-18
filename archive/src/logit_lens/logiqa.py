"""
logit_lens_logiqa_v2_final.py
──────────────────────────────
Logit lens on LogiQA v2, tracking two distinct metrics:

  1. Predicts WRITTEN answer  — does the logit lens predict what the model
                                actually wrote in \boxed{}? (extracted_answer)
  2. Predicts GROUND TRUTH    — does the logit lens predict the correct answer?

For correct examples these are the same.
For incorrect examples they differ — the model wrote the wrong letter,
and the logit lens should predict that wrong letter (metric 1) even though
it's not the ground truth (metric 2).

Key findings to look for:
  - Metric 1 should be high for ALL examples (correct + incorrect) in late
    layers — the model is committing to whatever it wrote
  - Metric 2 should be high for correct examples only — the model "knew"
    the right answer
  - Gap between metric 2 (correct) and metric 2 (incorrect) is the signal

Also excludes layer 36 from clean plots — confirmed artifact ({} prediction).

Probe position: boxed_pos + 1 (the '{' token, no prompt offset)
Confirmed token structure: ... '\\' 'boxed' '{' 'X' '}' '<|im_end|>'
"""

import os, glob, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================
BASE_DIR    = "/playpen-ssd/yuvraj/llm-probing-project"
ACT_DIR     = os.path.join(BASE_DIR, "activations/logiqa_v2")
MODEL_PATH  = "/playpen-ssd/shared/pretrained_models/models--Qwen--Qwen3-4B"
RESULTS_DIR = os.path.join(BASE_DIR, "results/logit_lens/logiqa_v2_final")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
N_HEATMAP_EXAMPLES = 40
N_SAMPLES          = None
SKIP_LAST_LAYER    = True   # layer 36 is a known artifact

print(f"Device: {DEVICE}")

# ==========================================
# LOAD MODEL COMPONENTS
# ==========================================
print(f"Loading tokenizer + model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    device_map="cpu", trust_remote_code=True
)
model.eval()
lm_head    = model.lm_head.to(DEVICE).eval()
final_norm = model.model.norm.to(DEVICE).eval()
print(f"lm_head: {lm_head.weight.shape}")
del model
torch.cuda.empty_cache()
print("Model freed — only norm + lm_head kept\n")

# Resolve A/B/C/D token IDs
LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
answer_token_ids = {}
for letter in ["A", "B", "C", "D"]:
    ids = tokenizer.encode(letter, add_special_tokens=False)
    assert len(ids) == 1, f"Expected single token for '{letter}', got {ids}"
    answer_token_ids[letter] = ids[0]
print(f"Answer token IDs: {answer_token_ids}")
abcd_ids = torch.tensor(list(answer_token_ids.values()), dtype=torch.long)

# ==========================================
# LOGIT LENS
# ==========================================
@torch.no_grad()
def logit_lens(hidden_state):
    """hidden_state: (1, hidden_dim) → top5 ids, top5 probs, full logits"""
    normed = final_norm(hidden_state.to(DEVICE))
    logits = lm_head(normed).float()
    probs  = torch.softmax(logits, dim=-1)
    top5_probs, top5_ids = probs.topk(5, dim=-1)
    return top5_ids[0].cpu(), top5_probs[0].cpu(), logits[0].cpu()

# ==========================================
# FIND PROBE POSITION
# ==========================================
def find_probe_position(generation, seq_len):
    """
    Returns (probe_pos, written_tok_id) where:
      probe_pos      = boxed_pos + 1  (the '{' token, no prompt offset)
      written_tok_id = token at boxed_pos + 2 (what the model actually wrote)
    """
    tokens  = tokenizer.encode(generation, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]

    boxed_pos = None
    for i, t in enumerate(decoded):
        if 'boxed' in t:
            boxed_pos = i
            break

    if boxed_pos is None:
        return None, None

    probe_pos      = boxed_pos + 1
    written_tok_pos = boxed_pos + 2

    if written_tok_pos >= len(tokens) or probe_pos >= seq_len:
        return None, None

    return probe_pos, tokens[written_tok_pos]

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
df = df[df["extracted_answer"].isin(["A", "B", "C", "D"])]
df = df[df["is_correct"].isin([0, 1])]

def tensor_valid(row):
    p = os.path.join(ACT_DIR, row["tensor_file"])
    return os.path.exists(p) and os.path.getsize(p) > 1024
df = df[df.apply(tensor_valid, axis=1)].reset_index(drop=True)

if N_SAMPLES is not None:
    df = df.head(N_SAMPLES)

print(f"Valid examples: {len(df)}")
print(f"Correct: {df['is_correct'].sum()} | Incorrect: {(df['is_correct']==0).sum()}")

sample_act = torch.load(os.path.join(ACT_DIR, df.iloc[0]["tensor_file"]), map_location="cpu")
num_layers  = sample_act.shape[0]
del sample_act

layers_to_eval = list(range(num_layers - 1)) if SKIP_LAST_LAYER else list(range(num_layers))
print(f"Num layers: {num_layers} | Evaluating: {len(layers_to_eval)} "
      f"({'skipping layer 36' if SKIP_LAST_LAYER else 'all'})")

# ==========================================
# PRE-COMPUTE POSITIONS
# ==========================================
print("Finding probe positions...")
probe_positions  = []
written_tok_ids  = []   # what the model actually wrote
gt_tok_ids       = []   # ground truth letter
valid_mask       = []
n_fallback       = 0

for _, row in df.iterrows():
    probe_pos, written_id = find_probe_position(
        str(row["model_generation"]), int(row["sequence_length"])
    )
    gt_id = answer_token_ids[row["ground_truth"]]

    if probe_pos is None:
        probe_positions.append(None)
        written_tok_ids.append(None)
        gt_tok_ids.append(None)
        valid_mask.append(False)
        continue

    # Sanity check: written token should be A/B/C/D
    written_letter = row["extracted_answer"]
    expected_written_id = answer_token_ids.get(written_letter)
    if written_id != expected_written_id:
        # Fallback
        probe_pos  = max(0, int(row["sequence_length"]) - 8)
        written_id = expected_written_id
        n_fallback += 1

    probe_positions.append(probe_pos)
    written_tok_ids.append(written_id)
    gt_tok_ids.append(gt_id)
    valid_mask.append(True)

print(f"Valid: {sum(valid_mask)} | Fallbacks: {n_fallback}")

# ==========================================
# ACCUMULATORS
# Split by correct/incorrect
# Track: predicts_written, predicts_gt (top-1 and ABCD-restricted)
# ==========================================
def make_acc():
    return np.zeros(num_layers)

# Top-1 in full vocab
top1_written_correct   = make_acc()  # correct examples: predicts what was written (=gt)
top1_written_incorrect = make_acc()  # incorrect examples: predicts what was written (wrong)
top1_gt_incorrect      = make_acc()  # incorrect examples: predicts gt (right answer)

# ABCD-restricted
abcd_written_correct   = make_acc()
abcd_written_incorrect = make_acc()
abcd_gt_incorrect      = make_acc()

# Top-5
top5_written_correct   = make_acc()
top5_written_incorrect = make_acc()
top5_gt_incorrect      = make_acc()

n_correct   = sum(1 for i, (_, row) in enumerate(df.iterrows())
                  if valid_mask[i] and row["is_correct"] == 1)
n_incorrect = sum(1 for i, (_, row) in enumerate(df.iterrows())
                  if valid_mask[i] and row["is_correct"] == 0)
print(f"n_correct={n_correct} | n_incorrect={n_incorrect}\n")

# For heatmap
all_top1_tokens    = []
heatmap_is_correct = []
heatmap_gt_strs    = []
heatmap_written_strs = []

# ==========================================
# MAIN LOOP
# ==========================================
for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Examples")):
    if not valid_mask[i]:
        continue

    act        = torch.load(os.path.join(ACT_DIR, row["tensor_file"]), map_location="cpu")
    probe_pos  = probe_positions[i]
    written_id = written_tok_ids[i]
    gt_id      = gt_tok_ids[i]
    is_cor     = int(row["is_correct"])
    gt_label   = LABEL_MAP[row["ground_truth"]]
    wr_label   = LABEL_MAP[row["extracted_answer"]]

    example_top1_tokens = []

    for layer in layers_to_eval:
        hidden = act[layer, probe_pos, :].unsqueeze(0)
        top5_ids, top5_probs, full_logits = logit_lens(hidden)

        top1_id  = top5_ids[0].item()
        top1_str = tokenizer.decode([top1_id]).strip()
        example_top1_tokens.append(top1_str)

        # ABCD restricted
        abcd_logits   = full_logits[abcd_ids]
        abcd_pred_idx = abcd_logits.argmax().item()

        if is_cor:
            # Correct: written == gt, track both the same
            if top1_id == written_id:
                top1_written_correct[layer] += 1
            if written_id in top5_ids.tolist():
                top5_written_correct[layer] += 1
            if abcd_pred_idx == gt_label:
                abcd_written_correct[layer] += 1
        else:
            # Incorrect: written != gt, track separately
            if top1_id == written_id:
                top1_written_incorrect[layer] += 1
            if top1_id == gt_id:
                top1_gt_incorrect[layer] += 1
            if written_id in top5_ids.tolist():
                top5_written_incorrect[layer] += 1
            if gt_id in top5_ids.tolist():
                top5_gt_incorrect[layer] += 1
            if abcd_pred_idx == wr_label:
                abcd_written_incorrect[layer] += 1
            if abcd_pred_idx == gt_label:
                abcd_gt_incorrect[layer] += 1

    all_top1_tokens.append(example_top1_tokens)
    heatmap_is_correct.append(is_cor)
    heatmap_gt_strs.append(tokenizer.decode([gt_id]).strip())
    heatmap_written_strs.append(tokenizer.decode([written_id]).strip())
    del act

# Normalize
for layer in layers_to_eval:
    nc = n_correct   if n_correct   > 0 else 1
    ni = n_incorrect if n_incorrect > 0 else 1
    top1_written_correct[layer]   /= nc
    top5_written_correct[layer]   /= nc
    abcd_written_correct[layer]   /= nc
    top1_written_incorrect[layer] /= ni
    top1_gt_incorrect[layer]      /= ni
    top5_written_incorrect[layer] /= ni
    top5_gt_incorrect[layer]      /= ni
    abcd_written_incorrect[layer] /= ni
    abcd_gt_incorrect[layer]      /= ni

# ==========================================
# SAVE
# ==========================================
pd.DataFrame({
    "layer":                    layers_to_eval,
    "top1_written_correct":     top1_written_correct[layers_to_eval],
    "top5_written_correct":     top5_written_correct[layers_to_eval],
    "abcd_written_correct":     abcd_written_correct[layers_to_eval],
    "top1_written_incorrect":   top1_written_incorrect[layers_to_eval],
    "top1_gt_incorrect":        top1_gt_incorrect[layers_to_eval],
    "top5_written_incorrect":   top5_written_incorrect[layers_to_eval],
    "top5_gt_incorrect":        top5_gt_incorrect[layers_to_eval],
    "abcd_written_incorrect":   abcd_written_incorrect[layers_to_eval],
    "abcd_gt_incorrect":        abcd_gt_incorrect[layers_to_eval],
}).to_csv(os.path.join(RESULTS_DIR, "logit_lens_summary.csv"), index=False)

# ==========================================
# PLOT 1: Main results — 3 metrics, clean separation
# ==========================================
L = layers_to_eval
COLORS = {
    "correct_written": "#E69F00",
    "incorrect_written": "#56B4E9",
    "incorrect_gt": "#CC79A7",
}

fig, axes = plt.subplots(1, 3, figsize=(21, 5))
fig.suptitle(
    "Logit Lens — Does the Model Know the Answer Before Writing It?\n"
    "(Qwen3-4B on LogiQA v2, probe at '{' token inside \\boxed{}, layer 36 excluded)",
    fontsize=13
)

metrics = [
    (top1_written_correct,   top1_written_incorrect,   top1_gt_incorrect,
     "Top-1 Accuracy (full vocab)"),
    (top5_written_correct,   top5_written_incorrect,   top5_gt_incorrect,
     "Top-5 Accuracy (full vocab)"),
    (abcd_written_correct,   abcd_written_incorrect,   abcd_gt_incorrect,
     "ABCD-restricted Accuracy"),
]

for ax, (corr_wr, incorr_wr, incorr_gt, title) in zip(axes, metrics):
    ax.plot(L, corr_wr[L],   marker='o', markersize=3,
            color=COLORS["correct_written"],
            label=f"Correct — predicts written=gt (n={n_correct})")
    ax.plot(L, incorr_wr[L], marker='s', markersize=3,
            color=COLORS["incorrect_written"],
            label=f"Incorrect — predicts written answer (n={n_incorrect})")
    ax.plot(L, incorr_gt[L], marker='^', markersize=3,
            color=COLORS["incorrect_gt"],
            label=f"Incorrect — predicts gt answer (n={n_incorrect})")
    ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (25%)')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Layer"); ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(0, num_layers, 2))
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "logit_lens_accuracy_curves.png"), dpi=300)
plt.close()
print("Saved: logit_lens_accuracy_curves.png")

# ==========================================
# PLOT 2: Heatmap — sorted correct first
# ==========================================
n_show     = min(N_HEATMAP_EXAMPLES, len(all_top1_tokens))
sort_order = np.argsort([-c for c in heatmap_is_correct[:n_show]])

token_matrix     = [all_top1_tokens[j]      for j in sort_order]
is_cor_sorted    = [heatmap_is_correct[j]   for j in sort_order]
gt_sorted        = [heatmap_gt_strs[j]      for j in sort_order]
written_sorted   = [heatmap_written_strs[j] for j in sort_order]

# Color: green = predicts written answer, red = neither, yellow = predicts gt (for incorrect)
color_matrix = np.zeros((n_show, len(L)))
for i in range(n_show):
    for j, layer in enumerate(L):
        tok = token_matrix[i][j]
        if tok == written_sorted[i]:
            color_matrix[i, j] = 1   # predicts written
        elif tok == gt_sorted[i] and is_cor_sorted[i] == 0:
            color_matrix[i, j] = 2   # predicts gt (only meaningful for incorrect)
        else:
            color_matrix[i, j] = 0   # neither

fig, ax = plt.subplots(figsize=(max(16, len(L) * 0.5), max(8, n_show * 0.35)))
cmap = mcolors.ListedColormap(["#D55E00", "#009E73", "#F0E442"])
ax.imshow(color_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=2, interpolation='none')

for i in range(n_show):
    for j in range(len(L)):
        ax.text(j, i, token_matrix[i][j][:4], ha='center', va='center',
                fontsize=5, color='white', fontweight='bold')

ylabels = [
    f"{'✓' if c==1 else '✗'} gt={gt_sorted[i]} wr={written_sorted[i]}"
    for i, c in enumerate(is_cor_sorted)
]
ax.set_yticks(range(n_show)); ax.set_yticklabels(ylabels, fontsize=7)
ax.set_xticks(range(0, len(L), 2)); ax.set_xticklabels(L[::2])
ax.set_xlabel("Layer"); ax.set_ylabel("Example")
ax.set_title("Logit Lens Top-1 Token at '{' inside \\boxed{}\n"
             "(Qwen3-4B on LogiQA v2, layer 36 excluded)")
ax.legend(handles=[
    Patch(facecolor="#009E73", label="Predicts written answer"),
    Patch(facecolor="#F0E442", label="Predicts gt (incorrect examples only)"),
    Patch(facecolor="#D55E00", label="Predicts neither"),
], loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "logit_lens_heatmap.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: logit_lens_heatmap.png")

# ==========================================
# SUMMARY
# ==========================================
print(f"\n{'='*60}")
print(f"n_correct={n_correct} | n_incorrect={n_incorrect}")
best = lambda arr: L[int(np.argmax(arr[L]))]
print(f"Best layer — top1 written correct:   {best(top1_written_correct)}")
print(f"Best layer — top1 written incorrect: {best(top1_written_incorrect)}")
print(f"Best layer — top1 gt incorrect:      {best(top1_gt_incorrect)}")
print(f"Best layer — abcd written correct:   {best(abcd_written_correct)}")
print(f"Best layer — abcd written incorrect: {best(abcd_written_incorrect)}")
print(f"Best layer — abcd gt incorrect:      {best(abcd_gt_incorrect)}")
print(f"Results → {RESULTS_DIR}/")
print(f"{'='*60}")