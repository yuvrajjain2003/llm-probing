"""
logit_lens_token_heatmap.py
────────────────────────────
For a small set of examples (correct and incorrect), apply the unembedding
matrix at every layer and visualize the top-5 predicted tokens + their
probabilities as a readable heatmap.

This lets you SEE what the model is predicting at each layer — no
aggregated metrics, just raw token predictions.
"""

import os, glob, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# CONFIG
# ==========================================
BASE_DIR    = "/playpen-ssd/yuvraj/llm-probing-project"
ACT_DIR     = os.path.join(BASE_DIR, "activations/logiqa_v2")
MODEL_PATH  = "/playpen-ssd/shared/pretrained_models/models--Qwen--Qwen3-4B"
RESULTS_DIR = os.path.join(BASE_DIR, "results/logit_lens/logiqa_v2_heatmap")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
N_CORRECT   = 5    # how many correct examples to show
N_INCORRECT = 5    # how many incorrect examples to show
TOP_K       = 5    # top-k tokens to show per layer

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
del model
torch.cuda.empty_cache()
print("Model loaded\n")

# ==========================================
# LOGIT LENS
# ==========================================
@torch.no_grad()
def get_top_k(hidden_state, k=5):
    """hidden: (hidden_dim,) → list of (token_str, prob)"""
    h      = hidden_state.unsqueeze(0).bfloat16().to(DEVICE)
    normed = final_norm(h)
    logits = lm_head(normed).float()
    probs  = torch.softmax(logits, dim=-1)
    top_p, top_i = probs.topk(k, dim=-1)
    tokens = [tokenizer.decode([idx.item()]) for idx in top_i[0]]
    probs  = top_p[0].tolist()
    return list(zip(tokens, probs))

# ==========================================
# FIND PROBE POSITION
# ==========================================
def find_probe_position(generation, seq_len):
    tokens  = tokenizer.encode(generation, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]
    for i, t in enumerate(decoded):
        if 'boxed' in t:
            probe_pos      = i + 1   # '{' token
            written_tok_pos = i + 2  # answer letter
            if written_tok_pos >= len(tokens) or probe_pos >= seq_len:
                return None, None, None
            return probe_pos, tokens[written_tok_pos], decoded[written_tok_pos]
    return None, None, None

# ==========================================
# LOAD DATA
# ==========================================
dfs = [pd.read_csv(f) for f in glob.glob(os.path.join(ACT_DIR, "master_log_worker*.csv"))]
df  = pd.concat(dfs, ignore_index=True).drop_duplicates("example_id")
if "resumed" in df.columns:
    df = df[df["resumed"] == False].reset_index(drop=True)

df = df[df["model_generation"].str.contains("boxed", na=False)]
df = df[df["ground_truth"].isin(["A","B","C","D"])]
df = df[df["extracted_answer"].isin(["A","B","C","D"])]
df = df[df["is_correct"].isin([0, 1])]

def tensor_valid(row):
    p = os.path.join(ACT_DIR, row["tensor_file"])
    return os.path.exists(p) and os.path.getsize(p) > 1024
df = df[df.apply(tensor_valid, axis=1)].reset_index(drop=True)

correct_df   = df[df["is_correct"] == 1].head(N_CORRECT)
incorrect_df = df[df["is_correct"] == 0].head(N_INCORRECT)
examples     = pd.concat([correct_df, incorrect_df]).reset_index(drop=True)
print(f"Showing {N_CORRECT} correct + {N_INCORRECT} incorrect examples\n")

# ==========================================
# GENERATE HEATMAP PER EXAMPLE
# ==========================================
for _, row in examples.iterrows():
    gen     = str(row["model_generation"])
    seq_len = int(row["sequence_length"])
    is_cor  = int(row["is_correct"])
    gt      = row["ground_truth"]
    written = row["extracted_answer"]

    probe_pos, written_tok_id, written_tok_str = find_probe_position(gen, seq_len)
    if probe_pos is None:
        print(f"Skipping example {row['example_id']} — no boxed found")
        continue

    act        = torch.load(os.path.join(ACT_DIR, row["tensor_file"]), map_location="cpu")
    num_layers = act.shape[0]
    layers     = list(range(num_layers - 1))  # skip layer 36 (known artifact)

    # Collect top-k per layer
    # layer_data[layer] = [(token_str, prob), ...]
    layer_data = {}
    for layer in layers:
        hidden = act[layer, probe_pos, :]
        layer_data[layer] = get_top_k(hidden, k=TOP_K)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(len(layers) * 0.55, TOP_K * 1.1 + 1.5))

    # Build color matrix: green if token matches written answer, gold if matches gt (for incorrect)
    color_matrix = np.zeros((TOP_K, len(layers)))
    label_matrix = [["" for _ in layers] for _ in range(TOP_K)]
    prob_matrix  = np.zeros((TOP_K, len(layers)))

    for j, layer in enumerate(layers):
        for k, (tok, prob) in enumerate(layer_data[layer]):
            tok_clean = tok.strip()
            label_matrix[k][j] = f"{tok_clean[:6]}\n{prob:.2f}"
            prob_matrix[k][j]  = prob
            if tok_clean == written:
                color_matrix[k][j] = 2   # written answer
            elif tok_clean == gt and not is_cor:
                color_matrix[k][j] = 1   # gt (only meaningful for incorrect)
            else:
                color_matrix[k][j] = 0

    cmap = mcolors.ListedColormap(["#f5f5f5", "#F0E442", "#009E73"])
    ax.imshow(color_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=2,
              interpolation='none', alpha=0.7)

    # Add text labels
    for k in range(TOP_K):
        for j in range(len(layers)):
            ax.text(j, k, label_matrix[k][j], ha='center', va='center',
                    fontsize=6.5, color='black')

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers], fontsize=7)
    ax.set_yticks(range(TOP_K))
    ax.set_yticklabels([f"Top-{k+1}" for k in range(TOP_K)], fontsize=8)
    ax.set_xlabel("Layer", fontsize=10)

    status = "✓ CORRECT" if is_cor else "✗ INCORRECT"
    title  = (f"{status} | example_id={row['example_id']} | "
              f"ground_truth={gt} | written={written}\n"
              f"Probe at '{{' token (pos={probe_pos}) — predicting next token")
    ax.set_title(title, fontsize=9, pad=10)

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#009E73", alpha=0.7, label=f"Written answer ({written})"),
        Patch(facecolor="#f5f5f5", alpha=0.7, label="Other token", edgecolor='gray'),
    ]
    if not is_cor:
        legend.insert(1, Patch(facecolor="#F0E442", alpha=0.7,
                               label=f"Ground truth ({gt}) — model wrote {written}"))
    ax.legend(handles=legend, loc='upper left',
              bbox_to_anchor=(0, -0.12), ncol=3, fontsize=8)

    plt.tight_layout()
    fname = f"example_{row['example_id']}_{'correct' if is_cor else 'incorrect'}.png"
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname} | gt={gt} written={written} probe_pos={probe_pos}")
    del act

print(f"\nAll heatmaps saved → {RESULTS_DIR}/")