"""
precompute_boxed_positions.py
─────────────────────────────
Adds a 'boxed_token_pos' column to the master_log.csv.
This is the index of the 'boxed' token in the tokenized generation,
which we use to truncate activations at the right position during training.

For examples where \boxed{} is not found, boxed_token_pos is set to -1
and those examples will be excluded from training.
"""

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

CSV_PATH   = "/playpen-ssd/yuvraj/llm-probing-project/activations/gsm8k/master_log.csv"
MODEL_PATH = "/playpen-ssd/shared/pretrained_models/models--Qwen--Qwen3-4B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
print(f"Total examples: {len(df)}")

def find_boxed_token_pos(generation):
    """
    Returns the token index of the first 'boxed' token in the generation.
    Returns -1 if not found.
    Note: this is the position in the GENERATION tokens, not the activation tensor.
    The activation tensor has an offset of +1 (activation[0] = last prompt token).
    """
    if not isinstance(generation, str):
        return -1
    tokens  = tokenizer.encode(generation, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]
    for i, t in enumerate(decoded):
        if 'boxed' in t:
            return i  # token index in generation
    return -1

print("Computing boxed positions...")
tqdm.pandas()
df['boxed_token_pos'] = df['model_generation'].progress_apply(find_boxed_token_pos)

completed  = (df['boxed_token_pos'] >= 0).sum()
missing    = (df['boxed_token_pos'] == -1).sum()
print(f"Found \\boxed{{}} in {completed} / {len(df)} examples ({missing} missing, will be excluded)")
print(f"Mean boxed position: {df[df['boxed_token_pos'] >= 0]['boxed_token_pos'].mean():.1f} tokens")
print(f"Min boxed position:  {df[df['boxed_token_pos'] >= 0]['boxed_token_pos'].min()}")
print(f"Max boxed position:  {df[df['boxed_token_pos'] >= 0]['boxed_token_pos'].max()}")

df.to_csv(CSV_PATH, index=False)
print(f"\nSaved updated CSV to {CSV_PATH}")
print("New column 'boxed_token_pos' added.")