"""
extract_activations_gsm8k_v3.py
────────────────────────────────
Extracts activations for GSM8K with Qwen3-4B thinking mode.
Instead of saving all tokens, saves 100 evenly-spaced token positions per example.

Two versions saved per example:
  - sampled:  hidden state AT each of the 100 positions
              shape: (num_layers, 100, hidden_dim)
              OR (num_layers, N, hidden_dim) if sequence < 100 tokens
  - averaged: mean of hidden states WITHIN each 1% window
              shape: (num_layers, 100, hidden_dim)
              OR (num_layers, N, hidden_dim) if sequence < 100 tokens

Saved to:
  /playpen-ssd/yuvraj/llm-probing-project/activations/gsm8k_v3/sampled/
  /playpen-ssd/yuvraj/llm-probing-project/activations/gsm8k_v3/averaged/
"""

import os
import re
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse

# ==========================================
# ARGS
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--worker_id",   type=int, default=0)
parser.add_argument("--num_workers", type=int, default=9)
parser.add_argument("--gpu_id",      type=int, default=0)
args = parser.parse_args()

# ==========================================
# CONFIG
# ==========================================
BASE_DIR      = "/playpen-ssd/yuvraj/llm-probing-project/activations/gsm8k_v3"
SAMPLED_DIR   = os.path.join(BASE_DIR, "sampled")
AVERAGED_DIR  = os.path.join(BASE_DIR, "averaged")
MODEL_PATH    = "/playpen-ssd/shared/pretrained_models/models--Qwen--Qwen3-4B"
N_POSITIONS   = 100       # number of evenly-spaced token positions to save
MAX_NEW_TOKENS = 4096

os.makedirs(SAMPLED_DIR,  exist_ok=True)
os.makedirs(AVERAGED_DIR, exist_ok=True)

# ==========================================
# LOAD MODEL
# ==========================================
print(f"Worker {args.worker_id}: loading model on cuda:{args.gpu_id}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=f"cuda:{args.gpu_id}",
)
model.eval()

# ==========================================
# DATASET SPLIT
# ==========================================
dataset   = load_dataset("gsm8k", "main", split="train")
total     = len(dataset)
chunk     = total // args.num_workers
start_idx = args.worker_id * chunk
end_idx   = total if args.worker_id == args.num_workers - 1 else start_idx + chunk
subset    = dataset.select(range(start_idx, end_idx))
print(f"Worker {args.worker_id}: examples {start_idx} to {end_idx-1} on cuda:{args.gpu_id}")

# ==========================================
# PROMPT & PARSING
# ==========================================
SYSTEM_PROMPT = (
    "You are a math problem solver. "
    "Solve the problem step by step using simple prose, no LaTeX formatting, no headers. "
    "End your answer with: The final answer is \\boxed{YOUR_ANSWER}."
)

def build_prompt(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

def extract_ground_truth(answer_str):
    return answer_str.split("#### ")[-1].strip()

def extract_model_answer(generation_str):
    match = re.search(r"\\boxed\{(.*?)\}", generation_str)
    if match:
        return match.group(1).strip()
    return None

def extract_thinking(generation_str):
    match = re.search(r"<think>(.*?)</think>", generation_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def find_think_end_position(generation_str):
    """Token index of </think> in the generation."""
    tokens  = tokenizer.encode(generation_str, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]
    for i, t in enumerate(decoded):
        if '</think>' in t or ('/' in t and 'think' in ''.join(decoded[max(0, i-2):i+2])):
            return i
    return None

# ==========================================
# SPARSE SAMPLING FUNCTIONS
# ==========================================
def get_sample_indices(seq_len, n_positions=N_POSITIONS):
    """
    Returns n_positions evenly-spaced indices into [0, seq_len).
    If seq_len < n_positions, returns all indices (0 to seq_len-1).
    Uses np.linspace to guarantee exactly n_positions unique indices.
    """
    if seq_len <= n_positions:
        return list(range(seq_len))
    # linspace from 0 to seq_len-1, rounded to nearest int, deduplicated
    indices = np.linspace(0, seq_len - 1, n_positions)
    return sorted(set(int(round(i)) for i in indices))


def get_window_boundaries(seq_len, n_positions=N_POSITIONS):
    """
    Splits [0, seq_len) into n_positions contiguous windows.
    Returns list of (start, end) tuples (end is exclusive).
    If seq_len < n_positions, returns one window per token.
    """
    if seq_len <= n_positions:
        return [(i, i + 1) for i in range(seq_len)]
    boundaries = []
    edges = np.linspace(0, seq_len, n_positions + 1)
    for i in range(n_positions):
        start = int(round(edges[i]))
        end   = int(round(edges[i + 1]))
        # Ensure at least one token per window
        if end <= start:
            end = start + 1
        # Clamp to valid range
        start = min(start, seq_len - 1)
        end   = min(end,   seq_len)
        boundaries.append((start, end))
    return boundaries


def build_sampled_tensor(all_layer_tokens, sample_indices, num_layers):
    """
    Build sparse sampled tensor.
    all_layer_tokens: list of length num_layers, each a list of tensors (one per token)
    sample_indices: list of int indices to keep
    Returns: tensor of shape (num_layers, len(sample_indices), hidden_dim)
    """
    layers = []
    for layer_idx in range(num_layers):
        layer_stack = torch.stack(all_layer_tokens[layer_idx])  # (seq_len, hidden_dim)
        sampled     = layer_stack[sample_indices, :]             # (n_pos, hidden_dim)
        layers.append(sampled)
    return torch.stack(layers)  # (num_layers, n_pos, hidden_dim)


def build_averaged_tensor(all_layer_tokens, windows, num_layers):
    """
    Build window-averaged tensor.
    windows: list of (start, end) tuples
    Returns: tensor of shape (num_layers, len(windows), hidden_dim)
    """
    layers = []
    for layer_idx in range(num_layers):
        layer_stack = torch.stack(all_layer_tokens[layer_idx])  # (seq_len, hidden_dim)
        averaged    = []
        for (start, end) in windows:
            window_mean = layer_stack[start:end, :].mean(dim=0)  # (hidden_dim,)
            averaged.append(window_mean)
        layers.append(torch.stack(averaged))  # (n_windows, hidden_dim)
    return torch.stack(layers)  # (num_layers, n_windows, hidden_dim)


# ==========================================
# MAIN LOOP
# ==========================================
log_data = []

with torch.no_grad():
    for idx, example in enumerate(tqdm(subset, desc=f"Worker {args.worker_id}")):
        global_idx   = start_idx + idx
        question     = example["question"]
        ground_truth = extract_ground_truth(example["answer"])

        # Check if already processed (resume support)
        sampled_path  = os.path.join(SAMPLED_DIR,  f"example_{global_idx}.pt")
        averaged_path = os.path.join(AVERAGED_DIR, f"example_{global_idx}.pt")
        if os.path.exists(sampled_path) and os.path.exists(averaged_path) \
                and os.path.getsize(sampled_path) > 1024 \
                and os.path.getsize(averaged_path) > 1024:
            continue

        prompt       = build_prompt(question)
        inputs       = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]

        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|im_end|>"),
                ],
            )
        except torch.cuda.OutOfMemoryError:
            print(f"WARNING: OOM on example {global_idx}, skipping.")
            torch.cuda.empty_cache()
            continue

        generated_tokens = outputs.sequences[0][input_length:]
        # Keep special tokens so we can find </think>
        full_generation  = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        model_answer  = extract_model_answer(full_generation)
        thinking_text = extract_thinking(full_generation)
        think_end_pos = find_think_end_position(full_generation)
        is_correct    = 1 if model_answer == ground_truth else 0

        num_layers = len(outputs.hidden_states[0])
        num_steps  = len(outputs.hidden_states)
        seq_len    = num_steps  # one activation per generated token

        # Build per-layer list of token tensors
        # Position 0 = last prompt token (predicts first generated token)
        # Position k = hidden state after generating token k
        all_layer_tokens = []
        for layer_idx in range(num_layers):
            layer_tokens = [outputs.hidden_states[0][layer_idx][0, -1, :].cpu().clone()]
            for step_idx in range(1, num_steps):
                layer_tokens.append(
                    outputs.hidden_states[step_idx][layer_idx][0, 0, :].cpu().clone()
                )
            all_layer_tokens.append(layer_tokens)

        # Compute sampling indices and window boundaries
        sample_indices = get_sample_indices(seq_len)
        windows        = get_window_boundaries(seq_len)

        # Build and save tensors
        sampled_tensor  = build_sampled_tensor(all_layer_tokens, sample_indices, num_layers)
        averaged_tensor = build_averaged_tensor(all_layer_tokens, windows, num_layers)

        # Validate shapes
        assert sampled_tensor.shape[0]  == num_layers, "Sampled layer dim mismatch"
        assert averaged_tensor.shape[0] == num_layers, "Averaged layer dim mismatch"
        assert sampled_tensor.shape[2]  == averaged_tensor.shape[2], "Hidden dim mismatch"

        torch.save(sampled_tensor,  sampled_path)
        torch.save(averaged_tensor, averaged_path)

        # Free GPU memory
        del outputs, all_layer_tokens, sampled_tensor, averaged_tensor
        torch.cuda.empty_cache()

        log_data.append({
            "example_id":            global_idx,
            "question":              question,
            "ground_truth":          ground_truth,
            "model_generation":      full_generation,
            "extracted_answer":      model_answer,
            "thinking":              thinking_text,
            "think_end_pos":         think_end_pos,
            "is_correct":            is_correct,
            "sampled_tensor_file":   f"example_{global_idx}.pt",
            "averaged_tensor_file":  f"example_{global_idx}.pt",
            "sequence_length":       seq_len,
            "n_sampled_positions":   len(sample_indices),
            "resumed":               False,
        })

        # Incremental save every 50 examples
        if (idx + 1) % 50 == 0:
            pd.DataFrame(log_data).to_csv(
                os.path.join(BASE_DIR, f"master_log_worker{args.worker_id}.csv"),
                index=False
            )

# Final save
pd.DataFrame(log_data).to_csv(
    os.path.join(BASE_DIR, f"master_log_worker{args.worker_id}.csv"),
    index=False
)
print(f"Worker {args.worker_id} done! Processed {len(log_data)} examples.")