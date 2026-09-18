"""
extract_activations_logiqa_v3.py
─────────────────────────────────
Extracts activations for LogiQA with Qwen3-4B thinking mode.
Same sparse sampling strategy as gsm8k_v3:
  - sampled:  hidden state AT each of the 100 evenly-spaced positions
  - averaged: mean of hidden states WITHIN each 1% window

Saved to:
  /playpen-ssd/yuvraj/llm-probing-project/activations/logiqa_v3/sampled/
  /playpen-ssd/yuvraj/llm-probing-project/activations/logiqa_v3/averaged/
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
BASE_DIR      = "/playpen-ssd/yuvraj/llm-probing-project/activations/logiqa_v3"
SAMPLED_DIR   = os.path.join(BASE_DIR, "sampled")
AVERAGED_DIR  = os.path.join(BASE_DIR, "averaged")
MODEL_PATH    = "/playpen-ssd/shared/pretrained_models/models--Qwen--Qwen3-4B"
N_POSITIONS   = 100
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
LOGIQA_URL = "hf://datasets/lucasmccabe/logiqa@refs/convert/parquet/default/train/*.parquet"
dataset    = load_dataset("parquet", data_files={"train": LOGIQA_URL}, split="train")
total      = len(dataset)
chunk      = total // args.num_workers
start_idx  = args.worker_id * chunk
end_idx    = total if args.worker_id == args.num_workers - 1 else start_idx + chunk
subset     = dataset.select(range(start_idx, end_idx))
print(f"Worker {args.worker_id}: examples {start_idx} to {end_idx-1} on cuda:{args.gpu_id}")

# ==========================================
# PROMPT & PARSING
# ==========================================
SYSTEM_PROMPT = (
    "You are a logical reasoning expert. "
    "Read the passage carefully and answer the multiple-choice question by selecting "
    "the single best option (A, B, C, or D). "
    "Think step by step using simple prose, no LaTeX formatting, no headers. "
    "End your answer with: The final answer is \\boxed{LETTER} where LETTER is A, B, C, or D."
)

LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

def build_prompt(context, question, options):
    user_content = (
        f"Passage: {context}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"C. {options[2]}\n"
        f"D. {options[3]}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

def extract_ground_truth(example):
    return LABEL_MAP[example["correct_option"]]

def extract_model_answer(generation_str):
    match = re.search(r"\\boxed\{([A-D])\}", generation_str)
    if match:
        return match.group(1).strip()
    match = re.search(r"(?:answer is|answer:)\s*([A-D])\b", generation_str, re.IGNORECASE)
    if match:
        return match.group(1).strip().upper()
    return None

def extract_thinking(generation_str):
    match = re.search(r"<think>(.*?)</think>", generation_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def find_think_end_position(generation_str):
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
    if seq_len <= n_positions:
        return list(range(seq_len))
    indices = np.linspace(0, seq_len - 1, n_positions)
    return sorted(set(int(round(i)) for i in indices))


def get_window_boundaries(seq_len, n_positions=N_POSITIONS):
    if seq_len <= n_positions:
        return [(i, i + 1) for i in range(seq_len)]
    boundaries = []
    edges = np.linspace(0, seq_len, n_positions + 1)
    for i in range(n_positions):
        start = int(round(edges[i]))
        end   = int(round(edges[i + 1]))
        if end <= start:
            end = start + 1
        start = min(start, seq_len - 1)
        end   = min(end,   seq_len)
        boundaries.append((start, end))
    return boundaries


def build_sampled_tensor(all_layer_tokens, sample_indices, num_layers):
    layers = []
    for layer_idx in range(num_layers):
        layer_stack = torch.stack(all_layer_tokens[layer_idx])
        sampled     = layer_stack[sample_indices, :]
        layers.append(sampled)
    return torch.stack(layers)


def build_averaged_tensor(all_layer_tokens, windows, num_layers):
    layers = []
    for layer_idx in range(num_layers):
        layer_stack = torch.stack(all_layer_tokens[layer_idx])
        averaged    = []
        for (start, end) in windows:
            averaged.append(layer_stack[start:end, :].mean(dim=0))
        layers.append(torch.stack(averaged))
    return torch.stack(layers)


# ==========================================
# MAIN LOOP
# ==========================================
log_data = []

with torch.no_grad():
    for idx, example in enumerate(tqdm(subset, desc=f"Worker {args.worker_id}")):
        global_idx   = start_idx + idx
        context      = example["context"]
        question     = example["query"]
        options      = example["options"]
        ground_truth = extract_ground_truth(example)

        sampled_path  = os.path.join(SAMPLED_DIR,  f"example_{global_idx}.pt")
        averaged_path = os.path.join(AVERAGED_DIR, f"example_{global_idx}.pt")
        if os.path.exists(sampled_path) and os.path.exists(averaged_path) \
                and os.path.getsize(sampled_path) > 1024 \
                and os.path.getsize(averaged_path) > 1024:
            continue

        prompt       = build_prompt(context, question, options)
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
        full_generation  = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        model_answer  = extract_model_answer(full_generation)
        thinking_text = extract_thinking(full_generation)
        think_end_pos = find_think_end_position(full_generation)
        is_correct    = 1 if model_answer == ground_truth else 0

        num_layers = len(outputs.hidden_states[0])
        num_steps  = len(outputs.hidden_states)
        seq_len    = num_steps

        all_layer_tokens = []
        for layer_idx in range(num_layers):
            layer_tokens = [outputs.hidden_states[0][layer_idx][0, -1, :].cpu().clone()]
            for step_idx in range(1, num_steps):
                layer_tokens.append(
                    outputs.hidden_states[step_idx][layer_idx][0, 0, :].cpu().clone()
                )
            all_layer_tokens.append(layer_tokens)

        sample_indices = get_sample_indices(seq_len)
        windows        = get_window_boundaries(seq_len)

        sampled_tensor  = build_sampled_tensor(all_layer_tokens, sample_indices, num_layers)
        averaged_tensor = build_averaged_tensor(all_layer_tokens, windows, num_layers)

        assert sampled_tensor.shape[0]  == num_layers
        assert averaged_tensor.shape[0] == num_layers
        assert sampled_tensor.shape[2]  == averaged_tensor.shape[2]

        torch.save(sampled_tensor,  sampled_path)
        torch.save(averaged_tensor, averaged_path)

        del outputs, all_layer_tokens, sampled_tensor, averaged_tensor
        torch.cuda.empty_cache()

        log_data.append({
            "example_id":            global_idx,
            "context":               context,
            "question":              question,
            "options":               options,
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

        if (idx + 1) % 50 == 0:
            pd.DataFrame(log_data).to_csv(
                os.path.join(BASE_DIR, f"master_log_worker{args.worker_id}.csv"),
                index=False
            )

pd.DataFrame(log_data).to_csv(
    os.path.join(BASE_DIR, f"master_log_worker{args.worker_id}.csv"),
    index=False
)
print(f"Worker {args.worker_id} done! Processed {len(log_data)} examples.")