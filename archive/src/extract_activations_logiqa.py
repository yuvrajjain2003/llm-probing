import os
import re
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse

# ==========================================
# 1. Setup & Initialization
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--worker_id", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=15)
parser.add_argument("--gpu_id", type=int, default=0)
args = parser.parse_args()

SAVE_DIR   = "/playpen-ssd/yuvraj/llm-probing-project/activations/logiqa_v2"
MODEL_PATH = "/playpen-ssd/shared/pretrained_models/models--Qwen--Qwen3-4B"

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=f"cuda:{args.gpu_id}"
)
model.eval()

LOGIQA_TRAIN_URL = "hf://datasets/lucasmccabe/logiqa@refs/convert/parquet/default/train/*.parquet"
dataset = load_dataset("parquet", data_files={"train": LOGIQA_TRAIN_URL}, split="train")

# Split dataset among workers
total      = len(dataset)
chunk_size = total // args.num_workers
start_idx  = args.worker_id * chunk_size
end_idx    = total if args.worker_id == args.num_workers - 1 else start_idx + chunk_size
subset     = dataset.select(range(start_idx, end_idx))

print(f"Worker {args.worker_id}: processing examples {start_idx} to {end_idx} on cuda:{args.gpu_id}")


# ==========================================
# 2. Prompting & Parsing
# ==========================================
SYSTEM_PROMPT = (
    "You are a logical reasoning expert. "
    "Read the passage carefully and answer the multiple-choice question by selecting the single best option (A, B, C, or D). "
    "Think step by step. "
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
# 3. Generation & Activation Extraction
# ==========================================
log_data = []

with torch.no_grad():
    for idx, example in enumerate(tqdm(subset, desc=f"Worker {args.worker_id}")):
        global_idx   = start_idx + idx
        context      = example["context"]
        question     = example["query"]
        options      = example["options"]
        ground_truth = extract_ground_truth(example)

        prompt       = build_prompt(context, question, options)
        inputs       = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ],
        )

        generated_tokens = outputs.sequences[0][input_length:]
        full_generation  = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        model_answer  = extract_model_answer(full_generation)
        thinking_text = extract_thinking(full_generation)
        think_end_pos = find_think_end_position(full_generation)
        is_correct    = 1 if model_answer == ground_truth else 0

        num_layers = len(outputs.hidden_states[0])
        num_steps  = len(outputs.hidden_states)

        layer_tensors = []
        for layer_idx in range(num_layers):
            layer_tokens = []
            prompt_last_token = outputs.hidden_states[0][layer_idx][0, -1, :].cpu().clone()
            layer_tokens.append(prompt_last_token)
            for step_idx in range(1, num_steps):
                gen_token = outputs.hidden_states[step_idx][layer_idx][0, 0, :].cpu().clone()
                layer_tokens.append(gen_token)
            layer_tensors.append(torch.stack(layer_tokens))

        activations_tensor = torch.stack(layer_tensors)

        tensor_filename = f"example_{global_idx}.pt"
        torch.save(activations_tensor, os.path.join(SAVE_DIR, tensor_filename))

        log_data.append({
            "example_id":       global_idx,
            "context":          context,
            "question":         question,
            "options":          options,
            "ground_truth":     ground_truth,
            "model_generation": full_generation,
            "extracted_answer": model_answer,
            "thinking":         thinking_text,
            "think_end_pos":    think_end_pos,
            "is_correct":       is_correct,
            "tensor_file":      tensor_filename,
            "sequence_length":  activations_tensor.shape[1],
        })

        # Incremental save every 50 examples in case of crash
        if (idx + 1) % 50 == 0:
            pd.DataFrame(log_data).to_csv(
                os.path.join(SAVE_DIR, f"master_log_worker{args.worker_id}.csv"), index=False
            )

df_log = pd.DataFrame(log_data)
df_log.to_csv(os.path.join(SAVE_DIR, f"master_log_worker{args.worker_id}.csv"), index=False)
print(f"Worker {args.worker_id} done! Processed {len(log_data)} examples.")