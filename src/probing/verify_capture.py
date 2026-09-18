#!/usr/bin/env python3
"""Correctness gate for EasySteer activation capture.

EasySteer is a *forked* vLLM. If its layer indexing or position convention differs
from what we assume, every downstream probe is silently garbage. So we do not
assume: we capture ALL layers at chosen positions from both EasySteer and stock HF
transformers on the identical string, then match them by cosine similarity and
report the discovered mapping.

  --stage hf   : reference pass with transformers, saves hf_ref.npz
  --stage es   : EasySteer capture, saves es_cap.npz
  --stage cmp  : match layers/positions, print the mapping and verdict

Run hf and es as separate processes: each loads a 7B model and vLLM reserves a
large share of the GPU.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np

# src/ on the path, so `generation.*` resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUT = Path("runs/goal2/_verify"); OUT.mkdir(parents=True, exist_ok=True)
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
GPQA = Path("runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")


def build_case():
    """One GPQA CoT -> (full_text, n_prompt_tokens, positions) using Goal 1's own helpers."""
    import pandas as pd
    from transformers import AutoTokenizer
    from generation.run_eval import render_mc_prompt
    from generation.checkpoint import extract_cot

    tok = AutoTokenizer.from_pretrained(MODEL)
    ex = pd.read_csv(GPQA / "raw" / "examples.csv").set_index("example_id")
    recs = [json.loads(l) for l in open(GPQA / "raw" / "sample_records.jsonl")]
    # shortest CoT that closed </think>, to keep the reference pass small
    cand = [r for r in recs if "</think>" in r["raw_completion"]]
    rec = min(cand, key=lambda r: r["reasoning_tokens"])
    row = ex.loc[rec["example_id"]]
    prompt = render_mc_prompt(row["question"], json.loads(row["choices"]), tok)
    think = extract_cot(rec["raw_completion"])
    full = prompt + think
    n_prompt = len(tok.encode(prompt, add_special_tokens=False))
    n_full = len(tok.encode(full, add_special_tokens=False))
    # 8 positions spread through the think span
    pos = np.linspace(n_prompt, n_full - 1, 8).round().astype(int).tolist()
    return dict(full=full, n_prompt=n_prompt, n_full=n_full, positions=pos,
                example_id=int(rec["example_id"]), cot_id=int(rec["sample_id"]))


def stage_hf(case):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0").eval()
    ids = tok(case["full"], return_tensors="pt", add_special_tokens=False).to(model.device)
    print(f"  HF input tokens: {ids.input_ids.shape[1]} (expected {case['n_full']})")
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    hs = out.hidden_states                      # tuple: [embed, layer1..layer28]
    print(f"  HF hidden_states entries: {len(hs)}  (embeddings + {len(hs)-1} layers)")
    arr = np.stack([hs[l][0].float().cpu().numpy() for l in range(len(hs))])
    np.savez(OUT / "hf_ref.npz", acts=arr, positions=np.array(case["positions"]),
             input_ids=ids.input_ids[0].cpu().numpy())
    print(f"  saved hf_ref.npz  shape {arr.shape}  (n_hf_entries, n_pos, hidden)")


def stage_es(case):
    from vllm import LLM
    from easysteer.hidden_states import get_all_hidden_states_generate
    llm = LLM(model=MODEL, gpu_memory_utilization=0.60, enforce_eager=True,
              max_model_len=8192)
    # v0.17.1 has no positions/layers args: it returns every layer at every
    # token as (seq_len, hidden). We slice client-side.
    hs, outs = get_all_hidden_states_generate(
        llm, [case["full"]], max_tokens=1, split_by_samples=True)
    per_layer = hs[0]
    print(f"  EasySteer returned {len(per_layer)} layers; "
          f"layer0 tensor shape {tuple(per_layer[0].shape)}  "
          f"(expected seq_len {case['n_full']})")
    arr = np.stack([t.float().cpu().numpy() for t in per_layer])
    np.savez(OUT / "es_cap.npz", acts=arr, positions=np.array(case["positions"]))
    print(f"  saved es_cap.npz  shape {arr.shape}")


def stage_cmp():
    hf = np.load(OUT / "hf_ref.npz"); es = np.load(OUT / "es_cap.npz")
    A, B = hf["acts"], es["acts"]          # (n_entries, seq, hidden)
    print(f"  HF {A.shape}   EasySteer {B.shape}")

    def cos(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))

    # probe positions well inside both sequences
    probes = [int(x) for x in np.linspace(20, min(A.shape[1], B.shape[1]) - 20, 5)]
    print(f"  probe positions: {probes}")
    best = None
    for dl in range(-2, 3):               # HF entry = ES layer + dl
        for dp in range(-3, 4):           # HF pos   = ES pos   + dp
            sims = []
            for j in range(B.shape[0]):
                i = j + dl
                if not (0 <= i < A.shape[0]):
                    sims = []; break
                for q in probes:
                    if 0 <= q + dp < A.shape[1] and q < B.shape[1]:
                        sims.append(cos(A[i, q + dp], B[j, q]))
            if sims:
                m = float(np.mean(sims))
                if best is None or m > best[0]:
                    best = (m, dl, dp, float(np.min(sims)))
    m, dl, dp, mn = best
    print(f"\n  BEST ALIGNMENT: HF_entry = ES_layer + {dl},  HF_pos = ES_pos + {dp}")

    # The FINAL decoder layer is expected to disagree: vLLM exposes it before the
    # model's final RMSNorm, HF after. Both are valid; judge it separately rather
    # than letting it fail the gate for the other 27 layers.
    body, last = [], []
    for j in range(B.shape[0]):
        i = j + dl
        if not (0 <= i < A.shape[0]):
            continue
        c = [cos(A[i, q + dp], B[j, q]) for q in probes
             if 0 <= q + dp < A.shape[1] and q < B.shape[1]]
        (last if j == B.shape[0] - 1 else body).extend(c)
    print(f"    layers 0..{B.shape[0]-2}: min cosine {min(body):.6f}  mean {np.mean(body):.6f}")
    print(f"    final layer {B.shape[0]-1}: min cosine {min(last):.6f}  "
          f"(expected < 1: pre- vs post-final-RMSNorm)")
    ok = min(body) > 0.99
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'}")
    print(f"    decoder layer L (1-indexed)  ->  ES layer index L-1")
    print(f"    token position t             ->  ES position t+{-dp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["hf", "es", "cmp"])
    a = p.parse_args()
    if a.stage == "cmp":
        stage_cmp()
    else:
        case = build_case()
        (OUT / "case.json").write_text(json.dumps(
            {k: v for k, v in case.items() if k != "full"}, indent=2))
        print(f"  case: example {case['example_id']} cot {case['cot_id']}  "
              f"prompt_tokens={case['n_prompt']} full_tokens={case['n_full']}")
        print(f"  positions: {case['positions']}")
        (stage_hf if a.stage == "hf" else stage_es)(case)
