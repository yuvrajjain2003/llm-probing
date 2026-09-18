#!/usr/bin/env python3
"""Goal 2 step 1: capture hidden states over the think phase of GPQA base CoTs.

We are NOT generating anything. Goal 1 already produced every CoT; this re-reads
that text with max_tokens=1 (prompt-only forward) and records what the model's
internals look like at each point. Goal 1's continuation statistics become the
LABELS for these activations, joined later by (example_id, cot_id).

Conventions verified by src/verify_capture.py against stock HF transformers
(min cosine 0.9969 over layers 0..26):

    decoder layer L (1-indexed)  ->  EasySteer layer index L-1
    token position t             ->  EasySteer position t+1

The +1 is because vLLM prepends a BOS on top of the one the chat template already
carries. Getting either convention wrong yields plausible-looking garbage, so both
are asserted at runtime, not assumed.

Note: the final decoder layer (28) is exposed by vLLM BEFORE the model's final
RMSNorm, unlike HF which applies it. Documented in run_config.json.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

# src/ on the path, so `generation.*` resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
ES_POS_OFFSET = 1          # verified
DECODER_LAYERS = (14, 21, 28)   # 1-indexed; ES indices are these minus 1


def build_cases(run_dir: Path, num_shards: int, shard_index: int):
    """One case per usable base CoT, using Goal 1's own prompt/CoT helpers."""
    from transformers import AutoTokenizer
    from generation.run_eval import render_mc_prompt
    from generation.checkpoint import extract_cot

    tok = AutoTokenizer.from_pretrained(MODEL)
    ex = pd.read_csv(run_dir / "raw" / "examples.csv").set_index("example_id")
    recs = [json.loads(l) for l in open(run_dir / "raw" / "sample_records.jsonl")]

    cases = []
    for r in recs:
        eid = int(r["example_id"])
        if num_shards > 1 and eid % num_shards != shard_index:
            continue
        think = extract_cot(r["raw_completion"])
        if think is None:                      # never closed </think>; unusable
            continue
        row = ex.loc[eid]
        choices = json.loads(row["choices"]) if isinstance(row["choices"], str) else None
        prompt = render_mc_prompt(row["question"], choices, tok)
        n_prompt = len(tok.encode(prompt, add_special_tokens=False))
        n_full = len(tok.encode(prompt + think, add_special_tokens=False))
        if n_full <= n_prompt:
            continue
        cases.append(dict(example_id=eid, cot_id=int(r["sample_id"]),
                          text=prompt + think, n_prompt=n_prompt, n_full=n_full))
    return cases


def run(args):
    from vllm import LLM
    from easysteer.hidden_states import get_all_hidden_states_generate

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f".shard{args.shard_index:02d}"
    act_path = out_dir / f"activations{sfx}.npy"
    idx_path = out_dir / f"index{sfx}.parquet"
    if act_path.exists() and idx_path.exists() and not args.overwrite:
        print(f"  {act_path.name} exists; nothing to do (use --overwrite)")
        return

    cases = build_cases(run_dir, args.num_shards, args.shard_index)
    print(f"  shard {args.shard_index}/{args.num_shards}: {len(cases)} CoTs")
    if not cases:
        return

    es_layers = [L - 1 for L in DECODER_LAYERS]
    # Prefix caching MUST be off. CoTs of the same question share an identical
    # prompt prefix; with caching on, vLLM serves those positions from cache and
    # never recomputes them, so no hidden states are captured for them -- the
    # returned sequence comes back short. Chunked prefill is off for the same
    # reason (a split prefill returns only part of the sequence).
    llm = LLM(model=MODEL, gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=True, max_model_len=args.max_model_len,
              enable_prefix_caching=False, enable_chunked_prefill=False)

    acts, rows, t0 = [], [], time.time()
    for bstart in range(0, len(cases), args.batch_size):
        batch = cases[bstart:bstart + args.batch_size]
        hs, _ = get_all_hidden_states_generate(
            llm, [c["text"] for c in batch], max_tokens=1, split_by_samples=True)

        for c, per_layer in zip(batch, hs):
            seq = per_layer[0].shape[0]
            # assert the two conventions rather than trusting them
            assert len(per_layer) == 28, f"expected 28 layers, got {len(per_layer)}"
            assert seq == c["n_full"] + ES_POS_OFFSET, (
                f"seq {seq} != n_full {c['n_full']} + {ES_POS_OFFSET}")

            # think-phase token positions, every Nth, in TOKEN coords
            tpos = np.arange(c["n_prompt"], c["n_full"], args.stride)
            epos = tpos + ES_POS_OFFSET                      # -> EasySteer coords
            sel = np.stack([per_layer[L][epos, :].float().cpu().numpy()
                            for L in es_layers], axis=1)     # (n_pos, 3, hidden)
            acts.append(sel.astype(np.float16))

            n_think = c["n_full"] - c["n_prompt"]
            for k, t in enumerate(tpos):
                rows.append(dict(example_id=c["example_id"], cot_id=c["cot_id"],
                                 token_index=int(t), n_prompt_tokens=c["n_prompt"],
                                 n_think_tokens=int(n_think),
                                 frac_through_think=float((t - c["n_prompt"]) / n_think)))
        done = bstart + len(batch)
        el = time.time() - t0
        print(f"    {done}/{len(cases)} CoTs  {sum(a.shape[0] for a in acts):,} positions  "
              f"{el/60:.1f} min  eta {el/done*(len(cases)-done)/60:.1f} min", flush=True)

    arr = np.concatenate(acts, axis=0)
    np.save(act_path, arr)
    idx = pd.DataFrame(rows); idx["row_idx"] = np.arange(len(idx))
    idx.to_parquet(idx_path, index=False)
    assert len(idx) == arr.shape[0], "index/activation row mismatch"
    print(f"  wrote {act_path.name} {arr.shape} {arr.nbytes/2**30:.2f} GB "
          f"and {idx_path.name} ({len(idx):,} rows)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--out-dir", default="runs/goal2/activations/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--stride", type=int, default=4, help="capture every Nth think token")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    p.add_argument("--max-model-len", type=int, default=24576)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if not 0 <= a.shard_index < a.num_shards:
        p.error("shard-index out of range")
    run(a)


if __name__ == "__main__":
    main()
