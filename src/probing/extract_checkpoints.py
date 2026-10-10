#!/usr/bin/env python3
"""Goal 2 capture, checkpoint-exact and all-layer.

Replaces ``extract_activations.py`` for the Goal-1-label probing question.  Two
differences, both of which were handicapping the activation side of the
comparison in ``probe_goal1_labels.py``:

  EXACT POSITION.  The old store captured every 4th think token, so the probe had
  to settle for the nearest captured token at or before each checkpoint -- up to
  3 tokens early, while the TF-IDF text baseline it was being compared against
  got the exact prefix.  Here the captured position IS the checkpoint token.

  EVERY LAYER.  The old store kept decoder layers 14/21/28 out of 28.  EasySteer
  already returns all 28 on every call, so the other 25 were computed and thrown
  away.  Keeping them is free at capture time.

Why one forward pass per CoT is enough, rather than one per checkpoint: the model
is causal, so the hidden states at positions 0..t are identical whether the input
was ``prompt + prefix`` (what ``continuations.py`` actually fed when sampling the
K continuations) or ``prompt + longer_prefix``.  We therefore feed each CoT's
LONGEST checkpoint prefix once and read all 5 checkpoint positions out of it.

And why one vector per checkpoint, not per continuation: for the same reason, the
activation at a checkpoint token does not depend on which continuation was drawn
from it.  All K continuations of a prefix would yield bit-identical vectors.  The
K continuations supply the LABEL; the prefix supplies the FEATURE.  Output rows
are keyed (example_id, cot_id, fraction) -- 1:1 with ``checkpoint_labels.csv``.

Conventions are inherited from ``extract_activations.py`` and were verified by
``verify_capture.py`` against stock HF transformers (min cosine 0.9969):

    decoder layer L (1-indexed)  ->  EasySteer layer index L-1
    token position t             ->  EasySteer position t+1   (vLLM double-BOS)

Both are asserted at runtime.  Note the final decoder layer (28) is exposed by
vLLM BEFORE the model's final RMSNorm, unlike HF, which applies it.

Output (per shard, resumable -- a finished shard is skipped):
  activations.shardNN.npy    (n_checkpoints, 28, hidden) float16
  index.shardNN.parquet      example_id, cot_id, fraction, token_pos, ...

Usage
-----
    python src/probing/extract_checkpoints.py --num-shards 8 --shard-index 0
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# src/ on the path, so `generation.*` resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
ES_POS_OFFSET = 1           # verified by verify_capture.py
N_LAYERS = 28               # decoder layers 1..28 -> ES indices 0..27


def build_cases(run_dir: Path, num_shards: int, shard_index: int):
    """One case per CoT: the longest checkpoint prefix + every checkpoint position.

    Positions are computed by tokenizing ``prompt + prefix_text`` directly rather
    than as ``n_prompt + token_offset``.  The two agree on this run (checked over
    300 sampled checkpoints), but only the direct form is exact by construction --
    tokenization is not additive across the prompt/prefix boundary in general.
    """
    from transformers import AutoTokenizer
    from generation.run_eval import render_mc_prompt

    tok = AutoTokenizer.from_pretrained(MODEL)
    ex = pd.read_csv(run_dir / "raw" / "examples.csv").set_index("example_id")

    # Checkpoints are append-only and shards overlap on resume, so dedup.
    by_cot: dict[tuple[int, int], dict[float, str]] = defaultdict(dict)
    with (run_dir / "raw" / "checkpoints.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            eid = int(r["example_id"])
            if num_shards > 1 and eid % num_shards != shard_index:
                continue
            by_cot[(eid, int(r["cot_id"]))].setdefault(float(r["fraction"]),
                                                       r["prefix_text"])

    prompt_cache: dict[int, tuple[str, int]] = {}
    cases = []
    for (eid, cid), per_frac in sorted(by_cot.items()):
        if eid not in prompt_cache:
            row = ex.loc[eid]
            choices = (json.loads(row["choices"])
                       if isinstance(row["choices"], str) else None)
            p = render_mc_prompt(row["question"], choices, tok)
            prompt_cache[eid] = (p, len(tok.encode(p, add_special_tokens=False)))
        prompt, n_prompt = prompt_cache[eid]

        # Exact last-token index of each prefix within prompt+prefix.
        pos = {f: len(tok.encode(prompt + txt, add_special_tokens=False)) - 1
               for f, txt in per_frac.items()}
        fracs = sorted(per_frac)
        longest = max(fracs, key=lambda f: pos[f])
        text = prompt + per_frac[longest]
        n_text = pos[longest] + 1
        # Positions must be inside the sequence we are about to feed; an empty
        # prefix (fraction 0) sits on the last prompt token, n_prompt - 1.
        assert all(n_prompt - 1 <= pos[f] < n_text for f in fracs), (eid, cid)

        cases.append(dict(example_id=eid, cot_id=cid, text=text,
                          n_prompt=n_prompt, n_text=n_text,
                          fracs=fracs, pos=[pos[f] for f in fracs],
                          prefix_tokens=[pos[f] - n_prompt + 1 for f in fracs]))
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
    n_ckpt = sum(len(c["fracs"]) for c in cases)
    print(f"  shard {args.shard_index}/{args.num_shards}: "
          f"{len(cases)} CoTs, {n_ckpt} checkpoints")
    if not cases:
        return

    # Prefix caching MUST be off: CoTs of one question share a prompt prefix, and
    # cached positions are never recomputed, so no hidden states come back for
    # them and the returned sequence is short.  Chunked prefill likewise returns
    # only part of the sequence.  Either one silently truncates the capture.
    llm = LLM(model=MODEL, gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=True, max_model_len=args.max_model_len,
              enable_prefix_caching=False, enable_chunked_prefill=False)

    acts, rows, t0 = [], [], time.time()
    for bstart in range(0, len(cases), args.batch_size):
        batch = cases[bstart:bstart + args.batch_size]
        hs, _ = get_all_hidden_states_generate(
            llm, [c["text"] for c in batch], max_tokens=1, split_by_samples=True)

        for c, per_layer in zip(batch, hs):
            assert len(per_layer) == N_LAYERS, \
                f"expected {N_LAYERS} layers, got {len(per_layer)}"
            seq = per_layer[0].shape[0]
            assert seq == c["n_text"] + ES_POS_OFFSET, \
                f"seq {seq} != n_text {c['n_text']} + {ES_POS_OFFSET}"

            epos = np.asarray(c["pos"]) + ES_POS_OFFSET      # -> EasySteer coords
            # (n_ckpt, n_layers, hidden)
            sel = np.stack([per_layer[L][epos, :].float().cpu().numpy()
                            for L in range(N_LAYERS)], axis=1)
            acts.append(sel.astype(np.float16))

            for f, tp, pt in zip(c["fracs"], c["pos"], c["prefix_tokens"]):
                rows.append(dict(example_id=c["example_id"], cot_id=c["cot_id"],
                                 fraction=float(f), token_pos=int(tp),
                                 n_prompt_tokens=c["n_prompt"],
                                 prefix_tokens=int(pt)))
        done = bstart + len(batch)
        el = time.time() - t0
        print(f"    {done}/{len(cases)} CoTs  {sum(a.shape[0] for a in acts):,} "
              f"checkpoints  {el/60:.1f} min  "
              f"eta {el/done*(len(cases)-done)/60:.1f} min", flush=True)

    arr = np.concatenate(acts, axis=0)
    idx = pd.DataFrame(rows); idx["row_idx"] = np.arange(len(idx))
    assert len(idx) == arr.shape[0], "index/activation row mismatch"
    np.save(act_path, arr)
    idx.to_parquet(idx_path, index=False)
    print(f"  wrote {act_path.name} {arr.shape} {arr.nbytes/2**30:.2f} GB "
          f"and {idx_path.name} ({len(idx):,} rows)")

    cfg = dict(model=MODEL, run_dir=str(run_dir), layers="all 1..28 (1-indexed)",
               es_pos_offset=ES_POS_OFFSET, dtype="float16",
               positions="exact checkpoint token (last token of prefix)",
               unit="one vector per (example_id, cot_id, fraction)",
               final_layer_note="decoder layer 28 is pre-RMSNorm in vLLM",
               num_shards=args.num_shards, shard_index=args.shard_index,
               max_model_len=args.max_model_len)
    (out_dir / f"capture_config{sfx}.json").write_text(json.dumps(cfg, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--out-dir", default="runs/goal2/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
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
