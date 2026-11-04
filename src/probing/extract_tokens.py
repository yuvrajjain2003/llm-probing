#!/usr/bin/env python3
"""All-token activation capture, for weighted-pooling probes (round 7, item 7).

Sibling of ``extract_checkpoints.py``.  That script keeps ONE vector per
(example_id, cot_id, fraction) -- the last token of the prefix.  Attention
pooling (Reasoning Theater, arXiv 2603.05488) needs EVERY token, so this script
keeps the reasoning tokens of each CoT's longest prefix.

Why one block per CoT serves all five cuts: the model is causal, so the hidden
state at position t is the same whatever follows it.  The tokens of the 10% cut
are a prefix of the tokens of the 90% cut, in the same array, in order.  So a
row (example_id, cot_id, fraction) reads ``A[cot_tok_start : cut_tok_end]``.

Conventions are inherited from ``extract_checkpoints.py`` and ``verify_capture.py``:

    decoder layer L (1-indexed)  ->  EasySteer layer index L-1
    token position t             ->  EasySteer position t+1   (vLLM double-BOS)

Both are asserted at runtime.  ``build_cases`` is imported from
``extract_checkpoints`` unchanged, so prompt rendering and tokenisation are
identical to the published store by construction.

Output (per shard, resumable -- a finished shard is skipped):
  tokens.shardNN.npy      (n_tokens, n_layers, hidden) float16
  index.shardNN.parquet   one row per (example_id, cot_id, fraction):
                          example_id, cot_id, fraction, token_pos, n_prompt_tokens,
                          prefix_tokens, cot_tok_start, cut_tok_end, n_tok_cut, shard
  capture_config.shardNN.json

Usage
-----
    # capture (EasySteer env, 1 GPU)
    $PY_ES src/probing/extract_tokens.py --run-dir R --out-dir O \
        --layers 21 --stride 1 --num-shards 16 --shard-index 0 [--cot-max 5]

    # verify against the published last-token store (CPU, no GPU, no model)
    $PY src/probing/extract_tokens.py --verify --out-dir O --ckpt-dir C [--sample 400]
"""
from __future__ import annotations

import argparse, glob, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.extract_checkpoints import build_cases, MODEL, ES_POS_OFFSET, N_LAYERS  # noqa: E402

KEYS = ["example_id", "cot_id", "fraction"]


def plan(cases, layers, stride, cot_max):
    """Attach the capture positions to each case and count tokens.

    Positions run over the REASONING tokens of the longest prefix,
    ``n_prompt .. pos[longest]``, with the given stride.  Every checkpoint
    position is forced in even when the stride would skip it, so each cut's
    LAST token is always present and equals the published store's vector.
    """
    if cot_max is not None:
        cases = [c for c in cases if c["cot_id"] < cot_max]
    kept = []
    for c in cases:
        last = max(c["pos"])
        pos = sorted(set(range(c["n_prompt"], last + 1, stride)) | set(c["pos"]))
        if not pos:
            continue
        c["cap_pos"] = pos
        # index of each cut's last token inside this CoT's block (exclusive end)
        rank = {p: i for i, p in enumerate(pos)}
        c["cut_end"] = [rank[p] + 1 for p in c["pos"]]
        kept.append(c)
    n_tok = sum(len(c["cap_pos"]) for c in kept)
    return kept, n_tok


_LLM = {}


def get_llm(args):
    """One engine per process, so a multi-shard job loads the weights once."""
    if "llm" not in _LLM:
        from vllm import LLM
        _LLM["llm"] = LLM(model=MODEL, gpu_memory_utilization=args.gpu_memory_utilization,
                          enforce_eager=True, max_model_len=args.max_model_len,
                          enable_prefix_caching=False, enable_chunked_prefill=False)
    return _LLM["llm"]


def run(args, shard_index):
    from easysteer.hidden_states import get_all_hidden_states_generate

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f".shard{shard_index:02d}"
    tok_path = out_dir / f"tokens{sfx}.npy"
    idx_path = out_dir / f"index{sfx}.parquet"
    if tok_path.exists() and idx_path.exists() and not args.overwrite:
        print(f"  {tok_path.name} exists; nothing to do (use --overwrite)")
        return

    layers = [int(x) for x in args.layers.split(",")]
    assert all(1 <= L <= N_LAYERS for L in layers), f"layers must be 1..{N_LAYERS}"
    cases = build_cases(run_dir, args.num_shards, shard_index)
    cases, n_tok = plan(cases, layers, args.stride, args.cot_max)
    print(f"  shard {shard_index}/{args.num_shards}: {len(cases)} CoTs, "
          f"{n_tok:,} tokens, layers {layers}, stride {args.stride}")
    if not cases:
        # still write empty artefacts so the shard counts as done
        np.save(tok_path, np.zeros((0, len(layers), 1), dtype=np.float16))
        pd.DataFrame(columns=KEYS).to_parquet(idx_path, index=False)
        return

    # Prefix caching and chunked prefill MUST stay off: both silently truncate the
    # returned hidden-state sequence (see extract_checkpoints.py).
    llm = get_llm(args)

    A = None
    rows, cur, t0 = [], 0, time.time()
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

            epos = np.asarray(c["cap_pos"]) + ES_POS_OFFSET   # -> EasySteer coords
            assert epos.max() < seq, (c["example_id"], c["cot_id"], epos.max(), seq)
            sel = np.stack([per_layer[L - 1][epos, :].float().cpu().numpy()
                            for L in layers], axis=1)          # (n_tok, n_layers, hidden)
            if A is None:
                A = np.lib.format.open_memmap(
                    tok_path.with_suffix(".npy.part"), mode="w+", dtype=np.float16,
                    shape=(n_tok, len(layers), sel.shape[2]))
            A[cur:cur + sel.shape[0]] = sel.astype(np.float16)

            for f, tp, pt, ce in zip(c["fracs"], c["pos"], c["prefix_tokens"], c["cut_end"]):
                rows.append(dict(example_id=c["example_id"], cot_id=c["cot_id"],
                                 fraction=float(f), token_pos=int(tp),
                                 n_prompt_tokens=c["n_prompt"], prefix_tokens=int(pt),
                                 cot_tok_start=cur, cut_tok_end=cur + int(ce),
                                 n_tok_cut=int(ce), shard=shard_index))
            cur += sel.shape[0]

        done = bstart + len(batch)
        el = time.time() - t0
        print(f"    {done}/{len(cases)} CoTs  {cur:,}/{n_tok:,} tokens  {el/60:.1f} min  "
              f"eta {el/done*(len(cases)-done)/60:.1f} min", flush=True)

    assert cur == n_tok, f"wrote {cur} tokens, planned {n_tok}"
    A.flush(); del A
    tok_path.with_suffix(".npy.part").rename(tok_path)
    idx = pd.DataFrame(rows)
    idx.to_parquet(idx_path, index=False)
    print(f"  wrote {tok_path.name} ({n_tok:,}, {len(layers)}, ...) "
          f"and {idx_path.name} ({len(idx):,} rows)")

    cfg = dict(model=MODEL, run_dir=str(run_dir), layers=layers, stride=args.stride,
               cot_max=args.cot_max, es_pos_offset=ES_POS_OFFSET, dtype="float16",
               positions="reasoning tokens n_prompt..pos[longest cut], stride applied, "
                         "every checkpoint position forced in",
               unit="one row per token; index gives each (example_id, cot_id, fraction) "
                    "the range [cot_tok_start, cut_tok_end)",
               num_shards=args.num_shards, shard_index=shard_index,
               max_model_len=args.max_model_len, n_tokens=int(n_tok))
    (out_dir / f"capture_config{sfx}.json").write_text(json.dumps(cfg, indent=2))


# -- the correctness gate ------------------------------------------------------

def verify(args):
    """The last token of each cut's range must equal the PUBLISHED last-token vector.

    This is the whole guarantee that the new store holds the same quantity as
    ``activations_ckpt``, on the same coordinates.  Requires no GPU and no model.
    """
    out_dir, ckpt_dir = Path(args.out_dir), Path(args.ckpt_dir)
    cfg = json.loads(sorted(out_dir.glob("capture_config.shard*.json"))[0].read_text())
    layers = cfg["layers"]

    NEW = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("index.shard*.parquet"))],
                    ignore_index=True)
    PUB = []
    for ip in sorted(glob.glob(str(ckpt_dir / "index.shard*.parquet"))):
        i = pd.read_parquet(ip).sort_values("row_idx")
        i["_shard"] = ip.split("index.")[1].replace(".parquet", "")
        i["_row"] = i.row_idx
        PUB.append(i[KEYS + ["_shard", "_row"]])
    PUB = pd.concat(PUB, ignore_index=True).drop_duplicates(KEYS)
    for d in (NEW, PUB):
        d["fraction"] = d["fraction"].round(6)
    m = NEW.merge(PUB, on=KEYS, how="left")
    missing = int(m._row.isna().sum())
    print(f"  new store rows: {len(NEW):,}   not found in published store: {missing}")
    if missing:
        raise SystemExit("FAIL: rows in the new store are absent from the published store")

    rng = np.random.default_rng(0)
    samp = m.iloc[rng.permutation(len(m))[:args.sample]].copy()
    cos, maxabs, nbit = [], [], 0
    cache_new, cache_pub = {}, {}
    for _, r in samp.iterrows():
        sh = int(r["shard"])
        if sh not in cache_new:
            cache_new[sh] = np.load(out_dir / f"tokens.shard{sh:02d}.npy", mmap_mode="r")
        if r["_shard"] not in cache_pub:
            cache_pub[r["_shard"]] = np.load(ckpt_dir / f"activations.{r['_shard']}.npy",
                                             mmap_mode="r")
        a = np.asarray(cache_new[sh][int(r["cut_tok_end"]) - 1], dtype=np.float32)   # (n_layers, h)
        b = np.asarray(cache_pub[r["_shard"]][int(r["_row"])], dtype=np.float32)     # (28, h)
        for j, L in enumerate(layers):
            u, v = a[j], b[L - 1]
            cos.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30)))
            maxabs.append(float(np.abs(u - v).max()))
            nbit += int(np.array_equal(cache_new[sh][int(r['cut_tok_end']) - 1][j],
                                       cache_pub[r['_shard']][int(r['_row'])][L - 1]))
    cos, maxabs = np.array(cos), np.array(maxabs)
    print(f"  sampled {len(samp)} rows x {len(layers)} layer(s) = {len(cos)} comparisons")
    print(f"  cosine   min {cos.min():.6f}  mean {cos.mean():.6f}")
    print(f"  max |diff| over all comparisons: {maxabs.max():.3e}")
    print(f"  bitwise identical: {nbit}/{len(cos)}")
    # token-count sanity: prefix_tokens is the published count of reasoning tokens
    if cfg["stride"] == 1:
        bad = int((NEW.n_tok_cut != NEW.prefix_tokens).sum())
        print(f"  stride 1: n_tok_cut != prefix_tokens on {bad} rows")
    ok = cos.min() > 0.9999
    print("  VERDICT:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit("FAIL: new store does not reproduce the published last-token vectors")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layers", default="21", help="comma-separated 1-indexed decoder layers")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--cot-max", type=int, default=None, help="keep only cot_id < N (pilot)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", default="0",
                   help="comma-separated shard indices to do in THIS process")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    p.add_argument("--max-model-len", type=int, default=24576)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--ckpt-dir", default=None, help="published activations_ckpt dir, for --verify")
    p.add_argument("--sample", type=int, default=400)
    a = p.parse_args()
    if a.verify:
        if not a.ckpt_dir:
            p.error("--verify needs --ckpt-dir")
        verify(a)
        return
    shards = [int(x) for x in str(a.shard_index).split(",")]
    if not all(0 <= s < a.num_shards for s in shards):
        p.error("shard-index out of range")
    for s in shards:
        run(a, s)


if __name__ == "__main__":
    main()
