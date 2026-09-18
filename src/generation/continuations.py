#!/usr/bin/env python3
"""Goal 1 step 3: prefix-conditioned continuation sampling.

Reads a base-CoT run produced by ``run_eval.py`` (one dataset x model dir), cuts
each terminated CoT into checkpoints (``checkpoint.py``), and from every prefix
generates K continuations by feeding ``prompt + prefix_text`` to vLLM as a raw
string -- no chat-template re-application on the partial assistant turn.

Two append-only JSONL outputs, both resumable by count like ``run_eval.py``:

  checkpoints.jsonl      one row per (example_id, cot_id, checkpoint); the prefix
                         text + offsets.  Cheap to recompute but stored so label
                         code and the surface-form-leakage analysis need no model.
  continuations.jsonl    one row per (example_id, cot_id, checkpoint, cont_id);
                         the continuation text, parsed answer, correctness, and
                         remaining-reasoning length.

Keys everywhere: (dataset, model, example_id, cot_id, fraction, cont_id).
``cot_id`` is the base CoT's ``sample_id``; ``fraction`` identifies the checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Running this file as a script puts src/<package>/ on sys.path, not src/, so the
# cross-package imports below would not resolve.  Put src/ on the path explicitly;
# this also lets the module be imported as `<package>.<module>`.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from generation.checkpoint import DEFAULT_FRACTIONS, extract_cot, make_checkpoints
# Import run_eval before vllm: run_eval sets OUTLINES_CACHE_DIR/HF_HOME/etc via
# os.environ.setdefault at module load, and vllm imports outlines (which reads
# those cache-dir vars at its own import time) as a transitive dependency. If
# vllm/outlines import first, outlines silently falls back to its default
# ~/.cache location instead of the repo's writable cache dir.
from generation.run_eval import (
    DATASETS,
    _append_jsonl,
    _load_jsonl,
    grade,
    parse_output,
    render_freeform_prompt,
    render_mc_prompt,
)
from vllm import LLM, SamplingParams

THINK_CLOSE = "</think>"


def _remaining_reasoning_tokens(continuation: str, tokenizer) -> int:
    """Tokens the continuation spends on reasoning before ``</think>``.

    If the continuation never closes the think block it hit the cap; we return
    the full continuation length and flag it separately via ``terminated``.
    """
    span = continuation.split(THINK_CLOSE, 1)[0] if THINK_CLOSE in continuation else continuation
    return len(tokenizer.encode(span, add_special_tokens=False))


def _build_prompt(row: pd.Series, is_mc: bool, tokenizer) -> str:
    # examples.csv round-trips an empty choices cell (freeform datasets) as NaN
    # (float), not None -- and NaN is truthy in Python, so a plain truthiness check
    # would try json.loads(nan) and crash. isinstance() only accepts real strings.
    raw_choices = row.get("choices")
    choices = (json.loads(raw_choices)
               if isinstance(raw_choices, str) and raw_choices.strip() else None)
    return (render_mc_prompt(row["question"], choices, tokenizer) if is_mc
            else render_freeform_prompt(row["question"], tokenizer))


def run(args: argparse.Namespace) -> None:
    # Base CoTs live under base_dir/raw/ (see run_eval.py's layout note); this
    # stage's own outputs go in the same raw/ + config/ split under out_dir.
    base_dir = Path(args.base_dir)
    base_records = base_dir / "raw" / "sample_records.jsonl"
    examples_csv = base_dir / "raw" / "examples.csv"
    if not base_records.exists():
        raise FileNotFoundError(f"no base CoTs at {base_records}")
    if not examples_csv.exists():
        raise FileNotFoundError(f"no examples.csv at {examples_csv} (needed for choices)")

    dataset = args.dataset
    cfg = DATASETS[dataset]
    is_mc = cfg["is_mc"]

    out_dir = Path(args.output_dir) if args.output_dir else base_dir
    raw_dir, config_dir = out_dir / "raw", out_dir / "config"
    raw_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    # Sharded workers must never append to one file concurrently: each writes its
    # own shard and merge_shards.py concatenates them into the unsuffixed names
    # that labels.py / robustness.py / plot_goal1.py already expect.
    sfx = "" if args.num_shards == 1 else f".shard{args.shard_index:02d}"
    ckpt_path = raw_dir / f"checkpoints{sfx}.jsonl"
    cont_path = raw_dir / f"continuations{sfx}.jsonl"

    print(f"\n{'='*60}")
    print(f"  base run : {base_dir}")
    print(f"  dataset  : {dataset}   model: {args.model.split('/')[-1]}")
    print(f"  out dir  : {out_dir}")
    print(f"  K        : {args.continuations_per_prefix}   fractions: {args.fractions}")
    print(f"{'='*60}\n")

    examples = pd.read_csv(examples_csv).set_index("example_id")
    base = _load_jsonl(base_records)
    print(f"  {len(base)} base CoTs loaded.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
        # Every K continuations of a prefix share the whole (prompt+prefix), and
        # nested checkpoints of a CoT share the growing prefix -- prefix caching
        # turns that shared span into a KV-cache hit instead of a recompute.
        enable_prefix_caching=True,
        # The model advertises a 131072 context, which forces a huge KV cache and
        # blocks low gpu-memory-utilization.  A continuation's context is prompt +
        # prefix (up to ~all of an 8k CoT at the 90% checkpoint) + up to max_tokens
        # generated, so cap at a value that covers that worst case, not 131072.
        max_model_len=args.max_model_len,
        # Cap concurrent sequences to what the KV cache can hold: continuations
        # run to thousands of tokens, so admitting hundreds at once forces
        # preempt-to-CPU and exhausts swap.  max_num_seqs bounds the running set;
        # vLLM streams the rest via continuous batching.  swap_space is headroom.
        max_num_seqs=args.max_num_seqs,
        swap_space=args.swap_space,
    )

    # ── Resume: count continuations already saved per (eid, cot_id, fraction) ──
    done: dict[tuple[int, int, float], int] = {}
    if args.resume:
        # Scan EVERY shard file, not just our own. Re-splitting a shard's leftover
        # work across more GPUs gives the new workers different filenames, and if
        # they only read their own file they would regenerate everything already
        # done. Shards partition examples, so scanning all of them cannot
        # double-count. The merged continuations.jsonl is deliberately excluded --
        # it is a copy of the shards, and counting both would halve the true count.
        shard_files = sorted(raw_dir.glob("continuations.shard*.jsonl"))
        frames = [_load_jsonl(f) for f in shard_files]
        frames = [f for f in frames if not f.empty]
        prev = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(shard_files) > 1:
            print(f"  resume: scanning {len(shard_files)} shard files")
        if not prev.empty:
            g = prev.groupby(["example_id", "cot_id", "fraction"])["cont_id"].count()
            done = {(int(a), int(b), float(c)): int(v) for (a, b, c), v in g.items()}
            print(f"  Resuming: {len(prev)} continuations already saved.")
    elif cont_path.exists():
        raise FileExistsError(f"{cont_path} exists; pass --resume or delete out dir.")

    # Checkpoints are cheap to recompute; only append ones we have not written.
    ckpt_seen: set[tuple[int, int, float]] = set()
    if ckpt_path.exists():
        prevc = _load_jsonl(ckpt_path)
        if not prevc.empty:
            ckpt_seen = {(int(r.example_id), int(r.cot_id), float(r.fraction))
                         for r in prevc.itertuples()}

    (config_dir / "continuations_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    _subset_keep = {int(x) for x in str(args.subset_keep).split(",") if x.strip() != ""}
    if args.subset_mod > 1:
        print(f"  subset: keeping examples where id % {args.subset_mod} in "
              f"{sorted(_subset_keep)}")

    fractions = tuple(args.fractions)
    n_truncated = 0
    K = args.continuations_per_prefix

    # ── Pass 1: enumerate every prefix, write checkpoints, collect work items ──
    # A work item is one prefix that still needs continuations.  We generate all
    # its `remaining` continuations in a single request via SamplingParams(n=...),
    # so vLLM shares the prefix's KV cache across them.
    work: list[dict] = []
    for rec in base.itertuples():
        eid = int(rec.example_id)
        # Subset filter picks WHICH examples this phase covers (e.g. a 20% slice to
        # measure K-sufficiency on before committing a K for the rest); sharding
        # then splits that subset across GPUs. The two compose.
        if args.subset_mod > 1 and (eid % args.subset_mod) not in _subset_keep:
            continue
        # Shard by example_id so every checkpoint of every CoT of one example stays
        # on one worker -- nested prefixes of a CoT share a growing span, which
        # enable_prefix_caching turns into KV hits only if they land together.
        if args.num_shards > 1 and eid % args.num_shards != args.shard_index:
            continue
        cot_id = int(rec.sample_id)
        cot = extract_cot(rec.raw_completion)
        if cot is None:
            n_truncated += 1
            continue

        ex_row = examples.loc[eid]
        prompt = _build_prompt(ex_row, is_mc, tokenizer)
        n_prompt_tok = len(tokenizer.encode(prompt, add_special_tokens=False))
        _, gold_norm, _ = grade(cfg, None, rec.gold_answer)

        for ck in make_checkpoints(cot, tokenizer, fractions):
            key = (eid, cot_id, float(ck.fraction))
            if key not in ckpt_seen:
                _append_jsonl(ckpt_path, [dict(
                    dataset=dataset, model=args.model,
                    example_id=eid, cot_id=cot_id, fraction=ck.fraction,
                    sentence_idx=ck.sentence_idx, n_sentences=ck.n_sentences,
                    char_offset=ck.char_offset, token_offset=ck.token_offset,
                    prefix_text=ck.prefix_text,
                )])
                ckpt_seen.add(key)

            have = done.get(key, 0)
            remaining = K - have
            if remaining <= 0:
                continue
            work.append(dict(
                eid=eid, cot_id=cot_id, fraction=float(ck.fraction),
                prefix_text=ck.prefix_text, cont_prompt=prompt + ck.prefix_text,
                gold_answer=rec.gold_answer, gold_norm=gold_norm,
                have=have, remaining=remaining,
                # prompt + prefix, in tokens: sets this prefix's generation budget
                n_ctx_tok=n_prompt_tok + ck.token_offset,
            ))

    total_conts = sum(w["remaining"] for w in work)
    print(f"  {len(work)} prefixes need work, {total_conts} continuations to generate.")

    # ── Pass 2: generate in chunks of prefixes (write per chunk for resumability) ──
    # A single global max_tokens cannot serve every checkpoint: a 10% prefix is
    # short but needs the whole remaining CoT, while a 90% prefix of a 16k CoT
    # leaves only ~1.5k to write but eats most of the context window. Give each
    # prefix whatever the context window still allows, capped at args.max_tokens.
    # (AIME's longest usable CoT is 16,077 tokens; its 90% prefix + a flat 8,192
    # would overrun a 20,480 window.)
    MIN_BUDGET = 256

    def _sp(n: int, n_ctx_tok: int) -> SamplingParams:
        budget = min(args.max_tokens, args.max_model_len - n_ctx_tok - 64)
        return SamplingParams(temperature=args.temperature, top_p=args.top_p,
                              max_tokens=max(MIN_BUDGET, budget), n=n)

    # Split K into sub-requests. vLLM schedules SamplingParams(n=N) as ONE sequence
    # group and must hold all N resident: at K=100 that is 100*(prefix+max_tokens)
    # ~= 2.3x this box's 420k-token KV cache, so vLLM preempts to CPU swap and the
    # GPU thrashes at 0% util (K=10 needed only 0.23x, which is why the pilot was
    # fine). enable_prefix_caching keeps the shared prefix KV hot across the
    # sub-requests, so splitting costs nothing but makes every group schedulable.
    def _split(w: dict) -> list[tuple[dict, int, int]]:
        subs, off = [], 0
        while off < w["remaining"]:
            n = min(args.max_n_per_request, w["remaining"] - off)
            subs.append((w, n, off))
            off += n
        return subs

    pbatch = args.prefix_batch_size
    for start in tqdm(range(0, len(work), pbatch), desc="prefix batches"):
        chunk = work[start:start + pbatch]
        subreqs = [sub for w in chunk for sub in _split(w)]
        prompts = [w["cont_prompt"] for w, _, _ in subreqs]
        sps = [_sp(n, w["n_ctx_tok"]) for w, n, _ in subreqs]
        outputs = llm.generate(prompts, sps)

        records = []
        for (w, _n, off), out in zip(subreqs, outputs):
            for j, comp in enumerate(out.outputs):
                cont = comp.text
                parsed = parse_output(w["prefix_text"] + cont)
                pred_norm, _, is_correct = grade(cfg, parsed["final_answer"], w["gold_answer"])
                records.append(dict(
                    dataset=dataset, model=args.model,
                    example_id=w["eid"], cot_id=w["cot_id"], fraction=w["fraction"],
                    cont_id=w["have"] + off + j,
                    gold_answer=w["gold_answer"], gold_answer_norm=w["gold_norm"],
                    continuation_text=cont,
                    final_answer=parsed["final_answer"],
                    final_answer_norm=pred_norm,
                    is_correct=is_correct,
                    parse_success=parsed["parse_success"],
                    parse_error=parsed["parse_error"],
                    terminated=THINK_CLOSE in cont,
                    # what this prefix was actually allowed to generate: an
                    # unterminated continuation that hit its budget is a context
                    # limit, not the model failing to converge. Kept separate so
                    # the truncation metric stays honest.
                    max_tokens_budget=max(MIN_BUDGET, min(
                        args.max_tokens, args.max_model_len - w["n_ctx_tok"] - 64)),
                    n_ctx_tok=w["n_ctx_tok"],
                    remaining_reasoning_tokens=_remaining_reasoning_tokens(cont, tokenizer),
                    continuation_chars=len(cont),
                ))
        _append_jsonl(cont_path, records)

    print(f"\n{'─'*60}")
    print(f"  base CoTs excluded (no </think>): {n_truncated}/{len(base)}")
    print(f"  checkpoints file : {ckpt_path}")
    print(f"  continuations    : {cont_path}")
    print(f"{'─'*60}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", required=True,
                   help="run_eval output dir for one dataset x model "
                        "(contains sample_records.jsonl + examples.csv)")
    p.add_argument("--model", required=True, help="same HF model ID used for base CoTs")
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--output-dir", default=None,
                   help="defaults to --base-dir (writes alongside base CoTs)")
    p.add_argument("--continuations-per-prefix", type=int, default=10, help="K")
    p.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    p.add_argument("--prefix-batch-size", type=int, default=64,
                   help="prefixes submitted per llm.generate call; each expands "
                        "to up to K sequences. Records are flushed per batch.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-model-len", type=int, default=20480,
                   help="KV-cache context cap; must cover prompt + longest prefix "
                        "+ max-tokens. Default 20480 (~short prompt + full 8k CoT "
                        "prefix + 8k generated).")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=48,
                   help="max concurrent sequences in the running set; keep the "
                        "product with max-tokens within KV-cache capacity.")
    p.add_argument("--swap-space", type=int, default=16,
                   help="GB of CPU swap space for preemption headroom.")
    p.add_argument("--subset-mod", type=int, default=1,
                   help="keep only examples with example_id %% THIS in --subset-keep "
                        "(1 = no filter). Used to run a slice first.")
    p.add_argument("--subset-keep", default="0",
                   help="comma-separated residues to keep, paired with --subset-mod")
    p.add_argument("--max-n-per-request", type=int, default=20,
                   help="max continuations per vLLM request; K is split into "
                        "ceil(K/this) requests so no sequence group exceeds the "
                        "KV cache. Prefix caching keeps the shared prefix hot.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="split work across this many single-GPU workers")
    p.add_argument("--shard-index", type=int, default=0,
                   help="which shard this worker owns (0 <= i < num-shards)")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    args = p.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        p.error(f"--shard-index {args.shard_index} out of range for "
                f"--num-shards {args.num_shards}")
    run(args)


if __name__ == "__main__":
    main()
