#!/usr/bin/env python3
"""Dynamic GPU work queue -- the thing the static shard split should have been.

A fixed shard-per-GPU split leaves GPUs idle the moment they finish, while one
unlucky shard runs for hours. This hands the next task to whichever GPU is free,
so the tail collapses to roughly one task instead of one shard.

Tasks here are (num_shards, shard_index) pairs. num_shards=32 with index 0,8,16,24
covers exactly the examples of the old 8-way shard 0 (id%8==0), split four ways --
so leftover work from a crashed shard can be spread without regenerating anything
(continuations.py resume scans every shard file).

Single-box tool: it assumes every GPU on the machine is yours and finds free ones
by scanning /proc. Under SLURM use a job array instead (runs/gpqa25/), which lets
the scheduler play this role.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = os.environ.get("PY") or sys.executable


def gpu_busy(gpu: int) -> bool:
    """True if any continuations.py process already owns this GPU."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            if "continuations.py" not in cmd:
                continue
            env = Path(f"/proc/{pid}/environ").read_bytes().replace(b"\0", b"\n").decode()
            for line in env.splitlines():
                if line.startswith("CUDA_VISIBLE_DEVICES=") and line.split("=", 1)[1].strip() == str(gpu):
                    return True
        except Exception:
            continue
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--indices", required=True, help="comma-separated shard indices")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=24576)
    p.add_argument("--max-n", type=int, default=4)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--subset-mod", type=int, default=1)
    p.add_argument("--subset-keep", default="0")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    args = p.parse_args()

    tasks = [int(x) for x in args.indices.split(",") if x.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    logdir = REPO / "runs/goal1/full/logs"; logdir.mkdir(parents=True, exist_ok=True)
    running: dict[int, subprocess.Popen] = {}
    print(f"pool: {len(tasks)} tasks over GPUs {gpus}", flush=True)

    while tasks or running:
        for g in list(running):
            if running[g].poll() is not None:
                print(f"  GPU {g}: task finished (rc={running[g].returncode})", flush=True)
                del running[g]
        for g in gpus:
            if not tasks:
                break
            if g in running or gpu_busy(g):
                continue
            idx = tasks.pop(0)
            log = logdir / f"cont_{args.dataset}_pool_shard{idx:02d}.log"
            cmd = [PY, "src/generation/continuations.py",
                   "--base-dir", args.run_dir, "--model", args.model,
                   "--dataset", args.dataset,
                   "--continuations-per-prefix", str(args.k),
                   "--fractions", "0.1", "0.3", "0.5", "0.7", "0.9",
                   "--prefix-batch-size", "16",
                   "--temperature", "0.7", "--top-p", "0.95",
                   "--max-tokens", str(args.max_tokens),
                   "--max-model-len", str(args.max_model_len),
                   "--max-n-per-request", str(args.max_n),
                   "--max-num-seqs", str(args.max_num_seqs),
                   "--tensor-parallel-size", "1",
                   "--gpu-memory-utilization", "0.85", "--swap-space", "32",
                   "--subset-mod", str(args.subset_mod),
                   "--subset-keep", args.subset_keep,
                   "--num-shards", str(args.num_shards),
                   "--shard-index", str(idx), "--resume"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            running[g] = subprocess.Popen(cmd, cwd=REPO, env=env,
                                          stdout=log.open("w"), stderr=subprocess.STDOUT)
            print(f"  GPU {g}: launched shard {idx}/{args.num_shards} -> {log.name}",
                  flush=True)
        time.sleep(20)
    print("pool: all tasks complete", flush=True)


if __name__ == "__main__":
    main()
