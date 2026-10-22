#!/usr/bin/env python3
"""Sentence-embed every checkpoint prefix (baseline 1: does the TEXT carry what the activations do?).

Text embedded = question + choices + prefix_text, i.e. what the probed model had read at the
checkpoint (the activation probe sees the same). Also embeds the 0% rows (question only;
cot_id -1) from --f0-input. Over-long inputs keep their END (truncation_side='left') and the
count is logged.

Output (sweep.py `npy` contract):  <out>/<name>.npy  (m, d) float32, L2-normalised
                                   <out>/<name>.keys.csv  example_id, cot_id, fraction
Resumable: finished chunks are kept in <out>/<name>.chunks/ and skipped on rerun.

    $PY src/probing/embed_prefixes.py --model Qwen/Qwen3-Embedding-8B --name qwen3_emb_8b

"Other LLM" control (--layer L): a plain LLM's hidden state after decoder layer L (1-indexed,
same convention as our activations: HF hidden_states[L]) at the last token, NOT normalised,
with --no-instruct. E.g. --model Qwen/Qwen2.5-Math-7B --layer 21 --no-instruct --max-len 4096
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

INSTRUCT = "Instruct: Retrieve semantically similar text\nQuery: "   # fixed before seeing results


def load_rows(run_dir: Path, f0_input: Path | None, instruct: str = INSTRUCT) -> pd.DataFrame:
    ex = pd.read_csv(run_dir / "raw/examples.csv").set_index("example_id")
    cps = pd.read_json(run_dir / "raw/checkpoints.jsonl", lines=True)
    if f0_input is not None:
        cps = pd.concat([cps, pd.read_json(f0_input / "raw/checkpoints.jsonl", lines=True)])
    cps = cps[["example_id", "cot_id", "fraction", "prefix_text"]].reset_index(drop=True)
    assert not cps.duplicated(["example_id", "cot_id", "fraction"]).any()

    def header(eid):
        q = ex.loc[eid, "question"].strip()
        ch = ex.loc[eid, "choices"]
        if isinstance(ch, str):
            q += "\n" + "\n".join(json.loads(ch))     # already labelled "A) ..."
        return q

    cps["text"] = [instruct + header(e) + "\n\n" + (p or "")
                   for e, p in zip(cps.example_id, cps.prefix_text)]
    return cps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--run-dir", default="runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--f0-input", default="runs/goal2/gpqa25/_f0_input")
    p.add_argument("--out-dir", default="runs/goal2/gpqa25/embeddings")
    p.add_argument("--max-len", type=int, default=32768)
    p.add_argument("--tokens-per-batch", type=int, default=65536)
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument("--limit", type=int, default=0, help="only first N rows (testing)")
    p.add_argument("--layer", type=int, help="LLM control: hidden state after this decoder layer")
    p.add_argument("--no-instruct", action="store_true")
    p.add_argument("--pooling", choices=["last", "cls"], default="last",
                   help="cls: first-token pooling for BERT-style encoders (right padding)")
    a = p.parse_args()

    rows = load_rows(Path(a.run_dir), Path(a.f0_input) if a.f0_input else None,
                     "" if a.no_instruct else INSTRUCT)
    if a.limit:
        rows = rows.iloc[: a.limit]
    out = Path(a.out_dir); cdir = out / f"{a.name}.chunks"; cdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model, truncation_side="left",
                                        padding_side="right" if a.pooling == "cls" else "left")
    if tok.pad_token is None:                       # e.g. Mistral; pads are masked, so unused
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                      trust_remote_code=True).cuda().eval()   # remote code: UltraBERT
    lens = np.array([len(t) for t in tok(rows.text.tolist(), add_special_tokens=True)["input_ids"]])
    print(f"{len(rows)} texts, tokens median {np.median(lens):.0f} max {lens.max()}, "
          f"truncated (> {a.max_len}): {(lens > a.max_len).sum()}", flush=True)

    order = np.argsort(-lens)                      # longest first: OOM shows up immediately
    t0 = time.time()
    for c0 in range(0, len(order), a.chunk):
        f = cdir / f"{c0:06d}.npy"
        if f.exists():
            continue
        idx = order[c0: c0 + a.chunk]
        embs, i = [], 0
        while i < len(idx):                         # length-sorted, token-budgeted batches
            L = min(lens[idx[i]], a.max_len)
            bs = max(1, a.tokens_per_batch // L)
            b = idx[i: i + bs]; i += bs
            enc = tok(rows.text.iloc[b].tolist(), padding=True, truncation=True,
                      max_length=a.max_len, return_tensors="pt").to("cuda")
            with torch.no_grad():
                if a.pooling == "cls":
                    h = F.normalize(model(**enc).last_hidden_state[:, 0].float(), dim=-1)   # CLS (right padded)
                elif a.layer is None:
                    h = F.normalize(model(**enc).last_hidden_state[:, -1].float(), dim=-1)  # last token (left padded)
                else:
                    h = model(**enc, output_hidden_states=True).hidden_states[a.layer][:, -1].float()
            embs.append(h.cpu().numpy())
        np.save(f, np.concatenate(embs))
        print(f"  chunk {c0}: {c0 + len(idx)}/{len(order)} done, {(time.time()-t0)/60:.1f} min", flush=True)

    parts = [np.load(cdir / f"{c0:06d}.npy") for c0 in range(0, len(order), a.chunk)]
    X = np.zeros((len(rows), parts[0].shape[1]), dtype=np.float32)
    X[order] = np.concatenate(parts)
    assert np.isfinite(X).all()
    np.save(out / f"{a.name}.npy", X)
    rows[["example_id", "cot_id", "fraction"]].to_csv(out / f"{a.name}.keys.csv", index=False)
    print(f"wrote {out / a.name}.npy {X.shape}", flush=True)


if __name__ == "__main__":
    main()
