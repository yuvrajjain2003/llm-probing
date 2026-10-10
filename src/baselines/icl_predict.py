#!/usr/bin/env python3
"""R1: in-context-learning baseline. An LLM sees k labelled examples (question + choices + CoT prefix
-> 5 targets) and predicts the targets of a new prefix. Same rows, folds and targets as sweep.py.

    $PY_ICL src/baselines/icl_predict.py plan  --model gemma4_31b            # CPU: token counts, max k, cost
    $PY_ICL src/baselines/icl_predict.py dry   --model gemma4_31b --k 10     # CPU: build prompts, print one
    $PY_ICL src/baselines/icl_predict.py run   --model gemma4_31b --k 10     # GPU: vLLM, resumable per fold
    $PY_ICL src/baselines/icl_predict.py score --model gemma4_31b --k 10     # CPU: parse + R2/acc + calibrated

Design: for outer fold f the examples come ONLY from questions with fold != f (sweep.fixed_fold_of), one
fixed ordered list per fold (seeded), balanced over the 6 checkpoints (0,10,..90%) by round robin; k uses its
first k items (nested). Block order = list order. Everything fold-specific comes FIRST in the prompt, the test
prefix LAST, so vLLM prefix caching computes the example block once per fold.
Thinking: default OFF (--thinking off): Qwen3.5/Gemma via chat-template enable_thinking=False, our R1 model by
prefilling "</think>\n\n". --thinking on: pass 1 thinks until its end-of-thinking token or --think-cap, then pass 2
appends the end tag and forces the JSON answer (vLLM structured outputs). Answers are always forced JSON.
"""
from __future__ import annotations

import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.embed_prefixes import load_rows            # question + choices + prefix text
from probing.sweep import fixed_fold_of

RUN_DIR = Path("runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
F0_INPUT = Path("runs/goal2/gpqa25/_f0_input")
OUT = Path("runs/goal2/gpqa25/icl")
KEYS = ["example_id", "cot_id", "fraction"]
TARGETS = ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std", "committed_by_here"]
JSON_KEYS = ["answer_entropy_bits", "p_mode", "rem_len_mean", "rem_len_std", "committed"]   # same order
FRACS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.0]                  # round-robin order of the example list
MODELS = {   # ctx = max_model_len used; tp = GPUs
    "r1qwen7b":   dict(id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", ctx=131072, tp=1, end="</think>"),
    "qwen35_9b":  dict(id="Qwen/Qwen3.5-9B", ctx=262144, tp=1, end="</think>"),
    "gemma4_31b": dict(id="google/gemma-4-31B-it", ctx=262144, tp=2, end="<channel|>"),
}
YARN_CTX = 1010000   # Qwen3.5 with --yarn (README: factor 4.0, original 262144)
YARN = {"text_config": {"rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
        "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
        "factor": 4.0, "original_max_position_embeddings": 262144}}}
ANSWER_BUDGET = 128

INSTR = """You predict how a reasoning model's answer to a multiple-choice question will develop.
A small reasoning model was given a GPQA-Diamond question and began writing a step-by-step reasoning chain. Each item shows the question with its answer choices, then the part of the chain written so far (the "prefix"; empty at checkpoint 0%), and the checkpoint (how far into the full chain the prefix ends). From the end of the prefix the model was continued 25 times independently until it finished. Five numbers were measured over those 25 continuations:
- answer_entropy_bits: entropy in bits (Miller-Madow corrected) of the distribution of final answers; 0 means all 25 agree.
- p_mode: fraction of the 25 continuations that give the most common final answer.
- rem_len_mean, rem_len_std: mean and standard deviation, over the 25 continuations, of the number of further reasoning tokens written before the model finishes thinking.
- committed: 1 if, at this checkpoint and every later one (10%, 30%, 50%, 70%, 90%), the most common answer is the same and at least 80% of continuations give it; otherwise 0 (always 0 at checkpoint 0%).
"""
INSTR_EX = "Labelled examples follow, then one query item. "
INSTR_END = ("Reply with one JSON object and nothing else, with exactly these keys and numeric values: "
             + ", ".join(JSON_KEYS) + ".")
INSTR_QUERY = ("Predict the measured values for the QUERY item above. Do not solve the question itself. "
               + INSTR_END)   # repeated after the query: long prompts make models lose the task (smoke test 2026-10-09)
NUM = r"-?[0-9]{1,6}(\.[0-9]{1,4})?"
JSON_RE = r"\{" + ", ".join(f'"{k}": {NUM}' for k in JSON_KEYS) + r"\}"   # forced compact JSON answer (vLLM
# structured outputs); a regex, not a JSON schema: with a schema our R1 model ran away in whitespace/long digits


# ---------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    rows = load_rows(RUN_DIR, F0_INPUT, "")
    rows["fraction"] = rows.fraction.round(6)
    lab = pd.concat([pd.read_csv(RUN_DIR / "labels/checkpoint_labels.csv"),
                     pd.read_csv(RUN_DIR / "labels/checkpoint_labels_f0.csv")])
    lab["fraction"] = lab.fraction.round(6)
    cm = pd.read_csv(RUN_DIR / "labels/commitment.csv")
    d = rows.merge(lab, on=KEYS).merge(cm[["example_id", "cot_id", "commitment_fraction"]],
                                       on=["example_id", "cot_id"], how="left")
    d["committed_by_here"] = (d.commitment_fraction.notna() & (d.fraction >= d.commitment_fraction)).astype(float)
    d["label_commit"] = d.committed_by_here                   # 0% rows: 0 by definition (shown in examples)
    d.loc[d.fraction == 0, "committed_by_here"] = np.nan      # not scored at 0%
    d = d.dropna(subset=["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std"])   # 9 rows, no valid continuation
    d["fold"] = d.example_id.map(fixed_fold_of(RUN_DIR, 5, 0))
    return d.sort_values(KEYS).reset_index(drop=True)


def label_json(r) -> str:
    return json.dumps({"answer_entropy_bits": round(float(r.answer_entropy_bits_mm), 2),
                       "p_mode": round(float(r.p_mode), 2), "rem_len_mean": int(round(r.rem_len_mean)),
                       "rem_len_std": int(round(r.rem_len_std)), "committed": int(r.label_commit)})


def item(r) -> str:
    return f"[checkpoint {int(round(r.fraction * 100))}%]\n{r.text}"


def example_order(d: pd.DataFrame, fold: int, n: int, seed: int = 1000) -> list[int]:
    """Row indices (into d) of the first n examples for `fold`: training-fold questions only, one new
    question per item, checkpoint by round robin over FRACS, random CoT; no row used twice."""
    rng = np.random.default_rng(seed + fold)
    qs = rng.permutation(d.loc[d.fold != fold, "example_id"].unique())
    idx = {(r.example_id, r.cot_id, r.fraction): i for i, r in enumerate(d[KEYS].itertuples())}
    cots = {q: sorted(d.loc[(d.example_id == q) & (d.fraction > 0), "cot_id"].unique()) for q in qs}
    out, used, j, qi = [], set(), 0, 0
    while len(out) < n:
        q, f = qs[qi % len(qs)], FRACS[j % len(FRACS)]
        qi += 1
        cot = -1 if f == 0 else int(rng.choice(cots[q]))
        if (q, cot, f) in used or (q, cot, f) not in idx:
            continue
        used.add((q, cot, f)); out.append(idx[(q, cot, f)]); j += 1
    return out


def test_rows(d: pd.DataFrame, n_cot: int, seed: int = 7) -> pd.DataFrame:
    """n_cot = 0: all rows. Else n_cot random CoTs per question (all their checkpoints) + the 0% row."""
    if n_cot == 0:
        return d
    keep = {}
    for q in d.example_id.unique():
        keep[q] = set(np.random.default_rng(seed + int(q)).permutation(25)[:n_cot].tolist()) | {-1}
    return d[[c in keep[q] for q, c in zip(d.example_id, d.cot_id)]]


def build_text(tok, cfg, d, ex_idx, row, thinking: bool) -> str:
    body = INSTR + (INSTR_EX if ex_idx else "") + INSTR_END
    parts = [body]
    for i, j in enumerate(ex_idx, 1):
        parts.append(f"=== EXAMPLE {i} ===\n{item(d.iloc[j])}\n=== MEASURED VALUES ===\n{label_json(d.iloc[j])}")
    parts.append(f"=== QUERY ===\n{item(row)}\n\n{INSTR_QUERY}\n=== MEASURED VALUES (JSON only) ===")
    s = tok.apply_chat_template([{"role": "user", "content": "\n\n".join(parts)}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=thinking)
    if "R1-Distill" in cfg["id"] and not thinking:
        s += "</think>\n\n"                                   # template ends with "<think>\n"
    return s


def encode(tok, s):                                          # the template already holds BOS
    return tok(s, add_special_tokens=False)["input_ids"]


# ---------------------------------------------------------------- plan / dry
def token_counts(tok, d, tag):
    f = OUT / f"tokcounts_{tag}.csv"
    if f.exists():
        return pd.read_csv(f)["n"].to_numpy()
    n = np.array([len(x) for x in tok([item(r) for r in d.itertuples()], add_special_tokens=False)["input_ids"]])
    OUT.mkdir(parents=True, exist_ok=True); pd.DataFrame({"n": n}).to_csv(f, index=False)
    return n


def cmd_plan(a):
    from transformers import AutoTokenizer
    cfg = MODELS[a.model]; ctx = YARN_CTX if a.yarn else cfg["ctx"]
    tok = AutoTokenizer.from_pretrained(cfg["id"])
    d = load_data(); nt = token_counts(tok, d, a.model)
    lab = np.array([len(x) for x in tok([label_json(r) for r in d.itertuples()], add_special_tokens=False)["input_ids"]])
    per = nt + lab + 24                                       # item + label + markers (approx.; verified exactly below)
    ovh = len(encode(tok, build_text(tok, cfg, d, [], d.iloc[0], False))) - len(encode(tok, item(d.iloc[0])))
    print(f"{a.model}: ctx {ctx}, rows {len(d)}, item tokens mean {nt.mean():.0f} median {np.median(nt):.0f} max {nt.max()}, "
          f"fixed overhead {ovh} tokens")
    sub = test_rows(d, a.subsample_cots); rows = []
    best, cums = {}, {}
    for f in range(5):
        cums[f] = np.cumsum(per[example_order(d, f, 600)])
        room = ctx - ANSWER_BUDGET - ovh - nt[d.index[d.fold == f]].max() - 500
        best[f] = int((cums[f] <= room).sum())
    kmax = min(best.values())
    for f in range(5):
        cum = cums[f]; tf = sub[sub.fold == f]
        for k in [0, 10, 30, kmax]:
            B = int(cum[k - 1]) if k else 0
            rows.append(dict(fold=f, k=k, block_tokens=B, n_test_full=int((d.fold == f).sum()), n_test_sub=len(tf),
                             prefill_cached_full=B + int(nt[d.index[d.fold == f]].sum()) + (d.fold == f).sum() * ovh,
                             prefill_cached_sub=B + int(nt[tf.index].sum()) + len(tf) * ovh,
                             prefill_nocache_sub=len(tf) * B + int(nt[tf.index].sum()) + len(tf) * ovh))
    print("max k per fold:", best, "-> use", min(best.values()))
    R = pd.DataFrame(rows); R.to_csv(OUT / f"plan_{a.model}{'_yarn' if a.yarn else ''}.csv", index=False)
    g = R.groupby("k").sum(numeric_only=True)
    print(g[["block_tokens", "prefill_cached_full", "prefill_cached_sub", "prefill_nocache_sub"]].div(1e6).round(2),
          "\n(million tokens, summed over the 5 folds; k=%d is 'max')" % kmax)
    # exact check: longest test prompt of each fold at k = kmax
    for f in range(5):
        r = d.loc[d[d.fold == f].index[nt[d.index[d.fold == f]].argmax()]]
        n = len(encode(tok, build_text(tok, cfg, d, example_order(d, f, kmax), r, False)))
        print(f"fold {f} k={kmax}: longest prompt {n} tokens (+{ANSWER_BUDGET} answer) vs ctx {ctx}: "
              f"{'OK' if n + ANSWER_BUDGET <= ctx else 'TOO LONG'}")


def cmd_dry(a):
    from transformers import AutoTokenizer
    cfg = MODELS[a.model]; tok = AutoTokenizer.from_pretrained(cfg["id"])
    d = load_data(); f = a.fold
    ex = example_order(d, f, a.k)
    assert (d.iloc[ex].fold != f).all()
    print("example fractions:", d.iloc[ex].fraction.value_counts().sort_index().to_dict(),
          "| distinct questions:", d.iloc[ex].example_id.nunique())
    T = test_rows(d, a.subsample_cots); T = T[T.fold == f]
    assert not set(T.example_id) & set(d.iloc[ex].example_id)
    lens = [len(encode(tok, build_text(tok, cfg, d, ex, r, a.thinking == "on"))) for r in T.head(a.n_dry).itertuples()]
    s = build_text(tok, cfg, d, ex, T.iloc[0], a.thinking == "on")
    print(f"fold {f} k={a.k}: {len(T)} test rows; first {len(lens)} prompts: tokens min {min(lens)} max {max(lens)}")
    print("---- prompt (head / tail) ----\n" + s[:2500] + "\n  [...]\n" + s[-900:] + "\n---- end ----")


# ---------------------------------------------------------------- run (GPU)
def cmd_run(a):
    import os
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    cfg = MODELS[a.model]; ctx = YARN_CTX if a.yarn else cfg["ctx"]
    tok = AutoTokenizer.from_pretrained(cfg["id"])
    d = load_data(); T = test_rows(d, a.subsample_cots)
    if a.max_rows:
        T = T.head(a.max_rows)
    tags = {k: f"{a.model}{'_yarn' if a.yarn else ''}_{a.thinking}_k{k}{'_smoke' if a.max_rows else ''}" for k in a.k}
    out = OUT / "raw"; out.mkdir(parents=True, exist_ok=True)
    todo = [(k, f) for k in a.k for f in range(5) if not (out / f"{tags[k]}_fold{f}.csv").exists() and (T.fold == f).any()]
    if not todo:
        return
    if a.yarn:
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    llm = LLM(cfg["id"], max_model_len=ctx, tensor_parallel_size=cfg["tp"], enable_prefix_caching=True,
              gpu_memory_utilization=0.92, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=16384,
              hf_overrides=YARN if a.yarn else None,
              **({} if "R1-Distill" in cfg["id"] else {"limit_mm_per_prompt": {"image": 0, "video": 0}}))
    from vllm.sampling_params import StructuredOutputsParams
    so = StructuredOutputsParams(regex=JSON_RE)
    end_id = tok.convert_tokens_to_ids(cfg["end"])
    close = encode(tok, cfg["end"] + "\n\n")
    for f in sorted({f for _, f in todo}):                  # fold-major: a fold's examples stay cached across its k values
        for k in [k for k, ff in todo if ff == f]:
            ex = example_order(d, f, k); Tf = T[T.fold == f]
            prompts = [TokensPrompt(prompt_token_ids=encode(tok, build_text(tok, cfg, d, ex, r, a.thinking == "on")))
                       for r in Tf.itertuples()]
            assert max(len(p["prompt_token_ids"]) for p in prompts) + ANSWER_BUDGET <= ctx
            t0 = time.time()
            llm.generate(prompts[:1], SamplingParams(temperature=0, max_tokens=1))      # fill the prefix cache first
            df = Tf[KEYS].copy()
            df["n_prompt"] = [len(p["prompt_token_ids"]) for p in prompts]
            thought = [""] * len(prompts)
            if a.thinking == "on":   # pass 1: think until the end-of-thinking token or the cap (shrunk to fit ctx)
                sp = [SamplingParams(temperature=0, stop_token_ids=[end_id], max_tokens=max(1, min(
                      a.think_cap, ctx - len(p["prompt_token_ids"]) - len(close) - ANSWER_BUDGET))) for p in prompts]
                res = llm.generate(prompts, sp)
                ids = [[t for t in r.outputs[0].token_ids if t != end_id] for r in res]
                df["think_tokens"] = [len(x) for x in ids]
                df["think_capped"] = [r.outputs[0].finish_reason == "length" for r in res]
                thought = [r.outputs[0].text + "</think>" for r in res]
                prompts = [TokensPrompt(prompt_token_ids=p["prompt_token_ids"] + x + close) for p, x in zip(prompts, ids)]
            # pass 2 (or only pass): forced JSON answer
            res = llm.generate(prompts, SamplingParams(temperature=0, structured_outputs=so, max_tokens=ANSWER_BUDGET))
            df["output"] = [t + r.outputs[0].text for t, r in zip(thought, res)]
            df.to_csv(out / f"{tags[k]}_fold{f}.csv", index=False)
            print(f"k={k} fold {f}: {len(df)} rows, {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- score
def parse(text: str):
    text = text.split("</think>")[-1]
    for m in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            j = json.loads(m)
            return [float(j[k]) for k in JSON_KEYS]
        except Exception:
            continue
    return [np.nan] * 5


def cmd_score(a):
    tag = f"{a.model}{'_yarn' if a.yarn else ''}_{a.thinking}_k{a.k}"
    d = load_data()[KEYS + ["fold"] + TARGETS]
    R = pd.concat([pd.read_csv(f) for f in sorted((OUT / "raw").glob(f"{tag}_fold?.csv"))], ignore_index=True)
    R["fraction"] = R.fraction.round(6)
    P = pd.DataFrame([parse(str(t)) for t in R.output], columns=TARGETS, index=R.index)
    P["committed_by_here"] = P.committed_by_here.clip(0, 1)
    m = R[KEYS].join(P).merge(d, on=KEYS, suffixes=("_pred", ""))
    print(f"{tag}: {len(m)} rows, parse failures {m[[t + '_pred' for t in TARGETS]].isna().any(axis=1).mean():.3f}")
    # calibration: per target and checkpoint, y = a + b*pred fitted on the OTHER folds' predictions
    long = []
    for t in TARGETS:
        for fr, g in m.groupby("fraction"):
            ok = g[t].notna() & g[t + "_pred"].notna()
            for f in range(5):
                te, tr = g[(g.fold == f) & ok], g[(g.fold != f) & ok]
                b, a0 = np.polyfit(tr[t + "_pred"], tr[t], 1) if tr[t + "_pred"].nunique() > 1 else (0.0, tr[t].mean())
                for probe, yp in (("icl_raw", te[t + "_pred"]), ("icl_cal", a0 + b * te[t + "_pred"])):
                    long.append(pd.DataFrame(dict(feature=tag, probe=probe, target=t, example_id=te.example_id,
                                                  cot_id=te.cot_id, fraction=fr, y_true=te[t], y_pred=yp)))
    L = pd.concat(long)
    L.to_csv(OUT / f"preds_{tag}.csv", index=False)
    res = []
    for (p, t, fr), g in L.groupby(["probe", "target", "fraction"]):
        y, yp = g.y_true.to_numpy(), g.y_pred.to_numpy()
        if t == "committed_by_here":
            res.append(dict(probe=p, target=t, fraction=fr, n=len(g), metric="acc", test=((yp > .5) == (y > .5)).mean()))
        else:
            res.append(dict(probe=p, target=t, fraction=fr, n=len(g), metric="r2",
                            test=1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum(), mse=((y - yp) ** 2).mean()))
    res = pd.DataFrame(res); res.insert(0, "feature", tag); res.to_csv(OUT / f"results_{tag}.csv", index=False)
    print(res.pivot_table(index=["probe", "target"], columns="fraction", values="test").round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["plan", "dry", "run", "score"])
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--k", type=int, nargs="+", default=[10], help="run: several k share one engine load")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--thinking", choices=["off", "on"], default="off")
    p.add_argument("--think-cap", type=int, default=16384)
    p.add_argument("--yarn", action="store_true", help="Qwen3.5 only: YaRN x4, context 1.01M")
    p.add_argument("--subsample-cots", type=int, default=1, help="CoTs per question to test (0 = all 25)")
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--n-dry", type=int, default=200)
    a = p.parse_args()
    if a.cmd != "run":
        a.k = a.k[0]
    {"plan": cmd_plan, "dry": cmd_dry, "run": cmd_run, "score": cmd_score}[a.cmd](a)
