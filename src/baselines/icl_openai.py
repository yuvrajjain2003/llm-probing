#!/usr/bin/env python3
"""R1 with a frontier API model (OpenAI). Same prompts, examples and folds as icl_predict.py, on a 100-row
subset of its test rows (20 per checkpoint 10..90%, 4 per fold). Two versions:
  ckpt   : as icl_predict.py (every item shows "[checkpoint N%]")
  nockpt : the examples hide their checkpoint; the query still shows it
Answers are forced JSON (structured outputs). Resumable: one JSON line per finished call in raw/<tag>.jsonl.
Key read from ~/.frontier_api_key (never printed).

    $PY src/baselines/icl_openai.py run   --k 10 50 100 --version ckpt nockpt
    $PY src/baselines/icl_openai.py score --k 10 50 100 --version ckpt nockpt
"""
from __future__ import annotations

import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines import icl_predict as ip

PRICES = {"gpt-6-astra": dict(inp=10.0, cached=1.0, out=50.0),    # $/1M tokens, <=272k-token prompts (checked 2026-10-15)
          "gpt-6-luna": dict(inp=0.10, cached=0.01, out=0.50)}    # checked 2026-10-21
MODEL, EFFORT = "gpt-6-astra", "high"              # MODEL set by --model
NAME, PRICE = "gpt6_" + MODEL.split("-")[-1], PRICES[MODEL]   # gpt6_astra
NOCKPT_OLD = ", and the checkpoint (how far into the full chain the prefix ends)."
NOCKPT_NEW = (". The labelled examples do not show their checkpoint; the query item shows its checkpoint "
              "(how far into the full chain the prefix ends).")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ip.JSON_KEYS,
          "properties": {k: {"type": "integer" if k == "committed" else "number"} for k in ip.JSON_KEYS}}


def subset(d: pd.DataFrame, small: bool = False, seed: int = 11) -> pd.DataFrame:
    """100 rows: 4 per fold x checkpoint. small: 20 rows, 1 per fold x checkpoint, minus fold i at checkpoint i."""
    T = ip.test_rows(d, 1)
    T = T[T.fraction > 0]
    T = T.groupby(["fold", "fraction"], group_keys=False).apply(lambda g: g.sample(4, random_state=seed))
    if small:   # 20 different questions; fold i skips checkpoint i
        F = [0.1, 0.3, 0.5, 0.7, 0.9]; rng = np.random.default_rng(seed); keep = []
        for f in range(5):
            qs = rng.permutation(T[T.fold == f].example_id.unique())
            keep += [(q, fr) for q, fr in zip(qs, [x for i, x in enumerate(F) if i != f])]
        T = T[[(q, fr) in set(keep) for q, fr in zip(T.example_id, T.fraction)]].drop_duplicates(["example_id", "fraction"])
    return T.sort_values(["fold"] + ip.KEYS)


DOMAIN_CSV = Path("data/gpqa_diamond/csv/train.csv")   # row i = example_id i (checked: same question text)


def subset50(d: pd.DataFrame, draws: int = 5000, seed: int = 5) -> pd.DataFrame:
    """The 20 small rows + 30 new: 2 per fold x checkpoint (10 per checkpoint, 10 per fold), 50 different
    questions, one CoT per question (R1 subsample). Among `draws` random fills, keep the one whose target
    distribution (mean, sd, quartiles) and subject mix (Chemistry/Physics/Biology) are closest to all
    10-90% rows. Uses labels and metadata only, never predictions."""
    S = subset(d, True)
    dom = pd.read_csv(DOMAIN_CSV)["High-level domain"]
    T = ip.test_rows(d, 1); T = T[(T.fraction > 0) & ~T.example_id.isin(S.example_id)]
    pop = d[d.fraction > 0]
    stats = lambda x: np.concatenate([[x[t].mean(), x[t].std()] + list(x[t].quantile([.25, .5, .75])) for t in ip.TARGETS])
    sd = np.repeat(pop[ip.TARGETS].std().to_numpy(), 5)
    p_dom = dom.value_counts(normalize=True)
    need = {(f, fr): 2 - ((S.fold == f) & (S.fraction == fr)).sum() for f in range(5) for fr in [0.1, 0.3, 0.5, 0.7, 0.9]}
    Y, Ys = T[ip.TARGETS].to_numpy(), S[ip.TARGETS].to_numpy()
    D, Ds = T.example_id.map(dom).to_numpy(), S.example_id.map(dom).to_numpy()
    cells = [(np.flatnonzero(((T.fold == f) & (T.fraction == fr)).to_numpy()), n) for (f, fr), n in need.items()]
    q = T.example_id.to_numpy(); ref = stats(pop)
    def npstats(A):
        return np.concatenate([[np.nanmean(a), np.nanstd(a, ddof=1)] + list(np.nanquantile(a, [.25, .5, .75])) for a in A.T])
    rng = np.random.default_rng(seed); best, best_gap = None, np.inf
    for _ in range(draws):
        used, pick = set(), []
        for idx, n in cells:
            c = [i for i in rng.permutation(idx) if q[i] not in used][:n]
            used |= {q[i] for i in c}; pick += c
        A = np.vstack([Ys, Y[pick]]); Dm = np.concatenate([Ds, D[pick]])
        share = pd.Series(Dm).value_counts(normalize=True).reindex(p_dom.index, fill_value=0)
        gap = max(np.abs((npstats(A) - ref) / sd).max(), (share - p_dom).abs().max() / 0.1)
        if gap < best_gap:
            best, best_gap = pd.concat([S, T.iloc[pick]]), gap
    print(f"subset50: best gap {best_gap:.3f} (max |diff| / population sd; subject share / 0.1)")
    return best.sort_values(["fold"] + ip.KEYS)


def subset10(d: pd.DataFrame) -> pd.DataFrame:
    """10 of the 20 small rows: 2 per fold, 2 per checkpoint, 10 different questions. Among all such picks,
    the one whose targets (mean, sd, quartiles) and subject mix are closest to all 10-90% rows (as subset50)."""
    import itertools
    S = subset(d, True).reset_index(drop=True)
    dom = pd.read_csv(DOMAIN_CSV)["High-level domain"]; p_dom = dom.value_counts(normalize=True)
    pop = d[d.fraction > 0]; sd = np.repeat(pop[ip.TARGETS].std().to_numpy(), 5)
    stats = lambda x: np.concatenate([[x[t].mean(), x[t].std()] + list(x[t].quantile([.25, .5, .75])) for t in ip.TARGETS])
    ref = stats(pop); best, best_gap = None, np.inf
    for pick in itertools.product(*[itertools.combinations(np.flatnonzero(S.fold == f), 2) for f in range(5)]):
        x = S.iloc[[i for c in pick for i in c]]
        if (x.fraction.round(1).value_counts() != 2).any():
            continue
        share = x.example_id.map(dom).value_counts(normalize=True).reindex(p_dom.index, fill_value=0)
        gap = max(np.abs((stats(x) - ref) / sd).max(), (share - p_dom).abs().max() / 0.1)
        if gap < best_gap:
            best, best_gap = x, gap
    print(f"subset10: best gap {best_gap:.3f}")
    return best.sort_values(["fold"] + ip.KEYS)


def prompt(d, ex, row, version: str) -> tuple[str, str]:
    instr = ip.INSTR
    if version == "nockpt":
        assert NOCKPT_OLD in instr; instr = instr.replace(NOCKPT_OLD, NOCKPT_NEW)
    parts = [instr + (ip.INSTR_EX if ex else "") + ip.INSTR_END]
    for i, j in enumerate(ex, 1):
        r = d.iloc[j]
        parts.append(f"=== EXAMPLE {i} ===\n{r.text if version == 'nockpt' else ip.item(r)}\n"
                     f"=== MEASURED VALUES ===\n{ip.label_json(r)}")
    query = f"=== QUERY ===\n{ip.item(row)}\n\n{ip.INSTR_QUERY}\n=== MEASURED VALUES (JSON only) ==="
    return "\n\n".join(parts) + "\n\n", query       # (shared block, query); joined = icl_predict's prompt text


def cost(u) -> float:
    c = u["cached_tokens"]
    return ((u["input_tokens"] - c) * PRICE["inp"] + c * PRICE["cached"] + u["output_tokens"] * PRICE["out"]) / 1e6


def cmd_run(a):
    from openai import OpenAI
    client = OpenAI(api_key=(Path.home() / ".frontier_api_key").read_text().strip(), timeout=3600, max_retries=5)
    d = ip.load_data(); T = subset50(d) if a.n50 else subset10(d) if a.n10 else subset(d, a.small)
    if a.max_rows:
        T = T.head(a.max_rows)
    out = ip.OUT / "raw"; out.mkdir(parents=True, exist_ok=True)
    spent, misses = 0.0, 0
    for v in a.version:
        for k in a.k:
            tag = f"{NAME}_{EFFORT}{'_nockpt' if v == 'nockpt' else ''}_k{k}{'_smoke' if a.max_rows else ''}"
            f = out / f"{tag}.jsonl"
            done = {tuple(json.loads(l)["key"]) for l in f.open()} if f.exists() else set()
            spent += sum(json.loads(l)["cost"] for l in f.open()) if f.exists() else 0
            for fold in range(5):
                ex = ip.example_order(d, fold, k)
                rows = [r for r in T[T.fold == fold].itertuples()
                        if (int(r.example_id), int(r.cot_id), round(r.fraction, 6)) not in done]

                def call(r):
                    t0 = time.time()
                    block, query = prompt(d, ex, r, v)
                    content = [{"type": "input_text", "text": block,   # explicit cache point after the examples
                                "prompt_cache_breakpoint": {"mode": "explicit"}},
                               {"type": "input_text", "text": query}]
                    resp = client.responses.create(
                        model=MODEL, input=[{"role": "user", "content": content}], reasoning={"effort": EFFORT},
                        extra_body={"prompt_cache_options": {"mode": "explicit"}},
                        text={"format": {"type": "json_schema", "name": "values", "schema": SCHEMA, "strict": True}},
                        max_output_tokens=a.max_output, prompt_cache_key=f"{tag}_f{fold}")
                    u = resp.usage
                    us = dict(input_tokens=u.input_tokens, cached_tokens=u.input_tokens_details.cached_tokens,
                              output_tokens=u.output_tokens, reasoning_tokens=u.output_tokens_details.reasoning_tokens)
                    return dict(key=[int(r.example_id), int(r.cot_id), round(r.fraction, 6)], fold=fold,
                                status=resp.status, output=resp.output_text, usage=us, cost=cost(us),
                                seconds=round(time.time() - t0, 1))

                with f.open("a") as fh:
                    for first, b in [(True, rows[:1]), (False, rows[1:])]:   # first call alone: caches the example block for the rest
                        with ThreadPoolExecutor(a.workers) as pool:
                            for res in pool.map(call, b):
                                fh.write(json.dumps(res) + "\n"); fh.flush(); spent += res["cost"]
                                print(f"{tag} fold {fold} {res['key']} {res['status']} in {res['usage']['input_tokens']}"
                                      f" cached {res['usage']['cached_tokens']} reason {res['usage']['reasoning_tokens']}"
                                      f" ${res['cost']:.2f} {res['seconds']}s | total ${spent:.2f}", flush=True)
                                if not first and res["usage"]["cached_tokens"] < res["usage"]["input_tokens"] / 2:
                                    misses += 1
                                if spent > a.budget or misses >= 2:   # in-flight calls (<= workers - 1) still finish
                                    print(f"stop: spent ${spent:.2f} (budget ${a.budget}), cache misses {misses}", flush=True); fh.flush(); os._exit(1)


def cmd_score(a):
    for v in a.version:
        for k in a.k:
            tag = f"{NAME}_{EFFORT}{'_nockpt' if v == 'nockpt' else ''}_k{k}"
            R = pd.DataFrame([json.loads(l) for l in (ip.OUT / "raw" / f"{tag}.jsonl").open()])
            R[ip.KEYS] = pd.DataFrame(R.key.tolist(), index=R.index)
            print(f"{tag}: {len(R)} calls, status {R.status.value_counts().to_dict()}, cost ${R.cost.sum():.0f}, "
                  f"reasoning tokens median {R.usage.map(lambda u: u['reasoning_tokens']).median():.0f}")
            for fold, g in R.groupby("fold"):        # icl_predict.cmd_score reads raw/<tag>_fold<f>.csv
                g[ip.KEYS + ["output"]].to_csv(ip.OUT / "raw" / f"{tag}_fold{fold}.csv", index=False)
            ip.cmd_score(argparse.Namespace(model=NAME, yarn=False, thinking=tag[len(NAME) + 1:].rsplit("_k", 1)[0], k=k))


def cmd_compare(a):
    """Pooled over the 20 rows (4 per checkpoint is too few per checkpoint): raw ICL predictions vs probes and
    earlier ICL runs on the SAME rows. 95% CI: question bootstrap (2000 draws), paired difference vs ridge."""
    T = (subset10 if a.n10 else lambda d: subset(d, True))(ip.load_data())[ip.KEYS]
    S = Path("runs/goal2/gpqa25/sweep/ref_fixed")
    P = pd.concat([pd.read_csv(f) for f in S.glob("preds*.csv")])
    src = {"ridge L21 (ours)": (P, "act_L21", {"ridge_act", "logistic"}), "TF-IDF": (P, "tfidf", {"ridge_text", "logistic"}),
           "prefix_len": (P, "prefix_len", {"ridge_len", "logistic"})}
    names = ["gpt6_astra"] + (["gpt6_luna"] if a.n10 else [])
    for tag in [f"{n}_{EFFORT}{v}_k{k}" for n in names for v in ["", "_nockpt"] for k in [10, 50]] + \
               ["gemma4_31b_off_k0", "gemma4_31b_off_k10", "gemma4_31b_off_k30"]:
        src[tag] = (pd.read_csv(ip.OUT / f"preds_{tag}.csv"), tag, {"icl_raw"})
    rng = np.random.default_rng(0); B = 2000
    out = []
    for t in ip.TARGETS:
        cols = {}
        for lab, (D, feat, probes) in src.items():
            x = D[(D.feature == feat) & D.probe.isin(probes) & (D.target == t)].copy()
            x["fraction"] = x.fraction.round(6)
            m = T.merge(x.drop_duplicates(ip.KEYS), on=ip.KEYS, how="left"); assert m.y_pred.notna().all(), (lab, t)
            cols[lab] = m.y_pred.to_numpy(); y = m.y_true.to_numpy()
        idx = rng.integers(0, len(y), (B, len(y)))   # every row a different question
        def score(p, i=slice(None)):
            yy, pp = y[i], p[i]
            if t == "committed_by_here":
                return ((pp > .5) == (yy > .5)).mean(-1)
            return 1 - ((yy - pp) ** 2).sum(-1) / ((yy - yy.mean(-1, keepdims=True)) ** 2).sum(-1)
        ref = score(cols["ridge L21 (ours)"], idx)
        for lab, p in cols.items():
            dd = score(p, idx) - ref
            out.append(dict(label=lab, target=t, score=score(p), diff_vs_ridge=score(p) - score(cols["ridge L21 (ours)"]),
                            ci_lo=np.percentile(dd, 2.5), ci_hi=np.percentile(dd, 97.5)))
    R = pd.DataFrame(out); R.to_csv(ip.OUT / (f"compare_gpt6_{len(T)}rows.csv" if a.n10 else f"compare_{NAME}_20rows.csv"), index=False)
    R["cell"] = [f"{s:.2f} ({d:+.2f} [{lo:+.2f},{hi:+.2f}])" for s, d, lo, hi in zip(R.score, R.diff_vs_ridge, R.ci_lo, R.ci_hi)]
    pd.set_option("display.width", 250)
    print(R.pivot(index="label", columns="target", values="cell")[ip.TARGETS].reindex(list(src)).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "score", "compare"])
    p.add_argument("--k", type=int, nargs="+", default=[10, 50, 100])
    p.add_argument("--version", nargs="+", choices=["ckpt", "nockpt"], default=["ckpt", "nockpt"])
    p.add_argument("--small", action="store_true", help="20-row subset")
    p.add_argument("--n50", action="store_true", help="50-row subset (the 20 + 30, matched to the population)")
    p.add_argument("--n10", action="store_true", help="10 of the 20 small rows, matched to the population")
    p.add_argument("--model", choices=list(PRICES), default="gpt-6-astra")
    p.add_argument("--max-rows", type=int, default=0, help="smoke test: first N rows (fold 0 first)")
    p.add_argument("--max-output", type=int, default=100000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--budget", type=float, default=1500, help="stop when estimated spend (all tags) exceeds this ($)")
    a = p.parse_args()
    MODEL = a.model; NAME, PRICE = "gpt6_" + MODEL.split("-")[-1], PRICES[MODEL]
    {"run": cmd_run, "score": cmd_score, "compare": cmd_compare}[a.cmd](a)
