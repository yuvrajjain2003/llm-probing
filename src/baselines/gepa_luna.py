#!/usr/bin/env python3
"""GEPA prompt optimisation (DSPy) of the zero-shot Luna in-context prompt (icl_openai.py, k=0).

Rows: icl_predict's 1-CoT-per-question set (1,178 rows incl. 0%). For outer fold f, GEPA gets the rows of
the OTHER folds' questions (~940), split by question 70/30 into GEPA train / val; the test rows are fold f's
rows, the same rows the Luna runs and the probes were scored on. Two prompt versions, as icl_openai.py:
  ckpt  : the item shows "[checkpoint N%]"
  blind : no checkpoint shown anywhere
Task and reflection model: gpt-6-luna, reasoning effort high. Key from ~/.frontier_api_key (never printed).

    $PY      src/baselines/gepa_luna.py prep                               # rows -> gepa/rows.jsonl
    $PY_DSPY src/baselines/gepa_luna.py optimize --fold 0 --version ckpt   # resumable (log_dir)
    $PY_DSPY src/baselines/gepa_luna.py test --fold 0 --version ckpt --which seed gepa
    $PY      src/baselines/gepa_luna.py score --version ckpt --which seed gepa   # needs all 5 folds' tests
$PY_DSPY = envs/dspy/bin/python (dspy 3.4.1, gepa 0.1.4). Raw test outputs go to icl/raw/<tag>_fold<f>.csv
in icl_predict's format, so its scorer (raw + calibrated R2 / acc) is reused unchanged.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("runs/goal2/gpqa25/gepa")
RAW = Path("runs/goal2/gpqa25/icl/raw")
KEYS = ["example_id", "cot_id", "fraction"]
REG = ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std"]   # scored as R2
OUTS = ["answer_entropy_bits", "p_mode", "rem_len_mean", "rem_len_std", "committed"]   # = icl_predict.JSON_KEYS
PRICE = dict(inp=0.10, cached=0.01, out=0.50)     # gpt-6-luna $/1M tokens (icl_openai.PRICES)


PCT = ["entropy_pct", "p_mode_pct", "rem_len_mean_pct", "rem_len_std_pct"]   # --pct mode outputs, 0-100
PCT_INSTR = ("For each of the four measured numbers, give its percentile from 0 to 100 among items cut at the same "
             "point of their chains: 0 = lowest of all such items, 50 = typical, 100 = highest. Also give committed "
             "(0 or 1). Do not solve the question itself.")


def tag_of(which: str, version: str, pct: bool = False, rank: bool = False, model: str = "luna",
           all_rows: bool = False, setup: str = "") -> str:
    mode = "rank" if rank else "pct" if pct else ""
    return (f"{'gepa' if which == 'gepa' else 'dspyseed'}{mode}{setup}_{model}{'all' if all_rows else ''}"
            f"_{version}_k0")


def rows_file(a) -> Path:
    return OUT / ("rows_all.jsonl" if a.all else "rows.jsonl")   # = icl_predict.cmd_score's tag


# ---------------------------------------------------------------- prep / score ($PY: needs icl_predict's deps)
def cmd_prep(a):
    from baselines import icl_predict as ip
    d = ip.test_rows(ip.load_data(), 0 if a.all else 1)
    d = d.assign(item=[ip.item(r) for r in d.itertuples()])
    cols = KEYS + ["fold", "text", "item"] + REG + ["committed_by_here", "label_commit"]
    OUT.mkdir(parents=True, exist_ok=True)
    d[cols].to_json(rows_file(a), orient="records", lines=True)
    # Seed instruction = icl_openai's k=0 prompt text, minus the JSON-format lines (DSPy's typed output
    # fields replace them).
    from baselines.icl_openai import NOCKPT_OLD, BLIND_NEW
    for v in ["ckpt", "blind"]:
        s = ip.INSTR if v == "ckpt" else ip.INSTR.replace(NOCKPT_OLD, BLIND_NEW)
        (OUT / f"seed_instructions_{v}.txt").write_text(
            s.strip() + "\nPredict the measured values for the item. Do not solve the question itself.")
    print(f"wrote {rows_file(a)}: {len(d)} rows, per fold {d.groupby('fold').size().tolist()}")


def cmd_score(a):
    from baselines import icl_predict as ip
    for w in a.which:
        tag = tag_of(w, a.version, a.pct, a.rank, a.model, a.all, a.setup)   # model_thinking_k0
        model, thinking = tag.rsplit("_k0", 1)[0].split(f"_{a.version}")[0], a.version
        ip.cmd_score(argparse.Namespace(model=model, yarn=False, thinking=thinking, k=0))


# ---------------------------------------------------------------- DSPy side ($PY_DSPY)
def instructions(version: str, pct: bool = False) -> str:
    s = (OUT / f"seed_instructions_{version}.txt").read_text()     # written by prep
    return s.replace("Predict the measured values for the item. Do not solve the question itself.", PCT_INSTR) if pct else s


def run_dir(a) -> Path:
    mode = "_rank" if a.rank else "_pct" if a.pct else ""
    extra = (f"_{a.model}" if a.model != "luna" else "") + ("_all" if a.all else "") + (f"_{a.setup}" if a.setup else "")
    return OUT / f"{a.version}_fold{a.fold}{mode}{extra}{'_dummy' if a.dummy else ''}"


def pctile(vals: np.ndarray, y: float) -> float:
    """Mid-rank percentile of y among vals (ties share the middle)."""
    return 100 * ((vals < y).sum() + 0.5 * (vals == y).sum()) / len(vals)


def split(rows: pd.DataFrame, fold: int, seed: int = 0):
    rows = rows.assign(fraction=rows.fraction.round(6))     # json round trip gives 0.30000000000000004
    tr = rows[rows.fold != fold]
    qs = np.random.default_rng(seed + fold).permutation(tr.example_id.unique())
    val_q = set(qs[: round(0.3 * len(qs))])
    return tr[~tr.example_id.isin(val_q)], tr[tr.example_id.isin(val_q)], rows[rows.fold == fold]


def make_lm(dummy: bool = False, model: str = "luna", reflection: bool = False):
    import dspy
    import litellm; litellm.completion   # load fully here: a first import from many threads at once races
    if dummy:   # offline check of the plumbing: fixed plausible answer
        from dspy.utils.dummies import DummyLM
        return DummyLM([{"answer_entropy_bits": "0.5", "p_mode": "0.8", "rem_len_mean": "2000", "entropy_pct": "40", "p_mode_pct": "60", "rem_len_mean_pct": "50", "rem_len_std_pct": "50",
                         "rem_len_std": "1500", "committed": "0"}] * 100000)
    if model == "gemma":   # local vLLM server (runs/gepa/gemma_job.sh), thinking off as in the R1 runs
        return dspy.LM("openai/google/gemma-4-31B-it", api_base=os.environ.get("GEMMA_URL", "http://localhost:8000/v1"),
                       api_key="EMPTY", temperature=1.0 if reflection else 0.0,
                       max_tokens=8000 if reflection else 1024, num_retries=5,
                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    key = (Path.home() / ".frontier_api_key").read_text().strip()
    return dspy.LM("openai/gpt-6-luna", api_key=key, model_type="responses", temperature=1.0,
                   max_tokens=32000, reasoning_effort="high", num_retries=5)


def program(version: str, pct: bool = False):
    import dspy

    class Tagged(dspy.Predict):   # stamps each prediction with its instruction, for the monitor log
        def forward(self, **kw):
            r = super().forward(**kw)
            r.instr = str(abs(hash(self.signature.instructions)))
            return r

    if pct:
        class P(dspy.Signature):
            item: str = dspy.InputField()
            entropy_pct: float = dspy.OutputField()
            p_mode_pct: float = dspy.OutputField()
            rem_len_mean_pct: float = dspy.OutputField()
            rem_len_std_pct: float = dspy.OutputField()
            committed: int = dspy.OutputField()
        return Tagged(P.with_instructions(instructions(version, True)))

    class Predict(dspy.Signature):
        item: str = dspy.InputField()
        answer_entropy_bits: float = dspy.OutputField()
        p_mode: float = dspy.OutputField()
        rem_len_mean: float = dspy.OutputField()
        rem_len_std: float = dspy.OutputField()
        committed: int = dspy.OutputField()

    return Tagged(Predict.with_instructions(instructions(version)))


def examples(df: pd.DataFrame, version: str):
    import dspy
    col = "item" if version == "ckpt" else "text"
    return [dspy.Example(item=r[col], **{k: r[k] for k in KEYS + REG + ["committed_by_here"]}).with_inputs("item")
            for r in df.to_dict("records")]


def make_metric(train: pd.DataFrame):
    """Per-row score in [0, 1]: mean over targets of (s + 1) / 2, s = max(-1, 1 - err^2 / var), var = the
    target's variance at that checkpoint in GEPA-train (so the mean of s over a checkpoint is its R2, before
    clipping); committed: s = +1 right / -1 wrong, skipped at 0%. Feedback names each error and the typical
    range at that checkpoint."""
    import dspy
    stats = {(t, fr): (g[t].var(), g[t].median(), g[t].quantile(.1), g[t].quantile(.9))
             for fr, g in train.groupby("fraction") for t in REG}

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        fr = round(float(gold.fraction), 6)
        s, fb = [], [f"Checkpoint {round(fr * 100)}%."]
        for t, o in zip(REG, OUTS):
            y, yp = float(gold[t]), getattr(pred, o, None)
            var, med, lo, hi = stats[(t, fr)]
            try:
                yp = float(yp)
            except (TypeError, ValueError):
                s.append(-1.0); fb.append(f"{o}: missing or not a number."); continue
            s.append(max(-1.0, 1 - (yp - y) ** 2 / var))
            fb.append(f"{o}: predicted {yp:.3g}, measured {y:.3g} ("
                      + ("about right" if abs(yp - y) < 0.25 * var ** .5 else
                         f"too {'high' if yp > y else 'low'} by {abs(yp - y):.3g}")
                      + f"). At this checkpoint the training median is {med:.3g}, 10-90% range {lo:.3g} to {hi:.3g}.")
        c = gold.committed_by_here
        if c is not None and not pd.isna(c):
            ok = (float(getattr(pred, "committed", 0) or 0) > .5) == (c > .5)
            s.append(1.0 if ok else -1.0)
            fb.append(f"committed: predicted {getattr(pred, 'committed', None)}, measured {int(c)} "
                      f"({'right' if ok else 'wrong'}).")
        return dspy.Prediction(score=float(np.mean([(x + 1) / 2 for x in s])), feedback=" ".join(fb))
    return metric


def make_metric_pct(train: pd.DataFrame, log: Path):
    """--pct mode. Per target s = max(-1, 1 - (p - t)^2 / var(t)), p = predicted percentile, t = the row's
    true percentile among GEPA-train rows at that checkpoint. A percentile has no scale to learn, so only
    ranking earns score. Feedback names the error, never typical values. Every call is logged for monitoring."""
    import dspy, threading
    vals = {(t, fr): g[t].to_numpy() for fr, g in train.groupby("fraction") for t in REG}
    var = {k: np.var([pctile(v, y) for y in v]) for k, v in vals.items()}
    lock = threading.Lock()
    words = dict(answer_entropy_bits_mm="answer entropy", p_mode="modal-answer share",
                 rem_len_mean="mean remaining length", rem_len_std="spread of remaining length")

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        fr = round(float(gold.fraction), 6)
        s, fb, rec = [], [f"Checkpoint {round(fr * 100)}%."], {}
        for t, o in zip(REG, PCT):
            tp = pctile(vals[(t, fr)], float(gold[t]))
            try:
                p = float(getattr(pred, o, None))
            except (TypeError, ValueError):
                s.append(-1.0); fb.append(f"{o}: missing or not a number."); continue
            rec[o] = [p, tp]
            s.append(max(-1.0, 1 - (p - tp) ** 2 / var[(t, fr)]))
            fb.append(f"{o}: predicted {p:.0f}, true percentile {tp:.0f} ("
                      + ("about right" if abs(p - tp) < 10 else
                         f"too {'high' if p > tp else 'low'}: this item's {words[t]} is "
                         f"{'lower' if p > tp else 'higher'} than you ranked it")
                      + ").")
        c = gold.committed_by_here
        if c is not None and not pd.isna(c):
            ok = (float(getattr(pred, "committed", 0) or 0) > .5) == (c > .5)
            s.append(1.0 if ok else -1.0)
            fb.append(f"committed: predicted {getattr(pred, 'committed', None)}, measured {int(c)} "
                      f"({'right' if ok else 'wrong'}).")
        score = float(np.mean([(x + 1) / 2 for x in s]))
        with lock, open(log, "a") as f:
            f.write(json.dumps(dict(instr=getattr(pred, "instr", None), example_id=int(gold.example_id),
                                    cot_id=int(gold.cot_id), fraction=fr, score=score, **rec)) + "\n")
        return dspy.Prediction(score=score, feedback=" ".join(fb))
    return metric


def patch_batch_rank_score(train: pd.DataFrame, log: Path):
    """--rank mode. Replace GEPA's per-row scores with ONE batch score given to every row of the batch: the
    mean over the four targets of Spearman(predicted percentile, true within-checkpoint percentile), pooled
    over the batch (both sides are within-checkpoint percentiles, so pooling across checkpoints is fair),
    mapped to [0, 1] as (rho + 1) / 2. Rank-only: invariant to any monotone rescaling, so neither scale nor
    shrinkage toward the middle can earn score. Missing outputs count as 50. Committed is not scored.
    Per-row feedback for reflection still comes from make_metric_pct."""
    from scipy.stats import spearmanr
    from dspy.teleprompt.gepa import gepa_utils
    vals = {(t, fr): g[t].to_numpy() for fr, g in train.groupby("fraction") for t in REG}
    orig = gepa_utils.DspyAdapter.evaluate

    def evaluate(self, batch, candidate, capture_traces=False, num_threads=None):
        eb = orig(self, batch, candidate, capture_traces=capture_traces, num_threads=num_threads)
        rhos = []
        for t, o in zip(REG, PCT):
            tp = [pctile(vals[(t, round(float(ex.fraction), 6))], float(ex[t])) for ex in batch]
            pp = []
            for out in eb.outputs:
                try:
                    pp.append(float(getattr(out, o)))
                except (TypeError, ValueError, AttributeError):
                    pp.append(50.0)
            r = spearmanr(pp, tp)[0] if len(batch) > 2 else 0.0
            rhos.append(0.0 if np.isnan(r) else r)
        score = (float(np.mean(rhos)) + 1) / 2
        with open(log, "a") as f:
            f.write(json.dumps(dict(n=len(batch), score=score, rhos=rhos)) + "\n")
        eb.scores = [score] * len(batch)
        return eb
    gepa_utils.DspyAdapter.evaluate = evaluate
    orig_refl = gepa_utils.DspyAdapter.make_reflective_dataset

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        """Keep reflection inside a 32k context: at most 10 examples, each item cut to its first 1,200
        characters (checkpoint tag + question + choices) and last 800 (end of the reasoning so far)."""
        out = orig_refl(self, candidate, eval_batch, components_to_update)
        rng = np.random.default_rng(len(eval_batch.outputs))
        for k, recs in out.items():
            recs = [recs[i] for i in sorted(rng.choice(len(recs), min(10, len(recs)), replace=False))]
            for r in recs:
                it = str(r["Inputs"].get("item", ""))
                if len(it) > 2000:
                    r["Inputs"]["item"] = it[:1200] + "\n[... middle of the reasoning omitted ...]\n" + it[-800:]
            out[k] = recs
        return out
    gepa_utils.DspyAdapter.make_reflective_dataset = make_reflective_dataset


def usage_cost(lm) -> tuple[int, float]:
    n, c = 0, 0.0
    for h in getattr(lm, "history", []):
        u = h.get("usage") or {}
        n += 1
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        c += ((u.get("prompt_tokens", 0) - cached) * PRICE["inp"] + cached * PRICE["cached"]
              + u.get("completion_tokens", 0) * PRICE["out"]) / 1e6
    return n, c


def cmd_optimize(a):
    import dspy
    rows = pd.read_json(rows_file(a), lines=True)
    rows = rows[rows.fraction == 0] if a.frac0 else rows
    tr, va, _ = split(rows, a.fold)
    run = run_dir(a)
    run.mkdir(parents=True, exist_ok=True)
    task, refl = make_lm(a.dummy, a.model), make_lm(a.dummy, a.model, reflection=True)
    dspy.configure(lm=task)
    if a.dummy:   # reflection must return an instruction in ``` blocks
        from dspy.utils.dummies import DummyLM
        txt = instructions(a.version, a.pct or a.rank) + " (dummy edit)"
        refl = DummyLM([{"new_instruction": txt}] * 1000, adapter=dspy.JSONAdapter())
    (run / "split.json").write_text(json.dumps(dict(val_q=sorted(int(q) for q in va.example_id.unique()))))
    metric = make_metric_pct(tr, run / "metric_log.jsonl") if a.pct or a.rank else make_metric(tr)
    if a.rank:
        patch_batch_rank_score(tr, run / "batch_log.jsonl")
        budget = dict(max_metric_calls=a.budget or 2500, reflection_minibatch_size=a.minibatch,
                      warn_on_score_mismatch=False)
    else:
        budget = dict(auto=a.auto)
    opt = dspy.GEPA(metric=metric, reflection_lm=refl, num_threads=a.threads, **budget,
                    failure_score=0.0, perfect_score=1.0, log_dir=str(run), track_stats=True, seed=0)
    print(f"fold {a.fold} {a.version}: GEPA train {len(tr)} rows / {tr.example_id.nunique()} q, "
          f"val {len(va)} / {va.example_id.nunique()} q", flush=True)
    best = opt.compile(program(a.version, a.pct or a.rank), trainset=examples(tr, a.version), valset=examples(va, a.version))
    best.save(str(run / "best_program.json"))
    (run / "best_instructions.txt").write_text(best.signature.instructions)
    r = best.detailed_results
    pd.DataFrame(dict(candidate=range(len(r.val_aggregate_scores)), val_score=r.val_aggregate_scores)).to_csv(
        run / "candidates.csv", index=False)
    n1, c1 = usage_cost(task); n2, c2 = usage_cost(refl)
    print(f"done: best val {max(r.val_aggregate_scores):.4f} (seed {r.val_aggregate_scores[0]:.4f}); "
          f"task calls {n1} ${c1:.2f}, reflection calls {n2} ${c2:.2f}", flush=True)


def cmd_test(a):
    import dspy
    from concurrent.futures import ThreadPoolExecutor
    rows = pd.read_json(rows_file(a), lines=True)
    rows = rows[rows.fraction == 0] if a.frac0 else rows
    tr, _, te = split(rows, a.fold)
    vals = {(t, fr): g[t].to_numpy() for fr, g in tr.groupby("fraction") for t in REG}
    lm = make_lm(a.dummy, a.model); dspy.configure(lm=lm)
    for w in a.which:
        prog = program(a.version, a.pct or a.rank)
        if w == "gepa":
            prog.load(str(run_dir(a) / "best_program.json"))
        f = RAW / f"{tag_of(w, a.version, a.pct, a.rank, a.model, a.all, a.setup)}{'_dummy' if a.dummy else ''}_fold{a.fold}.csv"
        done = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=KEYS + ["output"])
        key = lambda r: (int(r["example_id"]), int(r["cot_id"]), round(float(r["fraction"]), 6))
        seen = {key(r) for r in done.to_dict("records")}
        todo = [r for r in examples(te, a.version) if key(r) not in seen]

        def call(ex):
            try:
                p = prog(item=ex.item)
                if a.pct or a.rank:   # percentile -> value via the GEPA-train rows' quantiles at that checkpoint
                    fr = round(float(ex.fraction), 6)
                    out = json.dumps({o: float(np.quantile(vals[(t, fr)], min(max(float(getattr(p, po)), 0), 100) / 100))
                                      for t, o, po in zip(REG, OUTS, PCT)} | {"committed": p.committed})
                else:
                    out = json.dumps({o: getattr(p, o) for o in OUTS})
            except Exception as e:     # parse failure / content refusal: scored as missing
                out = f"ERROR {type(e).__name__}: {str(e)[:150]}"
            return {**{k: ex[k] for k in KEYS}, "output": out}

        with ThreadPoolExecutor(a.threads) as pool:
            res = list(pool.map(call, todo))
        pd.concat([done, pd.DataFrame(res)]).to_csv(f, index=False)
        n, c = usage_cost(lm)
        print(f"{f}: +{len(res)} rows (total {len(done) + len(res)}/{len(te)}); calls so far {n} ${c:.2f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prep", "optimize", "test", "score"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--version", choices=["ckpt", "blind"], default="ckpt")
    p.add_argument("--which", nargs="+", choices=["seed", "gepa"], default=["seed", "gepa"])
    p.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--pct", action="store_true", help="predict within-checkpoint percentiles (scale-free metric)")
    p.add_argument("--rank", action="store_true", help="as --pct, but GEPA scores whole batches by Spearman")
    p.add_argument("--model", choices=["luna", "gemma"], default="luna")
    p.add_argument("--all", action="store_true", help="whole dataset (all 25 chains per question), rows_all.jsonl")
    p.add_argument("--budget", type=int, default=0, help="--rank: max metric calls (default 2500)")
    p.add_argument("--minibatch", type=int, default=30, help="--rank: reflection minibatch size")
    p.add_argument("--setup", default="", help="label for this setup; goes into the run dir and output tags")
    p.add_argument("--frac0", action="store_true", help="only the 0%% checkpoint rows (adds 'f0' to --setup)")
    p.add_argument("--dummy", action="store_true", help="offline plumbing check with a fake LM (no API calls)")
    a = p.parse_args()
    a.setup += "f0" if a.frac0 else ""
    dict(prep=cmd_prep, optimize=cmd_optimize, test=cmd_test, score=cmd_score)[a.cmd](a)
