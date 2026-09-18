#!/usr/bin/env python3
"""Probe the Goal 1 behavioural labels from CoT-prefix activations.

This is the project's actual research question: do internal representations at a
CoT prefix predict properties of the DISTRIBUTION OVER FUTURE CONTINUATIONS
(entropy, remaining length, commitment) beyond what the prefix text already gives?

The Goal 1 labels are ground truth in the strong sense -- they were measured by
actually sampling K continuations from each prefix, not estimated.

Two design decisions that make or break this:

  WITHIN-CHECKPOINT TRAINING. Entropy falls 0.955 -> 0.403 and remaining length
  4735 -> 952 across the 5 checkpoints, and depth is itself decodable from the
  activations (0.82). A probe pooled across checkpoints could score well by
  reading depth alone and predicting the per-depth mean, while knowing nothing
  about the individual CoT. Training separately at each checkpoint holds depth
  constant by construction, so that shortcut is unavailable.

  TEXT BASELINE ON THE PREFIX. The claim is "beyond the prefix text", so the
  competitor is TF-IDF on the prefix text itself -- not just the question.
  A prefix-length baseline is included too, since remaining length is trivially
  related to how far in we already are.

Metric is out-of-fold R^2 for regression (0 = no better than predicting the
training mean) and accuracy for the binary target, always with 5-fold CV grouped
by example_id: the 3 CoTs of a question share its text, so ungrouped CV leaks.
"""
from __future__ import annotations

import argparse, builtins, functools, glob, json
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(builtins.print, flush=True)
LAYERS = {0: 14, 1: 21, 2: 28}
REG_TARGETS = ["answer_entropy_bits", "rem_len_mean", "rem_len_std", "p_mode"]


def align(act_dir: Path, run_dir: Path):
    """One activation vector per (CoT, checkpoint), taken at the checkpoint's own
    token. checkpoints.jsonl stores token_offset = tokens in the prefix, and the
    activation index stores absolute token_index plus n_prompt_tokens, so the
    target absolute position is n_prompt + token_offset - 1. We take the nearest
    captured token at or before it (capture stride is 4)."""
    cps = pd.DataFrame([json.loads(l) for l in open(run_dir / "raw" / "checkpoints.jsonl")])
    cps = cps.drop_duplicates(["example_id", "cot_id", "fraction"])[
        ["example_id", "cot_id", "fraction", "token_offset", "prefix_text"]]

    rows, vecs = [], []
    for ip in sorted(glob.glob(str(act_dir / "index.shard*.parquet"))):
        sfx = ip.split("index.")[1].replace(".parquet", "")
        arr = np.load(act_dir / f"activations.{sfx}.npy", mmap_mode="r")
        idx = pd.read_parquet(ip)
        for (eid, cid), g in idx.groupby(["example_id", "cot_id"]):
            sub = cps[(cps.example_id == eid) & (cps.cot_id == cid)]
            if sub.empty:
                continue
            ti = g.token_index.to_numpy()
            npt = int(g.n_prompt_tokens.iloc[0])
            order = np.argsort(ti); ti_s = ti[order]
            rid = g.row_idx.to_numpy()[order]
            for r in sub.itertuples():
                want = npt + int(r.token_offset) - 1
                j = np.searchsorted(ti_s, want, side="right") - 1
                if j < 0:
                    continue
                vecs.append(np.asarray(arr[rid[j]], dtype=np.float32))
                rows.append(dict(example_id=eid, cot_id=cid, fraction=float(r.fraction),
                                 token_index=int(ti_s[j]), prefix_tokens=int(r.token_offset),
                                 prefix_text=r.prefix_text))
    return np.stack(vecs), pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--act-dir", default="runs/goal2/activations/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--run-dir", default="runs/goal1/full/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--out-dir", default="runs/goal2/probes/gpqa_diamond")
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()

    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import r2_score

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    cache = out / "aligned_checkpoints.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        X, M = z["X"], pd.DataFrame(z["M"], columns=z["cols"])
        for c in ("example_id", "cot_id", "token_index", "prefix_tokens"):
            M[c] = M[c].astype(int)
        M["fraction"] = M["fraction"].astype(float)
        print(f"  loaded cached aligned activations {X.shape}")
    else:
        print("  aligning activations to checkpoint token offsets ...")
        X, M = align(Path(a.act_dir), Path(a.run_dir))
        np.savez(cache, X=X, M=M.to_numpy(), cols=np.array(M.columns))
        print(f"  aligned {X.shape}")

    lab = pd.read_csv(Path(a.run_dir) / "labels" / "checkpoint_labels.csv")
    cm = pd.read_csv(Path(a.run_dir) / "labels" / "commitment.csv")
    M = M.merge(lab, on=["example_id", "cot_id", "fraction"], how="left")
    M = M.merge(cm[["example_id", "cot_id", "commitment_fraction"]],
                on=["example_id", "cot_id"], how="left")
    # binary: has this CoT committed by this checkpoint?
    M["committed_by_here"] = (M.commitment_fraction.notna() &
                              (M.fraction >= M.commitment_fraction)).astype(int)
    print(f"  joined labels: {len(M)} (CoT, checkpoint) rows, "
          f"{M.example_id.nunique()} questions")

    res = []
    for frac in sorted(M.fraction.unique()):
        m = (M.fraction == frac).to_numpy()
        Mi, Xi = M[m].reset_index(drop=True), X[m]
        g = Mi.example_id.to_numpy()
        cv = list(GroupKFold(n_splits=a.folds).split(Xi, np.zeros(len(Mi)), g))
        print(f"\n=== checkpoint {frac:.0%}   n={len(Mi)} CoTs, {len(set(g))} questions ===")

        for tgt in REG_TARGETS + ["committed_by_here"]:
            y = Mi[tgt].to_numpy(dtype=float)
            ok = ~np.isnan(y)
            if ok.sum() < 50 or (tgt == "committed_by_here" and len(np.unique(y[ok])) < 2):
                continue
            binary = tgt == "committed_by_here"

            def score(get_train, get_test, kind):
                """kind: 'act'/'len'/'text' -> fit and return mean out-of-fold score."""
                s = []
                for tr, te in cv:
                    tr = tr[ok[tr]]; te = te[ok[te]]
                    if len(te) == 0: continue
                    Xtr, Xte = get_train(tr), get_test(te)
                    if binary:
                        mdl = make_pipeline(StandardScaler(with_mean=(kind != "text")),
                                            LogisticRegression(max_iter=2000, C=0.01))
                        mdl.fit(Xtr, y[tr])
                        s.append(float((mdl.predict(Xte) == y[te]).mean()))
                    else:
                        mdl = make_pipeline(StandardScaler(with_mean=(kind != "text")),
                                            Ridge(alpha=1000.0))
                        mdl.fit(Xtr, y[tr])
                        s.append(float(r2_score(y[te], mdl.predict(Xte))))
                return float(np.mean(s))

            base = (max(y[ok].mean(), 1 - y[ok].mean()) if binary else 0.0)
            # prefix-length baseline (1 feature)
            L = Mi.prefix_tokens.to_numpy().reshape(-1, 1).astype(float)
            s_len = score(lambda t: L[t], lambda t: L[t], "len")
            # text baseline on the PREFIX text
            txt = Mi.prefix_text.fillna("").to_numpy()
            s_txt = []
            for tr, te in cv:
                tr = tr[ok[tr]]; te = te[ok[te]]
                if len(te) == 0: continue
                v = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
                Xt = v.fit_transform(txt[tr]); Xe = v.transform(txt[te])
                if binary:
                    mdl = LogisticRegression(max_iter=2000, C=1.0).fit(Xt, y[tr])
                    s_txt.append(float((mdl.predict(Xe) == y[te]).mean()))
                else:
                    mdl = Ridge(alpha=1.0).fit(Xt, y[tr])
                    s_txt.append(float(r2_score(y[te], mdl.predict(Xe))))
            s_txt = float(np.mean(s_txt))
            row = dict(fraction=frac, target=tgt, n=int(ok.sum()),
                       baseline=round(base, 3), prefix_len=round(s_len, 3),
                       text=round(s_txt, 3))
            for col, Lname in LAYERS.items():
                row[f"act_L{Lname}"] = round(
                    score(lambda t, c=col: Xi[t, c, :], lambda t, c=col: Xi[t, c, :], "act"), 3)
            res.append(row)
            metric = "acc" if binary else "R2"
            print(f"   {tgt:<20} [{metric}]  len {row['prefix_len']:>6}  text {row['text']:>6}  "
                  f"act14 {row['act_L14']:>6}  act21 {row['act_L21']:>6}  act28 {row['act_L28']:>6}")

    df = pd.DataFrame(res)
    df.to_csv(out / "goal1_label_probes.csv", index=False)
    print(f"\n  wrote {out/'goal1_label_probes.csv'}")


if __name__ == "__main__":
    main()
