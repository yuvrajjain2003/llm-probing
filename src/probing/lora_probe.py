#!/usr/bin/env python3
"""R8: LoRA + MLP-head probe on OUR model, scored like the sweep probes.

Same as the frozen probes: input = rendered prompt + prefix text (extract_checkpoints.py), read at the
last prefix token, hidden state after decoder layer 21 (HF hidden_states[21]), 5 targets, fixed
per-question folds (sweep.fixed_fold_of), early stopping on held-out TRAINING questions.
Differences: layers 1-21 are fine-tuned with LoRA; layers 22-28 and the LM head are dropped (norm ->
Identity, so last_hidden_state IS hidden_states[21]); the head is 3584 -> h -> ReLU -> dropout -> 5
(4 standardised regressions, MSE; 1 logit for committed_by_here, BCE).

PER CHECKPOINT, like the sweep probes: one model per (fraction, fold), trained on that fraction's training
rows only (one sequence = prompt + that checkpoint's prefix, read at its last token). --check feeds each
CoT's longest prefix once and reads all its checkpoints (causal), as in extract_checkpoints.py.

    $PY src/probing/lora_probe.py --check                       # GPU: layer-21 state vs stored activations
    $PY src/probing/lora_probe.py --fractions 0.5 --folds-run 0 --epochs 1   # pilot
    $PY src/probing/lora_probe.py                               # all 5 folds (resumable per fold)
    $PY src/probing/lora_probe.py --tiny --max-questions 15 --max-cots 2 --folds-run 0,1   # CPU smoke test

Output (--out-dir): preds.f{fraction}.fold{i}.csv per finished (fraction, fold); when all folds exist, preds.csv and results.csv
in sweep.py format (feature act_L21_lora, probe lora_mlp). `train` = score on a train-CoT subsample.
"""
from __future__ import annotations

import argparse, copy, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.sweep import Act, KEYS, fixed_fold_of, load_meta, score        # noqa: E402

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
REG = ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std"]
TARGETS = REG + ["committed_by_here"]
ROOT = "runs/goal2/gpqa25"
LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_cases(M, run_dir, tok, double_bos=False):
    """One case per CoT: ids of prompt + longest prefix, and (row index in M, token position)."""
    from generation.run_eval import render_mc_prompt
    ex = pd.read_csv(Path(run_dir) / "raw/examples.csv").set_index("example_id")
    prompts, cases = {}, []
    for (eid, cid), d in M.groupby(["example_id", "cot_id"]):
        if eid not in prompts:
            ch = ex.loc[eid, "choices"]
            prompts[eid] = render_mc_prompt(ex.loc[eid, "question"], json.loads(ch) if isinstance(ch, str) else None, tok)
        last = d.loc[d.token_pos.idxmax()]
        ids = tok.encode(prompts[eid] + last.prefix_text, add_special_tokens=False)
        assert len(ids) == last.token_pos + 1, (eid, cid, len(ids), last.token_pos)   # same tokens as stored
        off = 0
        if double_bos:                                  # what vLLM/EasySteer actually fed (see learnings.md)
            ids, off = [tok.bos_token_id] + ids, 1
        cases.append(dict(eid=int(eid), cid=int(cid), ids=ids, rows=d.index.to_numpy(),
                          pos=d.token_pos.to_numpy() + off))
    return cases


def load_trunk(n_layers, tiny, device):
    """Qwen2 trunk with layers 1..n_layers; norm = Identity so the output is hidden_states[n_layers]."""
    from transformers import AutoModel, AutoConfig
    if tiny:
        from transformers import Qwen2Config, Qwen2Model
        torch.manual_seed(0)
        full = Qwen2Model(Qwen2Config(vocab_size=152064, hidden_size=64, intermediate_size=128, num_hidden_layers=n_layers + 1,
                                      num_attention_heads=4, num_key_value_heads=2)).to(torch.float32)
    else:
        full = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    full.eval()
    m = copy.deepcopy(full) if tiny else full
    m.layers = m.layers[:n_layers]
    m.config.num_hidden_layers = n_layers
    if getattr(m.config, "layer_types", None):
        m.config.layer_types = m.config.layer_types[:n_layers]
    m.norm = torch.nn.Identity()
    m.config.use_cache = False
    return m.to(device), full


@torch.no_grad()
def states(model, c, device):
    x = torch.tensor([c["ids"]], device=device)
    return model(input_ids=x).last_hidden_state[0, c["pos"]]


def check(args, M, cases, device):
    """With LoRA off and no head, trunk state must equal the stored activations (cosine >= 0.99)."""
    A = Act(M, dict(act_dir=args.act_dir), [21]).raw
    model, _ = load_trunk(21, False, device)
    rng = np.random.default_rng(0)
    cos = []
    for i in rng.choice(len(cases), args.check_n, replace=False):
        c = cases[i]
        h = states(model, c, device).float().cpu().numpy()
        a = A[c["rows"]]
        cs = (h * a).sum(1) / (np.linalg.norm(h, axis=1) * np.linalg.norm(a, axis=1))
        cos += cs.tolist(); print(f"  eid {c['eid']} cot {c['cid']} len {len(c['ids'])} cos {np.round(cs, 4)}", flush=True)
    print(f"CHECK min cosine {min(cos):.4f} mean {np.mean(cos):.4f}  ->", "PASS" if min(cos) >= 0.99 else "FAIL")
    sys.exit(0 if min(cos) >= 0.99 else 1)


class Head(torch.nn.Module):
    def __init__(self, d, hidden, dropout, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu)); self.register_buffer("sd", torch.tensor(sd))
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden, 5))

    def forward(self, h):
        return self.net((h.float() - self.mu) / self.sd)


def loss_fn(out, y, ok):
    """y: (n,5) standardised regressions + 0/1; ok: (n,5) bool. Mean over labelled entries of each part."""
    l = 0.0
    if ok[:, :4].any():
        l = l + ((out[:, :4] - y[:, :4]) ** 2)[ok[:, :4]].mean()
    return l + torch.nn.functional.binary_cross_entropy_with_logits(out[:, 4], y[:, 4])


def batches(cases, accum, rng):
    """Length-sorted groups of `accum` sequences (similar token counts per step), groups shuffled."""
    order = sorted(range(len(cases)), key=lambda i: len(cases[i]["ids"]))
    g = [order[i:i + accum] for i in range(0, len(order), accum)]
    return [g[i] for i in rng.permutation(len(g))]


def run_fold(args, frac, fold, M, cases, fold_of, mu_sd, device):
    from peft import LoraConfig, get_peft_model
    qf = np.array([fold_of[c["eid"]] for c in cases])
    tr_all, te = np.where(qf != fold)[0], np.where(qf == fold)[0]
    rng = np.random.default_rng(args.seed + fold)
    qs = np.unique([cases[i]["eid"] for i in tr_all])
    vq = set(rng.choice(qs, max(1, int(args.val_frac * len(qs))), replace=False).tolist())
    va = np.array([i for i in tr_all if cases[i]["eid"] in vq]); tr = np.array([i for i in tr_all if cases[i]["eid"] not in vq])
    print(f"fraction {frac} fold {fold}: train {len(tr)} CoTs, val {len(va)}, test {len(te)}", flush=True)

    rows_tr = np.concatenate([cases[i]["rows"] for i in tr])
    Y = M[TARGETS].to_numpy(float)
    ym, ys = np.nanmean(Y[rows_tr, :4], 0), np.nanstd(Y[rows_tr, :4], 0)
    Z = Y.copy(); Z[:, :4] = (Y[:, :4] - ym) / ys
    Zt, Ok = torch.tensor(np.nan_to_num(Z), dtype=torch.float32), torch.tensor(~np.isnan(Y))

    trunk, _ = load_trunk(args.layers, args.tiny, device)
    torch.manual_seed(args.seed + fold)
    model = get_peft_model(trunk, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=LORA_MODULES))
    if not args.tiny:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
    head = Head(trunk.config.hidden_size, args.hidden, args.dropout, *mu_sd(rows_tr)).to(device)
    lora_p = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([dict(params=lora_p, lr=args.lr), dict(params=head.parameters(), lr=args.head_lr)],
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    print(f"  trainable LoRA params {sum(p.numel() for p in lora_p)/1e6:.1f}M, head {sum(p.numel() for p in head.parameters())/1e6:.2f}M", flush=True)

    def seq_loss(i):
        c = cases[i]; r = c["rows"]
        out = head(model(input_ids=torch.tensor([c["ids"]], device=device)).last_hidden_state[0, c["pos"]])
        return loss_fn(out, Zt[r].to(device), Ok[r].to(device))

    def val_loss():
        model.eval(); head.eval(); s = 0.0
        with torch.no_grad():
            for i in va:
                s += float(seq_loss(i))
        model.train(); head.train()
        return s / len(va)

    def predict(idx):
        model.eval(); head.eval(); rows, P = [], []
        with torch.no_grad():
            for i in idx:
                c = cases[i]
                P.append(head(model(input_ids=torch.tensor([c["ids"]], device=device)).last_hidden_state[0, c["pos"]]).cpu().numpy())
                rows.append(c["rows"])
        model.train(); head.train()
        rows, P = np.concatenate(rows), np.concatenate(P)
        P[:, :4] = P[:, :4] * ys + ym; P[:, 4] = 1 / (1 + np.exp(-P[:, 4]))
        return rows, P

    snap = lambda: ([p.detach().cpu().clone() for p in lora_p], copy.deepcopy(head.state_dict()))
    best, best_state, bad, step, t0 = np.inf, snap(), 0, 0, time.time()
    log = []
    model.train(); head.train()
    done = False
    for ep in range(args.epochs):
        for grp in batches([cases[i] for i in tr], args.accum, rng):
            opt.zero_grad()
            tl = 0.0
            for j in grp:
                l = seq_loss(tr[j]) / len(grp); l.backward(); tl += float(l)
            torch.nn.utils.clip_grad_norm_(lora_p + list(head.parameters()), 1.0)
            opt.step(); sched.step(); step += 1
            if step % args.eval_every == 0:
                v = val_loss(); log.append(dict(fraction=frac, fold=fold, epoch=ep, step=step, train_loss=tl, val_loss=v, min=(time.time() - t0) / 60))
                print(f"  ep {ep} step {step} train {tl:.4f} val {v:.4f} ({log[-1]['min']:.1f} min)", flush=True)
                if v < best:
                    best, best_state, bad = v, snap(), 0
                else:
                    bad += 1
                    if bad >= args.patience:
                        done = True; break
        if done:
            break
    for p, s in zip(lora_p, best_state[0]):
        p.data.copy_(s.to(p.device))
    head.load_state_dict(best_state[1])

    out = []
    sub = rng.choice(tr, min(args.train_score_cots, len(tr)), replace=False)
    for split, idx in [("test", te), ("train", sub)]:
        rows, P = predict(idx)
        d = M.loc[rows, KEYS].copy()
        for k, t in enumerate(TARGETS):
            d[f"y_{t}"], d[f"p_{t}"] = M.loc[rows, t].to_numpy(), P[:, k]
        d["split"] = split; d["fold"] = fold
        out.append(d)
    pd.concat(out).to_csv(Path(args.out_dir) / f"preds.f{frac}.fold{fold}.csv", index=False)
    pd.DataFrame(log).to_csv(Path(args.out_dir) / f"log.f{frac}.fold{fold}.csv", index=False)
    print(f"fraction {frac} fold {fold} done: {step} steps, best val {best:.4f}, {(time.time()-t0)/60:.1f} min, "
          f"peak GPU mem {torch.cuda.max_memory_allocated()/2**30 if device == 'cuda' else 0:.1f} GiB", flush=True)


def finalize(args, M):
    fs = sorted(Path(args.out_dir).glob("preds.f*.fold*.csv"))
    D = pd.concat([pd.read_csv(f) for f in fs])
    te = D[D.split == "test"]; tr = D[D.split == "train"]
    P, R = [], []
    for f, d in te.groupby("fraction"):
        for t in TARGETS:
            ok = d[f"y_{t}"].notna()
            kind = "bin" if t == "committed_by_here" else "reg"
            y = d.loc[ok, f"y_{t}"].to_numpy(); p = d.loc[ok, f"p_{t}"].to_numpy()
            if kind == "bin":
                y = y.astype(int)
            trd = tr[(np.isclose(tr.fraction, f)) & tr[f"y_{t}"].notna()]
            ytr = trd[f"y_{t}"].to_numpy(); ytr = ytr.astype(int) if kind == "bin" else ytr
            r = dict(feature="act_L21_lora", probe="lora_mlp", fraction=f, target=t, n=int(ok.sum()),
                     metric="acc" if kind == "bin" else "R2", test=score(kind, y, p),
                     train=score(kind, ytr, trd[f"p_{t}"].to_numpy()), test_auc=np.nan,
                     hp=json.dumps(vars(args)), runtime_s=np.nan)
            if kind == "bin":
                from sklearn.metrics import roc_auc_score
                r["test_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
            R.append(r)
            q = d.loc[ok, KEYS].copy()
            q.insert(0, "target", t); q.insert(0, "probe", "lora_mlp"); q.insert(0, "feature", "act_L21_lora")
            q["y_true"], q["y_pred"] = y, p
            P.append(q)
    pd.DataFrame(R).to_csv(Path(args.out_dir) / "results.csv", index=False)
    pd.concat(P).to_csv(Path(args.out_dir) / "preds.csv", index=False)
    print(f"wrote results.csv ({len(R)} rows, {len(fs)} fraction-fold files) and preds.csv")
    print(pd.DataFrame(R)[["fraction", "target", "test", "train"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--act-dir", default=f"{ROOT}/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--run-dir", default="runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--out-dir", default=f"{ROOT}/lora/main")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--folds-run", default="")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--layers", type=int, default=21)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--accum", type=int, default=16, help="sequences per optimizer step")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=40, help="optimizer steps between validation passes")
    ap.add_argument("--patience", type=int, default=3, help="validation passes without improvement")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--train-score-cots", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--check-n", type=int, default=6)
    ap.add_argument("--double-bos", action="store_true", help="feed BOS twice, as EasySteer did (check only)")
    ap.add_argument("--tiny", action="store_true", help="CPU smoke test: tiny random Qwen2")
    ap.add_argument("--max-questions", type=int); ap.add_argument("--max-cots", type=int)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    device = "cpu" if args.tiny else "cuda"
    M = load_meta(Path(args.act_dir), Path(args.run_dir))
    if args.max_questions:
        M = M[M.example_id.isin(sorted(M.example_id.unique())[:args.max_questions])]
    if args.max_cots:
        M = M[M.cot_id < args.max_cots]
    M = M.reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if args.check:
        cases = build_cases(M, args.run_dir, tok, args.double_bos)
        print(f"{len(M)} rows, {len(cases)} CoT sequences", flush=True)
        return check(args, M, cases, device)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    fold_of = fixed_fold_of(Path(args.run_dir), args.folds, 0)
    if args.tiny:                                       # no stored activations of a random model
        d = 64; mu_sd = lambda rows: (np.zeros(d, np.float32), np.ones(d, np.float32))
    else:
        A = Act(M, dict(act_dir=args.act_dir), [21]).raw
        def mu_sd(rows):                                # train-row stats of the stored layer-21 states
            sd = A[rows].std(0); sd[sd < 1e-8] = 1.0
            return A[rows].mean(0), sd
    todo = [int(f) for f in args.folds_run.split(",")] if args.folds_run else range(args.folds)
    for frac in args.fractions:
        cases = None
        for f in todo:
            if (Path(args.out_dir) / f"preds.f{frac}.fold{f}.csv").exists():
                continue
            if cases is None:
                cases = build_cases(M[np.isclose(M.fraction, frac)], args.run_dir, tok)
                print(f"fraction {frac}: {len(cases)} sequences, {sum(len(c['ids']) for c in cases)/1e6:.2f}M tokens", flush=True)
            run_fold(args, frac, f, M, cases, fold_of, mu_sd, device)
    if len(list(Path(args.out_dir).glob("preds.f*.fold*.csv"))) >= len(args.fractions) * len(todo):
        finalize(args, M)


if __name__ == "__main__":
    main()
