#!/usr/bin/env python3
"""Weighted-pooling probes over the all-token store (round 7, item 7).

The question: does a LEARNED weighting over the prefix tokens beat reading the
last token, which is what every published probe in this project does?  The
Reasoning Theater paper (arXiv 2603.05488) compares its attention probe only
against a last-token probe, which cannot separate "the learned weighting helps"
from "having more than one token helps".  So four poolings are run on identical
rows and identical folds:

    last   the published probe's input (one vector per row)
    mean   average over the prefix's reasoning tokens
    max    elementwise maximum over them
    attn   z = W_v H . softmax(W_q H)      (the paper's Eq. 1)

and each is run BOTH closed form (ridge, via the project's own sweep.py, with
the published grids) and trained with the paper's Adam recipe, so "is the gain
the pooling or the optimiser" is answered by construction.

ONE CODE PATH FOR ALL FOUR.  A pooled feature is fed as a one-token sequence, so
the attention softmax is identically 1 and the model degenerates to a linear
head.  Same initialisation, same optimiser, same epoch budget, same best-val
rule for all four, so a difference between them is the pooling and nothing else.

Modes
-----
    pool     one pass over the token store -> pooled_{mean,max,last}.npy + keys.parquet,
             and the correctness gate: pooled_last must equal the published
             activations_ckpt vector for EVERY row (not a sample).
    actdir   a filtered activation-index directory (symlinks + a subset index), so
             sweep.py can score the published ridge probe on exactly these rows.
    train    the Adam-trained probes: 5 folds x 5 seeds, stacked and trained in one
             pass over the data, best-val-epoch per model, seed-ensembled
             out-of-fold predictions in the project's preds.csv schema.

Usage
-----
    $PY src/probing/attn_probe.py pool  --tag gpqa25 --tok-sub activations_tokens_pilot5
    $PY src/probing/attn_probe.py actdir --tag gpqa25 --cot-max 5 --act-out runs/.../_pilot_act
    $PY src/probing/attn_probe.py train --tag gpqa25 --tok-sub activations_tokens_pilot5 \
        --pooling attn --out-dir runs/goal2/gpqa25/sweep/r7_pooling
"""
from __future__ import annotations

import argparse, builtins, functools, glob, json, os, shutil, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.sweep import load_meta, fixed_fold_of          # noqa: E402  (imported, never edited)
from probing.probe_checkpoints import REG_TARGETS, BIN_TARGETS  # noqa: E402

print = functools.partial(builtins.print, flush=True)
KEYS = ["example_id", "cot_id", "fraction"]
REPO = Path(__file__).resolve().parents[2]
TOK_ROOT = Path(os.environ.get("R7_TOK_ROOT", "/work/users/y/u/yuvraj/llm-probing/runs/goal2"))
MS = "DeepSeek-R1-Distill-Qwen-7B"

SPECS = {
    "gpqa25":      dict(ds="gpqa_diamond",
                        targets=["answer_entropy_bits_mm", "p_mode", "rem_len_mean",
                                 "rem_len_std", "committed_by_here"]),
    "mmlupro_v1":  dict(ds="mmlu_pro",
                        targets=["answer_entropy_bits_mm", "answer_entropy_bits_mm_norm",
                                 "p_mode", "rem_len_mean", "rem_len_std", "committed_by_here"]),
    "globalop_v1": dict(ds="global_opinions",
                        targets=["answer_entropy_bits_mm", "answer_entropy_bits_mm_norm",
                                 "p_mode", "rem_len_mean", "rem_len_std", "committed_by_here"]),
}
FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]
# Poolings that read the TOKEN store. `meantok` is uniform weighting over exactly the
# same tokens, standardised exactly the same way, trained by exactly the same optimiser,
# so `attn` minus `meantok` isolates the LEARNED weighting and nothing else. The
# precomputed `mean` pooling is NOT that control: its per-dimension normalisation is
# fitted on already-averaged vectors, whose across-row spread is much smaller than a
# single token's, so it hands the optimiser a differently scaled problem.
TOKEN_POOLINGS = ("attn", "meantok")


def paths(tag, tok_sub="activations_tokens"):
    ds = SPECS[tag]["ds"]
    return dict(run_dir=REPO / f"runs/goal1/{tag}/{ds}/{MS}",
                act_dir=REPO / f"runs/goal2/{tag}/activations_ckpt/{ds}/{MS}",
                tok_dir=TOK_ROOT / f"{tag}/{tok_sub}/{ds}/{MS}")


def token_index(tok_dir: Path) -> pd.DataFrame:
    ps = sorted(tok_dir.glob("index.shard*.parquet"))
    if not ps:
        raise SystemExit(f"no token index under {tok_dir}")
    T = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)
    T["fraction"] = T.fraction.round(6)
    return T


def rows_frame(tag, tok_sub, labels_file="checkpoint_labels.csv"):
    """M: the token-store rows joined to the published labels, in a fixed order.

    The label join is sweep.load_meta, unchanged, so targets, their NaN pattern
    and the row set are exactly what ref_fixed used (intersected with whatever
    CoTs the token store covers).
    """
    P = paths(tag, tok_sub)
    M = load_meta(P["act_dir"], P["run_dir"], labels_file)
    M["fraction"] = M.fraction.round(6)
    T = token_index(P["tok_dir"])
    cols = KEYS + ["cot_tok_start", "cut_tok_end", "n_tok_cut", "shard"]
    M = M.merge(T[cols], on=KEYS, how="inner")
    M = M[M.fraction.isin([round(f, 6) for f in FRACS])]
    return M.sort_values(KEYS).reset_index(drop=True)


# -- mode: pool ----------------------------------------------------------------

def mode_pool(args):
    P = paths(args.tag, args.tok_sub)
    out = Path(args.out_dir or P["tok_dir"])
    M = rows_frame(args.tag, args.tok_sub)
    print(f"  {len(M):,} rows, {M.example_id.nunique()} questions, "
          f"{M.groupby(['example_id','cot_id']).ngroups:,} CoTs")

    cfg = json.loads(sorted(P["tok_dir"].glob("capture_config.shard*.json"))[0].read_text())
    layers = cfg["layers"]
    assert len(layers) == 1, f"pooling is written for one layer, got {layers}"

    d = None
    mean = max_ = last = None
    pos = {k: i for i, k in enumerate(zip(M.example_id, M.cot_id, M.fraction))}
    t0 = time.time()
    for sh, g in M.groupby("shard"):
        A = np.load(P["tok_dir"] / f"tokens.shard{int(sh):02d}.npy", mmap_mode="r")
        if d is None:
            d = A.shape[2]
            mean = np.empty((len(M), d), np.float32)
            max_ = np.empty((len(M), d), np.float32)
            last = np.empty((len(M), d), np.float16)
        for (eid, cid), gc in g.groupby(["example_id", "cot_id"]):
            gc = gc.sort_values("fraction")
            s = int(gc.cot_tok_start.iloc[0])
            e_last = int(gc.cut_tok_end.max())
            blk = np.asarray(A[s:e_last, 0, :])            # fp16, one read per CoT
            blkf = blk.astype(np.float32)
            run_sum = np.zeros(d, np.float64)
            run_max = np.full(d, -np.inf, np.float32)
            prev = 0
            for _, r in gc.iterrows():
                e = int(r.cut_tok_end) - s
                if e > prev:
                    run_sum += blkf[prev:e].sum(0, dtype=np.float64)
                    run_max = np.maximum(run_max, blkf[prev:e].max(0))
                    prev = e
                i = pos[(r.example_id, r.cot_id, r.fraction)]
                mean[i] = (run_sum / e).astype(np.float32)
                max_[i] = run_max
                last[i] = blk[e - 1]
        print(f"    shard {int(sh)}: {len(g):,} rows  {(time.time()-t0)/60:.1f} min")

    out.mkdir(parents=True, exist_ok=True)
    K = M[KEYS].copy()
    K.to_parquet(out / "pooled_keys.parquet", index=False)
    np.save(out / "pooled_mean.npy", mean)
    np.save(out / "pooled_max.npy", max_)
    np.save(out / "pooled_last.npy", last.astype(np.float32))
    print(f"  wrote pooled_{{mean,max,last}}.npy ({len(M):,}, {d}) to {out}")

    # THE GATE, on every row: pooled_last must be the published vector.
    PUB = []
    for ip in sorted(glob.glob(str(P["act_dir"] / "index.shard*.parquet"))):
        i = pd.read_parquet(ip).sort_values("row_idx")
        i["_shard"] = ip.split("index.")[1].replace(".parquet", "")
        i["_row"] = i.row_idx
        PUB.append(i[KEYS + ["_shard", "_row"]])
    PUB = pd.concat(PUB, ignore_index=True)
    PUB["fraction"] = PUB.fraction.round(6)
    m = M[KEYS].merge(PUB.drop_duplicates(KEYS), on=KEYS, how="left")
    assert m._row.notna().all(), "rows absent from the published store"
    L = int(layers[0])
    nbad = nbit = 0; worst = 0.0; mincos = 1.0
    for shp, g in m.groupby("_shard"):
        B = np.load(P["act_dir"] / f"activations.{shp}.npy", mmap_mode="r")
        r = g._row.to_numpy().astype(int)
        b = np.asarray(B[r, L - 1, :])
        a = last[g.index.to_numpy()]
        nbit += int((a == b).all(1).sum())
        af, bf = a.astype(np.float32), b.astype(np.float32)
        dif = np.abs(af - bf).max(1)
        cs = (af * bf).sum(1) / (np.linalg.norm(af, axis=1) * np.linalg.norm(bf, axis=1) + 1e-30)
        worst = max(worst, float(dif.max())); mincos = min(mincos, float(cs.min()))
        nbad += int((cs <= 0.9999).sum())
    print(f"  GATE over ALL {len(m):,} rows at decoder layer {L}:")
    print(f"    bitwise identical: {nbit:,}/{len(m):,}")
    print(f"    min cosine {mincos:.8f}   max |diff| {worst:.3e}   rows with cosine <= 0.9999: {nbad}")
    stride_ok = "n/a" if cfg["stride"] != 1 else str(int((M.n_tok_cut != M.prefix_tokens).sum()))
    print(f"    stride-1 token-count mismatches (n_tok_cut vs prefix_tokens): {stride_ok}")
    ok = (nbad == 0)
    print("    VERDICT:", "PASS" if ok else "FAIL")
    (out / "gate.json").write_text(json.dumps(dict(
        rows=len(m), bitwise_identical=nbit, min_cosine=mincos, max_abs_diff=worst,
        rows_below_threshold=nbad, stride1_token_count_mismatches=stride_ok,
        layer=L, verdict="PASS" if ok else "FAIL"), indent=2))
    if not ok:
        raise SystemExit("GATE FAILED")


# -- mode: actdir --------------------------------------------------------------

def mode_actdir(args):
    """A published-store view restricted to the CoTs the token store covers, so
    sweep.py scores the reference ridge probe on exactly the same rows."""
    P = paths(args.tag, args.tok_sub)
    out = Path(args.act_out); out.mkdir(parents=True, exist_ok=True)
    keep = None
    if args.cot_max is None:
        ti = token_index(P["tok_dir"])
        keep = {(int(a), int(b)) for a, b in zip(ti.example_id, ti.cot_id)}
    n = 0
    for ip in sorted(glob.glob(str(P["act_dir"] / "index.shard*.parquet"))):
        name = Path(ip).name
        sh = name.split("index.")[1].replace(".parquet", "")
        i = pd.read_parquet(ip)
        i = (i[i.cot_id < args.cot_max] if args.cot_max is not None
             else i[[((int(a), int(b)) in keep) for a, b in zip(i.example_id, i.cot_id)]])
        i.to_parquet(out / name, index=False)        # row_idx preserved -> same npy rows
        src = P["act_dir"] / f"activations.{sh}.npy"
        lnk = out / f"activations.{sh}.npy"
        if lnk.is_symlink() or lnk.exists():
            lnk.unlink()
        lnk.symlink_to(src)
        n += len(i)
    shutil.copy(sorted(P["act_dir"].glob("capture_config.shard*.json"))[0],
                out / "capture_config.shard00.json")
    print(f"  wrote {out}: {n:,} index rows, activations symlinked to the published store")


# -- mode: train ---------------------------------------------------------------

class Provider:
    """Row index -> (flat tokens, segment ids).  One token per row for the pooled
    poolings, the whole prefix for `attn`."""
    def __init__(self, pooling, M, tok_dir, pooled_dir):
        self.pooling = pooling
        if pooling in TOKEN_POOLINGS:
            self.A = {int(s): np.load(tok_dir / f"tokens.shard{int(s):02d}.npy", mmap_mode="r")
                      for s in M["shard"].unique()}
            self.sh = M["shard"].to_numpy().astype(int)
            self.st = M.cot_tok_start.to_numpy().astype(np.int64)
            self.en = M.cut_tok_end.to_numpy().astype(np.int64)
            self.ntok = (self.en - self.st).astype(np.int64)
            self.d = next(iter(self.A.values())).shape[2]
        else:
            K = pd.read_parquet(pooled_dir / "pooled_keys.parquet")
            K["fraction"] = K.fraction.round(6)
            X = np.load(pooled_dir / f"pooled_{pooling}.npy", mmap_mode="r")
            o = (K.reset_index().set_index(KEYS)
                 .loc[list(zip(M.example_id, M.cot_id, M.fraction)), "index"].to_numpy())
            self.X = np.asarray(X[o], dtype=np.float32)
            self.ntok = np.ones(len(M), np.int64)
            self.d = self.X.shape[1]

    def get(self, idx):
        if self.pooling not in TOKEN_POOLINGS:
            return self.X[idx], np.arange(len(idx), dtype=np.int64)
        n = self.ntok[idx]
        out = np.empty((int(n.sum()), self.d), np.float16)
        seg = np.repeat(np.arange(len(idx), dtype=np.int64), n)
        c = 0
        for j in idx:
            k = int(self.ntok[j])
            out[c:c + k] = self.A[self.sh[j]][self.st[j]:self.en[j], 0, :]
            c += k
        return out, seg


def batches(ntok, order, max_samples, tok_budget):
    b, tot = [], 0
    for j in order:
        k = int(ntok[j])
        if b and (len(b) >= max_samples or tot + k > tok_budget):
            yield np.array(b); b, tot = [], 0
        b.append(j); tot += k
    if b:
        yield np.array(b)


def mode_train(args):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    P = paths(args.tag, args.tok_sub)
    pooled_dir = Path(args.pooled_dir or P["tok_dir"])
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    targets = [t for t in SPECS[args.tag]["targets"]]
    nf, ns, nt = args.folds, args.seeds, len(targets)
    fold_of = fixed_fold_of(P["run_dir"], nf, args.fold_seed)

    M = rows_frame(args.tag, args.tok_sub)
    prov = Provider(args.pooling, M, P["tok_dir"], pooled_dir)
    dmod = prov.d
    print(f"  {args.pooling}: {len(M):,} rows, d={dmod}, device={dev}, "
          f"targets={targets}")

    # validation questions: a seeded 1/5 of each fold's TRAINING questions. Fold
    # dependent, seed independent, so the 5 seeds differ only by initialisation.
    qs = np.sort(M.example_id.unique())
    vq = {}
    for f in range(nf):
        tr = np.array([q for q in qs if fold_of[int(q)] != f])
        p = np.random.default_rng(1000 + f).permutation(len(tr))
        vq[f] = set(int(q) for q in tr[p[:max(1, len(tr) // 5)]])

    all_rows, results, preds = [], [], []
    for frac in (args.fractions or FRACS):
        rows = np.where(np.isclose(M.fraction.to_numpy(), frac))[0]
        if not len(rows):
            continue
        eid = M.example_id.to_numpy()[rows]
        fo = np.array([fold_of[int(q)] for q in eid])
        isval = np.array([[int(q) in vq[f] for f in range(nf)] for q in eid])
        TE = np.stack([fo == f for f in range(nf)], 1)          # (n, nf)
        VA = (~TE) & isval
        TR = (~TE) & (~isval)
        Y = np.stack([M[t].to_numpy(float)[rows] for t in targets], 1)   # (n, nt)
        OK = ~np.isnan(Y)
        if OK.sum(0).min() < 50:
            print(f"   cut {frac}: a target has <50 usable rows; "
                  f"counts {OK.sum(0).tolist()}")

        # per (fold, target) standardisation of the regression targets on non-test rows
        mu = np.zeros((nf, nt)); sd = np.ones((nf, nt))
        for f in range(nf):
            for j, t in enumerate(targets):
                m_ = (~TE[:, f]) & OK[:, j]
                if t in BIN_TARGETS:
                    continue
                mu[f, j] = Y[m_, j].mean(); sd[f, j] = Y[m_, j].std() or 1.0
        Ys = np.where(OK[:, None, :], (Y[:, None, :] - mu[None]) / sd[None], 0.0)  # (n, nf, nt)

        # Input normalisation, fitted on a sample of each fold's NON-TEST rows.
        #   global  one scalar mean and one scalar std per fold (the paper's option,
        #           which is off by default in their config)
        #   perdim  one mean and std per dimension per fold -- what sweep.py does for
        #           ridge, so the trained probes and the closed-form probes see the
        #           same preprocessing and a difference between them is the optimiser
        # Either way the correction is folded into the parameters at forward time, so
        # the token store is never rewritten.
        nmu = np.zeros((nf, dmod), np.float32); nsd = np.ones((nf, dmod), np.float32)
        rg = np.random.default_rng(7)
        for f in range(nf):
            cand = np.where(~TE[:, f])[0]
            s = cand[rg.permutation(len(cand))[:min(args.norm_rows, len(cand))]]
            xs, _ = prov.get(rows[s] if args.pooling in TOKEN_POOLINGS else s)
            if args.feat_norm == "global":
                nmu[f] = float(xs.astype(np.float32).mean())
                nsd[f] = float(xs.astype(np.float32).std()) or 1.0
            else:
                xf = xs.astype(np.float32)
                nmu[f] = xf.mean(0)
                sdv = xf.std(0); sdv[sdv < 1e-8] = 1.0; nsd[f] = sdv
        print(f"   cut {frac}: n={len(rows)}  {args.feat_norm} norm, sampled "
              f"{min(args.norm_rows, int((~TE[:, 0]).sum()))} rows/fold; "
              f"mean {nmu.mean():.4f} std {nsd.mean():.4f} "
              f"(std range {nsd.min():.3f}..{nsd.max():.3f})")

        # Stacked models.  Index m = g*nm0 + f*(ns*nt) + s*nt + j, where g indexes the
        # (lr, weight decay) grid.  All of them are trained in ONE pass over the data,
        # which is what makes the attention probe affordable: the token store is read
        # once per epoch no matter how many folds, seeds or settings are fitted.
        HP = [(float(a), float(b)) for a in args.lr_grid for b in args.wd_grid]
        G, nm0 = len(HP), nf * ns * nt
        nm = G * nm0
        m_f = np.tile(np.repeat(np.arange(nf), ns * nt), G)
        m_s = np.tile(np.tile(np.repeat(np.arange(ns), nt), nf), G)
        m_t = np.tile(np.tile(np.arange(nt), nf * ns), G)
        m_g = np.repeat(np.arange(G), nm0)
        T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
        # X arrives as fp16 from the store; cast on the GPU so only half the bytes move
        TX = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev).float()
        MUV, SDV = T(nmu[m_f]), T(nsd[m_f])        # (nm, d)
        IS_BIN = T(np.array([targets[j] in BIN_TARGETS for j in m_t], float))

        bound = 1.0 / np.sqrt(dmod)                        # nn.Linear default
        g_ = torch.Generator(device="cpu")
        Q0 = torch.empty(nm0, dmod); V0 = torch.empty(nm0, dmod)
        for s_ in range(ns):
            g_.manual_seed(args.seed0 + s_)
            sel = np.where(np.tile(np.repeat(np.arange(ns), nt), nf) == s_)[0]
            Q0[sel] = (torch.rand(len(sel), dmod, generator=g_) * 2 - 1) * bound
            V0[sel] = (torch.rand(len(sel), dmod, generator=g_) * 2 - 1) * bound
        Qs = [Q0.clone().to(dev).requires_grad_() for _ in HP]
        Vs = [V0.clone().to(dev).requires_grad_() for _ in HP]
        QBs = [torch.zeros(nm0, device=dev, requires_grad=True) for _ in HP]
        VBs = [torch.zeros(nm0, device=dev, requires_grad=True) for _ in HP]
        opt = torch.optim.Adam([
            dict(params=[Qs[i], Vs[i], QBs[i], VBs[i]], lr=lr_, weight_decay=wd_)
            for i, (lr_, wd_) in enumerate(HP)])

        TRt, VAt = T(TR.astype(float)), T(VA.astype(float))
        Yst, OKt = T(Ys), T(OK.astype(float))
        best = dict(val=torch.full((nm,), float("inf"), device=dev),
                    Q=torch.cat(Qs).detach().clone(), V=torch.cat(Vs).detach().clone(),
                    QB=torch.cat(QBs).detach().clone(), VB=torch.cat(VBs).detach().clone(),
                    ep=np.zeros(nm, int))

        def forward(X, seg, S, Q, V, QB, VB):
            Qn, Vn = Q / SDV, V / SDV                       # x' = (x - mu)/sd, folded in
            lg = X @ Qn.T - (Qn * MUV).sum(1) + QB          # (N, nm)
            vl = X @ Vn.T - (Vn * MUV).sum(1) + VB
            if args.pooling == "attn":
                mx = torch.full((S, nm), -1e30, device=dev).index_reduce_(
                    0, seg, lg, "amax", include_self=False)
                e = torch.exp(lg - mx[seg])
            elif args.pooling == "meantok":
                e = torch.ones_like(lg)          # uniform weighting over the same tokens
            else:
                return vl
            den = torch.zeros(S, nm, device=dev).index_add_(0, seg, e)
            num = torch.zeros(S, nm, device=dev).index_add_(0, seg, e * vl)
            return num / (den + 1e-8)

        def split_loss(z, idx):
            """Per-model train loss (for the step) and val loss (for selection)."""
            zz = z.view(-1, G, nf, ns, nt)
            y = Yst[idx][:, None]                            # (S, 1, nf, nt)
            ok = OKt[idx]                                    # (S, nt)
            ib = IS_BIN.view(G, nf, ns, nt)
            yt = y[:, :, :, None, :].expand_as(zz)
            el = ((zz - yt) ** 2) * (1 - ib) + torch.nn.functional. \
                binary_cross_entropy_with_logits(zz, yt.clamp(0, 1), reduction="none") * ib
            mtr = TRt[idx][:, None, :, None, None] * ok[:, None, None, None, :]
            mva = VAt[idx][:, None, :, None, None] * ok[:, None, None, None, :]
            f = lambda a: a.reshape(a.shape[0], -1).sum(0)
            return (f(el * mtr), f(mtr.expand_as(el)),
                    f(el.detach() * mva), f(mva.expand_as(el)))

        ntok_rows = (prov.ntok[rows] if args.pooling in TOKEN_POOLINGS
                     else np.ones(len(rows), np.int64))
        t0 = time.time()
        for ep in range(args.epochs):
            order = np.random.default_rng(5000 + ep).permutation(len(rows))
            vs = torch.zeros(nm, device=dev); vn = torch.zeros(nm, device=dev)
            for idx in batches(ntok_rows, order, args.batch_size, args.tok_budget):
                X, seg = prov.get(rows[idx] if args.pooling in TOKEN_POOLINGS else idx)
                z = forward(TX(X), torch.as_tensor(seg, device=dev), len(idx),
                            torch.cat(Qs), torch.cat(Vs), torch.cat(QBs), torch.cat(VBs))
                ltr, ctr, lva, cva = split_loss(z, T(idx, torch.long))
                opt.zero_grad(set_to_none=True)
                (ltr / ctr.clamp(min=1)).sum().backward()
                opt.step()
                vs += lva; vn += cva
            val = vs / vn.clamp(min=1)
            imp = val < best["val"]
            if bool(imp.any()):
                best["val"] = torch.where(imp, val, best["val"])
                cat = dict(Q=torch.cat(Qs), V=torch.cat(Vs),
                           QB=torch.cat(QBs), VB=torch.cat(VBs))
                for k in ("Q", "V", "QB", "VB"):
                    best[k] = torch.where(imp.view(-1, *([1] * (cat[k].dim() - 1))),
                                          cat[k].detach(), best[k])
                best["ep"][imp.cpu().numpy()] = ep
            if ep % args.log_every == 0 or ep == args.epochs - 1:
                print(f"    cut {frac} epoch {ep}: mean val {float(val.mean()):.4f}  "
                      f"best-so-far {float(best['val'].mean()):.4f}  "
                      f"improved {int(imp.sum())}/{nm}  {(time.time()-t0)/60:.1f} min")

        # Select (lr, wd) per (fold, seed, target) on validation loss, then predict
        # that fold's test rows with the chosen model's best-val-epoch weights and
        # average over the 5 seeds.
        bval = best["val"].cpu().numpy()
        key = m_f * (ns * nt) + m_s * nt + m_t
        chosen = np.zeros(nm0, int)
        for k in range(nm0):
            cand = np.where(key == k)[0]
            chosen[k] = cand[int(np.argmin(bval[cand]))]
        Z = np.zeros((len(rows), nm0), np.float32)
        Qb, Vb = best["Q"][chosen], best["V"][chosen]
        QBb, VBb = best["QB"][chosen], best["VB"][chosen]
        MUV_s, SDV_s, ISB_s = MUV[chosen], SDV[chosen], IS_BIN[chosen]
        MUV, SDV, IS_BIN = MUV_s, SDV_s, ISB_s
        nm_full, nm = nm, nm0
        with torch.no_grad():
            for idx in batches(ntok_rows, np.arange(len(rows)), args.batch_size, args.tok_budget):
                X, seg = prov.get(rows[idx] if args.pooling in TOKEN_POOLINGS else idx)
                Z[idx] = forward(TX(X), torch.as_tensor(seg, device=dev), len(idx),
                                 Qb, Vb, QBb, VBb).cpu().numpy()
        nm = nm_full
        Zr = Z.reshape(len(rows), nf, ns, nt)
        hp_pick = [HP[m_g[chosen[k]]] for k in range(nm0)]
        for j, t in enumerate(targets):
            yp = np.full(len(rows), np.nan)
            for f in range(nf):
                sel = TE[:, f] & OK[:, j]
                v = (1 / (1 + np.exp(-Zr[sel, f, :, j])) if t in BIN_TARGETS
                     else Zr[sel, f, :, j] * sd[f, j] + mu[f, j])
                yp[sel] = v.mean(1)
            ok = OK[:, j] & ~np.isnan(yp)
            preds.append(pd.DataFrame(dict(
                feature=f"tok_L21_{args.pooling}", probe=f"{args.pooling}_adam",
                target=t, example_id=eid[ok], cot_id=M.cot_id.to_numpy()[rows][ok],
                fraction=frac, y_true=Y[ok, j], y_pred=yp[ok])))
            if t in BIN_TARGETS:
                sc = float(((yp[ok] >= .5) == Y[ok, j]).mean())
            else:
                sc = float(1 - ((yp[ok] - Y[ok, j]) ** 2).sum() /
                           ((Y[ok, j] - Y[ok, j].mean()) ** 2).sum())
            picks = [hp_pick[f * ns * nt + s_ * nt + j] for f in range(nf) for s_ in range(ns)]
            eps = [int(best["ep"][chosen[f * ns * nt + s_ * nt + j]])
                   for f in range(nf) for s_ in range(ns)]
            results.append(dict(feature=f"tok_L21_{args.pooling}", probe=f"{args.pooling}_adam",
                                fraction=frac, target=t, n=int(ok.sum()),
                                metric="acc" if t in BIN_TARGETS else "r2", test=sc,
                                hp=json.dumps(picks), best_epochs=json.dumps(eps),
                                runtime_s=round(time.time() - t0, 1)))
            print(f"    cut {frac} {t}: {sc:.4f}   lr/wd picked {sorted(set(picks))}")

        if args.save_states and args.pooling == "attn":
            sdir = Path(args.save_states); sdir.mkdir(parents=True, exist_ok=True)
            tj = [targets.index(t) for t in args.state_targets if t in targets]
            sel_m = [(f, s_, j) for j in tj for f in range(nf) for s_ in range(ns)]
            cols = [f * ns * nt + s_ * nt + j for (f, s_, j) in sel_m]
            Qp = Qb[cols] / SDV[cols]                      # (K, d), normalisation folded in
            Qoff = -(Qb[cols] / SDV[cols] * MUV[cols]).sum(1) + QBb[cols]
            K = len(cols)
            POOL = np.zeros((len(rows), K, dmod), np.float16)
            ST = []
            with torch.no_grad():
                for idx in batches(ntok_rows, np.arange(len(rows)),
                                   args.batch_size, args.tok_budget):
                    X, seg = prov.get(rows[idx])
                    Xt = TX(X); segt = torch.as_tensor(seg, device=dev)
                    lg = Xt @ Qp.T + Qoff                  # (N, K)
                    S_ = len(idx)
                    mx = torch.full((S_, K), -1e30, device=dev).index_reduce_(
                        0, segt, lg, "amax", include_self=False)
                    e = torch.exp(lg - mx[segt])
                    den = torch.zeros(S_, K, device=dev).index_add_(0, segt, e)
                    a = e / (den[segt] + 1e-8)             # attention weights
                    for k in range(K):
                        pk = torch.zeros(S_, dmod, device=dev).index_add_(
                            0, segt, Xt * a[:, k:k + 1])
                        POOL[idx, k] = pk.cpu().numpy().astype(np.float16)
                    # diagnostics: effective number of attended tokens, and where the mass sits
                    ent = torch.zeros(S_, K, device=dev).index_add_(
                        0, segt, -a * torch.log(a + 1e-12))
                    amax = torch.full((S_, K), 0.0, device=dev).index_reduce_(
                        0, segt, a, "amax", include_self=False)
                    nt_ = torch.as_tensor(prov.ntok[rows[idx]], dtype=torch.float32, device=dev)
                    last_w = a[torch.cumsum(nt_.long(), 0) - 1]
                    ST.append(pd.DataFrame(dict(
                        row=np.repeat(idx, K), k=np.tile(np.arange(K), len(idx)),
                        n_tok=np.repeat(prov.ntok[rows[idx]], K),
                        eff_tokens=torch.exp(ent).cpu().numpy().ravel(),
                        max_weight=amax.cpu().numpy().ravel(),
                        last_weight=last_w.cpu().numpy().ravel())))
            for n_, (f, s_, j) in enumerate(sel_m):
                np.savez(sdir / f"states.frac{frac}.fold{f}.seed{s_}.{targets[j]}.npz",
                         rows=rows, X=POOL[:, n_], target=targets[j], fold=f, seed=s_,
                         frac=frac)
            SS = pd.concat(ST, ignore_index=True)
            SS["fold"] = [sel_m[k][0] for k in SS.k]
            SS["seed"] = [sel_m[k][1] for k in SS.k]
            SS["target"] = [targets[sel_m[k][2]] for k in SS.k]
            SS["fraction"] = frac
            SS.drop(columns="k").to_csv(sdir / f"attn_stats.frac{frac}.csv", index=False)
            print(f"    cut {frac}: wrote {len(sel_m)} state sets + attention stats; "
                  f"eff_tokens median {SS.eff_tokens.median():.1f} of n_tok median "
                  f"{SS.n_tok.median():.0f}; last-token weight median {SS.last_weight.median():.4f}")

    pd.DataFrame(results).to_csv(out / f"results_{args.pooling}.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(out / f"preds_{args.pooling}.csv", index=False)
    print(f"  wrote {out}/results_{args.pooling}.csv and preds_{args.pooling}.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["pool", "actdir", "train"])
    p.add_argument("--tag", required=True, choices=list(SPECS))
    p.add_argument("--tok-sub", default="activations_tokens")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--pooled-dir", default=None)
    p.add_argument("--act-out", default=None)
    p.add_argument("--cot-max", type=int, default=None)
    p.add_argument("--pooling", default="attn",
                   choices=["attn", "meantok", "mean", "max", "last"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--log-every", type=int, default=5)
    # The paper's single setting is lr 1e-3, weight decay 0.01 (experiment_config.py).
    # It is the first cell of the grid; the rest is selected per model on validation loss,
    # which is what this project does for every other probe.
    p.add_argument("--lr-grid", type=float, nargs="+", default=[1e-3, 1e-4])
    p.add_argument("--wd-grid", type=float, nargs="+",
                   default=[0.01, 1.0, 100.0, 1000.0, 10000.0])
    p.add_argument("--fractions", type=float, nargs="+", default=None,
                   help="restrict to these cuts (calibration only)")
    p.add_argument("--batch-size", type=int, default=16)   # paper's default
    p.add_argument("--tok-budget", type=int, default=65536)
    p.add_argument("--norm-rows", type=int, default=300)
    p.add_argument("--feat-norm", default="perdim", choices=["perdim", "global"])
    p.add_argument("--save-states", default=None,
                   help="dir for the attention-POOLED vectors, per (cut, fold, seed, target), "
                        "so a closed-form ridge can read them out (see attn_ridge.py)")
    p.add_argument("--state-targets", nargs="+",
                   default=["answer_entropy_bits_mm", "rem_len_mean"])
    a = p.parse_args()
    {"pool": mode_pool, "actdir": mode_actdir, "train": mode_train}[a.mode](a)


if __name__ == "__main__":
    main()
