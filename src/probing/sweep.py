#!/usr/bin/env python3
"""Sweep: every (feature set x probe x hyper-parameter grid x target x checkpoint) -> one tidy CSV.

USAGE
    cd <repo> && source env.sh
    $PY src/probing/sweep.py --config configs/sweep_baseline.yaml            # run / resume
    $PY src/probing/sweep.py --config C.yaml --shard 0/4                     # 1 of 4 parallel jobs
    $PY src/probing/sweep.py --config C.yaml --merge                         # shards -> results.csv
    $PY src/probing/sweep.py --config C.yaml --export out.csv --pred act_L21/ridge_act \
        --base text=tfidf/ridge_text --base prefixlen=prefix_len/ridge_len   # for bootstrap_ci.py
    Extra flags: --out-dir DIR (override), --max-questions N (small smoke test).
    Run on Slurm CPU nodes (sbatch -p small ...), never on the login node.

CONFIG (YAML; see configs/sweep_baseline.yaml)
    act_dir / run_dir / out_dir, folds, fractions (optional), targets (optional)
    features: {name: {type: act|prefix_len|tfidf|npy, ...params}}
              any dense feature also takes `pca: k` (PCA fit on each training split only,
              k standardised scores); reuse of a feature name with a different pca needs a new name
    probes:   {name: {class: <registered probe>, grid: {hp: [..]}, fixed: {hp: v}}}
              a grid value may be {logspace: [start, stop, num]}; grids are crossed.
    runs:     [{features: [..], probes: [..]}]   -> product of the two lists
    Probes whose `kinds` do not cover a target (e.g. ridge on a binary target) are skipped.
    Combo identity = (feature, probe name, fraction, target). Rows already in the output are
    skipped, so CHANGE THE PROBE NAME (or delete rows) when you change its grid.

PROTOCOL
    One model per checkpoint. Outer folds are FIXED per question (default): each example_id
    gets fold = rank in a seeded permutation of all questions in examples.csv, mod `folds`
    (config `fold_seed`, default 0) -- same fold at every checkpoint, feature and probe.
    `fixed_folds: false` restores probe_checkpoints.py's per-checkpoint GroupKFold (only used
    to reproduce probe_checkpoints.csv; there only 74/198 questions keep their fold). Hyper-parameters are picked per outer fold by inner GroupKFold(3) on the
    training part only (sum of inner R2, or accuracy for binary targets). `test` is pooled
    out-of-fold R2 / accuracy; `train` is the mean over outer folds of the score of the final
    fit on its own training rows. Features are standardised on the training rows only (tfidf:
    vectorizer fitted on training rows only). Per-combo `runtime_s` excludes the shared
    feature/fold preparation (standardise / tf-idf are built once per unit; SVDs are cached
    on first use, so the first ridge/logistic combo of a unit pays for them).

OUTPUT (out_dir)
    results[.shardIofN].csv  feature, probe, fraction, target, n, metric, test, train,
                             test_auc, hp (json: chosen hp per outer fold), runtime_s
    preds[.shardIofN].csv    long out-of-fold predictions (disable per run with `preds: false`)

ADD A PROBE   subclass Probe, give it `kinds`, fit(X, y, groups=None) / predict(X) (reg: value,
              bin: P(y=1)), decorate with @register_probe("name"); optionally override
              classmethod fit_path to fit a whole grid cheaply. Then add a `probes:` entry.
ADD A FEATURE subclass Feature (self.raw = row-aligned array in the order of M; optionally
              override prep(train, test) for train-only fitting), @register_feature("type").
              Precomputed features (e.g. sentence embeddings) need no code: use type `npy`.

NPY FEATURE CONTRACT  {type: npy, path: X.npy, keys: K.csv|K.parquet}
    X.npy is (m, d); K has m rows with columns example_id, cot_id, fraction where row j of K
    describes row j of X (K may be in any order, may cover more rows than we use). Rows are
    joined to the activation index on (example_id, cot_id, round(fraction, 6)); it is an error
    if any of our rows is missing in K or if K has duplicate keys. Row order is never assumed.
"""
from __future__ import annotations

import argparse, builtins, functools, glob, itertools, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.probe_checkpoints import (RidgePath, _r2, text_features,   # noqa: E402
                                       REG_TARGETS, BIN_TARGETS)

print = functools.partial(builtins.print, flush=True)
KEYS = ["example_id", "cot_id", "fraction"]


# -- data (mirrors probe_checkpoints.load, but without reading activations) -----

def load_meta(act_dir: Path, run_dir: Path) -> pd.DataFrame:
    """One row per (CoT, checkpoint), same inner joins/order as probe_checkpoints.load().
    Extra columns `_shard`, `_row` locate the activation row in its shard file."""
    idxs = []
    for ip in sorted(glob.glob(str(act_dir / "index.shard*.parquet"))):
        i = pd.read_parquet(ip).sort_values("row_idx")
        i["_shard"] = ip.split("index.")[1].replace(".parquet", "")
        i["_row"] = i.row_idx
        idxs.append(i.drop(columns="row_idx"))
    if not idxs:
        raise SystemExit(f"no activation shards under {act_dir}")
    M = pd.concat(idxs, ignore_index=True)
    lab = pd.read_csv(run_dir / "labels" / "checkpoint_labels.csv")
    cm = pd.read_csv(run_dir / "labels" / "commitment.csv")
    M = M.merge(lab, on=KEYS, how="inner")
    M = M.merge(cm[["example_id", "cot_id", "commitment_fraction"]],
                on=["example_id", "cot_id"], how="inner")
    M["committed_by_here"] = (M.commitment_fraction.notna() &
                              (M.fraction >= M.commitment_fraction)).astype(int)
    M["will_commit"] = M.commitment_fraction.notna().astype(int)
    cps = pd.DataFrame([json.loads(l) for l in open(run_dir / "raw" / "checkpoints.jsonl")])
    cps = cps.drop_duplicates(KEYS)
    M = M.merge(cps[KEYS + ["prefix_text"]], on=KEYS, how="left")
    return M.reset_index(drop=True)


# -- feature providers -----------------------------------------------------------

FEATURES: dict = {}


def register_feature(name):
    def deco(cls):
        FEATURES[name] = cls
        return cls
    return deco


class Feature:
    """self.raw: row-aligned with M (dense float32 (n, d), or an object array of texts)."""
    def prep(self, A_tr, A_te):
        """Fit on train rows only; return model inputs (default: standardise dense)."""
        mu, sd = A_tr.mean(0), A_tr.std(0)
        sd[sd < 1e-8] = 1.0
        Z_tr, Z_te = (A_tr - mu) / sd, (A_te - mu) / sd
        k = getattr(self, "pca", None)
        if not k:
            return Z_tr, Z_te
        # PCA fitted on train rows only; the k scores are standardised (equal footing for probes)
        Vt = np.linalg.svd(Z_tr, full_matrices=False)[2][:k]
        P_tr, P_te = Z_tr @ Vt.T, Z_te @ Vt.T
        s = P_tr.std(0); s[s < 1e-8] = 1.0
        return P_tr / s, P_te / s


@register_feature("act")
class Act(Feature):
    """layers: 1-indexed decoder layers (act_L21 <-> layers [21]); several are concatenated.
    Only those layers are read, via mmap."""
    def __init__(self, M, cfg, layers):
        for sh, rows in M.groupby("_shard").indices.items():
            a = np.load(Path(cfg["act_dir"]) / f"activations.{sh}.npy", mmap_mode="r")
            r = M._row.to_numpy()[rows]
            x = np.concatenate([a[r, L - 1, :] for L in layers], 1)
            if not hasattr(self, "raw"):
                self.raw = np.empty((len(M), x.shape[1]), np.float32)
            self.raw[rows] = x


@register_feature("prefix_len")
class PrefixLen(Feature):
    def __init__(self, M, cfg):
        self.raw = M.prefix_tokens.to_numpy(np.float32).reshape(-1, 1)


@register_feature("tfidf")
class Tfidf(Feature):
    def __init__(self, M, cfg):
        self.raw = M.prefix_text.fillna("").to_numpy(object)

    def prep(self, A_tr, A_te):
        return text_features(A_tr, A_te)          # same vectoriser as probe_checkpoints


@register_feature("npy")
class Npy(Feature):
    def __init__(self, M, cfg, path, keys):
        X = np.load(path, mmap_mode="r")
        K = pd.read_parquet(keys) if str(keys).endswith(".parquet") else pd.read_csv(keys)
        assert len(K) == len(X), f"{keys}: {len(K)} key rows vs {len(X)} feature rows"
        K = K[KEYS].copy(); K["fraction"] = K.fraction.round(6); K["_k"] = np.arange(len(K))
        assert not K.duplicated(KEYS).any(), f"{keys}: duplicate (example_id, cot_id, fraction)"
        q = M[KEYS].copy(); q["fraction"] = q.fraction.round(6)
        j = q.merge(K, on=KEYS, how="left")
        assert len(j) == len(M) and j._k.notna().all(), \
            f"{keys}: {int(j._k.isna().sum())} of our rows have no feature row"
        self.raw = np.asarray(X[j._k.to_numpy(int)], dtype=np.float32)


# -- probes ----------------------------------------------------------------------

PROBES: dict = {}


def register_probe(name):
    def deco(cls):
        PROBES[name] = cls
        return cls
    return deco


def get_svd(X, cache):
    """Economy SVD of a (standardised) training block, shared by every probe/target using it."""
    if "svd" not in cache:
        cache["svd"] = np.linalg.svd(X, full_matrices=False)
    return cache["svd"]


class Probe:
    kinds = ("reg", "bin")                       # targets this probe can handle

    def __init__(self, task, **hp):
        self.task, self.hp = task, hp

    def fit(self, X, y, groups=None):            # -> self
        raise NotImplementedError

    def predict(self, X):                        # reg: value; bin: P(y=1)
        raise NotImplementedError

    @classmethod
    def fit_path(cls, task, X, y, hps, cache, groups=None):
        """One fitted model per hp dict. Override to share work across the grid."""
        return [cls(task, **hp).fit(X, y, groups) for hp in hps]


class _Fn:                                       # wraps a predict function as a model
    def __init__(self, f): self.predict = f


@register_probe("ridge")
class Ridge(Probe):
    """hp: alpha. Dense inputs: whole alpha grid from one cached SVD (RidgePath)."""
    kinds = ("reg",)

    def fit(self, X, y, groups=None):
        from sklearn.linear_model import Ridge as SkRidge
        self.m = SkRidge(alpha=self.hp["alpha"]).fit(X, y)
        self.predict = self.m.predict
        return self

    @classmethod
    def fit_path(cls, task, X, y, hps, cache, groups=None):
        if not isinstance(X, np.ndarray):
            return super().fit_path(task, X, y, hps, cache, groups)
        U, s, Vt = get_svd(X, cache)
        rp = RidgePath.__new__(RidgePath)        # RidgePath state from the cached SVD
        rp.mu, rp.sd, rp.s, rp.Vt = 0.0, 1.0, s, Vt
        rp.ybar = float(y.mean()); rp.Uty = U.T @ (y - rp.ybar)
        return [_Fn(lambda Z, a=hp["alpha"]: rp.predict(Z, a)) for hp in hps]


@register_probe("logistic")
class Logistic(Probe):
    """hp: C. Dense inputs are rotated into the training row space (exact, much cheaper;
    see probe_checkpoints._rowspace)."""
    kinds = ("bin",)

    def fit(self, X, y, groups=None):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(max_iter=3000, C=self.hp["C"]).fit(X, y)
        self.predict = lambda Z: self.m.predict_proba(Z)[:, 1]
        return self

    @classmethod
    def fit_path(cls, task, X, y, hps, cache, groups=None):
        if not isinstance(X, np.ndarray):
            return super().fit_path(task, X, y, hps, cache, groups)
        U, s, Vt = get_svd(X, cache)
        keep = s > s.max() * 1e-10
        V, P = Vt[keep].T, U[:, keep] * s[keep]
        out = []
        for hp in hps:
            m = cls(task, **hp).fit(P, y)
            out.append(_Fn(lambda Z, m=m: m.predict(Z @ V)))
        return out


@register_probe("mlp")
class MLP(Probe):
    """One-hidden-layer torch MLP on CPU (dense inputs). hp: hidden, dropout, wd (AdamW decoupled
    weight decay), epochs, lr, batch, patience (0 = no early stopping, run all epochs),
    val_frac (share of training QUESTIONS held out for early stopping), seed.
    Regression targets are standardised internally; binary uses BCE-with-logits."""
    def __init__(self, task, hidden=8, dropout=0.0, wd=0.0, epochs=100, lr=1e-3, batch=256,
                 patience=10, val_frac=0.15, seed=0):
        super().__init__(task, hidden=hidden, dropout=dropout, wd=wd, epochs=epochs, lr=lr,
                         batch=batch, patience=patience, val_frac=val_frac, seed=seed)

    def fit(self, X, y, groups=None):
        import copy, torch
        h = SimpleNamespace(**self.hp)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
        torch.manual_seed(h.seed); rng = np.random.default_rng(h.seed)
        reg = self.task == "reg"
        self.ym, self.ys = (float(y.mean()), float(y.std()) or 1.0) if reg else (0.0, 1.0)
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor((y - self.ym) / self.ys, dtype=torch.float32)
        va = np.zeros(len(y), bool)
        if h.patience > 0:
            if groups is not None:
                u = np.unique(groups)
                va = np.isin(groups, rng.choice(u, max(1, int(h.val_frac * len(u))), replace=False))
            else:
                va[rng.choice(len(y), max(1, int(h.val_frac * len(y))), replace=False)] = True
        tr = np.where(~va)[0]
        self.net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], h.hidden), torch.nn.ReLU(),
                                       torch.nn.Dropout(h.dropout), torch.nn.Linear(h.hidden, 1))
        opt = torch.optim.AdamW(self.net.parameters(), lr=h.lr, weight_decay=h.wd)
        loss_fn = torch.nn.MSELoss() if reg else torch.nn.BCEWithLogitsLoss()
        best, best_state, bad = np.inf, None, 0
        for _ in range(h.epochs):
            self.net.train()
            for b in np.array_split(rng.permutation(tr), max(1, len(tr) // h.batch)):
                opt.zero_grad()
                loss_fn(self.net(Xt[b]).squeeze(1), yt[b]).backward()
                opt.step()
            if h.patience > 0:
                self.net.eval()
                with torch.no_grad():
                    v = float(loss_fn(self.net(Xt[va]).squeeze(1), yt[va]))
                if v < best:
                    best, best_state, bad = v, copy.deepcopy(self.net.state_dict()), 0
                else:
                    bad += 1
                    if bad >= h.patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            o = self.net(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()
        return o * self.ys + self.ym if self.task == "reg" else 1 / (1 + np.exp(-o))


@register_probe("rf")
class RF(Probe):
    """sklearn random forest (dense inputs). hp: n_estimators, max_depth, max_features,
    min_samples_leaf, n_jobs, seed."""
    def __init__(self, task, n_estimators=200, max_depth=None, max_features="sqrt",
                 min_samples_leaf=1, n_jobs=1, seed=0):
        super().__init__(task, n_estimators=n_estimators, max_depth=max_depth,
                         max_features=max_features, min_samples_leaf=min_samples_leaf,
                         n_jobs=n_jobs, seed=seed)

    def fit(self, X, y, groups=None):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        hp = dict(self.hp); hp["random_state"] = hp.pop("seed")
        self.m = (RandomForestRegressor if self.task == "reg" else RandomForestClassifier)(**hp)
        self.m.fit(X, y)
        self.predict = (self.m.predict if self.task == "reg"
                        else lambda Z: self.m.predict_proba(Z)[:, list(self.m.classes_).index(1)])
        return self


@register_probe("hgb")
class HGB(Probe):
    """sklearn HistGradientBoosting (dense inputs). hp: learning_rate, max_iter, max_leaf_nodes,
    min_samples_leaf, l2_regularization, max_features, val_frac, seed. Early stopping (if
    val_frac > 0) holds out val_frac of the training QUESTIONS, never a random row split."""
    def __init__(self, task, learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
                 min_samples_leaf=20, l2_regularization=0.0, max_features=1.0, val_frac=0.15, seed=0):
        super().__init__(task, learning_rate=learning_rate, max_iter=max_iter,
                         max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                         l2_regularization=l2_regularization, max_features=max_features,
                         val_frac=val_frac, seed=seed)

    def fit(self, X, y, groups=None):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        hp = dict(self.hp); vf = hp.pop("val_frac"); seed = hp.pop("seed")
        tr, kw = np.arange(len(y)), {}
        if vf > 0 and groups is not None:
            u = np.unique(groups)
            va = np.isin(groups, np.random.default_rng(seed).choice(u, max(1, int(vf * len(u))), replace=False))
            tr, kw = np.where(~va)[0], dict(X_val=X[va], y_val=y[va])
        cls = HistGradientBoostingRegressor if self.task == "reg" else HistGradientBoostingClassifier
        self.m = cls(**hp, early_stopping=bool(kw), random_state=seed).fit(X[tr], y[tr], **kw)
        self.predict = (self.m.predict if self.task == "reg"
                        else lambda Z: self.m.predict_proba(Z)[:, list(self.m.classes_).index(1)])
        return self


# -- folds, views, nested selection -----------------------------------------------

def _group_folds(n, g, k):
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=k).split(np.zeros(n), g, g))


def fixed_fold_of(run_dir: Path, k: int, seed: int) -> dict:
    """example_id -> outer fold, from ALL questions in examples.csv (not the rows present), so a
    question has the same fold at every checkpoint, for every feature and probe, and at 0%."""
    ids = np.sort(pd.read_csv(run_dir / "raw/examples.csv").example_id.unique())
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {int(q): int(r % k) for q, r in zip(ids[perm], range(len(ids)))}


def _fixed_folds(g, fold_of, k):
    f = np.array([fold_of[int(q)] for q in g])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


class Views:
    """Everything that depends on (feature, row subset, folds) but not on probe/target:
    prepared train/test inputs per outer fold and per inner fold, plus an SVD cache."""
    def __init__(self, feat, A, g, outer, n_inner=3):
        self.splits = []
        for tr, te in outer:
            Xtr, Xte = feat.prep(A[tr], A[te])
            sp = SimpleNamespace(tr=tr, te=te, Xtr=Xtr, Xte=Xte, cache={}, inners=[])
            k = min(n_inner, len(np.unique(g[tr])))
            if k >= 2:
                for itr, ite in _group_folds(len(tr), g[tr], k):
                    Itr, Ite = feat.prep(A[tr][itr], A[tr][ite])
                    sp.inners.append(SimpleNamespace(itr=itr, ite=ite, Xtr=Itr, Xte=Ite, cache={}))
            self.splits.append(sp)


def score(task, y, p):
    return _r2(y, p) if task == "reg" else float(((p >= 0.5).astype(int) == y).mean())


def run_combo(views, cls, hps, y, g, task):
    """-> (oof predictions, mean train score, chosen hp per outer fold)."""
    oof, train, chosen = np.full(len(y), np.nan), [], []
    for sp in views.splits:
        ytr, gtr = y[sp.tr], g[sp.tr]
        best = hps[len(hps) // 2]
        if len(hps) > 1 and sp.inners:
            sc = np.zeros(len(hps))
            for inn in sp.inners:
                if task == "bin" and len(np.unique(ytr[inn.itr])) < 2:
                    continue
                ms = cls.fit_path(task, inn.Xtr, ytr[inn.itr], hps, inn.cache, gtr[inn.itr])
                sc += [score(task, ytr[inn.ite], m.predict(inn.Xte)) for m in ms]
            best = hps[int(np.argmax(sc))]
        m = cls.fit_path(task, sp.Xtr, ytr, [best], sp.cache, gtr)[0]
        oof[sp.te] = m.predict(sp.Xte)
        train.append(score(task, ytr, m.predict(sp.Xtr)))
        chosen.append(best)
    return oof, float(np.mean(train)), chosen


# -- config / driver ---------------------------------------------------------------

def expand(grid, fixed):
    """{hp: [values] | {logspace: [a,b,n]}} crossed, plus fixed -> list of hp dicts."""
    vals = {k: (np.logspace(v["logspace"][0], v["logspace"][1], int(v["logspace"][2])).tolist()
                if isinstance(v, dict) else v) for k, v in (grid or {}).items()}
    return [dict(fixed or {}, **dict(zip(vals, c))) for c in itertools.product(*vals.values())]


def append_csv(df, path):
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def merge(out):
    fs = sorted(glob.glob(str(out / "results*.csv")))
    df = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(
        ["feature", "probe", "fraction", "target"], keep="last")
    df.to_csv(out / "results.csv", index=False)
    print(f"wrote {out/'results.csv'} ({len(df)} rows from {len(fs)} files)")


def export(out, dest, pred, bases):
    """Wide predictions.csv in the format bootstrap_ci.py reads:
    <target>__true, <target>__pred, <target>__pred_<name> per baseline."""
    P = pd.concat([pd.read_csv(f) for f in glob.glob(str(out / "preds*.csv"))])
    wide = None
    for col, combo in [("pred", pred)] + [(f"pred_{k}", v) for k, v in bases]:
        f, p = combo.split("/")
        s = P[(P.feature == f) & (P.probe == p)]
        assert len(s), f"no predictions for {combo}"
        for t, d in s.groupby("target"):
            d = d.drop_duplicates(KEYS + ["target"]).set_index(KEYS)
            part = pd.DataFrame({f"{t}__{col}": d.y_pred})
            if col == "pred":
                part[f"{t}__true"] = d.y_true
            wide = part if wide is None else wide.join(part, how="outer")
    wide.reset_index().to_csv(dest, index=False)
    print(f"wrote {dest} ({len(wide)} rows)")


def main():
    import yaml
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", default="0/1", help="i/N: run units with index %% N == i")
    ap.add_argument("--out-dir")
    ap.add_argument("--max-questions", type=int)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--export"); ap.add_argument("--pred")
    ap.add_argument("--base", action="append", default=[], help="name=feature/probe")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    out = Path(a.out_dir or cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    if a.merge:
        return merge(out)
    if a.export:
        return export(out, a.export, a.pred, [b.split("=", 1) for b in a.base])
    si, sn = map(int, a.shard.split("/"))
    sfx = "" if sn == 1 else f".shard{si}of{sn}"

    M = load_meta(Path(cfg["act_dir"]), Path(cfg["run_dir"]))
    if a.max_questions:
        M = M[M.example_id.isin(sorted(M.example_id.unique())[:a.max_questions])].reset_index(drop=True)
    print(f"  {len(M)} (CoT, checkpoint) rows, {M.example_id.nunique()} questions")

    combos = []                                   # (feature, probe, save_preds), config order
    for r in cfg["runs"]:
        for f, p in itertools.product(r["features"], r["probes"]):
            if not any(c[:2] == (f, p) for c in combos):
                combos.append((f, p, r.get("preds", True)))
    done = set()
    for f in glob.glob(str(out / "results*.csv")):
        done |= {(r.feature, r.probe, round(r.fraction, 6), r.target)
                 for r in pd.read_csv(f).itertuples()}
    res_path, pred_path = out / f"results{sfx}.csv", out / f"preds{sfx}.csv"
    targets = cfg.get("targets") or REG_TARGETS + BIN_TARGETS
    k_out = cfg.get("folds", 5)                   # fixed_folds: false = old per-checkpoint GroupKFold
    fold_of = (fixed_fold_of(Path(cfg["run_dir"]), k_out, cfg.get("fold_seed", 0))
               if cfg.get("fixed_folds", True) else None)
    fractions = cfg.get("fractions") or sorted(M.fraction.unique())
    feats, unit = {}, -1
    for frac in fractions:
        Mf = np.where(np.isclose(M.fraction.to_numpy(), frac))[0]
        groups = {}                  # usable-row mask -> targets (grouping as in probe_checkpoints)
        for t in targets:
            ya = M[t].to_numpy(float)[Mf]; ok = ~np.isnan(ya)
            if ok.sum() < 50 or (t in BIN_TARGETS and len(np.unique(ya[ok])) < 2):
                print(f"   {frac} {t} skipped (n={int(ok.sum())})"); continue
            groups.setdefault(ok.tobytes(), []).append(t)
        for okb, tgts in groups.items():
            rows = Mf[np.frombuffer(okb, dtype=bool)]
            g = M.example_id.to_numpy()[rows]
            outer = (_fixed_folds(g, fold_of, k_out) if fold_of is not None
                     else _group_folds(len(rows), g, k_out))
            for fname in dict.fromkeys(c[0] for c in combos):
                todo = [(p, t, sv) for f, p, sv in combos if f == fname for t in tgts
                        if ("bin" if t in BIN_TARGETS else "reg")
                        in PROBES[cfg["probes"][p]["class"]].kinds]
                if not todo:
                    continue
                unit += 1                 # numbered BEFORE the done-filter, so shards agree
                if unit % sn != si:
                    continue
                todo = [c for c in todo if (fname, c[0], round(frac, 6), c[1]) not in done]
                if not todo:
                    continue
                if fname not in feats:
                    fc = dict(cfg["features"][fname]); ty = fc.pop("type"); k = fc.pop("pca", None)
                    print(f"  loading feature {fname} ({ty})")
                    feats[fname] = FEATURES[ty](M, cfg, **fc) if fc else FEATURES[ty](M, cfg)
                    feats[fname].pca = k
                feat = feats[fname]
                t0 = time.time()
                views = Views(feat, feat.raw[rows], g, outer)
                print(f"=== {frac} {fname} n={len(rows)} views {time.time()-t0:.0f}s ===")
                for p, t, save in todo:
                    pc = cfg["probes"][p]; cls = PROBES[pc["class"]]
                    task = "bin" if t in BIN_TARGETS else "reg"
                    y = M[t].to_numpy(float)[rows]
                    yi = y.astype(int) if task == "bin" else y
                    t0 = time.time()
                    oof, tr_s, chosen = run_combo(views, cls, expand(pc.get("grid"), pc.get("fixed")),
                                                  yi, g, task)
                    row = dict(feature=fname, probe=p, fraction=frac, target=t, n=len(rows),
                               metric="acc" if task == "bin" else "R2",
                               test=score(task, yi, oof), train=tr_s, test_auc=np.nan,
                               hp=json.dumps(chosen), runtime_s=round(time.time() - t0, 1))
                    if task == "bin":
                        from sklearn.metrics import roc_auc_score
                        row["test_auc"] = float(roc_auc_score(yi, oof))
                    append_csv(pd.DataFrame([row]), res_path)
                    if save:
                        pr = M.iloc[rows][KEYS].copy()
                        pr.insert(0, "target", t); pr.insert(0, "probe", p); pr.insert(0, "feature", fname)
                        pr["y_true"], pr["y_pred"] = y, oof
                        append_csv(pr, pred_path)
                    print(f"   {p:<14}{t:<24}{row['metric']} test {row['test']:.4f} "
                          f"train {row['train']:.4f}  {row['runtime_s']}s")
                del views


if __name__ == "__main__":
    main()
