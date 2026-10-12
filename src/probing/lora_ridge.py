# R8: ridge / logistic (same grids + inner CV as ref_fixed) on the LoRA-ADAPTED layer-21 states saved by
# lora_probe.py --save-states. Fold i's states come from the LoRA trained on fold i's training questions only.
#   $PY src/probing/lora_ridge.py <lora run dir> <out dir>    -> <out dir>/preds.csv (lora_compare.py format)
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "src")
from probing.sweep import KEYS, Feature, Ridge, Logistic, Views, run_combo, fixed_fold_of, load_meta

RUN = "runs/goal1/gpqa25/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B"
ACT = "runs/goal2/gpqa25/activations_ckpt/gpqa_diamond/DeepSeek-R1-Distill-Qwen-7B"
T5 = ["answer_entropy_bits_mm", "p_mode", "rem_len_mean", "rem_len_std", "committed_by_here"]
src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
M = load_meta(Path(ACT), Path(RUN)).reset_index(drop=True)      # same row order as lora_probe.py
fold_of = fixed_fold_of(Path(RUN), 5, 0)
grid = {"reg": (Ridge, [dict(alpha=a) for a in np.logspace(0, 6, 13)]), "bin": (Logistic, [dict(C=c) for c in np.logspace(-4, 1, 6)])}
P = []
for f in sorted(src.glob("states.f*.fold*.npz")):
    z = np.load(f); rows, A = z["rows"], z["X"].astype(np.float32)
    fold = int(f.stem.split("fold")[1]); m = M.loc[rows].reset_index(drop=True)
    g = m.example_id.to_numpy(); te = np.array([fold_of[int(q)] == fold for q in g])
    for t in T5:
        ok = m[t].notna().to_numpy(); task = "bin" if t == "committed_by_here" else "reg"
        y = m[t].to_numpy()[ok]; y = y.astype(int) if task == "bin" else y.astype(float)
        cls, hps = grid[task]
        v = Views(Feature(), A[ok], g[ok], [(np.where(~te[ok])[0], np.where(te[ok])[0])])
        oof, _, chosen = run_combo(v, cls, hps, y, g[ok], task)
        q = m.loc[ok, KEYS][te[ok]].copy(); q.insert(0, "target", t)
        q["y_true"], q["y_pred"], q["hp"] = y[te[ok]], oof[te[ok]], str(chosen[0])
        P.append(q)
    print(f"{f.name}: {len(rows)} rows, {te.sum()} test", flush=True)
pd.concat(P).assign(feature="act_L21_lora", probe="ridge_logistic").to_csv(out / "preds.csv", index=False)
print("wrote", out / "preds.csv")
