#!/usr/bin/env python3
"""Multi-layer features (configs/sweep_multilayer.yaml) vs our layer-21 ridge: same question bootstrap and
paired differences as round3_ci.py (it is reused with a different SPEC). Mean over 10-90%.

    sbatch -p small -c 2 --mem=16G --wrap '. ./env.sh; $PY src/viz/multilayer_ci.py'
Outputs: runs/goal2/gpqa25/sweep/_multilayer/{ci_scores.csv, ci_paired.csv}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import round3_ci as r

r.OUT = r.S / "_multilayer"; r.OUT.mkdir(exist_ok=True)
r.SPEC = {
    "Our model L21": ("ref_fixed", "act_L21"),
    "L21 PCA64": ("4b_pca", "act_L21_pca64"),
    "L21 PCA192": ("multilayer", "act_L21_pca192"),
    "L14 PCA64": ("multilayer", "act_L14_pca64"),
    "L14 PCA192": ("multilayer", "act_L14_pca192"),
    "L13-15 PCA64 each": ("multilayer", "act_L13_15_pp64"),
    "L20-22 PCA64 each": ("multilayer", "act_L20_22_pp64"),
    "L14 full": ("multilayer", "act_L14"),
    "L13-15 full": ("multilayer", "act_L13_15"),
    "L20-22 full": ("multilayer", "act_L20_22"),
    "TF-IDF": ("ref_fixed", "tfidf"),
}
r.REFS = ["Our model L21"]

if __name__ == "__main__":
    r.main()
