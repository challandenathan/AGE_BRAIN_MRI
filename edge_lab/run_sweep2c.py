"""Sweep 2c: feature engineering on the block -- PCA of block, PCs appended
to the selection, pairwise interactions of top features. Singles, 5x2 CV."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab.pipeline import (SEED, load_data, compute_block_mask, build_model,
                          cv_scores, report)

OUT = Path(__file__).parent / "results"
X, y, Xt, _, _ = load_data()
NAMES = np.asarray(X.columns, dtype=object)
BLOCK = compute_block_mask(pd.concat([X, Xt], axis=0).to_numpy(float))

BASE = dict(mask="adakurt", block="trans", k=180, score="f")

CONFIGS = [
    ("svr block180 (ref)", dict(BASE), "svr"),
    ("svr pca=30 block", dict(mask="adakurt", block="trans", pca=30), "svr"),
    ("svr pca=60 block", dict(mask="adakurt", block="trans", pca=60), "svr"),
    ("svr pca=100 block", dict(mask="adakurt", block="trans", pca=100), "svr"),
    ("xgb pca=60 block", dict(mask="adakurt", block="trans", pca=60), "xgb"),
    ("svr k180+pca30", dict(BASE, pca_add=30), "svr"),
    ("svr k180+pca10", dict(BASE, pca_add=10), "svr"),
    ("xgb k180+pca30", dict(BASE, pca_add=30), "xgb"),
    ("svr k180+inter m=10", dict(BASE, inter_m=10), "svr"),
    ("svr k180+inter m=20", dict(BASE, inter_m=20), "svr"),
    ("xgb k180+inter m=20", dict(BASE, inter_m=20), "xgb"),
    ("xgb k300+inter m=20", dict(BASE, k=300, inter_m=20), "xgb"),
    ("gp k180+pca30", dict(BASE, pca_add=30), "gp"),
]

rows = []
t0 = time.time()
for name, o, model in CONFIGS:
    try:
        m = build_model(model, o, BLOCK, NAMES)
        s = cv_scores(m, X, y, 5, 2, seed=SEED)
        report(name, s, rows)
    except Exception as e:
        print(f"  {name:<40} FAILED: {type(e).__name__} {str(e)[:120]}", flush=True)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_csv(OUT / "sweep2c_feateng.csv", index=False)
print("\n=== summary ===")
print(pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_string(index=False))
