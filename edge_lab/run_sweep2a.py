"""Sweep 2a: single models on the block preproc (5x2 CV), PCA-of-block and
interaction variants."""
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
print(f"block: {BLOCK.sum()} features")

BASE = dict(mask="adakurt", block="trans", k=180, score="f")

CONFIGS = []
# singles on block k=180
for m in ["svr", "gp", "xgb", "lgbm", "cat", "histgb", "et", "knn", "ridge", "mlp"]:
    CONFIGS.append((f"{m} block180", dict(BASE), m))
# k variants for trees / svr
for k in [150, 200, 229, 300]:
    CONFIGS.append((f"svr block k={k}", dict(BASE, k=k), "svr"))
for k in [229, 300]:
    CONFIGS.append((f"xgb block k={k}", dict(BASE, k=k), "xgb"))
    CONFIGS.append((f"lgbm block k={k}", dict(BASE, k=k), "lgbm"))
CONFIGS.append(("xgb all832 nomask-sel", dict(mask="adakurt", block="none", k=10000), "xgb"))
CONFIGS.append(("lgbm all832", dict(mask="adakurt", block="none", k=10000), "lgbm"))

rows = []
t0 = time.time()
for name, o, model in CONFIGS:
    try:
        m = build_model(model, o, BLOCK, NAMES)
        s = cv_scores(m, X, y, 5, 2, seed=SEED)
        report(name, s, rows)
    except Exception as e:
        print(f"  {name:<44} FAILED: {type(e).__name__} {e}", flush=True)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_csv(OUT / "sweep2a_singles.csv", index=False)
print("\n=== summary ===")
print(pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_string(index=False))
