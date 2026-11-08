"""Sweep 2d: diverse-preprocessing stacks (the promising ensemble direction)."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab.pipeline import (SEED, load_data, compute_block_mask, build_stack,
                          cv_scores, report)

OUT = Path(__file__).parent / "results"
X, y, Xt, _, _ = load_data()
NAMES = np.asarray(X.columns, dtype=object)
BLOCK = compute_block_mask(pd.concat([X, Xt], axis=0).to_numpy(float))

BASE = dict(mask="adakurt", block="trans", k=180, score="f")
LOGMAD = dict(mask="logmad", log=True, block="trans", k=180, score="f")

CONFIGS = [
    ("stack2 ref svr+gp+xgb k180", ["svr", "gp", "xgb"], BASE, None),
    ("stack-diverse-k (trees k=300)", ["svr", "gp", "xgb", "lgbm"], BASE,
     {"xgb": dict(BASE, k=300), "lgbm": dict(BASE, k=300)}),
    ("stack-diverse-mask (svr logmad)", ["svr", "gp", "xgb"], BASE,
     {"svr": LOGMAD}),
    ("stack svr-logmad+gp+xgb-lgbm k300", ["svr", "gp", "xgb", "lgbm"], BASE,
     {"svr": LOGMAD, "xgb": dict(BASE, k=300), "lgbm": dict(BASE, k=300)}),
    ("stack2 k=200", ["svr", "gp", "xgb"], dict(BASE, k=200), None),
    ("stack svr+gp only", ["svr", "gp"], BASE, None),
    ("stack svr+xgb only", ["svr", "xgb"], BASE, None),
]

rows = []
t0 = time.time()
for name, members, o, per in CONFIGS:
    try:
        m = build_stack(members, o, BLOCK, NAMES, per_member_o=per)
        s = cv_scores(m, X, y, 5, 2, seed=SEED)
        report(name, s, rows)
    except Exception as e:
        print(f"  {name:<44} FAILED: {type(e).__name__} {str(e)[:120]}", flush=True)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_csv(OUT / "sweep2d_diverse.csv", index=False)
print("\n=== summary ===")
print(pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_string(index=False))
