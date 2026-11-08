"""Sweep 2b: ensembles (stacks/votes) on the block preproc, 5x2 CV."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab.pipeline import (SEED, load_data, compute_block_mask, build_stack,
                          build_vote, cv_scores, report)

OUT = Path(__file__).parent / "results"
X, y, Xt, _, _ = load_data()
NAMES = np.asarray(X.columns, dtype=object)
BLOCK = compute_block_mask(pd.concat([X, Xt], axis=0).to_numpy(float))

BASE = dict(mask="adakurt", block="trans", k=180, score="f")

CONFIGS = [
    ("stack2 svr+gp+xgb block180", ["svr", "gp", "xgb"], BASE, None),
    ("vote svr+gp+xgb block180", ["svr", "gp", "xgb"], BASE, "vote"),
    ("stack3 +lgbm", ["svr", "gp", "xgb", "lgbm"], BASE, None),
    ("stack5 +lgbm+cat", ["svr", "gp", "xgb", "lgbm", "cat"], BASE, None),
    ("stack6 +histgb+et", ["svr", "gp", "xgb", "lgbm", "histgb", "et"], BASE, None),
    ("stack-diverse-k", ["svr", "gp", "xgb", "lgbm", "cat"], BASE,
     {"xgb": dict(BASE, k=300), "lgbm": dict(BASE, k=300), "cat": dict(BASE, k=300)}),
    ("stack-diverse-mask", ["svr", "gp", "xgb", "lgbm"], BASE,
     {"svr": dict(mask="logmad", log=True, block="trans", k=180, score="f"),
      "gp": dict(BASE),
      "xgb": dict(BASE, k=300), "lgbm": dict(BASE, k=300)}),
    ("stack2 k=200", ["svr", "gp", "xgb"], dict(BASE, k=200), None),
]

rows = []
t0 = time.time()
for name, members, o, per in CONFIGS:
    try:
        if per == "vote":
            m = build_vote(members, o, BLOCK, NAMES)
        else:
            m = build_stack(members, o, BLOCK, NAMES, per_member_o=per)
        s = cv_scores(m, X, y, 5, 2, seed=SEED)
        report(name, s, rows)
    except Exception as e:
        print(f"  {name:<40} FAILED: {type(e).__name__} {str(e)[:120]}", flush=True)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_csv(OUT / "sweep2b_ensembles.csv", index=False)
print("\n=== summary ===")
print(pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_string(index=False))
print(f"total {time.time()-t0:.0f}s")
