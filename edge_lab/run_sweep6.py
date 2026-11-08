"""Sweep 6: finalist stacks with tuned member params and k=200-ish blocks,
5x2 CV (5x3 for the very finalists in run_final.py)."""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab.pipeline as P
from lab.pipeline import (SEED, load_data, compute_block_mask, build_stack,
                          build_model, cv_scores, report)

OUT = Path(__file__).parent / "results"
X, y, Xt, _, _ = load_data()
NAMES = np.asarray(X.columns, dtype=object)
BLOCK = compute_block_mask(pd.concat([X, Xt], axis=0).to_numpy(float))

BASE = dict(mask="adakurt", block="trans", k=200, score="f")
LOGMAD = dict(mask="logmad", log=True, block="trans", k=200, score="f")

svr_t = json.load(open(OUT / "svr_tuned.json"))
xgb_t = json.load(open(OUT / "xgb_tuned.json"))
P.SVR_PARAMS.update({"C": svr_t["model__regressor__C"],
                     "gamma": svr_t["model__regressor__gamma"],
                     "epsilon": svr_t["model__regressor__epsilon"]})
P.XGB_OVERRIDES.update({k.replace("model__", ""): v for k, v in xgb_t.items()
                        if k != "pre__select__k"})
print("svr params:", P.SVR_PARAMS, " xgb overrides:", P.XGB_OVERRIDES, flush=True)

rows = []
t0 = time.time()
CONFIGS = [
    # (name, members, base opts, per-member opts)
    ("stack2 k=200 adakurt (ref)", ["svr", "gp", "xgb"], BASE, None),
    ("stack2 k=220", ["svr", "gp", "xgb"], dict(BASE, k=220), None),
    ("stack2 k=200 tuned members", ["svr", "gp", "xgb"], BASE, None),  # params already overridden
    ("stack tuned k-svr200 xgb229", ["svr", "gp", "xgb"], dict(BASE, k=229),
     {"svr": dict(BASE, k=200)}),
    ("stack svr-logmad k200 + gp + xgb k300", ["svr", "gp", "xgb"], BASE,
     {"svr": LOGMAD, "xgb": dict(BASE, k=300)}),
    ("stack svr-logmad+gp+xgb+lgbm k200", ["svr", "gp", "xgb", "lgbm"], BASE,
     {"svr": LOGMAD, "xgb": dict(BASE, k=300), "lgbm": dict(BASE, k=300)}),
    ("stack svr+gp+xgb+lgbm k200", ["svr", "gp", "xgb", "lgbm"], BASE, None),
    ("stack2 k=200 spearman", ["svr", "gp", "xgb"], dict(BASE, score="spearman"), None),
]
for name, members, o, per in CONFIGS:
    try:
        m = build_stack(members, o, BLOCK, NAMES, per_member_o=per)
        s = cv_scores(m, X, y, 5, 2, seed=SEED)
        report(name, s, rows)
    except Exception as e:
        print(f"  {name:<44} FAILED: {type(e).__name__} {str(e)[:120]}", flush=True)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_csv(OUT / "sweep6.csv", index=False)
print(pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_string(index=False))
