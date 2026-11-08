"""Sweep 9: exploit the low-dim expert member. Winner stack + one (or two)
low-dimensional members (SVR/ridge on top-k features only)."""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab.pipeline as P
from lab.pipeline import (SEED, load_data, compute_block_mask, build_stack,
                          cv_scores, report)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

OUT = Path(__file__).parent / "results"
X, y, Xt, _, _ = load_data()
NAMES = np.asarray(X.columns, dtype=object)
BLOCK = compute_block_mask(pd.concat([X, Xt], axis=0).to_numpy(float))

svr_t = json.load(open(OUT / "svr_tuned.json"))
xgb_t = json.load(open(OUT / "xgb_tuned.json"))
P.SVR_PARAMS.update({"C": svr_t["model__regressor__C"],
                     "gamma": svr_t["model__regressor__gamma"],
                     "epsilon": svr_t["model__regressor__epsilon"]})
P.XGB_OVERRIDES.update({k.replace("model__", ""): v for k, v in xgb_t.items()
                        if k != "pre__select__k"})

BASE = dict(mask="adakurt", block="trans", k=200, score="f")
K229 = dict(BASE, k=229)
K300 = dict(BASE, k=300)
LOGMAD200 = dict(mask="logmad", log=True, mask_base=4.0, block="trans", k=200, score="f")

def stack(members, per):
    ests = []
    for name in members:
        o, reg = per[name]
        ests.append((name, Pipeline([("pre", P._pre(o, BLOCK, NAMES)), ("model", reg())])))
    return StackingRegressor(estimators=ests, final_estimator=RidgeCV(alphas=P.RIDGE_ALPHAS),
                             cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)

WIN = {"svr": (LOGMAD200, P._svr), "gp": (K229, P._gp), "xgb": (K300, P._xgb)}

rows = []
t0 = time.time()
def run(name, extra):
    per = dict(WIN)
    per.update(extra)
    members = list(per.keys())
    try:
        m = stack(members, per)
        s = cv_scores(m, X, y, 5, 2, seed=SEED)
        report(name, s, rows)
    except Exception as e:
        print(f"  {name:<46} FAILED: {type(e).__name__} {str(e)[:100]}", flush=True)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

def lowdim(k, mask="adakurt", model="svr"):
    o = dict(mask=mask, log=(mask == "logmad"), block="trans", k=k, score="spearman")
    reg = {"svr": P._svr, "ridge": P._ridge, "xgb": P._xgb}[model]
    return (o, reg)

# winner reference
run("W ref (svr-logmad200+gp229+xgb300)", {})
# + one lowdim svr member, sweep k
for k in [5, 10, 15, 20, 30, 50]:
    run(f"  + lowdim svr top-{k}", {f"low{k}": lowdim(k, model="svr")})
# lowdim ridge member
for k in [15, 30]:
    run(f"  + lowdim ridge top-{k}", {f"low{k}": lowdim(k, model="ridge")})
# lowdim with logmad mask
run("  + lowdim svr top-15 logmad", {"low15": lowdim(15, mask="logmad", model="svr")})
run("  + lowdim svr top-15 nomask", {"low15": lowdim(15, mask="none", model="svr")})
# lowdim xgb member
run("  + lowdim xgb top-15", {"low15": lowdim(15, model="xgb")})
# two lowdim members
run("  + lowdim svr 15 + 50", {"low15": lowdim(15, model="svr"), "low50": lowdim(50, model="svr")})
run("  + lowdim svr 10 + 30 + 50", {"low10": lowdim(10, model="svr"),
                                    "low30": lowdim(30, model="svr"),
                                    "low50": lowdim(50, model="svr")})

pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_csv(OUT / "sweep9.csv", index=False)
print(pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_string(index=False))
print(f"total {time.time()-t0:.0f}s")
