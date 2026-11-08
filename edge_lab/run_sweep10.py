"""Sweep 10: WIDE members (all 828 features, no block filter) added to the
winner stack -- this gave 0.6084 in sweep8. Explore wide-member variants."""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab.pipeline as P
from lab.pipeline import (SEED, load_data, compute_block_mask, cv_scores,
                          report, build_model)
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
WIDE = lambda mask="adakurt": dict(mask=mask, log=(mask == "logmad"), block="none",
                                   k=10000, score="f")

def stack(entries):
    """entries: list of (name, opts, block_mask_or_None, regressor_factory)"""
    ests = [(nm, Pipeline([("pre", P._pre(o, bm, NAMES)), ("model", reg())]))
            for nm, o, bm, reg in entries]
    return StackingRegressor(estimators=ests, final_estimator=RidgeCV(alphas=P.RIDGE_ALPHAS),
                             cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)

WIN = [("svr", LOGMAD200, BLOCK, P._svr), ("gp", K229, BLOCK, P._gp), ("xgb", K300, BLOCK, P._xgb)]

rows = []
t0 = time.time()
def run(name, entries):
    try:
        m = stack(entries)
        s = cv_scores(m, X, y, 5, 2, seed=SEED)
        report(name, s, rows)
    except Exception as e:
        print(f"  {name:<46} FAILED: {type(e).__name__} {str(e)[:100]}", flush=True)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

# singles for reference
run("single svr all828 adakurt", [("svr", WIDE(), None, P._svr)])
run("W ref", WIN)
# wide members
run("W + svr wide828 adakurt", WIN + [("wsvr", WIDE(), None, P._svr)])
run("W + svr wide828 logmad", WIN + [("wsvr", WIDE("logmad"), None, P._svr)])
run("W + svr wide828 nomask", WIN + [("wsvr", WIDE("none"), None, P._svr)])
run("W + xgb wide828", WIN + [("wxgb", WIDE(), None, P._xgb)])
run("W + ridge wide828", WIN + [("wridge", WIDE(), None, P._ridge)])
run("W + svr wide828 + lowdim svr30", WIN + [("wsvr", WIDE(), None, P._svr),
     ("low30", dict(BASE, k=30, score="spearman"), BLOCK, P._svr)])
run("W + xgb wide828 + svr wide828", WIN + [("wxgb", WIDE(), None, P._xgb),
                                            ("wsvr", WIDE(), None, P._svr)])
run("W + svr wide828, svr wide with C=8", WIN + [("wsvr", WIDE(), None, P._svr)])

pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_csv(OUT / "sweep10.csv", index=False)
print(pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_string(index=False))
print(f"total {time.time()-t0:.0f}s")
