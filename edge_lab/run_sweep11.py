"""Sweep 11: push the wide-member direction (W + svr wide nomask = 0.6117)."""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab.pipeline as P
from lab.pipeline import (SEED, load_data, compute_block_mask, cv_scores, report)
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
NOMASK = lambda k=10000: dict(mask="none", log=False, block="none", k=k, score="f")

def svr_with(C, gamma, eps=0.005):
    old = dict(P.SVR_PARAMS)
    P.SVR_PARAMS.update({"C": C, "gamma": gamma, "epsilon": eps})
    est = P._svr()
    P.SVR_PARAMS.update(old)
    return est

def stack(entries):
    ests = []
    for nm, o, bm, reg in entries:
        ests.append((nm, Pipeline([("pre", P._pre(o, bm, NAMES)), ("model", reg)])))
    return StackingRegressor(estimators=ests, final_estimator=RidgeCV(alphas=P.RIDGE_ALPHAS),
                             cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)

WIN = [("svr", LOGMAD200, BLOCK, P._svr()), ("gp", K229, BLOCK, P._gp()),
       ("xgb", K300, BLOCK, P._xgb())]
WIDE_NOMASK = ("wsvr", NOMASK(), None, P._svr())

rows = []
t0 = time.time()
def run(name, entries):
    try:
        m = stack(entries)
        s = cv_scores(m, X, y, 5, 2, seed=SEED)
        report(name, s, rows)
    except Exception as e:
        print(f"  {name:<48} FAILED: {type(e).__name__} {str(e)[:100]}", flush=True)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

run("W + wide svr nomask (ref)", WIN + [WIDE_NOMASK])
run("  wide svr C=8 gamma=0.02", WIN + [("wsvr", NOMASK(), None, svr_with(8.0, 0.02))])
run("  wide svr C=200 gamma=0.005", WIN + [("wsvr", NOMASK(), None, svr_with(200.0, 0.005))])
run("  wide svr k=600 nomask", WIN + [("wsvr", NOMASK(600), None, P._svr())])
run("  wide svr k=400 nomask", WIN + [("wsvr", NOMASK(400), None, P._svr())])
run("  wide xgb nomask", WIN + [("wxgb", NOMASK(), None, P._xgb())])
run("  wide svr nomask + wide svr adakurt", WIN + [("wsvr", NOMASK(), None, P._svr()),
     ("wsvr2", dict(mask="adakurt", block="none", k=10000, score="f"), None, P._svr())])
run("  wide svr nomask + lowdim svr30", WIN + [("wsvr", NOMASK(), None, P._svr()),
     ("low30", dict(BASE, k=30, score="spearman"), BLOCK, P._svr())])
run("  wide svr nomask + wide xgb nomask", WIN + [("wsvr", NOMASK(), None, P._svr()),
     ("wxgb", NOMASK(), None, P._xgb())])
run("5-mem: + low30 + wide", WIN + [("wsvr", NOMASK(), None, P._svr()),
     ("low30", dict(BASE, k=30, score="spearman"), BLOCK, P._svr())])
run("W no-gp + wide svr nomask + low30", [("svr", LOGMAD200, BLOCK, P._svr()),
     ("xgb", K300, BLOCK, P._xgb()), ("wsvr", NOMASK(), None, P._svr()),
     ("low30", dict(BASE, k=30, score="spearman"), BLOCK, P._svr())])

pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_csv(OUT / "sweep11.csv", index=False)
print(pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_string(index=False))
print(f"total {time.time()-t0:.0f}s")
