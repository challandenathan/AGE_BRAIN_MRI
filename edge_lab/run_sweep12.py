"""Sweep 12: fine-tune the wide member's k, add knn member, score variants."""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab.pipeline as P
from lab.pipeline import (SEED, load_data, compute_block_mask, cv_scores, report)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV
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
LOW30 = dict(BASE, k=30, score="spearman")
NOMASK = lambda k=10000, score="f": dict(mask="none", log=False, block="none", k=k, score=score)

def stack(entries, meta=None):
    ests = [(nm, Pipeline([("pre", P._pre(o, bm, NAMES)), ("model", reg)]))
            for nm, o, bm, reg in entries]
    return StackingRegressor(estimators=ests,
                             final_estimator=meta or RidgeCV(alphas=P.RIDGE_ALPHAS),
                             cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)

WIN = [("svr", LOGMAD200, BLOCK, P._svr()), ("gp", K229, BLOCK, P._gp()),
       ("xgb", K300, BLOCK, P._xgb())]
LOW = ("low30", LOW30, BLOCK, P._svr())

rows = []
t0 = time.time()
def run(name, entries, meta=None):
    try:
        m = stack(entries, meta)
        s = cv_scores(m, X, y, 5, 2, seed=SEED)
        report(name, s, rows)
    except Exception as e:
        print(f"  {name:<48} FAILED: {type(e).__name__} {str(e)[:100]}", flush=True)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

run("BEST W+wide828+low30 (ref)", WIN + [("wsvr", NOMASK(), None, P._svr()), LOW])
for k in [500, 550, 650, 700, 750, 828]:
    run(f"  wide k={k} nomask + low30", WIN + [("wsvr", NOMASK(k), None, P._svr()), LOW])
run("  wide828 spearman + low30", WIN + [("wsvr", NOMASK(10000, "spearman"), None, P._svr()), LOW])
run("  wide828 nomask + low30 + knn", WIN + [("wsvr", NOMASK(), None, P._svr()), LOW,
     ("knn", dict(BASE, k=200), BLOCK, P._knn())])
run("  wide828 nomask, no low30, meta ENet", WIN + [("wsvr", NOMASK(), None, P._svr())],
    meta=ElasticNetCV(alphas=np.logspace(-3, 3, 20), l1_ratio=[0.1, 0.5, 0.9], cv=5))
run("  W+wide828+low30, drop xgb", [WIN[0], WIN[1], ("wsvr", NOMASK(), None, P._svr()), LOW])
run("  W+wide828+low30, drop gp", [WIN[0], WIN[2], ("wsvr", NOMASK(), None, P._svr()), LOW])
run("  W+wide828+low30, low30->low50", WIN + [("wsvr", NOMASK(), None, P._svr()),
     ("low50", dict(BASE, k=50, score="spearman"), BLOCK, P._svr())])
run("  W+wide828+low30, low30->low20", WIN + [("wsvr", NOMASK(), None, P._svr()),
     ("low20", dict(BASE, k=20, score="spearman"), BLOCK, P._svr())])

pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_csv(OUT / "sweep12.csv", index=False)
print(pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_string(index=False))
print(f"total {time.time()-t0:.0f}s")
