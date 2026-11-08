"""Sweep 7: push the winning direction -- per-member preproc diversity,
k sweep for the stack, logmad base, QT variants, meta-learner, passthrough."""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab.pipeline as P
from lab.pipeline import (SEED, load_data, compute_block_mask, build_stack,
                          cv_scores, report)
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

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
LOGMAD = lambda base=4.0, k=200: dict(mask="logmad", log=True, mask_base=base,
                                       block="trans", k=k, score="f")

rows = []
t0 = time.time()
CONFIGS = [
    ("WINNER svr-logmad200+gp200+xgb300", ["svr", "gp", "xgb"], BASE,
     {"svr": LOGMAD(), "xgb": dict(BASE, k=300)}),
    ("  + xgb k=229", ["svr", "gp", "xgb"], BASE,
     {"svr": LOGMAD(), "xgb": dict(BASE, k=229)}),
    ("  + gp k=229", ["svr", "gp", "xgb"], BASE,
     {"svr": LOGMAD(), "gp": dict(BASE, k=229), "xgb": dict(BASE, k=300)}),
    ("  + lgbm k=229 as 4th", ["svr", "gp", "xgb", "lgbm"], BASE,
     {"svr": LOGMAD(), "xgb": dict(BASE, k=300), "lgbm": dict(BASE, k=229)}),
    ("  logmad base 3.5", ["svr", "gp", "xgb"], BASE,
     {"svr": LOGMAD(3.5), "xgb": dict(BASE, k=300)}),
    ("  logmad base 5.0", ["svr", "gp", "xgb"], BASE,
     {"svr": LOGMAD(5.0), "xgb": dict(BASE, k=300)}),
    ("stack all k=229", ["svr", "gp", "xgb"], dict(BASE, k=229), None),
    ("stack all k=250", ["svr", "gp", "xgb"], dict(BASE, k=250), None),
    ("stack all k=210", ["svr", "gp", "xgb"], dict(BASE, k=210), None),
    ("WINNER + n_quantiles=1000", ["svr", "gp", "xgb"], dict(BASE, n_quantiles=1000),
     {"svr": dict(LOGMAD(), n_quantiles=1000), "xgb": dict(BASE, k=300, n_quantiles=1000)}),
]
for name, members, o, per in CONFIGS:
    try:
        m = build_stack(members, o, BLOCK, NAMES, per_member_o=per)
        s = cv_scores(m, X, y, 5, 2, seed=SEED)
        report(name, s, rows)
    except Exception as e:
        print(f"  {name:<44} FAILED: {type(e).__name__} {str(e)[:120]}", flush=True)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

# meta-learner + passthrough variants on the winner
def custom_stack(per, meta=None, passthrough=False):
    ests = []
    for mname in ["svr", "gp", "xgb"]:
        mo = per.get(mname, BASE)
        ests.append((mname, Pipeline([("pre", P._pre(mo, BLOCK, NAMES)),
                                      ("model", P._REGRESSORS[mname]())])))
    return StackingRegressor(estimators=ests,
                             final_estimator=meta or RidgeCV(alphas=P.RIDGE_ALPHAS),
                             cv=KFold(5, shuffle=True, random_state=SEED),
                             passthrough=passthrough, n_jobs=-1)

WINNER_PER = {"svr": LOGMAD(), "xgb": dict(BASE, k=300)}
for name, meta, pt in [
    ("WINNER meta ElasticNetCV", ElasticNetCV(alphas=np.logspace(-3, 3, 20), l1_ratio=[0.1, 0.5, 0.9], cv=5), False),
    ("WINNER passthrough=True", RidgeCV(alphas=P.RIDGE_ALPHAS), True),
]:
    try:
        m = custom_stack(WINNER_PER, meta=meta, passthrough=pt)
        s = cv_scores(m, X, y, 5, 2, seed=SEED)
        report(name, s, rows)
    except Exception as e:
        print(f"  {name:<44} FAILED: {type(e).__name__} {str(e)[:120]}", flush=True)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

# svr-hyperparam-diverse stack: 3 SVRs with different (C,gamma) + gp + xgb
def svr_variant(C, gamma, eps=0.005):
    old = dict(P.SVR_PARAMS)
    P.SVR_PARAMS.update({"C": C, "gamma": gamma, "epsilon": eps})
    est = Pipeline([("pre", P._pre(LOGMAD(), BLOCK, NAMES)), ("model", P._svr())])
    P.SVR_PARAMS.update(old)
    return est

try:
    ests = [("svr_a", svr_variant(26.1, 0.00825)),
            ("svr_b", svr_variant(80.0, 0.0104)),
            ("svr_c", svr_variant(8.0, 0.015)),
            ("gp", Pipeline([("pre", P._pre(BASE, BLOCK, NAMES)), ("model", P._gp())])),
            ("xgb", Pipeline([("pre", P._pre(dict(BASE, k=300), BLOCK, NAMES)), ("model", P._xgb())]))]
    m = StackingRegressor(estimators=ests, final_estimator=RidgeCV(alphas=P.RIDGE_ALPHAS),
                          cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    s = cv_scores(m, X, y, 5, 2, seed=SEED)
    report("svr x3 hyper-diverse + gp + xgb", s, rows)
except Exception as e:
    print(f"  svr-diverse FAILED: {e}", flush=True)

pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_csv(OUT / "sweep7.csv", index=False)
print(pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_string(index=False))
print(f"total {time.time()-t0:.0f}s")
