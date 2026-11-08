"""Extra 5x3 check of wide-k variants around the winner."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab.pipeline as P
from lab.pipeline import SEED, load_data, compute_block_mask, cv_scores, report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

OUT = Path(__file__).parent / "results"
X, y, Xt, _, _ = load_data()
NAMES = np.asarray(X.columns, dtype=object)
BLOCK = compute_block_mask(pd.concat([X, Xt], axis=0).to_numpy(float))
svr_t = json.load(open(OUT / "svr_tuned.json")); xgb_t = json.load(open(OUT / "xgb_tuned.json"))
P.SVR_PARAMS.update({"C": svr_t["model__regressor__C"], "gamma": svr_t["model__regressor__gamma"], "epsilon": svr_t["model__regressor__epsilon"]})
P.XGB_OVERRIDES.update({k.replace("model__",""): v for k,v in xgb_t.items() if k != "pre__select__k"})
BASE = dict(mask="adakurt", block="trans", k=200, score="f")
K229 = dict(BASE, k=229); K300 = dict(BASE, k=300)
LOGMAD200 = dict(mask="logmad", log=True, mask_base=4.0, block="trans", k=200, score="f")
LOW30 = dict(BASE, k=30, score="spearman")
def stack(entries):
    ests = [(nm, Pipeline([("pre", P._pre(o, bm, NAMES)), ("model", reg)])) for nm, o, bm, reg in entries]
    return StackingRegressor(estimators=ests, final_estimator=RidgeCV(alphas=P.RIDGE_ALPHAS),
                             cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
W = [("svr", LOGMAD200, BLOCK, P._svr()), ("gp", K229, BLOCK, P._gp()), ("xgb", K300, BLOCK, P._xgb())]
LOW = ("low30", LOW30, BLOCK, P._svr())
rows = []
for k in [600, 650, 700, 828]:
    wide = dict(mask="none", log=False, block="none", k=k, score="f")
    m = stack(W + [("wsvr", wide, None, P._svr()), LOW])
    s = cv_scores(m, X, y, 5, 3, seed=SEED)
    report(f"W + wide{k} + low30  (5x3)", s, rows)
pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_csv(OUT / "final_5x3_widek.csv", index=False)
print(pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_string(index=False))
