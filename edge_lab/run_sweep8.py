"""Sweep 8: refine the winner (svr-logmad + gp-k229 + xgb-k300) and try
group-of-features expert stacks + raw-feature tree members + pca_add."""
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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

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
LOGMAD229 = dict(LOGMAD200, k=229)
WINNER = {"svr": LOGMAD200, "gp": K229, "xgb": K300}

rows = []
t0 = time.time()

CONFIGS = [
    ("W svr-logmad200+gp229+xgb300 (ref)", ["svr", "gp", "xgb"], BASE, WINNER),
    ("  svr logmad k=229", ["svr", "gp", "xgb"], BASE,
     {"svr": LOGMAD229, "gp": K229, "xgb": K300}),
    ("  svr adakurt k=200", ["svr", "gp", "xgb"], BASE,
     {"svr": BASE, "gp": K229, "xgb": K300}),
    ("  svr nomask k=200", ["svr", "gp", "xgb"], BASE,
     {"svr": dict(mask="none", log=False, block="trans", k=200, score="f"), "gp": K229, "xgb": K300}),
    ("  xgb k=400", ["svr", "gp", "xgb"], BASE,
     {"svr": LOGMAD200, "gp": K229, "xgb": dict(BASE, k=400)}),
    ("  + xgb pca_add=20", ["svr", "gp", "xgb"], BASE,
     {"svr": LOGMAD200, "gp": K229, "xgb": dict(K300, pca_add=20)}),
    ("  gp pca_add=20 k229", ["svr", "gp", "xgb"], BASE,
     {"svr": LOGMAD200, "gp": dict(K229, pca_add=20), "xgb": K300}),
    ("  svr logmad pca_add=20", ["svr", "gp", "xgb"], BASE,
     {"svr": dict(LOGMAD200, pca_add=20), "gp": K229, "xgb": K300}),
    ("  all n_quantiles=1000", ["svr", "gp", "xgb"], dict(BASE, n_quantiles=1000),
     {"svr": dict(LOGMAD200, n_quantiles=1000), "gp": dict(K229, n_quantiles=1000),
      "xgb": dict(K300, n_quantiles=1000)}),
]
for name, members, o, per in CONFIGS:
    try:
        m = build_stack(members, o, BLOCK, NAMES, per_member_o=per)
        s = cv_scores(m, X, y, 5, 2, seed=SEED)
        report(name, s, rows)
    except Exception as e:
        print(f"  {name:<40} FAILED: {type(e).__name__} {str(e)[:100]}", flush=True)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

# --- group-of-features experts: cluster block by |corr|>0.5, per-group SVR on
# its top-15 features, stacked with the winner members ---
try:
    from scipy.stats import rankdata
    Xf = X.to_numpy(float)
    med = np.nanmedian(Xf, axis=0)
    Xi = np.where(np.isnan(Xf), med, Xf)
    R = rankdata(Xi, axis=0)
    C = np.corrcoef(R, rowvar=False); np.fill_diagonal(C, 0)
    A = np.abs(C)
    blk_idx = np.where(BLOCK)[0]
    sub = A[np.ix_(blk_idx, blk_idx)]
    nc, lab = connected_components(csr_matrix(sub > 0.5), directed=False)
    groups = [blk_idx[lab == c] for c in range(nc) if (lab == c).sum() >= 8]
    print(f"  block clusters at |corr|>0.5 with >=8 feats: {len(groups)}, sizes {sorted([len(g) for g in groups], reverse=True)[:10]}", flush=True)
    rho = np.array([abs(P._spearman_scores(Xi[:, [j]], y)[0]) for j in blk_idx])
    # per-group pipelines via fixed column-subset masks
    ests = []
    for gi, g in enumerate(groups):
        top = g[np.argsort(-rho[np.isin(blk_idx, g)])][:15]
        mask = np.zeros(len(NAMES), bool); mask[top] = True
        pre = P.make_preprocessor(dict(mask="adakurt", block="none", k=10000, score="f"), mask, NAMES)
        # BlockRestrictedSelector with block_mask=mask and k=10000 keeps exactly mask cols
        ests.append((f"grp{gi}", Pipeline([("pre", pre), ("model", P._svr())])))
    ests += [("svr", Pipeline([("pre", P._pre(LOGMAD200, BLOCK, NAMES)), ("model", P._svr())])),
             ("gp", Pipeline([("pre", P._pre(K229, BLOCK, NAMES)), ("model", P._gp())])),
             ("xgb", Pipeline([("pre", P._pre(K300, BLOCK, NAMES)), ("model", P._xgb())]))]
    m = StackingRegressor(estimators=ests, final_estimator=RidgeCV(alphas=P.RIDGE_ALPHAS),
                          cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    s = cv_scores(m, X, y, 5, 2, seed=SEED)
    report(f"group-experts ({len(groups)} svr groups) + svr + gp + xgb", s, rows)
except Exception as e:
    print(f"  group-experts FAILED: {type(e).__name__} {str(e)[:150]}", flush=True)

pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_csv(OUT / "sweep8.csv", index=False)
print(pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_string(index=False))
print(f"total {time.time()-t0:.0f}s")
