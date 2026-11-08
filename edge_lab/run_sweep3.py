"""Sweep 3: PLS on block, GP variants, and hyperparameter tuning of the
members (RandomizedSearchCV under 5x2)."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab.pipeline import (SEED, load_data, compute_block_mask, build_model,
                          cv_scores, report, make_preprocessor, _pre)
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.model_selection import cross_val_score

OUT = Path(__file__).parent / "results"
X, y, Xt, _, _ = load_data()
NAMES = np.asarray(X.columns, dtype=object)
BLOCK = compute_block_mask(pd.concat([X, Xt], axis=0).to_numpy(float))
BASE = dict(mask="adakurt", block="trans", k=180, score="f")

rows = []
t0 = time.time()

# --- PLS on block ---
for nc in [10, 30, 60, 100, 150]:
    m = Pipeline([("pre", _pre(dict(mask="adakurt", block="trans", k=10000), BLOCK, NAMES)),
                  ("model", PLSRegression(n_components=nc))])
    s = cv_scores(m, X, y, 5, 2, seed=SEED)
    report(f"pls block229 ncomp={nc}", s, rows)

# --- GP variants ---
def gp_model(kernel, alpha, restarts, k=180):
    o = dict(BASE, k=k)
    pre = _pre(o, BLOCK, NAMES)
    return Pipeline([("pre", pre),
                     ("model", GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                                        alpha=alpha, n_restarts_optimizer=restarts,
                                                        random_state=SEED))])

kernels = {
    "matern2.5": ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0,
                  length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(1e-2, (1e-6, 1e1)),
    "rbf": ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0,
             length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(1e-2, (1e-6, 1e1)),
    "matern1.5": ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0,
                   length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(1e-2, (1e-6, 1e1)),
}
for kn, kern in kernels.items():
    for alpha, rs in [(1e-10, 2), (1e-6, 2), (1e-10, 5)]:
        s = cv_scores(gp_model(kern, alpha, rs), X, y, 5, 2, seed=SEED)
        report(f"gp {kn} alpha={alpha:g} rs={rs}", s, rows)
s = cv_scores(gp_model(kernels["matern2.5"], 1e-10, 2, k=100), X, y, 5, 2, seed=SEED)
report("gp matern2.5 k=100", s, rows)

print(f"    ({time.time()-t0:.0f}s)", flush=True)

# --- SVR joint tuning with k (RandomizedSearchCV, 5x2) ---
svr_pipe = build_model("svr", BASE, BLOCK, NAMES)
param_dist = {
    "pre__select__k": [150, 170, 180, 200, 220],
    "model__regressor__C": np.logspace(1, 3, 25),
    "model__regressor__gamma": np.logspace(-3, -1, 25),
    "model__regressor__epsilon": [0.005, 0.01, 0.02, 0.05],
}
rs = RandomizedSearchCV(svr_pipe, param_dist, n_iter=40, cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED),
                        scoring="r2", random_state=SEED, n_jobs=-1)
rs.fit(X, y)
print(f"  SVR tuned: best R2={rs.best_score_:.4f} params={rs.best_params_}")
rows.append({"name": "svr tuned (RS 40)", "mean_r2": rs.best_score_, "std_r2": np.nan,
             "min_r2": np.nan, "max_r2": np.nan, "n_folds": 10})
import json
json.dump({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in rs.best_params_.items()},
          open(OUT / "svr_tuned.json", "w"), indent=1)

# --- XGB tuning ---
xgb_pipe = build_model("xgb", BASE, BLOCK, NAMES)
xgb_dist = {
    "pre__select__k": [150, 180, 229, 300],
    "model__n_estimators": [400, 800, 1200],
    "model__max_depth": [3, 4, 5, 6],
    "model__learning_rate": [0.01, 0.02, 0.04],
    "model__subsample": [0.6, 0.7, 0.8],
    "model__colsample_bytree": [0.5, 0.6, 0.8],
    "model__reg_lambda": [1.0, 3.0, 6.0],
}
rs2 = RandomizedSearchCV(xgb_pipe, xgb_dist, n_iter=30, cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED),
                         scoring="r2", random_state=SEED, n_jobs=-1)
rs2.fit(X, y)
print(f"  XGB tuned: best R2={rs2.best_score_:.4f} params={rs2.best_params_}")
rows.append({"name": "xgb tuned (RS 30)", "mean_r2": rs2.best_score_, "std_r2": np.nan,
             "min_r2": np.nan, "max_r2": np.nan, "n_folds": 10})
json.dump({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in rs2.best_params_.items()},
          open(OUT / "xgb_tuned.json", "w"), indent=1)

# --- LGBM tuning ---
lgbm_pipe = build_model("lgbm", BASE, BLOCK, NAMES)
lgbm_dist = {
    "pre__select__k": [180, 229, 300],
    "model__n_estimators": [600, 1000, 1500],
    "model__max_depth": [4, 5, 6, -1],
    "model__num_leaves": [31, 63, 127],
    "model__learning_rate": [0.02, 0.03, 0.05],
    "model__subsample": [0.6, 0.7, 0.8],
    "model__colsample_bytree": [0.5, 0.6, 0.8],
    "model__reg_lambda": [1.0, 3.0, 6.0],
}
rs3 = RandomizedSearchCV(lgbm_pipe, lgbm_dist, n_iter=30, cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED),
                         scoring="r2", random_state=SEED, n_jobs=-1)
rs3.fit(X, y)
print(f"  LGBM tuned: best R2={rs3.best_score_:.4f} params={rs3.best_params_}")
rows.append({"name": "lgbm tuned (RS 30)", "mean_r2": rs3.best_score_, "std_r2": np.nan,
             "min_r2": np.nan, "max_r2": np.nan, "n_folds": 10})
json.dump({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in rs3.best_params_.items()},
          open(OUT / "lgbm_tuned.json", "w"), indent=1)

pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_csv(OUT / "sweep3.csv", index=False)
print("\n=== summary ===")
print(pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_string(index=False))
print(f"total {time.time()-t0:.0f}s")
