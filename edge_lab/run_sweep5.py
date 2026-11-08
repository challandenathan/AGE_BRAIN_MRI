"""Sweep 5: variance-reduction ensembles -- bagged SVR, seed-averaged stacks,
kernel ridge with tuned gamma, and a 2-seed stack average."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab.pipeline import (SEED, load_data, compute_block_mask, build_model,
                          build_stack, cv_scores, report, _pre, XGB_PARAMS)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingRegressor, VotingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

OUT = Path(__file__).parent / "results"
X, y, Xt, _, _ = load_data()
NAMES = np.asarray(X.columns, dtype=object)
BLOCK = compute_block_mask(pd.concat([X, Xt], axis=0).to_numpy(float))
BASE = dict(mask="adakurt", block="trans", k=180, score="f")

rows = []
t0 = time.time()

# --- bagged SVR ---
svr_pipe = build_model("svr", BASE, BLOCK, NAMES)
bag = BaggingRegressor(svr_pipe, n_estimators=25, max_samples=0.8, random_state=SEED, n_jobs=-1)
s = cv_scores(bag, X, y, 5, 2, seed=SEED)
report("bagged svr x25 (80%)", s, rows)

# --- kernel ridge tuned gamma (5x2) ---
kr_pipe = Pipeline([("pre", _pre(BASE, BLOCK, NAMES)),
                    ("model", KernelRidge(kernel="rbf"))])
kr_dist = {"model__alpha": np.logspace(-3, 1, 12), "model__gamma": np.logspace(-4, -1, 12)}
rs = RandomizedSearchCV(kr_pipe, kr_dist, n_iter=24, cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED),
                        scoring="r2", random_state=SEED, n_jobs=-1)
rs.fit(X, y)
print(f"  kernel-ridge tuned: {rs.best_score_:.4f} {rs.best_params_}")
rows.append({"name": "kernel-ridge tuned", "mean_r2": rs.best_score_, "std_r2": np.nan,
             "min_r2": np.nan, "max_r2": np.nan, "n_folds": 10})

# --- seed-averaged stack2: average OOF/test predictions of 3 stacks with
# different inner-CV seeds (variance reduction of the ensemble itself) ---
import lab.pipeline as P

def stack_with_seed(seed):
    ests = []
    for m in ["svr", "gp", "xgb"]:
        ests.append((m, Pipeline([("pre", _pre(BASE, BLOCK, NAMES)), ("model", P._REGRESSORS[m]())])))
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import RidgeCV
    return StackingRegressor(estimators=ests, final_estimator=RidgeCV(alphas=P.RIDGE_ALPHAS),
                             cv=KFold(5, shuffle=True, random_state=seed), n_jobs=-1)

# evaluate seed-averaged stack via manual CV (cross_val_score can't average models)
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED)
scores1, scores3 = [], []
for tr, va in cv.split(X, y):
    preds = []
    for sd in [42]:
        m = stack_with_seed(sd)
        m.fit(X.iloc[tr], y[tr])
        preds.append(m.predict(X.iloc[va]))
        scores1.append(r2_score(y[va], preds[-1]))
    for sd in [7, 123]:
        m = stack_with_seed(sd)
        m.fit(X.iloc[tr], y[tr])
        preds.append(m.predict(X.iloc[va]))
    scores3.append(r2_score(y[va], np.mean(preds, axis=0)))
report("stack2 seed42 (manual)", np.array(scores1), rows)
report("stack2 avg seeds 42/7/123", np.array(scores3), rows)

print(f"    ({time.time()-t0:.0f}s)", flush=True)
pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_csv(OUT / "sweep5.csv", index=False)
print(pd.DataFrame(rows).sort_values("mean_r2", ascending=False).to_string(index=False))
