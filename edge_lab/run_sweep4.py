"""Sweep 4: (a) re-impute the ORIGINAL missing cells on the selected top-k
features (KNN / iterative) instead of the global median -- probe with a fixed
feature set (slight selection leakage, flagged; only to decide if worth a
leak-free pipeline version); (b) transductive self-training simulation --
does adding pseudo-labeled 'unseen' samples to the training fold improve
honest CV R2?
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab.pipeline import SEED, load_data, compute_block_mask, build_model, build_stack

OUT = Path(__file__).parent / "results"
X, y, Xt, _, _ = load_data()
NAMES = np.asarray(X.columns, dtype=object)
BLOCK = compute_block_mask(pd.concat([X, Xt], axis=0).to_numpy(float))
BASE = dict(mask="adakurt", block="trans", k=180, score="f")

from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor

svr = TransformedTargetRegressor(SVR(C=80.03, gamma=0.010403, epsilon=0.01),
                                 transformer=StandardScaler())

def oof_r2(model, Xk, tag):
    oofp = cross_val_predict(model, Xk, y, cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    r2 = 1 - ((y - oofp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  {tag}: OOF R2 = {r2:.4f}", flush=True)
    return r2

# ---------------------------------------------------------------------------
print("=== (a) re-imputation of original NaNs on the selected 180 (probe) ===", flush=True)
Xf = X.to_numpy(float)
nan_mask = np.isnan(Xf)
Xi = SimpleImputer(strategy="median").fit_transform(Xf)
Q = QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=SEED).fit_transform(Xi)
sel = SelectKBest(f_regression, k=180).fit(Q, y)
cols = sel.get_support()
Q180_med = Q[:, cols]                       # median-imputed reference
Q180_nan = Q180_med.copy()
Q180_nan[nan_mask[:, cols]] = np.nan        # original missing cells restored

oof_r2(svr, Q180_med, "svr top180 median (ref)")
for k in [3, 5, 10]:
    oof_r2(svr, KNNImputer(n_neighbors=k).fit_transform(Q180_nan), f"svr top180 KNN{k} re-impute")
oof_r2(svr, IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=SEED)
        .fit_transform(Q180_nan), "svr top180 iterative re-impute")
# same for xgb
xgb = XGBRegressor(n_estimators=800, max_depth=4, learning_rate=0.02, subsample=0.7,
                   colsample_bytree=0.6, reg_lambda=3.0, random_state=SEED, n_jobs=-1)
oof_r2(xgb, Q180_med, "xgb top180 median (ref)")
for k in [5, 10]:
    oof_r2(xgb, KNNImputer(n_neighbors=k).fit_transform(Q180_nan), f"xgb top180 KNN{k} re-impute")

# ---------------------------------------------------------------------------
# (b) self-training simulation
# ---------------------------------------------------------------------------
print("\n=== (b) self-training simulation (honest outer CV) ===", flush=True)

def make_stack():
    return build_stack(["svr", "xgb"], BASE, BLOCK, NAMES)

outer = RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED)
results = {w: [] for w in [0.0, 0.3, 0.7, 1.0]}
for tr, va in outer.split(X, y):
    inner = KFold(5, shuffle=True, random_state=SEED)
    A_idx, B_idx = next(iter(inner.split(X.iloc[tr], y[tr])))
    A, B = tr[A_idx], tr[B_idx]
    m0 = make_stack()
    m0.fit(X.iloc[A], y[A])
    predB = m0.predict(X.iloc[B])
    for w in results:
        m = make_stack()
        Xall = pd.concat([X.iloc[A], X.iloc[B]], axis=0)
        yall = np.concatenate([y[A], predB])
        sw = np.concatenate([np.ones(len(A)), np.full(len(B), w)])
        try:
            m.fit(Xall, yall, model__sample_weight=sw)
        except Exception as e:
            # stacking may not route sample_weight; fall back to per-member fit
            print("  sample_weight routing failed:", type(e).__name__, str(e)[:80])
            break
        pred = m.predict(X.iloc[va])
        results[w].append(r2_score(y[va], pred))
    # reference: train on full outer-train (no pseudo)
    mref = make_stack()
    mref.fit(X.iloc[tr], y[tr])
    results.setdefault("full-tr-ref", []).append(r2_score(y[va], mref.predict(X.iloc[va])))

for w, sc in results.items():
    if sc:
        print(f"  w={w}: mean R2 = {np.mean(sc):.4f} +/- {np.std(sc):.4f}", flush=True)
