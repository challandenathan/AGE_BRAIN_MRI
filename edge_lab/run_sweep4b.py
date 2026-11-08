"""Sweep 4b: self-training simulation with single models (SVR / XGB support
sample_weight directly) + a manual OOF-weighted vote of the two."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab.pipeline import SEED, load_data, compute_block_mask, build_model

OUT = Path(__file__).parent / "results"
X, y, Xt, _, _ = load_data()
NAMES = np.asarray(X.columns, dtype=object)
BLOCK = compute_block_mask(pd.concat([X, Xt], axis=0).to_numpy(float))
BASE = dict(mask="adakurt", block="trans", k=200, score="f")

from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV


def fit_predict(model, Xtr, ytr, Xva, sw=None):
    if sw is None:
        model.fit(Xtr, ytr)
    else:
        model.fit(Xtr, ytr, model__sample_weight=sw)
    return model.predict(Xva)


outer = RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED)
WEIGHTS = [0.0, 0.3, 0.7, 1.0]
res = {("svr", w): [] for w in WEIGHTS}
res.update({("xgb", w): [] for w in WEIGHTS})
res.update({("vote", w): [] for w in WEIGHTS})
res[("full-tr", 0.0)] = []

for tr, va in outer.split(X, y):
    A_idx, B_idx = next(iter(KFold(5, shuffle=True, random_state=SEED).split(X.iloc[tr], y[tr])))
    A, B = tr[A_idx], tr[B_idx]
    # stage 1: fit on A, pseudo-label B
    ms, mx = build_model("svr", BASE, BLOCK, NAMES), build_model("xgb", BASE, BLOCK, NAMES)
    ps = fit_predict(ms, X.iloc[A], y[A], X.iloc[B])
    px = fit_predict(mx, X.iloc[A], y[A], X.iloc[B])
    # vote weights from OOF on A
    from sklearn.model_selection import cross_val_predict
    oofs = cross_val_predict(build_model("svr", BASE, BLOCK, NAMES), X.iloc[A], y[A],
                             cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    oofx = cross_val_predict(build_model("xgb", BASE, BLOCK, NAMES), X.iloc[A], y[A],
                             cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    meta = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(np.column_stack([oofs, oofx]), y[A])
    wv = meta.coef_
    for w in WEIGHTS:
        sw = np.concatenate([np.ones(len(A)), np.full(len(B), w)])
        Xall = pd.concat([X.iloc[A], X.iloc[B]], axis=0)
        yall = np.concatenate([y[A], (wv[0] * ps + wv[1] * px) / wv.sum()])
        ps2 = fit_predict(build_model("svr", BASE, BLOCK, NAMES), Xall, yall, X.iloc[va], sw)
        px2 = fit_predict(build_model("xgb", BASE, BLOCK, NAMES), Xall, yall, X.iloc[va], sw)
        res[("svr", w)].append(r2_score(y[va], ps2))
        res[("xgb", w)].append(r2_score(y[va], px2))
        res[("vote", w)].append(r2_score(y[va], wv[0] * ps2 + wv[1] * px2))
    res[("full-tr", 0.0)].append(r2_score(y[va], fit_predict(
        build_model("svr", BASE, BLOCK, NAMES), X.iloc[tr], y[tr], X.iloc[va])))

print("=== self-training simulation (honest 5x2 outer CV) ===")
for (tag, w), sc in sorted(res.items()):
    print(f"  {tag:<6} w={w:.1f}: mean R2 = {np.mean(sc):.4f} +/- {np.std(sc):.4f}")
