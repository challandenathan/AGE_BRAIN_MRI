"""Per-member fine-tuning of the winner stack (the "fine-tuning part").

Each model family is tuned with RandomizedSearchCV (SVR, XGBoost) or
GridSearchCV (GP) over its *full member pipeline*, under honest
RepeatedKFold(5x2, seed 42). Results are cached in tuned_params.json —
delete that file to re-run the searches.
"""
from __future__ import annotations

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedKFold

from .pipeline import (MEMBER_SPECS, SEED, build_model, load_params, report)

CV = lambda: RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED)  # noqa: E731


def _kernel(name):
    base = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) if name == "rbf" \
        else Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    return ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(1e-2, (1e-6, 1e1))


def tune_svr(X, y, block_mask, names, params, n_iter=40, n_jobs=-1):
    pipe = build_model("svr", MEMBER_SPECS["svr"], block_mask, names, params)
    dist = {
        "model__regressor__C": np.logspace(0.5, 3, 30),
        "model__regressor__gamma": np.logspace(-3.2, -1.2, 30),
        "model__regressor__epsilon": [0.002, 0.005, 0.01, 0.02, 0.05],
        "pre__select__k": [180, 200, 220],
    }
    rs = RandomizedSearchCV(pipe, dist, n_iter=n_iter, cv=CV(), scoring="r2",
                            random_state=SEED, n_jobs=n_jobs)
    rs.fit(X, y)
    bp = rs.best_params_
    out = {"C": float(bp["model__regressor__C"]),
           "gamma": float(bp["model__regressor__gamma"]),
           "epsilon": float(bp["model__regressor__epsilon"])}
    spec = dict(MEMBER_SPECS["svr"])
    spec["k"] = int(bp["pre__select__k"])
    print(f"  SVR  best 5x2 R2 = {rs.best_score_:.4f}  params = {out}  k = {spec['k']}")
    return out, spec


def tune_xgb(X, y, block_mask, names, params, n_iter=30, n_jobs=-1):
    pipe = build_model("xgb", MEMBER_SPECS["xgb"], block_mask, names, params)
    dist = {
        "model__n_estimators": [400, 800, 1200],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.02, 0.03, 0.04, 0.06],
        "model__subsample": [0.6, 0.7, 0.8],
        "model__colsample_bytree": [0.4, 0.5, 0.6, 0.8],
        "model__reg_lambda": [1.0, 3.0, 6.0],
        "pre__select__k": [229, 300, 400],
    }
    rs = RandomizedSearchCV(pipe, dist, n_iter=n_iter, cv=CV(), scoring="r2",
                            random_state=SEED, n_jobs=n_jobs)
    rs.fit(X, y)
    bp = rs.best_params_
    out = {k.replace("model__", ""): (v.tolist() if hasattr(v, "tolist") else v)
           for k, v in bp.items() if k.startswith("model__")}
    out.update({"random_state": SEED, "n_jobs": -1})
    spec = dict(MEMBER_SPECS["xgb"])
    spec["k"] = int(bp["pre__select__k"])
    print(f"  XGB  best 5x2 R2 = {rs.best_score_:.4f}  params = {out}  k = {spec['k']}")
    return out, spec


def tune_gp(X, y, block_mask, names, params, n_jobs=-1):
    pipe = build_model("gp", MEMBER_SPECS["gp"], block_mask, names, params)
    grid = {
        "model__alpha": [1e-10, 1e-8, 1e-6],
        "model__kernel": [_kernel("matern2.5"), _kernel("rbf")],
        "pre__select__k": [180, 229],
    }
    gs = GridSearchCV(pipe, grid, cv=CV(), scoring="r2", n_jobs=n_jobs)
    gs.fit(X, y)
    bp = gs.best_params_
    # kernel is ConstantKernel * base + WhiteKernel (a Sum): dig out the base
    kern = bp["model__kernel"]
    base = kern.k1.k2 if hasattr(kern, "k1") else kern  # Sum -> Product -> base
    kern_name = "rbf" if isinstance(base, RBF) else "matern2.5"
    out = {"alpha": float(bp["model__alpha"]), "kernel": kern_name, "n_restarts": 2}
    spec = dict(MEMBER_SPECS["gp"])
    spec["k"] = int(bp["pre__select__k"])
    print(f"  GP   best 5x2 R2 = {gs.best_score_:.4f}  params = {out}  k = {spec['k']}")
    return out, spec


def tune_all(X, y, block_mask, names, n_jobs=-1, n_iter_svr=40, n_iter_xgb=30):
    """Fine-tune every model family; returns (params, specs) and is cached by
    the caller via save_params."""
    params = load_params()
    print("fine-tuning members under RepeatedKFold(5x2, seed 42) ...", flush=True)
    svr_p, svr_spec = tune_svr(X, y, block_mask, names, params, n_iter_svr, n_jobs)
    xgb_p, xgb_spec = tune_xgb(X, y, block_mask, names, params, n_iter_xgb, n_jobs)
    gp_p, gp_spec = tune_gp(X, y, block_mask, names, params, n_jobs)
    return {"svr": svr_p, "xgb": xgb_p, "gp": gp_p}, \
           {"svr": svr_spec, "xgb": xgb_spec, "gp": gp_spec}
