"""Hyperparameter tuning for the final (vote) model — leak-free.

Each ensemble member is tuned with RandomizedSearchCV where the *whole
pipeline* (preprocessing included) is refit inside every fold. The vote
weights are then chosen on out-of-fold predictions of the tuned members.

Run from the CLI:  python solution.py --tune
Writes experiments/tuned_params.json and prints the best parameters.
"""
import itertools
import json

import numpy as np
from scipy.stats import loguniform
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict

from .models import HAS_XGB, build_model

# parameter distributions; keys are pipeline param paths
DISTS = {
    "svr": {
        "pre__select__k": [150, 180, 200],
        "model__regressor__C": loguniform(3, 3000),
        "model__regressor__gamma": loguniform(1e-3, 0.1),
        "model__regressor__epsilon": [0.01, 0.03, 0.05, 0.1],
    },
    "xgb": {
        "pre__select__k": [150, 180, 200],
        "model__n_estimators": [200, 400, 600, 800],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_lambda": [0.3, 1.0, 3.0],
    },
    "gp": {
        "pre__select__k": [150, 180, 200],
        "model__alpha": [1e-10, 1e-8, 1e-6, 1e-4],
    },
}
N_ITER = {"svr": 40, "xgb": 40, "gp": 12}


def tune_member(name, X, y, cfg, n_iter=None):
    """RandomizedSearchCV over one member's full pipeline. Returns the search."""
    model = build_model(name, cfg)
    search = RandomizedSearchCV(
        model, DISTS[name], n_iter=n_iter or N_ITER[name], cv=cfg.inner_cv(),
        scoring="r2", n_jobs=-1, random_state=cfg.seed, refit=True)
    search.fit(X, y)
    print(f"  {name:<4} best CV R2 = {search.best_score_:.4f}  "
          f"params = {search.best_params_}", flush=True)
    return search


def apply_best_params(name, cfg, best_params):
    """Write a member's best params back into the Config (in place)."""
    k = best_params.get("pre__select__k")
    if k is not None:
        cfg.n_features = int(k)
    if name == "svr":
        cfg.svr_C = float(best_params["model__regressor__C"])
        cfg.svr_gamma = float(best_params["model__regressor__gamma"])
        cfg.svr_epsilon = float(best_params["model__regressor__epsilon"])
    elif name == "xgb":
        cfg.xgb.update({k.replace("model__", ""): v
                        for k, v in best_params.items() if k.startswith("model__")})
    elif name == "gp":
        cfg.gp_alpha = float(best_params["model__alpha"])


def tune_vote_weights(X, y, cfg, members=("svr", "gp", "boost")):
    """Grid-search the vote weights on out-of-fold predictions of the members
    (each member's OOF comes from the same fold split, so the blend is honest).
    `boost` maps to the xgb/histgb pipeline used inside the vote."""
    from .models import make_gp, make_svr, make_xgb, make_histgb
    factories = {"svr": make_svr, "gp": make_gp,
                 "boost": (make_xgb if HAS_XGB else make_histgb)}
    oof = {}
    for name in members:
        m = factories[name](cfg)
        if hasattr(m, "n_jobs"):
            m.set_params(n_jobs=1)
        oof[name] = cross_val_predict(m, X, y, cv=cfg.inner_cv(), n_jobs=-1)
        print(f"  OOF {name:<5} R2 = {r2_score(y, oof[name]):.4f}", flush=True)

    best = (None, -np.inf)
    for w in itertools.product([0.0, 0.25, 0.5, 0.75, 1.0], repeat=len(members)):
        if sum(w) == 0:
            continue
        w = np.asarray(w) / sum(w)
        pred = sum(wi * oof[n] for wi, n in zip(w, members))
        s = r2_score(y, pred)
        if s > best[1]:
            best = (w, s)
    weights, score = best
    print(f"  best vote weights {dict(zip(members, weights.round(3)))} "
          f"-> OOF R2 = {score:.4f}", flush=True)
    return tuple(float(x) for x in weights), float(score)


def tune_all(X, y, cfg, members=("svr", "xgb", "gp")):
    """Tune every vote member, then the vote weights. Returns a JSON-able dict.

    k (n_features) is shared: the SVR member's best k wins, since it is the
    strongest member and the one the earlier k-sweep was run for."""
    out = {"seed": cfg.seed, "members": {}}
    k_svr = cfg.n_features
    for name in members:
        if name == "xgb" and not HAS_XGB:
            print("  xgb not importable on this machine — skipping", flush=True)
            continue
        search = tune_member(name, X, y, cfg)
        apply_best_params(name, cfg, search.best_params_)
        if name == "svr":
            k_svr = cfg.n_features
        out["members"][name] = {"best_cv_r2": float(search.best_score_),
                                "best_params": {k: (v if isinstance(v, (int, float, str)) else str(v))
                                                for k, v in search.best_params_.items()}}
    cfg.n_features = k_svr
    weights, wscore = tune_vote_weights(X, y, cfg)
    cfg.vote_weights = weights
    out["vote_weights"] = {"weights": dict(zip(("svr", "gp", "boost"), weights)),
                           "oof_r2": wscore}
    return out
