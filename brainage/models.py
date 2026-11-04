"""Model factories (subtask 3: age regression).

Every model is a *full* Pipeline that starts with the shared preprocessor, so
cross-validation re-fits the preprocessing inside every fold (no leakage) and
train/test always see identical transformations.

Available models (keys of MODELS):
    ridge   linear baseline
    xgb     gradient boosting
    histgb  sklearn histogram GBM (fast baseline)
    svr     support vector regression, RBF kernel, tuned hyperparameters
    stack   Ridge + boosted trees (XGBoost, else sklearn HistGBM) + GP,
            blended by a RidgeCV meta-learner on their out-of-fold predictions
            (the notebook's stacking idea, with the kernel fixed to an
            isotropic one — features are ~N(0,1) after the quantile transform,
            so a single length_scale is correct)
"""
import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .config import RIDGE_ALPHAS
from .preprocessing import make_preprocessor

try:  # xgboost needs a native OpenMP runtime that is not always present
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover - depends on the machine
    XGBRegressor = None
    HAS_XGB = False


def _with_pre(cfg, regressor):
    """Full pipeline = shared preprocessor + regressor."""
    return Pipeline([("pre", make_preprocessor(cfg)), ("model", regressor)])


def make_ridge(cfg):
    return _with_pre(cfg, RidgeCV(alphas=RIDGE_ALPHAS))


def _boost_regressor(cfg):
    """XGBoost when its native library loads, otherwise sklearn's HistGBM."""
    if HAS_XGB:
        return XGBRegressor(**cfg.xgb)
    return HistGradientBoostingRegressor(
        max_iter=cfg.xgb["n_estimators"], max_depth=cfg.xgb["max_depth"],
        learning_rate=cfg.xgb["learning_rate"],
        l2_regularization=cfg.xgb["reg_lambda"], random_state=cfg.seed)


def make_xgb(cfg):
    if not HAS_XGB:
        raise RuntimeError("xgboost is not importable on this machine "
                           "(missing OpenMP runtime); use --model histgb or stack")
    return _with_pre(cfg, XGBRegressor(**cfg.xgb))


def make_histgb(cfg):
    return _with_pre(cfg, HistGradientBoostingRegressor(
        max_iter=400, max_depth=3, learning_rate=0.05,
        l2_regularization=1.0, random_state=cfg.seed))


def make_svr(cfg):
    # y is standardized so that C / epsilon are on a meaningful scale
    svr = TransformedTargetRegressor(
        SVR(C=cfg.svr_C, gamma=cfg.svr_gamma, epsilon=cfg.svr_epsilon),
        transformer=StandardScaler(),
    )
    return _with_pre(cfg, svr)


def make_stack(cfg):
    """StackingRegressor over Ridge + XGBoost + GP on the same preprocessed
    features. The meta-learner sees only out-of-fold predictions of the base
    learners, so the blend itself is leak-free."""
    ridge = Pipeline([("pre", clone(make_preprocessor(cfg))),
                      ("model", RidgeCV(alphas=RIDGE_ALPHAS))])
    boost = Pipeline([("pre", clone(make_preprocessor(cfg))),
                      ("model", _boost_regressor(cfg))])
    gp = Pipeline([("pre", clone(make_preprocessor(cfg))),
                   ("model", GaussianProcessRegressor(
                       kernel=ConstantKernel(1.0, (1e-3, 1e3))
                              * Matern(length_scale=1.0,
                                       length_scale_bounds=(1e-2, 1e2), nu=2.5)
                              + WhiteKernel(noise_level=1e-2,
                                            noise_level_bounds=(1e-6, 1e1)),
                       normalize_y=True, alpha=1e-8,
                       n_restarts_optimizer=cfg.gp_n_restarts,
                       random_state=cfg.seed))])
    return StackingRegressor(
        estimators=[("ridge", ridge), ("boost", boost), ("gp", gp)],
        final_estimator=RidgeCV(alphas=RIDGE_ALPHAS),
        cv=cfg.inner_cv(),
        n_jobs=-1,
    )


MODELS = {
    "ridge": make_ridge,
    "histgb": make_histgb,
    "svr": make_svr,
    "stack": make_stack,
}
if HAS_XGB:
    MODELS["xgb"] = make_xgb


def build_model(name: str, cfg):
    """Build the full pipeline for a model name."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](cfg)
