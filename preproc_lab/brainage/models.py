"""Model factories (subtask 3: age regression).

Every model is a *full* Pipeline that starts with the shared preprocessor, so
cross-validation re-fits the preprocessing inside every fold (no leakage) and
train/test always see identical transformations.

Available models (keys of MODELS):
    ridge       linear baseline
    xgb         gradient boosting (needs the native OpenMP runtime)
    histgb      sklearn histogram GBM (fast baseline)
    svr         support vector regression, RBF kernel, tuned hyperparameters
    stack       Ridge + boosted trees + GP, blended by a RidgeCV meta-learner
                on their out-of-fold predictions (the notebook's stacking
                idea, kernel fixed to isotropic — features are ~N(0,1) after
                the quantile transform, so one length_scale is correct)
    gp          Gaussian process alone
    knn         k-nearest neighbours (distance-weighted)
    rf / et     random forest / extra trees
    elasticnet  ElasticNetCV
    huber       robust linear regression
    pls         partial least squares (latent components)
"""
import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor,
                              VotingRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import ElasticNetCV, HuberRegressor, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor

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


def _gp_kernel():
    """Isotropic Matern + white noise: correct after the quantile transform,
    which maps every feature to ~N(0,1) (a single length_scale then applies)."""
    return (ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1)))


def make_gp(cfg):
    return _with_pre(cfg, GaussianProcessRegressor(
        kernel=_gp_kernel(), normalize_y=True, alpha=cfg.gp_alpha,
        n_restarts_optimizer=cfg.gp_n_restarts, random_state=cfg.seed))


def make_knn(cfg):
    # after the quantile transform, euclidean distance is meaningful
    return _with_pre(cfg, KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1))


def make_rf(cfg):
    return _with_pre(cfg, RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, max_features="sqrt",
        random_state=cfg.seed, n_jobs=-1))


def make_et(cfg):
    return _with_pre(cfg, ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=cfg.seed, n_jobs=-1))


def make_elasticnet(cfg):
    return _with_pre(cfg, ElasticNetCV(alphas=RIDGE_ALPHAS, l1_ratio=[0.1, 0.5, 0.9],
                                       cv=5, max_iter=10000, random_state=cfg.seed))


def make_huber(cfg):
    # robust linear model: insensitive to residual outliers
    return _with_pre(cfg, HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=500))


def make_pls(cfg):
    # latent-component regression: handles the correlated feature blocks
    return _with_pre(cfg, PLSRegression(n_components=30))


def make_stack(cfg):
    """StackingRegressor over Ridge + boosted trees + GP on the same
    preprocessed features. The meta-learner sees only out-of-fold predictions
    of the base learners, so the blend itself is leak-free."""
    ridge = Pipeline([("pre", clone(make_preprocessor(cfg))),
                      ("model", RidgeCV(alphas=RIDGE_ALPHAS))])
    boost = Pipeline([("pre", clone(make_preprocessor(cfg))),
                      ("model", _boost_regressor(cfg))])
    gp = Pipeline([("pre", clone(make_preprocessor(cfg))),
                   ("model", GaussianProcessRegressor(
                       kernel=_gp_kernel(), normalize_y=True, alpha=cfg.gp_alpha,
                       n_restarts_optimizer=cfg.gp_n_restarts,
                       random_state=cfg.seed))])
    return StackingRegressor(
        estimators=[("ridge", ridge), ("boost", boost), ("gp", gp)],
        final_estimator=RidgeCV(alphas=RIDGE_ALPHAS),
        cv=cfg.inner_cv(),
        n_jobs=-1,
    )


def make_vote(cfg):
    """Weighted average of SVR + GP + boosted trees — three model families
    whose errors are only ~0.9 correlated, so averaging buys diversity.
    Weights default to uniform; brainage.tuning picks them on OOF predictions."""
    return VotingRegressor([
        ("svr", make_svr(cfg)),
        ("gp", make_gp(cfg)),
        ("boost", _with_pre(cfg, _boost_regressor(cfg))),
    ], weights=list(cfg.vote_weights), n_jobs=-1)


def make_mlp(cfg):
    from sklearn.neural_network import MLPRegressor
    return _with_pre(cfg, MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), activation="relu",
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
        max_iter=500, random_state=cfg.seed))


def make_gbr(cfg):
    from sklearn.ensemble import GradientBoostingRegressor
    return _with_pre(cfg, GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=cfg.seed))


def make_kridge(cfg):
    from sklearn.kernel_ridge import KernelRidge
    return _with_pre(cfg, KernelRidge(kernel="rbf", alpha=1.0, gamma=cfg.svr_gamma))


def make_linsvr(cfg):
    from sklearn.svm import LinearSVR
    return _with_pre(cfg, TransformedTargetRegressor(
        LinearSVR(C=1.0, epsilon=cfg.svr_epsilon, max_iter=5000, random_state=cfg.seed),
        transformer=StandardScaler()))


def make_stack2(cfg):
    """Stacking (RidgeCV meta on OOF preds) over SVR + GP + boost — the stacking
    counterpart of `vote`."""
    svr = Pipeline([("pre", clone(make_preprocessor(cfg))), ("model", make_svr(cfg).named_steps["model"])])
    boost = Pipeline([("pre", clone(make_preprocessor(cfg))),
                      ("model", _boost_regressor(cfg))])
    gp = Pipeline([("pre", clone(make_preprocessor(cfg))),
                   ("model", GaussianProcessRegressor(
                       kernel=_gp_kernel(), normalize_y=True, alpha=cfg.gp_alpha,
                       n_restarts_optimizer=cfg.gp_n_restarts, random_state=cfg.seed))])
    return StackingRegressor(
        estimators=[("svr", svr), ("boost", boost), ("gp", gp)],
        final_estimator=RidgeCV(alphas=RIDGE_ALPHAS), cv=cfg.inner_cv(), n_jobs=-1)


MODELS = {
    "ridge": make_ridge,
    "histgb": make_histgb,
    "svr": make_svr,
    "stack": make_stack,
    "stack2": make_stack2,
    "vote": make_vote,
    "gp": make_gp,
    "knn": make_knn,
    "rf": make_rf,
    "et": make_et,
    "elasticnet": make_elasticnet,
    "huber": make_huber,
    "pls": make_pls,
    "mlp": make_mlp,
    "gbr": make_gbr,
    "kridge": make_kridge,
    "linsvr": make_linsvr,
}
if HAS_XGB:
    MODELS["xgb"] = make_xgb
try:
    from lightgbm import LGBMRegressor
    def make_lgbm(cfg):
        return _with_pre(cfg, LGBMRegressor(
            n_estimators=800, max_depth=4, learning_rate=0.02, subsample=0.7,
            colsample_bytree=0.6, reg_lambda=3.0, random_state=cfg.seed,
            n_jobs=-1, verbose=-1))
    MODELS["lgbm"] = make_lgbm
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    def make_ngboost(cfg):
        return _with_pre(cfg, NGBRegressor(
            Dist=Normal, n_estimators=500, learning_rate=0.05,
            random_state=cfg.seed, verbose=False))
    MODELS["ngboost"] = make_ngboost
    HAS_NGB = True
except Exception:
    HAS_NGB = False


def build_model(name: str, cfg):
    """Build the full pipeline for a model name."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](cfg)
