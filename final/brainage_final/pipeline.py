"""The optimal pipeline, self-contained in `final/`.

Preprocessing (every step is a scikit-learn transformer, so inside a CV fold
it is always refit on the training part only — no leakage):

    VarianceThreshold(0)                     drop the 4 constant features
    [SignedLog]            optional sign(x)*log1p(|x|): makes the one-sided
                           MULTIPLICATIVE spikes additive (SVR member only)
    [LogMADMasker | AdaptiveMADMasker]        spikes -> NaN (subtask 1)
    SimpleImputer(median)                    fill NaNs (subtask 0)
    QuantileTransformer(normal)              rank -> N(0,1); neutralizes the
                           rescaled (~1e6 / 1e22) feature blocks and squashes
                           residual spikes to the boundary
    [BlockFilter]        keep the structural signal block (229 features,
                           largest |spearman corr|>0.35 connected component,
                           computed from train+test FEATURES only — no y)
    BlockRestrictedSelector                  top-k by F or |spearman| (subtask 2)

The only transductive piece is the block mask; it is identical whether
computed on train, test, or both (verified in edge_lab forensics).

Model: a 5-member StackingRegressor, RidgeCV meta-learner on the members'
out-of-fold predictions (inner KFold(5, shuffle, seed 42)):

    svr    signed-log + LogMAD(4) -> impute -> QT -> block -> top-200 F   -> SVR
    gp     adaptive MAD(5)        -> impute -> QT -> block -> all 229    -> GP (Matern 2.5)
    xgb    adaptive MAD(5)        -> impute -> QT -> block -> top-300 F  -> XGBoost
    wide   no mask                -> impute -> QT -> top-650 of all 828  -> SVR
    low30  adaptive MAD(5)        -> impute -> QT -> block -> top-30 |rho|-> SVR

The `wide` member alone scores ~0 R^2 (828 features overwhelm the SVR), but
its out-of-fold errors correlate only ~0.6-0.7 with the other members, so the
meta-learner uses it as a decorrelated correction (+0.017 R^2); the low-dim
member adds a different bias again.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.feature_selection import VarianceThreshold, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parent.parent          # final/
DATA_DIR = ROOT / "data"
PARAMS_PATH = ROOT / "tuned_params.json"
SEED = 42
RIDGE_ALPHAS = np.logspace(-3, 3, 20)

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_data():
    X = pd.read_csv(DATA_DIR / "X_train.csv").drop(columns="id")
    Xt = pd.read_csv(DATA_DIR / "X_test.csv").drop(columns="id")
    y = pd.read_csv(DATA_DIR / "y_train.csv")["y"].to_numpy(float)
    ids_tr = pd.read_csv(DATA_DIR / "X_train.csv")["id"].to_numpy()
    ids_te = pd.read_csv(DATA_DIR / "X_test.csv")["id"].to_numpy()
    return X, y, Xt, ids_tr, ids_te


def _rank_corr(M):
    R = rankdata(M, axis=0)
    sd = R.std(axis=0)
    R = np.divide(R - R.mean(axis=0), sd, out=np.zeros_like(R), where=sd > 0)
    return (R.T @ R) / M.shape[0]


def compute_block_mask(X_all, thr=0.35):
    """Boolean mask of the structural signal block: largest connected
    component of the graph |spearman corr| > thr, on median-imputed features.
    Unsupervised (no y) — safe to compute on train+test."""
    M = np.asarray(X_all, dtype=float)
    med = np.nanmedian(M, axis=0)
    Mi = np.where(np.isnan(M), med, M)
    A = np.abs(_rank_corr(Mi))
    np.fill_diagonal(A, 0.0)
    ncomp, lab = connected_components(csr_matrix(A > thr), directed=False)
    return lab == np.argmax(np.bincount(lab))


def _names_of(X):
    return np.asarray(X.columns, dtype=object) if hasattr(X, "columns") else None


def _names_or_pos(X):
    n = _names_of(X)
    if n is not None:
        return n
    return np.asarray([f"x{i}" for i in range(np.asarray(X).shape[1])], dtype=object)


def _out(out, X, names):
    """Wrap as DataFrame when the input was one, so column names survive the
    whole pipeline (needed to align the precomputed block mask by name)."""
    if hasattr(X, "columns"):
        return pd.DataFrame(out, columns=np.asarray(names, dtype=object), index=X.index)
    return out


def _align_mask(mask, mask_names, names):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == len(names):
        return mask
    idx = pd.Index(np.asarray(mask_names, dtype=object)).get_indexer(names)
    if (idx < 0).any():
        raise ValueError("mask names do not cover the current columns")
    return mask[idx]


# ---------------------------------------------------------------------------
# transformers
# ---------------------------------------------------------------------------

class SignedLog(BaseEstimator, TransformerMixin):
    """sign(x) * log1p(|x|) — monotone; turns multiplicative spikes into
    additive ones. Rank-invariant, so a following QuantileTransformer is
    unaffected."""

    def fit(self, X, y=None):
        self.feature_names_in_ = _names_or_pos(X)
        return self

    def transform(self, X):
        names = _names_of(X)
        out = np.sign(np.asarray(X, float)) * np.log1p(np.abs(np.asarray(X, float)))
        return _out(out, X, names if names is not None else self.feature_names_in_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object) if input_features is not None \
            else self.feature_names_in_


class _BaseMADMasker(BaseEstimator, TransformerMixin):
    """Adaptive per-feature MAD masking: right-skewed/heavy-tailed features get
    a looser high-side threshold; `base` robust sigmas on the low side."""

    def __init__(self, base=5.0, skew_bonus=1.5, skew_cut=1.0, kurt_cut=10.0, kurt_bonus=3.0):
        self.base = base
        self.skew_bonus = skew_bonus
        self.skew_cut = skew_cut
        self.kurt_cut = kurt_cut
        self.kurt_bonus = kurt_bonus

    def fit(self, X, y=None):
        self.feature_names_in_ = _names_or_pos(X)
        Xa = np.asarray(X, dtype=float)
        self.med_ = np.nanmedian(Xa, axis=0)
        mad = np.nanmedian(np.abs(Xa - self.med_), axis=0)
        q1, q3 = np.nanpercentile(Xa, [25, 75], axis=0)
        scale = np.where(mad > 0, mad / 0.6745, (q3 - q1) / 1.349)
        self.scale_ = np.where(scale > 0, scale, np.inf)
        Xc = np.where(np.isnan(Xa), self.med_, Xa)
        mu = Xc.mean(axis=0)
        sd = np.where(Xc.std(axis=0) > 0, Xc.std(axis=0), 1.0)
        m3 = ((Xc - mu) ** 3).mean(axis=0) / sd**3
        m4 = ((Xc - mu) ** 4).mean(axis=0) / sd**4 - 3.0
        hi = np.full(Xa.shape[1], self.base)
        hi = np.where((m3 > self.skew_cut) | (m4 > self.kurt_cut), hi + self.kurt_bonus, hi)
        self.thresh_hi_ = np.where(m3 > self.skew_cut, hi * self.skew_bonus, hi)
        self.thresh_lo_ = np.full(Xa.shape[1], self.base)
        return self

    def transform(self, X):
        names = _names_of(X)
        Xa = np.array(X, dtype=float)  # copy; never mutate the caller's data
        z = (Xa - self.med_) / self.scale_
        Xa[(z > self.thresh_hi_) | (z < -self.thresh_lo_)] = np.nan
        return _out(Xa, X, names if names is not None else self.feature_names_in_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object) if input_features is not None \
            else self.feature_names_in_


class LogMADMasker(_BaseMADMasker):
    """Adaptive MAD masking in signed-log space (place after SignedLog): the
    2-26x multiplicative spikes become additive log-shifts, caught precisely."""


class AdaptiveMADMasker(_BaseMADMasker):
    """Adaptive MAD masking in raw space."""


class BlockFilter(BaseEstimator, TransformerMixin):
    """Keep only the structural signal-block columns. `mask` is precomputed
    (transductive, train+test features only) and aligned by column name."""

    def __init__(self, mask=None, mask_names=None, thr=0.35):
        self.mask = mask
        self.mask_names = mask_names
        self.thr = thr

    def fit(self, X, y=None):
        names = _names_or_pos(X)
        if self.mask is None:
            self.mask_ = compute_block_mask(np.asarray(X, dtype=float), self.thr)
        else:
            self.mask_ = _align_mask(self.mask, self.mask_names, names)
        self.names_ = names[self.mask_]
        self.idx_ = np.where(self.mask_)[0]
        return self

    def transform(self, X):
        return _out(np.asarray(X, dtype=float)[:, self.idx_], X, self.names_)

    def get_feature_names_out(self, input_features=None):
        return self.names_


def _spearman_scores(X, y):
    Xr = rankdata(X, axis=0)
    xstd = Xr.std(axis=0)
    Xz = np.divide(Xr - Xr.mean(axis=0), xstd, out=np.zeros_like(Xr), where=xstd > 0)
    ry = rankdata(y)
    yz = (ry - ry.mean()) / ry.std()
    return np.abs((Xz * yz[:, None]).mean(axis=0))


class BlockRestrictedSelector(BaseEstimator, TransformerMixin):
    """Top-k features by a univariate score (F or |spearman|). When block_mask
    is given, non-block features are forced to the bottom."""

    def __init__(self, k=180, score="f", block_mask=None, mask_names=None):
        self.k = k
        self.score = score
        self.block_mask = block_mask
        self.mask_names = mask_names

    def fit(self, X, y=None):
        names = _names_or_pos(X)
        Xa = np.asarray(X, dtype=float)
        scores = f_regression(Xa, y)[0] if self.score == "f" else _spearman_scores(Xa, y)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf)
        if self.block_mask is not None:
            bm = _align_mask(self.block_mask, self.mask_names, names)
            scores = np.where(bm, scores, -np.inf)
        k = min(int(self.k), len(names))
        self.idx_ = np.sort(np.argsort(-scores)[:k])
        self.names_ = names[self.idx_]
        return self

    def transform(self, X):
        return _out(np.asarray(X, dtype=float)[:, self.idx_], X, self.names_)

    def get_feature_names_out(self, input_features=None):
        return self.names_


# ---------------------------------------------------------------------------
# preprocessor factory
# ---------------------------------------------------------------------------

def make_preprocessor(o, block_mask=None, block_names=None):
    """o keys: mask ('logmad'|'adakurt'|'none'), log, mask_base, k, score,
    block ('trans'|'fold'|'none'), n_quantiles."""
    o = dict(o)
    mask_mode = o.get("mask", "adakurt")
    use_log = o.get("log", mask_mode == "logmad")
    k = o.get("k", 200)
    score = o.get("score", "f")
    block = o.get("block", "trans")
    nq = o.get("n_quantiles", 500)

    steps = [("var", VarianceThreshold(0.0).set_output(transform="pandas"))]
    if use_log:
        steps.append(("slog", SignedLog()))
    if mask_mode == "logmad":
        steps.append(("outlier", LogMADMasker(base=o.get("mask_base", 4.0))))
    elif mask_mode == "adakurt":
        steps.append(("outlier", AdaptiveMADMasker(base=o.get("mask_base", 5.0))))
    elif mask_mode != "none":
        raise ValueError(mask_mode)
    steps += [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)
         .set_output(transform="pandas")),
        ("qt", QuantileTransformer(output_distribution="normal", n_quantiles=nq,
                                   random_state=SEED).set_output(transform="pandas")),
    ]
    if block == "trans":
        steps.append(("block", BlockFilter(mask=block_mask, mask_names=block_names)))
        sel_mask = None
    elif block == "fold":
        steps.append(("block", BlockFilter(mask=None)))
        sel_mask = None
    else:
        sel_mask = block_mask
    steps.append(("select", BlockRestrictedSelector(k=k, score=score, block_mask=sel_mask,
                                                     mask_names=block_names)))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# regressors + member specs
# ---------------------------------------------------------------------------

# default (already-tuned) hyperparameters — see tuning.py / tuned_params.json
DEFAULT_PARAMS = {
    "svr": {"C": 26.10157215682536, "gamma": 0.008254041852680182, "epsilon": 0.005},
    "xgb": {"n_estimators": 800, "max_depth": 5, "learning_rate": 0.04,
            "subsample": 0.6, "colsample_bytree": 0.5, "reg_lambda": 3.0,
            "random_state": SEED, "n_jobs": -1},
    "gp": {"alpha": 1e-10, "kernel": "matern2.5", "n_restarts": 2},
}


def svr_regressor(p):
    return TransformedTargetRegressor(SVR(**p), transformer=StandardScaler())


def xgb_regressor(p):
    from xgboost import XGBRegressor
    return XGBRegressor(**p)


def _gp_kernel(name):
    if name == "rbf":
        from sklearn.gaussian_process.kernels import RBF
        base = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    else:
        base = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    return (ConstantKernel(1.0, (1e-3, 1e3)) * base
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1)))


def gp_regressor(p):
    return GaussianProcessRegressor(kernel=_gp_kernel(p.get("kernel", "matern2.5")),
                                    normalize_y=True, alpha=p.get("alpha", 1e-10),
                                    n_restarts_optimizer=p.get("n_restarts", 2),
                                    random_state=SEED)


def load_params():
    """Tuned hyperparameters (+ optional member-spec overrides); falls back to
    the shipped defaults."""
    if PARAMS_PATH.exists():
        try:
            p = json.load(open(PARAMS_PATH))
            if "svr" in p and "xgb" in p and "gp" in p:
                return p
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_PARAMS))


def save_params(params):
    json.dump(params, open(PARAMS_PATH, "w"), indent=1)


# member preprocessing specs of the winner stack
MEMBER_SPECS = {
    "svr":   dict(mask="logmad", log=True, mask_base=4.0, block="trans", k=200, score="f"),
    "gp":    dict(mask="adakurt", block="trans", k=229, score="f"),
    "xgb":   dict(mask="adakurt", block="trans", k=300, score="f"),
    "wide":  dict(mask="none", log=False, block="none", k=650, score="f"),
    "low30": dict(mask="adakurt", block="trans", k=30, score="spearman"),
}


def build_model(name, o, block_mask=None, block_names=None, params=None):
    """Single member = preprocessor + regressor (used by the tuning searches)."""
    params = params or DEFAULT_PARAMS
    if name == "svr":
        reg = svr_regressor(params["svr"])
    elif name == "xgb":
        reg = xgb_regressor(params["xgb"])
    elif name == "gp":
        reg = gp_regressor(params["gp"])
    else:
        raise ValueError(name)
    return Pipeline([("pre", make_preprocessor(o, block_mask, block_names)),
                     ("model", reg)])


def build_winner(params=None, block_mask=None, block_names=None, inner_seed=SEED, specs=None):
    """The optimal 5-member stacking ensemble (RidgeCV meta-learner on the
    members' out-of-fold predictions)."""
    params = params or load_params()
    specs = specs or params.get("specs") or MEMBER_SPECS
    regs = {"svr": svr_regressor(params["svr"]),
            "xgb": xgb_regressor(params["xgb"]),
            "gp": gp_regressor(params["gp"])}
    ests = []
    for name in ["svr", "gp", "xgb", "wide", "low30"]:
        o = specs.get(name, MEMBER_SPECS[name])
        bm = block_mask if o["block"] == "trans" else None
        reg = regs[name] if name in regs else svr_regressor(params["svr"])
        ests.append((name, Pipeline([("pre", make_preprocessor(o, bm, block_names)),
                                     ("model", reg)])))
    return StackingRegressor(estimators=ests,
                             final_estimator=RidgeCV(alphas=RIDGE_ALPHAS),
                             cv=KFold(5, shuffle=True, random_state=inner_seed),
                             n_jobs=-1)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def cv_scores(model, X, y, n_splits=5, n_repeats=3, seed=SEED, n_jobs=-1):
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=n_jobs)


def report(name, scores, rows=None):
    if rows is not None:
        rows.append({"name": name, "mean_r2": scores.mean(), "std_r2": scores.std(),
                     "min_r2": scores.min(), "max_r2": scores.max(), "n_folds": len(scores)})
    print(f"  {name:<40} R2 = {scores.mean():.4f} +/- {scores.std():.4f} "
          f"(min {scores.min():.3f} max {scores.max():.3f})", flush=True)
    return rows
