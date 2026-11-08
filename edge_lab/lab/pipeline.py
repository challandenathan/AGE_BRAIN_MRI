"""edge_lab pipeline builders.

Preprocessing chain (all steps leak-free inside CV; the only transductive
piece is the signal-block mask, computed from train+test *features only* --
unsupervised, no y -- and shown to be identical whether computed on train,
test, or both):

    VarianceThreshold(0)
    [SignedLog]                optional: sign(x)*log1p(|x|) -- makes the
                               one-sided MULTIPLICATIVE spikes additive
    [LogMADMasker | AdaptiveMADMasker]   spikes -> NaN
    SimpleImputer(median)
    QuantileTransformer(normal)  rank -> N(0,1); neutralizes the ~1e6/1e22
                               rescaled features (rank-invariant to SignedLog)
    [BlockFilter]              keep only the structural signal block (229)
    [PairAverager]             optional: average |corr|>thr groups (denoise)
    BlockRestrictedSelector    top-k by F | spearman *within the block*
    [ReliabilityWeighter]     optional: scale each kept feature by |rho_y|

All transformers preserve pandas DataFrames end-to-end so that precomputed
masks (defined over the original 832 column names) can be aligned by name
even after VarianceThreshold drops the constant columns.

Models: each is Pipeline(pre, regressor). Ensembles stack several with a
RidgeCV meta-learner on OOF predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              StackingRegressor, VotingRegressor)
from sklearn.feature_selection import VarianceThreshold, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
SEED = 42
RIDGE_ALPHAS = np.logspace(-3, 3, 20)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

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
    component of the graph |spearman corr| > thr, on median-imputed
    features. Unsupervised (no y)."""
    M = np.asarray(X_all, dtype=float)
    med = np.nanmedian(M, axis=0)
    Mi = np.where(np.isnan(M), med, M)
    A = np.abs(_rank_corr(Mi))
    np.fill_diagonal(A, 0.0)
    ncomp, lab = connected_components(csr_matrix(A > thr), directed=False)
    sizes = np.bincount(lab)
    return lab == np.argmax(sizes)


def _names_of(X):
    if hasattr(X, "columns"):
        return np.asarray(X.columns, dtype=object)
    return None


def _names_or_pos(X):
    n = _names_of(X)
    if n is not None:
        return n
    return np.asarray([f"x{i}" for i in range(np.asarray(X).shape[1])], dtype=object)


def _out(out, X, names):
    """Wrap `out` as a DataFrame when X was one (keeps column names alive)."""
    if hasattr(X, "columns"):
        return pd.DataFrame(out, columns=np.asarray(names, dtype=object), index=X.index)
    return out


def _align_mask(mask, mask_names, names):
    """Re-express a boolean mask defined over `mask_names` onto `names`."""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == len(names):
        return mask
    idx = pd.Index(np.asarray(mask_names, dtype=object)).get_indexer(names)
    if (idx < 0).any():
        raise ValueError("mask names do not cover the current columns")
    return mask[idx]


# ----------------------------------------------------------------------------
# transformers
# ----------------------------------------------------------------------------

class SignedLog(BaseEstimator, TransformerMixin):
    """sign(x) * log1p(|x|) -- monotone; turns multiplicative spikes into
    additive ones. Rank-invariant, so a following QuantileTransformer is
    unaffected."""

    def fit(self, X, y=None):
        self.feature_names_in_ = _names_or_pos(X)
        return self

    def transform(self, X):
        names = _names_of(X)
        Xa = np.asarray(X, dtype=float)
        out = np.sign(Xa) * np.log1p(np.abs(Xa))
        return _out(out, X, names if names is not None else self.feature_names_in_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


class _BaseMADMasker(BaseEstimator, TransformerMixin):
    """Adaptive per-feature MAD masking: right-skewed/heavy-tailed features
    get a looser high-side threshold. `base` robust sigmas on the low side."""

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
        sd = Xc.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
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
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


class LogMADMasker(_BaseMADMasker):
    """Adaptive MAD masking in signed-log space (put after SignedLog): the
    2-26x multiplicative spikes become additive log-shifts and are caught
    precisely."""


class AdaptiveMADMasker(_BaseMADMasker):
    """Adaptive MAD masking in raw space (same rule as the main project)."""


class BlockFilter(BaseEstimator, TransformerMixin):
    """Keep only the structural signal-block columns.

    mask: precomputed boolean mask over mask_names (transductive, from
          train+test features) or None -> compute from X at fit time
          (within-fold, unsupervised).
    """

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
        names = _names_of(X)
        Xa = np.asarray(X, dtype=float)[:, self.idx_]
        return _out(Xa, X, self.names_)

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
    """Top-k features by a univariate score (F or |spearman|). When
    block_mask is given, non-block features are forced to the bottom, so the
    kept set is the k best *block* features."""

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
        Xa = np.asarray(X, dtype=float)[:, self.idx_]
        return _out(Xa, X, self.names_)

    def get_feature_names_out(self, input_features=None):
        return self.names_


class PairAverager(BaseEstimator, TransformerMixin):
    """Average groups of features whose |corr| > thr (connected components);
    denoises near-duplicate feature groups. Fitted on the training fold only."""

    def __init__(self, thr=0.9):
        self.thr = thr

    def fit(self, X, y=None):
        names = _names_or_pos(X)
        Xa = np.asarray(X, dtype=float)
        C = np.corrcoef(Xa, rowvar=False)
        A = np.abs(np.nan_to_num(C))
        np.fill_diagonal(A, 0.0)
        ncomp, lab = connected_components(csr_matrix(A > self.thr), directed=False)
        self.groups_ = [np.where(lab == c)[0] for c in range(ncomp)]
        self.names_ = np.asarray(
            [names[g[0]] if len(g) == 1 else f"avg{c}" for c, g in enumerate(self.groups_)],
            dtype=object)
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        out = np.column_stack([Xa[:, g].mean(axis=1) for g in self.groups_])
        return _out(out, X, self.names_)

    def get_feature_names_out(self, input_features=None):
        return self.names_


class SelectPlusPCA(BaseEstimator, TransformerMixin):
    """Keep top-k features by score AND append the first n_pc principal
    components of the whole (block) feature set -- the PCs capture the shared
    signal direction while the selected features keep the specifics."""

    def __init__(self, k=180, score="f", n_pc=30, block_mask=None, mask_names=None):
        self.k = k
        self.score = score
        self.n_pc = n_pc
        self.block_mask = block_mask
        self.mask_names = mask_names

    def fit(self, X, y=None):
        from sklearn.decomposition import PCA
        names = _names_or_pos(X)
        Xa = np.asarray(X, dtype=float)
        scores = f_regression(Xa, y)[0] if self.score == "f" else _spearman_scores(Xa, y)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf)
        if self.block_mask is not None:
            bm = _align_mask(self.block_mask, self.mask_names, names)
            scores = np.where(bm, scores, -np.inf)
        k = min(int(self.k), len(names))
        self.idx_ = np.sort(np.argsort(-scores)[:k])
        self.pca_ = PCA(n_components=min(int(self.n_pc), Xa.shape[1]),
                        random_state=SEED).fit(Xa)
        self.names_ = np.concatenate([names[self.idx_],
                                      np.asarray([f"pc{i+1}" for i in range(self.pca_.n_components_)],
                                                 dtype=object)])
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        out = np.column_stack([Xa[:, self.idx_], self.pca_.transform(Xa)])
        return _out(out, X, self.names_)

    def get_feature_names_out(self, input_features=None):
        return self.names_


class SelectPlusInteractions(BaseEstimator, TransformerMixin):
    """Keep top-k features by score, plus all pairwise products of the top-m
    features among them (captures ratio/interaction signal)."""

    def __init__(self, k=180, score="f", m=15, block_mask=None, mask_names=None):
        self.k = k
        self.score = score
        self.m = m
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
        top_m = np.argsort(-scores[self.idx_])[:min(int(self.m), len(self.idx_))]
        self.pair_idx_ = [(self.idx_[a], self.idx_[b])
                          for a in top_m for b in top_m if a < b]
        base_names = names[self.idx_]
        self.names_ = np.concatenate([
            base_names,
            np.asarray([f"{names[a]}*{names[b]}" for a, b in self.pair_idx_], dtype=object)])
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        inter = np.column_stack([Xa[:, a] * Xa[:, b] for a, b in self.pair_idx_]) \
            if self.pair_idx_ else np.empty((Xa.shape[0], 0))
        out = np.column_stack([Xa[:, self.idx_], inter])
        return _out(out, X, self.names_)

    def get_feature_names_out(self, input_features=None):
        return self.names_


class ReliabilityWeighter(BaseEstimator, TransformerMixin):
    """Scale each feature by its |spearman rho| with y (soft selection:
    keeps every selected feature but down-weights noisy ones)."""

    def __init__(self, power=1.0):
        self.power = power

    def fit(self, X, y=None):
        names = _names_of(X)
        Xa = np.asarray(X, dtype=float)
        r = _spearman_scores(Xa, y)
        self.w_ = np.nan_to_num(r) ** self.power
        self.feature_names_in_ = names if names is not None else \
            np.asarray([f"x{i}" for i in range(Xa.shape[1])], dtype=object)
        return self

    def transform(self, X):
        names = _names_of(X)
        out = np.asarray(X, dtype=float) * self.w_
        return _out(out, X, names if names is not None else self.feature_names_in_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


# ----------------------------------------------------------------------------
# preprocessor factory
# ----------------------------------------------------------------------------

def make_preprocessor(o, block_mask=None, block_names=None):
    """o: dict with keys
        mask      'logmad' | 'adakurt' | 'none'   (outlier handling)
        log       bool  (signed-log before masking; only matters with a mask)
        mask_base float (robust-sigma base for the masker)
        k         int   (features kept; large = all)
        score     'f' | 'spearman'
        block     'trans' (use block_mask) | 'fold' (compute in fit) | 'none'
        pairavg   None | float thr  (before selection)
        weight    bool  (ReliabilityWeighter after selection)
        n_quantiles int
    """
    o = dict(o)
    mask_mode = o.get("mask", "logmad")
    use_log = o.get("log", mask_mode == "logmad")
    k = o.get("k", 180)
    score = o.get("score", "f")
    block = o.get("block", "trans")
    pairavg = o.get("pairavg", None)
    weight = o.get("weight", False)
    nq = o.get("n_quantiles", 500)

    var = VarianceThreshold(0.0).set_output(transform="pandas")
    steps = [("var", var)]
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
        sel_mask = None  # already restricted to the block
    elif block == "fold":
        steps.append(("block", BlockFilter(mask=None)))
        sel_mask = None
    else:  # 'none': no filter; the selector can still restrict by mask
        sel_mask = block_mask
    if pairavg is not None:
        steps.append(("pairavg", PairAverager(thr=pairavg)))
    if o.get("pca_add"):
        steps.append(("select", SelectPlusPCA(k=k, score=score, n_pc=o["pca_add"],
                                              block_mask=sel_mask, mask_names=block_names)))
    elif o.get("inter_m"):
        steps.append(("select", SelectPlusInteractions(k=k, score=score, m=o["inter_m"],
                                                       block_mask=sel_mask, mask_names=block_names)))
    elif o.get("pca"):
        from sklearn.decomposition import PCA
        steps.append(("pca", PCA(n_components=int(o["pca"]), random_state=SEED)
                      .set_output(transform="pandas")))
    else:
        steps.append(("select", BlockRestrictedSelector(k=k, score=score, block_mask=sel_mask,
                                                         mask_names=block_names)))
    if weight:
        steps.append(("weight", ReliabilityWeighter()))
    return Pipeline(steps)


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------

def _pre(o, block_mask, block_names):
    return make_preprocessor(o, block_mask, block_names)


SVR_PARAMS = dict(C=80.03, gamma=0.010403, epsilon=0.01)


def _svr():
    return TransformedTargetRegressor(SVR(**SVR_PARAMS), transformer=StandardScaler())


def _gp():
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-10,
                                   n_restarts_optimizer=2, random_state=SEED)


XGB_PARAMS = dict(n_estimators=800, max_depth=4, learning_rate=0.02, subsample=0.7,
                  colsample_bytree=0.6, reg_lambda=3.0, random_state=SEED, n_jobs=-1)

LGBM_PARAMS = dict(n_estimators=1000, max_depth=5, learning_rate=0.03, num_leaves=31,
                   subsample=0.7, colsample_bytree=0.6, reg_lambda=3.0,
                   random_state=SEED, n_jobs=-1, verbose=-1)

CAT_PARAMS = dict(iterations=1200, depth=5, learning_rate=0.03, subsample=0.7,
                  colsample_bylevel=0.6, l2_leaf_reg=3.0, random_state=SEED,
                  verbose=0, thread_count=-1)

# runtime overrides (set by tuning scripts before building models)
XGB_OVERRIDES: dict = {}
LGBM_OVERRIDES: dict = {}


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(**{**XGB_PARAMS, **XGB_OVERRIDES})


def _lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**{**LGBM_PARAMS, **LGBM_OVERRIDES})


def _cat():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(**CAT_PARAMS)


def _histgb():
    return HistGradientBoostingRegressor(max_iter=500, max_depth=4, learning_rate=0.04,
                                         l2_regularization=1.0, random_state=SEED)


def _ridge():
    return RidgeCV(alphas=RIDGE_ALPHAS)


def _knn():
    return KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)


def _et():
    return ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                               random_state=SEED, n_jobs=-1)


def _mlp():
    return MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1.0, early_stopping=True,
                        validation_fraction=0.15, max_iter=3000, tol=1e-5,
                        random_state=SEED, n_iter_no_change=30)


_REGRESSORS = {
    "svr": _svr, "gp": _gp, "xgb": _xgb, "lgbm": _lgbm, "cat": _cat,
    "histgb": _histgb, "ridge": _ridge, "knn": _knn, "et": _et, "mlp": _mlp,
}


def build_model(name, o, block_mask=None, block_names=None):
    """Single model = lab preprocessor + regressor."""
    return Pipeline([("pre", _pre(o, block_mask, block_names)), ("model", _REGRESSORS[name]())])


def build_stack(member_names, o, block_mask=None, block_names=None, per_member_o=None, meta=None):
    """StackingRegressor of full pipelines (each with its own preprocessor
    options via per_member_o); RidgeCV meta-learner on OOF predictions."""
    ests = []
    for m in member_names:
        mo = (per_member_o or {}).get(m, o)
        ests.append((m, Pipeline([("pre", _pre(mo, block_mask, block_names)),
                                  ("model", _REGRESSORS[m]())])))
    return StackingRegressor(
        estimators=ests,
        final_estimator=meta or RidgeCV(alphas=RIDGE_ALPHAS),
        cv=KFold(5, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )


def build_vote(member_names, o, block_mask=None, block_names=None, weights=None):
    ests = [(m, Pipeline([("pre", _pre(o, block_mask, block_names)),
                          ("model", _REGRESSORS[m]())]))
            for m in member_names]
    return VotingRegressor(ests, weights=weights, n_jobs=-1)


MODELS = {name: (lambda n: lambda o, bm=None, bn=None: build_model(n, o, bm, bn))(name)
          for name in _REGRESSORS}


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

def cv_scores(model, X, y, n_splits=5, n_repeats=2, seed=SEED, n_jobs=-1):
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=n_jobs)


def report(name, scores, rows):
    rows.append({"name": name, "mean_r2": scores.mean(), "std_r2": scores.std(),
                 "min_r2": scores.min(), "max_r2": scores.max(), "n_folds": len(scores)})
    print(f"  {name:<44} R2 = {scores.mean():.4f} +/- {scores.std():.4f} "
          f"(min {scores.min():.3f} max {scores.max():.3f})", flush=True)
    return rows
