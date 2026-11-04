"""Preprocessing for subtasks 0-2 of the slides, as leak-free pipeline steps.

Order matters and is identical to the explored notebooks:

    VarianceThreshold(0)      drop constant features (no information)
    MADOutlierMasker          spikes -> NaN        (subtask 1: outlier detection)
    SimpleImputer(median)     fill NaNs            (subtask 0: missing values)
    QuantileTransformer       rank -> normal       (neutralizes the ~1e6-scale
                                                    features and residual spikes)
    SelectKBest               keep top-k features  (subtask 2: feature selection)

Every step learns its statistics only from the data it is fitted on, so inside
a CV fold the validation part never influences the training part.
"""
from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (SelectKBest, VarianceThreshold,
                                       f_regression, mutual_info_regression)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from .config import RIDGE_ALPHAS


class MADOutlierMasker(BaseEstimator, TransformerMixin):
    """Set entries more than `thresh` robust sigmas from the column median to NaN.

    Robust sigma = MAD / 0.6745, with a scaled-IQR fallback when MAD = 0 and
    `inf` (never flagged) for constant columns. Median/MAD are not distorted by
    the spikes themselves, so the ~1e23 entries are caught reliably.
    thresh=np.inf disables the step.
    """

    def __init__(self, thresh=8.0):
        self.thresh = thresh

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.med_ = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - self.med_), axis=0)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        scale = np.where(mad > 0, mad / 0.6745, (q3 - q1) / 1.349)
        self.scale_ = np.where(scale > 0, scale, np.inf)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)  # copy, never modify the caller's data
        X[np.abs(X - self.med_) / self.scale_ > self.thresh] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_",
                       np.asarray([f"x{i}" for i in range(len(self.med_))], dtype=object))


def _spearman_scores(X, y):
    """|Spearman rho| of every feature with y — robust, catches monotonic
    non-linear signal; constant columns score 0."""
    from scipy.stats import rankdata
    Xr = rankdata(X, axis=0)
    xstd = Xr.std(axis=0)
    Xz = np.divide(Xr - Xr.mean(axis=0), xstd, out=np.zeros_like(Xr), where=xstd > 0)
    ry = rankdata(y)
    yz = (ry - ry.mean()) / ry.std()
    return np.abs((Xz * yz[:, None]).mean(axis=0))


class AdaptiveMADMasker(BaseEstimator, TransformerMixin):
    """Per-feature MAD masking with a distribution-aware threshold: features
    that are right-skewed or heavy-tailed (where large values are legitimate)
    get a looser high-side threshold; light-tailed features get the base one.
    Low side always uses the base threshold. Fitted on the training fold only."""

    def __init__(self, base=5.0, skew_bonus=1.5, skew_cut=1.0, kurt_cut=10.0, kurt_bonus=3.0):
        self.base = base
        self.skew_bonus = skew_bonus
        self.skew_cut = skew_cut
        self.kurt_cut = kurt_cut
        self.kurt_bonus = kurt_bonus

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.med_ = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - self.med_), axis=0)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        scale = np.where(mad > 0, mad / 0.6745, (q3 - q1) / 1.349)
        self.scale_ = np.where(scale > 0, scale, np.inf)
        Xc = np.where(np.isnan(X), self.med_, X)
        mu = Xc.mean(axis=0)
        sd = Xc.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        m3 = ((Xc - mu) ** 3).mean(axis=0) / sd**3
        m4 = ((Xc - mu) ** 4).mean(axis=0) / sd**4 - 3.0
        hi = np.full(X.shape[1], self.base)
        hi = np.where((m3 > self.skew_cut) | (m4 > self.kurt_cut), hi + self.kurt_bonus, hi)
        self.thresh_hi_ = np.where(m3 > self.skew_cut, hi * self.skew_bonus, hi)
        self.thresh_lo_ = np.full(X.shape[1], self.base)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        z = (X - self.med_) / self.scale_
        X[(z > self.thresh_hi_) | (z < -self.thresh_lo_)] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_",
                       np.asarray([f"x{i}" for i in range(len(self.med_))], dtype=object))


def make_outlier_step(cfg):
    if cfg.outlier_thresh == np.inf:
        return None
    if cfg.outlier_mode == "adakurt":
        return AdaptiveMADMasker(cfg.outlier_thresh)
    if cfg.outlier_mode == "mad":
        return MADOutlierMasker(cfg.outlier_thresh)
    raise ValueError(f"unknown outlier_mode {cfg.outlier_mode!r}")


def make_score_func(name: str, seed: int):
    """Deterministic scoring function for SelectKBest."""
    if name == "f_regression":
        return f_regression
    if name == "mutual_info":
        return partial(mutual_info_regression, random_state=seed)
    if name == "spearman":
        return _spearman_scores
    raise ValueError(f"unknown score_func {name!r} "
                     "(use 'f_regression', 'mutual_info' or 'spearman')")


def make_selector(cfg, k=None):
    """The feature-selection step (subtask 2).

    selection='kbest': SelectKBest with cfg.score_func, keep top-k.
    selection='lasso': SelectFromModel(LassoCV) keeping the top-k features by
                       |coefficient| — a multivariate, correlated-features-aware
                       alternative to univariate ranking.
    """
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LassoCV
    k = cfg.n_features if k is None else k
    if cfg.selection == "lasso":
        # threshold=-inf so the max_features cap ranks ALL features by |coef|
        # (the default threshold would keep only the nonzero ones, i.e. far
        # fewer than k)
        return SelectFromModel(
            LassoCV(alphas=RIDGE_ALPHAS, cv=5, random_state=cfg.seed, max_iter=10000),
            max_features=None if k >= 832 else int(k), threshold=-np.inf)
    if cfg.selection != "kbest":
        raise ValueError(f"unknown selection {cfg.selection!r} (use 'kbest' or 'lasso')")
    k = "all" if k >= 832 else int(k)  # SelectKBest rejects k > n_features
    return SelectKBest(make_score_func(cfg.score_func, cfg.seed), k=k)


def make_preprocessor(cfg, k=None):
    """The shared preprocessing chain (everything except the regressor).

    `k` overrides cfg.n_features, which is handy for the k-sweep experiments.
    """
    outlier = make_outlier_step(cfg)
    steps = [("var", VarianceThreshold(0.0))]
    if outlier is not None:
        steps.append(("outlier", outlier))
    steps += [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("qt", QuantileTransformer(output_distribution="normal",
                                   n_quantiles=cfg.n_quantiles,
                                   random_state=cfg.seed)),
        ("select", make_selector(cfg, k)),
    ]
    return Pipeline(steps)


def selected_feature_names(cfg, X, y):
    """Fit the preprocessing chain on all training data and return the kept
    feature names (for inspection / reuse in notebooks)."""
    pre = make_preprocessor(cfg)
    pre.fit(X, y)
    return pre[-1].get_feature_names_out(input_features=pre[:-1].get_feature_names_out())
