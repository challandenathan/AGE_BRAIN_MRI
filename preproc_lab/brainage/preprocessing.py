"""LAB COPY — extended preprocessing for the preproc_lab sweeps.

Same leak-free design as the main brainage package (everything inside one
Pipeline), but with many alternative techniques for the three subtasks:

  outlier_mode : mad | mad_high | iqr | winsor | zscore | iforest | none
  impute       : median | mean | zero      (+ optional missing indicators)
  knn_impute   : KNNImputer applied AFTER quantile transform + selection
                 (distances are only meaningful on the ~N(0,1) scale)
  selection    : kbest | lasso | tree | pca | corr_kbest

All statistics are learned on the training fold only.
"""
from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.feature_selection import (SelectFromModel, SelectKBest,
                                       VarianceThreshold, f_regression,
                                       mutual_info_regression)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from .config import RIDGE_ALPHAS


class _Columnwise(BaseEstimator, TransformerMixin):
    """Base for pass-through-column transformers: feature names are preserved."""

    def _set_names(self, X):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            n = np.asarray(X).shape[1]
            self.feature_names_in_ = np.asarray([f"x{i}" for i in range(n)], dtype=object)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_", None)


def _robust_scale(X):
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0)
    q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
    scale = np.where(mad > 0, mad / 0.6745, (q3 - q1) / 1.349)
    return med, np.where(scale > 0, scale, np.inf)


class MADOutlierMasker(_Columnwise):
    """Entries beyond `thresh` robust sigmas from the median -> NaN.
    side='high' masks only the high side (the spikes in this data are one-sided)."""

    def __init__(self, thresh=8.0, side="both"):
        self.thresh = thresh
        self.side = side

    def fit(self, X, y=None):
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        self.med_, self.scale_ = _robust_scale(X)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        z = (X - self.med_) / self.scale_
        if self.side == "high":
            mask = z > self.thresh
        else:
            mask = np.abs(z) > self.thresh
        X[mask] = np.nan
        return X


class IQRMasker(_Columnwise):
    """Mask entries beyond `thresh` scaled-IQRs from the median (no MAD)."""

    def __init__(self, thresh=8.0):
        self.thresh = thresh

    def fit(self, X, y=None):
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        self.med_ = np.nanmedian(X, axis=0)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        scale = (q3 - q1) / 1.349
        self.scale_ = np.where(scale > 0, scale, np.inf)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        X[np.abs((X - self.med_) / self.scale_) > self.thresh] = np.nan
        return X


class ZMasker(_Columnwise):
    """Classic (non-robust) z-score masking — expected to be distorted by the spikes."""

    def __init__(self, thresh=8.0):
        self.thresh = thresh

    def fit(self, X, y=None):
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_ = np.where(self.std_ > 0, self.std_, np.inf)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        X[np.abs((X - self.mean_) / self.std_) > self.thresh] = np.nan
        return X


class WinsorClipper(_Columnwise):
    """Clip to median ± thresh*robust_sigma instead of masking to NaN."""

    def __init__(self, thresh=8.0):
        self.thresh = thresh

    def fit(self, X, y=None):
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        self.med_, self.scale_ = _robust_scale(X)
        self.lo_ = self.med_ - self.thresh * self.scale_
        self.hi_ = self.med_ + self.thresh * self.scale_
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


class RowNeutralizer(_Columnwise):
    """Row-level outlier detection (IsolationForest, fit on the training fold
    after imputation+scaling). Flagged rows are replaced by the column medians,
    i.e. neutralized toward the 'average' subject instead of dropped."""

    def __init__(self, contamination=0.02, seed=42):
        self.contamination = contamination
        self.seed = seed

    def fit(self, X, y=None):
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        self.med_ = np.nanmedian(X, axis=0)
        self.if_ = IsolationForest(contamination=self.contamination,
                                   random_state=self.seed, n_jobs=-1)
        self.if_.fit(np.nan_to_num(X, nan=0.0))
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        flagged = self.if_.predict(np.nan_to_num(X, nan=0.0)) == -1
        X[flagged] = self.med_
        return X


def make_outlier_step(cfg):
    mode = cfg.outlier_mode
    if mode == "none" or cfg.outlier_thresh == np.inf:
        return None
    if mode == "mad":
        return MADOutlierMasker(cfg.outlier_thresh, side="both")
    if mode == "mad_high":
        return MADOutlierMasker(cfg.outlier_thresh, side="high")
    if mode == "iqr":
        return IQRMasker(cfg.outlier_thresh)
    if mode == "zscore":
        return ZMasker(cfg.outlier_thresh)
    if mode == "winsor":
        return WinsorClipper(cfg.outlier_thresh)
    if mode == "iforest":
        return None  # handled after qt by RowNeutralizer
    raise ValueError(f"unknown outlier_mode {mode!r}")


def _spearman_scores(X, y):
    from scipy.stats import rankdata
    Xr = rankdata(X, axis=0)
    xstd = Xr.std(axis=0)
    Xz = np.divide(Xr - Xr.mean(axis=0), xstd, out=np.zeros_like(Xr), where=xstd > 0)
    ry = rankdata(y)
    yz = (ry - ry.mean()) / ry.std()
    return np.abs((Xz * yz[:, None]).mean(axis=0))


def make_score_func(name, seed):
    if name == "f_regression":
        return f_regression
    if name == "mutual_info":
        return partial(mutual_info_regression, random_state=seed)
    if name == "spearman":
        return _spearman_scores
    raise ValueError(f"unknown score_func {name!r}")


class CorrPruner(_Columnwise):
    """Greedily drop features whose |Pearson corr| with an already-kept feature
    exceeds `threshold` (removes near-duplicates before the k-best step)."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        names = self.feature_names_in_
        C = np.corrcoef(np.nan_to_num(X, nan=0.0), rowvar=False)
        C = np.nan_to_num(C)
        keep = []
        for j in range(X.shape[1]):
            if all(abs(C[j, k]) < self.threshold for k in keep):
                keep.append(j)
        self.keep_ = np.array(keep)
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[self.keep_] = True
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]

    def get_support(self):
        return self.support_

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features, dtype=object) if input_features is not None \
            else self.feature_names_in_
        return names[self.support_]


def make_selector(cfg, k=None):
    k = cfg.n_features if k is None else k
    sel = cfg.selection
    if sel == "pca":
        return PCA(n_components=min(int(k), 800), random_state=cfg.seed)
    if sel == "kbest":
        k = "all" if k >= 832 else int(k)
        return SelectKBest(make_score_func(cfg.score_func, cfg.seed), k=k)
    if sel == "lasso":
        return SelectFromModel(
            LassoCV(alphas=RIDGE_ALPHAS, cv=5, random_state=cfg.seed, max_iter=10000),
            max_features=None if k >= 832 else int(k), threshold=-np.inf)
    if sel == "tree":
        return SelectFromModel(
            ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                random_state=cfg.seed, n_jobs=-1),
            max_features=None if k >= 832 else int(k), threshold=-np.inf)
    raise ValueError(f"unknown selection {sel!r}")


def make_imputer(cfg):
    if cfg.impute == "median":
        return SimpleImputer(strategy="median", keep_empty_features=True,
                             add_indicator=cfg.add_indicator)
    if cfg.impute == "mean":
        return SimpleImputer(strategy="mean", keep_empty_features=True,
                             add_indicator=cfg.add_indicator)
    if cfg.impute == "zero":
        return SimpleImputer(strategy="constant", fill_value=0.0,
                             keep_empty_features=True, add_indicator=cfg.add_indicator)
    raise ValueError(f"unknown impute {cfg.impute!r}")


class SignedLog(_Columnwise):
    """sign(x) * log1p(|x|) — compresses the multiplicative spikes instead of
    deleting them (information-preserving alternative to masking)."""

    def fit(self, X, y=None):
        self._set_names(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.sign(X) * np.log1p(np.abs(X))


class SmartImputeSelector(_Columnwise):
    """Select top-k features, then RECONSTRUCT the missing/masked cells on the
    selected columns from the other selected features (KNN or iterative
    imputation) instead of replacing them by the column median.

    Internally: median-impute -> quantile-transform -> SelectKBest(k) ->
    restore the original NaN mask on the selected columns -> fill with the
    chosen imputer. Everything is fitted on the training fold only."""

    def __init__(self, k=180, score_func=f_regression, method="knn", n_neighbors=10,
                 max_iter=3, n_quantiles=500, seed=42):
        self.k = k
        self.score_func = score_func
        self.method = method
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.n_quantiles = n_quantiles
        self.seed = seed

    def fit(self, X, y=None):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        self.nan_mask_ = np.isnan(X)
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        self.med_ = med
        Xi = np.where(self.nan_mask_, med, X)
        self.qt_ = QuantileTransformer(output_distribution="normal",
                                       n_quantiles=self.n_quantiles,
                                       random_state=self.seed).fit(Xi)
        Xq = self.qt_.transform(Xi)
        k = "all" if self.k >= Xq.shape[1] else int(self.k)
        self.sel_ = SelectKBest(self.score_func, k=k).fit(Xq, y)
        Xs = self.sel_.transform(Xq)
        Xs[self.nan_mask_[:, self.sel_.get_support()]] = np.nan
        if self.method == "knn":
            self.filler_ = KNNImputer(n_neighbors=self.n_neighbors,
                                      weights="distance").fit(Xs)
        else:
            self.filler_ = IterativeImputer(
                estimator=BayesianRidge(), max_iter=self.max_iter,
                random_state=self.seed, skip_complete=True).fit(Xs)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mask = np.isnan(X)
        Xi = np.where(mask, self.med_, X)
        Xq = self.qt_.transform(Xi)
        Xs = self.sel_.transform(Xq)
        Xs[mask[:, self.sel_.get_support()]] = np.nan
        out = self.filler_.transform(Xs)
        return np.nan_to_num(out, nan=0.0)  # safety: all-NaN rows

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features, dtype=object) if input_features is not None \
            else self.feature_names_in_
        return self.sel_.get_feature_names_out(names)


def make_preprocessor(cfg, k=None):
    """The full preprocessing chain; steps are chosen by cfg.

    With knn_impute > 0 the median imputer is replaced by a KNNImputer applied
    AFTER the quantile transform (QuantileTransformer passes NaN through, and
    on the ~N(0,1) scale euclidean distance is meaningful), so the NaNs are
    filled from similar subjects instead of the column median."""
    steps = [("var", VarianceThreshold(0.0))]
    out = make_outlier_step(cfg)
    if out is not None:
        steps.append(("outlier", out))
    if cfg.smart_impute != "none":
        # selection + info-preserving reconstruction of the deleted cells
        steps.append(("smart", SmartImputeSelector(
            k=cfg.n_features, score_func=make_score_func(cfg.score_func, cfg.seed),
            method=cfg.smart_impute, n_neighbors=10, seed=cfg.seed,
            n_quantiles=cfg.n_quantiles)))
        return Pipeline(steps)
    if cfg.knn_impute and cfg.knn_impute > 0:
        steps.append(("qt", QuantileTransformer(output_distribution="normal",
                                                n_quantiles=cfg.n_quantiles,
                                                random_state=cfg.seed)))
        steps.append(("knn", KNNImputer(n_neighbors=int(cfg.knn_impute),
                                         weights="distance")))
    else:
        steps.append(("impute", make_imputer(cfg)))
        if cfg.prescale == "logscale":
            steps.append(("log", SignedLog()))
            steps.append(("scale", __import__("sklearn.preprocessing",
                                              fromlist=["StandardScaler"]).StandardScaler()))
        else:
            steps.append(("qt", QuantileTransformer(output_distribution="normal",
                                                    n_quantiles=cfg.n_quantiles,
                                                    random_state=cfg.seed)))
    if cfg.outlier_mode == "iforest":
        steps.append(("rows", RowNeutralizer(cfg.row_contamination, cfg.seed)))
    if cfg.selection == "corr_kbest":
        steps.append(("corr", CorrPruner(cfg.corr_thresh)))
        steps.append(("select", SelectKBest(make_score_func(cfg.score_func, cfg.seed),
                                             k="all" if (k or cfg.n_features) >= 832
                                             else int(k or cfg.n_features))))
    else:
        steps.append(("select", make_selector(cfg, k)))
    return Pipeline(steps)


def selected_feature_names(cfg, X, y):
    pre = make_preprocessor(cfg)
    pre.fit(X, y)
    last_name, last = pre.steps[-1]
    if last_name == "select":
        return last.get_feature_names_out(pre[:-1].get_feature_names_out())
    return pre[:-1].get_feature_names_out()
