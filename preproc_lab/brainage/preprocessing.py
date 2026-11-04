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


class PreciseSpikeMasker(_Columnwise):
    """Case-by-case spike rule matching the known corruption: mask a high-side
    entry only if BOTH (a) it is beyond `z_hi` robust sigmas AND (b) it is a
    multiplicative spike, x > ratio * median (only when median > 0); plus any
    entry beyond `z_abs` sigmas on either side. Legitimate tail values of
    heavy-tailed features are kept because they fail the ratio condition."""

    def __init__(self, z_hi=5.0, ratio=3.0, z_abs=12.0):
        self.z_hi = z_hi
        self.ratio = ratio
        self.z_abs = z_abs

    def fit(self, X, y=None):
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        self.med_, self.scale_ = _robust_scale(X)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        z = (X - self.med_) / self.scale_
        ratio_cond = np.where(self.med_ > 0, X > self.ratio * self.med_, False)
        mask = ((z > self.z_hi) & ratio_cond) | (np.abs(z) > self.z_abs)
        X[mask] = np.nan
        return X


class RatioMasker(_Columnwise):
    """Pure multiplicative-spike rule: mask x > ratio * median (median > 0 only)."""

    def __init__(self, ratio=4.0):
        self.ratio = ratio

    def fit(self, X, y=None):
        self._set_names(X)
        self.med_ = np.nanmedian(np.asarray(X, dtype=float), axis=0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        cond = np.where(self.med_ > 0, X > self.ratio * self.med_, False)
        X[cond] = np.nan
        return X


class TukeyMasker(_Columnwise):
    """Tukey upper/lower fences per feature: mask x outside [q1 - k*IQR, q3 + k*IQR]."""

    def __init__(self, k=3.0):
        self.k = k

    def fit(self, X, y=None):
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        self.q1_, self.q3_ = np.nanpercentile(X, [25, 75], axis=0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        iqr = self.q3_ - self.q1_
        X[(X < self.q1_ - self.k * iqr) | (X > self.q3_ + self.k * iqr)] = np.nan
        return X


class AdaptiveMADMasker(_Columnwise):
    """Per-feature threshold: heavy-tailed / right-skewed features get a looser
    high-side threshold, light-tailed ones a tighter one."""

    def __init__(self, base=5.0, skew_bonus=1.5, skew_cut=1.0, kurt_cut=10.0, kurt_bonus=3.0):
        self.base = base
        self.skew_bonus = skew_bonus
        self.skew_cut = skew_cut
        self.kurt_cut = kurt_cut
        self.kurt_bonus = kurt_bonus

    def fit(self, X, y=None):
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        self.med_, self.scale_ = _robust_scale(X)
        Xc = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        mu = Xc.mean(axis=0)
        sd = Xc.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        m3 = ((Xc - mu) ** 3).mean(axis=0) / sd**3
        m4 = ((Xc - mu) ** 4).mean(axis=0) / sd**4 - 3.0
        hi = np.full(X.shape[1], self.base)
        hi = np.where((m3 > self.skew_cut) | (m4 > self.kurt_cut), hi + self.kurt_bonus, hi)
        hi = np.where(m3 > self.skew_cut, hi * self.skew_bonus, hi)
        self.thresh_hi_ = hi
        self.thresh_lo_ = np.full(X.shape[1], self.base)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        z = (X - self.med_) / self.scale_
        X[(z > self.thresh_hi_) | (z < -self.thresh_lo_)] = np.nan
        return X


class EmpiricalWinsor(_Columnwise):
    """Clip each feature at its empirical [1-q, q] quantiles instead of masking."""

    def __init__(self, q=0.995):
        self.q = q

    def fit(self, X, y=None):
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = np.nanpercentile(X, [100 * (1 - self.q), 100 * self.q], axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


class RowNeutralizer(_Columnwise):
    """Row-level outlier detection (iforest | lof | mcd), fit on the training
    fold after imputation+scaling. Flagged rows are replaced by the column
    medians, i.e. neutralized toward the 'average' subject instead of dropped."""

    def __init__(self, contamination=0.02, seed=42, method="iforest"):
        self.contamination = contamination
        self.seed = seed
        self.method = method

    def fit(self, X, y=None):
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        self.med_ = np.nanmedian(X, axis=0)
        Xn = np.nan_to_num(X, nan=0.0)
        if self.method == "iforest":
            from sklearn.ensemble import IsolationForest
            self.det_ = IsolationForest(contamination=self.contamination,
                                        random_state=self.seed, n_jobs=-1)
            self.det_.fit(Xn)
        elif self.method == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            self.det_ = LocalOutlierFactor(n_neighbors=20,
                                           contamination=self.contamination,
                                           novelty=True, n_jobs=-1)
            self.det_.fit(Xn)
        elif self.method == "mcd":
            from sklearn.covariance import EllipticEnvelope
            self.det_ = EllipticEnvelope(contamination=self.contamination,
                                         random_state=self.seed)
            self.det_.fit(Xn)
        else:
            raise ValueError(self.method)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        flagged = self.det_.predict(np.nan_to_num(X, nan=0.0)) == -1
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
    if mode == "lof":
        return None
    if mode == "mcd":
        return None
    if mode == "spike":
        return PreciseSpikeMasker(cfg.outlier_thresh, cfg.outlier_ratio)
    if mode == "ratio":
        return RatioMasker(cfg.outlier_ratio)
    if mode == "tukey":
        return TukeyMasker(cfg.outlier_thresh)
    if mode == "adakurt":
        return AdaptiveMADMasker(cfg.outlier_thresh)
    if mode == "winsor_q":
        return EmpiricalWinsor(cfg.cap_quantile)
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


def _scores(score_func, X, y):
    s = score_func(X, y)
    if isinstance(s, tuple):  # f_regression returns (F, p-values)
        s = s[0]
    return np.nan_to_num(np.asarray(s, dtype=float), nan=-np.inf)


class _SelectorBase(_Columnwise):
    def get_support(self):
        return self.support_

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features, dtype=object) if input_features is not None \
            else self.feature_names_in_
        return names[self.support_]


class BorutaSelector(_SelectorBase):
    """Boruta-style: shadow (permuted) copies of every feature; a real feature is
    kept if its ExtraTrees importance beats the best shadow importance; the
    kept set is capped at k by importance rank."""

    def __init__(self, k=180, n_estimators=200, seed=42):
        self.k = k
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, X, y=None):
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        Xs = np.hstack([X, np.apply_along_axis(rng.permutation, 0, X)])
        et = ExtraTreesRegressor(n_estimators=self.n_estimators, min_samples_leaf=2,
                                 random_state=self.seed, n_jobs=-1).fit(Xs, y)
        imp = et.feature_importances_
        n = X.shape[1]
        thresh = imp[n:].max()
        keep = np.where(imp[:n] > thresh)[0]
        if len(keep) > self.k:  # cap by importance
            keep = keep[np.argsort(imp[keep])[::-1][: self.k]]
        self.support_ = np.zeros(n, dtype=bool)
        self.support_[keep] = True
        return self


class StabilitySelector(_SelectorBase):
    """Stability selection: on many row subsamples, take the top-k by the score
    function; keep features selected in at least `freq` of the subsamples
    (capped at k by mean rank)."""

    def __init__(self, k=180, score_func=f_regression, n_sub=20, frac=0.8, freq=0.5, seed=42):
        self.k = k
        self.score_func = score_func
        self.n_sub = n_sub
        self.frac = frac
        self.freq = freq
        self.seed = seed

    def fit(self, X, y=None):
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        count = np.zeros(X.shape[1])
        score_sum = np.zeros(X.shape[1])
        for _ in range(self.n_sub):
            idx = rng.choice(n, size=int(self.frac * n), replace=False)
            s = _scores(self.score_func, X[idx], y[idx])
            top = np.argsort(s)[::-1][: self.k]
            count[top] += 1
            score_sum += s
        keep = np.where(count >= self.freq * self.n_sub)[0]
        if len(keep) > self.k:
            keep = keep[np.argsort(score_sum[keep])[::-1][: self.k]]
        if len(keep) == 0:  # fallback: top-k by mean score
            keep = np.argsort(score_sum)[::-1][: self.k]
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[keep] = True
        return self


class ClusterSelector(_SelectorBase):
    """Diversity-aware selection: cluster correlated features (|corr| > thresh),
    keep the best-scoring feature of each cluster, then fill the remaining
    budget with the next-best unclustered features — avoids spending k on
    near-duplicates."""

    def __init__(self, k=180, score_func=f_regression, thresh=0.8):
        self.k = k
        self.score_func = score_func
        self.thresh = thresh

    def fit(self, X, y=None):
        self._set_names(X)
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        s = _scores(self.score_func, X, y)
        C = np.nan_to_num(np.corrcoef(X, rowvar=False))
        order = np.argsort(s)[::-1]
        keep, clustered = [], np.zeros(n, dtype=bool)
        for j in order:  # greedy: take j unless a kept feature is its correlate
            if not clustered[j]:
                keep.append(j)
                clustered |= np.abs(C[j]) > self.thresh
        keep = np.array(keep[: self.k])
        self.support_ = np.zeros(n, dtype=bool)
        self.support_[keep] = True
        return self


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
    if sel == "ridge":
        from sklearn.linear_model import Ridge
        return SelectFromModel(Ridge(alpha=100.0),
                               max_features=None if k >= 832 else int(k),
                               threshold=-np.inf)
    if sel == "boruta":
        return BorutaSelector(k if k < 832 else 832, seed=cfg.seed)
    if sel == "stability":
        return StabilitySelector(k if k < 832 else 832,
                                 score_func=make_score_func(cfg.score_func, cfg.seed),
                                 seed=cfg.seed)
    if sel == "cluster":
        return ClusterSelector(k if k < 832 else 832,
                               score_func=make_score_func(cfg.score_func, cfg.seed),
                               thresh=cfg.corr_thresh)
    if sel == "rfecv":
        from sklearn.feature_selection import RFECV
        from sklearn.linear_model import Ridge
        return RFECV(Ridge(alpha=10.0), step=0.2, cv=3,
                     min_features_to_select=50, scoring="r2", n_jobs=-1)
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
    if cfg.outlier_mode in ("iforest", "lof", "mcd"):
        steps.append(("rows", RowNeutralizer(cfg.row_contamination, cfg.seed,
                                             method=cfg.outlier_mode)))
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
