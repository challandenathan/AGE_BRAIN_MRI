# Task 1 — Predicting brain age from MRI features

## Data

- `X_train.csv`: 1212 samples x 832 features (`x0`–`x831`), `y_train.csv`: age (42–97, mean 69.9). `X_test.csv`: 776 samples.
- The features were perturbed in several ways, which we identified during exploration:
  - **Missing values**: ~7.6% of all entries are NaN, spread over all 832 features and all 1212 rows (test: ~6.3%).
  - **Outliers**: a handful of features contain one-sided multiplicative spikes (4–17x the column median; e.g. `x742` has 120 such entries, `x115` 51, `x641` 20), plus a few extreme entries up to ~1e23 (`x173`, `x665`, `x596`). 226 entries exceed 8 robust sigmas.
  - **Rescaled features**: 100 features sit entirely on a ~1e6 scale (same columns in train and test) — a unit/scale perturbation.
  - **Noise features**: only ~200 of the 832 features have |Spearman rho| >= 0.1 with age; 4 features are constant.

## Preprocessing

All preprocessing steps live inside a single scikit-learn `Pipeline`, so their statistics are re-estimated on the training part of every CV fold (no leakage into the validation folds).

1. **Outlier detection — median/MAD masking.** For each feature we compute the median and the MAD (median absolute deviation) on the training fold. Any entry more than `thresh = 8` robust sigmas away (`|x - median| / (MAD/0.6745) > 8`, falling back to scaled IQR when MAD = 0) is set to NaN. This catches the multiplicative spikes and the 1e23-scale entries without being distorted by them, because median/MAD are robust. A threshold sweep (3, 5, 8, 12, off) showed 5–8 is optimal and clearly better than no masking; winsorizing at the 0.5%/99.5% percentiles was also tested and performed slightly worse.
2. **Imputation of missing values — per-feature median.** After masking, every NaN (original ~7.6% plus masked outliers) is replaced by the column median of the training fold. Median was chosen over mean and k-NN imputation: it is robust to the remaining outliers and scored best in CV (R^2 0.563 vs 0.560 mean / 0.551 kNN at k=150).
3. **Feature selection — `SelectKBest` with `f_regression`, k = 180.** Features are ranked by the absolute F-statistic of their linear regression on age (computed after imputation and a quantile transform, see below) and the top 180 are kept. A sweep over k showed a sharp optimum: R^2 ~0.57 for k in 150–200, collapsing to ~0.30 at k=300, ~0.07 at k=400 and ~0 at k=832 — the majority of features are pure noise and hurt the SVR. Ranking by |Spearman rho| instead of F gave virtually identical results.

Between imputation and selection, a `QuantileTransformer` (output ~ normal, 500 quantiles) maps each feature to its rank-based normal quantile. This neutralizes the 100 features that were rescaled to ~1e6 and squashes any residual extreme values to the boundary of the distribution, which is why a plain standard/robust scaler performed worse (R^2 0.555 / 0.469).

## Regression model

**Support Vector Regression with an RBF kernel**, wrapped in `TransformedTargetRegressor` with a `StandardScaler` on the target so that `C`/`epsilon` are on a meaningful scale. Hyperparameters were tuned with `RandomizedSearchCV` (60 samples over k, outlier threshold, C, gamma, epsilon; 5-fold CV):

```
SVR(C=166.5, gamma=0.00907, epsilon=0.03)   on standardized y
```

Alternatives evaluated with the same protocol: ElasticNetCV (R^2 0.46), k-NN (0.43), HistGradientBoosting (0.51–0.52), and a stacking ensemble of SVR + GB + k-NN — none beat the tuned SVR (residual correlations between models were >0.86, so blending added little). Final pipeline:

```
MADOutlierMasker(thresh=8) -> SimpleImputer(median) -> QuantileTransformer(normal)
  -> SelectKBest(f_regression, k=180) -> SVR(RBF, C=166.5, gamma=0.00907, epsilon=0.03)
```

## Evaluation

The metric is the coefficient of determination

```
R^2 = 1 - sum_i (y_i - yhat_i)^2 / sum_i (y_i - mean(y))^2
```

estimated with **RepeatedKFold (5 folds x 3 repeats, shuffled, seed 42)**, scoring `r2`:

| model (same preprocessing) | CV R^2 |
|---|---|
| tuned SVR, k=180, thresh=8 (final) | **0.573 ± 0.035** |
| SVR, k=150, no masking | 0.561 ± 0.035 |
| HistGradientBoosting, k=150 | 0.515 ± 0.038 |
| ElasticNetCV, k=200 | 0.457 ± 0.044 |
| k-NN (k=15), k=180 | 0.435 ± 0.047 |
| constant mean predictor (sample.csv) | 0.000 |

Per-fold scores of the final model: 0.53–0.65. The final pipeline is then fitted on all 1212 training samples and predictions for the 776 test samples (clipped to the observed age range [42, 97]) are written to `submission.csv` in the `sample.csv` format. Reproduce with:

```
python3 solution.py
```
