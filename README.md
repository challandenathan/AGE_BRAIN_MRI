# Brain Age Prediction from MRI Features — AML 2026, Task 1

Predict chronological age (42–97) from 832 FreeSurfer-derived anatomical MRI
features. The data is deliberately perturbed: ~7.6% missing values, one-sided
multiplicative outlier spikes (up to ~1e23), ~100 features rescaled to a ~1e6
scale, and ~600 irrelevant noise features. Scored by **R²** on the test set
(public baseline for grade 4.0: **R² = 0.5**; see `AML_2026_Project_1.pdf`).

## Quick start

```bash
pip install -r requirements.txt
python solution.py               # CV-evaluate, fit on all data, write submission.csv
python solution.py --no-submit   # CV evaluation only
python solution.py --compare     # benchmark all candidate models -> experiments/cv_results.csv
python solution.py --tune        # tune the vote members + weights -> experiments/tuned_params.json
python solution.py --model svr   # any model from brainage.models.MODELS
python -m unittest discover -s tests   # test suite (stdlib unittest, no extra deps)
```

`submission.csv` is written in the exact `sample.csv` format (`id,y`, 776 rows),
predictions clipped to the observed age range [42, 97].

## Layout

```
├── solution.py            # CLI entry point (thin wrapper around the package)
├── brainage/              # the project package
│   ├── config.py          #   paths, seed, hyperparameters — the only place to tweak
│   ├── data.py            #   loading + integrity checks, submission writer
│   ├── preprocessing.py   #   outlier masker, imputation, scaling, feature selection
│   ├── models.py          #   model factories (12 models incl. stack + vote)
│   └── evaluate.py        #   repeated-CV evaluation, model comparison, OOF diagnostics
├── tests/test_pipeline.py # leak-free / shape / selection-mode / submission tests
├── experiments/           # CV results (cv_results.csv, cv_grid_sweep.csv, cv_scores_final.csv)
├── data/                  # X_train.csv, y_train.csv, X_test.csv
├── *.ipynb                # exploration notebooks (evidence for every choice below)
├── selected_features.csv  # the k=180 features kept by the final pipeline
└── submission.csv         # predictions for X_test (sample.csv format)
```

## Method — the four subtasks as one leak-free pipeline

All preprocessing lives inside a single scikit-learn `Pipeline`, so every CV
fold re-estimates all statistics on its training part only (no leakage):

```
VarianceThreshold(0)                       drop the 4 constant features
→ AdaptiveMADMasker(base=5)                subtask 1: entries beyond a per-feature
                                             robust-sigma threshold (median/MAD) are
                                             set to NaN — skewed/heavy-tailed features
                                             get a looser high-side threshold, so only
                                             true spikes (4–17x median, 1e23 values)
                                             are masked, legitimate tails are kept
                                             (best outlier rule in preproc_lab sweep #3)
→ SimpleImputer(strategy="median")         subtask 0: fill NaNs (original ~7.6%
                                             + masked outliers) per feature
→ QuantileTransformer(output="normal")     rank → normal map; neutralizes the
                                             ~1e6-scale feature block and squashes
                                             residual extremes to the boundary
→ SelectKBest(f_regression, k=180)         subtask 2: keep the top-180 features
→ StackingRegressor[ SVR + GP + XGB ]      subtask 3: RidgeCV meta-learner blends
      with RidgeCV meta-learner               the members' out-of-fold predictions
                                             (final model, see below)
```

## Robustness sweeps (what we tested so no signal is lost)

All under honest CV (RepeatedKFold, whole pipeline refit per fold). Full tables
in `experiments/cv_grid_sweep.csv` (34 runs) and `experiments/cv_results.csv`.

**Outlier handling** — MAD-mask threshold swept over {5, 8, 12, off}: at k=180
the score barely moves (0.570–0.574), because the quantile transform already
squashes spikes to the distribution boundary; the mask is kept as a safety net
(thresh=8). k swept over {110, 180, 250, 400, 832}: sharp optimum at **k=180**
(0.57; k=250 → 0.48, k=400 → 0.20, k=832 → 0.0) — the extra features are pure
noise and actively hurt the SVR.

**Feature selection** — SelectKBest with F-test (0.5633), Spearman |ρ|
(0.5638), mutual information (0.4663) and Lasso `SelectFromModel` (0.3288).
F/Spearman keep the redundant-but-informative correlated features; **MI and
Lasso top-180 lose signal** (Lasso keeps only one of each correlated group),
so F-test k=180 is the default; alternatives one flag away
(`--score-func spearman|mutual_info`, `--selection lasso`).

**Model zoo** (k=180, thresh=8, 5×2 CV):

| model                              | CV R² (5×2)      |
|------------------------------------|------------------|
| **vote = avg(SVR + GP + XGB)**     | **0.5747 ± 0.032** |
| SVR (RBF, C=166.5, γ=0.00907, ε=0.03) | 0.5701 ± 0.033 |
| stack (Ridge+XGB+GP + RidgeCV meta) | 0.5645 ± 0.034   |
| GP alone                           | 0.5644 ± 0.033   |
| XGBoost                            | 0.5298 ± 0.036   |
| HistGradientBoosting               | 0.5094 ± 0.036   |
| ExtraTrees                         | 0.5066 ± 0.033   |
| RandomForest                       | 0.4813 ± 0.030   |
| ElasticNetCV                       | 0.4443 ± 0.063   |
| RidgeCV                            | 0.4401 ± 0.066   |
| kNN (k=15, distance)               | 0.4341 ± 0.043   |
| Huber                              | 0.3726 ± 0.055   |
| PLS (30 components)                | 0.3706 ± 0.065   |

The three ensemble members have OOF error correlations of only ~0.91–0.98
(XGB the most complementary), so their average is the most robust choice —
and it is still just a 3-member `VotingRegressor`.

## Results

**Fine-tuning** (`python solution.py --tune`, RandomizedSearchCV over each vote
member's *full pipeline* under KFold(5, seed 42), then vote weights on OOF
predictions; results in `experiments/tuned_params.json`):

| member | best params (abbreviated) | member CV R² (5-fold) |
|---|---|---|
| SVR | C=80.0, γ=0.0104, ε=0.01, k=180 | 0.5622 |
| XGBoost | 800 trees, depth 4, lr 0.02, subsample 0.7, colsample 0.6, λ=3.0 | 0.5470 |
| GP | jitter α=1e-10, k=180 (kernel bounds self-optimize at fit) | 0.5635 |
| vote weights | (svr, gp, boost) = **(0.3, 0.4, 0.3)** | OOF 0.5778 |

The tuned values are the defaults in `brainage/config.py`.

Final model (**stack2**: SVR + GP + XGB blended by a RidgeCV meta-learner on
their out-of-fold predictions, k=180, adaptive MAD base=5), full protocol
**RepeatedKFold(5 folds × 3 repeats, seed 42)**: **CV R² = 0.5847 ± 0.0341**
(per-fold scores in `experiments/cv_scores_final.csv`). Reference scores under
the same 5×3 protocol: adaptive-MAD off (plain MAD5) 0.5812 ± 0.0331, vote
ensemble 0.5788 ± 0.0330, tuned SVR alone 0.5728 ± 0.0281; under 5×2 with
stack2: adaptive MAD 0.5838, plain MAD5 0.5804. Comfortably above the public
baseline of 0.5. Note: tuning and final evaluation share the seed, so the
headline carries the usual mild optimism of model selection.

The preprocessing alternatives (imputation / outlier / selection methods) were
swept in **`preproc_lab/`** — see its README for the full table. Short version:
the quantile transform already neutralizes the spikes (masking details barely
matter), median ≈ mean imputation, and univariate F/Spearman selection at
k=180 beats tree-importance, Lasso, PCA and KNN variants.

### Why the predicted range is narrower than y's range

`y_train` spans 42–97 (mean 69.9, std 9.7) but predictions span only
~50–88 (std ~7.5). This is **regression to the mean**, not clipping — zero
predictions sit at the clip bounds, and out-of-fold predictions show the same
shrinkage. Extreme ages are rare (31 subjects < 50, 18 > 90) and their MAE is
12–19 years vs 3.4 in the 60–80 bulk, so an R²-optimal model deliberately
shrinks toward the mean at the extremes; stretching predictions back to y's
spread (linear calibration on OOF) gains only +0.002 R², so we don't.

## Reproducibility & robustness notes

- Fixed seed (42) everywhere; `xgboost` is optional — if its native OpenMP
  runtime is missing, the code falls back to sklearn's `HistGradientBoosting`.
- `load_data` validates id alignment, duplicate ids and train/test column
  consistency; `write_submission` refuses to write if ids don't match
  `sample.csv`, and rejects NaN/inf predictions.
- The outlier masker never mutates its input and never flags constant columns.
- No AutoML, no test-set peeking: the test set is only touched once, at
  submission time. (Slides: max 10 submissions/day — don't overfit the public
  leaderboard, it is only part of the test set.)

## Hand-in checklist (from the slides)

1. Join the Kaggle competition with your ETH account and form a team.
2. Upload `submission.csv` (format of `sample.csv`) and select it for grading.
3. Fill in the Moodle form individually: Kaggle team name, legi number, your
   own description of the method, and this code. A Kaggle submission alone is
   not enough; keep the code reproducible.

## Draft method description (adapt in your own words for the Moodle form)

> We predict brain age from the 832 perturbed MRI features with a scikit-learn
> pipeline evaluated by repeated 5-fold cross-validation (R² = 0.585 ± 0.034
> on the training set). Preprocessing, all refit inside every CV fold:
> (1) outliers are detected per feature with a median/MAD rule using a
> distribution-aware threshold (looser for skewed/heavy-tailed features) and
> set to NaN; (2) missing values (~7.6%) are imputed with the
> per-feature median; (3) a quantile transform to a normal distribution makes
> the features scale-invariant, which neutralizes the block of features
> rescaled to ~1e6; (4) univariate F-test feature selection keeps the top 180
> of 832 features (a CV sweep showed a sharp optimum at k≈180; mutual
> information, Lasso and PCA-based selection all scored lower, and
> reconstructing the masked outlier cells from other features was worse than
> median imputation). The regressor is a stacking ensemble: RBF-kernel SVR, a
> Gaussian process with a Matérn kernel, and XGBoost, blended by a RidgeCV
> meta-learner on their out-of-fold predictions — hyperparameters tuned by
> randomized CV search, member errors only ~0.9 correlated. Predictions on the
> test set are clipped to the observed age range [42, 97].
