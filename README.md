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
│   ├── models.py          #   model factories: ridge / xgb / histgb / svr / stack
│   └── evaluate.py        #   repeated-CV evaluation, model comparison, OOF diagnostics
├── tests/test_pipeline.py # leak-free / shape / submission-format tests
├── experiments/           # CV results (cv_results.csv, cv_scores_final.csv)
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
→ MADOutlierMasker(thresh=8)               subtask 1: entries > 8 robust sigmas
                                             (median/MAD) from the column median
                                             are set to NaN — catches the 4–17x
                                             spikes and 1e23 values without being
                                             distorted by them
→ SimpleImputer(strategy="median")         subtask 0: fill NaNs (original ~7.6%
                                             + masked outliers) per feature
→ QuantileTransformer(output="normal")     rank → normal map; neutralizes the
                                             ~1e6-scale feature block and squashes
                                             residual extremes to the boundary
→ SelectKBest(f_regression, k=180)         subtask 2: keep the top-180 features;
                                             k chosen by a CV sweep (sharp optimum
                                             at k≈150–200, collapse beyond k≈300)
→ SVR(RBF, C=166.5, gamma=0.00907, eps=0.03)  subtask 3: age regression, on
                                             standardized y (TransformedTargetRegressor)
```

Hyperparameters (C, gamma, epsilon, k, outlier threshold) were tuned with
`RandomizedSearchCV` under the same CV protocol; a confirmation mini-grid is in
the exploration notebooks.

## Results (honest CV on the 1212 training samples)

Evaluation: **RepeatedKFold(5 folds × 3 repeats, seed 42)**, scoring R², the
whole pipeline refit inside every fold.

| model (same preprocessing)        | CV R² (5×2)      |
|-----------------------------------|------------------|
| **SVR k=180, thresh=8 (final)**   | **0.5701 ± 0.033** |
| stack (Ridge + XGB + GP) k=180    | 0.5645 ± 0.034   |
| stack k=110                       | 0.5556 ± 0.041   |
| SVR k=110                         | 0.5366 ± 0.043   |
| XGBoost k=180                     | 0.5298 ± 0.036   |
| HistGradientBoosting k=180        | 0.5094 ± 0.036   |
| Ridge k=180                       | 0.4401 ± 0.066   |

Final model, full protocol (5×3): **CV R² = 0.5728 ± 0.0348**
(per-fold scores in `experiments/cv_scores_final.csv`), comfortably above the
public baseline of 0.5. The stack is kept one flag away
(`python solution.py --model stack`): its out-of-fold errors correlate 0.97
with the SVR's, so blending the two adds only ~+0.002 R² — within noise.

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

> We predict brain age from the 832 perturbed MRI features with a single
> scikit-learn pipeline evaluated by repeated 5-fold cross-validation
> (R² = 0.57 ± 0.03 on the training set). Preprocessing, all refit inside
> every CV fold: (1) outliers are detected per feature with a median/MAD
> rule (8 robust sigmas) and set to NaN; (2) missing values (~7.6%) are
> imputed with the per-feature median; (3) a quantile transform to a normal
> distribution makes the features scale-invariant, which neutralizes the
> block of features rescaled to ~1e6; (4) univariate F-test feature
> selection keeps the top 180 of 832 features. The regressor is an RBF-kernel
> SVR (C=166.5, γ=0.00907, ε=0.03) on standardized age. Predictions on the
> test set are clipped to the observed age range [42, 97].
