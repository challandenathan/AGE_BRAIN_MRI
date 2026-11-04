# preproc_lab — preprocessing technique sweep

A self-contained copy of `brainage/` (the main project is untouched) extended
with alternative techniques for the three preprocessing subtasks, plus
`run_sweeps.py`, which evaluates them all under honest CV.

**Protocol.** Model fixed to the already-tuned configuration (no re-tuning with
the new techniques): the sweep uses the tuned **SVR** (fast, ~1.3 s/fold) under
RepeatedKFold(5×2, seed 42); the top-3 variants are then re-scored with the full
tuned **vote** ensemble (SVR+GP+XGB) under the same 5×2 protocol. Every variant
is a full pipeline, so all statistics are refit inside each CV fold.

Run: `python preproc_lab/run_sweeps.py` → `results/preproc_sweep.csv`,
`results/preproc_finalists.csv`.

## Tested techniques

- **Outliers**: MAD-mask at thresh {5, 8, 12}, one-sided MAD (high side only),
  winsorize (clip at med±8σ), IQR-mask, non-robust z-score mask, no masking,
  row-level IsolationForest neutralization (contamination 0.01/0.03).
- **NaN imputation**: median (baseline), mean, zero-constant, median + missing
  indicator columns, KNNImputer(k=3/5/10) applied after the quantile transform
  (NaNs survive QT, and on the ~N(0,1) scale distances are meaningful).
- **Feature selection**: SelectKBest F-test k=180 (baseline), Spearman |ρ| k=180,
  ExtraTrees-importance top-180, PCA (100/180 components), correlation pruning
  at 0.95 + F-test top-180, F-test k=150/200, Lasso top-180.

## Results — sweep with tuned SVR, RepeatedKFold(5×2)

| variant | CV R² |
|---|---|
| **outlier MAD thresh=5** | **0.5725 ± 0.028** |
| select Spearman k=180 | 0.5709 ± 0.033 |
| outlier IQR thresh=8 | 0.5706 ± 0.034 |
| outlier z-score thresh=8 | 0.5704 ± 0.034 |
| outlier: none | 0.5699 ± 0.034 |
| outlier MAD thresh=12 | 0.5697 ± 0.034 |
| select corr0.95 + F-kbest180 | 0.5696 ± 0.033 |
| **baseline: MAD8 / median / F-kbest180** | 0.5691 ± 0.033 |
| outlier one-sided MAD8 | 0.5691 ± 0.033 |
| impute mean | 0.5681 ± 0.033 |
| select F k=150 | 0.5666 ± 0.035 |
| outlier winsorize 8 | 0.5644 ± 0.034 |
| impute median + indicators | 0.5632 ± 0.035 |
| select F k=200 | 0.5621 ± 0.034 |
| outlier IsolationForest rows (0.01) | 0.5546 ± 0.042 |
| outlier IsolationForest rows (0.03) | 0.5404 ± 0.046 |
| select ExtraTrees top-180 | 0.5345 ± 0.036 |
| impute KNN k=10 (post-QT) | 0.4518 ± 0.103 |
| impute KNN k=3 | 0.4256 ± 0.101 |
| impute KNN k=5 | 0.4200 ± 0.106 |
| select Lasso top-180 | 0.3723 ± 0.050 |
| select PCA 100 | 0.3011 ± 0.038 |
| impute zero | 0.0586 ± 0.010 |
| select PCA 180 | 0.0914 ± 0.019 |

## Finalists with the tuned vote ensemble (5×2)

| variant | CV R² |
|---|---|
| **outlier MAD thresh=5** | **0.5777 ± 0.029** |
| select Spearman k=180 | 0.5775 ± 0.030 |
| outlier IQR thresh=8 | 0.5768 ± 0.031 |
| baseline (MAD8) | 0.5764 ± 0.030 |

## Conclusions

1. **The quantile transform is doing the heavy lifting.** Once it is in the
   pipeline, the outlier-masking details barely matter (0.569–0.573 across
   mad5/iqr/zscore/none) — QT squashes spikes to the distribution boundary.
   MAD thresh=5 is the best and lowest-variance choice, adopted in the main
   project (`outlier_thresh=5`).
2. **Median ≈ mean imputation** (0.569 vs 0.568). Zero-filling is catastrophic
   (0.06) because the raw features live on huge scales. Missing indicators add
   nothing (missingness is ~unrelated to y). KNN imputation on all 828
   quantile-transformed features is *much worse* (0.42–0.45) — the curse of
   dimensionality makes 828-dim euclidean neighbours meaningless.
3. **Univariate ranking (F or Spearman) at k=180 beats everything else.**
   Tree-importance selection (0.53), Lasso top-180 (0.37) and PCA (0.09–0.30)
   all lose signal: they drop correlated-but-informative features or project
   onto variance directions that are mostly noise. Correlation pruning before
   k-best is neutral (0.5696).
4. Row-level outlier removal (IsolationForest) hurts — neutralizing whole
   subjects toward the median loses information.

Net: the current main pipeline is already near the practical optimum for this
data; the only adopted change is MAD thresh 8 → 5.

## Sweep #2 — does masking delete important data? + new models

`run_sweeps2.py` → `results/sweep2_all.csv`, `results/sweep2_finalists.csv`.
Same protocol (tuned model fixed, RepeatedKFold 5×2; finalists re-scored with
the tuned vote).

**A. Info-preserving outlier handling** (tuned SVR):

| variant | CV R² |
|---|---|
| **BASE: MAD thresh=5 + median impute + QT (current)** | **0.5725 ± 0.028** |
| no masking at all (QT squashes spikes) | 0.5699 |
| signed-log scaling instead of QT + MAD5 | 0.5683 |
| signed-log, no mask | 0.5655 |
| winsorize at 8σ instead of delete | 0.5644 |
| MAD thresh=3 (more aggressive deleting) | 0.5446 |
| reconstruct masked cells with KNN k=10 on the selected 180 | 0.4809 |
| reconstruct masked cells with iterative (BayesianRidge) | 0.4607 |

Conclusion: **we are not deleting important data.** The masked cells are
genuine corruption — reconstructing them from the other features is much
*worse* than the median fill, and simply keeping the spikes (no mask) is also
slightly worse. MAD-5 + QT is the best of all worlds. With the vote ensemble:
BASE 0.5777 vs no-mask 0.5775 — a tie.

**B. New models** on the base preprocessing (5×2):

| model | CV R² |
|---|---|
| **stack2: SVR+GP+boost, RidgeCV meta-learner on OOF preds** | **0.5804 ± 0.030** |
| vote: avg(SVR+GP+XGB) (reference) | 0.5777 ± 0.029 |
| XGBoost | 0.5355 |
| LightGBM | 0.5196 |
| HistGradientBoosting | 0.5170 |
| GradientBoosting | 0.5140 |
| LinearSVR k=180 / k=400 | 0.371 / 0.147 |
| MLP (256,128,64) | −0.67 (overfits) |
| KernelRidge RBF | −1.95 (bad gamma) |
| NGBoost | NaN (numerically unstable here) |

The stacking meta-learner (RidgeCV on the members' out-of-fold predictions)
beats the fixed 0.3/0.4/0.3 average: **0.5804 vs 0.5777**. Verified under the
full 5×3 protocol before adoption (see git history / main README).

## Sweep #3 — precise case-by-case outlier rules, more selectors, exotic models

`run_sweeps3.py` + follow-ups → `results/sweep3_*.csv` (5×2; C/D with tuned SVR,
E with each model itself, finalists with stack2).

**C. Precise outlier rules** (tuned SVR): adaptive per-feature MAD (looser
high-side threshold for skewed/heavy-tailed features) **0.5772** > Tukey fence
k=3 0.5738 > base MAD5 0.5725 > precise "spike" rules (z>5 AND x>ratio·med)
0.564–0.566 > empirical winsor q99.5 0.5590 > LOF rows 0.5361. MCD/EllipticEnvelope
is singular at p≈n — not viable. The ratio/spike conditions are *too* precise:
they miss genuine outliers, so the plain adaptive MAD wins.

**D. Selection methods** (tuned SVR): stability selection (top-k on 20
subsamples, keep freq≥0.5) 0.5726 ± 0.0274 ≈ base F-180 0.5725; Spearman 0.5743;
cluster-diverse 0.5164; Boruta 0.4724; ridge-coef 0.4124; RFECV 0.3803. No
alternative beats plain univariate F-test top-180.

**E. Exotic models** (base preprocessing): stack2 0.5804 (ref) > stack3 (+ET
member) 0.5791 > monotonic-XGB (rho-sign constraints) 0.5355 = plain XGB >
CatBoost 0.5024 > ARD 0.4404 > OMP 0.4158 > RANSAC 0.3208 > QuantileRegressor
−0.002. None beats stack2.

**Finalists with stack2 (5×2)**: base MAD5 0.5804, **adaptive MAD 0.5838**,
stability 0.5790, adaptive+stability 0.5827 → **adaptive per-feature MAD adopted
in the main project** (`outlier_mode="adakurt"`); stability selection not
adopted (no gain with the ensemble).
