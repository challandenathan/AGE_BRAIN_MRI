# edge_lab — new experiments to beat the 0.58 baseline

A separate lab (the main project is untouched) built after reading the data
structure carefully. Everything is evaluated under honest repeated CV
(`RepeatedKFold(5x2, seed 42)` for sweeps, `5x3` for finalists); the whole
pipeline is refit inside every fold, so there is no leakage. The only
transductive step is the signal-block mask, computed from train+test
*features only* (no y) — it is identical whether computed on train, test, or
both.

## What the data forensics showed (`forensics*.py`)

- **There is a structural signal block of exactly 229 features**: the largest
  connected component of the |Spearman corr| > 0.35 graph. It contains 100%
  of the features with |rho_y| > 0.15 (median |rho| 0.225) and is perfectly
  stable across train / test / train+test. The other 603 features are the
  injected noise (325 of them uniform-distributed, max |rho_y| = 0.084).
- **Outliers are cell-level, not row-level**: 222 cells beyond 8 robust
  sigmas (0.02% of all cells), multiplicative 2x–26x, concentrated in ~20
  features. One row (740) has 8 spiked cells, but dropping the few spike-heavy
  rows does **not** improve CV (0.5774 vs 0.5776 baseline) — so we drop no
  rows, as recommended; the quantile transform squashes the spikes anyway.
- Missingness is MCAR (~7.6%, uncorrelated with y); median imputation stays
  best (KNN/iterative re-imputation of the original NaNs on the selected
  features is worse: 0.41–0.45 vs 0.571 for SVR).
- The y–feature relation is monotone (binned means increase smoothly; no
  quadratic/cubic signal), and pairwise products of the top features carry no
  extra signal (max |rho| 0.078 ≈ noise). Pairwise *differences* look strong
  (max 0.645) but are linearly redundant — adding them to the models does not
  help (XGB 0.542 vs 0.546 ref; SVR collapses).
- Self-training / pseudo-labeling of held-out folds does **not** help
  (w=0: 0.5465 vs w=1: 0.5471 for SVR) — not adopted.

## What worked (sweeps 1–12, all 5x2 CV)

| finding | 5x2 R² |
|---|---|
| main project stack2 k=180 (reference) | 0.5830 |
| + block-restricted selection at k=200–229, tuned members (SVR C=26.1 γ=0.0083 ε=0.005; XGB depth5 lr0.04 k=229) | 0.5895 |
| + per-member preprocessing diversity (SVR member uses **signed-log + MAD mask**, GP on the full 229-block, XGB on k=300) | 0.5944 |
| + a **wide member** (SVR on all 828 features, no outlier mask — alone it scores −0.01, but its OOF errors correlate only ~0.6–0.7 with the others, so the meta-learner uses it as a decorrelated correction) | **0.6117** |
| + a **low-dim member** (SVR on top-30 block features by |spearman|) | **0.6145** |
| wide member at k=650 instead of 828 | 0.6182 |

## Headline result — RepeatedKFold(5x3, seed 42), `results/final_5x3.csv`

| finalist | 5x3 R² |
|---|---|
| main project stack2 k=180 (reference; README: 0.5847) | **0.5849** ± 0.036 |
| W (svr-logmad200 + gp229 + xgb300) | 0.5935 ± 0.033 |
| W + wide828-nomask | 0.6144 ± 0.031 |
| W + wide828-nomask + low30 | 0.6153 ± 0.032 |
| **W + wide650-nomask + low30 (winner)** | **0.6208 ± 0.035** |
| svr + gp + wide828 + low30 (no xgb) | 0.6153 ± 0.030 |

Wide-member k at 5x3 (`results/final_5x3_widek.csv`): 650 → **0.6208**,
600 → 0.6207, 700 → 0.6188, 828 → 0.6153.

That is **+0.036 R² over the main project** (0.585 → 0.621 on the same
protocol). The submission (`edge_lab/submission.csv`, written by
`run_final.py`) averages 3 fits of the winner stack with different inner-CV
seeds (42/7/123) to reduce test-time variance.

Rejected: PCA of the block, pair-averaging, reliability weighting, PLS,
kernel-ridge (tuned 0.483), MLP, bagged SVR (0.538), KNN as a member,
LGBM/CatBoost/HistGB as extra members, row dropping, pseudo-labeling,
feature interactions, PCA-augmented members.

## Final model (see `run_final.py`)

A 5-member `StackingRegressor` (RidgeCV meta-learner on OOF predictions,
inner KFold(5, seed 42)):

```
svr  : SignedLog -> LogMADMasker(4) -> median impute -> QuantileTransformer(normal)
       -> block filter -> top-200 by F             -> SVR (tuned)
gp   : adaptive MAD(5) -> impute -> QT -> block -> all 229           -> GP (Matern 2.5)
xgb  : adaptive MAD(5) -> impute -> QT -> block -> top-300 by F      -> XGBoost (tuned)
wide : no mask -> impute -> QT -> top-650 of ALL 828 by F            -> SVR (tuned)
low30: adaptive MAD(5) -> impute -> QT -> block -> top-30 by |rho|   -> SVR (tuned)
```

Run:

```bash
python3 edge_lab/run_final.py --no-submit   # 5x3 CV of the finalists
python3 edge_lab/run_final.py               # + fit on all data, write edge_lab/submission.csv
```

Files: `lab/pipeline.py` (all transformers/builders), `run_sweep*.py`
(experiments), `results/*.csv|json|log` (all scores), `submission.csv`
(sample.csv format, predictions clipped to [42, 97]).
