# final — optimal brain-age pipeline (self-contained)

Everything needed to run and reproduce the best model lives in this folder:

```
final/
├── final.ipynb            # the whole solution as a notebook: data → forensics → preprocessing
│                          #   → model → per-member FINE-TUNING → 5x3 CV evaluation → submission
├── solution.py            # equivalent CLI: --tune | --evaluate | --submit | --quick
├── brainage_final/        # the package (preprocessing transformers + model builders + tuning)
│   ├── pipeline.py
│   └── tuning.py
├── tuned_params.json      # tuned hyperparameters (cache; delete to re-run the searches)
├── data/                  # X_train.csv, y_train.csv, X_test.csv
├── sample.csv             # submission format reference
├── requirements.txt
└── submission.csv         # written by solution.py / the notebook (sample.csv format)
```

## Quick start

```bash
cd final
pip install -r requirements.txt
python3 solution.py            # 5x3 CV evaluation + fit on all data + write submission.csv
# or step by step:
python3 solution.py --tune     # fine-tune SVR / XGBoost / GP (RepeatedKFold 5x2), cache tuned_params.json
python3 solution.py --evaluate # honest RepeatedKFold(5x3) CV of the winner stack
python3 solution.py --submit   # fit (3 inner seeds averaged) + write submission.csv
python3 solution.py --quick    # fast check: 5x2 CV + single-seed submission
# notebook:
jupyter nbconvert --to notebook --execute --inplace final.ipynb
```

## Result

| model | RepeatedKFold(5×3, seed 42) CV R² |
|---|---|
| previous main-project stack (k=180) — reference | 0.5849 ± 0.036 |
| **this pipeline (5-member stack, wide650 + low30)** | **≈ 0.62 (0.616–0.621 across runs) ± 0.04** |

(5×2 quick check: 0.6182 ± 0.033. Per-fold scores differ by ~±0.005 between runs
because XGBoost/GP fits are not bit-reproducible under `n_jobs=-1`; the mean is
stable at ≈ 0.62. Detailed sweep tables: `../edge_lab/results/`.)

## Method

1. **Outliers (subtask 1)** — cell-level, never row-level: per-feature adaptive
   MAD rule (looser high-side threshold for skewed/heavy-tailed features) sets
   spiked cells to NaN; the `svr` member first applies `sign(x)·log1p(|x|)` so the
   one-sided multiplicative spikes (2–26×) become additive and are caught precisely.
2. **Missing values (subtask 0)** — per-feature median imputation (~7.6% MCAR).
3. **Scaling** — `QuantileTransformer(normal)`: rank → N(0,1), which neutralizes the
   rescaled (~1e6/1e22) feature blocks and squashes residual spikes to the boundary.
4. **Feature selection (subtask 2)** — a transductive *signal-block mask* (largest
   |spearman corr|>0.35 connected component = 229 features, computed from train+test
   **features only**, no y) + top-k by F-test / |spearman| within the block.
5. **Regressor (subtask 3)** — `StackingRegressor`, RidgeCV meta-learner on the members'
   out-of-fold predictions, with five members of deliberately different views:
   `svr` (log-MAD, block top-200 F), `gp` (block, all 229), `xgb` (block top-300 F),
   `wide` (no mask, top-650 of all 828 — alone ≈ 0 R² but its OOF errors correlate
   only ~0.6–0.7 with the others, so the meta-learner uses it as a decorrelated
   correction), `low30` (block top-30 |ρ|).
6. **Submission** — 3 fits with different inner-CV seeds (42/7/123), predictions averaged,
   clipped to [42, 97], written in `sample.csv` format with id verification.

All statistics are refit inside every CV fold (no leakage); fixed seed 42 throughout.
Experiments and rejected alternatives are documented in `../edge_lab/README.md`.
