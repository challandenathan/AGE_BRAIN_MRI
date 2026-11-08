"""brainage_final — self-contained package for the optimal brain-age pipeline.

See final.ipynb / README.md for the full description. The pipeline:

  1. outlier masking (adaptive MAD, or signed-log + MAD for the SVR member)
  2. median imputation of missing values
  3. per-feature QuantileTransformer to a normal distribution
  4. structural signal-block filter (229 features) + top-k selection
  5. a 5-member stacking ensemble (SVR / GP / XGBoost / wide-SVR / low-dim SVR)
     blended by a RidgeCV meta-learner on out-of-fold predictions
"""
from .pipeline import (  # noqa: F401
    ROOT, DATA_DIR, SEED, load_data, compute_block_mask, make_preprocessor,
    build_model, build_winner, cv_scores, report, load_params, save_params,
    DEFAULT_PARAMS, MEMBER_SPECS,
)
from .tuning import tune_all  # noqa: F401
