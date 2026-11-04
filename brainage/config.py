"""Central configuration: paths, seed and hyperparameters in one place.

Everything that might need tweaking lives here, so experiments and the final
submission always use the exact same settings.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

RIDGE_ALPHAS = np.logspace(-3, 3, 20)


@dataclass
class Config:
    # --- data ---
    data_dir: Path = ROOT / "data"

    # --- reproducibility ---
    seed: int = 42

    # --- subtask 1: outlier detection ---
    outlier_thresh: float = 5.0     # robust sigmas (5 chosen by the preproc_lab sweep:
                                    # best mean and lowest std in both SVR and vote
                                    # protocols; np.inf disables masking)

    # --- preprocessing ---
    n_quantiles: int = 500          # QuantileTransformer resolution

    # --- subtask 2: feature selection ---
    selection: str = "kbest"        # "kbest" (SelectKBest) | "lasso" (SelectFromModel)
    n_features: int = 180           # k features kept
    score_func: str = "f_regression"  # "f_regression" | "mutual_info" | "spearman"

    # --- evaluation ---
    n_splits: int = 5
    n_repeats: int = 3              # RepeatedKFold(5 folds x 3 repeats)

    # --- subtask 3: regression ---
    model: str = "stack2"           # key of models.MODELS; "stack2" = SVR + GP + boost
                                    # blended by a RidgeCV meta-learner on their OOF
                                    # predictions (best in the preproc_lab sweeps:
                                    # 0.5812 5x3 vs vote 0.5788, svr 0.5728)
    # Tuned by `python solution.py --tune` (RandomizedSearchCV, KFold(5, seed 42),
    # whole pipeline refit per fold) — see experiments/tuned_params.json
    svr_C: float = 80.03
    svr_gamma: float = 0.010403
    svr_epsilon: float = 0.01

    xgb: dict = field(default_factory=lambda: dict(
        n_estimators=800, max_depth=4, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.6, reg_lambda=3.0,
        random_state=42, n_jobs=-1,
    ))
    gp_n_restarts: int = 2
    gp_alpha: float = 1e-10         # jitter on the GP kernel diagonal (tuned)
    vote_weights: tuple = (0.3, 0.4, 0.3)   # (svr, gp, boost), tuned on OOF preds

    # --- paths (derived) ---
    @property
    def x_train_path(self) -> Path:
        return self.data_dir / "X_train.csv"

    @property
    def y_train_path(self) -> Path:
        return self.data_dir / "y_train.csv"

    @property
    def x_test_path(self) -> Path:
        return self.data_dir / "X_test.csv"

    @property
    def sample_path(self) -> Path:
        return ROOT / "sample.csv"

    @property
    def submission_path(self) -> Path:
        return ROOT / "submission.csv"

    @property
    def selected_features_path(self) -> Path:
        return ROOT / "selected_features.csv"

    @property
    def results_dir(self) -> Path:
        return ROOT / "experiments"

    @property
    def results_path(self) -> Path:
        return self.results_dir / "cv_results.csv"

    def cv(self):
        """RepeatedKFold used for every honest evaluation in the project."""
        from sklearn.model_selection import RepeatedKFold
        return RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats,
                             random_state=self.seed)

    def inner_cv(self):
        """KFold used inside the stacking meta-learner and for OOF diagnostics."""
        from sklearn.model_selection import KFold
        return KFold(self.n_splits, shuffle=True, random_state=self.seed)
