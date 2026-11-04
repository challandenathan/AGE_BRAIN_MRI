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
    outlier_thresh: float = 8.0     # robust sigmas; np.inf disables masking

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
    model: str = "vote"             # key of models.MODELS; "vote" = average of
                                    # SVR + GP + boosted trees (best repeated-CV
                                    # score; "svr" alone is within noise of it)
    svr_C: float = 166.5            # tuned by RandomizedSearchCV (see README)
    svr_gamma: float = 0.00907
    svr_epsilon: float = 0.03

    xgb: dict = field(default_factory=lambda: dict(
        n_estimators=400, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
    ))
    gp_n_restarts: int = 2

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
