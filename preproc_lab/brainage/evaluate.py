"""Honest evaluation helpers: repeated K-fold CV and model comparison.

The metric is the coefficient of determination R^2 (the competition metric),
estimated with RepeatedKFold so the score does not depend on one lucky split.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_val_score


def cross_validate(model, X, y, cfg, scoring="r2", n_jobs=-1) -> np.ndarray:
    """R^2 of `model` under the configured RepeatedKFold (whole pipeline refit
    inside every fold)."""
    return cross_val_score(model, X, y, cv=cfg.cv(), scoring=scoring, n_jobs=n_jobs)


def compare_models(models: dict, X, y, cfg, n_jobs=-1) -> pd.DataFrame:
    """CV-score every model in `models` and return a sorted summary table."""
    rows = []
    for name, model in models.items():
        s = cross_validate(model, X, y, cfg, n_jobs=n_jobs)
        rows.append({
            "model": name,
            "mean_r2": s.mean(),
            "std_r2": s.std(),
            "min_r2": s.min(),
            "max_r2": s.max(),
            "n_folds": len(s),
        })
        print(f"  {name:<8} CV R2 = {s.mean():.4f} +/- {s.std():.4f}")
    return pd.DataFrame(rows).sort_values("mean_r2", ascending=False).reset_index(drop=True)


def oof_predictions(model, X, y, cfg) -> np.ndarray:
    """Out-of-fold predictions (single shuffled KFold) — used to check whether
    two models make different errors and are worth blending."""
    return cross_val_predict(model, X, y, cv=cfg.inner_cv(), n_jobs=-1)


def residual_correlation(models: dict, X, y, cfg) -> pd.DataFrame:
    """Correlation of the OOF *errors* of each pair of models.

    Low correlation => the models fail on different subjects => stacking /
    averaging can add value. High correlation (>0.8) => they are redundant.
    """
    oof = pd.DataFrame({name: oof_predictions(m, X, y, cfg) for name, m in models.items()})
    resid = y[:, None] - oof.to_numpy()
    return pd.DataFrame(np.corrcoef(resid, rowvar=False),
                        index=oof.columns, columns=oof.columns)
