"""brainage — AML 2026 Task 1: predict brain age from 832 perturbed MRI features.

The package implements the four subtasks from the project slides as explicit,
leak-free pipeline steps:

    0. imputation          SimpleImputer(median)            (after outlier masking)
    1. outlier detection   MADOutlierMasker (median/MAD)    (spikes -> NaN -> imputed)
    2. feature selection   SelectKBest(f_regression, k)
    3. regression          SVR (RBF) or a Ridge+XGB+GP stack

All steps live inside a single scikit-learn Pipeline, so their statistics are
re-estimated on the training part of every CV fold (no leakage).
"""
from .config import Config
from .data import Data, load_data, write_submission
from .models import build_model, MODELS
from .evaluate import cross_validate, compare_models

__all__ = [
    "Config",
    "Data",
    "load_data",
    "write_submission",
    "build_model",
    "MODELS",
    "cross_validate",
    "compare_models",
]
