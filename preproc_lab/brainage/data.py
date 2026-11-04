"""Loading and validation of the competition data, and submission writing."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Data:
    X: pd.DataFrame        # training features, id column removed
    y: np.ndarray          # ages, shape (n_train,)
    X_test: pd.DataFrame   # test features, id column removed
    train_ids: np.ndarray
    test_ids: np.ndarray


def load_data(cfg) -> Data:
    """Load X_train/y_train/X_test and validate their integrity.

    Raises ValueError on any inconsistency, so a corrupted download fails
    loudly instead of silently producing a wrong submission.
    """
    xtr = pd.read_csv(cfg.x_train_path)
    ytr = pd.read_csv(cfg.y_train_path)
    xte = pd.read_csv(cfg.x_test_path)

    if "id" not in xtr.columns or "id" not in ytr.columns or "id" not in xte.columns:
        raise ValueError("all files must contain an 'id' column")
    if not (xtr["id"].to_numpy() == ytr["id"].to_numpy()).all():
        raise ValueError("id mismatch between X_train.csv and y_train.csv")
    if xtr["id"].duplicated().any() or xte["id"].duplicated().any():
        raise ValueError("duplicate ids found")
    if "y" not in ytr.columns:
        raise ValueError("y_train.csv must contain a 'y' column")

    X = xtr.drop(columns="id")
    X_test = xte.drop(columns="id")
    if list(X.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test feature columns differ")
    if not np.isfinite(ytr["y"].to_numpy(dtype=float)).all():
        raise ValueError("y_train contains non-finite values")

    return Data(
        X=X,
        y=ytr["y"].to_numpy(dtype=float),
        X_test=X_test,
        train_ids=xtr["id"].to_numpy(),
        test_ids=xte["id"].to_numpy(),
    )


def write_submission(path, ids, preds, sample_path) -> pd.DataFrame:
    """Write predictions in the exact sample.csv format (columns id,y).

    The ids are checked against sample.csv so a re-ordered test set cannot
    silently scramble the submission.
    """
    sample = pd.read_csv(sample_path)
    if list(sample.columns) != ["id", "y"]:
        raise ValueError(f"sample.csv must have columns ['id', 'y'], got {list(sample.columns)}")
    if not (sample["id"].to_numpy() == np.asarray(ids)).all():
        raise ValueError("ids do not match sample.csv — refusing to write a scrambled submission")
    if len(preds) != len(sample):
        raise ValueError(f"expected {len(sample)} predictions, got {len(preds)}")
    if not np.isfinite(preds).all():
        raise ValueError("predictions contain NaN/inf")

    out = sample[["id"]].copy()
    out["y"] = np.asarray(preds, dtype=float)
    out.to_csv(path, index=False)
    return out
