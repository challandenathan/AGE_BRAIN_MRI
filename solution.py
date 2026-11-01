"""AML project — Task 1: predict brain age from (perturbed) MRI features.

Pipeline (all steps fitted inside each CV fold, so evaluation is leak-free):
  1. Outlier detection : MADOutlierMasker — entries more than `thresh` robust
                         sigmas (median / MAD) from their column median are set
                         to NaN and later imputed.
  2. Imputation        : per-feature median (robust to the remaining outliers).
  3. Scaling           : QuantileTransformer -> normal output (handles the ~100
                         features that were rescaled to ~1e6 and the spike
                         outliers by mapping them to ranks).
  4. Feature selection : SelectKBest with f_regression, k=180 (832 -> 180).
  5. Regression        : SVR with RBF kernel on standardized y
                         (C=166.5, gamma=0.00907, epsilon=0.03), hyperparameters
                         tuned by RandomizedSearchCV (see DESCRIPTION.md).

Evaluation: R^2 via RepeatedKFold(5 folds x 3 repeats). Then the pipeline is
fitted on all of X_train and predictions for X_test are written to
submission.csv in the sample.csv format.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR


class MADOutlierMasker(BaseEstimator, TransformerMixin):
    """Set entries more than `thresh` robust sigmas from the column median to NaN.

    Median/MAD are learned on the training fold only. Columns with zero MAD fall
    back to the scaled IQR; constant columns are never flagged. thresh=np.inf
    disables the step.
    """

    def __init__(self, thresh=8.0):
        self.thresh = thresh

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.med_ = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - self.med_), axis=0)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        scale = np.where(mad > 0, mad / 0.6745, (q3 - q1) / 1.349)
        self.scale_ = np.where(scale > 0, scale, np.inf)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)  # copy
        X[np.abs(X - self.med_) / self.scale_ > self.thresh] = np.nan
        return X


def make_pipeline(k=180, thresh=8.0):
    model = TransformedTargetRegressor(
        SVR(C=166.5, gamma=0.00907, epsilon=0.03),
        transformer=StandardScaler(),  # standardize y so C/epsilon are meaningful
    )
    return Pipeline([
        ('outlier', MADOutlierMasker(thresh)),                            # 1. outlier detection
        ('impute', SimpleImputer(strategy='median')),                     # 2. imputation
        ('scale', QuantileTransformer(output_distribution='normal',
                                      n_quantiles=500, random_state=0)), # robust scaling
        ('select', SelectKBest(f_regression, k=k)),                       # 3. feature selection
        ('model', model),                                                 # 4. regression
    ])


def main():
    Xtr = pd.read_csv('X_train.csv')
    ytr = pd.read_csv('y_train.csv')
    Xte = pd.read_csv('X_test.csv')
    X = Xtr.drop(columns='id')
    y = ytr['y'].values

    pipe = make_pipeline()

    # --- honest evaluation: repeated 5-fold CV, R^2 ---
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='r2', n_jobs=-1)
    print(f'CV R^2 per fold : {np.round(scores, 4)}')
    print(f'CV R^2 = {scores.mean():.4f} ± {scores.std():.4f}  (5 folds x 3 repeats)')

    # --- fit on all training data, predict the test set ---
    pipe.fit(X, y)
    pred = np.clip(pipe.predict(Xte.drop(columns='id')), y.min(), y.max())

    sub = pd.read_csv('sample.csv')
    assert (sub['id'] == Xte['id']).all(), 'id mismatch with sample.csv'
    sub['y'] = pred
    sub.to_csv('submission.csv', index=False)
    print(f'wrote submission.csv ({len(sub)} rows), '
          f'pred range [{pred.min():.2f}, {pred.max():.2f}], y range [{y.min():.0f}, {y.max():.0f}]')


if __name__ == '__main__':
    main()
