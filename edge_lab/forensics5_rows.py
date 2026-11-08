"""Forensics round 5: are there outlier ROWS (slides subtask 1 says outliers
are *samples* that do not follow the data)? Look at per-row spike counts,
robust Mahalanobis distance in the signal block, and CV residuals."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).parent / "results"

X = pd.read_csv(DATA / "X_train.csv").drop(columns="id")
y = pd.read_csv(DATA / "y_train.csv")["y"].to_numpy(float)
Xf = X.to_numpy(float)
n, p = Xf.shape

in_block = np.load(OUT / "in_block.npy")
BLOCK = in_block.copy()
# align to 828 after dropping constants: constants are x104,x129,x489,x530
const = np.where(np.nanstd(Xf, axis=0) == 0)[0]
print("constant cols:", const, [X.columns[c] for c in const])
keep = np.ones(p, bool); keep[const] = False
BLOCK828 = BLOCK[keep]

med = np.nanmedian(Xf, axis=0)
mad = np.nanmedian(np.abs(Xf - med), axis=0) / 0.6745
scale = np.where(mad > 0, mad, np.inf)
Z = (Xf - med) / scale
nspike = (np.abs(Z) > 8).sum(axis=1)
print(f"\nper-row spike count (>8 sigma): max={nspike.max()}, rows with >=1: {(nspike>=1).sum()}, "
      f">=3: {(nspike>=3).sum()}, >=5: {(nspike>=5).sum()}")
top_rows = np.argsort(-nspike)[:20]
print("rows with most spikes (row, nspikes, y):")
for i in top_rows[:20]:
    print(f"  row {i} (id={i}): nspike={nspike[i]}, y={y[i]}")

# robust Mahalanobis in block space on rank-transformed block features
Xi = np.where(np.isnan(Xf), med, Xf)
Xb = pd.DataFrame(Xi[:, keep]).rank().to_numpy()
Xb = Xb[:, BLOCK828]
mcd = MinCovDet(random_state=42, support_fraction=0.9).fit(Xb)
d2 = mcd.mahalanobis(Xb)
thr = chi2.ppf(0.999, Xb.shape[1])
print(f"\nMCD Mahalanobis (block, {Xb.shape[1]} dims): thr(0.999 chi2)={thr:.0f}")
flag = d2 > thr
print(f"rows beyond thr: {flag.sum()}")
order = np.argsort(-d2)
print("top-25 rows by d2 (row, d2, y, nspike):")
for i in order[:25]:
    print(f"  row {i}: d2={d2[i]:.0f} y={y[i]} nspike={nspike[i]}")
print("d2 quantiles:", np.round(np.percentile(d2, [50, 90, 95, 99, 99.5, 100]), 0))

# CV residuals of a quick SVR on block top features -> consensus outlier rows
from sklearn.feature_selection import SelectKBest, f_regression
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("qt", QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=42)),
    ("sel", SelectKBest(f_regression, k=180)),
    ("m", TransformedTargetRegressor(SVR(C=80.03, gamma=0.010403, epsilon=0.01),
                                      transformer=StandardScaler())),
])
oof = cross_val_predict(pipe, X.to_numpy(float), y, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1)
res = np.abs(y - oof)
print(f"\nOOF SVR R2={1 - ((y-oof)**2).sum()/((y-y.mean())**2).sum():.4f}")
print("top-25 rows by |residual| (row, res, y, pred, nspike, d2):")
for i in np.argsort(-res)[:25]:
    print(f"  row {i}: res={res[i]:.1f} y={y[i]} pred={oof[i]:.1f} nspike={nspike[i]} d2={d2[i]:.0f}")
print("res quantiles:", np.round(np.percentile(res, [50, 90, 95, 99, 100]), 1))

# overlap: rows flagged by both d2 and residual
both = set(np.where(flag)[0]) & set(np.argsort(-res)[:30])
print(f"\nrows in top-30 residual AND MCD-flagged: {sorted(both)}")
spiky_rows = set(np.where(nspike >= 3)[0])
print(f"rows with >=3 spikes: {sorted(spiky_rows)}")
print(f"overlap spiky & residual-top30: {sorted(spiky_rows & set(np.argsort(-res)[:30]))}")

np.save(OUT / "row_mahal.npy", d2)
np.save(OUT / "row_nspike.npy", nspike)
np.save(OUT / "row_resid.npy", res)
pd.DataFrame({"row": np.arange(n), "y": y, "nspike": nspike, "mahal_d2": d2,
              "abs_resid": res}).to_csv(OUT / "row_flags.csv", index=False)
print("\nsaved row_flags.csv")
