"""Inspect the spike-heavy rows and test (manual CV) whether dropping the few
clearly corrupted rows from the *training* folds helps."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab.pipeline import load_data, compute_block_mask, build_model, SEED

OUT = Path(__file__).parent / "results"
X, y, Xt, _, _ = load_data()
NAMES = np.asarray(X.columns, dtype=object)
Xf = X.to_numpy(float)
n, p = Xf.shape
BLOCK = compute_block_mask(pd.concat([X, Xt], axis=0).to_numpy(float))

med = np.nanmedian(Xf, axis=0)
mad = np.nanmedian(np.abs(Xf - med), axis=0) / 0.6745
scale = np.where(mad > 0, mad, np.inf)
Z = (Xf - med) / scale

for r in [740, 245, 937, 899, 681]:
    cells = np.where(np.abs(Z[r]) > 8)[0]
    print(f"row {r} y={y[r]}: spiked cells:")
    for c in cells:
        print(f"   {NAMES[c]:>6} val={Xf[r,c]:.4g} med={med[c]:.4g} z={Z[r,c]:+.1f}")

# how extreme is 8 spikes under a random model? expected spikes per row:
print(f"\nexpected rows with >=3 spikes if independent p(>8sigma)~0.02%*832=0.17/row: "
      f"{n*0.0017:.1f}")

# manual CV: baseline SVR, with and without dropping flagged rows from train folds
o = dict(mask="adakurt", block="none", k=180, score="f")
model = build_model("svr", o, BLOCK, NAMES)

def manual_cv(model, X, y, drop_rows, n_splits=5, n_repeats=2, seed=SEED):
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = []
    for tr, va in cv.split(X, y):
        tr_eff = np.setdiff1d(tr, drop_rows)
        m = model
        m.fit(X.iloc[tr_eff], y[tr_eff])
        pred = m.predict(X.iloc[va])
        scores.append(r2_score(y[va], pred))
    return np.array(scores)

base = manual_cv(model, X, y, drop_rows=[])
for drop in [[740], [740, 245, 937], [740, 245, 937, 899, 681, 646, 45, 524, 729, 655]]:
    s = manual_cv(model, X, y, drop_rows=drop)
    print(f"drop {drop}: R2 = {s.mean():.4f} +/- {s.std():.4f}   (base {base.mean():.4f} +/- {base.std():.4f})")
