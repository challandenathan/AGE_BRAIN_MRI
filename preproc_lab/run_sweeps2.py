"""preproc_lab sweep #2 — info-preserving outlier handling + new models.

Section A tests ways to KEEP the outlier information instead of deleting it
(mask -> NaN -> median): signed-log scaling instead of the quantile transform,
winsorizing, reconstructing masked cells from the other selected features
(KNN / iterative imputation on the k=180 selected columns), no masking.
Section B tests new models on the base preprocessing.

Model hyperparameters are FIXED (tuned defaults) — no re-tuning here.
Protocol: RepeatedKFold(5x2, seed 42), whole pipeline refit per fold.
Sweep model: tuned SVR (fast). Finalists: tuned vote ensemble.

Run:  python preproc_lab/run_sweeps2.py
Results: preproc_lab/results/sweep2_*.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parent
REPO = LAB.parent
sys.path.insert(0, str(LAB))

from brainage.config import Config
from brainage.data import load_data
from brainage.evaluate import cross_validate
from brainage.models import build_model

RESULTS = LAB / "results"


def cfg_of(**kw):
    c = Config()
    c.data_dir = REPO / "data"
    c.n_repeats = 2
    c.gp_n_restarts = 1
    c.outlier_thresh = 5.0          # adopted from sweep #1
    for k, v in kw.items():
        setattr(c, k, v)
    return c


data = load_data(cfg_of())
X, y = data.X, data.y
rows = []
KW = {}


def run(tag, model_name, **kw):
    c = cfg_of(**kw)
    m = build_model(model_name, c)
    if hasattr(m, "n_jobs"):
        m.set_params(n_jobs=1)
    t = time.time()
    s = cross_validate(m, X, y, c, n_jobs=-1)
    row = dict(variant=tag, model=model_name, mean_r2=s.mean(), std_r2=s.std(),
               secs=round(time.time() - t))
    row.update({k: str(v) for k, v in kw.items()})
    rows.append(row)
    KW[tag] = dict(kw)
    print(f"{tag:<38} {model_name:<8} R2={s.mean():.4f}+/-{s.std():.4f} "
          f"({row['secs']}s)", flush=True)
    return row


BASE = dict(outlier_mode="mad", outlier_thresh=5.0, impute="median",
            selection="kbest", score_func="f_regression", n_features=180)

print("=== A. outlier handling that PRESERVES the data (tuned SVR, 5x2) ===", flush=True)
run("A0 BASE mad5/median/QT/kbestF180", "svr", **BASE)
run("A1 no mask (QT squashes spikes)", "svr", **{**BASE, "outlier_mode": "none",
                                                  "outlier_thresh": np.inf})
run("A2 winsor8 instead of delete", "svr", **{**BASE, "outlier_mode": "winsor",
                                               "outlier_thresh": 8.0})
run("A3 signed-log scale, no QT, mad5", "svr", **{**BASE, "prescale": "logscale"})
run("A4 signed-log scale, no QT, no mask", "svr", **{**BASE, "prescale": "logscale",
                                                      "outlier_mode": "none",
                                                      "outlier_thresh": np.inf})
run("A5 reconstruct cells: KNN k=10 on sel180", "svr", **{**BASE, "smart_impute": "knn"})
run("A6 reconstruct cells: iterative (BR,3)", "svr", **{**BASE, "smart_impute": "iter"})
run("A7 mad3 (aggressive delete, contrast)", "svr", **{**BASE, "outlier_thresh": 3.0})
run("A8 mad5 + iter reconstruct + no QT? -> logscale+iter", "svr",
    **{**BASE, "smart_impute": "iter", "prescale": "logscale"})

print("=== B. new models on BASE preprocessing (5x2) ===", flush=True)
run("B0 vote (reference)", "vote", **BASE)
run("B1 mlp (256,128,64)", "mlp", **BASE)
run("B2 gbr (300x d3 lr.05)", "gbr", **BASE)
run("B3 kernel ridge rbf", "kridge", **BASE)
run("B4 linear SVR k=180", "linsvr", **BASE)
run("B5 linear SVR k=400", "linsvr", **{**BASE, "n_features": 400})
run("B6 stack2 svr+gp+boost (RidgeCV meta)", "stack2", **BASE)
if "lgbm" in __import__("brainage.models", fromlist=["MODELS"]).MODELS:
    run("B7 lightgbm (tuned-ish defaults)", "lgbm", **BASE)
if "ngboost" in __import__("brainage.models", fromlist=["MODELS"]).MODELS:
    run("B8 ngboost Normal 500", "ngboost", **BASE)
run("B9 xgb (reference)", "xgb", **BASE)
run("B10 histgb (reference)", "histgb", **BASE)

df = pd.DataFrame(rows).sort_values("mean_r2", ascending=False).reset_index(drop=True)
RESULTS.mkdir(exist_ok=True)
df.to_csv(RESULTS / "sweep2_all.csv", index=False)
print("\n=== RANKED ===", flush=True)
print(df[["variant", "model", "mean_r2", "std_r2"]].round(4).to_string(index=False),
      flush=True)

# finalists: best 2 from section A and best 2 new models from B, re-scored with vote
print("\n=== finalists with tuned VOTE (5x2) ===", flush=True)
secA = df[df["variant"].str.startswith("A")].head(2)
secB = df[df["variant"].str.startswith("B")].head(2)
fin = []
for _, r in list(secA.iterrows()) + list(secB.iterrows()):
    kw = KW[r["variant"]]
    model_name = "vote" if r["variant"].startswith("A") else r["model"]
    row = run(f"FIN {r['variant']}", model_name, **kw)
    fin.append(row)
pd.DataFrame(fin).to_csv(RESULTS / "sweep2_finalists.csv", index=False)
print("DONE", flush=True)
