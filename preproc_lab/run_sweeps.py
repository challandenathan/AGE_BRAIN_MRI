"""preproc_lab — sweep alternative NaN / outlier / feature-selection techniques.

The model is FIXED to the already-tuned configuration (no re-tuning with the
new techniques): the sweep uses the tuned SVR (fast, ~1.3 s/fold) under
RepeatedKFold(5x2), then the top-3 preprocessing variants are re-scored with
the full tuned vote ensemble (SVR+GP+XGB) under the same 5x2 protocol.

Run:  python preproc_lab/run_sweeps.py     (from the repo root)
Results: preproc_lab/results/preproc_sweep.csv, preproc_finalists.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parent
REPO = LAB.parent
sys.path.insert(0, str(LAB))          # use the LAB copy of brainage, not the main one

from brainage.config import Config
from brainage.data import load_data
from brainage.evaluate import cross_validate
from brainage.models import build_model

RESULTS = LAB / "results"


def cfg_of(**kw):
    c = Config()
    c.data_dir = REPO / "data"        # lab reads the repo's data
    c.n_repeats = 2                   # 5x2 CV for the sweep
    c.gp_n_restarts = 1
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def run(tag, model_name, **kw):
    c = cfg_of(**kw)
    m = build_model(model_name, c)
    if hasattr(m, "n_jobs"):
        m.set_params(n_jobs=1)
    t = time.time()
    s = cross_validate(m, X, y, c, n_jobs=-1)
    row = dict(variant=tag, model=model_name, mean_r2=s.mean(), std_r2=s.std(),
               secs=round(time.time() - t))
    row.update({k: (str(v)) for k, v in kw.items()})
    rows.append(row)
    KW[tag] = dict(kw)
    print(f"{tag:<34} {model_name:<5} R2={s.mean():.4f}+/-{s.std():.4f} "
          f"({row['secs']}s)", flush=True)
    return row


data = load_data(cfg_of())
X, y = data.X, data.y
rows = []
KW = {}

BASE = dict(outlier_mode="mad", outlier_thresh=8.0, impute="median",
            selection="kbest", score_func="f_regression", n_features=180)

print("=== 0. baseline (current pipeline, tuned SVR) ===", flush=True)
run("BASE mad8/median/kbestF180", "svr", **BASE)

print("=== 1. outlier handling variants ===", flush=True)
for mode, th in [("mad", 5.0), ("mad", 12.0), ("mad_high", 8.0), ("winsor", 8.0),
                 ("iqr", 8.0), ("zscore", 8.0), ("none", np.inf)]:
    run(f"outlier {mode} th={th}", "svr", **{**BASE, "outlier_mode": mode,
                                             "outlier_thresh": th})
for cont in [0.01, 0.03]:
    run(f"outlier iforest cont={cont}", "svr", **{**BASE, "outlier_mode": "iforest",
                                                  "row_contamination": cont})

print("=== 2. NaN imputation variants (mad8 fixed) ===", flush=True)
for imp in ["mean", "zero"]:
    run(f"impute {imp}", "svr", **{**BASE, "impute": imp})
run("impute median+indicator", "svr", **{**BASE, "add_indicator": True})
for k in [3, 5, 10]:
    run(f"impute knn{k} on QT'd feats", "svr", **{**BASE, "knn_impute": k})

print("=== 3. feature selection variants (mad8/median fixed) ===", flush=True)
run("select kbest spearman 180", "svr", **{**BASE, "score_func": "spearman"})
run("select tree(ET) top180", "svr", **{**BASE, "selection": "tree"})
run("select pca 100", "svr", **{**BASE, "selection": "pca", "n_features": 100})
run("select pca 180", "svr", **{**BASE, "selection": "pca", "n_features": 180})
run("select corr0.95+kbestF180", "svr", **{**BASE, "selection": "corr_kbest",
                                           "corr_thresh": 0.95})
run("select kbestF150", "svr", **{**BASE, "n_features": 150})
run("select kbestF200", "svr", **{**BASE, "n_features": 200})
run("select lasso top180", "svr", **{**BASE, "selection": "lasso"})

sweep = pd.DataFrame(rows).sort_values("mean_r2", ascending=False).reset_index(drop=True)
RESULTS.mkdir(exist_ok=True)
sweep.to_csv(RESULTS / "preproc_sweep.csv", index=False)
print("\n=== SWEEP RANKED (tuned SVR, 5x2) ===", flush=True)
print(sweep[["variant", "model", "mean_r2", "std_r2"]].round(4).to_string(index=False),
      flush=True)

# confirm the top-3 distinct variants with the full tuned vote ensemble
print("\n=== finalists with the tuned VOTE ensemble (5x2) ===", flush=True)
fin_rows = []
top = sweep.head(3)
for _, r in top.iterrows():
    kw = KW[r["variant"]]
    run(f"FINALIST {r['variant']}", "vote", **kw)
    fin_rows.append(rows[-1])
fin = pd.DataFrame(fin_rows)
fin.to_csv(RESULTS / "preproc_finalists.csv", index=False)
print("\n=== FINALISTS (tuned vote, 5x2) ===", flush=True)
print(fin[["variant", "mean_r2", "std_r2"]].round(4).to_string(index=False), flush=True)
print("DONE", flush=True)
