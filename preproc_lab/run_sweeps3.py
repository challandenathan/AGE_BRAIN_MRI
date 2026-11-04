"""preproc_lab sweep #3 — precise case-by-case outlier rules, more selectors,
exotic models. Model hyperparameters stay FIXED (tuned defaults).

Protocol: RepeatedKFold(5x2, seed 42), whole pipeline refit per fold.
  Section C: precise outlier rules   (sweep model: tuned SVR, fast)
  Section D: selection methods       (sweep model: tuned SVR)
  Section E: exotic models           (each model itself, base preprocessing)
Finalists re-scored with the stack2 ensemble (5x2).

Run:  python preproc_lab/run_sweeps3.py
Results: preproc_lab/results/sweep3_*.csv
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
from brainage.models import build_model, MODELS

RESULTS = LAB / "results"


def cfg_of(**kw):
    c = Config()
    c.data_dir = REPO / "data"
    c.n_repeats = 2
    c.gp_n_restarts = 1
    c.outlier_thresh = 5.0
    for k, v in kw.items():
        setattr(c, k, v)
    return c


data = load_data(cfg_of())
X, y = data.X, data.y
rows, KW = [], {}


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
    print(f"{tag:<44} {model_name:<9} R2={s.mean():.4f}+/-{s.std():.4f} "
          f"({row['secs']}s)", flush=True)
    return row


BASE = dict(outlier_mode="mad", outlier_thresh=5.0, impute="median",
            selection="kbest", score_func="f_regression", n_features=180)

print("=== C. precise case-by-case outlier rules (tuned SVR, 5x2) ===", flush=True)
run("C0 BASE mad5", "svr", **BASE)
run("C1 spike: z>5 AND x>3*med, |z|>12", "svr", **{**BASE, "outlier_mode": "spike",
                                                    "outlier_ratio": 3.0})
run("C2 spike ratio=2", "svr", **{**BASE, "outlier_mode": "spike", "outlier_ratio": 2.0})
run("C3 spike ratio=4", "svr", **{**BASE, "outlier_mode": "spike", "outlier_ratio": 4.0})
run("C4 ratio-only x>4*med", "svr", **{**BASE, "outlier_mode": "ratio", "outlier_ratio": 4.0})
run("C5 tukey fence k=3", "svr", **{**BASE, "outlier_mode": "tukey", "outlier_thresh": 3.0})
run("C6 adaptive kurt/skew MAD", "svr", **{**BASE, "outlier_mode": "adakurt"})
run("C7 winsor at empirical q99.5", "svr", **{**BASE, "outlier_mode": "winsor_q",
                                               "cap_quantile": 0.995})
run("C8 winsor at empirical q99.9", "svr", **{**BASE, "outlier_mode": "winsor_q",
                                               "cap_quantile": 0.999})
run("C9 LOF rows cont=0.02", "svr", **{**BASE, "outlier_mode": "lof",
                                        "row_contamination": 0.02})
# C10 MCD skipped: EllipticEnvelope/MCD is singular at p=828 ~ n (slogdet spam, no result)
print("C10 MCD skipped (p~=n, singular covariance)", flush=True)

print("=== D. selection methods (tuned SVR, base outliers) ===", flush=True)
run("D0 kbest F 180 (BASE)", "svr", **BASE)
run("D1 boruta shadow ET", "svr", **{**BASE, "selection": "boruta"})
run("D2 ridge-coef top180", "svr", **{**BASE, "selection": "ridge"})
run("D3 stability selection freq=0.5", "svr", **{**BASE, "selection": "stability"})
run("D4 cluster-diverse corr0.8", "svr", **{**BASE, "selection": "cluster",
                                             "corr_thresh": 0.8})
run("D5 RFECV ridge min50", "svr", **{**BASE, "selection": "rfecv"})
run("D6 kbest spearman 180", "svr", **{**BASE, "score_func": "spearman"})

print("=== E. exotic models (base preprocessing, each 5x2) ===", flush=True)
run("E0 stack2 (reference)", "stack2", **BASE)
run("E1 ARD regression", "ard", **BASE)
run("E2 OMP 50 coefs", "omp", **BASE)
run("E3 quantile reg median", "qreg", **BASE)
run("E4 RANSAC linear", "ransac", **BASE)
run("E5 monotonic XGB (rho signs)", "monoxgb", **BASE)
run("E6 stack3 (+ET member)", "stack3", **BASE)
if "catboost" in MODELS:
    run("E7 catboost", "catboost", **BASE)

df = pd.DataFrame(rows).sort_values("mean_r2", ascending=False).reset_index(drop=True)
RESULTS.mkdir(exist_ok=True)
df.to_csv(RESULTS / "sweep3_all.csv", index=False)
print("\n=== RANKED ===", flush=True)
print(df[["variant", "model", "mean_r2", "std_r2"]].round(4).to_string(index=False),
      flush=True)

print("\n=== finalists with stack2 (5x2) ===", flush=True)
fin = []
for section in ["C", "D"]:
    for _, r in df[df["variant"].str.startswith(section)].head(2).iterrows():
        fin.append(run(f"FIN {r['variant']}", "stack2", **KW[r["variant"]]))
pd.DataFrame(fin).to_csv(RESULTS / "sweep3_finalists.csv", index=False)
print("DONE", flush=True)
