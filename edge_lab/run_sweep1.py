"""Sweep 1: feature-selection strategies with the tuned SVR (fast), 5x2 CV.

Compares the main project's F-top-180-from-all-832 against selecting
*within the structural signal block* at various k, log-space masking,
pair-averaging and reliability weighting.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab.pipeline import (SEED, load_data, compute_block_mask, build_model,
                          cv_scores, report)

OUT = Path(__file__).parent / "results"
X, y, Xt, _, _ = load_data()
NAMES = np.asarray(X.columns, dtype=object)

# transductive block mask: features of train+test only, no y involved
BLOCK = compute_block_mask(pd.concat([X, Xt], axis=0).to_numpy(float))
print(f"signal block: {BLOCK.sum()} / {len(BLOCK)} features")

CONFIGS = [
    # name, preprocessor opts, model
    ("base F-180 all (main proj)", dict(mask="adakurt", block="none", k=180, score="f"), "svr"),
    ("block k=180 F",              dict(mask="adakurt", block="trans", k=180, score="f"), "svr"),
    ("block k=200 F",              dict(mask="adakurt", block="trans", k=200, score="f"), "svr"),
    ("block k=229 F (all block)",  dict(mask="adakurt", block="trans", k=229, score="f"), "svr"),
    ("block k=229 spearman",       dict(mask="adakurt", block="trans", k=229, score="spearman"), "svr"),
    ("block k=180 spearman",       dict(mask="adakurt", block="trans", k=180, score="spearman"), "svr"),
    ("block fold k=229 F",         dict(mask="adakurt", block="fold", k=229, score="f"), "svr"),
    ("block k=229 no mask",        dict(mask="none", log=False, block="trans", k=229, score="f"), "svr"),
    ("logmad4 block k=229 F",      dict(mask="logmad", log=True, block="trans", k=229, score="f"), "svr"),
    ("logmad4 block k=180 F",      dict(mask="logmad", log=True, block="trans", k=180, score="f"), "svr"),
    ("logmad3 block k=229 F",      dict(mask="logmad", log=True, mask_base=3.0, block="trans", k=229, score="f"), "svr"),
    ("logmad4 nomask+log block229", dict(mask="none", log=True, block="trans", k=229, score="f"), "svr"),
    ("adakurt block229 pairavg.9", dict(mask="adakurt", block="trans", pairavg=0.9, k=10000, score="f"), "svr"),
    ("adakurt block229 pairavg.8", dict(mask="adakurt", block="trans", pairavg=0.8, k=10000, score="f"), "svr"),
    ("logmad4 block229 weight",    dict(mask="logmad", log=True, block="trans", k=229, score="f", weight=True), "svr"),
    ("adakurt all832 xgb",         dict(mask="adakurt", block="none", k=10000, score="f"), "xgb"),
    ("adakurt block229 xgb",       dict(mask="adakurt", block="trans", k=229, score="f"), "xgb"),
]

rows = []
t0 = time.time()
for name, o, model in CONFIGS:
    m = build_model(model, o, BLOCK, NAMES)
    s = cv_scores(m, X, y, n_splits=5, n_repeats=2, seed=SEED)
    report(f"{name} [{model}]", s, rows)
    print(f"    ({time.time()-t0:.0f}s elapsed)", flush=True)

df = pd.DataFrame(rows).sort_values("mean_r2", ascending=False)
df.to_csv(OUT / "sweep1_selection.csv", index=False)
print("\n=== sweep1 summary ===")
print(df.to_string(index=False))
print(f"\ntotal {time.time()-t0:.0f}s; saved results/sweep1_selection.csv")
