"""Final evaluation: score the finalists under the headline protocol
RepeatedKFold(5x3, seed 42), pick the winner, fit on all training data and
write edge_lab/submission.csv (sample.csv format, clipped to [42, 97]).

Usage: python3 run_final.py [--no-submit]
"""
import sys, time, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab.pipeline as P
from lab.pipeline import (SEED, load_data, compute_block_mask, cv_scores, report)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).parent / "results"
X, y, Xt, ids_tr, ids_te = load_data()
NAMES = np.asarray(X.columns, dtype=object)
BLOCK = compute_block_mask(pd.concat([X, Xt], axis=0).to_numpy(float))
print(f"block: {BLOCK.sum()} features; train {X.shape}, test {Xt.shape}", flush=True)

# tuned member params from sweep3
svr_t = json.load(open(OUT / "svr_tuned.json"))
xgb_t = json.load(open(OUT / "xgb_tuned.json"))
P.SVR_PARAMS.update({"C": svr_t["model__regressor__C"],
                     "gamma": svr_t["model__regressor__gamma"],
                     "epsilon": svr_t["model__regressor__epsilon"]})
P.XGB_OVERRIDES.update({k.replace("model__", ""): v for k, v in xgb_t.items()
                        if k != "pre__select__k"})

BASE180 = dict(mask="adakurt", block="trans", k=180, score="f")
BASE = dict(mask="adakurt", block="trans", k=200, score="f")
K229 = dict(BASE, k=229)
K300 = dict(BASE, k=300)
LOGMAD200 = dict(mask="logmad", log=True, mask_base=4.0, block="trans", k=200, score="f")
LOW30 = dict(BASE, k=30, score="spearman")
WIDE = dict(mask="none", log=False, block="none", k=10000, score="f")
WIDE650 = dict(WIDE, k=650)

def stack(entries):
    ests = [(nm, Pipeline([("pre", P._pre(o, bm, NAMES)), ("model", reg)]))
            for nm, o, bm, reg in entries]
    return StackingRegressor(estimators=ests, final_estimator=RidgeCV(alphas=P.RIDGE_ALPHAS),
                             cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)

SVR, GP, XGB = P._svr(), P._gp(), P._xgb()
MAIN = [("svr", BASE180, BLOCK, SVR), ("gp", BASE180, BLOCK, GP), ("xgb", BASE180, BLOCK, XGB)]
W = [("svr", LOGMAD200, BLOCK, SVR), ("gp", K229, BLOCK, GP), ("xgb", K300, BLOCK, XGB)]

FINALISTS = {
    "A main stack2 k180 (ref)": MAIN,
    "B W svr-logmad200+gp229+xgb300": W,
    "C W + wide828-nomask": W + [("wsvr", WIDE, None, SVR)],
    "D W + wide828-nomask + low30": W + [("wsvr", WIDE, None, SVR), ("low30", LOW30, BLOCK, SVR)],
    "E W + wide650-nomask + low30": W + [("wsvr", WIDE650, None, SVR), ("low30", LOW30, BLOCK, SVR)],
    "G svr+gp + wide828 + low30 (no xgb)": [W[0], W[1], ("wsvr", WIDE, None, SVR),
                                             ("low30", LOW30, BLOCK, SVR)],
}

ap = argparse.ArgumentParser()
ap.add_argument("--no-submit", action="store_true")
ap.add_argument("--only", default=None, help="comma-separated finalist keys to run")
args = ap.parse_args()

rows = []
t0 = time.time()
models = {}
for name, entries in FINALISTS.items():
    if args.only and not any(k in name for k in args.only.split(",")):
        continue
    m = stack(entries)
    s = cv_scores(m, X, y, n_splits=5, n_repeats=3, seed=SEED)
    report(name, s, rows)
    models[name] = (entries, s)
    print(f"    ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows).sort_values("mean_r2", ascending=False)
df.to_csv(OUT / "final_5x3.csv", index=False)
print("\n=== 5x3 finalists ===")
print(df.to_string(index=False))

best = df.iloc[0]["name"]
entries, s = models[best]
print(f"\nWINNER: {best}  CV R2 = {s.mean():.4f} +/- {s.std():.4f}", flush=True)
np.save(OUT / "winner_fold_scores.npy", s)

if not args.no_submit:
    # seed-averaged submission: average 3 stacks with different inner-CV seeds
    preds = []
    for sd in [42, 7, 123]:
        ests = [(nm, Pipeline([("pre", P._pre(o, bm, NAMES)), ("model", reg)]))
                for nm, o, bm, reg in entries]
        m = StackingRegressor(estimators=ests, final_estimator=RidgeCV(alphas=P.RIDGE_ALPHAS),
                              cv=KFold(5, shuffle=True, random_state=sd), n_jobs=-1)
        m.fit(X, y)
        preds.append(m.predict(Xt))
        print(f"  fit seed {sd} done", flush=True)
    preds = np.clip(np.mean(preds, axis=0), 42, 97)
    sample = pd.read_csv(ROOT / "sample.csv")
    assert (sample["id"].to_numpy() == ids_te).all()
    out = sample[["id"]].copy()
    out["y"] = preds
    dest = Path(__file__).parent / "submission.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest} ({len(out)} rows, pred range [{preds.min():.1f}, {preds.max():.1f}])")
    main = pd.read_csv(ROOT / "submission.csv")
    if (main["id"].to_numpy() == ids_te).all():
        print(f"corr with main submission.csv predictions: "
              f"{np.corrcoef(main['y'], preds)[0, 1]:.4f}")
