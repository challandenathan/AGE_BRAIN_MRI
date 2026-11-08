#!/usr/bin/env python3
"""Run the optimal brain-age pipeline from the `final/` folder.

    python3 solution.py --tune        # fine-tune each member (5x2 CV), cache tuned_params.json
    python3 solution.py --evaluate    # honest RepeatedKFold(5x3) CV of the winner stack
    python3 solution.py --submit      # fit on all data (3 inner seeds averaged), write submission.csv
    python3 solution.py               # evaluate + submit
    python3 solution.py --quick       # fast check: 5x2 CV + single-seed submission

All preprocessing lives inside sklearn pipelines, so CV refits everything on
the training part of each fold (no leakage). The only transductive step is the
signal-block mask, computed from train+test *features only* (no y).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from brainage_final import (SEED, ROOT, build_winner, compute_block_mask, cv_scores,
                            load_data, load_params, report, save_params, tune_all)

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="fine-tune each member and cache params")
    ap.add_argument("--evaluate", action="store_true", help="5x3 CV evaluation")
    ap.add_argument("--submit", action="store_true", help="fit on all data, write submission.csv")
    ap.add_argument("--quick", action="store_true", help="5x2 CV + single-seed submission")
    args = ap.parse_args()
    if not (args.tune or args.evaluate or args.submit or args.quick):
        args.evaluate = args.submit = True

    t0 = time.time()
    X, y, Xt, ids_tr, ids_te = load_data()
    NAMES = np.asarray(X.columns, dtype=object)
    BLOCK = compute_block_mask(pd.concat([X, Xt], axis=0).to_numpy(float))
    print(f"train {X.shape}, test {Xt.shape}, signal block: {BLOCK.sum()} features")

    params = load_params()
    if args.tune:
        p, specs = tune_all(X, y, BLOCK, NAMES)
        params = {**params, **p, "specs": specs}
        save_params(params)
        print(f"saved {ROOT / 'tuned_params.json'}")

    if args.evaluate or args.quick:
        reps = 2 if args.quick else 3
        model = build_winner(params, BLOCK, NAMES)
        s = cv_scores(model, X, y, n_splits=5, n_repeats=reps, seed=SEED)
        report(f"winner stack 5x{reps} CV", s)
        np.save(OUT / f"fold_scores_5x{reps}.npy", s)
        pd.DataFrame({"name": ["winner"], "mean_r2": [s.mean()], "std_r2": [s.std()],
                      "min_r2": [s.min()], "max_r2": [s.max()]}).to_csv(
            OUT / f"cv_5x{reps}.csv", index=False)

    if args.submit or args.quick:
        seeds = [42] if args.quick else [42, 7, 123]
        preds = []
        for sd in seeds:
            m = build_winner(params, BLOCK, NAMES, inner_seed=sd)
            m.fit(X, y)
            preds.append(m.predict(Xt))
            print(f"  fitted (inner seed {sd})", flush=True)
        preds = np.clip(np.mean(preds, axis=0), 42.0, 97.0)
        sample = pd.read_csv(ROOT / "sample.csv")
        assert list(sample.columns) == ["id", "y"] and (sample["id"].to_numpy() == ids_te).all()
        assert np.isfinite(preds).all()
        out = sample[["id"]].copy()
        out["y"] = preds
        dest = ROOT / "submission.csv"
        out.to_csv(dest, index=False)
        print(f"wrote {dest} ({len(out)} rows, range [{preds.min():.1f}, {preds.max():.1f}])")

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
