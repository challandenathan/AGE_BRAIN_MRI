#!/usr/bin/env python3
"""AML 2026 — Task 1: predict brain age from (perturbed) MRI features.

Entry point of the project. Reproduces the submission end-to-end:

    python solution.py               # CV-evaluate the final model, fit on all
                                     # training data, write submission.csv
    python solution.py --no-submit   # CV evaluation only
    python solution.py --compare     # benchmark all candidate models, write
                                     # experiments/cv_results.csv
    python solution.py --model stack --n-features 110 --repeats 2

Pipeline (all steps fitted inside every CV fold, so evaluation is leak-free):
    VarianceThreshold -> MADOutlierMasker -> SimpleImputer(median)
    -> QuantileTransformer(normal) -> SelectKBest(f_regression, k)
    -> regressor (SVR-RBF by default, or the Ridge+XGB+GP stack)

Metric: R^2 via RepeatedKFold(5 x 3, seed 42). Predictions on X_test are
clipped to the observed age range and written in the sample.csv format.
"""
import argparse

import numpy as np
import pandas as pd

from brainage.config import Config
from brainage.data import load_data, write_submission
from brainage.evaluate import compare_models, cross_validate, residual_correlation
from brainage.models import MODELS, build_model
from brainage.preprocessing import selected_feature_names


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", action="store_true",
                   help="benchmark all models in brainage.models.MODELS instead of submitting")
    p.add_argument("--no-submit", action="store_true", help="only run CV, do not write submission.csv")
    p.add_argument("--model", default=None, choices=sorted(MODELS),
                   help="regressor to use (default: cfg.model)")
    p.add_argument("--n-features", type=int, default=None, help="override SelectKBest k")
    p.add_argument("--outlier-thresh", type=float, default=None,
                   help="robust-sigma threshold of the outlier masker (inf disables)")
    p.add_argument("--repeats", type=int, default=None, help="override CV repeats")
    p.add_argument("--seed", type=int, default=None, help="override the seed")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.model is not None:
        cfg.model = args.model
    if args.n_features is not None:
        cfg.n_features = args.n_features
    if args.outlier_thresh is not None:
        cfg.outlier_thresh = args.outlier_thresh
    if args.repeats is not None:
        cfg.n_repeats = args.repeats
    if args.seed is not None:
        cfg.seed = args.seed

    data = load_data(cfg)
    print(f"X_train {data.X.shape}, y {data.y.shape} "
          f"(range {data.y.min():.0f}-{data.y.max():.0f}), X_test {data.X_test.shape}")

    if args.compare:
        cfg.results_dir.mkdir(exist_ok=True)
        models = {name: factory(cfg) for name, factory in MODELS.items()}
        print(f"benchmarking {len(models)} models, "
              f"RepeatedKFold({cfg.n_splits} x {cfg.n_repeats}, seed {cfg.seed}) ...")
        table = compare_models(models, data.X, data.y, cfg)
        table.to_csv(cfg.results_path, index=False)
        print(f"\n{table.round(4).to_string(index=False)}")
        print(f"residual correlation of OOF errors:\n"
              f"{residual_correlation(models, data.X, data.y, cfg).round(3).to_string()}")
        print(f"\nwrote {cfg.results_path}")
        return

    model = build_model(cfg.model, cfg)
    print(f"model: {cfg.model} | k={cfg.n_features} | outlier_thresh={cfg.outlier_thresh} | "
          f"CV: RepeatedKFold({cfg.n_splits} x {cfg.n_repeats}, seed {cfg.seed})")
    scores = cross_validate(model, data.X, data.y, cfg)
    print(f"CV R^2 per fold : {np.round(scores, 4)}")
    print(f"CV R^2 = {scores.mean():.4f} +/- {scores.std():.4f}")
    cfg.results_dir.mkdir(exist_ok=True)
    pd.DataFrame({"fold": np.arange(len(scores)), "r2": scores}).to_csv(
        cfg.results_dir / "cv_scores_final.csv", index=False)

    names = selected_feature_names(cfg, data.X, data.y)
    pd.Series(names).to_csv(cfg.selected_features_path, index=False)
    print(f"selected {len(names)} features -> {cfg.selected_features_path}")

    if args.no_submit:
        return

    model.fit(data.X, data.y)
    pred = np.clip(model.predict(data.X_test), data.y.min(), data.y.max())
    write_submission(cfg.submission_path, data.test_ids, pred, cfg.sample_path)
    print(f"wrote {cfg.submission_path} ({len(pred)} rows), "
          f"pred range [{pred.min():.2f}, {pred.max():.2f}]")


if __name__ == "__main__":
    main()
