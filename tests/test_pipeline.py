"""Tests for the brainage package — run with:  python -m unittest discover -s tests -v

They use a subsample of the real data (fast) and cover the properties that make
the project robust: no leakage through the pipeline, shape/id safety, outlier
masking behaviour, and submission format.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainage.config import Config
from brainage.data import load_data, write_submission
from brainage.evaluate import cross_validate
from brainage.models import build_model
from brainage.preprocessing import MADOutlierMasker, make_preprocessor, selected_feature_names

ROOT = Path(__file__).resolve().parent.parent


def small_cfg():
    cfg = Config(data_dir=ROOT / "data")
    cfg.n_repeats = 1          # keep the CV smoke test fast
    cfg.gp_n_restarts = 0
    return cfg


class TestMADOutlierMasker(unittest.TestCase):
    def test_spikes_become_nan_constant_column_untouched(self):
        X = np.array([[1.0, 5.0], [1.1, 5.0], [0.9, 5.0], [1.0, 5.0], [1e23, 5.0]])
        masker = MADOutlierMasker(thresh=8.0).fit(X)
        out = masker.transform(X)
        self.assertTrue(np.isnan(out[4, 0]), "1e23 spike must be masked to NaN")
        self.assertFalse(np.isnan(out[:4]).any(), "normal entries must survive")
        self.assertFalse(np.isnan(out[:, 1]).any(), "constant column must never be flagged")

    def test_does_not_modify_input(self):
        X = np.array([[1.0], [1e23]])
        X_copy = X.copy()
        MADOutlierMasker(thresh=8.0).fit(X).transform(X)
        np.testing.assert_array_equal(X, X_copy)

    def test_inf_threshold_disables(self):
        X = np.array([[1.0], [1e23]])
        out = MADOutlierMasker(thresh=np.inf).fit(X).transform(X)
        self.assertFalse(np.isnan(out).any())


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = small_cfg()
        data = load_data(cls.cfg)
        # deterministic subsample to keep the tests fast
        rng = np.random.default_rng(0)
        idx = rng.choice(len(data.y), size=250, replace=False)
        cls.X, cls.y = data.X.iloc[idx].reset_index(drop=True), data.y[idx]
        cls.X_test = data.X_test.iloc[:50].reset_index(drop=True)

    def test_preprocessor_output_shape_and_finite(self):
        pre = make_preprocessor(self.cfg).fit(self.X, self.y)
        Xt = pre.transform(self.X_test)
        self.assertEqual(Xt.shape, (50, self.cfg.n_features))
        self.assertTrue(np.isfinite(Xt).all(), "no NaN/inf may survive imputation")

    def test_no_leakage_between_folds(self):
        # a pipeline fitted on fold A must not use fold B statistics:
        # here we simply check the imputer medians differ from full-data medians
        # when fitted on a subset (i.e. they are refit, not frozen)
        from sklearn.model_selection import KFold
        tr, va = next(KFold(2, shuffle=True, random_state=1).split(self.X))
        pre = make_preprocessor(self.cfg).fit(self.X.iloc[tr], self.y[tr])
        # impute must see the *outlier-masked* training fold, and be refit per fold:
        masked = pre.named_steps["outlier"].transform(
            pre.named_steps["var"].transform(self.X.iloc[tr]))
        expected = np.nanmedian(masked, axis=0)
        np.testing.assert_allclose(pre.named_steps["impute"].statistics_, expected,
                                   rtol=1e-6, equal_nan=True)
        # fitting on the other half must give different statistics (not frozen globals)
        pre2 = make_preprocessor(self.cfg).fit(self.X.iloc[va], self.y[va])
        self.assertFalse(np.allclose(pre.named_steps["impute"].statistics_,
                                     pre2.named_steps["impute"].statistics_))
        self.assertEqual(pre.transform(self.X.iloc[va]).shape[1], self.cfg.n_features)

    def test_selected_feature_names_count(self):
        names = selected_feature_names(self.cfg, self.X, self.y)
        self.assertEqual(len(names), self.cfg.n_features)

    def test_alternative_selection_modes(self):
        for sel, score in [("lasso", "f_regression"), ("kbest", "spearman")]:
            with self.subTest(selection=sel, score=score):
                cfg = small_cfg()
                cfg.selection = sel
                cfg.score_func = score
                pre = make_preprocessor(cfg).fit(self.X, self.y)
                out = pre.transform(self.X_test)
                self.assertEqual(out.shape, (50, cfg.n_features))
                self.assertTrue(np.isfinite(out).all())

    def test_each_model_fits_and_predicts(self):
        for name in ["ridge", "xgb", "histgb", "svr", "vote", "stack2"]:
            with self.subTest(model=name):
                m = build_model(name, self.cfg)
                m.fit(self.X, self.y)
                pred = m.predict(self.X_test)
                self.assertEqual(pred.shape, (50,))
                self.assertTrue(np.isfinite(pred).all())

    def test_cv_smoke(self):
        scores = cross_validate(build_model("ridge", self.cfg), self.X, self.y, self.cfg)
        self.assertEqual(len(scores), self.cfg.n_splits * self.cfg.n_repeats)
        self.assertTrue(np.isfinite(scores).all())

    def test_tune_member_smoke(self):
        from brainage.tuning import tune_member, apply_best_params
        cfg = small_cfg()
        cfg.n_splits = 3
        search = tune_member("svr", self.X, self.y, cfg, n_iter=2)
        self.assertTrue(np.isfinite(search.best_score_))
        apply_best_params("svr", cfg, search.best_params_)
        self.assertGreater(cfg.svr_C, 0)
        # the tuned pipeline still fits and predicts
        m = build_model("svr", cfg)
        m.fit(self.X, self.y)
        self.assertEqual(m.predict(self.X_test).shape, (50,))


class TestDataAndSubmission(unittest.TestCase):
    def test_load_data_validates(self):
        cfg = small_cfg()
        data = load_data(cfg)
        self.assertEqual(data.X.shape[1], 832)
        self.assertEqual(len(data.y), len(data.X))
        self.assertEqual(list(data.X.columns), list(data.X_test.columns))

    def test_write_submission_format(self):
        import tempfile
        cfg = small_cfg()
        data = load_data(cfg)
        preds = np.full(len(data.test_ids), data.y.mean())
        with tempfile.TemporaryDirectory() as d:
            out_path = Path(d) / "sub.csv"
            out = write_submission(out_path, data.test_ids, preds, cfg.sample_path)
            self.assertEqual(list(out.columns), ["id", "y"])
            self.assertEqual(len(out), len(pd.read_csv(cfg.sample_path)))
            # scrambled ids must be rejected
            with self.assertRaises(ValueError):
                write_submission(out_path, data.test_ids[::-1], preds, cfg.sample_path)


if __name__ == "__main__":
    unittest.main()
