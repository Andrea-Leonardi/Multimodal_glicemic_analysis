"""Checks for chronological supervision, frozen inputs and calibration."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from src.personal_data import configuration
from src.personal_experiment import folds, metrics
from src.personal_models import TabularModel, feature_columns
from src.personal_evaluate import ensemble_weights, fit_intervals, apply_intervals
from src.personal_timestral import Predictor, PHYSICAL, calendar_covariates
from src.personal_boosting import DirectMAEModel, training_window


class PersonalTrainingTests(unittest.TestCase):
    def test_recent_training_window_uses_the_boundary_not_validation_outcomes(self):
        dataset = SimpleNamespace(times=np.arange(200.) * 86400)
        positions = np.arange(150)
        recent = training_window(dataset, positions, {"train_window_days": 90}, 150 * 86400)
        np.testing.assert_array_equal(recent, np.arange(60, 150))
        np.testing.assert_array_equal(training_window(dataset, positions, {}, 150 * 86400), positions)

    def test_direct_mae_fits_separate_available_labels_and_respects_offset(self):
        import importlib.util
        if importlib.util.find_spec("catboost") is None:
            self.skipTest("Install catboost to run CatBoost estimator test")
        x = np.arange(40., dtype=np.float32)[:, None]
        glucose = np.full(40, 150.)
        y = glucose[:, None] + x * np.array([[.1, .2, .3, .4]])
        y[np.arange(40), np.arange(40) % 4] = np.nan
        self.assertFalse(np.isfinite(y).all(axis=1).any())
        model = DirectMAEModel({"depth": 2, "iterations": 5, "l2": 15}, threads=1)
        model.fit(x, y, glucose)
        prediction = model.predict(x[:3], glucose[:3])
        self.assertEqual(prediction.shape, (3, 4))
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_allclose(model.predict(x[:3], glucose[:3] + 50), prediction + 50)

    def test_expanding_folds_purge_all_training_targets(self):
        times = pd.date_range("2025-11-28", "2026-08-01", freq="1h", tz="UTC").as_unit("ns").asi8 / 1e9
        dataset = SimpleNamespace(times=times, eligible=np.ones(len(times), dtype=bool))
        config = configuration()
        previous_size = 0
        for start, train, valid in folds(dataset, config):
            boundary = pd.Timestamp(start, tz="Europe/Rome").timestamp()
            self.assertTrue((times[train] + 180 * 60 + 90 < boundary).all())
            self.assertTrue((times[valid] >= boundary).all())
            self.assertGreater(len(train), previous_size)
            self.assertEqual(len(np.intersect1d(train, valid)), 0)
            previous_size = len(train)

    def test_feature_ablations_do_not_leak_physical_inputs_into_glucose(self):
        names = ["glucose", "clock_sin_24", "iob_bolus_d360_p75", "iob_bolus_d300_p75", "carbs_sum_30m", "carb_remaining_proxy_60"]
        self.assertEqual(feature_columns(names, "glucose").tolist(), [0, 1])
        self.assertEqual(feature_columns(names, "raw").tolist(), [0, 1, 4])
        self.assertNotIn(3, feature_columns(names, "humalog360"))

    def test_dense_training_keeps_the_same_validation_origins(self):
        times = pd.date_range("2025-11-28", "2026-08-01", freq="1h", tz="UTC").as_unit("ns").asi8 / 1e9
        dataset = SimpleNamespace(times=times, eligible=np.ones(len(times), dtype=bool))
        config = configuration()
        for (_, sparse, a), (_, dense, b) in zip(folds(dataset, config), folds(dataset, config | {"train_stride_readings": 1})):
            np.testing.assert_array_equal(a, b)
            self.assertGreater(len(dense), len(sparse))

    def test_supervised_residual_offset_is_multi_horizon(self):
        x = np.arange(40., dtype=np.float32)[:, None] / 40
        offset = np.full((40, 4), [100., 130., 160., 190.])
        target = offset + 3 + 5 * x
        model = TabularModel({"family": "ridge", "alpha": 1e-8})
        model.fit(x, target, offset)
        np.testing.assert_allclose(model.predict(x, offset), target, atol=1e-4)

    def test_fitted_transforms_are_not_changed_by_prediction_rows(self):
        x = np.arange(40., dtype=np.float32)[:, None]
        glucose = np.full(40, 150.)
        y = np.repeat((150 + x), 4, axis=1)
        model = TabularModel({"family": "ridge", "alpha": 100.})
        model.fit(x, y, glucose)
        baseline = model.predict(x[:2], glucose[:2])
        model.predict(np.full((20, 1), 1e8), np.full(20, 150.))
        np.testing.assert_array_equal(baseline, model.predict(x[:2], glucose[:2]))

    def test_convex_ensemble_selects_exact_candidate_without_unbounded_weights(self):
        y = np.arange(100.)
        predictions = np.column_stack([y + 10, y, y - 20])
        weights = ensemble_weights(predictions, y)
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertTrue((weights >= 0).all())
        np.testing.assert_allclose(predictions @ weights, y, atol=1e-12)

    def test_conditional_calibration_preserves_point_predictions(self):
        prediction = np.full((400, 4), 150.)
        meal = np.r_[np.zeros(200, dtype=bool), np.ones(200, dtype=bool)]
        actual = prediction + np.where(meal[:, None], 30., 5.)
        model = fit_intervals(prediction, actual, meal)
        lo, hi = apply_intervals(prediction, meal, model)
        np.testing.assert_array_equal(hi[:200], np.full((200, 4), 155.))
        np.testing.assert_array_equal(hi[200:], np.full((200, 4), 180.))
        self.assertTrue((prediction == 150).all())

    def test_missing_targets_are_excluded_per_horizon(self):
        y = np.array([[100., np.nan], [100., 120.]])
        p = np.array([[110., 999.], [90., 130.]])
        result = metrics(y, p)
        self.assertEqual(result["n"], [2, 1])
        self.assertEqual(result["mae"], [10., 10.])

    def test_calendar_shape_and_current_local_clock(self):
        origin = pd.Timestamp("2026-07-01T12:00:00+02:00").timestamp()
        c = calendar_covariates(np.array([origin]), 12, 36, "Europe/Rome")
        self.assertEqual(c.shape, (1, 2, 48))
        self.assertAlmostEqual(float(c[0, 0, 11]), 0., places=6)
        self.assertAlmostEqual(float(c[0, 1, 11]), -1., places=6)

    def test_timestral_inputs_unchanged_when_future_observations_change(self):
        class Wrapped:
            def __init__(self, array): self.array = array
            def __getitem__(self, index): return Wrapped(self.array[index])
            def cpu(self): return self
            def numpy(self): return self.array
        captured = []
        def forecast(x, **kwargs):
            captured.append((x.copy(), {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}))
            return SimpleNamespace(quantiles=Wrapped(np.ones((len(x), 1, 36, 21))))
        dataset = SimpleNamespace(times=np.arange(100.) * 300, glucose=np.arange(100., dtype=np.float32),
                                  features=np.ones((100, 4), dtype=np.float32), feature_names=PHYSICAL)
        predictor = object.__new__(Predictor)
        predictor.dataset, predictor.config = dataset, configuration()
        predictor.adapter = SimpleNamespace(forecaster=SimpleNamespace(forecast=forecast))
        spec = {"mode": "physical", "context_hours": 1}
        predictor.predict(np.array([75]), spec)
        dataset.glucose[76:] = 999
        dataset.features[76:] = 999
        predictor.predict(np.array([75]), spec)
        np.testing.assert_array_equal(captured[0][0], captured[1][0])
        for key in ["past_covariates", "known_covariates"]:
            np.testing.assert_array_equal(captured[0][1][key], captured[1][1][key])


if __name__ == "__main__":
    unittest.main()
