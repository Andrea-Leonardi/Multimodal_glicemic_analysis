from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "multimodal_glycemic_analysis" / "Scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import GlookoDataCleaning as glooko_cleaning
import MergeData as merge_data
import MLapplications as ml
import Processing_and_descriptive as processing
import config as cfg


def _sample_daily_frame(periods: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2025-10-01", periods=periods, freq="D")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "training": rng.integers(0, 2, size=periods).astype(float),
            "bg_mean": np.linspace(170.0, 200.0, periods),
            "bg_sd": np.linspace(25.0, 35.0, periods),
            "bg_max": np.linspace(220.0, 280.0, periods),
            "bg_tir_%": np.linspace(75.0, 55.0, periods),
            "bg_auc_above_limit_rate": np.linspace(4.0, 12.0, periods),
            "cartridge_loaded": np.zeros(periods),
            "carb_daily": np.linspace(120.0, 180.0, periods),
            "ins_tot": np.linspace(35.0, 45.0, periods),
            "ins_bolo_tot": np.linspace(15.0, 22.0, periods),
            "steps": np.linspace(5000.0, 9000.0, periods),
            "calories": np.linspace(2000.0, 2500.0, periods),
            "avg_rest_hr": np.linspace(56.0, 62.0, periods),
            "sl_tot_duration": np.linspace(380.0, 460.0, periods),
            "sl_score": np.linspace(68.0, 84.0, periods),
            "sl_bedtime_mfm": np.linspace(-60.0, 30.0, periods),
        },
        index=dates,
    )


class PipelineHelpersTest(unittest.TestCase):
    def test_normalize_daily_index(self) -> None:
        df = pd.DataFrame({"x": [1, 2]}, index=["2026-01-02 13:00", "2026-01-01 08:00"])
        normalized = merge_data.normalize_daily_index(df)

        self.assertEqual(
            list(normalized.index),
            [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
        )

    def test_compute_daily_bg_stats_exact_values(self) -> None:
        group = pd.DataFrame(
            {
                "Bg": [100.0, 250.0],
                "dt_s": [300.0, 300.0],
            }
        )
        stats = glooko_cleaning.compute_daily_bg_stats(group)

        self.assertAlmostEqual(stats["mean"], 175.0, places=6)
        self.assertAlmostEqual(stats["tir_%"], 50.0, places=6)
        self.assertAlmostEqual(stats["tar_%"], 50.0, places=6)
        # auc_above = 50 * 5 min = 250; auc_total = (100 + 250) * 5 = 1750 -> 250/1750 * 100
        self.assertAlmostEqual(stats["auc_above_limit_rate"], 100.0 * 250.0 / 1750.0, places=6)

    def test_detect_multimodal_start_trims_multiweek_wearable_gap(self) -> None:
        dates = pd.date_range("2025-05-01", periods=100, freq="D")
        steps = pd.Series(np.nan, index=dates)
        steps.iloc[0:3] = 6000.0  # early isolated days
        steps.iloc[50:] = 7500.0  # contiguous recording block after 47-day gap
        df = pd.DataFrame({"steps": steps}, index=dates)

        mm_start = processing.detect_multimodal_start(df, gap_threshold_days=14)
        self.assertEqual(mm_start, dates[50])

    def test_build_lagged_dataset_has_no_same_day_leakage(self) -> None:
        df_base = _sample_daily_frame(periods=15)
        df_perturbed = df_base.copy()
        target_date = df_base.index[-1]

        # Perturb every same-day measurement on the final day
        for col in df_perturbed.columns:
            df_perturbed.loc[target_date, col] = 99999.0

        feat_base = processing.build_lagged_dataset(df_base, lag_days=2)
        feat_perturbed = processing.build_lagged_dataset(df_perturbed, lag_days=2)

        X_base, _ = ml.prepare_training_data(feat_base, lag_days=2, feature_set="multimodal")
        X_perturbed, _ = ml.prepare_training_data(feat_perturbed, lag_days=2, feature_set="multimodal")

        pd.testing.assert_series_equal(
            X_base.loc[target_date],
            X_perturbed.loc[target_date],
            check_names=False,
        )

    def test_feature_set_ablation_excludes_xiaomi_in_glooko_only(self) -> None:
        df = _sample_daily_frame(periods=15)
        engineered = processing.build_lagged_dataset(df, lag_days=2)

        X_mm, _ = ml.prepare_training_data(engineered, lag_days=2, feature_set="multimodal")
        X_gl, _ = ml.prepare_training_data(engineered, lag_days=2, feature_set="glooko_only")

        self.assertTrue(any(ml.is_xiaomi_feature(c) for c in X_mm.columns))
        self.assertFalse(any(ml.is_xiaomi_feature(c) for c in X_gl.columns))
        self.assertGreater(X_mm.shape[1], X_gl.shape[1])

    def test_resolve_glooko_exports_returns_sorted_directories(self) -> None:
        exports = cfg.resolve_glooko_exports()

        self.assertTrue(exports)
        self.assertEqual(exports, sorted(exports))
        self.assertTrue(all(path.is_dir() for path in exports))

    def test_format_results_creates_readable_sections(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "feature_set": "multimodal",
                    "task": "regression",
                    "model": "lasso",
                    "quantile": pd.NA,
                    "metric": "r2",
                    "cv_score": 0.1234,
                    "holdout_score": 0.0512,
                    "cv_mae": 2.15,
                    "holdout_mae": 2.31,
                    "holdout_rmse": 2.95,
                    "best_params": '{"model__alpha": 0.1}',
                    "n_features": 40,
                    "n_train": 100,
                    "n_holdout": 28,
                },
                {
                    "feature_set": "multimodal",
                    "task": "classification",
                    "model": "svm_rbf",
                    "quantile": 0.65,
                    "metric": "roc_auc",
                    "cv_score": 0.6789,
                    "holdout_score": 0.7123,
                    "holdout_pr_auc": 0.654,
                    "holdout_brier": pd.NA,
                    "best_params": '{"model__C": 10.0, "model__gamma": 0.001}',
                    "threshold": 10.25,
                    "n_features": 40,
                    "n_train": 100,
                    "n_holdout": 28,
                },
            ]
        )

        formatted = ml.format_results(results)

        self.assertIn("Training Summary", formatted)
        self.assertIn("Regression", formatted)
        self.assertIn("Classification", formatted)
        self.assertIn("alpha=0.1", formatted)
        self.assertIn("cutoff=10.250", formatted)


if __name__ == "__main__":
    unittest.main()
