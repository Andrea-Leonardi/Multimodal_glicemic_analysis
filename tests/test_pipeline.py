from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "multimodal_glycemic_analysis" / "Scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import MergeData as merge_data
import Processing_and_descriptive as processing
import config as cfg


class PipelineHelpersTest(unittest.TestCase):
    def test_normalize_daily_index(self) -> None:
        df = pd.DataFrame({"x": [1, 2]}, index=["2026-01-02 13:00", "2026-01-01 08:00"])
        normalized = merge_data.normalize_daily_index(df)

        self.assertEqual(list(normalized.index), [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")])

    def test_build_lagged_dataset(self) -> None:
        df = pd.DataFrame(
            {
                "training": [0, 1, 0],
                "bg_mean": [100.0, 110.0, 120.0],
                "bg_sd": [10.0, 11.0, 12.0],
                "bg_max": [140.0, 150.0, 160.0],
                "bg_auc_above_limit_rate": [5.0, 6.0, 7.0],
                "cartridge_loaded": [0, 1, 0],
                "carb_daily": [20.0, 30.0, 40.0],
                "sl_tot_duration": [400.0, 410.0, 420.0],
                "sl_score": [70.0, 71.0, 72.0],
                "sl_bedtime_mfm": [-30.0, -20.0, -10.0],
                "steps": [5000.0, 6000.0, 7000.0],
            },
            index=pd.date_range("2026-01-01", periods=3, freq="D"),
        )

        lagged = processing.build_lagged_dataset(df, lag_days=2)

        self.assertIn("bg_mean_lag1", lagged.columns)
        self.assertIn("bg_mean_lag2", lagged.columns)
        self.assertEqual(lagged.loc[pd.Timestamp("2026-01-03"), "bg_mean_lag1"], 110.0)
        self.assertEqual(lagged.loc[pd.Timestamp("2026-01-03"), "bg_mean_lag2"], 100.0)

    def test_resolve_glooko_exports_returns_sorted_directories(self) -> None:
        exports = cfg.resolve_glooko_exports()

        self.assertTrue(exports)
        self.assertEqual(exports, sorted(exports))
        self.assertTrue(all(path.is_dir() for path in exports))


if __name__ == "__main__":
    unittest.main()
