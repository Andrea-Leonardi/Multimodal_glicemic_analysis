"""Equivalence and memory-allocation regressions for the personal dataset."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import personal_data as current
import personal_dense_reference as reference


def assert_dataset_equal(left, right):
    """All model inputs/labels, with equal missingness and float32 tolerance."""
    if left.feature_names != right.feature_names:
        raise AssertionError("Feature names/order changed")
    differences = {}
    for name in ["times", "glucose", "features", "targets", "target_indices", "sequence", "eligible"]:
        a, b = getattr(left, name), getattr(right, name)
        np.testing.assert_equal(np.isnan(a), np.isnan(b), err_msg=name)
        if np.issubdtype(a.dtype, np.inexact):
            np.testing.assert_allclose(a, b, rtol=2e-7, atol=1e-6, equal_nan=True, err_msg=name)
            finite = np.isfinite(a) & np.isfinite(b)
            differences[name] = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.
        else:
            np.testing.assert_array_equal(a, b, err_msg=name)
            differences[name] = 0.
    pd.testing.assert_frame_equal(left.event_table, right.event_table)
    for name in left.streams:
        pd.testing.assert_frame_equal(left.streams[name], right.streams[name])
    return differences


def compare_sample(rows=64, block_size=17):
    read_csv = pd.read_csv
    def limited_csv(path, *args, **kwargs):
        frame = read_csv(path, *args, **kwargs)
        return frame.iloc[:rows].copy() if Path(path).name == "cleaned-CGM.csv" else frame
    with patch.object(pd, "read_csv", side_effect=limited_csv):
        dense = reference.build()
        chunked = current.build(block_size=block_size)
    return assert_dataset_equal(chunked, dense)


class PersonalMemoryTests(unittest.TestCase):
    def test_timestamp_units_are_explicit(self):
        dates = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
        microseconds = current.epoch_seconds(dates.as_unit("us"))
        nanoseconds = current.epoch_seconds(dates.as_unit("ns"))
        np.testing.assert_array_equal(microseconds, nanoseconds)
        np.testing.assert_array_equal(np.diff(microseconds), [300, 300])
        self.assertGreater(microseconds[0], 1_700_000_000)

    def test_curve_matches_original_expressions_across_internal_blocks(self):
        values = np.linspace(-10, 500, 70_001, dtype=np.float64)
        for duration, peak, delay in [(240, 55, 10), (360, 75, 10), (420, 90, 10)]:
            for original, chunked in zip(reference.insulin_curve(values, duration, peak, delay), current.insulin_curve(values, duration, peak, delay)):
                np.testing.assert_array_equal(original, chunked)
        for value in [0., 10., np.nan]:
            for original, chunked in zip(reference.insulin_curve(value), current.insulin_curve(value)):
                np.testing.assert_equal(original, chunked)

    def test_curve_normalization_and_remaining_derivative(self):
        ages = np.arange(0., 431., .05)
        remaining, activity = current.insulin_curve(ages)
        self.assertAlmostEqual(float(np.trapezoid(activity, ages)), 1., places=6)
        np.testing.assert_allclose(-np.gradient(remaining, .05)[220:7000], activity[220:7000], atol=2e-8)
        self.assertEqual(remaining[0], 1)
        self.assertEqual(remaining[-1], 0)

    def test_exposure_never_includes_future_events(self):
        times, doses = np.array([0., 300., 900.]), np.array([1., 2., 100.])
        values, ages = current.exposure(times, doses, np.array([750.]))
        self.assertEqual(values.sum(), 3)
        np.testing.assert_array_equal(ages, [[7.5, 12.5]])

    def test_exposure_rejects_giant_shape_before_dense_allocation(self):
        with self.assertRaisesRegex(MemoryError, "check time units"):
            current.exposure(np.arange(1001.), np.ones(1001), np.full(1001, 2000.))

    def test_invalid_block_sizes(self):
        for size in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                list(current.row_blocks(10, size))

    def test_entire_sample_dataset_matches_dense_reference(self):
        compare_sample(rows=64, block_size=17)

    def test_historical_inputs_match_dense_lookup(self):
        dataset = type("D", (), {})()
        dataset.times = np.arange(100.) * 300
        dataset.glucose = np.arange(100., dtype=np.float32)
        positions = np.array([0, 11, 50, 99])
        actual_values, actual_indices = current.historical_inputs(dataset, positions, 1, block_size=2)
        queries = dataset.times[positions, None] - np.arange(11, -1, -1)[None, :] * 300
        expected_indices = reference.nearest_indices(dataset.times, queries, 90, positions[:, None])
        np.testing.assert_array_equal(actual_indices, expected_indices)
        np.testing.assert_equal(actual_values, reference.sample(dataset.glucose, expected_indices))


if __name__ == "__main__":
    unittest.main()
