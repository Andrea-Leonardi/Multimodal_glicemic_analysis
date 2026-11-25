"""Shared causal data access, provenance, bounded writes and artifact handling."""

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

from src.data import ROOT, sha256
from src.personal_data import configuration, nearest_indices, sample, split_positions
from src.personal_experiment import load_dataset, folds, all_validation, utc_boundary


def settings():
    return json.loads((ROOT / "config/personal-v3.json").read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def atomic_array(path, array):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def atomic_joblib(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def context():
    cfg = settings()
    output = ROOT / cfg["output"]
    output.mkdir(parents=True, exist_ok=True)
    return load_dataset(ROOT / cfg["dataset"]), configuration(), cfg, output


def stage(output, name, dataset, cfg, sources):
    sources = ["src/v3/common.py", "src/v3/run.py", "scripts/run_v3.py", "config/personal-v3.json", *sources]
    fingerprint = {"settings": cfg, "data": dataset.metadata["fingerprint"],
        "code": {p: sha256(ROOT / p) for p in sorted(set(sources))}}
    path = output / (name + "-manifest.json")
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != fingerprint:
            raise ValueError(f"{name} source/config differs; use a new experiment output")
    else:
        atomic_json(path, {"fingerprint": fingerprint, "created_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "catboost", "torch", "timestral"]}})
    return fingerprint


def score(actual, prediction):
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predictions must not change the evaluation cohort")
    return {"mae": np.nanmean(np.abs(error), axis=0).tolist(),
            "rmse": np.sqrt(np.nanmean(error ** 2, axis=0)).tolist(), "n": np.isfinite(error).sum(axis=0).tolist()}


def regimes(dataset, positions, cfg):
    slope = np.asarray(dataset.features[positions, dataset.feature_names.index("glucose_delta_15m")]) / 15
    meal = np.asarray(dataset.features[positions, dataset.feature_names.index("carbs_sum_180m")]) > 0
    return np.where(np.abs(slope) >= cfg["rapid_threshold_mgdl_min"], 2, meal.astype(int))


def origin_weights(dataset, positions, cfg, weighted=True):
    group = regimes(dataset, positions, cfg)
    return np.where(group == 2, cfg["rapid_weight"], np.where(group == 1, cfg["meal_weight"], 1.)).astype(np.float32) if weighted else np.ones(len(positions), np.float32)


def dense_targets(dataset, positions, base, steps=36):
    positions = np.asarray(positions)
    result = np.empty((len(positions), steps), np.float32)
    for start in range(0, len(positions), 512):
        part = positions[start:start + 512]
        indices = nearest_indices(dataset.times, dataset.times[part, None] + np.arange(1, steps + 1)[None, :] * 300,
                                  base["target_tolerance_seconds"])
        result[start:start + len(part)] = sample(dataset.glucose, indices)
    return result


def all_positions(dataset, base):
    return np.flatnonzero(dataset.eligible & (np.arange(len(dataset.times)) % base["train_stride_readings"] == 0))


def periods(dataset, base):
    return {"calibration": split_positions(dataset, utc_boundary(base["calibration_start"], base), utc_boundary(base["test_start"], base), base["evaluation_stride_readings"], training=True),
            "test": split_positions(dataset, utc_boundary(base["test_start"], base), utc_boundary(base["test_end"], base), base["evaluation_stride_readings"], training=True)}


def training_positions(dataset, base, cutoff, stride=None):
    result = split_positions(dataset, pd.Timestamp(float(dataset.times[0]), unit="s", tz="UTC"),
                             pd.Timestamp(cutoff, unit="s", tz="UTC"), stride or base["train_stride_readings"], training=True)
    if len(result) and not (dataset.times[result] + 180 * 60 + base["target_tolerance_seconds"] < cutoff).all():
        raise ValueError("Training target was not yet available at model cutoff")
    return result


def save_forecast(path, positions, prediction, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite forecast")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, positions=positions, prediction=np.asarray(prediction, np.float32), **extra)
    temporary.replace(path)


def load_forecast(path, positions):
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["positions"], positions)
        result = saved["prediction"].copy()
    if not np.isfinite(result).all():
        raise ValueError(f"Invalid saved forecast: {path}")
    return result


class LongInputs:
    def __init__(self, dataset, output):
        self.dataset = dataset
        folder = output / "long-cache"
        if not (folder / "complete.json").exists():
            raise ValueError("Complete the v3 cache stage first")
        self.positions = np.load(folder / "positions.npy", mmap_mode="r")
        self.prediction = np.load(folder / "prediction.npy", mmap_mode="r")
        self.quantiles = np.load(folder / "quantiles.npy", mmap_mode="r")

    def get(self, positions, residual=False):
        x = np.asarray(self.dataset.features[positions], np.float32)
        current = np.asarray(self.dataset.glucose[positions])
        if not residual:
            return x, current
        index = np.searchsorted(self.positions, positions)
        if (index >= len(self.positions)).any() or not np.array_equal(self.positions[index], positions):
            raise ValueError("Long-context cache misses a requested origin")
        offset = np.asarray(self.prediction[index])
        q = np.asarray(self.quantiles[index])
        extra = [offset - current[:, None], q[:, :, 18] - q[:, :, 2]]
        return np.column_stack([x, *extra]).astype(np.float32), offset

    def native_width(self, positions):
        index = np.searchsorted(self.positions, positions)
        if (index >= len(self.positions)).any() or not np.array_equal(self.positions[index], positions):
            raise ValueError("Native-width cache misses an origin")
        q = np.asarray(self.quantiles[index])
        return q[:, :, 18] - q[:, :, 2]
