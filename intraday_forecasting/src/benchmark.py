"""Chronological selection, held-out calibration and immutable test forecasts.

Run: python -m src.benchmark [--baselines-only] [--output outputs/benchmark]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ROOT, boundary, prepare, read_config, sha256
from src.forecasting import FEATURES, QUANTILES, TimestralAdapter, baseline, calibrate, intervals


def origins(grid, start, end, config):
    """Common eligible origins for ALL candidates; no filtering on outcomes.

    Full horizon stays inside its split. Missing future targets are excluded
    separately at scoring time, never interpolated.
    """
    freq = config["frequency_minutes"]
    if freq != 5:
        raise ValueError("This first benchmark uses 5-minute steps")
    h = max(config["horizons_minutes"]) // freq
    n = max(config["context_hours"]) * 60 // freq
    x = grid.glucose.to_numpy(float)
    coverage = pd.Series(np.isfinite(x)).rolling(n, min_periods=n).mean().to_numpy()
    recent_ok = pd.Series(np.isfinite(x)).rolling(4, min_periods=4).sum().eq(4).to_numpy()
    idx = np.arange(len(grid))
    eligible = ((grid.index >= start) & (grid.index + pd.Timedelta(minutes=h * freq) < end)
                & (idx >= n - 1) & (idx + h < len(grid)) & recent_ok
                & (coverage >= 1 - config["max_missing_context_fraction"]))
    stride = config["origin_stride_minutes"] // freq
    if stride < 1 or config["origin_stride_minutes"] % freq:
        raise ValueError("Origin stride must be a positive multiple of frequency")
    return idx[eligible & (idx % stride == 0)]


def historical_context(grid, positions, length, config, mode):
    indices = positions[:, None] - np.arange(length - 1, -1, -1)
    x = grid.glucose.to_numpy(float)[indices].copy()
    # Measured arrival timestamps do not exist in exports: explicitly simulated
    # latency. Reconstruct the latest causal grid and omit not-yet-arrived cells.
    observed = pd.to_datetime(grid.observed_at, utc=True).to_numpy(dtype="datetime64[ns]")
    arrival = observed[indices] + np.timedelta64(int(config["cgm_latency_minutes"] * 60), "s")
    now = grid.index.to_numpy(dtype="datetime64[ns]")[positions, None]
    x[arrival > now] = np.nan
    names = FEATURES[mode]
    cov = None
    if names:
        cov = grid[names].to_numpy(float)[indices].transpose(0, 2, 1)
    return x, cov


def forecast_set(grid, positions, name, context_hours, mode, config, adapter, path):
    """Persist model output BEFORE obtaining any corresponding target array."""
    if path.exists():
        with np.load(path) as existing:
            if not np.array_equal(existing["positions"], positions):
                raise ValueError("Cached forecast origins differ")
            return existing["quantiles"]
    length = context_hours * 12
    h = max(config["horizons_minutes"]) // 5
    batches = []
    started = time.perf_counter()
    for offset in range(0, len(positions), config["batch_size"]):
        batch = positions[offset:offset + config["batch_size"]]
        x, cov = historical_context(grid, batch, length, config, mode)
        if name.startswith("timestral"):
            predicted = adapter.predict(x, h, cov)
        else:
            if config["cgm_latency_minutes"]:
                raise ValueError("Baseline latency>0 requires a timestamp-aware adapter; no silent forward fill")
            predicted = baseline(x, h, name)
        batches.append(predicted.astype(np.float32))
        if (offset // config["batch_size"] + 1) % 10 == 0:
            print(f"  {path.stem}: {min(offset + len(batch), len(positions))}/{len(positions)}", flush=True)
    if not batches:
        raise ValueError("No eligible origins")
    predictions = np.concatenate(batches)
    np.savez_compressed(path, quantiles=predictions, positions=positions,
                        origins_utc=grid.index.asi8[positions], levels=QUANTILES)
    elapsed = time.perf_counter() - started
    path.with_suffix(".json").write_text(json.dumps({"model": name, "mode": mode, "context_hours": context_hours,
        "origins": len(positions), "seconds": elapsed, "seconds_per_origin": elapsed / len(positions)}, indent=2))
    print(f"{path.stem}: {len(positions)} previsioni, {elapsed:.1f}s", flush=True)
    return predictions


def targets(grid, positions, steps):
    return grid.glucose.to_numpy(float)[positions[:, None] + np.arange(1, steps + 1)]


def make_rows(grid, positions, prediction, target, calibration, name, mode, context_hours, config):
    calibrated = intervals(prediction, calibration)
    raw = intervals(prediction)
    rows = []
    for horizon in config["horizons_minutes"]:
        k = horizon // 5 - 1
        origin_glucose = grid.glucose.iloc[positions].to_numpy()
        delta15 = (origin_glucose - grid.glucose.iloc[positions - 3].to_numpy()) / 15
        # This cohort uses known past meal reports, not future meals.
        recent_meal = grid.meal_carbs_g.rolling(36, min_periods=1).sum().iloc[positions].gt(0).to_numpy()
        d = pd.DataFrame({
            "origin": grid.index[positions], "target_time": grid.index[positions + k + 1],
            "observed_target_time": grid.observed_at.iloc[positions + k + 1].to_numpy(),
            "model": name, "mode": mode, "context_hours": context_hours,
            "horizon_minutes": horizon, "actual": target[:, k], "median": prediction[:, k, 10],
            "origin_glucose": origin_glucose, "slope_15min": delta15, "recent_meal": recent_meal,
            "rapid_change": np.abs(delta15) >= 2,
        })
        for coverage in [.8, .9]:
            suffix = str(int(coverage * 100))
            d["lower" + suffix] = calibrated[coverage][0][:, k]
            d["upper" + suffix] = calibrated[coverage][1][:, k]
            d["raw_lower" + suffix] = raw[coverage][0][:, k]
            d["raw_upper" + suffix] = raw[coverage][1][:, k]
        error = target[:, k, None] - prediction[:, k]
        d["pinball_raw"] = np.maximum(QUANTILES * error, (QUANTILES - 1) * error).mean(axis=1)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def statistics(d):
    d = d.loc[d.actual.notna()]
    if d.empty:
        return {"n": 0}
    err = d["median"] - d.actual
    result = {"n": len(d), "mae": float(err.abs().mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
              "bias": float(err.mean())}
    for level in [80, 90]:
        for prefix in ["", "raw_"]:
            if prefix and not str(d.model.iloc[0]).startswith("timestral"):
                continue
            lo, hi = d[prefix + f"lower{level}"], d[prefix + f"upper{level}"]
            result[f"{prefix}coverage{level}"] = float(d.actual.between(lo, hi).mean())
            result[f"{prefix}width{level}"] = float((hi - lo).mean())
    if str(d.model.iloc[0]).startswith("timestral"):
        result["pinball_raw"] = float(d.pinball_raw.mean())
    return result


def summarize(rows, config, output):
    keys = ["model", "mode", "horizon_minutes"]
    summaries, cohorts = [], []
    rng = np.random.default_rng(config["seed"])
    persistence = rows.loc[rows.model.eq("persistence"), ["origin", "horizon_minutes", "actual", "median"]]
    for key, d in rows.groupby(keys, sort=False):
        header = dict(zip(keys, key))
        metrics = statistics(d)
        valid = d.loc[d.actual.notna()].copy()
        valid["day"] = pd.to_datetime(valid.origin, utc=True).dt.tz_convert(config["timezone"]).dt.date
        valid["ae"] = (valid["median"] - valid.actual).abs()
        daily = valid.groupby("day").ae.agg(["sum", "count"])
        draws = rng.integers(0, len(daily), size=(config["bootstrap_repetitions"], len(daily)))
        bootstrap = daily["sum"].to_numpy()[draws].sum(axis=1) / daily["count"].to_numpy()[draws].sum(axis=1)
        metrics["mae_daily_bootstrap_low95"], metrics["mae_daily_bootstrap_high95"] = np.quantile(bootstrap, [.025, .975])
        paired = valid.merge(persistence, on=["origin", "horizon_minutes"], suffixes=("", "_baseline"))
        paired["improvement"] = (paired.median_baseline - paired.actual).abs() - paired.ae
        improvements = paired.groupby("day").improvement.agg(["sum", "count"])
        draws = rng.integers(0, len(improvements), size=(config["bootstrap_repetitions"], len(improvements)))
        distribution = improvements["sum"].to_numpy()[draws].sum(axis=1) / improvements["count"].to_numpy()[draws].sum(axis=1)
        metrics["mae_improvement_vs_persistence"] = float(paired.improvement.mean())
        metrics["improvement_daily_low95"], metrics["improvement_daily_high95"] = np.quantile(distribution, [.025, .975])
        summaries.append(header | metrics)
        masks = {
            "recent_meal": d.recent_meal, "no_recent_meal": ~d.recent_meal,
            "rapid_change": d.rapid_change, "stable_change": ~d.rapid_change,
            "origin_below_70": d.origin_glucose.lt(70),
            "origin_70_99": d.origin_glucose.between(70, 99),
            "origin_100_180": d.origin_glucose.between(100, 180),
            "origin_above_180": d.origin_glucose.gt(180),
            "target_below_70": d.actual.lt(70),
        }
        for cohort, mask in masks.items():
            cohorts.append(header | {"cohort": cohort} | statistics(d.loc[mask]))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(cohorts).to_csv(output / "cohorts.csv", index=False)
    return summary


def run(config_path=None, output=None, baselines_only=False):
    config = read_config(config_path)
    if baselines_only:
        config["models"] = ["persistence", "damped_trend"]
    if config["cgm_latency_minutes"] != 0:
        raise ValueError("This benchmark currently supports zero CGM delay only; real arrival delays remain to measure")
    output = Path(output or ROOT / "outputs/benchmark")
    output.mkdir(parents=True, exist_ok=True)
    grid, _, audit = prepare(config)
    fingerprint = {"config": config, "source_sha256": audit["source_sha256"],
                   "code_sha256": {p.name: sha256(p) for p in [ROOT / "src/data.py", ROOT / "src/forecasting.py", ROOT / "src/benchmark.py"]}}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Output contains a different experiment. Choose a new --output directory.")
    versions = {}
    for package in ["numpy", "pandas", "torch", "timestral", "safetensors", "huggingface_hub"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "python": platform.python_version(), "versions": versions}, indent=2))
    tz = config["timezone"]
    split_bounds = [boundary(config[key], tz) for key in ["validation_start", "calibration_start", "test_start", "test_end"]]
    for left, right in zip(split_bounds, split_bounds[1:]):
        if left >= right:
            raise ValueError("Split boundaries must be chronological")
    split_positions = {name: origins(grid, a, b, config | ({"origin_stride_minutes": config["validation_origin_stride_minutes"]} if name == "validation" else {}))
                       for name, a, b in zip(["validation", "calibration", "test"], split_bounds, split_bounds[1:])}
    for split, pos in split_positions.items():
        print(f"{split}: {len(pos)} origini ammissibili", flush=True)
    validation_positions = split_positions["validation"]
    selected, selection_rows, adapters = [], [], {}
    steps = max(config["horizons_minutes"]) // 5
    # Select context only on July. No test outcomes are accessed in this phase.
    for name in config["models"]:
        probabilistic = name.startswith("timestral")
        adapter = None
        if probabilistic:
            adapter = TimestralAdapter(name, config["model_revisions"][name], ROOT / ".cache/huggingface", config["torch_threads"])
            adapters[name] = adapter
        for mode in config["covariate_modes"] if probabilistic else ["glucose"]:
            choices = []
            for hours in config["context_hours"] if probabilistic else [min(config["context_hours"])]:
                path = output / f"validation_{name}_{mode}_{hours}h.npz"
                pred = forecast_set(grid, validation_positions, name, hours, mode, config, adapter, path)
                y = targets(grid, validation_positions, steps)
                score = float(np.nanmean(np.abs(pred[:, np.array(config["horizons_minutes"]) // 5 - 1, 10] - y[:, np.array(config["horizons_minutes"]) // 5 - 1])))
                candidate = {"model": name, "mode": mode, "context_hours": hours, "validation_mae": score}
                choices.append(candidate)
                selection_rows.append(candidate)
            selected.append(min(choices, key=lambda c: c["validation_mae"]))
    (output / "selection.json").write_text(json.dumps({"candidates": selection_rows, "selected": selected}, indent=2))
    all_rows = []
    calibration_rows = []
    for choice in selected:
        name, mode, hours = [choice[k] for k in ["model", "mode", "context_hours"]]
        adapter = adapters.get(name)
        predictions = {}
        for split in ["calibration", "test"]:
            predictions[split] = forecast_set(grid, split_positions[split], name, hours, mode, config, adapter,
                                              output / f"{split}_{name}_{mode}_{hours}h.npz")
        y_cal = targets(grid, split_positions["calibration"], steps)
        calibration = calibrate(predictions["calibration"], y_cal, name.startswith("timestral"))
        (output / f"calibration_{name}_{mode}.json").write_text(json.dumps(calibration, indent=2))
        # Calibration diagnostics are in-sample and labelled accordingly.
        calibration_rows.append(make_rows(grid, split_positions["calibration"], predictions["calibration"], y_cal,
                                           calibration, name, mode, hours, config))
        y_test = targets(grid, split_positions["test"], steps)
        all_rows.append(make_rows(grid, split_positions["test"], predictions["test"], y_test,
                                  calibration, name, mode, hours, config))
    rows = pd.concat(all_rows, ignore_index=True)
    rows.to_csv(output / "test_predictions.csv.gz", index=False)
    pd.concat(calibration_rows).to_csv(output / "calibration_diagnostics_in_sample.csv.gz", index=False)
    summary = summarize(rows, config, output)
    print(summary[["model", "mode", "horizon_minutes", "n", "mae", "rmse", "coverage80", "coverage90"]].to_string(index=False))
    from src.report import write_report
    write_report(output, summary, audit, config)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.baselines_only)
