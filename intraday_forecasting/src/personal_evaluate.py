"""Freeze validation choices, refit before August, calibrate, then score test."""

from __future__ import annotations

import gc
import json
from pathlib import Path
try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]
import time

import joblib
import numpy as np
import pandas as pd

from src.data import ROOT, sha256
from src.forecasting import finite_sample_radius
from src.personal_data import split_positions
from src.personal_experiment import Inputs, all_validation, init_stage, metrics, utc_boundary
from src.personal_boosting import create, training_window


def family_key(spec):
    if spec["family"] == "timestral":
        return spec["model"]
    if spec.get("base"):
        return "timestral_readout" if spec.get("embedding") else "timestral_residual"
    if spec.get("features", "all") != "all":
        return "ablation_" + spec["features"]
    if spec["family"] == "neural":
        return "tcn" if spec.get("sequence") else "mlp"
    return spec["family"]


def ensemble_weights(predictions, y):
    """Small convex grid on at most three candidates; no test/calibration labels."""
    count = predictions.shape[-1]
    choices = []
    if count == 1:
        return np.ones(1)
    for first in range(11):
        for second in range(11 - first) if count == 3 else [10 - first]:
            weight = np.array([first, second, 10 - first - second] if count == 3 else [first, second]) / 10
            score = float(np.nanmean(np.abs(predictions @ weight - y)))
            choices.append((score, weight))
    return min(choices, key=lambda item: item[0])[1]


def choose(output, dataset, config):
    candidates = []
    for stage in ["tabular", "timestral", "hybrid"]:
        candidates += json.loads((output / f"{stage}-selection.json").read_text())["candidates"]
    extended = output / "timestral-extended-selection.json"
    if extended.exists():
        candidates += json.loads(extended.read_text())["candidates"]
    long = output / "timestral-long-selection.json"
    if long.exists():
        candidates += json.loads(long.read_text())["candidates"]
    dense = output / "dense-training-selection.json"
    if dense.exists():
        candidates += json.loads(dense.read_text())["candidates"]
    direct = output / "direct-boosting-selection.json"
    if direct.exists():
        candidates += json.loads(direct.read_text())["candidates"]
    # Keep the best configuration per model family, including explicit ablations.
    selected = {}
    for candidate in candidates:
        family = family_key(candidate["spec"])
        if family not in selected or np.mean(candidate["metrics"]["mae"]) < np.mean(selected[family]["metrics"]["mae"]):
            selected[family] = candidate
    selected = list(selected.values())
    positions, fold_ids = all_validation(dataset, config)
    prediction_map = {}
    for choice in selected:
        name = choice["spec"]["name"]
        with np.load(output / "oof" / (name + ".npz")) as saved:
            if not np.array_equal(saved["positions"], positions):
                raise ValueError("Candidates must have the same OOF origins")
            prediction_map[name] = saved["prediction"]
            if prediction_map[name].shape != (len(positions), len(config["horizons_minutes"])) or not np.isfinite(prediction_map[name]).all():
                raise ValueError(f"Invalid validation predictions for {name}; cannot silently change the comparison cohort")
    # Calendar baselines are references, not candidates for the personal ensemble.
    ensembles, diagnostics = [], []
    y = dataset.targets[positions]
    for k, horizon in enumerate(config["horizons_minutes"]):
        top = sorted(selected, key=lambda c: c["metrics"]["mae"][k])[:3]
        names = [c["spec"]["name"] for c in top]
        matrix = np.column_stack([prediction_map[n][:, k] for n in names])
        cross_fitted = np.full(len(y), np.nan)
        for fold in np.unique(fold_ids):
            fitted = ensemble_weights(matrix[fold_ids != fold], y[fold_ids != fold, k])
            cross_fitted[fold_ids == fold] = matrix[fold_ids == fold] @ fitted
        single_mae = float(np.nanmean(np.abs(matrix[:, 0] - y[:, k])))
        crossfit_mae = float(np.nanmean(np.abs(cross_fitted - y[:, k])))
        weights = ensemble_weights(matrix, y[:, k]) if crossfit_mae < single_mae - .05 else np.array([1., 0., 0.])
        ensembles.append({"horizon_minutes": horizon, "models": names, "weights": weights.tolist()})
        diagnostics.append({"horizon_minutes": horizon, "best_single_selection_mae": single_mae,
                            "ensemble_crossfit_diagnostic_mae": crossfit_mae})
    return {"selected": selected, "ensemble": ensembles, "ensemble_diagnostics": diagnostics,
            "policy": "Families/configurations from April/June/July only; August for intervals; Sep-Oct is a reused retrospective test. No fresh blind evaluation claim."}


def recent_meal(dataset, positions):
    return np.asarray(dataset.features[positions, dataset.feature_names.index("carbs_sum_180m")]) > 0


def fit_intervals(predictions, target, meal):
    """Absolute residual calibration by horizon and known recent-meal regime."""
    result = {}
    for level in [.8, .9]:
        radius = np.empty((2, target.shape[1]))
        counts = np.zeros_like(radius, dtype=int)
        for regime in [0, 1]:
            for k in range(target.shape[1]):
                errors = np.abs(predictions[:, k] - target[:, k])
                mask = (meal == bool(regime)) & np.isfinite(errors)
                counts[regime, k] = mask.sum()
                if mask.sum() < 100:
                    mask = np.isfinite(errors)
                radius[regime, k] = finite_sample_radius(errors[mask], level)
        result[str(level)] = {"radii": radius.tolist(), "regime_counts": counts.tolist()}
    return result


def apply_intervals(prediction, meal, calibrator, level=.9):
    radii = np.asarray(calibrator[str(level)]["radii"])[meal.astype(int)]
    return prediction - radii, prediction + radii


def make_prediction(spec, model, inputs, dataset, positions, config, output, stage):
    folder = output / "predictions"
    folder.mkdir(exist_ok=True)
    path = folder / f"{stage}_{spec['name']}.npz"
    if path.exists():
        with np.load(path) as saved:
            if not np.array_equal(saved["positions"], positions):
                raise ValueError("Prediction cache origins differ")
            return saved["prediction"]
    if spec["family"] == "timestral":
        from src.personal_timestral import Predictor, forecast_file
        predictor = Predictor(dataset, config, spec["model"])
        result = forecast_file(predictor, positions, spec, path)
        del predictor
        gc.collect()
        return result
    outputs = []
    for first in range(0, len(positions), 1024):
        batch = positions[first:first + 1024]
        x, offset = inputs.get(spec, batch)
        sequence = np.asarray(dataset.sequence[batch]) if spec.get("sequence") else None
        outputs.append(model.predict(x, offset, sequence=sequence))
    result = np.concatenate(outputs).astype(np.float32)
    np.savez_compressed(path, positions=positions, prediction=result)
    return result


def evaluate(name, predictions, actual, positions, dataset, config, calibrator):
    meal = recent_meal(dataset, positions)
    lower80, upper80 = apply_intervals(predictions, meal, calibrator, .8)
    lower90, upper90 = apply_intervals(predictions, meal, calibrator, .9)
    rows = []
    slope = np.asarray(dataset.features[positions, dataset.feature_names.index("glucose_delta_15m")]) / 15
    for k, horizon in enumerate(config["horizons_minutes"]):
        target_index = dataset.target_indices[positions, k]
        target_time = np.where(target_index >= 0, dataset.times[np.maximum(target_index, 0)], np.nan)
        rows.append(pd.DataFrame({"model": name, "origin": pd.to_datetime(dataset.times[positions], unit="s", utc=True),
            "horizon_minutes": horizon, "observed_target_time": pd.to_datetime(target_time, unit="s", utc=True),
            "actual": actual[:, k], "median": predictions[:, k], "lower80": lower80[:, k], "upper80": upper80[:, k],
            "lower90": lower90[:, k], "upper90": upper90[:, k], "recent_meal": meal,
            "rapid_change": np.abs(slope) >= 2, "origin_glucose": dataset.glucose[positions]}))
    return pd.concat(rows, ignore_index=True)


def summarize(rows, config):
    rng = np.random.default_rng(config["seed"])
    reference = rows.loc[rows.model.eq("timestral-28m_glucose_24h"), ["origin", "horizon_minutes", "median"]]
    if reference.empty:
        raise ValueError("Keep the fixed 28M/24h/glucose benchmark on identical test origins")
    summaries, cohorts = [], []
    for (name, horizon), raw in rows.groupby(["model", "horizon_minutes"], sort=False):
        data = raw.loc[raw.actual.notna()].copy()
        data["ae"] = (data["median"] - data.actual).abs()
        row = {"model": name, "horizon_minutes": horizon, "n": len(data), "mae": data.ae.mean(),
               "rmse": np.sqrt(np.mean((data["median"] - data.actual) ** 2)),
               "bias": (data["median"] - data.actual).mean(), "p90_absolute_error": data.ae.quantile(.9)}
        for level in [80, 90]:
            row[f"coverage{level}"] = data.actual.between(data[f"lower{level}"], data[f"upper{level}"]).mean()
            row[f"width{level}"] = (data[f"upper{level}"] - data[f"lower{level}"]).mean()
            alpha = 1 - level / 100
            interval_score = (data[f"upper{level}"] - data[f"lower{level}"]
                + 2 / alpha * (data[f"lower{level}"] - data.actual).clip(lower=0)
                + 2 / alpha * (data.actual - data[f"upper{level}"]).clip(lower=0))
            row[f"interval_score{level}"] = interval_score.mean()
        data["day"] = data.origin.dt.tz_convert(config["timezone"]).dt.date
        paired = data.merge(reference, on=["origin", "horizon_minutes"], suffixes=("", "_reference"))
        paired["improvement"] = (paired.median_reference - paired.actual).abs() - paired.ae
        daily = paired.groupby("day").agg(ae_sum=("ae", "sum"), gain=("improvement", "sum"), count=("ae", "count"))
        draws = rng.integers(0, len(daily), (config["bootstrap_repetitions"], len(daily)))
        denominator = daily["count"].to_numpy()[draws].sum(axis=1)
        for key, field in [("mae", "ae_sum"), ("improvement_vs_timestral28", "gain")]:
            distribution = daily[field].to_numpy()[draws].sum(axis=1) / denominator
            row[key + "_low95"], row[key + "_high95"] = np.quantile(distribution, [.025, .975])
        row["improvement_vs_timestral28"] = paired.improvement.mean()
        summaries.append(row)
        masks = {"recent_meal": raw.recent_meal, "no_recent_meal": ~raw.recent_meal,
                 "rapid_change": raw.rapid_change, "stable_change": ~raw.rapid_change,
                 "origin_75_140": raw.origin_glucose.between(75, 140),
                 "origin_140_180": raw.origin_glucose.gt(140) & raw.origin_glucose.le(180),
                 "origin_180_250": raw.origin_glucose.gt(180) & raw.origin_glucose.le(250),
                 "origin_above250": raw.origin_glucose.gt(250)}
        for cohort, mask in masks.items():
            d = raw.loc[mask & raw.actual.notna()]
            cohorts.append({"model": name, "horizon_minutes": horizon, "cohort": cohort, "n": len(d),
                "mae": (d["median"] - d.actual).abs().mean(),
                "coverage90": d.actual.between(d.lower90, d.upper90).mean(), "width90": (d.upper90 - d.lower90).mean()})
    return pd.DataFrame(summaries), pd.DataFrame(cohorts)


def finalize(dataset, config, output):
    output = Path(output)
    fingerprint = {"config": config, "dataset": dataset.metadata["fingerprint"],
                   "code": {str(p.relative_to(ROOT)): sha256(p) for p in [ROOT / "src/personal_evaluate.py", ROOT / "src/personal_experiment.py", ROOT / "src/personal_models.py", ROOT / "src/personal_network.py", ROOT / "src/personal_timestral.py", ROOT / "src/personal_timestral_extended.py", ROOT / "src/personal_timestral_long.py", ROOT / "src/personal_dense_training.py", ROOT / "src/personal_boosting.py", ROOT / "src/personal_predict.py", ROOT / "src/personal_cuda.py", ROOT / "scripts/run_cuda.py"]}}
    init_stage(output, "final", fingerprint)
    selection = choose(output, dataset, config)
    # Include the unmodified 28M/24h/glucose benchmark regardless of which
    # longer-context configuration won validation.
    baseline_spec = {"name": "timestral-28m_glucose_24h", "family": "timestral", "model": "timestral-28m", "context_hours": 24, "mode": "glucose"}
    if baseline_spec["name"] not in [c["spec"]["name"] for c in selection["selected"]]:
        selection["selected"].append({"spec": baseline_spec})
    frozen = output / "selection-frozen.json"
    if frozen.exists() and json.loads(frozen.read_text()) != selection:
        raise ValueError("Previously frozen selection differs")
    frozen.write_text(json.dumps(selection, indent=2))
    beginning = pd.Timestamp(float(dataset.times[0]), unit="s", tz="UTC")
    a, b, c = [utc_boundary(config[key], config) for key in ["calibration_start", "test_start", "test_end"]]
    train = split_positions(dataset, beginning, a, config["train_stride_readings"], training=True)
    calibration_pos = split_positions(dataset, a, b, config["evaluation_stride_readings"], training=True)
    test_pos = split_positions(dataset, b, c, config["evaluation_stride_readings"], training=True)
    inputs = Inputs(dataset, output)
    (output / "models").mkdir(exist_ok=True)
    predictions = {"calibration": {}, "test": {}}
    training_rows_per_model = {}
    for choice in selection["selected"]:
        spec = choice["spec"]
        model = None
        if spec["family"] != "timestral":
            model_train = split_positions(dataset, beginning, a, spec.get("train_stride_readings", config["train_stride_readings"]), training=True)
            model_train = training_window(dataset, model_train, spec, a.timestamp())
            training_rows_per_model[spec["name"]] = len(model_train)
            path = output / "models" / (spec["name"] + ".joblib")
            if path.exists():
                model = joblib.load(path)["model"]
            else:
                started = time.perf_counter()
                x, offset = inputs.get(spec, model_train)
                model = create(spec, config["seed"], config["threads"])
                sequence = np.asarray(dataset.sequence[model_train]) if spec.get("sequence") else None
                epochs = [f["epochs"] for f in choice.get("folds", []) if f.get("epochs")]
                fixed_epochs = int(round(np.mean(epochs))) if epochs else None
                model.fit(x, dataset.targets[model_train], offset, sequence=sequence, fixed_epochs=fixed_epochs)
                temporary = path.with_suffix(".tmp")
                joblib.dump({"model": model, "spec": spec, "feature_names": dataset.feature_names,
                             "train_end": config["calibration_start"], "training_rows": len(model_train),
                             "training_first_origin_unix_s": float(dataset.times[model_train[0]]),
                             "training_last_origin_unix_s": float(dataset.times[model_train[-1]]),
                             "training_complete_targets": int(np.isfinite(dataset.targets[model_train]).all(axis=1).sum()),
                             "training_valid_targets_per_horizon": np.isfinite(dataset.targets[model_train]).sum(axis=0).tolist(),
                             "fingerprint": fingerprint}, temporary, compress=3)
                temporary.replace(path)
                del x, sequence
                print(f"Refit {spec['name']}: {len(model_train)} esempi, {time.perf_counter() - started:.1f}s", flush=True)
        for stage, pos in [("calibration", calibration_pos), ("test", test_pos)]:
            predictions[stage][spec["name"]] = make_prediction(spec, model, inputs, dataset, pos, config, output, stage)
        del model
        gc.collect()
    for stage, pos in [("calibration", calibration_pos), ("test", test_pos)]:
        predictions[stage]["persistence"] = np.repeat(np.asarray(dataset.glucose[pos])[:, None], 4, axis=1)
        slope = np.asarray(dataset.features[pos, dataset.feature_names.index("glucose_delta_15m")]) / 15
        horizon = np.asarray(config["horizons_minutes"])
        predictions[stage]["damped_trend"] = (np.asarray(dataset.glucose[pos])[:, None]
            + slope[:, None] * 30 * (1 - np.exp(-horizon[None, :] / 30))).astype(np.float32)
        combined = np.empty((len(pos), 4), dtype=np.float32)
        for k, ensemble in enumerate(selection["ensemble"]):
            combined[:, k] = sum(weight * predictions[stage][name][:, k] for name, weight in zip(ensemble["models"], ensemble["weights"]))
        predictions[stage]["selected_personal"] = combined
        for name in ["persistence", "damped_trend", "selected_personal"]:
            np.savez_compressed(output / "predictions" / f"{stage}_{name}.npz", positions=pos, prediction=predictions[stage][name])
    # Outcomes are read only after prediction artifacts and choices are fixed.
    for stage, pos in [("calibration", calibration_pos), ("test", test_pos)]:
        for name, prediction in predictions[stage].items():
            if prediction.shape != (len(pos), len(config["horizons_minutes"])) or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid {stage} predictions for {name}; keep all models on identical origins")
    actual_cal, actual_test = dataset.targets[calibration_pos], dataset.targets[test_pos]
    results = []
    for name in predictions["test"]:
        calibrator = fit_intervals(predictions["calibration"][name], actual_cal, recent_meal(dataset, calibration_pos))
        (output / "predictions" / ("intervals_" + name + ".json")).write_text(json.dumps(calibrator, indent=2))
        results.append(evaluate(name, predictions["test"][name], actual_test, test_pos, dataset, config, calibrator))
    rows = pd.concat(results, ignore_index=True)
    rows.to_csv(output / "test-predictions.csv.gz", index=False)
    summary, cohorts = summarize(rows, config)
    summary.to_csv(output / "test-metrics.csv", index=False)
    cohorts.to_csv(output / "test-cohorts.csv", index=False)
    (output / "completed.json").write_text(json.dumps({"training_rows_default": len(train), "training_rows_per_model": training_rows_per_model, "calibration_origins": len(calibration_pos), "test_origins": len(test_pos),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource is not None else 0.0, "fingerprint": fingerprint}, indent=2))
    from src.personal_report import write_report
    write_report(output, dataset, config, selection, summary)
    print(summary[["model", "horizon_minutes", "mae", "rmse", "coverage90", "width90"]].round(3).to_string(index=False), flush=True)
