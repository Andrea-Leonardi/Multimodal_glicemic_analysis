"""Adaptive pointwise intervals, selected on chronological development periods.

Rolling calibration only reads errors whose latest possible observation time is
strictly before the update cutoff. It never changes the point prediction.
"""

import json

import joblib
import numpy as np

from src.data import ROOT, sha256
from src.v3.common import (LongInputs, all_validation, atomic_joblib, atomic_json,
    load_forecast, regimes, stage, utc_boundary)


SCALE_FEATURES = ["glucose", "glucose_delta_5m", "glucose_delta_15m", "glucose_acceleration",
    "glucose_std_30m", "glucose_std_120m", "carbs_sum_60m", "carbs_sum_180m", "carbs_minutes_since",
    "bolus_sum_60m", "iob_bolus_d360_p75", "activity_bolus_d360_p75", "activity_basal_d360_p75",
    "clock_sin_24", "clock_cos_24", "basal_unknown_6h"]


def scale_inputs(dataset, positions, prediction, inputs):
    columns = [dataset.feature_names.index(name) for name in SCALE_FEATURES]
    return np.column_stack([dataset.features[positions][:, columns],
        prediction - dataset.glucose[positions, None], inputs.native_width(positions)]).astype(np.float32)


class ErrorScale:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, x, prediction, actual):
        from catboost import CatBoostRegressor
        self.models = []
        for k in range(actual.shape[1]):
            valid = np.isfinite(actual[:, k])
            y = np.log1p(np.abs(prediction[valid, k] - actual[valid, k]))
            model = CatBoostRegressor(iterations=self.cfg["intervals"]["scale_iterations"], depth=self.cfg["intervals"]["scale_depth"],
                learning_rate=.05, loss_function="Quantile:alpha=0.8", l2_leaf_reg=10,
                random_seed=self.cfg["seed"], thread_count=self.cfg["threads"], verbose=False, allow_writing_files=False)
            model.fit(x[valid], y)
            self.models.append(model)
        return self

    def predict(self, x):
        result = np.column_stack([np.expm1(model.predict(x)) for model in self.models])
        return np.clip(result, self.cfg["intervals"]["minimum_scale_mgdl"], 250.)


def finite_quantile(values, level):
    values = np.sort(np.asarray(values)[np.isfinite(values)])
    if len(values) < 10:
        raise ValueError("Insufficient causally available calibration errors")
    rank = min(len(values), int(np.ceil((len(values) + 1) * level)))
    return float(values[rank - 1])


def conformal_factors(prediction, actual, scale, group, level, minimum):
    factors = np.empty((3, actual.shape[1]), np.float64)
    counts = np.zeros_like(factors, int)
    for k in range(actual.shape[1]):
        ratio = np.abs(prediction[:, k] - actual[:, k]) / scale[:, k]
        for regime in range(3):
            mask = (group == regime) & np.isfinite(ratio)
            counts[regime, k] = mask.sum()
            if mask.sum() < minimum:
                mask = np.isfinite(ratio)
            factors[regime, k] = finite_quantile(ratio[mask], level)
    return factors, counts


def available_history(times, cutoff, horizon_minutes, tolerance, window_days=28, minimum=60):
    times = np.asarray(times)
    available = times + horizon_minutes * 60 + tolerance < cutoff
    recent = available & (times >= cutoff - window_days * 86400)
    return recent if recent.sum() >= minimum else available


def interval_score(actual, lower, upper, level=.9):
    alpha = 1 - level
    return upper - lower + 2 / alpha * np.maximum(lower - actual, 0) + 2 / alpha * np.maximum(actual - upper, 0)


def bands(mode, prediction, scale, group, calibration, cfg, base, times=None, ledger=None, anchor=None):
    """Return bands and the latest label-availability upper bound used per row."""
    result = {}
    last_used = np.full((len(prediction), 4), np.nan)
    for level in cfg["intervals"]["levels"]:
        if mode != "adaptive_rolling":
            factors, counts = conformal_factors(calibration["prediction"], calibration["actual"], calibration["scale"],
                calibration["group"], level, cfg["intervals"]["minimum_calibration_rows"])
            radius = scale * factors[group]
        else:
            if times is None or ledger is None or anchor is None:
                raise ValueError("Rolling intervals require times and an observed-error ledger")
            cutoffs = anchor + np.floor((times - anchor) / (cfg["weekly"]["update_days"] * 86400)) * cfg["weekly"]["update_days"] * 86400
            radius = np.empty_like(prediction, np.float64)
            for cutoff in np.unique(cutoffs):
                rows = cutoffs == cutoff
                for k, horizon in enumerate(base["horizons_minutes"]):
                    mask = available_history(ledger["times"], cutoff, horizon, base["target_tolerance_seconds"],
                        cfg["intervals"]["rolling_days"], cfg["intervals"]["minimum_calibration_rows"])
                    # Actual outcomes are accessed only after the causal mask.
                    ratio = np.abs(ledger["actual"][mask, k] - ledger["prediction"][mask, k]) / ledger["scale"][mask, k]
                    past_groups = ledger["group"][mask]
                    observed = np.isfinite(ratio)
                    used_times = ledger["times"][mask][observed]
                    if len(used_times):
                        last_used[rows, k] = np.max(used_times + horizon * 60 + base["target_tolerance_seconds"])
                    for regime in range(3):
                        pool = observed & (past_groups == regime)
                        if pool.sum() < cfg["intervals"]["minimum_calibration_rows"]:
                            pool = observed
                        quantile = finite_quantile(ratio[pool], level)
                        target_rows = rows & (group == regime)
                        radius[target_rows, k] = scale[target_rows, k] * quantile
            if not (last_used < times[:, None]).all():
                raise ValueError("Rolling interval calibration crossed forecast origin")
        result[str(level)] = {"lower": prediction - radius, "upper": prediction + radius}
    return result, last_used


def describe(actual, prediction, intervals, group):
    result = []
    lo, hi = intervals["0.9"]["lower"], intervals["0.9"]["upper"]
    for k in range(actual.shape[1]):
        valid = np.isfinite(actual[:, k])
        rapid = valid & (group == 2)
        result.append({"coverage90": float(np.mean((actual[valid,k] >= lo[valid,k]) & (actual[valid,k] <= hi[valid,k]))),
            "width90": float(np.mean(hi[valid,k]-lo[valid,k])),
            "interval_score90": float(np.nanmean(interval_score(actual[:,k],lo[:,k],hi[:,k]))),
            "rapid_n": int(rapid.sum()),
            "rapid_coverage90": float(np.mean((actual[rapid,k] >= lo[rapid,k]) & (actual[rapid,k] <= hi[rapid,k]))) if rapid.any() else None})
    return result


def run(dataset, base, cfg, output, device):
    stage(output, "intervals", dataset, cfg, ["src/v3/uncertainty.py", "src/v3/ensemble.py"])
    if base != dataset.metadata["config"]:
        raise ValueError("Base configuration differs from the verified dataset")
    complete = output / "intervals-complete.json"
    if complete.exists():
        for name, expected in json.loads(complete.read_text())["sha256"].items():
            if sha256(output / name) != expected:
                raise ValueError(f"Interval selection artifact changed: {name}")
        print("Selezione fasce già completa e artefatti verificati", flush=True)
        return
    positions, _ = all_validation(dataset, base)
    prediction = load_forecast(output / "oof/selected_crossfit.npz", positions)
    inputs = LongInputs(dataset, output)
    x = scale_inputs(dataset, positions, prediction, inputs)
    times = dataset.times[positions]
    a = utc_boundary(cfg["intervals"]["selection_train_end"], base).timestamp()
    b = utc_boundary(cfg["intervals"]["selection_calibration_end"], base).timestamp()
    train = times + 10800 + base["target_tolerance_seconds"] < a
    calibration_rows = (times >= a) & (times + 10800 + base["target_tolerance_seconds"] < b)
    evaluation = times >= b
    y = np.asarray(dataset.targets[positions])
    scale_model = ErrorScale(cfg).fit(x[train], prediction[train], y[train])
    scales = scale_model.predict(x)
    group = regimes(dataset, positions, cfg)
    meal = (dataset.features[positions, dataset.feature_names.index("carbs_sum_180m")] > 0).astype(int)
    choices = []
    for mode in cfg["intervals"]["modes"]:
        mode_scale = np.ones_like(prediction) if mode == "constant_meal" else scales
        mode_group = meal if mode == "constant_meal" else group
        calibration = {"prediction": prediction[calibration_rows], "actual": y[calibration_rows],
                       "scale": mode_scale[calibration_rows], "group": mode_group[calibration_rows]}
        ledger = {"prediction": prediction, "actual": y, "scale": mode_scale, "group": mode_group, "times": times}
        intervals, _ = bands(mode, prediction[evaluation], mode_scale[evaluation], mode_group[evaluation], calibration,
                             cfg, base, times=times[evaluation], ledger=ledger, anchor=a)
        statistics = describe(y[evaluation], prediction[evaluation], intervals, group[evaluation])
        choices.append({"mode": mode, "metrics": statistics, "selection_score": float(np.mean([s["interval_score90"] for s in statistics]))})
        print(f"Fasce {mode}: interval score {choices[-1]['selection_score']:.2f}", flush=True)
    chosen = min(choices, key=lambda row: row["selection_score"])
    frozen = {"mode": chosen["mode"], "choices": choices, "train_before": cfg["intervals"]["selection_train_end"],
              "calibration_before": cfg["intervals"]["selection_calibration_end"], "evaluation": "second half of July; development diagnostic only"}
    path = output / "interval-selection-frozen.json"
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError("Previously frozen interval selection differs")
    atomic_json(path, frozen)
    final_scale = ErrorScale(cfg).fit(x, prediction, y)
    atomic_joblib(output / "interval-scale.joblib", {"model": final_scale, "training_positions": positions,
        "label_upper_bound": float(np.max(times + 10800 + base["target_tolerance_seconds"]))})
    atomic_json(complete, {"sha256": {name: sha256(output / name) for name in
        ["selection-frozen.json", "oof/selected_crossfit.npz", "interval-selection-frozen.json", "interval-scale.joblib"]}})
