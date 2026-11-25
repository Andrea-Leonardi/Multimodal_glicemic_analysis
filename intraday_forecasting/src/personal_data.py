"""Personal historical experiment: exact sensor origins, causal exposure features.

Sources for the mathematical insulin curve (not a dosing implementation):
https://github.com/LoopKit/LoopKit/blob/dev/LoopKit/InsulinKit/ExponentialInsulinModel.swift
https://github.com/openaps/oref0/blob/master/lib/iob/calculate.js
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ROOT, bolus_events, sha256, utc

DEFAULT_BLOCK_SIZE = 512
MAX_EXPOSURE_CELLS = 1_000_000


def epoch_seconds(values):
    """pandas 3 may store datetimes in us; asi8 alone does not imply ns."""
    return pd.DatetimeIndex(values).as_unit("ns").asi8 / 1e9


def row_blocks(length, block_size=DEFAULT_BLOCK_SIZE):
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    for start in range(0, length, block_size):
        yield slice(start, min(start + block_size, length))


def configuration(path=None):
    return json.loads(Path(path or ROOT / "config/personal-v2.json").read_text())


def insulin_curve(minutes, duration=360., peak=75., delay=10.):
    """Fraction of effect remaining and activity per minute for a unit dose.

    Mathematical exponential curve, independently expressed from its integral.
    These are model features, NOT measured plasma insulin or Tandem net IOB.
    Duration excludes the explicit onset delay, following LoopKit convention.
    """
    if not 0 < peak < duration / 2 or delay < 0:
        raise ValueError("Require 0 < peak < duration/2 and delay >= 0")
    age = np.asarray(minutes)
    # Return arrays have the requested shape; intermediate expressions are
    # restricted to 65,536 ages, including when called independently of build().
    remaining = np.empty(age.shape, dtype=np.float64)
    activity = np.empty(age.shape, dtype=np.float64)
    flat = age.reshape(-1)
    out_remaining, out_activity = remaining.reshape(-1), activity.reshape(-1)
    tau = peak * (1 - peak / duration) / (1 - 2 * peak / duration)
    a = 2 * tau / duration
    scale = 1 / (1 - a + (1 + a) * np.exp(-duration / tau))
    for block in row_blocks(flat.size, 65_536):
        values = flat[block].astype(np.float64, copy=False)
        t = np.clip(values - delay, 0, duration)
        active = scale / tau ** 2 * t * (1 - t / duration) * np.exp(-t / tau)
        cumulative = scale * ((1 - a) + (t * t / (tau * duration) - (1 - a) * (t / tau + 1)) * np.exp(-t / tau))
        residual = np.clip(1 - cumulative, 0, 1)
        out_remaining[block] = np.where(values < 0, 0, np.where(values <= delay, 1, np.where(values >= duration + delay, 0, residual)))
        out_activity[block] = np.where((values > delay) & (values < duration + delay), active, 0)
    return remaining, activity


def nearest_indices(times, queries, tolerance_seconds=90, max_indices=None):
    """Nearest observation within tolerance; optional per-origin causal cap.

    Matching a past context's internal nominal slots may use readings a few
    seconds on either side of that slot, but NEVER after the forecast origin.
    Future target matching is a separate use with no cap, for labels only.
    """
    times, queries = np.asarray(times), np.asarray(queries)
    right = np.searchsorted(times, queries)
    if max_indices is not None:
        right = np.minimum(right, max_indices)
    right = np.clip(right, 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    selected = np.where(np.abs(times[left] - queries) < np.abs(times[right] - queries), left, right)
    valid = np.abs(times[selected] - queries) <= tolerance_seconds
    return np.where(valid, selected, -1)


def sample(values, indices):
    result = np.asarray(values)[np.maximum(indices, 0)].astype(np.float32, copy=True)
    result[indices < 0] = np.nan
    return result


def exposure(times, amounts, origins, max_minutes=430):
    """Past doses and ages for one origin block; reject oversized matrices.

    build() calls this with small blocks. The allocation guard also catches a
    timestamp unit regression before allocating rows × months of deliveries.
    """
    times, amounts = np.asarray(times), np.asarray(amounts)
    origins = np.asarray(origins)
    keep = np.isfinite(amounts) & (amounts != 0)
    times, amounts = times[keep], amounts[keep]
    order = np.argsort(times, kind="stable")
    times, amounts = times[order], amounts[order]
    if not len(times):
        return np.zeros((len(origins), 1)), np.full((len(origins), 1), max_minutes + 1.)
    stop = np.searchsorted(times, origins, side="right")
    start = np.searchsorted(times, origins - max_minutes * 60, side="left")
    width = max(int((stop - start).max()), 1)
    if len(origins) * width > MAX_EXPOSURE_CELLS:
        raise MemoryError(f"Exposure shape ({len(origins)}, {width}) exceeds {MAX_EXPOSURE_CELLS:,} cells; split origins into smaller blocks and check time units")
    indices = stop[:, None] - 1 - np.arange(width)[None, :]
    valid = indices >= start[:, None]
    clipped = np.clip(indices, 0, len(times) - 1)
    ages = (origins[:, None] - times[clipped]) / 60
    values = np.where(valid, amounts[clipped], 0)
    ages = np.where(valid, ages, max_minutes + 1)
    return values, ages


def window_sum(times, amounts, origins, minutes):
    order = np.argsort(times, kind="stable")
    times, amounts = np.asarray(times)[order], np.asarray(amounts)[order]
    cs = np.r_[0., np.cumsum(np.nan_to_num(amounts))]
    return cs[np.searchsorted(times, origins, side="right")] - cs[np.searchsorted(times, origins - minutes * 60, side="right")]


def event_streams(raw_boluses, basal, timezone):
    events = bolus_events(raw_boluses, timezone)
    extended = events.delivery_mode.eq("Prolungato")
    start = events.event_time
    complete = events.completed_at
    # A meal is placed at its logged occurrence, with extended-bolus START used
    # rather than its delivery completion. User explicitly confirms no late meals.
    meals = pd.DataFrame({"time": start, "amount": events.carbs_meal_inferred_g})
    # Recover the immediate component. Final extended totals remain at completion
    # instead of inferring a delivery rate from a future total/interruption time.
    initial = events.insulin_delivered_u - events.extended_delivered_u.fillna(0)
    immediate = pd.DataFrame({"time": start, "amount": initial,
                              "automatic": events.delivery_mode.eq("Automatico")})
    tail = pd.DataFrame({"time": complete[extended], "amount": events.extended_delivered_u[extended].fillna(0), "automatic": False})
    bolus = pd.concat([immediate, tail], ignore_index=True).sort_values("time")
    if (bolus.amount < -.0001).any():
        raise ValueError("Extended amount exceeds total delivered insulin")
    basal = basal.copy()
    basal.columns = ["time", "amount"]
    basal["time"] = utc(basal.time, timezone)
    # The two discordant records remain unknown rather than being summed.
    bad = basal.time.duplicated(keep=False)
    basal.loc[bad, "amount"] = np.nan
    basal = basal.drop_duplicates("time")
    streams = {"carbs": meals, "manual": bolus.loc[~bolus.automatic, ["time", "amount"]],
               "automatic": bolus.loc[bolus.automatic, ["time", "amount"]],
               "bolus": bolus[["time", "amount"]], "basal": basal}
    for stream in streams.values():
        stream["seconds"] = epoch_seconds(stream.time)
    return streams, events


@dataclass
class Dataset:
    times: np.ndarray
    glucose: np.ndarray
    features: np.ndarray
    feature_names: list
    targets: np.ndarray
    target_indices: np.ndarray
    sequence: np.ndarray
    eligible: np.ndarray
    event_table: pd.DataFrame
    streams: dict
    metadata: dict


def build(config=None, *, block_size=None, max_rows=None, progress=False):
    config = config or configuration()
    block_size = config.get("build_block_size", DEFAULT_BLOCK_SIZE) if block_size is None else block_size
    # Validate before reading any data, even for an empty dataset.
    next(row_blocks(1, block_size))
    paths = [ROOT / "data/cleaned/cleaned-CGM.csv", ROOT / "data/cleaned/cleaned-Basal.csv",
             ROOT / "data/tandemsource_data/merged/merged-Bolo.csv"]
    cgm = pd.read_csv(paths[0])
    if max_rows is not None:
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
            raise ValueError("max_rows must be a positive integer")
        cgm = cgm.iloc[:max_rows].copy()
    cgm["time"] = utc(cgm.datetime, config["timezone"])
    cgm = cgm.sort_values("time").drop_duplicates("time")
    times = epoch_seconds(cgm.time)
    glucose = cgm.glucose.to_numpy(np.float32)
    streams, events = event_streams(pd.read_csv(paths[2]), pd.read_csv(paths[1]), config["timezone"])
    n = len(times)
    if progress:
        print(f"Costruzione: {n} letture, blocchi da {block_size}", flush=True)
    columns = {"glucose": glucose}
    offsets = sorted(set(list(range(5, 65, 5)) + list(range(75, 181, 15)) + [240, 300, 360, 720, 1440, 2880]))
    for minutes in offsets:
        indices = nearest_indices(times, times - minutes * 60, config["history_tolerance_seconds"], np.arange(n))
        columns[f"glucose_lag_{minutes}m"] = sample(glucose, indices)
        if minutes <= 60:
            columns[f"glucose_delta_{minutes}m"] = glucose - columns[f"glucose_lag_{minutes}m"]
    columns["glucose_acceleration"] = glucose - 2 * columns["glucose_lag_15m"] + columns["glucose_lag_30m"]
    for window in [15, 30, 60, 120, 180, 360]:
        for label in ["mean", "std", "min", "max", "missing"]:
            columns[f"glucose_{label}_{window}m"] = np.empty(n, dtype=np.float32)
        offsets = np.arange(0, window + 1, 5)[None, :] * 60
        for block in row_blocks(n, block_size):
            idx = nearest_indices(times, times[block, None] - offsets, config["history_tolerance_seconds"], np.arange(block.start, block.stop)[:, None])
            x = sample(glucose, idx)
            for label, function in [("mean", np.nanmean), ("std", np.nanstd), ("min", np.nanmin), ("max", np.nanmax)]:
                columns[f"glucose_{label}_{window}m"][block] = function(x, axis=1)
            columns[f"glucose_missing_{window}m"][block] = np.isnan(x).mean(axis=1)
        del idx, x
    local = pd.DatetimeIndex(cgm.time).tz_convert(config["timezone"])
    hour = local.hour + local.minute / 60 + local.second / 3600
    for period in [24, 12, 8]:
        columns[f"clock_sin_{period}"] = np.sin(2 * np.pi * hour / period)
        columns[f"clock_cos_{period}"] = np.cos(2 * np.pi * hour / period)
    columns["weekday_sin"] = np.sin(2 * np.pi * local.dayofweek / 7)
    columns["weekday_cos"] = np.cos(2 * np.pi * local.dayofweek / 7)
    for name, stream in streams.items():
        t, values = stream.seconds.to_numpy(), stream.amount.to_numpy()
        for minutes in [5, 15, 30, 60, 120, 180, 240, 360, 720, 1440]:
            columns[f"{name}_sum_{minutes}m"] = window_sum(t, values, times, minutes)
        positive = t[np.isfinite(values) & (values > 0)]
        previous = np.searchsorted(positive, times, side="right") - 1
        columns[f"{name}_minutes_since"] = np.where(previous >= 0, np.minimum((times - positive[np.maximum(previous, 0)]) / 60, 1440), 1440)
    # Modeled effect remaining includes basal separately. It is deliberately not
    # named pump IOB: clinical systems often report insulin relative to a basal.
    for name in ["bolus", "basal"]:
        stream = streams[name]
        # Allocate only final one-dimensional columns, not whole-dataset event
        # matrices. Keep the original expression/reduction order within each row.
        for duration, peak, delay in config["insulin_curves"]:
            key = f"{name}_d{duration}_p{peak}"
            columns[f"iob_{key}"] = np.empty(n, dtype=np.float32)
            columns[f"activity_{key}"] = np.empty(n, dtype=np.float32)
            for horizon in config["horizons_minutes"]:
                columns[f"past_insulin_effect_{key}_{horizon}m"] = np.empty(n, dtype=np.float32)
        for block in row_blocks(n, block_size):
            doses, ages = exposure(stream.seconds, stream.amount, times[block], max_minutes=430)
            for duration, peak, delay in config["insulin_curves"]:
                key = f"{name}_d{duration}_p{peak}"
                remaining, activity = insulin_curve(ages, duration, peak, delay)
                columns[f"iob_{key}"][block] = (doses * remaining).sum(axis=1)
                columns[f"activity_{key}"][block] = (doses * activity).sum(axis=1)
                for horizon in config["horizons_minutes"]:
                    future_remaining, _ = insulin_curve(ages + horizon, duration, peak, delay)
                    columns[f"past_insulin_effect_{key}_{horizon}m"][block] = (doses * (remaining - future_remaining)).sum(axis=1)
            del doses, ages, remaining, activity, future_remaining
        if progress:
            print(f"Attività insulina: {name} completata", flush=True)
    for tau in [30, 60, 90, 120]:
        columns[f"carb_remaining_proxy_{tau}"] = np.empty(n, dtype=np.float32)
        columns[f"carb_activity_proxy_{tau}"] = np.empty(n, dtype=np.float32)
        for horizon in config["horizons_minutes"]:
            columns[f"past_carb_effect_proxy_{tau}_{horizon}m"] = np.empty(n, dtype=np.float32)
    for block in row_blocks(n, block_size):
        doses, ages = exposure(streams["carbs"].seconds, streams["carbs"].amount, times[block], max_minutes=720)
        for tau in [30, 60, 90, 120]:
            z = ages / tau
            remaining = np.exp(-z) * (1 + z)
            columns[f"carb_remaining_proxy_{tau}"][block] = (doses * remaining).sum(axis=1)
            columns[f"carb_activity_proxy_{tau}"][block] = (doses * ages / tau ** 2 * np.exp(-z)).sum(axis=1)
            for horizon in config["horizons_minutes"]:
                z_future = (ages + horizon) / tau
                future_remaining = np.exp(-z_future) * (1 + z_future)
                columns[f"past_carb_effect_proxy_{tau}_{horizon}m"][block] = (doses * (remaining - future_remaining)).sum(axis=1)
        del doses, ages, z, remaining, z_future, future_remaining
    columns["basal_unknown_6h"] = window_sum(streams["basal"].seconds, streams["basal"].amount.isna().astype(float), times, 360)
    columns["basal_samples_6h"] = window_sum(streams["basal"].seconds, np.ones(len(streams["basal"])), times, 360)
    incomplete_basal = (columns["basal_samples_6h"] < 68) | (columns["basal_unknown_6h"] > 0)
    for key in columns:
        if key.startswith(("iob_basal_", "activity_basal_", "past_insulin_effect_basal_")):
            columns[key][incomplete_basal] = np.nan
    names = list(columns)
    # Avoid an additional full float64 feature matrix before the float32 cast.
    features = np.empty((n, len(names)), dtype=np.float32)
    for column, key in enumerate(names):
        features[:, column] = columns[key]
    target_indices = nearest_indices(times, times[:, None] + np.array(config["horizons_minutes"])[None, :] * 60,
                                     config["target_tolerance_seconds"])
    target = sample(glucose, target_indices)
    # Compact six-hour sequence for the temporal neural model, all channels causal.
    sequence_names = ["glucose", "carbs_sum_5m", "bolus_sum_5m", "basal_sum_5m", "iob_bolus_d360_p75", "activity_bolus_d360_p75"]
    sequence = np.empty((n, len(sequence_names), 72), dtype=np.float32)
    offsets = np.arange(71, -1, -1)[None, :] * 300
    for block in row_blocks(n, block_size):
        idx = nearest_indices(times, times[block, None] - offsets, config["history_tolerance_seconds"], np.arange(block.start, block.stop)[:, None])
        for channel, key in enumerate(sequence_names):
            sequence[block, channel, :] = sample(columns[key], idx)
    del idx, columns
    # Eligibility uses past only. Targets remain NaN when unavailable and are
    # excluded separately during fitting/scoring at each horizon.
    recent_ok = np.isfinite(features[:, [names.index(f"glucose_lag_{k}m") for k in [5, 10, 15]]]).all(axis=1)
    eligible = recent_ok & (times - times[0] >= config["history_days"] * 86400) & (np.isnan(sequence[:, 0]).mean(axis=1) < .15)
    metadata = {"config": config, "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
                "rows": n, "feature_count": len(names), "sequence_names": sequence_names,
                "build_block_size": block_size, "build_row_limit": max_rows, "timestamp_unit": "seconds since Unix epoch, explicitly normalized from pandas datetime resolution",
                "event_policy": "meal at logged occurrence/start; immediate extended component at start, final extended component at completion; no back-distributed final quantity",
                "insulin_curve_source": "https://github.com/LoopKit/LoopKit/blob/dev/LoopKit/InsulinKit/ExponentialInsulinModel.swift",
                "insulin_note": "Humalog model features; bolus and total basal residual kept separate; not measured pump IOB or dosing parameters",
                "carb_note": "Gamma-shape proxy hypotheses, not measured carbohydrate absorption",
                "test_note": "Sep-Oct was inspected in v1. Reused retrospective test, not a fresh blind test."}
    print(f"Dataset personale: {n} letture, {len(names)} variabili, {eligible.sum()} origini utilizzabili", flush=True)
    return Dataset(times, glucose, features, names, target, target_indices, sequence, eligible, events, streams, metadata)


def historical_inputs(dataset, positions, hours, tolerance=90, *, block_size=DEFAULT_BLOCK_SIZE):
    length = hours * 12
    positions = np.asarray(positions)
    indices = np.empty((len(positions), length), dtype=np.int64)
    values = np.empty((len(positions), length), dtype=np.float32)
    offsets = np.arange(length - 1, -1, -1)[None, :] * 300
    for block in row_blocks(len(positions), block_size):
        queries = dataset.times[positions[block], None] - offsets
        indices[block] = nearest_indices(dataset.times, queries, tolerance, positions[block, None])
        values[block] = sample(dataset.glucose, indices[block])
    return values, indices


def split_positions(dataset, start, end, stride=3, training=False):
    start, end = pd.Timestamp(start).timestamp(), pd.Timestamp(end).timestamp()
    selected = dataset.eligible & (dataset.times >= start) & (dataset.times < end)
    if training:
        # Purge all training labels that would cross into the next period,
        # including the tolerance of nearest target matching.
        selected &= dataset.times + 180 * 60 + 90 < end
    return np.flatnonzero(selected & (np.arange(len(dataset.times)) % stride == 0))
