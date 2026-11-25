"""Build auditable, causal experimental data without rewriting source exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RULES_VERSION = "personal-meal-correction-v1"


def read_config(path=None):
    return json.loads(Path(path or ROOT / "config/experiment.json").read_text())


def utc(values, timezone="Europe/Rome"):
    """Never silently shift ambiguous/nonexistent local times."""
    dates = pd.to_datetime(values)
    return dates.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")


def boundary(value, timezone):
    return pd.Timestamp(value, tz=timezone).tz_convert("UTC")


def classify_boluses(raw):
    """User's notebook rules, preserving precedence and local completion clock.

    Only deliberate change: zero-carb MANUAL boluses remain corrections, rather
    than being conflated with pump-generated automatic boluses.
    """
    t = pd.to_datetime(raw["Data/Ora di completamento"])
    minute = t.dt.hour * 60 + t.dt.minute + t.dt.second / 60
    carbs = raw["Quantità Carboidrati"]
    glucose = raw["Glicemia (mg/dL)"]
    automatic = raw["Modalità di erogazione del bolo"].eq("Automatico")
    evening = minute.between(1170, 1320) & (
        carbs.gt(41) | (glucose.lt(270) & carbs.gt(29)) | (glucose.lt(210) & carbs.gt(21))
    )
    lunch = minute.between(750, 900) & glucose.lt(250) & carbs.gt(21)
    meal = (carbs.gt(50) | glucose.lt(180)
            | raw["Modalità di erogazione del bolo"].eq("Prolungato") | evening | lunch)
    correction = (
        t.dt.hour.lt(5) | t.dt.hour.between(16, 18)
        | (glucose.gt(250) & carbs.lt(35))
        | (glucose.gt(200) & carbs.lt(21))
        | (glucose.gt(270) & carbs.lt(41))
        | (carbs.lt(20) & ~minute.between(360, 690))
    )
    result = pd.Series("Correzione", index=raw.index)
    result.loc[carbs.gt(0) & (meal | (~correction & carbs.gt(20)))] = "Alimenti"
    result.loc[automatic] = "Automatico"
    return result


def bolus_events(raw, timezone):
    complete = utc(raw["Data/Ora di completamento"], timezone)
    start = utc(raw["Orario/Data d'inizio del bolo esteso"], timezone)
    purpose = classify_boluses(raw)
    out = pd.DataFrame({
        "event_time": start.fillna(complete),
        # Export has no original arrival/entry time. Completion is conservative:
        # final delivered amount and interruption status cannot be known earlier.
        "available_at": complete,
        "completed_at": complete,
        "extended_started_at": start,
        "extended_completed_at": utc(raw["Completamento del bolo esteso"], timezone),
        "original_type": raw["Tipo di bolo"],
        "delivery_mode": raw["Modalità di erogazione del bolo"],
        "purpose_inferred": purpose,
        "classification_version": RULES_VERSION,
        "carbs_recorded_g": raw["Quantità Carboidrati"],
        "carbs_meal_inferred_g": raw["Quantità Carboidrati"].where(purpose.eq("Alimenti"), 0),
        "insulin_delivered_u": raw["Insulina erogata"],
        "extended_delivered_u": raw["Insulina erogata con bolo esteso"],
        "extended_duration_minutes": raw["Durata (min)"],
        "standard_percent": raw["Percentuale standard"],
        "completion_status": raw["Descrizione dello stato di completamento"],
        "extended_status": raw["Stato di completamento del bolo esteso"],
        "bolus_glucose_mgdl": raw["Glicemia (mg/dL)"].replace(0, np.nan),
    })
    out.insert(0, "event_id", [hashlib.sha256("|".join(map(str, row)).encode()).hexdigest()[:24]
                             for row in out.itertuples(index=False, name=None)])
    return out


def cgm_grid(cgm, frequency_minutes=5):
    """Last actual sample in (t-5min,t]; never fill missing target values."""
    d = cgm.sort_values("observed_at").copy()
    d["grid_time"] = d.observed_at.dt.ceil(f"{frequency_minutes}min")
    d = d.drop_duplicates("grid_time", keep="last").set_index("grid_time")
    index = pd.date_range(d.index.min(), d.index.max(), freq=f"{frequency_minutes}min")
    out = d.reindex(index)
    out.index.name = "datetime"
    out["glucose_observed"] = out.glucose.notna()
    out["sample_age_seconds"] = (out.index.to_series() - out.observed_at).dt.total_seconds()
    return out


def event_features(events, basal, index, config):
    """Inputs indexed by when known, not by retrospectively inferred start.

    manual: idealised complete manual logging of meals and delivered manual
    boluses. pump: additional automatic boluses and basal; NOT currently live
    accessible. Extended total appears only on completion; no fictitious IOB.
    """
    freq = f"{config['frequency_minutes']}min"
    result = pd.DataFrame(index=index)
    a = events.available_at + pd.Timedelta(minutes=config["manual_latency_minutes"])
    for name, values in {
        "meal_carbs_g": events.carbs_meal_inferred_g,
        "manual_bolus_confirmed_u": events.insulin_delivered_u.where(events.delivery_mode.ne("Automatico"), 0),
    }.items():
        series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(a).ceil(freq)).groupby(level=0).sum()
        result[name] = series.reindex(index, fill_value=0)
    a = events.available_at + pd.Timedelta(minutes=config["pump_latency_minutes"])
    values = events.insulin_delivered_u.where(events.delivery_mode.eq("Automatico"), 0)
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(a).ceil(freq)).groupby(level=0).sum()
    result["automatic_bolus_confirmed_u"] = series.reindex(index, fill_value=0)
    a = basal.observed_at + pd.Timedelta(minutes=config["pump_latency_minutes"])
    # min_count preserves unknown/missing basal instead of pretending zero.
    series = pd.Series(basal.basal_u.to_numpy(), index=pd.DatetimeIndex(a).ceil(freq)).groupby(level=0).sum(min_count=1)
    result["basal_recorded_u"] = series.reindex(index)
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(config=None, output=None):
    config = config or read_config()
    output = Path(output or ROOT / "data/processed")
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = ROOT / "data/tandemsource_data/merged"
    cgm_path = ROOT / "data/cleaned/cleaned-CGM.csv"
    cgm = pd.read_csv(cgm_path)
    cgm["observed_at"] = utc(cgm.pop("datetime"), config["timezone"])
    raw_bolus = pd.read_csv(raw_dir / "merged-Bolo.csv")
    events = bolus_events(raw_bolus, config["timezone"])
    basal = pd.read_csv(ROOT / "data/cleaned/cleaned-Basal.csv")
    basal.columns = ["observed_at", "basal_u"]
    basal["observed_at"] = utc(basal.observed_at, config["timezone"])
    conflict = basal.observed_at.duplicated(keep=False)
    conflicts = basal.loc[conflict].copy()
    # Both conflicting records become unknown. No arbitrary first/last/sum.
    basal.loc[conflict, "basal_u"] = np.nan
    basal = basal.drop_duplicates("observed_at")
    grid = cgm_grid(cgm, config["frequency_minutes"])
    grid = grid.join(event_features(events, basal, grid.index, config))
    grid.to_csv(output / "timeline.csv", index=True)
    events.to_csv(output / "bolus_events.csv", index=False)
    conflicts.to_csv(output / "basal_conflicts.csv", index=False)
    gaps = cgm.observed_at.diff().dt.total_seconds()
    gap_rows = cgm.loc[gaps.gt(450), ["observed_at"]].copy()
    gap_rows["previous_at"] = cgm.observed_at.shift().loc[gap_rows.index]
    gap_rows["gap_minutes"] = gaps.loc[gap_rows.index] / 60
    gap_rows.to_csv(output / "cgm_gaps.csv", index=False)
    hourly_path = raw_dir / "merged-Somministrazione oraria basale.csv"
    hourly = pd.read_csv(hourly_path).drop(columns="Numero di serie")
    h = hourly.melt(id_vars=["Data dell'evento"], var_name="hour", value_name="hourly_u").dropna()
    h["local_hour"] = pd.to_datetime(h["Data dell'evento"]) + pd.to_timedelta(h.hour.astype(int), unit="h")
    # Overlapping export dates may contain complementary partial days. Compare
    # only hourly values that agree across files, without choosing conflicts.
    consistent = h.groupby("local_hour").hourly_u.agg(["nunique", "first"])
    local = basal.observed_at.dt.tz_convert(config["timezone"]).dt.tz_localize(None)
    actual = basal.groupby(local.dt.floor("h")).basal_u.sum(min_count=1)
    compare = consistent.loc[consistent["nunique"].eq(1), "first"].to_frame("hourly_u").join(actual).dropna()
    audit = {
        "timezone_assumption": config["timezone"],
        "rules_version": RULES_VERSION,
        "cgm": {"rows": len(cgm), "start_utc": str(cgm.observed_at.min()), "end_utc": str(cgm.observed_at.max()),
                "min_mgdl": float(cgm.glucose.min()), "max_mgdl": float(cgm.glucose.max()),
                "below_70": int(cgm.glucose.lt(70).sum()), "grid_rows": len(grid),
                "missing_grid_cells": int(grid.glucose.isna().sum()),
                "utc_gaps_over_one_hour": int(gaps.gt(3600).sum()),
                "max_gap_minutes": float(gaps.max() / 60),
                "source_duplicates": int(cgm.duplicated("observed_at").sum())},
        "bolus": {"rows": len(events), "extended": int(events.delivery_mode.eq("Prolungato").sum()),
                  "interrupted": int(events.completion_status.ne("Completato").sum()),
                  "interrupted_delivered_u": float(events.loc[events.completion_status.ne("Completato"), "insulin_delivered_u"].sum()),
                  "manual_zero_carbs": int((events.delivery_mode.ne("Automatico") & events.carbs_recorded_g.eq(0)).sum()),
                  "meal_count_inferred": int(events.purpose_inferred.eq("Alimenti").sum()),
                  "carbs_recorded_g": float(events.carbs_recorded_g.sum()),
                  "carbs_meal_inferred_g": float(events.carbs_meal_inferred_g.sum())},
        "basal": {"conflicting_records": len(conflicts), "policy": "conflicting timestamp -> missing",
                  "hourly_comparison_hours": len(compare),
                  "hourly_mean_absolute_difference_u": float((compare.basal_u - compare.hourly_u).abs().mean())},
        "assumptions": [
            "Local naive timestamps assumed Europe/Rome; DST converted to UTC before gridding.",
            "Historical arrival times absent: configured delays are assumptions, not measured live latency.",
            "Bolus final totals/status available only at completion, including extended boluses.",
            "Meal grams inferred using user-confirmed personal rules; not independently verified consumption.",
            "Manual arm assumes complete logging; historical automatic/basal inputs are a separate retrospective arm.",
            "Absent event = zero only under assumed complete event capture; missing basal/CGM remains NaN.",
            "No measured IOB series and no pharmacological IOB reconstruction.",
        ],
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [
            cgm_path, ROOT / "data/cleaned/cleaned-Basal.csv", ROOT / "data/cleaned/cleaned-Bolo.csv",
            raw_dir / "merged-Bolo.csv", hourly_path]},
        "config": config,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "cgm": audit["cgm"], "bolus": audit["bolus"]}, indent=2))
    return grid, events, audit


if __name__ == "__main__":
    prepare()
