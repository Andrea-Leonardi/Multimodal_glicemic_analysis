from __future__ import annotations

"""
This script converts raw Glooko exports into a clean daily glycemic dataset.

Its role in the project is to aggregate continuous glucose, insulin, alarms,
and meal-related records into daily summaries that can be combined with
wearable-derived behavioral variables.

The workflow is:
1. load CGM, alarms, insulin, and bolus files from all available exports,
2. parse timestamps and numeric values,
3. compute time-weighted daily glucose statistics,
4. derive daily insulin and cartridge-change indicators,
5. estimate daily carbohydrate totals after excluding likely correction events,
6. save the result as a daily parquet dataset.

This file is intentionally organized as a readable data-preparation stage.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

# ==================================================
# Glucose thresholds used for daily summaries
# ==================================================

LOW_GLUCOSE = 70
HIGH_GLUCOSE = 200
CAP_SECONDS = 15 * 60


def parse_bound(value: str | None) -> pd.Timestamp | None:
    """Convert an optional CLI date to a pandas timestamp."""
    if value is None:
        return None
    return pd.Timestamp(value)


def compute_daily_bg_stats(group: pd.DataFrame) -> pd.Series:
    """Compute time-weighted glucose statistics for one calendar day."""

    weights_s = group["dt_s"].to_numpy(dtype=float)
    values = group["Bg"].to_numpy(dtype=float)

    mask = np.isfinite(weights_s) & np.isfinite(values) & (weights_s > 0)
    weights_s = weights_s[mask]
    values = values[mask]
    if weights_s.size == 0 or weights_s.sum() == 0:
        return pd.Series(dtype="float64")

    weights_min = weights_s / 60.0
    mean = np.average(values, weights=weights_s)
    variance = np.average((values - mean) ** 2, weights=weights_s)
    sd = np.sqrt(variance)
    cv = sd / mean if mean != 0 else np.nan

    total = weights_s.sum()
    tir = weights_s[(values >= LOW_GLUCOSE) & (values <= HIGH_GLUCOSE)].sum()
    tar = weights_s[values > HIGH_GLUCOSE].sum()
    auc_total = np.sum(values * weights_min)
    auc_above = np.sum(np.clip(values - HIGH_GLUCOSE, 0, None) * weights_min)

    return pd.Series(
        {
            "mean": mean,
            "sd": sd,
            "cv": cv,
            "min": np.nanmin(values),
            "median": np.nanmedian(values),
            "max": np.nanmax(values),
            "tir_%": 100 * tir / total,
            "tar_%": 100 * tar / total,
            "auc_above_limit_rate": 100 * auc_above / auc_total if auc_total else np.nan,
        }
    )


def clean_glooko_data(
    export_dirs: list[Path] | None = None,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a clean daily Glooko dataset from the raw exports."""

    exports = export_dirs or cfg.resolve_glooko_exports()
    if not exports:
        raise FileNotFoundError(f"No Glooko export directories found in {cfg.GLOOKO_DIR}")

    # ==================================================
    # 1. Load all CGM files across the available exports
    # ==================================================

    cgm_frames: list[pd.DataFrame] = []
    for exp_dir in exports:
        for path in sorted(exp_dir.glob(cfg.CGM_PATTERN)):
            cgm_frames.append(pd.read_csv(path, skiprows=1))
    if not cgm_frames:
        raise FileNotFoundError(f"No Glooko files matched {cfg.CGM_PATTERN!r}")

    df_bg = pd.concat(cgm_frames, ignore_index=True)

    # ==================================================
    # 2. Parse CGM timestamps and glucose values
    # ==================================================

    df_bg = df_bg.rename(
        columns={
            "Data e ora": "Time",
            "Valore glicemia CGM (mg/dl)": "Bg",
        }
    )
    df_bg["Time"] = pd.to_datetime(df_bg["Time"], dayfirst=True, errors="coerce")
    df_bg["Bg"] = df_bg["Bg"].astype(str).str.replace(",", ".", regex=False)
    df_bg["Bg"] = pd.to_numeric(df_bg["Bg"], errors="coerce")
    df_bg = df_bg.dropna(subset=["Time", "Bg"])
    df_bg = (
        df_bg.set_index("Time")
        .sort_index()
        .drop(columns="Numero di serie", errors="ignore")
    )
    df_bg = df_bg.groupby(level=0)["Bg"].mean().to_frame()

    # ==================================================
    # 3. Build time-weighted daily glucose summaries
    # ==================================================

    df = df_bg.copy()
    df["t_next"] = df.index.to_series().shift(-1)
    df["dt_s"] = (df["t_next"] - df.index.to_series()).dt.total_seconds()
    df["dt_s"] = df["dt_s"].clip(lower=0, upper=CAP_SECONDS)
    df = df.dropna(subset=["dt_s", "Bg"])
    df = df.loc[df["dt_s"] > 0].copy()

    daily_glooko = df.groupby(pd.Grouper(freq="D")).apply(compute_daily_bg_stats)
    daily_glooko.index = pd.to_datetime(daily_glooko.index).normalize()
    daily_glooko = daily_glooko.add_prefix("bg_").dropna(how="all")

    if start is not None:
        daily_glooko = daily_glooko.loc[daily_glooko.index >= start]
    if end is not None:
        daily_glooko = daily_glooko.loc[daily_glooko.index < end]

    # ==================================================
    # 4. Load cartridge replacement events
    # ==================================================

    alarm_frames: list[pd.DataFrame] = []
    for exp_dir in exports:
        path = exp_dir / cfg.ALARMS_FILE
        if path.exists():
            alarm_frames.append(pd.read_csv(path, skiprows=1))
    if not alarm_frames:
        raise FileNotFoundError(f"No Glooko files matched {cfg.ALARMS_FILE!r}")

    df_alarms = pd.concat(alarm_frames, ignore_index=True)
    df_alarms = df_alarms.rename(
        columns={
            "Data e ora": "Time",
            "Allarme/Evento": "Cart_load",
        }
    )
    df_alarms["Time"] = pd.to_datetime(df_alarms["Time"], dayfirst=True, errors="coerce")
    df_alarms = df_alarms.dropna(subset=["Time", "Cart_load"])
    df_alarms = (
        df_alarms.set_index("Time")
        .sort_index()
        .drop(columns="Numero di serie", errors="ignore")
    )
    if start is not None:
        df_alarms = df_alarms.loc[df_alarms.index >= start]
    if end is not None:
        df_alarms = df_alarms.loc[df_alarms.index < end]
    df_alarms = df_alarms.loc[df_alarms["Cart_load"].eq("Cartridge Loaded")].copy()
    df_alarms["cartridge_loaded"] = 1
    df_alarms["Date"] = pd.to_datetime(df_alarms.index).normalize()
    daily_cart = df_alarms.groupby("Date")["cartridge_loaded"].max().to_frame()

    daily_glooko = daily_glooko.join(daily_cart, how="left")
    daily_glooko["cartridge_loaded"] = daily_glooko["cartridge_loaded"].fillna(0).astype(int)

    # ==================================================
    # 5. Load daily insulin totals
    # ==================================================

    insulin_frames: list[pd.DataFrame] = []
    insulin_file = Path(cfg.INSULIN_SUBDIR) / cfg.INSULIN_FILE
    for exp_dir in exports:
        path = exp_dir / insulin_file
        if path.exists():
            insulin_frames.append(pd.read_csv(path, skiprows=1))
    if not insulin_frames:
        raise FileNotFoundError(f"No Glooko files matched {insulin_file!r}")

    df_ins = pd.concat(insulin_frames, ignore_index=True)
    df_ins = df_ins.rename(
        columns={
            "Data e ora": "Time",
            "Bolo totale (U)": "ins_bolo_tot",
            "Insulina totale (U)": "ins_tot",
            "Basale totale (U)": "ins_basal_tot",
        }
    )
    df_ins["Time"] = pd.to_datetime(df_ins["Time"], dayfirst=True, errors="coerce")
    for col in ["ins_bolo_tot", "ins_tot", "ins_basal_tot"]:
        df_ins[col] = df_ins[col].astype(str).str.replace(",", ".", regex=False)
        df_ins[col] = pd.to_numeric(df_ins[col], errors="coerce")
    df_ins = (
        df_ins.dropna(subset=["Time"])
        .set_index("Time")
        .sort_index()
        .drop(columns="Numero di serie", errors="ignore")
    )
    if start is not None:
        df_ins = df_ins.loc[df_ins.index >= start]
    if end is not None:
        df_ins = df_ins.loc[df_ins.index < end]
    df_ins["Date"] = pd.to_datetime(df_ins.index).normalize()
    daily_ins = df_ins.groupby("Date")[["ins_bolo_tot", "ins_tot", "ins_basal_tot"]].max()

    daily_glooko = daily_glooko.join(daily_ins, how="left")
    daily_glooko[["ins_bolo_tot", "ins_tot", "ins_basal_tot"]] = daily_glooko[
        ["ins_bolo_tot", "ins_tot", "ins_basal_tot"]
    ].fillna(0)

    # ==================================================
    # 6. Load bolus data and derive daily carbohydrate totals
    # ==================================================

    bolus_frames: list[pd.DataFrame] = []
    bolus_file = Path(cfg.INSULIN_SUBDIR) / cfg.BOLUS_FILE
    for exp_dir in exports:
        path = exp_dir / bolus_file
        if path.exists():
            bolus_frames.append(pd.read_csv(path, skiprows=1))
    if not bolus_frames:
        raise FileNotFoundError(f"No Glooko files matched {bolus_file!r}")

    df_carb = pd.concat(bolus_frames, ignore_index=True).fillna(0)
    df_carb = df_carb.rename(
        columns={
            "Data e ora": "Time",
            "Immissione glicemia (mg/dl)": "glicemia",
            "Consumo di carboidrati (g)": "carb",
            "Insulina erogata (U)": "insulina",
        }
    )
    df_carb["Time"] = pd.to_datetime(df_carb["Time"], dayfirst=True, errors="coerce")
    df_carb = (
        df_carb.set_index("Time")
        .sort_index()
        .drop(
            columns={
                "Numero di serie",
                "Tipo di insulina",
                "Rapporto carboidrati",
                "Erogazione iniziale (U)",
                "Erogazione estesa (U)",
            },
            errors="ignore",
        )
    )
    if start is not None:
        df_carb = df_carb.loc[df_carb.index >= start]
    if end is not None:
        df_carb = df_carb.loc[df_carb.index < end]

    for col in ["glicemia", "carb", "insulina"]:
        df_carb[col] = df_carb[col].astype(str).str.replace(",", ".", regex=False)
        df_carb[col] = pd.to_numeric(df_carb[col], errors="coerce")

    df_carb = df_carb.loc[(df_carb["insulina"] > 0) & (df_carb["carb"] > 0)].copy()
    df_carb["ora"] = pd.to_datetime(df_carb.index, errors="coerce")

    base_240 = (df_carb["carb"] <= 40) & (df_carb["glicemia"] > 240)
    base_200 = (df_carb["carb"] <= 40) & (df_carb["glicemia"] > 200)

    time_1530_1930 = df_carb["ora"].dt.time.between(
        pd.to_datetime("15:30").time(),
        pd.to_datetime("19:30").time(),
    )
    time_2230_2359 = df_carb["ora"].dt.time.between(
        pd.to_datetime("22:30").time(),
        pd.to_datetime("23:59").time(),
    )
    time_0001_0500 = df_carb["ora"].dt.time.between(
        pd.to_datetime("00:01").time(),
        pd.to_datetime("05:00").time(),
    )

    time_mask = time_1530_1930 | time_2230_2359 | time_0001_0500
    df_carb["is_correction"] = base_240 | (base_200 & time_mask)

    df_carb = df_carb.loc[~df_carb["is_correction"], ["carb"]].copy()
    df_carb.index = pd.to_datetime(df_carb.index, errors="coerce")
    daily_carbs = df_carb["carb"].resample("D").sum().to_frame("carb_daily")

    daily_glooko = daily_glooko.join(daily_carbs, how="left")
    daily_glooko["carb_daily"] = daily_glooko["carb_daily"].fillna(0)

    # ==================================================
    # 7. Final ordering
    # ==================================================

    return daily_glooko.sort_index()


def save_glooko_data(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Save the cleaned Glooko dataset."""
    output_path = output_path or cfg.GLOOKO_DATA_CLEANED
    cfg.ensure_processed_dir()
    df.to_parquet(output_path, engine="pyarrow", index=True)
    return output_path


def main() -> None:
    """CLI entry point used by the project pipeline."""
    parser = argparse.ArgumentParser(description="Clean and aggregate Glooko exports.")
    parser.add_argument("--start-date", dest="start_date")
    parser.add_argument("--end-date", dest="end_date")
    parser.add_argument("--output", type=Path, default=cfg.GLOOKO_DATA_CLEANED)
    args = parser.parse_args()

    df = clean_glooko_data(
        start=parse_bound(args.start_date),
        end=parse_bound(args.end_date),
    )
    saved_path = save_glooko_data(df, args.output)
    print(f"Saved: {saved_path} | rows={len(df)} | columns={len(df.columns)}")


if __name__ == "__main__":
    main()
