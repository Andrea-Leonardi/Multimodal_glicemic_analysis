from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

LOW_GLUCOSE = 70
HIGH_GLUCOSE = 200
CAP_SECONDS = 15 * 60


def _parse_bound(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def _resolve_exports(export_dirs: list[Path] | None = None) -> list[Path]:
    exports = export_dirs or cfg.resolve_glooko_exports()
    if not exports:
        raise FileNotFoundError(f"No Glooko export directories found in {cfg.GLOOKO_DIR}")
    return exports


def _read_export_csvs(
    export_dirs: list[Path],
    *,
    pattern: str | None = None,
    relative_file: Path | None = None,
    skiprows: int = 1,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for exp_dir in export_dirs:
        paths = sorted(exp_dir.glob(pattern)) if pattern else [exp_dir / relative_file]
        for path in paths:
            if path.exists():
                frames.append(pd.read_csv(path, skiprows=skiprows))
    if not frames:
        descriptor = pattern or str(relative_file)
        raise FileNotFoundError(f"No Glooko files matched {descriptor!r}")
    return pd.concat(frames, ignore_index=True)


def _apply_window(
    df: pd.DataFrame,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if start is not None:
        out = out.loc[out.index >= start]
    if end is not None:
        out = out.loc[out.index < end]
    return out


def _parse_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out[col].astype(str).str.replace(",", ".", regex=False)
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _build_daily_bg(df_bg: pd.DataFrame) -> pd.DataFrame:
    df_bg = df_bg.rename(
        columns={
            "Data e ora": "Time",
            "Valore glicemia CGM (mg/dl)": "Bg",
        }
    )
    df_bg["Time"] = pd.to_datetime(df_bg["Time"], dayfirst=True, errors="coerce")
    df_bg = _parse_numeric_columns(df_bg, ["Bg"])
    df_bg = df_bg.dropna(subset=["Time", "Bg"])
    df_bg = (
        df_bg.set_index("Time")
        .sort_index()
        .drop(columns="Numero di serie", errors="ignore")
    )
    df_bg = df_bg.groupby(level=0)["Bg"].mean().to_frame()

    df = df_bg.copy()
    df["t_next"] = df.index.to_series().shift(-1)
    df["dt_s"] = (df["t_next"] - df.index.to_series()).dt.total_seconds()
    df["dt_s"] = df["dt_s"].clip(lower=0, upper=CAP_SECONDS)
    df = df.dropna(subset=["dt_s", "Bg"])
    df = df.loc[df["dt_s"] > 0].copy()

    def daily_stats(group: pd.DataFrame) -> pd.Series:
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

    daily_bg = df.groupby(pd.Grouper(freq="D")).apply(daily_stats)
    daily_bg.index = pd.to_datetime(daily_bg.index).normalize()
    return daily_bg.add_prefix("bg_").dropna(how="all")


def _load_daily_cartridge_events(
    export_dirs: list[Path],
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    df_alarms = _read_export_csvs(export_dirs, relative_file=Path(cfg.ALARMS_FILE))
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
    df_alarms = _apply_window(df_alarms, start=start, end=end)
    df_alarms = df_alarms.loc[df_alarms["Cart_load"].eq("Cartridge Loaded")].copy()
    df_alarms["cartridge_loaded"] = 1
    df_alarms["Date"] = pd.to_datetime(df_alarms.index).normalize()
    return df_alarms.groupby("Date")["cartridge_loaded"].max().to_frame()


def _load_daily_insulin(
    export_dirs: list[Path],
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    df_ins = _read_export_csvs(
        export_dirs,
        relative_file=Path(cfg.INSULIN_SUBDIR) / cfg.INSULIN_FILE,
    )
    df_ins = df_ins.rename(
        columns={
            "Data e ora": "Time",
            "Bolo totale (U)": "ins_bolo_tot",
            "Insulina totale (U)": "ins_tot",
            "Basale totale (U)": "ins_basal_tot",
        }
    )
    df_ins["Time"] = pd.to_datetime(df_ins["Time"], dayfirst=True, errors="coerce")
    df_ins = _parse_numeric_columns(df_ins, ["ins_bolo_tot", "ins_tot", "ins_basal_tot"])
    df_ins = (
        df_ins.dropna(subset=["Time"])
        .set_index("Time")
        .sort_index()
        .drop(columns="Numero di serie", errors="ignore")
    )
    df_ins = _apply_window(df_ins, start=start, end=end)
    df_ins["Date"] = pd.to_datetime(df_ins.index).normalize()
    daily_ins = df_ins.groupby("Date")[["ins_bolo_tot", "ins_tot", "ins_basal_tot"]].max()
    return daily_ins.fillna(0)


def _load_daily_carbs(
    export_dirs: list[Path],
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    df_carb = _read_export_csvs(
        export_dirs,
        relative_file=Path(cfg.INSULIN_SUBDIR) / cfg.BOLUS_FILE,
    ).fillna(0)
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
    df_carb = _apply_window(df_carb, start=start, end=end)
    df_carb = _parse_numeric_columns(df_carb, ["glicemia", "carb", "insulina"])
    df_carb = df_carb.loc[(df_carb["insulina"] > 0) & (df_carb["carb"] > 0)].copy()
    df_carb["ora"] = pd.to_datetime(df_carb.index, errors="coerce")

    base_240 = (df_carb["carb"] <= 40) & (df_carb["glicemia"] > 240)
    base_200 = (df_carb["carb"] <= 40) & (df_carb["glicemia"] > 200)

    def in_time_window(series: pd.Series, start_time: str, end_time: str) -> pd.Series:
        start_clock = pd.to_datetime(start_time).time()
        end_clock = pd.to_datetime(end_time).time()
        return series.dt.time.between(start_clock, end_clock)

    time_mask = (
        in_time_window(df_carb["ora"], "15:30", "19:30")
        | in_time_window(df_carb["ora"], "22:30", "23:59")
        | in_time_window(df_carb["ora"], "00:01", "05:00")
    )
    df_carb["is_correction"] = base_240 | (base_200 & time_mask)
    df_carb = df_carb.loc[~df_carb["is_correction"], ["carb"]].copy()
    df_carb.index = pd.to_datetime(df_carb.index, errors="coerce")
    daily_carbs = df_carb["carb"].resample("D").sum().to_frame("carb_daily")
    return daily_carbs


def clean_glooko_data(
    export_dirs: list[Path] | None = None,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    exports = _resolve_exports(export_dirs)
    df_bg = _read_export_csvs(exports, pattern=cfg.CGM_PATTERN)
    daily_glooko = _build_daily_bg(df_bg)
    daily_glooko = _apply_window(daily_glooko, start=start, end=end)

    daily_cart = _load_daily_cartridge_events(exports, start=start, end=end)
    daily_ins = _load_daily_insulin(exports, start=start, end=end)
    daily_carbs = _load_daily_carbs(exports, start=start, end=end)

    daily_glooko = daily_glooko.join(daily_cart, how="left")
    daily_glooko["cartridge_loaded"] = daily_glooko["cartridge_loaded"].fillna(0).astype(int)
    daily_glooko = daily_glooko.join(daily_ins, how="left")
    daily_glooko[["ins_bolo_tot", "ins_tot", "ins_basal_tot"]] = daily_glooko[
        ["ins_bolo_tot", "ins_tot", "ins_basal_tot"]
    ].fillna(0)
    daily_glooko = daily_glooko.join(daily_carbs, how="left")
    daily_glooko["carb_daily"] = daily_glooko["carb_daily"].fillna(0)
    return daily_glooko.sort_index()


def save_glooko_data(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    output_path = output_path or cfg.GLOOKO_DATA_CLEANED
    cfg.ensure_processed_dir()
    df.to_parquet(output_path, engine="pyarrow", index=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and aggregate Glooko exports.")
    parser.add_argument("--start-date", dest="start_date")
    parser.add_argument("--end-date", dest="end_date")
    parser.add_argument("--output", type=Path, default=cfg.GLOOKO_DATA_CLEANED)
    args = parser.parse_args()

    df = clean_glooko_data(
        start=_parse_bound(args.start_date),
        end=_parse_bound(args.end_date),
    )
    saved_path = save_glooko_data(df, args.output)
    print(f"Saved: {saved_path} | rows={len(df)} | columns={len(df.columns)}")


if __name__ == "__main__":
    main()
