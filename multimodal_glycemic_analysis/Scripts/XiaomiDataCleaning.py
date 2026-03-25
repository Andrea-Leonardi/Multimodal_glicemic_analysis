from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config as cfg

EXCLUDED_KEYS = {"stress", "goal", "valid_stand", "spo2", "intensity"}
HEART_RATE_DROP_COLUMNS = [
    "latest_hr.time",
    "latest_hr.bpm",
    "anaerobic_hr_zone_duration",
    "fat_burning_hr_zone_duration",
    "aerobic_hr_zone_duration",
    "warm_up_hr_zone_duration",
    "extreme_hr_zone_duration",
    "abnormal_hr_count",
]
SLEEP_DROP_COLUMNS = [
    "total_snore_disturb",
    "breath_quality",
    "day_sleep_evaluation",
    "avg_spo2",
    "sleep_trace_duration",
    "sleep_algorithm_version",
    "total_body_move",
    "total_turn_over",
    "sleep_manually_duration",
    "total_snore",
]


def _parse_bound(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(cfg.TIMEZONE)
    return ts.tz_convert(cfg.TIMEZONE)


def _load_json_series(series: pd.Series) -> pd.Series:
    return series.dropna().map(json.loads)


def clean_xiaomi_data(
    fitness_path: Path | None = None,
    training_path: Path | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    fitness_path = fitness_path or (cfg.XIAOMI_DIR / cfg.XIAOMI_FITNESS_FILE)
    training_path = training_path or (cfg.XIAOMI_DIR / cfg.XIAOMI_TRAINING_FILE)

    df_x = pd.read_csv(cfg.require_existing_path(fitness_path))
    df_x["Time"] = pd.to_datetime(df_x["Time"], unit="s", utc=True).dt.tz_convert(cfg.TIMEZONE)

    df_x = df_x.loc[~df_x["Key"].isin(EXCLUDED_KEYS)].copy()
    df_x = df_x.loc[df_x["Tag"] != "daily_mark"].copy()
    df_x = df_x.drop(columns=["Uid", "Sid", "Tag", "UpdateTime"], errors="ignore")

    if start is not None:
        df_x = df_x.loc[df_x["Time"] >= start].copy()
    if end is not None:
        df_x = df_x.loc[df_x["Time"] < end].copy()

    df_x["Date"] = df_x["Time"].dt.date
    duplicates = df_x.groupby(["Date", "Key"]).size()
    duplicated_keys = duplicates.loc[duplicates > 1]
    if not duplicated_keys.empty:
        sample = duplicated_keys.head(5).to_dict()
        raise ValueError(f"Found duplicate Xiaomi daily aggregates: {sample}")

    df_wide = (
        df_x.pivot(index="Date", columns="Key", values="Value")
        .sort_index()
    )

    if "calories" in df_wide.columns:
        df_wide["calories"] = df_wide["calories"].map(
            lambda value: json.loads(value)["calories"] if pd.notna(value) else pd.NA
        )

    if "heart_rate" in df_wide.columns:
        hr_source = _load_json_series(df_wide["heart_rate"])
        hr_df = pd.json_normalize(hr_source)
        hr_df.index = hr_source.index
        hr_df = hr_df.drop(columns=[col for col in HEART_RATE_DROP_COLUMNS if col in hr_df.columns])
        hr_df = hr_df.rename(columns={"avg_rhr": "avg_rest_hr"})
        df_wide = df_wide.join(hr_df).drop(columns=["heart_rate"])

    if "sleep" in df_wide.columns:
        sleep_source = _load_json_series(df_wide["sleep"])
        sl_df = pd.json_normalize(sleep_source)
        sl_df.index = sleep_source.index

        def extract_bedtime(segment_details: object) -> pd.Series:
            if not isinstance(segment_details, list) or not segment_details:
                return pd.Series({"sleep_seg_bedtime": pd.NA})
            seg = pd.DataFrame(segment_details)
            bedtime = seg["bedtime"].min() if "bedtime" in seg.columns else pd.NA
            return pd.Series({"sleep_seg_bedtime": bedtime})

        if "segment_details" in sl_df.columns:
            seg_cols = sl_df["segment_details"].apply(extract_bedtime)
            seg_cols["sleep_seg_bedtime_dt"] = pd.to_datetime(
                seg_cols["sleep_seg_bedtime"], unit="s", utc=True
            ).dt.tz_convert(cfg.TIMEZONE)
            sl_df = sl_df.drop(columns=["segment_details"]).join(seg_cols[["sleep_seg_bedtime_dt"]])

        sl_df = sl_df.drop(columns=[col for col in SLEEP_DROP_COLUMNS if col in sl_df.columns])
        sl_df["wake_up_dt_calc"] = sl_df["sleep_seg_bedtime_dt"] + pd.to_timedelta(
            sl_df["total_duration"], unit="m"
        )
        sl_df = sl_df.rename(
            columns={
                "total_long_duration": "long_duration",
                "sleep_deep_duration": "deep_duration",
                "sleep_nap_duration": "nap_duration",
                "sleep_rem_duration": "rem_duration",
                "total_duration": "tot_duration",
                "sleep_score": "score",
                "sleep_awake_duration": "awake_duration",
                "sleep_light_duration": "light_duration",
                "long_sleep_evaluation": "long_score",
                "sleep_seg_bedtime_dt": "bedtime",
                "wake_up_dt_calc": "wakeup_time",
            }
        )
        sl_df["bedtime_mfm"] = (
            sl_df["bedtime"].dt.hour * 60
            + sl_df["bedtime"].dt.minute
            + sl_df["bedtime"].dt.second / 60
        )
        sl_df["wakeup_time_mfm"] = (
            sl_df["wakeup_time"].dt.hour * 60
            + sl_df["wakeup_time"].dt.minute
            + sl_df["wakeup_time"].dt.second / 60
        )
        sl_df["bedtime_mfm"] = sl_df["bedtime_mfm"].where(
            sl_df["bedtime_mfm"] <= 420,
            sl_df["bedtime_mfm"] - 24 * 60,
        )
        sl_df = sl_df.drop(columns=["bedtime", "wakeup_time"], errors="ignore").add_prefix("sl_")
        df_wide = df_wide.join(sl_df).drop(columns=["sleep"])

    if "steps" in df_wide.columns:
        step_source = _load_json_series(df_wide["steps"])
        step_df = pd.json_normalize(step_source)
        step_df.index = step_source.index
        if "steps" in step_df.columns:
            df_wide = df_wide.drop(columns=["steps"]).join(step_df["steps"])

    df_train = pd.read_csv(cfg.require_existing_path(training_path), usecols=["Time"])
    df_train["Time"] = pd.to_datetime(df_train["Time"], unit="s", utc=True).dt.tz_convert(cfg.TIMEZONE)
    if start is not None:
        df_train = df_train.loc[df_train["Time"] >= start].copy()
    if end is not None:
        df_train = df_train.loc[df_train["Time"] < end].copy()
    df_train["Date"] = df_train["Time"].dt.date
    df_train["training"] = 1
    df_train_day = df_train.groupby("Date", as_index=True)["training"].max().to_frame()

    df_wide = df_wide.join(df_train_day, how="left")
    df_wide["training"] = df_wide["training"].fillna(0).astype(int)
    df_wide.index = pd.to_datetime(df_wide.index).normalize()
    return df_wide.sort_index()


def save_xiaomi_data(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    output_path = output_path or cfg.XIAOMI_DATA_CLEANED
    cfg.ensure_processed_dir()
    df.to_parquet(output_path, engine="pyarrow", index=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Xiaomi daily aggregates.")
    parser.add_argument("--start-date", dest="start_date")
    parser.add_argument("--end-date", dest="end_date")
    parser.add_argument("--output", type=Path, default=cfg.XIAOMI_DATA_CLEANED)
    args = parser.parse_args()

    df = clean_xiaomi_data(
        start=_parse_bound(args.start_date),
        end=_parse_bound(args.end_date),
    )
    saved_path = save_xiaomi_data(df, args.output)
    print(f"Saved: {saved_path} | rows={len(df)} | columns={len(df.columns)}")


if __name__ == "__main__":
    main()
