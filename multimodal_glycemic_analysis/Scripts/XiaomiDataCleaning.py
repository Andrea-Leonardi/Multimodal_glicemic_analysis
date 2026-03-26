from __future__ import annotations

"""
This script converts the raw Xiaomi wearable export into a clean daily table.

Its role in the project is to transform heterogeneous activity and sleep data
into structured predictors that can later be merged with glycemic information.

The workflow is:
1. load raw Xiaomi fitness and training exports,
2. clean timestamps and remove non-informative records,
3. reshape the daily export from long format to wide format,
4. unpack JSON-like payloads for calories, heart rate, sleep, and steps,
5. build a daily training indicator,
6. save the result as a daily parquet dataset.

This file is meant to be readable as an ETL step inside the broader pipeline.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import config as cfg

# ==================================================
# Static cleaning rules
# ==================================================

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


def parse_bound(value: str | None) -> pd.Timestamp | None:
    """Convert an optional CLI date to a timezone-aware timestamp."""
    if value is None:
        return None

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(cfg.TIMEZONE)
    return ts.tz_convert(cfg.TIMEZONE)


def clean_xiaomi_data(
    fitness_path: Path | None = None,
    training_path: Path | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a clean daily Xiaomi dataset from the raw exports."""

    fitness_path = fitness_path or (cfg.XIAOMI_DIR / cfg.XIAOMI_FITNESS_FILE)
    training_path = training_path or (cfg.XIAOMI_DIR / cfg.XIAOMI_TRAINING_FILE)

    # ==================================================
    # 1. Load raw Xiaomi files
    # ==================================================

    df_x = pd.read_csv(cfg.require_existing_path(fitness_path))
    df_x["Time"] = pd.to_datetime(df_x["Time"], unit="s", utc=True).dt.tz_convert(cfg.TIMEZONE)

    df_train = pd.read_csv(cfg.require_existing_path(training_path), usecols=["Time"])
    df_train["Time"] = pd.to_datetime(df_train["Time"], unit="s", utc=True).dt.tz_convert(cfg.TIMEZONE)

    # ==================================================
    # 2. Remove unwanted rows and restrict the time window
    # ==================================================

    df_x = df_x.loc[~df_x["Key"].isin(EXCLUDED_KEYS)].copy()
    df_x = df_x.loc[df_x["Tag"] != "daily_mark"].copy()
    df_x = df_x.drop(columns=["Uid", "Sid", "Tag", "UpdateTime"], errors="ignore")

    if start is not None:
        df_x = df_x.loc[df_x["Time"] >= start].copy()
        df_train = df_train.loc[df_train["Time"] >= start].copy()
    if end is not None:
        df_x = df_x.loc[df_x["Time"] < end].copy()
        df_train = df_train.loc[df_train["Time"] < end].copy()

    # ==================================================
    # 3. Convert the long daily export into a wide table
    # ==================================================

    df_x["Date"] = df_x["Time"].dt.date
    duplicates = df_x.groupby(["Date", "Key"]).size()
    duplicated_keys = duplicates.loc[duplicates > 1]
    if not duplicated_keys.empty:
        sample = duplicated_keys.head(5).to_dict()
        raise ValueError(f"Found duplicate Xiaomi daily aggregates: {sample}")

    df_wide = df_x.pivot(index="Date", columns="Key", values="Value").sort_index()

    # ==================================================
    # 4. Expand the calories payload
    # ==================================================

    if "calories" in df_wide.columns:
        df_wide["calories"] = df_wide["calories"].map(
            lambda value: json.loads(value)["calories"] if pd.notna(value) else pd.NA
        )

    # ==================================================
    # 5. Expand the heart-rate payload
    # ==================================================

    if "heart_rate" in df_wide.columns:
        hr_source = df_wide["heart_rate"].dropna().map(json.loads)
        hr_df = pd.json_normalize(hr_source)
        hr_df.index = hr_source.index
        hr_df = hr_df.drop(columns=[col for col in HEART_RATE_DROP_COLUMNS if col in hr_df.columns])
        hr_df = hr_df.rename(columns={"avg_rhr": "avg_rest_hr"})
        df_wide = df_wide.join(hr_df).drop(columns=["heart_rate"])

    # ==================================================
    # 6. Expand the sleep payload
    # ==================================================

    if "sleep" in df_wide.columns:
        sleep_source = df_wide["sleep"].dropna().map(json.loads)
        sl_df = pd.json_normalize(sleep_source)
        sl_df.index = sleep_source.index

        if "segment_details" in sl_df.columns:
            bedtime_rows: list[dict[str, object]] = []
            for row_date, segment_details in sl_df["segment_details"].items():
                bedtime = pd.NA
                if isinstance(segment_details, list) and segment_details:
                    seg = pd.DataFrame(segment_details)
                    if "bedtime" in seg.columns:
                        bedtime = seg["bedtime"].min()
                bedtime_rows.append({"Date": row_date, "sleep_seg_bedtime": bedtime})

            seg_cols = pd.DataFrame(bedtime_rows).set_index("Date")
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

        # Bedtime is moved to the previous day when it occurs before 7:00 AM.
        sl_df["bedtime_mfm"] = sl_df["bedtime_mfm"].where(
            sl_df["bedtime_mfm"] <= 420,
            sl_df["bedtime_mfm"] - 24 * 60,
        )

        sl_df = sl_df.drop(columns=["bedtime", "wakeup_time"], errors="ignore").add_prefix("sl_")
        df_wide = df_wide.join(sl_df).drop(columns=["sleep"])

    # ==================================================
    # 7. Expand the steps payload
    # ==================================================

    if "steps" in df_wide.columns:
        step_source = df_wide["steps"].dropna().map(json.loads)
        step_df = pd.json_normalize(step_source)
        step_df.index = step_source.index
        if "steps" in step_df.columns:
            df_wide = df_wide.drop(columns=["steps"]).join(step_df["steps"])

    # ==================================================
    # 8. Add the daily training indicator
    # ==================================================

    df_train["Date"] = df_train["Time"].dt.date
    df_train["training"] = 1
    df_train_day = df_train.groupby("Date", as_index=True)["training"].max().to_frame()

    df_wide = df_wide.join(df_train_day, how="left")
    df_wide["training"] = df_wide["training"].fillna(0).astype(int)

    # ==================================================
    # 9. Final index normalization
    # ==================================================

    df_wide.index = pd.to_datetime(df_wide.index).normalize()
    return df_wide.sort_index()


def save_xiaomi_data(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Save the cleaned Xiaomi dataset."""
    output_path = output_path or cfg.XIAOMI_DATA_CLEANED
    cfg.ensure_processed_dir()
    df.to_parquet(output_path, engine="pyarrow", index=True)
    return output_path


def main() -> None:
    """CLI entry point used by the project pipeline."""
    parser = argparse.ArgumentParser(description="Clean Xiaomi daily aggregates.")
    parser.add_argument("--start-date", dest="start_date")
    parser.add_argument("--end-date", dest="end_date")
    parser.add_argument("--output", type=Path, default=cfg.XIAOMI_DATA_CLEANED)
    args = parser.parse_args()

    df = clean_xiaomi_data(
        start=parse_bound(args.start_date),
        end=parse_bound(args.end_date),
    )
    saved_path = save_xiaomi_data(df, args.output)
    print(f"Saved: {saved_path} | rows={len(df)} | columns={len(df.columns)}")


if __name__ == "__main__":
    main()
