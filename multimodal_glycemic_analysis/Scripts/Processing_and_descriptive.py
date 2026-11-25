from __future__ import annotations

"""
Feature engineering and descriptive helpers for the merged daily dataset.

The workflow is:
1. load or rebuild the merged daily dataset,
2. optionally detect and restrict to the contiguous multimodal overlap window
   (avoiding multi-week wearable recording gaps that would otherwise drop wearable features),
3. build leak-free lagged predictors, rolling window statistics (computed strictly
   on past days via shift(1)), short-vs-baseline deltas, sleep missingness indicators,
   and cyclical calendar features,
4. save the processed modeling table,
5. optionally display quick exploratory boxplots and correlation heatmaps.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as cfg
import MergeData as merge_data


def load_merged_data(path: Path | None = None) -> pd.DataFrame:
    """Load the merged daily dataset from disk."""
    return pd.read_parquet(cfg.require_existing_path(path or cfg.DATA_CLEANED)).sort_index()


def detect_multimodal_start(
    df: pd.DataFrame,
    *,
    wearable_col: str = "steps",
    gap_threshold_days: int = cfg.WEARABLE_GAP_THRESHOLD_DAYS,
) -> pd.Timestamp:
    """Find the start date of the latest contiguous multimodal recording block.

    If early isolated wearable days are separated from the main continuous recording
    period by a gap longer than `gap_threshold_days`, this returns the first valid
    wearable date after the last long outage.
    """
    ordered = df.sort_index()
    if wearable_col not in ordered.columns:
        return pd.Timestamp(ordered.index.min())

    valid_dates = ordered.index[ordered[wearable_col].notna()]
    if len(valid_dates) <= 1:
        return pd.Timestamp(ordered.index.min())

    gaps = valid_dates.to_series().diff().dt.days.fillna(0)
    long_gaps = gaps[gaps > gap_threshold_days]
    if long_gaps.empty:
        return pd.Timestamp(valid_dates[0])

    return pd.Timestamp(long_gaps.index[-1])


def boxplot(df: pd.DataFrame) -> None:
    """Show simple boxplots for the numeric columns."""
    step = 6
    numeric_df = df.select_dtypes("number")
    for start in range(0, numeric_df.shape[1], step):
        subset = numeric_df.iloc[:, start : start + step]
        fig, axes = plt.subplots(1, subset.shape[1], figsize=(3 * subset.shape[1], 4), sharey=False)
        if subset.shape[1] == 1:
            axes = [axes]

        for ax, col in zip(axes, subset.columns):
            ax.boxplot(subset[col].dropna().values)
            ax.set_title(col, fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()


def corrplot(df: pd.DataFrame, threshold: float) -> None:
    """Show a Spearman correlation heatmap with optional annotation threshold."""
    corr = df.select_dtypes("number").corr(method="spearman")
    annot = corr.round(2).astype(str)
    annot[np.abs(corr) < threshold] = ""

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=annot, fmt="", annot_kws={"size": 6}, vmin=-1, vmax=1, center=0)
    plt.tight_layout()
    plt.show()


def build_lagged_dataset(
    df: pd.DataFrame,
    *,
    lag_days: int = cfg.DEFAULT_LAG_DAYS,
    lag_columns: list[str] | None = None,
    rolling_columns: list[str] | None = None,
    rolling_windows: tuple[int, ...] = cfg.ROLLING_WINDOWS,
    include_calendar_features: bool = True,
    include_day_index: bool = False,
    restrict_multimodal: bool = False,
) -> pd.DataFrame:
    """Create leak-free lagged, rolling, delta, and calendar predictors.

    Every lagged and rolling feature at date `t` depends strictly on observations
    at dates `<= t - 1`.
    """
    if lag_days < 1:
        raise ValueError("lag_days must be at least 1.")

    out = df.copy().sort_index()
    if restrict_multimodal:
        mm_start = detect_multimodal_start(out)
        out = out.loc[out.index >= mm_start].copy()

    columns = lag_columns or [col for col in cfg.DEFAULT_LAG_COLUMNS if col in out.columns]
    if not columns:
        raise ValueError("No valid columns available for lag feature generation.")

    missing = sorted(set(columns) - set(out.columns))
    if missing:
        raise KeyError(f"Missing columns required for lagging: {missing}")

    new_features: dict[str, pd.Series] = {}

    if include_day_index and "day" not in out.columns:
        new_features["day"] = pd.Series(range(len(out)), index=out.index, dtype=float)

    if include_calendar_features and isinstance(out.index, pd.DatetimeIndex):
        dow = out.index.dayofweek.to_numpy(dtype=float)
        new_features["cal_dow_sin"] = pd.Series(np.sin(2 * np.pi * dow / 7.0), index=out.index)
        new_features["cal_dow_cos"] = pd.Series(np.cos(2 * np.pi * dow / 7.0), index=out.index)
        new_features["cal_is_weekend"] = pd.Series((dow >= 5).astype(float), index=out.index)

    # 1. Explicit daily lags
    for lag in range(1, lag_days + 1):
        for col in columns:
            new_features[f"{col}_lag{lag}"] = out[col].shift(lag)

    # 2. Leak-free rolling statistics computed on shifted (t - 1) history
    roll_cols = (
        rolling_columns
        if rolling_columns is not None
        else [col for col in cfg.ROLLING_COLUMNS if col in out.columns]
    )
    for col in roll_cols:
        past = out[col].shift(1)
        for window in rolling_windows:
            min_periods = max(2, window // 2)
            roll_mean = past.rolling(window=window, min_periods=min_periods).mean()
            new_features[f"{col}_roll{window}_mean"] = roll_mean
            if window >= 7:
                new_features[f"{col}_roll{window}_std"] = past.rolling(
                    window=window, min_periods=min_periods
                ).std()
                new_features[f"{col}_delta_7d"] = past - roll_mean

    # 3. Wearable sleep availability flag from previous night
    if "sl_tot_duration" in out.columns:
        new_features["sl_missing_lag1"] = out["sl_tot_duration"].shift(1).isna().astype(float)

    if new_features:
        out = pd.concat([out, pd.DataFrame(new_features, index=out.index)], axis=1)

    return out


def save_lagged_dataset(
    df: pd.DataFrame,
    parquet_path: Path | None = None,
    csv_path: Path | None = None,
) -> tuple[Path, Path]:
    """Persist the lagged dataset used by the modeling stage."""
    parquet_path = parquet_path or cfg.DATA_PROCESSED
    csv_path = csv_path or cfg.DATA_PROCESSED_CSV
    cfg.ensure_processed_dir()
    df.to_parquet(parquet_path, engine="pyarrow", index=True)
    df.to_csv(csv_path)

    # Keep legacy artifact synchronized for backward compatibility.
    if parquet_path != cfg.DATA_LAGGED:
        df.to_parquet(cfg.DATA_LAGGED, engine="pyarrow", index=True)

    return parquet_path, csv_path


def prepare_processed_dataset(
    *,
    rebuild_merged: bool = True,
    save: bool = True,
    lag_days: int = cfg.DEFAULT_LAG_DAYS,
    restrict_multimodal: bool = True,
) -> pd.DataFrame:
    """Build or reload the merged dataset and then create modeling features."""
    if rebuild_merged:
        merged, _, _ = merge_data.build_cleaned_dataset(save_intermediate=save)
    else:
        merged = load_merged_data()

    lagged = build_lagged_dataset(
        merged,
        lag_days=lag_days,
        restrict_multimodal=restrict_multimodal,
    )
    if save:
        save_lagged_dataset(lagged)
    return lagged


def main() -> None:
    """CLI entry point used by the project pipeline."""
    parser = argparse.ArgumentParser(description="Build lagged/rolling features and optional descriptive plots.")
    parser.add_argument(
        "--no-rebuild-merged",
        action="store_true",
        help="Reuse data_cleaned.parquet instead of rebuilding the merged dataset first.",
    )
    parser.add_argument("--lag-days", type=int, default=cfg.DEFAULT_LAG_DAYS)
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="Keep full merged date range instead of restricting to the contiguous multimodal window.",
    )
    parser.add_argument("--show-boxplot", action="store_true")
    parser.add_argument("--show-corrplot", action="store_true")
    parser.add_argument("--correlation-threshold", type=float, default=0.2)
    args = parser.parse_args()

    lagged = prepare_processed_dataset(
        rebuild_merged=not args.no_rebuild_merged,
        save=True,
        lag_days=args.lag_days,
        restrict_multimodal=not args.full_range,
    )

    base_df = load_merged_data()
    if args.show_boxplot:
        boxplot(base_df)
    if args.show_corrplot:
        corrplot(lagged, args.correlation_threshold)

    print(
        f"Saved: {cfg.DATA_PROCESSED} | rows={len(lagged)} | columns={len(lagged.columns)} "
        f"| range={lagged.index.min().date()}..{lagged.index.max().date()}"
    )


if __name__ == "__main__":
    main()
