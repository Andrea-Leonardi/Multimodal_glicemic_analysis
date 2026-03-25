from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as cfg
import MergeData as merge_data


def load_merged_data(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(cfg.require_existing_path(path or cfg.DATA_CLEANED)).sort_index()


def boxplot(df: pd.DataFrame) -> None:
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
    include_day_index: bool = True,
) -> pd.DataFrame:
    if lag_days < 1:
        raise ValueError("lag_days must be at least 1.")

    out = df.copy().sort_index()
    columns = lag_columns or [col for col in cfg.DEFAULT_LAG_COLUMNS if col in out.columns]
    if not columns:
        raise ValueError("No valid columns available for lag feature generation.")

    missing = sorted(set(columns) - set(out.columns))
    if missing:
        raise KeyError(f"Missing columns required for lagging: {missing}")

    if include_day_index and "day" not in out.columns:
        out["day"] = range(len(out))

    for lag in range(1, lag_days + 1):
        for col in columns:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)

    return out


def save_lagged_dataset(
    df: pd.DataFrame,
    parquet_path: Path | None = None,
    csv_path: Path | None = None,
) -> tuple[Path, Path]:
    parquet_path = parquet_path or cfg.DATA_PROCESSED
    csv_path = csv_path or cfg.DATA_PROCESSED_CSV
    cfg.ensure_processed_dir()
    df.to_parquet(parquet_path, engine="pyarrow", index=True)
    df.to_csv(csv_path)

    # Legacy artifact kept in sync for backward compatibility.
    if parquet_path != cfg.DATA_LAGGED:
        df.to_parquet(cfg.DATA_LAGGED, engine="pyarrow", index=True)

    return parquet_path, csv_path


def prepare_processed_dataset(
    *,
    rebuild_merged: bool = True,
    save: bool = True,
    lag_days: int = cfg.DEFAULT_LAG_DAYS,
) -> pd.DataFrame:
    if rebuild_merged:
        merged, _, _ = merge_data.build_cleaned_dataset(save_intermediate=save)
    else:
        merged = load_merged_data()

    lagged = build_lagged_dataset(merged, lag_days=lag_days)
    if save:
        save_lagged_dataset(lagged)
    return lagged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lagged features and optional descriptive plots.")
    parser.add_argument(
        "--no-rebuild-merged",
        action="store_true",
        help="Reuse data_cleaned.parquet instead of rebuilding the merged dataset first.",
    )
    parser.add_argument("--lag-days", type=int, default=cfg.DEFAULT_LAG_DAYS)
    parser.add_argument("--show-boxplot", action="store_true")
    parser.add_argument("--show-corrplot", action="store_true")
    parser.add_argument("--correlation-threshold", type=float, default=0.2)
    args = parser.parse_args()

    lagged = prepare_processed_dataset(
        rebuild_merged=not args.no_rebuild_merged,
        save=True,
        lag_days=args.lag_days,
    )

    base_df = load_merged_data()
    if args.show_boxplot:
        boxplot(base_df)
    if args.show_corrplot:
        corrplot(lagged, args.correlation_threshold)

    print(f"Saved: {cfg.DATA_PROCESSED} | rows={len(lagged)} | columns={len(lagged.columns)}")


if __name__ == "__main__":
    main()
