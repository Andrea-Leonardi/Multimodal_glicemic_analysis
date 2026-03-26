from __future__ import annotations

"""
This script merges the cleaned Xiaomi and Glooko daily datasets.

Its purpose is to create the main analytical table used by the rest of the
project. The merge happens after both data sources have been cleaned and
normalized to the same daily time index.

The workflow is:
1. build or load the cleaned Xiaomi dataset,
2. build or load the cleaned Glooko dataset,
3. align both datasets on a daily index,
4. join them into a single multimodal daily table,
5. save the merged dataset for feature engineering and modeling.
"""

import argparse
from pathlib import Path

import pandas as pd

import config as cfg
import GlookoDataCleaning as glooko_cleaning
import XiaomiDataCleaning as xiaomi_cleaning


def load_df(parquet_path: Path) -> pd.DataFrame:
    """Load an existing parquet artifact."""
    return pd.read_parquet(cfg.require_existing_path(parquet_path))


def normalize_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """Force every index value to midnight so daily joins are stable."""
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce").normalize()
    out = out.loc[~out.index.isna()].sort_index()
    return out


def build_cleaned_dataset(save_intermediate: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning workflow and return merged plus source tables."""

    # ==================================================
    # 1. Build the Xiaomi daily dataset
    # ==================================================

    xiaomi = normalize_daily_index(xiaomi_cleaning.clean_xiaomi_data())
    if xiaomi.empty:
        raise ValueError("Xiaomi cleaning produced an empty dataset.")

    # ==================================================
    # 2. Use the Xiaomi window to trim the Glooko export
    # ==================================================

    window_start = xiaomi.index.min()
    window_end = xiaomi.index.max() + pd.Timedelta(days=1)
    glooko = normalize_daily_index(
        glooko_cleaning.clean_glooko_data(start=window_start, end=window_end)
    )
    if glooko.empty:
        raise ValueError("Glooko cleaning produced an empty dataset.")

    # ==================================================
    # 3. Join both sources on the normalized daily index
    # ==================================================

    merged = glooko.join(xiaomi, how="outer", lsuffix="_glooko", rsuffix="_xiaomi").sort_index()

    # ==================================================
    # 4. Persist artifacts if requested
    # ==================================================

    if save_intermediate:
        xiaomi_cleaning.save_xiaomi_data(xiaomi)
        glooko_cleaning.save_glooko_data(glooko)
        save_merged_data(merged)

    return merged, glooko, xiaomi


def save_merged_data(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Save the merged daily dataset."""
    output_path = output_path or cfg.DATA_CLEANED
    cfg.ensure_processed_dir()
    df.to_parquet(output_path, engine="pyarrow", index=True)
    return output_path


def main() -> None:
    """CLI entry point used by the project pipeline."""
    parser = argparse.ArgumentParser(description="Build the merged daily dataset.")
    parser.add_argument(
        "--from-existing",
        action="store_true",
        help="Merge existing cleaned parquet files instead of rebuilding from raw data.",
    )
    parser.add_argument("--output", type=Path, default=cfg.DATA_CLEANED)
    args = parser.parse_args()

    if args.from_existing:
        glooko = normalize_daily_index(load_df(cfg.GLOOKO_DATA_CLEANED))
        xiaomi = normalize_daily_index(load_df(cfg.XIAOMI_DATA_CLEANED))
        merged = glooko.join(xiaomi, how="outer", lsuffix="_glooko", rsuffix="_xiaomi").sort_index()
        saved_path = save_merged_data(merged, args.output)
    else:
        merged, _, _ = build_cleaned_dataset(save_intermediate=True)
        saved_path = args.output
        if args.output != cfg.DATA_CLEANED:
            saved_path = save_merged_data(merged, args.output)

    print(f"Saved: {saved_path} | rows={len(merged)} | columns={len(merged.columns)}")


if __name__ == "__main__":
    main()
