from __future__ import annotations

"""
This script is the main command-line entry point of the project.

It does not contain the analytical logic itself. Instead, it orchestrates the
different stages of the workflow so the user can run the project step by step
or end to end from one place.

Available stages:
1. clean raw data and build the merged daily dataset,
2. create lagged features for modeling,
3. run regression and classification benchmarks,
4. execute the full pipeline in sequence.
"""

import argparse
import sys

import MergeData as merge_data
import MLapplications as ml
import Processing_and_descriptive as processing


def main() -> None:
    """CLI entry point that orchestrates the full project workflow."""
    parser = argparse.ArgumentParser(description="Run the glycemic analysis pipeline.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("clean", help="Rebuild cleaned and merged daily datasets.")

    features_parser = subparsers.add_parser("features", help="Build lagged features from the merged dataset.")
    features_parser.add_argument("--lag-days", type=int, default=2)
    features_parser.add_argument("--reuse-merged", action="store_true")

    train_parser = subparsers.add_parser("train", help="Run ML benchmarks on lagged features.")
    train_parser.add_argument("--lag-days", type=int, default=2)
    train_parser.add_argument("--holdout-days", type=int, default=14)
    train_parser.add_argument("--reuse-processed", action="store_true")
    train_parser.add_argument(
        "--task",
        choices=["all", "regression", "classification"],
        default="all",
    )

    all_parser = subparsers.add_parser("all", help="Run clean, feature engineering and training in sequence.")
    all_parser.add_argument("--lag-days", type=int, default=2)
    all_parser.add_argument("--holdout-days", type=int, default=14)
    all_parser.add_argument(
        "--task",
        choices=["all", "regression", "classification"],
        default="all",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        print(
            "\nExamples:\n"
            "  python pipeline.py clean\n"
            "  python pipeline.py features --lag-days 2\n"
            "  python pipeline.py train --task all --holdout-days 14\n"
            "  python pipeline.py all"
        )
        return

    args = parser.parse_args()

    # ==================================================
    # 1. Cleaning and merge only
    # ==================================================

    if args.command == "clean":
        merged, _, _ = merge_data.build_cleaned_dataset(save_intermediate=True)
        print(f"Built merged dataset | rows={len(merged)} | columns={len(merged.columns)}")
        return

    # ==================================================
    # 2. Feature engineering only
    # ==================================================

    if args.command == "features":
        lagged = processing.prepare_processed_dataset(
            rebuild_merged=not args.reuse_merged,
            save=True,
            lag_days=args.lag_days,
        )
        print(f"Built lagged dataset | rows={len(lagged)} | columns={len(lagged.columns)}")
        return

    # ==================================================
    # 3. Modeling only
    # ==================================================

    if args.command == "train":
        results = ml.run_modeling(
            rebuild_processed=not args.reuse_processed,
            lag_days=args.lag_days,
            holdout_days=args.holdout_days,
            task=args.task,
        )
        print(ml.format_results(results))
        print(f"\nSaved CSV: {ml.cfg.ML_RESULTS}")
        return

    # ==================================================
    # 4. Full pipeline: clean -> features -> modeling
    # ==================================================

    merge_data.build_cleaned_dataset(save_intermediate=True)
    processing.prepare_processed_dataset(rebuild_merged=False, save=True, lag_days=args.lag_days)
    results = ml.run_modeling(
        rebuild_processed=False,
        lag_days=args.lag_days,
        holdout_days=args.holdout_days,
        task=args.task,
    )
    print(ml.format_results(results))
    print(f"\nSaved CSV: {ml.cfg.ML_RESULTS}")


if __name__ == "__main__":
    main()
