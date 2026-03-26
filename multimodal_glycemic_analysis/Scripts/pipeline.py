from __future__ import annotations

import argparse
import sys

import MergeData as merge_data
import MLapplications as ml
import Processing_and_descriptive as processing


def main() -> None:
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

    if args.command == "clean":
        merged, _, _ = merge_data.build_cleaned_dataset(save_intermediate=True)
        print(f"Built merged dataset | rows={len(merged)} | columns={len(merged.columns)}")
        return

    if args.command == "features":
        lagged = processing.prepare_processed_dataset(
            rebuild_merged=not args.reuse_merged,
            save=True,
            lag_days=args.lag_days,
        )
        print(f"Built lagged dataset | rows={len(lagged)} | columns={len(lagged.columns)}")
        return

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
