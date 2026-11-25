from __future__ import annotations

"""
Main command-line entry point for the Multimodal Glycemic Analysis pipeline.

Available stages:
1. clean    - parse raw Glooko and Xiaomi exports and build the merged daily table,
2. features - restrict to the contiguous multimodal window and build lagged/rolling features,
3. train    - benchmark baselines + ML models, run Glooko-only vs multimodal ablation, and compute importance,
4. report   - generate publication-ready PNG figures in `reports/figures/`,
5. all      - execute the full workflow end to end.
"""

import argparse
import sys

import MergeData as merge_data
import MLapplications as ml
import Processing_and_descriptive as processing
import Visualizations as viz
import config as cfg


def main() -> None:
    """CLI entry point that orchestrates the full project workflow."""
    parser = argparse.ArgumentParser(description="Run the multimodal glycemic analysis pipeline.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("clean", help="Rebuild cleaned and merged daily datasets.")

    features_parser = subparsers.add_parser("features", help="Build lagged and rolling features.")
    features_parser.add_argument("--lag-days", type=int, default=cfg.DEFAULT_LAG_DAYS)
    features_parser.add_argument("--reuse-merged", action="store_true")
    features_parser.add_argument("--full-range", action="store_true")

    train_parser = subparsers.add_parser("train", help="Run ML benchmarks, ablation, and feature importance.")
    train_parser.add_argument("--lag-days", type=int, default=cfg.DEFAULT_LAG_DAYS)
    train_parser.add_argument("--holdout-days", type=int, default=cfg.DEFAULT_HOLDOUT_DAYS)
    train_parser.add_argument("--reuse-processed", action="store_true")
    train_parser.add_argument("--no-ablation", action="store_true")
    train_parser.add_argument(
        "--task",
        choices=["all", "regression", "classification"],
        default="all",
    )

    report_parser = subparsers.add_parser("report", help="Generate PNG figures in reports/figures/.")
    report_parser.add_argument("--lag-days", type=int, default=cfg.DEFAULT_LAG_DAYS)
    report_parser.add_argument("--holdout-days", type=int, default=cfg.DEFAULT_HOLDOUT_DAYS)

    all_parser = subparsers.add_parser("all", help="Run clean, features, train, and report in sequence.")
    all_parser.add_argument("--lag-days", type=int, default=cfg.DEFAULT_LAG_DAYS)
    all_parser.add_argument("--holdout-days", type=int, default=cfg.DEFAULT_HOLDOUT_DAYS)
    all_parser.add_argument(
        "--task",
        choices=["all", "regression", "classification"],
        default="all",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        print(
            "\nExamples:\n"
            "  python multimodal_glycemic_analysis/Scripts/pipeline.py clean\n"
            "  python multimodal_glycemic_analysis/Scripts/pipeline.py features --lag-days 2\n"
            "  python multimodal_glycemic_analysis/Scripts/pipeline.py train --task all --holdout-days 28\n"
            "  python multimodal_glycemic_analysis/Scripts/pipeline.py report\n"
            "  python multimodal_glycemic_analysis/Scripts/pipeline.py all"
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
            restrict_multimodal=not args.full_range,
        )
        print(
            f"Built feature dataset | rows={len(lagged)} | columns={len(lagged.columns)} "
            f"| range={lagged.index.min().date()}..{lagged.index.max().date()}"
        )
        return

    if args.command == "train":
        results = ml.run_modeling(
            rebuild_processed=not args.reuse_processed,
            lag_days=args.lag_days,
            holdout_days=args.holdout_days,
            task=args.task,
            run_ablation=not args.no_ablation,
        )
        print(ml.format_results(results))
        print(f"\nSaved CSV: {cfg.ML_RESULTS}")
        return

    if args.command == "report":
        paths = viz.generate_all_figures(lag_days=args.lag_days, holdout_days=args.holdout_days)
        for path in paths:
            print(f"Saved figure: {path}")
        return

    # Full pipeline: clean -> features -> train -> report
    merge_data.build_cleaned_dataset(save_intermediate=True)
    processing.prepare_processed_dataset(
        rebuild_merged=False,
        save=True,
        lag_days=args.lag_days,
        restrict_multimodal=True,
    )
    results = ml.run_modeling(
        rebuild_processed=False,
        lag_days=args.lag_days,
        holdout_days=args.holdout_days,
        task=args.task,
        run_ablation=True,
    )
    print(ml.format_results(results))
    print(f"\nSaved CSV: {cfg.ML_RESULTS}")
    paths = viz.generate_all_figures(lag_days=args.lag_days, holdout_days=args.holdout_days)
    for path in paths:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
