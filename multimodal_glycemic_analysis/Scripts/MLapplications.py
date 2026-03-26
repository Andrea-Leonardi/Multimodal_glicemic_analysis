from __future__ import annotations

"""
This script contains the modeling stage of the project.

The goal is not to build a production-ready medical predictor, but to compare
several reasonable statistical and machine learning models on the same daily
dataset in a fair and readable way.

The workflow is:
1. load the lagged daily dataset created by the preprocessing pipeline,
2. build a feature matrix X and a target vector y,
3. split the data chronologically into train and holdout sets,
4. benchmark multiple regression and classification models with TimeSeriesSplit,
5. print the results in a readable terminal report and save them to CSV.

The focus of this file is transparency:
- same input data for all models,
- same validation logic for all models,
- simple printed summaries that are easy to discuss in a portfolio or interview.
"""

import argparse
import json
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message="'penalty' was deprecated in version 1.8",
    category=FutureWarning,
    module="sklearn.linear_model._logistic",
)

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import Processing_and_descriptive as processing
import config as cfg

# ==================================================
# Global modeling settings
# ==================================================

DEFAULT_CLASSIFICATION_QUANTILES = (0.35, 0.50, 0.65)
GRID_N_JOBS = 1
RANDOM_STATE = 42


def load_processed_data(rebuild_processed: bool = True, lag_days: int = cfg.DEFAULT_LAG_DAYS) -> pd.DataFrame:
    """Load the processed lagged dataset, or rebuild it from the pipeline."""

    if rebuild_processed:
        return processing.prepare_processed_dataset(
            rebuild_merged=True,
            save=True,
            lag_days=lag_days,
        )

    return pd.read_parquet(cfg.require_existing_path(cfg.DATA_PROCESSED)).sort_index()


def prepare_training_data(
    df: pd.DataFrame,
    *,
    target_col: str = cfg.DEFAULT_TARGET_COLUMN,
    lag_days: int = cfg.DEFAULT_LAG_DAYS,
    min_availability: float = cfg.MIN_FEATURE_AVAILABILITY,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create the feature matrix X and the target vector y."""

    # ==================================================
    # 1. Keep only lagged predictors
    # ==================================================

    lagged_columns = []
    for col in df.columns:
        if any(col.endswith(f"_lag{lag}") for lag in range(1, lag_days + 1)):
            if df[col].notna().mean() >= min_availability:
                lagged_columns.append(col)

    lagged_columns = sorted(lagged_columns)
    if "day" in df.columns:
        lagged_columns = ["day", *lagged_columns]

    if not lagged_columns:
        raise ValueError("No lagged feature columns were found in the processed dataset.")

    # ==================================================
    # 2. Drop the first lagged days and missing targets
    # ==================================================

    modeling_df = df.sort_index().iloc[lag_days:].copy()
    modeling_df = modeling_df.loc[modeling_df[target_col].notna()].copy()

    X = modeling_df[lagged_columns]
    y = modeling_df[target_col]
    return X, y


def split_train_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    holdout_days: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a chronological train/holdout split."""

    if holdout_days < 1:
        raise ValueError("holdout_days must be at least 1.")
    if len(X) <= holdout_days + 10:
        raise ValueError("Not enough samples to create a meaningful holdout split.")

    X_train = X.iloc[:-holdout_days]
    X_holdout = X.iloc[-holdout_days:]
    y_train = y.iloc[:-holdout_days]
    y_holdout = y.iloc[-holdout_days:]
    return X_train, X_holdout, y_train, y_holdout


def benchmark_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
) -> pd.DataFrame:
    """Run the regression benchmark."""

    # ==================================================
    # 1. Define time-aware cross-validation
    # ==================================================

    n_splits = min(5, max(2, len(X_train) // 20))
    if len(X_train) <= n_splits:
        raise ValueError("Not enough samples for TimeSeriesSplit.")
    cv = TimeSeriesSplit(n_splits=n_splits)

    # ==================================================
    # 2. Define the regression models to compare
    # ==================================================

    max_components = max(2, min(10, X_train.shape[1]))
    model_specs = [
        {
            "name": "lasso",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Lasso(max_iter=200000, tol=1e-4, random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                "model__alpha": np.logspace(-3, 2, 20),
            },
        },
        {
            "name": "elastic_net",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        ElasticNet(max_iter=200000, tol=1e-4, random_state=RANDOM_STATE),
                    ),
                ]
            ),
            "param_grid": {
                "model__alpha": np.logspace(-3, 2, 16),
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        },
        {
            "name": "pcr_ridge",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("pca", PCA()),
                    ("model", Ridge()),
                ]
            ),
            "param_grid": {
                "pca__n_components": list(range(2, max_components + 1)),
                "model__alpha": np.logspace(-2, 4, 12),
            },
        },
        {
            "name": "pls",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", PLSRegression()),
                ]
            ),
            "param_grid": {
                "model__n_components": list(range(1, min(8, X_train.shape[1]) + 1)),
            },
        },
        {
            "name": "hist_gb",
            "estimator": Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            early_stopping=False,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3, None],
                "model__max_leaf_nodes": [7, 15, 31],
                "model__min_samples_leaf": [5, 10, 20],
            },
        },
    ]

    # ==================================================
    # 3. Fit each model and store the results
    # ==================================================

    rows: list[dict] = []

    for spec in model_specs:
        search = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            cv=cv,
            scoring="r2",
            n_jobs=GRID_N_JOBS,
            error_score=np.nan,
        )
        search.fit(X_train, y_train)

        rows.append(
            {
                "task": "regression",
                "model": spec["name"],
                "quantile": np.nan,
                "metric": "r2",
                "cv_score": float(search.best_score_),
                "holdout_score": float(search.best_estimator_.score(X_holdout, y_holdout)),
                "best_params": json.dumps(search.best_params_, sort_keys=True, default=str),
                "n_train": len(X_train),
                "n_holdout": len(X_holdout),
            }
        )

    return pd.DataFrame(rows).sort_values(["cv_score", "holdout_score"], ascending=False)


def benchmark_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    *,
    quantiles: tuple[float, ...] = DEFAULT_CLASSIFICATION_QUANTILES,
) -> pd.DataFrame:
    """Run the classification benchmark at different target thresholds."""

    # ==================================================
    # 1. Define time-aware cross-validation
    # ==================================================

    n_splits = min(5, max(2, len(X_train) // 20))
    if len(X_train) <= n_splits:
        raise ValueError("Not enough samples for TimeSeriesSplit.")
    cv = TimeSeriesSplit(n_splits=n_splits)

    # ==================================================
    # 2. Define the classification models to compare
    # ==================================================

    model_specs = [
        {
            "name": "logistic",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=10000,
                            class_weight="balanced",
                            solver="liblinear",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__C": np.logspace(-2, 2, 10),
            },
        },
        {
            "name": "logistic_elasticnet",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=50000,
                            class_weight="balanced",
                            solver="saga",
                            penalty="elasticnet",
                            tol=1e-3,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__C": np.logspace(-2, 2, 8),
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        },
        {
            "name": "naive_bayes",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", GaussianNB()),
                ]
            ),
            "param_grid": {},
        },
        {
            "name": "lda",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LinearDiscriminantAnalysis()),
                ]
            ),
            "param_grid": {},
        },
        {
            "name": "svm_linear",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="linear", class_weight="balanced")),
                ]
            ),
            "param_grid": {
                "model__C": np.logspace(-2, 2, 10),
            },
        },
        {
            "name": "svm_rbf",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="rbf", class_weight="balanced")),
                ]
            ),
            "param_grid": {
                "model__C": np.logspace(-1, 2, 8),
                "model__gamma": np.logspace(-4, -1, 8),
            },
        },
        {
            "name": "hist_gb",
            "estimator": Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            early_stopping=False,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3, None],
                "model__max_leaf_nodes": [7, 15, 31],
                "model__min_samples_leaf": [5, 10, 20],
            },
        },
    ]

    # ==================================================
    # 3. Loop over the target thresholds
    # ==================================================

    rows: list[dict] = []

    for quantile in quantiles:
        # The binary target is created using the training set only.
        threshold = float(y_train.quantile(quantile))
        y_train_bin = (y_train >= threshold).astype(int)
        y_holdout_bin = (y_holdout >= threshold).astype(int)

        # ==================================================
        # 4. Fit each classifier for the current threshold
        # ==================================================

        for spec in model_specs:
            search = GridSearchCV(
                estimator=spec["estimator"],
                param_grid=spec["param_grid"],
                cv=cv,
                scoring="roc_auc",
                n_jobs=GRID_N_JOBS,
                error_score=np.nan,
            )
            search.fit(X_train, y_train_bin)

            if y_holdout_bin.nunique() < 2:
                holdout_auc = np.nan
            elif hasattr(search.best_estimator_, "decision_function"):
                scores = search.best_estimator_.decision_function(X_holdout)
                holdout_auc = float(roc_auc_score(y_holdout_bin, scores))
            else:
                scores = search.best_estimator_.predict_proba(X_holdout)[:, 1]
                holdout_auc = float(roc_auc_score(y_holdout_bin, scores))

            rows.append(
                {
                    "task": "classification",
                    "model": spec["name"],
                    "quantile": quantile,
                    "metric": "roc_auc",
                    "cv_score": float(search.best_score_),
                    "holdout_score": holdout_auc,
                    "best_params": json.dumps(search.best_params_, sort_keys=True, default=str),
                    "threshold": threshold,
                    "n_train": len(X_train),
                    "n_holdout": len(X_holdout),
                }
            )

    return pd.DataFrame(rows).sort_values(["cv_score", "holdout_score"], ascending=False)


def format_results(results: pd.DataFrame) -> str:
    """Format the results as a readable terminal report."""

    if results.empty:
        return "No results available."

    # ==================================================
    # 1. Small local helpers used only for printing
    # ==================================================

    def format_metric(value: float | int | None) -> str:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.3f}"

    def format_param_value(value: object) -> str:
        if isinstance(value, float):
            if value == 0:
                return "0"
            if abs(value) >= 100 or abs(value) < 0.01:
                return f"{value:.3g}"
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value)

    def format_best_params(raw_params: str) -> str:
        if not raw_params:
            return "-"
        params = json.loads(raw_params)
        if not params:
            return "-"

        parts = []
        for key, value in params.items():
            clean_key = key.replace("model__", "").replace("pca__", "")
            parts.append(f"{clean_key}={format_param_value(value)}")
        return ", ".join(parts)

    def build_text_table(headers: list[str], rows: list[list[str]]) -> str:
        widths = [len(header) for header in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(cell))

        formatted_rows = []
        header_row = " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers)))
        separator = "-+-".join("-" * width for width in widths)
        formatted_rows.append(header_row)
        formatted_rows.append(separator)

        for row in rows:
            formatted_rows.append(
                " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(row)))
            )

        return "\n".join(formatted_rows)

    # ==================================================
    # 2. Global header
    # ==================================================

    lines: list[str] = []
    n_train = int(results["n_train"].iloc[0])
    n_holdout = int(results["n_holdout"].iloc[0])
    lines.append("Training Summary")
    lines.append(f"train rows: {n_train} | holdout rows: {n_holdout}")

    # ==================================================
    # 3. Regression block
    # ==================================================

    regression = results.loc[results["task"] == "regression"].copy()
    if not regression.empty:
        regression_rows = []
        for _, row in regression.sort_values(["cv_score", "holdout_score"], ascending=False).iterrows():
            regression_rows.append(
                [
                    str(row["model"]),
                    format_metric(row["cv_score"]),
                    format_metric(row["holdout_score"]),
                    format_best_params(str(row["best_params"])),
                ]
            )

        lines.append("")
        lines.append("Regression")
        lines.append(
            build_text_table(
                ["model", "cv_r2", "holdout_r2", "best params"],
                regression_rows,
            )
        )

    # ==================================================
    # 4. Classification block
    # ==================================================

    classification = results.loc[results["task"] == "classification"].copy()
    if not classification.empty:
        lines.append("")
        lines.append("Classification")

        for quantile, subset in classification.groupby("quantile", sort=True):
            threshold = subset["threshold"].iloc[0]
            classification_rows = []

            for _, row in subset.sort_values(["cv_score", "holdout_score"], ascending=False).iterrows():
                classification_rows.append(
                    [
                        str(row["model"]),
                        format_metric(row["cv_score"]),
                        format_metric(row["holdout_score"]),
                        format_best_params(str(row["best_params"])),
                    ]
                )

            lines.append("")
            lines.append(f"q={float(quantile):.2f} | cutoff={format_metric(threshold)}")
            lines.append(
                build_text_table(
                    ["model", "cv_auc", "holdout_auc", "best params"],
                    classification_rows,
                )
            )

    return "\n".join(lines)


def save_results(results: pd.DataFrame, output_path=cfg.ML_RESULTS) -> None:
    """Save the benchmark table to CSV."""

    cfg.ensure_processed_dir()
    results.to_csv(output_path, index=False)


def run_modeling(
    *,
    rebuild_processed: bool = True,
    lag_days: int = cfg.DEFAULT_LAG_DAYS,
    holdout_days: int = 14,
    task: str = "all",
) -> pd.DataFrame:
    """Run the full modeling workflow."""

    # ==================================================
    # 1. Load data and create train / holdout splits
    # ==================================================

    processed_df = load_processed_data(rebuild_processed=rebuild_processed, lag_days=lag_days)
    X, y = prepare_training_data(processed_df, lag_days=lag_days)
    X_train, X_holdout, y_train, y_holdout = split_train_holdout(
        X,
        y,
        holdout_days=holdout_days,
    )

    # ==================================================
    # 2. Run the requested benchmark blocks
    # ==================================================

    outputs: list[pd.DataFrame] = []

    if task in {"all", "regression"}:
        outputs.append(benchmark_regression(X_train, y_train, X_holdout, y_holdout))

    if task in {"all", "classification"}:
        outputs.append(benchmark_classification(X_train, y_train, X_holdout, y_holdout))

    # ==================================================
    # 3. Merge and save the results
    # ==================================================

    results = pd.concat(outputs, ignore_index=True).sort_values(
        ["task", "cv_score", "holdout_score"],
        ascending=[True, False, False],
    )
    save_results(results)
    return results


def main() -> None:
    """CLI entry point used by the project pipeline."""

    parser = argparse.ArgumentParser(description="Run time-aware ML benchmarks on lagged daily features.")
    parser.add_argument(
        "--reuse-processed",
        action="store_true",
        help="Reuse data_processed.parquet instead of rebuilding the pipeline first.",
    )
    parser.add_argument("--lag-days", type=int, default=cfg.DEFAULT_LAG_DAYS)
    parser.add_argument("--holdout-days", type=int, default=14)
    parser.add_argument(
        "--task",
        choices=["all", "regression", "classification"],
        default="all",
    )
    args = parser.parse_args()

    results = run_modeling(
        rebuild_processed=not args.reuse_processed,
        lag_days=args.lag_days,
        holdout_days=args.holdout_days,
        task=args.task,
    )

    print(format_results(results))
    print(f"\nSaved CSV: {cfg.ML_RESULTS}")


if __name__ == "__main__":
    main()
