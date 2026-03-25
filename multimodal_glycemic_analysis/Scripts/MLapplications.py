from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config as cfg
import Processing_and_descriptive as processing

DEFAULT_CLASSIFICATION_QUANTILES = (0.35, 0.50, 0.65)
GRID_N_JOBS = 1


def load_processed_data(rebuild_processed: bool = True, lag_days: int = cfg.DEFAULT_LAG_DAYS) -> pd.DataFrame:
    if rebuild_processed:
        return processing.prepare_processed_dataset(rebuild_merged=True, save=True, lag_days=lag_days)
    return pd.read_parquet(cfg.require_existing_path(cfg.DATA_PROCESSED)).sort_index()


def select_feature_columns(
    df: pd.DataFrame,
    lag_days: int = cfg.DEFAULT_LAG_DAYS,
    min_availability: float = cfg.MIN_FEATURE_AVAILABILITY,
) -> list[str]:
    lagged_cols = [
        col
        for col in df.columns
        if any(col.endswith(f"_lag{lag}") for lag in range(1, lag_days + 1))
    ]
    feature_cols = sorted(
        col for col in lagged_cols if df[col].notna().mean() >= min_availability
    )
    if "day" in df.columns:
        feature_cols = ["day", *feature_cols]
    if not feature_cols:
        raise ValueError("No lagged feature columns were found in the processed dataset.")
    return feature_cols


def prepare_training_data(
    df: pd.DataFrame,
    *,
    target_col: str = cfg.DEFAULT_TARGET_COLUMN,
    lag_days: int = cfg.DEFAULT_LAG_DAYS,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = select_feature_columns(df, lag_days=lag_days)
    modeling_df = df.sort_index().iloc[lag_days:].copy()
    modeling_df = modeling_df.loc[modeling_df[target_col].notna()].copy()
    return modeling_df[feature_cols], modeling_df[target_col]


def split_train_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    holdout_days: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if holdout_days < 1:
        raise ValueError("holdout_days must be at least 1.")
    if len(X) <= holdout_days + 10:
        raise ValueError("Not enough samples to create a meaningful holdout split.")
    return X.iloc[:-holdout_days], X.iloc[-holdout_days:], y.iloc[:-holdout_days], y.iloc[-holdout_days:]


def _make_cv(n_samples: int) -> TimeSeriesSplit:
    n_splits = min(5, max(2, n_samples // 20))
    if n_samples <= n_splits:
        raise ValueError("Not enough samples for TimeSeriesSplit.")
    return TimeSeriesSplit(n_splits=n_splits)


def _serialise_params(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def _holdout_auc(estimator, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    if y_test.nunique() < 2:
        return float("nan")
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_test)
    else:
        scores = estimator.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, scores))


def benchmark_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
) -> pd.DataFrame:
    cv = _make_cv(len(X_train))
    max_components = max(2, min(10, X_train.shape[1]))

    model_specs = [
        (
            "lasso",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Lasso(max_iter=200000, tol=1e-4)),
                ]
            ),
            {"model__alpha": np.logspace(-3, 2, 20)},
        ),
        (
            "pcr_ridge",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("pca", PCA()),
                    ("model", Ridge()),
                ]
            ),
            {
                "pca__n_components": list(range(2, max_components + 1)),
                "model__alpha": np.logspace(-2, 4, 12),
            },
        ),
        (
            "pls",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", PLSRegression()),
                ]
            ),
            {"model__n_components": list(range(1, min(8, X_train.shape[1]) + 1))},
        ),
    ]

    rows: list[dict] = []
    for model_name, estimator, param_grid in model_specs:
        search = GridSearchCV(
            estimator,
            param_grid=param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=GRID_N_JOBS,
            error_score=np.nan,
        )
        search.fit(X_train, y_train)
        rows.append(
            {
                "task": "regression",
                "model": model_name,
                "quantile": np.nan,
                "metric": "r2",
                "cv_score": float(search.best_score_),
                "holdout_score": float(search.best_estimator_.score(X_holdout, y_holdout)),
                "best_params": _serialise_params(search.best_params_),
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
    cv = _make_cv(len(X_train))
    model_specs = [
        (
            "logistic",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=10000,
                            class_weight="balanced",
                            solver="liblinear",
                        ),
                    ),
                ]
            ),
            {"model__C": np.logspace(-2, 2, 10)},
        ),
        (
            "naive_bayes",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", GaussianNB()),
                ]
            ),
            {},
        ),
        (
            "lda",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LinearDiscriminantAnalysis()),
                ]
            ),
            {},
        ),
        (
            "svm_linear",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="linear", class_weight="balanced")),
                ]
            ),
            {"model__C": np.logspace(-2, 2, 10)},
        ),
        (
            "svm_rbf",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="rbf", class_weight="balanced")),
                ]
            ),
            {
                "model__C": np.logspace(-1, 2, 8),
                "model__gamma": np.logspace(-4, -1, 8),
            },
        ),
    ]

    rows: list[dict] = []
    for quantile in quantiles:
        threshold = float(y_train.quantile(quantile))
        y_train_bin = (y_train >= threshold).astype(int)
        y_holdout_bin = (y_holdout >= threshold).astype(int)

        for model_name, estimator, param_grid in model_specs:
            search = GridSearchCV(
                estimator,
                param_grid=param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=GRID_N_JOBS,
                error_score=np.nan,
            )
            search.fit(X_train, y_train_bin)
            rows.append(
                {
                    "task": "classification",
                    "model": model_name,
                    "quantile": quantile,
                    "metric": "roc_auc",
                    "cv_score": float(search.best_score_),
                    "holdout_score": _holdout_auc(search.best_estimator_, X_holdout, y_holdout_bin),
                    "best_params": _serialise_params(search.best_params_),
                    "threshold": threshold,
                    "n_train": len(X_train),
                    "n_holdout": len(X_holdout),
                }
            )

    return pd.DataFrame(rows).sort_values(["cv_score", "holdout_score"], ascending=False)


def save_results(results: pd.DataFrame, output_path=cfg.ML_RESULTS) -> None:
    cfg.ensure_processed_dir()
    results.to_csv(output_path, index=False)


def run_modeling(
    *,
    rebuild_processed: bool = True,
    lag_days: int = cfg.DEFAULT_LAG_DAYS,
    holdout_days: int = 14,
    task: str = "all",
) -> pd.DataFrame:
    processed_df = load_processed_data(rebuild_processed=rebuild_processed, lag_days=lag_days)
    X, y = prepare_training_data(processed_df, lag_days=lag_days)
    X_train, X_holdout, y_train, y_holdout = split_train_holdout(X, y, holdout_days=holdout_days)

    outputs: list[pd.DataFrame] = []
    if task in {"all", "regression"}:
        outputs.append(benchmark_regression(X_train, y_train, X_holdout, y_holdout))
    if task in {"all", "classification"}:
        outputs.append(benchmark_classification(X_train, y_train, X_holdout, y_holdout))

    results = pd.concat(outputs, ignore_index=True).sort_values(
        ["task", "cv_score", "holdout_score"], ascending=[True, False, False]
    )
    save_results(results)
    return results


def main() -> None:
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

    print(results.head(10).to_string(index=False))
    print(f"\nSaved: {cfg.ML_RESULTS}")


if __name__ == "__main__":
    main()
