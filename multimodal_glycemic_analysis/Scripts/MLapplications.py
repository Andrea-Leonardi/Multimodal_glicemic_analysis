from __future__ import annotations

"""
Time-aware modeling, baseline comparison, modality ablation, and interpretability.

The workflow is:
1. load or rebuild the engineered daily dataset (restricted by default to the
   contiguous multimodal recording window so wearable features are genuinely active),
2. construct leak-free feature matrices for either `multimodal` (Glooko + Xiaomi)
   or `glooko_only` (ablation control),
3. split chronologically into train and holdout windows,
4. benchmark explicit baselines (mean/prior and persistence y_{t-1}) alongside
   linear, kernel, and tree-based models using TimeSeriesSplit,
5. compute permutation importance on the holdout set for interpretability,
6. export benchmark, ablation, and feature-importance CSVs and print a clean summary.
"""

import argparse
import json
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
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

DEFAULT_CLASSIFICATION_QUANTILES = (0.50, 0.65)
GRID_N_JOBS = 1
RANDOM_STATE = 42


def _elasticnet_logistic_kwargs() -> dict[str, object]:
    """Return version-safe LogisticRegression kwargs for Elastic Net regularization."""
    parts = sklearn.__version__.split(".")
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    kwargs: dict[str, object] = {
        "max_iter": 5000,
        "class_weight": "balanced",
        "solver": "saga",
        "tol": 1e-2,
        "random_state": RANDOM_STATE,
    }
    if minor < 8:
        kwargs["penalty"] = "elasticnet"
    return kwargs


def is_xiaomi_feature(col: str) -> bool:
    """Return True if an engineered feature originates from Xiaomi wearable signals."""
    if col.startswith("sl_"):
        return True
    return any(col.startswith(f"{base}_") for base in cfg.XIAOMI_LAG_COLUMNS)


def is_engineered_predictor(col: str, lag_days: int) -> bool:
    """Return True if `col` is a leak-free engineered predictor."""
    if col.startswith("cal_") or col == "sl_missing_lag1":
        return True
    if any(col.endswith(f"_lag{lag}") for lag in range(1, lag_days + 1)):
        return True
    if "_roll" in col and (col.endswith("_mean") or col.endswith("_std")):
        return True
    if col.endswith("_delta_7d"):
        return True
    return False


def load_processed_data(
    rebuild_processed: bool = True,
    lag_days: int = cfg.DEFAULT_LAG_DAYS,
    restrict_multimodal: bool = True,
) -> pd.DataFrame:
    """Load the processed feature dataset, or rebuild it from the pipeline."""
    if rebuild_processed:
        return processing.prepare_processed_dataset(
            rebuild_merged=True,
            save=True,
            lag_days=lag_days,
            restrict_multimodal=restrict_multimodal,
        )

    return pd.read_parquet(cfg.require_existing_path(cfg.DATA_PROCESSED)).sort_index()


def prepare_training_data(
    df: pd.DataFrame,
    *,
    target_col: str = cfg.DEFAULT_TARGET_COLUMN,
    lag_days: int = cfg.DEFAULT_LAG_DAYS,
    min_availability: float = cfg.MIN_FEATURE_AVAILABILITY,
    feature_set: str = "multimodal",
) -> tuple[pd.DataFrame, pd.Series]:
    """Create the leak-free feature matrix X and target vector y."""
    if feature_set not in {"multimodal", "glooko_only"}:
        raise ValueError(f"Unsupported feature_set={feature_set!r}; expected 'multimodal' or 'glooko_only'.")

    warmup = max(lag_days, 3)
    modeling_df = df.sort_index().iloc[warmup:].copy()
    modeling_df = modeling_df.loc[modeling_df[target_col].notna()].copy()

    selected_columns: list[str] = []
    for col in modeling_df.columns:
        if not is_engineered_predictor(col, lag_days):
            continue
        if feature_set == "glooko_only" and is_xiaomi_feature(col):
            continue
        if modeling_df[col].notna().mean() >= min_availability:
            selected_columns.append(col)

    selected_columns = sorted(selected_columns)
    if not selected_columns:
        raise ValueError("No eligible engineered feature columns were found in the processed dataset.")

    X = modeling_df[selected_columns]
    y = modeling_df[target_col]
    return X, y


def split_train_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    holdout_days: int = cfg.DEFAULT_HOLDOUT_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a chronological train/holdout split."""
    if holdout_days < 1:
        raise ValueError("holdout_days must be at least 1.")
    if len(X) <= holdout_days + 15:
        raise ValueError("Not enough samples to create a meaningful holdout split.")

    X_train = X.iloc[:-holdout_days]
    X_holdout = X.iloc[-holdout_days:]
    y_train = y.iloc[:-holdout_days]
    y_holdout = y.iloc[-holdout_days:]
    return X_train, X_holdout, y_train, y_holdout


def _build_cv(n_train: int) -> TimeSeriesSplit:
    n_splits = min(5, max(2, n_train // 20))
    if n_train <= n_splits:
        raise ValueError("Not enough samples for TimeSeriesSplit.")
    return TimeSeriesSplit(n_splits=n_splits)


def _evaluate_persistence_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    cv: TimeSeriesSplit,
    target_col: str,
    feature_set: str,
) -> dict[str, object] | None:
    """Evaluate a naive persistence baseline predicting y_hat(t) = y(t - 1)."""
    lag1_col = f"{target_col}_lag1"
    if lag1_col not in X_train.columns:
        return None

    cv_r2s: list[float] = []
    cv_maes: list[float] = []
    for train_idx, val_idx in cv.split(X_train):
        fallback = float(y_train.iloc[train_idx].mean())
        preds = X_train[lag1_col].iloc[val_idx].fillna(fallback)
        cv_r2s.append(float(r2_score(y_train.iloc[val_idx], preds)))
        cv_maes.append(float(mean_absolute_error(y_train.iloc[val_idx], preds)))

    holdout_preds = X_holdout[lag1_col].fillna(float(y_train.mean()))
    return {
        "feature_set": feature_set,
        "task": "regression",
        "model": "baseline_persistence",
        "quantile": np.nan,
        "metric": "r2",
        "cv_score": float(np.mean(cv_r2s)),
        "holdout_score": float(r2_score(y_holdout, holdout_preds)),
        "cv_mae": float(np.mean(cv_maes)),
        "holdout_mae": float(mean_absolute_error(y_holdout, holdout_preds)),
        "holdout_rmse": float(root_mean_squared_error(y_holdout, holdout_preds)),
        "best_params": "{}",
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
        "n_holdout": len(X_holdout),
    }


def benchmark_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    *,
    target_col: str = cfg.DEFAULT_TARGET_COLUMN,
    feature_set: str = "multimodal",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the regression benchmark and return both summary table and fitted best model."""
    cv = _build_cv(len(X_train))
    max_components = max(2, min(10, X_train.shape[1]))

    model_specs = [
        {
            "name": "baseline_mean",
            "estimator": Pipeline([("model", DummyRegressor(strategy="mean"))]),
            "param_grid": {},
        },
        {
            "name": "ridge",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge()),
                ]
            ),
            "param_grid": {"model__alpha": np.logspace(-1, 4, 15)},
        },
        {
            "name": "lasso",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Lasso(max_iter=200000, tol=1e-4, random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {"model__alpha": np.logspace(-2, 2, 16)},
        },
        {
            "name": "elastic_net",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(max_iter=50000, tol=1e-3, random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                "model__alpha": np.logspace(-2, 1, 8),
                "model__l1_ratio": [0.2, 0.5, 0.8],
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
                "pca__n_components": sorted({2, 4, 6, 8, max_components}),
                "model__alpha": np.logspace(-1, 3, 6),
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
            "param_grid": {"model__n_components": list(range(1, min(7, X_train.shape[1]) + 1))},
        },
        {
            "name": "random_forest",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=100,
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__max_depth": [3, 6],
                "model__min_samples_leaf": [4, 8],
                "model__max_features": ["sqrt"],
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
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__max_leaf_nodes": [7, 15],
                "model__min_samples_leaf": [5, 15],
            },
        },
    ]

    rows: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {}

    persistence_row = _evaluate_persistence_regression(
        X_train, y_train, X_holdout, y_holdout, cv, target_col, feature_set
    )
    if persistence_row is not None:
        rows.append(persistence_row)

    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error"}

    for spec in model_specs:
        search = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            cv=cv,
            scoring=scoring,
            refit="r2",
            n_jobs=GRID_N_JOBS,
            error_score=np.nan,
        )
        search.fit(X_train, y_train)
        best_idx = int(search.best_index_)
        cv_mae = float(-search.cv_results_["mean_test_mae"][best_idx])
        holdout_preds = search.best_estimator_.predict(X_holdout)
        fitted_models[spec["name"]] = search.best_estimator_

        rows.append(
            {
                "feature_set": feature_set,
                "task": "regression",
                "model": spec["name"],
                "quantile": np.nan,
                "metric": "r2",
                "cv_score": float(search.best_score_),
                "holdout_score": float(r2_score(y_holdout, holdout_preds)),
                "cv_mae": cv_mae,
                "holdout_mae": float(mean_absolute_error(y_holdout, holdout_preds)),
                "holdout_rmse": float(root_mean_squared_error(y_holdout, holdout_preds)),
                "best_params": json.dumps(search.best_params_, sort_keys=True, default=str),
                "n_features": X_train.shape[1],
                "n_train": len(X_train),
                "n_holdout": len(X_holdout),
            }
        )

    df_res = pd.DataFrame(rows).sort_values(["cv_score", "holdout_score"], ascending=False)
    return df_res, fitted_models


def _score_binary_predictions(estimator: object, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract ranking scores (for ROC/PR AUC) and probabilities (for Brier) if available."""
    probs: np.ndarray | None = None
    if hasattr(estimator, "predict_proba"):
        probs = np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)
        return probs, probs
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=float)
        return scores, None
    preds = np.asarray(estimator.predict(X), dtype=float)
    return preds, None


def benchmark_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    *,
    quantiles: tuple[float, ...] = DEFAULT_CLASSIFICATION_QUANTILES,
    target_col: str = cfg.DEFAULT_TARGET_COLUMN,
    feature_set: str = "multimodal",
) -> tuple[pd.DataFrame, dict[tuple[float, str], object]]:
    """Run the binary glycemic-risk classification benchmark across target quantiles."""
    cv = _build_cv(len(X_train))

    model_specs = [
        {
            "name": "baseline_prior",
            "estimator": Pipeline([("model", DummyClassifier(strategy="prior"))]),
            "param_grid": {},
        },
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
            "param_grid": {"model__C": np.logspace(-2, 2, 10)},
        },
        {
            "name": "logistic_elasticnet",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(**_elasticnet_logistic_kwargs())),
                ]
            ),
            "param_grid": {
                "model__C": np.logspace(-2, 1, 5),
                "model__l1_ratio": [0.2, 0.5, 0.8],
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
                    ("model", SVC(kernel="linear", class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {"model__C": np.logspace(-2, 1, 6)},
        },
        {
            "name": "svm_rbf",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                "model__C": [0.1, 1.0, 10.0, 50.0],
                "model__gamma": [1e-3, 1e-2, 1e-1],
            },
        },
        {
            "name": "random_forest",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=100,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__max_depth": [3, 6],
                "model__min_samples_leaf": [4, 8],
                "model__max_features": ["sqrt"],
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
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__max_leaf_nodes": [7, 15],
                "model__min_samples_leaf": [5, 15],
            },
        },
    ]

    rows: list[dict[str, object]] = []
    fitted_models: dict[tuple[float, str], object] = {}
    lag1_col = f"{target_col}_lag1"

    for quantile in quantiles:
        threshold = float(y_train.quantile(quantile))
        y_train_bin = (y_train >= threshold).astype(int)
        y_holdout_bin = (y_holdout >= threshold).astype(int)
        has_both_classes = y_holdout_bin.nunique() >= 2

        # Persistence baseline using previous day's target value as risk score
        if lag1_col in X_train.columns:
            cv_aucs: list[float] = []
            for train_idx, val_idx in cv.split(X_train):
                y_val_fold = y_train_bin.iloc[val_idx]
                if y_val_fold.nunique() < 2:
                    continue
                fallback = float(y_train.iloc[train_idx].mean())
                s_val = X_train[lag1_col].iloc[val_idx].fillna(fallback)
                cv_aucs.append(float(roc_auc_score(y_val_fold, s_val)))

            s_hold = X_holdout[lag1_col].fillna(float(y_train.mean()))
            rows.append(
                {
                    "feature_set": feature_set,
                    "task": "classification",
                    "model": "baseline_persistence",
                    "quantile": quantile,
                    "metric": "roc_auc",
                    "cv_score": float(np.mean(cv_aucs)) if cv_aucs else np.nan,
                    "holdout_score": float(roc_auc_score(y_holdout_bin, s_hold)) if has_both_classes else np.nan,
                    "holdout_pr_auc": (
                        float(average_precision_score(y_holdout_bin, s_hold)) if has_both_classes else np.nan
                    ),
                    "holdout_brier": np.nan,
                    "best_params": "{}",
                    "threshold": threshold,
                    "n_features": X_train.shape[1],
                    "n_train": len(X_train),
                    "n_holdout": len(X_holdout),
                }
            )

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
            fitted_models[(quantile, spec["name"])] = search.best_estimator_

            if not has_both_classes:
                holdout_auc = np.nan
                holdout_pr_auc = np.nan
                holdout_brier = np.nan
            else:
                scores, probs = _score_binary_predictions(search.best_estimator_, X_holdout)
                holdout_auc = float(roc_auc_score(y_holdout_bin, scores))
                holdout_pr_auc = float(average_precision_score(y_holdout_bin, scores))
                holdout_brier = (
                    float(brier_score_loss(y_holdout_bin, probs)) if probs is not None else np.nan
                )

            rows.append(
                {
                    "feature_set": feature_set,
                    "task": "classification",
                    "model": spec["name"],
                    "quantile": quantile,
                    "metric": "roc_auc",
                    "cv_score": float(search.best_score_),
                    "holdout_score": holdout_auc,
                    "holdout_pr_auc": holdout_pr_auc,
                    "holdout_brier": holdout_brier,
                    "best_params": json.dumps(search.best_params_, sort_keys=True, default=str),
                    "threshold": threshold,
                    "n_features": X_train.shape[1],
                    "n_train": len(X_train),
                    "n_holdout": len(X_holdout),
                }
            )

    df_res = pd.DataFrame(rows).sort_values(["cv_score", "holdout_score"], ascending=False)
    return df_res, fitted_models


def compute_holdout_permutation_importance(
    estimator: object,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    *,
    task: str,
    model_name: str,
    scoring: str,
    n_repeats: int = 20,
) -> pd.DataFrame:
    """Compute permutation importance on the chronological holdout set."""
    perm = permutation_importance(
        estimator,
        X_holdout,
        y_holdout,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    rows = []
    for feature, mean_imp, std_imp in zip(
        X_holdout.columns, perm.importances_mean, perm.importances_std
    ):
        rows.append(
            {
                "task": task,
                "model": model_name,
                "feature": feature,
                "modality": "xiaomi" if is_xiaomi_feature(feature) else ("calendar" if feature.startswith("cal_") else "glooko"),
                "importance_mean": float(mean_imp),
                "importance_std": float(std_imp),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def format_results(results: pd.DataFrame) -> str:
    """Format benchmark results as a readable terminal report."""
    if results.empty:
        return "No results available."

    def format_metric(value: object) -> str:
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
        if not raw_params or raw_params == "nan":
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
        formatted_rows = [
            " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))),
            "-+-".join("-" * width for width in widths),
        ]
        for row in rows:
            formatted_rows.append(
                " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(row)))
            )
        return "\n".join(formatted_rows)

    lines: list[str] = []
    n_train = int(results["n_train"].iloc[0])
    n_holdout = int(results["n_holdout"].iloc[0])
    n_feat = int(results["n_features"].iloc[0]) if "n_features" in results.columns else "-"
    lines.append("Training Summary")
    lines.append(f"train rows: {n_train} | holdout rows: {n_holdout} | features: {n_feat}")

    regression = results.loc[results["task"] == "regression"].copy()
    if not regression.empty:
        regression_rows = []
        for _, row in regression.sort_values(["cv_score", "holdout_score"], ascending=False).iterrows():
            regression_rows.append(
                [
                    str(row["model"]),
                    format_metric(row.get("cv_score")),
                    format_metric(row.get("holdout_score")),
                    format_metric(row.get("cv_mae")),
                    format_metric(row.get("holdout_mae")),
                    format_metric(row.get("holdout_rmse")),
                    format_best_params(str(row.get("best_params", ""))),
                ]
            )
        lines.append("")
        lines.append("Regression")
        lines.append(
            build_text_table(
                ["model", "cv_r2", "hold_r2", "cv_mae", "hold_mae", "hold_rmse", "best params"],
                regression_rows,
            )
        )

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
                        format_metric(row.get("cv_score")),
                        format_metric(row.get("holdout_score")),
                        format_metric(row.get("holdout_pr_auc")),
                        format_best_params(str(row.get("best_params", ""))),
                    ]
                )
            lines.append("")
            lines.append(f"q={float(quantile):.2f} | cutoff={format_metric(threshold)}")
            lines.append(
                build_text_table(
                    ["model", "cv_auc", "hold_auc", "hold_pr", "best params"],
                    classification_rows,
                )
            )

    return "\n".join(lines)


def save_results(results: pd.DataFrame, output_path=cfg.ML_RESULTS) -> None:
    """Save a benchmark table to CSV."""
    cfg.ensure_processed_dir()
    results.to_csv(output_path, index=False)


def run_modeling(
    *,
    rebuild_processed: bool = True,
    lag_days: int = cfg.DEFAULT_LAG_DAYS,
    holdout_days: int = cfg.DEFAULT_HOLDOUT_DAYS,
    task: str = "all",
    feature_set: str = "multimodal",
    run_ablation: bool = True,
    compute_importance: bool = True,
) -> pd.DataFrame:
    """Run the full modeling workflow, optional ablation, and permutation importance."""
    processed_df = load_processed_data(
        rebuild_processed=rebuild_processed,
        lag_days=lag_days,
        restrict_multimodal=True,
    )

    X, y = prepare_training_data(processed_df, lag_days=lag_days, feature_set=feature_set)
    X_train, X_holdout, y_train, y_holdout = split_train_holdout(X, y, holdout_days=holdout_days)

    outputs: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    if task in {"all", "regression"}:
        reg_df, reg_models = benchmark_regression(
            X_train, y_train, X_holdout, y_holdout, feature_set=feature_set
        )
        outputs.append(reg_df)

        if compute_importance:
            non_base = reg_df.loc[~reg_df["model"].str.startswith("baseline_")]
            if not non_base.empty:
                best_reg_name = str(non_base.iloc[0]["model"])
                importance_frames.append(
                    compute_holdout_permutation_importance(
                        reg_models[best_reg_name],
                        X_holdout,
                        y_holdout,
                        task="regression",
                        model_name=best_reg_name,
                        scoring="neg_mean_absolute_error",
                    )
                )

    if task in {"all", "classification"}:
        clf_df, clf_models = benchmark_classification(
            X_train, y_train, X_holdout, y_holdout, feature_set=feature_set
        )
        outputs.append(clf_df)

        if compute_importance:
            target_q = max(DEFAULT_CLASSIFICATION_QUANTILES)
            q_sub = clf_df.loc[
                (clf_df["quantile"] == target_q) & (~clf_df["model"].str.startswith("baseline_"))
            ]
            if not q_sub.empty:
                best_clf_name = str(q_sub.iloc[0]["model"])
                threshold = float(y_train.quantile(target_q))
                y_hold_bin = (y_holdout >= threshold).astype(int)
                if y_hold_bin.nunique() >= 2:
                    importance_frames.append(
                        compute_holdout_permutation_importance(
                            clf_models[(target_q, best_clf_name)],
                            X_holdout,
                            y_hold_bin,
                            task=f"classification_q{int(target_q * 100)}",
                            model_name=best_clf_name,
                            scoring="roc_auc",
                        )
                    )

    results = pd.concat(outputs, ignore_index=True).sort_values(
        ["task", "cv_score", "holdout_score"],
        ascending=[True, False, False],
    )
    save_results(results, cfg.ML_RESULTS)

    if importance_frames:
        imp_df = pd.concat(importance_frames, ignore_index=True)
        save_results(imp_df, cfg.FEATURE_IMPORTANCE)

    if run_ablation and feature_set == "multimodal":
        X_gl, y_gl = prepare_training_data(processed_df, lag_days=lag_days, feature_set="glooko_only")
        Xg_tr, Xg_ho, yg_tr, yg_ho = split_train_holdout(X_gl, y_gl, holdout_days=holdout_days)
        ablation_parts: list[pd.DataFrame] = [results]
        if task in {"all", "regression"}:
            reg_gl, _ = benchmark_regression(Xg_tr, yg_tr, Xg_ho, yg_ho, feature_set="glooko_only")
            ablation_parts.append(reg_gl)
        if task in {"all", "classification"}:
            clf_gl, _ = benchmark_classification(Xg_tr, yg_tr, Xg_ho, yg_ho, feature_set="glooko_only")
            ablation_parts.append(clf_gl)
        ablation_df = pd.concat(ablation_parts, ignore_index=True)
        save_results(ablation_df, cfg.ABLATION_RESULTS)

    return results


def main() -> None:
    """CLI entry point used by the project pipeline."""
    parser = argparse.ArgumentParser(description="Run time-aware ML benchmarks on engineered daily features.")
    parser.add_argument(
        "--reuse-processed",
        action="store_true",
        help="Reuse data_processed.parquet instead of rebuilding the pipeline first.",
    )
    parser.add_argument("--lag-days", type=int, default=cfg.DEFAULT_LAG_DAYS)
    parser.add_argument("--holdout-days", type=int, default=cfg.DEFAULT_HOLDOUT_DAYS)
    parser.add_argument(
        "--task",
        choices=["all", "regression", "classification"],
        default="all",
    )
    parser.add_argument(
        "--feature-set",
        choices=["multimodal", "glooko_only"],
        default="multimodal",
    )
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()

    results = run_modeling(
        rebuild_processed=not args.reuse_processed,
        lag_days=args.lag_days,
        holdout_days=args.holdout_days,
        task=args.task,
        feature_set=args.feature_set,
        run_ablation=not args.no_ablation,
    )

    print(format_results(results))
    print(f"\nSaved CSV: {cfg.ML_RESULTS}")
    if not args.no_ablation and args.feature_set == "multimodal":
        print(f"Saved ablation CSV: {cfg.ABLATION_RESULTS}")
    print(f"Saved importance CSV: {cfg.FEATURE_IMPORTANCE}")


if __name__ == "__main__":
    main()
