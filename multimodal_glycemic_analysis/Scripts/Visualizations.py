from __future__ import annotations

"""
Generate portfolio-ready figures from processed datasets and ML benchmark outputs.

Outputs written to `reports/figures/`:
1. `01_glycemic_overview.png`          - Daily glucose trajectory, TIR, and wearable coverage window
2. `02_multimodal_correlations.png`    - Spearman correlation of lagged multimodal predictors vs glycemic outcomes
3. `03_model_benchmark_ablation.png`   - Baselines vs Glooko-only vs Multimodal benchmark comparison
4. `04_feature_importance_and_roc.png` - Holdout permutation importance + ROC curve for best classifier
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

import MLapplications as ml
import Processing_and_descriptive as processing
import config as cfg

PALETTE = {
    "primary": "#1f4e79",
    "secondary": "#e07a5f",
    "accent": "#2a9d8f",
    "muted": "#6c757d",
    "light_bg": "#f8f9fa",
    "gap": "#f4a261",
}


def _apply_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d0d7de",
            "grid.color": "#e9ecef",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
        }
    )


def plot_glycemic_overview(merged_df: pd.DataFrame, out_path: Path) -> Path:
    """Plot daily glucose mean/variability, Time in Range, and multimodal overlap."""
    df = merged_df.sort_index()
    mm_start = processing.detect_multimodal_start(df)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.0]})

    # Top panel: daily mean glucose +/- SD + 7-day rolling mean
    ax = axes[0]
    roll7 = df["bg_mean"].rolling(7, min_periods=3).mean()
    ax.fill_between(
        df.index,
        df["bg_mean"] - df["bg_sd"],
        df["bg_mean"] + df["bg_sd"],
        color=PALETTE["primary"],
        alpha=0.14,
        label="Daily Mean ± 1 SD",
    )
    ax.plot(df.index, df["bg_mean"], color=PALETTE["primary"], alpha=0.45, linewidth=1.0, label="Daily Mean Glucose")
    ax.plot(df.index, roll7, color=PALETTE["primary"], linewidth=2.2, label="7-Day Rolling Mean")
    ax.axhline(200, color=PALETTE["secondary"], linestyle="--", linewidth=1.1, label="Hyperglycemia Threshold (200 mg/dL)")
    ax.axvspan(df.index.min(), mm_start, color=PALETTE["gap"], alpha=0.14, label="Sparse Wearable Period (Excluded from Multimodal)")
    ax.axvline(mm_start, color=PALETTE["gap"], linestyle=":", linewidth=1.5)
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("Longitudinal CGM Trajectory & Contiguous Multimodal Window")
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=8.5)

    # Bottom panel: Time in Range (%) + wearable sleep availability ticks
    ax2 = axes[1]
    tir_roll7 = df["bg_tir_%"].rolling(7, min_periods=3).mean()
    ax2.bar(df.index, df["bg_tir_%"], width=0.85, color=PALETTE["accent"], alpha=0.35, label="Daily TIR (70–200 mg/dL)")
    ax2.plot(df.index, tir_roll7, color=PALETTE["accent"], linewidth=2.0, label="7-Day Rolling TIR")
    ax2.axvspan(df.index.min(), mm_start, color=PALETTE["gap"], alpha=0.14)
    ax2.axvline(mm_start, color=PALETTE["gap"], linestyle=":", linewidth=1.5)

    sleep_ok = df.index[df["sl_tot_duration"].notna()] if "sl_tot_duration" in df.columns else []
    if len(sleep_ok) > 0:
        ax2.scatter(
            sleep_ok,
            np.full(len(sleep_ok), 4.0),
            marker="|",
            s=36,
            color="#3d405b",
            alpha=0.8,
            label="Xiaomi Sleep Recorded",
        )

    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Time in Range (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", ncol=3, frameon=True, fontsize=8.5)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_multimodal_correlations(processed_df: pd.DataFrame, out_path: Path) -> Path:
    """Plot Spearman correlations between key lagged predictors and daily glycemic targets."""
    targets = ["bg_auc_above_limit_rate", "bg_mean", "bg_sd", "bg_tir_%"]
    candidate_predictors = [
        "bg_auc_above_limit_rate_lag1",
        "bg_mean_lag1",
        "bg_mean_roll7_mean",
        "bg_sd_lag1",
        "carb_daily_lag1",
        "ins_tot_lag1",
        "ins_bolo_tot_lag1",
        "steps_lag1",
        "steps_roll7_mean",
        "calories_lag1",
        "training_lag1",
        "avg_rest_hr_lag1",
        "sl_tot_duration_lag1",
        "sl_score_lag1",
        "sl_bedtime_mfm_lag1",
        "cal_is_weekend",
    ]
    targets = [t for t in targets if t in processed_df.columns]
    predictors = [p for p in candidate_predictors if p in processed_df.columns]

    corr_full = processed_df[predictors + targets].corr(method="spearman")
    corr_sub = corr_full.loc[predictors, targets]

    rename_targets = {
        "bg_auc_above_limit_rate": "Hyperglycemic\nAUC Rate",
        "bg_mean": "Daily Mean\nGlucose",
        "bg_sd": "Glucose\nVariability (SD)",
        "bg_tir_%": "Time in\nRange (%)",
    }
    corr_sub = corr_sub.rename(columns=rename_targets)

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    sns.heatmap(
        corr_sub,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0.0,
        vmin=-0.5,
        vmax=0.5,
        linewidths=0.5,
        cbar_kws={"label": "Spearman ρ"},
        ax=ax,
    )
    ax.set_title("Spearman Correlation: Lagged Multimodal Predictors vs. Next-Day Glycemic Targets")
    ax.set_ylabel("Lagged / Rolling Predictor (available before day t)")
    ax.set_xlabel("Day-t Glycemic Outcome")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_benchmark_and_ablation(ablation_df: pd.DataFrame, out_path: Path) -> Path:
    """Plot Regression (MAE) and Classification (ROC-AUC) comparing Glooko-only vs Multimodal."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # Left: Regression Holdout MAE (lower is better)
    reg = ablation_df.loc[ablation_df["task"] == "regression"].copy()
    if not reg.empty:
        order = (
            reg.loc[reg["feature_set"] == "multimodal"]
            .sort_values("holdout_mae", ascending=True)["model"]
            .tolist()
        )
        sns.barplot(
            data=reg,
            y="model",
            x="holdout_mae",
            hue="feature_set",
            order=order,
            palette={"glooko_only": PALETTE["muted"], "multimodal": PALETTE["primary"]},
            ax=axes[0],
        )
        axes[0].set_title("Regression Holdout MAE (Lower is Better)")
        axes[0].set_xlabel("Holdout MAE (AUC Above Limit Rate)")
        axes[0].set_ylabel("Model")
        axes[0].set_xlim(0, reg["holdout_mae"].max() * 1.18)
        axes[0].legend(title="Feature Set", loc="lower right", frameon=True)

    # Right: Classification TimeSeriesSplit CV ROC-AUC at highest quantile (higher is better)
    clf = ablation_df.loc[ablation_df["task"] == "classification"].copy()
    if not clf.empty:
        target_q = float(clf["quantile"].max())
        clf_q = clf.loc[clf["quantile"] == target_q].copy()
        order_clf = (
            clf_q.loc[clf_q["feature_set"] == "multimodal"]
            .sort_values("cv_score", ascending=False)["model"]
            .tolist()
        )
        sns.barplot(
            data=clf_q,
            y="model",
            x="cv_score",
            hue="feature_set",
            order=order_clf,
            palette={"glooko_only": PALETTE["muted"], "multimodal": PALETTE["accent"]},
            ax=axes[1],
        )
        axes[1].axvline(0.5, color=PALETTE["secondary"], linestyle="--", linewidth=1.2, label="Random (0.50)")
        axes[1].set_xlim(0.4, 0.85)
        axes[1].set_title(f"High-Risk Classification CV ROC-AUC (q={target_q:.2f})")
        axes[1].set_xlabel("TimeSeriesSplit CV ROC-AUC")
        axes[1].set_ylabel("")
        axes[1].legend(title="Feature Set", loc="lower right", frameon=True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_importance_and_roc(
    importance_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    out_path: Path,
    *,
    lag_days: int = cfg.DEFAULT_LAG_DAYS,
    holdout_days: int = cfg.DEFAULT_HOLDOUT_DAYS,
) -> Path:
    """Plot top permutation importances alongside holdout ROC curve for best classifier."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Left: Permutation importance for best regression model (richer non-zero feature spread)
    ax_imp = axes[0]
    reg_imp = importance_df.loc[importance_df["task"] == "regression"].copy()
    imp_sub = reg_imp if not reg_imp.empty else importance_df.copy()
    if not imp_sub.empty:
        top = imp_sub.sort_values("importance_mean", ascending=False).head(12).iloc[::-1]
        color_map = {"glooko": PALETTE["primary"], "xiaomi": PALETTE["accent"], "calendar": PALETTE["secondary"]}
        colors = [color_map.get(m, PALETTE["muted"]) for m in top["modality"]]
        ax_imp.barh(
            top["feature"],
            top["importance_mean"],
            xerr=top["importance_std"],
            color=colors,
            alpha=0.88,
            capsize=3,
        )
        ax_imp.axvline(0.0, color="#343a40", linewidth=0.9)
        model_label = str(top["model"].iloc[0])
        ax_imp.set_title(f"Holdout Permutation Importance ({model_label} Regression)")
        ax_imp.set_xlabel("Mean MAE Increase When Permuted")
        ax_imp.set_ylabel("")

        from matplotlib.patches import Patch

        legend_handles = [
            Patch(facecolor=PALETTE["primary"], label="Glooko (CGM / Insulin / Carbs)"),
            Patch(facecolor=PALETTE["accent"], label="Xiaomi (Sleep / Activity / HR)"),
            Patch(facecolor=PALETTE["secondary"], label="Calendar"),
        ]
        ax_imp.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=8.5)

    # Right: Holdout ROC curve comparing best multimodal classifier vs persistence baseline
    ax_roc = axes[1]
    X, y = ml.prepare_training_data(processed_df, lag_days=lag_days, feature_set="multimodal")
    X_train, X_holdout, y_train, y_holdout = ml.split_train_holdout(X, y, holdout_days=holdout_days)

    clf_rows = ablation_df.loc[
        (ablation_df["task"] == "classification") & (ablation_df["feature_set"] == "multimodal")
    ]
    if not clf_rows.empty:
        target_q = float(clf_rows["quantile"].max())
        threshold = float(y_train.quantile(target_q))
        y_hold_bin = (y_holdout >= threshold).astype(int)

        if y_hold_bin.nunique() >= 2:
            # Persistence ROC
            lag1_col = f"{cfg.DEFAULT_TARGET_COLUMN}_lag1"
            if lag1_col in X_holdout.columns:
                p_scores = X_holdout[lag1_col].fillna(float(y_train.mean()))
                fpr_p, tpr_p, _ = roc_curve(y_hold_bin, p_scores)
                auc_p = roc_auc_score(y_hold_bin, p_scores)
                ax_roc.plot(
                    fpr_p,
                    tpr_p,
                    linestyle="--",
                    color=PALETTE["muted"],
                    linewidth=1.8,
                    label=f"Persistence Baseline (AUC = {auc_p:.3f})",
                )

            # Best non-baseline classifier refit cleanly on X_train
            _, clf_models = ml.benchmark_classification(
                X_train, y_train, X_holdout, y_holdout, quantiles=(target_q,), feature_set="multimodal"
            )
            non_base = clf_rows.loc[
                (clf_rows["quantile"] == target_q) & (~clf_rows["model"].str.startswith("baseline_"))
            ].sort_values(["cv_score", "holdout_score"], ascending=False)
            if not non_base.empty:
                best_name = str(non_base.iloc[0]["model"])
                best_est = clf_models[(target_q, best_name)]
                scores, _ = ml._score_binary_predictions(best_est, X_holdout)
                fpr, tpr, _ = roc_curve(y_hold_bin, scores)
                auc_val = roc_auc_score(y_hold_bin, scores)
                ax_roc.plot(
                    fpr,
                    tpr,
                    color=PALETTE["accent"],
                    linewidth=2.4,
                    label=f"Best Multimodal: {best_name} (AUC = {auc_val:.3f})",
                )

        ax_roc.plot([0, 1], [0, 1], linestyle=":", color="#adb5bd", linewidth=1.2, label="Random Chance (AUC = 0.500)")
        ax_roc.set_title(f"Holdout ROC Curve (High-Risk Threshold q={target_q:.2f})")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_figures(
    *,
    lag_days: int = cfg.DEFAULT_LAG_DAYS,
    holdout_days: int = cfg.DEFAULT_HOLDOUT_DAYS,
) -> list[Path]:
    """Generate all report figures and return their paths."""
    _apply_style()
    fig_dir = cfg.ensure_figures_dir()

    merged_df = processing.load_merged_data()
    processed_df = pd.read_parquet(cfg.require_existing_path(cfg.DATA_PROCESSED)).sort_index()
    ablation_df = pd.read_csv(cfg.require_existing_path(cfg.ABLATION_RESULTS))
    importance_df = pd.read_csv(cfg.require_existing_path(cfg.FEATURE_IMPORTANCE))

    paths = [
        plot_glycemic_overview(merged_df, fig_dir / "01_glycemic_overview.png"),
        plot_multimodal_correlations(processed_df, fig_dir / "02_multimodal_correlations.png"),
        plot_benchmark_and_ablation(ablation_df, fig_dir / "03_model_benchmark_ablation.png"),
        plot_importance_and_roc(
            importance_df,
            processed_df,
            ablation_df,
            fig_dir / "04_feature_importance_and_roc.png",
            lag_days=lag_days,
            holdout_days=holdout_days,
        ),
    ]
    return paths


def main() -> None:
    paths = generate_all_figures()
    for path in paths:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
