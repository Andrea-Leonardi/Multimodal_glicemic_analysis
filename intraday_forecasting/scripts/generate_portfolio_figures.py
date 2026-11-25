from __future__ import annotations

"""
Generate portfolio figures for Track A (High-Resolution 5-min CGM + Pump Forecasting).

Outputs written to `<repo_root>/reports/figures/`:
1. `05_intraday_agp_profile.png`           - Ambulatory Glucose Profile (24h circadian percentiles from 94,237
                                             5-min readings) + representative 48h high-resolution CGM & bolus trace
2. `06_intraday_benchmark_uncertainty.png` - Multi-horizon MAE progression (v1 -> v2 -> v3 across 30/60/120/180 min)
                                             + 90% adaptive prediction-interval coverage & width diagnostics
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

INTRADAY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = INTRADAY_ROOT.parent
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

CGM_PATH = INTRADAY_ROOT / "data" / "cleaned" / "cleaned-CGM.csv"
BOLO_PATH = INTRADAY_ROOT / "data" / "cleaned" / "cleaned-Bolo.csv"

PALETTE = {
    "primary": "#1f4e79",
    "accent": "#2a9d8f",
    "warn": "#e07a5f",
    "gold": "#e9c46a",
    "muted": "#6c757d",
    "dark": "#264653",
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


def plot_intraday_agp_and_trace(out_path: Path) -> Path:
    """Plot 24-hour Ambulatory Glucose Profile (AGP) and a 48h 5-min CGM + bolus trace."""
    cgm = pd.read_csv(CGM_PATH)
    cgm["datetime"] = pd.to_datetime(cgm["datetime"])
    cgm["glucose"] = pd.to_numeric(cgm["glucose"], errors="coerce")
    cgm = cgm.dropna(subset=["datetime", "glucose"]).sort_values("datetime")

    # Bin time-of-day into 15-minute intervals for smooth circadian AGP
    cgm["minute_bin"] = ((cgm["datetime"].dt.hour * 60 + cgm["datetime"].dt.minute) // 15) * 15
    agp = (
        cgm.groupby("minute_bin")["glucose"]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        .unstack()
    )
    hours = agp.index.to_numpy() / 60.0

    # Load boluses for 48-hour trace overlay
    bolo = pd.read_csv(BOLO_PATH)
    bolo["datetime"] = pd.to_datetime(bolo["datetime"])
    bolo["carbs"] = pd.to_numeric(bolo["Quantità Carboidrati"], errors="coerce").fillna(0.0)
    bolo["insulin"] = pd.to_numeric(bolo["Insulina erogata"], errors="coerce").fillna(0.0)

    # Pick a clean 48-hour window inside the Sep-Oct test period (2026-09-02..2026-09-04)
    start_ts = pd.Timestamp("2026-09-02 00:00:00")
    end_ts = pd.Timestamp("2026-09-04 00:00:00")
    trace = cgm[(cgm["datetime"] >= start_ts) & (cgm["datetime"] < end_ts)]
    if len(trace) < 400:
        # Fallback to last 48h with dense coverage
        end_ts = cgm["datetime"].max()
        start_ts = end_ts - pd.Timedelta(hours=48)
        trace = cgm[(cgm["datetime"] >= start_ts) & (cgm["datetime"] <= end_ts)]
    bolo_trace = bolo[(bolo["datetime"] >= start_ts) & (bolo["datetime"] <= end_ts)]
    meals = bolo_trace[(bolo_trace["Tipo di bolo"] == "Alimenti") & (bolo_trace["carbs"] > 0)]
    manual_corr = bolo_trace[(bolo_trace["Tipo di bolo"] == "Correzione") & (bolo_trace["insulin"] > 0)]

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), gridspec_kw={"width_ratios": [1.05, 1.25]})

    # Left: Ambulatory Glucose Profile (AGP) across 94,237 readings
    ax = axes[0]
    ax.axhspan(70, 180, color=PALETTE["accent"], alpha=0.09, label="Clinical Target (70–180 mg/dL)")
    ax.fill_between(hours, agp[0.10], agp[0.90], color=PALETTE["primary"], alpha=0.14, label="10th–90th Percentile")
    ax.fill_between(hours, agp[0.25], agp[0.75], color=PALETTE["primary"], alpha=0.30, label="25th–75th Percentile (IQR)")
    ax.plot(hours, agp[0.50], color=PALETTE["primary"], linewidth=2.4, label="Circadian Median")
    ax.axhline(180, color=PALETTE["warn"], linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(70, color=PALETTE["warn"], linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.set_ylim(55, 310)
    ax.set_xlabel("Time of Day (Europe/Rome)")
    ax.set_ylabel("CGM Glucose (mg/dL)")
    ax.set_title(f"Ambulatory Glucose Profile (n = {len(cgm):,} 5-min readings)")
    ax.legend(loc="upper left", frameon=True, fontsize=8.3)

    # Right: 48-hour high-resolution CGM + meal & correction events
    ax2 = axes[1]
    ax2.axhspan(70, 180, color=PALETTE["accent"], alpha=0.09)
    ax2.plot(trace["datetime"], trace["glucose"], color=PALETTE["primary"], linewidth=1.8, label="5-min CGM Trace")
    if not meals.empty:
        meal_y = np.interp(
            meals["datetime"].astype("int64"),
            trace["datetime"].astype("int64"),
            trace["glucose"],
        )
        ax2.scatter(
            meals["datetime"],
            meal_y,
            color=PALETTE["warn"],
            s=48,
            zorder=5,
            edgecolors="white",
            linewidth=0.6,
            label="Meal Bolus (Carbs + Humalog)",
        )
    if not manual_corr.empty:
        corr_y = np.interp(
            manual_corr["datetime"].astype("int64"),
            trace["datetime"].astype("int64"),
            trace["glucose"],
        )
        ax2.scatter(
            manual_corr["datetime"],
            corr_y,
            color=PALETTE["dark"],
            marker="^",
            s=40,
            zorder=5,
            label="Manual Correction Bolus",
        )
    ax2.axhline(180, color=PALETTE["warn"], linestyle="--", linewidth=1.0, alpha=0.8)
    ax2.set_ylim(55, 310)
    ax2.set_xlabel("Timestamp (48h Sample Inside Sep–Oct Test Window)")
    ax2.set_ylabel("CGM Glucose (mg/dL)")
    ax2.set_title("High-Resolution 5-Minute CGM & Tandem Insulin Pump Events")
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax2.legend(loc="upper right", frameon=True, fontsize=8.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_intraday_benchmark_and_uncertainty(out_path: Path) -> Path:
    """Plot verified v1/v2/v3 multi-horizon MAE and 90% adaptive uncertainty coverage.

    Values are the audited test-set metrics across 4,878 Sep-Oct origins documented
    in `intraday_forecasting/RESUME.md` (verified by 63 tests and SHA-256 manifests).
    """
    horizons = np.array([30, 60, 120, 180])
    horizon_labels = ["30 min", "60 min", "120 min", "180 min"]

    mae_t24 = np.array([13.23, 22.31, 34.18, 41.57])
    mae_t28d = np.array([11.57, 18.71, 25.29, 27.82])
    mae_v2 = np.array([11.30, 18.06, 24.74, 27.21])
    mae_v3 = np.array([11.04, 17.58, 23.69, 25.86])

    cov_global = np.array([89.99, 90.13, 90.84, 89.51])
    cov_rapid = np.array([92.51, 87.57, 89.01, 85.71])
    widths_v3 = np.array([46.70, 74.86, 104.26, 108.51])

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))

    # Left: Multi-horizon MAE progression
    ax = axes[0]
    ax.plot(horizons, mae_t24, marker="s", linestyle="--", color=PALETTE["muted"], linewidth=1.8, label="Timestral 28M (glucose, 24h)")
    ax.plot(horizons, mae_t28d, marker="^", linestyle="-.", color=PALETTE["gold"], linewidth=2.0, label="Timestral 28M (covariates, 28d)")
    ax.plot(horizons, mae_v2, marker="o", color=PALETTE["primary"], linewidth=2.1, label="Personal Ensemble v2 (CatBoost + TCN)")
    ax.plot(horizons, mae_v3, marker="D", color=PALETTE["accent"], linewidth=2.6, label="Personal Ensemble v3 (25 models + Rolling)")

    for h, v3_val, t24_val in zip(horizons, mae_v3, mae_t24):
        pct = (v3_val - t24_val) / t24_val * 100
        ax.annotate(
            f"{v3_val:.2f}\n({pct:.0f}%)",
            (h, v3_val),
            textcoords="offset points",
            xytext=(0, -24),
            ha="center",
            fontsize=8.2,
            fontweight="bold",
            color=PALETTE["accent"],
        )

    ax.set_xticks(horizons)
    ax.set_xticklabels(horizon_labels)
    ax.set_ylim(5, 46)
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Test MAE (mg/dL) — Lower is Better")
    ax.set_title("Multi-Horizon Forecasting Benchmark (4,878 Test Origins)")
    ax.legend(loc="upper left", frameon=True, fontsize=8.4)

    # Right: 90% Adaptive Uncertainty Calibration + Interval Width
    ax2 = axes[1]
    x = np.arange(len(horizons))
    w = 0.34
    b1 = ax2.bar(x - w / 2, cov_global, width=w, color=PALETTE["primary"], alpha=0.88, label="Global Coverage (all origins)")
    b2 = ax2.bar(x + w / 2, cov_rapid, width=w, color=PALETTE["warn"], alpha=0.85, label="Rapid-Change Regime Coverage")
    ax2.axhline(90.0, color="#2b2d42", linestyle="--", linewidth=1.2, label="Nominal 90% Target")

    for rect in list(b1) + list(b2):
        h_val = rect.get_height()
        ax2.text(
            rect.get_x() + rect.get_width() / 2.0,
            h_val + 0.7,
            f"{h_val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )

    # Annotate v3 mean interval widths along bottom
    for idx, width_val in enumerate(widths_v3):
        ax2.text(
            idx,
            72.5,
            f"width:\n{width_val:.0f} mg/dL",
            ha="center",
            va="center",
            fontsize=7.8,
            color="#343a40",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#f8f9fa", "edgecolor": "#dee2e6"},
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(horizon_labels)
    ax2.set_ylim(68, 99)
    ax2.set_xlabel("Forecast Horizon")
    ax2.set_ylabel("Empirical Coverage (%)")
    ax2.set_title("90% Adaptive Uncertainty Calibration (Ensemble v3)")
    ax2.legend(loc="upper right", frameon=True, fontsize=8.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    _apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p1 = plot_intraday_agp_and_trace(FIGURES_DIR / "05_intraday_agp_profile.png")
    p2 = plot_intraday_benchmark_and_uncertainty(FIGURES_DIR / "06_intraday_benchmark_uncertainty.png")
    print(f"Saved figure: {p1}")
    print(f"Saved figure: {p2}")


if __name__ == "__main__":
    main()
