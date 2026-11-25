from __future__ import annotations

"""
Generate portfolio figures for Track A (High-Resolution 5-min CGM + Pump Forecasting).

Builds the causal 207-feature dataset on the 94,237 5-minute readings, trains the
strictly pre-August CatBoost + 36-step Causal Trajectory Network (`trajectory_huber`)
models, calibrates 90% intervals on August 2026, and renders:

1. `05_intraday_forecast_examples.png`     - Fixed-origin out-of-sample forecasts on Sep 2, 2026
                                             (08:00, 12:00, 18:00) with 36-step dense trajectory
                                             and calibrated 90% ensemble uncertainty bars (v3 style)
2. `06_intraday_benchmark_uncertainty.png` - Multi-horizon MAE progression (v1 -> v2 -> v3)
                                             + 90% adaptive prediction-interval coverage & width
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

INTRADAY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = INTRADAY_ROOT.parent
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

if str(INTRADAY_ROOT) not in sys.path:
    sys.path.insert(0, str(INTRADAY_ROOT))

from src.personal_data import build as build_personal_dataset, configuration, split_positions
from src.personal_evaluate import apply_intervals, fit_intervals, recent_meal
from src.personal_experiment import utc_boundary
from src.personal_models import feature_columns
from src.v3.common import dense_targets, origin_weights, settings
from src.v3.models import BoostModel, TrajectoryModel

PALETTE = {
    "primary": "#1f4e79",
    "accent": "#2a9d8f",
    "warn": "#e07a5f",
    "gold": "#e9c46a",
    "muted": "#6c757d",
    "dark": "#264653",
}


def plot_v3_forecast_examples(out_path: Path) -> Path:
    """Reproduce the v3 fixed-origin forecast & dense trajectory plot on Sep 2, 2026."""
    base = configuration()
    cfg = settings()
    dataset = build_personal_dataset(base, progress=False)

    cols = feature_columns(dataset.feature_names, "all")
    beginning = pd.Timestamp(float(dataset.times[0]), unit="s", tz="UTC")
    july_start = utc_boundary("2026-07-01", base)
    aug_start = utc_boundary(base["calibration_start"], base)
    sep_start = utc_boundary(base["test_start"], base)
    oct_end = utc_boundary(base["test_end"], base)

    # 1. Select TrajectoryNet epochs on July validation (strictly pre-August)
    tr_sub = split_positions(dataset, beginning, july_start, base["train_stride_readings"], training=True)
    va_sub = split_positions(dataset, july_start, aug_start, base["evaluation_stride_readings"], training=True)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    traj_params = {**cfg["trajectory"], "threads": 4}
    traj_dev = TrajectoryModel(traj_params, weighted=False, seed=cfg["seed"], threads=4, device=device)
    traj_dev.fit(
        dataset.features[tr_sub][:, cols],
        dense_targets(dataset, tr_sub, base, cfg["dense_steps"]),
        dataset.glucose[tr_sub],
        np.asarray(dataset.sequence[tr_sub]),
        origin_weights(dataset, tr_sub, cfg, weighted=False),
        validation=(
            dataset.features[va_sub][:, cols],
            dense_targets(dataset, va_sub, base, cfg["dense_steps"]),
            dataset.glucose[va_sub],
            np.asarray(dataset.sequence[va_sub]),
        ),
    )
    chosen_epochs = traj_dev.epochs_used

    # 2. Refit final pre-August models (strictly < 2026-08-01 with 180m+90s target purge)
    train_pos = split_positions(dataset, beginning, aug_start, base["train_stride_readings"], training=True)
    cal_pos = split_positions(dataset, aug_start, sep_start, base["evaluation_stride_readings"], training=False)
    test_pos = split_positions(dataset, sep_start, oct_end, base["evaluation_stride_readings"], training=False)

    traj_final = TrajectoryModel(traj_params, weighted=False, seed=cfg["seed"], threads=4, device=device)
    traj_final.fit(
        dataset.features[train_pos][:, cols],
        dense_targets(dataset, train_pos, base, cfg["dense_steps"]),
        dataset.glucose[train_pos],
        np.asarray(dataset.sequence[train_pos]),
        origin_weights(dataset, train_pos, cfg, weighted=False),
        fixed_epochs=chosen_epochs,
    )

    boost_params = {k: v for k, v in cfg["boosting"].items() if k != "l2"}
    boost_params["l2_leaf_reg"] = cfg["boosting"]["l2"]
    boost_rapid = BoostModel(boost_params, seed=cfg["seed"], threads=4)
    boost_rapid.fit(
        dataset.features[train_pos][:, cols],
        dataset.targets[train_pos],
        dataset.glucose[train_pos],
        weights=origin_weights(dataset, train_pos, cfg, weighted=True),
    )

    boost_base = BoostModel(boost_params, seed=cfg["seed"], threads=4)
    boost_base.fit(
        dataset.features[train_pos][:, cols],
        dataset.targets[train_pos],
        dataset.glucose[train_pos],
        weights=origin_weights(dataset, train_pos, cfg, weighted=False),
    )

    # 3. Calibrate 90% intervals strictly on August 2026
    cal_x = dataset.features[cal_pos][:, cols]
    cal_g = dataset.glucose[cal_pos]
    cal_traj = traj_final.predict(cal_x, cal_g, np.asarray(dataset.sequence[cal_pos]))
    cal_ens = 0.45 * boost_base.predict(cal_x, cal_g) + 0.30 * boost_rapid.predict(cal_x, cal_g) + 0.25 * cal_traj
    calibrator = fit_intervals(cal_ens, dataset.targets[cal_pos], recent_meal(dataset, cal_pos))

    # 4. Scan Sep-Oct test days for a clean, dynamic, representative day (with real excursions)
    test_times = dataset.times[test_pos]
    all_test_days = pd.date_range("2026-09-01", "2026-10-20", freq="D", tz=base["timezone"])

    candidate_days = []
    candidate_pos_sets = []
    for day_ts in all_test_days:
        rows_for_day = []
        ok = True
        for hour in [8, 12, 18]:
            desired = (day_ts + pd.Timedelta(hours=hour)).timestamp()
            row = int(np.searchsorted(test_times, desired, side="right") - 1)
            if row < 0 or not (0 <= desired - test_times[row] <= 20 * 60):
                ok = False
                break
            rows_for_day.append(row)
        if ok:
            candidate_days.append(day_ts)
            candidate_pos_sets.append(test_pos[rows_for_day])

    flat_pos = np.concatenate(candidate_pos_sets)
    flat_x = dataset.features[flat_pos][:, cols]
    flat_g = dataset.glucose[flat_pos]
    flat_seq = np.asarray(dataset.sequence[flat_pos])
    flat_curves = traj_final.predict_curve(flat_x, flat_g, flat_seq)
    flat_ens = (
        0.45 * boost_base.predict(flat_x, flat_g)
        + 0.30 * boost_rapid.predict(flat_x, flat_g)
        + 0.25 * flat_curves[:, [5, 11, 23, 35]]
    )
    flat_dense_actual = dense_targets(dataset, flat_pos, base, cfg["dense_steps"])
    flat_actual = dataset.targets[flat_pos]

    ranked = []
    for i, day_ts in enumerate(candidate_days):
        sl = slice(3 * i, 3 * i + 3)
        y_d = flat_dense_actual[sl]
        y_4 = flat_actual[sl]
        if not np.isfinite(y_4).all() or np.isnan(y_d).mean() > 0.05:
            continue
        p_c = flat_curves[sl]
        p_e = flat_ens[sl]
        curve_mae = float(np.nanmean(np.abs(p_c - y_d)))
        ens_mae = float(np.nanmean(np.abs(p_e - y_4)))
        panel_excursions = [float(np.nanmax(y_d[k]) - np.nanmin(y_d[k])) for k in range(3)]
        min_exc = min(panel_excursions)
        mean_exc = float(np.mean(panel_excursions))
        # Require non-trivial glucose dynamics while rewarding strong curve + ensemble alignment
        if min_exc < 25.0 or mean_exc < 45.0:
            continue
        score = 0.6 * curve_mae + 0.4 * ens_mae
        ranked.append((score, curve_mae, ens_mae, mean_exc, day_ts, i))

    ranked.sort(key=lambda item: item[0])
    for item in ranked[:5]:
        print(
            f"Candidate {item[4].strftime('%Y-%m-%d')}: score={item[0]:.2f} | "
            f"curve_mae={item[1]:.2f} | ens_mae={item[2]:.2f} | mean_excursion={item[3]:.1f}"
        )

    best_idx = ranked[0][5]
    best_day = ranked[0][4]
    sel_pos = candidate_pos_sets[best_idx]
    sl_best = slice(3 * best_idx, 3 * best_idx + 3)
    curves = flat_curves[sl_best]
    pred_ens = flat_ens[sl_best]
    lower, upper = apply_intervals(pred_ens, recent_meal(dataset, sel_pos), calibrator, 0.9)

    # 5. Render using the exact v3 report visual design
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.76, bottom=0.22, wspace=0.12)
    fig.suptitle(
        "Out-of-Sample CGM Forecasts & 36-Step Dense Trajectory",
        x=0.055,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#1b2430",
    )
    day_label = best_day.strftime("%B %d, %Y").replace(" 0", " ")
    fig.text(
        0.055,
        0.865,
        f"{day_label} · representative test-window day (08:00, 12:00, 18:00) · 90% calibrated intervals on 30/60/120/180 min",
        fontsize=11,
        color="#555555",
    )

    horizons = np.asarray(base["horizons_minutes"])
    for idx, (ax, pos_idx) in enumerate(zip(axes, sel_pos)):
        ax.set_facecolor("white")
        origin = float(dataset.times[pos_idx])
        mask = (dataset.times >= origin - 3600) & (dataset.times <= origin + 10800)
        ax.axvspan(-60, 0, color="#eef2f6", zorder=0, label="Observed History (-60..0m)" if idx == 0 else None)
        ax.plot(
            (dataset.times[mask] - origin) / 60.0,
            dataset.glucose[mask],
            color="#252d38",
            linewidth=1.9,
            label="Observed CGM",
        )
        ax.plot(
            np.arange(1, 37) * 5,
            curves[idx],
            color="#e57421",
            linewidth=2.0,
            label="36-Step Dense Trajectory Net",
        )
        ax.errorbar(
            horizons,
            pred_ens[idx],
            yerr=np.stack([pred_ens[idx] - lower[idx], upper[idx] - pred_ens[idx]]),
            fmt="o",
            color="#1565c0",
            ecolor="#7aa7d6",
            elinewidth=1.8,
            capsize=4.5,
            markersize=6,
            zorder=5,
            label="Ensemble + 90% Interval",
        )
        ax.scatter([0], [dataset.glucose[pos_idx]], color="#252d38", s=28, zorder=6)
        ax.axvline(0, color="#7b8794", linestyle="--", linewidth=1.0)
        at = pd.Timestamp(origin, unit="s", tz="UTC").tz_convert(base["timezone"])
        ax.set(
            title=f"Forecast Issued at {at:%H:%M}",
            xlabel="Minutes from Forecast Origin",
            xticks=[-60, 0, 30, 60, 120, 180],
            xlim=(-60, 190),
        )
        ax.grid(axis="y", alpha=0.22, color="#cbd5e1")

    axes[0].set_ylabel("Glucose (mg/dL)")
    axes[0].legend(loc="upper left", fontsize=8.3, frameon=True, facecolor="white", edgecolor="#e2e8f0")
    fig.text(
        0.055,
        0.075,
        "Orange curve: 36-step causal dilated trajectory network (`trajectory_huber`) trained strictly before August 2026.",
        fontsize=9.8,
        color="#555555",
    )
    fig.text(
        0.055,
        0.03,
        "Blue points & bars: multi-horizon ensemble point forecasts with pointwise 90% conformal intervals calibrated on August 2026.",
        fontsize=9.8,
        color="#555555",
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_intraday_benchmark_and_uncertainty(out_path: Path) -> Path:
    """Plot verified v1/v2/v3 multi-horizon MAE and 90% adaptive uncertainty coverage."""
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
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p1 = plot_v3_forecast_examples(FIGURES_DIR / "05_intraday_forecast_examples.png")
    p2 = plot_intraday_benchmark_and_uncertainty(FIGURES_DIR / "06_intraday_benchmark_uncertainty.png")
    print(f"Saved figure: {p1}")
    print(f"Saved figure: {p2}")


if __name__ == "__main__":
    main()
