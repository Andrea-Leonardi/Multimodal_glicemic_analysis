"""Plot saved validation forecasts without training or inspecting final-test labels."""

import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.data import ROOT, sha256
from src.personal_data import configuration
from src.personal_experiment import load_dataset, metrics


def main():
    output = ROOT / "outputs/personal-v2"
    folder = output / "validation-figures"
    folder.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    config, dataset = configuration(), load_dataset()
    horizons = np.asarray(config["horizons_minutes"])
    candidates = []
    for stage in ["tabular", "dense-training", "direct-boosting", "hybrid", "timestral", "timestral-extended"]:
        candidates += json.loads((output / f"{stage}-selection.json").read_text())["candidates"]
    best = min(candidates, key=lambda row: np.mean(row["metrics"]["mae"]))
    best_ts = min([r for r in candidates if r["spec"]["family"] == "timestral"], key=lambda r: np.mean(r["metrics"]["mae"]))
    names = [best["spec"]["name"], "tcn_huber", best_ts["spec"]["name"]]
    labels = ["CatBoost personale", "Rete temporale TCN", "Timestral 28M · covariate · 14 giorni"]
    colors = ["#1565c0", "#78909c", "#e57421"]
    predictions, positions, hashes = {}, None, {}
    for name in names:
        path = output / "oof" / (name + ".npz")
        with np.load(path) as saved:
            if positions is None:
                positions = saved["positions"]
            np.testing.assert_array_equal(positions, saved["positions"])
            predictions[name] = saved["prediction"].copy()
            assert predictions[name].shape == (len(positions), 4)
            assert np.isfinite(predictions[name]).all()
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    actual = np.asarray(dataset.targets[positions])
    scores = {name: metrics(actual, pred) for name, pred in predictions.items()}
    for candidate in [best, best_ts]:
        np.testing.assert_allclose(scores[candidate["spec"]["name"]]["mae"], candidate["metrics"]["mae"], atol=1e-5)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    fig.subplots_adjust(left=.09, right=.97, top=.83, bottom=.20)
    fig.suptitle("Quanto sbaglia il modello?", x=.09, ha="left", fontsize=20, fontweight="bold")
    ax.set_title("Validazione su aprile, giugno e luglio · 8.516 origini comuni", loc="left", fontsize=11, color="#555", pad=17)
    for name, label, color in zip(names, labels, colors):
        y = scores[name]["mae"]
        ax.plot(horizons, y, "o-", color=color, label=label, linewidth=2.5, markersize=7)
    for h, value in zip(horizons, scores[names[0]]["mae"]):
        ax.annotate(f"{value:.1f}".replace(".", ","), (h, value), xytext=(0, -20), textcoords="offset points", ha="center", color=colors[0], fontweight="bold")
    ax.set(xlabel="Quanto tempo prima viene fatta la previsione (minuti)", ylabel="Errore assoluto medio — MAE (mg/dL)", xticks=horizons, ylim=(0, 34))
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.text(.09, .065, "Più basso è meglio. Risultati usati per scegliere il modello; valutazione finale ancora da completare.", fontsize=10, color="#555")
    fig.savefig(folder / "accuracy.png", dpi=170)
    fig.savefig(folder / "accuracy.pdf")
    plt.close(fig)

    day = pd.Timestamp("2026-07-02", tz=config["timezone"])
    origins = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True)
    fig.subplots_adjust(left=.055, right=.985, top=.75, bottom=.22, wspace=.12)
    fig.suptitle("Previsioni e glicemia realmente osservata", x=.055, ha="left", fontsize=20, fontweight="bold")
    fig.text(.055, .87, "CatBoost personale · tre orari prefissati del 2 luglio 2026 · ogni pannello è una singola previsione", fontsize=11, color="#555")
    example_rows = []
    for ax, hour in zip(axes, [8, 12, 18]):
        wanted = (day + pd.Timedelta(hours=hour)).timestamp()
        row = np.searchsorted(dataset.times[positions], wanted, side="right") - 1
        index = int(positions[row])
        origin = float(dataset.times[index])
        assert 0 <= wanted - origin <= 20 * 60
        origins.append(index)
        interval = np.flatnonzero((dataset.times >= origin - 3600) & (dataset.times <= origin + 10800))
        minutes = (dataset.times[interval] - origin) / 60
        ax.axvspan(-60, 0, color="#eef2f6", zorder=0)
        ax.plot(minutes, dataset.glucose[interval], color="#252d38", linewidth=1.8, label="Glicemia osservata")
        ax.scatter(horizons, predictions[names[0]][row], color=colors[0], s=64, zorder=4, label="Previsioni CatBoost")
        ax.scatter([0], [dataset.glucose[index]], color="#252d38", s=30, zorder=4)
        ax.axvline(0, color="#7b8794", linestyle="--", linewidth=1)
        local = pd.Timestamp(origin, unit="s", tz="UTC").tz_convert(config["timezone"])
        ax.set(title=f"Emessa alle {local:%H:%M}", xlabel="Minuti dall'emissione", xticks=[-60, 0, 30, 60, 120, 180], xlim=(-60, 190))
        ax.grid(axis="y", alpha=.18)
        for k, h in enumerate(horizons):
            example_rows.append({"origin": local.isoformat(), "horizon_minutes": int(h), "prediction_mgdl": float(predictions[names[0]][row, k]), "actual_target_mgdl": float(actual[row, k])})
    axes[0].set_ylabel("Glicemia (mg/dL)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(.055, .08, "I quattro punti sono le previsioni a 30/60/120/180 minuti; la linea scura è la glicemia misurata.", fontsize=10, color="#555")
    fig.text(.055, .035, "Esempi scelti per orario, senza cercare previsioni riuscite. Fasce di incertezza ancora da calibrare.", fontsize=10, color="#555")
    fig.savefig(folder / "forecast-examples.png", dpi=160)
    fig.savefig(folder / "forecast-examples.pdf")
    plt.close(fig)
    pd.DataFrame(example_rows).to_csv(folder / "forecast-examples.csv", index=False)

    # Each plotted forecast here has a different origin and the indicated fixed
    # lead time. Matching target timestamps are used instead of nominal slots.
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.98, top=.86, bottom=.13, hspace=.19)
    fig.suptitle("Un giorno intero, quattro orizzonti", x=.08, ha="left", fontsize=20, fontweight="bold")
    fig.text(.08, .915, "2 luglio 2026 · CatBoost personale · previsioni aggiornate circa ogni 15 minuti", fontsize=11, color="#555")
    day_end = day + pd.Timedelta(days=1)
    visible = (dataset.times >= day.timestamp()) & (dataset.times < day_end.timestamp())
    clock = pd.to_datetime(dataset.times[visible], unit="s", utc=True).tz_convert(config["timezone"])
    for k, ax in enumerate(axes):
        target_index = dataset.target_indices[positions, k]
        valid = target_index >= 0
        target_seconds = dataset.times[np.maximum(target_index, 0)]
        mask = valid & (target_seconds >= day.timestamp()) & (target_seconds < day_end.timestamp())
        target_clock = pd.to_datetime(target_seconds[mask], unit="s", utc=True).tz_convert(config["timezone"])
        ax.plot(clock, dataset.glucose[visible], color="#252d38", linewidth=1.5, label="Glicemia osservata")
        ax.plot(target_clock, predictions[names[0]][mask, k], "o-", color=colors[0], linewidth=1.2, markersize=3, label=f"Previsione fatta {horizons[k]} min prima")
        ax.set_ylabel("mg/dL")
        ax.legend(loc="upper left", ncol=2, fontsize=9, frameon=False)
        ax.grid(alpha=.15)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=3, tz=ZoneInfo(config["timezone"])))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZoneInfo(config["timezone"])))
    axes[-1].set(xlim=(day, day_end), xlabel="Orario della glicemia da prevedere · Europe/Rome")
    fig.text(.08, .045, "Ogni punto blu proviene da una previsione diversa. Il modello usa solo ciò che è noto alla sua emissione.", fontsize=10, color="#555")
    fig.savefig(folder / "rolling-day.png", dpi=160)
    fig.savefig(folder / "rolling-day.pdf")
    plt.close(fig)
    details = {"best_completed_validation_model": best["spec"], "selection_criterion": "Mean MAE over four horizons, completed dense validation comparisons only",
               "origin_count": len(positions), "metrics": scores, "example_day_fixed": day.isoformat(),
               "example_hours_fixed": [8, 12, 18], "example_origin_indices": origins, "input_sha256": hashes,
               "plot_code_sha256": sha256(Path(__file__)), "scope": "Saved validation predictions only. No retraining, test scoring, uncertainty calibration or resumption of interrupted search."}
    (folder / "metadata.json").write_text(json.dumps(details, indent=2))
    print(json.dumps({"folder": str(folder), "best": names[0], "mae": scores[names[0]]["mae"], "images": ["accuracy.png", "forecast-examples.png", "rolling-day.png"]}, indent=2))


if __name__ == "__main__":
    main()
