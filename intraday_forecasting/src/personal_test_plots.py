"""Fixed-date final-test examples with calibrated pointwise 90% intervals.

Only reads saved predictions/calibration; never chooses or fits a model.
"""

import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.data import ROOT, sha256
from src.personal_experiment import load_dataset
from src.personal_evaluate import recent_meal, apply_intervals


def main():
    output = ROOT / "outputs/personal-v2"
    complete = json.loads((output / "completed.json").read_text())
    config = complete["fingerprint"]["config"]
    dataset = load_dataset()
    path = output / "predictions/test_selected_personal.npz"
    with np.load(path) as saved:
        positions, prediction = saved["positions"], saved["prediction"]
    interval_path = output / "predictions/intervals_selected_personal.json"
    calibrator = json.loads(interval_path.read_text())
    lo, hi = apply_intervals(prediction, recent_meal(dataset, positions), calibrator)
    day = pd.Timestamp("2026-09-02", tz=config["timezone"])
    horizons = np.asarray(config["horizons_minutes"])
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    examples = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.9), sharey=True)
    fig.subplots_adjust(left=.055, right=.985, top=.75, bottom=.23, wspace=.12)
    fig.suptitle("Previsioni finali con intervalli di incertezza", x=.055, ha="left", fontsize=20, fontweight="bold")
    fig.text(.055, .87, "2 settembre 2026 · tre orari prefissati · modello scelto sulle sole validazioni", fontsize=11, color="#555")
    for ax, hour in zip(axes, [8, 12, 18]):
        desired = (day + pd.Timedelta(hours=hour)).timestamp()
        row = int(np.searchsorted(dataset.times[positions], desired, side="right") - 1)
        assert row >= 0
        index = int(positions[row])
        origin = float(dataset.times[index])
        assert 0 <= desired - origin <= 20 * 60
        visible = np.flatnonzero((dataset.times >= origin - 3600) & (dataset.times <= origin + 10800))
        ax.axvspan(-60, 0, color="#eef2f6", zorder=0)
        ax.plot((dataset.times[visible] - origin) / 60, dataset.glucose[visible], color="#252d38", linewidth=1.8, label="Glicemia osservata")
        ax.errorbar(horizons, prediction[row], yerr=np.stack([prediction[row] - lo[row], hi[row] - prediction[row]]),
                    fmt="o", color="#1565c0", ecolor="#7aa7d6", capsize=5, linewidth=1.8, markersize=6, label="Previsione e fascia 90%")
        ax.axvline(0, color="#7b8794", linestyle="--", linewidth=1)
        ax.scatter([0], [dataset.glucose[index]], color="#252d38", s=30, zorder=4)
        local = pd.Timestamp(origin, unit="s", tz="UTC").tz_convert(config["timezone"])
        ax.set(title=f"Emessa alle {local:%H:%M}", xlabel="Minuti dall'emissione", xticks=[-60, 0, 30, 60, 120, 180], xlim=(-60, 190))
        ax.grid(axis="y", alpha=.18)
        examples.append({"position": index, "origin": local.isoformat(), "prediction_mgdl": prediction[row].tolist(),
                         "lower90": lo[row].tolist(), "upper90": hi[row].tolist()})
    axes[0].set_ylabel("Glicemia (mg/dL)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.text(.055, .09, "Ogni pannello mostra quattro previsioni emesse insieme. Le barre sono intervalli per il singolo orizzonte, calibrati su agosto.", fontsize=10, color="#555")
    fig.text(.055, .04, "Esempi fissati per data e orario, senza scegliere gli errori più piccoli. Test retrospettivo già consultato nella fase v1.", fontsize=10, color="#555")
    fig.savefig(output / "test-examples.png", dpi=160)
    fig.savefig(output / "test-examples.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.98, top=.86, bottom=.13, hspace=.19)
    fig.suptitle("Test finale: un giorno, quattro orizzonti", x=.08, ha="left", fontsize=20, fontweight="bold")
    fig.text(.08, .915, "2 settembre 2026 · modello selezionato · previsioni aggiornate circa ogni 15 minuti", fontsize=11, color="#555")
    day_end = day + pd.Timedelta(days=1)
    visible = (dataset.times >= day.timestamp()) & (dataset.times < day_end.timestamp())
    clock = pd.to_datetime(dataset.times[visible], unit="s", utc=True).tz_convert(config["timezone"])
    for k, ax in enumerate(axes):
        target_index = dataset.target_indices[positions, k]
        target_seconds = dataset.times[np.maximum(target_index, 0)]
        mask = (target_index >= 0) & (target_seconds >= day.timestamp()) & (target_seconds < day_end.timestamp())
        target_clock = pd.to_datetime(target_seconds[mask], unit="s", utc=True).tz_convert(config["timezone"])
        ax.plot(clock, dataset.glucose[visible], color="#252d38", linewidth=1.5, label="Glicemia osservata")
        ax.plot(target_clock, prediction[mask, k], "o-", color="#1565c0", linewidth=1.2, markersize=3, label=f"Previsione fatta {horizons[k]} min prima")
        ax.set_ylabel("mg/dL")
        ax.legend(loc="upper left", ncol=2, fontsize=9, frameon=False)
        ax.grid(alpha=.15)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=3, tz=ZoneInfo(config["timezone"])))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZoneInfo(config["timezone"])))
    axes[-1].set(xlim=(day, day_end), xlabel="Orario della glicemia da prevedere · Europe/Rome")
    fig.text(.08, .045, "Ogni punto blu proviene da un'origine diversa. La figura mostra errori osservati; non viene usata per cambiare modelli o pesi.", fontsize=10, color="#555")
    fig.savefig(output / "test-day.png", dpi=160)
    fig.savefig(output / "test-day.pdf")
    plt.close(fig)
    metadata = {"date_fixed_before_test_scoring": day.isoformat(), "hours_fixed": [8, 12, 18], "examples": examples,
                "prediction_sha256": sha256(path), "intervals_sha256": sha256(interval_path), "plot_code_sha256": sha256(Path(__file__))}
    (output / "test-examples.json").write_text(json.dumps(metadata, indent=2))
    report = output / "report.md"
    section = "\n## Esempi a orari prefissati nel test\n\nIl 2 settembre e gli orari 08/12/18 sono fissati nel codice prima dello scoring finale, senza selezionare esempi per errore. Le barre indicano intervalli per il singolo orizzonte. Nel grafico giornaliero ogni punto è una previsione diversa.\n\n![Tre previsioni con intervalli](test-examples.png)\n\n![Giornata completa](test-day.png)\n"
    if "## Esempi a orari prefissati nel test" not in report.read_text():
        report.write_text(report.read_text() + section)
    print("Grafici finali generati: test-examples.png/pdf e test-day.png/pdf", flush=True)


if __name__ == "__main__":
    main()
