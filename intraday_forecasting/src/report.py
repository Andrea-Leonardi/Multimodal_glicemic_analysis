"""Standalone figures and an Italian experiment report."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def markdown_table(frame):
    frame = frame.copy()
    for c in frame.select_dtypes("number"):
        integer = pd.api.types.is_integer_dtype(frame[c])
        frame[c] = frame[c].map(lambda x: (str(int(x)) if integer else f"{x:.2f}") if pd.notna(x) else "—")
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))


def write_report(output, summary, audit, config):
    import os
    from src.data import ROOT
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for (model, mode), d in summary.groupby(["model", "mode"], sort=False):
        label = f"{model} / {mode}"
        axes[0].plot(d.horizon_minutes, d.mae, marker="o", label=label)
        axes[1].plot(d.horizon_minutes, d.coverage90 * 100, marker="o", label=label)
    axes[0].set(ylabel="MAE (mg/dL)", xlabel="Orizzonte (minuti)", title="Errore sul test settembre–ottobre")
    axes[1].axhline(90, color="black", linestyle="--", linewidth=1)
    axes[1].set(ylabel="Copertura osservata (%)", xlabel="Orizzonte (minuti)", title="Fascia nominale al 90%, calibrata su agosto")
    for ax in axes:
        ax.set_xticks(config["horizons_minutes"])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(output / "comparison.png", dpi=170)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    selected = json.loads((output / "selection.json").read_text())["selected"]
    # Deterministic example: first test origin for each chosen configuration.
    # It is NOT selected because the resulting forecast looks good.
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    actual_plotted = False
    for choice in selected:
        name, mode, hours = [choice[k] for k in ["model", "mode", "context_hours"]]
        if mode != "glucose":
            continue
        with np.load(output / f"test_{name}_{mode}_{hours}h.npz") as item:
            pred = item["quantiles"][0]
            position = int(item["positions"][0])
            origin = pd.Timestamp(item["origins_utc"][0], tz="UTC").tz_convert(config["timezone"])
        radius = json.loads((output / f"calibration_{name}_{mode}.json").read_text())["0.9"]["radius"]
        minute = np.arange(1, len(pred) + 1) * 5
        line, = ax.plot(minute, pred[:, 10], label=name)
        ax.fill_between(minute, pred[:, 1] - radius, pred[:, -2] + radius, color=line.get_color(), alpha=.10)
        if not actual_plotted:
            from src.data import ROOT
            grid = pd.read_csv(ROOT / "data/processed/timeline.csv")
            ax.plot(minute, grid.glucose.iloc[position + 1:position + len(pred) + 1], color="black", linewidth=2, label="Osservata")
            actual_plotted = True
    ax.set(xlabel="Minuti dalla previsione", ylabel="Glicemia (mg/dL)", title=f"Prima origine del test: {origin:%d/%m/%Y %H:%M %Z}")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(output / "first_forecast.png", dpi=170)
    plt.close(fig)
    columns = ["model", "mode", "horizon_minutes", "n", "mae", "rmse", "coverage80", "coverage90", "width90"]
    text = """# Esperimento storico TimestralGluco

Previsione personale sperimentale della glicemia. Nessun calcolo di dosi e nessun comando alla pompa.

## Protocollo

- Selezione dei contesti su luglio 2026; calibrazione su agosto; test su settembre–ottobre.
- Le previsioni sono salvate prima di estrarre gli esiti corrispondenti per lo scoring.
- Griglia UTC a 5 minuti: ultimo campione realmente osservato in `(t−5 minuti,t]`. Celle vuote senza interpolazione. Il timestamp effettivo del target è conservato.
- Confronto su origini comuni, selezionate esclusivamente da dati passati. Ogni orizzonte resta nel proprio split.
- Ritardo CGM assunto zero; ritardi reali da misurare. Questo primo risultato non riproduce ancora la latenza misurata di Share/Follow.
- Le origini di calibrazione/test distano un'ora dalla successiva; in validazione si confrontano i contesti ogni tre ore per contenere il calcolo CPU. Tutte le curve hanno 36 passi. Non si simulano ancora aggiornamenti a ogni lettura/evento.
- Baseline di persistenza e tendenza recente smorzata (15 minuti di tendenza, smorzamento fissato a 30 minuti).
- Contesti Timestral scelti sul MAE medio dei quattro orizzonti; nessun fine-tuning.
- Intervalli 80%/90% calibrati per singolo passo, tramite residui assoluti per baseline e ampliamento CQR per Timestral.
- La copertura riguarda i singoli punti, non l'intera curva contemporaneamente. Dipendenza temporale e cambiamenti della distribuzione impediscono di interpretarla come garanzia.
- Intervalli di variabilità del MAE e del miglioramento sulla persistenza tramite bootstrap di giornate intere, con campioni appaiati. Dipendenze tra giorni possono restare.

## Significato dei confronti

- `glucose`: sola glicemia, disponibile dal percorso Dexcom previsto.
- `manual`: glicemia, carboidrati interpretati come mangiati e boli manuali confermati. Assume inserimento completo e disponibilità al completamento del bolo, non anticipa pasti futuri.
- `pump`: aggiunge boli automatici e basale storica; è un confronto retrospettivo con più informazioni, non un ingresso attualmente disponibile dal vivo.
- I carboidrati inseriti per richiedere insulina extra sono esclusi dai carboidrati-pasto secondo le regole personali confermate dall'utente. Dati originali ed etichette inferite sono conservati separatamente.
- Un bolo esteso compare come totale confermato solo al completamento; non ne viene inventato il profilo di erogazione minuto per minuto. Questa scelta ritarda l'informazione su pasto e insulina: il confronto manuale storico è un'approssimazione dell'inserimento dal vivo.
- Non sono disponibili pasti pianificati, IOB manuale storico né i tempi originali di arrivo dei dati.
- Le previsioni sono condizionate dal funzionamento storico di Control-IQ; le sue azioni future non sono fornite al modello.

## Risultati sul test

"""
    text += markdown_table(summary[columns])
    reference = summary.loc[summary.model.eq("persistence")].set_index("horizon_minutes")
    small = summary.loc[summary.model.eq("timestral-28m") & summary["mode"].eq("glucose")].set_index("horizon_minutes")
    if not small.empty:
        text += "\n\n### Interpretazione del primo esperimento\n\n"
        for horizon in config["horizons_minutes"]:
            base, value = reference.loc[horizon, "mae"], small.loc[horizon, "mae"]
            text += f"- A {horizon} minuti, Timestral 28M con sola glicemia passa da {base:.2f} a {value:.2f} mg/dL di MAE rispetto alla persistenza ({100 * (base - value) / base:.1f}% di riduzione).\n"
        text += f"- La fascia al 90% del 28M con sola glicemia è larga in media {small.loc[180, 'width90']:.1f} mg/dL a tre ore: il miglioramento dell'errore medio coesiste con un'incertezza ampia.\n"
        for model in ["timestral-28m", "timestral-209m"]:
            univariate = summary.loc[summary.model.eq(model) & summary["mode"].eq("glucose")].set_index("horizon_minutes")
            manual = summary.loc[summary.model.eq(model) & summary["mode"].eq("manual")].set_index("horizon_minutes")
            if not manual.empty:
                difference = (univariate.mae - manual.mae).abs().max()
                text += f"- {model}: l'aggiunta degli ingressi manuali cambia il MAE di al massimo {difference:.2f} mg/dL nei quattro orizzonti. È un contributo piccolo in questo test zero-shot; non dimostra che gli eventi siano inutili dopo un adattamento o con registrazioni migliori.\n"
        models = summary.loc[summary.model.str.startswith("timestral")]
        text += f"- Fra tutte le varianti Timestral, la copertura osservata delle fasce nominali al 90% va da {models.coverage90.min() * 100:.1f}% a {models.coverage90.max() * 100:.1f}%.\n"
        text += "- L'assenza di glicemie basse limita la valutazione: nessuna conclusione sulle prestazioni sotto 70 mg/dL.\n"
    text += "\n\n![Confronto](comparison.png)\n\n![Prima previsione](first_forecast.png)\n"
    text += "\n## Qualità dei dati e limiti\n\n"
    text += f"- {audit['cgm']['rows']} letture CGM, {audit['cgm']['missing_grid_cells']} celle mancanti dopo la griglia; alcuni campioni ricadono nello stesso intervallo.\n"
    text += f"- Minimo CGM {audit['cgm']['min_mgdl']:.0f} mg/dL: nessun esempio sotto 70 mg/dL. Le metriche in questa fascia non sono stimabili.\n"
    text += f"- Recuperati {audit['bolus']['interrupted']} boli interrotti e conservati {audit['bolus']['extended']} boli estesi.\n"
    text += "- Un timestamp basale discordante è marcato mancante, senza sommare o scegliere arbitrariamente i due valori.\n"
    text += "- Gli eventi pasto sono inferiti; i confronti vicino ai pasti misurano le etichette disponibili, non pasti verificati indipendentemente.\n"
    text += "- I risultati sul test sono ora consultati: ulteriori scelte basate su questi risultati richiederanno un nuovo periodo futuro per una valutazione finale indipendente.\n"
    text += "\n## Artefatti\n\n`manifest.json`: configurazione, hash sorgenti/codice e versioni. `selection.json`: tutte le scelte su validazione. `*.npz`: curve complete a 21 quantili prima dello scoring. `test_predictions.csv.gz`: esiti e intervalli per i quattro orizzonti. `metrics.csv`: errori, copertura, ampiezza e bootstrap. `cohorts.csv`: sottogruppi, inclusi quelli con zero osservazioni. `calibration_diagnostics_in_sample.csv.gz`: diagnostica su agosto, non valutazione indipendente.\n"
    text += "\n## Fonti tecniche\n\n[Timestral ufficiale](https://github.com/mistralai/timestral), [Mistral](https://mistral.ai/news/timestral/). Codice e pesi sono fissati alle revisioni nel manifest; inferenza locale CPU. La procedura di fine-tuning locale non è ancora implementata/verificata.\n"
    (output / "report.md").write_text(text)
