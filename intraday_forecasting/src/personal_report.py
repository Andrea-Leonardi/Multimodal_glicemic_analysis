"""Italian report and standalone figures for the supervised historical study."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ROOT
from src.report import markdown_table


def write_report(output, dataset, config, selection, summary):
    output = Path(output)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = summary.loc[summary.model.eq("selected_personal")].set_index("horizon_minutes")
    baseline = summary.loc[summary.model.eq("timestral-28m_glucose_24h")].set_index("horizon_minutes")
    wanted = ["persistence", "timestral-28m_glucose_24h", "selected_personal"]
    wanted += [c["spec"]["name"] for c in selection["selected"] if c["spec"]["family"] == "timestral"]
    wanted += [c["spec"]["name"] for c in selection["selected"] if c["spec"]["family"] in {"catboost", "hist", "extra_trees"} and not c["spec"].get("features") and not c["spec"].get("base")]
    wanted = list(dict.fromkeys(wanted))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for name in wanted:
        d = summary.loc[summary.model.eq(name)]
        axes[0].plot(d.horizon_minutes, d.mae, marker="o", linewidth=2 if name == "selected_personal" else 1, label=name)
        axes[1].plot(d.horizon_minutes, d.width90, marker="o", linewidth=2 if name == "selected_personal" else 1)
    axes[0].set(xlabel="Orizzonte (minuti)", ylabel="MAE (mg/dL)", title="Stesse origini e stessi target per tutti i modelli")
    axes[1].set(xlabel="Orizzonte (minuti)", ylabel="Ampiezza media (mg/dL)", title="Fasce 90%, calibrate su agosto per regime")
    axes[0].legend(fontsize=6.5)
    for ax in axes:
        ax.set_xticks(config["horizons_minutes"])
        ax.grid(alpha=.2)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    cohorts = pd.read_csv(output / "test-cohorts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for name in ["selected_personal", "timestral-28m_glucose_24h"]:
        for cohort, style in [("recent_meal", "-"), ("no_recent_meal", "--")]:
            d = cohorts.loc[cohorts.model.eq(name) & cohorts.cohort.eq(cohort)]
            axes[0].plot(d.horizon_minutes, d.mae, linestyle=style, marker="o", label=f"{name} / {cohort}")
            axes[1].plot(d.horizon_minutes, d.coverage90 * 100, linestyle=style, marker="o")
    axes[0].set(xlabel="Minuti", ylabel="MAE (mg/dL)", title="Errori dopo pasti e negli altri periodi")
    axes[1].set(xlabel="Minuti", ylabel="Copertura (%)", title="Copertura per regime della fascia nominale 90%")
    axes[1].axhline(90, color="black", linestyle=":")
    axes[0].legend(fontsize=6.5)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xticks(config["horizons_minutes"])
    fig.savefig(output / "regimes.png", dpi=180)
    plt.close(fig)
    # Fixed example: first eligible test origin, never selected for its error.
    with np.load(output / "predictions/test_selected_personal.npz") as saved:
        first = int(saved["positions"][0])
        personal_prediction = saved["prediction"][0]
    with np.load(output / "predictions/test_timestral-28m_glucose_24h.npz") as saved:
        timestral_prediction = saved["prediction"][0]
    intervals = json.loads((output / "predictions/intervals_selected_personal.json").read_text())
    regime = int(dataset.features[first, dataset.feature_names.index("carbs_sum_180m")] > 0)
    radius = np.asarray(intervals["0.9"]["radii"])[regime]
    ending = np.searchsorted(dataset.times, dataset.times[first] + 180 * 60, side="right")
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.plot((dataset.times[first:ending] - dataset.times[first]) / 60, dataset.glucose[first:ending], color="black", label="Glicemia osservata")
    ax.errorbar(config["horizons_minutes"], personal_prediction, yerr=radius, fmt="o", capsize=4, label="Personale: quattro previsioni e intervalli 90%")
    ax.plot(config["horizons_minutes"], timestral_prediction, "s--", label="Timestral 28M / 24h")
    origin = pd.Timestamp(float(dataset.times[first]), unit="s", tz="UTC").tz_convert(config["timezone"])
    ax.set(xlabel="Minuti dall'origine", ylabel="mg/dL", title=f"Prima origine del test: {origin:%d/%m/%Y %H:%M}")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(output / "first-origin.png", dpi=180)
    plt.close(fig)
    cv_records = []
    for stage in ["tabular", "hybrid", "timestral"]:
        record = json.loads((output / f"{stage}-selection.json").read_text())
        for candidate in record["candidates"]:
            for k, h in enumerate(config["horizons_minutes"]):
                cv_records.append({"model": candidate["spec"]["name"], "horizon_minutes": h, "validation_mae": candidate["metrics"]["mae"][k]})
    extended_path = output / "timestral-extended-selection.json"
    extended = json.loads(extended_path.read_text()) if extended_path.exists() else {"candidates": [], "candidates_sparse": []}
    long_path = output / "timestral-long-selection.json"
    if long_path.exists():
        longer = json.loads(long_path.read_text())
        extended["candidates"] += longer["candidates"]
        extended["candidates_sparse"] += longer["candidates_sparse"]
    for candidate in extended["candidates"]:
        for k, h in enumerate(config["horizons_minutes"]):
            cv_records.append({"model": candidate["spec"]["name"], "horizon_minutes": h, "validation_mae": candidate["metrics"]["mae"][k]})
    dense_path = output / "dense-training-selection.json"
    if dense_path.exists():
        for candidate in json.loads(dense_path.read_text())["candidates"]:
            for k, h in enumerate(config["horizons_minutes"]):
                cv_records.append({"model": candidate["spec"]["name"], "horizon_minutes": h, "validation_mae": candidate["metrics"]["mae"][k]})
    direct_path = output / "direct-boosting-selection.json"
    if direct_path.exists():
        for candidate in json.loads(direct_path.read_text())["candidates"]:
            for k, h in enumerate(config["horizons_minutes"]):
                cv_records.append({"model": candidate["spec"]["name"], "horizon_minutes": h, "validation_mae": candidate["metrics"]["mae"][k]})
    cv = pd.DataFrame(cv_records)
    cv.to_csv(output / "validation-metrics.csv", index=False)
    context_search = json.loads((output / "timestral-selection.json").read_text())["candidates_sparse"] + extended["candidates_sparse"]
    context_rows = [{"model": c["spec"]["name"], **{f"MAE_{h}min": c["metrics"]["mae"][k] for k, h in enumerate(config["horizons_minutes"])}} for c in context_search]
    context_table = pd.DataFrame(context_rows)
    context_table.to_csv(output / "timestral-context-search.csv", index=False)
    text = """# Modello personale storico — esperimento v2

## Risultato e perimetro

Modelli addestrati sullo storico personale, senza componente real time. L'orario del pasto è considerato l'orario effettivo del consumo, secondo la conferma dell'utente. Le regole personali separano carboidrati mangiati e carboidrati inseriti per ottenere insulina extra. Il dominio valutato è quello osservato, a partire da 75 mg/dL.

`selected_personal` è scelto prima del test usando solo le validazioni; può combinare modelli diversi ai quattro orizzonti. Il test settembre–ottobre è riutilizzato dopo v1 e la discussione dei suoi risultati: è un confronto retrospettivo, **non un nuovo test cieco**. Nessun iperparametro o peso dell'ensemble viene scelto sul test o sulla calibrazione di agosto.

"""
    compact = []
    for h in config["horizons_minutes"]:
        c, b = chosen.loc[h], baseline.loc[h]
        compact.append({"minuti": h, "Timestral28_24h_MAE": b.mae, "personale_MAE": c.mae,
                        "riduzione_%": 100 * (b.mae - c.mae) / b.mae, "personale_RMSE": c.rmse,
                        "copertura90": c.coverage90, "ampiezza90": c.width90})
    text += markdown_table(pd.DataFrame(compact))
    optimized_name = next((c["spec"]["name"] for c in selection["selected"]
        if c["spec"]["family"] == "timestral" and c["spec"]["model"] == "timestral-28m"
        and c["spec"]["name"] != "timestral-28m_glucose_24h"), "timestral-28m_glucose_24h")
    comparison_rows = []
    for name, label in [("timestral-28m_glucose_24h", "Timestral 28M: riferimento 24 ore"),
                        (optimized_name, "Timestral 28M: configurazione scelta in validazione"),
                        ("selected_personal", "Modello personale selezionato")]:
        values = summary.loc[summary.model.eq(name)].set_index("horizon_minutes")
        comparison_rows.append({"modello": label, **{f"MAE_{h}min": values.loc[h, "mae"] for h in config["horizons_minutes"]}})
    text += "\n\n## Confronto con Timestral configurato\n\n"
    text += markdown_table(pd.DataFrame(comparison_rows))
    text += "\n\nEstendere il contesto e trasformare gli ingressi porta Timestral molto più vicino al modello personale. Il guadagno aggiuntivo del modello personale va valutato rispetto anche a questo benchmark. I due modelli Timestral mantengono i pesi pubblici; il correttore personale e il readout sono adattamenti supervisionati separati.\n"
    uncertainty_rows = []
    rapid = cohorts.loc[cohorts.model.eq("selected_personal") & cohorts.cohort.eq("rapid_change")].set_index("horizon_minutes")
    for h in config["horizons_minutes"]:
        uncertainty_rows.append({"minuti": h, "copertura90_globale_%": 100 * chosen.loc[h, "coverage90"],
            "ampiezza90_mgdl": chosen.loc[h, "width90"], "n_variazioni_rapide": int(rapid.loc[h, "n"]),
            "copertura90_variazioni_rapide_%": 100 * rapid.loc[h, "coverage90"]})
    text += "\n## Limite delle fasce durante variazioni rapide\n\n"
    text += markdown_table(pd.DataFrame(uncertainty_rows))
    text += "\n\nLe fasce sono calibrate su agosto per pasto/non pasto. La copertura globale è vicina al nominale 90%, ma durante variazioni rapide la copertura a 60 minuti è " + f"{100 * rapid.loc[60, 'coverage90']:.1f}% su {int(rapid.loc[60, 'n'])} origini. "
    text += "Questo è un limite osservato: la copertura non è uniforme nei diversi regimi. Variazione rapida significa valore assoluto del delta degli ultimi 15 minuti diviso 15 almeno pari a 2 mg/dL/min. Le previsioni e le fasce non vengono modificate dopo aver visto questi errori; ulteriori calibrazioni richiedono un esperimento separato e conferma su nuovi dati.\n"
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    ax.plot(config["horizons_minutes"], chosen.loc[config["horizons_minutes"], "coverage90"] * 100, "o-", label="Tutte le origini", color="#1565c0")
    ax.plot(config["horizons_minutes"], rapid.loc[config["horizons_minutes"], "coverage90"] * 100, "o-", label="Variazioni rapide", color="#c65128")
    ax.axhline(90, color="black", linestyle=":", label="Nominale 90%")
    ax.set(xlabel="Orizzonte (minuti)", ylabel="Copertura osservata (%)", title="Le fasce coprono meno durante variazioni rapide", xticks=config["horizons_minutes"])
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(output / "uncertainty-coverage.png", dpi=180)
    fig.savefig(output / "uncertainty-coverage.pdf")
    plt.close(fig)
    text += "\n![Copertura durante variazioni rapide](uncertainty-coverage.png)\n"
    text += "\n\n![Confronto](comparison.png)\n\n![Regimi](regimes.png)\n\n![Prima origine del test](first-origin.png)\n"
    text += "\n## Selezione congelata prima del test\n\n"
    for item in selection["ensemble"]:
        members = [f"{n}: {w:.1f}" for n, w in zip(item["models"], item["weights"]) if w > 0]
        text += f"- {item['horizon_minutes']} minuti: " + "; ".join(members) + ".\n"
    ablations = cv.loc[cv.model.isin(["catboost_glucose_only", "catboost_raw_events", "catboost_d6", "catboost_humalog360"])]
    text += "\n## Utilità degli ingressi sulle validazioni\n\n"
    text += "Confronto con gli stessi parametri CatBoost depth=6: gli eventi e le curve si aggiungono alla glicemia/calendario. Sono differenze predittive osservate, non stime causali dell'effetto dell'insulina.\n\n"
    text += markdown_table(ablations.pivot(index="model", columns="horizon_minutes", values="validation_mae").reset_index().rename(columns={h: f"MAE_{h}min" for h in config["horizons_minutes"]}))
    text += "\n\n## Ricerca della configurazione Timestral\n\nTutte queste configurazioni condividono le origini di validazione preliminare circa orarie, distinte dalla valutazione densa dei modelli selezionati. La variante fisica include anche il calendario futuro noto.\n\n"
    text += markdown_table(context_table)
    text += """
## Dati e ingressi

- Origini sui timestamp CGM effettivi; target alla lettura più vicina al timestamp richiesto, con scarto massimo ±90 secondi. Timestamp del target conservato. Celle storiche mancanti non sono target interpolati.
- 207 variabili: glicemia passata, tendenze, statistiche, calendario noto, quantità in finestre temporali e tempo dagli eventi; residuo/attività di insulina e proxy dei carboidrati.
- Humalog (lispro): curva esponenziale con più ipotesi durata/picco. Boli e basale rapida vengono trattati separatamente. Le quantità stimate sono **variabili del predittore**, non l'IOB misurata della pompa né parametri terapeutici.
- Le proiezioni dell'effetto futuro usano esclusivamente insulina/carboidrati già registrati all'origine. Non includono pasti o somministrazioni future degli export.
- Nei boli prolungati: pasto e quota iniziale all'inizio; componente estesa finale confermata al completamento, senza inventare un profilo erogato sulla base della quantità finale. È un'approssimazione conservativa da tenere presente.
- Le altre azioni future di Control-IQ restano sconosciute.
- Dati preparati in blocchi da 512 letture; array riaperti con memory mapping. Processi di esperimento con limite di memoria e thread definiti nel progetto.

L'insulina residua stimata è calcolata come `somma(dose_i × R(minuti_dalla_dose_i))`; l'attività come `somma(dose_i × A(minuti_dalla_dose_i))`. `R` parte da 1 e scende a 0; `A` integra a 1 ed è la velocità di consumo dell'effetto. Boli e basale hanno serie separate. I parametri esplorati `(durata, picco del kernel, ritardo)` sono `(240,55,10)`, `(300,75,10)`, `(360,75,10)`, `(420,90,10)` minuti: durata e picco si riferiscono al kernel prima del ritardo. Sono ipotesi predittive, non una misura individuale dell'assorbimento o impostazioni della pompa.

## Protocollo

1. Finestre di validazione aprile, giugno e luglio 2026; ogni addestramento usa soltanto le date precedenti. Tutti i target di training vengono esclusi se attraversano l'inizio della validazione, inclusa la tolleranza temporale.
2. Campionamento ogni tre letture (circa 15 minuti) per i modelli personali. Ricerca Timestral iniziale ogni dodici letture (circa un'ora), poi confronto del vincitore sulle stesse origini dense degli altri modelli.
3. Imputazione, standardizzazione e reti sono adattate sui soli dati di training. Per le reti l'early stopping usa la validazione di quel fold: queste metriche servono alla selezione, non a una stima finale indipendente.
4. Scelta del migliore per famiglia e dei pesi dell'ensemble per orizzonte sulle sole validazioni. La scelta dell'ensemble include una diagnostica dei pesi stimati sugli altri due fold.
5. Refit finale fino a luglio; per le reti numero di epoche fissato dai fold precedenti. Modelli e selezione salvati prima di leggere gli esiti del test per lo scoring.
6. Agosto riservato alle fasce 80%/90%, calibrate tramite residui assoluti separatamente per pasto registrato nelle ultime tre ore e altri periodi; fallback globale sotto 100 esempi per regime/orizzonte.
7. Test retrospettivo settembre–ottobre con origini comuni, errori per orizzonte e regime, ampiezza/copertura, e bootstrap appaiato per giornate. Gli errori mancanti sono esclusi solo dallo scoring per quel target.

Le quantità di training riportate sono origini eleggibili: Ridge, Extra Trees e CatBoost multioutput usano quelle con tutti i target presenti; histogram boosting e CatBoost MAE separato usano i target disponibili per ciascun orizzonte; le reti mascherano i target mancanti nella loss. Gli artefatti finali riportano anche i conteggi effettivi dei target.

## Modelli confrontati

- Regressioni Ridge regolarizzate, Extra Trees, histogram gradient boosting e CatBoost. Per i finalisti CatBoost e convoluzionale si confronta anche il training su ogni lettura a 5 minuti rispetto al training ogni tre letture; la cadenza di validazione resta identica.
- MLP con due loss e rete con convoluzioni dilatate sullo storico di glicemia/eventi.
- Ablazioni CatBoost con sola glicemia/calendario, eventi grezzi, sola curva insulinica 360/75, e tutte le curve/variabili.
- Timestral 28M/209M inizialmente ai contesti 24/72/168 ore, poi 28M glicemia fino a 8/16 settimane e covariate fino a quattro settimane, 209M glicemia fino a quattro settimane. Varianti con covariate fisiche trasformate e calendario futuro noto; le due aggiunte sono una variante congiunta, non un test causale isolato.
- Timestral 28M/24h con correttore CatBoost personale e readout Ridge delle rappresentazioni del decoder congelato. Il readout è addestrato sui dati personali; **non è un fine-tuning completo dei pesi Timestral**.
- CatBoost con quattro loss MAE distinte, usando tutto lo storico oppure gli ultimi 90 giorni di training; stessa validazione, nessuna scelta sui target di agosto o del test.
- Persistenza, tendenza smorzata (costante di tempo fissa 30 minuti) e combinazione selezionata per orizzonte.

## Tutti i risultati finali sulle stesse origini

"""
    text += markdown_table(summary[["model", "horizon_minutes", "n", "mae", "rmse", "bias", "coverage90", "width90"]])
    text += "\n\n## Incertezza e validità delle conclusioni\n\n"
    text += "Le bande sono intervalli per il singolo orizzonte, non una garanzia simultanea sull'intera curva. La dipendenza temporale e i cambiamenti del comportamento personale richiedono conferma su un nuovo periodo indipendente. Il bootstrap per giornate riduce il problema delle letture sovrapposte, senza eliminare ogni dipendenza tra giornate. Una copertura globale vicina al 90% va letta insieme ai gruppi pasto/non pasto e all'ampiezza. `test-metrics.csv` include anche l'interval score 80%/90% (più basso è migliore), che penalizza sia intervalli larghi sia osservazioni esterne.\n"
    text += "\n## Artefatti e riproduzione\n\n"
    text += "`selection-frozen.json`: scelte antecedenti al test. `validation-metrics.csv`: tutti i confronti validati. `test-metrics.csv`, `test-cohorts.csv`, `test-predictions.csv.gz`: risultati e previsioni. `models/*.joblib`: modelli personali riutilizzabili e metadati. `timestral-personal-cache/`: rappresentazioni del backbone congelato, salvate a blocchi. I manifest registrano hash di codice/dati, configurazione e versioni. I CSV originali e il benchmark v1 restano conservati.\n"
    if (output / "cuda-equivalence.json").exists():
        text += "\nEsecuzione mista CPU/CUDA: una verifica su casi CPU già completi confronta mediane, quantili e rappresentazioni in float32 con TF32 disabilitato; differenze massime delle mediane <0.000046 mg/dL. `cuda-equivalence.json` registra il controllo, `execution-cuda-*.json` hardware/versioni e fasi accelerate. L'ambiente CUDA è separato; i risultati CPU completi restano in cache. La RAM GPU è gestita con limite fisico cgroup, non con limite degli indirizzi virtuali.\n"
    text += "\nI modelli personali emettono direttamente quattro orizzonti; una curva densa completa richiederà una successiva valutazione dei passi intermedi. Nessun calcolo di dosi o comando della pompa è implementato in questa fase.\n"
    text += "\n## Fonti delle formule e del modello\n\n[Modello esponenziale LoopKit](https://github.com/LoopKit/LoopKit/blob/dev/LoopKit/InsulinKit/ExponentialInsulinModel.swift), [formula OpenAPS](https://github.com/openaps/oref0/blob/master/lib/iob/calculate.js), [Timestral ufficiale](https://github.com/mistralai/timestral). Le ipotesi sui carboidrati sono proxy matematici gamma, non misure di assorbimento.\n"
    (output / "report.md").write_text(text)
