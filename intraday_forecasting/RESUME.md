# Stato del progetto — 25 ottobre 2026

## V3 completata e verificata — tutti i miglioramenti richiesti

**Questa sezione prevale sul checkpoint v2 sotto.** Nessun training resta attivo; non occorre rilanciare `all` o `final`. V2 e dati originali sono conservati. La v3 usa `src/v3/`, `config/personal-v3.json` e `outputs/personal-v3/`.

MAE in mg/dL sulle stesse 4.878 origini di settembre–ottobre:

| Modello | 30 min | 60 min | 120 min | 180 min |
|---|---:|---:|---:|---:|
| Ensemble v3 | 11,04 | 17,58 | 23,69 | 25,86 |
| Ensemble v2 | 11,30 | 18,06 | 24,74 | 27,21 |
| Timestral 28M, covariate, 28 giorni | 11,57 | 18,71 | 25,29 | 27,82 |

Guadagno rispetto al v2 circa 2–5%, maggiore sugli orizzonti lunghi. I confronti appaiati per giornate sono nel report. **Il periodo era già stato esaminato nella v2 e ha suggerito le ipotesi v3**: questo confronto è esplorativo, da confermare con nuovi dati indipendenti. Inoltre la v3 include refit settimanali che possono apprendere dagli esiti delle settimane precedenti del test; il v2 rimane statico.

Implementazione completata:

- Cache Timestral 28M/covariate/28 giorni su 30.271 origini, a blocchi e con riuso verificato degli archivi v2.
- Tre CatBoost: correttore forte, correttore pesato per variazioni rapide, diretto pesato; due reti causali a 36 passi con/senza loss sulle pendenze.
- Fine-tuning effettivo: testa con glicemia/24h (1.880.064 parametri), testa+ultimo blocco con covariate/72h (4.535.424 parametri). Equivalenza iniziale nativa scarto zero; pesi pubblici mai sovrascritti. A parità di ingressi, MAE di validazione a tre ore `41.79 → 37.57` e `31.13 → 29.19` rispettivamente. Non è il fine-tuning di tutto il backbone.
- Due politiche di refit settimanale con storico crescente e target già disponibili al cutoff: diretto e correttore Timestral lungo.
- Ensemble su 25 candidati compatibili senza filtro top-3; confronto dei pesi globali e per regime. Scelto `global`, MAE diagnostico di validazione `10.99/17.52/23.48/25.69`.
- Tre metodi di fasce confrontati: pasto/non pasto, scala appresa con gruppi di regime, rolling su errori già osservabili. Scelto `adaptive_group` sulla seconda metà di luglio; anche i risultati degli altri metodi sono conservati.
- Selezioni e hash congelati prima del nuovo scoring; training statico prima di agosto, calibrazione agosto, test settembre–ottobre. Nessun aggiustamento dopo il nuovo test.
- Sette nuovi modelli statici, 54 checkpoint settimanali (30 di validazione, 24 finali), previsioni e tracciati temporali. Modelli v2 riutilizzati senza sovrascriverli.

Fasce nominali al 90%: copertura globale **89,99 / 90,13 / 90,84 / 89,51%**; ampiezza media **46,70 / 74,86 / 104,26 / 108,51 mg/dL**, inferiore al v2 sui quattro orizzonti. Nelle variazioni rapide la copertura è **92,51 / 87,57 / 89,01 / 85,71%**: migliora a 60 minuti rispetto all'81,1% v2, ma rimane insufficiente in alcuni regimi, con pochi casi rapidi (185 a 60 minuti).

Modello denso separato: `trajectory_huber`. I suoi 36 punti **non sono l'ensemble dei quattro orizzonti** e le barre calibrate riguardano quest'ultimo. Il tempo del massimo nella finestra ha ancora errore medio di circa 43 minuti: non dichiarare risolto il problema dei picchi. Il massimo della finestra può cadere al bordo e non coincide necessariamente con un picco fisiologico completo.

Verifiche concluse: **63 test superati**; hash coerenti e v2 invariato; metriche ricalcolate con scarto massimo circa `1,3e-12 mg/dL`; tre replay identici rimuovendo `targets` e `target_indices`, scarto CPU/cache GPU massimo **0,0000153 mg/dL**. Verificati cutoff, 54 checkpoint settimanali, fasce rolling causali, curve dense e quantili. Grafici controllati visivamente. Esiti in `outputs/personal-v3/verified.json` e `test-suite.log`.

Artefatti principali in `outputs/personal-v3/`: `report.md`, `comparison.png/pdf`, `examples.png/pdf`, `uncertainty.png`; `test-metrics.csv`, `test-cohorts.csv`, `test-intervals.csv`, `test-dense-metrics.csv`, `validation-metrics.csv`, `finetune-matched-validation.csv`; selezioni congelate e `selection-before-scoring.json`; directory `models/`, `weekly-models/`, `finetune-models/`, `cv-models/`, `predictions/`, `oof/`, `intervals/`, `long-cache/`. Gli output sono ignorati da Git e restano locali. Nessun nuovo commit/push eseguito per questa richiesta.

Replay sulle origini storiche, dopo il completamento:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_limited.py -- .venv/bin/python -m src.v3.predict --at '2026-10-15T12:00:00+02:00'
```

Solo per una nuova verifica degli artefatti, se necessaria:

```bash
.venv/bin/python scripts/run_v3.py verify --device cpu
```

WSL 23,47 GiB; CPU 16 GiB virtuali/8 thread; GPU cgroup 16 GiB fisici, soglia 14 GiB, swap massimo 1 GiB, VRAM massima 4 GiB. Eventuale lavoro CPU parallelo alla GPU al massimo 3 GiB fisici, lasciando oltre 4 GiB al sistema. Picco finale misurato **2.872,66 MiB RSS / 421,96 MiB VRAM**. Non modificare Conda `arena`; `.venv-gpu` è separato. Il costruttore originale resta verificato a circa 490 MiB e sotto limite virtuale 2 GiB.

Per continuare: raccogliere un periodo nuovo e indipendente, fissare prima split e criteri e confrontare i modelli congelati. Nuove selezioni richiedono un output distinto; non cambiare sorgenti/configurazioni numeriche congelate. Pasti futuri e azioni future di Control-IQ restano sconosciuti; il dominio è quello osservato (minimo 75 mg/dL). Real time, dosaggio e comandi alla pompa restano fuori ambito. Leggere `tasks.md` prima di nuovi esperimenti.

## Checkpoint v2 concluso

**Esperimento storico v2 completato e verificato.** Il lavoro del checkpoint precedente è concluso: non occorre rilanciare `long` o `final`. Nessun processo di training è lasciato attivo.

## Risultati finali

MAE in mg/dL sulle stesse 4.878 origini di settembre–ottobre, con target mancanti esclusi per orizzonte:

| Modello | 30 min | 60 min | 120 min | 180 min |
|---|---:|---:|---:|---:|
| Modello personale selezionato | 11,30 | 18,06 | 24,74 | 27,21 |
| Timestral 28M, covariate, 28 giorni | 11,57 | 18,71 | 25,29 | 27,82 |
| Timestral 28M, glicemia, 24 ore | 13,23 | 22,31 | 34,18 | 41,57 |

Il vantaggio del modello personale è circa 15–35% rispetto al riferimento di 24 ore; Timestral configurato con più storico e ingressi trasformati è già vicino al modello personale. Non presentare tutto il miglioramento come effetto del solo addestramento personale.

**Limite delle fasce:** nominale 90%, copertura globale circa 89–90%, ma solo 81,1% a 60 minuti durante variazioni rapide. A tre ore ampiezza media complessiva circa 116 mg/dL. Nessun parametro viene cambiato dopo aver visto questi esiti.

Settembre–ottobre era già stato consultato nella v1: questo è un test retrospettivo riutilizzato, non un nuovo test cieco.

## Cosa è concluso

- Dataset verificato e riusato in memory mapping: 94.237 letture, 207 variabili; Humalog per boli/basale, pasti all'orario registrato, regole sui carboidrati fittizi conservate.
- Correzione us/ns e costruzione a blocchi: picco circa 490 MiB, CSV originali invariati.
- 23 configurazioni supervisionate e 18 configurazioni Timestral preliminari; validazioni cronologiche aprile/giugno/luglio.
- Selezione e pesi congelati prima dello scoring: `selection-frozen.json`, hash in `selection-before-scoring.json`.
- Training finale prima di agosto su 22.437 origini, 12 modelli salvati; agosto per le fasce, settembre–ottobre per la valutazione.
- Controlli su artefatti, hash, origini, target e metriche: `outputs/personal-v2/verified.json` passato. Tre previsioni senza array dei target restano identiche; scarto massimo CPU/cache CUDA 0,000026 mg/dL.
- 53 test della logica già superati; nuove verifiche sugli artefatti e resa visiva dei grafici.

Il modello combina CatBoost MAE e correttore Timestral a 30/60 minuti, CatBoost/Extra Trees/TCN a 120 e CatBoost/TCN a 180. I pesi del backbone Timestral restano pubblici: nessun fine-tuning completo eseguito.

## Artefatti

In `outputs/personal-v2/`:

- `report.md`: protocollo, confronti, limiti, risultati per regime e grafici.
- `test-metrics.csv`, `test-cohorts.csv`, `test-predictions.csv.gz`: errori, copertura/ampiezza, bootstrap e previsioni.
- `models/*.joblib`: 12 modelli personali riutilizzabili, circa 49 MiB complessivi.
- `test-examples.png/pdf`, `test-day.png/pdf`: esempi a date/orari prefissati con fasce e giornata completa del 2 settembre.
- `uncertainty-coverage.png/pdf`, `comparison.png/pdf`, `regimes.png`: confronto ed evidenza del limite nelle variazioni rapide.
- `completed.json`, `verified.json`, manifest e log CPU/CUDA: provenienza e verifiche.

Gli output sono ignorati da Git e conservati localmente. Codice/istruzioni sono versionabili.

## Riuso sullo storico

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_limited.py -- .venv/bin/python -m src.personal_predict --at '2026-10-15T12:00:00+02:00'
```

Verifica completa dei modelli e degli artefatti, se necessaria dopo nuove modifiche:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_limited.py -- .venv/bin/python scripts/verify_personal_final.py
```

I modelli attuali emettono quattro orizzonti, non una curva densa già validata. Il comando è un replay storico; per nuovi dataset occorre preservare i dati e le impronte dell'esperimento e predisporre un percorso di valutazione separato.

## Vincoli e prossime analisi

WSL circa 23,47 GiB; training CPU 16 GiB virtuali/8 thread, GPU cgroup 16 GiB fisici. Picco finale misurato circa 4,07 GiB RAM e 905 MiB VRAM. Ambiente `.venv-gpu` separato; non modificare Conda `arena`. I test completi richiedono il profilo virtuale training; il costruttore rimane verificato anche a 2 GiB.

Non cambiare il codice numerico/configurazione congelati senza un nuovo output: gli hash proteggono modelli e cache. Il rapporto e le nuove visualizzazioni sono stati aggiornati senza alterare previsioni o selezione.

Possibili fasi successive: nuovo periodo indipendente; incertezza condizionata anche alla variazione rapida; previsione diretta dei passi intermedi. Real time e dosaggio restano fuori ambito. Leggere `tasks.md` prima di avviare un nuovo esperimento; non riaprire il v2 già completato per scegliere parametri sul suo test.
