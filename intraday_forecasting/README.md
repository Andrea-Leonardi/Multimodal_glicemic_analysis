# TimestralGluco

Esperimento personale per confrontare previsioni glicemiche con Timestral e metodi semplici. Addestramento e inferenza locali, nessun calcolo di dosi o collegamento di comando alla pompa.

Il lavoro attuale è il modello storico supervisionato v2: [report e risultati](outputs/personal-v2/report.md), [protocollo e comandi](#addestramento-storico-personale-v2). Le sezioni iniziali conservano la documentazione del benchmark v1 e dei componenti live già predisposti; la raccolta live è rinviata.

Il registro del lavoro è [tasks.md](tasks.md). Il primo esperimento completo produce [report, metriche e grafici](outputs/experiment-v1/report.md).

## Ambiente e comandi

Ambiente verificato: Linux/WSL, Python 3.14, PyTorch CPU. L'ambiente `.venv` già preparato nel progetto usa pandas/numpy/matplotlib dell'ambiente Conda esistente e installa Timestral/PyTorch localmente; non modifica l'ambiente Conda. Per una nuova installazione isolata:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.download_models
```

Il download legge solo revisioni pubbliche di pesi/configurazioni. L'analisi e l'inferenza lavorano sui CSV locali. Per riprodurre:

```bash
.venv/bin/python -m src.data
HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 .venv/bin/python -m src.benchmark --output outputs/experiment-v1
```

Solo baseline, senza dipendenze Timestral:

```bash
python -m src.benchmark --baselines-only --output outputs/baselines-new
```

La configurazione è in `config/experiment.json`. Una directory di output conserva un singolo esperimento: il comando rifiuta di riusarla se cambiano configurazione, hash dei dati o codice del benchmark. Per un esperimento modificato scegliere un nuovo `--output`; le previsioni già calcolate sono riutilizzate solo con il manifest corrispondente.

## Dati e interpretazione

- `data/cleaned/` e gli export originali restano invariati. `src/data.py` legge CGM/basale puliti e recupera i dettagli dei boli dagli export originali in `data/tandemsource_data/merged/`.
- `data/processed/audit.json` contiene controlli, assunzioni, hash degli input e confronto dei totali basali con i riepiloghi orari.
- `data/processed/timeline.csv` contiene la griglia a 5 minuti, glicemia osservata, timestamp del campione, età, maschere dei mancanti e covariate.
- `data/processed/bolus_events.csv` conserva categoria originale, modalità, stato, orari, componenti estese e classificazione personale versionata.
- `carbs_recorded_g` conserva ciò che è stato inserito nella pompa; `carbs_meal_inferred_g` conserva i grammi interpretati come mangiati. Per l'uso personale dell'utente, le regole del notebook distinguono i carboidrati fittizi inseriti per una correzione. Non si scarta l'insulina erogata durante quelle correzioni.
- Un bolo manuale a zero carboidrati resta una correzione manuale. I boli automatici sono identificati dalla modalità, non dall'assenza di carboidrati.
- Anche i boli interrotti sono conservati con la quantità erogata riportata dall'export. Nei boli prolungati non viene inventata un'erogazione istantanea all'inizio né una curva farmacologica.
- Per lo storico il totale finale del bolo diventa disponibile al completamento. Nei dati mancano orari di inserimento/arrivo originali: questa è una convenzione conservativa, particolarmente approssimativa per i pasti associati ai boli prolungati.
- Fuso assunto `Europe/Rome`, esplicito nella configurazione; griglia e intervalli in UTC. Orari locali ambigui/non esistenti producono errore, senza correzione silenziosa.
- La griglia usa solo campioni nell'intervallo `(t−5min,t]`; non interpola i target. Se due letture cadono nella stessa cella conserva l'ultima e mantiene il timestamp reale. Celle CGM/basale vuote restano sconosciute; il timestamp basale discordante è segnato mancante.

## Protocollo storico

Novembre–giugno restano disponibili come storico di sviluppo; luglio seleziona i contesti da 6/12/24 ore. Agosto calibra le fasce 80%/90%. Settembre–ottobre costituiscono il test. La selezione su luglio usa origini ogni tre ore per contenere il costo CPU; calibrazione e test ogni ora. Tutti i candidati condividono i criteri di ammissibilità sul passato e non usano target futuri per scegliere le origini. L'intero orizzonte resta nel proprio split.

Tre ingressi Timestral, distinti nel report:

| Modalità | Ingressi | Interpretazione |
| --- | --- | --- |
| `glucose` | Glicemia | Primo riferimento utilizzabile con Dexcom |
| `manual` | Glicemia, pasti inferiti, boli manuali confermati | Assume registrazione manuale completa; storico approssimato dai completamenti |
| `pump` | Come manual, più boli automatici e basale | Confronto retrospettivo con più informazioni; non è attualmente replicabile con il solo Share |

I futuri pasti e le future azioni di Control-IQ non sono ingressi. IOB storico e piani annunciati prima del pasto/bolo non sono disponibili. Il ritardo CGM del benchmark iniziale è assunto zero: il programma rifiuta altre latenze CGM per evitare simulazioni incomplete. I ritardi reali vanno misurati e riprodotti in un esperimento successivo.

Si conservano curve complete di tre ore a 21 quantili, errori a 30/60/120/180 minuti, copertura e ampiezza degli intervalli, sottogruppi e bootstrap per giornate. I gruppi senza osservazioni sono riportati con `n=0`: nello storico non ci sono glicemie sotto 75 mg/dL. La copertura è punto per punto e non una garanzia sull'intera curva; dipendenza temporale e cambiamenti nel tempo richiedono verifica prospettica.

Le previsioni sono scritte prima di estrarre gli esiti per lo scoring. Configurazione, versioni e hash sono nel manifest. Dopo aver consultato il test, ogni ulteriore scelta ispirata dai suoi risultati richiede dati futuri nuovi per una valutazione indipendente.

## Registro manuale

Registro SQLite locale e append-only tramite l'applicazione, in `data/live/events.sqlite`. Ogni evento conserva `event_time` e `available_at`; i piani sono separati dalle conferme, che possono avere quantità e orario effettivi diversi. La stessa intenzione non può essere confermata due volte. Le unità sono g, U, mg/dL; l'IOB è un'osservazione manuale, non un calcolo farmacologico.

```bash
.venv/bin/python -m src.journal --help
.venv/bin/python -m src.journal add --help
.venv/bin/python -m src.journal confirm --help
.venv/bin/python -m src.journal list
```

`add` richiede `--kind` (`meal`, `insulin_plan`, `insulin_confirmed`, `iob`, `glucose`), `--amount` e `--at` con fuso esplicito, ad esempio un timestamp ISO che termini in `+02:00` o `+01:00` secondo la data. `confirm` richiede l'ID restituito per il piano, la quantità effettivamente somministrata e l'orario effettivo. Questi comandi registrano quantità decise dall'utente.

## Dexcom Share europeo

Il client `src/dexcom.py` riproduce il flusso del codice ESP32 fornito dall'utente: autenticazione, login con account ID, letture da `shareous1.dexcom.com`. Supporta le risposte ID sia stringa sia oggetto, mantiene la sessione e la rinnova una sola volta su errore di sessione. Verifica i certificati TLS, imposta timeout e non stampa credenziali, sessioni o risposte di autenticazione.

Usa variabili d'ambiente impostate localmente, senza inserire credenziali nei file versionati o nella chat:

```bash
read -r -p 'Account Dexcom: ' DEXCOM_USER
read -r -s -p 'Password Dexcom: ' DEXCOM_PASSWORD
export DEXCOM_USER DEXCOM_PASSWORD
.venv/bin/python -m src.dexcom --once
```

Raccolta continua, da fermare con Ctrl+C:

```bash
.venv/bin/python -m src.dexcom
```

Richiede le ultime 24 ore a ogni interrogazione, fino a 288 letture; registra solo campioni nuovi, conserva l'orario `WT` del sensore e il primo orario di ricezione. Lo stesso campione riottenuto non cambia la latenza registrata. Valori discordanti allo stesso timestamp sono rifiutati senza sovrascrivere. Polling ogni 60 secondi con attesa crescente sugli errori. Lo storico recuperato all'avvio ha un'età diversa dalla latenza in regime: le due condizioni vanno analizzate separatamente.

I limiti di recupero e gli endpoint sono riscontrabili nel [codice del client pydexcom](https://github.com/gagebenne/pydexcom/blob/main/pydexcom/const.py). Share è un percorso diverso dalla Developer API e il suo formato può cambiare. Nessuna autenticazione reale è stata eseguita durante l'implementazione: le credenziali effettive non erano disponibili.

## Previsioni prospettiche locali

Dopo il benchmark e l'avvio del collector, in un secondo terminale:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m src.live --watch
```

Il worker usa Timestral 28M e il contesto scelto su luglio. Attende un contesto sufficientemente completo, aggiorna la previsione quando cambia il registro e salva la curva nel database prima di generare `outputs/live/latest.json` e `latest.png`. Gli esiti vengono associati dopo il rispettivo orizzonte, con una tolleranza di arrivo di 15 minuti, nella tabella `forecast_scores`. I dati non ancora arrivati non possono entrare in una previsione passata.

Per includere pasti realmente inseriti e boli manuali confermati usare `--mode manual --manual-logging-since TIMESTAMP_CON_FUSO`: il parametro indica da quando l'inserimento è completo. Prima di quel momento le covariate restano sconosciute. Senza questa dichiarazione il worker rifiuta la modalità manuale. La modalità `pump` non è disponibile nel worker.

Il worker live ancora gli intervalli all'orario effettivo di emissione, mentre il benchmark iniziale usa una griglia fissa. Questa differenza, i ritardi Share e la qualità degli inserimenti richiedono nuova valutazione prospettica. Le bande derivate dallo storico sono mostrate come sperimentali, con copertura dal vivo da verificare.

**IOB e insulina pianificata vengono conservati ma non condizionano ancora il modello.** Il modello usa la glicemia e, nella modalità manuale, pasti/boli confermati. Il benchmark non contiene una serie IOB o piani storici; aggiungerli alla previsione, con esperimenti e distinzione delle curve condizionate a piani non confermati, rimane un'attività esplicita in `tasks.md`.

## Modello e fasi successive

Codice Timestral e pesi 28M/209M sono fissati a commit/revisioni in configurazione. [Repository ufficiale](https://github.com/mistralai/timestral), [annuncio Mistral](https://mistral.ai/news/timestral/). L'inferenza è verificata localmente; il repository documenta l'inferenza standalone. La procedura concreta di fine-tuning locale rimane da verificare prima di prometterne l'esecuzione.

Gli output generati, la cache, l'ambiente locale, il database live e gli eventuali file `.env` sono esclusi dai nuovi file Git tramite `.gitignore`. Il lavoro non cambia gli export o i file di pulizia preesistenti.

## Dataset personale v2: costruzione con memoria limitata

La fase attuale si concentra sullo storico, con Humalog e pasti all'orario effettivamente registrato. Il confronto v2 usa questa preparazione verificata; protocollo e comandi sono descritti nella sezione successiva.

È stato corretto un errore di unità: pandas 3 restituiva timestamp in microsecondi, mentre il codice presumeva nanosecondi. Gli intervalli risultavano 1.000 volte più brevi e `exposure()` tentava di creare una matrice da circa 60 GiB. Ora i timestamp sono normalizzati esplicitamente; statistiche, esposizioni, curve e sequenze sono elaborate in blocchi da 512 righe. Una guardia blocca esposizioni troppo grandi prima dell'allocazione.

Per verificare prima l'equivalenza su un campione e poi l'intero storico:

```bash
.venv/bin/python scripts/check_personal_memory.py sample
.venv/bin/python scripts/check_personal_memory.py full
```

Ogni verifica viene eseguita in un processo separato, con `RLIMIT_AS` a 2 GiB di spazio virtuale, pool numerici limitati a un thread e controllo preventivo di almeno 4 GiB disponibili per le altre applicazioni. Se le risorse non bastano, il processo si arresta; non aumenta il limite automaticamente. L'esecuzione completa richiede un campione superato sugli stessi hash di codice, configurazione e dati.

Risultato misurato: campione da 4.096 letture numericamente identico al riferimento vettoriale con unità corrette; costruzione completa da 94.237 letture e 207 variabili con picco RSS **490 MiB**, in circa **5,8 secondi**. I rapporti sono in `outputs/memory-check/`. Otto test dedicati verificano anche unità temporali, formule, assenza di eventi futuri e guardia alle allocazioni.

Gli array completi sono salvati senza compressione in `outputs/memory-check/dataset/`, pubblicando la directory soltanto a scrittura conclusa. Per la ripresa usare il memory mapping:

```python
from pathlib import Path
import numpy as np

folder = Path("outputs/memory-check/dataset")
X = np.load(folder / "features.npy", mmap_mode="r")
y = np.load(folder / "targets.npy", mmap_mode="r")
sequence = np.load(folder / "sequence.npy", mmap_mode="r")
```

Il memory mapping evita il caricamento integrale iniziale; selezioni avanzate e conversioni possono comunque creare copie. I successivi addestramenti devono conservare batch e limiti di memoria espliciti. Per esperimenti con codice/configurazione differenti usare un nuovo `--output`, senza sovrascrivere il dataset verificato.

Con l'impostazione WSL `memory=24GB` attiva, Linux riporta circa **23,47 GiB complessivi**. `config/resources.json` autorizza **16 GiB di spazio virtuale per processo e 8 thread per l'addestramento**, e **4 GiB e un thread per la costruzione**. All'avvio viene controllata la memoria disponibile: il limite effettivo si riduce se necessario per lasciare almeno **4 GiB disponibili** alle altre applicazioni. Questo è un controllo preventivo, non una prenotazione permanente della memoria di sistema. Le verifiche dedicate con `check_personal_memory.py` mantengono il limite conservativo di 2 GiB già superato dal dataset.

Usare `.venv/bin/python scripts/run_limited.py --` come prefisso al comando Python di addestramento. Il wrapper applica il limite prima degli import numerici e imposta i pool di thread. Eseguire un addestramento alla volta; processi indipendenti non condividono un budget RAM aggregato.

## Addestramento storico personale v2

Il confronto supervisionato usa gli ingressi già verificati in `outputs/memory-check/dataset/`, senza attivare il collector o il worker real time. Le previsioni sono dirette a 30, 60, 120 e 180 minuti; il dominio è quello presente nei dati. L'IOB modellata e l'attività di Humalog sono caratteristiche predittive, distinte dall'IOB mostrata dalla pompa.

Eseguire le fasi in ordine, una alla volta, con i limiti del progetto:

```bash
.venv/bin/python scripts/run_limited.py -- .venv/bin/python -m src.personal_experiment --stage tabular
.venv/bin/python scripts/run_limited.py -- .venv/bin/python -m src.personal_dense_training
.venv/bin/python scripts/run_limited.py -- .venv/bin/python -m src.personal_boosting
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_limited.py -- .venv/bin/python -m src.personal_experiment --stage timestral
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_limited.py -- .venv/bin/python -m src.personal_timestral_extended
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_limited.py -- .venv/bin/python -c 'from src.personal_timestral_long import run; from src.personal_experiment import load_dataset; from src.personal_data import configuration; from src.data import ROOT; run(load_dataset(), configuration(), ROOT / "outputs/personal-v2")'
.venv/bin/python scripts/run_limited.py -- .venv/bin/python -m src.personal_experiment --stage hybrid
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_limited.py -- .venv/bin/python -m src.personal_experiment --stage final
```

Le fasi conservano previsioni e manifest per riprendere un'esecuzione interrotta. Se cambia il codice numerico o la configurazione, scegliere un nuovo `--output` per tutte le fasi: i manifest impediscono di mescolare esperimenti diversi.

- `tabular`: 16 configurazioni tra Ridge, Extra Trees, histogram boosting, CatBoost, MLP e rete con convoluzioni dilatate; ablazioni degli ingressi con parametri CatBoost uguali.
- `personal_dense_training`: per i finalisti CatBoost e convoluzionale, confronto con tutte le letture di training a 5 minuti, mantenendo le medesime origini di validazione ogni tre letture.
- `personal_boosting`: quattro stimatori CatBoost distinti, loss MAE per orizzonte; confronto fra tutto il training e ultimi 90 giorni, sulle stesse origini di validazione.
- `timestral`: contesti 24/72/168 ore per 28M e 209M, più varianti con covariate fisiche e calendario futuro noto. Ricerca preliminare oraria, confronto del vincitore sulle stesse origini ogni tre letture degli altri modelli.
- `personal_timestral_extended`: prova anche 14/28 giorni con sola glicemia e 7/14 giorni con covariate per il 28M e approfondisce il vincitore se migliora sulle validazioni.
- `personal_timestral_long.run`: ulteriori 8/16 settimane di glicemia e quattro settimane di covariate per 28M; 2/4 settimane per 209M. Si resta entro il limite preaddestrato di contesto. Il wrapper CUDA espone questa fase come `long`.
- `hybrid`: correttore CatBoost e due readout Ridge delle rappresentazioni Timestral 28M congelate. Sono personalizzazioni supervisionate; i pesi del backbone rimangono quelli pubblici.
- `final`: selezione per famiglia e pesi per orizzonte sulle sole validazioni, refit prima di agosto, intervalli calibrati su agosto per regime pasto/non pasto, poi scoring retrospettivo settembre–ottobre.

Le finestre di validazione sono aprile, giugno e luglio. Ogni training usa solo il passato e scarta anche i target che attraverserebbero il confine del periodo. Imputazione e standardizzazione vengono adattate sui soli esempi di training. Non vengono forniti pasti, boli o modifiche della basale futuri; le uniche covariate future note di Timestral sono quelle del calendario.

I risultati finali vengono scritti in [outputs/personal-v2/report.md](outputs/personal-v2/report.md), con metriche e grafici, `selection-frozen.json`, modelli `.joblib`, previsioni e bootstrap per giornate. Il test settembre–ottobre era già stato consultato nel primo esperimento: questo resta un confronto retrospettivo e non viene presentato come un nuovo test cieco. Per le fasce viene riportata anche la copertura dopo i pasti.

Dopo la fase finale, una previsione storica può essere ricalcolata dai modelli salvati, senza leggere i target:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_limited.py -- .venv/bin/python -m src.personal_predict --at '2026-10-15T12:00:00+02:00'
```

Il comando usa l'ultima lettura realmente precedente all'orario richiesto, restituisce i quattro valori previsti con intervalli e rifiuta origini anteriori alla disponibilità del modello/calibrazione finale.

Dopo il completamento della fase finale, generare gli esempi del test a date/orari prefissati e verificare gli artefatti e il riuso dei modelli:

```bash
.venv/bin/python scripts/run_limited.py --profile build -- .venv/bin/python -m src.personal_test_plots
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_limited.py -- .venv/bin/python scripts/verify_personal_final.py
```

La verifica controlla provenienza di codice/dati, origini comuni, pesi congelati, separazione dei target di training, metriche e copertura ricalcolate. Riproduce tre previsioni dai modelli salvati anche rimuovendo gli array dei target futuri; conserva gli esiti e gli hash in `outputs/personal-v2/verified.json`. Non riaddestra né seleziona modelli.


## Accelerazione CUDA facoltativa

La RTX 3070 è utilizzabile da WSL. `.venv-gpu` è separato da `.venv`: riusa la build CUDA di Torch 2.14.1 dell'ambiente locale `arena` senza modificarlo; le dipendenze scientifiche necessarie sono allineate alle versioni del progetto. Python 3.13 e build CUDA 13.0 sono registrati nei log di esecuzione, insieme agli hash. Il percorso CPU sopra rimane disponibile.

Prima dell'accelerazione è richiesto il confronto CPU/CUDA su artefatti CPU già completi. Il controllo comprende mediana, tutti i quantili agli orizzonti valutati e rappresentazioni del decoder. Float32 con TF32 disabilitato; differenza massima osservata delle mediane <0.000046 mg/dL. Le cache CPU complete restano riutilizzate: `execution-cuda-*.json` documenta la continuazione su GPU, separatamente dai manifest scientifici.

```bash
.venv/bin/python scripts/run_cuda.py check
.venv/bin/python scripts/run_cuda.py timestral
.venv/bin/python scripts/run_cuda.py extended
.venv/bin/python scripts/run_cuda.py long
# Eseguire anche dense-training, personal_boosting e hybrid con i comandi CPU sopra, prima di final:
.venv/bin/python scripts/run_cuda.py final
```

Il wrapper richiede CUDA e una sessione systemd utente WSL. Applica un cgroup temporaneo con massimo 16 GiB di RAM fisica, soglia 14 GiB e massimo 1 GiB swap, abbassando il limite se non rimangono inizialmente 4 GiB disponibili. Limita la VRAM a 4 GiB e lascia almeno 1 GiB libero rispetto alla disponibilità iniziale; riduce i batch in caso di esaurimento VRAM. CUDA riserva ampie regioni virtuali, quindi qui non si applica `RLIMIT_AS`. Eseguire una sola fase pesante alla volta.

## Esperimento v3: miglioramenti supervisionati e incertezza adattiva

La v3 conserva il benchmark v2 e scrive separatamente in `outputs/personal-v3/`. Riutilizza il dataset verificato, le regole personali sui carboidrati e le medesime origini di valutazione. Configurazione: [config/personal-v3.json](config/personal-v3.json). Il confronto rimane storico; non avvia alcun servizio Dexcom o comando della pompa.

Le fasi sono riprendibili e vanno eseguite in quest'ordine, con una sola fase pesante alla volta:

```bash
.venv/bin/python scripts/run_v3.py cache
.venv/bin/python scripts/run_v3.py tabular --device cpu
.venv/bin/python scripts/run_v3.py trajectory
.venv/bin/python scripts/run_v3.py finetune
.venv/bin/python scripts/run_v3.py rolling --device cpu
.venv/bin/python scripts/run_v3.py ensemble --device cpu
.venv/bin/python scripts/run_v3.py intervals --device cpu
.venv/bin/python scripts/run_v3.py final
.venv/bin/python scripts/run_v3.py verify --device cpu
```

Il wrapper usa `.venv-gpu` e i limiti cgroup/VRAM sopra per CUDA, `.venv` e il profilo CPU del progetto con `--device cpu`. In alternativa `scripts/run_v3.py all` esegue le fasi in sequenza. Gli stadi già completati riusano le cache; non è necessario rilanciare i training dopo una verifica riuscita. Gli hash impediscono di mescolare sorgenti o configurazioni numeriche diverse: un nuovo esperimento richiede un nuovo output nella configurazione, preservando gli artefatti esistenti.

- `cache`: Timestral 28M con covariate fisiche e contesto di 28 giorni su tutte le origini necessarie al correttore; avanzamento salvato a blocchi e riuso degli archivi v2 compatibili.
- `tabular`: correttori CatBoost del Timestral lungo, con e senza pesatura delle variazioni rapide, più un modello diretto pesato.
- `trajectory`: due reti causali che emettono 36 passi a 5 minuti; confronto fra loss sui livelli e aggiunta di pesi sui regimi e loss sulle pendenze.
- `finetune`: aggiornamento effettivo dei pesi Timestral, limitato alla testa oppure alla testa e all'ultimo blocco del decoder. Il percorso differenziabile viene confrontato con quello nativo prima del training. I checkpoint contengono i soli parametri aggiornati e non sovrascrivono i pesi pubblici. Non è il fine-tuning di tutto il backbone.
- `rolling`: refit settimanali con storico crescente. Ogni aggiornamento esclude i target non ancora osservabili al suo cutoff. Durante il test questi modelli possono incorporare gli esiti già trascorsi del test, secondo la politica fissata prima della valutazione: confronto presequenziale, distinto dai modelli statici.
- `ensemble`: ricerca di combinazioni convesse su tutti i candidati compatibili, senza limitarli ai tre migliori individualmente; confronto fra pesi globali e pesi dipendenti da pasto/variazione rapida noti all'origine. Il v2 resta candidato e fallback per orizzonte.
- `intervals`: confronto fra fasce per pasto, scale d'errore apprese con gruppi di regime e calibrazione rolling. Il registro degli errori usa soltanto esiti disponibili prima di ciascun aggiornamento. La scelta si basa sulla seconda metà di luglio; nessuna scelta sui nuovi errori di settembre–ottobre.
- `final`: congela entrambe le selezioni, rifà il training prima di agosto per i modelli statici, calibra e archivia tutte le previsioni prima del confronto descrittivo. Le fasce sono per singolo orizzonte, non simultanee sull'intera curva.
- `verify`: controlla hash, protezione degli artefatti v2, disponibilità dei target di training e degli errori di calibrazione, ricostruzione dell'ensemble, metriche e replay dei modelli salvati anche rimuovendo gli array degli esiti.

Il report è [outputs/personal-v3/report.md](outputs/personal-v3/report.md), con errori e confronti appaiati per giornate, copertura e ampiezza delle fasce, risultati per regimi e metriche di tutti i 36 passi. Il modello denso è scelto separatamente dall'ensemble a quattro orizzonti; i grafici li distinguono. Un tentativo senza vantaggio resta documentato.

Replay storico dopo il completamento e la verifica:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_limited.py -- .venv/bin/python -m src.v3.predict --at '2026-10-15T12:00:00+02:00'
```

Il comando ricalcola i modelli sull'ultima origine del benchmark precedente all'orario richiesto (tolleranza 20 minuti), restituisce i quattro orizzonti, le fasce registrate nello stato storico e la curva del modello denso separato. Non serve avere i target futuri per calcolare la previsione.

**Settembre–ottobre era già stato esaminato nella v2 e ha suggerito alcune ipotesi v3.** Il nuovo confronto è esplorativo: le selezioni precedono il nuovo scoring ma una conferma indipendente richiede nuovi dati. `RESUME.md` e `tasks.md` riportano lo stato effettivamente raggiunto.
