# TimestralGluco — attività

Registro operativo aggiornato durante il lavoro. Ambito: esperimento personale di previsione; nessun calcolo di dosi, comando della pompa o modifica della terapia.

## Decisioni e informazioni confermate

- L'utente inserisce talvolta carboidrati non mangiati per ottenere insulina extra. Le regole del notebook distinguono pasti e correzioni secondo questo uso personale: vanno conservate e versionate, insieme ai dati originali.
- Conservare separatamente carboidrati registrati nella pompa e carboidrati interpretati come mangiati. La classificazione resta euristica.
- Un bolo manuale senza carboidrati non diventa automatico: distinguere origine/modalità e scopo inferito.
- Non sovrascrivere gli export, i CSV esistenti o il notebook di pulizia.
- Timestamp originali privi di fuso: assumere Europe/Rome, rendere esplicita l'ipotesi e usare UTC per intervalli e griglia.
- Separare eventi pianificati, confermati e dati effettivamente disponibili; nessuna informazione futura conosciuta solo a posteriori nei test.
- Sviluppo fino a luglio 2026 (validazione interna cronologica); agosto per calibrazione; settembre–ottobre per test finale.
- 2026-10-24, nuova priorità: solo modelli addestrati/valutati sullo storico; raccolta e previsione real time rinviate. Timestral rimane sempre un benchmark, anche nelle prove di personalizzazione.
- L'orario di un pasto registrato è l'orario effettivo del pasto, confermato dall'utente. Non ipotizzare inserimenti ritardati nello storico personale. Per i boli prolungati distinguere inizio e completamento dell'erogazione.
- Insulina nella pompa: **Humalog (lispro)**, sia boli sia basale. Generare serie di attività/insulina residua da modelli documentati, mantenendole distinte dall'IOB misurata/mostrata dalla pompa.
- Il dominio d'interesse è quello effettivamente presente nei dati (minimo 75 mg/dL). Non dedicare esperimenti a fasce inferiori assenti.

## 1. Analisi e dati

- [x] Esaminare CSV puliti, notebook ed export originali senza modificarli.
- [x] Rilevare riclassificazione di 237 boli manuali, 26 boli estesi senza dettagli nei CSV puliti e 5 boli interrotti esclusi.
- [x] Verificare assenza di glicemie sotto 75 mg/dL e quasi continuità della serie CGM.
- [x] Rendere riproducibile l'audit: copertura, duplicati, interruzioni, unità, fuso e disponibilità delle variabili.
- [x] Preparare eventi verificati conservando categorie originali, regole personali, componenti estese e somministrazioni interrotte.
- [x] Preparare griglia causale a 5 minuti con maschere dei mancanti e provenienza temporale.
- [x] Gestire esplicitamente il timestamp basale duplicato con valori discordanti e verificare i riepiloghi orari.

## 2. Esperimenti storici

- [x] Definire configurazione riproducibile, split cronologici e protezioni contro contaminazione dai dati futuri.
- [x] Implementare baseline di persistenza e tendenza smorzata.
- [x] Verificare e integrare Timestral locale con revisione del codice e dei pesi fissata.
- [x] Confrontare contesti storici e modelli su validazione interna, senza usare il test per scegliere.
- [x] Calibrare intervalli sul solo agosto e misurare errori a 30/60/120/180 minuti.
- [x] Salvare previsioni, metriche per orizzonte, copertura/ampiezza e risultati per pasti, variazioni rapide e fasce glicemiche.
- [x] Confrontare glicemia sola e covariate manuali idealizzate; separare il confronto con storico completo della pompa. Restano da misurare latenze e qualità degli inserimenti reali.
- [x] Quantificare variabilità per giornate (500 bootstrap con confronti appaiati) e produrre grafici PNG/PDF e report Markdown esportabili.

## 3. Verifiche e documentazione

- [x] Testare causalità della griglia, split, regole personali, boli estesi/interrotti e calibrazione; aggiunti test su Share, registro e worker live (33 test superati).
- [x] Documentare installazione, comandi, assunzioni e limiti effettivamente osservati.
- [x] Eseguire la pipeline e registrare qui esiti, problemi e nuove attività emerse.

## 4. Fasi successive

- [x] Registro locale di eventi manuali: IOB, pasti, boli pianificati e conferme, con event_time e available_at (`src/journal.py`).
- [x] Implementare il client Share europeo dal codice ESP32 fornito: recupero ultime 24 ore, deduplicazione, timestamp UTC/arrivo, sessioni e timeout (`src/dexcom.py`), con test simulati.
- [ ] Provare l'autenticazione e raccolta Share reale con credenziali impostate localmente; misurare ritardi in regime separati dal recupero iniziale. Credenziali effettive non disponibili durante l'implementazione.
- [x] Implementare salvataggio delle previsioni prima dell'esito, aggiornamento su eventi/letture e confronto differito (`src/live.py`); verificato anche un flusso completo con dati sintetici in `/tmp`, senza chiamate Dexcom.
- [ ] Raccogliere un periodo prospettico reale e valutarlo: ritardi misurati, griglia ancorata all'emissione e qualità degli inserimenti differiscono dal benchmark iniziale.
- [ ] Sperimentare il condizionamento su IOB manuale e scenari con boli pianificati, distinguendo intenzioni e somministrazioni confermate. Attualmente sono registrati, ma NON ingressi del modello.
- [x] Verificare fattibilità concreta del fine-tuning locale dopo il benchmark zero-shot: v3 aggiorna davvero testa e, in una variante, ultimo blocco del decoder; equivalenza nativa e gradienti verificati.

## Note di avanzamento

- 2026-10-23: avvio implementazione autorizzato dall'utente; recepita la spiegazione sui carboidrati fittizi usati per le correzioni.
- Preparazione riproducibile in `src/data.py`; artefatti in `data/processed/`. Recuperati tutti i 4.566 boli, inclusi 28 prolungati e 5 interrotti. Regole personali confrontate con tutte le etichette preesistenti; unica variazione intenzionale: origine manuale dei 237 boli senza carboidrati.
- Primo benchmark baseline in `outputs/baselines/`: 682 origini di validazione, 740 di calibrazione, 1.236 di test (una/ora, criteri comuni sul passato). Nessun dato futuro usato per escludere origini; target mancanti esclusi solo dallo scoring.
- Nuova attività: verificare tempi di arrivo reali Share/Follow. La prima valutazione assume ritardo CGM zero e non va presentata come replica della latenza dal vivo.
- Nuova attività: valutare sensibilità alla griglia (alcune letture consecutive ricadono nello stesso intervallo) e al ritardo conservativo degli eventi estesi, mantenendo separato un nuovo test futuro per eventuali scelte successive.
- L'utente ha fornito il codice ESP32. Portato il flusso Share europeo in Python senza copiare identificativi/credenziali nel progetto. Autenticazione vera ancora da provare; nessuna password è stata richiesta in chat.
- Timestral installato nell'ambiente locale `.venv`, con PyTorch CPU; codice `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`, pesi 28M `49b26409ac87b5935539a5d8ffe7f397b720db47`, 209M `9874d252ab0aae16f7f7c999f98f98d6ad7d8365`. Inferenza offline verificata anche per isolamento fra esempi di un batch.
- Esperimento principale `outputs/experiment-v1/`: 229 origini luglio ogni tre ore per selezione CPU; 740 agosto e 1.236 settembre–ottobre ogni ora. Scelto contesto 24 ore su luglio per tutti i confronti Timestral. Queste cadenze non rappresentano ancora un replay ad ogni lettura.
- Benchmark completo concluso: 36 archivi di curve, metriche su 8 configurazioni finali e 4 orizzonti, report e grafici in `outputs/experiment-v1/`. Il test è ora consultato: nuove decisioni ispirate ai suoi risultati richiedono un nuovo periodo futuro indipendente.
- MAE (mg/dL) a 30/60/120/180 minuti: persistenza `15.45 / 24.90 / 38.45 / 47.48`; Timestral 28M sola glicemia `14.03 / 22.96 / 34.83 / 41.87`; 209M sola glicemia `13.83 / 22.73 / 34.86 / 41.86`. Aggiungere gli ingressi manuali cambia il MAE di meno di 0.16 mg/dL nei confronti effettuati; nessuna promessa di grande beneficio zero-shot dalle covariate.
- Fasce Timestral nominali al 90%: copertura osservata 88.2–91.7% sui quattro orizzonti e tutte le varianti. Per il 28M sola glicemia la fascia a tre ore è larga in media circa 174 mg/dL. Nessuna valutazione possibile sotto 70 mg/dL.
- Verifica finale: 33 test superati; input originali invariati (hash verificati); codici del manifest corrispondenti; prova completa live con dati sintetici separati in `/tmp`. Nessun collector reale avviato: variabili credenziali non configurate.

## Revisione critica — 2026-10-24

- [x] Ricalcolare il MAE a 30 minuti direttamente dagli archivi di previsione e dai target: 14.03266075 mg/dL, corrispondente alle metriche salvate. Non individuati errori di indice del quantile mediano nel percorso esaminato.
- [x] Separare limiti zero-shot, approssimazioni temporali e limiti della valutazione. I mesi precedenti non hanno addestrato pesi o un modello personale; l'inferenza vede al massimo 24 ore e covariate impulsive grezze.
- [x] Verificare copertura per regime: 28M sola glicemia, 30 minuti, dopo pasto registrato MAE 19.34 e copertura 90% effettiva 82.9%; senza pasto registrato nelle ultime tre ore MAE 11.28 e copertura 93.2%. A 60 minuti, dopo pasto, copertura 83.6%. La sintesi iniziale della copertura globale non evidenziava abbastanza questa differenza.
- [x] Verificare allineamento dei target: la lettura usata per il target nominale a 30 minuti cade nel test circa 25.9–29.27 minuti dopo l'origine di previsione. Il timestamp reale è salvato ma questa convenzione richiede una revisione per confrontare orizzonti effettivi e live. Non spiega da sola gli errori elevati e può rendere il task nominale più facile.
- [x] Confermare che tutti i contesti testati (6/12/24 ore) risultano migliori al limite superiore scelto: non è stata esplorata la saturazione oltre 24 ore.
- [x] Chiarire orario del pasto: l'utente conferma coincidenza tra registrazione e consumo. L'ipotesi di inserimento tardivo è superata; resta la distinzione tecnica tra inizio/fine dei boli prolungati.
- [x] Rivedere i boli prolungati senza anticipare quantità finali/status futuri. I 28 eventi estesi sono circa il 3.2% degli 875 pasti inferiti: il problema temporale è reale, ma non dimostrato come causa principale dell'errore medio.
- [ ] Unificare il replay storico e il worker live, con stessa griglia, stessa gestione degli arrivi e valutazione degli orizzonti effettivi; tenere distinti ritardo di osservazione e ritardo di acquisizione.
- [x] Addestrare un primo modello personale semplice (regressione regolarizzata e/o gradient boosting) su lag glicemici, tendenze, ora del giorno e storia temporale di pasti/insulina, con sole informazioni disponibili all'origine.
- [x] Confrontare impulsi grezzi con somme su finestre, tempo dagli eventi e rappresentazioni distribuite nel tempo, senza confondere proxy stimati con IOB misurata della pompa.
- [x] Usare validazioni cronologiche ripetute su mesi successivi, verificando prima 30/60 minuti, e confrontare anche un correttore personale degli errori di Timestral. Nessuna promessa di miglioramento prima delle misure.
- [x] Provare contesti Timestral oltre 24 ore sullo sviluppo/validazione; il modello 209M non ha mostrato un guadagno tale da renderlo la priorità.
- [x] Valutare fasce condizionate al regime (pasti/andamento), ampiezza e punteggi probabilistici; evitare di trattare la copertura aggregata come affidabilità uniforme.
- [ ] Integrare piani annunciati come scenari condizionati e IOB con ingressi coerenti tra esperimenti e live; non introdurre eventi futuri degli export come se fossero stati annunciati.
- [x] Conservare v1 come riferimento; i risultati settembre–ottobre sono già stati consultati. Le nuove scelte richiedono validazione interna temporale e successiva conferma su un periodo futuro indipendente.

## Esperimento storico personale v2 — completato il 25 ottobre 2026

- [x] Preparare dataset ancorato alle letture CGM effettive, target con tolleranza temporale esplicita ed eventi pasto coerenti con il chiarimento; totale esteso confermato al completamento, senza redistribuirlo retroattivamente.
- [x] Implementare e verificare curve esponenziali di attività/residuo da fonti primarie OpenAPS/Loop senza cambiare impostazioni terapeutiche.
- [x] Confrontare il gruppo di quattro ipotesi durata/picco per Humalog con la sola curva 360/75, a parità di parametri CatBoost. Il guadagno rispetto agli eventi grezzi è limitato; nessuna durata individuale clinica identificata.
- [x] Creare ingressi temporali: lag, tendenze, calendario, pasti/insulina per finestre, attività/residuo e proiezioni delle sole somministrazioni già avvenute.
- [x] Confrontare modelli personali lineari, alberi/boosting e reti neurali con validazioni cronologiche ripetute, escludendo target oltre il confine di addestramento.
- [x] Provare Timestral con contesti estesi e covariate trasformate; verificare un adattamento personale e/o correttore supervisionato senza modificare il benchmark originale.
- [x] Selezionare modelli/ensemble solo sulle validazioni, calibrare su agosto e congelare le scelte prima della valutazione retrospettiva settembre–ottobre.
- [x] Salvare modelli riutilizzabili, previsioni, configurazioni, metriche e rapporto confrontabile sul medesimo insieme di origini.
- [x] Testare causalità, formule, split e riproducibilità; aggiornare gli esiti e le attività successive.

### Avanzamento addestramento storico

- [x] Completare 16 configurazioni supervisionate su tre finestre temporali (aprile, giugno, luglio): Ridge, Extra Trees, histogram boosting, CatBoost, MLP e convoluzioni dilatate. Inclusi confronti con sola glicemia, eventi grezzi e sola ipotesi insulinica 360/75.
- [x] Salvare previsioni di validazione, tempi, dimensioni di training e hash in `outputs/personal-v2/oof/` e `tabular-selection.json`; nessun target di training attraversa il confine di validazione.
- [x] Aggiungere dodici test su split, imputazione/standardizzazione, offset multiorizzonte, pesi convessi, calibrazione per regime e indipendenza degli ingressi Timestral dalle osservazioni future.
- [x] Completare ricerca Timestral 28M/209M con 24/72/168 ore e variabili fisiche/calendario. Se il limite superiore continua a migliorare, verificare 14/28 giorni per il 28M prima della scelta finale.
- [x] Addestrare correttore CatBoost e readout Ridge personale delle rappresentazioni Timestral congelate. Il readout non va descritto come fine-tuning completo dei pesi.
- [x] Congelare modelli e pesi, rifare il training fino a luglio, calibrare agosto per regime pasto/non pasto e valutare settembre–ottobre sulle stesse origini. Il test rimane retrospettivo già consultato, non nuovo cieco.
- Primo esito sulle sole validazioni: configurazioni migliori circa 11.5–11.7 mg/dL a 30 minuti e 26–27 mg/dL a tre ore. Sono metriche di selezione, non risultati finali sul test.
- Timestral 28M (validazione oraria): estendendo il contesto da 24 a 168 ore il MAE passa circa da `13.33/22.05/33.93/41.68` a `12.57/19.95/27.76/31.23`; esplorazione più lunga ancora in corso. Nessuna scelta viene fatta sul test.

## Blocco memoria WSL — priorità prima dell'addestramento

- [x] Individuare la causa: pandas restituisce `datetime64[us]`; `.asi8 / 1e9` comprimeva gli intervalli di 1000 volte. La matrice basale diventava `94.237 × 85.925` (circa 60.33 GiB per singolo array float64).
- [x] Normalizzare esplicitamente i timestamp a nanosecondi prima di convertirli in secondi Unix; testare equivalenza delle risoluzioni us/ns e intervallo reale di 300 secondi.
- [x] Elaborare a blocchi esposizioni, curve insuliniche, statistiche temporali e sequenze; evitare matrici float64 duplicate per assemblare gli ingressi.
- [x] Aggiungere guardia preventiva alle dimensioni di `exposure()` e limitare anche i temporanei di `insulin_curve()` e delle finestre Timestral.
- [x] Conservare un riferimento vettoriale limitato a 4.096 letture (`tests/personal_dense_reference.py`), correggendo soltanto l'unità temporale: il vecchio errore us/ns non è un risultato numerico da riprodurre.
- [x] Verificare formule e intero dataset su campione: differenza massima zero su tempi, ingressi, etichette, sequenze e maschere; otto test di regressione superati.
- [x] Ripetere verifica campione e costruzione completa sotto limite `RLIMIT_AS` di 2 GiB, riservando almeno 4 GiB disponibili alle altre applicazioni; registrare picco RSS e durata.
- [x] Salvare array `.npy` e metadati in modo atomico per riprendere l'addestramento senza ricostruzioni o copie integrali in RAM.
- Esiti protetti del 2026-10-24: equivalenza su 4.096 letture con differenza massima **zero** su tutti gli array; campione nuovo 162.32 MiB RSS. Costruzione completa: **490.06 MiB RSS**, **5.83 s**, 94.237 letture, 207 variabili, 90.797 origini utilizzabili; array finali 235.19 MiB. Prima del processo completo erano disponibili 13.112 MiB; limite di spazio virtuale 2.048 MiB e riserva richiesta 4.096 MiB. Tutti gli otto test dedicati passano sotto lo stesso limite.
- Rapporti: `outputs/memory-check/sample-equivalence.json`, `sample-build.json`, `full-build.json`. Includono hash di codice/configurazione/dati e limiti effettivamente applicati.
- Dataset completo pubblicato atomicamente in `outputs/memory-check/dataset/`: `features.npy`, `sequence.npy`, `times.npy`, `glucose.npy`, `targets.npy`, `target_indices.npy`, `eligible.npy`, ordine delle variabili e metadati. Nessun training ha ancora usato questi array.
- Stato di ripresa: addestramento v2 **non iniziato**. Prossimo passo: riaprire gli array con `numpy.load(..., mmap_mode='r')`, creare fold cronologici e avviare i confronti personali/Timestral con batch e limiti espliciti. Il dato corretto usa secondi Unix e intervalli sensore mediani di 300 secondi.
- Comandi di verifica ripetibili: `.venv/bin/python scripts/check_personal_memory.py sample`, poi `.venv/bin/python scripts/check_personal_memory.py full`. Il secondo rifiuta l'esecuzione se il campione non è stato verificato sugli stessi file. Per codice/dati cambiati usare un nuovo `--output` per conservare gli artefatti precedenti.
- [x] Audit separato delle serializzazioni `.asi8` nel benchmark v1: gli archivi memorizzano origini in microsecondi, ma il grafico le interpretava come nanosecondi (data 1970). Le metriche usano gli indici di riga, non quel metadato, e non cambiano. Gli artefatti v1 restano conservati; v2 usa secondi Unix espliciti e non eredita il problema.
- 2026-10-24: l'utente conferma 32 GB di RAM fisica e autorizza processi oltre 2 GiB. WSL espone attualmente circa 15.5 GiB. In `config/resources.json` configurato profilo addestramento a **8 GiB di spazio virtuale**, fino a 8 thread e almeno **4 GiB disponibili riservati** alle altre applicazioni; il limite effettivo viene ridotto se la memoria disponibile non basta.
- [x] Aggiungere `scripts/run_limited.py` per applicare il profilo prima degli import numerici. Il costruttore verificato mantiene 2 GiB: gli ingressi richiedono circa 490 MiB e non serve aumentarne il picco. Il nuovo profilo risorse è separato dagli hash del dataset, che rimane riutilizzabile.
- Per riprendere: `.venv/bin/python scripts/run_limited.py -- .venv/bin/python ...`. Eseguire un addestramento per volta con thread a memoria condivisa; evitare lavoratori indipendenti che moltiplichino il budget per processo.
- [x] Su richiesta dell'utente, impostare WSL a **24 GB**: aggiunto `memory=24GB` nella sezione `[wsl2]` di `C:\Users\Andrea\.wslconfig`, preservando rete mirrored, DNS tunneling e autoProxy. Scrittura e backup verificati.
- Backup Windows: `C:\Users\Andrea\.wslconfig.backup-20261024T160149731199Z`.
- [x] Attivare la nuova configurazione: riavvio eseguito dall'utente e verificato il 2026-10-24. `/proc/meminfo` riporta **23.47 GiB totali**, circa **20.25 GiB disponibili** al controllo e 6 GiB di swap; `.wslconfig` conferma `memory=24GB`.
- [x] Ampliare i profili del progetto dopo il riavvio: addestramento **16 GiB / 8 thread**, costruzione **4 GiB / 1 thread**, controllo preventivo di **4 GiB disponibili** per altre applicazioni. La verifica dedicata del costruttore resta a 2 GiB, come già validato; non cambiano logica, blocchi o dataset salvato.

### Accelerazione verificata e ripresa

- [x] Rilevare RTX 3070 (8 GiB) in WSL; creare `.venv-gpu` separato senza modificare l'ambiente CUDA esistente `arena`. Versioni scientifiche allineate al progetto; Python 3.13 e build Torch CUDA registrati a parte.
- [x] Verificare CPU/CUDA su 28M glicemia/24h, 28M covariate/72h e 209M glicemia/168h: scarto massimo previsioni <0.000046 mg/dL, quantili <0.000184 mg/dL, rappresentazioni <0.000001. Float32, TF32 disabilitato. Esiti in `outputs/personal-v2/cuda-equivalence.json`.
- [x] Limitare CUDA con cgroup systemd: massimo 16 GiB di RAM fisica, soglia 14 GiB, massimo 1 GiB swap; riserva iniziale di almeno 4 GiB disponibili. VRAM limitata a 4 GiB e almeno 1 GiB libero. `RLIMIT_AS` resta per CPU, non per CUDA.
- [x] Interrompere il solo job Timestral CPU dopo la verifica di equivalenza e riprendere le cache complete sulla GPU; aggiungere log separati di hardware/versioni/hash senza invalidare gli esperimenti CPU completati.
- [x] Confrontare training denso a 5 minuti per i finalisti CatBoost e TCN sulle stesse origini di validazione.
- [x] Completare gli stadi Timestral, contesti 14/28 giorni, ibridi, selezione/refit e valutazione finale.
- [x] Verificare inferenza dei modelli salvati senza accesso ai target e riportare metriche finali, fasce e limiti in `report.md`.

- [x] Confrontare loss MAE separata per orizzonte e finestra di training di 90 giorni, con split e origini identici; includere la scelta nella selezione prima del test.
- [x] Estendere anche Timestral 28M con covariate da 72 a 168/336 ore: il limite di 72 ore continuava a migliorare sulle validazioni.

- Training fitto completato: CatBoost MAE validazione `11.63/18.23/24.45/26.86`; TCN `11.75/18.10/24.16/26.73`. Non emerge un vantaggio sistematico rispetto al campionamento ogni tre letture.

- [x] Verificare saturazione Timestral 28M glicemia a 8/16 settimane (entro 32.768 passi), 28M covariate a 4 settimane e 209M glicemia a 2/4 settimane: a 4 settimane il 28M continuava a migliorare sulle validazioni, test non consultato.
- Nota risorse: prova parallela CPU con limite virtuale 3 GiB arrestata dal limite prima del training (nessun esaurimento RAM WSL). Ripresa con limite CPU fisico 3 GiB e virtuale 8 GiB; insieme al processo GPU fisico massimo 16 GiB, totale massimo 19 GiB, con oltre 4 GiB lasciati al sistema. Log distinti `execution-cpu-direct*.json`.

- CatBoost con quattro loss MAE e storico completo: validazione `11.28/17.82/24.01/26.46`, contro `11.46/18.16/24.59/26.81` con ultimi 90 giorni. La riduzione della finestra non aiuta in questo confronto. Picco RAM circa 525 MiB, cgroup fisico 3 GiB rispettato.

- [x] Verificare anche il codice Timestral installato nei due ambienti CPU/CUDA: file Python identici, hash conservati in `timestral-installed-sources.json`.

- [x] Completare 23 configurazioni supervisionate (16 iniziali, 2 training densi, 2 loss/finestra, 3 correttori/readout) su tre validazioni. Picco del processo ibrido: circa 1.55 GiB; nessun fine-tuning completo del backbone dichiarato.

- Verifica suite completa: il profilo virtuale build da 4 GiB è troppo stretto dopo aver caricato insieme le dipendenze di tutti i test e CatBoost; processo di test interrotto con SIGTERM dopo mancata risposta a SIGINT. RSS circa 267 MiB e contatori cgroup `oom=0`, `oom_kill=0`: nessun esaurimento RAM WSL. Ripetere i test completi nel profilo virtuale training da 16 GiB, con limite fisico separato 2 GiB. Costruttore e sua verifica dedicata restano validati nei limiti originali.

## Checkpoint richiesto dall'utente — prima del reset dei token

- [x] Fermare correttamente il solo processo GPU `long`, conservando tutti gli artefatti completi; stato del log `interrupted_by_user`. Ultimo avanzamento denso: 2.400/8.516, file incompleto da ripetere.
- [x] Verificare 53 test completi nel profilo virtuale training, con cgroup fisico 2 GiB: tutti superati in 1,7 secondi. Log `outputs/personal-v2/test-suite.log`.
- [x] Scrivere `RESUME.md` con conclusioni provvisorie, numeri comparabili, punto esatto e comandi di ripresa.
- [x] Riprendere `.venv/bin/python scripts/run_cuda.py long`; completare il confronto denso 28M/covariate/28 giorni e i contesti 209M a 14/28 giorni.
- [x] Eseguire `.venv/bin/python scripts/run_cuda.py final`, senza ulteriori scelte basate sul test; modelli finali e calibrazione salvati il 25 ottobre.
- [x] Verificare inferenza senza target, artefatti, grafici e report finale; aggiornare questo registro e comunicare le metriche finali.

Conclusione provvisoria sulle stesse origini di validazione: CatBoost MAE `11.28/17.82/24.01/26.46`; Timestral 28M/covariate/14 giorni `11.64/18.76/25.56/28.26`. Sono metriche di selezione: nessuna dichiarazione di accuratezza finale su nuovi dati.

### Immagini richieste dall'utente

- [x] Generare anteprime del miglior modello sui confronti completi: CatBoost MAE. Confronto errori, tre emissioni a orari prefissati e giornata completa del 2 luglio, PNG/PDF in `outputs/personal-v2/validation-figures/`.
- [x] Verificare corrispondenza delle origini, metriche ricalcolate uguali ai risultati salvati e resa visiva dei tre grafici. Codice numerico degli esperimenti invariato; nessun training ripreso o scoring del test. `RESUME.md` resta valido.

## Ripresa del 25 ottobre 2026

- [x] Rileggere `RESUME.md`, verificare memoria WSL (23,47 GiB), stato Git e hash del codice di tutti gli stadi: coerenti con le cache salvate.
- [x] Completare lo stadio `long`, riutilizzando la selezione preliminare e ripetendo il solo file denso non completato.
- [x] Congelare la selezione, rifare il training finale, calibrare agosto e valutare settembre–ottobre.
- [x] Verificare inferenza senza target, artefatti e grafici; aggiornare risultati e punto di ripresa.

- [x] Completare Timestral 28M/covariate/28 giorni sulle 8.516 origini comuni: MAE `11.47/18.38/24.92/27.33`. Migliora il contesto 14 giorni; ricerca 209M ancora in corso.

- [x] Completare entrambi i confronti Timestral lunghi e avviare `final`: selezione sulle sole validazioni, prima dello scoring settembre–ottobre.

## Esito finale — 25 ottobre 2026

- [x] Completare 18 configurazioni Timestral preliminari (contesti fino a 16 settimane per glicemia 28M e 4 settimane per covariate/209M), oltre alle 23 configurazioni supervisionate.
- [x] Congelare configurazioni e pesi prima dello scoring: `selection-frozen.json`, hash antecedente allo scoring in `selection-before-scoring.json`.
- [x] Rifare il training su 22.437 origini anteriori ad agosto, salvare 12 modelli, calibrare su 2.932 origini di agosto e valutare 18 modelli/riferimenti su 4.878 origini comuni settembre–ottobre.
- [x] Verificare provenienza di sorgenti/dati, origini, finitezza delle previsioni, ricostruzione dei pesi, separazione dei target e metriche/copertura ricalcolate: `verified.json` passato.
- [x] Ripetere tre previsioni CPU con e senza array dei target: identiche; differenza massima rispetto alle cache CPU/CUDA 0,000026 mg/dL.
- [x] Generare rapporto, confronto Timestral configurato, grafici dei regimi e della copertura nelle variazioni rapide, tre esempi del 2 settembre con fasce 90% e giornata completa.
- [x] Aggiornare `RESUME.md` con lo stato concluso e i comandi di riuso. Nessun training o scoring resta da terminare per v2.

MAE test a 30/60/120/180 minuti: modello selezionato `11.30/18.06/24.74/27.21`; Timestral 28M/covariate/28 giorni `11.57/18.71/25.29/27.82`; Timestral 28M/glicemia/24 ore `13.23/22.31/34.18/41.57`. Riduzione rispetto al riferimento 24 ore: 14,6% / 19,0% / 27,6% / 34,6%. Il confronto con il Timestral configurato mostra un vantaggio personale aggiuntivo più contenuto.

Fasce nominali 90%: copertura globale `90.14/90.09/89.16/88.83%`, ampiezza media `50.68/82.83/109.35/116.42 mg/dL`. Nelle variazioni rapide la copertura a 60 minuti è 81,1% su 185 origini: limite esplicito nel rapporto, nessuna ricalibrazione dopo aver visto il test.

Picco finale: circa 4,07 GiB RAM e 905 MiB VRAM, entro i limiti; verifica artefatti circa 747 MiB RAM. Il verificatore usa somme float64 come riferimento indipendente: massima differenza dalle metriche CSV circa 0,0000081 mg/dL. La prima riduzione su array float32 aveva introdotto un falso allarme di arrotondamento (~0,000134 sul RMSE della persistenza); nessuna modifica a previsioni, modelli o metriche originali. I 53 test della logica già superati restano validi; controlli aggiuntivi eseguiti su artefatti reali e grafici.

### Eventuali esperimenti successivi, separati da v2

- [ ] Confermare i risultati su un nuovo periodo indipendente: settembre–ottobre era già stato consultato nella v1.
- [x] Valutare fasce anche condizionate a variazione rapida con protocollo separato (v3); la conferma su nuovi dati resta l'attività indipendente sopra.
- [x] Valutare previsioni dirette ai passi intermedi per ottenere una curva densa oltre ai quattro orizzonti: v3 emette e confronta 36 passi.
- Real time, piani annunciati e fine-tuning completo del backbone restano fasi successive.

## Esperimento v3 — implementazione completa richiesta dall'utente

V2 resta congelato. Nuovi moduli in `src/v3/`, configurazione `config/personal-v3.json`, output separato `outputs/personal-v3`. Il test storico già consultato rimane esplorativo; conferma indipendente richiede nuovi dati.

- [x] Fissare protocollo, configurazioni e limiti prima dei nuovi confronti.
- [x] Preparare cache Timestral 28M/covariate/28 giorni per il training del correttore forte, riusando le previsioni v2 e salvando avanzamenti riprendibili.
- [x] Implementare e confrontare correttori forti e loss pesate per variazioni rapide.
- [x] Implementare e confrontare reti che emettono 36 passi, con loss su livelli e pendenze.
- [x] Implementare adattamento Timestral con testa e testa+ultimo blocco; verificare equivalenza iniziale e gradienti prima del training.
- [x] Simulare refit settimanali di modello diretto e correttore, con target di training già disponibili al cutoff.
- [x] Selezionare ensemble senza restrizione ai tre migliori individuali; confrontare pesi globali e dipendenti dal regime noto.
- [x] Confrontare fasce adattive, gruppi pasto/variazione rapida e calibrazione rolling con errori disponibili al momento.
- [x] Congelare scelte sulle validazioni, rifare training/calibrazione, valutare e confrontare v2/v3 senza adattarsi al test.
- [x] Verificare causalità, replay senza target, memoria, modelli salvati e grafici; aggiornare `RESUME.md` e report con tutti gli esiti.

- [x] Superare 10 test v3 preliminari: percorso Timestral differenziabile uguale al nativo, aggiornamento limitato a testa/ultimo blocco, ensemble complementare, target densi, refit settimanale e calibrazione rolling causali. Eseguiti sotto cgroup CPU 3 GiB mentre la cache GPU resta limitata a 16 GiB.
- [x] Eseguire la suite completa nel medesimo limite CPU 3 GiB: **63 test superati** in 1,68 secondi, log `outputs/personal-v3/test-suite.log`.

Esiti sulle sole validazioni v3 (MAE a 30/60/120/180 minuti, mg/dL): correttore Timestral lungo `11.075/17.688/24.024/26.584`; correttore pesato `11.053/17.633/24.021/26.555`; diretto pesato `11.335/17.861/24.033/26.370`; refit diretto settimanale `11.26/17.74/23.74/26.17`; correttore settimanale `11.07/17.69/23.96/26.44`. I modelli densi sono confrontati sulle medesime origini e conservano tutti i 36 passi.

- Fine-tuning completato: testa sola `12.706/20.565/29.635/37.565`, contro pesi pubblici con glicemia/24h `13.055/21.977/33.868/41.789`; testa+ultimo blocco `11.770/18.899/25.755/29.194`, contro pesi pubblici con covariate/72h `12.047/19.735/27.688/31.126`. Aggiornati rispettivamente 1.880.064 e 4.535.424 parametri; equivalenza iniziale con il percorso nativo: scarto massimo zero. Picchi 2.282,90 MiB RSS e 421,96 MiB VRAM.
- Ensemble congelato sulle validazioni: globale `10.99/17.52/23.48/25.69`; pesi contestuali non superano il margine prefissato. La curva densa è scelta separatamente. Nessun nuovo scoring del test prima della scelta delle fasce.
- Precalcolo finale dei due refit settimanali completato senza scoring del test, in CPU entro 3 GiB fisici; checkpoint e tracciati di disponibilità conservati.
- Fasce congelate: `adaptive_group`, scelta sulla seconda metà di luglio con interval score medio `124.40`, contro `137.47` per pasto/non pasto e `125.53` per rolling. La variante rolling resta valutata e archiviata. Prova degli hash delle due selezioni in `selection-before-scoring.json`: nessun file di metriche del nuovo test presente all'avvio.

### Conclusione v3 — 25 ottobre 2026

- MAE sulle stesse 4.878 origini del test: **`11.044/17.584/23.692/25.861 mg/dL`**, contro v2 `11.296/18.062/24.740/27.206`; guadagno circa 2–5%. Bootstrap appaiato per giornate nel report. La v3 include refit settimanali, il v2 è statico; il periodo già consultato mantiene valore esplorativo.
- Fasce scelte: copertura globale nominale 90% effettiva `89.99/90.13/90.84/89.51%`, ampiezza `46.70/74.86/104.26/108.51 mg/dL`. A 60 minuti in variazioni rapide copertura `87.57%` (v2 `81.1%`); a tre ore `85.71%`: limite ancora aperto, nessun aggiustamento sul test.
- Fine-tuning effettivo migliora il modello pubblico con identici ingressi; pesi finali riutilizzabili. Modello denso separato `trajectory_huber`, 36 passi a cinque minuti. L'errore medio del tempo del massimo nella finestra resta circa 43 minuti; non confondere questa diagnostica con la previsione di un picco fisiologico completo.
- **63 test passati**, verifica finale `passed`: sette modelli statici, 54 checkpoint settimanali, v2 invariato, tre replay identici senza target, scarto massimo CPU/cache GPU `0.0000153 mg/dL`, metriche ricalcolate entro `1.3e-12 mg/dL`. Picco finale `2872.66 MiB RSS / 421.96 MiB VRAM` nei limiti del progetto.
- Report e grafici controllati: `outputs/personal-v3/report.md`, `comparison.png/pdf`, `examples.png/pdf`, `uncertainty.png`. Rimossa una warning grafica sui target futuri interamente mancanti, senza cambiare dati, modelli o metriche; verificati nuovamente tutti gli artefatti dopo la rigenerazione del report.
- `RESUME.md` e README aggiornati. Nessun training attivo, nessun nuovo commit/push. V2 e input originali conservati.

- [ ] Confermare i modelli congelati su un periodo nuovo e indipendente prima di nuove scelte. Richiede nuovi dati; non è lavoro di training lasciato incompleto.
