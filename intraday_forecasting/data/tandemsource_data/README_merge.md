# Unione degli export Tandem

Fare doppio clic su **aggiorna_merged.bat**, oppure eseguire:

```powershell
python .\merge_csv.py
```

Richiede Python 3.9 o successivo, senza pacchetti aggiuntivi.
La cartella di partenza è sempre quella dello script, anche se viene avviato
da un'altra posizione. Per sceglierne una diversa:

```powershell
python .\merge_csv.py --root "C:\percorso\cartella"
```

Lo script cerca i CSV direttamente in tutte le sottocartelle, escludendo
`merged`, i collegamenti e le cartelle nascoste o con prefisso `__`.
Riconosce i nomi attuali come `23092026-22102026-Bolo.csv` e i nomi originali
degli export come `LeonardiAndrea_1257006_22ott2026_1851-Bolo.csv`.
Raggruppa i file per il suffisso, senza distinguere maiuscole e minuscole.

In `merged` genera un CSV per ciascun tipo, con una sola intestazione.
Ordina tutte le righe dalla data più antica alla più recente:

| Tipo | Colonna utilizzata |
| --- | --- |
| Bolo | Data/Ora di completamento |
| CGM, Dosi basali, GLICEMIA | Data e ora dell'evento |
| Somministrazione oraria basale | Data dell'evento |

I valori originali restano invariati. I CSV sono scritti in UTF-8 con BOM,
separatore virgola. Le osservazioni duplicate vengono conservate.
Se due righe hanno la stessa data, mantengono il loro ordine di lettura.

Aggiungere nuove cartelle accanto allo script e avviarlo di nuovo:
i risultati vengono ricostruiti dalle sorgenti, senza riutilizzare i CSV già
presenti in `merged`. I file sorgente non vengono modificati.
Se un tipo non è più presente nelle sorgenti, il suo precedente file in
`merged` non viene cancellato automaticamente.

Se una data è assente o non valida oppure le colonne di uno stesso tipo
non corrispondono, lo script segnala file e problema prima di scrivere i
risultati. I CSV devono avere intestazioni e codifica UTF-8, come gli export attuali.
