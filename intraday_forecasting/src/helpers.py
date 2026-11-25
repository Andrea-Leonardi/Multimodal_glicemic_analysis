import pandas as pd


def summarize_dataframe(df: pd.DataFrame) -> None:
    """Stampa nome, valori unici, esempi, missing (%) e dtype di ogni colonna.

    Conteggio ed esempi escludono i valori mancanti. Gli esempi sono i primi
    tre valori distinti disponibili. Per un DataFrame senza righe, missing
    vale 0%. Il DataFrame originale non viene modificato.
    """
    rows = []
    for name, column in df.items():
        unique_values = column.dropna().unique()
        missing_pct = column.isna().mean() * 100 if len(column) else 0.0
        rows.append({
            "Colonna": name,
            "Valori unici": len(unique_values),
            "Esempi (max 3)": unique_values[:3].tolist(),
            "Missing (%)": f"{missing_pct:.2f}%",
            "Tipo dati": str(column.dtype),
        })

    summary = pd.DataFrame(rows, columns=[
        "Colonna", "Valori unici", "Esempi (max 3)", "Missing (%)", "Tipo dati"
    ])
    print(summary.to_string(index=False))
    print(f"\nRighe: {len(df)} | Colonne: {len(df.columns)}")


def drop_uniform(df: pd.DataFrame) -> pd.DataFrame:
    """Restituisce una copia senza colonne con un solo valore distinto.

    Stampa nome e valore delle colonne rimosse. I missing contano come un
    valore distinto: una colonna interamente missing viene rimossa, mentre
    una con un valore costante e alcuni missing viene mantenuta.
    """
    uniform = df.nunique(dropna=False).eq(1)
    for name, column in df.loc[:, uniform].items():
        print(f"Colonna rimossa: {name} | Valore unico: {column.iloc[0]!r}")

    return df.loc[:, ~uniform].copy()
