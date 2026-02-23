import pandas as pd
import numpy as np
import config as cfg


def main():
    # ============================
    # 1) LOAD + CLEAN CGM (Glooko)
    # ============================
    
    EXPORTS = [
        cfg.GLOOKO_EXPORT_5,
        cfg.GLOOKO_EXPORT_4,
    ]
    
    dfs = []
    for exp_dir in EXPORTS:
        for f in sorted(exp_dir.glob(cfg.CGM_PATTERN)):
            tmp = pd.read_csv(f, skiprows=1)  # salta la prima riga (header extra di Glooko)
            dfs.append(tmp)
    
    df_bg = pd.concat(dfs, ignore_index=True)
    
    # rinomina colonne chiave
    df_bg = df_bg.rename(
        columns={
            "Data e ora": "Time",
            "Valore glicemia CGM (mg/dl)": "Bg",
        }
    )
    
    # parse datetime
    df_bg["Time"] = pd.to_datetime(df_bg["Time"], dayfirst=True, errors="coerce")
    
    # parse Bg (string -> float)
    df_bg["Bg"] = (
        df_bg["Bg"]
        .astype(str)
        .str.replace(",", ".", regex=False)   # nel caso di virgola decimale
    )
    df_bg["Bg"] = pd.to_numeric(df_bg["Bg"], errors="coerce")
    
    # tieni solo righe valide
    df_bg = df_bg.dropna(subset=["Time", "Bg"])
    
    # imposta index time e ordina
    df_bg = (
        df_bg.set_index("Time")
             .sort_index()
             .drop(columns="Numero di serie", errors="ignore")
    )
    
    # ===========================================
    # 2) LIMIT RANGE TO XIAOMI DATA WINDOW
    # ===========================================
    
    xiaomi_data = pd.read_parquet(cfg.XIAOMI_DATA_CLEANED)
    
    start_date = xiaomi_data.index.min()
    end_date = xiaomi_data.index.max()
    
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    
    df_bg = df_bg[(df_bg.index >= start_ts) & (df_bg.index < end_ts)]
    
    # ===========================================
    # 3) AVERAGE DUPLICATES (SAME TIMESTAMP)
    # ===========================================
    
    # Se per lo stesso timestamp hai più righe, fai la media della glicemia.
    df_bg = df_bg.groupby(level=0)["Bg"].mean().to_frame()
    
    # ============================
    # 4) BUILD dt (time weights)
    # ============================
    
    df = df_bg.sort_index().copy()
    
    df["t_next"] = df.index.to_series().shift(-1)
    df["dt_s"] = (df["t_next"] - df.index.to_series()).dt.total_seconds()
    
    # cap dei buchi (evita che un singolo valore "pesi" ore)
    CAP_SECONDS = 15 * 60
    df["dt_s"] = df["dt_s"].clip(lower=0, upper=CAP_SECONDS)
    
    # togli ultima riga (dt NaN) e righe strane
    df = df.dropna(subset=["dt_s", "Bg"])
    df = df[df["dt_s"] > 0]  # extra sicurezza
    
    # ============================
    # 5) DAILY STATS (time-weighted)
    # ============================
    LOW = 70
    HIGH = 200
    
    def daily_stats(g: pd.DataFrame) -> pd.Series:
        # pesi temporali (secondi) e glicemia
        w_s = g["dt_s"].to_numpy(dtype=float)
        x = g["Bg"].to_numpy(dtype=float)
    
        # filtra valori validi
        mask = np.isfinite(w_s) & np.isfinite(x) & (w_s > 0)
        w_s = w_s[mask]
        x = x[mask]
    
        if w_s.size == 0 or w_s.sum() == 0:
            return pd.Series(dtype="float64")
    
        # converto i pesi in MINUTI (comodo per AUC e interpretazione)
        w_min = w_s / 60.0
    
        # ======= STATISTICHE TIME-WEIGHTED =======
        wmean = np.average(x, weights=w_s)
    
        wvar = np.average((x - wmean) ** 2, weights=w_s)
        wsd = np.sqrt(wvar)
        cv = wsd / wmean if wmean != 0 else np.nan
    
        # ======= TIME IN RANGES (pesato) =======
        tir = w_s[(x >= LOW) & (x <= HIGH)].sum()
        tar = w_s[x > HIGH].sum()
    
    
        total = w_s.sum()
        tir_p = 100 * tir / total
        tar_p = 100 * tar / total
    
        # ======= AUC (excess) =======
        # AUC sopra soglia: somma di (Bg - HIGH) quando Bg > HIGH, pesata per tempo
        excess_high = np.clip(x - HIGH, 0, None)
        auc_above_high_mgdl_min = np.sum(excess_high * w_min)
    
        # (opzionale) AUC totale (area sotto la curva glicemia)
        auc_total_mgdl_min = np.sum(x * w_min)
        
        auc_above_rate = 100 * auc_above_high_mgdl_min / auc_total_mgdl_min
    
        return pd.Series({
            "mean": wmean,
            "sd": wsd,
            "cv": cv,
    
            "min": np.nanmin(x),
            "median": np.nanmedian(x),
            "max": np.nanmax(x),
    
            "tir_%": tir_p,
            "tar_%": tar_p,
    
            "auc_above_limit_rate": auc_above_rate
    
        })
    
    
    daily_bg = df.groupby(df.index.date).apply(daily_stats)
    
    daily_glooko = daily_bg.add_prefix("bg_")
    
    # ============================
    # IMPORT DATI SU CAMBI INSULINA
    # ============================
    
    dfs = []
    for exp_dir in EXPORTS:
        f = exp_dir / cfg.ALARMS_FILE
        if f.exists():
            dfs.append(pd.read_csv(f, skiprows=1))  # se serve davvero skiprows=1
    df_alarms = pd.concat(dfs, ignore_index=True)
    
    
    # rinomina colonne chiave
    df_alarms = df_alarms.rename(
        columns={
            "Data e ora": "Time",
            "Allarme/Evento": "Cart_load",
        }
    )
    
    # parse datetime
    df_alarms["Time"] = pd.to_datetime(df_alarms["Time"], dayfirst=True, errors="coerce")
    df_alarms = df_alarms.dropna(subset=["Time", "Cart_load"])
    
    # imposta index time e ordina
    df_alarms = (
        df_alarms.set_index("Time")
             .sort_index()
             .drop(columns="Numero di serie", errors="ignore")
    )
    
    df_alarms = df_alarms[(df_alarms.index >= start_ts) & (df_alarms.index < end_ts)]
    
    df_alarms= df_alarms[df_alarms["Cart_load"] == "Cartridge Loaded"] 
    
    # tieni solo gli eventi "Cartridge Loaded"
    df_alarms = df_alarms.loc[df_alarms["Cart_load"].eq("Cartridge Loaded")].copy()
    
    # feature numerica
    df_alarms["cartridge_loaded"] = 1
    
    # porta a livello giornaliero: index -> Date (stesso tipo di daily_bg)
    df_alarms["Date"] = df_alarms.index.date
    
    # se in un giorno ci sono più eventi, prendo max (quindi 1)
    daily_cart = df_alarms.groupby("Date")["cartridge_loaded"].max().to_frame()
    
    # join su indice giornaliero
    daily_glooko = daily_glooko.join(daily_cart, how="left")
    
    # giorni senza evento -> 0
    daily_glooko["cartridge_loaded"] = daily_glooko["cartridge_loaded"].fillna(0).astype(int)
    
    # ============================
    # IMPORT DATI SU INSULINA
    # ============================
    
    dfs = []
    for exp_dir in EXPORTS:
        f = exp_dir / cfg.INSULIN_SUBDIR / cfg.INSULIN_FILE
        if f.exists():
            dfs.append(pd.read_csv(f, skiprows=1))  # se serve davvero skiprows=1
    df_ins = pd.concat(dfs, ignore_index=True)
    
    
    # rinomina colonne chiave
    df_ins = df_ins.rename(columns={
        "Data e ora": "Time",
        "Bolo totale (U)":"ins_bolo_tot",
        "Insulina totale (U)":"ins_tot",
        "Basale totale (U)":"ins_basal_tot"
        })
    
    # parse datetime
    df_ins["Time"] = pd.to_datetime(df_ins["Time"], dayfirst=True, errors="coerce")
    
    # parse data(string -> float)
    toParse = ["ins_bolo_tot","ins_tot","ins_basal_tot"]
    
    for col in toParse:
        df_ins[col] = (
            df_ins[col]
            .astype(str)
            .str.replace(",", ".", regex=False)   # nel caso di virgola decimale
        )
        df_ins[col] = pd.to_numeric(df_ins[col], errors="coerce")
    
    
    # imposta index time e ordina
    df_ins = (
        df_ins.set_index("Time")
             .sort_index()
             .drop(columns="Numero di serie", errors="ignore")
    )
    
    df_ins = df_ins[(df_ins.index >= start_ts) & (df_ins.index < end_ts)]
    
    # porta a livello giornaliero: index -> Date (stesso tipo di daily_bg)
    df_ins.index = df_ins.index.date
    
    
    # join su indice giornaliero
    daily_glooko = daily_glooko.join(df_ins, how="left")
    
    
    # ============================
    # IMPORT DATI SU CARBOIDRATI
    # ============================
    
    dfs = []
    for exp_dir in EXPORTS:
        f = exp_dir / cfg.INSULIN_SUBDIR / cfg.BOLUS_FILE
        if f.exists():
            dfs.append(pd.read_csv(f, skiprows=1))  # se serve davvero skiprows=1
    df_carb = pd.concat(dfs, ignore_index=True).fillna(0)
    
    # rinomina colonne chiave
    df_carb = df_carb.rename(columns={
        "Data e ora": "Time",
        "Immissione glicemia (mg/dl)":"glicemia",
        "Consumo di carboidrati (g)":"carb",
        "Insulina erogata (U)":"insulina",
    })
    
    # parse datetime
    df_carb["Time"] = pd.to_datetime(df_carb["Time"], dayfirst=True, errors="coerce")
    
    # imposta index time e ordina
    df_carb = (
        df_carb.set_index("Time")
             .sort_index()
             .drop(columns={"Numero di serie",
                            "Tipo di insulina",
                            "Rapporto carboidrati",
                            "Erogazione iniziale (U)", 
                            "Erogazione estesa (U)"}, errors="ignore"))
    
    df_carb = df_carb[(df_carb.index >= start_ts) & (df_carb.index < end_ts)]
    
    # parse data(string -> float)
    toParse = ["glicemia","carb","insulina"]
    
    for col in toParse:
        df_carb[col] = (
            df_carb[col]
            .astype(str)
            .str.replace(",", ".", regex=False)   # nel caso di virgola decimale
        )
        df_carb[col] = pd.to_numeric(df_carb[col], errors="coerce")
    
    
    df_carb = df_carb[df_carb["insulina"]>0]
    df_carb = df_carb[df_carb["carb"]>0]
    
    
    #creo colonna con carboidrati reali
    
    # 0) Assicurati che l'indice sia datetime (se Time è l’indice)
    df_carb["ora"] = pd.to_datetime(df_carb.index, errors="coerce")
    
    # 1) Condizione base "carb piccoli + glicemia alta"
    base_240 = (df_carb["carb"] <= 40) & (df_carb["glicemia"] > 240)
    base_200 = (df_carb["carb"] <= 40) & (df_carb["glicemia"] > 200)
    
    # 2) Funzione helper per fascia oraria (solo hh:mm)
    def in_time_window(s_datetime, start_str, end_str):
        start_t = pd.to_datetime(start_str).time()
        end_t   = pd.to_datetime(end_str).time()
        return s_datetime.dt.time.between(start_t, end_t)
    
    # 3) Fasce orarie dove applichi la regola "base_200"
    t_1530_1930 = in_time_window(df_carb["ora"], "15:30", "19:30")
    t_2230_2359 = in_time_window(df_carb["ora"], "22:30", "23:59")
    t_0001_0500 = in_time_window(df_carb["ora"], "00:01", "05:00")
    
    time_mask = t_1530_1930 | t_2230_2359 | t_0001_0500
    
    # 4) Maschera finale "è correzione"
    df_carb["is_correction"] = base_240 | (base_200 & time_mask)
    
    # 5) Tieni solo i carboidrati "veri" (non correzioni)
    df_carb = df_carb.loc[~df_carb["is_correction"]].copy()
    
    df_carb = df_carb[["carb"]].copy()
    
    # carb per giorno
    df_carb.index = pd.to_datetime(df_carb.index, errors="coerce")
    carb_daily = df_carb["carb"].resample("D").sum().to_frame("carb_daily")
    
    # assicurati che daily_glooko abbia DatetimeIndex
    daily_glooko.index = pd.to_datetime(daily_glooko.index)
    
    # join
    daily_glooko = daily_glooko.join(carb_daily, how="left")
    
    # giorni senza carb -> 0
    daily_glooko["carb_daily"] = daily_glooko["carb_daily"].fillna(0)
    
    if True:
        try:
            df_export = daily_glooko.copy()
            df_export.index = pd.to_datetime(df_export.index)
            pq_path = cfg.GLOOKO_DATA_CLEANED
            df_export.to_parquet(pq_path, engine="pyarrow", index=True)
            print("Saved:", pq_path)
        except Exception as e:
            print("Parquet export failed:", repr(e))

if __name__ == "__main__": main()