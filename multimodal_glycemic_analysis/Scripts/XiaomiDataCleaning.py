import pandas as pd
import json
import config as cfg

def main():
    # importo il file con gli aggregati giornalieri--------------------------------------------------------------------------
        
    xiaomi_path = cfg.XIAOMI_DIR / cfg.XIAOMI_FITNESS_FILE
    df_x = pd.read_csv(xiaomi_path)
        
    #converto date e ore in modo utilizzabile-------------------------------------------------------------------------- 
        
    df_x["Time"] = (
        pd.to_datetime(df_x["Time"], unit="s", utc=True).dt.tz_convert("Europe/Rome")
            
            )
    
    #droppo righe inutili--------------------------------------------------------------------------
        
    df_x = df_x[~df_x["Key"].isin(                          #~ è negazione
        ["stress", "goal", "valid_stand", "spo2", "intensity"]
        )]
    
    df_x = df_x[df_x["Tag"] != "daily_mark"]
    
    #droppo colonne inutili--------------------------------------------------------------------------
        
    df_x.drop(columns = ["Uid","Sid","Tag","UpdateTime"], inplace = True)
        
    #elimino date fuori range obiettivo--------------------------------------------------------------------------
    
    start = pd.Timestamp("2025-09-01", tz="Europe/Rome")
    end   = pd.Timestamp("2026-02-01", tz="Europe/Rome")
        
    df_x = df_x[(df_x["Time"] >= start) & (df_x["Time"] < end)].copy()
        
    
    #passo a formato wide da long--------------------------------------------------------------------------
    
    # 1) crea la colonna Date (solo giorno, senza ora)
    df_x["Date"] = df_x["Time"].dt.date
    
    
    # 2) pivot: una colonna per ogni Key
    df_wide = (
        df_x.pivot_table(
            index="Date",
            columns="Key",
            values="Value",
            aggfunc="first"   # se per lo stesso giorno/key hai più righe, prende la prima
        )
        .reset_index()
    )
    
    df_wide = df_wide.set_index("Date") #imposto la data come indice di riga
    
    #pulisco i dati di calories--------------------------------------------------------------------------
    
    df_wide["calories"] = df_wide["calories"].apply(
        lambda x: json.loads(x)["calories"] if pd.notna(x) else pd.NA
    )  # apply applica ad ogni riga, lamda x è tipo funzione di x, x = stringa JSON → json.loads(x) = dizionario → ["calories"] = valore numerico
    
    #pulisco i dati di heart_rate--------------------------------------------------------------------------
    
    hr_notna = df_wide["heart_rate"].dropna()
    hr_parsed = hr_notna.apply(json.loads) #passo da striga a dizionario python
    hr_df = pd.json_normalize(hr_parsed) #trasformo gli elemtni del dizionario in colonne
    hr_df.index = hr_notna.index   #  reimposta l'indice Date
    
    # rimuovi colonne inutili 
    drop_cols = [
        "latest_hr.time",
        "latest_hr.bpm",
        "anaerobic_hr_zone_duration",
        "fat_burning_hr_zone_duration",
        "aerobic_hr_zone_duration",
        "warm_up_hr_zone_duration",
        "extreme_hr_zone_duration",
        "abnormal_hr_count"
    ]
    hr_df = hr_df.drop(columns=[c for c in drop_cols if c in hr_df.columns])
    
    # rinomine (solo se esistono)
    rename_map = {"avg_rhr": "avg_rest_hr"}
    hr_df = hr_df.rename(columns={k: v for k, v in rename_map.items() if k in hr_df.columns})
    
    ## hr_df = hr_df.add_prefix("hr_")  
    
    df_wide = df_wide.join(hr_df)          # join su Date (indice)
    df_wide = df_wide.drop(columns=["heart_rate"])
    
    
    #pulisco i dati di sleep--------------------------------------------------------------------------
        
    sleep_notna = df_wide["sleep"].dropna()
    sl_parsed = sleep_notna.apply(json.loads)
    
    sl_df = pd.json_normalize(sl_parsed)
    sl_df.index = sleep_notna.index  # indice = Date, join sicuro
    
    def extract_bed_wake(seg_list):
        # ritorna solo bedtime (min) e wake_up_time (max)
        if not isinstance(seg_list, list) or len(seg_list) == 0:
            return pd.Series({"sleep_seg_bedtime": pd.NA})
    
        seg = pd.DataFrame(seg_list)
        bedtime = seg["bedtime"].min() if "bedtime" in seg.columns else pd.NA
        return pd.Series({"sleep_seg_bedtime": bedtime})
    
    # estrai SOLO 2 colonne 
    seg_cols = sl_df["segment_details"].apply(extract_bed_wake)
    
    # converto subito a datetime Italia
    seg_cols["sleep_seg_bedtime_dt"] = pd.to_datetime(seg_cols["sleep_seg_bedtime"], unit="s", utc=True).dt.tz_convert("Europe/Rome")
    
    # tengo solo le due datetime
    seg_cols = seg_cols[["sleep_seg_bedtime_dt"]]
    
    # rimuovo segment_details dal sleep base e aggiungo le due nuove colonne
    sl_df = sl_df.drop(columns=["segment_details"])
    sl_df = sl_df.join(seg_cols)
    
    # qui droppi le altre colonne inutili
    sl_drop_cols = [
        'total_snore_disturb','breath_quality','day_sleep_evaluation','avg_spo2',
        'sleep_trace_duration','sleep_algorithm_version','total_body_move',
        'total_turn_over','sleep_manually_duration','total_snore'
    ]
    sl_df = sl_df.drop(columns=[c for c in sl_drop_cols if c in sl_df.columns])
    
    
    #calcolo manualmente l'orario di risveglio
    sl_df["wake_up_dt_calc"] = (
        sl_df["sleep_seg_bedtime_dt"]
        + pd.to_timedelta(sl_df["total_duration"], unit="m")
    )
    
    rename_map = {"total_long_duration": "long_duration",
                  "sleep_deep_duration": "deep_duration",
                  "sleep_nap_duration": "nap_duration",
                  "sleep_rem_duration": "rem_duration",
                  "total_duration": "tot_duration",
                  "sleep_score": "score",
                  "sleep_awake_duration": "awake_duration",
                  "sleep_light_duration": "light_duration",
                  "long_sleep_evaluation": "long_score",
                  "sleep_seg_bedtime_dt": "bedtime",
                  "wake_up_dt_calc": "wakeup_time"
                  
                  
                  }
    sl_df = sl_df.rename(columns={k: v for k, v in rename_map.items() if k in sl_df.columns})
    
    #trasformo l'orario di bedtime e di wakeup in minuti dalla mezzanotte con correzione per orari prima di mezzanotte
    
    
    sl_df["bedtime_mfm"] = (
        sl_df["bedtime"].dt.hour*60
        + sl_df["bedtime"].dt.minute
        + sl_df["bedtime"].dt.second/60
    )

    sl_df["wakeup_time_mfm"] = (
        sl_df["wakeup_time"].dt.hour*60
        + sl_df["wakeup_time"].dt.minute
        + sl_df["wakeup_time"].dt.second/60
    )

    sl_df["bedtime_mfm"] = sl_df["bedtime_mfm"].where(
        sl_df["bedtime_mfm"] <= 420,
        sl_df["bedtime_mfm"] - 24*60
    )
    
    sl_df = sl_df.drop(columns=["bedtime","wakeup_time"])
    
    # prefisso e join finale
    sl_df = sl_df.add_prefix("sl_")
    df_wide = df_wide.join(sl_df)
    df_wide = df_wide.drop(columns=["sleep"])
    
    
    #pulisco i dati sui passi--------------------------------------------------------------------------
    
    steps_notna = df_wide["steps"].dropna()
    ste_parsed = steps_notna.apply(json.loads)
    ste_df = pd.json_normalize(ste_parsed)
    ste_df.index = steps_notna.index
    
    ste_df = ste_df["steps"]
    
    
    df_wide = df_wide.drop(columns=["steps"])
    df_wide = df_wide.join(ste_df)
    
    
    #importo le date degli allenamenti---------------------------------------------------------------------
    
    
    train_path = cfg.XIAOMI_DIR / cfg.XIAOMI_TRAINING_FILE
    df_train = pd.read_csv(train_path)
    
    # tieni solo le colonne utili
    df_train = df_train[["Time"]]
    
    # converto Time in datetime con tz
    df_train["Time"] = pd.to_datetime(df_train["Time"], unit="s", utc=True).dt.tz_convert("Europe/Rome")
    
    # creo Date giornaliera 
    df_train["Date"] = df_train["Time"].dt.date
    
    # colonna booleana "training" per giorno:
    df_train["training"] = 1
    
    # aggrego per giorno 
    df_train_day = df_train.groupby("Date", as_index=True)["training"].max().to_frame()
    
    # join su Date 
    df_wide = df_wide.join(df_train_day)
    
    df_wide["training"] = df_wide["training"].fillna(0).astype(int)
    
    #export dati puliti 
    
    if True:
        try:
            df_export = df_wide.copy()
            df_export.index = pd.to_datetime(df_export.index)
            pq_path = cfg.XIAOMI_DATA_CLEANED
            df_export.to_parquet(pq_path, engine="pyarrow", index=True)
            print("Saved:", pq_path)
        except Exception as e:
            print("Parquet export failed:", repr(e))

if __name__ == "__main__": main()


