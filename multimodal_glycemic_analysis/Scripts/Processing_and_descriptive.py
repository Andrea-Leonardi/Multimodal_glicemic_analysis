import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg
import seaborn as sns
from pathlib import Path
import runpy





def run_script(path: Path) -> None:
    """
    Esegue uno script .py come se fosse lanciato da terminale.
    Questo fa scattare il blocco:
        if __name__ == "__main__": main()
    """
    runpy.run_path(str(path), run_name="__main__")

if True:
    scripts_dir = (Path(__file__).resolve().parents[0])
    run_script(scripts_dir / "MergeData.py")

df = pd.read_parquet(cfg.DATA_CLEANED)

#=============================
#BOXPLOT
#=============================

def boxplot(df):
    step = 6
    for start in range(0, df.shape[1], step):
        sub = df.iloc[:, start:start+step]      # fino a 6 colonne
        sub = sub.select_dtypes("number") 
        
        fig, axes = plt.subplots(1, sub.shape[1], figsize=(3*sub.shape[1], 4), sharey=False)
        if sub.shape[1] == 1:
            axes = [axes]
    
        for ax, col in zip(axes, sub.columns):
            ax.boxplot(sub[col].dropna().values)
            ax.set_title(col, fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
    
        plt.tight_layout()
        plt.show()

#=============================
#CORRPLOT
#=============================
    
def corrplot(df, r):

    corr = df.select_dtypes("number").corr(method="spearman")
    annot = corr.round(2).astype(str)
    annot[np.abs(corr) < r] = ""   # non scrive i numeri piccoli
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=annot, fmt="", annot_kws={"size": 6},
                vmin=-1, vmax=1, center=0)
    plt.tight_layout()
    plt.show()
    
#=============================
#FURTHER MANIPULATIONS ON MERGED DATASET
#=============================

def lag():
    
    #creo una colonna con il numero dei giorni sequenziale
    df["day"] = range(df.shape[0])

    #laggo i dati    
        
    k = 2
    toLag = ["training",
             "bg_mean",
             "bg_sd",
             "bg_max",
             "bg_auc_above_limit_rate",
             "cartridge_loaded",
             "carb_daily",
             "sl_tot_duration",
             "sl_score",
             "sl_bedtime_mfm",
             "steps"]
    
    df_lag = df.copy()
    
    for i in range(k):
        for col in toLag:
            df_lag[f"{col}_lag{i+1}"] = df_lag[col].shift(i+1)
        
    
    corrplot(df_lag, 0.2)
    
    #export lagged
    
    if True:
        try:
            df_export = df_lag.copy()
            df_export.index = pd.to_datetime(df_export.index)
            pq_path = cfg.DATA_PROCESSED
            df_export.to_parquet(pq_path, engine="pyarrow", index=True)
            print("Saved:", pq_path)
            
            df_export.to_csv(cfg.PROCESSED_SUBDIR/"dataproc.csv")
        except Exception as e:
            print("Parquet export failed:", repr(e))

if __name__ == "__lag__":
    lag()
    

