from __future__ import annotations

from pathlib import Path
import runpy
import pandas as pd
import config as cfg


# ============================
# Helpers
# ============================

def run_script(path: Path) -> None:
    runpy.run_path(str(path), run_name="__main__")


def load_df(parquet_path: Path) -> pd.DataFrame:

    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"[WARN] Parquet read failed ({parquet_path.name}): {repr(e)}")



def normalize_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uniforma l'indice a DatetimeIndex giornaliero (00:00), così il join è stabile.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce").normalize()
    out = out[~out.index.isna()].sort_index()
    return out


# ============================
# Main
# ============================

def main():
    scripts_dir = (Path(__file__).resolve().parents[0])

    xiaomi_script = scripts_dir / "XiaomiDataCleaning.py"
    glooko_script = scripts_dir / "GlookoDataCleaning.py"

    # 1) Esegui i cleaning scripts (rigenerano gli export)
    run_script(xiaomi_script)
    run_script(glooko_script)

    # 2) Carica gli output
    glooko_parquet = Path(cfg.GLOOKO_DATA_CLEANED)
    xiaomi_parquet = Path(cfg.XIAOMI_DATA_CLEANED)

    glooko = load_df(glooko_parquet)
    xiaomi = load_df(xiaomi_parquet)

    # 3) Allinea indice giornaliero
    glooko = normalize_daily_index(glooko)
    xiaomi = normalize_daily_index(xiaomi)

    # 4) Join (outer per non perdere giorni)
    df = glooko.join(xiaomi, how="outer", lsuffix="_glooko", rsuffix="_xiaomi").sort_index()
    
    # 5) Export merged

    try:
        df.to_parquet(cfg.DATA_CLEANED)
        print("Saved:", cfg.DATA_CLEANED)
    except Exception as e:
        print("Parquet export failed:", repr(e))

if __name__ == "__main__":
    main()


