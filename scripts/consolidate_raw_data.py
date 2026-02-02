"""
Consolidate raw data files into clean, single CSV files.

This script:
1. Consolidates IDM market data from multiple folders into single files
2. Consolidates DA market data from multiple year files
3. Renames and organizes files with English names
"""

import pandas as pd
import os
from pathlib import Path
import shutil

# Paths
RAW_DATA = Path("RawData")
CLEAN_DATA = Path("data/clean")

def consolidate_idm_data():
    """Consolidate IDM market data from multiple folders."""
    print("[*] Consolidating IDM market data...")

    idm_path = RAW_DATA / "IDM_MarketData"

    # Collect all 15-min files
    dfs_15min = []
    dfs_60min = []

    for folder in sorted(idm_path.iterdir()):
        if folder.is_dir() and folder.name.startswith("IDM_total_results"):
            f15 = folder / "15 min.csv"
            f60 = folder / "60 min.csv"

            if f15.exists():
                df = pd.read_csv(f15, sep=";", encoding="utf-8")
                dfs_15min.append(df)
                print(f"    [+] Read {f15.name} from {folder.name}")

            if f60.exists():
                df = pd.read_csv(f60, sep=";", encoding="utf-8")
                dfs_60min.append(df)

    # Concatenate and save
    if dfs_15min:
        df_15 = pd.concat(dfs_15min, ignore_index=True)
        # Sort by date and period
        df_15["Delivery day"] = pd.to_datetime(df_15["Delivery day"], format="%d.%m.%Y")
        df_15 = df_15.sort_values(["Delivery day", "Period number"]).reset_index(drop=True)
        df_15["Delivery day"] = df_15["Delivery day"].dt.strftime("%Y-%m-%d")

        out_path = CLEAN_DATA / "market" / "intraday" / "idm_15min.csv"
        df_15.to_csv(out_path, index=False, sep=";")
        print(f"    [+] Saved {out_path}: {len(df_15)} rows")

    if dfs_60min:
        df_60 = pd.concat(dfs_60min, ignore_index=True)
        df_60["Delivery day"] = pd.to_datetime(df_60["Delivery day"], format="%d.%m.%Y")
        df_60 = df_60.sort_values(["Delivery day", "Period number"]).reset_index(drop=True)
        df_60["Delivery day"] = df_60["Delivery day"].dt.strftime("%Y-%m-%d")

        out_path = CLEAN_DATA / "market" / "intraday" / "idm_60min.csv"
        df_60.to_csv(out_path, index=False, sep=";")
        print(f"    [+] Saved {out_path}: {len(df_60)} rows")

def consolidate_da_data():
    """Consolidate DA market data from multiple year files."""
    print("[*] Consolidating DA market data...")

    da_path = RAW_DATA / "Damas"
    dfs = []

    for f in sorted(da_path.glob("Celkove_vysledky_DT_*.csv")):
        # Read with correct encoding
        df = pd.read_csv(f, sep=";", encoding="utf-8")
        dfs.append(df)
        print(f"    [+] Read {f.name}: {len(df)} rows")

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)

        # Rename columns to English
        column_map = {
            "Obchodny den": "trading_day",
            "Cislo periody": "period_number",
            "Perioda": "period",
            "Perioda (min)": "period_minutes",
            "Cena SK (EUR/MWh)": "price_sk_eur_mwh",
            "Dopyt uspesny (MW)": "demand_matched_mw",
            "Ponuka uspesna (MW)": "supply_matched_mw",
            "Tok CZ -> SK (MW)": "flow_cz_to_sk_mw",
            "Tok SK -> CZ (MW)": "flow_sk_to_cz_mw",
            "Tok PL -> SK (MW)": "flow_pl_to_sk_mw",
            "Tok SK -> PL (MW)": "flow_sk_to_pl_mw",
            "Tok HU -> SK (MW)": "flow_hu_to_sk_mw",
            "Tok SK -> HU (MW)": "flow_sk_to_hu_mw",
            "Stav zverejnenia": "publication_status"
        }

        # Handle special characters in column names
        df_all.columns = [c.replace("\ufeff", "") for c in df_all.columns]

        # Try to rename (some columns may have slightly different names)
        for old, new in column_map.items():
            for col in df_all.columns:
                if old.lower() in col.lower():
                    df_all = df_all.rename(columns={col: new})
                    break

        # Parse and sort by date
        # Handle Slovak date format (D.M.YYYY)
        try:
            df_all["trading_day"] = pd.to_datetime(df_all.iloc[:, 0], format="%d.%m.%Y")
            df_all = df_all.sort_values(["trading_day", df_all.columns[1]]).reset_index(drop=True)
            df_all["trading_day"] = df_all["trading_day"].dt.strftime("%Y-%m-%d")
        except:
            print("    [-] Could not parse dates, keeping original format")

        out_path = CLEAN_DATA / "market" / "day_ahead" / "da_auction_results.csv"
        df_all.to_csv(out_path, index=False, sep=";")
        print(f"    [+] Saved {out_path}: {len(df_all)} rows")

def copy_and_rename_scada():
    """Copy SCADA files with cleaner English names."""
    print("[*] Creating symlinks/copies for SCADA files with clean names...")

    scada_map = {
        "3MIN_REG.csv": "regulation_1min.csv",
        "3MIN_Load.csv": "system_load_3min.csv",
        "3MIN_Prod.csv": "system_production_1min.csv",
        "3MIN_Export.csv": "real_balance_1min.csv",
        "3MIN_ACK_REAL_BALNCE.csv": "ack_real_balance_1min.csv",
        "ProductionPerType.csv": "production_by_type_hourly.csv",
    }

    weather_map = {
        "Claudes.csv": "cloud_cover_15min.csv",
        "Tampreture.csv": "temperature_15min.csv",
    }

    # Create name mapping file instead of copying large files
    mappings = []

    for old, new in scada_map.items():
        src = RAW_DATA / old
        if src.exists():
            mappings.append({
                "original_path": str(src),
                "clean_name": new,
                "category": "scada"
            })
            print(f"    [+] Mapped: {old} -> {new}")

    for old, new in weather_map.items():
        src = RAW_DATA / old
        if src.exists():
            mappings.append({
                "original_path": str(src),
                "clean_name": new,
                "category": "weather"
            })
            print(f"    [+] Mapped: {old} -> {new}")

    # Save mapping file
    df_map = pd.DataFrame(mappings)
    out_path = CLEAN_DATA / "file_mapping.csv"
    df_map.to_csv(out_path, index=False)
    print(f"    [+] Saved mapping to {out_path}")

def main():
    print("=" * 60)
    print("Data Consolidation Script")
    print("=" * 60)

    # Create output directories
    for subdir in ["market/day_ahead", "market/intraday", "scada", "weather"]:
        (CLEAN_DATA / subdir).mkdir(parents=True, exist_ok=True)

    consolidate_idm_data()
    consolidate_da_data()
    copy_and_rename_scada()

    print("\n" + "=" * 60)
    print("[+] Consolidation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
