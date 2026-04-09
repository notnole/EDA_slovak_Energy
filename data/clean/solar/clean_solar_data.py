# -*- coding: utf-8 -*-
"""
Clean Solar Data Pipeline
=========================
Combines DAMAS solar forecast (DA) and actual production into a single
clean hourly CSV covering 2024-2026.

Sources:
  Forecast (DA):  RawData/Damas/Vyroba FVE a Veternych - Predikcia *.xlsx
                  Column: "OZE - Slnecna DA [MW]"
  Actual:         RawData/Damas/Vyroba ES SR - Skutocnost *.xlsx
                  Column: "Solar [MW]"

Output:
  data/clean/solar/solar_hourly.csv
  Columns: datetime, solar_actual_mw, solar_forecast_da_mw, solar_surprise_mw
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Fix Windows console encoding for Slovak filenames
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent.parent.parent.parent

# --- Source files (ordered: 2024, 2025, Jan 2026) ---
# Auto-discover all matching files (covers all years)
DAMAS_DIR = BASE_DIR / "RawData" / "Damas"
FORECAST_FILES = sorted(DAMAS_DIR.glob("V\u00fdroba FVE a Vetern\u00fdch - Predikcia*.xlsx"))
ACTUAL_FILES = sorted(DAMAS_DIR.glob("V\u00fdroba ES SR - Skuto\u010dnos\u0165*.xlsx"))

OUTPUT_PATH = Path(__file__).parent / "solar_hourly.csv"


def load_forecast_files():
    """Load and concatenate solar DA forecast from FVE Predikcia files."""
    print("[*] Loading solar DA forecast files...")
    frames = []

    for fpath in FORECAST_FILES:
        df = pd.read_excel(fpath)
        print(f"    {fpath.name}: {len(df)} rows")

        # Find the solar DA column (contains "Slnecna DA" or similar)
        solar_col = [c for c in df.columns if "DA" in str(c) and "Sln" in str(c)]
        if not solar_col:
            solar_col = [c for c in df.columns if "DA" in str(c)]
        if not solar_col:
            print(f"    [!] No solar DA column found in {fpath.name}, skipping")
            continue
        solar_col = solar_col[0]

        # Date column is first unnamed column (DD.MM.YYYY), hour is second
        date_col = df.columns[0]
        hour_col = df.columns[1]

        # Forward-fill date
        df[date_col] = df[date_col].ffill()

        # Parse datetime
        df["datetime"] = pd.to_datetime(
            df[date_col].astype(str), format="%d.%m.%Y", errors="coerce"
        )
        # Hour is 1-24, convert to 0-23
        df["hour"] = df[hour_col].astype(int) - 1
        df["datetime"] = df["datetime"] + pd.to_timedelta(df["hour"], unit="h")

        df = df[["datetime", solar_col]].copy()
        df.columns = ["datetime", "solar_forecast_da_mw"]
        df = df.dropna(subset=["datetime"])

        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    print(f"[+] Forecast: {len(result)} records, {result['datetime'].min()} to {result['datetime'].max()}")
    return result


def load_actual_files():
    """Load and concatenate actual solar production from Skutocnost files."""
    print("\n[*] Loading actual solar production files...")
    frames = []

    for fpath in ACTUAL_FILES:
        df = pd.read_excel(fpath)
        print(f"    {fpath.name}: {len(df)} rows, cols: {len(df.columns)}")

        # Find Solar column
        solar_col = [c for c in df.columns if "Solar" in str(c)]
        if not solar_col:
            print(f"    [!] No Solar column in {fpath.name}, skipping")
            continue
        solar_col = solar_col[0]

        date_col = df.columns[0]
        hour_col = df.columns[1]

        df[date_col] = df[date_col].ffill()

        df["datetime"] = pd.to_datetime(
            df[date_col].astype(str), format="%d.%m.%Y", errors="coerce"
        )
        df["hour"] = df[hour_col].astype(int) - 1
        df["datetime"] = df["datetime"] + pd.to_timedelta(df["hour"], unit="h")

        df = df[["datetime", solar_col]].copy()
        df.columns = ["datetime", "solar_actual_mw"]
        df = df.dropna(subset=["datetime"])

        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    print(f"[+] Actual: {len(result)} records, {result['datetime'].min()} to {result['datetime'].max()}")
    return result


def main():
    print("=" * 60)
    print("Clean Solar Data Pipeline (DAMAS)")
    print("=" * 60)

    forecast_df = load_forecast_files()
    actual_df = load_actual_files()

    # Merge on datetime
    print("\n[*] Merging forecast and actual...")
    df = actual_df.merge(forecast_df, on="datetime", how="outer")
    df = df.sort_values("datetime").reset_index(drop=True)

    # Compute solar surprise
    df["solar_surprise_mw"] = df["solar_actual_mw"] - df["solar_forecast_da_mw"]

    print(f"[+] Merged: {len(df)} records")
    print(f"    Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"    Both actual+forecast: {df.dropna(subset=['solar_actual_mw', 'solar_forecast_da_mw']).shape[0]}")

    # Summer stats
    df_summer = df[df["datetime"].dt.month.isin([6, 7, 8])]
    df_summer_valid = df_summer.dropna(subset=["solar_actual_mw", "solar_forecast_da_mw"])
    print(f"    Summer (Jun-Aug) with both: {len(df_summer_valid)}")
    for year in sorted(df_summer_valid["datetime"].dt.year.unique()):
        yr = df_summer_valid[df_summer_valid["datetime"].dt.year == year]
        months = sorted(yr["datetime"].dt.month.unique())
        print(f"      {year}: {len(yr)} records, months: {months}")

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[+] Saved: {OUTPUT_PATH}")
    print(f"    Columns: {list(df.columns)}")

    print("\n" + "=" * 60)
    print("[+] Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
