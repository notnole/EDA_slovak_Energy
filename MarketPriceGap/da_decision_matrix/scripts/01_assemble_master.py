"""
01_assemble_master.py - Assemble master hourly dataset for DA decision matrix.

Merges 6 data sources:
1. hourly_market_prices.csv  (DA + IDM + imbalance + spreads)
2. load_data.csv             (load actual + forecast + error)
3. ProductionPerType.csv     (EDA format: solar, nuclear, hydro, gas, etc.)
4. temperature_15min.csv     (GFS forecast + actual, resampled to hourly)
5. da_prices.csv             (cross-border flows CZ/PL/HU)

Output: MarketPriceGap/da_decision_matrix/data/da_master_hourly.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# --- paths ---
ROOT = Path(__file__).resolve().parents[3]  # ipesoft_eda_data/
OUT_DIR = ROOT / "MarketPriceGap" / "da_decision_matrix" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARKET_FILE = ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv"
LOAD_FILE = ROOT / "features" / "DamasLoad" / "load_data.csv"
PRODUCTION_FILE = ROOT / "RawData" / "ProductionPerType.csv"
TEMP_FILE = ROOT / "data" / "clean" / "weather" / "temperature_15min.csv"
DA_PRICES_FILE = ROOT / "features" / "DamasPrices" / "data" / "da_prices.csv"


# ============================================================
# 1. Parse ProductionPerType.csv (Ipesoft EDA multi-series)
# ============================================================
def parse_production_eda(filepath):
    """Parse the 14-series EDA export of production by type."""
    print("[*] Parsing ProductionPerType.csv (EDA format, 14 series)...")

    with open(filepath, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Header: series names at even positions (0, 2, 4, ...)
    header_fields = lines[0].strip().split(",")
    series_names = [header_fields[i].strip() for i in range(0, len(header_fields), 2)
                    if i < len(header_fields) and header_fields[i].strip()]
    num_series = len(series_names)
    print(f"[*]   Found {num_series} series: {series_names}")

    # Normalize series names for column use
    col_names = []
    for s in series_names:
        name = s.lower().replace(".", "_").replace(" ", "_")
        col_names.append(name)

    rows = []
    parse_errors = 0

    for line_num, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line:
            continue

        fields = line.split(",")
        pos = 0
        row_dt = None
        row_values = {}

        try:
            for i in range(num_series):
                # datetime field
                if pos >= len(fields):
                    break
                dt_str = fields[pos].strip()
                pos += 1

                # ms field (always '000')
                if pos >= len(fields):
                    break
                pos += 1  # skip ms

                # value field
                if pos >= len(fields):
                    break
                val_str = fields[pos].strip()
                pos += 1

                if val_str == "(invalid)":
                    row_values[col_names[i]] = np.nan
                else:
                    int_part = val_str
                    # Check if next field is decimal part
                    if pos < len(fields):
                        next_f = fields[pos].strip()
                        if next_f and "/" not in next_f and next_f != "(invalid)":
                            # It's the decimal part
                            value = float(int_part + "." + next_f)
                            pos += 1
                        else:
                            value = float(int_part)
                    else:
                        value = float(int_part)
                    row_values[col_names[i]] = value

                # Use first series datetime as the row timestamp
                if i == 0:
                    row_dt = pd.to_datetime(dt_str, format="%d/%m/%Y %H:%M:%S")

            if row_dt is not None:
                row_values["datetime"] = row_dt
                rows.append(row_values)
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 5:
                print(f"[-]   Parse error line {line_num}: {e}")

    df = pd.DataFrame(rows)
    df = df.set_index("datetime").sort_index()

    if parse_errors > 0:
        print(f"[-]   Total parse errors: {parse_errors}")

    print(f"[+]   Parsed {len(df)} rows, {df.shape[1]} columns")
    print(f"[*]   Period: {df.index.min()} to {df.index.max()}")
    print(f"[*]   Columns: {list(df.columns)}")

    # Show valid count per column
    valid = df.notna().sum()
    for col in df.columns:
        pct = 100 * valid[col] / len(df)
        print(f"      {col}: {valid[col]} valid ({pct:.1f}%)")

    return df


# ============================================================
# 2. Load pre-cleaned data sources
# ============================================================
def load_market_prices():
    """Load hourly_market_prices.csv (DA + IDM + spreads)."""
    print("[*] Loading hourly_market_prices.csv...")
    df = pd.read_csv(MARKET_FILE, parse_dates=["timestamp_hour"])
    df = df.rename(columns={"timestamp_hour": "datetime"})
    df = df.set_index("datetime").sort_index()
    # Drop redundant calendar columns (will recreate if needed)
    drop_cols = ["date", "hour", "dayofweek", "month", "is_weekend"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    print(f"[+]   {len(df)} rows, {df.index.min()} to {df.index.max()}")
    return df


def load_load_data():
    """Load DAMAS load forecast data."""
    print("[*] Loading load_data.csv...")
    df = pd.read_csv(LOAD_FILE, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    # Keep only essential columns
    keep = ["actual_load_mw", "forecast_load_mw", "forecast_error_mw", "forecast_error_pct"]
    df = df[[c for c in keep if c in df.columns]]
    print(f"[+]   {len(df)} rows, {df.index.min()} to {df.index.max()}")
    return df


def load_temperature():
    """Load temperature data, resample 15min -> hourly."""
    print("[*] Loading temperature_15min.csv and resampling to hourly...")
    df = pd.read_csv(TEMP_FILE, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    # Resample 15min to hourly (mean)
    df_h = df.resample("h").mean()
    print(f"[+]   {len(df_h)} rows, {df_h.index.min()} to {df_h.index.max()}")
    return df_h


def load_da_features():
    """Load da_prices.csv for cross-border flow columns."""
    print("[*] Loading da_prices.csv (cross-border flows)...")
    df = pd.read_csv(DA_PRICES_FILE, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    # Only take cross-border flows (not already in market file)
    keep = ["net_flow_cz", "net_flow_pl", "net_flow_hu"]
    df = df[[c for c in keep if c in df.columns]]
    print(f"[+]   {len(df)} rows, {df.index.min()} to {df.index.max()}")
    return df


# ============================================================
# 3. Merge everything
# ============================================================
def main():
    print("=" * 70)
    print("DA Decision Matrix - Step 1: Assemble Master Dataset")
    print("=" * 70)

    # Load all sources
    df_market = load_market_prices()
    df_load = load_load_data()
    df_production = parse_production_eda(PRODUCTION_FILE)
    df_temp = load_temperature()
    df_flows = load_da_features()

    # Start with market prices as base
    print("\n[*] Merging datasets...")
    master = df_market.copy()

    # Prefix load columns to avoid conflicts
    df_load = df_load.rename(columns={
        "actual_load_mw": "load_actual_mw",
        "forecast_load_mw": "load_forecast_mw",
        "forecast_error_mw": "load_error_mw",
        "forecast_error_pct": "load_error_pct",
    })
    master = master.join(df_load, how="left")

    # Join production data
    master = master.join(df_production, how="left", rsuffix="_prod")

    # Join temperature
    df_temp = df_temp.rename(columns={
        "temp_forecast_gfs": "temp_forecast",
        "temp_actual": "temp_actual",
    })
    master = master.join(df_temp, how="left")

    # Join cross-border flows
    master = master.join(df_flows, how="left")

    # Add calendar features
    master["hour"] = master.index.hour
    master["date"] = master.index.date
    master["day_of_week"] = master.index.dayofweek
    master["month"] = master.index.month
    master["is_weekend"] = master.index.dayofweek.isin([5, 6]).astype(int)

    print(f"\n[+] Master dataset: {master.shape[0]} rows x {master.shape[1]} columns")
    print(f"[*] Period: {master.index.min()} to {master.index.max()}")

    # Show data availability summary
    print("\n--- Data Availability Summary ---")
    key_cols = [
        "da_price", "idm_vwap", "spread_da_idm",
        "load_actual_mw", "load_forecast_mw",
        "sk_a_solar", "sk_f_solar", "sk_a_nuclear",
        "sk_a_hydroreservoir", "sk_a_natgas",
        "temp_forecast", "temp_actual",
        "net_flow_cz", "net_flow_pl", "net_flow_hu",
    ]
    for col in key_cols:
        if col in master.columns:
            n_valid = master[col].notna().sum()
            first_valid = master[col].first_valid_index()
            last_valid = master.loc[master[col].notna()].index[-1] if n_valid > 0 else None
            print(f"  {col:30s}: {n_valid:6d} valid  ({first_valid} to {last_valid})")
        else:
            print(f"  {col:30s}: MISSING")

    # Identify overlap period (all key signals present)
    overlap_mask = (
        master["da_price"].notna()
        & master["idm_vwap"].notna()
        & master["load_forecast_mw"].notna()
        & master["sk_a_solar"].notna()
        & master["temp_forecast"].notna()
    )
    n_overlap = overlap_mask.sum()
    if n_overlap > 0:
        overlap_start = master.index[overlap_mask][0]
        overlap_end = master.index[overlap_mask][-1]
        print(f"\n[+] Full overlap period: {overlap_start} to {overlap_end} ({n_overlap} hours)")
    else:
        print("\n[-] WARNING: No period with all signals present!")

    # Save
    out_path = OUT_DIR / "da_master_hourly.csv"
    master.to_csv(out_path)
    print(f"\n[+] Saved: {out_path}")
    print(f"    Shape: {master.shape}")

    return master


if __name__ == "__main__":
    main()
