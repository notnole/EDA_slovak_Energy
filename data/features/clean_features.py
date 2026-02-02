"""
Clean 3-minute feature data by removing duplicate scrapes.

The legacy system scrapes every ~20 seconds, but data only updates every 3 minutes.
We keep only the first occurrence when a new value appears.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\RawData")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    'regulation': '3MIN_REG.csv',
    'load': '3MIN_Load.csv',
    'production': '3MIN_Prod.csv',
    'export_import': '3MIN_ACK_REAL_BALNCE.csv'
}

def parse_datetime(dt_str):
    """Parse datetime string."""
    try:
        return pd.to_datetime(dt_str, format='%d/%m/%Y %H:%M:%S')
    except:
        try:
            return pd.to_datetime(dt_str, format='%m/%d/%Y %H:%M:%S')
        except:
            return pd.NaT

def load_regulation_data(filepath):
    """Load regulation data with European decimal format."""
    print(f"  Loading {filepath.name}...")

    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            parts = line.strip().split(',')
            if len(parts) < 4:
                continue

            dt_str = parts[0]
            if '(invalid)' in line:
                continue

            try:
                int_part = parts[2].strip()
                dec_part = parts[3].strip() if len(parts) > 3 else '0'

                if int_part == '' or int_part == '(invalid)':
                    continue

                value = float(f"{int_part}.{dec_part}")
                dt = parse_datetime(dt_str)
                if pd.notna(dt):
                    rows.append({'datetime': dt, 'regulation_mw': value})
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)
    print(f"    Raw rows: {len(df):,}")
    return df

def load_standard_data(filepath, value_name):
    """Load standard format data."""
    print(f"  Loading {filepath.name}...")

    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            parts = line.strip().split(',')
            if len(parts) < 3:
                continue

            dt_str = parts[0]

            try:
                value = float(parts[2].strip())
                dt = parse_datetime(dt_str)
                if pd.notna(dt):
                    rows.append({'datetime': dt, value_name: value})
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)
    print(f"    Raw rows: {len(df):,}")
    return df

def remove_duplicates(df, value_col):
    """Keep only first occurrence when value changes."""
    df = df.sort_values('datetime').reset_index(drop=True)

    # Mark rows where value changed from previous
    df['changed'] = df[value_col] != df[value_col].shift(1)
    df.loc[0, 'changed'] = True  # First row always kept

    # Keep only changed rows
    df_clean = df[df['changed']].drop(columns=['changed']).reset_index(drop=True)

    print(f"    After dedup: {len(df_clean):,} rows ({len(df_clean)/len(df)*100:.1f}% kept)")
    return df_clean

def remove_outliers(df, value_col, threshold):
    """Remove rows where |value| > threshold."""
    before = len(df)
    df_clean = df[df[value_col].abs() <= threshold].reset_index(drop=True)
    removed = before - len(df_clean)
    if removed > 0:
        print(f"    Removed {removed} outliers (|{value_col}| > {threshold})")
    return df_clean

def main():
    print("=" * 60)
    print("CLEANING FEATURE DATA (removing duplicate scrapes)")
    print("=" * 60)

    # 1. Regulation
    print("\n[1/4] REGULATION")
    reg_df = load_regulation_data(RAW_DIR / FILES['regulation'])
    reg_df = remove_duplicates(reg_df, 'regulation_mw')
    reg_df = remove_outliers(reg_df, 'regulation_mw', 300)  # Same threshold as imbalance label
    reg_df.to_csv(OUTPUT_DIR / 'regulation_3min.csv', index=False)

    # 2. Load
    print("\n[2/4] LOAD")
    load_df = load_standard_data(RAW_DIR / FILES['load'], 'load_mw')
    load_df = remove_duplicates(load_df, 'load_mw')
    load_df.to_csv(OUTPUT_DIR / 'load_3min.csv', index=False)

    # 3. Production
    print("\n[3/4] PRODUCTION")
    prod_df = load_standard_data(RAW_DIR / FILES['production'], 'production_mw')
    prod_df = remove_duplicates(prod_df, 'production_mw')
    prod_df.to_csv(OUTPUT_DIR / 'production_3min.csv', index=False)

    # 4. Export/Import
    print("\n[4/4] EXPORT/IMPORT")
    exp_df = load_standard_data(RAW_DIR / FILES['export_import'], 'export_import_mw')
    exp_df = remove_duplicates(exp_df, 'export_import_mw')
    exp_df.to_csv(OUTPUT_DIR / 'export_import_3min.csv', index=False)

    # Summary
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob('*.csv')):
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df):,} rows")

if __name__ == '__main__':
    main()
