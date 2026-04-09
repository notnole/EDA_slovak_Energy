"""
Clean 3-minute feature data by removing duplicate scrapes.

The legacy system scrapes every ~20 seconds, but data only updates every 3 minutes.
We keep only the first occurrence when a new value appears.

Handles two raw file formats:
  Old (2024-2025): 24h time, European comma decimals: "01/01/2024 00:00:20,000,-12,788,"
  New (2026):      12h AM/PM time, dot decimals:      "1/1/2026 12:00:24 AM.000,41.797,"
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RAW_DIR = REPO_ROOT / "RawData"
OUTPUT_DIR = SCRIPT_DIR  # data/features/

# Old files (2024-2025) + New files (2026)
FILE_GROUPS = {
    'regulation': {
        'old': '3MIN_REG.csv',
        'new': 'Reg3Min26.csv',
        'col': 'regulation_mw',
        'parser': 'regulation',  # special European decimal parser
    },
    'load': {
        'old': '3MIN_Load.csv',
        'new': 'Load3Min26.csv',
        'col': 'load_mw',
        'parser': 'standard',
    },
    'production': {
        'old': '3MIN_Prod.csv',
        'new': 'Prod3Min26.csv',
        'col': 'production_mw',
        'parser': 'standard',
    },
    'export_import': {
        'old': '3MIN_ACK_REAL_BALNCE.csv',
        'new': 'ackRealBalance3Min26.csv',
        'col': 'export_import_mw',
        'parser': 'standard',
    },
}


def parse_old_datetime(dt_str):
    """Parse old format: '01/01/2024 00:00:20,000' (24h, comma before millis)."""
    # Remove the ,000 millisecond part
    dt_str = dt_str.strip()
    for fmt in ['%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
        try:
            return pd.to_datetime(dt_str, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def parse_new_datetime(dt_str):
    """Parse new format: '1/1/2026 12:00:24 AM.000' (12h AM/PM, dot before millis)."""
    dt_str = dt_str.strip()
    # Remove .000 millisecond suffix
    if '.000' in dt_str:
        dt_str = dt_str.replace('.000', '')
    for fmt in ['%m/%d/%Y %I:%M:%S %p', '%d/%m/%Y %I:%M:%S %p']:
        try:
            return pd.to_datetime(dt_str, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def detect_format(filepath):
    """Detect whether file uses old or new format by checking the second line."""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        f.readline()  # skip header
        line = f.readline().strip()
    if 'AM' in line or 'PM' in line:
        return 'new'
    return 'old'


def load_regulation_old(filepath):
    """Load old-format regulation: European comma decimal (e.g. -12,788)."""
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().rstrip(',').split(',')
            if len(parts) < 4:
                continue
            dt_str = parts[0]
            if '(invalid)' in line:
                continue
            try:
                # Format: datetime,millis,integer_part,decimal_part
                int_part = parts[2].strip()
                dec_part = parts[3].strip() if len(parts) > 3 else '0'
                if int_part == '' or int_part == '(invalid)':
                    continue
                value = float(f"{int_part}.{dec_part}")
                dt = parse_old_datetime(dt_str)
                if pd.notna(dt):
                    rows.append({'datetime': dt, 'regulation_mw': value})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


def load_regulation_new(filepath):
    """Load new-format regulation: dot decimal (e.g. 41.797)."""
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().rstrip(',').split(',')
            if len(parts) < 2:
                continue
            # Format: datetime,value,  (sometimes trailing comma)
            dt_str = parts[0].strip()
            if '(invalid)' in line or dt_str == '':
                continue
            try:
                value = float(parts[1].strip())
                dt = parse_new_datetime(dt_str)
                if pd.notna(dt):
                    rows.append({'datetime': dt, 'regulation_mw': value})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


def load_standard_old(filepath, value_name):
    """Load old-format standard data (load, production, export)."""
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().rstrip(',').split(',')
            if len(parts) < 3:
                continue
            dt_str = parts[0]
            try:
                value = float(parts[2].strip())
                dt = parse_old_datetime(dt_str)
                if pd.notna(dt):
                    rows.append({'datetime': dt, value_name: value})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


def load_standard_new(filepath, value_name):
    """Load new-format standard data."""
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().rstrip(',').split(',')
            if len(parts) < 2:
                continue
            dt_str = parts[0].strip()
            if dt_str == '':
                continue
            try:
                value = float(parts[1].strip())
                dt = parse_new_datetime(dt_str)
                if pd.notna(dt):
                    rows.append({'datetime': dt, value_name: value})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


def remove_duplicates(df, value_col):
    """Keep only first occurrence when value changes (dedup ~20s scrapes)."""
    df = df.sort_values('datetime').reset_index(drop=True)
    df['changed'] = df[value_col] != df[value_col].shift(1)
    df.loc[0, 'changed'] = True
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


def load_and_merge(name, config):
    """Load old + new files for a given signal, merge, dedup."""
    col = config['col']
    dfs = []

    for variant in ['old', 'new']:
        filename = config.get(variant)
        if filename is None:
            continue
        filepath = RAW_DIR / filename
        if not filepath.exists():
            print(f"    [{variant}] {filename} not found, skipping")
            continue

        fmt = detect_format(filepath)
        print(f"    [{variant}] {filename} (format: {fmt})")

        if config['parser'] == 'regulation':
            if fmt == 'old':
                df = load_regulation_old(filepath)
            else:
                df = load_regulation_new(filepath)
        else:
            if fmt == 'old':
                df = load_standard_old(filepath, col)
            else:
                df = load_standard_new(filepath, col)

        print(f"    [{variant}] Loaded {len(df):,} rows: {df['datetime'].min()} to {df['datetime'].max()}")
        dfs.append(df)

    if not dfs:
        print(f"    [!] No data loaded for {name}")
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values('datetime').drop_duplicates(subset='datetime', keep='last').reset_index(drop=True)
    print(f"    Merged: {len(merged):,} rows, {merged['datetime'].min()} to {merged['datetime'].max()}")
    return merged


def main():
    print("=" * 60)
    print("CLEANING FEATURE DATA (old + new formats)")
    print(f"Raw: {RAW_DIR}")
    print(f"Out: {OUTPUT_DIR}")
    print("=" * 60)

    for i, (name, config) in enumerate(FILE_GROUPS.items(), 1):
        col = config['col']
        print(f"\n[{i}/{len(FILE_GROUPS)}] {name.upper()}")

        df = load_and_merge(name, config)
        if df.empty:
            continue

        df = remove_duplicates(df, col)

        if name == 'regulation':
            df = remove_outliers(df, col, 300)

        outpath = OUTPUT_DIR / f'{name}_3min.csv'
        df.to_csv(outpath, index=False)
        print(f"    Saved: {outpath.name}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for f in sorted(OUTPUT_DIR.glob('*_3min.csv')):
        df = pd.read_csv(f, parse_dates=['datetime'])
        print(f"  {f.name}: {len(df):,} rows, {df['datetime'].min()} to {df['datetime'].max()}")


if __name__ == '__main__':
    main()
