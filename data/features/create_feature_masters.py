"""
Create master feature files with different coverage options.

Option 1: Core features (regulation + load) - Full 2024-2026 coverage
Option 2: All features - Limited to Oct 2025-Jan 2026 (only 109 days)

This allows choosing between more data vs more features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")

def main():
    print("=" * 70)
    print("CREATING MASTER FEATURE FILES")
    print("=" * 70)

    # Load all cleaned feature files
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
    load_df = pd.read_csv(FEATURES_DIR / 'load_3min.csv', parse_dates=['datetime'])
    prod_df = pd.read_csv(FEATURES_DIR / 'production_3min.csv', parse_dates=['datetime'])
    exp_df = pd.read_csv(FEATURES_DIR / 'export_import_3min.csv', parse_dates=['datetime'])

    # Load label data for alignment
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])

    print("\nData coverage:")
    print(f"  Regulation:    {reg_df['datetime'].min()} to {reg_df['datetime'].max()} ({len(reg_df):,} rows)")
    print(f"  Load:          {load_df['datetime'].min()} to {load_df['datetime'].max()} ({len(load_df):,} rows)")
    print(f"  Production:    {prod_df['datetime'].min()} to {prod_df['datetime'].max()} ({len(prod_df):,} rows)")
    print(f"  Export/Import: {exp_df['datetime'].min()} to {exp_df['datetime'].max()} ({len(exp_df):,} rows)")
    print(f"  Labels:        {label_df['datetime'].min()} to {label_df['datetime'].max()} ({len(label_df):,} rows)")

    # OPTION 1: Core features (regulation + load) - Maximum data coverage
    print("\n" + "-" * 70)
    print("OPTION 1: Core Features (regulation + load)")
    print("-" * 70)

    core_start = max(reg_df['datetime'].min(), load_df['datetime'].min())
    core_end = min(reg_df['datetime'].max(), load_df['datetime'].max())

    reg_core = reg_df[(reg_df['datetime'] >= core_start) & (reg_df['datetime'] <= core_end)]
    load_core = load_df[(load_df['datetime'] >= core_start) & (load_df['datetime'] <= core_end)]

    master_core = reg_core.merge(load_core, on='datetime', how='outer')
    master_core = master_core.sort_values('datetime').reset_index(drop=True)

    # Add time features
    master_core['hour'] = master_core['datetime'].dt.hour
    master_core['minute'] = master_core['datetime'].dt.minute
    master_core['day_of_week'] = master_core['datetime'].dt.dayofweek
    master_core['month'] = master_core['datetime'].dt.month

    # Calculate which 15-min settlement period this 3-min falls into
    master_core['settlement_term'] = ((master_core['hour'] * 60 + master_core['minute']) // 15) + 1
    master_core['minute_in_settlement'] = (master_core['hour'] * 60 + master_core['minute']) % 15

    master_core.to_csv(FEATURES_DIR / 'master_features_core.csv', index=False)

    print(f"  Date range: {master_core['datetime'].min()} to {master_core['datetime'].max()}")
    print(f"  Rows: {len(master_core):,}")
    print(f"  Days of data: {(master_core['datetime'].max() - master_core['datetime'].min()).days}")
    print(f"  Features: regulation_mw, load_mw + time features")
    print(f"  Saved: master_features_core.csv")

    # OPTION 2: All features - Limited coverage
    print("\n" + "-" * 70)
    print("OPTION 2: All Features (limited to Oct 2025+)")
    print("-" * 70)

    all_start = max(reg_df['datetime'].min(), load_df['datetime'].min(),
                    prod_df['datetime'].min(), exp_df['datetime'].min())
    all_end = min(reg_df['datetime'].max(), load_df['datetime'].max(),
                  prod_df['datetime'].max(), exp_df['datetime'].max())

    reg_all = reg_df[(reg_df['datetime'] >= all_start) & (reg_df['datetime'] <= all_end)]
    load_all = load_df[(load_df['datetime'] >= all_start) & (load_df['datetime'] <= all_end)]
    prod_all = prod_df[(prod_df['datetime'] >= all_start) & (prod_df['datetime'] <= all_end)]
    exp_all = exp_df[(exp_df['datetime'] >= all_start) & (exp_df['datetime'] <= all_end)]

    master_all = reg_all.merge(load_all, on='datetime', how='outer')
    master_all = master_all.merge(prod_all, on='datetime', how='outer')
    master_all = master_all.merge(exp_all, on='datetime', how='outer')
    master_all = master_all.sort_values('datetime').reset_index(drop=True)

    # Add time features
    master_all['hour'] = master_all['datetime'].dt.hour
    master_all['minute'] = master_all['datetime'].dt.minute
    master_all['day_of_week'] = master_all['datetime'].dt.dayofweek
    master_all['month'] = master_all['datetime'].dt.month
    master_all['settlement_term'] = ((master_all['hour'] * 60 + master_all['minute']) // 15) + 1
    master_all['minute_in_settlement'] = (master_all['hour'] * 60 + master_all['minute']) % 15

    master_all.to_csv(FEATURES_DIR / 'master_features_all.csv', index=False)

    print(f"  Date range: {master_all['datetime'].min()} to {master_all['datetime'].max()}")
    print(f"  Rows: {len(master_all):,}")
    print(f"  Days of data: {(master_all['datetime'].max() - master_all['datetime'].min()).days}")
    print(f"  Features: regulation_mw, load_mw, production_mw, export_import_mw + time features")
    print(f"  Saved: master_features_all.csv")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: TRADEOFF")
    print("=" * 70)
    print(f"\n  CORE (reg + load):  {(master_core['datetime'].max() - master_core['datetime'].min()).days} days of data, 2 features")
    print(f"  ALL (4 features):   {(master_all['datetime'].max() - master_all['datetime'].min()).days} days of data, 4 features")
    print(f"\n  Recommendation: Start with CORE for more training data.")
    print(f"  Regulation alone may capture most signal (it's the inverse of imbalance).")

if __name__ == '__main__':
    main()
