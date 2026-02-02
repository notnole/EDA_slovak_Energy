"""
Analyze how often feature values actually change vs being repeated.
Legacy scraping system captures data more frequently than it updates.
"""

import pandas as pd
import numpy as np
from pathlib import Path

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")

def analyze_changes(df, value_col, name):
    """Analyze how often values actually change."""
    print(f"\n{'='*60}")
    print(f"CHANGE FREQUENCY: {name}")
    print(f"{'='*60}")

    # Calculate changes
    df = df.sort_values('datetime').reset_index(drop=True)
    df['changed'] = df[value_col] != df[value_col].shift(1)
    df['changed'].iloc[0] = True  # First row always counts

    total_rows = len(df)
    changed_rows = df['changed'].sum()
    repeated_rows = total_rows - changed_rows

    print(f"  Total rows: {total_rows:,}")
    print(f"  Rows where value changed: {changed_rows:,}")
    print(f"  Repeated (no change): {repeated_rows:,} ({repeated_rows/total_rows*100:.1f}%)")

    # Calculate time between changes
    change_times = df[df['changed']]['datetime']
    if len(change_times) > 1:
        intervals = change_times.diff().dropna()

        print(f"\n  Time between actual changes:")
        print(f"    Mean: {intervals.mean()}")
        print(f"    Median: {intervals.median()}")
        print(f"    Min: {intervals.min()}")
        print(f"    Max: {intervals.max()}")

        # Distribution of intervals
        interval_mins = intervals.dt.total_seconds() / 60
        print(f"\n  Interval distribution (minutes):")
        for pct in [25, 50, 75, 90, 95]:
            print(f"    P{pct}: {interval_mins.quantile(pct/100):.1f} min")

    return df[df['changed']].drop(columns=['changed'])

def main():
    print("=" * 60)
    print("ANALYZING TRUE DATA UPDATE FREQUENCY")
    print("=" * 60)

    results = {}

    # Analyze each feature
    for fname, vcol in [
        ('regulation_3min.csv', 'regulation_mw'),
        ('load_3min.csv', 'load_mw'),
        ('production_3min.csv', 'production_mw'),
        ('export_import_3min.csv', 'export_import_mw')
    ]:
        fpath = FEATURES_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath, parse_dates=['datetime'])
            df_dedup = analyze_changes(df, vcol, vcol)
            results[vcol] = df_dedup

            # Save deduplicated version
            out_name = fname.replace('_3min.csv', '_dedup.csv')
            df_dedup.to_csv(FEATURES_DIR / out_name, index=False)
            print(f"  Saved: {out_name} ({len(df_dedup):,} rows)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, df in results.items():
        print(f"  {name}: {len(df):,} unique values")

if __name__ == '__main__':
    main()
