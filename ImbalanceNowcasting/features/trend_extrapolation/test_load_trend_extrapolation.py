"""
Test Load Trend Extrapolation

Question: If we learn trend+seasonal from 2024, can we extrapolate to 2025-2026
and still get a useful residual that correlates with imbalance?

This determines if STL residual is a usable feature in production.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from scipy import stats
from pathlib import Path

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\features\trend_extrapolation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load load feature and label."""
    load_df = pd.read_csv(FEATURES_DIR / 'load_3min.csv', parse_dates=['datetime'])
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(columns={'System Imbalance (MWh)': 'imbalance'})
    return load_df, label_df

def split_by_year(df, train_end='2024-12-31'):
    """Split data into train (2024) and test (2025-2026)."""
    train = df[df['datetime'] <= train_end].copy()
    test = df[df['datetime'] > train_end].copy()
    return train, test

def fit_stl_and_extract_seasonal(train_df, col, period=480):
    """Fit STL on training data and extract seasonal pattern."""
    # Resample to regular 3-min intervals
    train_resampled = train_df.set_index('datetime')[[col]].resample('3min').mean()
    train_resampled = train_resampled.interpolate(method='linear', limit=5).dropna()

    # Fit STL
    stl = STL(train_resampled[col], period=period)
    result = stl.fit()

    # Extract daily seasonal pattern (average by time of day)
    seasonal_df = pd.DataFrame({
        'seasonal': result.seasonal,
        'hour': result.seasonal.index.hour,
        'minute': result.seasonal.index.minute
    })
    seasonal_pattern = seasonal_df.groupby(['hour', 'minute'])['seasonal'].mean()

    # Extract trend (we'll use simple extrapolation)
    trend_series = result.trend

    return result, seasonal_pattern, trend_series

def apply_seasonal_to_test(test_df, seasonal_pattern, col):
    """Apply learned seasonal pattern to test data."""
    test_df = test_df.copy()
    test_df['hour'] = test_df['datetime'].dt.hour
    test_df['minute'] = test_df['datetime'].dt.minute

    # Map seasonal pattern
    test_df['seasonal'] = test_df.apply(
        lambda row: seasonal_pattern.get((row['hour'], row['minute']), 0), axis=1
    )

    # Simple trend: use mean from training
    # (In practice, trend extrapolation is risky - this tests if seasonal alone works)

    return test_df

def compute_residual_correlation(df, col, label_df):
    """Compute correlation between residual and imbalance."""
    # Merge with label
    merged = pd.merge_asof(
        label_df.sort_values('datetime'),
        df.sort_values('datetime'),
        on='datetime',
        direction='backward',
        tolerance=pd.Timedelta(minutes=15)
    )

    valid = merged[['residual', 'imbalance']].dropna()
    if len(valid) < 100:
        return np.nan, 0

    corr, pval = stats.pearsonr(valid['residual'], valid['imbalance'])
    return corr, len(valid)

def main():
    print("=" * 70)
    print("LOAD TREND EXTRAPOLATION TEST")
    print("=" * 70)
    print("\nQuestion: Can we learn seasonal pattern from 2024 and apply to 2025-2026?")

    load_df, label_df = load_data()
    col = 'load_mw'

    # Split data
    train_df, test_df = split_by_year(load_df)
    print(f"\nTrain (2024): {len(train_df):,} rows")
    print(f"Test (2025-2026): {len(test_df):,} rows")

    # Split labels too
    label_train, label_test = split_by_year(label_df)

    # Fit STL on 2024
    print("\nFitting STL on 2024 data...")
    stl_result, seasonal_pattern, trend_series = fit_stl_and_extract_seasonal(train_df, col)

    # Method 1: ToD deviation (simpler, no trend)
    print("\n" + "=" * 70)
    print("METHOD 1: Time-of-Day Deviation (no trend extrapolation)")
    print("=" * 70)

    # Compute ToD means from 2024
    train_df['hour'] = train_df['datetime'].dt.hour
    train_df['minute'] = train_df['datetime'].dt.minute
    train_df['dow'] = train_df['datetime'].dt.dayofweek
    train_df['is_weekend'] = train_df['dow'] >= 5

    tod_means_2024 = train_df.groupby(['hour', 'minute', 'is_weekend'])[col].mean()

    # Apply to test
    test_df['hour'] = test_df['datetime'].dt.hour
    test_df['minute'] = test_df['datetime'].dt.minute
    test_df['dow'] = test_df['datetime'].dt.dayofweek
    test_df['is_weekend'] = test_df['dow'] >= 5

    test_df['tod_mean'] = test_df.apply(
        lambda row: tod_means_2024.get((row['hour'], row['minute'], row['is_weekend']),
                                        tod_means_2024.get((row['hour'], row['minute'], False), np.nan)),
        axis=1
    )
    test_df['residual'] = test_df[col] - test_df['tod_mean']

    # Also apply to train for comparison
    train_df['tod_mean'] = train_df.apply(
        lambda row: tod_means_2024.get((row['hour'], row['minute'], row['is_weekend']),
                                        train_df[col].mean()),
        axis=1
    )
    train_df['residual'] = train_df[col] - train_df['tod_mean']

    # Correlations
    corr_train, n_train = compute_residual_correlation(train_df, col, label_train)
    corr_test, n_test = compute_residual_correlation(test_df, col, label_test)

    print(f"\n  ToD Deviation correlation with imbalance:")
    print(f"    2024 (in-sample):     r = {corr_train:.4f} (n={n_train:,})")
    print(f"    2025-2026 (out-of-sample): r = {corr_test:.4f} (n={n_test:,})")

    # Compare with raw load
    train_df['residual'] = train_df[col]
    test_df['residual'] = test_df[col]
    corr_raw_train, _ = compute_residual_correlation(train_df, col, label_train)
    corr_raw_test, _ = compute_residual_correlation(test_df, col, label_test)

    print(f"\n  Raw Load correlation with imbalance:")
    print(f"    2024 (in-sample):     r = {corr_raw_train:.4f}")
    print(f"    2025-2026 (out-of-sample): r = {corr_raw_test:.4f}")

    # Restore residual
    test_df['residual'] = test_df[col] - test_df['tod_mean']
    train_df['residual'] = train_df[col] - train_df['tod_mean']

    # Method 2: STL seasonal only (no trend)
    print("\n" + "=" * 70)
    print("METHOD 2: STL Seasonal Pattern (learned from 2024)")
    print("=" * 70)

    # Apply seasonal pattern to test
    test_df['stl_seasonal'] = test_df.apply(
        lambda row: seasonal_pattern.get((row['hour'], row['minute']), 0), axis=1
    )
    # Use training mean as "trend" proxy
    train_mean = train_df[col].mean()
    test_df['stl_residual'] = test_df[col] - train_mean - test_df['stl_seasonal']

    test_df['residual'] = test_df['stl_residual']
    corr_stl_test, n_stl = compute_residual_correlation(test_df, col, label_test)

    print(f"\n  STL residual (seasonal only, no trend) correlation:")
    print(f"    2025-2026: r = {corr_stl_test:.4f} (n={n_stl:,})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n  Correlation with imbalance on TEST data (2025-2026):")
    print(f"    Raw Load:           r = {corr_raw_test:.4f}")
    print(f"    ToD Deviation:      r = {corr_test:.4f}  {'[BETTER]' if abs(corr_test) > abs(corr_raw_test) else ''}")
    print(f"    STL Seasonal Only:  r = {corr_stl_test:.4f}")

    improvement = (abs(corr_test) - abs(corr_raw_test)) / abs(corr_raw_test) * 100
    print(f"\n  ToD Deviation improvement over raw: {improvement:+.1f}%")

    print("\n  CONCLUSION:")
    if abs(corr_test) > abs(corr_raw_test):
        print("    [OK] ToD Deviation extrapolates well - USE IT as feature")
    else:
        print("    [WARN] ToD Deviation doesn't improve on test - check assumptions")

    # Plot
    print("\nGenerating comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Load Trend Extrapolation Test: 2024 → 2025-2026', fontsize=14, fontweight='bold')

    # Plot 1: Raw load correlation (train vs test)
    ax = axes[0, 0]
    ax.bar(['2024\n(train)', '2025-2026\n(test)'], [corr_raw_train, corr_raw_test], color=['tab:blue', 'tab:orange'])
    ax.set_ylabel('Correlation with Imbalance')
    ax.set_title('Raw Load')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    for i, v in enumerate([corr_raw_train, corr_raw_test]):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center')

    # Plot 2: ToD Deviation correlation
    ax = axes[0, 1]
    ax.bar(['2024\n(train)', '2025-2026\n(test)'], [corr_train, corr_test], color=['tab:blue', 'tab:orange'])
    ax.set_ylabel('Correlation with Imbalance')
    ax.set_title('ToD Deviation (learned from 2024)')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    for i, v in enumerate([corr_train, corr_test]):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center')

    # Plot 3: Sample of ToD deviation on test
    ax = axes[1, 0]
    sample = test_df[test_df['datetime'] < '2025-01-15'].copy()
    ax.plot(sample['datetime'], sample[col], alpha=0.7, label='Actual Load')
    ax.plot(sample['datetime'], sample['tod_mean'], alpha=0.7, label='ToD Expected (from 2024)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Load (MW)')
    ax.set_title('ToD Pattern Extrapolation (Jan 2025 sample)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Comparison bar chart
    ax = axes[1, 1]
    methods = ['Raw Load', 'ToD Deviation', 'STL (seasonal)']
    corrs = [corr_raw_test, corr_test, corr_stl_test]
    colors = ['tab:gray', 'tab:green', 'tab:purple']
    ax.barh(methods, [abs(c) for c in corrs], color=colors)
    ax.set_xlabel('|Correlation| with Imbalance (2025-2026)')
    ax.set_title('Method Comparison on Test Data')
    for i, v in enumerate(corrs):
        ax.text(abs(v) + 0.002, i, f'{v:.3f}', va='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'load_trend_extrapolation_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: load_trend_extrapolation_test.png")

    print(f"\nOutput: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
