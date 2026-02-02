"""
Feature Decomposition Analysis

Decompose features into trend, seasonal, and residual components.
The residual (deviation from expected) may be more predictive than raw values.

Full period (regulation, load): STL decomposition
Short period (production, export_import): Simple ToD-based deviation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from pathlib import Path

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\features\decomposition")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_feature(name, fname, col):
    """Load a feature file."""
    fpath = FEATURES_DIR / fname
    df = pd.read_csv(fpath, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek
    # Time of day key for grouping (hour + minute/60)
    df['tod'] = df['hour'] + df['minute'] / 60
    # Weekday/weekend flag
    df['is_weekend'] = df['day_of_week'] >= 5
    return df, col

def compute_tod_deviation(df, col):
    """
    Compute deviation from Time-of-Day average.
    Separate averages for weekday vs weekend.
    """
    # Compute mean by (tod, is_weekend)
    tod_means = df.groupby(['tod', 'is_weekend'])[col].mean().reset_index()
    tod_means.columns = ['tod', 'is_weekend', f'{col}_tod_mean']

    # Merge back
    df = df.merge(tod_means, on=['tod', 'is_weekend'], how='left')

    # Compute deviation
    df[f'{col}_deviation'] = df[col] - df[f'{col}_tod_mean']

    return df

def decompose_stl(df, col, name, period=480):
    """
    STL decomposition for full-period features.
    period=480 = 1 day at 3-min resolution (480 = 24*60/3)
    """
    print(f"\n  STL decomposition for {name}...")

    # Need evenly spaced data - resample to 3min
    df_resampled = df.set_index('datetime')[[col]].resample('3min').mean()
    df_resampled = df_resampled.interpolate(method='linear', limit=5)

    # Drop NaN
    df_resampled = df_resampled.dropna()

    if len(df_resampled) < period * 3:
        print(f"    Not enough data for STL (need {period*3}, have {len(df_resampled)})")
        return None, None

    # STL decomposition
    stl = STL(df_resampled[col], period=period)
    result = stl.fit()

    # Create decomposed dataframe
    decomp_df = pd.DataFrame({
        'datetime': df_resampled.index,
        f'{col}_trend': result.trend,
        f'{col}_seasonal': result.seasonal,
        f'{col}_residual': result.resid
    }).reset_index(drop=True)

    # Variance explained
    total_var = df_resampled[col].var()
    trend_var = result.trend.var()
    seasonal_var = result.seasonal.var()
    resid_var = result.resid.var()

    print(f"    Variance decomposition:")
    print(f"      Trend: {trend_var/total_var*100:.1f}%")
    print(f"      Seasonal: {seasonal_var/total_var*100:.1f}%")
    print(f"      Residual: {resid_var/total_var*100:.1f}%")

    return result, decomp_df

def plot_stl_decomposition(result, name, col, output_path):
    """Plot STL decomposition results."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'STL Decomposition: {name.title()}', fontsize=14, fontweight='bold')

    # Original
    axes[0].plot(result.observed, linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Original')
    axes[0].grid(True, alpha=0.3)

    # Trend
    axes[1].plot(result.trend, linewidth=1, color='tab:orange')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)

    # Seasonal
    axes[2].plot(result.seasonal, linewidth=0.5, color='tab:green')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)

    # Residual
    axes[3].plot(result.resid, linewidth=0.5, color='tab:red', alpha=0.7)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)

    # Add variance info
    total_var = result.observed.var()
    axes[0].text(0.02, 0.95, f'Total Var: {total_var:.1f}', transform=axes[0].transAxes,
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].text(0.02, 0.95, f'Trend Var: {result.trend.var()/total_var*100:.1f}%', transform=axes[1].transAxes,
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[2].text(0.02, 0.95, f'Seasonal Var: {result.seasonal.var()/total_var*100:.1f}%', transform=axes[2].transAxes,
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[3].text(0.02, 0.95, f'Residual Var: {result.resid.var()/total_var*100:.1f}%', transform=axes[3].transAxes,
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_tod_deviation(df, col, name, output_path):
    """Plot Time-of-Day deviation analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Time-of-Day Deviation: {name.title()}', fontsize=14, fontweight='bold')

    # Original vs ToD mean
    ax = axes[0, 0]
    sample = df.head(480*7)  # 1 week
    ax.plot(sample['datetime'], sample[col], linewidth=0.8, alpha=0.7, label='Actual')
    ax.plot(sample['datetime'], sample[f'{col}_tod_mean'], linewidth=1, color='red', label='ToD Mean')
    ax.set_ylabel(f'{name.title()} (MW)')
    ax.set_title('Actual vs Time-of-Day Expected (1 week sample)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Deviation time series
    ax = axes[0, 1]
    ax.plot(sample['datetime'], sample[f'{col}_deviation'], linewidth=0.8, alpha=0.7, color='tab:purple')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Deviation (MW)')
    ax.set_title('Deviation from Expected (1 week sample)')
    ax.grid(True, alpha=0.3)

    # Deviation distribution
    ax = axes[1, 0]
    deviation = df[f'{col}_deviation'].dropna()
    ax.hist(deviation, bins=100, alpha=0.7, color='tab:purple', edgecolor='white')
    ax.axvline(deviation.mean(), color='black', linestyle='--', label=f'Mean: {deviation.mean():.1f}')
    ax.axvline(0, color='red', linestyle='-', alpha=0.5)
    ax.set_xlabel('Deviation (MW)')
    ax.set_ylabel('Count')
    ax.set_title(f'Deviation Distribution (std={deviation.std():.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Deviation by hour
    ax = axes[1, 1]
    hourly_dev_std = df.groupby('hour')[f'{col}_deviation'].std()
    ax.bar(hourly_dev_std.index, hourly_dev_std.values, color='tab:purple', alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Deviation Std (MW)')
    ax.set_title('Deviation Volatility by Hour')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 60)
    print("FEATURE DECOMPOSITION ANALYSIS")
    print("=" * 60)

    decomposed_features = {}

    # 1. Load - STL decomposition (full period)
    print("\n[1/4] LOAD - STL Decomposition")
    df_load, col_load = load_feature('load', 'load_3min.csv', 'load_mw')
    stl_result_load, decomp_load = decompose_stl(df_load, col_load, 'load', period=480)
    if stl_result_load:
        plot_stl_decomposition(stl_result_load, 'load', col_load, OUTPUT_DIR / '01_load_stl.png')
        print(f"    Saved: 01_load_stl.png")
        decomposed_features['load'] = decomp_load

    # Also compute ToD deviation for load
    df_load = compute_tod_deviation(df_load, col_load)
    plot_tod_deviation(df_load, col_load, 'load', OUTPUT_DIR / '02_load_tod_deviation.png')
    print(f"    Saved: 02_load_tod_deviation.png")
    df_load.to_csv(FEATURES_DIR / 'load_3min_with_deviation.csv', index=False)

    # 2. Regulation - STL decomposition (full period)
    print("\n[2/4] REGULATION - STL Decomposition")
    df_reg, col_reg = load_feature('regulation', 'regulation_3min.csv', 'regulation_mw')
    stl_result_reg, decomp_reg = decompose_stl(df_reg, col_reg, 'regulation', period=480)
    if stl_result_reg:
        plot_stl_decomposition(stl_result_reg, 'regulation', col_reg, OUTPUT_DIR / '03_regulation_stl.png')
        print(f"    Saved: 03_regulation_stl.png")
        decomposed_features['regulation'] = decomp_reg

    # ToD deviation for regulation
    df_reg = compute_tod_deviation(df_reg, col_reg)
    plot_tod_deviation(df_reg, col_reg, 'regulation', OUTPUT_DIR / '04_regulation_tod_deviation.png')
    print(f"    Saved: 04_regulation_tod_deviation.png")
    df_reg.to_csv(FEATURES_DIR / 'regulation_3min_with_deviation.csv', index=False)

    # 3. Production - ToD deviation only (short period)
    print("\n[3/4] PRODUCTION - ToD Deviation")
    df_prod, col_prod = load_feature('production', 'production_3min.csv', 'production_mw')
    df_prod = compute_tod_deviation(df_prod, col_prod)
    plot_tod_deviation(df_prod, col_prod, 'production', OUTPUT_DIR / '05_production_tod_deviation.png')
    print(f"    Saved: 05_production_tod_deviation.png")
    df_prod.to_csv(FEATURES_DIR / 'production_3min_with_deviation.csv', index=False)

    # 4. Export/Import - ToD deviation only (short period)
    print("\n[4/4] EXPORT/IMPORT - ToD Deviation")
    df_exp, col_exp = load_feature('export_import', 'export_import_3min.csv', 'export_import_mw')
    df_exp = compute_tod_deviation(df_exp, col_exp)
    plot_tod_deviation(df_exp, col_exp, 'export_import', OUTPUT_DIR / '06_export_import_tod_deviation.png')
    print(f"    Saved: 06_export_import_tod_deviation.png")
    df_exp.to_csv(FEATURES_DIR / 'export_import_3min_with_deviation.csv', index=False)

    # Summary
    print("\n" + "=" * 60)
    print("DECOMPOSITION SUMMARY")
    print("=" * 60)

    print("\nVariance decomposition (STL):")
    if stl_result_load:
        total = stl_result_load.observed.var()
        print(f"  Load: Trend {stl_result_load.trend.var()/total*100:.1f}%, "
              f"Seasonal {stl_result_load.seasonal.var()/total*100:.1f}%, "
              f"Residual {stl_result_load.resid.var()/total*100:.1f}%")
    if stl_result_reg:
        total = stl_result_reg.observed.var()
        print(f"  Regulation: Trend {stl_result_reg.trend.var()/total*100:.1f}%, "
              f"Seasonal {stl_result_reg.seasonal.var()/total*100:.1f}%, "
              f"Residual {stl_result_reg.resid.var()/total*100:.1f}%")

    print("\nToD deviation std:")
    print(f"  Load: {df_load[f'{col_load}_deviation'].std():.1f} MW")
    print(f"  Regulation: {df_reg[f'{col_reg}_deviation'].std():.1f} MW")
    print(f"  Production: {df_prod[f'{col_prod}_deviation'].std():.1f} MW")
    print(f"  Export/Import: {df_exp[f'{col_exp}_deviation'].std():.1f} MW")

    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Updated feature files with deviation columns in: {FEATURES_DIR}")

if __name__ == '__main__':
    main()
