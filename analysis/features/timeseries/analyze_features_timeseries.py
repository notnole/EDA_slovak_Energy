"""
Feature Analysis: Time Series Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\features\timeseries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_features():
    """Load all feature files."""
    features = {}
    for name, fname in [
        ('regulation', 'regulation_3min.csv'),
        ('load', 'load_3min.csv'),
        ('production', 'production_3min.csv'),
        ('export_import', 'export_import_3min.csv')
    ]:
        fpath = FEATURES_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath, parse_dates=['datetime'])
            features[name] = df
            print(f"Loaded {name}: {len(df):,} rows, {df['datetime'].min()} to {df['datetime'].max()}")
    return features

def plot_full_timeseries(features):
    """Plot full time series for all features."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    fig.suptitle('Feature Time Series (Full Range)', fontsize=14, fontweight='bold')

    configs = [
        ('regulation', 'regulation_mw', 'Regulation (MW)', 'tab:blue'),
        ('load', 'load_mw', 'Load (MW)', 'tab:green'),
        ('production', 'production_mw', 'Production (MW)', 'tab:orange'),
        ('export_import', 'export_import_mw', 'Export/Import (MW)', 'tab:red')
    ]

    for ax, (name, col, label, color) in zip(axes, configs):
        if name in features:
            df = features[name]
            ax.plot(df['datetime'], df[col], color=color, linewidth=0.3, alpha=0.7)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)

            # Add stats
            mean_val = df[col].mean()
            std_val = df[col].std()
            ax.axhline(mean_val, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(0.02, 0.95, f'Mean: {mean_val:.1f}, Std: {std_val:.1f}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'{name} not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel(label)

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_full_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 01_full_timeseries.png")

def plot_sample_week(features):
    """Plot one week of data to see detail."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Feature Time Series (Sample Week: Nov 2025)', fontsize=14, fontweight='bold')

    # Pick a sample week where all features exist (Oct 2025+)
    start = pd.Timestamp('2025-11-10')
    end = pd.Timestamp('2025-11-17')

    configs = [
        ('regulation', 'regulation_mw', 'Regulation (MW)', 'tab:blue'),
        ('load', 'load_mw', 'Load (MW)', 'tab:green'),
        ('production', 'production_mw', 'Production (MW)', 'tab:orange'),
        ('export_import', 'export_import_mw', 'Export/Import (MW)', 'tab:red')
    ]

    for ax, (name, col, label, color) in zip(axes, configs):
        if name in features:
            df = features[name]
            mask = (df['datetime'] >= start) & (df['datetime'] < end)
            df_week = df[mask]

            if len(df_week) > 0:
                ax.plot(df_week['datetime'], df_week[col], color=color, linewidth=0.8)
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data for this week', ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel(label)
        else:
            ax.text(0.5, 0.5, f'{name} not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel(label)

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_sample_week.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_sample_week.png")

def plot_sample_day(features):
    """Plot one day of data to see intraday patterns."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Feature Time Series (Sample Day: Nov 12, 2025)', fontsize=14, fontweight='bold')

    # Pick a sample day where all features exist (Oct 2025+)
    start = pd.Timestamp('2025-11-12')
    end = pd.Timestamp('2025-11-13')

    configs = [
        ('regulation', 'regulation_mw', 'Regulation (MW)', 'tab:blue'),
        ('load', 'load_mw', 'Load (MW)', 'tab:green'),
        ('production', 'production_mw', 'Production (MW)', 'tab:orange'),
        ('export_import', 'export_import_mw', 'Export/Import (MW)', 'tab:red')
    ]

    for ax, (name, col, label, color) in zip(axes, configs):
        if name in features:
            df = features[name]
            mask = (df['datetime'] >= start) & (df['datetime'] < end)
            df_day = df[mask]

            if len(df_day) > 0:
                ax.plot(df_day['datetime'], df_day[col], color=color, linewidth=1, marker='.', markersize=2)
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)

                # Mark quarter hours
                for h in range(24):
                    for q in [0, 15, 30, 45]:
                        t = start + pd.Timedelta(hours=h, minutes=q)
                        ax.axvline(t, color='gray', alpha=0.2, linewidth=0.5)
            else:
                ax.text(0.5, 0.5, f'No data for this day', ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel(label)
        else:
            ax.text(0.5, 0.5, f'{name} not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel(label)

    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_sample_day.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 03_sample_day.png")

def plot_distributions(features):
    """Plot distributions of each feature."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold')

    configs = [
        ('regulation', 'regulation_mw', 'Regulation (MW)', 'tab:blue'),
        ('load', 'load_mw', 'Load (MW)', 'tab:green'),
        ('production', 'production_mw', 'Production (MW)', 'tab:orange'),
        ('export_import', 'export_import_mw', 'Export/Import (MW)', 'tab:red')
    ]

    for ax, (name, col, label, color) in zip(axes.flat, configs):
        if name in features:
            df = features[name]
            values = df[col].dropna()

            ax.hist(values, bins=100, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
            ax.axvline(values.mean(), color='black', linestyle='--', label=f'Mean: {values.mean():.1f}')
            ax.axvline(values.median(), color='red', linestyle=':', label=f'Median: {values.median():.1f}')
            ax.set_xlabel(label)
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Add percentiles
            p5, p95 = values.quantile(0.05), values.quantile(0.95)
            ax.axvline(p5, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(p95, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f'{name.title()}\nP5={p5:.0f}, P95={p95:.0f}', fontsize=10)
        else:
            ax.text(0.5, 0.5, f'{name} not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name.title())

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 04_distributions.png")

def main():
    print("=" * 60)
    print("FEATURE TIME SERIES ANALYSIS")
    print("=" * 60)

    features = load_features()

    print("\nGenerating plots...")
    plot_full_timeseries(features)
    plot_sample_week(features)
    plot_sample_day(features)
    plot_distributions(features)

    print("\n" + "=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()
