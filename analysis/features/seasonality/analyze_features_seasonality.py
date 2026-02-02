"""
Feature Seasonality Analysis

Two groups:
1. Full period (2024-2026): regulation, load - daily, weekly, monthly, yearly
2. Short period (Oct 2025+): production, export_import - daily, weekly only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\features\seasonality")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_features():
    """Load all feature files."""
    features = {}
    for name, fname, col in [
        ('regulation', 'regulation_3min.csv', 'regulation_mw'),
        ('load', 'load_3min.csv', 'load_mw'),
        ('production', 'production_3min.csv', 'production_mw'),
        ('export_import', 'export_import_3min.csv', 'export_import_mw')
    ]:
        fpath = FEATURES_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath, parse_dates=['datetime'])
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['year'] = df['datetime'].dt.year
            features[name] = (df, col)
    return features

def plot_full_period_seasonality(features):
    """Plot seasonality for full period features (regulation, load)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Feature Seasonality - Full Period (2024-2026): Regulation & Load', fontsize=14, fontweight='bold')

    for row, (name, color) in enumerate([('regulation', 'tab:blue'), ('load', 'tab:green')]):
        if name not in features:
            continue
        df, col = features[name]

        # Daily (by hour)
        ax = axes[row, 0]
        hourly = df.groupby('hour')[col].agg(['mean', 'std'])
        ax.plot(hourly.index, hourly['mean'], color=color, linewidth=2)
        ax.fill_between(hourly.index, hourly['mean'] - hourly['std'],
                       hourly['mean'] + hourly['std'], color=color, alpha=0.2)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel(f'{name.title()} (MW)')
        ax.set_title('Daily Pattern')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 4))

        # Weekly (by day of week)
        ax = axes[row, 1]
        daily = df.groupby('day_of_week')[col].agg(['mean', 'std'])
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax.bar(range(7), daily['mean'], color=color, alpha=0.7, yerr=daily['std'], capsize=3)
        ax.set_xticks(range(7))
        ax.set_xticklabels(days)
        ax.set_xlabel('Day of Week')
        ax.set_ylabel(f'{name.title()} (MW)')
        ax.set_title('Weekly Pattern')
        ax.grid(True, alpha=0.3, axis='y')

        # Monthly
        ax = axes[row, 2]
        monthly = df.groupby('month')[col].agg(['mean', 'std'])
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.bar(range(1, 13), monthly['mean'], color=color, alpha=0.7, yerr=monthly['std'], capsize=2)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months, rotation=45, ha='right')
        ax.set_xlabel('Month')
        ax.set_ylabel(f'{name.title()} (MW)')
        ax.set_title('Monthly Pattern')
        ax.grid(True, alpha=0.3, axis='y')

        # Yearly
        ax = axes[row, 3]
        yearly = df.groupby('year')[col].agg(['mean', 'std'])
        ax.bar(yearly.index, yearly['mean'], color=color, alpha=0.7, yerr=yearly['std'], capsize=3)
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{name.title()} (MW)')
        ax.set_title('Yearly Pattern')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_seasonality_full_period.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 01_seasonality_full_period.png")

def plot_short_period_seasonality(features):
    """Plot seasonality for short period features (production, export_import)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Feature Seasonality - Short Period (Oct 2025+): Production & Export/Import', fontsize=14, fontweight='bold')

    for row, (name, color) in enumerate([('production', 'tab:orange'), ('export_import', 'tab:red')]):
        if name not in features:
            continue
        df, col = features[name]

        # Daily (by hour)
        ax = axes[row, 0]
        hourly = df.groupby('hour')[col].agg(['mean', 'std'])
        ax.plot(hourly.index, hourly['mean'], color=color, linewidth=2)
        ax.fill_between(hourly.index, hourly['mean'] - hourly['std'],
                       hourly['mean'] + hourly['std'], color=color, alpha=0.2)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel(f'{name.replace("_", "/").title()} (MW)')
        ax.set_title('Daily Pattern')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 4))

        # Weekly (by day of week)
        ax = axes[row, 1]
        daily = df.groupby('day_of_week')[col].agg(['mean', 'std'])
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax.bar(range(7), daily['mean'], color=color, alpha=0.7, yerr=daily['std'], capsize=3)
        ax.set_xticks(range(7))
        ax.set_xticklabels(days)
        ax.set_xlabel('Day of Week')
        ax.set_ylabel(f'{name.replace("_", "/").title()} (MW)')
        ax.set_title('Weekly Pattern')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_seasonality_short_period.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_seasonality_short_period.png")

def plot_hourly_by_daytype(features):
    """Plot hourly patterns split by weekday vs weekend."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Daily Patterns: Weekday vs Weekend', fontsize=14, fontweight='bold')

    configs = [
        ('regulation', 'tab:blue'),
        ('load', 'tab:green'),
        ('production', 'tab:orange'),
        ('export_import', 'tab:red')
    ]

    for ax, (name, color) in zip(axes.flat, configs):
        if name not in features:
            ax.text(0.5, 0.5, f'{name} not available', ha='center', va='center', transform=ax.transAxes)
            continue

        df, col = features[name]
        df['is_weekend'] = df['day_of_week'] >= 5

        weekday = df[~df['is_weekend']].groupby('hour')[col].agg(['mean', 'std'])
        weekend = df[df['is_weekend']].groupby('hour')[col].agg(['mean', 'std'])

        ax.plot(weekday.index, weekday['mean'], color=color, linewidth=2, label='Weekday')
        ax.fill_between(weekday.index, weekday['mean'] - weekday['std'],
                       weekday['mean'] + weekday['std'], color=color, alpha=0.15)

        ax.plot(weekend.index, weekend['mean'], color=color, linewidth=2, linestyle='--', label='Weekend')
        ax.fill_between(weekend.index, weekend['mean'] - weekend['std'],
                       weekend['mean'] + weekend['std'], color=color, alpha=0.1)

        ax.set_xlabel('Hour of Day')
        ax.set_ylabel(f'{name.replace("_", "/").title()} (MW)')
        ax.set_title(name.replace("_", "/").title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 4))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_hourly_weekday_weekend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 03_hourly_weekday_weekend.png")

def main():
    print("=" * 60)
    print("FEATURE SEASONALITY ANALYSIS")
    print("=" * 60)

    features = load_features()
    for name, (df, col) in features.items():
        print(f"  {name}: {len(df):,} rows")

    print("\nGenerating plots...")
    plot_full_period_seasonality(features)
    plot_short_period_seasonality(features)
    plot_hourly_by_daytype(features)

    print(f"\nOutput: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
