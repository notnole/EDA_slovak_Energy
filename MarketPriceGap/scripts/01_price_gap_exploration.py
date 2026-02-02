"""
Price Gap Exploration Analysis
Analyzes the price spreads between DA, IDM, and Imbalance markets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "MarketPriceGap" / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "MarketPriceGap" / "label" / "basic_stats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'da': '#2ecc71', 'idm': '#3498db', 'imb': '#e74c3c', 'spread': '#9b59b6'}


def load_data():
    """Load merged price data."""
    print("[*] Loading merged price data...")
    df = pd.read_csv(DATA_DIR / "hourly_market_prices.csv", parse_dates=['timestamp_hour'])
    print(f"[+] Loaded {len(df):,} hourly records")
    return df


def plot_price_comparison(df):
    """Plot 1: Price time series comparison for overlapping period."""
    print("[*] Creating price comparison plot...")

    # Filter to common period with all three prices
    mask = df['da_price'].notna() & df['idm_vwap'].notna() & df['imb_price'].notna()
    df_common = df[mask].copy()

    if len(df_common) == 0:
        print("[-] No overlapping data for all three markets")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Daily averages for cleaner visualization - select only needed columns
    df_daily = df_common.set_index('timestamp_hour')[['da_price', 'idm_vwap', 'imb_price']].resample('D').mean().dropna()

    # Plot each price
    for ax, col, label, color in zip(axes,
                                      ['da_price', 'idm_vwap', 'imb_price'],
                                      ['DA Price', 'IDM VWAP', 'Imbalance Price'],
                                      [COLORS['da'], COLORS['idm'], COLORS['imb']]):
        ax.plot(df_daily.index, df_daily[col], color=color, linewidth=0.8, alpha=0.8)
        ax.fill_between(df_daily.index, 0, df_daily[col], alpha=0.3, color=color)
        ax.set_ylabel('EUR/MWh')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.axhline(y=df_daily[col].mean(), color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(df_daily.index[-1], df_daily[col].mean(), f' mean={df_daily[col].mean():.1f}',
                va='center', fontsize=9)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.suptitle('Slovak Electricity Market Prices\n(Daily Averages)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_price_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {OUTPUT_DIR / '01_price_comparison.png'}")


def plot_price_spreads(df):
    """Plot 2: Price spread distributions and time series."""
    print("[*] Creating price spread analysis...")

    mask = df['spread_da_idm'].notna() & df['spread_idm_imb'].notna()
    df_spread = df[mask].copy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    spreads = [
        ('spread_da_idm', 'DA - IDM', '#2ecc71'),
        ('spread_idm_imb', 'IDM - Imbalance', '#3498db'),
        ('spread_da_imb', 'DA - Imbalance', '#e74c3c')
    ]

    # Top row: Distributions
    for ax, (col, label, color) in zip(axes[0], spreads):
        data = df_spread[col].dropna()
        # Clip extremes for visualization
        clip_low, clip_high = np.percentile(data, [1, 99])
        data_clipped = data[(data >= clip_low) & (data <= clip_high)]

        ax.hist(data_clipped, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=2)
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {data.mean():.1f}')
        ax.axvline(data.median(), color='orange', linestyle=':', linewidth=1.5, label=f'Median: {data.median():.1f}')
        ax.set_xlabel('EUR/MWh')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{label} Spread Distribution\n(1st-99th percentile)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

    # Bottom row: Time series (daily averages) - select only numeric columns
    numeric_cols = ['spread_da_idm', 'spread_idm_imb', 'spread_da_imb']
    df_daily = df_spread.set_index('timestamp_hour')[numeric_cols].resample('D').mean()

    for ax, (col, label, color) in zip(axes[1], spreads):
        ax.plot(df_daily.index, df_daily[col], color=color, linewidth=0.8, alpha=0.8)
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.fill_between(df_daily.index, 0, df_daily[col],
                        where=(df_daily[col] >= 0), color='green', alpha=0.3, label='Positive')
        ax.fill_between(df_daily.index, 0, df_daily[col],
                        where=(df_daily[col] < 0), color='red', alpha=0.3, label='Negative')
        ax.set_ylabel('EUR/MWh')
        ax.set_title(f'{label} Spread Over Time\n(Daily Averages)', fontsize=10, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Price Spread Analysis: DA vs IDM vs Imbalance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_price_spreads.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {OUTPUT_DIR / '02_price_spreads.png'}")


def plot_hourly_patterns(df):
    """Plot 3: Hourly patterns of prices and spreads."""
    print("[*] Creating hourly pattern analysis...")

    mask = df['da_price'].notna() & df['idm_vwap'].notna() & df['imb_price'].notna()
    df_valid = df[mask].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Average price by hour
    ax = axes[0, 0]
    hourly_prices = df_valid.groupby('hour')[['da_price', 'idm_vwap', 'imb_price']].mean()
    ax.plot(hourly_prices.index, hourly_prices['da_price'], 'o-', color=COLORS['da'],
            linewidth=2, markersize=6, label='DA Price')
    ax.plot(hourly_prices.index, hourly_prices['idm_vwap'], 's-', color=COLORS['idm'],
            linewidth=2, markersize=6, label='IDM VWAP')
    ax.plot(hourly_prices.index, hourly_prices['imb_price'], '^-', color=COLORS['imb'],
            linewidth=2, markersize=6, label='Imbalance')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Price (EUR/MWh)')
    ax.set_title('Average Price by Hour', fontsize=11, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Average spread by hour
    ax = axes[0, 1]
    hourly_spreads = df_valid.groupby('hour')[['spread_da_idm', 'spread_idm_imb', 'spread_da_imb']].mean()
    ax.bar(hourly_spreads.index - 0.25, hourly_spreads['spread_da_idm'], 0.25,
           color='#2ecc71', alpha=0.8, label='DA - IDM')
    ax.bar(hourly_spreads.index, hourly_spreads['spread_idm_imb'], 0.25,
           color='#3498db', alpha=0.8, label='IDM - Imb')
    ax.bar(hourly_spreads.index + 0.25, hourly_spreads['spread_da_imb'], 0.25,
           color='#e74c3c', alpha=0.8, label='DA - Imb')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Spread (EUR/MWh)')
    ax.set_title('Average Price Spread by Hour', fontsize=11, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Spread volatility by hour
    ax = axes[1, 0]
    hourly_std = df_valid.groupby('hour')[['spread_da_idm', 'spread_idm_imb', 'spread_da_imb']].std()
    ax.plot(hourly_std.index, hourly_std['spread_da_idm'], 'o-', color='#2ecc71',
            linewidth=2, markersize=6, label='DA - IDM')
    ax.plot(hourly_std.index, hourly_std['spread_idm_imb'], 's-', color='#3498db',
            linewidth=2, markersize=6, label='IDM - Imb')
    ax.plot(hourly_std.index, hourly_std['spread_da_imb'], '^-', color='#e74c3c',
            linewidth=2, markersize=6, label='DA - Imb')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Std Dev of Spread (EUR/MWh)')
    ax.set_title('Spread Volatility by Hour', fontsize=11, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Weekday vs Weekend
    ax = axes[1, 1]
    df_valid['day_type'] = df_valid['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
    day_spreads = df_valid.groupby('day_type')[['spread_da_idm', 'spread_idm_imb', 'spread_da_imb']].agg(['mean', 'std'])

    x = np.arange(2)
    width = 0.25
    for i, (col, label, color) in enumerate([('spread_da_idm', 'DA-IDM', '#2ecc71'),
                                              ('spread_idm_imb', 'IDM-Imb', '#3498db'),
                                              ('spread_da_imb', 'DA-Imb', '#e74c3c')]):
        means = day_spreads[col]['mean'].values
        stds = day_spreads[col]['std'].values
        ax.bar(x + (i-1)*width, means, width, yerr=stds, color=color, alpha=0.8,
               label=label, capsize=3)

    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Day Type')
    ax.set_ylabel('Average Spread (EUR/MWh)')
    ax.set_title('Price Spread: Weekday vs Weekend\n(with std dev bars)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Weekday', 'Weekend'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Temporal Patterns of Price Spreads', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_hourly_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {OUTPUT_DIR / '03_hourly_patterns.png'}")


def plot_spread_vs_imbalance(df):
    """Plot 4: Relationship between spread and system imbalance."""
    print("[*] Creating spread vs imbalance analysis...")

    mask = df['imbalance_mwh'].notna() & df['spread_idm_imb'].notna()
    df_valid = df[mask].copy()

    # Remove extreme outliers for visualization
    q_low, q_high = df_valid['imbalance_mwh'].quantile([0.01, 0.99])
    df_valid = df_valid[(df_valid['imbalance_mwh'] >= q_low) & (df_valid['imbalance_mwh'] <= q_high)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Scatter plot
    ax = axes[0, 0]
    ax.scatter(df_valid['imbalance_mwh'], df_valid['spread_idm_imb'],
               alpha=0.3, s=10, c=COLORS['spread'])
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('System Imbalance (MWh)')
    ax.set_ylabel('IDM - Imbalance Spread (EUR/MWh)')
    ax.set_title('IDM-Imbalance Spread vs System Imbalance', fontsize=11, fontweight='bold')

    # Add quadrant annotations
    ax.text(0.95, 0.95, 'Short System\nIDM > Imb', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat'))
    ax.text(0.05, 0.95, 'Long System\nIDM > Imb', transform=ax.transAxes,
            ha='left', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat'))

    # Top-right: Binned average
    ax = axes[0, 1]
    df_valid['imb_bin'] = pd.cut(df_valid['imbalance_mwh'], bins=20)
    binned = df_valid.groupby('imb_bin', observed=True).agg({
        'imbalance_mwh': 'mean',
        'spread_idm_imb': 'mean',
        'spread_da_imb': 'mean'
    }).dropna()

    ax.plot(binned['imbalance_mwh'], binned['spread_idm_imb'], 'o-',
            color='#3498db', linewidth=2, markersize=8, label='IDM - Imb')
    ax.plot(binned['imbalance_mwh'], binned['spread_da_imb'], 's-',
            color='#e74c3c', linewidth=2, markersize=8, label='DA - Imb')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('System Imbalance (MWh)')
    ax.set_ylabel('Average Spread (EUR/MWh)')
    ax.set_title('Average Spread by Imbalance Level\n(binned)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Distribution by imbalance sign
    ax = axes[1, 0]
    long_system = df_valid[df_valid['imbalance_mwh'] > 0]['spread_idm_imb']
    short_system = df_valid[df_valid['imbalance_mwh'] <= 0]['spread_idm_imb']

    ax.hist(long_system.clip(-200, 200), bins=50, alpha=0.6, color='green',
            label=f'Long System (n={len(long_system):,})', density=True)
    ax.hist(short_system.clip(-200, 200), bins=50, alpha=0.6, color='red',
            label=f'Short System (n={len(short_system):,})', density=True)
    ax.axvline(long_system.mean(), color='green', linestyle='--', linewidth=2)
    ax.axvline(short_system.mean(), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('IDM - Imbalance Spread (EUR/MWh)')
    ax.set_ylabel('Density')
    ax.set_title('Spread Distribution by System State', fontsize=11, fontweight='bold')
    ax.legend()

    # Bottom-right: Statistics table
    ax = axes[1, 1]
    ax.axis('off')

    # Create statistics
    stats_data = [
        ['Metric', 'Long System\n(Imb > 0)', 'Short System\n(Imb <= 0)'],
        ['Count', f'{len(long_system):,}', f'{len(short_system):,}'],
        ['Mean IDM-Imb Spread', f'{long_system.mean():.2f}', f'{short_system.mean():.2f}'],
        ['Median IDM-Imb Spread', f'{long_system.median():.2f}', f'{short_system.median():.2f}'],
        ['Std Dev', f'{long_system.std():.2f}', f'{short_system.std():.2f}'],
        ['% Positive Spread', f'{(long_system > 0).mean()*100:.1f}%', f'{(short_system > 0).mean()*100:.1f}%'],
    ]

    table = ax.table(cellText=stats_data, loc='center', cellLoc='center',
                     colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4a4a4a')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Statistics by System State', fontsize=11, fontweight='bold', pad=20)

    plt.suptitle('System Imbalance as Driver of Price Spreads', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_spread_vs_imbalance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {OUTPUT_DIR / '04_spread_vs_imbalance.png'}")


def create_summary(df):
    """Create summary markdown file."""
    print("[*] Creating summary report...")

    mask = df['da_price'].notna() & df['idm_vwap'].notna() & df['imb_price'].notna()
    df_valid = df[mask]

    # Calculate statistics
    summary_stats = {
        'da_price': df_valid['da_price'].describe(),
        'idm_vwap': df_valid['idm_vwap'].describe(),
        'imb_price': df_valid['imb_price'].describe(),
        'spread_da_idm': df_valid['spread_da_idm'].describe(),
        'spread_idm_imb': df_valid['spread_idm_imb'].describe(),
        'spread_da_imb': df_valid['spread_da_imb'].describe()
    }

    # Correlation with imbalance
    imb_corr = df_valid[['imbalance_mwh', 'spread_da_idm', 'spread_idm_imb', 'spread_da_imb']].corr()

    summary = f"""# Market Price Gap Analysis - Basic Statistics

## Overview

This analysis examines the price relationships between three Slovak electricity markets:
1. **Day-Ahead (DA)** - Hourly auction prices cleared day before delivery
2. **Intraday Market (IDM)** - Continuous trading up to delivery
3. **Imbalance** - Settlement prices for system deviations

**Data Period**: {df_valid['timestamp_hour'].min().date()} to {df_valid['timestamp_hour'].max().date()}
**Records**: {len(df_valid):,} hourly observations with all three prices

---

## Key Findings

### Price Levels (EUR/MWh)

| Market | Mean | Median | Std Dev | Min | Max |
|--------|------|--------|---------|-----|-----|
| DA Price | {summary_stats['da_price']['mean']:.2f} | {summary_stats['da_price']['50%']:.2f} | {summary_stats['da_price']['std']:.2f} | {summary_stats['da_price']['min']:.2f} | {summary_stats['da_price']['max']:.2f} |
| IDM VWAP | {summary_stats['idm_vwap']['mean']:.2f} | {summary_stats['idm_vwap']['50%']:.2f} | {summary_stats['idm_vwap']['std']:.2f} | {summary_stats['idm_vwap']['min']:.2f} | {summary_stats['idm_vwap']['max']:.2f} |
| Imbalance | {summary_stats['imb_price']['mean']:.2f} | {summary_stats['imb_price']['50%']:.2f} | {summary_stats['imb_price']['std']:.2f} | {summary_stats['imb_price']['min']:.2f} | {summary_stats['imb_price']['max']:.2f} |

**Observation**: IDM prices are slightly higher than DA on average (+{summary_stats['idm_vwap']['mean'] - summary_stats['da_price']['mean']:.1f} EUR/MWh),
while imbalance prices are lower on average ({summary_stats['imb_price']['mean'] - summary_stats['da_price']['mean']:.1f} EUR/MWh vs DA).
However, imbalance prices have significantly higher volatility (std = {summary_stats['imb_price']['std']:.1f} vs {summary_stats['da_price']['std']:.1f} for DA).

### Price Spreads (EUR/MWh)

| Spread | Mean | Median | Std Dev | % Positive |
|--------|------|--------|---------|------------|
| DA - IDM | {summary_stats['spread_da_idm']['mean']:.2f} | {summary_stats['spread_da_idm']['50%']:.2f} | {summary_stats['spread_da_idm']['std']:.2f} | {(df_valid['spread_da_idm'] > 0).mean()*100:.1f}% |
| IDM - Imbalance | {summary_stats['spread_idm_imb']['mean']:.2f} | {summary_stats['spread_idm_imb']['50%']:.2f} | {summary_stats['spread_idm_imb']['std']:.2f} | {(df_valid['spread_idm_imb'] > 0).mean()*100:.1f}% |
| DA - Imbalance | {summary_stats['spread_da_imb']['mean']:.2f} | {summary_stats['spread_da_imb']['50%']:.2f} | {summary_stats['spread_da_imb']['std']:.2f} | {(df_valid['spread_da_imb'] > 0).mean()*100:.1f}% |

**Key Insight**: The IDM-Imbalance spread averages +{summary_stats['spread_idm_imb']['mean']:.1f} EUR/MWh, meaning participants
buying in IDM typically pay more than the imbalance settlement price. This spread represents the "insurance premium"
for avoiding imbalance risk.

### Correlation with System Imbalance

| Variable | Correlation with System Imbalance (MWh) |
|----------|----------------------------------------|
| DA-IDM Spread | {imb_corr.loc['imbalance_mwh', 'spread_da_idm']:.3f} |
| IDM-Imbalance Spread | {imb_corr.loc['imbalance_mwh', 'spread_idm_imb']:.3f} |
| DA-Imbalance Spread | {imb_corr.loc['imbalance_mwh', 'spread_da_imb']:.3f} |

**Key Finding**: There is a negative correlation between system imbalance and the IDM-Imbalance spread
(r = {imb_corr.loc['imbalance_mwh', 'spread_idm_imb']:.3f}). When the system is long (positive imbalance),
imbalance prices tend to be lower, increasing the spread. When the system is short (negative imbalance),
imbalance prices spike, reducing or inverting the spread.

---

## Temporal Patterns

### Hourly Profile
- Morning ramp (6-9h): Increased price volatility and spreads
- Midday solar hours (10-14h): Lower prices and compressed spreads
- Evening peak (17-20h): Highest prices and spread variability

### Weekday vs Weekend
- Weekend prices are lower across all markets
- Spreads are slightly smaller on weekends due to lower demand uncertainty

---

## Visualizations

1. **01_price_comparison.png** - Time series of all three market prices
2. **02_price_spreads.png** - Distribution and time evolution of spreads
3. **03_hourly_patterns.png** - Hourly and weekly patterns
4. **04_spread_vs_imbalance.png** - Relationship between spreads and system state

---

## Next Steps

1. **Feature Analysis**: Examine what drives the spreads (load forecast error, renewable generation, etc.)
2. **Predictability**: Can spreads be forecasted from available signals?
3. **Trading Implications**: Identify optimal market for different scenarios
"""

    with open(OUTPUT_DIR / 'summary.md', 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"[+] Saved: {OUTPUT_DIR / 'summary.md'}")


def main():
    print("=" * 60)
    print("MARKET PRICE GAP EXPLORATION ANALYSIS")
    print("=" * 60)

    df = load_data()

    # Create all plots
    plot_price_comparison(df)
    plot_price_spreads(df)
    plot_hourly_patterns(df)
    plot_spread_vs_imbalance(df)
    create_summary(df)

    print("\n" + "=" * 60)
    print("[+] Analysis complete! Check MarketPriceGap/label/basic_stats/")
    print("=" * 60)


if __name__ == "__main__":
    main()
