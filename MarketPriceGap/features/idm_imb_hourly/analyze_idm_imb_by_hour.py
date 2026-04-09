"""
IDM vs Imbalance Strategy Analysis by Hour of Day

Analyzes the "sell IDM, buy imbalance" strategy performance by hour.
Key question: Did the strategy stop working in Jan 2026, or is it hour-dependent?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "hourly_market_prices.csv"
OUTPUT_DIR = Path(__file__).parent

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10


def load_data():
    """Load and prepare market price data."""
    print("[*] Loading data...")
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp_hour'])

    # Filter to rows with both IDM and Imbalance prices
    df = df.dropna(subset=['idm_vwap', 'imb_price'])

    # Ensure spread is calculated
    df['spread_idm_imb'] = df['idm_vwap'] - df['imb_price']

    # Time features
    df['date'] = pd.to_datetime(df['timestamp_hour']).dt.date
    df['hour'] = df['timestamp_hour'].dt.hour
    df['month'] = df['timestamp_hour'].dt.month
    df['year'] = df['timestamp_hour'].dt.year
    df['year_month'] = df['timestamp_hour'].dt.to_period('M')

    # Strategy outcome: profit when IDM > Imbalance (sell IDM, let go to imbalance)
    df['strategy_profitable'] = (df['spread_idm_imb'] > 0).astype(int)
    df['profit'] = df['spread_idm_imb']  # Profit per MWh

    print(f"[+] Loaded {len(df):,} hourly records with both prices")
    print(f"    Date range: {df['timestamp_hour'].min()} to {df['timestamp_hour'].max()}")

    return df


def analyze_by_hour_and_period(df):
    """Analyze strategy by hour and time period."""
    print("\n[*] Analyzing by hour and period...")

    # Create period labels
    df['period'] = 'Pre-2025'
    df.loc[df['year'] == 2025, 'period'] = '2025'
    df.loc[(df['year'] == 2026) & (df['month'] == 1), 'period'] = 'Jan 2026'

    # Overall stats by period
    period_stats = df.groupby('period').agg({
        'strategy_profitable': ['count', 'mean', 'sum'],
        'profit': ['mean', 'sum', 'std']
    }).round(3)
    period_stats.columns = ['n_hours', 'win_rate', 'n_wins', 'avg_profit', 'total_profit', 'profit_std']
    print("\n--- Overall Stats by Period ---")
    print(period_stats)

    # By hour of day
    hourly_stats = df.groupby(['period', 'hour']).agg({
        'strategy_profitable': ['count', 'mean'],
        'profit': ['mean', 'std', 'sum']
    })
    hourly_stats.columns = ['n_obs', 'win_rate', 'avg_profit', 'profit_std', 'total_profit']
    hourly_stats = hourly_stats.reset_index()

    # Pivot for comparison
    win_rate_by_hour = hourly_stats.pivot(index='hour', columns='period', values='win_rate')
    avg_profit_by_hour = hourly_stats.pivot(index='hour', columns='period', values='avg_profit')

    return period_stats, hourly_stats, win_rate_by_hour, avg_profit_by_hour


def analyze_monthly_trend(df):
    """Analyze monthly trend of the strategy."""
    print("\n[*] Analyzing monthly trend...")

    monthly = df.groupby('year_month').agg({
        'strategy_profitable': ['count', 'mean'],
        'profit': ['mean', 'sum']
    })
    monthly.columns = ['n_hours', 'win_rate', 'avg_profit', 'total_profit']
    monthly = monthly.reset_index()
    monthly['year_month_str'] = monthly['year_month'].astype(str)

    print("\n--- Monthly Performance ---")
    print(monthly.tail(15).to_string())

    return monthly


def create_visualizations(df, period_stats, hourly_stats, win_rate_by_hour, avg_profit_by_hour, monthly):
    """Create and save visualizations."""
    print("\n[*] Creating visualizations...")

    # Figure 1: Win Rate by Hour and Period
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1.1: Win rate by hour for each period
    ax = axes[0, 0]
    periods = ['2025', 'Jan 2026']
    colors = {'2025': '#2ecc71', 'Jan 2026': '#e74c3c'}

    for period in periods:
        if period in win_rate_by_hour.columns:
            ax.plot(win_rate_by_hour.index, win_rate_by_hour[period] * 100,
                   marker='o', label=period, color=colors.get(period, 'gray'), linewidth=2)

    ax.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Break-even (50%)')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Sell IDM / Buy Imbalance: Win Rate by Hour')
    ax.legend()
    ax.set_xticks(range(24))
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Plot 1.2: Average profit by hour
    ax = axes[0, 1]
    for period in periods:
        if period in avg_profit_by_hour.columns:
            ax.bar(np.arange(24) + (0.35 if period == 'Jan 2026' else 0),
                   avg_profit_by_hour[period] if period in avg_profit_by_hour.columns else [0]*24,
                   width=0.35, label=period, color=colors.get(period, 'gray'), alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Avg Profit (EUR/MWh)')
    ax.set_title('Average Profit by Hour')
    ax.legend()
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3)

    # Plot 1.3: Monthly win rate trend
    ax = axes[1, 0]
    monthly_plot = monthly[monthly['year_month'] >= pd.Period('2025-01')]
    x = range(len(monthly_plot))
    colors_bar = ['#2ecc71' if ym < pd.Period('2026-01') else '#e74c3c'
                  for ym in monthly_plot['year_month']]
    ax.bar(x, monthly_plot['win_rate'] * 100, color=colors_bar, alpha=0.8)
    ax.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Monthly Win Rate Trend')
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_plot['year_month_str'], rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Plot 1.4: Monthly total profit
    ax = axes[1, 1]
    colors_bar = ['#2ecc71' if p > 0 else '#e74c3c' for p in monthly_plot['total_profit']]
    ax.bar(x, monthly_plot['total_profit'], color=colors_bar, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Profit (EUR/MWh)')
    ax.set_title('Monthly Total Profit')
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_plot['year_month_str'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / '01_win_rate_by_hour.png', dpi=150, bbox_inches='tight')
    print(f"[+] Saved 01_win_rate_by_hour.png")
    plt.close()

    # Figure 2: Detailed hour-by-hour comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Heatmap-style comparison
    ax = axes[0]

    # Get data for 2025 and Jan 2026
    data_2025 = hourly_stats[hourly_stats['period'] == '2025'].set_index('hour')
    data_2026 = hourly_stats[hourly_stats['period'] == 'Jan 2026'].set_index('hour')

    hours = range(24)
    width = 0.35

    wr_2025 = [data_2025.loc[h, 'win_rate'] * 100 if h in data_2025.index else 0 for h in hours]
    wr_2026 = [data_2026.loc[h, 'win_rate'] * 100 if h in data_2026.index else 0 for h in hours]

    x = np.arange(24)
    bars1 = ax.bar(x - width/2, wr_2025, width, label='2025', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, wr_2026, width, label='Jan 2026', color='#e74c3c', alpha=0.8)

    ax.axhline(y=50, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Break-even')
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Win Rate Comparison: 2025 vs Jan 2026 by Hour', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(x)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (v1, v2) in enumerate(zip(wr_2025, wr_2026)):
        if v1 > 0:
            ax.text(i - width/2, v1 + 2, f'{v1:.0f}', ha='center', va='bottom', fontsize=8, color='#27ae60')
        if v2 > 0:
            ax.text(i + width/2, v2 + 2, f'{v2:.0f}', ha='center', va='bottom', fontsize=8, color='#c0392b')

    # Plot 2: Change in win rate
    ax = axes[1]

    change = []
    for h in hours:
        w25 = data_2025.loc[h, 'win_rate'] * 100 if h in data_2025.index else 50
        w26 = data_2026.loc[h, 'win_rate'] * 100 if h in data_2026.index else 50
        change.append(w26 - w25)

    colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in change]
    ax.bar(x, change, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Change in Win Rate (pp)', fontsize=12)
    ax.set_title('Win Rate Change: Jan 2026 vs 2025 (negative = worse in 2026)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, c in enumerate(change):
        ax.text(i, c + (1 if c >= 0 else -2), f'{c:+.0f}', ha='center', va='bottom' if c >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / '02_hourly_comparison.png', dpi=150, bbox_inches='tight')
    print(f"[+] Saved 02_hourly_comparison.png")
    plt.close()

    # Figure 3: Which hours still work?
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Identify profitable hours in Jan 2026
    ax = axes[0]
    profitable_hours = []
    unprofitable_hours = []

    for h in hours:
        if h in data_2026.index and data_2026.loc[h, 'win_rate'] >= 0.5:
            profitable_hours.append(h)
        else:
            unprofitable_hours.append(h)

    # Create a visual summary
    hour_status = ['Profitable' if h in profitable_hours else 'Unprofitable' for h in hours]
    colors = ['#2ecc71' if s == 'Profitable' else '#e74c3c' for s in hour_status]

    bars = ax.bar(hours, [data_2026.loc[h, 'win_rate'] * 100 if h in data_2026.index else 0 for h in hours],
                  color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=50, color='black', linestyle='--', linewidth=2, label='Break-even (50%)')
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Win Rate in Jan 2026 (%)', fontsize=12)
    ax.set_title('Which Hours Still Work in Jan 2026?', fontsize=14, fontweight='bold')
    ax.set_xticks(hours)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Summary text
    ax = axes[1]
    ax.axis('off')

    # Calculate summary stats
    n_profitable = len(profitable_hours)
    n_unprofitable = len(unprofitable_hours)

    avg_wr_2025 = data_2025['win_rate'].mean() * 100 if len(data_2025) > 0 else 0
    avg_wr_2026 = data_2026['win_rate'].mean() * 100 if len(data_2026) > 0 else 0

    best_hours_2026 = data_2026.nlargest(5, 'win_rate')[['win_rate', 'avg_profit']].reset_index()
    worst_hours_2026 = data_2026.nsmallest(5, 'win_rate')[['win_rate', 'avg_profit']].reset_index()

    summary_text = f"""
    SELL IDM / BUY IMBALANCE STRATEGY - HOURLY ANALYSIS
    ====================================================

    OVERALL CHANGE (Jan 2026 vs 2025)
    ---------------------------------
    Average Win Rate 2025:     {avg_wr_2025:.1f}%
    Average Win Rate Jan 2026: {avg_wr_2026:.1f}%
    Change:                    {avg_wr_2026 - avg_wr_2025:+.1f} percentage points

    HOURS STILL PROFITABLE (>= 50% win rate in Jan 2026)
    ----------------------------------------------------
    {n_profitable} hours: {profitable_hours if profitable_hours else 'NONE'}

    HOURS NO LONGER PROFITABLE (< 50% in Jan 2026)
    ----------------------------------------------
    {n_unprofitable} hours: {unprofitable_hours}

    TOP 5 HOURS IN JAN 2026
    -----------------------
    """
    for _, row in best_hours_2026.iterrows():
        summary_text += f"    Hour {int(row['hour']):02d}: {row['win_rate']*100:.1f}% win rate, {row['avg_profit']:.1f} EUR/MWh avg profit\n"

    summary_text += f"""
    WORST 5 HOURS IN JAN 2026
    -------------------------
    """
    for _, row in worst_hours_2026.iterrows():
        summary_text += f"    Hour {int(row['hour']):02d}: {row['win_rate']*100:.1f}% win rate, {row['avg_profit']:.1f} EUR/MWh avg profit\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / '03_profitable_hours.png', dpi=150, bbox_inches='tight')
    print(f"[+] Saved 03_profitable_hours.png")
    plt.close()

    return profitable_hours, best_hours_2026, worst_hours_2026


def generate_summary(df, period_stats, hourly_stats, monthly, profitable_hours, best_hours, worst_hours):
    """Generate summary markdown file."""
    print("\n[*] Generating summary...")

    # Get detailed stats
    data_2025 = hourly_stats[hourly_stats['period'] == '2025'].set_index('hour')
    data_2026 = hourly_stats[hourly_stats['period'] == 'Jan 2026'].set_index('hour')

    avg_wr_2025 = data_2025['win_rate'].mean() * 100 if len(data_2025) > 0 else 0
    avg_wr_2026 = data_2026['win_rate'].mean() * 100 if len(data_2026) > 0 else 0

    unprofitable_hours = [h for h in range(24) if h not in profitable_hours]

    summary = f"""# Sell IDM / Buy Imbalance Strategy - Hourly Analysis

## Key Question

**Did the "sell IDM, buy imbalance" strategy stop working in January 2026, or is it hour-dependent?**

---

## Executive Summary

| Metric | 2025 | Jan 2026 | Change |
|--------|------|----------|--------|
| Overall Win Rate | {avg_wr_2025:.1f}% | {avg_wr_2026:.1f}% | {avg_wr_2026 - avg_wr_2025:+.1f} pp |
| Hours Profitable (>=50%) | 24/24 | {len(profitable_hours)}/24 | -{24 - len(profitable_hours)} |

**Conclusion**: The strategy deteriorated significantly in Jan 2026, but **{len(profitable_hours)} hours still show positive edge**:
- Profitable hours in Jan 2026: **{profitable_hours if profitable_hours else 'NONE'}**
- Hours that flipped negative: **{unprofitable_hours}**

---

## Detailed Findings

### 1. Overall Strategy Performance

| Period | Hours | Win Rate | Avg Profit (EUR/MWh) | Total Profit |
|--------|-------|----------|---------------------|--------------|
"""

    for period in ['2025', 'Jan 2026']:
        p_data = hourly_stats[hourly_stats['period'] == period]
        if len(p_data) > 0:
            n_hours = p_data['n_obs'].sum()
            wr = p_data['win_rate'].mean() * 100
            avg_pnl = p_data['avg_profit'].mean()
            total_pnl = p_data['total_profit'].sum()
            summary += f"| {period} | {n_hours:,} | {wr:.1f}% | {avg_pnl:.2f} | {total_pnl:,.0f} |\n"

    summary += f"""
### 2. Performance by Hour of Day

#### Hours Still Profitable in Jan 2026 (Win Rate >= 50%)

| Hour | Win Rate 2025 | Win Rate Jan 2026 | Change | Avg Profit Jan 2026 |
|------|---------------|-------------------|--------|---------------------|
"""

    for h in sorted(profitable_hours):
        wr_25 = data_2025.loc[h, 'win_rate'] * 100 if h in data_2025.index else 50
        wr_26 = data_2026.loc[h, 'win_rate'] * 100 if h in data_2026.index else 0
        pnl_26 = data_2026.loc[h, 'avg_profit'] if h in data_2026.index else 0
        summary += f"| {h:02d}:00 | {wr_25:.1f}% | {wr_26:.1f}% | {wr_26 - wr_25:+.1f} pp | {pnl_26:.1f} EUR |\n"

    summary += f"""
#### Hours No Longer Profitable in Jan 2026 (Win Rate < 50%)

| Hour | Win Rate 2025 | Win Rate Jan 2026 | Change | Avg Profit Jan 2026 |
|------|---------------|-------------------|--------|---------------------|
"""

    for h in sorted(unprofitable_hours):
        wr_25 = data_2025.loc[h, 'win_rate'] * 100 if h in data_2025.index else 50
        wr_26 = data_2026.loc[h, 'win_rate'] * 100 if h in data_2026.index else 0
        pnl_26 = data_2026.loc[h, 'avg_profit'] if h in data_2026.index else 0
        summary += f"| {h:02d}:00 | {wr_25:.1f}% | {wr_26:.1f}% | {wr_26 - wr_25:+.1f} pp | {pnl_26:.1f} EUR |\n"

    summary += f"""
### 3. Best and Worst Hours in Jan 2026

#### Top 5 Hours (by Win Rate)

| Rank | Hour | Win Rate | Avg Profit |
|------|------|----------|------------|
"""

    for i, row in best_hours.iterrows():
        summary += f"| {i+1} | {int(row['hour']):02d}:00 | {row['win_rate']*100:.1f}% | {row['avg_profit']:.1f} EUR |\n"

    summary += f"""
#### Worst 5 Hours (by Win Rate)

| Rank | Hour | Win Rate | Avg Profit |
|------|------|----------|------------|
"""

    for i, row in worst_hours.iterrows():
        summary += f"| {i+1} | {int(row['hour']):02d}:00 | {row['win_rate']*100:.1f}% | {row['avg_profit']:.1f} EUR |\n"

    # Monthly trend
    monthly_2025_2026 = monthly[monthly['year_month'] >= pd.Period('2025-01')]

    summary += f"""
### 4. Monthly Trend

| Month | Win Rate | Avg Profit | Total Profit |
|-------|----------|------------|--------------|
"""

    for _, row in monthly_2025_2026.iterrows():
        summary += f"| {row['year_month_str']} | {row['win_rate']*100:.1f}% | {row['avg_profit']:.1f} | {row['total_profit']:.0f} |\n"

    summary += f"""
---

## Trading Recommendations

### IF Hourly Selection is Possible

1. **TRADE ONLY** during hours {profitable_hours} (hours with >= 50% win rate in Jan 2026)
2. **AVOID** hours {unprofitable_hours[:5]}... (worst performers)
3. **Monitor** edge decay - if profitable hours drop below 50%, stop trading

### IF Must Trade All Hours

The strategy has become marginally profitable or unprofitable:
- Overall win rate dropped from {avg_wr_2025:.1f}% to {avg_wr_2026:.1f}%
- Consider **pausing** until market regime changes
- Or implement stricter filters (spread size threshold)

---

## Possible Explanations for Jan 2026 Deterioration

1. **Market structure change**: More participants exploiting this spread
2. **Improved forecasting**: Better imbalance predictions reducing the premium
3. **Regulatory changes**: Settlement rule modifications
4. **Seasonal effect**: January patterns differ from rest of year
5. **Data anomaly**: Only ~25 days of data - may be too short

---

## Visualizations

1. **01_win_rate_by_hour.png** - Win rate and profit by hour, monthly trend
2. **02_hourly_comparison.png** - Direct 2025 vs Jan 2026 comparison with change
3. **03_profitable_hours.png** - Summary of which hours still work

---

## Data Coverage

- **2025**: Full year hourly data
- **Jan 2026**: 25 days (Jan 1-25, 2026)
- Hours analyzed: 0-23 (all delivery hours)
"""

    with open(OUTPUT_DIR / 'summary.md', 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"[+] Saved summary.md")
    return summary


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("IDM vs IMBALANCE STRATEGY - HOURLY ANALYSIS")
    print("="*60)

    # Load data
    df = load_data()

    # Analyze by hour and period
    period_stats, hourly_stats, win_rate_by_hour, avg_profit_by_hour = analyze_by_hour_and_period(df)

    # Monthly trend
    monthly = analyze_monthly_trend(df)

    # Create visualizations
    profitable_hours, best_hours, worst_hours = create_visualizations(
        df, period_stats, hourly_stats, win_rate_by_hour, avg_profit_by_hour, monthly
    )

    # Generate summary
    summary = generate_summary(df, period_stats, hourly_stats, monthly,
                               profitable_hours, best_hours, worst_hours)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("Files created:")
    print("  - 01_win_rate_by_hour.png")
    print("  - 02_hourly_comparison.png")
    print("  - 03_profitable_hours.png")
    print("  - summary.md")


if __name__ == "__main__":
    main()
