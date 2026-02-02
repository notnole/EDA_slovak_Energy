"""
Script to analyze seasonality patterns in the system imbalance label.
Creates visualizations for daily, weekly, monthly, and yearly patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
MASTER_FILE = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
OUTPUT_DIR = BASE_DIR / "analysis" / "label_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE = (14, 8)
COLORS = {
    'main': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'positive': '#C73E1D',
    'negative': '#3B7A57'
}


def load_data():
    """Load the master imbalance data."""
    print("Loading master imbalance data...")
    df = pd.read_csv(MASTER_FILE, parse_dates=['datetime', 'Date'])
    print(f"Loaded {len(df):,} rows")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    return df


def plot_time_series_overview(df):
    """Create overview time series plot."""
    print("\nCreating time series overview...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Full time series
    ax1 = axes[0]
    ax1.plot(df['datetime'], df['System Imbalance (MWh)'], linewidth=0.5, alpha=0.7, color=COLORS['main'])
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.fill_between(df['datetime'], df['System Imbalance (MWh)'], 0,
                     where=df['System Imbalance (MWh)'] > 0, alpha=0.3, color=COLORS['positive'], label='Positive (Surplus)')
    ax1.fill_between(df['datetime'], df['System Imbalance (MWh)'], 0,
                     where=df['System Imbalance (MWh)'] < 0, alpha=0.3, color=COLORS['negative'], label='Negative (Deficit)')
    ax1.set_title('System Imbalance - Full Time Series', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Imbalance (MWh)')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Daily aggregation
    daily_df = df.groupby('Date')['System Imbalance (MWh)'].agg(['mean', 'std', 'sum']).reset_index()
    ax2 = axes[1]
    ax2.plot(daily_df['Date'], daily_df['mean'], linewidth=1, color=COLORS['main'], label='Daily Mean')
    ax2.fill_between(daily_df['Date'],
                     daily_df['mean'] - daily_df['std'],
                     daily_df['mean'] + daily_df['std'],
                     alpha=0.3, color=COLORS['main'], label='±1 Std Dev')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Daily Aggregated Imbalance (Mean ± Std Dev)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Mean Imbalance (MWh)')
    ax2.legend(loc='upper right')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Distribution
    ax3 = axes[2]
    ax3.hist(df['System Imbalance (MWh)'].dropna(), bins=100, color=COLORS['main'], alpha=0.7, edgecolor='white')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.axvline(x=df['System Imbalance (MWh)'].mean(), color=COLORS['accent'], linestyle='--', linewidth=2, label=f'Mean: {df["System Imbalance (MWh)"].mean():.2f}')
    ax3.axvline(x=df['System Imbalance (MWh)'].median(), color=COLORS['secondary'], linestyle='--', linewidth=2, label=f'Median: {df["System Imbalance (MWh)"].median():.2f}')
    ax3.set_title('Distribution of System Imbalance', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Imbalance (MWh)')
    ax3.set_ylabel('Frequency')
    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_time_series_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_time_series_overview.png")


def plot_daily_seasonality(df):
    """Analyze and plot daily (intraday) seasonality patterns."""
    print("\nAnalyzing daily seasonality...")

    # Extract hour from settlement term (96 periods per day = 15 min each)
    # Note: DST days have 97-100 periods - cap hour at 23
    df['hour'] = ((df['Settlement Term'] - 1) * 15) // 60
    df['hour'] = df['hour'].clip(upper=23)  # Cap at 23 for DST transition days
    df['quarter_hour'] = df['Settlement Term'].clip(upper=96)  # Cap at 96 for DST

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Hourly pattern (boxplot)
    ax1 = axes[0, 0]
    hourly_data = [df[df['hour'] == h]['System Imbalance (MWh)'].dropna() for h in range(24)]
    bp1 = ax1.boxplot(hourly_data, positions=range(24), widths=0.6, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor(COLORS['main'])
        patch.set_alpha(0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Intraday Pattern by Hour (Boxplot)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Imbalance (MWh)')
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])

    # 15-min settlement period pattern (mean with confidence interval)
    ax2 = axes[0, 1]
    qh_stats = df.groupby('quarter_hour')['System Imbalance (MWh)'].agg(['mean', 'std', 'median'])
    ax2.plot(qh_stats.index, qh_stats['mean'], color=COLORS['main'], linewidth=2, label='Mean')
    ax2.fill_between(qh_stats.index,
                     qh_stats['mean'] - qh_stats['std'],
                     qh_stats['mean'] + qh_stats['std'],
                     alpha=0.3, color=COLORS['main'])
    ax2.plot(qh_stats.index, qh_stats['median'], color=COLORS['secondary'], linewidth=1, linestyle='--', label='Median')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Intraday Pattern by 15-min Settlement Period', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Settlement Period (1-96)')
    ax2.set_ylabel('Imbalance (MWh)')
    ax2.legend(loc='upper right')

    # Add vertical lines for hour boundaries
    for h in range(0, 97, 4):
        ax2.axvline(x=h, color='gray', linestyle=':', alpha=0.3)

    # Hourly mean pattern
    ax3 = axes[1, 0]
    hourly_stats = df.groupby('hour')['System Imbalance (MWh)'].agg(['mean', 'std', 'count'])
    bars = ax3.bar(hourly_stats.index, hourly_stats['mean'], color=COLORS['main'], alpha=0.7, edgecolor='white')
    ax3.errorbar(hourly_stats.index, hourly_stats['mean'], yerr=hourly_stats['std']/np.sqrt(hourly_stats['count']),
                 fmt='none', color='black', capsize=3, label='95% CI')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Mean Hourly Imbalance with 95% Confidence Interval', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Mean Imbalance (MWh)')
    ax3.set_xticks(range(0, 24))
    ax3.set_xticklabels([f'{h:02d}' for h in range(24)])

    # Heatmap: Hour vs Day of Week
    ax4 = axes[1, 1]
    df['day_of_week'] = df['datetime'].dt.dayofweek
    pivot_table = df.pivot_table(values='System Imbalance (MWh)',
                                  index='day_of_week',
                                  columns='hour',
                                  aggfunc='mean')
    im = ax4.imshow(pivot_table, cmap='RdYlGn_r', aspect='auto')
    ax4.set_title('Mean Imbalance Heatmap: Day of Week vs Hour', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Day of Week')
    ax4.set_xticks(range(0, 24, 2))
    ax4.set_xticklabels([f'{h:02d}' for h in range(0, 24, 2)])
    ax4.set_yticks(range(7))
    ax4.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.colorbar(im, ax=ax4, label='Mean Imbalance (MWh)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_daily_seasonality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_daily_seasonality.png")

    return df


def plot_weekly_seasonality(df):
    """Analyze and plot weekly seasonality patterns."""
    print("\nAnalyzing weekly seasonality...")

    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_name'] = df['datetime'].dt.day_name()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Day of week boxplot
    ax1 = axes[0, 0]
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_data = [df[df['day_name'] == day]['System Imbalance (MWh)'].dropna() for day in day_order]
    bp1 = ax1.boxplot(daily_data, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], patch_artist=True)
    colors_week = [COLORS['main']]*5 + [COLORS['secondary']]*2  # Different color for weekends
    for patch, color in zip(bp1['boxes'], colors_week):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Weekly Pattern by Day (Boxplot)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Day of Week')
    ax1.set_ylabel('Imbalance (MWh)')

    # Daily mean with error bars
    ax2 = axes[0, 1]
    daily_stats = df.groupby('day_of_week')['System Imbalance (MWh)'].agg(['mean', 'std', 'count'])
    bars = ax2.bar(range(7), daily_stats['mean'],
                   color=[COLORS['main']]*5 + [COLORS['secondary']]*2,
                   alpha=0.7, edgecolor='white')
    ax2.errorbar(range(7), daily_stats['mean'],
                 yerr=1.96*daily_stats['std']/np.sqrt(daily_stats['count']),
                 fmt='none', color='black', capsize=5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Mean Daily Imbalance with 95% CI', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Mean Imbalance (MWh)')
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    # Weekday vs Weekend comparison
    ax3 = axes[1, 0]
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    weekday_data = df[~df['is_weekend']]['System Imbalance (MWh)'].dropna()
    weekend_data = df[df['is_weekend']]['System Imbalance (MWh)'].dropna()
    bp3 = ax3.boxplot([weekday_data, weekend_data], labels=['Weekday', 'Weekend'], patch_artist=True)
    bp3['boxes'][0].set_facecolor(COLORS['main'])
    bp3['boxes'][1].set_facecolor(COLORS['secondary'])
    for patch in bp3['boxes']:
        patch.set_alpha(0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Weekday vs Weekend Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Imbalance (MWh)')

    # Add statistics annotation
    stats_text = f"Weekday: μ={weekday_data.mean():.2f}, σ={weekday_data.std():.2f}\n"
    stats_text += f"Weekend: μ={weekend_data.mean():.2f}, σ={weekend_data.std():.2f}"
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Weekly aggregated time series
    ax4 = axes[1, 1]
    df['week'] = df['datetime'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_stats = df.groupby('week')['System Imbalance (MWh)'].agg(['mean', 'std', 'sum'])
    ax4.plot(weekly_stats.index, weekly_stats['mean'], color=COLORS['main'], linewidth=1.5, marker='o', markersize=3)
    ax4.fill_between(weekly_stats.index,
                     weekly_stats['mean'] - weekly_stats['std'],
                     weekly_stats['mean'] + weekly_stats['std'],
                     alpha=0.3, color=COLORS['main'])
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_title('Weekly Aggregated Imbalance Over Time', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Week')
    ax4.set_ylabel('Mean Imbalance (MWh)')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_weekly_seasonality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_weekly_seasonality.png")

    return df


def plot_monthly_seasonality(df):
    """Analyze and plot monthly seasonality patterns."""
    print("\nAnalyzing monthly seasonality...")

    df['month'] = df['datetime'].dt.month
    df['month_name'] = df['datetime'].dt.month_name()
    df['year_month'] = df['datetime'].dt.to_period('M')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Monthly boxplot
    ax1 = axes[0, 0]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_data = [df[df['month'] == m]['System Imbalance (MWh)'].dropna() for m in range(1, 13)]
    bp1 = ax1.boxplot(monthly_data, labels=month_names, patch_artist=True)
    for i, patch in enumerate(bp1['boxes']):
        # Color gradient: blue for winter, orange for summer
        if i in [11, 0, 1]:  # Dec, Jan, Feb - Winter
            patch.set_facecolor('#3498db')
        elif i in [2, 3, 4]:  # Mar, Apr, May - Spring
            patch.set_facecolor('#2ecc71')
        elif i in [5, 6, 7]:  # Jun, Jul, Aug - Summer
            patch.set_facecolor('#e74c3c')
        else:  # Sep, Oct, Nov - Autumn
            patch.set_facecolor('#f39c12')
        patch.set_alpha(0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Monthly Pattern (Boxplot)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Imbalance (MWh)')

    # Monthly mean with error bars
    ax2 = axes[0, 1]
    monthly_stats = df.groupby('month')['System Imbalance (MWh)'].agg(['mean', 'std', 'count'])
    colors_month = ['#3498db', '#3498db', '#2ecc71', '#2ecc71', '#2ecc71',
                    '#e74c3c', '#e74c3c', '#e74c3c', '#f39c12', '#f39c12', '#f39c12', '#3498db']
    bars = ax2.bar(range(1, 13), monthly_stats['mean'], color=colors_month, alpha=0.7, edgecolor='white')
    ax2.errorbar(range(1, 13), monthly_stats['mean'],
                 yerr=1.96*monthly_stats['std']/np.sqrt(monthly_stats['count']),
                 fmt='none', color='black', capsize=5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Mean Monthly Imbalance with 95% CI', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Mean Imbalance (MWh)')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_names)

    # Monthly aggregated time series
    ax3 = axes[1, 0]
    ym_stats = df.groupby('year_month')['System Imbalance (MWh)'].agg(['mean', 'std', 'sum'])
    ym_stats.index = ym_stats.index.to_timestamp()
    ax3.plot(ym_stats.index, ym_stats['mean'], color=COLORS['main'], linewidth=2, marker='o', markersize=5)
    ax3.fill_between(ym_stats.index,
                     ym_stats['mean'] - ym_stats['std'],
                     ym_stats['mean'] + ym_stats['std'],
                     alpha=0.3, color=COLORS['main'])
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Monthly Aggregated Imbalance Over Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Mean Imbalance (MWh)')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # Monthly volatility (std)
    ax4 = axes[1, 1]
    monthly_vol = df.groupby('month')['System Imbalance (MWh)'].std()
    bars = ax4.bar(range(1, 13), monthly_vol, color=colors_month, alpha=0.7, edgecolor='white')
    ax4.set_title('Monthly Volatility (Standard Deviation)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Std Dev (MWh)')
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(month_names)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_monthly_seasonality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_monthly_seasonality.png")

    return df


def plot_yearly_seasonality(df):
    """Analyze and plot yearly seasonality patterns."""
    print("\nAnalyzing yearly seasonality...")

    df['year'] = df['datetime'].dt.year
    df['day_of_year'] = df['datetime'].dt.dayofyear

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Yearly comparison boxplot
    ax1 = axes[0, 0]
    years = sorted(df['year'].unique())
    yearly_data = [df[df['year'] == y]['System Imbalance (MWh)'].dropna() for y in years]
    bp1 = ax1.boxplot(yearly_data, labels=[str(y) for y in years], patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor(COLORS['main'])
        patch.set_alpha(0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Yearly Comparison (Boxplot)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Imbalance (MWh)')

    # Day of year pattern (all years overlaid)
    ax2 = axes[0, 1]
    for year in years:
        year_df = df[df['year'] == year]
        daily_mean = year_df.groupby('day_of_year')['System Imbalance (MWh)'].mean()
        # Apply smoothing
        daily_smooth = daily_mean.rolling(window=7, center=True, min_periods=1).mean()
        ax2.plot(daily_smooth.index, daily_smooth.values, alpha=0.7, linewidth=1.5, label=str(year))
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Day of Year Pattern (7-day Rolling Mean)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Day of Year')
    ax2.set_ylabel('Mean Imbalance (MWh)')
    ax2.legend(loc='upper right')
    ax2.set_xlim(1, 366)

    # Yearly statistics comparison
    ax3 = axes[1, 0]
    yearly_stats = df.groupby('year')['System Imbalance (MWh)'].agg(['mean', 'std', 'count'])
    x = np.arange(len(years))
    width = 0.35
    bars1 = ax3.bar(x - width/2, yearly_stats['mean'], width, label='Mean', color=COLORS['main'], alpha=0.7)
    bars2 = ax3.bar(x + width/2, yearly_stats['std'], width, label='Std Dev', color=COLORS['secondary'], alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Yearly Mean and Volatility', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('MWh')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(y) for y in years])
    ax3.legend()

    # Seasonal decomposition proxy: quarterly pattern
    ax4 = axes[1, 1]
    df['quarter'] = df['datetime'].dt.quarter
    quarterly_stats = df.groupby(['year', 'quarter'])['System Imbalance (MWh)'].mean().unstack()
    quarterly_stats.plot(kind='bar', ax=ax4, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_title('Quarterly Pattern by Year', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Mean Imbalance (MWh)')
    ax4.legend(title='Quarter', labels=['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)'])
    ax4.set_xticklabels([str(y) for y in years], rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_yearly_seasonality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_yearly_seasonality.png")

    return df


def generate_plot_descriptions(df):
    """Generate text descriptions for all plots."""
    descriptions = []

    # Time series overview (01)
    descriptions.append("=" * 80)
    descriptions.append("PLOT 01: TIME SERIES OVERVIEW")
    descriptions.append("=" * 80)
    descriptions.append("\nTop panel: Full 15-minute resolution time series")
    descriptions.append("  - Green shading: Positive imbalance (surplus/over-generation)")
    descriptions.append("  - Red shading: Negative imbalance (deficit/under-generation)")

    daily_df = df.groupby('Date')['System Imbalance (MWh)'].agg(['mean', 'std'])
    descriptions.append(f"\nMiddle panel: Daily aggregated mean +/- std dev")
    descriptions.append(f"  - Daily mean range: {daily_df['mean'].min():.2f} to {daily_df['mean'].max():.2f} MWh")
    descriptions.append(f"  - Avg daily volatility: {daily_df['std'].mean():.2f} MWh")

    imb = df['System Imbalance (MWh)']
    descriptions.append(f"\nBottom panel: Distribution histogram")
    descriptions.append(f"  - Mean: {imb.mean():.2f} MWh")
    descriptions.append(f"  - Median: {imb.median():.2f} MWh")
    descriptions.append(f"  - Std Dev: {imb.std():.2f} MWh")
    descriptions.append(f"  - Skewness: {imb.skew():.2f} ({'left-skewed' if imb.skew() < 0 else 'right-skewed'})")

    # Daily seasonality (02)
    descriptions.append("\n\n" + "=" * 80)
    descriptions.append("PLOT 02: DAILY (INTRADAY) SEASONALITY")
    descriptions.append("=" * 80)

    df['hour'] = ((df['Settlement Term'] - 1) * 15) // 60
    df['hour'] = df['hour'].clip(upper=23)
    hourly = df.groupby('hour')['System Imbalance (MWh)'].agg(['mean', 'std'])

    peak_hour = hourly['mean'].idxmax()
    trough_hour = hourly['mean'].idxmin()
    most_volatile_hour = hourly['std'].idxmax()

    descriptions.append(f"\nTop-left: Hourly boxplots showing distribution per hour")
    descriptions.append(f"Top-right: 15-min settlement period pattern (96 periods/day)")
    descriptions.append(f"\nKey findings:")
    descriptions.append(f"  - Peak hour: {peak_hour:02d}:00 (mean: {hourly.loc[peak_hour, 'mean']:.2f} MWh)")
    descriptions.append(f"  - Trough hour: {trough_hour:02d}:00 (mean: {hourly.loc[trough_hour, 'mean']:.2f} MWh)")
    descriptions.append(f"  - Most volatile hour: {most_volatile_hour:02d}:00 (std: {hourly.loc[most_volatile_hour, 'std']:.2f} MWh)")
    descriptions.append(f"  - Daily amplitude: {hourly['mean'].max() - hourly['mean'].min():.2f} MWh")

    descriptions.append(f"\nBottom-left: Hourly mean with 95% confidence interval")
    descriptions.append(f"Bottom-right: Heatmap of mean imbalance by hour and day of week")

    # Weekly seasonality (03)
    descriptions.append("\n\n" + "=" * 80)
    descriptions.append("PLOT 03: WEEKLY SEASONALITY")
    descriptions.append("=" * 80)

    df['day_of_week'] = df['datetime'].dt.dayofweek
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow = df.groupby('day_of_week')['System Imbalance (MWh)'].agg(['mean', 'std'])

    peak_day = dow['mean'].idxmax()
    trough_day = dow['mean'].idxmin()

    descriptions.append(f"\nTop-left: Day of week boxplots")
    descriptions.append(f"Top-right: Day of week means with error bars")
    descriptions.append(f"\nKey findings:")
    descriptions.append(f"  - Highest day: {day_names[peak_day]} (mean: {dow.loc[peak_day, 'mean']:.2f} MWh)")
    descriptions.append(f"  - Lowest day: {day_names[trough_day]} (mean: {dow.loc[trough_day, 'mean']:.2f} MWh)")

    weekday_mean = df[df['day_of_week'] < 5]['System Imbalance (MWh)'].mean()
    weekend_mean = df[df['day_of_week'] >= 5]['System Imbalance (MWh)'].mean()
    descriptions.append(f"  - Weekday mean: {weekday_mean:.2f} MWh")
    descriptions.append(f"  - Weekend mean: {weekend_mean:.2f} MWh")
    descriptions.append(f"  - Weekend effect: {weekend_mean - weekday_mean:+.2f} MWh")

    descriptions.append(f"\nBottom-left: Weekday vs Weekend comparison")
    descriptions.append(f"Bottom-right: Weekly aggregated time series")

    # Monthly seasonality (04)
    descriptions.append("\n\n" + "=" * 80)
    descriptions.append("PLOT 04: MONTHLY SEASONALITY")
    descriptions.append("=" * 80)

    df['month'] = df['datetime'].dt.month
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly = df.groupby('month')['System Imbalance (MWh)'].agg(['mean', 'std'])

    peak_month = monthly['mean'].idxmax()
    trough_month = monthly['mean'].idxmin()
    most_volatile_month = monthly['std'].idxmax()

    descriptions.append(f"\nTop-left: Monthly boxplots (colored by season)")
    descriptions.append(f"  - Blue: Winter (Dec-Feb)")
    descriptions.append(f"  - Green: Spring (Mar-May)")
    descriptions.append(f"  - Red: Summer (Jun-Aug)")
    descriptions.append(f"  - Orange: Autumn (Sep-Nov)")

    descriptions.append(f"\nKey findings:")
    descriptions.append(f"  - Peak month: {month_names[peak_month-1]} (mean: {monthly.loc[peak_month, 'mean']:.2f} MWh)")
    descriptions.append(f"  - Trough month: {month_names[trough_month-1]} (mean: {monthly.loc[trough_month, 'mean']:.2f} MWh)")
    descriptions.append(f"  - Most volatile: {month_names[most_volatile_month-1]} (std: {monthly.loc[most_volatile_month, 'std']:.2f} MWh)")
    descriptions.append(f"  - Seasonal amplitude: {monthly['mean'].max() - monthly['mean'].min():.2f} MWh")

    descriptions.append(f"\nTop-right: Monthly means with 95% CI")
    descriptions.append(f"Bottom-left: Monthly time series trend")
    descriptions.append(f"Bottom-right: Monthly volatility (std dev)")

    # Yearly seasonality (05)
    descriptions.append("\n\n" + "=" * 80)
    descriptions.append("PLOT 05: YEARLY SEASONALITY")
    descriptions.append("=" * 80)

    df['year'] = df['datetime'].dt.year
    yearly = df.groupby('year')['System Imbalance (MWh)'].agg(['mean', 'std', 'count'])

    descriptions.append(f"\nTop-left: Yearly comparison boxplots")
    descriptions.append(f"Top-right: Day-of-year pattern overlay (7-day rolling mean)")

    descriptions.append(f"\nYearly statistics:")
    for year in sorted(df['year'].unique()):
        y_data = yearly.loc[year]
        descriptions.append(f"  {year}: mean={y_data['mean']:.2f}, std={y_data['std']:.2f}, n={int(y_data['count']):,}")

    descriptions.append(f"\nBottom-left: Year-over-year mean and volatility")
    descriptions.append(f"Bottom-right: Quarterly pattern by year")

    return "\n".join(descriptions)


def generate_statistics_report(df):
    """Generate a statistics report for the label."""
    print("\nGenerating statistics report...")

    report = []
    report.append("=" * 80)
    report.append("SYSTEM IMBALANCE LABEL ANALYSIS REPORT")
    report.append("=" * 80)

    report.append("\n1. BASIC STATISTICS")
    report.append("-" * 40)
    imb = df['System Imbalance (MWh)']
    report.append(f"Total records: {len(df):,}")
    report.append(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    report.append(f"Days covered: {(df['datetime'].max() - df['datetime'].min()).days}")
    report.append(f"\nMean: {imb.mean():.4f} MWh")
    report.append(f"Median: {imb.median():.4f} MWh")
    report.append(f"Std Dev: {imb.std():.4f} MWh")
    report.append(f"Min: {imb.min():.4f} MWh")
    report.append(f"Max: {imb.max():.4f} MWh")
    report.append(f"Skewness: {imb.skew():.4f}")
    report.append(f"Kurtosis: {imb.kurtosis():.4f}")

    report.append("\n\n2. SIGN DISTRIBUTION")
    report.append("-" * 40)
    positive = (imb > 0).sum()
    negative = (imb < 0).sum()
    zero = (imb == 0).sum()
    report.append(f"Positive (Surplus): {positive:,} ({100*positive/len(imb):.2f}%)")
    report.append(f"Negative (Deficit): {negative:,} ({100*negative/len(imb):.2f}%)")
    report.append(f"Zero: {zero:,} ({100*zero/len(imb):.2f}%)")

    report.append("\n\n3. PERCENTILES")
    report.append("-" * 40)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        report.append(f"  {p}th percentile: {imb.quantile(p/100):.4f} MWh")

    report.append("\n\n4. HOURLY STATISTICS")
    report.append("-" * 40)
    df['hour'] = ((df['Settlement Term'] - 1) * 15) // 60
    hourly_stats = df.groupby('hour')['System Imbalance (MWh)'].agg(['mean', 'std'])
    report.append(f"{'Hour':<8} {'Mean':<12} {'Std Dev':<12}")
    for h in range(24):
        report.append(f"{h:02d}:00    {hourly_stats.loc[h, 'mean']:>10.4f}   {hourly_stats.loc[h, 'std']:>10.4f}")

    report.append("\n\n5. DAY OF WEEK STATISTICS")
    report.append("-" * 40)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = df['datetime'].dt.dayofweek
    dow_stats = df.groupby('day_of_week')['System Imbalance (MWh)'].agg(['mean', 'std'])
    report.append(f"{'Day':<12} {'Mean':<12} {'Std Dev':<12}")
    for d in range(7):
        report.append(f"{day_names[d]:<12} {dow_stats.loc[d, 'mean']:>10.4f}   {dow_stats.loc[d, 'std']:>10.4f}")

    report.append("\n\n6. MONTHLY STATISTICS")
    report.append("-" * 40)
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    df['month'] = df['datetime'].dt.month
    month_stats = df.groupby('month')['System Imbalance (MWh)'].agg(['mean', 'std'])
    report.append(f"{'Month':<12} {'Mean':<12} {'Std Dev':<12}")
    for m in range(1, 13):
        report.append(f"{month_names[m-1]:<12} {month_stats.loc[m, 'mean']:>10.4f}   {month_stats.loc[m, 'std']:>10.4f}")

    report.append("\n\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    report_text = "\n".join(report)

    # Save report
    report_path = OUTPUT_DIR / 'label_statistics_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)

    print("  Saved: label_statistics_report.txt")
    return report_text


def main():
    print("=" * 60)
    print("SYSTEM IMBALANCE LABEL SEASONALITY ANALYSIS")
    print("=" * 60)

    # Load data
    df = load_data()

    # Create visualizations
    plot_time_series_overview(df)
    df = plot_daily_seasonality(df)
    df = plot_weekly_seasonality(df)
    df = plot_monthly_seasonality(df)
    df = plot_yearly_seasonality(df)

    # Generate statistics report
    report = generate_statistics_report(df)

    # Generate plot descriptions
    print("\nGenerating plot descriptions...")
    plot_desc = generate_plot_descriptions(df)
    desc_path = OUTPUT_DIR / 'plot_descriptions.txt'
    with open(desc_path, 'w', encoding='utf-8') as f:
        f.write(plot_desc)
    print("  Saved: plot_descriptions.txt")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
