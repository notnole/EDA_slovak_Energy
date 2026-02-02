"""
Script to compare imbalance patterns between past (2024-2025) and current (2026).
Creates side-by-side visualizations to identify behavioral changes.
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
OUTPUT_DIR = BASE_DIR / "analysis" / "year_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'past': '#2E86AB',      # Blue for 2024
    'current': '#E94F37',   # Red for 2025-2026
    'past_light': '#7FB3D3',
    'current_light': '#F5A79B'
}

# Labels for plots
LABEL_PAST = '2024'
LABEL_CURRENT = '2025-2026'


def load_data():
    """Load and split data by year groups."""
    print("Loading master imbalance data...")
    df = pd.read_csv(MASTER_FILE, parse_dates=['datetime', 'Date'])

    # Add time features
    df['year'] = df['datetime'].dt.year
    df['hour'] = ((df['Settlement Term'] - 1) * 15) // 60
    df['hour'] = df['hour'].clip(upper=23)
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear

    # Split into 2024 (baseline) and 2025-2026 (current)
    df_past = df[df['year'] == 2024].copy()
    df_current = df[df['year'].isin([2025, 2026])].copy()

    print(f"Baseline (2024): {len(df_past):,} rows ({df_past['datetime'].min().date()} to {df_past['datetime'].max().date()})")
    print(f"Current (2025-2026): {len(df_current):,} rows ({df_current['datetime'].min().date()} to {df_current['datetime'].max().date()})")

    return df, df_past, df_current


def plot_distribution_comparison(df_past, df_current):
    """Compare distributions between periods."""
    print("\nCreating distribution comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Histogram comparison
    ax1 = axes[0, 0]
    bins = np.linspace(-100, 100, 80)
    ax1.hist(df_past['System Imbalance (MWh)'].clip(-100, 100), bins=bins,
             alpha=0.6, color=COLORS['past'], label=LABEL_PAST, density=True)
    ax1.hist(df_current['System Imbalance (MWh)'].clip(-100, 100), bins=bins,
             alpha=0.6, color=COLORS['current'], label=LABEL_CURRENT, density=True)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axvline(x=df_past['System Imbalance (MWh)'].mean(), color=COLORS['past'],
                linestyle='--', linewidth=2, label=f'Mean {LABEL_PAST}: {df_past["System Imbalance (MWh)"].mean():.2f}')
    ax1.axvline(x=df_current['System Imbalance (MWh)'].mean(), color=COLORS['current'],
                linestyle='--', linewidth=2, label=f'Mean {LABEL_CURRENT}: {df_current["System Imbalance (MWh)"].mean():.2f}')
    ax1.set_title('Distribution Comparison (Normalized)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Imbalance (MWh)')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-100, 100)

    # Box plot comparison
    ax2 = axes[0, 1]
    bp = ax2.boxplot([df_past['System Imbalance (MWh)'].dropna(),
                      df_current['System Imbalance (MWh)'].dropna()],
                     labels=[LABEL_PAST, LABEL_CURRENT], patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(COLORS['past'])
    bp['boxes'][1].set_facecolor(COLORS['current'])
    for patch in bp['boxes']:
        patch.set_alpha(0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Box Plot Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Imbalance (MWh)')

    # Statistics text
    stats_past = df_past['System Imbalance (MWh)'].describe()
    stats_curr = df_current['System Imbalance (MWh)'].describe()
    stats_text = f"{LABEL_PAST}:\n  μ={stats_past['mean']:.2f}, σ={stats_past['std']:.2f}\n  median={stats_past['50%']:.2f}\n\n"
    stats_text += f"{LABEL_CURRENT}:\n  μ={stats_curr['mean']:.2f}, σ={stats_curr['std']:.2f}\n  median={stats_curr['50%']:.2f}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Sign distribution comparison
    ax3 = axes[1, 0]
    past_pos = (df_past['System Imbalance (MWh)'] > 0).sum() / len(df_past) * 100
    past_neg = (df_past['System Imbalance (MWh)'] < 0).sum() / len(df_past) * 100
    curr_pos = (df_current['System Imbalance (MWh)'] > 0).sum() / len(df_current) * 100
    curr_neg = (df_current['System Imbalance (MWh)'] < 0).sum() / len(df_current) * 100

    x = np.arange(2)
    width = 0.35
    bars1 = ax3.bar(x - width/2, [past_pos, curr_pos], width, label='Positive (Surplus)',
                    color='#3B7A57', alpha=0.7)
    bars2 = ax3.bar(x + width/2, [past_neg, curr_neg], width, label='Negative (Deficit)',
                    color='#C73E1D', alpha=0.7)
    ax3.set_title('Sign Distribution Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([LABEL_PAST, LABEL_CURRENT])
    ax3.legend()
    ax3.set_ylim(0, 70)

    # Add percentage labels
    for bar, val in zip(bars1, [past_pos, curr_pos]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', fontsize=10)
    for bar, val in zip(bars2, [past_neg, curr_neg]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', fontsize=10)

    # Percentile comparison
    ax4 = axes[1, 1]
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    past_pct = [df_past['System Imbalance (MWh)'].quantile(p/100) for p in percentiles]
    curr_pct = [df_current['System Imbalance (MWh)'].quantile(p/100) for p in percentiles]

    x = np.arange(len(percentiles))
    width = 0.35
    ax4.bar(x - width/2, past_pct, width, label=LABEL_PAST, color=COLORS['past'], alpha=0.7)
    ax4.bar(x + width/2, curr_pct, width, label=LABEL_CURRENT, color=COLORS['current'], alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_title('Percentile Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Percentile')
    ax4.set_ylabel('Imbalance (MWh)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'P{p}' for p in percentiles])
    ax4.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_distribution_comparison.png")


def plot_daily_comparison(df_past, df_current):
    """Compare intraday patterns."""
    print("\nCreating daily seasonality comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Hourly mean comparison
    ax1 = axes[0, 0]
    past_hourly = df_past.groupby('hour')['System Imbalance (MWh)'].agg(['mean', 'std'])
    curr_hourly = df_current.groupby('hour')['System Imbalance (MWh)'].agg(['mean', 'std'])

    ax1.plot(past_hourly.index, past_hourly['mean'], color=COLORS['past'],
             linewidth=2, marker='o', markersize=5, label=LABEL_PAST)
    ax1.fill_between(past_hourly.index,
                     past_hourly['mean'] - past_hourly['std'],
                     past_hourly['mean'] + past_hourly['std'],
                     alpha=0.2, color=COLORS['past'])
    ax1.plot(curr_hourly.index, curr_hourly['mean'], color=COLORS['current'],
             linewidth=2, marker='s', markersize=5, label=LABEL_CURRENT)
    ax1.fill_between(curr_hourly.index,
                     curr_hourly['mean'] - curr_hourly['std'],
                     curr_hourly['mean'] + curr_hourly['std'],
                     alpha=0.2, color=COLORS['current'])
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Hourly Pattern Comparison (Mean ± Std)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Imbalance (MWh)')
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f'{h:02d}' for h in range(0, 24, 2)])
    ax1.legend()

    # Hourly boxplot side by side
    ax2 = axes[0, 1]
    positions_past = np.arange(24) - 0.2
    positions_curr = np.arange(24) + 0.2

    bp1 = ax2.boxplot([df_past[df_past['hour'] == h]['System Imbalance (MWh)'].dropna()
                       for h in range(24)],
                      positions=positions_past, widths=0.35, patch_artist=True,
                      showfliers=False)
    bp2 = ax2.boxplot([df_current[df_current['hour'] == h]['System Imbalance (MWh)'].dropna()
                       for h in range(24)],
                      positions=positions_curr, widths=0.35, patch_artist=True,
                      showfliers=False)

    for patch in bp1['boxes']:
        patch.set_facecolor(COLORS['past'])
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor(COLORS['current'])
        patch.set_alpha(0.7)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Hourly Boxplot Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Imbalance (MWh)')
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xticklabels([f'{h:02d}' for h in range(0, 24, 2)])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['past'], alpha=0.7, label=LABEL_PAST),
                       Patch(facecolor=COLORS['current'], alpha=0.7, label=LABEL_CURRENT)]
    ax2.legend(handles=legend_elements, loc='upper right')

    # Hourly volatility comparison
    ax3 = axes[1, 0]
    ax3.bar(past_hourly.index - 0.2, past_hourly['std'], 0.4,
            label=LABEL_PAST, color=COLORS['past'], alpha=0.7)
    ax3.bar(curr_hourly.index + 0.2, curr_hourly['std'], 0.4,
            label=LABEL_CURRENT, color=COLORS['current'], alpha=0.7)
    ax3.set_title('Hourly Volatility (Std Dev) Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Std Dev (MWh)')
    ax3.set_xticks(range(0, 24, 2))
    ax3.set_xticklabels([f'{h:02d}' for h in range(0, 24, 2)])
    ax3.legend()

    # Difference plot (2025-2026 - 2024)
    ax4 = axes[1, 1]
    diff_mean = curr_hourly['mean'] - past_hourly['mean']
    colors = [COLORS['current'] if d > 0 else COLORS['past'] for d in diff_mean]
    ax4.bar(diff_mean.index, diff_mean, color=colors, alpha=0.7, edgecolor='white')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_title(f'Difference: {LABEL_CURRENT} vs {LABEL_PAST} (Mean)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Difference (MWh)')
    ax4.set_xticks(range(0, 24, 2))
    ax4.set_xticklabels([f'{h:02d}' for h in range(0, 24, 2)])

    # Add annotation
    avg_diff = diff_mean.mean()
    ax4.text(0.02, 0.98, f'Avg difference: {avg_diff:+.2f} MWh', transform=ax4.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_daily_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_daily_comparison.png")


def plot_weekly_comparison(df_past, df_current):
    """Compare weekly patterns."""
    print("\nCreating weekly seasonality comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Day of week mean comparison
    ax1 = axes[0, 0]
    past_dow = df_past.groupby('day_of_week')['System Imbalance (MWh)'].agg(['mean', 'std'])
    curr_dow = df_current.groupby('day_of_week')['System Imbalance (MWh)'].agg(['mean', 'std'])

    x = np.arange(7)
    width = 0.35
    ax1.bar(x - width/2, past_dow['mean'], width, yerr=past_dow['std']/5,
            label=LABEL_PAST, color=COLORS['past'], alpha=0.7, capsize=3)
    ax1.bar(x + width/2, curr_dow['mean'], width, yerr=curr_dow['std']/5,
            label=LABEL_CURRENT, color=COLORS['current'], alpha=0.7, capsize=3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Day of Week Mean Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Day of Week')
    ax1.set_ylabel('Mean Imbalance (MWh)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(day_names)
    ax1.legend()

    # Boxplot comparison
    ax2 = axes[0, 1]
    positions_past = np.arange(7) - 0.2
    positions_curr = np.arange(7) + 0.2

    bp1 = ax2.boxplot([df_past[df_past['day_of_week'] == d]['System Imbalance (MWh)'].dropna()
                       for d in range(7)],
                      positions=positions_past, widths=0.35, patch_artist=True,
                      showfliers=False)
    bp2 = ax2.boxplot([df_current[df_current['day_of_week'] == d]['System Imbalance (MWh)'].dropna()
                       for d in range(7)],
                      positions=positions_curr, widths=0.35, patch_artist=True,
                      showfliers=False)

    for patch in bp1['boxes']:
        patch.set_facecolor(COLORS['past'])
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor(COLORS['current'])
        patch.set_alpha(0.7)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Day of Week Boxplot Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Imbalance (MWh)')
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(day_names)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['past'], alpha=0.7, label=LABEL_PAST),
                       Patch(facecolor=COLORS['current'], alpha=0.7, label=LABEL_CURRENT)]
    ax2.legend(handles=legend_elements, loc='upper right')

    # Weekday vs Weekend
    ax3 = axes[1, 0]
    past_weekday = df_past[df_past['day_of_week'] < 5]['System Imbalance (MWh)']
    past_weekend = df_past[df_past['day_of_week'] >= 5]['System Imbalance (MWh)']
    curr_weekday = df_current[df_current['day_of_week'] < 5]['System Imbalance (MWh)']
    curr_weekend = df_current[df_current['day_of_week'] >= 5]['System Imbalance (MWh)']

    data = [past_weekday.dropna(), past_weekend.dropna(),
            curr_weekday.dropna(), curr_weekend.dropna()]
    positions = [0.8, 1.2, 2.8, 3.2]
    bp3 = ax3.boxplot(data, positions=positions, widths=0.35, patch_artist=True, showfliers=False)

    bp3['boxes'][0].set_facecolor(COLORS['past'])
    bp3['boxes'][1].set_facecolor(COLORS['past_light'])
    bp3['boxes'][2].set_facecolor(COLORS['current'])
    bp3['boxes'][3].set_facecolor(COLORS['current_light'])
    for patch in bp3['boxes']:
        patch.set_alpha(0.7)

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Weekday vs Weekend Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Imbalance (MWh)')
    ax3.set_xticks([1, 3])
    ax3.set_xticklabels([LABEL_PAST, LABEL_CURRENT])

    legend_elements = [Patch(facecolor=COLORS['past'], alpha=0.7, label='Weekday'),
                       Patch(facecolor=COLORS['past_light'], alpha=0.7, label='Weekend')]
    ax3.legend(handles=legend_elements, loc='upper right')

    # Difference plot
    ax4 = axes[1, 1]
    diff_mean = curr_dow['mean'] - past_dow['mean']
    colors = [COLORS['current'] if d > 0 else COLORS['past'] for d in diff_mean]
    ax4.bar(range(7), diff_mean, color=colors, alpha=0.7, edgecolor='white')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_title(f'Difference: {LABEL_CURRENT} vs {LABEL_PAST} (Mean)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Day of Week')
    ax4.set_ylabel('Difference (MWh)')
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(day_names)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_weekly_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_weekly_comparison.png")


def plot_monthly_comparison(df_past, df_current):
    """Compare monthly patterns (only January available for 2026)."""
    print("\nCreating monthly seasonality comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Monthly pattern from past
    ax1 = axes[0, 0]
    past_monthly = df_past.groupby('month')['System Imbalance (MWh)'].agg(['mean', 'std'])

    curr_monthly = df_current.groupby('month')['System Imbalance (MWh)'].agg(['mean', 'std'])

    x = np.arange(1, 13)
    width = 0.35
    ax1.bar(x - width/2, past_monthly['mean'], width, color=COLORS['past'], alpha=0.7,
            yerr=past_monthly['std']/5, capsize=3, label=LABEL_PAST)

    # Plot 2025-2026 months that have data
    for month in curr_monthly.index:
        ax1.bar(month + width/2, curr_monthly.loc[month, 'mean'], width, color=COLORS['current'], alpha=0.7,
                yerr=curr_monthly.loc[month, 'std']/5, capsize=3)
    # Add label only once
    ax1.bar([], [], width, color=COLORS['current'], alpha=0.7, label=LABEL_CURRENT)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title(f'Monthly Pattern: {LABEL_PAST} vs {LABEL_CURRENT}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Mean Imbalance (MWh)')
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_names)
    ax1.legend()

    # January comparison across years
    ax2 = axes[0, 1]
    jan_2024 = df_past[df_past['month'] == 1]['System Imbalance (MWh)']
    jan_2025 = df_current[(df_current['year'] == 2025) & (df_current['month'] == 1)]['System Imbalance (MWh)']
    jan_2026 = df_current[(df_current['year'] == 2026) & (df_current['month'] == 1)]['System Imbalance (MWh)']

    bp2 = ax2.boxplot([jan_2024.dropna(), jan_2025.dropna(), jan_2026.dropna()],
                      labels=['Jan 2024', 'Jan 2025', 'Jan 2026'], patch_artist=True)
    bp2['boxes'][0].set_facecolor(COLORS['past'])
    bp2['boxes'][1].set_facecolor(COLORS['current'])
    bp2['boxes'][2].set_facecolor(COLORS['current_light'])
    for patch in bp2['boxes']:
        patch.set_alpha(0.7)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('January Comparison Across Years', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Imbalance (MWh)')

    # Statistics annotation
    stats_text = f"Jan 2024: μ={jan_2024.mean():.2f}, σ={jan_2024.std():.2f}\n"
    stats_text += f"Jan 2025: μ={jan_2025.mean():.2f}, σ={jan_2025.std():.2f}\n"
    stats_text += f"Jan 2026: μ={jan_2026.mean():.2f}, σ={jan_2026.std():.2f}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Day-of-year comparison (first 24 days)
    ax3 = axes[1, 0]
    max_day = min(df_current['day_of_year'].max(), 31)  # Compare up to available 2026 data

    for year in [2024, 2025]:
        year_data = df_past[df_past['year'] == year]
        daily = year_data[year_data['day_of_year'] <= max_day].groupby('day_of_year')['System Imbalance (MWh)'].mean()
        daily_smooth = daily.rolling(window=3, center=True, min_periods=1).mean()
        alpha = 0.5 if year == 2024 else 0.7
        ax3.plot(daily_smooth.index, daily_smooth.values, color=COLORS['past'],
                alpha=alpha, linewidth=1.5, label=str(year))

    daily_2026 = df_current.groupby('day_of_year')['System Imbalance (MWh)'].mean()
    daily_2026_smooth = daily_2026.rolling(window=3, center=True, min_periods=1).mean()
    ax3.plot(daily_2026_smooth.index, daily_2026_smooth.values, color=COLORS['current'],
            linewidth=2, label='2026')

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title(f'Daily Pattern Comparison (First {max_day} Days)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Day of Year')
    ax3.set_ylabel('Mean Imbalance (MWh)')
    ax3.legend()

    # Monthly volatility from past
    ax4 = axes[1, 1]
    past_vol = df_past.groupby('month')['System Imbalance (MWh)'].std()
    curr_vol = df_current.groupby('month')['System Imbalance (MWh)'].std()

    ax4.bar(x - width/2, past_vol, width, color=COLORS['past'], alpha=0.7, label=LABEL_PAST)
    for month in curr_vol.index:
        ax4.bar(month + width/2, curr_vol.loc[month], width, color=COLORS['current'], alpha=0.7)
    ax4.bar([], [], width, color=COLORS['current'], alpha=0.7, label=LABEL_CURRENT)

    ax4.set_title('Monthly Volatility (Std Dev)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Std Dev (MWh)')
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(month_names)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_monthly_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_monthly_comparison.png")


def plot_time_series_comparison(df, df_past, df_current):
    """Time series overview with period highlighting."""
    print("\nCreating time series comparison...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # Full time series with period coloring
    ax1 = axes[0]
    ax1.plot(df_past['datetime'], df_past['System Imbalance (MWh)'],
             linewidth=0.3, alpha=0.6, color=COLORS['past'], label=LABEL_PAST)
    ax1.plot(df_current['datetime'], df_current['System Imbalance (MWh)'],
             linewidth=0.3, alpha=0.8, color=COLORS['current'], label=LABEL_CURRENT)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Full Time Series by Period', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Imbalance (MWh)')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Daily aggregated comparison
    ax2 = axes[1]
    past_daily = df_past.groupby('Date')['System Imbalance (MWh)'].mean()
    curr_daily = df_current.groupby('Date')['System Imbalance (MWh)'].mean()

    ax2.plot(past_daily.index, past_daily.values, linewidth=1, alpha=0.7,
             color=COLORS['past'], label=LABEL_PAST)
    ax2.plot(curr_daily.index, curr_daily.values, linewidth=1.5,
             color=COLORS['current'], label=LABEL_CURRENT)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Daily Mean Imbalance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Mean Imbalance (MWh)')
    ax2.legend(loc='upper right')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Rolling statistics
    ax3 = axes[2]
    window = 7 * 96  # 7 days

    past_roll = df_past.set_index('datetime')['System Imbalance (MWh)'].rolling(window, min_periods=window//2).mean()
    curr_roll = df_current.set_index('datetime')['System Imbalance (MWh)'].rolling(window, min_periods=window//2).mean()

    ax3.plot(past_roll.index, past_roll.values, linewidth=2, color=COLORS['past'], label=f'{LABEL_PAST} (7-day rolling)')
    ax3.plot(curr_roll.index, curr_roll.values, linewidth=2, color=COLORS['current'], label=f'{LABEL_CURRENT} (7-day rolling)')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axhline(y=df_past['System Imbalance (MWh)'].mean(), color=COLORS['past'],
                linestyle='--', alpha=0.5, label=f'{LABEL_PAST} Mean: {df_past["System Imbalance (MWh)"].mean():.2f}')
    ax3.axhline(y=df_current['System Imbalance (MWh)'].mean(), color=COLORS['current'],
                linestyle='--', alpha=0.5, label=f'{LABEL_CURRENT} Mean: {df_current["System Imbalance (MWh)"].mean():.2f}')
    ax3.set_title('7-Day Rolling Mean', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Rolling Mean (MWh)')
    ax3.legend(loc='upper right')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_time_series_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_time_series_comparison.png")


def generate_comparison_report(df_past, df_current):
    """Generate comparison statistics report."""
    print("\nGenerating comparison report...")

    report = []
    report.append("=" * 80)
    report.append(f"YEAR COMPARISON REPORT: {LABEL_PAST} vs {LABEL_CURRENT}")
    report.append("=" * 80)

    imb_past = df_past['System Imbalance (MWh)']
    imb_curr = df_current['System Imbalance (MWh)']

    report.append("\n1. BASIC STATISTICS COMPARISON")
    report.append("-" * 40)
    report.append(f"{'Metric':<25} {LABEL_PAST:>15} {LABEL_CURRENT:>15} {'Diff':>12}")
    report.append("-" * 70)

    metrics = [
        ('Records', len(imb_past), len(imb_curr)),
        ('Mean (MWh)', imb_past.mean(), imb_curr.mean()),
        ('Median (MWh)', imb_past.median(), imb_curr.median()),
        ('Std Dev (MWh)', imb_past.std(), imb_curr.std()),
        ('Min (MWh)', imb_past.min(), imb_curr.min()),
        ('Max (MWh)', imb_past.max(), imb_curr.max()),
        ('Skewness', imb_past.skew(), imb_curr.skew()),
        ('Kurtosis', imb_past.kurtosis(), imb_curr.kurtosis()),
    ]

    for name, past_val, curr_val in metrics:
        if name == 'Records':
            diff = curr_val - past_val
            report.append(f"{name:<25} {past_val:>15,} {curr_val:>15,} {diff:>+12,}")
        else:
            diff = curr_val - past_val
            report.append(f"{name:<25} {past_val:>15.4f} {curr_val:>15.4f} {diff:>+12.4f}")

    report.append("\n\n2. SIGN DISTRIBUTION COMPARISON")
    report.append("-" * 40)
    past_pos = (imb_past > 0).sum() / len(imb_past) * 100
    past_neg = (imb_past < 0).sum() / len(imb_past) * 100
    curr_pos = (imb_curr > 0).sum() / len(imb_curr) * 100
    curr_neg = (imb_curr < 0).sum() / len(imb_curr) * 100

    report.append(f"{'Sign':<25} {LABEL_PAST:>15} {LABEL_CURRENT:>15} {'Diff':>12}")
    report.append("-" * 70)
    report.append(f"{'Positive (Surplus) %':<25} {past_pos:>14.2f}% {curr_pos:>14.2f}% {curr_pos-past_pos:>+11.2f}%")
    report.append(f"{'Negative (Deficit) %':<25} {past_neg:>14.2f}% {curr_neg:>14.2f}% {curr_neg-past_neg:>+11.2f}%")

    report.append("\n\n3. HOURLY COMPARISON")
    report.append("-" * 40)
    past_hourly = df_past.groupby('hour')['System Imbalance (MWh)'].mean()
    curr_hourly = df_current.groupby('hour')['System Imbalance (MWh)'].mean()

    report.append(f"{'Hour':<8} {LABEL_PAST:>12} {LABEL_CURRENT:>12} {'Diff':>12}")
    report.append("-" * 50)
    for h in range(24):
        past_val = past_hourly.get(h, 0)
        curr_val = curr_hourly.get(h, 0)
        report.append(f"{h:02d}:00    {past_val:>12.4f} {curr_val:>12.4f} {curr_val-past_val:>+12.4f}")

    report.append("\n\n4. DAY OF WEEK COMPARISON")
    report.append("-" * 40)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    past_dow = df_past.groupby('day_of_week')['System Imbalance (MWh)'].mean()
    curr_dow = df_current.groupby('day_of_week')['System Imbalance (MWh)'].mean()

    report.append(f"{'Day':<12} {LABEL_PAST:>12} {LABEL_CURRENT:>12} {'Diff':>12}")
    report.append("-" * 50)
    for d in range(7):
        past_val = past_dow.get(d, 0)
        curr_val = curr_dow.get(d, 0)
        report.append(f"{day_names[d]:<12} {past_val:>12.4f} {curr_val:>12.4f} {curr_val-past_val:>+12.4f}")

    report.append("\n\n5. KEY OBSERVATIONS")
    report.append("-" * 40)

    mean_diff = imb_curr.mean() - imb_past.mean()
    std_diff = imb_curr.std() - imb_past.std()

    if abs(mean_diff) > 0.5:
        direction = "higher" if mean_diff > 0 else "lower"
        report.append(f"- {LABEL_CURRENT} mean imbalance is {abs(mean_diff):.2f} MWh {direction} than {LABEL_PAST}")
    else:
        report.append("- Mean imbalance is similar between periods")

    if abs(std_diff) > 1:
        vol_change = "more volatile" if std_diff > 0 else "less volatile"
        report.append(f"- {LABEL_CURRENT} is {vol_change} (std diff: {std_diff:+.2f} MWh)")
    else:
        report.append("- Volatility is similar between periods")

    pos_diff = curr_pos - past_pos
    if abs(pos_diff) > 2:
        sign_change = "more surplus" if pos_diff > 0 else "more deficit"
        report.append(f"- {LABEL_CURRENT} shows {sign_change} tendency ({pos_diff:+.1f}% change)")

    report.append("\n\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    report_text = "\n".join(report)

    report_path = OUTPUT_DIR / 'year_comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)

    print("  Saved: year_comparison_report.txt")
    return report_text


def plot_microstructure_comparison(df_past, df_current):
    """Compare autocorrelation and microstructure between periods."""
    print("\nCreating microstructure comparison...")

    from statsmodels.tsa.stattools import acf, pacf

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    imb_past = df_past['System Imbalance (MWh)'].dropna().values
    imb_curr = df_current['System Imbalance (MWh)'].dropna().values

    # ACF comparison
    ax1 = axes[0, 0]
    max_lags = 96  # 24 hours worth of 15-min periods
    acf_past = acf(imb_past, nlags=max_lags, fft=True)
    acf_curr = acf(imb_curr, nlags=max_lags, fft=True)

    ax1.plot(range(max_lags + 1), acf_past, color=COLORS['past'], linewidth=2, label=LABEL_PAST)
    ax1.plot(range(max_lags + 1), acf_curr, color=COLORS['current'], linewidth=2, label=LABEL_CURRENT, alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=1.96/np.sqrt(len(imb_past)), color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=-1.96/np.sqrt(len(imb_past)), color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Autocorrelation Function (ACF) - 24h Lags', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Lag (15-min periods)')
    ax1.set_ylabel('ACF')
    ax1.legend()
    ax1.set_xlim(0, max_lags)

    # Add hour markers
    for h in range(0, 25, 4):
        ax1.axvline(x=h*4, color='gray', linestyle=':', alpha=0.3)

    # PACF comparison
    ax2 = axes[0, 1]
    pacf_lags = 48  # 12 hours
    pacf_past = pacf(imb_past, nlags=pacf_lags)
    pacf_curr = pacf(imb_curr, nlags=pacf_lags)

    ax2.plot(range(pacf_lags + 1), pacf_past, color=COLORS['past'], linewidth=2, label=LABEL_PAST)
    ax2.plot(range(pacf_lags + 1), pacf_curr, color=COLORS['current'], linewidth=2, label=LABEL_CURRENT, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=1.96/np.sqrt(len(imb_past)), color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=-1.96/np.sqrt(len(imb_past)), color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Partial Autocorrelation Function (PACF) - 12h Lags', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Lag (15-min periods)')
    ax2.set_ylabel('PACF')
    ax2.legend()
    ax2.set_xlim(0, pacf_lags)

    # Volatility clustering - squared returns ACF
    ax3 = axes[1, 0]
    sq_past = (imb_past - imb_past.mean()) ** 2
    sq_curr = (imb_curr - imb_curr.mean()) ** 2
    acf_sq_past = acf(sq_past, nlags=max_lags, fft=True)
    acf_sq_curr = acf(sq_curr, nlags=max_lags, fft=True)

    ax3.plot(range(max_lags + 1), acf_sq_past, color=COLORS['past'], linewidth=2, label=LABEL_PAST)
    ax3.plot(range(max_lags + 1), acf_sq_curr, color=COLORS['current'], linewidth=2, label=LABEL_CURRENT, alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Volatility Clustering (ACF of Squared Deviations)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Lag (15-min periods)')
    ax3.set_ylabel('ACF')
    ax3.legend()
    ax3.set_xlim(0, max_lags)

    # Rolling volatility comparison
    ax4 = axes[1, 1]
    window = 20  # 5 hours rolling window

    df_past_sorted = df_past.sort_values('datetime').copy()
    df_curr_sorted = df_current.sort_values('datetime').copy()

    roll_std_past = df_past_sorted['System Imbalance (MWh)'].rolling(window).std()
    roll_std_curr = df_curr_sorted['System Imbalance (MWh)'].rolling(window).std()

    bins = np.linspace(0, 40, 50)
    ax4.hist(roll_std_past.dropna(), bins=bins, alpha=0.6, color=COLORS['past'],
             label=f'{LABEL_PAST} (μ={roll_std_past.mean():.2f})', density=True)
    ax4.hist(roll_std_curr.dropna(), bins=bins, alpha=0.6, color=COLORS['current'],
             label=f'{LABEL_CURRENT} (μ={roll_std_curr.mean():.2f})', density=True)
    ax4.set_title('Distribution of Rolling Volatility (5h window)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Rolling Std Dev (MWh)')
    ax4.set_ylabel('Density')
    ax4.legend()

    # First difference comparison (changes)
    ax5 = axes[2, 0]
    diff_past = np.diff(imb_past)
    diff_curr = np.diff(imb_curr)

    bins = np.linspace(-50, 50, 80)
    ax5.hist(diff_past, bins=bins, alpha=0.6, color=COLORS['past'],
             label=f'{LABEL_PAST} (σ={diff_past.std():.2f})', density=True)
    ax5.hist(diff_curr, bins=bins, alpha=0.6, color=COLORS['current'],
             label=f'{LABEL_CURRENT} (σ={diff_curr.std():.2f})', density=True)
    ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax5.set_title('Distribution of Period-to-Period Changes', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Change in Imbalance (MWh)')
    ax5.set_ylabel('Density')
    ax5.legend()

    # ACF of changes (first differences)
    ax6 = axes[2, 1]
    acf_diff_past = acf(diff_past, nlags=max_lags, fft=True)
    acf_diff_curr = acf(diff_curr, nlags=max_lags, fft=True)

    ax6.plot(range(max_lags + 1), acf_diff_past, color=COLORS['past'], linewidth=2, label=LABEL_PAST)
    ax6.plot(range(max_lags + 1), acf_diff_curr, color=COLORS['current'], linewidth=2, label=LABEL_CURRENT, alpha=0.8)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax6.axhline(y=1.96/np.sqrt(len(diff_past)), color='gray', linestyle='--', alpha=0.5)
    ax6.axhline(y=-1.96/np.sqrt(len(diff_past)), color='gray', linestyle='--', alpha=0.5)
    ax6.set_title('ACF of First Differences (Stationarized)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Lag (15-min periods)')
    ax6.set_ylabel('ACF')
    ax6.legend()
    ax6.set_xlim(0, max_lags)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_microstructure_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_microstructure_comparison.png")

    # Calculate and return correlation metrics
    acf_correlation = np.corrcoef(acf_past, acf_curr)[0, 1]
    pacf_correlation = np.corrcoef(pacf_past, pacf_curr)[0, 1]
    vol_acf_correlation = np.corrcoef(acf_sq_past, acf_sq_curr)[0, 1]
    diff_acf_correlation = np.corrcoef(acf_diff_past, acf_diff_curr)[0, 1]

    return {
        'acf_correlation': acf_correlation,
        'pacf_correlation': pacf_correlation,
        'vol_acf_correlation': vol_acf_correlation,
        'diff_acf_correlation': diff_acf_correlation,
        'change_std_past': diff_past.std(),
        'change_std_curr': diff_curr.std(),
        'roll_vol_mean_past': roll_std_past.mean(),
        'roll_vol_mean_curr': roll_std_curr.mean()
    }


def generate_plot_descriptions(df_past, df_current, micro_stats):
    """Generate text descriptions for all comparison plots."""
    descriptions = []

    imb_past = df_past['System Imbalance (MWh)']
    imb_curr = df_current['System Imbalance (MWh)']

    # Distribution comparison (01)
    descriptions.append("=" * 80)
    descriptions.append("PLOT 01: DISTRIBUTION COMPARISON")
    descriptions.append("=" * 80)

    descriptions.append(f"\nTop-left: Normalized histograms overlay")
    descriptions.append(f"  {LABEL_PAST}: mean={imb_past.mean():.2f}, median={imb_past.median():.2f}")
    descriptions.append(f"  {LABEL_CURRENT}: mean={imb_curr.mean():.2f}, median={imb_curr.median():.2f}")
    descriptions.append(f"  Mean shift: {imb_curr.mean() - imb_past.mean():+.2f} MWh")

    descriptions.append(f"\nTop-right: Side-by-side boxplots")
    descriptions.append(f"  {LABEL_PAST}: IQR=[{imb_past.quantile(0.25):.2f}, {imb_past.quantile(0.75):.2f}]")
    descriptions.append(f"  {LABEL_CURRENT}: IQR=[{imb_curr.quantile(0.25):.2f}, {imb_curr.quantile(0.75):.2f}]")

    past_pos = (imb_past > 0).sum() / len(imb_past) * 100
    curr_pos = (imb_curr > 0).sum() / len(imb_curr) * 100
    descriptions.append(f"\nBottom-left: Sign distribution")
    descriptions.append(f"  {LABEL_PAST}: {past_pos:.1f}% surplus, {100-past_pos:.1f}% deficit")
    descriptions.append(f"  {LABEL_CURRENT}: {curr_pos:.1f}% surplus, {100-curr_pos:.1f}% deficit")
    descriptions.append(f"  Shift: {curr_pos - past_pos:+.1f}% toward {'surplus' if curr_pos > past_pos else 'deficit'}")

    descriptions.append(f"\nBottom-right: Percentile comparison (P1 to P99)")

    # Daily comparison (02)
    descriptions.append("\n\n" + "=" * 80)
    descriptions.append("PLOT 02: DAILY (INTRADAY) COMPARISON")
    descriptions.append("=" * 80)

    df_past['hour'] = ((df_past['Settlement Term'] - 1) * 15) // 60
    df_past['hour'] = df_past['hour'].clip(upper=23)
    df_current['hour'] = ((df_current['Settlement Term'] - 1) * 15) // 60
    df_current['hour'] = df_current['hour'].clip(upper=23)

    past_hourly = df_past.groupby('hour')['System Imbalance (MWh)'].mean()
    curr_hourly = df_current.groupby('hour')['System Imbalance (MWh)'].mean()
    diff_hourly = curr_hourly - past_hourly

    descriptions.append(f"\nTop-left: Hourly mean pattern with std bands")
    descriptions.append(f"Top-right: Side-by-side hourly boxplots")

    max_diff_hour = diff_hourly.abs().idxmax()
    min_diff_hour = diff_hourly.abs().idxmin()
    descriptions.append(f"\nKey differences:")
    descriptions.append(f"  Largest change: {max_diff_hour:02d}:00 ({diff_hourly[max_diff_hour]:+.2f} MWh)")
    descriptions.append(f"  Smallest change: {min_diff_hour:02d}:00 ({diff_hourly[min_diff_hour]:+.2f} MWh)")
    descriptions.append(f"  Average hourly diff: {diff_hourly.mean():+.2f} MWh")

    descriptions.append(f"\nBottom-left: Hourly volatility (std) comparison")
    descriptions.append(f"Bottom-right: Difference plot ({LABEL_CURRENT} - {LABEL_PAST})")

    # Weekly comparison (03)
    descriptions.append("\n\n" + "=" * 80)
    descriptions.append("PLOT 03: WEEKLY COMPARISON")
    descriptions.append("=" * 80)

    df_past['day_of_week'] = df_past['datetime'].dt.dayofweek
    df_current['day_of_week'] = df_current['datetime'].dt.dayofweek
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    past_dow = df_past.groupby('day_of_week')['System Imbalance (MWh)'].mean()
    curr_dow = df_current.groupby('day_of_week')['System Imbalance (MWh)'].mean()
    diff_dow = curr_dow - past_dow

    descriptions.append(f"\nTop-left: Day of week means with error bars")
    descriptions.append(f"Top-right: Side-by-side daily boxplots")

    descriptions.append(f"\nDay-by-day comparison:")
    for d in range(7):
        descriptions.append(f"  {day_names[d]}: {LABEL_PAST}={past_dow[d]:.2f}, {LABEL_CURRENT}={curr_dow[d]:.2f}, diff={diff_dow[d]:+.2f}")

    descriptions.append(f"\nBottom-left: Weekday vs Weekend breakdown")
    descriptions.append(f"Bottom-right: Difference plot by day")

    # Monthly comparison (04)
    descriptions.append("\n\n" + "=" * 80)
    descriptions.append("PLOT 04: MONTHLY COMPARISON")
    descriptions.append("=" * 80)

    descriptions.append(f"\nTop-left: Monthly pattern with both periods")
    descriptions.append(f"Top-right: January comparison across years (2024, 2025, 2026)")

    jan_2024 = df_past[df_past['datetime'].dt.month == 1]['System Imbalance (MWh)']
    jan_2025 = df_current[(df_current['datetime'].dt.year == 2025) & (df_current['datetime'].dt.month == 1)]['System Imbalance (MWh)']
    jan_2026 = df_current[(df_current['datetime'].dt.year == 2026) & (df_current['datetime'].dt.month == 1)]['System Imbalance (MWh)']

    descriptions.append(f"\nJanuary year-over-year:")
    descriptions.append(f"  Jan 2024: mean={jan_2024.mean():.2f}, std={jan_2024.std():.2f}")
    descriptions.append(f"  Jan 2025: mean={jan_2025.mean():.2f}, std={jan_2025.std():.2f}")
    descriptions.append(f"  Jan 2026: mean={jan_2026.mean():.2f}, std={jan_2026.std():.2f}")

    descriptions.append(f"\nBottom-left: Day-of-year pattern overlay")
    descriptions.append(f"Bottom-right: Monthly volatility comparison")

    # Time series comparison (05)
    descriptions.append("\n\n" + "=" * 80)
    descriptions.append("PLOT 05: TIME SERIES COMPARISON")
    descriptions.append("=" * 80)

    descriptions.append(f"\nTop: Full time series colored by period")
    descriptions.append(f"  {LABEL_PAST}: {len(df_past):,} observations")
    descriptions.append(f"  {LABEL_CURRENT}: {len(df_current):,} observations")

    descriptions.append(f"\nMiddle: Daily mean time series")
    descriptions.append(f"Bottom: 7-day rolling mean with period averages")

    # Microstructure comparison (06)
    descriptions.append("\n\n" + "=" * 80)
    descriptions.append("PLOT 06: MICROSTRUCTURE COMPARISON")
    descriptions.append("=" * 80)

    descriptions.append(f"\nTop-left: Autocorrelation Function (ACF)")
    descriptions.append(f"  Shows how current value correlates with past values")
    descriptions.append(f"  Correlation between periods: r={micro_stats['acf_correlation']:.4f}")
    descriptions.append(f"  INTERPRETATION: {'Nearly identical' if micro_stats['acf_correlation'] > 0.99 else 'Similar' if micro_stats['acf_correlation'] > 0.95 else 'Different'} memory structure")

    descriptions.append(f"\nTop-right: Partial Autocorrelation Function (PACF)")
    descriptions.append(f"  Shows direct correlation after removing intermediate effects")
    descriptions.append(f"  Correlation between periods: r={micro_stats['pacf_correlation']:.4f}")

    descriptions.append(f"\nMiddle-left: Volatility Clustering (ACF of squared deviations)")
    descriptions.append(f"  Shows if high/low volatility periods cluster together")
    descriptions.append(f"  Correlation between periods: r={micro_stats['vol_acf_correlation']:.4f}")
    descriptions.append(f"  INTERPRETATION: {'Similar' if micro_stats['vol_acf_correlation'] > 0.90 else 'Different'} volatility dynamics")

    descriptions.append(f"\nMiddle-right: Rolling volatility distribution (5h window)")
    descriptions.append(f"  {LABEL_PAST} mean rolling std: {micro_stats['roll_vol_mean_past']:.2f} MWh")
    descriptions.append(f"  {LABEL_CURRENT} mean rolling std: {micro_stats['roll_vol_mean_curr']:.2f} MWh")

    descriptions.append(f"\nBottom-left: Period-to-period changes distribution")
    descriptions.append(f"  {LABEL_PAST} change std: {micro_stats['change_std_past']:.2f} MWh")
    descriptions.append(f"  {LABEL_CURRENT} change std: {micro_stats['change_std_curr']:.2f} MWh")
    pct_diff = abs(micro_stats['change_std_past'] - micro_stats['change_std_curr']) / micro_stats['change_std_past'] * 100
    descriptions.append(f"  Difference: {pct_diff:.1f}%")

    descriptions.append(f"\nBottom-right: ACF of first differences (stationarized)")
    descriptions.append(f"  Correlation between periods: r={micro_stats['diff_acf_correlation']:.4f}")

    # Overall conclusion
    descriptions.append("\n\n" + "=" * 80)
    descriptions.append("OVERALL MICROSTRUCTURE CONCLUSION")
    descriptions.append("=" * 80)

    all_high = (micro_stats['acf_correlation'] > 0.99 and
                micro_stats['pacf_correlation'] > 0.99 and
                micro_stats['vol_acf_correlation'] > 0.90 and
                micro_stats['diff_acf_correlation'] > 0.99)

    if all_high:
        descriptions.append("\n[CONFIRMED] Microstructure is HIGHLY SIMILAR between periods")
        descriptions.append("  - Time series memory: IDENTICAL")
        descriptions.append("  - Volatility clustering: SIMILAR")
        descriptions.append("  - Change dynamics: IDENTICAL")
        descriptions.append("\n  => Safe to combine data for ML training")
        descriptions.append("  => Mean shift can be handled with relative features")
    else:
        descriptions.append("\n[CAUTION] Some microstructure differences detected")
        descriptions.append("  Review individual metrics above for details")

    return "\n".join(descriptions)


def generate_microstructure_report(micro_stats):
    """Add microstructure findings to report."""
    report = []
    report.append("\n\n6. MICROSTRUCTURE COMPARISON")
    report.append("-" * 40)
    report.append("\nAutocorrelation Structure Similarity:")
    report.append(f"  ACF correlation:           {micro_stats['acf_correlation']:.4f}")
    report.append(f"  PACF correlation:          {micro_stats['pacf_correlation']:.4f}")
    report.append(f"  Volatility ACF correlation:{micro_stats['vol_acf_correlation']:.4f}")
    report.append(f"  Differenced ACF correlation:{micro_stats['diff_acf_correlation']:.4f}")

    report.append("\nVolatility Metrics:")
    report.append(f"  Period-to-period change std ({LABEL_PAST}):    {micro_stats['change_std_past']:.4f} MWh")
    report.append(f"  Period-to-period change std ({LABEL_CURRENT}): {micro_stats['change_std_curr']:.4f} MWh")
    report.append(f"  Rolling volatility mean ({LABEL_PAST}):        {micro_stats['roll_vol_mean_past']:.4f} MWh")
    report.append(f"  Rolling volatility mean ({LABEL_CURRENT}):     {micro_stats['roll_vol_mean_curr']:.4f} MWh")

    report.append("\n7. RECOMMENDATION FOR ML TRAINING")
    report.append("-" * 40)

    # Decision logic
    high_similarity = (micro_stats['acf_correlation'] > 0.95 and
                       micro_stats['vol_acf_correlation'] > 0.90)

    if high_similarity:
        report.append("[YES] INCLUDE 2024 DATA - Microstructure is highly similar")
        report.append(f"  - ACF patterns match very well (r={micro_stats['acf_correlation']:.3f})")
        report.append(f"  - Volatility clustering is similar (r={micro_stats['vol_acf_correlation']:.3f})")
        report.append("  - The mean shift can be handled by relative features")
    else:
        report.append("[CAUTION] CONSIDER CAREFULLY - Some microstructure differences detected")
        report.append(f"  - ACF correlation: {micro_stats['acf_correlation']:.3f}")
        report.append(f"  - Volatility ACF correlation: {micro_stats['vol_acf_correlation']:.3f}")

    change_std_diff = abs(micro_stats['change_std_past'] - micro_stats['change_std_curr'])
    change_std_pct = change_std_diff / micro_stats['change_std_past'] * 100

    if change_std_pct < 10:
        report.append(f"[OK] Period-to-period dynamics are similar (diff: {change_std_pct:.1f}%)")
    else:
        report.append(f"[WARN] Period-to-period dynamics differ by {change_std_pct:.1f}%")

    return "\n".join(report)


def main():
    print("=" * 60)
    print(f"YEAR COMPARISON ANALYSIS: {LABEL_PAST} vs {LABEL_CURRENT}")
    print("=" * 60)

    # Load and split data
    df, df_past, df_current = load_data()

    # Create comparison visualizations
    plot_distribution_comparison(df_past, df_current)
    plot_daily_comparison(df_past, df_current)
    plot_weekly_comparison(df_past, df_current)
    plot_monthly_comparison(df_past, df_current)
    plot_time_series_comparison(df, df_past, df_current)

    # Microstructure analysis
    micro_stats = plot_microstructure_comparison(df_past, df_current)

    # Generate report
    report = generate_comparison_report(df_past, df_current)

    # Append microstructure findings to report
    micro_report = generate_microstructure_report(micro_stats)
    report_path = OUTPUT_DIR / 'year_comparison_report.txt'
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(micro_report)
    print("  Updated: year_comparison_report.txt with microstructure analysis")

    # Generate plot descriptions
    print("\nGenerating plot descriptions...")
    plot_desc = generate_plot_descriptions(df_past, df_current, micro_stats)
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
