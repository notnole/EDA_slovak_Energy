"""
Time series decomposition analysis for system imbalance.
Decomposes into trend, seasonal components, and residuals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
MASTER_FILE = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
OUTPUT_DIR = BASE_DIR / "analysis" / "decomposition"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'original': '#2E86AB',
    'trend': '#E94F37',
    'seasonal_daily': '#F6AE2D',
    'seasonal_weekly': '#86BA90',
    'residual': '#7A28CB'
}


def load_data():
    """Load the master imbalance data."""
    print("Loading master imbalance data...")
    df = pd.read_csv(MASTER_FILE, parse_dates=['datetime', 'Date'])
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"Loaded {len(df):,} rows")
    return df


def stl_decomposition(df):
    """Perform STL decomposition with daily seasonality."""
    print("\nPerforming STL decomposition (daily period=96)...")
    from statsmodels.tsa.seasonal import STL

    # Use a subset for speed if data is large, or full data
    series = df['System Imbalance (MWh)'].values

    # STL with daily period (96 = 24 hours * 4 quarters)
    stl = STL(series, period=96, robust=True)
    result = stl.fit()

    return result


def mstl_decomposition(df):
    """Perform MSTL decomposition with multiple seasonalities."""
    print("\nPerforming MSTL decomposition (daily=96, weekly=672)...")
    from statsmodels.tsa.seasonal import MSTL

    series = df['System Imbalance (MWh)'].values

    # MSTL with daily (96) and weekly (672) periods
    mstl = MSTL(series, periods=[96, 672])
    result = mstl.fit()

    return result


def spectral_analysis(df):
    """Perform spectral/Fourier analysis to find dominant frequencies."""
    print("\nPerforming spectral analysis...")

    series = df['System Imbalance (MWh)'].values
    n = len(series)

    # Remove mean
    series_centered = series - np.mean(series)

    # FFT
    yf = fft(series_centered)
    xf = fftfreq(n, d=1)  # d=1 means 1 sample = 15 minutes

    # Get positive frequencies only
    positive_freq_idx = xf > 0
    freqs = xf[positive_freq_idx]
    power = np.abs(yf[positive_freq_idx]) ** 2

    # Convert to periods (in number of 15-min samples)
    periods = 1 / freqs

    return freqs, periods, power


def plot_stl_decomposition(df, stl_result):
    """Plot STL decomposition results."""
    print("\nCreating STL decomposition plot...")

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    # Use subset for plotting (last 2 weeks = 2*7*96 = 1344 points)
    plot_len = min(1344, len(df))
    idx = range(len(df) - plot_len, len(df))
    dates = df['datetime'].iloc[idx]

    # Original
    ax1 = axes[0]
    ax1.plot(dates, df['System Imbalance (MWh)'].iloc[idx], color=COLORS['original'], linewidth=0.8)
    ax1.set_ylabel('Original\n(MWh)')
    ax1.set_title('STL Decomposition (Daily Period = 96)', fontsize=14, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Trend
    ax2 = axes[1]
    ax2.plot(dates, stl_result.trend[idx], color=COLORS['trend'], linewidth=1.5)
    ax2.set_ylabel('Trend\n(MWh)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Seasonal
    ax3 = axes[2]
    ax3.plot(dates, stl_result.seasonal[idx], color=COLORS['seasonal_daily'], linewidth=0.8)
    ax3.set_ylabel('Daily Seasonal\n(MWh)')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Residual
    ax4 = axes[3]
    ax4.plot(dates, stl_result.resid[idx], color=COLORS['residual'], linewidth=0.8)
    ax4.set_ylabel('Residual\n(MWh)')
    ax4.set_xlabel('Date')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_stl_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_stl_decomposition.png")


def plot_mstl_decomposition(df, mstl_result):
    """Plot MSTL decomposition results."""
    print("\nCreating MSTL decomposition plot...")

    fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)

    # Use subset for plotting (last 2 weeks)
    plot_len = min(1344, len(df))
    idx = range(len(df) - plot_len, len(df))
    dates = df['datetime'].iloc[idx]

    # Original
    ax1 = axes[0]
    ax1.plot(dates, df['System Imbalance (MWh)'].iloc[idx], color=COLORS['original'], linewidth=0.8)
    ax1.set_ylabel('Original\n(MWh)')
    ax1.set_title('MSTL Decomposition (Daily=96, Weekly=672)', fontsize=14, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Trend
    ax2 = axes[1]
    ax2.plot(dates, mstl_result.trend[idx], color=COLORS['trend'], linewidth=1.5)
    ax2.set_ylabel('Trend\n(MWh)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Daily Seasonal
    ax3 = axes[2]
    ax3.plot(dates, mstl_result.seasonal[:, 0][idx], color=COLORS['seasonal_daily'], linewidth=0.8)
    ax3.set_ylabel('Daily Seasonal\n(MWh)')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Weekly Seasonal
    ax4 = axes[3]
    ax4.plot(dates, mstl_result.seasonal[:, 1][idx], color=COLORS['seasonal_weekly'], linewidth=0.8)
    ax4.set_ylabel('Weekly Seasonal\n(MWh)')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Residual
    ax5 = axes[4]
    ax5.plot(dates, mstl_result.resid[idx], color=COLORS['residual'], linewidth=0.8)
    ax5.set_ylabel('Residual\n(MWh)')
    ax5.set_xlabel('Date')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_mstl_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_mstl_decomposition.png")


def plot_spectral_analysis(freqs, periods, power):
    """Plot spectral analysis results."""
    print("\nCreating spectral analysis plot...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Power spectrum (frequency domain)
    ax1 = axes[0, 0]
    # Focus on meaningful frequency range (up to 1 cycle per hour = 4 samples)
    freq_mask = (freqs > 0) & (freqs < 0.1)
    ax1.semilogy(freqs[freq_mask], power[freq_mask], color=COLORS['original'], linewidth=0.5)
    ax1.set_xlabel('Frequency (cycles per 15-min period)')
    ax1.set_ylabel('Power (log scale)')
    ax1.set_title('Power Spectrum', fontsize=14, fontweight='bold')

    # Mark key frequencies
    key_periods = {
        'Daily (96)': 1/96,
        '12h (48)': 1/48,
        '6h (24)': 1/24,
        'Weekly (672)': 1/672
    }
    for name, freq in key_periods.items():
        if freq < 0.1:
            ax1.axvline(x=freq, color='red', linestyle='--', alpha=0.5)
            ax1.text(freq, ax1.get_ylim()[1], name, rotation=90, va='top', fontsize=8)

    # Power spectrum (period domain) - zoomed on interesting periods
    ax2 = axes[0, 1]
    period_mask = (periods > 1) & (periods < 800)
    ax2.semilogy(periods[period_mask], power[period_mask], color=COLORS['original'], linewidth=0.5)
    ax2.set_xlabel('Period (number of 15-min intervals)')
    ax2.set_ylabel('Power (log scale)')
    ax2.set_title('Power Spectrum by Period', fontsize=14, fontweight='bold')

    # Mark key periods
    key_periods_vals = [4, 8, 24, 48, 96, 192, 672]  # 1h, 2h, 6h, 12h, 24h, 48h, 1 week
    key_periods_names = ['1h', '2h', '6h', '12h', '24h', '48h', '1wk']
    for p, name in zip(key_periods_vals, key_periods_names):
        ax2.axvline(x=p, color='red', linestyle='--', alpha=0.5)
        ax2.text(p, ax2.get_ylim()[1], name, rotation=90, va='top', fontsize=9)

    # Top frequencies
    ax3 = axes[1, 0]
    # Find top 20 peaks
    peak_idx = signal.find_peaks(power, distance=10)[0]
    peak_powers = power[peak_idx]
    top_20_idx = peak_idx[np.argsort(peak_powers)[-20:]]
    top_periods = periods[top_20_idx]
    top_powers = power[top_20_idx]

    # Sort by period
    sort_idx = np.argsort(top_periods)
    top_periods = top_periods[sort_idx]
    top_powers = top_powers[sort_idx]

    ax3.bar(range(len(top_periods)), top_powers, color=COLORS['original'], alpha=0.7)
    ax3.set_xticks(range(len(top_periods)))
    ax3.set_xticklabels([f'{p:.0f}' for p in top_periods], rotation=45, ha='right')
    ax3.set_xlabel('Period (15-min intervals)')
    ax3.set_ylabel('Power')
    ax3.set_title('Top 20 Dominant Periods', fontsize=14, fontweight='bold')

    # Cumulative power explained
    ax4 = axes[1, 1]
    sorted_power = np.sort(power)[::-1]
    cumsum_power = np.cumsum(sorted_power) / np.sum(power) * 100
    ax4.plot(range(1, len(cumsum_power) + 1), cumsum_power, color=COLORS['original'], linewidth=2)
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(y=90, color='gray', linestyle='--', alpha=0.5)

    # Find how many components for 50% and 90%
    n_50 = np.argmax(cumsum_power >= 50) + 1
    n_90 = np.argmax(cumsum_power >= 90) + 1
    ax4.axvline(x=n_50, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=n_90, color='red', linestyle='--', alpha=0.5)
    ax4.text(n_50, 45, f'{n_50} components\nfor 50%', fontsize=9)
    ax4.text(n_90, 85, f'{n_90} components\nfor 90%', fontsize=9)

    ax4.set_xlabel('Number of Frequency Components')
    ax4.set_ylabel('Cumulative Power Explained (%)')
    ax4.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax4.set_xlim(0, min(1000, len(cumsum_power)))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_spectral_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_spectral_analysis.png")

    return top_periods, top_powers, n_50, n_90


def plot_variance_decomposition(df, stl_result, mstl_result):
    """Plot variance decomposition."""
    print("\nCreating variance decomposition plot...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    original = df['System Imbalance (MWh)'].values
    total_var = np.var(original)

    # STL variance decomposition
    ax1 = axes[0, 0]
    stl_vars = {
        'Trend': np.var(stl_result.trend),
        'Daily Seasonal': np.var(stl_result.seasonal),
        'Residual': np.var(stl_result.resid)
    }
    stl_pcts = {k: v/total_var*100 for k, v in stl_vars.items()}

    colors_stl = [COLORS['trend'], COLORS['seasonal_daily'], COLORS['residual']]
    bars = ax1.bar(stl_pcts.keys(), stl_pcts.values(), color=colors_stl, alpha=0.7, edgecolor='white')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('STL Variance Decomposition', fontsize=14, fontweight='bold')
    for bar, pct in zip(bars, stl_pcts.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{pct:.1f}%',
                ha='center', fontsize=11, fontweight='bold')

    # MSTL variance decomposition
    ax2 = axes[0, 1]
    mstl_vars = {
        'Trend': np.var(mstl_result.trend),
        'Daily Seasonal': np.var(mstl_result.seasonal[:, 0]),
        'Weekly Seasonal': np.var(mstl_result.seasonal[:, 1]),
        'Residual': np.var(mstl_result.resid)
    }
    mstl_pcts = {k: v/total_var*100 for k, v in mstl_vars.items()}

    colors_mstl = [COLORS['trend'], COLORS['seasonal_daily'], COLORS['seasonal_weekly'], COLORS['residual']]
    bars = ax2.bar(mstl_pcts.keys(), mstl_pcts.values(), color=colors_mstl, alpha=0.7, edgecolor='white')
    ax2.set_ylabel('Variance Explained (%)')
    ax2.set_title('MSTL Variance Decomposition', fontsize=14, fontweight='bold')
    for bar, pct in zip(bars, mstl_pcts.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{pct:.1f}%',
                ha='center', fontsize=11, fontweight='bold')

    # Residual distribution comparison
    ax3 = axes[1, 0]
    ax3.hist(stl_result.resid, bins=80, alpha=0.6, color=COLORS['seasonal_daily'],
             label=f'STL Residual (σ={np.std(stl_result.resid):.2f})', density=True)
    ax3.hist(mstl_result.resid, bins=80, alpha=0.6, color=COLORS['seasonal_weekly'],
             label=f'MSTL Residual (σ={np.std(mstl_result.resid):.2f})', density=True)
    ax3.hist(original, bins=80, alpha=0.4, color=COLORS['original'],
             label=f'Original (σ={np.std(original):.2f})', density=True)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Value (MWh)')
    ax3.set_ylabel('Density')
    ax3.set_title('Residual Distribution Comparison', fontsize=14, fontweight='bold')
    ax3.legend()

    # Residual autocorrelation
    ax4 = axes[1, 1]
    from statsmodels.tsa.stattools import acf
    max_lags = 96

    acf_orig = acf(original, nlags=max_lags, fft=True)
    acf_stl = acf(stl_result.resid, nlags=max_lags, fft=True)
    acf_mstl = acf(mstl_result.resid, nlags=max_lags, fft=True)

    ax4.plot(range(max_lags + 1), acf_orig, color=COLORS['original'], linewidth=2, label='Original', alpha=0.5)
    ax4.plot(range(max_lags + 1), acf_stl, color=COLORS['seasonal_daily'], linewidth=2, label='STL Residual')
    ax4.plot(range(max_lags + 1), acf_mstl, color=COLORS['seasonal_weekly'], linewidth=2, label='MSTL Residual')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Lag (15-min periods)')
    ax4.set_ylabel('ACF')
    ax4.set_title('Residual Autocorrelation (lower = better decomposition)', fontsize=14, fontweight='bold')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_variance_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_variance_decomposition.png")

    return stl_pcts, mstl_pcts


def plot_seasonal_patterns(df, stl_result, mstl_result):
    """Plot extracted seasonal patterns."""
    print("\nCreating seasonal patterns plot...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Daily pattern from STL (average over all days)
    ax1 = axes[0, 0]
    seasonal = stl_result.seasonal
    n_days = len(seasonal) // 96
    daily_matrix = seasonal[:n_days*96].reshape(n_days, 96)
    daily_mean = daily_matrix.mean(axis=0)
    daily_std = daily_matrix.std(axis=0)

    hours = np.arange(96) / 4  # Convert to hours
    ax1.plot(hours, daily_mean, color=COLORS['seasonal_daily'], linewidth=2)
    ax1.fill_between(hours, daily_mean - daily_std, daily_mean + daily_std,
                     color=COLORS['seasonal_daily'], alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Seasonal Component (MWh)')
    ax1.set_title('STL Daily Seasonal Pattern (Mean ± Std)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 2))

    # Daily pattern from MSTL
    ax2 = axes[0, 1]
    seasonal_daily = mstl_result.seasonal[:, 0]
    daily_matrix = seasonal_daily[:n_days*96].reshape(n_days, 96)
    daily_mean = daily_matrix.mean(axis=0)
    daily_std = daily_matrix.std(axis=0)

    ax2.plot(hours, daily_mean, color=COLORS['seasonal_daily'], linewidth=2)
    ax2.fill_between(hours, daily_mean - daily_std, daily_mean + daily_std,
                     color=COLORS['seasonal_daily'], alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Seasonal Component (MWh)')
    ax2.set_title('MSTL Daily Seasonal Pattern (Mean ± Std)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 2))

    # Weekly pattern from MSTL
    ax3 = axes[1, 0]
    seasonal_weekly = mstl_result.seasonal[:, 1]
    n_weeks = len(seasonal_weekly) // 672
    if n_weeks > 0:
        weekly_matrix = seasonal_weekly[:n_weeks*672].reshape(n_weeks, 672)
        weekly_mean = weekly_matrix.mean(axis=0)
        weekly_std = weekly_matrix.std(axis=0)

        days = np.arange(672) / 96  # Convert to days
        ax3.plot(days, weekly_mean, color=COLORS['seasonal_weekly'], linewidth=1.5)
        ax3.fill_between(days, weekly_mean - weekly_std, weekly_mean + weekly_std,
                         color=COLORS['seasonal_weekly'], alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Day of Week')
        ax3.set_ylabel('Seasonal Component (MWh)')
        ax3.set_title('MSTL Weekly Seasonal Pattern (Mean ± Std)', fontsize=14, fontweight='bold')
        ax3.set_xlim(0, 7)
        ax3.set_xticks(range(8))
        ax3.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', ''])

        # Add day separators
        for d in range(8):
            ax3.axvline(x=d, color='gray', linestyle=':', alpha=0.3)

    # Combined daily pattern by day of week
    ax4 = axes[1, 1]
    df['hour'] = ((df['Settlement Term'] - 1) * 15) / 60
    df['hour'] = df['hour'].clip(upper=23.75)
    df['day_of_week'] = df['datetime'].dt.dayofweek

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    colors_days = plt.cm.viridis(np.linspace(0, 1, 7))

    for d in range(7):
        day_data = df[df['day_of_week'] == d].groupby('hour')['System Imbalance (MWh)'].mean()
        ax4.plot(day_data.index, day_data.values, color=colors_days[d], linewidth=1.5, label=day_names[d])

    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Mean Imbalance (MWh)')
    ax4.set_title('Original Daily Pattern by Day of Week', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', ncol=2)
    ax4.set_xlim(0, 24)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_seasonal_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_seasonal_patterns.png")


def generate_decomposition_report(df, stl_result, mstl_result, stl_pcts, mstl_pcts,
                                   top_periods, top_powers, n_50, n_90):
    """Generate decomposition analysis report."""
    print("\nGenerating decomposition report...")

    original = df['System Imbalance (MWh)'].values
    report = []

    report.append("=" * 80)
    report.append("TIME SERIES DECOMPOSITION ANALYSIS")
    report.append("=" * 80)

    report.append("\n\n" + "=" * 80)
    report.append("PLOT 01: STL DECOMPOSITION (Daily Period = 96)")
    report.append("=" * 80)
    report.append("\nDecomposes time series into:")
    report.append("  - Trend: Slowly varying component (local average)")
    report.append("  - Daily Seasonal: Repeating 24-hour pattern")
    report.append("  - Residual: What's left (unpredictable component)")
    report.append(f"\nVariance explained:")
    for comp, pct in stl_pcts.items():
        report.append(f"  {comp}: {pct:.1f}%")
    report.append(f"\nResidual statistics:")
    report.append(f"  Mean: {np.mean(stl_result.resid):.4f} MWh")
    report.append(f"  Std: {np.std(stl_result.resid):.2f} MWh")
    report.append(f"  Original std: {np.std(original):.2f} MWh")
    report.append(f"  Std reduction: {(1 - np.std(stl_result.resid)/np.std(original))*100:.1f}%")

    report.append("\n\n" + "=" * 80)
    report.append("PLOT 02: MSTL DECOMPOSITION (Daily=96, Weekly=672)")
    report.append("=" * 80)
    report.append("\nDecomposes time series into:")
    report.append("  - Trend: Slowly varying component")
    report.append("  - Daily Seasonal: Repeating 24-hour pattern")
    report.append("  - Weekly Seasonal: Repeating 7-day pattern")
    report.append("  - Residual: What's left")
    report.append(f"\nVariance explained:")
    for comp, pct in mstl_pcts.items():
        report.append(f"  {comp}: {pct:.1f}%")
    report.append(f"\nResidual statistics:")
    report.append(f"  Mean: {np.mean(mstl_result.resid):.4f} MWh")
    report.append(f"  Std: {np.std(mstl_result.resid):.2f} MWh")
    report.append(f"  Std reduction: {(1 - np.std(mstl_result.resid)/np.std(original))*100:.1f}%")

    report.append("\n\n" + "=" * 80)
    report.append("PLOT 03: SPECTRAL ANALYSIS")
    report.append("=" * 80)
    report.append("\nDominant periods (15-min intervals):")

    # Convert to meaningful units
    period_meanings = {
        4: '1 hour', 8: '2 hours', 12: '3 hours', 24: '6 hours',
        48: '12 hours', 96: '24 hours (daily)', 192: '2 days',
        672: '1 week'
    }

    for i, (period, power) in enumerate(sorted(zip(top_periods, top_powers), key=lambda x: -x[1])[:10]):
        period_int = int(round(period))
        meaning = period_meanings.get(period_int, f'{period_int*15/60:.1f} hours')
        report.append(f"  {i+1}. Period={period_int} ({meaning}), Power={power:.0f}")

    report.append(f"\nCumulative variance:")
    report.append(f"  50% explained by top {n_50} frequency components")
    report.append(f"  90% explained by top {n_90} frequency components")

    report.append("\n\n" + "=" * 80)
    report.append("PLOT 04: VARIANCE DECOMPOSITION")
    report.append("=" * 80)
    report.append("\nCompares how much variance each component explains.")
    report.append("Lower residual variance = better decomposition.")
    report.append(f"\nResidual comparison:")
    report.append(f"  STL residual std: {np.std(stl_result.resid):.2f} MWh")
    report.append(f"  MSTL residual std: {np.std(mstl_result.resid):.2f} MWh")
    report.append(f"  MSTL improvement: {(np.std(stl_result.resid) - np.std(mstl_result.resid)):.2f} MWh")

    report.append("\n\n" + "=" * 80)
    report.append("PLOT 05: SEASONAL PATTERNS")
    report.append("=" * 80)
    report.append("\nExtracted seasonal patterns that repeat regularly:")
    report.append("  - Daily pattern: Shows intraday cycle")
    report.append("  - Weekly pattern: Shows day-of-week effects")

    # Daily pattern stats
    seasonal_daily = mstl_result.seasonal[:, 0]
    n_days = len(seasonal_daily) // 96
    daily_matrix = seasonal_daily[:n_days*96].reshape(n_days, 96)
    daily_mean = daily_matrix.mean(axis=0)

    peak_hour = np.argmax(daily_mean) / 4
    trough_hour = np.argmin(daily_mean) / 4
    report.append(f"\nDaily pattern characteristics:")
    report.append(f"  Peak: {peak_hour:.1f}:00 ({daily_mean.max():.2f} MWh)")
    report.append(f"  Trough: {trough_hour:.1f}:00 ({daily_mean.min():.2f} MWh)")
    report.append(f"  Amplitude: {daily_mean.max() - daily_mean.min():.2f} MWh")

    report.append("\n\n" + "=" * 80)
    report.append("IMPLICATIONS FOR ML NOWCASTING")
    report.append("=" * 80)

    resid_var_pct = mstl_pcts['Residual']
    report.append(f"\n1. PREDICTABLE PORTION: {100-resid_var_pct:.1f}% of variance")
    report.append("   - Can be captured by trend + seasonal features")
    report.append("   - Use HoD, DoW, rolling means as features")

    report.append(f"\n2. UNPREDICTABLE PORTION: {resid_var_pct:.1f}% of variance")
    report.append("   - This is what your model needs to learn")
    report.append("   - Residual std: {:.2f} MWh (vs original {:.2f} MWh)".format(
        np.std(mstl_result.resid), np.std(original)))

    report.append("\n3. RECOMMENDED FEATURES:")
    report.append("   - Hour of day (captures daily seasonal)")
    report.append("   - Day of week (captures weekly seasonal)")
    report.append("   - Rolling mean/std (captures trend + local dynamics)")
    report.append("   - Lag features (residual has some autocorrelation)")

    report.append("\n4. EXPECTED BASELINE PERFORMANCE:")
    report.append(f"   - Naive (use last value): RMSE ~ {np.std(np.diff(original)):.2f} MWh")
    report.append(f"   - Seasonal naive (same hour yesterday): RMSE ~ {np.std(original - np.roll(original, 96)):.2f} MWh")
    report.append(f"   - Perfect decomposition: RMSE ~ {np.std(mstl_result.resid):.2f} MWh")

    report.append("\n\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    report_text = "\n".join(report)
    report_path = OUTPUT_DIR / 'decomposition_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("  Saved: decomposition_report.txt")

    return report_text


def main():
    print("=" * 60)
    print("TIME SERIES DECOMPOSITION ANALYSIS")
    print("=" * 60)

    # Load data
    df = load_data()

    # Perform decompositions
    stl_result = stl_decomposition(df)
    mstl_result = mstl_decomposition(df)

    # Spectral analysis
    freqs, periods, power = spectral_analysis(df)

    # Create plots
    plot_stl_decomposition(df, stl_result)
    plot_mstl_decomposition(df, mstl_result)
    top_periods, top_powers, n_50, n_90 = plot_spectral_analysis(freqs, periods, power)
    stl_pcts, mstl_pcts = plot_variance_decomposition(df, stl_result, mstl_result)
    plot_seasonal_patterns(df, stl_result, mstl_result)

    # Generate report
    generate_decomposition_report(df, stl_result, mstl_result, stl_pcts, mstl_pcts,
                                   top_periods, top_powers, n_50, n_90)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
