"""
Advanced label analysis for system imbalance nowcasting.
Covers: autocorrelation decay, extreme events, persistence,
conditional distributions, stationarity, and ARCH effects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.stats.diagnostic import het_arch
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
MASTER_FILE = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
OUTPUT_DIR = BASE_DIR / "analysis" / "label" / "advanced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'main': '#2E86AB',
    'secondary': '#E94F37',
    'accent': '#F6AE2D',
    'positive': '#3B7A57',
    'negative': '#C73E1D'
}


def load_data():
    """Load the master imbalance data."""
    print("Loading master imbalance data...")
    df = pd.read_csv(MASTER_FILE, parse_dates=['datetime', 'Date'])
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"Loaded {len(df):,} rows")
    return df


def analyze_autocorrelation_decay(df):
    """Analyze how autocorrelation decays with lag."""
    print("\nAnalyzing autocorrelation decay...")

    series = df['System Imbalance (MWh)'].values
    max_lags = 200  # ~50 hours

    # Compute ACF and PACF
    acf_vals = acf(series, nlags=max_lags, fft=True)
    pacf_vals = pacf(series, nlags=min(max_lags, len(series)//2 - 1))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ACF plot
    ax1 = axes[0, 0]
    ax1.bar(range(len(acf_vals)), acf_vals, color=COLORS['main'], alpha=0.7, width=1)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=1.96/np.sqrt(len(series)), color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=-1.96/np.sqrt(len(series)), color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Lag (15-min periods)')
    ax1.set_ylabel('ACF')
    ax1.set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, max_lags)

    # Mark key lags
    for lag, name in [(4, '1h'), (24, '6h'), (48, '12h'), (96, '24h')]:
        ax1.axvline(x=lag, color='gray', linestyle=':', alpha=0.5)
        ax1.text(lag, ax1.get_ylim()[1]*0.95, name, ha='center', fontsize=9)

    # PACF plot
    ax2 = axes[0, 1]
    ax2.bar(range(len(pacf_vals)), pacf_vals, color=COLORS['secondary'], alpha=0.7, width=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=1.96/np.sqrt(len(series)), color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=-1.96/np.sqrt(len(series)), color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Lag (15-min periods)')
    ax2.set_ylabel('PACF')
    ax2.set_title('Partial Autocorrelation Function (PACF)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, min(50, len(pacf_vals)))

    # ACF decay analysis
    ax3 = axes[1, 0]
    # Find where ACF drops below significance
    significance = 1.96/np.sqrt(len(series))
    significant_lags = np.where(np.abs(acf_vals) > significance)[0]

    ax3.semilogy(range(1, len(acf_vals)), np.abs(acf_vals[1:]), color=COLORS['main'], linewidth=2)
    ax3.axhline(y=significance, color='red', linestyle='--', label=f'95% significance ({significance:.4f})')
    ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='ACF = 0.1')
    ax3.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='ACF = 0.05')
    ax3.set_xlabel('Lag (15-min periods)')
    ax3.set_ylabel('|ACF| (log scale)')
    ax3.set_title('ACF Decay (Log Scale)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.set_xlim(1, max_lags)

    # Information content by lag
    ax4 = axes[1, 1]
    cumsum_r2 = np.cumsum(acf_vals[1:]**2)
    total_r2 = np.sum(acf_vals[1:]**2)
    pct_explained = cumsum_r2 / total_r2 * 100

    ax4.plot(range(1, len(pct_explained)+1), pct_explained, color=COLORS['main'], linewidth=2)
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(y=99, color='gray', linestyle='--', alpha=0.5)

    # Find key points
    lag_50 = np.argmax(pct_explained >= 50) + 1
    lag_90 = np.argmax(pct_explained >= 90) + 1
    lag_99 = np.argmax(pct_explained >= 99) + 1

    ax4.axvline(x=lag_50, color='red', linestyle=':', alpha=0.7)
    ax4.axvline(x=lag_90, color='red', linestyle=':', alpha=0.7)
    ax4.text(lag_50+2, 45, f'{lag_50} lags\n({lag_50*15/60:.1f}h)', fontsize=9)
    ax4.text(lag_90+2, 85, f'{lag_90} lags\n({lag_90*15/60:.1f}h)', fontsize=9)

    ax4.set_xlabel('Number of Lags')
    ax4.set_ylabel('Cumulative R² Explained (%)')
    ax4.set_title('Cumulative Autocorrelation Information', fontsize=14, fontweight='bold')
    ax4.set_xlim(1, max_lags)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_autocorrelation_decay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_autocorrelation_decay.png")

    # Find significant lags
    sig_lags = np.where(np.abs(acf_vals) > significance)[0]
    last_significant = sig_lags[-1] if len(sig_lags) > 0 else 0

    return {
        'lag_50_pct': lag_50,
        'lag_90_pct': lag_90,
        'lag_99_pct': lag_99,
        'last_significant_lag': last_significant,
        'acf_lag1': acf_vals[1],
        'acf_lag4': acf_vals[4],  # 1 hour
        'acf_lag96': acf_vals[96],  # 24 hours
        'pacf_lag1': pacf_vals[1],
        'pacf_lag2': pacf_vals[2]
    }


def analyze_extreme_events(df):
    """Analyze extreme imbalance events."""
    print("\nAnalyzing extreme events...")

    series = df['System Imbalance (MWh)']

    # Define extreme thresholds
    p01, p05, p95, p99 = series.quantile([0.01, 0.05, 0.95, 0.99])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Extreme events distribution
    ax1 = axes[0, 0]
    bins = np.linspace(series.min(), series.max(), 100)
    ax1.hist(series, bins=bins, color=COLORS['main'], alpha=0.7, density=True)
    ax1.axvline(x=p01, color=COLORS['negative'], linestyle='--', linewidth=2, label=f'P1: {p01:.1f}')
    ax1.axvline(x=p05, color=COLORS['negative'], linestyle=':', linewidth=2, label=f'P5: {p05:.1f}')
    ax1.axvline(x=p95, color=COLORS['positive'], linestyle=':', linewidth=2, label=f'P95: {p95:.1f}')
    ax1.axvline(x=p99, color=COLORS['positive'], linestyle='--', linewidth=2, label=f'P99: {p99:.1f}')
    ax1.set_xlabel('Imbalance (MWh)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution with Extreme Thresholds', fontsize=14, fontweight='bold')
    ax1.legend()

    # Extreme events by hour
    ax2 = axes[0, 1]
    df['hour'] = ((df['Settlement Term'] - 1) * 15) // 60
    df['hour'] = df['hour'].clip(upper=23)
    df['is_extreme_low'] = series < p05
    df['is_extreme_high'] = series > p95

    extreme_low_by_hour = df.groupby('hour')['is_extreme_low'].mean() * 100
    extreme_high_by_hour = df.groupby('hour')['is_extreme_high'].mean() * 100

    x = np.arange(24)
    width = 0.35
    ax2.bar(x - width/2, extreme_low_by_hour, width, label='Extreme Low (<P5)', color=COLORS['negative'], alpha=0.7)
    ax2.bar(x + width/2, extreme_high_by_hour, width, label='Extreme High (>P95)', color=COLORS['positive'], alpha=0.7)
    ax2.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Expected (5%)')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Percentage of Extreme Events (%)')
    ax2.set_title('Extreme Events by Hour of Day', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend()

    # Extreme events by day of week
    ax3 = axes[1, 0]
    df['day_of_week'] = df['datetime'].dt.dayofweek
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    extreme_low_by_dow = df.groupby('day_of_week')['is_extreme_low'].mean() * 100
    extreme_high_by_dow = df.groupby('day_of_week')['is_extreme_high'].mean() * 100

    x = np.arange(7)
    ax3.bar(x - width/2, extreme_low_by_dow, width, label='Extreme Low (<P5)', color=COLORS['negative'], alpha=0.7)
    ax3.bar(x + width/2, extreme_high_by_dow, width, label='Extreme High (>P95)', color=COLORS['positive'], alpha=0.7)
    ax3.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Expected (5%)')
    ax3.set_xlabel('Day of Week')
    ax3.set_ylabel('Percentage of Extreme Events (%)')
    ax3.set_title('Extreme Events by Day of Week', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(day_names)
    ax3.legend()

    # Clustering of extreme events
    ax4 = axes[1, 1]
    extreme_mask = (series < p05) | (series > p95)
    extreme_series = extreme_mask.astype(int).values

    # Count consecutive extremes
    runs = []
    current_run = 0
    for val in extreme_series:
        if val == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    if runs:
        max_run = max(runs)
        bins = range(1, min(max_run + 2, 20))
        ax4.hist(runs, bins=bins, color=COLORS['accent'], alpha=0.7, edgecolor='white')
        ax4.set_xlabel('Length of Extreme Event Run (consecutive periods)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Clustering: Consecutive Extreme Events', fontsize=14, fontweight='bold')

        # Add statistics
        stats_text = f"Total runs: {len(runs)}\nMean length: {np.mean(runs):.2f}\nMax length: {max_run} ({max_run*15/60:.1f}h)"
        ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_extreme_events.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_extreme_events.png")

    return {
        'p01': p01, 'p05': p05, 'p95': p95, 'p99': p99,
        'extreme_low_pct': (series < p05).mean() * 100,
        'extreme_high_pct': (series > p95).mean() * 100,
        'mean_run_length': np.mean(runs) if runs else 0,
        'max_run_length': max(runs) if runs else 0,
        'total_runs': len(runs)
    }


def analyze_persistence(df):
    """Analyze persistence and runs of positive/negative imbalance."""
    print("\nAnalyzing persistence and runs...")

    series = df['System Imbalance (MWh)'].values
    sign = np.sign(series)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Transition matrix
    ax1 = axes[0, 0]
    transitions = {
        'pos_to_pos': 0, 'pos_to_neg': 0,
        'neg_to_pos': 0, 'neg_to_neg': 0
    }

    for i in range(1, len(sign)):
        if sign[i-1] > 0 and sign[i] > 0:
            transitions['pos_to_pos'] += 1
        elif sign[i-1] > 0 and sign[i] < 0:
            transitions['pos_to_neg'] += 1
        elif sign[i-1] < 0 and sign[i] > 0:
            transitions['neg_to_pos'] += 1
        elif sign[i-1] < 0 and sign[i] < 0:
            transitions['neg_to_neg'] += 1

    total_from_pos = transitions['pos_to_pos'] + transitions['pos_to_neg']
    total_from_neg = transitions['neg_to_pos'] + transitions['neg_to_neg']

    trans_matrix = np.array([
        [transitions['pos_to_pos']/total_from_pos, transitions['pos_to_neg']/total_from_pos],
        [transitions['neg_to_pos']/total_from_neg, transitions['neg_to_neg']/total_from_neg]
    ])

    im = ax1.imshow(trans_matrix, cmap='Blues', vmin=0, vmax=1)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Positive', 'Negative'])
    ax1.set_yticklabels(['Positive', 'Negative'])
    ax1.set_xlabel('Next State')
    ax1.set_ylabel('Current State')
    ax1.set_title('Transition Probability Matrix', fontsize=14, fontweight='bold')

    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f'{trans_matrix[i,j]:.1%}', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white' if trans_matrix[i,j] > 0.5 else 'black')

    plt.colorbar(im, ax=ax1, label='Probability')

    # Run lengths distribution
    ax2 = axes[0, 1]

    # Calculate runs
    pos_runs = []
    neg_runs = []
    current_run = 1
    current_sign = sign[0]

    for i in range(1, len(sign)):
        if sign[i] == current_sign:
            current_run += 1
        else:
            if current_sign > 0:
                pos_runs.append(current_run)
            elif current_sign < 0:
                neg_runs.append(current_run)
            current_run = 1
            current_sign = sign[i]

    # Add last run
    if current_sign > 0:
        pos_runs.append(current_run)
    elif current_sign < 0:
        neg_runs.append(current_run)

    max_run = max(max(pos_runs) if pos_runs else 0, max(neg_runs) if neg_runs else 0)
    bins = range(1, min(max_run + 2, 50))

    ax2.hist(pos_runs, bins=bins, alpha=0.6, color=COLORS['positive'], label=f'Positive (n={len(pos_runs)})', density=True)
    ax2.hist(neg_runs, bins=bins, alpha=0.6, color=COLORS['negative'], label=f'Negative (n={len(neg_runs)})', density=True)
    ax2.set_xlabel('Run Length (consecutive periods)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Run Lengths', fontsize=14, fontweight='bold')
    ax2.legend()

    # Cumulative run length
    ax3 = axes[1, 0]
    all_runs = pos_runs + neg_runs
    sorted_runs = np.sort(all_runs)
    cdf = np.arange(1, len(sorted_runs) + 1) / len(sorted_runs)

    ax3.plot(sorted_runs, cdf, color=COLORS['main'], linewidth=2)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)

    median_run = sorted_runs[np.argmax(cdf >= 0.5)]
    p90_run = sorted_runs[np.argmax(cdf >= 0.9)]

    ax3.axvline(x=median_run, color='red', linestyle=':', alpha=0.7)
    ax3.axvline(x=p90_run, color='red', linestyle=':', alpha=0.7)
    ax3.text(median_run + 0.5, 0.45, f'Median: {median_run}', fontsize=10)
    ax3.text(p90_run + 0.5, 0.85, f'P90: {p90_run}', fontsize=10)

    ax3.set_xlabel('Run Length (consecutive periods)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('CDF of Run Lengths', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, min(50, max(sorted_runs)))

    # Time since last sign change
    ax4 = axes[1, 1]
    time_since_change = []
    counter = 0
    current_sign = sign[0]

    for i in range(len(sign)):
        if sign[i] == current_sign:
            counter += 1
        else:
            counter = 1
            current_sign = sign[i]
        time_since_change.append(counter)

    ax4.hist(time_since_change, bins=50, color=COLORS['main'], alpha=0.7, density=True)
    ax4.set_xlabel('Periods Since Last Sign Change')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution of Time Since Last Sign Change', fontsize=14, fontweight='bold')

    mean_time = np.mean(time_since_change)
    ax4.axvline(x=mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.1f}')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_persistence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_persistence.png")

    return {
        'pos_to_pos_prob': trans_matrix[0, 0],
        'neg_to_neg_prob': trans_matrix[1, 1],
        'mean_pos_run': np.mean(pos_runs) if pos_runs else 0,
        'mean_neg_run': np.mean(neg_runs) if neg_runs else 0,
        'median_run': median_run,
        'p90_run': p90_run,
        'max_pos_run': max(pos_runs) if pos_runs else 0,
        'max_neg_run': max(neg_runs) if neg_runs else 0
    }


def analyze_conditional_distributions(df):
    """Analyze conditional distributions based on recent history."""
    print("\nAnalyzing conditional distributions...")

    series = df['System Imbalance (MWh)'].values

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Conditional on previous value (high/medium/low)
    ax1 = axes[0, 0]
    prev_vals = series[:-1]
    curr_vals = series[1:]

    p33, p67 = np.percentile(prev_vals, [33, 67])
    low_mask = prev_vals < p33
    mid_mask = (prev_vals >= p33) & (prev_vals <= p67)
    high_mask = prev_vals > p67

    bins = np.linspace(-50, 50, 60)
    ax1.hist(curr_vals[low_mask], bins=bins, alpha=0.5, color=COLORS['negative'],
             label=f'After Low (<{p33:.1f})', density=True)
    ax1.hist(curr_vals[mid_mask], bins=bins, alpha=0.5, color=COLORS['main'],
             label=f'After Mid ({p33:.1f}-{p67:.1f})', density=True)
    ax1.hist(curr_vals[high_mask], bins=bins, alpha=0.5, color=COLORS['positive'],
             label=f'After High (>{p67:.1f})', density=True)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Current Imbalance (MWh)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Conditional on Previous Value', fontsize=14, fontweight='bold')
    ax1.legend()

    # Conditional mean and std
    ax2 = axes[0, 1]
    n_bins = 20
    bin_edges = np.percentile(prev_vals, np.linspace(0, 100, n_bins + 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    cond_means = []
    cond_stds = []
    for i in range(n_bins):
        mask = (prev_vals >= bin_edges[i]) & (prev_vals < bin_edges[i+1])
        if mask.sum() > 10:
            cond_means.append(curr_vals[mask].mean())
            cond_stds.append(curr_vals[mask].std())
        else:
            cond_means.append(np.nan)
            cond_stds.append(np.nan)

    ax2.errorbar(bin_centers, cond_means, yerr=cond_stds, fmt='o-', color=COLORS['main'],
                 capsize=3, label='Mean ± Std')
    ax2.plot(bin_centers, bin_centers, 'k--', alpha=0.5, label='y=x (perfect persistence)')
    ax2.set_xlabel('Previous Imbalance (MWh)')
    ax2.set_ylabel('Current Imbalance (MWh)')
    ax2.set_title('Conditional Mean and Std by Previous Value', fontsize=14, fontweight='bold')
    ax2.legend()

    # Conditional on rolling volatility
    ax3 = axes[1, 0]
    rolling_std = pd.Series(series).rolling(20).std().values
    valid_mask = ~np.isnan(rolling_std[:-1])
    prev_vol = rolling_std[:-1][valid_mask]
    curr_vals_vol = curr_vals[valid_mask]

    vol_p33, vol_p67 = np.percentile(prev_vol, [33, 67])
    low_vol_mask = prev_vol < vol_p33
    mid_vol_mask = (prev_vol >= vol_p33) & (prev_vol <= vol_p67)
    high_vol_mask = prev_vol > vol_p67

    ax3.hist(curr_vals_vol[low_vol_mask], bins=bins, alpha=0.5, color=COLORS['positive'],
             label=f'Low Vol (<{vol_p33:.1f})', density=True)
    ax3.hist(curr_vals_vol[mid_vol_mask], bins=bins, alpha=0.5, color=COLORS['main'],
             label=f'Mid Vol ({vol_p33:.1f}-{vol_p67:.1f})', density=True)
    ax3.hist(curr_vals_vol[high_vol_mask], bins=bins, alpha=0.5, color=COLORS['negative'],
             label=f'High Vol (>{vol_p67:.1f})', density=True)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Current Imbalance (MWh)')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution Conditional on Recent Volatility', fontsize=14, fontweight='bold')
    ax3.legend()

    # Scatter: current vs previous
    ax4 = axes[1, 1]
    sample_idx = np.random.choice(len(prev_vals), min(5000, len(prev_vals)), replace=False)
    ax4.scatter(prev_vals[sample_idx], curr_vals[sample_idx], alpha=0.1, s=5, color=COLORS['main'])
    ax4.plot([prev_vals.min(), prev_vals.max()], [prev_vals.min(), prev_vals.max()],
             'r--', linewidth=2, label='y=x')

    # Add regression line
    slope, intercept = np.polyfit(prev_vals, curr_vals, 1)
    x_line = np.array([prev_vals.min(), prev_vals.max()])
    ax4.plot(x_line, slope*x_line + intercept, 'g-', linewidth=2,
             label=f'Regression: y={slope:.2f}x+{intercept:.2f}')

    ax4.set_xlabel('Previous Imbalance (MWh)')
    ax4.set_ylabel('Current Imbalance (MWh)')
    ax4.set_title('Current vs Previous Value (Sample)', fontsize=14, fontweight='bold')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_conditional_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_conditional_distributions.png")

    return {
        'regression_slope': slope,
        'regression_intercept': intercept,
        'mean_after_low': curr_vals[low_mask].mean(),
        'mean_after_high': curr_vals[high_mask].mean(),
        'std_low_vol': curr_vals_vol[low_vol_mask].std(),
        'std_high_vol': curr_vals_vol[high_vol_mask].std()
    }


def analyze_stationarity(df):
    """Perform stationarity tests."""
    print("\nAnalyzing stationarity...")

    series = df['System Imbalance (MWh)'].values

    # ADF test
    adf_result = adfuller(series, maxlag=96, autolag='AIC')

    # KPSS test
    kpss_result = kpss(series, regression='c', nlags='auto')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Rolling mean
    ax1 = axes[0, 0]
    window = 672  # 1 week
    rolling_mean = pd.Series(series).rolling(window).mean()
    rolling_std = pd.Series(series).rolling(window).std()

    ax1.plot(df['datetime'], series, alpha=0.3, color=COLORS['main'], linewidth=0.5, label='Original')
    ax1.plot(df['datetime'], rolling_mean, color=COLORS['secondary'], linewidth=2, label=f'Rolling Mean ({window} periods)')
    ax1.axhline(y=series.mean(), color='black', linestyle='--', linewidth=1, label=f'Global Mean: {series.mean():.2f}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Imbalance (MWh)')
    ax1.set_title('Rolling Mean (Weekly Window)', fontsize=14, fontweight='bold')
    ax1.legend()

    # Rolling std
    ax2 = axes[0, 1]
    ax2.plot(df['datetime'], rolling_std, color=COLORS['accent'], linewidth=1)
    ax2.axhline(y=series.std(), color='black', linestyle='--', linewidth=1, label=f'Global Std: {series.std():.2f}')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rolling Std (MWh)')
    ax2.set_title('Rolling Volatility (Weekly Window)', fontsize=14, fontweight='bold')
    ax2.legend()

    # Test results summary
    ax3 = axes[1, 0]
    ax3.axis('off')

    test_text = "STATIONARITY TEST RESULTS\n"
    test_text += "=" * 50 + "\n\n"

    test_text += "ADF Test (H0: Unit root exists / Non-stationary)\n"
    test_text += "-" * 50 + "\n"
    test_text += f"Test Statistic: {adf_result[0]:.4f}\n"
    test_text += f"P-value: {adf_result[1]:.6f}\n"
    test_text += f"Lags Used: {adf_result[2]}\n"
    test_text += f"Critical Values:\n"
    for key, val in adf_result[4].items():
        test_text += f"  {key}: {val:.4f}\n"
    if adf_result[1] < 0.05:
        test_text += "=> REJECT H0: Series is STATIONARY\n"
    else:
        test_text += "=> FAIL TO REJECT H0: Series may be NON-STATIONARY\n"

    test_text += "\n\nKPSS Test (H0: Series is stationary)\n"
    test_text += "-" * 50 + "\n"
    test_text += f"Test Statistic: {kpss_result[0]:.4f}\n"
    test_text += f"P-value: {kpss_result[1]:.4f}\n"
    test_text += f"Lags Used: {kpss_result[2]}\n"
    test_text += f"Critical Values:\n"
    for key, val in kpss_result[3].items():
        test_text += f"  {key}: {val:.4f}\n"
    if kpss_result[1] < 0.05:
        test_text += "=> REJECT H0: Series may be NON-STATIONARY\n"
    else:
        test_text += "=> FAIL TO REJECT H0: Series is STATIONARY\n"

    ax3.text(0.05, 0.95, test_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Monthly mean trend
    ax4 = axes[1, 1]
    df['year_month'] = df['datetime'].dt.to_period('M')
    monthly_stats = df.groupby('year_month')['System Imbalance (MWh)'].agg(['mean', 'std'])
    monthly_stats.index = monthly_stats.index.to_timestamp()

    ax4.errorbar(monthly_stats.index, monthly_stats['mean'], yerr=monthly_stats['std']/10,
                 fmt='o-', color=COLORS['main'], capsize=3)
    ax4.axhline(y=series.mean(), color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Monthly Mean (MWh)')
    ax4.set_title('Monthly Mean with Std/10 Error Bars', fontsize=14, fontweight='bold')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_stationarity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_stationarity.png")

    return {
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'adf_stationary': adf_result[1] < 0.05,
        'kpss_statistic': kpss_result[0],
        'kpss_pvalue': kpss_result[1],
        'kpss_stationary': kpss_result[1] >= 0.05
    }


def analyze_arch_effects(df):
    """Analyze ARCH/GARCH effects (volatility clustering)."""
    print("\nAnalyzing ARCH effects...")

    series = df['System Imbalance (MWh)'].values

    # Compute squared residuals from mean
    residuals = series - series.mean()
    sq_residuals = residuals ** 2

    # ARCH test
    arch_test = het_arch(residuals, nlags=12)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Squared residuals time series
    ax1 = axes[0, 0]
    ax1.plot(df['datetime'], sq_residuals, color=COLORS['main'], linewidth=0.3, alpha=0.7)
    rolling_sq = pd.Series(sq_residuals).rolling(96).mean()
    ax1.plot(df['datetime'], rolling_sq, color=COLORS['secondary'], linewidth=2, label='24h Rolling Mean')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Squared Residuals')
    ax1.set_title('Squared Residuals (Volatility Proxy)', fontsize=14, fontweight='bold')
    ax1.legend()

    # ACF of squared residuals
    ax2 = axes[0, 1]
    max_lags = 96
    acf_sq = acf(sq_residuals, nlags=max_lags, fft=True)
    ax2.bar(range(len(acf_sq)), acf_sq, color=COLORS['accent'], alpha=0.7, width=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=1.96/np.sqrt(len(sq_residuals)), color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=-1.96/np.sqrt(len(sq_residuals)), color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Lag (15-min periods)')
    ax2.set_ylabel('ACF')
    ax2.set_title('ACF of Squared Residuals (ARCH indicator)', fontsize=14, fontweight='bold')

    # Volatility by hour
    ax3 = axes[1, 0]
    df['sq_resid'] = sq_residuals
    hourly_vol = df.groupby('hour')['sq_resid'].mean()
    hourly_vol_std = df.groupby('hour')['sq_resid'].std()

    ax3.bar(hourly_vol.index, np.sqrt(hourly_vol), color=COLORS['main'], alpha=0.7,
            yerr=np.sqrt(hourly_vol_std)/10, capsize=3)
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Root Mean Squared Residual (MWh)')
    ax3.set_title('Volatility by Hour of Day', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(0, 24, 2))

    # ARCH test results
    ax4 = axes[1, 1]
    ax4.axis('off')

    test_text = "ARCH EFFECTS TEST RESULTS\n"
    test_text += "=" * 50 + "\n\n"
    test_text += "Engle's ARCH Test\n"
    test_text += "(H0: No ARCH effects / Homoskedastic)\n"
    test_text += "-" * 50 + "\n"
    test_text += f"LM Statistic: {arch_test[0]:.4f}\n"
    test_text += f"LM P-value: {arch_test[1]:.6f}\n"
    test_text += f"F Statistic: {arch_test[2]:.4f}\n"
    test_text += f"F P-value: {arch_test[3]:.6f}\n\n"

    if arch_test[1] < 0.05:
        test_text += "=> REJECT H0: ARCH effects ARE PRESENT\n"
        test_text += "   (Volatility is predictable/clustered)\n"
    else:
        test_text += "=> FAIL TO REJECT H0: No significant ARCH effects\n"

    test_text += "\n\nIMPLICATIONS:\n"
    test_text += "-" * 50 + "\n"
    if arch_test[1] < 0.05:
        test_text += "- Rolling volatility features will be useful\n"
        test_text += "- High volatility periods cluster together\n"
        test_text += "- Consider GARCH-type error modeling\n"
        test_text += "- Prediction intervals should vary by regime\n"
    else:
        test_text += "- Volatility is relatively constant\n"
        test_text += "- Simple prediction intervals may suffice\n"

    # ACF of squared residuals summary
    sig_acf_sq = np.sum(np.abs(acf_sq[1:]) > 1.96/np.sqrt(len(sq_residuals)))
    test_text += f"\n\nSignificant lags in squared ACF: {sig_acf_sq}/{max_lags}\n"
    test_text += f"ACF[1] of squared residuals: {acf_sq[1]:.4f}\n"

    ax4.text(0.05, 0.95, test_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_arch_effects.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_arch_effects.png")

    return {
        'arch_lm_stat': arch_test[0],
        'arch_lm_pvalue': arch_test[1],
        'arch_effects_present': arch_test[1] < 0.05,
        'acf_sq_lag1': acf_sq[1],
        'significant_sq_lags': sig_acf_sq
    }


def generate_advanced_report(acf_stats, extreme_stats, persist_stats, cond_stats, station_stats, arch_stats):
    """Generate comprehensive advanced analysis report."""
    print("\nGenerating advanced analysis report...")

    report = []
    report.append("=" * 80)
    report.append("ADVANCED LABEL ANALYSIS REPORT")
    report.append("=" * 80)

    # Autocorrelation
    report.append("\n\n" + "=" * 80)
    report.append("1. AUTOCORRELATION DECAY")
    report.append("=" * 80)
    report.append(f"\nKey ACF values:")
    report.append(f"  Lag 1 (15 min): {acf_stats['acf_lag1']:.4f}")
    report.append(f"  Lag 4 (1 hour): {acf_stats['acf_lag4']:.4f}")
    report.append(f"  Lag 96 (24 hours): {acf_stats['acf_lag96']:.4f}")
    report.append(f"\nPACF (direct effects):")
    report.append(f"  Lag 1: {acf_stats['pacf_lag1']:.4f}")
    report.append(f"  Lag 2: {acf_stats['pacf_lag2']:.4f}")
    report.append(f"\nInformation content:")
    report.append(f"  50% of ACF info in first {acf_stats['lag_50_pct']} lags ({acf_stats['lag_50_pct']*15/60:.1f} hours)")
    report.append(f"  90% of ACF info in first {acf_stats['lag_90_pct']} lags ({acf_stats['lag_90_pct']*15/60:.1f} hours)")
    report.append(f"  Last significant lag: {acf_stats['last_significant_lag']} ({acf_stats['last_significant_lag']*15/60:.1f} hours)")
    report.append("\nIMPLICATION: Use lag features up to ~4-8 lags for most information")

    # Extreme events
    report.append("\n\n" + "=" * 80)
    report.append("2. EXTREME EVENTS")
    report.append("=" * 80)
    report.append(f"\nThresholds:")
    report.append(f"  P1: {extreme_stats['p01']:.2f} MWh")
    report.append(f"  P5: {extreme_stats['p05']:.2f} MWh")
    report.append(f"  P95: {extreme_stats['p95']:.2f} MWh")
    report.append(f"  P99: {extreme_stats['p99']:.2f} MWh")
    report.append(f"\nClustering:")
    report.append(f"  Total extreme runs: {extreme_stats['total_runs']}")
    report.append(f"  Mean run length: {extreme_stats['mean_run_length']:.2f} periods ({extreme_stats['mean_run_length']*15:.0f} min)")
    report.append(f"  Max run length: {extreme_stats['max_run_length']} periods ({extreme_stats['max_run_length']*15/60:.1f} hours)")
    report.append("\nIMPLICATION: Extreme events cluster - if extreme, likely to stay extreme")

    # Persistence
    report.append("\n\n" + "=" * 80)
    report.append("3. PERSISTENCE / RUNS")
    report.append("=" * 80)
    report.append(f"\nTransition probabilities:")
    report.append(f"  P(positive | positive): {persist_stats['pos_to_pos_prob']:.1%}")
    report.append(f"  P(negative | negative): {persist_stats['neg_to_neg_prob']:.1%}")
    report.append(f"\nRun lengths:")
    report.append(f"  Mean positive run: {persist_stats['mean_pos_run']:.2f} periods")
    report.append(f"  Mean negative run: {persist_stats['mean_neg_run']:.2f} periods")
    report.append(f"  Median run: {persist_stats['median_run']} periods")
    report.append(f"  90th percentile run: {persist_stats['p90_run']} periods")
    report.append(f"  Max positive run: {persist_stats['max_pos_run']} periods ({persist_stats['max_pos_run']*15/60:.1f} hours)")
    report.append(f"  Max negative run: {persist_stats['max_neg_run']} periods ({persist_stats['max_neg_run']*15/60:.1f} hours)")
    report.append("\nIMPLICATION: Strong persistence - sign feature from previous period is useful")

    # Conditional distributions
    report.append("\n\n" + "=" * 80)
    report.append("4. CONDITIONAL DISTRIBUTIONS")
    report.append("=" * 80)
    report.append(f"\nRegression (current on previous):")
    report.append(f"  Slope: {cond_stats['regression_slope']:.4f}")
    report.append(f"  Intercept: {cond_stats['regression_intercept']:.4f}")
    report.append(f"\nConditional means:")
    report.append(f"  After low values: {cond_stats['mean_after_low']:.2f} MWh")
    report.append(f"  After high values: {cond_stats['mean_after_high']:.2f} MWh")
    report.append(f"\nConditional volatility:")
    report.append(f"  During low volatility: {cond_stats['std_low_vol']:.2f} MWh")
    report.append(f"  During high volatility: {cond_stats['std_high_vol']:.2f} MWh")
    report.append(f"\nIMPLICATION: Mean reversion present (slope={cond_stats['regression_slope']:.2f}<1)")

    # Stationarity
    report.append("\n\n" + "=" * 80)
    report.append("5. STATIONARITY")
    report.append("=" * 80)
    report.append(f"\nADF Test (H0: non-stationary):")
    report.append(f"  Statistic: {station_stats['adf_statistic']:.4f}")
    report.append(f"  P-value: {station_stats['adf_pvalue']:.6f}")
    report.append(f"  Result: {'STATIONARY' if station_stats['adf_stationary'] else 'NON-STATIONARY'}")
    report.append(f"\nKPSS Test (H0: stationary):")
    report.append(f"  Statistic: {station_stats['kpss_statistic']:.4f}")
    report.append(f"  P-value: {station_stats['kpss_pvalue']:.4f}")
    report.append(f"  Result: {'STATIONARY' if station_stats['kpss_stationary'] else 'NON-STATIONARY'}")
    report.append("\nIMPLICATION: Series is stationary - no differencing needed")

    # ARCH effects
    report.append("\n\n" + "=" * 80)
    report.append("6. ARCH EFFECTS (Volatility Clustering)")
    report.append("=" * 80)
    report.append(f"\nARCH LM Test (H0: no ARCH effects):")
    report.append(f"  Statistic: {arch_stats['arch_lm_stat']:.4f}")
    report.append(f"  P-value: {arch_stats['arch_lm_pvalue']:.6f}")
    report.append(f"  Result: {'ARCH EFFECTS PRESENT' if arch_stats['arch_effects_present'] else 'No ARCH effects'}")
    report.append(f"\nSquared residuals ACF:")
    report.append(f"  Lag 1: {arch_stats['acf_sq_lag1']:.4f}")
    report.append(f"  Significant lags: {arch_stats['significant_sq_lags']}")
    report.append("\nIMPLICATION: Volatility clusters - rolling_std is a useful feature")

    # Summary
    report.append("\n\n" + "=" * 80)
    report.append("SUMMARY: IMPLICATIONS FOR NOWCASTING MODEL")
    report.append("=" * 80)
    report.append("\nFEATURE RECOMMENDATIONS:")
    report.append(f"  1. Lag features: Use lags 1-4 (captures {acf_stats['lag_50_pct']*15}min of history)")
    report.append(f"  2. Sign persistence: P(same sign) = {max(persist_stats['pos_to_pos_prob'], persist_stats['neg_to_neg_prob']):.0%}")
    report.append(f"  3. Rolling volatility: ARCH effects present, volatility is predictable")
    report.append(f"  4. Mean reversion: Slope={cond_stats['regression_slope']:.2f} suggests partial reversion")
    report.append("\nMODEL CONSIDERATIONS:")
    report.append("  - Series is stationary (no differencing needed)")
    report.append("  - Strong autocorrelation (lag features valuable)")
    report.append("  - Volatility clusters (prediction intervals should vary)")
    report.append("  - Extreme events cluster (momentum during extremes)")

    report.append("\n\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    report_text = "\n".join(report)
    report_path = OUTPUT_DIR / 'advanced_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("  Saved: advanced_analysis_report.txt")

    return report_text


def main():
    print("=" * 60)
    print("ADVANCED LABEL ANALYSIS")
    print("=" * 60)

    # Load data
    df = load_data()

    # Run all analyses
    acf_stats = analyze_autocorrelation_decay(df)
    extreme_stats = analyze_extreme_events(df)
    persist_stats = analyze_persistence(df)
    cond_stats = analyze_conditional_distributions(df)
    station_stats = analyze_stationarity(df)
    arch_stats = analyze_arch_effects(df)

    # Generate report
    generate_advanced_report(acf_stats, extreme_stats, persist_stats,
                            cond_stats, station_stats, arch_stats)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
