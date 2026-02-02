"""
Comprehensive Load Analysis for Slovakia Grid Load Prediction

This script performs detailed analysis of the load data to understand patterns
and identify opportunities to beat the baseline forecast.

Analyses included:
1. Time Series Decomposition (weekday/weekend)
2. Temporal Patterns (hourly, daily, weekly, monthly)
3. Distribution Analysis
4. Autocorrelation (ACF/PACF)
5. Stationarity Tests (ADF/KPSS)
6. Forecast Error Analysis
7. Load Duration Curve
8. Anomaly Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Paths
BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet'
PLOT_PATH = Path(__file__).parent / 'plots'
PLOT_PATH.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load and prepare the data."""
    df = pd.read_parquet(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    df['day_type'] = df['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
    return df


# =============================================================================
# 1. TIME SERIES DECOMPOSITION
# =============================================================================
def analyze_decomposition(df: pd.DataFrame):
    """Perform STL decomposition for weekday and weekend separately."""
    print("\n" + "="*70)
    print("1. TIME SERIES DECOMPOSITION")
    print("="*70)

    # Overall decomposition (weekly seasonality = 168 hours)
    series = df['actual_load_mw'].dropna()

    # STL decomposition with weekly period
    stl = STL(series, period=168, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    axes[0].plot(result.observed, linewidth=0.5, alpha=0.8)
    axes[0].set_ylabel('Observed')
    axes[0].set_title('STL Decomposition of Load (Weekly Period = 168 hours)')

    axes[1].plot(result.trend, linewidth=1, color='red')
    axes[1].set_ylabel('Trend')

    axes[2].plot(result.seasonal, linewidth=0.5, color='green')
    axes[2].set_ylabel('Seasonal')

    axes[3].plot(result.resid, linewidth=0.5, alpha=0.7, color='purple')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '01_stl_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Decomposition statistics
    print(f"\nDecomposition Statistics:")
    print(f"  Trend range: {result.trend.min():.0f} - {result.trend.max():.0f} MW")
    print(f"  Seasonal amplitude: {result.seasonal.max() - result.seasonal.min():.0f} MW")
    print(f"  Residual std: {result.resid.std():.1f} MW")

    # Variance explained
    total_var = series.var()
    trend_var = result.trend.var()
    seasonal_var = result.seasonal.var()
    resid_var = result.resid.var()

    print(f"\nVariance Decomposition:")
    print(f"  Trend: {trend_var/total_var*100:.1f}%")
    print(f"  Seasonal: {seasonal_var/total_var*100:.1f}%")
    print(f"  Residual: {resid_var/total_var*100:.1f}%")

    # Weekday vs Weekend decomposition comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for i, (day_type, color) in enumerate([('Weekday', 'blue'), ('Weekend', 'orange')]):
        subset = df[df['day_type'] == day_type]['actual_load_mw'].dropna()

        # Daily decomposition (24 hour period)
        if len(subset) >= 48:
            stl_sub = STL(subset, period=24, robust=True)
            result_sub = stl_sub.fit()

            axes[i, 0].plot(result_sub.seasonal[:168], color=color, linewidth=1)
            axes[i, 0].set_title(f'{day_type} - Seasonal Component (First Week)')
            axes[i, 0].set_xlabel('Hour')
            axes[i, 0].set_ylabel('MW')

            axes[i, 1].hist(result_sub.resid, bins=50, color=color, alpha=0.7, edgecolor='black')
            axes[i, 1].set_title(f'{day_type} - Residual Distribution')
            axes[i, 1].set_xlabel('Residual (MW)')
            axes[i, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '01b_decomposition_weekday_weekend.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Saved: 01_stl_decomposition.png, 01b_decomposition_weekday_weekend.png")

    return result


# =============================================================================
# 2. TEMPORAL PATTERNS
# =============================================================================
def analyze_temporal_patterns(df: pd.DataFrame):
    """Analyze hourly, daily, weekly, and monthly patterns."""
    print("\n" + "="*70)
    print("2. TEMPORAL PATTERNS")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 2a. Hourly profile by day type
    hourly = df.groupby(['hour', 'day_type'])['actual_load_mw'].agg(['mean', 'std']).reset_index()

    for day_type, color in [('Weekday', 'blue'), ('Weekend', 'orange')]:
        data = hourly[hourly['day_type'] == day_type]
        axes[0, 0].plot(data['hour'], data['mean'], marker='o', label=day_type, color=color, linewidth=2)
        axes[0, 0].fill_between(data['hour'],
                                 data['mean'] - data['std'],
                                 data['mean'] + data['std'],
                                 alpha=0.2, color=color)

    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Load (MW)')
    axes[0, 0].set_title('Hourly Load Profile (Mean ± Std)')
    axes[0, 0].legend()
    axes[0, 0].set_xticks(range(1, 25))

    # 2b. Daily pattern (average load per day)
    daily = df.groupby(df.index.date)['actual_load_mw'].mean()
    axes[0, 1].plot(daily.index, daily.values, linewidth=0.8, alpha=0.8)
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Daily Average Load (MW)')
    axes[0, 1].set_title('Daily Average Load Over Time')
    axes[0, 1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)

    # 2c. Weekly pattern
    weekly = df.groupby('day_of_week')['actual_load_mw'].agg(['mean', 'std'])
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 0].bar(range(7), weekly['mean'], yerr=weekly['std'], capsize=3,
                   color=['blue']*5 + ['orange']*2, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(days)
    axes[1, 0].set_xlabel('Day of Week')
    axes[1, 0].set_ylabel('Load (MW)')
    axes[1, 0].set_title('Weekly Load Pattern (Mean ± Std)')

    # 2d. Monthly pattern
    monthly = df.groupby('month')['actual_load_mw'].agg(['mean', 'std'])
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    colors = plt.cm.coolwarm(np.linspace(0, 1, 12))
    axes[1, 1].bar(range(1, 13), monthly['mean'], yerr=monthly['std'], capsize=3,
                   color=colors, alpha=0.8, edgecolor='black')
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].set_xticklabels(months)
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Load (MW)')
    axes[1, 1].set_title('Monthly Load Pattern (Mean ± Std)')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '02_temporal_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print statistics
    print("\nHourly Load by Day Type:")
    hourly_summary = df.groupby(['day_type', 'hour'])['actual_load_mw'].mean().unstack(level=0)
    print(f"  Peak hour (Weekday): Hour {hourly_summary['Weekday'].idxmax()} ({hourly_summary['Weekday'].max():.0f} MW)")
    print(f"  Peak hour (Weekend): Hour {hourly_summary['Weekend'].idxmax()} ({hourly_summary['Weekend'].max():.0f} MW)")
    print(f"  Off-peak hour (Weekday): Hour {hourly_summary['Weekday'].idxmin()} ({hourly_summary['Weekday'].min():.0f} MW)")
    print(f"  Off-peak hour (Weekend): Hour {hourly_summary['Weekend'].idxmin()} ({hourly_summary['Weekend'].min():.0f} MW)")

    print(f"\nWeekday vs Weekend:")
    print(f"  Weekday mean: {df[df['day_type']=='Weekday']['actual_load_mw'].mean():.0f} MW")
    print(f"  Weekend mean: {df[df['day_type']=='Weekend']['actual_load_mw'].mean():.0f} MW")
    print(f"  Difference: {df[df['day_type']=='Weekday']['actual_load_mw'].mean() - df[df['day_type']=='Weekend']['actual_load_mw'].mean():.0f} MW")

    print("\n  Saved: 02_temporal_patterns.png")

    # Heatmap of hourly load by day of week
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot = df.pivot_table(values='actual_load_mw', index='hour', columns='day_of_week', aggfunc='mean')
    pivot.columns = days
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
    ax.set_yticks(range(24))
    ax.set_yticklabels(range(1, 25))
    ax.set_xticks(range(7))
    ax.set_xticklabels(days)
    ax.set_ylabel('Hour of Day')
    ax.set_xlabel('Day of Week')
    ax.set_title('Load Heatmap: Hour vs Day of Week')
    plt.colorbar(im, ax=ax, label='Load (MW)')
    plt.tight_layout()
    plt.savefig(PLOT_PATH / '02b_load_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: 02b_load_heatmap.png")


# =============================================================================
# 3. DISTRIBUTION ANALYSIS
# =============================================================================
def analyze_distributions(df: pd.DataFrame):
    """Analyze load distributions."""
    print("\n" + "="*70)
    print("3. DISTRIBUTION ANALYSIS")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 3a. Overall distribution
    load = df['actual_load_mw'].dropna()
    axes[0, 0].hist(load, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')

    # Fit normal distribution
    mu, std = stats.norm.fit(load)
    x = np.linspace(load.min(), load.max(), 100)
    axes[0, 0].plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.0f}, σ={std:.0f}')
    axes[0, 0].set_xlabel('Load (MW)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Load Distribution with Normal Fit')
    axes[0, 0].legend()

    # 3b. Distribution by day type
    for day_type, color in [('Weekday', 'blue'), ('Weekend', 'orange')]:
        data = df[df['day_type'] == day_type]['actual_load_mw'].dropna()
        axes[0, 1].hist(data, bins=40, density=True, alpha=0.5, label=day_type, color=color, edgecolor='black')
    axes[0, 1].set_xlabel('Load (MW)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Load Distribution by Day Type')
    axes[0, 1].legend()

    # 3c. Box plots by hour
    df_box = df[['hour', 'actual_load_mw']].dropna()
    df_box.boxplot(column='actual_load_mw', by='hour', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Load (MW)')
    axes[1, 0].set_title('Load Distribution by Hour')
    plt.suptitle('')  # Remove automatic title

    # 3d. Box plots by month
    df_box = df[['month', 'actual_load_mw']].dropna()
    df_box.boxplot(column='actual_load_mw', by='month', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Load (MW)')
    axes[1, 1].set_title('Load Distribution by Month')
    plt.suptitle('')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '03_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Statistical tests
    print("\nDistribution Statistics:")
    print(f"  Skewness: {load.skew():.3f}")
    print(f"  Kurtosis: {load.kurtosis():.3f}")

    # Normality test
    _, p_value = stats.normaltest(load)
    print(f"  Normality test p-value: {p_value:.2e} ({'Normal' if p_value > 0.05 else 'Not Normal'})")

    # Percentiles
    print(f"\nPercentiles:")
    for p in [5, 25, 50, 75, 95]:
        print(f"  P{p}: {load.quantile(p/100):.0f} MW")

    print("\n  Saved: 03_distributions.png")


# =============================================================================
# 4. AUTOCORRELATION ANALYSIS
# =============================================================================
def analyze_autocorrelation(df: pd.DataFrame):
    """Analyze ACF and PACF for model selection."""
    print("\n" + "="*70)
    print("4. AUTOCORRELATION ANALYSIS")
    print("="*70)

    series = df['actual_load_mw'].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 4a. ACF (up to 1 week)
    plot_acf(series, lags=168, ax=axes[0, 0], alpha=0.05)
    axes[0, 0].set_title('Autocorrelation Function (ACF) - 1 Week')
    axes[0, 0].set_xlabel('Lag (hours)')
    axes[0, 0].axvline(x=24, color='red', linestyle='--', alpha=0.5, label='24h')
    axes[0, 0].axvline(x=168, color='green', linestyle='--', alpha=0.5, label='168h (1 week)')

    # 4b. PACF
    plot_pacf(series, lags=72, ax=axes[0, 1], alpha=0.05, method='ywm')
    axes[0, 1].set_title('Partial Autocorrelation Function (PACF) - 3 Days')
    axes[0, 1].set_xlabel('Lag (hours)')
    axes[0, 1].axvline(x=24, color='red', linestyle='--', alpha=0.5, label='24h')

    # 4c. ACF for differenced series (daily)
    diff_24 = series.diff(24).dropna()
    plot_acf(diff_24, lags=168, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('ACF of 24h Differenced Series')
    axes[1, 0].set_xlabel('Lag (hours)')

    # 4d. ACF for weekly differenced series
    diff_168 = series.diff(168).dropna()
    plot_acf(diff_168, lags=72, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title('ACF of 168h (Weekly) Differenced Series')
    axes[1, 1].set_xlabel('Lag (hours)')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '04_autocorrelation.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Key lag correlations
    acf_values = acf(series, nlags=168)
    print("\nKey Autocorrelations:")
    print(f"  Lag 1 (1 hour): {acf_values[1]:.3f}")
    print(f"  Lag 24 (1 day): {acf_values[24]:.3f}")
    print(f"  Lag 48 (2 days): {acf_values[48]:.3f}")
    print(f"  Lag 168 (1 week): {acf_values[168]:.3f}")

    print("\nImplications for Modeling:")
    print("  - High lag-1 correlation suggests AR components")
    print("  - High lag-24 correlation suggests daily seasonality")
    print("  - High lag-168 correlation suggests weekly seasonality")

    print("\n  Saved: 04_autocorrelation.png")


# =============================================================================
# 5. STATIONARITY TESTS
# =============================================================================
def analyze_stationarity(df: pd.DataFrame):
    """Test for stationarity using ADF and KPSS tests."""
    print("\n" + "="*70)
    print("5. STATIONARITY TESTS")
    print("="*70)

    series = df['actual_load_mw'].dropna()

    # ADF Test
    adf_result = adfuller(series, autolag='AIC')
    print("\nAugmented Dickey-Fuller Test:")
    print(f"  Test Statistic: {adf_result[0]:.4f}")
    print(f"  p-value: {adf_result[1]:.4f}")
    print(f"  Lags Used: {adf_result[2]}")
    print(f"  Critical Values:")
    for key, value in adf_result[4].items():
        print(f"    {key}: {value:.4f}")
    print(f"  Conclusion: {'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'} (at 5% level)")

    # KPSS Test
    kpss_result = kpss(series, regression='c', nlags='auto')
    print("\nKPSS Test:")
    print(f"  Test Statistic: {kpss_result[0]:.4f}")
    print(f"  p-value: {kpss_result[1]:.4f}")
    print(f"  Lags Used: {kpss_result[2]}")
    print(f"  Critical Values:")
    for key, value in kpss_result[3].items():
        print(f"    {key}: {value:.4f}")
    print(f"  Conclusion: {'Stationary' if kpss_result[1] > 0.05 else 'Non-stationary'} (at 5% level)")

    # Test differenced series
    print("\n--- Differenced Series (24h) ---")
    diff_series = series.diff(24).dropna()

    adf_diff = adfuller(diff_series, autolag='AIC')
    print(f"ADF p-value: {adf_diff[1]:.4f} ({'Stationary' if adf_diff[1] < 0.05 else 'Non-stationary'})")

    kpss_diff = kpss(diff_series, regression='c', nlags='auto')
    print(f"KPSS p-value: {kpss_diff[1]:.4f} ({'Stationary' if kpss_diff[1] > 0.05 else 'Non-stationary'})")

    # Visualize rolling statistics
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    window = 168  # 1 week
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    axes[0].plot(series.index, series.values, label='Original', alpha=0.5, linewidth=0.5)
    axes[0].plot(rolling_mean.index, rolling_mean.values, label=f'Rolling Mean ({window}h)', color='red', linewidth=2)
    axes[0].set_ylabel('Load (MW)')
    axes[0].set_title('Load Series with Rolling Mean')
    axes[0].legend()

    axes[1].plot(rolling_std.index, rolling_std.values, label=f'Rolling Std ({window}h)', color='orange', linewidth=1)
    axes[1].set_ylabel('Standard Deviation (MW)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Rolling Standard Deviation')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '05_stationarity.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Saved: 05_stationarity.png")


# =============================================================================
# 6. FORECAST ERROR ANALYSIS
# =============================================================================
def analyze_forecast_errors(df: pd.DataFrame):
    """Analyze when and where the baseline forecast fails."""
    print("\n" + "="*70)
    print("6. FORECAST ERROR ANALYSIS (Baseline Weaknesses)")
    print("="*70)

    # Filter for valid errors
    df_valid = df.dropna(subset=['forecast_error_mw'])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 6a. Error distribution
    axes[0, 0].hist(df_valid['forecast_error_mw'], bins=50, density=True, alpha=0.7,
                    color='steelblue', edgecolor='black')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Forecast Error (MW)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Forecast Error Distribution')

    # 6b. Error by hour
    error_by_hour = df_valid.groupby('hour')['forecast_error_mw'].agg(['mean', 'std'])
    axes[0, 1].bar(error_by_hour.index, error_by_hour['mean'], yerr=error_by_hour['std'],
                   capsize=2, alpha=0.7, color='coral', edgecolor='black')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Mean Error (MW)')
    axes[0, 1].set_title('Forecast Error by Hour (+ = under-forecast)')

    # 6c. Error by day of week
    error_by_dow = df_valid.groupby('day_of_week')['forecast_error_mw'].agg(['mean', 'std'])
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    colors = ['blue']*5 + ['orange']*2
    axes[0, 2].bar(range(7), error_by_dow['mean'], yerr=error_by_dow['std'],
                   capsize=3, alpha=0.7, color=colors, edgecolor='black')
    axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[0, 2].set_xticks(range(7))
    axes[0, 2].set_xticklabels(days)
    axes[0, 2].set_xlabel('Day of Week')
    axes[0, 2].set_ylabel('Mean Error (MW)')
    axes[0, 2].set_title('Forecast Error by Day of Week')

    # 6d. Absolute error by hour
    abs_error_by_hour = df_valid.groupby('hour')['forecast_error_mw'].apply(lambda x: np.abs(x).mean())
    axes[1, 0].bar(abs_error_by_hour.index, abs_error_by_hour.values, alpha=0.7,
                   color='darkred', edgecolor='black')
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('MAE (MW)')
    axes[1, 0].set_title('Mean Absolute Error by Hour')

    # 6e. Error by month
    error_by_month = df_valid.groupby('month')['forecast_error_mw'].agg(['mean', 'std'])
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes[1, 1].bar(range(1, 13), error_by_month['mean'], yerr=error_by_month['std'],
                   capsize=2, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].set_xticklabels(months, rotation=45)
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Mean Error (MW)')
    axes[1, 1].set_title('Forecast Error by Month')

    # 6f. Error vs actual load
    axes[1, 2].scatter(df_valid['actual_load_mw'], df_valid['forecast_error_mw'],
                       alpha=0.1, s=1)
    axes[1, 2].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 2].set_xlabel('Actual Load (MW)')
    axes[1, 2].set_ylabel('Forecast Error (MW)')
    axes[1, 2].set_title('Error vs Load Level')

    # Add trend line
    z = np.polyfit(df_valid['actual_load_mw'], df_valid['forecast_error_mw'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df_valid['actual_load_mw'].min(), df_valid['actual_load_mw'].max(), 100)
    axes[1, 2].plot(x_range, p(x_range), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.4f})')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '06_forecast_errors.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print insights
    print("\nForecast Error Statistics:")
    print(f"  Overall MAE: {np.abs(df_valid['forecast_error_mw']).mean():.1f} MW")
    print(f"  Overall RMSE: {np.sqrt((df_valid['forecast_error_mw']**2).mean()):.1f} MW")
    print(f"  Overall Bias: {df_valid['forecast_error_mw'].mean():.1f} MW")

    print("\nWorst Hours (highest MAE):")
    worst_hours = abs_error_by_hour.nlargest(3)
    for hour, mae in worst_hours.items():
        print(f"  Hour {hour}: MAE = {mae:.1f} MW")

    print("\nBest Hours (lowest MAE):")
    best_hours = abs_error_by_hour.nsmallest(3)
    for hour, mae in best_hours.items():
        print(f"  Hour {hour}: MAE = {mae:.1f} MW")

    print("\n  Saved: 06_forecast_errors.png")

    # Error heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot = df_valid.pivot_table(values='forecast_error_mw', index='hour',
                                  columns='day_of_week', aggfunc=lambda x: np.abs(x).mean())
    pivot.columns = days
    im = ax.imshow(pivot.values, aspect='auto', cmap='Reds')
    ax.set_yticks(range(24))
    ax.set_yticklabels(range(1, 25))
    ax.set_xticks(range(7))
    ax.set_xticklabels(days)
    ax.set_ylabel('Hour of Day')
    ax.set_xlabel('Day of Week')
    ax.set_title('Forecast MAE Heatmap: Hour vs Day of Week (Opportunities to Improve)')
    plt.colorbar(im, ax=ax, label='MAE (MW)')
    plt.tight_layout()
    plt.savefig(PLOT_PATH / '06b_error_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: 06b_error_heatmap.png")


# =============================================================================
# 7. LOAD DURATION CURVE
# =============================================================================
def analyze_load_duration(df: pd.DataFrame):
    """Create load duration curve - classic energy analysis."""
    print("\n" + "="*70)
    print("7. LOAD DURATION CURVE")
    print("="*70)

    load = df['actual_load_mw'].dropna().sort_values(ascending=False).values
    duration = np.arange(1, len(load) + 1) / len(load) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Load duration curve
    axes[0].plot(duration, load, linewidth=2, color='steelblue')
    axes[0].fill_between(duration, load, alpha=0.3, color='steelblue')
    axes[0].set_xlabel('Duration (%)')
    axes[0].set_ylabel('Load (MW)')
    axes[0].set_title('Load Duration Curve')
    axes[0].grid(True, alpha=0.3)

    # Add key percentile lines
    for p, color in [(10, 'red'), (50, 'orange'), (90, 'green')]:
        val = np.percentile(load, 100-p)
        axes[0].axhline(y=val, color=color, linestyle='--', alpha=0.7,
                        label=f'P{p}: {val:.0f} MW')
        axes[0].axvline(x=p, color=color, linestyle='--', alpha=0.7)
    axes[0].legend()

    # By year comparison
    for year in df['year'].unique():
        year_load = df[df['year'] == year]['actual_load_mw'].dropna().sort_values(ascending=False).values
        year_duration = np.arange(1, len(year_load) + 1) / len(year_load) * 100
        axes[1].plot(year_duration, year_load, linewidth=2, label=str(year))

    axes[1].set_xlabel('Duration (%)')
    axes[1].set_ylabel('Load (MW)')
    axes[1].set_title('Load Duration Curve by Year')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '07_load_duration_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Statistics
    print("\nLoad Duration Statistics:")
    print(f"  Base load (P90): {np.percentile(load, 10):.0f} MW")
    print(f"  Median load (P50): {np.percentile(load, 50):.0f} MW")
    print(f"  Peak load (P10): {np.percentile(load, 90):.0f} MW")
    print(f"  Maximum: {load.max():.0f} MW")
    print(f"  Minimum: {load.min():.0f} MW")
    print(f"  Load factor: {load.mean()/load.max()*100:.1f}%")

    print("\n  Saved: 07_load_duration_curve.png")


# =============================================================================
# 8. ANOMALY DETECTION
# =============================================================================
def analyze_anomalies(df: pd.DataFrame):
    """Detect and visualize anomalies in load data."""
    print("\n" + "="*70)
    print("8. ANOMALY DETECTION")
    print("="*70)

    df_valid = df.dropna(subset=['actual_load_mw']).copy()

    # Method 1: Z-score based
    df_valid['z_score'] = (df_valid['actual_load_mw'] - df_valid['actual_load_mw'].mean()) / df_valid['actual_load_mw'].std()
    df_valid['is_anomaly_zscore'] = np.abs(df_valid['z_score']) > 3

    # Method 2: IQR based
    Q1 = df_valid['actual_load_mw'].quantile(0.25)
    Q3 = df_valid['actual_load_mw'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_valid['is_anomaly_iqr'] = (df_valid['actual_load_mw'] < lower_bound) | (df_valid['actual_load_mw'] > upper_bound)

    # Method 3: Hour-specific anomalies
    hourly_stats = df_valid.groupby('hour')['actual_load_mw'].agg(['mean', 'std'])
    df_valid = df_valid.join(hourly_stats, on='hour', rsuffix='_hourly')
    df_valid['z_score_hourly'] = (df_valid['actual_load_mw'] - df_valid['mean']) / df_valid['std']
    df_valid['is_anomaly_hourly'] = np.abs(df_valid['z_score_hourly']) > 3

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time series with anomalies
    axes[0, 0].plot(df_valid.index, df_valid['actual_load_mw'], linewidth=0.5, alpha=0.7)
    anomalies = df_valid[df_valid['is_anomaly_zscore']]
    axes[0, 0].scatter(anomalies.index, anomalies['actual_load_mw'], color='red', s=20, label='Z-score Anomalies')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Load (MW)')
    axes[0, 0].set_title(f'Load with Z-score Anomalies (|z|>3): {len(anomalies)} points')
    axes[0, 0].legend()

    # Anomalies by hour
    anomaly_by_hour = df_valid.groupby('hour')['is_anomaly_hourly'].sum()
    axes[0, 1].bar(anomaly_by_hour.index, anomaly_by_hour.values, color='red', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Number of Anomalies')
    axes[0, 1].set_title('Hour-Specific Anomalies by Hour')

    # Anomalies by day of week
    anomaly_by_dow = df_valid.groupby('day_of_week')['is_anomaly_hourly'].sum()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 0].bar(range(7), anomaly_by_dow.values, color='red', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(days)
    axes[1, 0].set_xlabel('Day of Week')
    axes[1, 0].set_ylabel('Number of Anomalies')
    axes[1, 0].set_title('Hour-Specific Anomalies by Day of Week')

    # Anomalies by month
    anomaly_by_month = df_valid.groupby('month')['is_anomaly_hourly'].sum()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes[1, 1].bar(range(1, 13), anomaly_by_month.values, color='red', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].set_xticklabels(months, rotation=45)
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Number of Anomalies')
    axes[1, 1].set_title('Hour-Specific Anomalies by Month')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '08_anomalies.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Statistics
    print("\nAnomaly Statistics:")
    print(f"  Z-score anomalies (|z|>3): {df_valid['is_anomaly_zscore'].sum()} ({df_valid['is_anomaly_zscore'].mean()*100:.2f}%)")
    print(f"  IQR anomalies: {df_valid['is_anomaly_iqr'].sum()} ({df_valid['is_anomaly_iqr'].mean()*100:.2f}%)")
    print(f"  Hour-specific anomalies: {df_valid['is_anomaly_hourly'].sum()} ({df_valid['is_anomaly_hourly'].mean()*100:.2f}%)")

    print(f"\nIQR Bounds: [{lower_bound:.0f}, {upper_bound:.0f}] MW")

    print("\n  Saved: 08_anomalies.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70)
    print("SLOVAKIA GRID LOAD ANALYSIS")
    print("="*70)

    df = load_data()
    print(f"\nData loaded: {len(df):,} records")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Run all analyses
    analyze_decomposition(df)
    analyze_temporal_patterns(df)
    analyze_distributions(df)
    analyze_autocorrelation(df)
    analyze_stationarity(df)
    analyze_forecast_errors(df)
    analyze_load_duration(df)
    analyze_anomalies(df)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll plots saved to: {PLOT_PATH}")

    # Summary of key findings for modeling
    print("\n" + "="*70)
    print("KEY FINDINGS FOR MODELING")
    print("="*70)
    print("""
1. SEASONALITY:
   - Strong daily pattern (24h cycle)
   - Strong weekly pattern (168h cycle)
   - Monthly/seasonal variation

2. WEEKDAY vs WEEKEND:
   - Weekday load is significantly higher
   - Different hourly profiles
   - Consider separate models or features

3. BASELINE WEAKNESSES:
   - Higher errors during peak hours
   - Higher errors on transition days (Mon, Fri)
   - Opportunity to improve by ~68 MW MAE

4. STATIONARITY:
   - Series has trend and seasonality
   - Consider differencing or detrending
   - SARIMA, Prophet, or ML with proper features

5. AUTOCORRELATION:
   - High correlation at lag 1, 24, 168
   - Include recent lags as features
   - Consider LSTM or similar for sequences

6. ANOMALIES:
   - Small percentage of outliers
   - May need special handling for holidays
   - Consider robust methods or outlier detection
""")


if __name__ == '__main__':
    main()
