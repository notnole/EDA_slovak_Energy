"""
Day-Ahead Price Analysis

Analyzes:
1. Price decomposition (trend, daily/weekly seasonality)
2. Change from yesterday patterns
3. Price volatility analysis
4. Correlation with load
5. Negative price patterns
6. Cross-border flow impact
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import STL
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Paths
BASE_PATH = Path(__file__).parent.parent.parent
PRICE_PATH = Path(__file__).parent / 'da_prices.parquet'
LOAD_PATH = BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet'
PLOT_PATH = Path(__file__).parent / 'plots'


def load_data():
    """Load price and load data."""
    print("Loading data...")

    # Price data
    df_price = pd.read_parquet(PRICE_PATH)
    df_price['datetime'] = pd.to_datetime(df_price['datetime'])
    print(f"  Prices: {len(df_price)} records")

    # Load data
    df_load = pd.read_parquet(LOAD_PATH)
    df_load['datetime'] = pd.to_datetime(df_load['datetime'])
    print(f"  Load: {len(df_load)} records")

    return df_price, df_load


def basic_statistics(df: pd.DataFrame):
    """Print basic price statistics."""
    print("\n" + "=" * 70)
    print("BASIC PRICE STATISTICS")
    print("=" * 70)

    print(f"\nOverall:")
    print(f"  Mean:    {df['price_eur_mwh'].mean():.2f} EUR/MWh")
    print(f"  Median:  {df['price_eur_mwh'].median():.2f} EUR/MWh")
    print(f"  Std:     {df['price_eur_mwh'].std():.2f} EUR/MWh")
    print(f"  Min:     {df['price_eur_mwh'].min():.2f} EUR/MWh")
    print(f"  Max:     {df['price_eur_mwh'].max():.2f} EUR/MWh")

    # Percentiles
    print(f"\nPercentiles:")
    for p in [5, 25, 50, 75, 95]:
        val = df['price_eur_mwh'].quantile(p/100)
        print(f"  P{p}:     {val:.2f} EUR/MWh")

    # Negative prices
    neg = df[df['price_eur_mwh'] < 0]
    print(f"\nNegative prices:")
    print(f"  Count:   {len(neg)} ({len(neg)/len(df)*100:.2f}%)")
    print(f"  Mean:    {neg['price_eur_mwh'].mean():.2f} EUR/MWh")
    print(f"  Min:     {neg['price_eur_mwh'].min():.2f} EUR/MWh")

    # By year
    print(f"\nBy year:")
    yearly = df.groupby('year')['price_eur_mwh'].agg(['mean', 'std', 'min', 'max'])
    print(yearly.round(2).to_string())


def decompose_price(df: pd.DataFrame):
    """Decompose price series using STL."""
    print("\n" + "=" * 70)
    print("PRICE DECOMPOSITION (STL)")
    print("=" * 70)

    # Use 2024 data (complete year, hourly)
    df_2024 = df[df['year'] == 2024].copy()
    df_2024 = df_2024.set_index('datetime').sort_index()

    # STL decomposition with weekly period (168 hours)
    stl = STL(df_2024['price_eur_mwh'], period=168, robust=True)
    result = stl.fit()

    # Variance decomposition
    total_var = df_2024['price_eur_mwh'].var()
    trend_var = result.trend.var()
    seasonal_var = result.seasonal.var()
    resid_var = result.resid.var()

    print(f"\nVariance Decomposition (2024):")
    print(f"  Trend:     {trend_var/total_var*100:.1f}%")
    print(f"  Seasonal:  {seasonal_var/total_var*100:.1f}%")
    print(f"  Residual:  {resid_var/total_var*100:.1f}%")

    return result, df_2024


def temporal_patterns(df: pd.DataFrame):
    """Analyze temporal patterns."""
    print("\n" + "=" * 70)
    print("TEMPORAL PATTERNS")
    print("=" * 70)

    # Hourly pattern
    hourly = df.groupby('hour')['price_eur_mwh'].agg(['mean', 'std', 'median'])
    print(f"\nHourly pattern:")
    print(f"  Peak hour:     {hourly['mean'].idxmax()} ({hourly['mean'].max():.1f} EUR/MWh)")
    print(f"  Off-peak hour: {hourly['mean'].idxmin()} ({hourly['mean'].min():.1f} EUR/MWh)")
    print(f"  Spread:        {hourly['mean'].max() - hourly['mean'].min():.1f} EUR/MWh")

    # Weekday vs Weekend
    wkday = df[df['is_weekend'] == 0]['price_eur_mwh'].mean()
    wkend = df[df['is_weekend'] == 1]['price_eur_mwh'].mean()
    print(f"\nWeekday vs Weekend:")
    print(f"  Weekday mean:  {wkday:.1f} EUR/MWh")
    print(f"  Weekend mean:  {wkend:.1f} EUR/MWh")
    print(f"  Difference:    {wkday - wkend:.1f} EUR/MWh")

    # Monthly pattern
    monthly = df.groupby('month')['price_eur_mwh'].mean()
    print(f"\nMonthly pattern:")
    print(f"  Highest:  Month {monthly.idxmax()} ({monthly.max():.1f} EUR/MWh)")
    print(f"  Lowest:   Month {monthly.idxmin()} ({monthly.min():.1f} EUR/MWh)")

    return hourly


def change_from_yesterday(df: pd.DataFrame):
    """Analyze day-to-day price changes."""
    print("\n" + "=" * 70)
    print("CHANGE FROM YESTERDAY ANALYSIS")
    print("=" * 70)

    df_valid = df[df['price_change_24h'].notna()].copy()

    print(f"\n24-hour price change:")
    print(f"  Mean:    {df_valid['price_change_24h'].mean():.2f} EUR/MWh")
    print(f"  Std:     {df_valid['price_change_24h'].std():.2f} EUR/MWh")
    print(f"  Min:     {df_valid['price_change_24h'].min():.2f} EUR/MWh")
    print(f"  Max:     {df_valid['price_change_24h'].max():.2f} EUR/MWh")

    # Percentage change
    print(f"\n24-hour percentage change:")
    print(f"  Mean:    {df_valid['price_change_24h_pct'].mean():.1f}%")
    print(f"  Std:     {df_valid['price_change_24h_pct'].std():.1f}%")

    # Changes by hour
    hourly_change = df_valid.groupby('hour')['price_change_24h'].agg(['mean', 'std'])
    print(f"\nMost volatile hours (highest std):")
    top3 = hourly_change.nlargest(3, 'std')
    for hour, row in top3.iterrows():
        print(f"  Hour {hour}: std = {row['std']:.1f}, mean = {row['mean']:.1f}")

    # Autocorrelation of changes
    acf_1 = df_valid['price_change_24h'].autocorr(lag=1)
    acf_24 = df_valid['price_change_24h'].autocorr(lag=24)
    print(f"\nAutocorrelation of 24h changes:")
    print(f"  Lag 1h:  {acf_1:.3f}")
    print(f"  Lag 24h: {acf_24:.3f}")

    return df_valid


def price_load_correlation(df_price: pd.DataFrame, df_load: pd.DataFrame):
    """Analyze correlation between price and load."""
    print("\n" + "=" * 70)
    print("PRICE-LOAD CORRELATION")
    print("=" * 70)

    # Merge on datetime
    df = df_price.merge(
        df_load[['datetime', 'actual_load_mw', 'forecast_load_mw']],
        on='datetime',
        how='inner'
    )
    print(f"\nMatched records: {len(df)}")

    # Overall correlation
    corr_actual = df['price_eur_mwh'].corr(df['actual_load_mw'])
    corr_forecast = df['price_eur_mwh'].corr(df['forecast_load_mw'])
    print(f"\nOverall correlation:")
    print(f"  Price vs Actual Load:   {corr_actual:.3f}")
    print(f"  Price vs Forecast Load: {corr_forecast:.3f}")

    # Correlation by hour
    hourly_corr = df.groupby('hour').apply(
        lambda x: x['price_eur_mwh'].corr(x['actual_load_mw'])
    )
    print(f"\nCorrelation by hour:")
    print(f"  Highest: Hour {hourly_corr.idxmax()} ({hourly_corr.max():.3f})")
    print(f"  Lowest:  Hour {hourly_corr.idxmin()} ({hourly_corr.min():.3f})")

    # Correlation with load forecast error
    df['load_error'] = df['actual_load_mw'] - df['forecast_load_mw']
    corr_error = df['price_eur_mwh'].corr(df['load_error'])
    print(f"\nPrice vs Load forecast error: {corr_error:.3f}")

    return df


def negative_price_analysis(df: pd.DataFrame):
    """Analyze negative price patterns."""
    print("\n" + "=" * 70)
    print("NEGATIVE PRICE ANALYSIS")
    print("=" * 70)

    neg = df[df['price_eur_mwh'] < 0].copy()
    pos = df[df['price_eur_mwh'] >= 0].copy()

    print(f"\nNegative prices: {len(neg)} hours ({len(neg)/len(df)*100:.2f}%)")

    # Hourly distribution
    neg_hourly = neg.groupby('hour').size()
    print(f"\nMost common hours for negative prices:")
    top3 = neg_hourly.nlargest(3)
    for hour, count in top3.items():
        pct = count / len(neg) * 100
        print(f"  Hour {hour}: {count} ({pct:.1f}%)")

    # Weekend vs weekday
    neg_weekend_pct = neg['is_weekend'].mean() * 100
    print(f"\nNegative prices on weekend: {neg_weekend_pct:.1f}%")
    print(f"(vs {df['is_weekend'].mean()*100:.1f}% of all data)")

    # Net import during negative prices
    neg_import = neg['net_import'].mean()
    pos_import = pos['net_import'].mean()
    print(f"\nNet import (MW):")
    print(f"  During negative prices: {neg_import:.1f}")
    print(f"  During positive prices: {pos_import:.1f}")

    return neg


def create_plots(df_price: pd.DataFrame, df_load: pd.DataFrame, stl_result, df_2024):
    """Create analysis plots."""
    print("\n" + "=" * 70)
    print("CREATING PLOTS")
    print("=" * 70)

    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    # 1. STL Decomposition
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    axes[0].plot(df_2024.index, df_2024['price_eur_mwh'], 'b-', linewidth=0.5)
    axes[0].set_ylabel('Price (EUR/MWh)')
    axes[0].set_title('Original Price Series (2024)')

    axes[1].plot(df_2024.index, stl_result.trend, 'g-', linewidth=1)
    axes[1].set_ylabel('Trend')
    axes[1].set_title('Trend Component')

    axes[2].plot(df_2024.index, stl_result.seasonal, 'orange', linewidth=0.5)
    axes[2].set_ylabel('Seasonal')
    axes[2].set_title('Seasonal Component (168h period)')

    axes[3].plot(df_2024.index, stl_result.resid, 'r-', linewidth=0.5, alpha=0.7)
    axes[3].set_ylabel('Residual')
    axes[3].set_title('Residual Component')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '01_price_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_price_decomposition.png")

    # 2. Temporal patterns
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Hourly pattern
    hourly = df_price.groupby(['hour', 'is_weekend'])['price_eur_mwh'].mean().unstack()
    axes[0, 0].plot(hourly.index, hourly[0], 'b-o', label='Weekday', markersize=4)
    axes[0, 0].plot(hourly.index, hourly[1], 'r-o', label='Weekend', markersize=4)
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Price (EUR/MWh)')
    axes[0, 0].set_title('Average Price by Hour')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Daily pattern (by day of week)
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow = df_price.groupby('day_of_week')['price_eur_mwh'].agg(['mean', 'std'])
    axes[0, 1].bar(range(7), dow['mean'], yerr=dow['std'], capsize=3, color='steelblue', edgecolor='black')
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(dow_names)
    axes[0, 1].set_ylabel('Price (EUR/MWh)')
    axes[0, 1].set_title('Average Price by Day of Week')

    # Monthly pattern
    monthly = df_price.groupby('month')['price_eur_mwh'].agg(['mean', 'std'])
    axes[1, 0].bar(range(1, 13), monthly['mean'], yerr=monthly['std'], capsize=3, color='coral', edgecolor='black')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Price (EUR/MWh)')
    axes[1, 0].set_title('Average Price by Month')
    axes[1, 0].set_xticks(range(1, 13))

    # Price distribution
    axes[1, 1].hist(df_price['price_eur_mwh'], bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(df_price['price_eur_mwh'].mean(), color='red', linestyle='--', label=f"Mean: {df_price['price_eur_mwh'].mean():.1f}")
    axes[1, 1].axvline(0, color='black', linestyle='-', linewidth=2)
    axes[1, 1].set_xlabel('Price (EUR/MWh)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Price Distribution')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '02_temporal_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_temporal_patterns.png")

    # 3. Change from yesterday
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df_valid = df_price[df_price['price_change_24h'].notna()]

    # Change distribution
    axes[0, 0].hist(df_valid['price_change_24h'], bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('24h Price Change (EUR/MWh)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of 24h Price Changes')

    # Change by hour
    hourly_change = df_valid.groupby('hour')['price_change_24h'].agg(['mean', 'std'])
    axes[0, 1].bar(hourly_change.index, hourly_change['std'], color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Std of 24h Change (EUR/MWh)')
    axes[0, 1].set_title('Price Change Volatility by Hour')

    # Daily volatility over time
    daily_vol = df_price.groupby('date')['price_eur_mwh'].std()
    axes[1, 0].plot(pd.to_datetime(daily_vol.index), daily_vol.values, 'b-', linewidth=0.8)
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Intraday Std (EUR/MWh)')
    axes[1, 0].set_title('Daily Price Volatility Over Time')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Yesterday vs Today scatter
    sample = df_valid.sample(min(5000, len(df_valid)), random_state=42)
    axes[1, 1].scatter(sample['price_lag24'], sample['price_eur_mwh'], alpha=0.2, s=5)
    axes[1, 1].plot([sample['price_lag24'].min(), sample['price_lag24'].max()],
                    [sample['price_lag24'].min(), sample['price_lag24'].max()], 'r--', linewidth=2)
    corr = df_valid['price_lag24'].corr(df_valid['price_eur_mwh'])
    axes[1, 1].set_xlabel('Yesterday Price (EUR/MWh)')
    axes[1, 1].set_ylabel('Today Price (EUR/MWh)')
    axes[1, 1].set_title(f'Today vs Yesterday Price (corr={corr:.3f})')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '03_price_changes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_price_changes.png")

    # 4. Price-Load correlation
    df_merged = df_price.merge(
        df_load[['datetime', 'actual_load_mw', 'forecast_load_mw']],
        on='datetime',
        how='inner'
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Scatter: Price vs Load
    sample = df_merged.sample(min(5000, len(df_merged)), random_state=42)
    axes[0, 0].scatter(sample['actual_load_mw'], sample['price_eur_mwh'], alpha=0.2, s=5)
    corr = df_merged['actual_load_mw'].corr(df_merged['price_eur_mwh'])
    axes[0, 0].set_xlabel('Actual Load (MW)')
    axes[0, 0].set_ylabel('Price (EUR/MWh)')
    axes[0, 0].set_title(f'Price vs Load (corr={corr:.3f})')

    # Correlation by hour
    hourly_corr = df_merged.groupby('hour').apply(
        lambda x: x['price_eur_mwh'].corr(x['actual_load_mw'])
    )
    colors = ['green' if c > 0 else 'red' for c in hourly_corr]
    axes[0, 1].bar(hourly_corr.index, hourly_corr.values, color=colors, edgecolor='black')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].set_title('Price-Load Correlation by Hour')
    axes[0, 1].axhline(0, color='black')

    # Sample week - both series
    start = df_merged['datetime'].min() + pd.Timedelta(days=30)
    sample_week = df_merged[(df_merged['datetime'] >= start) & (df_merged['datetime'] < start + pd.Timedelta(days=7))]

    ax1 = axes[1, 0]
    ax2 = ax1.twinx()
    ax1.plot(sample_week['datetime'], sample_week['actual_load_mw'], 'b-', label='Load', linewidth=1)
    ax2.plot(sample_week['datetime'], sample_week['price_eur_mwh'], 'r-', label='Price', linewidth=1)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Load (MW)', color='blue')
    ax2.set_ylabel('Price (EUR/MWh)', color='red')
    ax1.set_title('Sample Week: Load and Price')
    ax1.tick_params(axis='x', rotation=45)

    # Heatmap: Hour x Day of Week for price
    pivot = df_merged.pivot_table(values='price_eur_mwh', index='hour', columns='day_of_week', aggfunc='mean')
    im = axes[1, 1].imshow(pivot.values, aspect='auto', cmap='RdYlGn_r')
    axes[1, 1].set_xlabel('Day of Week')
    axes[1, 1].set_ylabel('Hour')
    axes[1, 1].set_title('Average Price Heatmap')
    axes[1, 1].set_xticks(range(7))
    axes[1, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.colorbar(im, ax=axes[1, 1], label='EUR/MWh')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '04_price_load_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_price_load_correlation.png")

    # 5. Negative prices analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    neg = df_price[df_price['price_eur_mwh'] < 0]

    # Negative prices by hour
    neg_hourly = neg.groupby('hour').size()
    all_hourly = df_price.groupby('hour').size()
    neg_pct = (neg_hourly / all_hourly * 100).fillna(0)
    axes[0, 0].bar(neg_pct.index, neg_pct.values, color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('% of Hours with Negative Price')
    axes[0, 0].set_title('Negative Price Frequency by Hour')

    # Negative prices by month
    neg_monthly = neg.groupby('month').size()
    all_monthly = df_price.groupby('month').size()
    neg_pct_m = (neg_monthly / all_monthly * 100).fillna(0)
    axes[0, 1].bar(neg_pct_m.index, neg_pct_m.values, color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('% of Hours with Negative Price')
    axes[0, 1].set_title('Negative Price Frequency by Month')
    axes[0, 1].set_xticks(range(1, 13))

    # Net import vs price
    axes[1, 0].scatter(df_price['net_import'], df_price['price_eur_mwh'], alpha=0.1, s=5)
    corr = df_price['net_import'].corr(df_price['price_eur_mwh'])
    axes[1, 0].set_xlabel('Net Import (MW)')
    axes[1, 0].set_ylabel('Price (EUR/MWh)')
    axes[1, 0].set_title(f'Price vs Net Import (corr={corr:.3f})')
    axes[1, 0].axhline(0, color='red', linestyle='--')

    # Negative price magnitude distribution
    axes[1, 1].hist(neg['price_eur_mwh'], bins=50, color='red', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Negative Price (EUR/MWh)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Distribution of Negative Prices (n={len(neg)})')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '05_negative_prices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_negative_prices.png")

    # 6. Autocorrelation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # ACF of prices
    plot_acf(df_2024['price_eur_mwh'].dropna(), lags=200, ax=axes[0], alpha=0.05)
    axes[0].set_xlabel('Lag (hours)')
    axes[0].set_title('Price Autocorrelation (ACF)')
    axes[0].axvline(24, color='red', linestyle='--', alpha=0.5, label='24h')
    axes[0].axvline(168, color='green', linestyle='--', alpha=0.5, label='168h')
    axes[0].legend()

    # PACF
    plot_pacf(df_2024['price_eur_mwh'].dropna(), lags=50, ax=axes[1], alpha=0.05)
    axes[1].set_xlabel('Lag (hours)')
    axes[1].set_title('Partial Autocorrelation (PACF)')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '06_price_autocorrelation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_price_autocorrelation.png")


def main():
    """Run complete price analysis."""
    df_price, df_load = load_data()

    # Basic statistics
    basic_statistics(df_price)

    # Decomposition
    stl_result, df_2024 = decompose_price(df_price)

    # Temporal patterns
    temporal_patterns(df_price)

    # Change from yesterday
    change_from_yesterday(df_price)

    # Price-load correlation
    price_load_correlation(df_price, df_load)

    # Negative prices
    negative_price_analysis(df_price)

    # Create plots
    create_plots(df_price, df_load, stl_result, df_2024)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
