"""
Analyze how the prediction signal evolves over time.
Focus on late 2025 and early 2026.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_PATH = Path(__file__).parent.parent.parent.parent
OUTPUT_PATH = Path(__file__).parent.parent


def load_all_data():
    """Load all data including early 2026."""
    print("[*] Loading data...")

    # Imbalance data
    imb = pd.read_csv(BASE_PATH / 'data' / 'master' / 'master_imbalance_data.csv')
    imb['datetime'] = pd.to_datetime(imb['datetime'])
    imb['hour_start'] = imb['datetime'].dt.floor('h')

    hourly_imb = imb.groupby('hour_start').agg({
        'System Imbalance (MWh)': 'sum',
        'Imbalance Settlement Price (EUR/MWh)': 'mean',
    }).reset_index()
    hourly_imb.columns = ['datetime', 'imbalance_mwh', 'imb_price']

    # IDM prices
    idm = pd.read_csv(BASE_PATH / 'data' / 'clean' / 'market' / 'intraday' / 'idm_60min.csv',
                      sep=';', decimal=',')
    idm['datetime'] = pd.to_datetime(idm['Delivery day']) + \
                      pd.to_timedelta((idm['Period number'] - 1), unit='h')
    idm = idm[['datetime', 'Weighted average price of all trades (EUR/MWh)']].copy()
    idm.columns = ['datetime', 'idm_price']
    idm['idm_price'] = pd.to_numeric(idm['idm_price'], errors='coerce')

    # H+2 predictions (H2 2025)
    pred_h2_2025 = pd.read_parquet(OUTPUT_PATH / 'data' / 'predictions_h2.parquet')
    pred_h2_2025['datetime'] = pd.to_datetime(pred_h2_2025['datetime'])
    pred_h2_2025['error_datetime'] = pred_h2_2025['datetime'] + pd.Timedelta(hours=2)

    # Check if we have Jan 2026 predictions from the other model
    jan2026_path = BASE_PATH / 'LoadAnalysis' / 'nowcast_5h' / 'data'
    has_jan2026 = False

    # For now, let's also load the load data to compute actual errors for Jan 2026
    load_data = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    load_data['datetime'] = pd.to_datetime(load_data['datetime'])
    load_data['error'] = load_data['actual_load_mw'] - load_data['forecast_load_mw']

    print(f"    Imbalance: {hourly_imb['datetime'].min()} to {hourly_imb['datetime'].max()}")
    print(f"    IDM: {idm['datetime'].min()} to {idm['datetime'].max()}")
    print(f"    Predictions: {pred_h2_2025['datetime'].min()} to {pred_h2_2025['datetime'].max()}")
    print(f"    Load data: {load_data['datetime'].min()} to {load_data['datetime'].max()}")

    # Merge predictions with imbalance and IDM
    merged = pred_h2_2025.merge(hourly_imb, left_on='error_datetime', right_on='datetime',
                                 suffixes=('_pred', '_imb'))
    merged = merged.merge(idm, left_on='error_datetime', right_on='datetime',
                          suffixes=('', '_idm'))
    merged['spread'] = merged['imb_price'] - merged['idm_price']
    merged['month'] = merged['error_datetime'].dt.to_period('M')
    merged['week'] = merged['error_datetime'].dt.to_period('W')

    print(f"    Merged H2 2025: {len(merged):,} records")

    # Also create a dataset with actual errors for comparison (including Jan 2026)
    load_with_prices = load_data.merge(hourly_imb, on='datetime', how='inner')
    load_with_prices = load_with_prices.merge(idm, on='datetime', how='inner')
    load_with_prices['spread'] = load_with_prices['imb_price'] - load_with_prices['idm_price']
    load_with_prices['month'] = load_with_prices['datetime'].dt.to_period('M')

    print(f"    Load with prices (for actual error analysis): {len(load_with_prices):,} records")

    return merged, load_with_prices


def analyze_signal_by_month(df, error_col='stage2_pred', label='Predicted'):
    """Analyze signal strength by month."""
    print(f"\n[*] Analyzing {label} error signal by month...")

    results = []

    for month in sorted(df['month'].unique()):
        month_data = df[df['month'] == month]

        if len(month_data) < 50:
            continue

        # Overall stats
        stats = {
            'month': str(month),
            'n': len(month_data),
            'mean_spread': month_data['spread'].mean(),
            'prob_pos_spread': (month_data['spread'] > 0).mean(),
        }

        # Signal at threshold -100
        neg_100 = month_data[month_data[error_col] < -100]
        if len(neg_100) >= 5:
            stats['n_neg100'] = len(neg_100)
            stats['spread_neg100'] = neg_100['spread'].mean()
            stats['prob_pos_neg100'] = (neg_100['spread'] > 0).mean()
        else:
            stats['n_neg100'] = 0
            stats['spread_neg100'] = np.nan
            stats['prob_pos_neg100'] = np.nan

        # Signal at threshold +100
        pos_100 = month_data[month_data[error_col] > 100]
        if len(pos_100) >= 5:
            stats['n_pos100'] = len(pos_100)
            stats['spread_pos100'] = pos_100['spread'].mean()
            stats['prob_pos_pos100'] = (pos_100['spread'] > 0).mean()
        else:
            stats['n_pos100'] = 0
            stats['spread_pos100'] = np.nan
            stats['prob_pos_pos100'] = np.nan

        # Imbalance direction signal
        neg_100_imb = month_data[month_data[error_col] < -100]
        if len(neg_100_imb) >= 5:
            stats['imb_mean_neg100'] = neg_100_imb['imbalance_mwh'].mean()
            stats['prob_pos_imb_neg100'] = (neg_100_imb['imbalance_mwh'] > 0).mean()

        results.append(stats)

    return pd.DataFrame(results)


def analyze_rolling_signal(df, error_col='stage2_pred', window_days=14):
    """Analyze signal with rolling window."""
    print(f"\n[*] Analyzing rolling signal (window={window_days} days)...")

    df = df.sort_values('error_datetime').copy()
    df['date'] = df['error_datetime'].dt.date

    # Daily aggregation first
    daily = df.groupby('date').apply(lambda x: pd.Series({
        'n': len(x),
        'mean_spread': x['spread'].mean(),
        'n_neg100': (x[error_col] < -100).sum(),
        'spread_neg100': x[x[error_col] < -100]['spread'].mean() if (x[error_col] < -100).sum() >= 3 else np.nan,
        'prob_pos_neg100': (x[x[error_col] < -100]['spread'] > 0).mean() if (x[error_col] < -100).sum() >= 3 else np.nan,
        'imb_mean_neg100': x[x[error_col] < -100]['imbalance_mwh'].mean() if (x[error_col] < -100).sum() >= 3 else np.nan,
    }), include_groups=False).reset_index()

    daily['date'] = pd.to_datetime(daily['date'])

    # Rolling stats
    daily['spread_roll'] = daily['mean_spread'].rolling(window_days, min_periods=7).mean()
    daily['spread_neg100_roll'] = daily['spread_neg100'].rolling(window_days, min_periods=5).mean()
    daily['prob_pos_neg100_roll'] = daily['prob_pos_neg100'].rolling(window_days, min_periods=5).mean()

    return daily


def create_evolution_plots(monthly_stats, rolling_stats, load_monthly, output_path):
    """Create time evolution plots."""
    print("\n[*] Creating plots...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Signal Evolution: Late 2025 - Early 2026', fontsize=14, fontweight='bold')

    # 1. Monthly spread for pred < -100
    ax = axes[0, 0]
    months = monthly_stats['month'].values
    spread_neg100 = monthly_stats['spread_neg100'].values
    colors = ['green' if x > 0 else 'red' for x in spread_neg100]
    bars = ax.bar(range(len(months)), spread_neg100, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.set_ylabel('Mean Spread (EUR/MWh)')
    ax.set_title('Signal: Spread when Pred < -100 MW')
    # Add count labels
    for i, (bar, n) in enumerate(zip(bars, monthly_stats['n_neg100'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={int(n)}', ha='center', va='bottom', fontsize=8)

    # 2. Monthly P(spread > 0) for pred < -100
    ax = axes[0, 1]
    prob_pos = monthly_stats['prob_pos_neg100'].values * 100
    ax.bar(range(len(months)), prob_pos, color='steelblue', alpha=0.7)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.set_ylabel('P(Imb > IDM) %')
    ax.set_title('Signal: P(Positive Spread) when Pred < -100 MW')
    ax.legend()
    ax.set_ylim(0, 100)

    # 3. Rolling spread signal
    ax = axes[1, 0]
    ax.plot(rolling_stats['date'], rolling_stats['spread_neg100_roll'], 'b-', linewidth=2,
            label='Spread when pred<-100 (14d roll)')
    ax.plot(rolling_stats['date'], rolling_stats['spread_roll'], 'gray', linewidth=1,
            alpha=0.5, label='Overall spread (14d roll)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Spread (EUR/MWh)')
    ax.set_title('Rolling Signal Strength (14-day window)')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

    # 4. Rolling probability
    ax = axes[1, 1]
    ax.plot(rolling_stats['date'], rolling_stats['prob_pos_neg100_roll'] * 100, 'b-', linewidth=2)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
    ax.set_xlabel('Date')
    ax.set_ylabel('P(Imb > IDM) %')
    ax.set_title('Rolling P(Positive Spread) when Pred < -100 MW')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=45)

    # 5. Compare predicted vs actual error signal (using load data)
    ax = axes[2, 0]
    if load_monthly is not None and len(load_monthly) > 0:
        months_load = load_monthly['month'].astype(str).values
        spread_actual = load_monthly['spread_neg100'].values
        colors = ['green' if x > 0 else 'red' for x in spread_actual]
        ax.bar(range(len(months_load)), spread_actual, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(len(months_load)))
        ax.set_xticklabels(months_load, rotation=45, ha='right')
        ax.set_ylabel('Mean Spread (EUR/MWh)')
        ax.set_title('Actual Error < -100 MW: Spread (Full Period)')

    # 6. Imbalance direction signal by month
    ax = axes[2, 1]
    imb_mean = monthly_stats['imb_mean_neg100'].values
    colors = ['green' if x > 0 else 'red' for x in imb_mean]
    ax.bar(range(len(months)), imb_mean, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.set_ylabel('Mean Imbalance (MWh)')
    ax.set_title('Imbalance when Pred < -100 MW')

    plt.tight_layout()
    plt.savefig(output_path / 'plots' / 'signal_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    [+] Saved: signal_evolution.png")


def main():
    print("=" * 70)
    print("SIGNAL EVOLUTION ANALYSIS: Late 2025 - Early 2026")
    print("=" * 70)

    merged, load_with_prices = load_all_data()

    # Monthly analysis for predictions
    monthly_stats = analyze_signal_by_month(merged, 'stage2_pred', 'Predicted')

    print("\n" + "-" * 70)
    print("MONTHLY SIGNAL STRENGTH (Predicted Error)")
    print("-" * 70)
    print(monthly_stats.to_string(index=False))

    # Monthly analysis for actual errors (to see Jan 2026)
    load_monthly = analyze_signal_by_month(load_with_prices, 'error', 'Actual')

    print("\n" + "-" * 70)
    print("MONTHLY SIGNAL STRENGTH (Actual Error - includes Jan 2026)")
    print("-" * 70)
    # Filter to relevant period
    load_monthly_filtered = load_monthly[load_monthly['month'] >= '2025-07']
    print(load_monthly_filtered.to_string(index=False))

    # Rolling analysis
    rolling_stats = analyze_rolling_signal(merged, 'stage2_pred', window_days=14)

    # Create plots
    create_evolution_plots(monthly_stats, rolling_stats, load_monthly_filtered, OUTPUT_PATH)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SIGNAL STABILITY")
    print("=" * 70)

    # Compare early vs late period
    early = merged[merged['error_datetime'] < '2025-10-01']
    late = merged[merged['error_datetime'] >= '2025-10-01']

    print("\n  PREDICTED ERROR SIGNAL (pred < -100 MW):")
    print(f"\n  Jul-Sep 2025:")
    early_neg = early[early['stage2_pred'] < -100]
    if len(early_neg) > 0:
        print(f"    n = {len(early_neg)}")
        print(f"    Mean spread: {early_neg['spread'].mean():+.1f} EUR/MWh")
        print(f"    P(Imb > IDM): {(early_neg['spread'] > 0).mean()*100:.0f}%")
        print(f"    Mean imbalance: {early_neg['imbalance_mwh'].mean():+.1f} MWh")

    print(f"\n  Oct-Dec 2025:")
    late_neg = late[late['stage2_pred'] < -100]
    if len(late_neg) > 0:
        print(f"    n = {len(late_neg)}")
        print(f"    Mean spread: {late_neg['spread'].mean():+.1f} EUR/MWh")
        print(f"    P(Imb > IDM): {(late_neg['spread'] > 0).mean()*100:.0f}%")
        print(f"    Mean imbalance: {late_neg['imbalance_mwh'].mean():+.1f} MWh")

    # January 2026 from actual errors
    jan2026 = load_with_prices[load_with_prices['datetime'].dt.to_period('M') == '2026-01']
    if len(jan2026) > 0:
        print(f"\n  Jan 2026 (using ACTUAL errors):")
        jan_neg = jan2026[jan2026['error'] < -100]
        if len(jan_neg) > 0:
            print(f"    n = {len(jan_neg)}")
            print(f"    Mean spread: {jan_neg['spread'].mean():+.1f} EUR/MWh")
            print(f"    P(Imb > IDM): {(jan_neg['spread'] > 0).mean()*100:.0f}%")
            print(f"    Mean imbalance: {jan_neg['imbalance_mwh'].mean():+.1f} MWh")

    return monthly_stats, rolling_stats, load_monthly


if __name__ == "__main__":
    monthly, rolling, load_monthly = main()
