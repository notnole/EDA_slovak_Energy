"""
Separate analysis for 2026 data.
Uses actual DAMAS errors since our model predictions only cover H2 2025.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_PATH = Path(__file__).parent.parent.parent.parent
OUTPUT_PATH = Path(__file__).parent.parent


def load_2026_data():
    """Load all 2026 data."""
    print("[*] Loading 2026 data...")

    # Load data
    load_data = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    load_data['datetime'] = pd.to_datetime(load_data['datetime'])
    load_data['error'] = load_data['actual_load_mw'] - load_data['forecast_load_mw']

    # Imbalance
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

    # Merge
    df = load_data.merge(hourly_imb, on='datetime', how='inner')
    df = df.merge(idm, on='datetime', how='inner')
    df['spread'] = df['imb_price'] - df['idm_price']
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['week'] = df['datetime'].dt.isocalendar().week

    # Filter to 2026
    df_2026 = df[df['year'] == 2026].copy()

    print(f"    2026 records: {len(df_2026):,}")
    print(f"    Date range: {df_2026['datetime'].min()} to {df_2026['datetime'].max()}")

    # Also get 2025 for comparison
    df_2025 = df[df['year'] == 2025].copy()
    print(f"    2025 records (for comparison): {len(df_2025):,}")

    return df_2026, df_2025


def analyze_error_imbalance_relationship(df, label='2026'):
    """Analyze relationship between DAMAS error and imbalance."""
    print(f"\n[*] Analyzing error-imbalance relationship ({label})...")

    results = {}

    # Basic correlations
    results['corr_error_imb'] = df['error'].corr(df['imbalance_mwh'])
    results['corr_error_spread'] = df['error'].corr(df['spread'])

    print(f"    Correlation (error vs imbalance): {results['corr_error_imb']:.4f}")
    print(f"    Correlation (error vs spread):    {results['corr_error_spread']:.4f}")

    # By error bucket
    bins = [-np.inf, -150, -100, -50, 0, 50, 100, 150, np.inf]
    labels = ['<-150', '-150:-100', '-100:-50', '-50:0', '0:50', '50:100', '100:150', '>150']
    df['error_bucket'] = pd.cut(df['error'], bins=bins, labels=labels)

    bucket_stats = df.groupby('error_bucket', observed=True).agg({
        'imbalance_mwh': ['mean', 'std', 'count'],
        'spread': ['mean', 'median'],
        'imb_price': 'mean',
        'idm_price': 'mean',
    }).round(2)

    bucket_stats.columns = ['imb_mean', 'imb_std', 'count', 'spread_mean', 'spread_median',
                            'imb_price', 'idm_price']

    # Probabilities
    bucket_probs = df.groupby('error_bucket', observed=True).apply(
        lambda x: pd.Series({
            'prob_pos_imb': (x['imbalance_mwh'] > 0).mean(),
            'prob_pos_spread': (x['spread'] > 0).mean(),
        }), include_groups=False
    )

    return bucket_stats, bucket_probs, results


def analyze_extreme_errors(df, label='2026'):
    """Analyze what happens at extreme errors."""
    print(f"\n[*] Analyzing extreme errors ({label})...")

    results = {}

    for threshold in [100, 150, 200]:
        neg = df[df['error'] < -threshold]
        pos = df[df['error'] > threshold]

        if len(neg) >= 5:
            results[f'neg_{threshold}'] = {
                'n': len(neg),
                'error_mean': neg['error'].mean(),
                'imb_mean': neg['imbalance_mwh'].mean(),
                'prob_pos_imb': (neg['imbalance_mwh'] > 0).mean(),
                'spread_mean': neg['spread'].mean(),
                'spread_median': neg['spread'].median(),
                'prob_pos_spread': (neg['spread'] > 0).mean(),
                'imb_price_mean': neg['imb_price'].mean(),
                'idm_price_mean': neg['idm_price'].mean(),
            }

        if len(pos) >= 5:
            results[f'pos_{threshold}'] = {
                'n': len(pos),
                'error_mean': pos['error'].mean(),
                'imb_mean': pos['imbalance_mwh'].mean(),
                'prob_pos_imb': (pos['imbalance_mwh'] > 0).mean(),
                'spread_mean': pos['spread'].mean(),
                'spread_median': pos['spread'].median(),
                'prob_pos_spread': (pos['spread'] > 0).mean(),
                'imb_price_mean': pos['imb_price'].mean(),
                'idm_price_mean': pos['idm_price'].mean(),
            }

    return results


def analyze_by_week(df):
    """Weekly breakdown for 2026."""
    print("\n[*] Analyzing by week...")

    weekly = df.groupby('week').apply(lambda x: pd.Series({
        'n': len(x),
        'mean_error': x['error'].mean(),
        'mean_imb': x['imbalance_mwh'].mean(),
        'mean_spread': x['spread'].mean(),
        'n_neg100': (x['error'] < -100).sum(),
        'spread_neg100': x[x['error'] < -100]['spread'].mean() if (x['error'] < -100).sum() >= 3 else np.nan,
        'prob_pos_spread_neg100': (x[x['error'] < -100]['spread'] > 0).mean() if (x['error'] < -100).sum() >= 3 else np.nan,
    }), include_groups=False)

    return weekly


def create_2026_plots(df_2026, df_2025, bucket_stats_2026, bucket_stats_2025, output_path):
    """Create comparison plots."""
    print("\n[*] Creating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('2026 Analysis: DAMAS Error vs Imbalance & Prices', fontsize=14, fontweight='bold')

    # 1. Mean imbalance by error bucket - 2026 vs 2025
    ax = axes[0, 0]
    x = np.arange(len(bucket_stats_2026))
    width = 0.35
    ax.bar(x - width/2, bucket_stats_2026['imb_mean'], width, label='2026', alpha=0.7)
    ax.bar(x + width/2, bucket_stats_2025['imb_mean'], width, label='2025', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_stats_2026.index, rotation=45, ha='right')
    ax.set_xlabel('Error Bucket (MW)')
    ax.set_ylabel('Mean Imbalance (MWh)')
    ax.set_title('Imbalance by Error Bucket')
    ax.legend()

    # 2. Mean spread by error bucket - 2026 vs 2025
    ax = axes[0, 1]
    ax.bar(x - width/2, bucket_stats_2026['spread_mean'], width, label='2026', alpha=0.7)
    ax.bar(x + width/2, bucket_stats_2025['spread_mean'], width, label='2025', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_stats_2026.index, rotation=45, ha='right')
    ax.set_xlabel('Error Bucket (MW)')
    ax.set_ylabel('Mean Spread: Imb-IDM (EUR/MWh)')
    ax.set_title('Price Spread by Error Bucket')
    ax.legend()

    # 3. Scatter: Error vs Imbalance - 2026
    ax = axes[0, 2]
    ax.scatter(df_2026['error'], df_2026['imbalance_mwh'], alpha=0.3, s=10, label='2026')
    # Add trend line
    z = np.polyfit(df_2026['error'], df_2026['imbalance_mwh'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_2026['error'].min(), df_2026['error'].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.3f})')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('DAMAS Error (MW)')
    ax.set_ylabel('Imbalance (MWh)')
    ax.set_title('Error vs Imbalance (2026)')
    ax.legend()

    # 4. Scatter: Error vs Spread - 2026
    ax = axes[1, 0]
    ax.scatter(df_2026['error'], df_2026['spread'], alpha=0.3, s=10)
    # Binned mean
    df_2026['error_bin'] = pd.cut(df_2026['error'], bins=20)
    binned = df_2026.groupby('error_bin', observed=True).agg({
        'error': 'mean', 'spread': 'mean'
    })
    ax.plot(binned['error'], binned['spread'], 'r-', linewidth=2, label='Binned mean')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('DAMAS Error (MW)')
    ax.set_ylabel('Spread: Imb-IDM (EUR/MWh)')
    ax.set_title('Error vs Price Spread (2026)')
    ax.legend()

    # 5. Price comparison 2026
    ax = axes[1, 1]
    ax.scatter(df_2026['idm_price'], df_2026['imb_price'], alpha=0.3, s=10)
    lims = [min(df_2026['idm_price'].min(), df_2026['imb_price'].min()),
            max(df_2026['idm_price'].max(), df_2026['imb_price'].max())]
    ax.plot(lims, lims, 'r--', alpha=0.7, label='Parity')
    ax.set_xlabel('IDM Price (EUR/MWh)')
    ax.set_ylabel('Imbalance Price (EUR/MWh)')
    ax.set_title('Imbalance vs IDM Price (2026)')
    ax.legend()

    # 6. Distribution comparison
    ax = axes[1, 2]
    ax.hist(df_2026['spread'], bins=50, alpha=0.5, label='2026', density=True)
    ax.hist(df_2025['spread'], bins=50, alpha=0.5, label='2025', density=True)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=df_2026['spread'].mean(), color='blue', linestyle='--',
               label=f"2026 mean: {df_2026['spread'].mean():.1f}")
    ax.axvline(x=df_2025['spread'].mean(), color='orange', linestyle='--',
               label=f"2025 mean: {df_2025['spread'].mean():.1f}")
    ax.set_xlabel('Spread: Imb-IDM (EUR/MWh)')
    ax.set_ylabel('Density')
    ax.set_title('Spread Distribution: 2026 vs 2025')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path / 'plots' / 'analysis_2026.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    [+] Saved: analysis_2026.png")


def main():
    print("=" * 70)
    print("2026 SEPARATE ANALYSIS")
    print("=" * 70)

    df_2026, df_2025 = load_2026_data()

    if len(df_2026) == 0:
        print("[!] No 2026 data available")
        return

    # Basic stats
    print("\n" + "-" * 70)
    print("BASIC STATISTICS")
    print("-" * 70)

    print(f"\n  2026:")
    print(f"    Records: {len(df_2026):,}")
    print(f"    DAMAS error: mean={df_2026['error'].mean():.1f}, std={df_2026['error'].std():.1f} MW")
    print(f"    Imbalance:   mean={df_2026['imbalance_mwh'].mean():.1f}, std={df_2026['imbalance_mwh'].std():.1f} MWh")
    print(f"    IDM price:   mean={df_2026['idm_price'].mean():.1f}, std={df_2026['idm_price'].std():.1f} EUR/MWh")
    print(f"    Imb price:   mean={df_2026['imb_price'].mean():.1f}, std={df_2026['imb_price'].std():.1f} EUR/MWh")
    print(f"    Spread:      mean={df_2026['spread'].mean():.1f}, std={df_2026['spread'].std():.1f} EUR/MWh")

    print(f"\n  2025 (for comparison):")
    print(f"    Records: {len(df_2025):,}")
    print(f"    DAMAS error: mean={df_2025['error'].mean():.1f}, std={df_2025['error'].std():.1f} MW")
    print(f"    Imbalance:   mean={df_2025['imbalance_mwh'].mean():.1f}, std={df_2025['imbalance_mwh'].std():.1f} MWh")
    print(f"    Spread:      mean={df_2025['spread'].mean():.1f}, std={df_2025['spread'].std():.1f} EUR/MWh")

    # Error-imbalance relationship
    bucket_stats_2026, bucket_probs_2026, corr_2026 = analyze_error_imbalance_relationship(df_2026, '2026')
    bucket_stats_2025, bucket_probs_2025, corr_2025 = analyze_error_imbalance_relationship(df_2025, '2025')

    print("\n" + "-" * 70)
    print("ERROR BUCKET ANALYSIS - 2026")
    print("-" * 70)
    print(bucket_stats_2026.to_string())
    print("\nProbabilities:")
    print(bucket_probs_2026.round(3).to_string())

    print("\n" + "-" * 70)
    print("ERROR BUCKET ANALYSIS - 2025 (comparison)")
    print("-" * 70)
    print(bucket_stats_2025.to_string())

    # Extreme errors
    extreme_2026 = analyze_extreme_errors(df_2026, '2026')
    extreme_2025 = analyze_extreme_errors(df_2025, '2025')

    print("\n" + "-" * 70)
    print("EXTREME ERROR ANALYSIS - 2026")
    print("-" * 70)

    for key, val in extreme_2026.items():
        print(f"\n  {key} (n={val['n']}):")
        print(f"    Mean error:     {val['error_mean']:+.1f} MW")
        print(f"    Mean imbalance: {val['imb_mean']:+.1f} MWh")
        print(f"    P(imb > 0):     {val['prob_pos_imb']*100:.0f}%")
        print(f"    Mean spread:    {val['spread_mean']:+.1f} EUR/MWh")
        print(f"    P(spread > 0):  {val['prob_pos_spread']*100:.0f}%")

    # Weekly analysis
    weekly = analyze_by_week(df_2026)
    print("\n" + "-" * 70)
    print("WEEKLY BREAKDOWN - 2026")
    print("-" * 70)
    print(weekly.round(2).to_string())

    # Create plots
    create_2026_plots(df_2026, df_2025, bucket_stats_2026, bucket_stats_2025, OUTPUT_PATH)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: 2026 vs 2025 COMPARISON")
    print("=" * 70)

    print(f"""
    KEY METRICS:
                                    2026        2025
    ------------------------------------------------
    Mean spread (Imb-IDM):         {df_2026['spread'].mean():+.1f}       {df_2025['spread'].mean():+.1f} EUR/MWh
    P(Imb > IDM):                  {(df_2026['spread']>0).mean()*100:.0f}%        {(df_2025['spread']>0).mean()*100:.0f}%
    Corr(error, imbalance):        {corr_2026['corr_error_imb']:.3f}      {corr_2025['corr_error_imb']:.3f}
    Corr(error, spread):           {corr_2026['corr_error_spread']:.3f}      {corr_2025['corr_error_spread']:.3f}
    """)

    # Signal comparison at error < -100
    neg100_2026 = df_2026[df_2026['error'] < -100]
    neg100_2025 = df_2025[df_2025['error'] < -100]

    print(f"""
    SIGNAL WHEN ERROR < -100 MW:
                                    2026        2025
    ------------------------------------------------
    n:                             {len(neg100_2026)}          {len(neg100_2025)}
    Mean imbalance:                {neg100_2026['imbalance_mwh'].mean():+.1f}       {neg100_2025['imbalance_mwh'].mean():+.1f} MWh
    P(imb > 0):                    {(neg100_2026['imbalance_mwh']>0).mean()*100:.0f}%         {(neg100_2025['imbalance_mwh']>0).mean()*100:.0f}%
    Mean spread:                   {neg100_2026['spread'].mean():+.1f}       {neg100_2025['spread'].mean():+.1f} EUR/MWh
    P(spread > 0):                 {(neg100_2026['spread']>0).mean()*100:.0f}%         {(neg100_2025['spread']>0).mean()*100:.0f}%
    """)

    return df_2026, df_2025, bucket_stats_2026, extreme_2026


if __name__ == "__main__":
    df_2026, df_2025, bucket_stats, extreme = main()
