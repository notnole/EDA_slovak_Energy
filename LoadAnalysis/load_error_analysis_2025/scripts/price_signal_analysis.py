"""
Analyze relationship between predicted error, imbalance price, and IDM price.

Questions:
1. Does predicted error signal imbalance price direction?
2. Can we predict if imbalance price will be above/below IDM price?
3. What's the trading value of the signal?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_PATH = Path(__file__).parent.parent.parent.parent
OUTPUT_PATH = Path(__file__).parent.parent


def load_all_data():
    """Load predictions, imbalance data, and IDM prices."""
    print("[*] Loading data...")

    # 1. Imbalance data (15-min)
    imb = pd.read_csv(BASE_PATH / 'data' / 'master' / 'master_imbalance_data.csv')
    imb['datetime'] = pd.to_datetime(imb['datetime'])
    imb['hour_start'] = imb['datetime'].dt.floor('h')

    # Aggregate to hourly
    hourly_imb = imb.groupby('hour_start').agg({
        'System Imbalance (MWh)': 'sum',
        'Imbalance Settlement Price (EUR/MWh)': 'mean',  # VWAP would be better but this works
        'SREC (EUR/MWh)': 'mean',
    }).reset_index()
    hourly_imb.columns = ['datetime', 'imbalance_mwh', 'imb_price', 'srec']
    print(f"    Imbalance: {len(hourly_imb):,} hourly records")

    # 2. IDM prices (hourly)
    idm = pd.read_csv(BASE_PATH / 'data' / 'clean' / 'market' / 'intraday' / 'idm_60min.csv',
                      sep=';', decimal=',')
    idm['datetime'] = pd.to_datetime(idm['Delivery day']) + \
                      pd.to_timedelta((idm['Period number'] - 1), unit='h')
    idm = idm[['datetime', 'Weighted average price of all trades (EUR/MWh)']].copy()
    idm.columns = ['datetime', 'idm_price']
    idm['idm_price'] = pd.to_numeric(idm['idm_price'], errors='coerce')
    print(f"    IDM: {len(idm):,} hourly records")
    print(f"    IDM range: {idm['datetime'].min()} to {idm['datetime'].max()}")

    # 3. Predictions
    pred = pd.read_parquet(OUTPUT_PATH / 'data' / 'predictions_h2.parquet')
    pred['datetime'] = pd.to_datetime(pred['datetime'])
    pred['error_datetime'] = pred['datetime'] + pd.Timedelta(hours=2)
    print(f"    Predictions: {len(pred):,} records")

    # Merge all
    merged = pred.merge(hourly_imb, left_on='error_datetime', right_on='datetime',
                        suffixes=('_pred', '_imb'))
    merged = merged.merge(idm, left_on='error_datetime', right_on='datetime',
                          suffixes=('', '_idm'))

    # Calculate spread
    merged['spread'] = merged['imb_price'] - merged['idm_price']  # imbalance - IDM

    print(f"    Merged: {len(merged):,} records")

    # Basic stats
    print(f"\n    Price stats:")
    print(f"      IDM price:       mean={merged['idm_price'].mean():.1f}, std={merged['idm_price'].std():.1f} EUR/MWh")
    print(f"      Imbalance price: mean={merged['imb_price'].mean():.1f}, std={merged['imb_price'].std():.1f} EUR/MWh")
    print(f"      Spread (imb-IDM): mean={merged['spread'].mean():.1f}, std={merged['spread'].std():.1f} EUR/MWh")

    return merged


def analyze_price_by_prediction(df):
    """Analyze prices by prediction buckets."""
    print("\n[*] Analyzing prices by prediction bucket...")

    bins = [-np.inf, -150, -100, -50, 0, 50, 100, 150, np.inf]
    labels = ['<-150', '-150:-100', '-100:-50', '-50:0', '0:50', '50:100', '100:150', '>150']

    df['pred_bucket'] = pd.cut(df['stage2_pred'], bins=bins, labels=labels)

    # Stats by bucket
    stats = df.groupby('pred_bucket', observed=True).agg({
        'imbalance_mwh': 'mean',
        'imb_price': ['mean', 'std'],
        'idm_price': 'mean',
        'spread': ['mean', 'median', 'std'],
        'stage2_pred': 'count',
    }).round(2)

    stats.columns = ['imb_mwh', 'imb_price_mean', 'imb_price_std',
                     'idm_price_mean', 'spread_mean', 'spread_median', 'spread_std', 'count']

    return stats


def analyze_spread_signal(df):
    """Analyze if predicted error signals spread direction."""
    print("\n[*] Analyzing spread signal...")

    results = {}

    # Overall spread stats
    results['overall'] = {
        'spread_mean': df['spread'].mean(),
        'spread_std': df['spread'].std(),
        'prob_positive_spread': (df['spread'] > 0).mean(),
    }

    # By prediction threshold
    for threshold in [50, 100, 150, 200]:
        pos = df[df['stage2_pred'] > threshold]
        neg = df[df['stage2_pred'] < -threshold]

        if len(pos) >= 10:
            results[f'pred_gt_{threshold}'] = {
                'n': len(pos),
                'spread_mean': pos['spread'].mean(),
                'spread_median': pos['spread'].median(),
                'prob_positive_spread': (pos['spread'] > 0).mean(),
                'imb_price_mean': pos['imb_price'].mean(),
                'idm_price_mean': pos['idm_price'].mean(),
            }

        if len(neg) >= 10:
            results[f'pred_lt_{-threshold}'] = {
                'n': len(neg),
                'spread_mean': neg['spread'].mean(),
                'spread_median': neg['spread'].median(),
                'prob_positive_spread': (neg['spread'] > 0).mean(),
                'imb_price_mean': neg['imb_price'].mean(),
                'idm_price_mean': neg['idm_price'].mean(),
            }

    return results


def analyze_trading_value(df):
    """Analyze hypothetical trading value of the signal."""
    print("\n[*] Analyzing trading value...")

    results = {}

    # Strategy: When we predict large positive error (under-forecast -> system short)
    # expect imbalance price > IDM (positive spread)
    # Action: Sell on IDM, buy back at imbalance = profit if spread is positive

    # Strategy 1: Trade when |prediction| > threshold
    for threshold in [100, 150]:
        # Long imbalance when pred > threshold (expect system short -> high imb price)
        long_signal = df[df['stage2_pred'] > threshold].copy()
        if len(long_signal) > 0:
            # Profit = imb_price - idm_price (if we sold IDM and bought imbalance)
            # But actually for SHORT position: profit when imb_price > idm_price
            long_signal['pnl'] = long_signal['spread']  # simplified
            results[f'long_thresh_{threshold}'] = {
                'n_trades': len(long_signal),
                'mean_pnl': long_signal['pnl'].mean(),
                'total_pnl': long_signal['pnl'].sum(),
                'win_rate': (long_signal['pnl'] > 0).mean(),
                'sharpe': long_signal['pnl'].mean() / long_signal['pnl'].std() if long_signal['pnl'].std() > 0 else 0,
            }

        # Short imbalance when pred < -threshold (expect system long -> low imb price)
        short_signal = df[df['stage2_pred'] < -threshold].copy()
        if len(short_signal) > 0:
            # Profit = idm_price - imb_price (if we bought IDM and sold at imbalance)
            short_signal['pnl'] = -short_signal['spread']  # opposite sign
            results[f'short_thresh_{threshold}'] = {
                'n_trades': len(short_signal),
                'mean_pnl': short_signal['pnl'].mean(),
                'total_pnl': short_signal['pnl'].sum(),
                'win_rate': (short_signal['pnl'] > 0).mean(),
                'sharpe': short_signal['pnl'].mean() / short_signal['pnl'].std() if short_signal['pnl'].std() > 0 else 0,
            }

    return results


def create_price_plots(df, bucket_stats, output_path):
    """Create price analysis plots."""
    print("\n[*] Creating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Predicted Error vs Imbalance Price & IDM Spread', fontsize=14, fontweight='bold')

    # 1. Mean spread by prediction bucket
    ax = axes[0, 0]
    stats = bucket_stats.reset_index()
    colors = ['green' if x > 0 else 'red' for x in stats['spread_mean']]
    ax.bar(range(len(stats)), stats['spread_mean'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats['pred_bucket'], rotation=45, ha='right')
    ax.set_xlabel('Predicted Error Bucket (MW)')
    ax.set_ylabel('Mean Spread: Imb - IDM (EUR/MWh)')
    ax.set_title('Price Spread by Prediction')

    # 2. Scatter: Predicted error vs Spread
    ax = axes[0, 1]
    ax.scatter(df['stage2_pred'], df['spread'], alpha=0.2, s=5)

    # Binned means
    df['pred_bin'] = pd.cut(df['stage2_pred'], bins=20)
    binned = df.groupby('pred_bin', observed=True).agg({
        'stage2_pred': 'mean',
        'spread': 'mean'
    })
    ax.plot(binned['stage2_pred'], binned['spread'], 'r-', linewidth=2, label='Binned mean')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted Error (MW)')
    ax.set_ylabel('Spread: Imb - IDM (EUR/MWh)')
    ax.set_title('Prediction vs Price Spread')
    ax.legend()

    # 3. Probability of positive spread by bucket
    ax = axes[0, 2]
    bins = [-np.inf, -150, -100, -50, 0, 50, 100, 150, np.inf]
    labels = ['<-150', '-150:-100', '-100:-50', '-50:0', '0:50', '50:100', '100:150', '>150']
    df['pred_bucket'] = pd.cut(df['stage2_pred'], bins=bins, labels=labels)

    prob_pos = df.groupby('pred_bucket', observed=True)['spread'].apply(lambda x: (x > 0).mean())
    ax.bar(range(len(prob_pos)), prob_pos.values, color='steelblue', alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
    ax.set_xticks(range(len(prob_pos)))
    ax.set_xticklabels(prob_pos.index, rotation=45, ha='right')
    ax.set_xlabel('Predicted Error Bucket (MW)')
    ax.set_ylabel('P(Imb Price > IDM Price)')
    ax.set_title('Probability of Positive Spread')
    ax.legend()
    ax.set_ylim(0, 1)

    # 4. Imbalance price vs IDM price
    ax = axes[1, 0]
    ax.scatter(df['idm_price'], df['imb_price'], alpha=0.2, s=5)
    lims = [min(df['idm_price'].min(), df['imb_price'].min()),
            max(df['idm_price'].max(), df['imb_price'].max())]
    ax.plot(lims, lims, 'r--', alpha=0.7, label='Parity')
    ax.set_xlabel('IDM Price (EUR/MWh)')
    ax.set_ylabel('Imbalance Price (EUR/MWh)')
    ax.set_title('Imbalance vs IDM Price')
    ax.legend()

    # 5. Signal strength at thresholds
    ax = axes[1, 1]
    thresholds = [25, 50, 75, 100, 125, 150, 175, 200]
    pos_spread = []
    neg_spread = []

    for t in thresholds:
        pos_data = df[df['stage2_pred'] > t]
        neg_data = df[df['stage2_pred'] < -t]

        if len(pos_data) > 10:
            pos_spread.append(pos_data['spread'].mean())
        else:
            pos_spread.append(np.nan)

        if len(neg_data) > 10:
            neg_spread.append(neg_data['spread'].mean())
        else:
            neg_spread.append(np.nan)

    ax.plot(thresholds, pos_spread, 'b-o', label='Pred > +threshold', markersize=6)
    ax.plot(thresholds, neg_spread, 'r-o', label='Pred < -threshold', markersize=6)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Threshold (MW)')
    ax.set_ylabel('Mean Spread (EUR/MWh)')
    ax.set_title('Spread Signal at Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Spread distribution by extreme predictions
    ax = axes[1, 2]
    extreme_pos = df[df['stage2_pred'] > 100]['spread']
    extreme_neg = df[df['stage2_pred'] < -100]['spread']
    baseline = df[(df['stage2_pred'] >= -50) & (df['stage2_pred'] <= 50)]['spread']

    ax.hist(baseline, bins=30, alpha=0.5, label=f'Baseline |pred|<50 (n={len(baseline)})', density=True)
    ax.hist(extreme_pos, bins=30, alpha=0.5, label=f'Pred>100 (n={len(extreme_pos)})', density=True)
    ax.hist(extreme_neg, bins=30, alpha=0.5, label=f'Pred<-100 (n={len(extreme_neg)})', density=True)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Spread: Imb - IDM (EUR/MWh)')
    ax.set_ylabel('Density')
    ax.set_title('Spread Distribution by Prediction')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path / 'plots' / 'price_signal_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    [+] Saved: price_signal_analysis.png")


def main():
    print("=" * 70)
    print("PRICE SIGNAL ANALYSIS: Predicted Error vs Imbalance & IDM Prices")
    print("=" * 70)

    df = load_all_data()

    if len(df) == 0:
        print("[!] No merged data available")
        return

    # Bucket analysis
    bucket_stats = analyze_price_by_prediction(df)
    print("\n" + "-" * 70)
    print("PRICE STATS BY PREDICTION BUCKET")
    print("-" * 70)
    print(bucket_stats.to_string())

    # Spread signal analysis
    spread_results = analyze_spread_signal(df)

    print("\n" + "-" * 70)
    print("SPREAD SIGNAL BY PREDICTION THRESHOLD")
    print("-" * 70)

    print(f"\n  Overall baseline:")
    print(f"    Mean spread (Imb-IDM): {spread_results['overall']['spread_mean']:+.2f} EUR/MWh")
    print(f"    P(Imb > IDM):          {spread_results['overall']['prob_positive_spread']*100:.1f}%")

    for key, val in spread_results.items():
        if key == 'overall':
            continue
        print(f"\n  {key} (n={val['n']}):")
        print(f"    Mean spread:   {val['spread_mean']:+.2f} EUR/MWh")
        print(f"    Median spread: {val['spread_median']:+.2f} EUR/MWh")
        print(f"    P(Imb > IDM):  {val['prob_positive_spread']*100:.1f}%")

    # Trading value
    trading_results = analyze_trading_value(df)

    print("\n" + "-" * 70)
    print("HYPOTHETICAL TRADING VALUE (EUR/MWh per trade)")
    print("-" * 70)

    for key, val in trading_results.items():
        print(f"\n  Strategy: {key}")
        print(f"    Trades:    {val['n_trades']}")
        print(f"    Mean P&L:  {val['mean_pnl']:+.2f} EUR/MWh")
        print(f"    Win rate:  {val['win_rate']*100:.1f}%")
        print(f"    Sharpe:    {val['sharpe']:.2f}")

    # Create plots
    create_price_plots(df, bucket_stats, OUTPUT_PATH)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: PRICE SIGNAL INTERPRETATION")
    print("=" * 70)

    print("""
    MECHANISM:
    - Positive pred error -> expect load under-forecast -> actual > forecast
    - System likely SHORT (deficit) -> needs more generation -> HIGH imbalance price
    - Therefore: pred > 0 should correlate with imb_price > idm_price (positive spread)

    - Negative pred error -> expect load over-forecast -> actual < forecast
    - System likely LONG (surplus) -> excess generation -> LOW imbalance price
    - Therefore: pred < 0 should correlate with imb_price < idm_price (negative spread)
    """)

    # Key findings
    if 'pred_gt_150' in spread_results and 'pred_lt_-150' in spread_results:
        pos = spread_results['pred_gt_150']
        neg = spread_results['pred_lt_-150']

        print(f"    KEY FINDINGS:")
        print(f"    When pred > +150 MW:")
        print(f"      -> Mean spread: {pos['spread_mean']:+.1f} EUR/MWh")
        print(f"      -> P(Imb > IDM): {pos['prob_positive_spread']*100:.0f}%")
        print(f"    When pred < -150 MW:")
        print(f"      -> Mean spread: {neg['spread_mean']:+.1f} EUR/MWh")
        print(f"      -> P(Imb > IDM): {neg['prob_positive_spread']*100:.0f}%")

    return df, bucket_stats, spread_results, trading_results


if __name__ == "__main__":
    df, bucket_stats, spread_results, trading_results = main()
