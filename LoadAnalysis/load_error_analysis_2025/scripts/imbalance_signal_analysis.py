"""
Analyze how predicted error magnitude signals imbalance direction and size.

Question: When we predict a large error (e.g., +200 MW), what does that tell us
about expected imbalance direction and magnitude?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_PATH = Path(__file__).parent.parent.parent.parent
OUTPUT_PATH = Path(__file__).parent.parent

def load_data():
    """Load and merge predictions with imbalance."""
    print("[*] Loading data...")

    # Imbalance
    imb = pd.read_csv(BASE_PATH / 'data' / 'master' / 'master_imbalance_data.csv')
    imb['datetime'] = pd.to_datetime(imb['datetime'])
    imb['hour_start'] = imb['datetime'].dt.floor('h')

    hourly_imb = imb.groupby('hour_start').agg({
        'System Imbalance (MWh)': 'sum',
    }).reset_index()
    hourly_imb.columns = ['datetime', 'imbalance_mwh']

    # Predictions
    pred = pd.read_parquet(OUTPUT_PATH / 'data' / 'predictions_h2.parquet')
    pred['datetime'] = pd.to_datetime(pred['datetime'])
    pred['error_datetime'] = pred['datetime'] + pd.Timedelta(hours=2)

    # Merge
    merged = pred.merge(hourly_imb, left_on='error_datetime', right_on='datetime',
                        suffixes=('_pred', '_imb'))

    print(f"    Records: {len(merged):,}")
    return merged


def analyze_by_buckets(df):
    """Analyze imbalance by predicted error buckets."""
    print("\n[*] Analyzing by prediction buckets...")

    # Create buckets
    bins = [-np.inf, -150, -100, -50, -25, 0, 25, 50, 100, 150, np.inf]
    labels = ['<-150', '-150:-100', '-100:-50', '-50:-25', '-25:0',
              '0:25', '25:50', '50:100', '100:150', '>150']

    df['pred_bucket'] = pd.cut(df['stage2_pred'], bins=bins, labels=labels)

    # Stats by bucket
    bucket_stats = df.groupby('pred_bucket', observed=True).agg({
        'imbalance_mwh': ['mean', 'median', 'std', 'count'],
        'stage2_pred': 'mean',
    }).round(2)

    bucket_stats.columns = ['imb_mean', 'imb_median', 'imb_std', 'count', 'pred_mean']

    # Probability of negative imbalance (deficit)
    bucket_probs = df.groupby('pred_bucket', observed=True).apply(
        lambda x: pd.Series({
            'prob_negative': (x['imbalance_mwh'] < 0).mean(),
            'prob_positive': (x['imbalance_mwh'] > 0).mean(),
            'prob_large_neg': (x['imbalance_mwh'] < -50).mean(),
            'prob_large_pos': (x['imbalance_mwh'] > 50).mean(),
        })
    )

    return bucket_stats, bucket_probs


def analyze_extreme_predictions(df, threshold=100):
    """Analyze what happens when we predict extreme errors."""
    print(f"\n[*] Analyzing extreme predictions (|pred| > {threshold} MW)...")

    large_pos = df[df['stage2_pred'] > threshold]
    large_neg = df[df['stage2_pred'] < -threshold]
    small = df[df['stage2_pred'].abs() <= 50]

    results = {}

    # Large positive prediction (under-forecast expected)
    if len(large_pos) > 0:
        results['large_pos'] = {
            'n': len(large_pos),
            'pred_mean': large_pos['stage2_pred'].mean(),
            'imb_mean': large_pos['imbalance_mwh'].mean(),
            'imb_median': large_pos['imbalance_mwh'].median(),
            'imb_std': large_pos['imbalance_mwh'].std(),
            'prob_neg_imb': (large_pos['imbalance_mwh'] < 0).mean(),
            'prob_large_neg': (large_pos['imbalance_mwh'] < -50).mean(),
        }
        print(f"\n    When predicted error > +{threshold} MW (n={len(large_pos)}):")
        print(f"      Mean prediction:     {results['large_pos']['pred_mean']:+.1f} MW")
        print(f"      Mean imbalance:      {results['large_pos']['imb_mean']:+.1f} MWh")
        print(f"      Median imbalance:    {results['large_pos']['imb_median']:+.1f} MWh")
        print(f"      P(imbalance < 0):    {results['large_pos']['prob_neg_imb']*100:.1f}%")
        print(f"      P(imbalance < -50):  {results['large_pos']['prob_large_neg']*100:.1f}%")

    # Large negative prediction (over-forecast expected)
    if len(large_neg) > 0:
        results['large_neg'] = {
            'n': len(large_neg),
            'pred_mean': large_neg['stage2_pred'].mean(),
            'imb_mean': large_neg['imbalance_mwh'].mean(),
            'imb_median': large_neg['imbalance_mwh'].median(),
            'imb_std': large_neg['imbalance_mwh'].std(),
            'prob_pos_imb': (large_neg['imbalance_mwh'] > 0).mean(),
            'prob_large_pos': (large_neg['imbalance_mwh'] > 50).mean(),
        }
        print(f"\n    When predicted error < -{threshold} MW (n={len(large_neg)}):")
        print(f"      Mean prediction:     {results['large_neg']['pred_mean']:+.1f} MW")
        print(f"      Mean imbalance:      {results['large_neg']['imb_mean']:+.1f} MWh")
        print(f"      Median imbalance:    {results['large_neg']['imb_median']:+.1f} MWh")
        print(f"      P(imbalance > 0):    {results['large_neg']['prob_pos_imb']*100:.1f}%")
        print(f"      P(imbalance > +50):  {results['large_neg']['prob_large_pos']*100:.1f}%")

    # Baseline (small prediction)
    if len(small) > 0:
        results['small'] = {
            'n': len(small),
            'imb_mean': small['imbalance_mwh'].mean(),
            'imb_std': small['imbalance_mwh'].std(),
        }
        print(f"\n    Baseline (|pred| <= 50 MW, n={len(small)}):")
        print(f"      Mean imbalance:      {results['small']['imb_mean']:+.1f} MWh")
        print(f"      Std imbalance:       {results['small']['imb_std']:.1f} MWh")

    return results


def analyze_quantiles(df):
    """Analyze imbalance by prediction quantiles."""
    print("\n[*] Analyzing by prediction quantiles...")

    # Deciles
    df['pred_decile'] = pd.qcut(df['stage2_pred'], 10, labels=False)

    decile_stats = df.groupby('pred_decile').agg({
        'stage2_pred': ['mean', 'min', 'max'],
        'imbalance_mwh': ['mean', 'median', 'std'],
    }).round(2)

    decile_probs = df.groupby('pred_decile').apply(
        lambda x: (x['imbalance_mwh'] < 0).mean()
    )

    return decile_stats, decile_probs


def create_signal_plots(df, bucket_stats, output_path):
    """Create visualization of signal strength."""
    print("\n[*] Creating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Predicted Error as Signal for Imbalance Direction', fontsize=14, fontweight='bold')

    # 1. Mean imbalance by prediction bucket
    ax = axes[0, 0]
    bucket_stats_plot = bucket_stats.reset_index()
    colors = ['red' if x < 0 else 'green' for x in bucket_stats_plot['imb_mean']]
    bars = ax.bar(range(len(bucket_stats_plot)), bucket_stats_plot['imb_mean'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(len(bucket_stats_plot)))
    ax.set_xticklabels(bucket_stats_plot['pred_bucket'], rotation=45, ha='right')
    ax.set_xlabel('Predicted Error Bucket (MW)')
    ax.set_ylabel('Mean Imbalance (MWh)')
    ax.set_title('Mean Imbalance by Prediction Bucket')

    # 2. Probability of negative imbalance by bucket
    ax = axes[0, 1]
    bins = [-np.inf, -150, -100, -50, -25, 0, 25, 50, 100, 150, np.inf]
    labels = ['<-150', '-150:-100', '-100:-50', '-50:-25', '-25:0',
              '0:25', '25:50', '50:100', '100:150', '>150']
    df['pred_bucket'] = pd.cut(df['stage2_pred'], bins=bins, labels=labels)

    prob_neg = df.groupby('pred_bucket', observed=True)['imbalance_mwh'].apply(
        lambda x: (x < 0).mean()
    )
    ax.bar(range(len(prob_neg)), prob_neg.values, color='steelblue', alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
    ax.set_xticks(range(len(prob_neg)))
    ax.set_xticklabels(prob_neg.index, rotation=45, ha='right')
    ax.set_xlabel('Predicted Error Bucket (MW)')
    ax.set_ylabel('P(Imbalance < 0)')
    ax.set_title('Probability of Negative Imbalance')
    ax.legend()
    ax.set_ylim(0, 1)

    # 3. Box plot by bucket
    ax = axes[0, 2]
    bucket_order = labels
    df_plot = df[df['pred_bucket'].isin(bucket_order)]
    df_plot['pred_bucket'] = pd.Categorical(df_plot['pred_bucket'], categories=bucket_order, ordered=True)
    df_plot = df_plot.sort_values('pred_bucket')

    bucket_data = [df_plot[df_plot['pred_bucket'] == b]['imbalance_mwh'].values
                   for b in bucket_order if b in df_plot['pred_bucket'].values]
    bucket_labels_present = [b for b in bucket_order if b in df_plot['pred_bucket'].values]

    bp = ax.boxplot(bucket_data, labels=bucket_labels_present, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.set_xticklabels(bucket_labels_present, rotation=45, ha='right')
    ax.set_xlabel('Predicted Error Bucket (MW)')
    ax.set_ylabel('Imbalance (MWh)')
    ax.set_title('Imbalance Distribution by Bucket')

    # 4. Scatter with trend line
    ax = axes[1, 0]
    ax.scatter(df['stage2_pred'], df['imbalance_mwh'], alpha=0.2, s=5)

    # Add binned means
    df['pred_bin'] = pd.cut(df['stage2_pred'], bins=20)
    binned_mean = df.groupby('pred_bin', observed=True).agg({
        'stage2_pred': 'mean',
        'imbalance_mwh': 'mean'
    })
    ax.plot(binned_mean['stage2_pred'], binned_mean['imbalance_mwh'],
            'r-', linewidth=2, label='Binned mean')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted Error (MW)')
    ax.set_ylabel('Imbalance (MWh)')
    ax.set_title('Trend: Predicted Error vs Imbalance')
    ax.legend()

    # 5. Signal strength at thresholds
    ax = axes[1, 1]
    thresholds = [25, 50, 75, 100, 125, 150, 175, 200]
    pos_signal = []
    neg_signal = []

    for t in thresholds:
        pos_data = df[df['stage2_pred'] > t]
        neg_data = df[df['stage2_pred'] < -t]

        if len(pos_data) > 10:
            pos_signal.append(pos_data['imbalance_mwh'].mean())
        else:
            pos_signal.append(np.nan)

        if len(neg_data) > 10:
            neg_signal.append(neg_data['imbalance_mwh'].mean())
        else:
            neg_signal.append(np.nan)

    ax.plot(thresholds, pos_signal, 'b-o', label='Pred > +threshold', markersize=6)
    ax.plot(thresholds, neg_signal, 'r-o', label='Pred < -threshold', markersize=6)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Threshold (MW)')
    ax.set_ylabel('Mean Imbalance (MWh)')
    ax.set_title('Signal Strength at Different Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Count at each threshold
    ax = axes[1, 2]
    counts_pos = [len(df[df['stage2_pred'] > t]) for t in thresholds]
    counts_neg = [len(df[df['stage2_pred'] < -t]) for t in thresholds]

    x = np.arange(len(thresholds))
    width = 0.35
    ax.bar(x - width/2, counts_pos, width, label='Pred > +threshold', color='blue', alpha=0.7)
    ax.bar(x + width/2, counts_neg, width, label='Pred < -threshold', color='red', alpha=0.7)
    ax.set_xlabel('Threshold (MW)')
    ax.set_ylabel('Count')
    ax.set_title('Sample Size at Thresholds')
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'plots' / 'imbalance_signal_strength.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    [+] Saved: imbalance_signal_strength.png")


def main():
    print("=" * 70)
    print("PREDICTED ERROR AS IMBALANCE SIGNAL")
    print("=" * 70)

    df = load_data()

    # Bucket analysis
    bucket_stats, bucket_probs = analyze_by_buckets(df)

    print("\n" + "-" * 70)
    print("IMBALANCE STATS BY PREDICTION BUCKET")
    print("-" * 70)
    print(bucket_stats.to_string())

    print("\n" + "-" * 70)
    print("PROBABILITY OF IMBALANCE DIRECTION BY BUCKET")
    print("-" * 70)
    print(bucket_probs.round(3).to_string())

    # Extreme prediction analysis
    extreme_100 = analyze_extreme_predictions(df, threshold=100)
    extreme_150 = analyze_extreme_predictions(df, threshold=150)

    # Quantile analysis
    decile_stats, decile_probs = analyze_quantiles(df)

    print("\n" + "-" * 70)
    print("IMBALANCE BY PREDICTION DECILE")
    print("-" * 70)
    print(decile_stats.to_string())

    print("\nP(imbalance < 0) by decile:")
    print(decile_probs.round(3).to_string())

    # Create plots
    create_signal_plots(df, bucket_stats, OUTPUT_PATH)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SIGNAL INTERPRETATION")
    print("=" * 70)

    print("""
    INTERPRETATION OF PREDICTED ERROR AS SIGNAL:

    Remember: Error = Actual - Forecast
      - Positive error (+) = Load UNDER-forecast (actual > expected)
      - Negative error (-) = Load OVER-forecast (actual < expected)

    And: Imbalance convention (typically):
      - Negative imbalance = System SHORT (needs more generation)
      - Positive imbalance = System LONG (excess generation)

    SIGNAL STRENGTH:
    """)

    # Calculate signal at key thresholds
    for threshold in [100, 150, 200]:
        pos = df[df['stage2_pred'] > threshold]
        neg = df[df['stage2_pred'] < -threshold]

        if len(pos) >= 10:
            print(f"    When pred > +{threshold} MW (n={len(pos):,}):")
            print(f"      -> Mean imbalance: {pos['imbalance_mwh'].mean():+.1f} MWh")
            print(f"      -> P(negative imb): {(pos['imbalance_mwh'] < 0).mean()*100:.0f}%")

        if len(neg) >= 10:
            print(f"    When pred < -{threshold} MW (n={len(neg):,}):")
            print(f"      -> Mean imbalance: {neg['imbalance_mwh'].mean():+.1f} MWh")
            print(f"      -> P(positive imb): {(neg['imbalance_mwh'] > 0).mean()*100:.0f}%")
        print()

    return df, bucket_stats


if __name__ == "__main__":
    df, stats = main()
