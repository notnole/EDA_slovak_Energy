"""
Analyze correlation between H+2 DAMAS error predictions and system imbalance.

Hypothesis: If we can predict DAMAS load forecast errors 2 hours ahead,
can this help predict system imbalance volume?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

BASE_PATH = Path(__file__).parent.parent.parent.parent  # ipesoft_eda_data
OUTPUT_PATH = Path(__file__).parent.parent  # load_error_analysis_2025

def load_imbalance_data():
    """Load and aggregate imbalance data to hourly."""
    print("[*] Loading imbalance data...")

    imb = pd.read_csv(BASE_PATH / 'data' / 'master' / 'master_imbalance_data.csv')
    imb['datetime'] = pd.to_datetime(imb['datetime'])

    print(f"    Raw records: {len(imb):,}")
    print(f"    Date range: {imb['datetime'].min()} to {imb['datetime'].max()}")

    # Aggregate to hourly (sum of 4 settlement periods)
    imb['hour_start'] = imb['datetime'].dt.floor('h')

    hourly_imb = imb.groupby('hour_start').agg({
        'System Imbalance (MWh)': 'sum',  # Sum of 4 quarters
    }).reset_index()
    hourly_imb.columns = ['datetime', 'imbalance_mwh']

    # Also compute absolute imbalance
    hourly_imb['abs_imbalance'] = np.abs(hourly_imb['imbalance_mwh'])

    print(f"    Hourly records: {len(hourly_imb):,}")

    return hourly_imb


def load_predictions():
    """Load H+2 predictions from our model."""
    print("[*] Loading H+2 predictions...")

    pred = pd.read_parquet(OUTPUT_PATH / 'data' / 'predictions_h2.parquet')
    pred['datetime'] = pd.to_datetime(pred['datetime'])

    print(f"    Prediction records: {len(pred):,}")
    print(f"    Date range: {pred['datetime'].min()} to {pred['datetime'].max()}")

    return pred


def analyze_correlation(pred, imb):
    """Analyze correlation between error predictions and imbalance."""
    print("\n[*] Merging datasets...")

    # Merge on datetime
    # Note: pred datetime is when prediction is made, target_h2 is the error 2h ahead
    # So the actual error occurs at datetime + 2h
    pred['error_datetime'] = pred['datetime'] + pd.Timedelta(hours=2)

    merged = pred.merge(imb, left_on='error_datetime', right_on='datetime',
                        suffixes=('_pred', '_imb'))

    print(f"    Merged records: {len(merged):,}")

    if len(merged) == 0:
        print("    [!] No overlapping data!")
        return None

    # Key columns
    # - target_h2: actual DAMAS error (actual - forecast)
    # - stage2_pred: our predicted error
    # - imbalance_mwh: system imbalance

    results = {}

    # 1. Correlation: Actual DAMAS error vs Imbalance
    corr_actual = merged['target_h2'].corr(merged['imbalance_mwh'])
    results['actual_error_vs_imbalance'] = corr_actual
    print(f"\n    Actual DAMAS error vs Imbalance: r = {corr_actual:.4f}")

    # 2. Correlation: Predicted error vs Imbalance
    corr_pred = merged['stage2_pred'].corr(merged['imbalance_mwh'])
    results['predicted_error_vs_imbalance'] = corr_pred
    print(f"    Predicted error vs Imbalance:    r = {corr_pred:.4f}")

    # 3. Correlation: Absolute values
    merged['abs_actual_error'] = np.abs(merged['target_h2'])
    merged['abs_pred_error'] = np.abs(merged['stage2_pred'])

    corr_abs_actual = merged['abs_actual_error'].corr(merged['abs_imbalance'])
    corr_abs_pred = merged['abs_pred_error'].corr(merged['abs_imbalance'])
    results['abs_actual_vs_abs_imbalance'] = corr_abs_actual
    results['abs_pred_vs_abs_imbalance'] = corr_abs_pred
    print(f"    |Actual error| vs |Imbalance|:   r = {corr_abs_actual:.4f}")
    print(f"    |Predicted error| vs |Imbalance|: r = {corr_abs_pred:.4f}")

    # 4. Sign agreement
    sign_actual = np.sign(merged['target_h2'])
    sign_pred = np.sign(merged['stage2_pred'])
    sign_imb = np.sign(merged['imbalance_mwh'])

    sign_match_actual = (sign_actual == sign_imb).mean()
    sign_match_pred = (sign_pred == sign_imb).mean()
    results['sign_match_actual'] = sign_match_actual
    results['sign_match_pred'] = sign_match_pred
    print(f"\n    Sign match (actual error vs imb):    {sign_match_actual*100:.1f}%")
    print(f"    Sign match (predicted error vs imb): {sign_match_pred*100:.1f}%")

    # 5. Statistical tests
    r, p = stats.pearsonr(merged['stage2_pred'].dropna(),
                          merged.loc[merged['stage2_pred'].notna(), 'imbalance_mwh'])
    results['pearson_p_value'] = p
    print(f"\n    Pearson p-value: {p:.2e}")

    # 6. Regression: Can predicted error explain imbalance variance?
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    X = merged[['stage2_pred']].values
    y = merged['imbalance_mwh'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    results['r2_simple'] = r2
    print(f"    R-squared (pred error -> imbalance): {r2:.4f}")

    # 7. With hour features
    merged['hour'] = merged['datetime_pred'].dt.hour
    merged['dow'] = merged['datetime_pred'].dt.dayofweek

    X_full = merged[['stage2_pred', 'hour', 'dow']].values
    model_full = LinearRegression()
    model_full.fit(X_full, y)
    y_pred_full = model_full.predict(X_full)
    r2_full = r2_score(y, y_pred_full)
    results['r2_with_time'] = r2_full
    print(f"    R-squared (pred + hour + dow):       {r2_full:.4f}")

    return merged, results


def create_plots(merged, output_path):
    """Create visualization plots."""
    print("\n[*] Creating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('H+2 DAMAS Error Prediction vs System Imbalance', fontsize=14, fontweight='bold')

    # 1. Scatter: Actual error vs Imbalance
    ax = axes[0, 0]
    ax.scatter(merged['target_h2'], merged['imbalance_mwh'], alpha=0.3, s=5)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    corr = merged['target_h2'].corr(merged['imbalance_mwh'])
    ax.set_xlabel('Actual DAMAS Error (MW)')
    ax.set_ylabel('System Imbalance (MWh)')
    ax.set_title(f'Actual Error vs Imbalance (r={corr:.3f})')

    # 2. Scatter: Predicted error vs Imbalance
    ax = axes[0, 1]
    ax.scatter(merged['stage2_pred'], merged['imbalance_mwh'], alpha=0.3, s=5)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    corr = merged['stage2_pred'].corr(merged['imbalance_mwh'])
    ax.set_xlabel('Predicted DAMAS Error (MW)')
    ax.set_ylabel('System Imbalance (MWh)')
    ax.set_title(f'Predicted Error vs Imbalance (r={corr:.3f})')

    # 3. Scatter: Absolute values
    ax = axes[0, 2]
    ax.scatter(merged['abs_pred_error'], merged['abs_imbalance'], alpha=0.3, s=5)
    corr = merged['abs_pred_error'].corr(merged['abs_imbalance'])
    ax.set_xlabel('|Predicted Error| (MW)')
    ax.set_ylabel('|Imbalance| (MWh)')
    ax.set_title(f'Absolute Values (r={corr:.3f})')

    # 4. Time series comparison (sample month)
    ax = axes[1, 0]
    sample = merged[(merged['datetime_pred'].dt.month == 8) &
                    (merged['datetime_pred'].dt.year == 2025)].head(168)  # 1 week
    if len(sample) > 0:
        ax.plot(sample['datetime_pred'], sample['imbalance_mwh'], label='Imbalance', alpha=0.7)
        ax.plot(sample['datetime_pred'], sample['stage2_pred'] * 0.5, label='Pred Error (scaled)', alpha=0.7)
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('MWh / MW')
        ax.set_title('Sample Week (Aug 2025)')
        ax.tick_params(axis='x', rotation=45)

    # 5. By hour correlation
    ax = axes[1, 1]
    hourly_corr = merged.groupby(merged['datetime_pred'].dt.hour).apply(
        lambda x: x['stage2_pred'].corr(x['imbalance_mwh'])
    )
    ax.bar(hourly_corr.index, hourly_corr.values)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation by Hour')
    ax.set_xticks(range(0, 24, 2))

    # 6. Distribution comparison
    ax = axes[1, 2]
    ax.hist(merged['imbalance_mwh'], bins=50, alpha=0.5, label='Imbalance', density=True)
    ax.hist(merged['stage2_pred'], bins=50, alpha=0.5, label='Pred Error', density=True)
    ax.legend()
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison')

    plt.tight_layout()
    plt.savefig(output_path / 'plots' / 'imbalance_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    [+] Saved: imbalance_correlation.png")


def main():
    print("=" * 70)
    print("H+2 DAMAS ERROR vs SYSTEM IMBALANCE ANALYSIS")
    print("=" * 70)

    # Load data
    imb = load_imbalance_data()
    pred = load_predictions()

    # Analyze correlation
    result = analyze_correlation(pred, imb)

    if result is None:
        return

    merged, stats_results = result

    # Create plots
    create_plots(merged, OUTPUT_PATH)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Correlation between H+2 DAMAS error prediction and system imbalance:

    Signed correlation:
      - Actual error vs Imbalance:    r = {stats_results['actual_error_vs_imbalance']:.4f}
      - Predicted error vs Imbalance: r = {stats_results['predicted_error_vs_imbalance']:.4f}

    Absolute correlation:
      - |Actual error| vs |Imbalance|:    r = {stats_results['abs_actual_vs_abs_imbalance']:.4f}
      - |Predicted error| vs |Imbalance|: r = {stats_results['abs_pred_vs_abs_imbalance']:.4f}

    Sign agreement:
      - Actual error sign = Imbalance sign:    {stats_results['sign_match_actual']*100:.1f}%
      - Predicted error sign = Imbalance sign: {stats_results['sign_match_pred']*100:.1f}%

    Predictive power (R-squared):
      - Predicted error alone: {stats_results['r2_simple']:.4f} ({stats_results['r2_simple']*100:.1f}% variance explained)
      - With hour + dow:       {stats_results['r2_with_time']:.4f} ({stats_results['r2_with_time']*100:.1f}% variance explained)

    Statistical significance: p = {stats_results['pearson_p_value']:.2e}
    """)

    # Interpretation
    print("    INTERPRETATION:")
    if abs(stats_results['predicted_error_vs_imbalance']) > 0.3:
        print("    [+] Moderate correlation - error predictions have some predictive value")
    elif abs(stats_results['predicted_error_vs_imbalance']) > 0.1:
        print("    [~] Weak correlation - limited predictive value")
    else:
        print("    [-] Very weak correlation - error predictions don't predict imbalance well")

    if stats_results['sign_match_pred'] > 0.6:
        print("    [+] Sign prediction better than random (50%)")
    else:
        print("    [-] Sign prediction near random")

    return merged, stats_results


if __name__ == "__main__":
    merged, stats = main()
