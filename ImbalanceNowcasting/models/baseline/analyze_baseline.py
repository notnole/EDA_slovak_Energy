"""
Baseline Model Analysis

Implements and analyzes the deterministic baseline predictor defined in models/BaseLine.md.
Prediction = -0.25 × weighted_avg(regulation) at each lead time.

The model transitions from heuristic estimation to precise measurement as QH progresses.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\models\baseline")
DATA_DIR = OUTPUT_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load regulation and labels."""
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
        columns={'System Imbalance (MWh)': 'imbalance'}
    )
    return reg_df, label_df


def create_qh_observations(reg_df):
    """
    Create dataset with regulation observations at each minute within QH.
    Minutes: 0, 3, 6, 9, 12 (5 observations per 15-min period)

    NOTE: Label datetime is the START of settlement period (e.g., 00:00 for period 00:00-00:15)
    Regulation at 00:03 belongs to settlement starting at 00:00 (ending at 00:15)
    """
    reg_df = reg_df.copy()

    # Floor to 3-minute marks
    reg_df['datetime'] = reg_df['datetime'].dt.floor('3min')

    # Settlement period end (ceiling to 15 min)
    reg_df['settlement_end'] = reg_df['datetime'].dt.ceil('15min')

    # Handle edge case where datetime is exactly on 15-min mark
    mask = reg_df['datetime'] == reg_df['settlement_end']
    reg_df.loc[mask, 'settlement_end'] = reg_df.loc[mask, 'datetime'] + pd.Timedelta(minutes=15)

    # Settlement START = END - 15 min (to match label datetime format)
    reg_df['settlement_start'] = reg_df['settlement_end'] - pd.Timedelta(minutes=15)

    # Minute within settlement (0, 3, 6, 9, 12)
    reg_df['minute_in_qh'] = (reg_df['datetime'] - reg_df['settlement_start']).dt.total_seconds() / 60
    reg_df['minute_in_qh'] = reg_df['minute_in_qh'].astype(int)

    # Pivot to get one row per settlement period with all observations
    pivot_df = reg_df.pivot_table(
        index='settlement_start',
        columns='minute_in_qh',
        values='regulation_mw',
        aggfunc='first'
    ).reset_index()

    pivot_df.columns = ['settlement_start'] + [f'reg_min{int(c)}' for c in pivot_df.columns[1:]]

    return pivot_df


def compute_baseline_predictions(df):
    """
    Compute baseline predictions at each lead time per the spec:
    - 15 min lead (:00): 0.25 * current (last known = previous QH's last)
    - 12 min lead (:03): 0.25 * reg_min0
    - 9 min lead (:06): 0.25 * (0.8*reg_min3 + 0.2*reg_min0)
    - 6 min lead (:09): 0.25 * (0.6*reg_min6 + 0.2*reg_min3 + 0.2*reg_min0)
    - 3 min lead (:12): 0.25 * (0.4*reg_min9 + 0.2*reg_min6 + 0.2*reg_min3 + 0.2*reg_min0)

    Note: Imbalance = -regulation (inverse relationship), so prediction = -0.25 * weighted_reg
    """
    df = df.copy()

    # Previous QH's last observation (for 15-min lead prediction)
    df['reg_prev_qh'] = df['reg_min12'].shift(1)

    # Lead 15 min (at :00, before any observation in current QH)
    # Use last known value from previous QH
    df['pred_lead15'] = -0.25 * df['reg_prev_qh']

    # Lead 12 min (at :03, after first observation)
    df['pred_lead12'] = -0.25 * df['reg_min0']

    # Lead 9 min (at :06, after second observation)
    df['pred_lead9'] = -0.25 * (0.8 * df['reg_min3'] + 0.2 * df['reg_min0'])

    # Lead 6 min (at :09, after third observation)
    df['pred_lead6'] = -0.25 * (0.6 * df['reg_min6'] + 0.2 * df['reg_min3'] + 0.2 * df['reg_min0'])

    # Lead 3 min (at :12, after fourth observation)
    df['pred_lead3'] = -0.25 * (0.4 * df['reg_min9'] + 0.2 * df['reg_min6'] + 0.2 * df['reg_min3'] + 0.2 * df['reg_min0'])

    # Lead 0 min (at :15, with all 5 observations - "perfect" measurement)
    df['pred_lead0'] = -0.25 * (
        0.2 * df['reg_min12'] + 0.2 * df['reg_min9'] +
        0.2 * df['reg_min6'] + 0.2 * df['reg_min3'] + 0.2 * df['reg_min0']
    )

    return df


def evaluate_predictions(df, label_df):
    """Merge with labels and compute error metrics."""
    # Merge (settlement_start matches label datetime)
    df = df.rename(columns={'settlement_start': 'datetime'})
    merged = pd.merge(df, label_df, on='datetime', how='inner')

    # Compute errors for each lead time
    lead_times = [15, 12, 9, 6, 3, 0]
    results = []

    for lead in lead_times:
        pred_col = f'pred_lead{lead}'
        valid = merged[['imbalance', pred_col]].dropna()

        if len(valid) < 100:
            continue

        y_true = valid['imbalance']
        y_pred = valid[pred_col]

        error = y_true - y_pred

        mae = np.abs(error).mean()
        rmse = np.sqrt((error ** 2).mean())
        mbe = error.mean()  # Mean Bias Error
        r2 = 1 - (error ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()
        corr = np.corrcoef(y_true, y_pred)[0, 1]

        results.append({
            'lead_time_min': lead,
            'n_samples': len(valid),
            'mae': mae,
            'rmse': rmse,
            'mbe': mbe,
            'r2': r2,
            'correlation': corr
        })

        # Store errors for later analysis
        merged[f'error_lead{lead}'] = merged['imbalance'] - merged[pred_col]

    return merged, pd.DataFrame(results)


def analyze_error_patterns(df, results_df):
    """Analyze when and where baseline errors occur."""
    # Add time features
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['dow'] >= 5
    df['month'] = df['datetime'].dt.month

    patterns = {}

    # Focus on lead 12 and lead 3 (where baseline has most/least error)
    for lead in [12, 3]:
        error_col = f'error_lead{lead}'
        valid = df[['hour', 'dow', 'is_weekend', 'imbalance', error_col]].dropna()

        # By hour
        hour_errors = valid.groupby('hour')[error_col].agg(['mean', 'std', 'count'])
        hour_errors.columns = ['mean_error', 'std_error', 'count']
        patterns[f'lead{lead}_by_hour'] = hour_errors.reset_index()

        # By imbalance magnitude
        valid['imb_abs'] = valid['imbalance'].abs()
        valid['imb_quartile'] = pd.qcut(valid['imb_abs'], 4, labels=['Small', 'Med-Small', 'Med-Large', 'Large'])
        imb_errors = valid.groupby('imb_quartile', observed=True)[error_col].agg(['mean', 'std', 'count'])
        imb_errors.columns = ['mean_error', 'std_error', 'count']
        patterns[f'lead{lead}_by_imbalance'] = imb_errors.reset_index()

    return patterns


def main():
    print("=" * 70)
    print("BASELINE MODEL ANALYSIS")
    print("=" * 70)
    print("\nModel: Prediction = -0.25 * weighted_avg(regulation)")
    print("Based on specification in models/BaseLine.md")

    reg_df, label_df = load_data()
    print(f"\nLoaded {len(reg_df):,} regulation observations")
    print(f"Loaded {len(label_df):,} imbalance labels")

    # Create QH observations
    qh_df = create_qh_observations(reg_df)
    print(f"\nCreated {len(qh_df):,} quarter-hour periods")

    # Check observation coverage
    obs_cols = [c for c in qh_df.columns if c.startswith('reg_min')]
    coverage = qh_df[obs_cols].notna().mean()
    print("\nObservation coverage per minute:")
    for col, cov in coverage.items():
        print(f"  {col}: {cov*100:.1f}%")

    # Compute baseline predictions
    qh_df = compute_baseline_predictions(qh_df)

    # Evaluate
    merged_df, results_df = evaluate_predictions(qh_df, label_df)
    print(f"\nEvaluated on {len(merged_df):,} settlement periods")

    # =================================================================
    # RESULTS BY LEAD TIME
    # =================================================================
    print("\n" + "=" * 70)
    print("BASELINE PERFORMANCE BY LEAD TIME")
    print("=" * 70)

    print("\n| Lead Time | MAE (MWh) | RMSE (MWh) | R2    | Bias  |")
    print("|-----------|-----------|------------|-------|-------|")
    for _, row in results_df.iterrows():
        print(f"| {int(row['lead_time_min']):2d} min     | {row['mae']:9.2f} | {row['rmse']:10.2f} | {row['r2']:.3f} | {row['mbe']:+.2f} |")

    results_df.to_csv(DATA_DIR / 'baseline_performance.csv', index=False)

    # =================================================================
    # ERROR PATTERNS
    # =================================================================
    print("\n" + "=" * 70)
    print("ERROR PATTERN ANALYSIS")
    print("=" * 70)

    patterns = analyze_error_patterns(merged_df, results_df)

    # Save patterns
    for name, pattern_df in patterns.items():
        pattern_df.to_csv(DATA_DIR / f'error_{name}.csv', index=False)

    # Report key patterns
    print("\nError by Imbalance Magnitude (Lead 12 min):")
    imb_df = patterns['lead12_by_imbalance']
    for _, row in imb_df.iterrows():
        print(f"  {row['imb_quartile']:10s}: MAE ~ {abs(row['mean_error']):.2f}, Std = {row['std_error']:.2f}")

    print("\nError by Imbalance Magnitude (Lead 3 min):")
    imb_df = patterns['lead3_by_imbalance']
    for _, row in imb_df.iterrows():
        print(f"  {row['imb_quartile']:10s}: MAE ~ {abs(row['mean_error']):.2f}, Std = {row['std_error']:.2f}")

    # =================================================================
    # VISUALIZATION
    # =================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Baseline Model Analysis: -0.25 * weighted_avg(regulation)',
                 fontsize=14, fontweight='bold')

    # Plot 1: MAE by lead time
    ax = axes[0, 0]
    ax.bar(results_df['lead_time_min'], results_df['mae'], color='tab:blue', alpha=0.7)
    ax.set_xlabel('Lead Time (minutes)')
    ax.set_ylabel('MAE (MWh)')
    ax.set_title('MAE by Lead Time')
    ax.invert_xaxis()  # Higher lead time on left
    for i, row in results_df.iterrows():
        ax.text(row['lead_time_min'], row['mae'] + 0.2, f"{row['mae']:.2f}", ha='center')

    # Plot 2: R2 by lead time
    ax = axes[0, 1]
    ax.bar(results_df['lead_time_min'], results_df['r2'], color='tab:green', alpha=0.7)
    ax.set_xlabel('Lead Time (minutes)')
    ax.set_ylabel('R2')
    ax.set_title('R2 by Lead Time')
    ax.invert_xaxis()
    ax.set_ylim(0, 1)
    for i, row in results_df.iterrows():
        ax.text(row['lead_time_min'], row['r2'] + 0.02, f"{row['r2']:.2f}", ha='center')

    # Plot 3: Predicted vs Actual (Lead 3 min)
    ax = axes[0, 2]
    sample = merged_df[['imbalance', 'pred_lead3']].dropna().sample(min(3000, len(merged_df)), random_state=42)
    ax.scatter(sample['imbalance'], sample['pred_lead3'], alpha=0.3, s=10)
    ax.plot([-60, 60], [-60, 60], 'r--', label='Perfect')
    ax.set_xlabel('Actual Imbalance (MWh)')
    ax.set_ylabel('Predicted (MWh)')
    ax.set_title('Predicted vs Actual (Lead 3 min)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)

    # Plot 4: Predicted vs Actual (Lead 12 min)
    ax = axes[1, 0]
    sample = merged_df[['imbalance', 'pred_lead12']].dropna().sample(min(3000, len(merged_df)), random_state=42)
    ax.scatter(sample['imbalance'], sample['pred_lead12'], alpha=0.3, s=10)
    ax.plot([-60, 60], [-60, 60], 'r--', label='Perfect')
    ax.set_xlabel('Actual Imbalance (MWh)')
    ax.set_ylabel('Predicted (MWh)')
    ax.set_title('Predicted vs Actual (Lead 12 min)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)

    # Plot 5: Error distribution by lead time
    ax = axes[1, 1]
    error_cols = [f'error_lead{l}' for l in [12, 9, 6, 3]]
    error_data = [merged_df[col].dropna().values for col in error_cols]
    bp = ax.boxplot(error_data, tick_labels=['12 min', '9 min', '6 min', '3 min'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#e74c3c', '#f39c12', '#f1c40f', '#27ae60']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lead Time')
    ax.set_ylabel('Error (MWh)')
    ax.set_title('Error Distribution by Lead Time')
    ax.set_ylim(-30, 30)

    # Plot 6: Error by hour (Lead 12 min)
    ax = axes[1, 2]
    hour_df = patterns['lead12_by_hour']
    ax.bar(hour_df['hour'], hour_df['mean_error'], color='tab:purple', alpha=0.7)
    ax.fill_between(hour_df['hour'],
                    hour_df['mean_error'] - hour_df['std_error']/np.sqrt(hour_df['count']),
                    hour_df['mean_error'] + hour_df['std_error']/np.sqrt(hour_df['count']),
                    alpha=0.3, color='tab:purple')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Mean Error (MWh)')
    ax.set_title('Error Pattern by Hour (Lead 12 min)')
    ax.set_xticks(range(0, 24, 3))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'baseline_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: baseline_analysis.png")

    # =================================================================
    # TIME SERIES OF ERRORS
    # =================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Baseline Model: Sample Predictions', fontsize=14, fontweight='bold')

    # Sample week
    sample_start = '2025-11-10'
    sample_end = '2025-11-17'
    sample = merged_df[(merged_df['datetime'] >= sample_start) & (merged_df['datetime'] < sample_end)]

    # Plot actual vs predicted (lead 12)
    ax = axes[0]
    ax.plot(sample['datetime'], sample['imbalance'], 'b-', label='Actual', alpha=0.7)
    ax.plot(sample['datetime'], sample['pred_lead12'], 'r--', label='Predicted (12 min)', alpha=0.7)
    ax.set_ylabel('Imbalance (MWh)')
    ax.set_title('Lead 12 min: Early Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot actual vs predicted (lead 3)
    ax = axes[1]
    ax.plot(sample['datetime'], sample['imbalance'], 'b-', label='Actual', alpha=0.7)
    ax.plot(sample['datetime'], sample['pred_lead3'], 'g--', label='Predicted (3 min)', alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Imbalance (MWh)')
    ax.set_title('Lead 3 min: Near-Measurement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'baseline_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: baseline_timeseries.png")

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    lead12 = results_df[results_df['lead_time_min'] == 12].iloc[0]
    lead3 = results_df[results_df['lead_time_min'] == 3].iloc[0]

    print(f"\nBaseline Performance:")
    print(f"  Lead 12 min (early):  MAE = {lead12['mae']:.2f} MWh, R2 = {lead12['r2']:.3f}")
    print(f"  Lead 3 min (late):    MAE = {lead3['mae']:.2f} MWh, R2 = {lead3['r2']:.3f}")

    print(f"\nModel Characteristics:")
    print(f"  - At 3 min lead: effectively measuring, not predicting")
    print(f"  - At 12 min lead: heuristic, cannot anticipate reversals")
    print(f"  - Improvement from 12->3 min: {(lead12['mae'] - lead3['mae'])/lead12['mae']*100:.0f}% MAE reduction")

    print(f"\nThis is the baseline to beat with ML models at 12+ min lead times.")
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
