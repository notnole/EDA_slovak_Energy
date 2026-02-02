"""
Production Performance Analysis for Imbalance Nowcasting Model
Analyzes real-world deployment performance from Jan 29-30, 2026

IMPORTANT: Predictions are in UTC, actuals are in CET (UTC+1)
Settlement period is derived from: timestamp + lead_time = period END
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "ProductionData"

def load_predictions():
    """Load model predictions from production (UTC timestamps)."""
    df = pd.read_csv(DATA_DIR / "beam_lite_predictions.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # quarter_hour is a string like "15:15" representing the settlement period in UTC
    return df

def load_actuals():
    """Load actual imbalance values from OKTE (CET timezone)."""
    df = pd.read_csv(DATA_DIR / "SystemImbalance_2026-01-28_2026-01-30.csv", sep=';')
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    # Settlement term 1-96 maps to 00:00-23:45 CET in 15-min intervals
    # Keep in CET (naive datetime) for matching with predictions
    df['settlement_start'] = df['Date'] + pd.to_timedelta((df['Settlement Term'] - 1) * 15, unit='min')
    df['actual_imbalance'] = df['System Imbalance (MWh)']
    return df[['settlement_start', 'actual_imbalance', 'Settlement Term']]

def merge_predictions_actuals(preds, actuals):
    """Merge predictions with actual values by settlement period.

    IMPORTANT: quarter_hour in predictions is in UTC.
    OKTE actuals are in CET (UTC+1 in winter).
    So we add 1 hour to quarter_hour to convert to CET for matching.
    """
    preds = preds.copy()

    # quarter_hour is in UTC, convert to CET by adding 1 hour
    preds['pred_date'] = preds['timestamp'].dt.strftime('%Y-%m-%d')
    preds['settlement_start'] = pd.to_datetime(
        preds['pred_date'] + ' ' + preds['quarter_hour'].astype(str)
    ) + pd.Timedelta(hours=1)  # UTC -> CET

    # Merge with actuals (which use settlement_start in CET)
    merged = preds.merge(actuals, on='settlement_start', how='left')
    return merged

def calculate_metrics(df, groupby_col=None):
    """Calculate MAE, RMSE, and bias metrics."""
    df = df.dropna(subset=['prediction_mwh', 'actual_imbalance'])

    if len(df) == 0:
        return pd.DataFrame()

    df['error'] = df['prediction_mwh'] - df['actual_imbalance']
    df['abs_error'] = df['error'].abs()
    df['squared_error'] = df['error'] ** 2

    # Baseline errors
    df['baseline_error'] = df['baseline_pred'] - df['actual_imbalance']
    df['baseline_abs_error'] = df['baseline_error'].abs()

    if groupby_col:
        metrics = df.groupby(groupby_col).agg(
            count=('error', 'count'),
            mae=('abs_error', 'mean'),
            rmse=('squared_error', lambda x: np.sqrt(x.mean())),
            bias=('error', 'mean'),
            baseline_mae=('baseline_abs_error', 'mean'),
        ).round(3)
        metrics['improvement_pct'] = ((metrics['baseline_mae'] - metrics['mae']) / metrics['baseline_mae'] * 100).round(1)
    else:
        metrics = {
            'count': len(df),
            'mae': df['abs_error'].mean(),
            'rmse': np.sqrt(df['squared_error'].mean()),
            'bias': df['error'].mean(),
            'baseline_mae': df['baseline_abs_error'].mean(),
        }
        metrics['improvement_pct'] = (metrics['baseline_mae'] - metrics['mae']) / metrics['baseline_mae'] * 100

    return metrics

def main():
    print("=" * 60)
    print("PRODUCTION PERFORMANCE ANALYSIS")
    print("Imbalance Nowcasting Model - Real Deployment Results")
    print("=" * 60)

    # Load data
    print("\n[*] Loading data...")
    preds = load_predictions()
    actuals = load_actuals()

    print(f"   Predictions: {len(preds)} records")
    print(f"   Date range: {preds['timestamp'].min()} to {preds['timestamp'].max()}")
    print(f"   Actual imbalances: {len(actuals)} settlement periods")

    # Merge
    merged = merge_predictions_actuals(preds, actuals)
    valid = merged.dropna(subset=['actual_imbalance'])
    print(f"   Matched predictions: {len(valid)}")

    # Overall metrics
    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE")
    print("=" * 60)
    overall = calculate_metrics(valid)
    print(f"\n   Total predictions evaluated: {overall['count']}")
    print(f"   Model MAE:    {overall['mae']:.3f} MWh")
    print(f"   Model RMSE:   {overall['rmse']:.3f} MWh")
    print(f"   Model Bias:   {overall['bias']:.3f} MWh")
    print(f"   Baseline MAE: {overall['baseline_mae']:.3f} MWh")
    print(f"   Improvement:  {overall['improvement_pct']:.1f}%")

    # By lead time
    print("\n" + "=" * 60)
    print("PERFORMANCE BY LEAD TIME")
    print("=" * 60)
    by_lead = calculate_metrics(valid, 'lead_time_min')
    print("\n" + by_lead.to_string())

    # Statistical significance check
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS")
    print("=" * 60)

    # Check for systematic patterns
    print("\n[*] Error distribution:")
    print(f"   Mean error:   {valid['prediction_mwh'].mean() - valid['actual_imbalance'].mean():.3f} MWh")
    print(f"   Std error:    {(valid['prediction_mwh'] - valid['actual_imbalance']).std():.3f} MWh")

    # Hour-of-day patterns
    valid['hour'] = valid['timestamp'].dt.hour
    by_hour = valid.groupby('hour').apply(
        lambda x: pd.Series({
            'mae': (x['prediction_mwh'] - x['actual_imbalance']).abs().mean(),
            'count': len(x)
        })
    )

    print("\n[*] MAE by hour of day:")
    print("   Hour | MAE (MWh) | Count")
    print("   " + "-" * 30)
    for hour, row in by_hour.iterrows():
        print(f"   {hour:4d} | {row['mae']:9.3f} | {int(row['count']):5d}")

    # Lead 12 specific analysis (hardest case)
    print("\n" + "=" * 60)
    print("LEAD 12 ANALYSIS (Most Challenging)")
    print("=" * 60)
    lead12 = valid[valid['lead_time_min'] == 12]
    if len(lead12) > 0:
        lead12_metrics = calculate_metrics(lead12)
        print(f"\n   Predictions: {lead12_metrics['count']}")
        print(f"   Model MAE:    {lead12_metrics['mae']:.3f} MWh")
        print(f"   Baseline MAE: {lead12_metrics['baseline_mae']:.3f} MWh")
        print(f"   Improvement:  {lead12_metrics['improvement_pct']:.1f}%")

        # Compare to historical baseline MAE of ~6.6 MWh
        historical_baseline = 6.6
        print(f"\n   Historical baseline MAE: ~{historical_baseline} MWh")
        if lead12_metrics['mae'] < historical_baseline:
            print(f"   [OK] Model beats historical baseline by {(historical_baseline - lead12_metrics['mae']):.2f} MWh")
        else:
            print(f"   [!!] Model underperforms historical baseline by {(lead12_metrics['mae'] - historical_baseline):.2f} MWh")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n   Deployment period: ~1.5 days of live predictions")
    print(f"   Total settlement periods covered: {len(valid['settlement_start'].unique())}")

    if overall['improvement_pct'] > 0:
        print(f"\n   [OK] Model OUTPERFORMS baseline by {overall['improvement_pct']:.1f}%")
    else:
        print(f"\n   [!!] Model UNDERPERFORMS baseline by {-overall['improvement_pct']:.1f}%")

    return valid, by_lead

if __name__ == "__main__":
    valid, by_lead = main()
