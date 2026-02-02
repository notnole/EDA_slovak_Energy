"""
Quarterly Nowcast Model (15-minute update frequency)
====================================================

At each quarter (Q0, Q1, Q2, Q3), we predict the same 5-hour horizon.
But for H+1 (current hour), we have partial actual load data!

Strategy:
- H+1: Use partial 3-min load data to extrapolate full hour load
- H+2-H+5: Use model predictions (potentially improved with H+1 estimate)

The more of the current hour that has elapsed, the better our H+1 estimate.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent

print("=" * 70)
print("QUARTERLY NOWCAST MODEL")
print("=" * 70)


def load_data():
    """Load hourly and 3-minute data."""
    print("\n[*] Loading data...")

    # Hourly data
    df_hourly = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
    df_hourly = df_hourly.sort_values('datetime').reset_index(drop=True)
    df_hourly['error'] = df_hourly['actual_load_mw'] - df_hourly['forecast_load_mw']

    # 3-minute data
    df_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'load_3min.csv')
    df_3min['datetime'] = pd.to_datetime(df_3min['datetime'])
    df_3min = df_3min.sort_values('datetime').reset_index(drop=True)
    df_3min['hour_start'] = df_3min['datetime'].dt.floor('h')

    print(f"    Hourly: {len(df_hourly):,} records")
    print(f"    3-min:  {len(df_3min):,} records")

    return df_hourly, df_3min


def compute_partial_hour_estimate(df_3min, hour_start, minutes_elapsed):
    """
    Estimate full hour load using partial 3-min data.

    Args:
        df_3min: 3-minute load data
        hour_start: Start of the hour (e.g., 11:00)
        minutes_elapsed: How many minutes into the hour (15, 30, or 45)

    Returns:
        Estimated full hour average load (MW)
    """
    # Get 3-min observations for this hour up to minutes_elapsed
    hour_data = df_3min[
        (df_3min['hour_start'] == hour_start) &
        (df_3min['datetime'] < hour_start + pd.Timedelta(minutes=minutes_elapsed))
    ]

    if len(hour_data) == 0:
        return np.nan

    # Simple extrapolation: assume remaining load equals average so far
    partial_mean = hour_data['load_mw'].mean()

    # Could also use trend-based extrapolation:
    # trend = (hour_data['load_mw'].iloc[-1] - hour_data['load_mw'].iloc[0]) / len(hour_data)
    # But simple mean works well for load

    return partial_mean


def compute_h1_error_estimate(df_3min, hour_start, minutes_elapsed, forecast_load):
    """
    Estimate H+1 error using partial actual load data.

    error = actual_load - forecast_load

    We estimate actual_load from partial 3-min data.
    """
    estimated_load = compute_partial_hour_estimate(df_3min, hour_start, minutes_elapsed)

    if pd.isna(estimated_load):
        return np.nan

    return estimated_load - forecast_load


def evaluate_quarterly_h1(df_hourly, df_3min):
    """
    Evaluate H+1 prediction accuracy at each quarter.

    Q0 (0 min):  Use model prediction only (baseline)
    Q1 (15 min): Extrapolate from first 15 min of 3-min data
    Q2 (30 min): Extrapolate from first 30 min
    Q3 (45 min): Extrapolate from first 45 min
    """
    print("\n" + "=" * 70)
    print("H+1 PREDICTION BY QUARTER POSITION (DAMAS BASELINE)")
    print("=" * 70)

    results = []

    # Get hours where we have BOTH hourly and 3-min data in 2025
    df_3min['hour_start'] = df_3min['datetime'].dt.floor('h')
    hours_with_3min = set(df_3min[df_3min['datetime'].dt.year >= 2025]['hour_start'].unique())

    test_df = df_hourly[df_hourly['datetime'].dt.year >= 2025].copy()
    test_df = test_df[test_df['datetime'].isin(hours_with_3min)]

    print(f"  [*] Testing on {len(test_df)} hours with 3-min data")

    for quarter, minutes in [(0, 0), (1, 15), (2, 30), (3, 45)]:
        errors_model = []  # Model-only prediction error
        errors_extrap = []  # Extrapolation prediction error

        for _, row in test_df.iterrows():
            hour_start = row['datetime']
            actual_load = row['actual_load_mw']
            forecast_load = row['forecast_load_mw']
            actual_error = actual_load - forecast_load

            # Model prediction (baseline DAMAS error = 0, or use our model)
            # For now, use DAMAS baseline (predict error = 0)
            model_pred_error = 0
            errors_model.append(abs(actual_error - model_pred_error))

            if minutes > 0:
                # Extrapolation from partial data
                estimated_load = compute_partial_hour_estimate(df_3min, hour_start, minutes)
                if not pd.isna(estimated_load):
                    estimated_error = estimated_load - forecast_load
                    errors_extrap.append(abs(actual_error - estimated_error))

        mae_model = np.nanmean(errors_model) if errors_model else np.nan
        mae_extrap = np.nanmean(errors_extrap) if errors_extrap else np.nan

        results.append({
            'quarter': quarter,
            'minutes_elapsed': minutes,
            'mae_baseline': mae_model,
            'mae_extrapolation': mae_extrap,
            'improvement': (mae_model - mae_extrap) / mae_model * 100 if mae_extrap else 0,
            'n_samples': len(errors_extrap) if minutes > 0 else len(errors_model)
        })

        print(f"\n  Q{quarter} ({minutes:2d} min elapsed):")
        print(f"    Baseline (DAMAS):      {mae_model:.1f} MW MAE")
        if minutes > 0:
            print(f"    Extrapolation:         {mae_extrap:.1f} MW MAE")
            print(f"    Improvement:           {(mae_model - mae_extrap) / mae_model * 100:+.1f}%")

    return pd.DataFrame(results)


def evaluate_with_model_baseline(df_hourly, df_3min):
    """
    Compare extrapolation against our trained model baseline.
    """
    print("\n" + "=" * 70)
    print("COMPARING EXTRAPOLATION VS MODEL PREDICTION")
    print("=" * 70)

    # Load our trained model predictions (if available)
    # For now, simulate with simple persistence model

    test_data = df_hourly[df_hourly['datetime'].dt.year >= 2025].copy()
    test_data['error_lag1'] = test_data['error'].shift(1)  # Previous hour error
    test_data = test_data.dropna()

    results = []

    for quarter, minutes in [(0, 0), (1, 15), (2, 30), (3, 45)]:
        errors_model = []
        errors_extrap = []
        errors_combined = []

        for _, row in test_data.iterrows():
            hour_start = row['datetime']
            actual_error = row['error']
            forecast_load = row['forecast_load_mw']

            # Model prediction: use previous hour error as simple model
            model_pred = row['error_lag1']
            errors_model.append(abs(actual_error - model_pred))

            if minutes > 0:
                # Extrapolation
                estimated_load = compute_partial_hour_estimate(df_3min, hour_start, minutes)
                if not pd.isna(estimated_load):
                    extrap_pred = estimated_load - forecast_load
                    errors_extrap.append(abs(actual_error - extrap_pred))

                    # Combined: weighted average of model and extrapolation
                    # Weight extrapolation more as we get more data
                    weight_extrap = minutes / 60  # 0.25 at Q1, 0.50 at Q2, 0.75 at Q3
                    combined_pred = weight_extrap * extrap_pred + (1 - weight_extrap) * model_pred
                    errors_combined.append(abs(actual_error - combined_pred))

        mae_model = np.mean(errors_model)
        mae_extrap = np.mean(errors_extrap) if errors_extrap else np.nan
        mae_combined = np.mean(errors_combined) if errors_combined else np.nan

        results.append({
            'quarter': quarter,
            'minutes': minutes,
            'mae_model': mae_model,
            'mae_extrapolation': mae_extrap,
            'mae_combined': mae_combined,
        })

        print(f"\n  Q{quarter} ({minutes:2d} min):")
        print(f"    Model (lag1):          {mae_model:.1f} MW")
        if minutes > 0:
            print(f"    Extrapolation only:    {mae_extrap:.1f} MW")
            print(f"    Combined (weighted):   {mae_combined:.1f} MW")

    return pd.DataFrame(results)


def analyze_extrapolation_accuracy(df_hourly, df_3min):
    """
    Deep dive: How accurate is extrapolation at each quarter?
    """
    print("\n" + "=" * 70)
    print("EXTRAPOLATION ACCURACY ANALYSIS")
    print("=" * 70)

    # Get hours where we have 3-min data
    df_3min['hour_start'] = df_3min['datetime'].dt.floor('h')
    hours_with_3min = set(df_3min[df_3min['datetime'].dt.year >= 2025]['hour_start'].unique())

    test_data = df_hourly[df_hourly['datetime'].dt.year >= 2025].copy()
    test_data = test_data[test_data['datetime'].isin(hours_with_3min)]

    for minutes in [15, 30, 45]:
        errors = []

        for _, row in test_data.iterrows():
            hour_start = row['datetime']
            actual_load = row['actual_load_mw']

            estimated_load = compute_partial_hour_estimate(df_3min, hour_start, minutes)
            if not pd.isna(estimated_load):
                errors.append(actual_load - estimated_load)

        errors = np.array(errors)

        print(f"\n  {minutes} min extrapolation ({minutes}/60 = {minutes/60:.0%} of hour):")
        print(f"    MAE:  {np.nanmean(np.abs(errors)):.1f} MW")
        print(f"    RMSE: {np.sqrt(np.nanmean(errors**2)):.1f} MW")
        print(f"    Bias: {np.nanmean(errors):+.1f} MW")
        print(f"    Std:  {np.nanstd(errors):.1f} MW")


def main():
    df_hourly, df_3min = load_data()

    # Basic quarterly evaluation
    results_basic = evaluate_quarterly_h1(df_hourly, df_3min)

    # Compare with model
    results_model = evaluate_with_model_baseline(df_hourly, df_3min)

    # Extrapolation accuracy
    analyze_extrapolation_accuracy(df_hourly, df_3min)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: H+1 PREDICTION BY QUARTER")
    print("=" * 70)
    print("")
    print("  Quarter | Min Elapsed | Model MAE | Extrap MAE | Combined MAE")
    print("  --------|-------------|-----------|------------|-------------")
    for _, r in results_model.iterrows():
        extrap = f"{r['mae_extrapolation']:.1f} MW" if not pd.isna(r['mae_extrapolation']) else "N/A"
        combined = f"{r['mae_combined']:.1f} MW" if not pd.isna(r['mae_combined']) else "N/A"
        print(f"  Q{int(r['quarter'])}      | {int(r['minutes']):2d} min       | {r['mae_model']:.1f} MW   | {extrap:10s} | {combined}")
    print("")

    print("""
RECOMMENDATION:
- Q0: Use trained model (no partial data available)
- Q1-Q3: Blend model prediction with extrapolation
- Weight extrapolation by (minutes_elapsed / 60)
- At Q3 (45 min), extrapolation dominates (~75% weight)

For H+2 to H+5:
- No partial data available
- Use trained model predictions
- Could adjust based on H+1 extrapolation trend
""")

    return results_basic, results_model


if __name__ == "__main__":
    results_basic, results_model = main()
