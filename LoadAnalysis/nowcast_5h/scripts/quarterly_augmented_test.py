"""
Quarterly Augmented Model Test
==============================

Keep true error_lag0, ADD extrapolated error as additional features.

Features:
- error_lag0: TRUE error from previous hour (always available)
- quarter: which point in hour (0, 1, 2, 3)
- extrap_q1: extrapolated H+1 error from 15 min data (0 at Q0, filled at Q1+)
- extrap_q2: extrapolated H+1 error from 30 min data (0 at Q0-Q1, filled at Q2+)
- extrap_q3: extrapolated H+1 error from 45 min data (0 at Q0-Q2, filled at Q3)

The model learns WHEN to use the extrapolated signals.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent

print("=" * 70)
print("QUARTERLY AUGMENTED MODEL TEST")
print("Keep true error_lag0 + add extrapolated features")
print("=" * 70)


def load_data():
    """Load hourly and 3-minute data."""
    print("\n[*] Loading data...")

    df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek

    df_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'load_3min.csv')
    df_3min['datetime'] = pd.to_datetime(df_3min['datetime'])
    df_3min = df_3min.sort_values('datetime').reset_index(drop=True)
    df_3min['hour_start'] = df_3min['datetime'].dt.floor('h')

    print(f"    Hourly: {len(df):,} records")
    print(f"    3-min:  {len(df_3min):,} records")

    return df, df_3min


def create_base_features(df):
    """Create base features (without quarter-specific ones)."""
    df = df.copy()

    for lag in range(0, 25):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    df['error_roll_mean_24h'] = df['error'].shift(1).rolling(24).mean()
    df['error_roll_std_24h'] = df['error'].shift(1).rolling(24).std()

    for h in range(1, 6):
        df[f'target_h{h}'] = df['error'].shift(-h)

    return df


def compute_extrapolated_error(df_3min, hour_start, minutes_elapsed, forecast_load):
    """Get extrapolated error for current hour from partial 3-min data."""
    # Get the NEXT hour's partial data (H+1 is the hour starting at hour_start + 1h)
    next_hour = hour_start + pd.Timedelta(hours=1)

    hour_data = df_3min[
        (df_3min['hour_start'] == next_hour) &
        (df_3min['datetime'] < next_hour + pd.Timedelta(minutes=minutes_elapsed))
    ]

    if len(hour_data) == 0:
        return np.nan

    estimated_load = hour_data['load_mw'].mean()

    # We need the forecast for the NEXT hour (H+1)
    # For now, approximate - in production you'd look up the actual forecast
    return estimated_load - forecast_load  # This is approximate


def compute_extrapolated_error_current_hour(df_3min, current_hour, minutes_elapsed, forecast_load):
    """Get extrapolated error for the CURRENT hour being predicted."""
    hour_data = df_3min[
        (df_3min['hour_start'] == current_hour) &
        (df_3min['datetime'] < current_hour + pd.Timedelta(minutes=minutes_elapsed))
    ]

    if len(hour_data) == 0:
        return np.nan

    estimated_load = hour_data['load_mw'].mean()
    return estimated_load - forecast_load


def build_quarterly_dataset(df, df_3min):
    """
    Build a dataset with quarterly features.
    Each hourly record gets expanded into 4 records (one per quarter).
    """
    print("\n[*] Building quarterly dataset...")

    df_3min['hour_start'] = df_3min['datetime'].dt.floor('h')

    # Create a lookup for H+1 forecast
    df['h1_forecast'] = df['forecast_load_mw'].shift(-1)

    records = []

    # Process each hour
    for _, row in df.iterrows():
        hour_start = row['datetime']  # This is the prediction time (e.g., 10:00 meaning hour 10-11 just ended)
        # H+1 is the NEXT hour (11:00-12:00)
        h1_hour = hour_start + pd.Timedelta(hours=1)

        # Get CORRECT H+1 forecast from the next row
        h1_forecast = row['h1_forecast']
        if pd.isna(h1_forecast):
            continue  # Skip if no H+1 forecast available

        for quarter in [0, 1, 2, 3]:
            minutes = quarter * 15

            rec = {
                'datetime': hour_start,
                'quarter': quarter,
                'year': row['year'],
                'month': row['month'],
                'hour': row['hour'],
                'dow': row['dow'],
                'forecast_load_mw': row['forecast_load_mw'],
                'h1_forecast': h1_forecast,
            }

            # Copy base features
            for col in df.columns:
                if col.startswith('error_lag') or col.startswith('error_roll') or col.startswith('target_'):
                    rec[col] = row[col]

            # Add extrapolated features (progressively filled)
            # Using CORRECT H+1 forecast for extrapolation
            # extrap_q1: available at Q1, Q2, Q3
            # extrap_q2: available at Q2, Q3
            # extrap_q3: available at Q3

            if quarter >= 1:
                rec['extrap_q1'] = compute_extrapolated_error_current_hour(
                    df_3min, h1_hour, 15, h1_forecast
                )
            else:
                rec['extrap_q1'] = 0.0

            if quarter >= 2:
                rec['extrap_q2'] = compute_extrapolated_error_current_hour(
                    df_3min, h1_hour, 30, h1_forecast
                )
            else:
                rec['extrap_q2'] = 0.0

            if quarter >= 3:
                rec['extrap_q3'] = compute_extrapolated_error_current_hour(
                    df_3min, h1_hour, 45, h1_forecast
                )
            else:
                rec['extrap_q3'] = 0.0

            records.append(rec)

    df_quarterly = pd.DataFrame(records)
    print(f"    Created {len(df_quarterly):,} quarterly records")

    return df_quarterly


def train_and_evaluate(df_quarterly):
    """Train augmented models and evaluate per quarter."""
    print("\n[*] Training augmented models...")

    base_features = [
        'error_lag0', 'error_lag1', 'error_lag2', 'error_lag3',
        'error_lag24', 'error_roll_mean_24h', 'hour', 'dow'
    ]

    augmented_features = base_features + ['quarter', 'extrap_q1', 'extrap_q2', 'extrap_q3']

    # Train on 2024
    train = df_quarterly[df_quarterly['year'] == 2024].copy()
    train = train.dropna(subset=augmented_features + ['target_h1'])

    # Test on 2025+
    test = df_quarterly[df_quarterly['year'] >= 2025].copy()
    test = test.dropna(subset=augmented_features + ['target_h1'])

    print(f"    Train: {len(train):,} records")
    print(f"    Test:  {len(test):,} records")

    results = []

    for h in range(1, 6):
        target_col = f'target_h{h}'

        # Train BASE model (without quarter features)
        model_base = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            verbosity=-1, random_state=42
        )
        model_base.fit(train[base_features], train[target_col])

        # Train AUGMENTED model (with quarter features)
        model_aug = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            verbosity=-1, random_state=42
        )
        model_aug.fit(train[augmented_features], train[target_col])

        # Evaluate per quarter
        for quarter in [0, 1, 2, 3]:
            test_q = test[test['quarter'] == quarter].copy()

            if len(test_q) == 0:
                continue

            pred_base = model_base.predict(test_q[base_features])
            pred_aug = model_aug.predict(test_q[augmented_features])

            actual = test_q[target_col].values

            mae_base = np.nanmean(np.abs(actual - pred_base))
            mae_aug = np.nanmean(np.abs(actual - pred_aug))

            results.append({
                'horizon': h,
                'quarter': quarter,
                'mae_base': mae_base,
                'mae_augmented': mae_aug,
                'improvement': mae_base - mae_aug
            })

    return pd.DataFrame(results)


def print_results(results_df):
    """Print results in a nice table."""
    print("\n" + "=" * 70)
    print("RESULTS: BASE vs AUGMENTED MODEL")
    print("=" * 70)

    print("\n  Base model: error_lag0 + standard features")
    print("  Augmented:  Base + quarter + extrap_q1/q2/q3")

    for h in range(1, 6):
        print(f"\n  --- H+{h} ---")
        print(f"  Quarter | Base MAE | Aug MAE  | Improvement")
        print(f"  --------|----------|----------|------------")

        for q in range(4):
            row = results_df[(results_df['horizon'] == h) & (results_df['quarter'] == q)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            print(f"  Q{q}      | {row['mae_base']:6.1f}   | {row['mae_augmented']:6.1f}   | {row['improvement']:+.1f} MW")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Average improvement by quarter")
    print("=" * 70)

    for q in range(4):
        q_results = results_df[results_df['quarter'] == q]
        avg_imp = q_results['improvement'].mean()
        print(f"  Q{q}: {avg_imp:+.1f} MW average improvement across H+1 to H+5")


def main():
    df, df_3min = load_data()

    # Create base features
    df = create_base_features(df)

    # Build quarterly dataset
    df_quarterly = build_quarterly_dataset(df, df_3min)

    # Train and evaluate
    results = train_and_evaluate(df_quarterly)

    # Print results
    print_results(results)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
If augmented model shows improvement at Q1-Q3:
  -> The model learns to use extrapolated features when available
  -> Single model handles all quarters

If no improvement or degradation:
  -> May need separate models per quarter
  -> Or different feature engineering
""")

    return results


if __name__ == "__main__":
    results = main()
