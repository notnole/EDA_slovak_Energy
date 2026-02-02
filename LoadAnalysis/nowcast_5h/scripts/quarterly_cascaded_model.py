"""
Quarterly Cascaded Model with Seasonal Extrapolation
=====================================================

Improvements over basic augmented model:
1. Cascade estimated H+1 error to H+2-H+5 predictions
2. Seasonal (HoD) adjustment for extrapolation bias
3. Trend indicator from partial load data
4. Better feature engineering for downstream horizons
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent

print("=" * 70)
print("QUARTERLY CASCADED MODEL")
print("With seasonal extrapolation and H+1 cascade")
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
    """Create base features."""
    df = df.copy()

    for lag in range(0, 25):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    df['error_roll_mean_24h'] = df['error'].shift(1).rolling(24).mean()
    df['error_roll_std_24h'] = df['error'].shift(1).rolling(24).std()

    # H+1 forecast for extrapolation
    df['h1_forecast'] = df['forecast_load_mw'].shift(-1)
    df['h1_actual_load'] = df['actual_load_mw'].shift(-1)
    df['h1_actual_error'] = df['error'].shift(-1)

    for h in range(1, 6):
        df[f'target_h{h}'] = df['error'].shift(-h)

    return df


def get_partial_load_stats(df_3min, hour_start, minutes_elapsed):
    """Get load statistics from partial 3-min data."""
    hour_data = df_3min[
        (df_3min['hour_start'] == hour_start) &
        (df_3min['datetime'] < hour_start + pd.Timedelta(minutes=minutes_elapsed))
    ]

    if len(hour_data) == 0:
        return np.nan, np.nan, np.nan

    mean_load = hour_data['load_mw'].mean()

    # Trend: slope of load over time
    if len(hour_data) >= 2:
        x = np.arange(len(hour_data))
        y = hour_data['load_mw'].values
        trend = np.polyfit(x, y, 1)[0]  # MW per 3-min interval
    else:
        trend = 0.0

    # Volatility
    std_load = hour_data['load_mw'].std() if len(hour_data) > 1 else 0.0

    return mean_load, trend, std_load


def compute_seasonal_extrap_bias(df, df_3min, train_years=[2024]):
    """
    Compute hour-specific extrapolation bias from training data.
    bias[hour][minutes] = mean(actual_load - extrapolated_load)
    """
    print("\n[*] Computing seasonal extrapolation bias...")

    df_3min['hour_start'] = df_3min['datetime'].dt.floor('h')

    bias = {}
    for minutes in [15, 30, 45]:
        bias[minutes] = {}

        for hour in range(24):
            errors = []

            # Use training data only
            train_df = df[(df['year'].isin(train_years)) & (df['hour'] == hour)]

            for _, row in train_df.iterrows():
                h1_hour = row['datetime'] + pd.Timedelta(hours=1)
                h1_actual = row['h1_actual_load']

                if pd.isna(h1_actual):
                    continue

                # Get extrapolated load
                partial_data = df_3min[
                    (df_3min['hour_start'] == h1_hour) &
                    (df_3min['datetime'] < h1_hour + pd.Timedelta(minutes=minutes))
                ]

                if len(partial_data) > 0:
                    extrap_load = partial_data['load_mw'].mean()
                    errors.append(h1_actual - extrap_load)

            bias[minutes][hour] = np.mean(errors) if errors else 0.0

        print(f"    {minutes} min: bias range [{min(bias[minutes].values()):.1f}, {max(bias[minutes].values()):.1f}] MW")

    return bias


def build_quarterly_dataset(df, df_3min, seasonal_bias):
    """Build dataset with cascaded features."""
    print("\n[*] Building quarterly dataset with cascaded features...")

    df_3min['hour_start'] = df_3min['datetime'].dt.floor('h')

    records = []

    for idx, row in df.iterrows():
        hour_start = row['datetime']
        h1_hour = hour_start + pd.Timedelta(hours=1)
        h1_forecast = row['h1_forecast']
        prediction_hour = row['hour']  # Hour when prediction is made

        if pd.isna(h1_forecast):
            continue

        for quarter in [0, 1, 2, 3]:
            minutes = quarter * 15

            rec = {
                'datetime': hour_start,
                'quarter': quarter,
                'year': row['year'],
                'month': row['month'],
                'hour': row['hour'],
                'dow': row['dow'],
                'h1_forecast': h1_forecast,
            }

            # Copy base features
            for col in df.columns:
                if col.startswith('error_lag') or col.startswith('error_roll') or col.startswith('target_'):
                    rec[col] = row[col]

            # === EXTRAPOLATION FEATURES ===

            # Raw extrapolated values (progressively filled)
            extrap_load_q1, trend_q1, vol_q1 = (np.nan, np.nan, np.nan)
            extrap_load_q2, trend_q2, vol_q2 = (np.nan, np.nan, np.nan)
            extrap_load_q3, trend_q3, vol_q3 = (np.nan, np.nan, np.nan)

            if quarter >= 1:
                extrap_load_q1, trend_q1, vol_q1 = get_partial_load_stats(df_3min, h1_hour, 15)
            if quarter >= 2:
                extrap_load_q2, trend_q2, vol_q2 = get_partial_load_stats(df_3min, h1_hour, 30)
            if quarter >= 3:
                extrap_load_q3, trend_q3, vol_q3 = get_partial_load_stats(df_3min, h1_hour, 45)

            # Apply seasonal bias correction
            # H+1 hour is prediction_hour + 1
            h1_hod = (prediction_hour + 1) % 24

            # Bias-corrected extrapolated load
            if not pd.isna(extrap_load_q1):
                bias_q1 = seasonal_bias.get(15, {}).get(h1_hod, 0)
                extrap_load_q1_corrected = extrap_load_q1 + bias_q1
            else:
                extrap_load_q1_corrected = np.nan

            if not pd.isna(extrap_load_q2):
                bias_q2 = seasonal_bias.get(30, {}).get(h1_hod, 0)
                extrap_load_q2_corrected = extrap_load_q2 + bias_q2
            else:
                extrap_load_q2_corrected = np.nan

            if not pd.isna(extrap_load_q3):
                bias_q3 = seasonal_bias.get(45, {}).get(h1_hod, 0)
                extrap_load_q3_corrected = extrap_load_q3 + bias_q3
            else:
                extrap_load_q3_corrected = np.nan

            # Compute extrapolated H+1 ERROR (bias-corrected)
            if not pd.isna(extrap_load_q1_corrected):
                rec['extrap_h1_error_q1'] = extrap_load_q1_corrected - h1_forecast
            else:
                rec['extrap_h1_error_q1'] = 0.0

            if not pd.isna(extrap_load_q2_corrected):
                rec['extrap_h1_error_q2'] = extrap_load_q2_corrected - h1_forecast
            else:
                rec['extrap_h1_error_q2'] = 0.0

            if not pd.isna(extrap_load_q3_corrected):
                rec['extrap_h1_error_q3'] = extrap_load_q3_corrected - h1_forecast
            else:
                rec['extrap_h1_error_q3'] = 0.0

            # === CASCADED FEATURE: Best available H+1 estimate ===
            # This is the key feature for H+2-H+5!
            if quarter == 0:
                # No extrapolation available, use model-based estimate (error_lag0 as proxy)
                rec['est_h1_error'] = row['error_lag0'] if not pd.isna(row['error_lag0']) else 0.0
            elif quarter == 1:
                rec['est_h1_error'] = rec['extrap_h1_error_q1']
            elif quarter == 2:
                rec['est_h1_error'] = rec['extrap_h1_error_q2']
            else:  # quarter == 3
                rec['est_h1_error'] = rec['extrap_h1_error_q3']

            # === TREND FEATURES ===
            rec['trend_q1'] = trend_q1 if not pd.isna(trend_q1) else 0.0
            rec['trend_q2'] = trend_q2 if not pd.isna(trend_q2) else 0.0
            rec['trend_q3'] = trend_q3 if not pd.isna(trend_q3) else 0.0

            # Best available trend
            if quarter == 0:
                rec['best_trend'] = 0.0
            elif quarter == 1:
                rec['best_trend'] = rec['trend_q1']
            elif quarter == 2:
                rec['best_trend'] = rec['trend_q2']
            else:
                rec['best_trend'] = rec['trend_q3']

            # === VOLATILITY FEATURES ===
            rec['vol_q1'] = vol_q1 if not pd.isna(vol_q1) else 0.0
            rec['vol_q2'] = vol_q2 if not pd.isna(vol_q2) else 0.0
            rec['vol_q3'] = vol_q3 if not pd.isna(vol_q3) else 0.0

            records.append(rec)

    df_quarterly = pd.DataFrame(records)
    print(f"    Created {len(df_quarterly):,} quarterly records")

    return df_quarterly


def train_and_evaluate(df_quarterly):
    """Train cascaded models and evaluate."""
    print("\n[*] Training cascaded models...")

    base_features = [
        'error_lag0', 'error_lag1', 'error_lag2', 'error_lag3',
        'error_lag24', 'error_roll_mean_24h', 'hour', 'dow'
    ]

    # Cascaded features include the estimated H+1 error!
    cascaded_features = base_features + [
        'quarter',
        'extrap_h1_error_q1', 'extrap_h1_error_q2', 'extrap_h1_error_q3',
        'est_h1_error',  # KEY: Best available H+1 estimate
        'best_trend',
    ]

    train = df_quarterly[df_quarterly['year'] == 2024].copy()
    train = train.dropna(subset=cascaded_features + ['target_h1'])

    test = df_quarterly[df_quarterly['year'] >= 2025].copy()
    test = test.dropna(subset=cascaded_features + ['target_h1'])

    print(f"    Train: {len(train):,} records")
    print(f"    Test:  {len(test):,} records")

    results = []

    for h in range(1, 6):
        target_col = f'target_h{h}'

        # Train BASE model
        model_base = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            verbosity=-1, random_state=42
        )
        model_base.fit(train[base_features], train[target_col])

        # Train CASCADED model
        model_casc = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            verbosity=-1, random_state=42
        )
        model_casc.fit(train[cascaded_features], train[target_col])

        # Evaluate per quarter
        for quarter in [0, 1, 2, 3]:
            test_q = test[test['quarter'] == quarter].copy()

            if len(test_q) == 0:
                continue

            pred_base = model_base.predict(test_q[base_features])
            pred_casc = model_casc.predict(test_q[cascaded_features])

            actual = test_q[target_col].values

            mae_base = np.nanmean(np.abs(actual - pred_base))
            mae_casc = np.nanmean(np.abs(actual - pred_casc))

            results.append({
                'horizon': h,
                'quarter': quarter,
                'mae_base': mae_base,
                'mae_cascaded': mae_casc,
                'improvement': mae_base - mae_casc
            })

        # Print feature importance for this horizon
        if h == 2:  # Show for H+2
            importance = pd.DataFrame({
                'feature': cascaded_features,
                'importance': model_casc.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\n    H+2 Top features:")
            for _, row in importance.head(8).iterrows():
                print(f"      {row['feature']}: {row['importance']}")

    return pd.DataFrame(results)


def print_results(results_df):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("RESULTS: BASE vs CASCADED MODEL")
    print("=" * 70)

    print("\n  Cascaded model adds: est_h1_error (best H+1 estimate), trend, seasonal bias")

    for h in range(1, 6):
        print(f"\n  --- H+{h} ---")
        print(f"  Quarter | Base MAE | Cascaded | Improvement")
        print(f"  --------|----------|----------|------------")

        for q in range(4):
            row = results_df[(results_df['horizon'] == h) & (results_df['quarter'] == q)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            print(f"  Q{q}      | {row['mae_base']:6.1f}   | {row['mae_cascaded']:6.1f}   | {row['improvement']:+.1f} MW")

    # Compare H+2 improvement across models
    print("\n" + "=" * 70)
    print("H+2 IMPROVEMENT COMPARISON")
    print("=" * 70)

    print("\n  Previous augmented model H+2:")
    print("    Q0: 42.0 -> 42.2 (-0.2 MW)")
    print("    Q1: 42.1 -> 38.0 (+4.1 MW)")
    print("    Q2: 42.1 -> 37.2 (+4.9 MW)")
    print("    Q3: 42.1 -> 36.4 (+5.7 MW)")

    print("\n  New cascaded model H+2:")
    for q in range(4):
        row = results_df[(results_df['horizon'] == 2) & (results_df['quarter'] == q)].iloc[0]
        print(f"    Q{q}: {row['mae_base']:.1f} -> {row['mae_cascaded']:.1f} ({row['improvement']:+.1f} MW)")


def main():
    df, df_3min = load_data()

    # Create base features
    df = create_base_features(df)

    # Compute seasonal extrapolation bias from training data
    seasonal_bias = compute_seasonal_extrap_bias(df, df_3min, train_years=[2024])

    # Build quarterly dataset with cascaded features
    df_quarterly = build_quarterly_dataset(df, df_3min, seasonal_bias)

    # Train and evaluate
    results = train_and_evaluate(df_quarterly)

    # Print results
    print_results(results)

    return results


if __name__ == "__main__":
    results = main()
