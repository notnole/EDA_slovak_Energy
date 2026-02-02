"""
Test: Does using 3-min aggregated load instead of hourly load affect accuracy?

In production, we may only have live 3-min data, not hourly actuals.
Question: Can we compute error_lag0 from mean(3-min load) instead of hourly actual?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent

print("=" * 70)
print("TEST: 3-MIN AGGREGATED vs HOURLY LOAD")
print("=" * 70)


def load_data():
    """Load both datasets."""
    print("\n[*] Loading data...")

    # Hourly data
    df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 3-min data
    df_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'load_3min.csv')
    df_3min['datetime'] = pd.to_datetime(df_3min['datetime'])
    df_3min = df_3min.sort_values('datetime').reset_index(drop=True)
    df_3min['hour_start'] = df_3min['datetime'].dt.floor('h')

    print(f"    Hourly: {len(df):,} records")
    print(f"    3-min:  {len(df_3min):,} records")

    return df, df_3min


def compare_hourly_vs_aggregated(df, df_3min):
    """Compare hourly actual load vs mean of 3-min data."""
    print("\n[*] Comparing hourly load vs 3-min aggregated...")

    # Aggregate 3-min to hourly
    hourly_from_3min = df_3min.groupby('hour_start')['load_mw'].agg(['mean', 'count']).reset_index()
    hourly_from_3min.columns = ['datetime', 'load_3min_mean', 'n_obs']

    # Merge with hourly data
    comparison = df[['datetime', 'actual_load_mw', 'forecast_load_mw']].merge(
        hourly_from_3min,
        on='datetime',
        how='inner'
    )

    # Calculate difference
    comparison['diff'] = comparison['actual_load_mw'] - comparison['load_3min_mean']
    comparison['diff_abs'] = comparison['diff'].abs()
    comparison['diff_pct'] = comparison['diff'] / comparison['actual_load_mw'] * 100

    print(f"\n    Matched hours: {len(comparison):,}")
    print(f"\n    Difference (Hourly - 3min Mean):")
    print(f"      Mean:   {comparison['diff'].mean():.2f} MW")
    print(f"      Std:    {comparison['diff'].std():.2f} MW")
    print(f"      MAE:    {comparison['diff_abs'].mean():.2f} MW")
    print(f"      Max:    {comparison['diff_abs'].max():.2f} MW")
    print(f"      MAPE:   {comparison['diff_pct'].abs().mean():.3f}%")

    # Check hours with incomplete 3-min data
    incomplete = comparison[comparison['n_obs'] < 20]
    print(f"\n    Hours with <20 observations: {len(incomplete)} ({len(incomplete)/len(comparison)*100:.1f}%)")

    # Full hours only
    full_hours = comparison[comparison['n_obs'] >= 20]
    print(f"\n    Full hours (>=20 obs) only:")
    print(f"      MAE: {full_hours['diff_abs'].mean():.2f} MW")
    print(f"      Max: {full_hours['diff_abs'].max():.2f} MW")

    return comparison


def test_model_with_3min_error(df, df_3min):
    """Test if using 3-min derived error affects model accuracy."""
    print("\n[*] Testing model accuracy with 3-min derived error...")

    # Aggregate 3-min to hourly
    hourly_from_3min = df_3min.groupby('hour_start')['load_mw'].mean().reset_index()
    hourly_from_3min.columns = ['datetime', 'load_3min']

    # Merge
    df = df.merge(hourly_from_3min, on='datetime', how='left')

    # Create both error versions
    df['error_hourly'] = df['actual_load_mw'] - df['forecast_load_mw']
    df['error_3min'] = df['load_3min'] - df['forecast_load_mw']

    df['year'] = df['datetime'].dt.year
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek

    # Create features using HOURLY error
    df['error_lag0_hourly'] = df['error_hourly'].shift(0)
    df['error_lag1_hourly'] = df['error_hourly'].shift(1)
    df['error_lag2_hourly'] = df['error_hourly'].shift(2)
    df['error_lag24_hourly'] = df['error_hourly'].shift(24)
    df['error_roll_mean_24h_hourly'] = df['error_hourly'].shift(1).rolling(24).mean()

    # Create features using 3-MIN error
    df['error_lag0_3min'] = df['error_3min'].shift(0)
    df['error_lag1_3min'] = df['error_3min'].shift(1)
    df['error_lag2_3min'] = df['error_3min'].shift(2)
    df['error_lag24_3min'] = df['error_3min'].shift(24)
    df['error_roll_mean_24h_3min'] = df['error_3min'].shift(1).rolling(24).mean()

    # Target (always use true hourly error)
    df['target_h1'] = df['error_hourly'].shift(-1)
    df['target_h2'] = df['error_hourly'].shift(-2)

    # Feature sets
    features_hourly = [
        'error_lag0_hourly', 'error_lag1_hourly', 'error_lag2_hourly',
        'error_lag24_hourly', 'error_roll_mean_24h_hourly',
        'hour', 'dow'
    ]

    features_3min = [
        'error_lag0_3min', 'error_lag1_3min', 'error_lag2_3min',
        'error_lag24_3min', 'error_roll_mean_24h_3min',
        'hour', 'dow'
    ]

    # Train/test split
    train = df[df['year'] == 2024].dropna(subset=features_hourly + features_3min + ['target_h1', 'target_h2'])
    test = df[df['year'] >= 2025].dropna(subset=features_hourly + features_3min + ['target_h1', 'target_h2'])

    print(f"\n    Train: {len(train):,}")
    print(f"    Test:  {len(test):,}")

    results = []

    for h, target_col in [(1, 'target_h1'), (2, 'target_h2')]:
        # Model with hourly error features
        model_hourly = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            verbosity=-1, random_state=42
        )
        model_hourly.fit(train[features_hourly], train[target_col])
        pred_hourly = model_hourly.predict(test[features_hourly])
        mae_hourly = np.nanmean(np.abs(test[target_col].values - pred_hourly))

        # Model with 3-min error features
        model_3min = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            verbosity=-1, random_state=42
        )
        model_3min.fit(train[features_3min], train[target_col])
        pred_3min = model_3min.predict(test[features_3min])
        mae_3min = np.nanmean(np.abs(test[target_col].values - pred_3min))

        diff = mae_3min - mae_hourly

        results.append({
            'horizon': h,
            'mae_hourly': mae_hourly,
            'mae_3min': mae_3min,
            'diff': diff
        })

        print(f"\n    H+{h}:")
        print(f"      Hourly features MAE: {mae_hourly:.2f} MW")
        print(f"      3-min features MAE:  {mae_3min:.2f} MW")
        print(f"      Difference: {diff:+.2f} MW")

    return pd.DataFrame(results)


def test_error_correlation(df, df_3min):
    """Check correlation between hourly and 3-min derived errors."""
    print("\n[*] Error correlation analysis...")

    # Aggregate 3-min to hourly
    hourly_from_3min = df_3min.groupby('hour_start')['load_mw'].mean().reset_index()
    hourly_from_3min.columns = ['datetime', 'load_3min']

    # Merge
    df = df.merge(hourly_from_3min, on='datetime', how='inner')

    # Calculate errors
    df['error_hourly'] = df['actual_load_mw'] - df['forecast_load_mw']
    df['error_3min'] = df['load_3min'] - df['forecast_load_mw']

    # Correlation
    corr = df['error_hourly'].corr(df['error_3min'])
    print(f"\n    Correlation(hourly error, 3-min error): {corr:.6f}")

    # Error of errors
    df['error_diff'] = df['error_hourly'] - df['error_3min']
    print(f"\n    Error difference (hourly - 3min):")
    print(f"      Mean: {df['error_diff'].mean():.2f} MW")
    print(f"      Std:  {df['error_diff'].std():.2f} MW")
    print(f"      MAE:  {df['error_diff'].abs().mean():.2f} MW")

    return df


def main():
    df, df_3min = load_data()

    # Compare raw load values
    comparison = compare_hourly_vs_aggregated(df, df_3min)

    # Check error correlation
    df_with_errors = test_error_correlation(df.copy(), df_3min)

    # Test model accuracy
    results = test_model_with_3min_error(df.copy(), df_3min)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    load_mae = comparison['diff_abs'].mean()
    model_diff = results['diff'].mean()

    print(f"""
    Load difference (hourly vs 3-min mean): {load_mae:.2f} MW MAE

    Model accuracy impact: {model_diff:+.2f} MW

    """)

    if abs(model_diff) < 0.5:
        print("    [+] SAFE TO USE 3-min aggregated data")
        print("        Difference is negligible (<0.5 MW)")
    elif abs(model_diff) < 1.0:
        print("    [~] ACCEPTABLE to use 3-min aggregated data")
        print("        Small difference (<1 MW)")
    else:
        print("    [-] CAUTION: Noticeable accuracy impact")
        print(f"        Difference is {abs(model_diff):.1f} MW")

    return comparison, results


if __name__ == "__main__":
    comparison, results = main()
