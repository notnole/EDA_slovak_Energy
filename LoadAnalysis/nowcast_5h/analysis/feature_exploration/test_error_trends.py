"""
Test Error Trend Features for 5-Hour Nowcasting
================================================
Test if error trend/momentum features improve the model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent.parent  # ipesoft_eda_data


def load_and_prepare():
    """Load and prepare data."""
    df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['year'] = df['datetime'].dt.year
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']

    # 3-min load
    load_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'load_3min.csv')
    load_3min['datetime'] = pd.to_datetime(load_3min['datetime'])
    load_3min['hour_start'] = load_3min['datetime'].dt.floor('H')
    load_hourly = load_3min.groupby('hour_start').agg({'load_mw': ['std', 'first', 'last']}).reset_index()
    load_hourly.columns = ['datetime', 'load_std_3min', 'load_first', 'load_last']
    load_hourly['load_trend_3min'] = load_hourly['load_last'] - load_hourly['load_first']
    df = df.merge(load_hourly[['datetime', 'load_std_3min', 'load_trend_3min']], on='datetime', how='left')

    # Regulation
    reg_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'regulation_3min.csv')
    reg_3min['datetime'] = pd.to_datetime(reg_3min['datetime'])
    reg_3min['hour_start'] = reg_3min['datetime'].dt.floor('H')
    reg_hourly = reg_3min.groupby('hour_start').agg({'regulation_mw': ['mean', 'std']}).reset_index()
    reg_hourly.columns = ['datetime', 'reg_mean', 'reg_std']
    df = df.merge(reg_hourly, on='datetime', how='left')

    return df


def create_features(df):
    """Create all features including error trends."""
    df = df.copy()

    # 1. Error lags
    for lag in range(1, 13):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    # 2. Rolling statistics
    for window in [3, 6, 12, 24]:
        df[f'error_roll_mean_{window}h'] = df['error'].shift(1).rolling(window).mean()
        df[f'error_roll_std_{window}h'] = df['error'].shift(1).rolling(window).std()

    # 3. ERROR TREND FEATURES (what we're testing)
    # Short-term trend (3h)
    df['error_trend_3h'] = df['error_lag1'] - df['error_lag3']
    # Medium-term trend (6h)
    df['error_trend_6h'] = df['error_lag1'] - df['error_lag6']
    # Original trend (5h)
    df['error_trend_5h'] = df['error_lag1'] - df['error_lag5']
    # Long-term trend (12h)
    df['error_trend_12h'] = df['error_lag1'] - df['error_lag12']

    # Error acceleration (is trend speeding up or slowing down?)
    df['error_accel'] = (df['error_lag1'] - df['error_lag2']) - (df['error_lag2'] - df['error_lag3'])

    # Error momentum (weighted recent changes)
    df['error_momentum'] = 0.5 * (df['error_lag1'] - df['error_lag2']) + \
                           0.3 * (df['error_lag2'] - df['error_lag3']) + \
                           0.2 * (df['error_lag3'] - df['error_lag4'])

    # Sign changes (volatility of direction)
    df['error_sign_lag1'] = np.sign(df['error_lag1'])
    df['error_sign_lag2'] = np.sign(df['error_lag2'])
    df['error_sign_change'] = (df['error_sign_lag1'] != df['error_sign_lag2']).astype(int)

    # 4. 3-min features
    df['load_volatility_lag1'] = df['load_std_3min'].shift(1)
    df['load_trend_lag1'] = df['load_trend_3min'].shift(1)

    # 5. Regulation
    for lag in range(1, 4):
        df[f'reg_mean_lag{lag}'] = df['reg_mean'].shift(lag)

    # 6. Seasonal
    seasonal = df.groupby(['dow', 'hour'])['error'].transform('mean')
    df['seasonal_error'] = seasonal

    # 7. Time
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)

    # Target
    df['target_h1'] = df['error'].shift(-1)

    return df


def test_model(df, features, name):
    """Test a feature set."""
    target = 'target_h1'
    available = [f for f in features if f in df.columns]

    df_model = df.dropna(subset=[target] + available)
    train = df_model[df_model['year'] < 2025]
    test = df_model[df_model['year'] >= 2025]

    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=8, num_leaves=50,
        min_child_samples=30, random_state=42, verbosity=-1
    )
    model.fit(train[available], train[target])
    pred = model.predict(test[available])

    baseline_mae = np.abs(test[target]).mean()
    model_mae = np.abs(test[target] - pred).mean()
    improvement = (baseline_mae - model_mae) / baseline_mae * 100

    # Get feature importance for trend features
    importance = dict(zip(available, model.feature_importances_))

    return {
        'name': name,
        'mae': model_mae,
        'improvement': improvement,
        'n_features': len(available),
        'importance': importance
    }


def main():
    print("=" * 70)
    print("ERROR TREND FEATURES TEST (H+1)")
    print("=" * 70)

    df = load_and_prepare()
    df = create_features(df)

    # Base features (without trends)
    base_features = [
        'error_lag1', 'error_lag2', 'error_lag3', 'error_lag4', 'error_lag5',
        'error_lag6', 'error_lag7', 'error_lag8',
        'error_roll_mean_3h', 'error_roll_std_3h',
        'error_roll_mean_6h', 'error_roll_std_6h',
        'error_roll_mean_12h', 'error_roll_std_12h',
        'error_roll_mean_24h', 'error_roll_std_24h',
        'load_volatility_lag1', 'load_trend_lag1',
        'reg_mean_lag1', 'reg_mean_lag2', 'reg_mean_lag3',
        'seasonal_error',
        'hour', 'hour_sin', 'hour_cos', 'dow', 'is_weekend',
    ]

    # Trend features to test
    trend_features = [
        'error_trend_3h',
        'error_trend_5h',
        'error_trend_6h',
        'error_trend_12h',
        'error_accel',
        'error_momentum',
        'error_sign_change',
    ]

    print("\nTesting feature combinations...")

    results = []

    # 1. Base model (no trends)
    result = test_model(df, base_features, "Base (no trends)")
    results.append(result)
    print(f"\n  Base (no trends): MAE={result['mae']:.2f}, Improvement={result['improvement']:+.1f}%")

    # 2. Add individual trend features
    for trend_feat in trend_features:
        features = base_features + [trend_feat]
        result = test_model(df, features, f"+ {trend_feat}")
        results.append(result)
        trend_imp = result['importance'].get(trend_feat, 0)
        print(f"  + {trend_feat:20}: MAE={result['mae']:.2f}, Improvement={result['improvement']:+.1f}%, Importance={trend_imp:.0f}")

    # 3. Add all trend features
    all_features = base_features + trend_features
    result = test_model(df, all_features, "All trends")
    results.append(result)
    print(f"\n  All trends: MAE={result['mae']:.2f}, Improvement={result['improvement']:+.1f}%")

    # Feature importance for trend features
    print("\n" + "=" * 70)
    print("TREND FEATURE IMPORTANCE (All trends model)")
    print("=" * 70)

    for feat in trend_features:
        imp = result['importance'].get(feat, 0)
        print(f"  {feat:25}: {imp:6.0f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    base_mae = results[0]['mae']
    print(f"\n  Base model MAE: {base_mae:.2f} MW")
    print(f"\n  Impact of adding trend features:")

    for r in results[1:]:
        diff = base_mae - r['mae']
        if diff > 0:
            print(f"    {r['name']:25}: {diff:+.2f} MW better")
        else:
            print(f"    {r['name']:25}: {diff:+.2f} MW")

    best = min(results, key=lambda x: x['mae'])
    print(f"\n  Best: {best['name']} with MAE={best['mae']:.2f} MW ({best['improvement']:+.1f}%)")


if __name__ == "__main__":
    main()
