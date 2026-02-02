"""
Test Hour-Specific Models for 5-Hour Nowcasting
================================================
Given huge hourly variation (+8.8% to +48.9%), test if
separate models for different periods improve results.
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
    load_hourly = load_3min.groupby('hour_start').agg({
        'load_mw': ['std', 'first', 'last']
    }).reset_index()
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

    # Features
    for lag in range(1, 9):
        df[f'error_lag{lag}'] = df['error'].shift(lag)
    for window in [3, 6, 12, 24]:
        df[f'error_roll_mean_{window}h'] = df['error'].shift(1).rolling(window).mean()
        df[f'error_roll_std_{window}h'] = df['error'].shift(1).rolling(window).std()
    df['load_volatility_lag1'] = df['load_std_3min'].shift(1)
    df['load_trend_lag1'] = df['load_trend_3min'].shift(1)
    for lag in range(1, 4):
        df[f'reg_mean_lag{lag}'] = df['reg_mean'].shift(lag)
    seasonal = df.groupby(['dow', 'hour'])['error'].transform('mean')
    df['seasonal_error'] = seasonal
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)

    # Target
    df['target_h1'] = df['error'].shift(-1)

    return df


def get_features():
    return [
        'error_lag1', 'error_lag2', 'error_lag3', 'error_lag4', 'error_lag5',
        'error_roll_mean_3h', 'error_roll_std_3h',
        'error_roll_mean_6h', 'error_roll_std_6h',
        'error_roll_mean_12h', 'error_roll_std_12h',
        'error_roll_mean_24h', 'error_roll_std_24h',
        'load_volatility_lag1', 'load_trend_lag1',
        'reg_mean_lag1', 'reg_mean_lag2', 'reg_mean_lag3',
        'seasonal_error', 'hour_sin', 'hour_cos', 'dow', 'is_weekend',
    ]


def train_global_model(df, features):
    """Train single global model."""
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

    test = test.copy()
    test['pred'] = model.predict(test[available])

    return test


def train_hour_specific_models(df, features):
    """Train separate model for each hour."""
    target = 'target_h1'
    available = [f for f in features + ['hour'] if f in df.columns]

    df_model = df.dropna(subset=[target] + available)
    train = df_model[df_model['year'] < 2025]
    test = df_model[df_model['year'] >= 2025].copy()

    test['pred'] = np.nan

    for hour in range(24):
        train_hour = train[train['hour'] == hour]
        test_hour = test[test['hour'] == hour]

        if len(train_hour) < 50 or len(test_hour) < 10:
            continue

        # For hour-specific, remove hour from features
        hour_features = [f for f in available if f not in ['hour', 'hour_sin', 'hour_cos']]

        model = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=6,
            min_child_samples=20, random_state=42, verbosity=-1
        )
        model.fit(train_hour[hour_features], train_hour[target])
        test.loc[test['hour'] == hour, 'pred'] = model.predict(test_hour[hour_features])

    return test


def train_period_specific_models(df, features):
    """Train separate model for each time period."""
    target = 'target_h1'
    available = [f for f in features if f in df.columns]

    # Define periods
    periods = {
        'night': [0, 1, 2, 3, 4, 5],
        'morning': [6, 7, 8, 9],
        'midday': [10, 11, 12, 13, 14, 15, 16],
        'evening': [17, 18, 19, 20, 21, 22, 23],
    }

    df_model = df.dropna(subset=[target] + available)
    train = df_model[df_model['year'] < 2025]
    test = df_model[df_model['year'] >= 2025].copy()

    test['pred'] = np.nan

    for period_name, hours in periods.items():
        train_period = train[train['hour'].isin(hours)]
        test_period = test[test['hour'].isin(hours)]

        model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.04, max_depth=7,
            min_child_samples=25, random_state=42, verbosity=-1
        )
        model.fit(train_period[available], train_period[target])
        test.loc[test['hour'].isin(hours), 'pred'] = model.predict(test_period[available])

    return test


def evaluate(test, name):
    """Evaluate predictions."""
    baseline_mae = np.abs(test['target_h1']).mean()
    model_mae = np.abs(test['target_h1'] - test['pred']).mean()
    improvement = (baseline_mae - model_mae) / baseline_mae * 100
    return {'name': name, 'baseline': baseline_mae, 'mae': model_mae, 'improvement': improvement}


def main():
    print("=" * 70)
    print("HOUR-SPECIFIC MODEL COMPARISON (H+1)")
    print("=" * 70)

    df = load_and_prepare()
    features = get_features()

    print("\nTraining models...")

    # 1. Global model
    test_global = train_global_model(df, features)
    result_global = evaluate(test_global, "Global Model")

    # 2. 24 hour-specific models
    test_hour = train_hour_specific_models(df, features)
    result_hour = evaluate(test_hour, "24 Hour Models")

    # 3. 4 period-specific models
    test_period = train_period_specific_models(df, features)
    result_period = evaluate(test_period, "4 Period Models")

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n  {'Model':<20} {'Baseline':<12} {'MAE':<12} {'Improvement'}")
    print("  " + "-" * 56)

    for r in [result_global, result_period, result_hour]:
        print(f"  {r['name']:<20} {r['baseline']:<12.1f} {r['mae']:<12.1f} {r['improvement']:+.1f}%")

    # Detailed comparison by hour
    print("\n" + "=" * 70)
    print("BY-HOUR COMPARISON: GLOBAL vs HOUR-SPECIFIC")
    print("=" * 70)

    test_global['pred_global'] = test_global['pred']
    comparison = test_global[['hour', 'target_h1', 'pred_global']].copy()
    comparison['pred_hour'] = test_hour['pred'].values

    print(f"\n  {'Hour':<6} {'Global MAE':<12} {'Hour MAE':<12} {'Diff':<10}")
    print("  " + "-" * 42)

    for hour in range(24):
        subset = comparison[comparison['hour'] == hour]
        global_mae = np.abs(subset['target_h1'] - subset['pred_global']).mean()
        hour_mae = np.abs(subset['target_h1'] - subset['pred_hour']).mean()
        diff = hour_mae - global_mae
        better = "+" if diff < 0 else "-" if diff > 0 else "="
        print(f"  {hour:<6} {global_mae:<12.1f} {hour_mae:<12.1f} {diff:+.1f} {better}")

    # Overall winner
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    best = min([result_global, result_period, result_hour], key=lambda x: x['mae'])
    print(f"\n  Best approach: {best['name']}")
    print(f"  MAE: {best['mae']:.1f} MW ({best['improvement']:+.1f}% vs baseline)")


if __name__ == "__main__":
    main()
