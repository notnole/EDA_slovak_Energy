"""
Systematic Bias Features Test
=============================
Test if adding features about DAMAS systematic bias improves predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent.parent  # ipesoft_eda_data


def main():
    print("=" * 70)
    print("SYSTEMATIC BIAS FEATURES TEST")
    print("=" * 70)

    # Load data
    df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['date'] = df['datetime'].dt.date
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']

    # 3-minute load
    load_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'load_3min.csv')
    load_3min['datetime'] = pd.to_datetime(load_3min['datetime'])
    load_3min['hour_start'] = load_3min['datetime'].dt.floor('h')
    load_hourly = load_3min.groupby('hour_start').agg({
        'load_mw': ['std', 'first', 'last']
    }).reset_index()
    load_hourly.columns = ['datetime', 'load_std_3min', 'load_first', 'load_last']
    load_hourly['load_trend_3min'] = load_hourly['load_last'] - load_hourly['load_first']
    df = df.merge(load_hourly[['datetime', 'load_std_3min', 'load_trend_3min']], on='datetime', how='left')

    # Regulation
    reg_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'regulation_3min.csv')
    reg_3min['datetime'] = pd.to_datetime(reg_3min['datetime'])
    reg_3min['hour_start'] = reg_3min['datetime'].dt.floor('h')
    reg_hourly = reg_3min.groupby('hour_start').agg({'regulation_mw': ['mean', 'std']}).reset_index()
    reg_hourly.columns = ['datetime', 'reg_mean', 'reg_std']
    df = df.merge(reg_hourly, on='datetime', how='left')

    print("\nCreating features...")

    # Standard features
    for lag in range(1, 9):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    for window in [3, 6, 12, 24]:
        df[f'error_roll_mean_{window}h'] = df['error'].shift(1).rolling(window).mean()
        df[f'error_roll_std_{window}h'] = df['error'].shift(1).rolling(window).std()

    df['error_trend_3h'] = df['error_lag1'] - df['error_lag3']
    df['error_trend_6h'] = df['error_lag1'] - df['error_lag6']
    df['error_momentum'] = (0.5 * (df['error_lag1'] - df['error_lag2']) +
                            0.3 * (df['error_lag2'] - df['error_lag3']) +
                            0.2 * (df['error_lag3'] - df['error_lag4']))

    df['load_volatility_lag1'] = df['load_std_3min'].shift(1)
    df['load_trend_lag1'] = df['load_trend_3min'].shift(1)
    for lag in range(1, 4):
        df[f'reg_mean_lag{lag}'] = df['reg_mean'].shift(lag)
    df['reg_std_lag1'] = df['reg_std'].shift(1)

    df['seasonal_error'] = df.groupby(['dow', 'hour'])['error'].transform('mean')
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)

    # ===== NEW: SYSTEMATIC BIAS FEATURES =====
    print("\nAdding systematic bias features...")

    # 1. Daily mean error from PREVIOUS day
    daily_error = df.groupby('date')['error'].mean().reset_index()
    daily_error.columns = ['date', 'daily_mean_error']
    daily_error = daily_error.sort_values('date')
    daily_error['prev_day_mean_error'] = daily_error['daily_mean_error'].shift(1)
    daily_error['daily_roll_3d'] = daily_error['daily_mean_error'].shift(1).rolling(3).mean()
    daily_error['daily_roll_7d'] = daily_error['daily_mean_error'].shift(1).rolling(7).mean()

    df = df.merge(daily_error[['date', 'prev_day_mean_error', 'daily_roll_3d', 'daily_roll_7d']],
                  on='date', how='left')

    # 2. Same-day cumulative bias
    df['same_day_running_mean'] = df.groupby('date')['error'].transform(
        lambda x: x.expanding().mean().shift(1))

    # 3. Early day bias (for hours >= 6)
    early_mask = df['hour'] < 6
    early_df = df[early_mask].groupby('date')['error'].mean().reset_index()
    early_df.columns = ['date', 'early_day_bias']
    df = df.merge(early_df, on='date', how='left')
    df.loc[df['hour'] < 6, 'early_day_bias'] = np.nan

    # 4. Sign persistence
    df['bias_sign_lag24'] = np.sign(df['error'].shift(24))
    df['sign_match_24h'] = (np.sign(df['error'].shift(1)) == df['bias_sign_lag24']).astype(float)

    # Targets
    for h in range(1, 6):
        df[f'target_h{h}'] = df['error'].shift(-h)

    # Define feature sets
    base_features = [
        'error_lag1', 'error_lag2', 'error_lag3', 'error_lag4', 'error_lag5', 'error_lag6',
        'error_roll_mean_3h', 'error_roll_std_3h',
        'error_roll_mean_6h', 'error_roll_std_6h',
        'error_roll_mean_12h', 'error_roll_std_12h',
        'error_roll_mean_24h', 'error_roll_std_24h',
        'error_trend_3h', 'error_trend_6h', 'error_momentum',
        'load_volatility_lag1', 'load_trend_lag1',
        'reg_mean_lag1', 'reg_mean_lag2', 'reg_mean_lag3', 'reg_std_lag1',
        'seasonal_error',
        'hour', 'hour_sin', 'hour_cos', 'dow', 'is_weekend',
    ]

    bias_features = [
        'prev_day_mean_error',
        'daily_roll_3d',
        'daily_roll_7d',
        'same_day_running_mean',
        'early_day_bias',
        'sign_match_24h',
    ]

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    target = 'target_h1'

    # Without bias features
    features_base = [f for f in base_features if f in df.columns]
    df_model = df.dropna(subset=[target] + features_base).copy()
    train = df_model[df_model['year'] == 2024]
    test = df_model[df_model['year'] >= 2025].copy()

    model_base = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=8, num_leaves=50,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=-1
    )
    model_base.fit(train[features_base], train[target])
    test['pred_base'] = model_base.predict(test[features_base])

    baseline_mae = np.abs(test[target]).mean()
    base_mae = np.abs(test[target] - test['pred_base']).mean()
    base_imp = (baseline_mae - base_mae) / baseline_mae * 100

    print(f"\nBASELINE (no model): {baseline_mae:.1f} MW")
    print(f"Base model (no bias features): {base_mae:.1f} MW ({base_imp:+.1f}%)")

    # With bias features
    all_features = base_features + bias_features
    features_all = [f for f in all_features if f in df.columns]

    df_model2 = df.dropna(subset=[target] + features_all).copy()
    train2 = df_model2[df_model2['year'] == 2024]
    test2 = df_model2[df_model2['year'] >= 2025].copy()

    model_bias = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=8, num_leaves=50,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=-1
    )
    model_bias.fit(train2[features_all], train2[target])
    test2['pred_bias'] = model_bias.predict(test2[features_all])

    bias_mae = np.abs(test2[target] - test2['pred_bias']).mean()
    bias_imp = (baseline_mae - bias_mae) / baseline_mae * 100

    print(f"Model + bias features: {bias_mae:.1f} MW ({bias_imp:+.1f}%)")
    print(f"\nGain from bias features: {base_mae - bias_mae:.2f} MW ({(base_mae - bias_mae)/base_mae*100:+.1f}%)")

    # Feature importance
    importance = pd.DataFrame({
        'feature': features_all,
        'importance': model_bias.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 20 features with bias:")
    for _, row in importance.head(20).iterrows():
        tag = '[BIAS]' if row['feature'] in bias_features else ''
        print(f"  {row['feature']:<30}: {row['importance']:6.0f} {tag}")

    print("\nBias feature importance:")
    for feat in bias_features:
        if feat in importance['feature'].values:
            imp = importance[importance['feature'] == feat]['importance'].values[0]
            print(f"  {feat:<30}: {imp:.0f}")

    # Check by hour
    print("\n" + "=" * 70)
    print("BIAS FEATURE IMPACT BY HOUR")
    print("=" * 70)

    test2['error_base'] = np.abs(test2[target] - test2['pred_base'].values[:len(test2)] if 'pred_base' in test2.columns else test2[target])
    test2['error_bias'] = np.abs(test2[target] - test2['pred_bias'])

    # Retrain base model on same test set for fair comparison
    test2['pred_base_fair'] = model_base.predict(test2[features_base])
    test2['error_base_fair'] = np.abs(test2[target] - test2['pred_base_fair'])

    print(f"\n{'Hour':<6} {'Base MAE':<12} {'Bias MAE':<12} {'Gain':<10}")
    print("-" * 45)
    by_hour = test2.groupby('hour').agg({
        'error_base_fair': 'mean',
        'error_bias': 'mean'
    }).reset_index()
    for _, row in by_hour.iterrows():
        gain = row['error_base_fair'] - row['error_bias']
        print(f"{int(row['hour']):<6} {row['error_base_fair']:<12.1f} {row['error_bias']:<12.1f} {gain:+.1f}")

    return df, model_bias


if __name__ == "__main__":
    df, model = main()
