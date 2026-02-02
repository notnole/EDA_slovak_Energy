"""
Stage 2 Residual Deep Analysis
==============================
Look for any remaining patterns in the final residuals.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent.parent  # ipesoft_eda_data


def main():
    print("=" * 70)
    print("STAGE 2 RESIDUAL DEEP ANALYSIS")
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

    # 3-min load
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

    # Stage 1 features
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
    df['target'] = df['error'].shift(-1)

    stage1_features = [
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

    # Train Stage 1
    df_model = df.dropna(subset=['target'] + stage1_features).copy()
    train_s1 = df_model[df_model['year'] == 2024]

    model_s1 = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=8, num_leaves=50,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=-1
    )
    model_s1.fit(train_s1[stage1_features], train_s1['target'])
    df_model['stage1_pred'] = model_s1.predict(df_model[stage1_features])
    df_model['residual_h1'] = df_model['target'] - df_model['stage1_pred']

    # Stage 2 features
    for lag in range(1, 7):
        df_model[f'residual_lag{lag}'] = df_model['residual_h1'].shift(lag)
    for window in [3, 6]:
        df_model[f'residual_roll_mean_{window}h'] = df_model['residual_h1'].shift(1).rolling(window).mean()
        df_model[f'residual_roll_std_{window}h'] = df_model['residual_h1'].shift(1).rolling(window).std()
    df_model['residual_trend_3h'] = df_model['residual_lag1'] - df_model['residual_lag3']

    stage2_features = [
        'residual_lag1', 'residual_lag2', 'residual_lag3',
        'residual_roll_mean_3h', 'residual_roll_mean_6h',
        'residual_roll_std_3h', 'residual_roll_std_6h',
        'residual_trend_3h',
    ]

    # Train Stage 2
    train_s2 = df_model[(df_model['year'] == 2024) & (df_model['month'] > 6)]
    train_s2 = train_s2.dropna(subset=stage2_features)
    test = df_model[df_model['year'] >= 2025].dropna(subset=stage2_features).copy()

    model_s2 = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5, num_leaves=30,
        min_child_samples=20, random_state=42, verbosity=-1
    )
    model_s2.fit(train_s2[stage2_features], train_s2['residual_h1'])

    test['residual_pred'] = model_s2.predict(test[stage2_features])
    test['stage2_pred'] = test['stage1_pred'] + test['residual_pred']
    test['final_residual'] = test['target'] - test['stage2_pred']

    mae_s1 = np.abs(test['residual_h1']).mean()
    mae_s2 = np.abs(test['final_residual']).mean()
    print(f"\nStage 1 MAE: {mae_s1:.1f} MW")
    print(f"Stage 2 MAE: {mae_s2:.1f} MW")

    # Final residual statistics
    print(f"\nFinal residual stats:")
    print(f"  Mean: {test['final_residual'].mean():.2f} MW")
    print(f"  Std: {test['final_residual'].std():.1f} MW")

    # Autocorrelation structure
    print(f"\nFinal residual autocorrelation:")
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        corr = test['final_residual'].corr(test['final_residual'].shift(lag))
        print(f"  Lag {lag:2d}h: {corr:+.3f}")

    # By hour
    print(f"\nFinal residual MAE by hour:")
    by_hour = test.groupby('hour')['final_residual'].apply(lambda x: np.abs(x).mean())
    worst_hours = by_hour.nlargest(5)
    best_hours = by_hour.nsmallest(5)
    print("  Worst hours:")
    for h, mae in worst_hours.items():
        print(f"    Hour {h:2d}: {mae:.1f} MW")
    print("  Best hours:")
    for h, mae in best_hours.items():
        print(f"    Hour {h:2d}: {mae:.1f} MW")

    # Distribution analysis
    print(f"\nResidual distribution analysis:")
    print(f"  Skewness: {stats.skew(test['final_residual'].dropna()):.3f}")
    print(f"  Kurtosis: {stats.kurtosis(test['final_residual'].dropna()):.3f}")

    # Percentile analysis
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nResidual percentiles:")
    for p in percentiles:
        val = np.percentile(test['final_residual'].dropna(), p)
        print(f"  P{p:2d}: {val:+.0f} MW")

    # Extreme residual analysis
    extreme = test[np.abs(test['final_residual']) > 100]
    print(f"\n|Residual| > 100 MW: {len(extreme)} cases ({len(extreme)/len(test)*100:.1f}%)")
    if len(extreme) > 0:
        print(f"  Hour distribution: {extreme['hour'].value_counts().head(5).to_dict()}")
        print(f"  DOW distribution: {extreme['dow'].value_counts().to_dict()}")

    # Does large residual predict next large residual?
    test['abs_residual'] = np.abs(test['final_residual'])
    test['next_abs_residual'] = test['abs_residual'].shift(-1)
    print(f"\nDoes large residual predict next large residual?")
    large_now = test[test['abs_residual'] > 50]
    normal_now = test[test['abs_residual'] <= 50]
    print(f"  After large residual: next MAE = {large_now['next_abs_residual'].mean():.1f} MW")
    print(f"  After normal residual: next MAE = {normal_now['next_abs_residual'].mean():.1f} MW")

    # Stage 3 test - can we predict the final residual?
    print("\n" + "=" * 70)
    print("STAGE 3 TEST (residual of residual)")
    print("=" * 70)

    for lag in range(1, 5):
        test[f'final_resid_lag{lag}'] = test['final_residual'].shift(lag)
    test['final_resid_roll_mean_3h'] = test['final_residual'].shift(1).rolling(3).mean()

    stage3_features = ['final_resid_lag1', 'final_resid_lag2', 'final_resid_lag3',
                       'final_resid_roll_mean_3h']

    test_s3 = test.dropna(subset=stage3_features).copy()
    train_s3 = test_s3.iloc[:len(test_s3)//3]
    eval_s3 = test_s3.iloc[len(test_s3)//3:]

    model_s3 = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=4,
        random_state=42, verbosity=-1
    )
    model_s3.fit(train_s3[stage3_features], train_s3['final_residual'])
    eval_s3['s3_pred'] = model_s3.predict(eval_s3[stage3_features])
    eval_s3['s3_residual'] = eval_s3['final_residual'] - eval_s3['s3_pred']

    mae_before = np.abs(eval_s3['final_residual']).mean()
    mae_after = np.abs(eval_s3['s3_residual']).mean()
    print(f"\nStage 2 residual MAE: {mae_before:.1f} MW")
    print(f"Stage 3 residual MAE: {mae_after:.1f} MW")
    print(f"Gain: {mae_before - mae_after:.2f} MW ({(mae_before-mae_after)/mae_before*100:+.1f}%)")

    # Final autocorrelation
    s3_autocorr = eval_s3['s3_residual'].corr(eval_s3['s3_residual'].shift(1))
    print(f"\nStage 3 residual autocorr (lag1): {s3_autocorr:.3f}")

    return test


if __name__ == "__main__":
    test = main()
