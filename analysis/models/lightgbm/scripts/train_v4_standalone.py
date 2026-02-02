"""
LightGBM V4 - Standalone Training Script
=========================================
Run this directly with: python train_v4_standalone.py

This script replicates the notebook but runs without Jupyter.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'seed': 42,
}

LEAD_TIMES = [12, 9, 6, 3, 0]

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
print(f"Regulation: {len(reg_df):,} rows ({reg_df['datetime'].min()} to {reg_df['datetime'].max()})")

load_df = pd.read_csv(FEATURES_DIR / 'load_3min.csv', parse_dates=['datetime'])
print(f"Load: {len(load_df):,} rows")

label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
    columns={'System Imbalance (MWh)': 'imbalance'}
)
print(f"Labels: {len(label_df):,} rows")

# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def compute_load_expected(load_df):
    """Compute expected load by time-of-day from 2024 data."""
    load_df = load_df.copy()
    load_df['hour'] = load_df['datetime'].dt.hour
    load_df['minute'] = load_df['datetime'].dt.minute
    load_df['is_weekend'] = load_df['datetime'].dt.dayofweek >= 5
    train_mask = load_df['datetime'].dt.year == 2024
    expected = load_df[train_mask].groupby(['hour', 'minute', 'is_weekend'])['load_mw'].mean()
    expected.name = 'expected_load'
    return expected


def add_historical_regulation_features(reg_df):
    """Add historical regulation features from raw 3-min data."""
    reg_df = reg_df.sort_values('datetime').copy()

    reg_df['reg_hist_mean_10'] = reg_df['regulation_mw'].shift(1).rolling(10).mean()
    reg_df['reg_hist_std_10'] = reg_df['regulation_mw'].shift(1).rolling(10).std()
    reg_df['reg_hist_min_10'] = reg_df['regulation_mw'].shift(1).rolling(10).min()
    reg_df['reg_hist_max_10'] = reg_df['regulation_mw'].shift(1).rolling(10).max()
    reg_df['reg_hist_range_10'] = reg_df['reg_hist_max_10'] - reg_df['reg_hist_min_10']
    reg_df['reg_hist_trend_10'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(10)
    reg_df['reg_hist_mean_20'] = reg_df['regulation_mw'].shift(1).rolling(20).mean()
    reg_df['reg_hist_std_20'] = reg_df['regulation_mw'].shift(1).rolling(20).std()
    reg_df['reg_momentum'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(2)
    reg_df['reg_acceleration'] = reg_df['reg_momentum'] - reg_df['reg_momentum'].shift(1)

    return reg_df


def create_base_features(reg_df, load_df, label_df, load_expected):
    """Create base feature dataframe by aligning 3-min data to 15-min periods."""

    reg_df = reg_df.copy()
    reg_df['datetime_floor'] = reg_df['datetime'].dt.floor('3min')
    reg_df['settlement_end'] = reg_df['datetime_floor'].dt.ceil('15min')
    mask = reg_df['datetime_floor'] == reg_df['settlement_end']
    reg_df.loc[mask, 'settlement_end'] = reg_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    reg_df['settlement_start'] = reg_df['settlement_end'] - pd.Timedelta(minutes=15)
    reg_df['minute_in_qh'] = (reg_df['datetime_floor'] - reg_df['settlement_start']).dt.total_seconds() / 60

    pivot_reg = reg_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='regulation_mw', aggfunc='first'
    ).reset_index()
    pivot_reg.columns = ['datetime'] + [f'reg_min{int(c)}' for c in pivot_reg.columns[1:]]

    hist_cols = ['reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_min_10', 'reg_hist_max_10',
                 'reg_hist_range_10', 'reg_hist_trend_10', 'reg_hist_mean_20', 'reg_hist_std_20',
                 'reg_momentum', 'reg_acceleration']
    reg_min0 = reg_df[reg_df['minute_in_qh'] == 0][['settlement_start'] + hist_cols].copy()
    reg_min0 = reg_min0.rename(columns={'settlement_start': 'datetime'})

    df = pd.merge(label_df, pivot_reg, on='datetime', how='inner')
    df = pd.merge(df, reg_min0, on='datetime', how='left')

    load_df = load_df.copy()
    load_df['datetime_floor'] = load_df['datetime'].dt.floor('3min')
    load_df['settlement_end'] = load_df['datetime_floor'].dt.ceil('15min')
    mask = load_df['datetime_floor'] == load_df['settlement_end']
    load_df.loc[mask, 'settlement_end'] = load_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    load_df['settlement_start'] = load_df['settlement_end'] - pd.Timedelta(minutes=15)
    load_df['minute_in_qh'] = (load_df['datetime_floor'] - load_df['settlement_start']).dt.total_seconds() / 60

    load_pivot = load_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='load_mw', aggfunc='first'
    ).reset_index()
    load_pivot.columns = ['datetime'] + [f'load_min{int(c)}' for c in load_pivot.columns[1:]]
    df = pd.merge(df, load_pivot, on='datetime', how='left')

    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def add_proxy_lag_features(df):
    """Add proxy-based lag features."""
    df = df.sort_values('datetime').copy()

    reg_cols = ['reg_min0', 'reg_min3', 'reg_min6', 'reg_min9', 'reg_min12']
    available_cols = [c for c in reg_cols if c in df.columns]
    df['period_proxy'] = -0.25 * df[available_cols].mean(axis=1)

    df['proxy_lag1'] = df['period_proxy'].shift(1)
    df['proxy_lag2'] = df['period_proxy'].shift(2)
    df['proxy_lag3'] = df['period_proxy'].shift(3)
    df['proxy_lag4'] = df['period_proxy'].shift(4)

    df['proxy_rolling_mean4'] = df['period_proxy'].shift(1).rolling(4).mean()
    df['proxy_rolling_std4'] = df['period_proxy'].shift(1).rolling(4).std()
    df['proxy_rolling_mean10'] = df['period_proxy'].shift(1).rolling(10).mean()
    df['proxy_rolling_std10'] = df['period_proxy'].shift(1).rolling(10).std()
    df['proxy_rolling_min10'] = df['period_proxy'].shift(1).rolling(10).min()
    df['proxy_rolling_max10'] = df['period_proxy'].shift(1).rolling(10).max()

    df['proxy_last_sign'] = np.sign(df['period_proxy'].shift(1))
    df['proxy_last_positive'] = (df['period_proxy'].shift(1) > 0).astype(int)
    df['proxy_last_negative'] = (df['period_proxy'].shift(1) < 0).astype(int)

    signs = np.sign(df['period_proxy'])
    sign_change = (signs != signs.shift(1)).astype(int)
    sign_change.iloc[0] = 1
    groups = sign_change.cumsum()
    df['proxy_consecutive_same_sign'] = groups.groupby(groups).cumcount().shift(1)

    positive = (df['period_proxy'] > 0).astype(float)
    df['proxy_prop_positive_4'] = positive.shift(1).rolling(4).mean()
    df['proxy_prop_positive_10'] = positive.shift(1).rolling(10).mean()

    df['proxy_momentum'] = df['period_proxy'].shift(1) - df['period_proxy'].shift(2)
    df['proxy_acceleration'] = df['proxy_momentum'] - df['proxy_momentum'].shift(1)
    df['proxy_deviation_from_mean'] = df['period_proxy'].shift(1) - df['proxy_rolling_mean10']

    short_std = df['period_proxy'].shift(1).rolling(4).std()
    long_std = df['period_proxy'].shift(1).rolling(20, min_periods=5).std()
    df['proxy_volatility_ratio'] = short_std / long_std.clip(lower=0.1)
    df['proxy_high_volatility'] = (df['proxy_volatility_ratio'] > 1.5).astype(int)

    return df


def compute_lead_features(df, lead_time, load_expected):
    """Compute lead-time specific features."""
    result = df.copy()

    available_minutes = {
        12: [0], 9: [0, 3], 6: [0, 3, 6], 3: [0, 3, 6, 9], 0: [0, 3, 6, 9, 12]
    }

    mins = available_minutes[lead_time]
    reg_cols = [f'reg_min{m}' for m in mins]
    load_cols = [f'load_min{m}' for m in mins if f'load_min{m}' in df.columns]

    result['reg_cumulative_mean'] = df[reg_cols].mean(axis=1)

    if lead_time == 12:
        result['baseline_pred'] = -0.25 * df['reg_min0']
    elif lead_time == 9:
        result['baseline_pred'] = -0.25 * (0.8 * df['reg_min3'] + 0.2 * df['reg_min0'])
    elif lead_time == 6:
        result['baseline_pred'] = -0.25 * (0.6 * df['reg_min6'] + 0.2 * df['reg_min3'] + 0.2 * df['reg_min0'])
    elif lead_time == 3:
        result['baseline_pred'] = -0.25 * (0.4 * df['reg_min9'] + 0.2 * df['reg_min6'] +
                                            0.2 * df['reg_min3'] + 0.2 * df['reg_min0'])
    else:
        result['baseline_pred'] = -0.25 * df[reg_cols].mean(axis=1)

    if len(load_cols) > 0:
        load_mean = df[load_cols].mean(axis=1)
        temp = df[['hour', 'is_weekend']].copy()
        temp['minute'] = mins[0]
        temp = temp.merge(load_expected.reset_index(), on=['hour', 'minute', 'is_weekend'], how='left')
        result['load_deviation'] = load_mean - temp['expected_load'].values
    else:
        result['load_deviation'] = 0

    if len(reg_cols) >= 2:
        result['reg_std'] = df[reg_cols].std(axis=1)
        result['reg_range'] = df[reg_cols].max(axis=1) - df[reg_cols].min(axis=1)
        result['reg_trend'] = df[reg_cols[-1]] - df[reg_cols[0]]
        result['reg_min'] = df[reg_cols].min(axis=1)
        result['reg_max'] = df[reg_cols].max(axis=1)
    else:
        result['reg_std'] = 0
        result['reg_range'] = 0
        result['reg_trend'] = 0
        result['reg_min'] = df[reg_cols[0]]
        result['reg_max'] = df[reg_cols[0]]

    return result


def get_features_for_lead(lead_time):
    """Get optimized feature set for each lead time."""

    core = ['baseline_pred', 'reg_cumulative_mean']
    within_period = ['reg_std', 'reg_range', 'reg_trend']
    hist_reg_basic = ['reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_trend_10']
    hist_reg_extended = ['reg_hist_min_10', 'reg_hist_max_10', 'reg_hist_range_10',
                         'reg_hist_mean_20', 'reg_hist_std_20']
    hist_reg_momentum = ['reg_momentum', 'reg_acceleration']
    proxy_basic = ['proxy_lag1', 'proxy_rolling_mean4']
    proxy_extended = ['proxy_lag2', 'proxy_lag3', 'proxy_rolling_std4',
                      'proxy_rolling_mean10', 'proxy_rolling_std10']
    proxy_sign = ['proxy_last_sign', 'proxy_last_positive', 'proxy_consecutive_same_sign',
                  'proxy_prop_positive_4', 'proxy_prop_positive_10']
    proxy_momentum = ['proxy_momentum', 'proxy_acceleration', 'proxy_deviation_from_mean']
    proxy_volatility = ['proxy_volatility_ratio', 'proxy_high_volatility']
    time_basic = ['hour_sin', 'hour_cos']
    time_extended = ['is_weekend', 'dow_sin', 'dow_cos']
    other = ['load_deviation']

    if lead_time == 12:
        features = (core + hist_reg_basic + hist_reg_extended + hist_reg_momentum +
                   proxy_basic + proxy_extended + proxy_sign + proxy_momentum +
                   proxy_volatility + time_basic + time_extended + other)
    elif lead_time == 9:
        features = (core + within_period + hist_reg_basic + hist_reg_extended +
                   proxy_basic + proxy_extended + proxy_sign[:3] + proxy_momentum[:2] +
                   time_basic + other)
    elif lead_time == 6:
        features = (core + within_period + hist_reg_basic[:2] +
                   proxy_basic + proxy_extended[:2] + proxy_sign[:2] +
                   time_basic)
    elif lead_time == 3:
        features = (core + within_period + hist_reg_basic[:1] +
                   proxy_basic + proxy_sign[:1] + time_basic[:1])
    else:
        features = (core + within_period + proxy_basic[:1] + proxy_sign[:1])

    return features


# =============================================================================
# MAIN TRAINING
# =============================================================================
if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    load_expected = compute_load_expected(load_df)
    print(f"Expected load patterns: {len(load_expected)}")

    reg_df = add_historical_regulation_features(reg_df)
    print("Historical regulation features added")

    df = create_base_features(reg_df, load_df, label_df, load_expected)
    print(f"Base features: {len(df):,} periods, {len(df.columns)} columns")

    df = add_proxy_lag_features(df)
    print(f"Proxy features added: {len(df.columns)} total columns")

    # Train/test split
    test_start = pd.Timestamp('2025-11-01')
    print(f"\nTrain/Test split: {test_start}")

    # =============================================================================
    # TRAINING LOOP
    # =============================================================================
    models = {}
    results = []
    feature_importance_all = []

    for lead in LEAD_TIMES:
        print(f"\n{'='*70}")
        print(f"TRAINING LEAD TIME: {lead} MINUTES")
        print(f"{'='*70}")

        lead_df = compute_lead_features(df, lead, load_expected)

        train_df = lead_df[lead_df['datetime'] < test_start].copy()
        test_df = lead_df[lead_df['datetime'] >= test_start].copy()

        feature_cols = get_features_for_lead(lead)
        feature_cols = [c for c in feature_cols if c in lead_df.columns]

        print(f"Features: {len(feature_cols)}")

        train_clean = train_df.dropna(subset=feature_cols + ['imbalance'])
        X_train = train_clean[feature_cols].values
        y_train = train_clean['imbalance'].values

        print(f"Training samples: {len(X_train):,}")

        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(PARAMS, train_data, num_boost_round=500)
        models[lead] = model

        test_clean = test_df.dropna(subset=feature_cols + ['imbalance'])
        X_test = test_clean[feature_cols].values
        y_test = test_clean['imbalance'].values
        baseline = test_clean['baseline_pred'].values

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        dir_acc = np.mean(np.sign(y_test) == np.sign(y_pred)) * 100

        baseline_mae = mean_absolute_error(y_test, baseline)
        baseline_dir_acc = np.mean(np.sign(y_test) == np.sign(baseline)) * 100

        results.append({
            'lead_time': lead,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'dir_acc': dir_acc,
            'baseline_mae': baseline_mae,
            'baseline_dir_acc': baseline_dir_acc,
            'n_features': len(feature_cols),
            'n_samples': len(y_test)
        })

        importance = model.feature_importance(importance_type='gain')
        total_imp = importance.sum()
        for feat, imp in zip(feature_cols, importance):
            feature_importance_all.append({
                'lead_time': lead,
                'feature': feat,
                'importance': imp,
                'importance_pct': imp / total_imp * 100
            })

        print(f"MAE: {mae:.3f} (Baseline: {baseline_mae:.3f}) | Dir Acc: {dir_acc:.1f}%")

    # =============================================================================
    # RESULTS SUMMARY
    # =============================================================================
    print("\n" + "=" * 90)
    print("PERFORMANCE SUMMARY: V4 MODEL vs BASELINE")
    print("=" * 90)

    results_df = pd.DataFrame(results)
    results_df['improvement'] = (1 - results_df['mae'] / results_df['baseline_mae']) * 100

    print(f"\n{'Lead':<6} {'Feats':<8} {'V4 MAE':<10} {'Base MAE':<12} {'Improve':<12} {'V4 Dir%':<10} {'Base Dir%':<10}")
    print("-" * 90)

    for _, row in results_df.iterrows():
        print(f"{int(row['lead_time']):<6} {int(row['n_features']):<8} {row['mae']:<10.3f} {row['baseline_mae']:<12.3f} "
              f"{row['improvement']:>+10.1f}%  {row['dir_acc']:<10.1f} {row['baseline_dir_acc']:<10.1f}")

    print("-" * 90)
    print(f"\nAverage MAE improvement: {results_df['improvement'].mean():.1f}%")

    # Feature importance by lead time
    print("\n" + "=" * 70)
    print("TOP 5 FEATURES PER LEAD TIME")
    print("=" * 70)

    importance_df = pd.DataFrame(feature_importance_all)
    for lead in LEAD_TIMES:
        lead_imp = importance_df[importance_df['lead_time'] == lead].nlargest(5, 'importance_pct')
        print(f"\nLead {lead} min:")
        for _, row in lead_imp.iterrows():
            print(f"  {row['feature']:<35} {row['importance_pct']:>6.2f}%")

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'v4_standalone_results.csv', index=False)
    importance_df.to_csv(OUTPUT_DIR / 'v4_standalone_feature_importance.csv', index=False)

    print(f"\n\nResults saved to: {OUTPUT_DIR}")
    print("\nTRAINING COMPLETE!")
