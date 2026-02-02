"""
LightGBM v3 - Optimized Features per Lead Time

Key changes:
1. Use ACTUAL past imbalance values for lag features (not just proxy)
2. More aggressive feature selection per lead
3. Add cross-period patterns (same hour yesterday, same hour last week)
4. Simpler model for short leads, richer model for long leads
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\models\lightgbm")


def load_data():
    """Load all data."""
    print("Loading data...")
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
    load_df = pd.read_csv(FEATURES_DIR / 'load_3min.csv', parse_dates=['datetime'])
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
        columns={'System Imbalance (MWh)': 'imbalance'}
    )
    return reg_df, load_df, label_df


def compute_load_tod_expected(load_df):
    """Compute expected load by time-of-day."""
    load_df = load_df.copy()
    load_df['hour'] = load_df['datetime'].dt.hour
    load_df['minute'] = load_df['datetime'].dt.minute
    load_df['is_weekend'] = load_df['datetime'].dt.dayofweek >= 5
    train_mask = load_df['datetime'].dt.year == 2024
    expected = load_df[train_mask].groupby(['hour', 'minute', 'is_weekend'])['load_mw'].mean()
    expected.name = 'expected_load'
    return expected


def create_base_features(reg_df, load_df, label_df, load_expected):
    """Create base feature dataframe with regulation pivoted."""
    print("\nCreating base features...")

    # Align regulation to settlement periods
    reg_df = reg_df.copy()
    reg_df['datetime_floor'] = reg_df['datetime'].dt.floor('3min')
    reg_df['settlement_end'] = reg_df['datetime_floor'].dt.ceil('15min')
    mask = reg_df['datetime_floor'] == reg_df['settlement_end']
    reg_df.loc[mask, 'settlement_end'] = reg_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    reg_df['settlement_start'] = reg_df['settlement_end'] - pd.Timedelta(minutes=15)
    reg_df['minute_in_qh'] = (reg_df['datetime_floor'] - reg_df['settlement_start']).dt.total_seconds() / 60

    pivot = reg_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='regulation_mw', aggfunc='first'
    ).reset_index()
    pivot.columns = ['datetime'] + [f'reg_min{int(c)}' for c in pivot.columns[1:]]

    df = pd.merge(label_df, pivot, on='datetime', how='inner')

    # Pivot load
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

    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    print(f"  Base features: {len(df):,} rows")
    return df


def add_actual_imbalance_lags(df):
    """
    Add lag features using ACTUAL past imbalance values.
    This is the key difference from v2 - we use real imbalance, not proxy.
    """
    df = df.sort_values('datetime').copy()

    # Actual imbalance lags (these are known at prediction time)
    df['imb_lag1'] = df['imbalance'].shift(1)  # Previous 15-min period
    df['imb_lag2'] = df['imbalance'].shift(2)
    df['imb_lag3'] = df['imbalance'].shift(3)
    df['imb_lag4'] = df['imbalance'].shift(4)

    # Rolling statistics on actual imbalance
    df['imb_rolling_mean4'] = df['imbalance'].shift(1).rolling(4).mean()
    df['imb_rolling_std4'] = df['imbalance'].shift(1).rolling(4).std()
    df['imb_rolling_mean10'] = df['imbalance'].shift(1).rolling(10).mean()
    df['imb_rolling_std10'] = df['imbalance'].shift(1).rolling(10).std()
    df['imb_rolling_min10'] = df['imbalance'].shift(1).rolling(10).min()
    df['imb_rolling_max10'] = df['imbalance'].shift(1).rolling(10).max()

    # Trend over last 10 periods
    df['imb_trend_10'] = df['imbalance'].shift(1) - df['imbalance'].shift(10)

    # Sign-based features on actual imbalance
    df['last_imb_sign'] = np.sign(df['imbalance'].shift(1))
    df['last_imb_positive'] = (df['imbalance'].shift(1) > 0).astype(int)

    # Consecutive same-sign count
    signs = np.sign(df['imbalance'])
    sign_change = (signs != signs.shift(1)).astype(int)
    sign_change.iloc[0] = 1
    groups = sign_change.cumsum()
    df['consecutive_same_sign'] = groups.groupby(groups).cumcount().shift(1)

    # Proportion positive
    positive = (df['imbalance'] > 0).astype(float)
    df['prop_positive_4'] = positive.shift(1).rolling(4).mean()
    df['prop_positive_10'] = positive.shift(1).rolling(10).mean()

    # Momentum on actual imbalance
    df['imb_momentum'] = df['imbalance'].shift(1) - df['imbalance'].shift(2)
    df['imb_acceleration'] = df['imb_momentum'] - df['imb_momentum'].shift(1)

    # Same hour yesterday (96 periods ago for 15-min data)
    df['imb_same_hour_yesterday'] = df['imbalance'].shift(96)

    # Same hour, same day last week (96*7 = 672 periods)
    df['imb_same_hour_last_week'] = df['imbalance'].shift(672)

    # Hour-of-day mean imbalance (computed from training data)
    hour_means = df.groupby('hour')['imbalance'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df['hour_mean_imbalance'] = hour_means

    return df


def compute_lead_features(df, lead_time, load_expected):
    """Compute lead-time specific features."""
    result = df.copy()
    result['lead_time'] = lead_time

    available_minutes = {
        12: [0], 9: [0, 3], 6: [0, 3, 6], 3: [0, 3, 6, 9], 0: [0, 3, 6, 9, 12]
    }

    mins = available_minutes[lead_time]
    reg_cols = [f'reg_min{m}' for m in mins]
    load_cols = [f'load_min{m}' for m in mins if f'load_min{m}' in df.columns]

    # Core features
    result['reg_cumulative_mean'] = df[reg_cols].mean(axis=1)

    # Baseline prediction
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

    # Load deviation
    if len(load_cols) > 0:
        load_mean = df[load_cols].mean(axis=1)
        temp = df[['hour', 'is_weekend']].copy()
        temp['minute'] = mins[0]
        temp = temp.merge(load_expected.reset_index(), on=['hour', 'minute', 'is_weekend'], how='left')
        result['load_deviation'] = load_mean - temp['expected_load'].values
    else:
        result['load_deviation'] = 0

    # Within-period stats
    if len(reg_cols) >= 2:
        result['reg_std'] = df[reg_cols].std(axis=1)
        result['reg_range'] = df[reg_cols].max(axis=1) - df[reg_cols].min(axis=1)
        result['reg_trend'] = df[reg_cols[-1]] - df[reg_cols[0]]
    else:
        result['reg_std'] = 0
        result['reg_range'] = 0
        result['reg_trend'] = 0

    return result


def get_features_for_lead(lead_time):
    """Get optimized feature set for each lead time."""

    # Features available at all leads
    core = ['baseline_pred', 'reg_cumulative_mean']
    within_period = ['reg_std', 'reg_range', 'reg_trend']
    time = ['hour_sin', 'hour_cos', 'is_weekend']

    # Actual imbalance lag features
    imb_lags_basic = ['imb_lag1', 'imb_rolling_mean4']
    imb_lags_extended = ['imb_lag2', 'imb_lag3', 'imb_lag4', 'imb_rolling_std4']
    imb_lags_long = ['imb_rolling_mean10', 'imb_rolling_std10', 'imb_rolling_min10',
                     'imb_rolling_max10', 'imb_trend_10']

    # Sign features
    sign_basic = ['last_imb_sign', 'last_imb_positive']
    sign_extended = ['consecutive_same_sign', 'prop_positive_4', 'prop_positive_10']

    # Momentum
    momentum = ['imb_momentum', 'imb_acceleration']

    # Cross-period
    cross_period = ['imb_same_hour_yesterday', 'imb_same_hour_last_week', 'hour_mean_imbalance']

    other = ['load_deviation']

    if lead_time == 12:
        # Maximum features for longest lead
        features = (core + imb_lags_basic + imb_lags_extended + imb_lags_long +
                   sign_basic + sign_extended + momentum + cross_period +
                   time + other)
    elif lead_time == 9:
        features = (core + within_period + imb_lags_basic + imb_lags_extended +
                   imb_lags_long[:3] + sign_basic + sign_extended[:2] +
                   momentum + cross_period[:2] + time + other)
    elif lead_time == 6:
        features = (core + within_period + imb_lags_basic + imb_lags_extended[:2] +
                   imb_lags_long[:2] + sign_basic + momentum[:1] +
                   cross_period[:1] + time[:2])
    elif lead_time == 3:
        features = (core + within_period + imb_lags_basic + sign_basic[:1] +
                   cross_period[:1] + time[:2])
    else:  # lead 0
        features = core + within_period + imb_lags_basic[:1] + sign_basic[:1]

    return features


def train_model(train_df, feature_cols):
    """Train LightGBM."""
    X = train_df[feature_cols].values
    y = train_df['imbalance'].values

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isinf(X).any(axis=1)
    X, y = X[valid], y[valid]

    params = {
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

    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=500)
    return model


def evaluate(model, test_df, feature_cols):
    """Evaluate model."""
    X = test_df[feature_cols].values
    y_true = test_df['imbalance'].values

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y_true) & ~np.isinf(X).any(axis=1)
    X, y_true = X[valid], y_true[valid]

    y_pred = model.predict(X)

    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'directional_accuracy': (np.sign(y_pred) == np.sign(y_true)).mean(),
        'n_samples': len(y_true),
    }


def main():
    print("=" * 70)
    print("LIGHTGBM v3 - ACTUAL IMBALANCE LAGS + CROSS-PERIOD FEATURES")
    print("=" * 70)

    reg_df, load_df, label_df = load_data()
    load_expected = compute_load_tod_expected(load_df)
    df = create_base_features(reg_df, load_df, label_df, load_expected)

    # Add actual imbalance lag features (before splitting by lead)
    print("\nAdding actual imbalance lag features...")
    df = add_actual_imbalance_lags(df)

    lead_times = [12, 9, 6, 3, 0]
    test_start = pd.Timestamp('2025-10-01')

    results = []
    models = {}

    for lead in lead_times:
        print(f"\n{'='*60}")
        print(f"LEAD TIME: {lead} MINUTES")
        print(f"{'='*60}")

        lead_df = compute_lead_features(df, lead, load_expected)
        train_df = lead_df[lead_df['datetime'] < test_start]
        test_df = lead_df[lead_df['datetime'] >= test_start]

        feature_cols = get_features_for_lead(lead)
        feature_cols = [c for c in feature_cols if c in lead_df.columns]

        print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
        print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

        model = train_model(train_df, feature_cols)
        metrics = evaluate(model, test_df, feature_cols)

        print(f"\nTest Performance:")
        print(f"  MAE: {metrics['mae']:.3f} MWh")
        print(f"  R2: {metrics['r2']:.3f}")
        print(f"  Dir.Acc: {metrics['directional_accuracy']:.1%}")

        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importance})
        imp_df = imp_df.sort_values('importance', ascending=False)
        total = imp_df['importance'].sum()

        print(f"\nTop Features:")
        for _, row in imp_df.head(8).iterrows():
            print(f"  {row['feature']:<30} {row['importance']/total*100:>5.1f}%")

        results.append({'lead_time': lead, **metrics})
        models[lead] = model

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'lightgbm_v3_results.csv', index=False)

    with open(OUTPUT_DIR / 'lightgbm_models_v3.pkl', 'wb') as f:
        pickle.dump(models, f)

    # Comparison
    print("\n" + "=" * 70)
    print("V1 vs V2 vs V3 COMPARISON")
    print("=" * 70)

    v1 = pd.read_csv(OUTPUT_DIR / 'lightgbm_results.csv')
    v2 = pd.read_csv(OUTPUT_DIR / 'lightgbm_v2_results.csv')

    print(f"\n{'Lead':<6} {'V1 MAE':<10} {'V2 MAE':<10} {'V3 MAE':<10} {'V1 Dir':<10} {'V2 Dir':<10} {'V3 Dir':<10}")
    print("-" * 66)

    for _, v3_row in results_df.iterrows():
        lead = int(v3_row['lead_time'])
        v1_row = v1[v1['lead_time'] == lead].iloc[0]
        v2_row = v2[v2['lead_time'] == lead].iloc[0]

        print(f"{lead:<6} {v1_row['mae']:<10.3f} {v2_row['mae']:<10.3f} {v3_row['mae']:<10.3f} "
              f"{v1_row['directional_accuracy']*100:<10.1f} {v2_row['directional_accuracy']*100:<10.1f} "
              f"{v3_row['directional_accuracy']*100:<10.1f}")

    # Best improvements
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUMMARY (vs V1 baseline)")
    print("=" * 70)

    print(f"\n{'Lead':<8} {'V1 MAE':<12} {'Best MAE':<12} {'Improvement':<12} {'Best Version':<12}")
    print("-" * 56)

    for lead in lead_times:
        v1_mae = v1[v1['lead_time'] == lead]['mae'].values[0]
        v2_mae = v2[v2['lead_time'] == lead]['mae'].values[0]
        v3_mae = results_df[results_df['lead_time'] == lead]['mae'].values[0]

        best_mae = min(v1_mae, v2_mae, v3_mae)
        best_ver = 'V1' if best_mae == v1_mae else ('V2' if best_mae == v2_mae else 'V3')
        improvement = (v1_mae - best_mae) / v1_mae * 100

        print(f"{lead:<8} {v1_mae:<12.3f} {best_mae:<12.3f} {improvement:>+10.1f}% {best_ver:<12}")


if __name__ == '__main__':
    main()
