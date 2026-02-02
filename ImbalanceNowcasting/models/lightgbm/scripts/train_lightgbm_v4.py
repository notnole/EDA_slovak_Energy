"""
LightGBM v4 - REAL-TIME ONLY FEATURES

CRITICAL: Actual imbalance values are NOT available until next day!
We can ONLY use:
1. Regulation data (3-min real-time)
2. Load data (3-min real-time)
3. Time features
4. Proxy-based features (computed from regulation)
5. Historical REGULATION statistics (last 10 3-min updates)
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
    """Load data."""
    print("Loading data...")
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
    load_df = pd.read_csv(FEATURES_DIR / 'load_3min.csv', parse_dates=['datetime'])
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
        columns={'System Imbalance (MWh)': 'imbalance'}
    )
    return reg_df, load_df, label_df


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
    """
    Add historical regulation features from the RAW 3-min data.
    These are statistics over the last N 3-minute observations BEFORE the current period.
    """
    reg_df = reg_df.sort_values('datetime').copy()

    # Statistics over last 10 observations (30 minutes of history)
    reg_df['reg_hist_mean_10'] = reg_df['regulation_mw'].shift(1).rolling(10).mean()
    reg_df['reg_hist_std_10'] = reg_df['regulation_mw'].shift(1).rolling(10).std()
    reg_df['reg_hist_min_10'] = reg_df['regulation_mw'].shift(1).rolling(10).min()
    reg_df['reg_hist_max_10'] = reg_df['regulation_mw'].shift(1).rolling(10).max()
    reg_df['reg_hist_range_10'] = reg_df['reg_hist_max_10'] - reg_df['reg_hist_min_10']

    # Trend: last observation minus 10 observations ago
    reg_df['reg_hist_trend_10'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(10)

    # Statistics over last 20 observations (1 hour of history)
    reg_df['reg_hist_mean_20'] = reg_df['regulation_mw'].shift(1).rolling(20).mean()
    reg_df['reg_hist_std_20'] = reg_df['regulation_mw'].shift(1).rolling(20).std()

    # Recent momentum (change in regulation)
    reg_df['reg_momentum'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(2)
    reg_df['reg_acceleration'] = reg_df['reg_momentum'] - reg_df['reg_momentum'].shift(1)

    return reg_df


def create_base_features(reg_df, load_df, label_df, load_expected):
    """Create base feature dataframe."""
    print("\nCreating base features...")

    # First add historical regulation features to raw 3-min data
    reg_df = add_historical_regulation_features(reg_df)

    # Align regulation to settlement periods
    reg_df['datetime_floor'] = reg_df['datetime'].dt.floor('3min')
    reg_df['settlement_end'] = reg_df['datetime_floor'].dt.ceil('15min')
    mask = reg_df['datetime_floor'] == reg_df['settlement_end']
    reg_df.loc[mask, 'settlement_end'] = reg_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    reg_df['settlement_start'] = reg_df['settlement_end'] - pd.Timedelta(minutes=15)
    reg_df['minute_in_qh'] = (reg_df['datetime_floor'] - reg_df['settlement_start']).dt.total_seconds() / 60

    # Pivot regulation values
    pivot_reg = reg_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='regulation_mw', aggfunc='first'
    ).reset_index()
    pivot_reg.columns = ['datetime'] + [f'reg_min{int(c)}' for c in pivot_reg.columns[1:]]

    # Pivot historical features (take from first observation of each period = minute 0)
    hist_cols = ['reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_min_10', 'reg_hist_max_10',
                 'reg_hist_range_10', 'reg_hist_trend_10', 'reg_hist_mean_20', 'reg_hist_std_20',
                 'reg_momentum', 'reg_acceleration']

    # Get historical features at minute 0 of each period (the first observation)
    reg_min0 = reg_df[reg_df['minute_in_qh'] == 0][['settlement_start'] + hist_cols].copy()
    reg_min0 = reg_min0.rename(columns={'settlement_start': 'datetime'})

    # Merge with labels
    df = pd.merge(label_df, pivot_reg, on='datetime', how='inner')
    df = pd.merge(df, reg_min0, on='datetime', how='left')

    # Load features
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
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    print(f"  Base features: {len(df):,} rows")
    return df


def add_proxy_lag_features(df):
    """
    Add proxy-based lag features.
    Proxy = -0.25 * mean(regulation) for each prior period.
    """
    df = df.sort_values('datetime').copy()

    # Compute proxy for each period using all 5 regulation observations
    reg_cols = ['reg_min0', 'reg_min3', 'reg_min6', 'reg_min9', 'reg_min12']
    available_cols = [c for c in reg_cols if c in df.columns]
    df['period_proxy'] = -0.25 * df[available_cols].mean(axis=1)

    # Lag features based on proxy
    df['proxy_lag1'] = df['period_proxy'].shift(1)
    df['proxy_lag2'] = df['period_proxy'].shift(2)
    df['proxy_lag3'] = df['period_proxy'].shift(3)
    df['proxy_lag4'] = df['period_proxy'].shift(4)

    # Rolling statistics on proxy
    df['proxy_rolling_mean4'] = df['period_proxy'].shift(1).rolling(4).mean()
    df['proxy_rolling_std4'] = df['period_proxy'].shift(1).rolling(4).std()
    df['proxy_rolling_mean10'] = df['period_proxy'].shift(1).rolling(10).mean()
    df['proxy_rolling_std10'] = df['period_proxy'].shift(1).rolling(10).std()
    df['proxy_rolling_min10'] = df['period_proxy'].shift(1).rolling(10).min()
    df['proxy_rolling_max10'] = df['period_proxy'].shift(1).rolling(10).max()

    # Sign-based features on proxy
    df['proxy_last_sign'] = np.sign(df['period_proxy'].shift(1))
    df['proxy_last_positive'] = (df['period_proxy'].shift(1) > 0).astype(int)
    df['proxy_last_negative'] = (df['period_proxy'].shift(1) < 0).astype(int)

    # Consecutive same-sign counter
    signs = np.sign(df['period_proxy'])
    sign_change = (signs != signs.shift(1)).astype(int)
    sign_change.iloc[0] = 1
    groups = sign_change.cumsum()
    df['proxy_consecutive_same_sign'] = groups.groupby(groups).cumcount().shift(1)

    # Proportion positive in last N periods
    positive = (df['period_proxy'] > 0).astype(float)
    df['proxy_prop_positive_4'] = positive.shift(1).rolling(4).mean()
    df['proxy_prop_positive_10'] = positive.shift(1).rolling(10).mean()

    # Momentum on proxy
    df['proxy_momentum'] = df['period_proxy'].shift(1) - df['period_proxy'].shift(2)
    df['proxy_acceleration'] = df['proxy_momentum'] - df['proxy_momentum'].shift(1)

    # Deviation from rolling mean
    df['proxy_deviation_from_mean'] = df['period_proxy'].shift(1) - df['proxy_rolling_mean10']

    # Volatility ratio
    short_std = df['period_proxy'].shift(1).rolling(4).std()
    long_std = df['period_proxy'].shift(1).rolling(20, min_periods=5).std()
    df['proxy_volatility_ratio'] = short_std / long_std.clip(lower=0.1)
    df['proxy_high_volatility'] = (df['proxy_volatility_ratio'] > 1.5).astype(int)

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

    # Baseline prediction (weighted by recency)
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

    # Within-period regulation stats
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

    # Core (always include)
    core = ['baseline_pred', 'reg_cumulative_mean']

    # Within-period regulation stats (need 2+ observations)
    within_period = ['reg_std', 'reg_range', 'reg_trend']

    # Historical regulation stats (from last 10 3-min observations BEFORE current period)
    hist_reg_basic = ['reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_trend_10']
    hist_reg_extended = ['reg_hist_min_10', 'reg_hist_max_10', 'reg_hist_range_10',
                         'reg_hist_mean_20', 'reg_hist_std_20']
    hist_reg_momentum = ['reg_momentum', 'reg_acceleration']

    # Proxy-based lag features
    proxy_basic = ['proxy_lag1', 'proxy_rolling_mean4']
    proxy_extended = ['proxy_lag2', 'proxy_lag3', 'proxy_rolling_std4',
                      'proxy_rolling_mean10', 'proxy_rolling_std10']
    proxy_sign = ['proxy_last_sign', 'proxy_last_positive', 'proxy_consecutive_same_sign',
                  'proxy_prop_positive_4', 'proxy_prop_positive_10']
    proxy_momentum = ['proxy_momentum', 'proxy_acceleration', 'proxy_deviation_from_mean']
    proxy_volatility = ['proxy_volatility_ratio', 'proxy_high_volatility']

    # Time features
    time_basic = ['hour_sin', 'hour_cos']
    time_extended = ['is_weekend', 'dow_sin', 'dow_cos']

    # Other
    other = ['load_deviation']

    if lead_time == 12:
        # Maximum features - we have minimal current-period info
        features = (core + hist_reg_basic + hist_reg_extended + hist_reg_momentum +
                   proxy_basic + proxy_extended + proxy_sign + proxy_momentum +
                   proxy_volatility + time_basic + time_extended + other)
    elif lead_time == 9:
        # Add within-period stats, still need historical
        features = (core + within_period + hist_reg_basic + hist_reg_extended +
                   proxy_basic + proxy_extended + proxy_sign[:3] + proxy_momentum[:2] +
                   time_basic + other)
    elif lead_time == 6:
        # More current info, reduce historical dependency
        features = (core + within_period + hist_reg_basic[:2] +
                   proxy_basic + proxy_extended[:2] + proxy_sign[:2] +
                   time_basic)
    elif lead_time == 3:
        # Strong current-period info
        features = (core + within_period + hist_reg_basic[:1] +
                   proxy_basic + proxy_sign[:1] + time_basic[:1])
    else:  # lead 0
        # Best current-period info, minimal history needed
        features = (core + within_period + proxy_basic[:1] + proxy_sign[:1])

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
    print("LIGHTGBM v4 - REAL-TIME FEATURES ONLY")
    print("(No actual imbalance values - proxy and regulation history only)")
    print("=" * 70)

    reg_df, load_df, label_df = load_data()
    load_expected = compute_load_expected(load_df)
    df = create_base_features(reg_df, load_df, label_df, load_expected)

    # Add proxy-based lag features
    print("\nAdding proxy-based lag features...")
    df = add_proxy_lag_features(df)

    lead_times = [12, 9, 6, 3, 0]
    test_start = pd.Timestamp('2025-10-01')

    results = []
    models = {}
    all_importance = []

    for lead in lead_times:
        print(f"\n{'='*60}")
        print(f"LEAD TIME: {lead} MINUTES")
        print(f"{'='*60}")

        lead_df = compute_lead_features(df, lead, load_expected)
        train_df = lead_df[lead_df['datetime'] < test_start]
        test_df = lead_df[lead_df['datetime'] >= test_start]

        feature_cols = get_features_for_lead(lead)
        feature_cols = [c for c in feature_cols if c in lead_df.columns]

        print(f"\nFeatures ({len(feature_cols)}):")
        print(f"  {feature_cols}")
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

        print(f"\nTop 10 Features:")
        for _, row in imp_df.head(10).iterrows():
            all_importance.append({'lead_time': lead, 'feature': row['feature'],
                                   'importance': row['importance'], 'pct': row['importance']/total*100})
            print(f"  {row['feature']:<35} {row['importance']/total*100:>5.1f}%")

        results.append({'lead_time': lead, **metrics})
        models[lead] = model

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'lightgbm_v4_results.csv', index=False)

    importance_df = pd.DataFrame(all_importance)
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance_v4.csv', index=False)

    with open(OUTPUT_DIR / 'lightgbm_models_v4.pkl', 'wb') as f:
        pickle.dump(models, f)

    # Comparison with V1 and V2
    print("\n" + "=" * 70)
    print("COMPARISON: V1 vs V2 vs V4 (V3 invalid - used actual imbalance)")
    print("=" * 70)

    v1 = pd.read_csv(OUTPUT_DIR / 'lightgbm_results.csv')
    v2 = pd.read_csv(OUTPUT_DIR / 'lightgbm_v2_results.csv')

    print(f"\n{'Lead':<6} {'V1 MAE':<10} {'V2 MAE':<10} {'V4 MAE':<10} {'V4 vs V1':<12} {'V1 Dir':<8} {'V4 Dir':<8}")
    print("-" * 72)

    for _, v4_row in results_df.iterrows():
        lead = int(v4_row['lead_time'])
        v1_row = v1[v1['lead_time'] == lead].iloc[0]
        v2_row = v2[v2['lead_time'] == lead].iloc[0]

        v4_vs_v1 = (v4_row['mae'] - v1_row['mae']) / v1_row['mae'] * 100

        print(f"{lead:<6} {v1_row['mae']:<10.3f} {v2_row['mae']:<10.3f} {v4_row['mae']:<10.3f} "
              f"{v4_vs_v1:>+10.1f}% {v1_row['directional_accuracy']*100:<8.1f} "
              f"{v4_row['directional_accuracy']*100:<8.1f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
