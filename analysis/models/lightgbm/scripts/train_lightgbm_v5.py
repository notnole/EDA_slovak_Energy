"""
LightGBM v5 - HYBRID MODEL

- Lead 12, 9: Rich feature set with historical patterns, load features, regime detection
- Lead 6, 3, 0: Simpler feature set focused on current-period info

All features are REAL-TIME available (no actual imbalance values).
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


def add_3min_historical_features(reg_df, load_df):
    """
    Add historical features computed from raw 3-minute data.
    These capture patterns BEFORE the current settlement period.
    """
    reg_df = reg_df.sort_values('datetime').copy()
    load_df = load_df.sort_values('datetime').copy()

    # ================================================================
    # REGULATION HISTORICAL FEATURES
    # ================================================================

    # Basic stats over last 10 observations (30 min)
    reg_df['reg_hist_mean_10'] = reg_df['regulation_mw'].shift(1).rolling(10).mean()
    reg_df['reg_hist_std_10'] = reg_df['regulation_mw'].shift(1).rolling(10).std()
    reg_df['reg_hist_min_10'] = reg_df['regulation_mw'].shift(1).rolling(10).min()
    reg_df['reg_hist_max_10'] = reg_df['regulation_mw'].shift(1).rolling(10).max()
    reg_df['reg_hist_range_10'] = reg_df['reg_hist_max_10'] - reg_df['reg_hist_min_10']
    reg_df['reg_hist_trend_10'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(10)

    # Stats over last 20 observations (1 hour)
    reg_df['reg_hist_mean_20'] = reg_df['regulation_mw'].shift(1).rolling(20).mean()
    reg_df['reg_hist_std_20'] = reg_df['regulation_mw'].shift(1).rolling(20).std()

    # Momentum and acceleration
    reg_df['reg_momentum'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(2)
    reg_df['reg_acceleration'] = reg_df['reg_momentum'] - reg_df['reg_momentum'].shift(1)

    # Distribution features
    # Skewness approximation (using standardized 3rd moment)
    def rolling_skew(x):
        return x.rolling(10).apply(lambda v: ((v - v.mean()) ** 3).mean() / (v.std() ** 3 + 1e-6), raw=True)
    reg_df['reg_hist_skew_10'] = rolling_skew(reg_df['regulation_mw'].shift(1))

    # Percentile rank: where does current value stand vs last 100 observations?
    reg_df['reg_percentile_100'] = reg_df['regulation_mw'].shift(1).rolling(100).apply(
        lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    )

    # Stability indicator (low std = stable)
    long_std = reg_df['regulation_mw'].shift(1).rolling(50, min_periods=10).std()
    reg_df['reg_stability'] = 1 / (1 + reg_df['reg_hist_std_10'] / long_std.clip(lower=0.1))

    # ================================================================
    # LOAD HISTORICAL FEATURES
    # ================================================================

    # Basic stats
    load_df['load_hist_mean_10'] = load_df['load_mw'].shift(1).rolling(10).mean()
    load_df['load_hist_std_10'] = load_df['load_mw'].shift(1).rolling(10).std()
    load_df['load_hist_trend_10'] = load_df['load_mw'].shift(1) - load_df['load_mw'].shift(10)

    # Load momentum
    load_df['load_momentum'] = load_df['load_mw'].shift(1) - load_df['load_mw'].shift(2)
    load_df['load_acceleration'] = load_df['load_momentum'] - load_df['load_momentum'].shift(1)

    return reg_df, load_df


def create_base_features(reg_df, load_df, label_df, load_expected):
    """Create base feature dataframe with settlement-period alignment."""
    print("\nCreating base features...")

    # Add 3-min historical features first
    reg_df, load_df = add_3min_historical_features(reg_df, load_df)

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

    # Get historical regulation features at minute 0 of each period
    hist_reg_cols = ['reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_min_10', 'reg_hist_max_10',
                     'reg_hist_range_10', 'reg_hist_trend_10', 'reg_hist_mean_20', 'reg_hist_std_20',
                     'reg_momentum', 'reg_acceleration', 'reg_hist_skew_10', 'reg_percentile_100',
                     'reg_stability']
    reg_min0 = reg_df[reg_df['minute_in_qh'] == 0][['settlement_start'] + hist_reg_cols].copy()
    reg_min0 = reg_min0.rename(columns={'settlement_start': 'datetime'})

    # Merge with labels
    df = pd.merge(label_df, pivot_reg, on='datetime', how='inner')
    df = pd.merge(df, reg_min0, on='datetime', how='left')

    # Align load to settlement periods
    load_df['datetime_floor'] = load_df['datetime'].dt.floor('3min')
    load_df['settlement_end'] = load_df['datetime_floor'].dt.ceil('15min')
    mask = load_df['datetime_floor'] == load_df['settlement_end']
    load_df.loc[mask, 'settlement_end'] = load_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    load_df['settlement_start'] = load_df['settlement_end'] - pd.Timedelta(minutes=15)
    load_df['minute_in_qh'] = (load_df['datetime_floor'] - load_df['settlement_start']).dt.total_seconds() / 60

    # Pivot load values
    load_pivot = load_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='load_mw', aggfunc='first'
    ).reset_index()
    load_pivot.columns = ['datetime'] + [f'load_min{int(c)}' for c in load_pivot.columns[1:]]

    # Get historical load features at minute 0
    hist_load_cols = ['load_hist_mean_10', 'load_hist_std_10', 'load_hist_trend_10',
                      'load_momentum', 'load_acceleration']
    load_min0 = load_df[load_df['minute_in_qh'] == 0][['settlement_start'] + hist_load_cols].copy()
    load_min0 = load_min0.rename(columns={'settlement_start': 'datetime'})

    df = pd.merge(df, load_pivot, on='datetime', how='left')
    df = pd.merge(df, load_min0, on='datetime', how='left')

    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Hour group indicators
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 11)).astype(int)
    df['is_peak'] = ((df['hour'] >= 11) & (df['hour'] < 15)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 15) & (df['hour'] < 21)).astype(int)
    df['is_evening'] = ((df['hour'] >= 21) & (df['hour'] < 24)).astype(int)

    print(f"  Base features: {len(df):,} rows")
    return df


def add_proxy_features(df):
    """Add proxy-based features (computed from prior periods' regulation)."""
    df = df.sort_values('datetime').copy()

    # Compute period proxy using all 5 observations
    reg_cols = ['reg_min0', 'reg_min3', 'reg_min6', 'reg_min9', 'reg_min12']
    available = [c for c in reg_cols if c in df.columns]
    df['period_proxy'] = -0.25 * df[available].mean(axis=1)

    # Basic lags
    df['proxy_lag1'] = df['period_proxy'].shift(1)
    df['proxy_lag2'] = df['period_proxy'].shift(2)
    df['proxy_lag3'] = df['period_proxy'].shift(3)
    df['proxy_lag4'] = df['period_proxy'].shift(4)

    # Rolling stats
    df['proxy_rolling_mean4'] = df['period_proxy'].shift(1).rolling(4).mean()
    df['proxy_rolling_std4'] = df['period_proxy'].shift(1).rolling(4).std()
    df['proxy_rolling_mean10'] = df['period_proxy'].shift(1).rolling(10).mean()
    df['proxy_rolling_std10'] = df['period_proxy'].shift(1).rolling(10).std()
    df['proxy_rolling_min10'] = df['period_proxy'].shift(1).rolling(10).min()
    df['proxy_rolling_max10'] = df['period_proxy'].shift(1).rolling(10).max()

    # Sign features
    df['proxy_last_sign'] = np.sign(df['period_proxy'].shift(1))
    df['proxy_last_positive'] = (df['period_proxy'].shift(1) > 0).astype(int)

    # Consecutive same-sign counter
    signs = np.sign(df['period_proxy'])
    sign_change = (signs != signs.shift(1)).astype(int)
    sign_change.iloc[0] = 1
    groups = sign_change.cumsum()
    df['proxy_consecutive_same_sign'] = groups.groupby(groups).cumcount().shift(1)

    # Proportion positive
    positive = (df['period_proxy'] > 0).astype(float)
    df['proxy_prop_positive_4'] = positive.shift(1).rolling(4).mean()
    df['proxy_prop_positive_10'] = positive.shift(1).rolling(10).mean()

    # Momentum
    df['proxy_momentum'] = df['period_proxy'].shift(1) - df['period_proxy'].shift(2)
    df['proxy_acceleration'] = df['proxy_momentum'] - df['proxy_momentum'].shift(1)

    # Deviation from mean
    df['proxy_deviation_from_mean'] = df['period_proxy'].shift(1) - df['proxy_rolling_mean10']

    # Volatility regime
    short_std = df['period_proxy'].shift(1).rolling(4).std()
    long_std = df['period_proxy'].shift(1).rolling(20, min_periods=5).std()
    df['proxy_volatility_ratio'] = short_std / long_std.clip(lower=0.1)
    df['proxy_high_volatility'] = (df['proxy_volatility_ratio'] > 1.5).astype(int)

    # Extreme value indicators
    df['proxy_is_extreme_high'] = (df['period_proxy'].shift(1) > df['proxy_rolling_mean10'] + 2 * df['proxy_rolling_std10']).astype(int)
    df['proxy_is_extreme_low'] = (df['period_proxy'].shift(1) < df['proxy_rolling_mean10'] - 2 * df['proxy_rolling_std10']).astype(int)

    # ================================================================
    # TIME-PATTERN FEATURES (same hour yesterday, last week)
    # ================================================================

    # Same hour yesterday (96 periods ago for 15-min data)
    df['proxy_same_hour_yesterday'] = df['period_proxy'].shift(96)

    # Same hour, same day last week (96*7 = 672 periods)
    df['proxy_same_hour_last_week'] = df['period_proxy'].shift(672)

    # Hour-of-day typical proxy (expanding mean from training period)
    df['hour_proxy_mean'] = df.groupby('hour')['period_proxy'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Day-of-week typical proxy
    df['dow_proxy_mean'] = df.groupby('day_of_week')['period_proxy'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Hour × weekend interaction proxy mean
    df['hour_weekend_key'] = df['hour'].astype(str) + '_' + df['is_weekend'].astype(str)
    df['hour_weekend_proxy_mean'] = df.groupby('hour_weekend_key')['period_proxy'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df = df.drop(columns=['hour_weekend_key'])

    # ================================================================
    # LOAD-REGULATION RELATIONSHIP
    # ================================================================

    # Load-regulation ratio (recent)
    if 'load_hist_mean_10' in df.columns:
        df['load_reg_ratio'] = df['load_hist_mean_10'] / df['reg_hist_mean_10'].clip(lower=1)
        df['load_reg_ratio'] = df['load_reg_ratio'].clip(-100, 100)

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
    """
    HYBRID FEATURE SELECTION:
    - Lead 12, 9: Rich features (historical, time patterns, regime)
    - Lead 6, 3, 0: Simple features (current-period focused)
    """

    # Core (all leads)
    core = ['baseline_pred', 'reg_cumulative_mean']
    within_period = ['reg_std', 'reg_range', 'reg_trend']

    # ================================================================
    # LEAD 12: MAXIMUM FEATURES
    # ================================================================
    if lead_time == 12:
        return core + [
            # Historical regulation (from 3-min data)
            'reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_min_10', 'reg_hist_max_10',
            'reg_hist_range_10', 'reg_hist_trend_10', 'reg_hist_mean_20', 'reg_hist_std_20',
            'reg_momentum', 'reg_acceleration', 'reg_hist_skew_10', 'reg_percentile_100',
            'reg_stability',
            # Historical load
            'load_hist_mean_10', 'load_hist_std_10', 'load_hist_trend_10',
            'load_momentum', 'load_acceleration',
            # Proxy lags and stats
            'proxy_lag1', 'proxy_lag2', 'proxy_lag3', 'proxy_lag4',
            'proxy_rolling_mean4', 'proxy_rolling_std4',
            'proxy_rolling_mean10', 'proxy_rolling_std10',
            'proxy_rolling_min10', 'proxy_rolling_max10',
            # Proxy sign features
            'proxy_last_sign', 'proxy_last_positive',
            'proxy_consecutive_same_sign', 'proxy_prop_positive_4', 'proxy_prop_positive_10',
            # Proxy momentum and regime
            'proxy_momentum', 'proxy_acceleration', 'proxy_deviation_from_mean',
            'proxy_volatility_ratio', 'proxy_high_volatility',
            'proxy_is_extreme_high', 'proxy_is_extreme_low',
            # Time patterns
            'proxy_same_hour_yesterday', 'proxy_same_hour_last_week',
            'hour_proxy_mean', 'dow_proxy_mean', 'hour_weekend_proxy_mean',
            # Load-reg relationship
            'load_reg_ratio',
            # Time features
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
            'is_night', 'is_morning', 'is_peak', 'is_afternoon', 'is_evening',
            # Other
            'load_deviation',
        ]

    # ================================================================
    # LEAD 9: RICH FEATURES (slightly less than 12)
    # ================================================================
    elif lead_time == 9:
        return core + within_period + [
            # Historical regulation
            'reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_min_10', 'reg_hist_max_10',
            'reg_hist_trend_10', 'reg_hist_mean_20', 'reg_hist_std_20',
            'reg_momentum', 'reg_stability',
            # Historical load
            'load_hist_mean_10', 'load_hist_trend_10', 'load_momentum',
            # Proxy
            'proxy_lag1', 'proxy_lag2', 'proxy_lag3',
            'proxy_rolling_mean4', 'proxy_rolling_std4', 'proxy_rolling_mean10',
            'proxy_last_sign', 'proxy_consecutive_same_sign', 'proxy_prop_positive_4',
            'proxy_momentum', 'proxy_deviation_from_mean', 'proxy_volatility_ratio',
            # Time patterns
            'proxy_same_hour_yesterday', 'hour_proxy_mean',
            # Time
            'hour_sin', 'hour_cos', 'is_weekend',
            'load_deviation',
        ]

    # ================================================================
    # LEAD 6: MODERATE FEATURES
    # ================================================================
    elif lead_time == 6:
        return core + within_period + [
            'reg_hist_mean_10', 'reg_hist_std_10',
            'proxy_lag1', 'proxy_rolling_mean4',
            'proxy_last_sign', 'proxy_momentum',
            'hour_sin', 'hour_cos',
        ]

    # ================================================================
    # LEAD 3: SIMPLE FEATURES
    # ================================================================
    elif lead_time == 3:
        return core + within_period + [
            'proxy_lag1', 'proxy_rolling_mean4',
            'hour_sin', 'hour_cos',
        ]

    # ================================================================
    # LEAD 0: MINIMAL FEATURES
    # ================================================================
    else:  # lead 0
        return core + within_period + ['proxy_lag1']


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
    print("LIGHTGBM v5 - HYBRID MODEL")
    print("Lead 12,9: Rich features | Lead 6,3,0: Simple features")
    print("=" * 70)

    reg_df, load_df, label_df = load_data()
    load_expected = compute_load_expected(load_df)
    df = create_base_features(reg_df, load_df, label_df, load_expected)

    print("\nAdding proxy-based features...")
    df = add_proxy_features(df)

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
        # Print in groups of 6 for readability
        for i in range(0, len(feature_cols), 6):
            print(f"  {feature_cols[i:i+6]}")

        print(f"\nTrain: {len(train_df):,} | Test: {len(test_df):,}")

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

        print(f"\nTop 15 Features:")
        for i, (_, row) in enumerate(imp_df.head(15).iterrows()):
            all_importance.append({
                'lead_time': lead, 'feature': row['feature'],
                'importance': row['importance'], 'pct': row['importance']/total*100
            })
            print(f"  {row['feature']:<35} {row['importance']/total*100:>5.1f}%")

        results.append({'lead_time': lead, **metrics})
        models[lead] = model

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'lightgbm_v5_results.csv', index=False)

    importance_df = pd.DataFrame(all_importance)
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance_v5.csv', index=False)

    with open(OUTPUT_DIR / 'lightgbm_models_v5.pkl', 'wb') as f:
        pickle.dump(models, f)

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: V1 vs V2 vs V4 vs V5")
    print("=" * 70)

    v1 = pd.read_csv(OUTPUT_DIR / 'lightgbm_results.csv')
    v2 = pd.read_csv(OUTPUT_DIR / 'lightgbm_v2_results.csv')
    v4 = pd.read_csv(OUTPUT_DIR / 'lightgbm_v4_results.csv')

    print(f"\n{'Lead':<6} {'V1':<8} {'V2':<8} {'V4':<8} {'V5':<8} {'Best':<8} {'vs V1':<10}")
    print("-" * 56)

    for _, v5_row in results_df.iterrows():
        lead = int(v5_row['lead_time'])
        v1_mae = v1[v1['lead_time'] == lead]['mae'].values[0]
        v2_mae = v2[v2['lead_time'] == lead]['mae'].values[0]
        v4_mae = v4[v4['lead_time'] == lead]['mae'].values[0]
        v5_mae = v5_row['mae']

        best_mae = min(v1_mae, v2_mae, v4_mae, v5_mae)
        best_ver = {v1_mae: 'V1', v2_mae: 'V2', v4_mae: 'V4', v5_mae: 'V5'}[best_mae]
        improvement = (v1_mae - best_mae) / v1_mae * 100

        print(f"{lead:<6} {v1_mae:<8.3f} {v2_mae:<8.3f} {v4_mae:<8.3f} {v5_mae:<8.3f} "
              f"{best_ver:<8} {improvement:>+8.1f}%")

    # Directional accuracy
    print(f"\n{'Lead':<6} {'V1 Dir':<10} {'V5 Dir':<10} {'Change':<10}")
    print("-" * 36)

    for _, v5_row in results_df.iterrows():
        lead = int(v5_row['lead_time'])
        v1_dir = v1[v1['lead_time'] == lead]['directional_accuracy'].values[0] * 100
        v5_dir = v5_row['directional_accuracy'] * 100
        change = v5_dir - v1_dir

        print(f"{lead:<6} {v1_dir:<10.1f} {v5_dir:<10.1f} {change:>+8.1f}pp")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
