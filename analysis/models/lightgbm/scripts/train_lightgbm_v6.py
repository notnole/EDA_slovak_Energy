"""
LightGBM v6 - Based on Error Analysis

New features for Lead 12, 9:
1. Lag 2 and Lag 4 explicit features (30-min and 1-hour cycles)
2. Transition hour indicators (hours 9-11, 15-17)
3. Sign-specific features (negative imbalances are harder)
4. Error-correction proxy (recent trend direction)
5. 2-hour (8 periods) and 4-hour (16 periods) rolling windows
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
    """Add historical features from raw 3-minute data."""
    reg_df = reg_df.sort_values('datetime').copy()
    load_df = load_df.sort_values('datetime').copy()

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

    # Momentum
    reg_df['reg_momentum'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(2)

    # Load features
    load_df['load_hist_mean_10'] = load_df['load_mw'].shift(1).rolling(10).mean()
    load_df['load_hist_trend_10'] = load_df['load_mw'].shift(1) - load_df['load_mw'].shift(10)
    load_df['load_momentum'] = load_df['load_mw'].shift(1) - load_df['load_mw'].shift(2)

    return reg_df, load_df


def create_base_features(reg_df, load_df, label_df, load_expected):
    """Create base feature dataframe."""
    print("\nCreating base features...")

    reg_df, load_df = add_3min_historical_features(reg_df, load_df)

    # Align regulation to settlement periods
    reg_df['datetime_floor'] = reg_df['datetime'].dt.floor('3min')
    reg_df['settlement_end'] = reg_df['datetime_floor'].dt.ceil('15min')
    mask = reg_df['datetime_floor'] == reg_df['settlement_end']
    reg_df.loc[mask, 'settlement_end'] = reg_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    reg_df['settlement_start'] = reg_df['settlement_end'] - pd.Timedelta(minutes=15)
    reg_df['minute_in_qh'] = (reg_df['datetime_floor'] - reg_df['settlement_start']).dt.total_seconds() / 60

    # Pivot regulation
    pivot_reg = reg_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='regulation_mw', aggfunc='first'
    ).reset_index()
    pivot_reg.columns = ['datetime'] + [f'reg_min{int(c)}' for c in pivot_reg.columns[1:]]

    # Get historical regulation features at minute 0
    hist_reg_cols = ['reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_min_10', 'reg_hist_max_10',
                     'reg_hist_range_10', 'reg_hist_trend_10', 'reg_hist_mean_20', 'reg_hist_std_20',
                     'reg_momentum']
    reg_min0 = reg_df[reg_df['minute_in_qh'] == 0][['settlement_start'] + hist_reg_cols].copy()
    reg_min0 = reg_min0.rename(columns={'settlement_start': 'datetime'})

    df = pd.merge(label_df, pivot_reg, on='datetime', how='inner')
    df = pd.merge(df, reg_min0, on='datetime', how='left')

    # Load alignment
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

    hist_load_cols = ['load_hist_mean_10', 'load_hist_trend_10', 'load_momentum']
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

    print(f"  Base features: {len(df):,} rows")
    return df


def add_enhanced_proxy_features(df):
    """
    Add enhanced proxy-based features based on error analysis.
    """
    df = df.sort_values('datetime').copy()

    # Compute period proxy
    reg_cols = ['reg_min0', 'reg_min3', 'reg_min6', 'reg_min9', 'reg_min12']
    available = [c for c in reg_cols if c in df.columns]
    df['period_proxy'] = -0.25 * df[available].mean(axis=1)

    # ================================================================
    # 1. BASIC LAG FEATURES
    # ================================================================
    df['proxy_lag1'] = df['period_proxy'].shift(1)
    df['proxy_lag2'] = df['period_proxy'].shift(2)
    df['proxy_lag3'] = df['period_proxy'].shift(3)
    df['proxy_lag4'] = df['period_proxy'].shift(4)

    # ================================================================
    # 2. LAG 2 AND LAG 4 SPECIFIC FEATURES (30-min and 1-hour cycles)
    # ================================================================
    # Difference between current and lag 2 (captures 30-min cycle)
    df['proxy_diff_lag2'] = df['proxy_lag1'] - df['proxy_lag2']

    # Difference between lag 2 and lag 4 (captures 1-hour pattern)
    df['proxy_diff_lag2_lag4'] = df['proxy_lag2'] - df['proxy_lag4']

    # Average of lag 2 and lag 4 (1-hour pattern)
    df['proxy_avg_lag2_lag4'] = (df['proxy_lag2'] + df['proxy_lag4']) / 2

    # Is current similar to 30-min ago?
    df['proxy_similar_to_lag2'] = (np.abs(df['proxy_lag1'] - df['proxy_lag2']) < 1).astype(int)

    # Is current similar to 1-hour ago?
    df['proxy_similar_to_lag4'] = (np.abs(df['proxy_lag1'] - df['proxy_lag4']) < 1).astype(int)

    # ================================================================
    # 3. ROLLING WINDOWS: 4, 8 (2-hour), 16 (4-hour)
    # ================================================================
    # 4 periods (1 hour)
    df['proxy_rolling_mean4'] = df['period_proxy'].shift(1).rolling(4).mean()
    df['proxy_rolling_std4'] = df['period_proxy'].shift(1).rolling(4).std()
    df['proxy_rolling_min4'] = df['period_proxy'].shift(1).rolling(4).min()
    df['proxy_rolling_max4'] = df['period_proxy'].shift(1).rolling(4).max()

    # 8 periods (2 hours)
    df['proxy_rolling_mean8'] = df['period_proxy'].shift(1).rolling(8).mean()
    df['proxy_rolling_std8'] = df['period_proxy'].shift(1).rolling(8).std()
    df['proxy_rolling_min8'] = df['period_proxy'].shift(1).rolling(8).min()
    df['proxy_rolling_max8'] = df['period_proxy'].shift(1).rolling(8).max()
    df['proxy_rolling_range8'] = df['proxy_rolling_max8'] - df['proxy_rolling_min8']

    # 16 periods (4 hours)
    df['proxy_rolling_mean16'] = df['period_proxy'].shift(1).rolling(16).mean()
    df['proxy_rolling_std16'] = df['period_proxy'].shift(1).rolling(16).std()
    df['proxy_rolling_min16'] = df['period_proxy'].shift(1).rolling(16).min()
    df['proxy_rolling_max16'] = df['period_proxy'].shift(1).rolling(16).max()
    df['proxy_rolling_range16'] = df['proxy_rolling_max16'] - df['proxy_rolling_min16']

    # Trend over 2 hours and 4 hours
    df['proxy_trend_8'] = df['proxy_lag1'] - df['period_proxy'].shift(8)
    df['proxy_trend_16'] = df['proxy_lag1'] - df['period_proxy'].shift(16)

    # ================================================================
    # 4. SIGN-SPECIFIC FEATURES
    # ================================================================
    df['proxy_last_sign'] = np.sign(df['period_proxy'].shift(1))
    df['proxy_last_positive'] = (df['period_proxy'].shift(1) > 0).astype(int)
    df['proxy_last_negative'] = (df['period_proxy'].shift(1) < 0).astype(int)

    # Consecutive same-sign counter
    signs = np.sign(df['period_proxy'])
    sign_change = (signs != signs.shift(1)).astype(int)
    sign_change.iloc[0] = 1
    groups = sign_change.cumsum()
    df['proxy_consecutive_same_sign'] = groups.groupby(groups).cumcount().shift(1)

    # Proportion positive in different windows
    positive = (df['period_proxy'] > 0).astype(float)
    df['proxy_prop_positive_4'] = positive.shift(1).rolling(4).mean()
    df['proxy_prop_positive_8'] = positive.shift(1).rolling(8).mean()
    df['proxy_prop_positive_16'] = positive.shift(1).rolling(16).mean()

    # Sign-weighted features (amplify negative patterns since they have higher error)
    df['proxy_lag1_if_negative'] = df['proxy_lag1'] * df['proxy_last_negative']
    df['proxy_lag1_if_positive'] = df['proxy_lag1'] * df['proxy_last_positive']

    # Recent negative count
    df['proxy_negative_count_4'] = (df['period_proxy'] < 0).astype(int).shift(1).rolling(4).sum()
    df['proxy_negative_count_8'] = (df['period_proxy'] < 0).astype(int).shift(1).rolling(8).sum()

    # ================================================================
    # 5. ERROR-CORRECTION / TREND DIRECTION FEATURES
    # ================================================================
    # Momentum
    df['proxy_momentum'] = df['proxy_lag1'] - df['proxy_lag2']
    df['proxy_acceleration'] = df['proxy_momentum'] - df['proxy_momentum'].shift(1)

    # Trend direction (is it increasing or decreasing?)
    df['proxy_trend_direction'] = np.sign(df['proxy_momentum'])

    # Consecutive trend direction
    trend_signs = np.sign(df['proxy_momentum'])
    trend_change = (trend_signs != trend_signs.shift(1)).astype(int)
    trend_change.iloc[0] = 1
    trend_groups = trend_change.cumsum()
    df['proxy_trend_persistence'] = trend_groups.groupby(trend_groups).cumcount().shift(1)

    # Deviation from rolling means
    df['proxy_deviation_from_mean4'] = df['proxy_lag1'] - df['proxy_rolling_mean4']
    df['proxy_deviation_from_mean8'] = df['proxy_lag1'] - df['proxy_rolling_mean8']
    df['proxy_deviation_from_mean16'] = df['proxy_lag1'] - df['proxy_rolling_mean16']

    # Mean reversion indicator (far from mean = likely to revert)
    df['proxy_zscore_4'] = df['proxy_deviation_from_mean4'] / df['proxy_rolling_std4'].clip(lower=0.1)
    df['proxy_zscore_8'] = df['proxy_deviation_from_mean8'] / df['proxy_rolling_std8'].clip(lower=0.1)

    # ================================================================
    # 6. TRANSITION HOUR INDICATORS
    # ================================================================
    df['is_morning_transition'] = ((df['hour'] >= 9) & (df['hour'] <= 11)).astype(int)
    df['is_afternoon_transition'] = ((df['hour'] >= 15) & (df['hour'] <= 17)).astype(int)
    df['is_transition_hour'] = (df['is_morning_transition'] | df['is_afternoon_transition']).astype(int)

    # Hour group dummies
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 11)).astype(int)
    df['is_midday'] = ((df['hour'] >= 11) & (df['hour'] < 15)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 15) & (df['hour'] < 21)).astype(int)
    df['is_evening'] = ((df['hour'] >= 21) & (df['hour'] < 24)).astype(int)

    # ================================================================
    # 7. TIME PATTERN FEATURES
    # ================================================================
    # Same hour yesterday (96 periods)
    df['proxy_same_hour_yesterday'] = df['period_proxy'].shift(96)

    # Hour-of-day mean (expanding)
    df['hour_proxy_mean'] = df.groupby('hour')['period_proxy'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Deviation from hour mean
    df['proxy_deviation_from_hour_mean'] = df['proxy_lag1'] - df['hour_proxy_mean']

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
    """Get feature set for each lead time."""

    core = ['baseline_pred', 'reg_cumulative_mean']
    within_period = ['reg_std', 'reg_range', 'reg_trend']

    if lead_time == 12:
        return core + [
            # Historical regulation
            'reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_min_10', 'reg_hist_max_10',
            'reg_hist_range_10', 'reg_hist_trend_10', 'reg_hist_mean_20', 'reg_hist_std_20',
            'reg_momentum',
            # Historical load
            'load_hist_mean_10', 'load_hist_trend_10', 'load_momentum',
            # Basic proxy lags
            'proxy_lag1', 'proxy_lag2', 'proxy_lag3', 'proxy_lag4',
            # Lag 2 and Lag 4 specific (30-min and 1-hour cycles)
            'proxy_diff_lag2', 'proxy_diff_lag2_lag4', 'proxy_avg_lag2_lag4',
            'proxy_similar_to_lag2', 'proxy_similar_to_lag4',
            # Rolling 4 (1 hour)
            'proxy_rolling_mean4', 'proxy_rolling_std4', 'proxy_rolling_min4', 'proxy_rolling_max4',
            # Rolling 8 (2 hours)
            'proxy_rolling_mean8', 'proxy_rolling_std8', 'proxy_rolling_min8', 'proxy_rolling_max8',
            'proxy_rolling_range8',
            # Rolling 16 (4 hours)
            'proxy_rolling_mean16', 'proxy_rolling_std16', 'proxy_rolling_min16', 'proxy_rolling_max16',
            'proxy_rolling_range16',
            # Trends
            'proxy_trend_8', 'proxy_trend_16',
            # Sign features
            'proxy_last_sign', 'proxy_last_positive', 'proxy_last_negative',
            'proxy_consecutive_same_sign',
            'proxy_prop_positive_4', 'proxy_prop_positive_8', 'proxy_prop_positive_16',
            'proxy_lag1_if_negative', 'proxy_lag1_if_positive',
            'proxy_negative_count_4', 'proxy_negative_count_8',
            # Momentum and trend
            'proxy_momentum', 'proxy_acceleration',
            'proxy_trend_direction', 'proxy_trend_persistence',
            'proxy_deviation_from_mean4', 'proxy_deviation_from_mean8', 'proxy_deviation_from_mean16',
            'proxy_zscore_4', 'proxy_zscore_8',
            # Transition hours
            'is_morning_transition', 'is_afternoon_transition', 'is_transition_hour',
            'is_night', 'is_morning', 'is_midday', 'is_afternoon', 'is_evening',
            # Time patterns
            'proxy_same_hour_yesterday', 'hour_proxy_mean', 'proxy_deviation_from_hour_mean',
            # Time encoding
            'hour_sin', 'hour_cos', 'is_weekend',
            # Other
            'load_deviation',
        ]

    elif lead_time == 9:
        return core + within_period + [
            # Historical
            'reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_mean_20',
            'reg_hist_trend_10', 'reg_momentum',
            'load_hist_mean_10', 'load_hist_trend_10',
            # Proxy lags
            'proxy_lag1', 'proxy_lag2', 'proxy_lag3', 'proxy_lag4',
            'proxy_diff_lag2', 'proxy_diff_lag2_lag4', 'proxy_avg_lag2_lag4',
            # Rolling windows
            'proxy_rolling_mean4', 'proxy_rolling_std4',
            'proxy_rolling_mean8', 'proxy_rolling_std8', 'proxy_rolling_range8',
            'proxy_rolling_mean16', 'proxy_rolling_std16',
            'proxy_trend_8', 'proxy_trend_16',
            # Sign
            'proxy_last_sign', 'proxy_consecutive_same_sign',
            'proxy_prop_positive_4', 'proxy_prop_positive_8',
            'proxy_negative_count_4',
            # Momentum
            'proxy_momentum', 'proxy_trend_direction',
            'proxy_deviation_from_mean4', 'proxy_deviation_from_mean8',
            # Transition
            'is_morning_transition', 'is_afternoon_transition',
            # Time
            'proxy_same_hour_yesterday', 'hour_proxy_mean',
            'hour_sin', 'hour_cos', 'is_weekend',
            'load_deviation',
        ]

    elif lead_time == 6:
        return core + within_period + [
            'reg_hist_mean_10', 'reg_hist_std_10',
            'proxy_lag1', 'proxy_lag2',
            'proxy_rolling_mean4', 'proxy_rolling_std4',
            'proxy_rolling_mean8',
            'proxy_last_sign', 'proxy_momentum',
            'hour_sin', 'hour_cos',
        ]

    elif lead_time == 3:
        return core + within_period + [
            'proxy_lag1', 'proxy_rolling_mean4',
            'hour_sin', 'hour_cos',
        ]

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
    print("LIGHTGBM v6 - ERROR ANALYSIS BASED FEATURES")
    print("=" * 70)

    reg_df, load_df, label_df = load_data()
    load_expected = compute_load_expected(load_df)
    df = create_base_features(reg_df, load_df, label_df, load_expected)

    print("\nAdding enhanced proxy features...")
    df = add_enhanced_proxy_features(df)

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

        print(f"\nFeatures ({len(feature_cols)})")
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

        print(f"\nTop 20 Features:")
        for i, (_, row) in enumerate(imp_df.head(20).iterrows()):
            all_importance.append({
                'lead_time': lead, 'feature': row['feature'],
                'importance': row['importance'], 'pct': row['importance']/total*100
            })
            print(f"  {row['feature']:<40} {row['importance']/total*100:>5.1f}%")

        results.append({'lead_time': lead, **metrics})
        models[lead] = model

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'lightgbm_v6_results.csv', index=False)

    importance_df = pd.DataFrame(all_importance)
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance_v6.csv', index=False)

    with open(OUTPUT_DIR / 'lightgbm_models_v6.pkl', 'wb') as f:
        pickle.dump(models, f)

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON: ALL VERSIONS")
    print("=" * 70)

    v1 = pd.read_csv(OUTPUT_DIR / 'lightgbm_results.csv')
    v2 = pd.read_csv(OUTPUT_DIR / 'lightgbm_v2_results.csv')
    v4 = pd.read_csv(OUTPUT_DIR / 'lightgbm_v4_results.csv')

    print(f"\n{'Lead':<6} {'V1':<8} {'V2':<8} {'V4':<8} {'V6':<8} {'Best':<6} {'vs V1':<10} {'vs V4':<10}")
    print("-" * 74)

    for _, v6_row in results_df.iterrows():
        lead = int(v6_row['lead_time'])
        v1_mae = v1[v1['lead_time'] == lead]['mae'].values[0]
        v2_mae = v2[v2['lead_time'] == lead]['mae'].values[0]
        v4_mae = v4[v4['lead_time'] == lead]['mae'].values[0]
        v6_mae = v6_row['mae']

        best_mae = min(v1_mae, v2_mae, v4_mae, v6_mae)
        best_ver = {v1_mae: 'V1', v2_mae: 'V2', v4_mae: 'V4', v6_mae: 'V6'}[best_mae]
        vs_v1 = (v1_mae - v6_mae) / v1_mae * 100
        vs_v4 = (v4_mae - v6_mae) / v4_mae * 100

        print(f"{lead:<6} {v1_mae:<8.3f} {v2_mae:<8.3f} {v4_mae:<8.3f} {v6_mae:<8.3f} "
              f"{best_ver:<6} {vs_v1:>+8.1f}% {vs_v4:>+8.1f}%")

    # Directional accuracy comparison
    print(f"\n{'Lead':<6} {'V1 Dir':<10} {'V4 Dir':<10} {'V6 Dir':<10}")
    print("-" * 36)

    for _, v6_row in results_df.iterrows():
        lead = int(v6_row['lead_time'])
        v1_dir = v1[v1['lead_time'] == lead]['directional_accuracy'].values[0] * 100
        v4_dir = v4[v4['lead_time'] == lead]['directional_accuracy'].values[0] * 100
        v6_dir = v6_row['directional_accuracy'] * 100

        print(f"{lead:<6} {v1_dir:<10.1f} {v4_dir:<10.1f} {v6_dir:<10.1f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
