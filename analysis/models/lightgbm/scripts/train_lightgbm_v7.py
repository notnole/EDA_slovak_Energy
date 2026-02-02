"""
LightGBM v7 - Smart Error-Correction Features

Instead of 74 features, add 2-3 smart features that directly target known error patterns:
1. Proxy bias correction (by hour + sign)
2. Cyclical residual (lag 2 and lag 4 patterns)
3. Error momentum proxy
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
    """Compute expected load by time-of-day."""
    load_df = load_df.copy()
    load_df['hour'] = load_df['datetime'].dt.hour
    load_df['minute'] = load_df['datetime'].dt.minute
    load_df['is_weekend'] = load_df['datetime'].dt.dayofweek >= 5
    train_mask = load_df['datetime'].dt.year == 2024
    expected = load_df[train_mask].groupby(['hour', 'minute', 'is_weekend'])['load_mw'].mean()
    expected.name = 'expected_load'
    return expected


def add_3min_features(reg_df):
    """Add historical features from 3-min regulation."""
    reg_df = reg_df.sort_values('datetime').copy()
    reg_df['reg_hist_mean_10'] = reg_df['regulation_mw'].shift(1).rolling(10).mean()
    reg_df['reg_hist_std_10'] = reg_df['regulation_mw'].shift(1).rolling(10).std()
    reg_df['reg_hist_mean_20'] = reg_df['regulation_mw'].shift(1).rolling(20).mean()
    reg_df['reg_momentum'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(2)
    return reg_df


def create_base_features(reg_df, load_df, label_df, load_expected):
    """Create base feature dataframe."""
    print("\nCreating base features...")

    reg_df = add_3min_features(reg_df)

    # Align to settlement periods
    reg_df['datetime_floor'] = reg_df['datetime'].dt.floor('3min')
    reg_df['settlement_end'] = reg_df['datetime_floor'].dt.ceil('15min')
    mask = reg_df['datetime_floor'] == reg_df['settlement_end']
    reg_df.loc[mask, 'settlement_end'] = reg_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    reg_df['settlement_start'] = reg_df['settlement_end'] - pd.Timedelta(minutes=15)
    reg_df['minute_in_qh'] = (reg_df['datetime_floor'] - reg_df['settlement_start']).dt.total_seconds() / 60

    # Pivot
    pivot_reg = reg_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='regulation_mw', aggfunc='first'
    ).reset_index()
    pivot_reg.columns = ['datetime'] + [f'reg_min{int(c)}' for c in pivot_reg.columns[1:]]

    hist_cols = ['reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_mean_20', 'reg_momentum']
    reg_min0 = reg_df[reg_df['minute_in_qh'] == 0][['settlement_start'] + hist_cols].copy()
    reg_min0 = reg_min0.rename(columns={'settlement_start': 'datetime'})

    df = pd.merge(label_df, pivot_reg, on='datetime', how='inner')
    df = pd.merge(df, reg_min0, on='datetime', how='left')

    # Load
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


def compute_proxy_bias_lookup(df, train_end):
    """
    Compute expected proxy bias by (hour, sign) from training data.

    The proxy systematically over/under-predicts depending on hour and sign.
    We learn this pattern from training data and use it as a correction.
    """
    train_df = df[df['datetime'] < train_end].copy()

    # Compute proxy for each period
    reg_cols = ['reg_min0', 'reg_min3', 'reg_min6', 'reg_min9', 'reg_min12']
    available = [c for c in reg_cols if c in train_df.columns]
    train_df['proxy'] = -0.25 * train_df[available].mean(axis=1)

    # Proxy error = proxy - actual
    train_df['proxy_error'] = train_df['proxy'] - train_df['imbalance']

    # Sign of proxy
    train_df['proxy_sign'] = np.sign(train_df['proxy'])

    # Compute average error by (hour, sign)
    bias_by_hour_sign = train_df.groupby(['hour', 'proxy_sign'])['proxy_error'].agg(['mean', 'std']).reset_index()
    bias_by_hour_sign.columns = ['hour', 'proxy_sign', 'expected_bias', 'bias_std']

    # Also compute by hour only (for cases where sign lookup fails)
    bias_by_hour = train_df.groupby('hour')['proxy_error'].mean().reset_index()
    bias_by_hour.columns = ['hour', 'expected_bias_hour']

    return bias_by_hour_sign, bias_by_hour


def add_smart_error_features(df, bias_by_hour_sign, bias_by_hour, train_end):
    """
    Add smart features that directly target known error patterns.
    """
    df = df.sort_values('datetime').copy()

    # Compute period proxy
    reg_cols = ['reg_min0', 'reg_min3', 'reg_min6', 'reg_min9', 'reg_min12']
    available = [c for c in reg_cols if c in df.columns]
    df['period_proxy'] = -0.25 * df[available].mean(axis=1)

    # Basic lags
    df['proxy_lag1'] = df['period_proxy'].shift(1)
    df['proxy_lag2'] = df['period_proxy'].shift(2)
    df['proxy_lag3'] = df['period_proxy'].shift(3)
    df['proxy_lag4'] = df['period_proxy'].shift(4)

    # Rolling mean
    df['proxy_rolling_mean4'] = df['period_proxy'].shift(1).rolling(4).mean()
    df['proxy_rolling_std4'] = df['period_proxy'].shift(1).rolling(4).std()

    # Sign features
    df['proxy_last_sign'] = np.sign(df['period_proxy'].shift(1))

    # ================================================================
    # SMART FEATURE 1: Expected Proxy Bias Correction
    # ================================================================
    # This tells the model: "given the hour and sign, the proxy typically
    # over/under-predicts by this much"

    df['proxy_sign'] = np.sign(df['period_proxy'].shift(1))

    # Merge bias lookup
    df = df.merge(bias_by_hour_sign, on=['hour', 'proxy_sign'], how='left')
    df = df.merge(bias_by_hour, on='hour', how='left')

    # Use hour+sign bias if available, otherwise hour-only
    df['expected_proxy_bias'] = df['expected_bias'].fillna(df['expected_bias_hour'])

    # The corrected proxy (what the proxy SHOULD predict after bias adjustment)
    df['proxy_corrected'] = df['proxy_lag1'] - df['expected_proxy_bias']

    # Clean up
    df = df.drop(columns=['proxy_sign', 'expected_bias', 'bias_std', 'expected_bias_hour'], errors='ignore')

    # ================================================================
    # SMART FEATURE 2: Cyclical Residual (30-min and 1-hour patterns)
    # ================================================================
    # Errors at lag 2 (r=0.155) and lag 4 (r=0.220) are correlated.
    # This captures the cyclical pattern in the data.

    # 30-minute cycle: how much did proxy change from 30 min ago?
    df['cycle_30min'] = df['proxy_lag1'] - df['proxy_lag2']

    # 1-hour cycle: how much did proxy change from 1 hour ago?
    df['cycle_60min'] = df['proxy_lag1'] - df['proxy_lag4']

    # Combined cycle indicator: average of lag2 and lag4 deviations
    # This captures: "if proxy was similar 30min and 60min ago, expect similar now"
    df['cycle_pattern'] = (df['proxy_lag2'] + df['proxy_lag4']) / 2 - df['proxy_lag1']

    # Persistence of cycle: is the 30-min change consistent with 60-min change?
    df['cycle_consistency'] = np.sign(df['cycle_30min']) == np.sign(df['cycle_60min'])
    df['cycle_consistency'] = df['cycle_consistency'].astype(int)

    # ================================================================
    # SMART FEATURE 3: Error Momentum Proxy
    # ================================================================
    # The error tends to persist. If proxy was "wrong" in a certain direction,
    # it's likely to be wrong again. We approximate this with proxy momentum.

    # Momentum: direction and magnitude of recent proxy change
    df['proxy_momentum'] = df['proxy_lag1'] - df['proxy_lag2']

    # Acceleration: is the momentum changing?
    df['proxy_acceleration'] = df['proxy_momentum'] - (df['proxy_lag2'] - df['proxy_lag3'])

    # Second-order difference (captures the curvature of the proxy)
    df['proxy_curvature'] = df['proxy_lag1'] - 2*df['proxy_lag2'] + df['proxy_lag3']

    # ================================================================
    # SMART FEATURE 4: Regime Feature
    # ================================================================
    # If proxy has been consistently positive or negative, there may be a regime

    # Proportion positive in last 4 periods
    positive = (df['period_proxy'] > 0).astype(float)
    df['proxy_prop_positive_4'] = positive.shift(1).rolling(4).mean()

    # Strong positive/negative regime
    df['positive_regime'] = (df['proxy_prop_positive_4'] > 0.75).astype(int)
    df['negative_regime'] = (df['proxy_prop_positive_4'] < 0.25).astype(int)

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
    """Get feature set - V4 base + smart error features."""

    core = ['baseline_pred', 'reg_cumulative_mean']
    within_period = ['reg_std', 'reg_range', 'reg_trend']

    # Smart error-correction features
    smart_features = [
        'expected_proxy_bias',    # What bias to expect based on hour+sign
        'proxy_corrected',        # Bias-corrected proxy
        'cycle_30min',            # 30-min cycle pattern
        'cycle_60min',            # 1-hour cycle pattern
        'cycle_pattern',          # Combined cycle indicator
        'proxy_momentum',         # Error momentum proxy
        'proxy_curvature',        # Second-order pattern
    ]

    if lead_time == 12:
        return core + [
            # From V4 (proven useful)
            'reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_mean_20', 'reg_momentum',
            'proxy_lag1', 'proxy_lag2', 'proxy_lag3', 'proxy_lag4',
            'proxy_rolling_mean4', 'proxy_rolling_std4',
            'proxy_last_sign',
            'hour_sin', 'hour_cos', 'is_weekend',
            'load_deviation',
            # Smart error features
            'expected_proxy_bias',
            'proxy_corrected',
            'cycle_30min',
            'cycle_60min',
            'cycle_pattern',
            'proxy_momentum',
            'proxy_curvature',
            'positive_regime',
            'negative_regime',
        ]

    elif lead_time == 9:
        return core + within_period + [
            'reg_hist_mean_10', 'reg_hist_mean_20',
            'proxy_lag1', 'proxy_lag2', 'proxy_lag3',
            'proxy_rolling_mean4', 'proxy_rolling_std4',
            'proxy_last_sign',
            'hour_sin', 'hour_cos', 'is_weekend',
            'load_deviation',
            # Smart features
            'expected_proxy_bias',
            'proxy_corrected',
            'cycle_30min',
            'cycle_60min',
            'proxy_momentum',
        ]

    elif lead_time == 6:
        return core + within_period + [
            'reg_hist_mean_10',
            'proxy_lag1', 'proxy_lag2',
            'proxy_rolling_mean4',
            'hour_sin', 'hour_cos',
            'expected_proxy_bias',
            'cycle_30min',
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
    print("LIGHTGBM v7 - SMART ERROR-CORRECTION FEATURES")
    print("=" * 70)

    reg_df, load_df, label_df = load_data()
    load_expected = compute_load_expected(load_df)
    df = create_base_features(reg_df, load_df, label_df, load_expected)

    test_start = pd.Timestamp('2025-10-01')

    # Compute proxy bias lookup from training data
    print("\nComputing proxy bias lookup from training data...")
    bias_by_hour_sign, bias_by_hour = compute_proxy_bias_lookup(df, test_start)

    print("\nProxy bias by hour and sign (sample):")
    print(bias_by_hour_sign.head(10))

    # Add smart features
    print("\nAdding smart error-correction features...")
    df = add_smart_error_features(df, bias_by_hour_sign, bias_by_hour, test_start)

    lead_times = [12, 9, 6, 3, 0]
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

        print(f"\nFeature Importance:")
        for _, row in imp_df.iterrows():
            pct = row['importance']/total*100
            if pct > 0.5:  # Only show features with >0.5% contribution
                print(f"  {row['feature']:<30} {pct:>5.1f}%")

        results.append({'lead_time': lead, **metrics})
        models[lead] = model

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'lightgbm_v7_results.csv', index=False)

    with open(OUTPUT_DIR / 'lightgbm_models_v7.pkl', 'wb') as f:
        pickle.dump(models, f)

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON: V4 vs V7")
    print("=" * 70)

    v1 = pd.read_csv(OUTPUT_DIR / 'lightgbm_results.csv')
    v4 = pd.read_csv(OUTPUT_DIR / 'lightgbm_v4_results.csv')

    print(f"\n{'Lead':<6} {'V1':<8} {'V4':<8} {'V7':<8} {'V7 vs V4':<10} {'V1 Dir':<8} {'V4 Dir':<8} {'V7 Dir':<8}")
    print("-" * 78)

    for _, v7_row in results_df.iterrows():
        lead = int(v7_row['lead_time'])
        v1_mae = v1[v1['lead_time'] == lead]['mae'].values[0]
        v4_mae = v4[v4['lead_time'] == lead]['mae'].values[0]
        v7_mae = v7_row['mae']

        v1_dir = v1[v1['lead_time'] == lead]['directional_accuracy'].values[0] * 100
        v4_dir = v4[v4['lead_time'] == lead]['directional_accuracy'].values[0] * 100
        v7_dir = v7_row['directional_accuracy'] * 100

        vs_v4 = (v4_mae - v7_mae) / v4_mae * 100

        print(f"{lead:<6} {v1_mae:<8.3f} {v4_mae:<8.3f} {v7_mae:<8.3f} {vs_v4:>+8.1f}% "
              f"{v1_dir:<8.1f} {v4_dir:<8.1f} {v7_dir:<8.1f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
