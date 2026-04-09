"""
LightGBM v8 - QUANTILE REGRESSION NOWCASTING

Based on V4 (best point-prediction model). Same real-time-only features,
but trains 5 quantile models (0.1, 0.25, 0.5, 0.75, 0.9) per lead time
for probabilistic forecasting and confidence-aware trading.

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

# --- Configuration ---
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent.parent.parent  # ipesoft_eda_data/
FEATURES_DIR = PROJECT_DIR / 'data' / 'features'
MASTER_DIR = PROJECT_DIR / 'data' / 'master'
OUTPUT_DIR = SCRIPT_DIR.parent / 'outputs'


# ===========================================================================
# Feature engineering (identical to V4)
# ===========================================================================

def load_data():
    """Load data."""
    print("[*] Loading data...")
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
    shifted = reg_df['regulation_mw'].shift(1)

    # Statistics over last 10 observations (30 minutes of history)
    reg_df['reg_hist_mean_10'] = shifted.rolling(10).mean()
    reg_df['reg_hist_std_10'] = shifted.rolling(10).std()
    reg_df['reg_hist_min_10'] = shifted.rolling(10).min()
    reg_df['reg_hist_max_10'] = shifted.rolling(10).max()
    reg_df['reg_hist_range_10'] = reg_df['reg_hist_max_10'] - reg_df['reg_hist_min_10']

    # Trend: last observation minus 10 observations ago
    reg_df['reg_hist_trend_10'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(10)

    # Statistics over last 20 observations (1 hour of history)
    reg_df['reg_hist_mean_20'] = shifted.rolling(20).mean()
    reg_df['reg_hist_std_20'] = shifted.rolling(20).std()

    # Recent momentum (change in regulation)
    reg_df['reg_momentum'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(2)
    reg_df['reg_acceleration'] = reg_df['reg_momentum'] - reg_df['reg_momentum'].shift(1)

    # --- V8 NEW: Extended volatility features ---

    # Longer volatility windows (2hr, 4hr)
    reg_df['reg_hist_std_40'] = shifted.rolling(40).std()
    reg_df['reg_hist_std_80'] = shifted.rolling(80).std()
    reg_df['reg_hist_range_40'] = shifted.rolling(40).max() - shifted.rolling(40).min()

    # Regulation magnitude (system stress indicator)
    abs_shifted = reg_df['regulation_mw'].abs().shift(1)
    reg_df['reg_abs_mean_10'] = abs_shifted.rolling(10).mean()
    reg_df['reg_abs_mean_20'] = abs_shifted.rolling(20).mean()

    # Volatility ratio: short-term vs long-term regulation volatility
    reg_df['reg_vol_ratio'] = reg_df['reg_hist_std_10'] / reg_df['reg_hist_std_40'].clip(lower=1.0)

    return reg_df


def add_historical_load_features(load_df):
    """
    Add historical load volatility features from raw 3-min load data.
    V8 NEW: Load volatility was entirely missing from V4.
    """
    load_df = load_df.sort_values('datetime').copy()
    shifted = load_df['load_mw'].shift(1)

    # Load volatility over 30min and 1hr
    load_df['load_hist_std_10'] = shifted.rolling(10).std()
    load_df['load_hist_std_20'] = shifted.rolling(20).std()
    load_df['load_hist_range_10'] = shifted.rolling(10).max() - shifted.rolling(10).min()

    # Load ramp rate (absolute mean of 3-min changes)
    load_ramp = load_df['load_mw'].diff()
    load_df['load_ramp_abs_mean_10'] = load_ramp.abs().shift(1).rolling(10).mean()

    return load_df


def create_base_features(reg_df, load_df, label_df, load_expected):
    """Create base feature dataframe."""
    print("[*] Creating base features...")

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
                 'reg_momentum', 'reg_acceleration',
                 # V8 NEW: extended volatility
                 'reg_hist_std_40', 'reg_hist_std_80', 'reg_hist_range_40',
                 'reg_abs_mean_10', 'reg_abs_mean_20', 'reg_vol_ratio']

    # Get historical features at minute 0 of each period (the first observation)
    reg_min0 = reg_df[reg_df['minute_in_qh'] == 0][['settlement_start'] + hist_cols].copy()
    reg_min0 = reg_min0.rename(columns={'settlement_start': 'datetime'})

    # Merge with labels
    df = pd.merge(label_df, pivot_reg, on='datetime', how='inner')
    df = pd.merge(df, reg_min0, on='datetime', how='left')

    # Add historical load features to raw 3-min load data
    load_df = add_historical_load_features(load_df)

    # Align load to settlement periods
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

    # V8 NEW: Pivot load volatility features (from minute 0 of each period)
    load_hist_cols = ['load_hist_std_10', 'load_hist_std_20', 'load_hist_range_10',
                      'load_ramp_abs_mean_10']
    load_min0 = load_df[load_df['minute_in_qh'] == 0][['settlement_start'] + load_hist_cols].copy()
    load_min0 = load_min0.rename(columns={'settlement_start': 'datetime'})
    df = pd.merge(df, load_min0, on='datetime', how='left')

    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    print(f"[+]   Base features: {len(df):,} rows")
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

    # V8 NEW: Vol-of-vol (is the system transitioning between regimes?)
    df['proxy_vol_of_vol'] = df['proxy_rolling_std4'].rolling(4).std()

    # V8 NEW: Proxy absolute magnitude (large |proxy| = large expected imbalance = more uncertainty)
    df['proxy_abs_mean_4'] = df['period_proxy'].abs().shift(1).rolling(4).mean()
    df['proxy_abs_mean_10'] = df['period_proxy'].abs().shift(1).rolling(10).mean()

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

    # V8 NEW: Enhanced volatility features for interval sharpness
    reg_vol_extended = ['reg_hist_std_40', 'reg_hist_std_80', 'reg_hist_range_40',
                        'reg_abs_mean_10', 'reg_abs_mean_20', 'reg_vol_ratio']
    load_volatility = ['load_hist_std_10', 'load_hist_std_20', 'load_hist_range_10',
                       'load_ramp_abs_mean_10']
    proxy_vol_extended = ['proxy_vol_of_vol', 'proxy_abs_mean_4', 'proxy_abs_mean_10']

    # Time features
    time_basic = ['hour_sin', 'hour_cos']
    time_extended = ['is_weekend', 'dow_sin', 'dow_cos']

    # Other
    other = ['load_deviation']

    if lead_time == 12:
        # Maximum features - we have minimal current-period info
        # V8: add all new volatility features here
        features = (core + hist_reg_basic + hist_reg_extended + hist_reg_momentum +
                   reg_vol_extended + load_volatility +
                   proxy_basic + proxy_extended + proxy_sign + proxy_momentum +
                   proxy_volatility + proxy_vol_extended +
                   time_basic + time_extended + other)
    elif lead_time == 9:
        # Add within-period stats, still need historical
        # V8: add volatility features here too
        features = (core + within_period + hist_reg_basic + hist_reg_extended +
                   reg_vol_extended[:4] + load_volatility[:2] +
                   proxy_basic + proxy_extended + proxy_sign[:3] + proxy_momentum[:2] +
                   proxy_vol_extended[:2] +
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


# ===========================================================================
# Training and evaluation (modified for quantile regression)
# ===========================================================================

def train_model(train_df, feature_cols, alpha=0.5):
    """Train LightGBM quantile regression model."""
    X = train_df[feature_cols].values
    y = train_df['imbalance'].values

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isinf(X).any(axis=1)
    X, y = X[valid], y[valid]

    # Extra regularization for tail quantiles
    is_tail = alpha <= 0.1 or alpha >= 0.9
    min_data = 80 if is_tail else 50
    reg_lambda = 0.3 if is_tail else 0.1

    params = {
        'objective': 'quantile',
        'alpha': alpha,
        'metric': 'quantile',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': min_data,
        'reg_alpha': 0.1,
        'reg_lambda': reg_lambda,
        'verbose': -1,
        'seed': 42,
    }

    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=500)
    return model


def get_valid_test_data(test_df, feature_cols):
    """Extract valid (non-NaN, non-Inf) test data."""
    X = test_df[feature_cols].values
    y_true = test_df['imbalance'].values
    dt = test_df['datetime'].values

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y_true) & ~np.isinf(X).any(axis=1)
    return X[valid], y_true[valid], dt[valid]


def pinball_loss(y_true, y_pred, alpha):
    """Compute pinball (quantile) loss."""
    diff = y_true - y_pred
    return np.mean(np.where(diff >= 0, alpha * diff, (alpha - 1) * diff))


def evaluate_single(y_true, y_pred, alpha):
    """Evaluate a single quantile model."""
    return {
        'quantile': alpha,
        'pinball_loss': pinball_loss(y_true, y_pred, alpha),
        'coverage': float(np.mean(y_true <= y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'n_samples': len(y_true),
    }


def evaluate_quantile_set(predictions, y_true):
    """
    Evaluate a full set of quantile predictions for one lead time.

    Parameters
    ----------
    predictions : dict {alpha: y_pred array}
    y_true : array
    """
    alphas = sorted(predictions.keys())

    # --- Quantile crossing rate (before sorting) ---
    stacked = np.column_stack([predictions[a] for a in alphas])
    diffs = np.diff(stacked, axis=1)
    crossing_rate = float(np.mean(np.any(diffs < 0, axis=1)))

    # --- Sort to fix crossings ---
    sorted_stacked = np.sort(stacked, axis=1)
    sorted_preds = {a: sorted_stacked[:, i] for i, a in enumerate(alphas)}

    # --- CRPS approximation (discrete quantile decomposition) ---
    # CRPS ~= (2/K) * sum_k pinball(alpha_k, y, q_k)
    crps = (2.0 / len(alphas)) * sum(
        pinball_loss(y_true, sorted_preds[a], a) for a in alphas
    )

    # --- Interval metrics ---
    results = {
        'crps': crps,
        'crossing_rate': crossing_rate,
    }

    # 50% interval: Q25-Q75
    if 0.25 in sorted_preds and 0.75 in sorted_preds:
        q25 = sorted_preds[0.25]
        q75 = sorted_preds[0.75]
        results['sharpness_50'] = float(np.mean(q75 - q25))
        results['coverage_50'] = float(np.mean((y_true >= q25) & (y_true <= q75)))

    # 80% interval: Q10-Q90
    if 0.1 in sorted_preds and 0.9 in sorted_preds:
        q10 = sorted_preds[0.1]
        q90 = sorted_preds[0.9]
        results['sharpness_80'] = float(np.mean(q90 - q10))
        results['coverage_80'] = float(np.mean((y_true >= q10) & (y_true <= q90)))

    return results, sorted_preds


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("[*] LIGHTGBM v8 - QUANTILE REGRESSION NOWCASTING")
    print("[*] Based on V4 features, quantiles: %s" % QUANTILES)
    print("=" * 70)

    reg_df, load_df, label_df = load_data()
    load_expected = compute_load_expected(load_df)
    df = create_base_features(reg_df, load_df, label_df, load_expected)

    print("[*] Adding proxy-based lag features...")
    df = add_proxy_lag_features(df)

    lead_times = [12, 9, 6, 3, 0]
    test_start = pd.Timestamp('2025-10-01')

    all_results = []          # per-quantile metrics
    all_interval_results = [] # per-lead interval metrics
    all_models = {}           # {lead: {alpha: model}}
    all_importance = []       # feature importance (q50 only)
    all_predictions = []      # test set predictions

    for lead in lead_times:
        print(f"\n{'='*60}")
        print(f"[*] LEAD TIME: {lead} MINUTES")
        print(f"{'='*60}")

        lead_df = compute_lead_features(df, lead, load_expected)
        train_df = lead_df[lead_df['datetime'] < test_start]
        test_df = lead_df[lead_df['datetime'] >= test_start]

        feature_cols = get_features_for_lead(lead)
        feature_cols = [c for c in feature_cols if c in lead_df.columns]

        print(f"[*] Features ({len(feature_cols)}): {feature_cols}")
        print(f"[*] Train: {len(train_df):,} | Test: {len(test_df):,}")

        # Get valid test data once (shared across quantiles)
        X_test, y_test, dt_test = get_valid_test_data(test_df, feature_cols)
        print(f"[*] Valid test samples: {len(y_test):,}")

        # Train all quantile models for this lead time
        lead_models = {}
        lead_preds = {}

        for alpha in QUANTILES:
            print(f"[*]   Training quantile {alpha:.2f}...", end=" ")
            model = train_model(train_df, feature_cols, alpha=alpha)
            y_pred = model.predict(X_test)

            metrics = evaluate_single(y_test, y_pred, alpha)
            metrics['lead_time'] = lead
            all_results.append(metrics)
            print(f"pinball={metrics['pinball_loss']:.3f}  coverage={metrics['coverage']:.3f}  "
                  f"mae={metrics['mae']:.3f}")

            lead_models[alpha] = model
            lead_preds[alpha] = y_pred

            # Feature importance for q50 only
            if alpha == 0.5:
                importance = model.feature_importance(importance_type='gain')
                imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importance})
                imp_df = imp_df.sort_values('importance', ascending=False)
                total = imp_df['importance'].sum()
                for _, row in imp_df.iterrows():
                    all_importance.append({
                        'lead_time': lead,
                        'feature': row['feature'],
                        'importance': row['importance'],
                        'pct': row['importance'] / total * 100,
                    })

        all_models[lead] = lead_models

        # Evaluate cross-quantile metrics (with sorting)
        interval_metrics, sorted_preds = evaluate_quantile_set(lead_preds, y_test)
        interval_metrics['lead_time'] = lead

        # q50 MAE for V4 comparison
        interval_metrics['q50_mae'] = mean_absolute_error(y_test, sorted_preds[0.5])
        all_interval_results.append(interval_metrics)

        print(f"\n[+] Interval results (lead {lead}):")
        print(f"[+]   CRPS:             {interval_metrics['crps']:.3f}")
        print(f"[+]   Crossing rate:    {interval_metrics['crossing_rate']:.1%}")
        if 'sharpness_50' in interval_metrics:
            print(f"[+]   50%% interval:    width={interval_metrics['sharpness_50']:.2f}  "
                  f"coverage={interval_metrics['coverage_50']:.1%}")
        if 'sharpness_80' in interval_metrics:
            print(f"[+]   80%% interval:    width={interval_metrics['sharpness_80']:.2f}  "
                  f"coverage={interval_metrics['coverage_80']:.1%}")

        # Collect predictions for this lead time
        pred_df = pd.DataFrame({'datetime': dt_test, 'lead_time': lead, 'y_true': y_test})
        for alpha in QUANTILES:
            pred_df[f'q{int(alpha*100):02d}'] = sorted_preds[alpha]
        all_predictions.append(pred_df)

        # Print top 10 features (from q50)
        lead_imp = [x for x in all_importance if x['lead_time'] == lead]
        lead_imp.sort(key=lambda x: x['importance'], reverse=True)
        print(f"\n[*] Top 10 features (q50):")
        for feat in lead_imp[:10]:
            print(f"[*]   {feat['feature']:<35} {feat['pct']:>5.1f}%")

    # --- Save outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'lightgbm_v8_results.csv', index=False)

    interval_df = pd.DataFrame(all_interval_results)
    interval_df.to_csv(OUTPUT_DIR / 'lightgbm_v8_interval_results.csv', index=False)

    importance_df = pd.DataFrame(all_importance)
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance_v8.csv', index=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv(OUTPUT_DIR / 'lightgbm_v8_predictions.csv', index=False)

    with open(OUTPUT_DIR / 'lightgbm_models_v8.pkl', 'wb') as f:
        pickle.dump(all_models, f)

    # --- V4 comparison ---
    print("\n" + "=" * 70)
    print("[*] COMPARISON: V8 q50 (median) vs V4 (mean regression)")
    print("=" * 70)

    v4_results_path = OUTPUT_DIR / 'lightgbm_v4_results.csv'
    if v4_results_path.exists():
        v4 = pd.read_csv(v4_results_path)
        print(f"\n{'Lead':<6} {'V4 MAE':<10} {'V8 q50':<10} {'Diff':<10} {'CRPS':<8} "
              f"{'80% Cov':<8} {'80% Width':<10} {'Crossing':<10}")
        print("-" * 72)

        for _, irow in interval_df.iterrows():
            lead = int(irow['lead_time'])
            v4_row = v4[v4['lead_time'] == lead]
            if len(v4_row) > 0:
                v4_mae = v4_row.iloc[0]['mae']
                v8_mae = irow['q50_mae']
                diff_pct = (v8_mae - v4_mae) / v4_mae * 100
                cov_80 = irow.get('coverage_80', float('nan'))
                sharp_80 = irow.get('sharpness_80', float('nan'))
                print(f"{lead:<6} {v4_mae:<10.3f} {v8_mae:<10.3f} {diff_pct:>+8.1f}%  "
                      f"{irow['crps']:<8.3f} {cov_80:<8.1%} {sharp_80:<10.2f} {irow['crossing_rate']:<10.1%}")
    else:
        print("[-] V4 results not found at %s, skipping comparison" % v4_results_path)

    # --- Calibration summary ---
    print("\n" + "=" * 70)
    print("[*] CALIBRATION SUMMARY (expected vs observed coverage)")
    print("=" * 70)
    print(f"\n{'Lead':<6}", end="")
    for alpha in QUANTILES:
        print(f"  q{int(alpha*100):02d}", end="")
    print()
    print("-" * 40)

    for lead in lead_times:
        lead_res = [r for r in all_results if r['lead_time'] == lead]
        print(f"{lead:<6}", end="")
        for r in sorted(lead_res, key=lambda x: x['quantile']):
            print(f"  {r['coverage']:.2f}", end="")
        print()

    print(f"\n{'Target':<6}", end="")
    for alpha in QUANTILES:
        print(f"  {alpha:.2f}", end="")
    print()

    print(f"\n[+] All outputs saved to: {OUTPUT_DIR}")
    print(f"[+]   lightgbm_v8_results.csv          - per-quantile metrics (25 rows)")
    print(f"[+]   lightgbm_v8_interval_results.csv  - interval metrics (5 rows)")
    print(f"[+]   lightgbm_v8_predictions.csv       - test set predictions")
    print(f"[+]   feature_importance_v8.csv          - feature importance (q50)")
    print(f"[+]   lightgbm_models_v8.pkl             - 25 trained models")


if __name__ == '__main__':
    main()
