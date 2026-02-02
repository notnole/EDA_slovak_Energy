"""
LightGBM Nowcasting Model v2 - Enhanced Features

New features for longer lead times:
1. Historical regulation stats (last 10 periods): std, trend, min, max
2. Sign-based features: last sign flag, consecutive same-sign count
3. Momentum features: change, acceleration
4. Regime features: proportion positive, volatility regime
5. Extended lags for longer leads
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Paths
FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\models\lightgbm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load regulation, load, and imbalance data."""
    print("Loading data...")
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
    load_df = pd.read_csv(FEATURES_DIR / 'load_3min.csv', parse_dates=['datetime'])
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
        columns={'System Imbalance (MWh)': 'imbalance'}
    )
    print(f"  Regulation: {len(reg_df):,} rows")
    print(f"  Load: {len(load_df):,} rows")
    print(f"  Labels: {len(label_df):,} rows")
    return reg_df, load_df, label_df


def compute_load_tod_expected(load_df):
    """Compute expected load by time-of-day."""
    load_df = load_df.copy()
    load_df['hour'] = load_df['datetime'].dt.hour
    load_df['minute'] = load_df['datetime'].dt.minute
    load_df['is_weekend'] = load_df['datetime'].dt.dayofweek >= 5
    train_mask = load_df['datetime'].dt.year == 2024
    train_load = load_df[train_mask]
    expected = train_load.groupby(['hour', 'minute', 'is_weekend'])['load_mw'].mean()
    expected.name = 'expected_load'
    return expected


def create_quarter_hour_features(reg_df, load_df, label_df, load_expected):
    """Create features for each quarter-hour settlement period."""
    print("\nCreating quarter-hour features...")

    # Align regulation to settlement periods
    reg_df = reg_df.copy()
    reg_df['datetime_floor'] = reg_df['datetime'].dt.floor('3min')
    reg_df['settlement_end'] = reg_df['datetime_floor'].dt.ceil('15min')
    mask = reg_df['datetime_floor'] == reg_df['settlement_end']
    reg_df.loc[mask, 'settlement_end'] = reg_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    reg_df['settlement_start'] = reg_df['settlement_end'] - pd.Timedelta(minutes=15)
    reg_df['minute_in_qh'] = (reg_df['datetime_floor'] - reg_df['settlement_start']).dt.total_seconds() / 60

    # Pivot regulation
    pivot = reg_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='regulation_mw', aggfunc='first'
    ).reset_index()
    pivot.columns = ['datetime'] + [f'reg_min{int(c)}' for c in pivot.columns[1:]]

    # Merge with labels
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
    df['month'] = df['datetime'].dt.month

    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    print(f"  Created {len(df):,} settlement period rows")
    return df


def compute_features_for_lead_time(df, lead_time, load_expected):
    """Compute features available at a specific lead time."""
    result = df[['datetime', 'imbalance', 'hour', 'day_of_week', 'is_weekend',
                 'month', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']].copy()
    result['lead_time'] = lead_time

    available_minutes = {
        12: [0],
        9: [0, 3],
        6: [0, 3, 6],
        3: [0, 3, 6, 9],
        0: [0, 3, 6, 9, 12],
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
    elif lead_time == 0:
        result['baseline_pred'] = -0.25 * df[reg_cols].mean(axis=1)

    # Load deviation
    if len(load_cols) > 0:
        load_mean = df[load_cols].mean(axis=1)
        result['load_mean'] = load_mean
        temp = df[['hour', 'is_weekend']].copy()
        temp['minute'] = mins[0]
        temp = temp.merge(load_expected.reset_index(), on=['hour', 'minute', 'is_weekend'], how='left')
        result['load_deviation'] = load_mean - temp['expected_load'].values
    else:
        result['load_deviation'] = 0

    # Within-period regulation stats (only if 2+ observations)
    if len(reg_cols) >= 2:
        result['reg_std'] = df[reg_cols].std(axis=1)
        result['reg_range'] = df[reg_cols].max(axis=1) - df[reg_cols].min(axis=1)
        result['reg_trend'] = df[reg_cols[-1]] - df[reg_cols[0]]
    else:
        result['reg_std'] = 0
        result['reg_range'] = 0
        result['reg_trend'] = 0

    return result


def add_enhanced_lag_features(df):
    """
    Add enhanced lag features including:
    - Basic lags (1-4)
    - Historical stats (last 10 periods)
    - Sign-based features
    - Momentum features
    - Regime features
    """
    df = df.sort_values(['lead_time', 'datetime']).copy()
    lag_features = []

    for lead in df['lead_time'].unique():
        lead_df = df[df['lead_time'] == lead].copy()

        # Imbalance proxy = baseline prediction (our best estimate of prior imbalance)
        proxy = lead_df['baseline_pred']

        # ============================================================
        # 1. BASIC LAG FEATURES
        # ============================================================
        lead_df['imb_proxy_lag1'] = proxy.shift(1)
        lead_df['imb_proxy_lag2'] = proxy.shift(2)
        lead_df['imb_proxy_lag3'] = proxy.shift(3)
        lead_df['imb_proxy_lag4'] = proxy.shift(4)

        # Rolling stats over last 4 periods
        lead_df['imb_proxy_rolling_mean4'] = proxy.shift(1).rolling(4).mean()
        lead_df['imb_proxy_rolling_std4'] = proxy.shift(1).rolling(4).std()

        # ============================================================
        # 2. HISTORICAL STATS (last 10 periods = 2.5 hours)
        # ============================================================
        lead_df['hist_mean_10'] = proxy.shift(1).rolling(10).mean()
        lead_df['hist_std_10'] = proxy.shift(1).rolling(10).std()
        lead_df['hist_min_10'] = proxy.shift(1).rolling(10).min()
        lead_df['hist_max_10'] = proxy.shift(1).rolling(10).max()
        lead_df['hist_range_10'] = lead_df['hist_max_10'] - lead_df['hist_min_10']

        # Trend over last 10 periods (linear regression slope approximation)
        # Using simple difference: last - first of window
        lead_df['hist_trend_10'] = proxy.shift(1) - proxy.shift(10)

        # ============================================================
        # 3. SIGN-BASED FEATURES
        # ============================================================
        # Last sign (1 = positive, -1 = negative, 0 = zero)
        lead_df['last_sign'] = np.sign(proxy.shift(1))

        # Is last positive/negative (binary flags)
        lead_df['last_was_positive'] = (proxy.shift(1) > 0).astype(int)
        lead_df['last_was_negative'] = (proxy.shift(1) < 0).astype(int)

        # Consecutive same-sign counter
        signs = np.sign(proxy)
        sign_change = (signs != signs.shift(1)).astype(int)
        sign_change.iloc[0] = 1  # First row starts a new sequence

        # Group consecutive same-sign periods
        groups = sign_change.cumsum()
        consecutive_count = groups.groupby(groups).cumcount() + 1

        # Shift to get "how many consecutive same-sign periods BEFORE current"
        lead_df['consecutive_same_sign'] = consecutive_count.shift(1)

        # Proportion positive in last N periods
        positive_flags = (proxy > 0).astype(float)
        lead_df['prop_positive_4'] = positive_flags.shift(1).rolling(4).mean()
        lead_df['prop_positive_10'] = positive_flags.shift(1).rolling(10).mean()

        # ============================================================
        # 4. MOMENTUM FEATURES
        # ============================================================
        # Change from lag2 to lag1 (momentum)
        lead_df['momentum'] = proxy.shift(1) - proxy.shift(2)

        # Acceleration (change in momentum)
        momentum = proxy.shift(1) - proxy.shift(2)
        lead_df['acceleration'] = momentum - momentum.shift(1)

        # Rate of change (percentage)
        lead_df['pct_change'] = proxy.pct_change().shift(1)
        lead_df['pct_change'] = lead_df['pct_change'].clip(-10, 10)  # Clip extreme values

        # ============================================================
        # 5. REGIME FEATURES
        # ============================================================
        # Volatility regime (high/low based on rolling std vs long-term median)
        rolling_std_10 = proxy.shift(1).rolling(10).std()
        long_term_std = proxy.shift(1).rolling(100, min_periods=20).std()
        lead_df['volatility_ratio'] = rolling_std_10 / long_term_std.clip(lower=0.1)
        lead_df['high_volatility'] = (lead_df['volatility_ratio'] > 1.5).astype(int)

        # Mean reversion indicator (how far from rolling mean)
        lead_df['deviation_from_mean'] = proxy.shift(1) - lead_df['hist_mean_10']

        # ============================================================
        # 6. TIME-BASED INTERACTIONS (for longer leads)
        # ============================================================
        # Hour group (categorical effect)
        hour = lead_df['hour']
        lead_df['is_night'] = ((hour >= 0) & (hour < 6)).astype(int)
        lead_df['is_morning'] = ((hour >= 6) & (hour < 11)).astype(int)
        lead_df['is_peak'] = ((hour >= 11) & (hour < 15)).astype(int)
        lead_df['is_afternoon'] = ((hour >= 15) & (hour < 21)).astype(int)
        lead_df['is_evening'] = ((hour >= 21) & (hour < 24)).astype(int)

        lag_features.append(lead_df)

    return pd.concat(lag_features, ignore_index=True)


def get_feature_columns_for_lead(lead_time):
    """
    Return optimized feature columns for each lead time.
    Longer leads get more lag/historical features.
    Shorter leads focus on current-period info.
    """
    # Core features (all leads)
    core = [
        'reg_cumulative_mean',
        'baseline_pred',
    ]

    # Within-period stats (only useful when we have 2+ observations)
    within_period = ['reg_std', 'reg_range', 'reg_trend']

    # Basic lag features
    basic_lags = [
        'imb_proxy_lag1',
        'imb_proxy_rolling_mean4',
    ]

    # Extended lag features (more useful at longer leads)
    extended_lags = [
        'imb_proxy_lag2',
        'imb_proxy_lag3',
        'imb_proxy_lag4',
        'imb_proxy_rolling_std4',
    ]

    # Historical stats (last 10 periods)
    historical = [
        'hist_mean_10',
        'hist_std_10',
        'hist_min_10',
        'hist_max_10',
        'hist_range_10',
        'hist_trend_10',
    ]

    # Sign-based features
    sign_features = [
        'last_sign',
        'last_was_positive',
        'consecutive_same_sign',
        'prop_positive_4',
        'prop_positive_10',
    ]

    # Momentum features
    momentum = [
        'momentum',
        'acceleration',
    ]

    # Regime features
    regime = [
        'volatility_ratio',
        'high_volatility',
        'deviation_from_mean',
    ]

    # Time features
    time_features = [
        'hour_sin',
        'hour_cos',
        'is_weekend',
    ]

    # Hour group features
    hour_groups = [
        'is_night',
        'is_morning',
        'is_peak',
        'is_afternoon',
        'is_evening',
    ]

    # Other
    other = [
        'load_deviation',
    ]

    # Build feature set based on lead time
    if lead_time == 12:
        # Longest lead: maximize use of historical/lag features
        # No within-period stats (only 1 observation)
        features = (core + basic_lags + extended_lags + historical +
                   sign_features + momentum + regime + time_features +
                   hour_groups + other)
    elif lead_time == 9:
        # Within-period stats now available
        features = (core + within_period + basic_lags + extended_lags +
                   historical + sign_features + momentum + regime +
                   time_features + other)
    elif lead_time == 6:
        # More current info available, reduce lag dependency
        features = (core + within_period + basic_lags + extended_lags[:2] +
                   historical[:4] + sign_features[:3] + momentum[:1] +
                   regime[:1] + time_features)
    elif lead_time == 3:
        # Even more current info
        features = (core + within_period + basic_lags +
                   historical[:2] + sign_features[:2] + time_features[:2])
    else:  # lead_time == 0
        # Most current info available, minimal lag features needed
        features = (core + within_period + basic_lags[:1] +
                   sign_features[:1])

    return features


def train_lightgbm_model(train_df, feature_cols, target='imbalance'):
    """Train LightGBM model."""
    X = train_df[feature_cols].values
    y = train_df[target].values

    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isinf(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

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


def evaluate_model(model, test_df, feature_cols, target='imbalance'):
    """Evaluate model on test set."""
    X = test_df[feature_cols].values
    y_true = test_df[target].values

    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y_true) & ~np.isinf(X).any(axis=1)
    X = X[valid_mask]
    y_true = y_true[valid_mask]
    test_valid = test_df[valid_mask].copy()

    y_pred = model.predict(X)

    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'bias': np.mean(y_pred - y_true),
        'n_samples': len(y_true),
        'directional_accuracy': (np.sign(y_pred) == np.sign(y_true)).mean(),
    }

    return metrics, y_pred, y_true, test_valid


def main():
    print("=" * 70)
    print("LIGHTGBM v2 - ENHANCED FEATURES")
    print("=" * 70)

    # Load data
    reg_df, load_df, label_df = load_data()
    load_expected = compute_load_tod_expected(load_df)
    qh_df = create_quarter_hour_features(reg_df, load_df, label_df, load_expected)

    # Create features for each lead time
    lead_times = [12, 9, 6, 3, 0]
    all_data = []

    for lead in lead_times:
        print(f"\nProcessing lead time {lead} min...")
        lead_df = compute_features_for_lead_time(qh_df, lead, load_expected)
        all_data.append(lead_df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Add enhanced lag features
    print("\nAdding enhanced lag features...")
    combined_df = add_enhanced_lag_features(combined_df)

    # Replace inf with nan
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)

    print(f"\nTotal samples: {len(combined_df):,}")

    # Train/test split
    test_start = pd.Timestamp('2025-10-01')

    # Results storage
    all_results = []
    models = {}
    feature_importance_all = []

    # Train and evaluate for each lead time
    for lead in lead_times:
        print(f"\n{'='*60}")
        print(f"LEAD TIME: {lead} MINUTES")
        print(f"{'='*60}")

        lead_df = combined_df[combined_df['lead_time'] == lead].copy()
        train_df = lead_df[lead_df['datetime'] < test_start]
        test_df = lead_df[lead_df['datetime'] >= test_start]

        # Get feature columns for this lead
        feature_cols = get_feature_columns_for_lead(lead)

        # Filter to only available columns
        feature_cols = [c for c in feature_cols if c in lead_df.columns]

        print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
        print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

        # Train model
        print("\nTraining LightGBM...")
        model = train_lightgbm_model(train_df, feature_cols)

        # Evaluate
        metrics, y_pred, y_true, test_valid = evaluate_model(model, test_df, feature_cols)

        print(f"\nTest Set Performance:")
        print(f"  MAE: {metrics['mae']:.3f} MWh")
        print(f"  RMSE: {metrics['rmse']:.3f} MWh")
        print(f"  R2: {metrics['r2']:.3f}")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.1%}")

        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        for feat, imp in zip(feature_cols, importance):
            feature_importance_all.append({
                'lead_time': lead,
                'feature': feat,
                'importance': imp
            })

        # Top 10 features
        imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importance})
        imp_df = imp_df.sort_values('importance', ascending=False)
        print(f"\nTop 10 Features:")
        for _, row in imp_df.head(10).iterrows():
            pct = row['importance'] / imp_df['importance'].sum() * 100
            print(f"  {row['feature']:<30} {pct:>6.1f}%")

        all_results.append({'lead_time': lead, **metrics})
        models[lead] = model

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'lightgbm_v2_results.csv', index=False)

    importance_df = pd.DataFrame(feature_importance_all)
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance_v2.csv', index=False)

    with open(OUTPUT_DIR / 'lightgbm_models_v2.pkl', 'wb') as f:
        pickle.dump(models, f)

    # Load v1 results for comparison
    v1_results = pd.read_csv(OUTPUT_DIR / 'lightgbm_results.csv')

    # Summary comparison
    print("\n" + "=" * 70)
    print("V1 vs V2 COMPARISON")
    print("=" * 70)

    print(f"\n{'Lead':<8} {'V1 MAE':<12} {'V2 MAE':<12} {'Change':<12} {'V1 Dir.Acc':<12} {'V2 Dir.Acc':<12}")
    print("-" * 68)

    for _, v2_row in results_df.iterrows():
        lead = int(v2_row['lead_time'])
        v1_row = v1_results[v1_results['lead_time'] == lead].iloc[0]

        v1_mae = v1_row['mae']
        v2_mae = v2_row['mae']
        change = (v2_mae - v1_mae) / v1_mae * 100

        v1_dir = v1_row['directional_accuracy'] * 100
        v2_dir = v2_row['directional_accuracy'] * 100

        print(f"{lead:<8} {v1_mae:<12.3f} {v2_mae:<12.3f} {change:>+10.1f}% {v1_dir:<12.1f} {v2_dir:<12.1f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
