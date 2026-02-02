"""
LightGBM Nowcasting Model for System Imbalance

Train 5 separate LightGBM models (one per lead time) to predict
15-minute system imbalance from 3-minute regulation data.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
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
    """Load regulation and imbalance data."""
    print("Loading data...")

    # Load 3-min regulation data
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])

    # Load 3-min load data
    load_df = pd.read_csv(FEATURES_DIR / 'load_3min.csv', parse_dates=['datetime'])

    # Load 15-min imbalance labels
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
        columns={'System Imbalance (MWh)': 'imbalance'}
    )

    print(f"  Regulation: {len(reg_df):,} rows")
    print(f"  Load: {len(load_df):,} rows")
    print(f"  Labels: {len(label_df):,} rows")

    return reg_df, load_df, label_df


def compute_load_tod_expected(load_df):
    """
    Compute expected load by time-of-day (hour, minute, is_weekend).
    Returns lookup table from training period (2024).
    """
    load_df = load_df.copy()
    load_df['hour'] = load_df['datetime'].dt.hour
    load_df['minute'] = load_df['datetime'].dt.minute
    load_df['is_weekend'] = load_df['datetime'].dt.dayofweek >= 5

    # Use 2024 data for expected values
    train_mask = load_df['datetime'].dt.year == 2024
    train_load = load_df[train_mask]

    expected = train_load.groupby(['hour', 'minute', 'is_weekend'])['load_mw'].mean()
    expected.name = 'expected_load'

    return expected


def create_quarter_hour_features(reg_df, load_df, label_df, load_expected):
    """
    Create features for each quarter-hour settlement period.

    Returns a dataframe with one row per (settlement_period, lead_time) combination.
    """
    print("\nCreating quarter-hour features...")

    # Align regulation to settlement periods
    reg_df = reg_df.copy()
    reg_df['datetime_floor'] = reg_df['datetime'].dt.floor('3min')

    # Determine settlement period (label datetime = START of period)
    reg_df['settlement_end'] = reg_df['datetime_floor'].dt.ceil('15min')
    mask = reg_df['datetime_floor'] == reg_df['settlement_end']
    reg_df.loc[mask, 'settlement_end'] = reg_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    reg_df['settlement_start'] = reg_df['settlement_end'] - pd.Timedelta(minutes=15)

    # Minute within quarter-hour (0, 3, 6, 9, 12)
    reg_df['minute_in_qh'] = (reg_df['datetime_floor'] - reg_df['settlement_start']).dt.total_seconds() / 60

    # Pivot to get one row per settlement period with columns for each minute
    pivot = reg_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='regulation_mw', aggfunc='first'
    ).reset_index()
    pivot.columns = ['datetime'] + [f'reg_min{int(c)}' for c in pivot.columns[1:]]

    # Merge with labels
    df = pd.merge(label_df, pivot, on='datetime', how='inner')

    # Also get load data aligned to settlement periods
    load_df = load_df.copy()
    load_df['datetime_floor'] = load_df['datetime'].dt.floor('3min')
    load_df['settlement_end'] = load_df['datetime_floor'].dt.ceil('15min')
    mask = load_df['datetime_floor'] == load_df['settlement_end']
    load_df.loc[mask, 'settlement_end'] = load_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    load_df['settlement_start'] = load_df['settlement_end'] - pd.Timedelta(minutes=15)
    load_df['minute_in_qh'] = (load_df['datetime_floor'] - load_df['settlement_start']).dt.total_seconds() / 60

    # Pivot load
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

    # Cyclical hour encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    print(f"  Created {len(df):,} settlement period rows")

    return df


def compute_features_for_lead_time(df, lead_time, load_expected):
    """
    Compute features available at a specific lead time.

    Lead time = minutes until end of settlement period.
    - Lead 12: only reg_min0 available (observation at minute 3)
    - Lead 9: reg_min0, reg_min3 available
    - Lead 6: reg_min0, reg_min3, reg_min6 available
    - Lead 3: reg_min0, reg_min3, reg_min6, reg_min9 available
    - Lead 0: all 5 observations available
    """
    result = df[['datetime', 'imbalance', 'hour', 'day_of_week', 'is_weekend',
                 'month', 'hour_sin', 'hour_cos']].copy()
    result['lead_time'] = lead_time

    # Determine which regulation minutes are available
    # At lead_time X, observations up to minute (15-X-3) are available
    # Lead 12: minute 0 (first obs at 00:03 tells us reg at 00:00-00:03)
    # Wait - need to reconsider timing
    #
    # Settlement period 00:00-00:15:
    # - Observation at 00:03 -> reg_min0 available, lead time = 12 min
    # - Observation at 00:06 -> reg_min3 available, lead time = 9 min
    # - Observation at 00:09 -> reg_min6 available, lead time = 6 min
    # - Observation at 00:12 -> reg_min9 available, lead time = 3 min
    # - Observation at 00:15 -> reg_min12 available, lead time = 0 min

    available_minutes = {
        12: [0],              # Only first observation
        9: [0, 3],            # First two observations
        6: [0, 3, 6],         # First three observations
        3: [0, 3, 6, 9],      # First four observations
        0: [0, 3, 6, 9, 12],  # All five observations
    }

    mins = available_minutes[lead_time]
    reg_cols = [f'reg_min{m}' for m in mins]
    load_cols = [f'load_min{m}' for m in mins if f'load_min{m}' in df.columns]

    # 2.1 Cumulative regulation mean
    result['reg_cumulative_mean'] = df[reg_cols].mean(axis=1)

    # 2.3 Baseline prediction (weighted average as per BaseLine.md)
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

    # 2.4 Load deviation
    if len(load_cols) > 0:
        load_mean = df[load_cols].mean(axis=1)
        # Get expected load for each row
        result['load_mean'] = load_mean

        # Lookup expected load
        temp = df[['hour', 'is_weekend']].copy()
        # Use minute 0 for simplicity (or could use mean minute)
        temp['minute'] = mins[0]
        temp = temp.merge(
            load_expected.reset_index(),
            on=['hour', 'minute', 'is_weekend'],
            how='left'
        )
        result['load_deviation'] = load_mean - temp['expected_load'].values
    else:
        result['load_deviation'] = 0

    # 2.5 Rolling statistics on current period regulation
    if len(reg_cols) >= 2:
        result['reg_std'] = df[reg_cols].std(axis=1)
        result['reg_max'] = df[reg_cols].max(axis=1)
        result['reg_min'] = df[reg_cols].min(axis=1)
        result['reg_range'] = result['reg_max'] - result['reg_min']
        result['reg_trend'] = df[reg_cols[-1]] - df[reg_cols[0]]  # Last - first
    else:
        result['reg_std'] = 0
        result['reg_max'] = df[reg_cols[0]]
        result['reg_min'] = df[reg_cols[0]]
        result['reg_range'] = 0
        result['reg_trend'] = 0

    return result


def add_lag_features(df):
    """
    Add lag features: imbalance proxy from previous periods.

    imbalance_proxy_lag1 = -0.25 * mean(previous period's regulation)
    """
    df = df.sort_values(['lead_time', 'datetime']).copy()

    # Group by lead_time and compute lags within each group
    lag_features = []

    for lead in df['lead_time'].unique():
        lead_df = df[df['lead_time'] == lead].copy()

        # Lag 1: previous period's baseline prediction as proxy
        lead_df['imb_proxy_lag1'] = lead_df['baseline_pred'].shift(1)

        # Lag 2
        lead_df['imb_proxy_lag2'] = lead_df['baseline_pred'].shift(2)

        # Lag 3
        lead_df['imb_proxy_lag3'] = lead_df['baseline_pred'].shift(3)

        # Rolling mean of proxy over last 4 periods
        lead_df['imb_proxy_rolling_mean4'] = lead_df['baseline_pred'].shift(1).rolling(4).mean()

        # Rolling std of proxy
        lead_df['imb_proxy_rolling_std4'] = lead_df['baseline_pred'].shift(1).rolling(4).std()

        lag_features.append(lead_df)

    return pd.concat(lag_features, ignore_index=True)


def prepare_train_test_split(df, test_start='2025-10-01'):
    """
    Time-based train/test split.
    Train: before test_start
    Test: test_start onwards
    """
    test_start = pd.Timestamp(test_start)

    train = df[df['datetime'] < test_start].copy()
    test = df[df['datetime'] >= test_start].copy()

    print(f"\n  Train: {len(train):,} rows ({train['datetime'].min()} to {train['datetime'].max()})")
    print(f"  Test: {len(test):,} rows ({test['datetime'].min()} to {test['datetime'].max()})")

    return train, test


def get_feature_columns():
    """Return list of feature columns to use."""
    return [
        'reg_cumulative_mean',
        'baseline_pred',
        'load_deviation',
        'reg_std',
        'reg_range',
        'reg_trend',
        'hour_sin',
        'hour_cos',
        'is_weekend',
        'imb_proxy_lag1',
        'imb_proxy_lag2',
        'imb_proxy_rolling_mean4',
        'imb_proxy_rolling_std4',
    ]


def train_lightgbm_model(train_df, feature_cols, target='imbalance'):
    """
    Train LightGBM model with time series cross-validation.
    """
    X = train_df[feature_cols].values
    y = train_df[target].values

    # Handle NaN from lag features
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    # LightGBM parameters (initial conservative settings)
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

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        val_pred = model.predict(X_val)
        cv_scores.append(mean_absolute_error(y_val, val_pred))

    # Train final model on all data
    train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
    )

    return final_model, np.mean(cv_scores)


def evaluate_model(model, test_df, feature_cols, target='imbalance'):
    """Evaluate model on test set."""
    X = test_df[feature_cols].values
    y_true = test_df[target].values

    # Handle NaN
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y_true)
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
    }

    # Directional accuracy (sign prediction)
    sign_correct = (np.sign(y_pred) == np.sign(y_true)).mean()
    metrics['directional_accuracy'] = sign_correct

    return metrics, y_pred, y_true, test_valid


def stratified_evaluation(test_df, y_pred, y_true, lead_time):
    """Evaluate by magnitude, hour, and sign."""
    results = {'lead_time': lead_time}

    df = test_df.copy()
    df['y_pred'] = y_pred
    df['y_true'] = y_true
    df['error'] = y_pred - y_true
    df['abs_error'] = np.abs(df['error'])

    # By imbalance magnitude
    df['imb_magnitude'] = pd.cut(
        np.abs(df['y_true']),
        bins=[0, 2, 5, 10, 1000],
        labels=['small', 'medium', 'large', 'extreme']
    )

    mag_results = df.groupby('imb_magnitude', observed=True).agg({
        'abs_error': 'mean',
        'y_true': 'count'
    }).rename(columns={'abs_error': 'mae', 'y_true': 'n'})

    for mag in ['small', 'medium', 'large', 'extreme']:
        if mag in mag_results.index:
            results[f'mae_{mag}'] = mag_results.loc[mag, 'mae']
            results[f'n_{mag}'] = mag_results.loc[mag, 'n']

    # By hour group
    df['hour_group'] = pd.cut(
        df['hour'],
        bins=[-1, 5, 10, 14, 20, 24],
        labels=['night', 'morning', 'peak', 'afternoon', 'evening']
    )

    hour_results = df.groupby('hour_group', observed=True)['abs_error'].mean()
    for hg in ['night', 'morning', 'peak', 'afternoon', 'evening']:
        if hg in hour_results.index:
            results[f'mae_{hg}'] = hour_results[hg]

    # By sign
    pos_mask = df['y_true'] > 0
    neg_mask = df['y_true'] < 0

    if pos_mask.sum() > 0:
        results['mae_positive'] = df.loc[pos_mask, 'abs_error'].mean()
    if neg_mask.sum() > 0:
        results['mae_negative'] = df.loc[neg_mask, 'abs_error'].mean()

    return results


def main():
    print("=" * 70)
    print("LIGHTGBM NOWCASTING MODEL")
    print("=" * 70)

    # Load data
    reg_df, load_df, label_df = load_data()

    # Compute load expected values from 2024
    load_expected = compute_load_tod_expected(load_df)

    # Create base features
    qh_df = create_quarter_hour_features(reg_df, load_df, label_df, load_expected)

    # Create features for each lead time
    lead_times = [12, 9, 6, 3, 0]
    all_data = []

    for lead in lead_times:
        print(f"\nProcessing lead time {lead} min...")
        lead_df = compute_features_for_lead_time(qh_df, lead, load_expected)
        all_data.append(lead_df)

    # Combine all lead times
    combined_df = pd.concat(all_data, ignore_index=True)

    # Add lag features
    combined_df = add_lag_features(combined_df)

    # Drop rows with NaN
    combined_df = combined_df.dropna()

    print(f"\nTotal samples: {len(combined_df):,}")

    # Feature columns
    feature_cols = get_feature_columns()
    print(f"Features: {feature_cols}")

    # Results storage
    all_results = []
    models = {}

    # Train and evaluate for each lead time
    for lead in lead_times:
        print(f"\n{'='*50}")
        print(f"LEAD TIME: {lead} MINUTES")
        print(f"{'='*50}")

        lead_df = combined_df[combined_df['lead_time'] == lead].copy()

        # Train/test split
        train_df, test_df = prepare_train_test_split(lead_df)

        # Train model
        print("\nTraining LightGBM...")
        model, cv_mae = train_lightgbm_model(train_df, feature_cols)
        print(f"  CV MAE: {cv_mae:.3f}")

        # Evaluate on test set
        metrics, y_pred, y_true, test_valid = evaluate_model(model, test_df, feature_cols)

        print(f"\nTest Set Performance:")
        print(f"  MAE: {metrics['mae']:.3f} MWh")
        print(f"  RMSE: {metrics['rmse']:.3f} MWh")
        print(f"  R2: {metrics['r2']:.3f}")
        print(f"  Bias: {metrics['bias']:.3f} MWh")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.1%}")

        # Stratified evaluation
        strat_results = stratified_evaluation(test_valid, y_pred, y_true, lead)

        print(f"\nMAE by Imbalance Magnitude:")
        for mag in ['small', 'medium', 'large', 'extreme']:
            key = f'mae_{mag}'
            if key in strat_results:
                print(f"  {mag}: {strat_results[key]:.2f} MWh (n={strat_results.get(f'n_{mag}', 0):.0f})")

        print(f"\nMAE by Hour Group:")
        for hg in ['night', 'morning', 'peak', 'afternoon', 'evening']:
            key = f'mae_{hg}'
            if key in strat_results:
                print(f"  {hg}: {strat_results[key]:.2f} MWh")

        # Store results
        result_row = {
            'lead_time': lead,
            **metrics,
            **strat_results,
        }
        all_results.append(result_row)
        models[lead] = model

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'lightgbm_results.csv', index=False)
    print(f"\nResults saved to: {OUTPUT_DIR / 'lightgbm_results.csv'}")

    # Save models
    with open(OUTPUT_DIR / 'lightgbm_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print(f"Models saved to: {OUTPUT_DIR / 'lightgbm_models.pkl'}")

    # Summary comparison with baseline
    print("\n" + "=" * 70)
    print("SUMMARY: LIGHTGBM vs BASELINE")
    print("=" * 70)

    baseline_mae = {12: 6.59, 9: 4.74, 6: 3.41, 3: 2.52, 0: 2.15}
    baseline_r2 = {12: 0.307, 9: 0.631, 6: 0.781, 3: 0.842, 0: 0.860}

    print(f"\n{'Lead':<8} {'Baseline MAE':<14} {'LightGBM MAE':<14} {'Improvement':<12} {'Baseline R2':<12} {'LightGBM R2':<12}")
    print("-" * 72)

    for _, row in results_df.iterrows():
        lead = int(row['lead_time'])
        bl_mae = baseline_mae[lead]
        lgb_mae = row['mae']
        improvement = (bl_mae - lgb_mae) / bl_mae * 100
        bl_r2 = baseline_r2[lead]
        lgb_r2 = row['r2']

        print(f"{lead:<8} {bl_mae:<14.2f} {lgb_mae:<14.2f} {improvement:>+10.1f}% {bl_r2:<12.3f} {lgb_r2:<12.3f}")

    # Feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Lead 12 model)")
    print("=" * 70)

    importance = models[12].feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    for _, row in importance_df.iterrows():
        print(f"  {row['feature']:<30} {row['importance']:>10.1f}")

    importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
