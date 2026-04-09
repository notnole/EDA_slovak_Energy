"""
Comprehensive Hyperparameter Tuning for Q0 H+2 Model
=====================================================
Tune EVERYTHING to see if it's worth the effort for all 20 models.

Tuning categories:
1. LightGBM hyperparameters (12+ params)
2. Feature engineering (lag depths, rolling windows, etc.)
3. Training strategy (fine-tune weight, fine-tune window)

Uses Optuna with time-series cross-validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_PATH = Path(__file__).parent.parent.parent.parent  # ipesoft_eda_data

# Global data cache
_DATA_CACHE = {}


def load_data():
    """Load all data sources (cached)."""
    if 'df' in _DATA_CACHE:
        return _DATA_CACHE['df'].copy()

    print("[*] Loading data...")

    df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']

    # 3-minute load
    load_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'load_3min.csv')
    load_3min['datetime'] = pd.to_datetime(load_3min['datetime'])
    load_3min['hour_start'] = load_3min['datetime'].dt.floor('h')
    load_hourly = load_3min.groupby('hour_start').agg({
        'load_mw': ['std', 'first', 'last', 'mean', 'min', 'max']
    }).reset_index()
    load_hourly.columns = ['datetime', 'load_std_3min', 'load_first', 'load_last',
                           'load_mean_3min', 'load_min_3min', 'load_max_3min']
    load_hourly['load_trend_3min'] = load_hourly['load_last'] - load_hourly['load_first']
    load_hourly['load_range_3min'] = load_hourly['load_max_3min'] - load_hourly['load_min_3min']
    df = df.merge(load_hourly[['datetime', 'load_std_3min', 'load_trend_3min',
                                'load_mean_3min', 'load_range_3min']],
                  on='datetime', how='left')

    # Regulation
    reg_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'regulation_3min.csv')
    reg_3min['datetime'] = pd.to_datetime(reg_3min['datetime'])
    reg_3min['hour_start'] = reg_3min['datetime'].dt.floor('h')
    reg_hourly = reg_3min.groupby('hour_start').agg({
        'regulation_mw': ['mean', 'std', 'min', 'max']
    }).reset_index()
    reg_hourly.columns = ['datetime', 'reg_mean', 'reg_std', 'reg_min', 'reg_max']
    reg_hourly['reg_range'] = reg_hourly['reg_max'] - reg_hourly['reg_min']
    df = df.merge(reg_hourly, on='datetime', how='left')

    print(f"    Loaded {len(df):,} records")
    _DATA_CACHE['df'] = df
    return df.copy()


def create_features(df, params, train_subset=None):
    """Create features based on tunable parameters."""
    df = df.copy()

    # === ERROR LAGS (tunable depth) ===
    max_lag = params.get('max_error_lag', 8)
    for lag in range(1, max_lag + 1):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    # === ROLLING WINDOWS (tunable) ===
    windows = []
    if params.get('use_roll_3h', True):
        windows.append(3)
    if params.get('use_roll_6h', True):
        windows.append(6)
    if params.get('use_roll_12h', True):
        windows.append(12)
    if params.get('use_roll_24h', True):
        windows.append(24)
    if params.get('use_roll_48h', False):
        windows.append(48)

    for window in windows:
        df[f'error_roll_mean_{window}h'] = df['error'].shift(1).rolling(window).mean()
        df[f'error_roll_std_{window}h'] = df['error'].shift(1).rolling(window).std()
        if params.get('use_roll_median', False):
            df[f'error_roll_median_{window}h'] = df['error'].shift(1).rolling(window).median()

    # === ERROR TRENDS (tunable) ===
    if params.get('use_error_trends', True):
        df['error_trend_3h'] = df['error_lag1'] - df.get('error_lag3', df['error_lag1'])
        df['error_trend_6h'] = df['error_lag1'] - df.get('error_lag6', df['error_lag1'])

    # === MOMENTUM (tunable weights) ===
    if params.get('use_momentum', True):
        w1 = params.get('momentum_w1', 0.5)
        w2 = params.get('momentum_w2', 0.3)
        w3 = 1.0 - w1 - w2
        df['error_momentum'] = (w1 * (df['error_lag1'] - df['error_lag2']) +
                                w2 * (df['error_lag2'] - df.get('error_lag3', df['error_lag2'])) +
                                w3 * (df.get('error_lag3', df['error_lag2']) - df.get('error_lag4', df['error_lag2'])))

    # === 3-MIN LOAD FEATURES (tunable) ===
    if params.get('use_load_volatility', True):
        df['load_volatility_lag1'] = df['load_std_3min'].shift(1)
    if params.get('use_load_trend', True):
        df['load_trend_lag1'] = df['load_trend_3min'].shift(1)
    if params.get('use_load_range', False):
        df['load_range_lag1'] = df['load_range_3min'].shift(1)

    # === REGULATION FEATURES (tunable depth) ===
    reg_lag_depth = params.get('reg_lag_depth', 3)
    for lag in range(1, reg_lag_depth + 1):
        df[f'reg_mean_lag{lag}'] = df['reg_mean'].shift(lag)
    if params.get('use_reg_std', True):
        df['reg_std_lag1'] = df['reg_std'].shift(1)
    if params.get('use_reg_range', False):
        df['reg_range_lag1'] = df['reg_range'].shift(1)

    # === SEASONAL ERROR (from training data only) ===
    if params.get('use_seasonal_error', True):
        if train_subset is not None:
            seasonal_map = train_subset.groupby(['dow', 'hour'])['error'].mean()
            df['seasonal_error'] = df.apply(
                lambda r: seasonal_map.get((r['dow'], r['hour']), 0), axis=1)
        else:
            df['seasonal_error'] = df.groupby(['dow', 'hour'])['error'].transform('mean')

    # === TIME FEATURES (tunable) ===
    if params.get('use_hour_cyclical', True):
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    if params.get('use_dow_cyclical', False):
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    if params.get('use_is_weekend', True):
        df['is_weekend'] = (df['dow'] >= 5).astype(int)
    if params.get('use_hour_raw', True):
        pass  # hour already exists
    if params.get('use_dow_raw', True):
        pass  # dow already exists

    # === INTERACTION FEATURES (tunable) ===
    if params.get('use_hour_error_interaction', False):
        df['hour_x_error_lag1'] = df['hour'] * df['error_lag1'] / 100

    # === SAME-HOUR FEATURES (tunable) ===
    if params.get('use_same_hour_yesterday', True):
        df['error_same_hour_yesterday'] = df['error'].shift(24)
    if params.get('use_same_hour_2d', False):
        df['error_same_hour_2d'] = df['error'].shift(48)
    if params.get('use_same_hour_week', False):
        df['error_same_hour_week'] = df['error'].shift(168)

    # === ERROR SIGN FEATURES (tunable) ===
    if params.get('use_error_sign', False):
        df['error_lag1_sign'] = np.sign(df['error_lag1'])
        df['error_sign_streak'] = (df['error_lag1'] * df['error_lag2'] > 0).astype(int)

    # === FORECAST FEATURES (tunable) ===
    if params.get('use_forecast_level', False):
        df['forecast_load'] = df['forecast_load_mw']
    if params.get('use_forecast_diff', False):
        df['forecast_diff_1h'] = df['forecast_load_mw'] - df['forecast_load_mw'].shift(1)
        df['forecast_diff_24h'] = df['forecast_load_mw'] - df['forecast_load_mw'].shift(24)

    # Target
    df['target_h2'] = df['error'].shift(-2)

    return df


def get_feature_list(df, params):
    """Get list of features based on params."""
    features = []

    # Error lags
    max_lag = params.get('max_error_lag', 8)
    for lag in range(1, max_lag + 1):
        if f'error_lag{lag}' in df.columns:
            features.append(f'error_lag{lag}')

    # Rolling features
    for window in [3, 6, 12, 24, 48]:
        if f'error_roll_mean_{window}h' in df.columns:
            features.append(f'error_roll_mean_{window}h')
            features.append(f'error_roll_std_{window}h')
        if f'error_roll_median_{window}h' in df.columns:
            features.append(f'error_roll_median_{window}h')

    # Trends and momentum
    for col in ['error_trend_3h', 'error_trend_6h', 'error_momentum']:
        if col in df.columns:
            features.append(col)

    # Load features
    for col in ['load_volatility_lag1', 'load_trend_lag1', 'load_range_lag1']:
        if col in df.columns:
            features.append(col)

    # Regulation features
    for lag in range(1, 6):
        if f'reg_mean_lag{lag}' in df.columns:
            features.append(f'reg_mean_lag{lag}')
    for col in ['reg_std_lag1', 'reg_range_lag1']:
        if col in df.columns:
            features.append(col)

    # Seasonal
    if 'seasonal_error' in df.columns:
        features.append('seasonal_error')

    # Time features
    for col in ['hour', 'dow', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']:
        if col in df.columns and params.get(f'use_{col}', col in ['hour', 'dow']):
            features.append(col)

    # Same-hour features
    for col in ['error_same_hour_yesterday', 'error_same_hour_2d', 'error_same_hour_week']:
        if col in df.columns:
            features.append(col)

    # Interaction and sign features
    for col in ['hour_x_error_lag1', 'error_lag1_sign', 'error_sign_streak']:
        if col in df.columns:
            features.append(col)

    # Forecast features
    for col in ['forecast_load', 'forecast_diff_1h', 'forecast_diff_24h']:
        if col in df.columns:
            features.append(col)

    return list(set(features))


def evaluate_model(params, df, verbose=False):
    """
    Evaluate model with given parameters using time-series CV.

    Training: 2024 + 2025
    Test: Jan 2026
    """
    # Training strategy params
    finetune_weight = params.get('finetune_weight', 3.0)
    finetune_days = params.get('finetune_days', 90)

    test_start = pd.Timestamp('2026-01-01')
    finetune_start = test_start - pd.Timedelta(days=finetune_days)

    # Create features with training-only seasonal error
    train_mask = df['year'].isin([2024, 2025])
    train_subset = df[train_mask].copy()

    df_feat = create_features(df.copy(), params, train_subset=train_subset)
    features = get_feature_list(df_feat, params)

    if len(features) < 5:
        return 999.0  # Invalid config

    # Filter valid data
    df_valid = df_feat.dropna(subset=['target_h2'] + features).copy()

    # Training data
    train = df_valid[df_valid['year'].isin([2024, 2025])].copy()
    if len(train) < 1000:
        return 999.0

    # Sample weights for fine-tuning
    is_recent = train['datetime'] >= finetune_start
    sample_weights = np.where(is_recent, finetune_weight, 1.0)

    # Test data
    test = df_valid[(df_valid['year'] == 2026) & (df_valid['month'] == 1)].copy()
    if len(test) < 100:
        return 999.0

    # LightGBM params
    lgb_params = {
        'n_estimators': params.get('n_estimators', 300),
        'learning_rate': params.get('learning_rate', 0.03),
        'max_depth': params.get('max_depth', 8),
        'num_leaves': params.get('num_leaves', 50),
        'min_child_samples': params.get('min_child_samples', 30),
        'subsample': params.get('subsample', 0.8),
        'colsample_bytree': params.get('colsample_bytree', 0.8),
        'reg_alpha': params.get('reg_alpha', 0.1),
        'reg_lambda': params.get('reg_lambda', 0.1),
        'random_state': 42,
        'verbosity': -1,
    }

    # Train model
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(train[features], train['target_h2'], sample_weight=sample_weights)

    # Evaluate
    pred = model.predict(test[features])
    mae = np.abs(test['target_h2'] - pred).mean()

    if verbose:
        print(f"    Features: {len(features)}, Train: {len(train)}, Test: {len(test)}, MAE: {mae:.2f}")

    return mae


def objective(trial):
    """Optuna objective function."""

    params = {
        # === LightGBM Hyperparameters ===
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),

        # === Feature Engineering ===
        'max_error_lag': trial.suggest_int('max_error_lag', 4, 24),
        'use_roll_3h': trial.suggest_categorical('use_roll_3h', [True, False]),
        'use_roll_6h': trial.suggest_categorical('use_roll_6h', [True, False]),
        'use_roll_12h': trial.suggest_categorical('use_roll_12h', [True, False]),
        'use_roll_24h': trial.suggest_categorical('use_roll_24h', [True, False]),
        'use_roll_48h': trial.suggest_categorical('use_roll_48h', [True, False]),
        'use_roll_median': trial.suggest_categorical('use_roll_median', [True, False]),

        'use_error_trends': trial.suggest_categorical('use_error_trends', [True, False]),
        'use_momentum': trial.suggest_categorical('use_momentum', [True, False]),
        'momentum_w1': trial.suggest_float('momentum_w1', 0.3, 0.7),
        'momentum_w2': trial.suggest_float('momentum_w2', 0.1, 0.4),

        'use_load_volatility': trial.suggest_categorical('use_load_volatility', [True, False]),
        'use_load_trend': trial.suggest_categorical('use_load_trend', [True, False]),
        'use_load_range': trial.suggest_categorical('use_load_range', [True, False]),

        'reg_lag_depth': trial.suggest_int('reg_lag_depth', 1, 5),
        'use_reg_std': trial.suggest_categorical('use_reg_std', [True, False]),
        'use_reg_range': trial.suggest_categorical('use_reg_range', [True, False]),

        'use_seasonal_error': trial.suggest_categorical('use_seasonal_error', [True, False]),
        'use_hour_cyclical': trial.suggest_categorical('use_hour_cyclical', [True, False]),
        'use_dow_cyclical': trial.suggest_categorical('use_dow_cyclical', [True, False]),
        'use_is_weekend': trial.suggest_categorical('use_is_weekend', [True, False]),
        'use_hour_raw': trial.suggest_categorical('use_hour_raw', [True, False]),
        'use_dow_raw': trial.suggest_categorical('use_dow_raw', [True, False]),

        'use_hour_error_interaction': trial.suggest_categorical('use_hour_error_interaction', [True, False]),
        'use_same_hour_yesterday': trial.suggest_categorical('use_same_hour_yesterday', [True, False]),
        'use_same_hour_2d': trial.suggest_categorical('use_same_hour_2d', [True, False]),
        'use_same_hour_week': trial.suggest_categorical('use_same_hour_week', [True, False]),

        'use_error_sign': trial.suggest_categorical('use_error_sign', [True, False]),
        'use_forecast_level': trial.suggest_categorical('use_forecast_level', [True, False]),
        'use_forecast_diff': trial.suggest_categorical('use_forecast_diff', [True, False]),

        # === Training Strategy ===
        'finetune_weight': trial.suggest_float('finetune_weight', 1.0, 10.0),
        'finetune_days': trial.suggest_int('finetune_days', 30, 180),
    }

    df = load_data()
    mae = evaluate_model(params, df)

    return mae


def get_baseline_params():
    """Get baseline (current) parameters."""
    return {
        # LightGBM (from two_stage_model.py)
        'n_estimators': 300,
        'learning_rate': 0.03,
        'max_depth': 8,
        'num_leaves': 50,
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,

        # Features (current defaults)
        'max_error_lag': 8,
        'use_roll_3h': True,
        'use_roll_6h': True,
        'use_roll_12h': True,
        'use_roll_24h': True,
        'use_roll_48h': False,
        'use_roll_median': False,
        'use_error_trends': True,
        'use_momentum': True,
        'momentum_w1': 0.5,
        'momentum_w2': 0.3,
        'use_load_volatility': True,
        'use_load_trend': True,
        'use_load_range': False,
        'reg_lag_depth': 3,
        'use_reg_std': True,
        'use_reg_range': False,
        'use_seasonal_error': True,
        'use_hour_cyclical': True,
        'use_dow_cyclical': False,
        'use_is_weekend': True,
        'use_hour_raw': True,
        'use_dow_raw': True,
        'use_hour_error_interaction': False,
        'use_same_hour_yesterday': False,
        'use_same_hour_2d': False,
        'use_same_hour_week': False,
        'use_error_sign': False,
        'use_forecast_level': False,
        'use_forecast_diff': False,

        # Training
        'finetune_weight': 3.0,
        'finetune_days': 90,
    }


def main():
    print("=" * 70)
    print("COMPREHENSIVE TUNING: Q0 H+2 Model")
    print("=" * 70)
    print("\nTuning 40+ parameters to see if it's worth the effort.")
    print("Test set: January 2026")

    # Load data once
    df = load_data()

    # Evaluate baseline
    print("\n" + "-" * 70)
    print("BASELINE MODEL (current parameters)")
    print("-" * 70)
    baseline_params = get_baseline_params()
    baseline_mae = evaluate_model(baseline_params, df, verbose=True)
    print(f"\n    Baseline MAE: {baseline_mae:.2f} MW")

    # Run Optuna
    print("\n" + "-" * 70)
    print("OPTUNA TUNING (200 trials)")
    print("-" * 70)

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )

    # Add baseline as first trial
    study.enqueue_trial(baseline_params)

    study.optimize(objective, n_trials=200, show_progress_bar=True)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    best_mae = study.best_value
    improvement = baseline_mae - best_mae
    pct_improvement = improvement / baseline_mae * 100

    print(f"\n    Baseline MAE:  {baseline_mae:.2f} MW")
    print(f"    Best MAE:      {best_mae:.2f} MW")
    print(f"    Improvement:   {improvement:.2f} MW ({pct_improvement:.1f}%)")

    # Best parameters
    print("\n" + "-" * 70)
    print("BEST PARAMETERS")
    print("-" * 70)

    best_params = study.best_params

    print("\n  LightGBM:")
    for key in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves',
                'min_child_samples', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
        baseline_val = baseline_params.get(key, 'N/A')
        best_val = best_params.get(key, 'N/A')
        changed = "*" if baseline_val != best_val else ""
        print(f"    {key}: {baseline_val} -> {best_val} {changed}")

    print("\n  Feature Engineering:")
    feat_keys = [k for k in best_params.keys() if k.startswith('use_') or k.startswith('max_') or
                 k.startswith('reg_lag') or k.startswith('momentum')]
    for key in sorted(feat_keys):
        baseline_val = baseline_params.get(key, 'N/A')
        best_val = best_params.get(key, 'N/A')
        changed = "*" if baseline_val != best_val else ""
        print(f"    {key}: {baseline_val} -> {best_val} {changed}")

    print("\n  Training Strategy:")
    for key in ['finetune_weight', 'finetune_days']:
        baseline_val = baseline_params.get(key, 'N/A')
        best_val = best_params.get(key, 'N/A')
        changed = "*" if baseline_val != best_val else ""
        print(f"    {key}: {baseline_val} -> {best_val} {changed}")

    # Top 10 trials
    print("\n" + "-" * 70)
    print("TOP 10 TRIALS")
    print("-" * 70)
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value').head(10)
    print(f"\n{'Rank':<6}{'MAE':<10}{'n_est':<8}{'lr':<10}{'depth':<8}{'leaves':<8}")
    print("-" * 50)
    for i, (_, row) in enumerate(trials_df.iterrows(), 1):
        print(f"{i:<6}{row['value']:<10.2f}{row['params_n_estimators']:<8}"
              f"{row['params_learning_rate']:<10.4f}{row['params_max_depth']:<8}"
              f"{row['params_num_leaves']:<8}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if pct_improvement > 5:
        print(f"\n    [+] TUNING IS WORTH IT!")
        print(f"        {pct_improvement:.1f}% improvement ({improvement:.2f} MW)")
        print(f"        Recommend tuning all 20 models")
    elif pct_improvement > 2:
        print(f"\n    [~] MODERATE BENEFIT")
        print(f"        {pct_improvement:.1f}% improvement ({improvement:.2f} MW)")
        print(f"        Consider tuning critical horizons (H+1, H+2)")
    else:
        print(f"\n    [-] LIMITED BENEFIT")
        print(f"        Only {pct_improvement:.1f}% improvement ({improvement:.2f} MW)")
        print(f"        Default parameters are already good")

    return study, baseline_mae, best_mae


if __name__ == "__main__":
    study, baseline_mae, best_mae = main()
