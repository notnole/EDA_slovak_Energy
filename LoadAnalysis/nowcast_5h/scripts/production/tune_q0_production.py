"""
Production-Ready Tuning for Q0 Models (H+1 to H+5)
===================================================

KEY INSIGHT: We want to optimize for FUTURE predictions (Feb 2026+), not past.

Setup for production:
- Hyperparameter Selection: Use walk-forward CV on historical data
- Final Training: Use ALL available data (2024 + 2025 + Jan 2026)
- Validation: Jan 2026 used ONLY for hyperparameter selection, then included in final training

Walk-Forward Cross-Validation:
  Fold 1: Train 2024-Q1-Q3, Validate 2024-Q4
  Fold 2: Train 2024, Validate 2025-Q1
  Fold 3: Train 2024 + 2025-H1, Validate 2025-Q3
  Fold 4: Train 2024 + 2025, Validate Jan 2026

This mimics production: always predicting FUTURE data you haven't seen.

Trial Count Analysis (from previous 200-trial run):
  - Trials 1-50:   52.98 -> 50.0 MW (major gains)
  - Trials 50-100: 50.0 -> 49.05 MW (good gains)
  - Trials 100-150: 49.05 -> 48.14 MW (moderate gains)
  - Trials 150-200: 48.14 -> 48.09 MW (diminishing returns)

  RECOMMENDATION: 100-150 trials per model (best ROI)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_PATH = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent.parent / 'models_tuned'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cache
_DATA_CACHE = {}


def load_data():
    """Load all data sources."""
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
        'load_mw': ['std', 'first', 'last', 'min', 'max']
    }).reset_index()
    load_hourly.columns = ['datetime', 'load_std_3min', 'load_first', 'load_last',
                           'load_min_3min', 'load_max_3min']
    load_hourly['load_trend_3min'] = load_hourly['load_last'] - load_hourly['load_first']
    load_hourly['load_range_3min'] = load_hourly['load_max_3min'] - load_hourly['load_min_3min']
    df = df.merge(load_hourly[['datetime', 'load_std_3min', 'load_trend_3min', 'load_range_3min']],
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

    _DATA_CACHE['df'] = df
    return df.copy()


def create_features(df, params, train_subset=None):
    """Create features based on tunable parameters."""
    df = df.copy()

    # Error lags
    max_lag = params.get('max_error_lag', 8)
    for lag in range(1, max_lag + 1):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    # Rolling windows
    windows = []
    if params.get('use_roll_3h', True): windows.append(3)
    if params.get('use_roll_6h', True): windows.append(6)
    if params.get('use_roll_12h', True): windows.append(12)
    if params.get('use_roll_24h', True): windows.append(24)
    if params.get('use_roll_48h', False): windows.append(48)

    for window in windows:
        df[f'error_roll_mean_{window}h'] = df['error'].shift(1).rolling(window).mean()
        df[f'error_roll_std_{window}h'] = df['error'].shift(1).rolling(window).std()
        if params.get('use_roll_median', False):
            df[f'error_roll_median_{window}h'] = df['error'].shift(1).rolling(window).median()

    # Trends
    if params.get('use_error_trends', True):
        if 'error_lag3' in df.columns:
            df['error_trend_3h'] = df['error_lag1'] - df['error_lag3']
        if 'error_lag6' in df.columns:
            df['error_trend_6h'] = df['error_lag1'] - df['error_lag6']

    # Momentum
    if params.get('use_momentum', True):
        w1 = params.get('momentum_w1', 0.5)
        w2 = params.get('momentum_w2', 0.3)
        w3 = max(0, 1.0 - w1 - w2)
        lag2 = df.get('error_lag2', df['error_lag1'])
        lag3 = df.get('error_lag3', lag2)
        lag4 = df.get('error_lag4', lag3)
        df['error_momentum'] = w1 * (df['error_lag1'] - lag2) + w2 * (lag2 - lag3) + w3 * (lag3 - lag4)

    # Load features
    if params.get('use_load_volatility', True):
        df['load_volatility_lag1'] = df['load_std_3min'].shift(1)
    if params.get('use_load_trend', True):
        df['load_trend_lag1'] = df['load_trend_3min'].shift(1)
    if params.get('use_load_range', False):
        df['load_range_lag1'] = df['load_range_3min'].shift(1)

    # Regulation
    reg_lag_depth = params.get('reg_lag_depth', 3)
    for lag in range(1, reg_lag_depth + 1):
        df[f'reg_mean_lag{lag}'] = df['reg_mean'].shift(lag)
    if params.get('use_reg_std', True):
        df['reg_std_lag1'] = df['reg_std'].shift(1)
    if params.get('use_reg_range', False):
        df['reg_range_lag1'] = df['reg_range'].shift(1)

    # Seasonal error
    if params.get('use_seasonal_error', True):
        if train_subset is not None:
            seasonal_map = train_subset.groupby(['dow', 'hour'])['error'].mean().to_dict()
            df['seasonal_error'] = df.apply(lambda r: seasonal_map.get((r['dow'], r['hour']), 0), axis=1)
        else:
            df['seasonal_error'] = df.groupby(['dow', 'hour'])['error'].transform('mean')

    # Time features
    if params.get('use_hour_cyclical', True):
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    if params.get('use_dow_cyclical', False):
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    if params.get('use_is_weekend', True):
        df['is_weekend'] = (df['dow'] >= 5).astype(int)

    # Same-hour features
    if params.get('use_same_hour_yesterday', False):
        df['error_same_hour_yesterday'] = df['error'].shift(24)
    if params.get('use_same_hour_2d', False):
        df['error_same_hour_2d'] = df['error'].shift(48)
    if params.get('use_same_hour_week', False):
        df['error_same_hour_week'] = df['error'].shift(168)

    # Error sign
    if params.get('use_error_sign', False):
        df['error_lag1_sign'] = np.sign(df['error_lag1'])
        df['error_sign_streak'] = (df['error_lag1'] * df.get('error_lag2', df['error_lag1']) > 0).astype(int)

    # Forecast features
    if params.get('use_forecast_level', False):
        df['forecast_load'] = df['forecast_load_mw']
    if params.get('use_forecast_diff', False):
        df['forecast_diff_1h'] = df['forecast_load_mw'] - df['forecast_load_mw'].shift(1)
        df['forecast_diff_24h'] = df['forecast_load_mw'] - df['forecast_load_mw'].shift(24)

    return df


def get_feature_list(df, params):
    """Get feature list based on created features."""
    features = []

    # Error lags
    for lag in range(1, params.get('max_error_lag', 8) + 1):
        if f'error_lag{lag}' in df.columns:
            features.append(f'error_lag{lag}')

    # Rolling
    for window in [3, 6, 12, 24, 48]:
        for stat in ['mean', 'std', 'median']:
            col = f'error_roll_{stat}_{window}h'
            if col in df.columns:
                features.append(col)

    # Other features
    other_cols = [
        'error_trend_3h', 'error_trend_6h', 'error_momentum',
        'load_volatility_lag1', 'load_trend_lag1', 'load_range_lag1',
        'reg_std_lag1', 'reg_range_lag1', 'seasonal_error',
        'hour', 'dow', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
        'error_same_hour_yesterday', 'error_same_hour_2d', 'error_same_hour_week',
        'error_lag1_sign', 'error_sign_streak',
        'forecast_load', 'forecast_diff_1h', 'forecast_diff_24h'
    ]
    for col in other_cols:
        if col in df.columns:
            features.append(col)

    # Regulation lags
    for lag in range(1, 6):
        if f'reg_mean_lag{lag}' in df.columns:
            features.append(f'reg_mean_lag{lag}')

    return list(set(features))


def walk_forward_cv(df, params, horizon):
    """
    Walk-forward cross-validation for hyperparameter selection.

    Folds:
      1: Train 2024-Q1-Q3, Val 2024-Q4
      2: Train 2024, Val 2025-Q1-Q2
      3: Train 2024 + 2025-H1, Val 2025-H2
      4: Train 2024 + 2025, Val Jan 2026
    """
    target = f'target_h{horizon}'

    folds = [
        # (train_end, val_start, val_end)
        ('2024-09-30', '2024-10-01', '2024-12-31'),
        ('2024-12-31', '2025-01-01', '2025-06-30'),
        ('2025-06-30', '2025-07-01', '2025-12-31'),
        ('2025-12-31', '2026-01-01', '2026-01-31'),
    ]

    fold_maes = []

    for train_end, val_start, val_end in folds:
        train_end = pd.Timestamp(train_end)
        val_start = pd.Timestamp(val_start)
        val_end = pd.Timestamp(val_end)

        # Create features with train-only seasonal error
        train_subset = df[df['datetime'] <= train_end]
        df_feat = create_features(df.copy(), params, train_subset=train_subset)

        # Add target
        df_feat[target] = df_feat['error'].shift(-horizon)

        features = get_feature_list(df_feat, params)
        if len(features) < 5:
            return 999.0

        df_valid = df_feat.dropna(subset=[target] + features)

        # Split
        train = df_valid[df_valid['datetime'] <= train_end].copy()
        val = df_valid[(df_valid['datetime'] >= val_start) & (df_valid['datetime'] <= val_end)].copy()

        if len(train) < 500 or len(val) < 100:
            continue

        # Fine-tuning weights
        finetune_days = params.get('finetune_days', 90)
        finetune_weight = params.get('finetune_weight', 3.0)
        finetune_start = train_end - pd.Timedelta(days=finetune_days)
        is_recent = train['datetime'] >= finetune_start
        sample_weights = np.where(is_recent, finetune_weight, 1.0)

        # Train
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

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(train[features], train[target], sample_weight=sample_weights)

        # Validate
        pred = model.predict(val[features])
        mae = np.abs(val[target] - pred).mean()
        fold_maes.append(mae)

    if len(fold_maes) == 0:
        return 999.0

    # Return weighted average (more weight on recent folds)
    weights = [1, 1.5, 2, 3]  # More weight on recent validation periods
    weighted_mae = sum(m * w for m, w in zip(fold_maes, weights[:len(fold_maes)])) / sum(weights[:len(fold_maes)])

    return weighted_mae


def create_objective(horizon):
    """Create Optuna objective for specific horizon."""

    def objective(trial):
        params = {
            # LightGBM
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),

            # Features
            'max_error_lag': trial.suggest_int('max_error_lag', 6, 24),
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
            'reg_lag_depth': trial.suggest_int('reg_lag_depth', 1, 4),
            'use_reg_std': trial.suggest_categorical('use_reg_std', [True, False]),
            'use_reg_range': trial.suggest_categorical('use_reg_range', [True, False]),
            'use_seasonal_error': trial.suggest_categorical('use_seasonal_error', [True, False]),
            'use_hour_cyclical': trial.suggest_categorical('use_hour_cyclical', [True, False]),
            'use_dow_cyclical': trial.suggest_categorical('use_dow_cyclical', [True, False]),
            'use_is_weekend': trial.suggest_categorical('use_is_weekend', [True, False]),
            'use_same_hour_yesterday': trial.suggest_categorical('use_same_hour_yesterday', [True, False]),
            'use_same_hour_2d': trial.suggest_categorical('use_same_hour_2d', [True, False]),
            'use_same_hour_week': trial.suggest_categorical('use_same_hour_week', [True, False]),
            'use_error_sign': trial.suggest_categorical('use_error_sign', [True, False]),
            'use_forecast_level': trial.suggest_categorical('use_forecast_level', [True, False]),
            'use_forecast_diff': trial.suggest_categorical('use_forecast_diff', [True, False]),

            # Training
            'finetune_weight': trial.suggest_float('finetune_weight', 1.0, 8.0),
            'finetune_days': trial.suggest_int('finetune_days', 45, 150),
        }

        df = load_data()
        return walk_forward_cv(df, params, horizon)

    return objective


def train_final_model(df, params, horizon):
    """
    Train final production model on ALL available data.
    """
    target = f'target_h{horizon}'

    # Use ALL data for training (2024 + 2025 + Jan 2026)
    train_subset = df.copy()
    df_feat = create_features(df.copy(), params, train_subset=train_subset)
    df_feat[target] = df_feat['error'].shift(-horizon)

    features = get_feature_list(df_feat, params)
    df_valid = df_feat.dropna(subset=[target] + features)

    # Fine-tuning on last N days
    finetune_days = params.get('finetune_days', 90)
    finetune_weight = params.get('finetune_weight', 3.0)
    max_date = df_valid['datetime'].max()
    finetune_start = max_date - pd.Timedelta(days=finetune_days)
    is_recent = df_valid['datetime'] >= finetune_start
    sample_weights = np.where(is_recent, finetune_weight, 1.0)

    # Train
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

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(df_valid[features], df_valid[target], sample_weight=sample_weights)

    return model, features


def get_baseline_cv_mae(df, horizon):
    """Get baseline CV MAE for comparison."""
    baseline_params = {
        'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8,
        'num_leaves': 50, 'min_child_samples': 30, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'max_error_lag': 8, 'use_roll_3h': True, 'use_roll_6h': True,
        'use_roll_12h': True, 'use_roll_24h': True, 'use_roll_48h': False,
        'use_roll_median': False, 'use_error_trends': True, 'use_momentum': True,
        'momentum_w1': 0.5, 'momentum_w2': 0.3, 'use_load_volatility': True,
        'use_load_trend': True, 'use_load_range': False, 'reg_lag_depth': 3,
        'use_reg_std': True, 'use_reg_range': False, 'use_seasonal_error': True,
        'use_hour_cyclical': True, 'use_dow_cyclical': False, 'use_is_weekend': True,
        'use_same_hour_yesterday': False, 'use_same_hour_2d': False,
        'use_same_hour_week': False, 'use_error_sign': False,
        'use_forecast_level': False, 'use_forecast_diff': False,
        'finetune_weight': 3.0, 'finetune_days': 90,
    }
    return walk_forward_cv(df, baseline_params, horizon)


def save_trial_history_csv(study, horizon, output_dir):
    """Save all trials to CSV for inspection."""
    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        row = {
            'trial': t.number,
            'cv_mae': t.value,
            'state': t.state.name,
            'datetime': t.datetime_complete.isoformat() if t.datetime_complete else None,
            'duration_sec': t.duration.total_seconds() if t.duration else None,
        }
        # Add all parameters
        for k, v in t.params.items():
            row[f'param_{k}'] = v
        rows.append(row)

    df_trials = pd.DataFrame(rows)

    # Add best_so_far column
    df_trials = df_trials.sort_values('trial')
    df_trials['best_so_far'] = df_trials['cv_mae'].cummin()
    df_trials['is_best'] = df_trials['cv_mae'] == df_trials['best_so_far']

    # Save
    csv_path = output_dir / f'q0_h{horizon}_trial_history.csv'
    df_trials.to_csv(csv_path, index=False)

    return df_trials


def main(n_trials=100):
    print("=" * 70)
    print("PRODUCTION-READY Q0 MODEL TUNING")
    print("=" * 70)
    print(f"\nTrials per model: {n_trials}")
    print("Validation: Walk-forward CV (4 folds)")
    print("Final training: ALL data (2024 + 2025 + Jan 2026)")

    df = load_data()

    # Initialize logging
    all_results = {
        'config': {
            'n_trials': n_trials,
            'cv_folds': 4,
            'cv_method': 'walk_forward',
            'final_train_data': '2024 + 2025 + Jan 2026',
        },
        'horizons': {}
    }

    results = {}
    best_params = {}
    final_models = {}
    all_studies = {}
    all_trial_dfs = {}

    for horizon in range(1, 6):
        print(f"\n{'='*70}")
        print(f"TUNING Q0 H+{horizon}")
        print("=" * 70)

        # Get baseline for comparison
        print("  Computing baseline CV MAE...")
        baseline_cv = get_baseline_cv_mae(df, horizon)
        print(f"  Baseline CV MAE: {baseline_cv:.2f} MW")

        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42 + horizon)
        )

        # Optimize
        objective = create_objective(horizon)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        all_studies[horizon] = study

        best_params[horizon] = study.best_params
        cv_mae = study.best_value
        improvement = baseline_cv - cv_mae
        improvement_pct = improvement / baseline_cv * 100

        print(f"\n  Best CV MAE: {cv_mae:.2f} MW")
        print(f"  Improvement: {improvement:.2f} MW ({improvement_pct:.1f}%)")

        # Save trial history CSV immediately (in case of crash)
        trial_df = save_trial_history_csv(study, horizon, OUTPUT_DIR)
        all_trial_dfs[horizon] = trial_df
        print(f"  Trial history saved: q0_h{horizon}_trial_history.csv")

        # Train final model on all data
        print(f"  Training final model on all data...")
        model, features = train_final_model(df, study.best_params, horizon)
        final_models[horizon] = {'model': model, 'features': features, 'params': study.best_params}

        # Get feature importance
        feature_importance = dict(zip(features, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

        # Save model
        joblib.dump(model, OUTPUT_DIR / f'q0_h{horizon}_tuned.joblib')

        # Extract trial history for convergence analysis
        trial_history = []
        best_so_far = float('inf')
        for t in study.trials:
            if t.value is not None:
                best_so_far = min(best_so_far, t.value)
                trial_history.append({
                    'trial': t.number,
                    'value': t.value,
                    'best_so_far': best_so_far
                })

        # Store results
        results[horizon] = {
            'baseline_cv_mae': baseline_cv,
            'tuned_cv_mae': cv_mae,
            'improvement_mw': improvement,
            'improvement_pct': improvement_pct,
            'n_features': len(features),
            'best_params': study.best_params,
        }

        # Log detailed results
        all_results['horizons'][f'h{horizon}'] = {
            'baseline_cv_mae': float(baseline_cv),
            'tuned_cv_mae': float(cv_mae),
            'improvement_mw': float(improvement),
            'improvement_pct': float(improvement_pct),
            'n_features': len(features),
            'features_used': features,
            'top_10_features': [(f, float(imp)) for f, imp in top_features],
            'best_params': {k: (int(v) if isinstance(v, np.integer) else
                               float(v) if isinstance(v, np.floating) else v)
                           for k, v in study.best_params.items()},
            'convergence': trial_history,
            'n_trials': len(study.trials),
        }

    # Save comprehensive results
    with open(OUTPUT_DIR / 'q0_tuning_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save best params (compact version)
    with open(OUTPUT_DIR / 'q0_best_params.json', 'w') as f:
        params_json = {}
        for h, p in best_params.items():
            params_json[str(h)] = {k: (int(v) if isinstance(v, np.integer) else
                                       float(v) if isinstance(v, np.floating) else v)
                                  for k, v in p.items()}
        json.dump(params_json, f, indent=2)

    # Save feature lists
    with open(OUTPUT_DIR / 'q0_feature_lists.json', 'w') as f:
        feature_lists = {str(h): final_models[h]['features'] for h in range(1, 6)}
        json.dump(feature_lists, f, indent=2)

    # Generate summary report
    report_lines = [
        "# Q0 Tuning Results",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Trials per model: {n_trials}",
        "",
        "## Summary",
        "",
        "| Horizon | Baseline | Tuned | Improvement | Features |",
        "|---------|----------|-------|-------------|----------|",
    ]

    total_improvement = 0
    for h in range(1, 6):
        r = results[h]
        report_lines.append(
            f"| H+{h} | {r['baseline_cv_mae']:.2f} MW | {r['tuned_cv_mae']:.2f} MW | "
            f"{r['improvement_mw']:.2f} MW ({r['improvement_pct']:.1f}%) | {r['n_features']} |"
        )
        total_improvement += r['improvement_mw']

    avg_improvement = total_improvement / 5
    report_lines.extend([
        "",
        f"**Average improvement: {avg_improvement:.2f} MW**",
        "",
        "## Top Features by Horizon",
        "",
    ])

    for h in range(1, 6):
        top_feats = all_results['horizons'][f'h{h}']['top_10_features'][:5]
        report_lines.append(f"### H+{h}")
        for feat, imp in top_feats:
            report_lines.append(f"- {feat}: {imp:.0f}")
        report_lines.append("")

    report_lines.extend([
        "## Key Parameter Changes vs Baseline",
        "",
    ])

    # Compare key params
    baseline_key = {'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 8,
                    'max_error_lag': 8, 'finetune_weight': 3.0, 'finetune_days': 90}

    for h in range(1, 6):
        bp = best_params[h]
        changes = []
        for k, base_v in baseline_key.items():
            if k in bp:
                new_v = bp[k]
                if isinstance(new_v, float):
                    if abs(new_v - base_v) / max(abs(base_v), 0.001) > 0.1:
                        changes.append(f"{k}: {base_v} -> {new_v:.3f}")
                elif new_v != base_v:
                    changes.append(f"{k}: {base_v} -> {new_v}")
        if changes:
            report_lines.append(f"**H+{h}:** {', '.join(changes[:3])}")

    report_lines.append("")
    report_lines.append(f"## Files Generated")
    report_lines.append(f"- `q0_h{{1-5}}_tuned.joblib` - Tuned models")
    report_lines.append(f"- `q0_tuning_results.json` - Full results with convergence")
    report_lines.append(f"- `q0_best_params.json` - Best parameters")
    report_lines.append(f"- `q0_feature_lists.json` - Feature lists per horizon")

    with open(OUTPUT_DIR / 'q0_tuning_summary.md', 'w') as f:
        f.write('\n'.join(report_lines))

    # Create combined trial history (all horizons)
    combined_trials = []
    for h, trial_df in all_trial_dfs.items():
        trial_df = trial_df.copy()
        trial_df['horizon'] = h
        combined_trials.append(trial_df)

    combined_df = pd.concat(combined_trials, ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / 'q0_all_trials_history.csv', index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  {'Horizon':<10} {'Baseline':<12} {'Tuned':<12} {'Improvement':<15} {'Features':<10}")
    print("  " + "-" * 60)
    for h in range(1, 6):
        r = results[h]
        print(f"  H+{h:<8} {r['baseline_cv_mae']:<12.2f} {r['tuned_cv_mae']:<12.2f} "
              f"{r['improvement_mw']:+.2f} MW ({r['improvement_pct']:.1f}%)  {r['n_features']:<10}")

    print(f"\n  Average improvement: {avg_improvement:.2f} MW")

    print(f"\n  Files saved to: {OUTPUT_DIR}")
    print(f"    - q0_h{{1-5}}_tuned.joblib      (models)")
    print(f"    - q0_h{{1-5}}_trial_history.csv (per-horizon trials)")
    print(f"    - q0_all_trials_history.csv    (combined trials)")
    print(f"    - q0_tuning_results.json       (full results)")
    print(f"    - q0_best_params.json          (parameters)")
    print(f"    - q0_feature_lists.json        (features)")
    print(f"    - q0_tuning_summary.md         (report)")

    return results, final_models, all_studies


if __name__ == "__main__":
    import sys
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    results, models, studies = main(n_trials=n_trials)
