"""
Optuna Hyperparameter Tuning - Stage 2 Models
==============================================
Trains on OUT-OF-SAMPLE Stage 1 residuals to avoid leakage.

For each CV fold:
1. Train Stage 1 on first half of training period
2. Get OOF predictions on second half (out-of-sample residuals)
3. Tune Stage 2 on these residuals
4. Evaluate combined model on validation period

Usage:
    python optuna_stage2.py --horizon 2 --n_trials 80
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path(__file__).parent.parent.parent.parent
TUNING_PATH = Path(__file__).parent


def load_data():
    """Load and prepare base data."""
    print("Loading data...")

    df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']

    # 3-minute load features
    load_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'load_3min.csv')
    load_3min['datetime'] = pd.to_datetime(load_3min['datetime'])
    load_3min['hour_start'] = load_3min['datetime'].dt.floor('h')
    load_hourly = load_3min.groupby('hour_start').agg({
        'load_mw': ['std', 'first', 'last']
    }).reset_index()
    load_hourly.columns = ['datetime', 'load_std_3min', 'load_first', 'load_last']
    load_hourly['load_trend_3min'] = load_hourly['load_last'] - load_hourly['load_first']
    df = df.merge(load_hourly[['datetime', 'load_std_3min', 'load_trend_3min']],
                  on='datetime', how='left')

    # Regulation features
    reg_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'regulation_3min.csv')
    reg_3min['datetime'] = pd.to_datetime(reg_3min['datetime'])
    reg_3min['hour_start'] = reg_3min['datetime'].dt.floor('h')
    reg_hourly = reg_3min.groupby('hour_start').agg({
        'regulation_mw': ['mean', 'std']
    }).reset_index()
    reg_hourly.columns = ['datetime', 'reg_mean', 'reg_std']
    df = df.merge(reg_hourly, on='datetime', how='left')

    print(f"  Loaded {len(df):,} rows")
    return df


def create_stage1_features(df, train_mask, horizon, feature_list):
    """Create Stage 1 features based on the tuned feature list."""
    df = df.copy()

    # Error lags (1-8)
    for lag in range(1, 9):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    # Rolling statistics
    for window in [3, 6, 12, 24]:
        df[f'error_roll_mean_{window}h'] = df['error'].shift(1).rolling(window).mean()
        df[f'error_roll_std_{window}h'] = df['error'].shift(1).rolling(window).std()

    # Error trends
    df['error_trend_3h'] = df['error_lag1'] - df['error_lag3']
    df['error_trend_6h'] = df['error_lag1'] - df['error_lag6']
    df['error_momentum'] = (0.5 * (df['error_lag1'] - df['error_lag2']) +
                            0.3 * (df['error_lag2'] - df['error_lag3']) +
                            0.2 * (df['error_lag3'] - df['error_lag4']))

    # 3-min features
    df['load_volatility_lag1'] = df['load_std_3min'].shift(1)
    df['load_trend_lag1'] = df['load_trend_3min'].shift(1)

    # Regulation lags
    for lag in range(1, 4):
        df[f'reg_mean_lag{lag}'] = df['reg_mean'].shift(lag)
    df['reg_std_lag1'] = df['reg_std'].shift(1)

    # Seasonal error (from training data only)
    train_data = df[train_mask]
    seasonal_means = train_data.groupby(['dow', 'hour'])['error'].mean()
    df['seasonal_error'] = df.set_index(['dow', 'hour']).index.map(
        lambda x: seasonal_means.get(x, 0)
    ).values

    # Time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)

    # Target
    df[f'target_h{horizon}'] = df['error'].shift(-horizon)

    return df


def create_residual_features(df, horizon):
    """
    Create features from Stage 1 residuals.
    CRITICAL: Shift by horizon to avoid data leakage!
    """
    df = df.copy()

    # Residual lags (shifted by horizon)
    for lag in range(1, 7):
        df[f'residual_lag{lag}'] = df['residual'].shift(horizon + lag - 1)

    # Rolling stats on residuals
    for window in [3, 6, 12]:
        df[f'residual_roll_mean_{window}h'] = df['residual'].shift(horizon).rolling(window).mean()
        df[f'residual_roll_std_{window}h'] = df['residual'].shift(horizon).rolling(window).std()

    # Residual trends
    df['residual_trend_3h'] = df['residual_lag1'] - df['residual_lag3']
    df['residual_trend_6h'] = df['residual_lag1'] - df['residual_lag6']

    return df


def get_stage2_feature_selection(trial):
    """Get Stage 2 feature selection from trial."""
    features = []

    # Residual lags
    if trial.suggest_categorical('s2_include_residual_lags', [True, False]):
        max_lag = trial.suggest_int('s2_max_residual_lag', 2, 6)
        features.extend([f'residual_lag{i}' for i in range(1, max_lag + 1)])

    # Rolling stats on residuals
    if trial.suggest_categorical('s2_include_residual_rolling', [True, False]):
        windows = trial.suggest_categorical('s2_residual_windows', ['3_6', '3_6_12', '6_12'])
        window_list = [int(w) for w in windows.split('_')]

        if trial.suggest_categorical('s2_include_residual_roll_mean', [True, False]):
            features.extend([f'residual_roll_mean_{w}h' for w in window_list])
        if trial.suggest_categorical('s2_include_residual_roll_std', [True, False]):
            features.extend([f'residual_roll_std_{w}h' for w in window_list])

    # Residual trends
    if trial.suggest_categorical('s2_include_residual_trends', [True, False]):
        features.append('residual_trend_3h')
        if trial.suggest_categorical('s2_include_trend_6h', [True, False]):
            features.append('residual_trend_6h')

    return features


def get_stage2_lgb_params(trial):
    """Get Stage 2 LightGBM params (simpler model than Stage 1)."""
    return {
        'n_estimators': trial.suggest_int('s2_n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('s2_learning_rate', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('s2_max_depth', 3, 8),
        'num_leaves': trial.suggest_int('s2_num_leaves', 10, 50),
        'min_child_samples': trial.suggest_int('s2_min_child_samples', 5, 30),
        'subsample': trial.suggest_float('s2_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('s2_colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('s2_reg_alpha', 1e-3, 1.0, log=True),
        'reg_lambda': trial.suggest_float('s2_reg_lambda', 1e-3, 1.0, log=True),
        'random_state': 42,
        'verbosity': -1,
    }


def get_sample_weights(dates, train_end_date, recency_months, recency_weight):
    """Compute sample weights with recency bias."""
    days_ago = (train_end_date - dates).dt.days
    months_ago = days_ago / 30.0
    weights = np.where(months_ago <= recency_months, recency_weight, 1.0)
    return weights / weights.mean()


def define_cv_folds():
    """Same CV folds as Stage 1."""
    return [
        ('2024-01-01', '2025-01-01', '2025-01-01', '2025-07-01'),
        ('2024-01-01', '2025-07-01', '2025-07-01', '2025-10-01'),
        ('2024-01-01', '2025-10-01', '2025-10-01', '2026-01-01'),
    ]


def evaluate_stage2_fold(df, fold_dates, s1_features, s1_params, s2_features, s2_lgb_params,
                         recency_months, recency_weight, horizon):
    """
    Evaluate Stage 2 on a single CV fold.

    Key: Stage 1 trained on first half of training period,
         Stage 2 trained on second half (out-of-sample residuals).
    """
    train_start, train_end, val_start, val_end = fold_dates
    target = f'target_h{horizon}'

    # Parse dates
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)

    # Split training period: S1 trains on first half, S2 trains on second half
    train_mid_dt = train_start_dt + (train_end_dt - train_start_dt) / 2
    train_mid = train_mid_dt.strftime('%Y-%m-%d')

    # Create Stage 1 features (seasonal_error from S1 training period only)
    s1_train_mask = (df['datetime'] >= train_start) & (df['datetime'] < train_mid)
    df_fold = create_stage1_features(df.copy(), s1_train_mask, horizon, s1_features)

    # Filter available features
    s1_avail = [f for f in s1_features if f in df_fold.columns]
    df_fold = df_fold.dropna(subset=[target] + s1_avail)

    # Get data splits
    s1_train = df_fold[(df_fold['datetime'] >= train_start) & (df_fold['datetime'] < train_mid)]
    s2_train_period = df_fold[(df_fold['datetime'] >= train_mid) & (df_fold['datetime'] < train_end)]
    val_data = df_fold[(df_fold['datetime'] >= val_start) & (df_fold['datetime'] < val_end)]

    if len(s1_train) < 100 or len(s2_train_period) < 100 or len(val_data) < 50:
        return {'mae': 999999, 'stage1_mae': 999999}

    # Extract Stage 1 LightGBM params
    s1_lgb_params = {k: v for k, v in s1_params.items()
                     if k in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves',
                              'min_child_samples', 'subsample', 'colsample_bytree',
                              'reg_alpha', 'reg_lambda']}
    s1_lgb_params['random_state'] = 42
    s1_lgb_params['verbosity'] = -1

    # Sample weights for Stage 1
    s1_weights = get_sample_weights(s1_train['datetime'], train_mid_dt,
                                     s1_params.get('recency_months', 3),
                                     s1_params.get('recency_weight', 2.0))

    # Train Stage 1
    model_s1 = lgb.LGBMRegressor(**s1_lgb_params)
    model_s1.fit(s1_train[s1_avail], s1_train[target], sample_weight=s1_weights)

    # Get Stage 1 predictions on S2 training period and validation
    s2_train_period = s2_train_period.copy()
    s2_train_period['s1_pred'] = model_s1.predict(s2_train_period[s1_avail])
    s2_train_period['residual'] = s2_train_period[target] - s2_train_period['s1_pred']

    val_data = val_data.copy()
    val_data['s1_pred'] = model_s1.predict(val_data[s1_avail])
    val_data['residual'] = val_data[target] - val_data['s1_pred']

    # Stage 1 MAE on validation
    stage1_mae = np.abs(val_data['residual']).mean()

    # Create residual features for Stage 2
    # Combine S2 train and val for feature creation, then split
    combined = pd.concat([s2_train_period, val_data]).sort_values('datetime')
    combined = create_residual_features(combined, horizon)

    # Split back
    s2_train_data = combined[combined['datetime'] < val_start].dropna(subset=s2_features)
    s2_val_data = combined[combined['datetime'] >= val_start].dropna(subset=s2_features)

    if len(s2_train_data) < 50 or len(s2_val_data) < 30:
        return {'mae': stage1_mae, 'stage1_mae': stage1_mae}

    # Sample weights for Stage 2
    s2_weights = get_sample_weights(s2_train_data['datetime'], pd.to_datetime(train_end),
                                     recency_months, recency_weight)

    # Train Stage 2
    s2_avail = [f for f in s2_features if f in s2_train_data.columns]
    if len(s2_avail) == 0:
        return {'mae': stage1_mae, 'stage1_mae': stage1_mae}

    model_s2 = lgb.LGBMRegressor(**s2_lgb_params)
    model_s2.fit(s2_train_data[s2_avail], s2_train_data['residual'], sample_weight=s2_weights)

    # Final predictions
    s2_val_data = s2_val_data.copy()
    s2_val_data['s2_pred'] = model_s2.predict(s2_val_data[s2_avail])
    s2_val_data['final_pred'] = s2_val_data['s1_pred'] + s2_val_data['s2_pred']
    s2_val_data['final_error'] = s2_val_data[target] - s2_val_data['final_pred']

    stage2_mae = np.abs(s2_val_data['final_error']).mean()

    return {
        'mae': stage2_mae,
        'stage1_mae': stage1_mae,
        'n_s1_train': len(s1_train),
        'n_s2_train': len(s2_train_data),
        'n_val': len(s2_val_data),
    }


class Stage2Objective:
    """Optuna objective for Stage 2 tuning."""

    def __init__(self, df, horizon, s1_params, s1_features):
        self.df = df
        self.horizon = horizon
        self.s1_params = s1_params
        self.s1_features = s1_features
        self.cv_folds = define_cv_folds()

    def __call__(self, trial):
        # Get Stage 2 feature selection
        s2_features = get_stage2_feature_selection(trial)
        if len(s2_features) < 2:
            return 999999

        # Get Stage 2 LightGBM params
        s2_lgb_params = get_stage2_lgb_params(trial)

        # Sample weighting for Stage 2
        recency_months = trial.suggest_int('s2_recency_months', 1, 4)
        recency_weight = trial.suggest_float('s2_recency_weight', 1.0, 4.0)

        # Evaluate on all folds
        fold_maes = []
        fold_s1_maes = []

        for fold_idx, fold_dates in enumerate(self.cv_folds):
            result = evaluate_stage2_fold(
                self.df, fold_dates, self.s1_features, self.s1_params,
                s2_features, s2_lgb_params, recency_months, recency_weight,
                self.horizon
            )
            fold_maes.append(result['mae'])
            fold_s1_maes.append(result['stage1_mae'])

            # Pruning
            trial.report(np.mean(fold_maes), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_mae = np.mean(fold_maes)
        mean_s1_mae = np.mean(fold_s1_maes)

        trial.set_user_attr('fold_maes', fold_maes)
        trial.set_user_attr('fold_s1_maes', fold_s1_maes)
        trial.set_user_attr('s2_features', s2_features)
        trial.set_user_attr('improvement', (mean_s1_mae - mean_mae) / mean_s1_mae * 100)

        return mean_mae


def run_stage2_tuning(horizon, n_trials, output_dir):
    """Run Stage 2 hyperparameter tuning."""

    print("=" * 70)
    print(f"STAGE 2 OPTUNA TUNING - H+{horizon}")
    print("=" * 70)

    # Load Stage 1 best params
    s1_params_path = output_dir / 'stage1_best_params.json'
    if not s1_params_path.exists():
        print(f"ERROR: Stage 1 params not found at {s1_params_path}")
        print("Run Stage 1 tuning first!")
        return None, None

    with open(s1_params_path) as f:
        s1_best = json.load(f)

    s1_features = s1_best['features']
    s1_params = s1_best['params']

    print(f"\nLoaded Stage 1 params (MAE: {s1_best['mae']:.2f} MW)")
    print(f"Stage 1 features: {len(s1_features)}")

    # Load data
    df = load_data()

    # Create study
    study_path = output_dir / f'stage2_study.db'
    study = optuna.create_study(
        study_name=f'stage2_h{horizon}',
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=10, n_startup_trials=5),
        storage=f'sqlite:///{study_path}',
        load_if_exists=True,
    )

    # Create objective
    objective = Stage2Objective(df, horizon, s1_params, s1_features)

    # Run optimization
    print(f"\nRunning {n_trials} trials...")
    start_time = datetime.now()

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    best = study.best_trial
    print(f"\nBest trial: #{best.number}")
    print(f"Best Stage 2 MAE: {best.value:.2f} MW")
    print(f"Stage 1 MAE: {np.mean(best.user_attrs.get('fold_s1_maes', [])):.2f} MW")
    print(f"Improvement: {best.user_attrs.get('improvement', 0):.1f}%")
    print(f"Fold MAEs: {best.user_attrs.get('fold_maes', [])}")
    print(f"S2 Features: {best.user_attrs.get('s2_features', [])}")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/n_trials:.1f}s per trial)")

    # Save best params
    best_params = {
        'horizon': horizon,
        'stage2_mae': best.value,
        'stage1_mae': np.mean(best.user_attrs.get('fold_s1_maes', [])),
        'improvement': best.user_attrs.get('improvement', 0),
        'fold_maes': best.user_attrs.get('fold_maes', []),
        's2_features': best.user_attrs.get('s2_features', []),
        's2_params': best.params,
        'trial_number': best.number,
        'timestamp': datetime.now().isoformat(),
    }

    params_path = output_dir / 'stage2_best_params.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to: {params_path}")

    return study, best_params


def main():
    parser = argparse.ArgumentParser(description='Stage 2 Optuna Tuning')
    parser.add_argument('--horizon', type=int, default=2, help='Forecast horizon (1-5)')
    parser.add_argument('--n_trials', type=int, default=80, help='Number of Optuna trials')
    args = parser.parse_args()

    output_dir = TUNING_PATH / f'h{args.horizon}'

    study, best_params = run_stage2_tuning(args.horizon, args.n_trials, output_dir)

    if best_params:
        print("\n" + "=" * 70)
        print("DONE")
        print("=" * 70)
        print(f"\nFinal results for H+{args.horizon}:")
        print(f"  Stage 1 MAE: {best_params['stage1_mae']:.2f} MW")
        print(f"  Stage 2 MAE: {best_params['stage2_mae']:.2f} MW")
        print(f"  Improvement: {best_params['improvement']:.1f}%")


if __name__ == '__main__':
    main()
