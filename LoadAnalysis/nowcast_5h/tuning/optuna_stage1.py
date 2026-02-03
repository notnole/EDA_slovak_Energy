"""
Optuna Hyperparameter Tuning - Stage 1 Models
==============================================
Walk-forward CV with 4 folds, tuning:
- Feature selection (groups + max-lag)
- LightGBM hyperparameters
- Sample weighting (recency)

Usage:
    python optuna_stage1.py --horizon 2 --n_trials 160
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
BASE_PATH = Path(__file__).parent.parent.parent.parent  # ipesoft_eda_data
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

    print(f"  Loaded {len(df):,} rows, {df['datetime'].min()} to {df['datetime'].max()}")
    return df


def create_all_features(df, train_mask, horizon):
    """
    Create all possible features. Selection happens later.

    Args:
        df: Base dataframe
        train_mask: Boolean mask for training data (for seasonal_error computation)
        horizon: Prediction horizon (1-5)
    """
    df = df.copy()

    # Error lags (1-8)
    for lag in range(1, 9):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    # Rolling statistics (all windows)
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

    # Seasonal error (computed ONLY from training data)
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


def get_selected_features(trial):
    """
    Get feature list based on trial parameters.
    Returns list of feature names to use.
    """
    features = []

    # Error lags (controlled by max_error_lag)
    if trial.suggest_categorical('include_error_lags', [True, False]):
        max_lag = trial.suggest_int('max_error_lag', 2, 8)
        features.extend([f'error_lag{i}' for i in range(1, max_lag + 1)])

    # Rolling windows (categorical selection)
    if trial.suggest_categorical('include_rolling_stats', [True, False]):
        rolling_config = trial.suggest_categorical('rolling_windows',
            ['3_6', '3_6_12', '3_6_12_24', '6_12_24'])
        windows = [int(w) for w in rolling_config.split('_')]

        include_means = trial.suggest_categorical('include_rolling_means', [True, False])
        include_stds = trial.suggest_categorical('include_rolling_stds', [True, False])

        if include_means:
            features.extend([f'error_roll_mean_{w}h' for w in windows])
        if include_stds:
            features.extend([f'error_roll_std_{w}h' for w in windows])

    # Error trends
    if trial.suggest_categorical('include_error_trends', [True, False]):
        features.extend(['error_trend_3h', 'error_trend_6h'])
        if trial.suggest_categorical('include_momentum', [True, False]):
            features.append('error_momentum')

    # 3-min features
    if trial.suggest_categorical('include_3min_features', [True, False]):
        features.extend(['load_volatility_lag1', 'load_trend_lag1'])

    # Regulation features
    if trial.suggest_categorical('include_regulation', [True, False]):
        max_reg_lag = trial.suggest_int('max_reg_lag', 1, 3)
        features.extend([f'reg_mean_lag{i}' for i in range(1, max_reg_lag + 1)])
        features.append('reg_std_lag1')

    # Seasonal error (individual toggle - important to test)
    if trial.suggest_categorical('include_seasonal_error', [True, False]):
        features.append('seasonal_error')

    # Time features
    if trial.suggest_categorical('include_time_features', [True, False]):
        features.extend(['hour', 'hour_sin', 'hour_cos', 'dow', 'is_weekend'])

    return features


def get_lgb_params(trial):
    """Get LightGBM hyperparameters from trial."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
        'random_state': 42,
        'verbosity': -1,
    }


def get_sample_weights(dates, train_end_date, recency_months, recency_weight):
    """
    Compute sample weights with recency bias.

    Args:
        dates: Series of datetime values
        train_end_date: End of training period
        recency_months: How many recent months get higher weight
        recency_weight: Weight multiplier for recent data

    Returns:
        Normalized weight array
    """
    days_ago = (train_end_date - dates).dt.days
    months_ago = days_ago / 30.0
    weights = np.where(months_ago <= recency_months, recency_weight, 1.0)
    return weights / weights.mean()  # Normalize to mean=1


def define_cv_folds():
    """
    Define 3-fold walk-forward CV structure.
    Minimum 12 months training data to capture full seasonality.

    Returns list of (train_start, train_end, val_start, val_end) tuples.
    Dates are inclusive start, exclusive end.
    """
    return [
        # Fold 1: Train full 2024 (12 months), Val H1 2025
        ('2024-01-01', '2025-01-01', '2025-01-01', '2025-07-01'),
        # Fold 2: Train 2024 + H1 2025 (18 months), Val Q3 2025
        ('2024-01-01', '2025-07-01', '2025-07-01', '2025-10-01'),
        # Fold 3: Train 2024 + 3Q 2025 (21 months), Val Q4 2025
        ('2024-01-01', '2025-10-01', '2025-10-01', '2026-01-01'),
    ]


def evaluate_fold(df, fold_idx, fold_dates, features, lgb_params,
                  recency_months, recency_weight, horizon):
    """
    Evaluate model on a single CV fold.

    Returns:
        dict with MAE, predictions, and other metrics
    """
    train_start, train_end, val_start, val_end = fold_dates
    target = f'target_h{horizon}'

    # Create train mask on original df for seasonal_error computation
    train_mask_orig = (df['datetime'] >= train_start) & (df['datetime'] < train_end)

    # Create features with seasonal_error from training data only
    df_fold = create_all_features(df.copy(), train_mask_orig, horizon)

    # Filter to rows with all features available
    available_features = [f for f in features if f in df_fold.columns]
    if len(available_features) == 0:
        return {'mae': 999999, 'n_train': 0, 'n_val': 0}

    df_fold = df_fold.dropna(subset=[target] + available_features)

    # Create masks on the processed df_fold (after dropna)
    train_data = df_fold[(df_fold['datetime'] >= train_start) & (df_fold['datetime'] < train_end)]
    val_data = df_fold[(df_fold['datetime'] >= val_start) & (df_fold['datetime'] < val_end)]

    if len(train_data) < 100 or len(val_data) < 50:
        return {'mae': 999999, 'n_train': len(train_data), 'n_val': len(val_data)}

    # Sample weights
    train_end_date = pd.to_datetime(train_end)
    weights = get_sample_weights(train_data['datetime'], train_end_date,
                                  recency_months, recency_weight)

    # Train model
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(train_data[available_features], train_data[target], sample_weight=weights)

    # Predict on validation
    val_preds = model.predict(val_data[available_features])
    mae = np.abs(val_data[target].values - val_preds).mean()

    return {
        'mae': mae,
        'n_train': len(train_data),
        'n_val': len(val_data),
        'predictions': val_preds,
        'actuals': val_data[target].values,
        'datetimes': val_data['datetime'].values,
    }


class Stage1Objective:
    """Optuna objective for Stage 1 tuning."""

    def __init__(self, df, horizon):
        self.df = df
        self.horizon = horizon
        self.cv_folds = define_cv_folds()
        self.best_trial_results = None

    def __call__(self, trial):
        # Get feature selection
        features = get_selected_features(trial)
        if len(features) < 3:
            return 999999  # Too few features

        # Get LightGBM params
        lgb_params = get_lgb_params(trial)

        # Get sample weighting params
        recency_months = trial.suggest_int('recency_months', 1, 6)
        recency_weight = trial.suggest_float('recency_weight', 1.0, 5.0)

        # Evaluate on all folds
        fold_maes = []
        fold_results = []

        for fold_idx, fold_dates in enumerate(self.cv_folds):
            result = evaluate_fold(
                self.df, fold_idx, fold_dates, features, lgb_params,
                recency_months, recency_weight, self.horizon
            )
            fold_maes.append(result['mae'])
            fold_results.append(result)

            # Report intermediate value for pruning
            trial.report(np.mean(fold_maes), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_mae = np.mean(fold_maes)

        # Store fold results in trial
        trial.set_user_attr('fold_maes', fold_maes)
        trial.set_user_attr('features', features)
        trial.set_user_attr('n_features', len(features))

        return mean_mae


def run_tuning(horizon, n_trials, output_dir):
    """Run Stage 1 hyperparameter tuning."""

    print("=" * 70)
    print(f"STAGE 1 OPTUNA TUNING - H+{horizon}")
    print("=" * 70)

    # Load data
    df = load_data()

    # Create study
    study_path = output_dir / f'stage1_study.db'
    study = optuna.create_study(
        study_name=f'stage1_h{horizon}',
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=20, n_startup_trials=10),
        storage=f'sqlite:///{study_path}',
        load_if_exists=True,
    )

    # Create objective
    objective = Stage1Objective(df, horizon)

    # Run optimization
    print(f"\nRunning {n_trials} trials...")
    print(f"Study storage: {study_path}")
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
    print(f"Best MAE: {best.value:.2f} MW")
    print(f"Fold MAEs: {best.user_attrs.get('fold_maes', [])}")
    print(f"Features ({best.user_attrs.get('n_features', 0)}): {best.user_attrs.get('features', [])}")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/n_trials:.1f}s per trial)")

    # Save best params
    best_params = {
        'horizon': horizon,
        'mae': best.value,
        'fold_maes': best.user_attrs.get('fold_maes', []),
        'features': best.user_attrs.get('features', []),
        'params': best.params,
        'trial_number': best.number,
        'timestamp': datetime.now().isoformat(),
    }

    params_path = output_dir / 'stage1_best_params.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to: {params_path}")

    # Print top 5 trials
    print("\n" + "-" * 70)
    print("TOP 5 TRIALS")
    print("-" * 70)

    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value').head(5)
    for _, row in trials_df.iterrows():
        print(f"  Trial {int(row['number']):3d}: MAE={row['value']:.2f} MW")

    return study, best_params


def main():
    parser = argparse.ArgumentParser(description='Stage 1 Optuna Tuning')
    parser.add_argument('--horizon', type=int, default=2, help='Forecast horizon (1-5)')
    parser.add_argument('--n_trials', type=int, default=160, help='Number of Optuna trials')
    args = parser.parse_args()

    output_dir = TUNING_PATH / f'h{args.horizon}'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'fold_predictions').mkdir(exist_ok=True)

    study, best_params = run_tuning(args.horizon, args.n_trials, output_dir)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Review results in {output_dir}")
    print(f"  2. Run Stage 2 tuning with: python optuna_stage2.py --horizon {args.horizon}")


if __name__ == '__main__':
    main()
