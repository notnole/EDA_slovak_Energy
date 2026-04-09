"""
Generate Walk-Forward Out-of-Sample Load Nowcast Predictions
============================================================

Uses the same two-stage model framework as evaluate_holdout.py, but walks
forward through time so that every prediction is genuinely out-of-sample:
the model that produces each prediction was trained only on strictly prior data.

Folds (expanding window, matching optuna CV structure):
  Fold 1: Train [2024-01 to 2025-01], Predict [2025-01 to 2025-07]
  Fold 2: Train [2024-01 to 2025-07], Predict [2025-07 to 2025-10]
  Fold 3: Train [2024-01 to 2025-10], Predict [2025-10 to 2026-01]
  Fold 4: Train [2024-01 to 2026-01], Predict [2026-01 to 2026-02]

For each fold, Stage 1 trains on the first portion of the training window,
Stage 2 trains on OOS residuals from the second portion.

Output: h{horizon}_oos_predictions.csv with columns:
  datetime, actual_error, predicted_error
  (matching the format expected by the imbalance predictor)

Only generates H+2 by default (the horizon needed for the imbalance model).
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

BASE_PATH = Path(__file__).parent.parent.parent.parent
TUNING_PATH = Path(__file__).parent
OUTPUT_PATH = TUNING_PATH / 'oos_predictions'
OUTPUT_PATH.mkdir(exist_ok=True)

# Walk-forward folds: (train_end, predict_start, predict_end)
# Training always starts at data start (2024-01)
FOLDS = [
    ('2025-01-01', '2025-01-01', '2025-07-01'),
    ('2025-07-01', '2025-07-01', '2025-10-01'),
    ('2025-10-01', '2025-10-01', '2026-01-01'),
    ('2026-01-01', '2026-01-01', '2026-02-01'),
    ('2026-02-01', '2026-02-01', '2026-03-01'),
    ('2026-03-01', '2026-03-01', '2026-04-01'),
    ('2026-04-01', '2026-04-01', '2026-05-01'),
]


def load_data():
    """Load and prepare base data (same as evaluate_holdout)."""
    print("[*] Loading data...")
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

    print(f"[+] Loaded {len(df):,} rows, {df['datetime'].min()} to {df['datetime'].max()}")
    return df


def create_all_features(df, train_mask, horizon):
    """Create all features for Stage 1 (same as evaluate_holdout)."""
    df = df.copy()

    for lag in range(1, 9):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    for window in [3, 6, 12, 24]:
        df[f'error_roll_mean_{window}h'] = df['error'].shift(1).rolling(window).mean()
        df[f'error_roll_std_{window}h'] = df['error'].shift(1).rolling(window).std()

    df['error_trend_3h'] = df['error_lag1'] - df['error_lag3']
    df['error_trend_6h'] = df['error_lag1'] - df['error_lag6']
    df['error_momentum'] = (0.5 * (df['error_lag1'] - df['error_lag2']) +
                            0.3 * (df['error_lag2'] - df['error_lag3']) +
                            0.2 * (df['error_lag3'] - df['error_lag4']))

    df['load_volatility_lag1'] = df['load_std_3min'].shift(1)
    df['load_trend_lag1'] = df['load_trend_3min'].shift(1)

    for lag in range(1, 4):
        df[f'reg_mean_lag{lag}'] = df['reg_mean'].shift(lag)
    df['reg_std_lag1'] = df['reg_std'].shift(1)

    # Seasonal error from training data only
    train_data = df[train_mask]
    seasonal_means = train_data.groupby(['dow', 'hour'])['error'].mean()
    df['seasonal_error'] = df.set_index(['dow', 'hour']).index.map(
        lambda x: seasonal_means.get(x, 0)
    ).values

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)

    df[f'target_h{horizon}'] = df['error'].shift(-horizon)
    return df


def create_residual_features(df, horizon):
    """Create Stage 2 residual features (same as evaluate_holdout)."""
    df = df.copy()
    for lag in range(1, 7):
        df[f'residual_lag{lag}'] = df['residual'].shift(horizon + lag - 1)
    for window in [3, 6, 12]:
        df[f'residual_roll_mean_{window}h'] = df['residual'].shift(horizon).rolling(window).mean()
        df[f'residual_roll_std_{window}h'] = df['residual'].shift(horizon).rolling(window).std()
    df['residual_trend_3h'] = df['residual_lag1'] - df['residual_lag3']
    df['residual_trend_6h'] = df['residual_lag1'] - df['residual_lag6']
    return df


def get_sample_weights(dates, train_end_date, recency_months, recency_weight):
    """Compute sample weights with recency bias."""
    days_ago = (train_end_date - dates).dt.days
    months_ago = days_ago / 30.0
    weights = np.where(months_ago <= recency_months, recency_weight, 1.0)
    return weights / weights.mean()


def predict_fold(df, horizon, train_end, predict_start, predict_end,
                 s1_features, s1_params, s2_features, s2_params):
    """Train on data before train_end, predict [predict_start, predict_end)."""
    target = f'target_h{horizon}'

    # Stage 1 trains on first portion, Stage 2 on OOS residuals from second
    train_start_dt = pd.to_datetime(df['datetime'].min())
    train_end_dt = pd.to_datetime(train_end)
    s1_train_end_dt = train_start_dt + (train_end_dt - train_start_dt) * 2 // 3
    s1_train_end = s1_train_end_dt.strftime('%Y-%m-%d')
    s2_train_start = s1_train_end

    # Create features using only training data for seasonal means
    train_mask = df['datetime'] < train_end
    df_feat = create_all_features(df.copy(), train_mask, horizon)
    df_feat = df_feat.dropna(subset=[target])

    # --- Stage 1 ---
    s1_train = df_feat[df_feat['datetime'] < s1_train_end]
    s1_avail = [f for f in s1_features if f in s1_train.columns]

    s1_lgb_params = {k: v for k, v in s1_params.items()
                     if k in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves',
                              'min_child_samples', 'subsample', 'colsample_bytree',
                              'reg_alpha', 'reg_lambda']}
    s1_lgb_params['random_state'] = 42
    s1_lgb_params['verbosity'] = -1

    s1_weights = get_sample_weights(
        s1_train['datetime'], pd.to_datetime(s1_train_end),
        s1_params.get('recency_months', 3), s1_params.get('recency_weight', 2.0))

    s1_train_clean = s1_train.dropna(subset=s1_avail)
    model_s1 = lgb.LGBMRegressor(**s1_lgb_params)
    model_s1.fit(s1_train_clean[s1_avail], s1_train_clean[target], sample_weight=
                 get_sample_weights(s1_train_clean['datetime'], pd.to_datetime(s1_train_end),
                                    s1_params.get('recency_months', 3), s1_params.get('recency_weight', 2.0)))

    # --- Stage 2: Get OOS residuals ---
    s2_train_period = df_feat[(df_feat['datetime'] >= s2_train_start) &
                               (df_feat['datetime'] < train_end)].copy()
    if len(s2_train_period) == 0:
        # Fall back to S1-only predictions
        test_data = df_feat[(df_feat['datetime'] >= predict_start) &
                             (df_feat['datetime'] < predict_end)].copy()
        test_clean = test_data.dropna(subset=s1_avail)
        test_clean['predicted_error'] = model_s1.predict(test_clean[s1_avail])
        return test_clean[['datetime', target, 'predicted_error']].rename(
            columns={target: 'actual_error'})

    s2_train_period['s1_pred'] = model_s1.predict(s2_train_period[s1_avail].fillna(0))
    s2_train_period['residual'] = s2_train_period[target] - s2_train_period['s1_pred']

    s2_train_period = create_residual_features(s2_train_period, horizon)
    s2_avail = [f for f in s2_features if f in s2_train_period.columns]
    s2_train_data = s2_train_period.dropna(subset=s2_avail)

    if len(s2_train_data) < 50:
        # Not enough data for Stage 2, use Stage 1 only
        test_data = df_feat[(df_feat['datetime'] >= predict_start) &
                             (df_feat['datetime'] < predict_end)].copy()
        test_clean = test_data.dropna(subset=s1_avail)
        test_clean['predicted_error'] = model_s1.predict(test_clean[s1_avail])
        return test_clean[['datetime', target, 'predicted_error']].rename(
            columns={target: 'actual_error'})

    s2_lgb_params = {
        'n_estimators': s2_params.get('s2_n_estimators', 200),
        'learning_rate': s2_params.get('s2_learning_rate', 0.05),
        'max_depth': s2_params.get('s2_max_depth', 5),
        'num_leaves': s2_params.get('s2_num_leaves', 31),
        'min_child_samples': s2_params.get('s2_min_child_samples', 30),
        'subsample': s2_params.get('s2_subsample', 0.8),
        'colsample_bytree': s2_params.get('s2_colsample_bytree', 0.8),
        'reg_alpha': s2_params.get('s2_reg_alpha', 0.1),
        'reg_lambda': s2_params.get('s2_reg_lambda', 1.0),
        'random_state': 42,
        'verbosity': -1,
    }

    s2_weights = get_sample_weights(
        s2_train_data['datetime'], pd.to_datetime(train_end),
        s2_params.get('s2_recency_months', 2), s2_params.get('s2_recency_weight', 1.5))

    model_s2 = lgb.LGBMRegressor(**s2_lgb_params)
    model_s2.fit(s2_train_data[s2_avail], s2_train_data['residual'], sample_weight=s2_weights)

    # --- Predict on test period ---
    test_data = df_feat[(df_feat['datetime'] >= predict_start) &
                         (df_feat['datetime'] < predict_end)].copy()
    test_data['s1_pred'] = model_s1.predict(test_data[s1_avail].fillna(0))
    test_data['residual'] = test_data[target] - test_data['s1_pred']

    test_data = create_residual_features(test_data, horizon)
    test_clean = test_data.dropna(subset=s2_avail)

    test_clean['s2_pred'] = model_s2.predict(test_clean[s2_avail])
    test_clean['predicted_error'] = test_clean['s1_pred'] + test_clean['s2_pred']

    return test_clean[['datetime', target, 'predicted_error']].rename(
        columns={target: 'actual_error'})


def main():
    print("=" * 70)
    print("WALK-FORWARD OOS LOAD NOWCAST PREDICTIONS")
    print("=" * 70)

    df = load_data()

    horizon = 2  # H+2 for the imbalance predictor

    # Load best params from tuning
    h_dir = TUNING_PATH / f'h{horizon}'
    with open(h_dir / 'stage1_best_params.json') as f:
        s1_best = json.load(f)
    with open(h_dir / 'stage2_best_params.json') as f:
        s2_best = json.load(f)

    s1_features = s1_best['features']
    s1_params = s1_best['params']
    s2_features = s2_best['s2_features']
    s2_params = s2_best['s2_params']

    print(f"\n[*] Generating H+{horizon} OOS predictions across {len(FOLDS)} folds")
    print(f"    S1 features: {len(s1_features)}, S2 features: {len(s2_features)}")

    all_preds = []

    for i, (train_end, pred_start, pred_end) in enumerate(FOLDS):
        print(f"\n--- Fold {i+1}: Train < {train_end}, Predict [{pred_start}, {pred_end}) ---")

        fold_preds = predict_fold(
            df, horizon, train_end, pred_start, pred_end,
            s1_features, s1_params, s2_features, s2_params
        )

        if fold_preds is not None and len(fold_preds) > 0:
            mae = (fold_preds['actual_error'] - fold_preds['predicted_error']).abs().mean()
            print(f"    Predictions: {len(fold_preds)} hours, MAE: {mae:.1f} MW")
            all_preds.append(fold_preds)
        else:
            print("    [!] No predictions generated for this fold")

    # Concatenate all folds
    oos = pd.concat(all_preds, ignore_index=True)
    oos = oos.sort_values('datetime').reset_index(drop=True)
    oos = oos.drop_duplicates(subset='datetime', keep='last')

    # Compute overall metrics
    overall_mae = (oos['actual_error'] - oos['predicted_error']).abs().mean()
    corr = oos['actual_error'].corr(oos['predicted_error'])

    print(f"\n{'=' * 70}")
    print(f"OOS SUMMARY (H+{horizon})")
    print(f"{'=' * 70}")
    print(f"  Total predictions: {len(oos)}")
    print(f"  Date range: {oos['datetime'].min()} to {oos['datetime'].max()}")
    print(f"  Overall MAE: {overall_mae:.1f} MW")
    print(f"  Correlation: {corr:.3f}")

    # Save in the format expected by the imbalance predictor
    out_path = OUTPUT_PATH / f'h{horizon}_oos_predictions.csv'
    oos.to_csv(out_path, index=False)
    print(f"\n[+] Saved: {out_path}")

    # Also save to the error_analysis dir for compatibility
    compat_dir = Path(__file__).parent.parent / 'scripts' / 'error_analysis'
    if compat_dir.exists():
        compat_path = compat_dir / f'h{horizon}_oos_predictions.csv'
        oos.to_csv(compat_path, index=False)
        print(f"[+] Saved: {compat_path}")


if __name__ == "__main__":
    main()
