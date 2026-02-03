"""
Evaluate Stage 2 with Cascade Features
======================================
Uses existing tuned params but adds cascade features to Stage 2.

Key: Stage 2 trains on OOF predictions to avoid leakage.

Training setup:
- Stage 1: Train on 2024 + H1 2025
- Stage 1 OOF: Predictions on H2 2025 (out-of-sample for S1)
- Stage 2: Train on H2 2025 using OOF residuals + cascade features
- Evaluate: January 2026
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

BASE_PATH = Path(__file__).parent.parent.parent.parent
TUNING_PATH = Path(__file__).parent
OUTPUT_PATH = TUNING_PATH / 'cascade_evaluation'
OUTPUT_PATH.mkdir(exist_ok=True)


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

    # 3-minute features
    load_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'load_3min.csv')
    load_3min['datetime'] = pd.to_datetime(load_3min['datetime'])
    load_3min['hour_start'] = load_3min['datetime'].dt.floor('h')
    load_hourly = load_3min.groupby('hour_start').agg({
        'load_mw': ['std', 'first', 'last']
    }).reset_index()
    load_hourly.columns = ['datetime', 'load_std_3min', 'load_first', 'load_last']
    load_hourly['load_trend_3min'] = load_hourly['load_last'] - load_hourly['load_first']
    df = df.merge(load_hourly[['datetime', 'load_std_3min', 'load_trend_3min']], on='datetime', how='left')

    # Regulation
    reg_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'regulation_3min.csv')
    reg_3min['datetime'] = pd.to_datetime(reg_3min['datetime'])
    reg_3min['hour_start'] = reg_3min['datetime'].dt.floor('h')
    reg_hourly = reg_3min.groupby('hour_start').agg({'regulation_mw': ['mean', 'std']}).reset_index()
    reg_hourly.columns = ['datetime', 'reg_mean', 'reg_std']
    df = df.merge(reg_hourly, on='datetime', how='left')

    print(f"  Loaded {len(df):,} rows")
    return df


def create_all_features(df, train_mask):
    """Create all Stage 1 features."""
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

    for h in range(1, 6):
        df[f'target_h{h}'] = df['error'].shift(-h)

    return df


def create_residual_features(df, residual_col, horizon):
    """Create Stage 2 residual features."""
    df = df.copy()
    for lag in range(1, 7):
        df[f'residual_lag{lag}'] = df[residual_col].shift(horizon + lag - 1)
    for window in [3, 6]:
        df[f'residual_roll_mean_{window}h'] = df[residual_col].shift(horizon).rolling(window).mean()
        df[f'residual_roll_std_{window}h'] = df[residual_col].shift(horizon).rolling(window).std()
    df['residual_trend_3h'] = df['residual_lag1'] - df['residual_lag3']
    df['residual_trend_6h'] = df['residual_lag1'] - df['residual_lag6']
    return df


def get_sample_weights(dates, train_end_date, recency_months, recency_weight):
    days_ago = (train_end_date - dates).dt.days
    months_ago = days_ago / 30.0
    weights = np.where(months_ago <= recency_months, recency_weight, 1.0)
    return weights / weights.mean()


def main():
    print("=" * 70)
    print("STAGE 2 WITH CASCADE FEATURES")
    print("=" * 70)
    print("\nSetup:")
    print("  - Stage 1: Train on 2024 + H1 2025")
    print("  - Stage 2: Train on H2 2025 (OOF residuals) + cascade features")
    print("  - Evaluate: January 2026")

    df = load_data()

    # Time splits
    s1_train_end = '2025-07-01'  # Stage 1 trains up to here
    s2_train_start = '2025-07-01'  # Stage 2 trains on H2 2025 (OOF for S1)
    s2_train_end = '2026-01-01'
    test_start = '2026-01-01'
    test_end = '2026-02-01'

    # Create features
    train_mask = df['datetime'] < s2_train_end
    df = create_all_features(df, train_mask)

    # Load tuned params
    print("\nLoading tuned parameters...")
    s1_params = {}
    s1_features = {}
    s2_params = {}
    s2_features = {}

    for h in range(1, 6):
        h_dir = TUNING_PATH / f'h{h}'
        with open(h_dir / 'stage1_best_params.json') as f:
            s1_best = json.load(f)
        with open(h_dir / 'stage2_best_params.json') as f:
            s2_best = json.load(f)
        s1_params[h] = s1_best['params']
        s1_features[h] = s1_best['features']
        s2_params[h] = s2_best['s2_params']
        s2_features[h] = s2_best['s2_features']

    # ========== STAGE 1: Train all models ==========
    print("\n" + "=" * 70)
    print("STAGE 1 TRAINING")
    print("=" * 70)

    models_s1 = {}
    s1_train_data = df[df['datetime'] < s1_train_end]

    for h in range(1, 6):
        target = f'target_h{h}'
        feats = [f for f in s1_features[h] if f in df.columns]

        train = s1_train_data.dropna(subset=[target] + feats)

        lgb_params = {k: v for k, v in s1_params[h].items()
                      if k in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves',
                               'min_child_samples', 'subsample', 'colsample_bytree',
                               'reg_alpha', 'reg_lambda']}
        lgb_params['random_state'] = 42
        lgb_params['verbosity'] = -1

        weights = get_sample_weights(train['datetime'], pd.to_datetime(s1_train_end),
                                      s1_params[h].get('recency_months', 3),
                                      s1_params[h].get('recency_weight', 2.0))

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(train[feats], train[target], sample_weight=weights)
        models_s1[h] = model

        print(f"  H+{h}: {len(train):,} samples, {len(feats)} features")

    # ========== GET ALL STAGE 1 PREDICTIONS ==========
    print("\n" + "=" * 70)
    print("STAGE 1 PREDICTIONS (for all data)")
    print("=" * 70)

    for h in range(1, 6):
        target = f'target_h{h}'
        feats = [f for f in s1_features[h] if f in df.columns]
        mask = df[feats].notna().all(axis=1)
        df.loc[mask, f's1_pred_h{h}'] = models_s1[h].predict(df.loc[mask, feats])
        df[f'residual_h{h}'] = df[target] - df[f's1_pred_h{h}']

    # ========== CREATE CASCADE FEATURES ==========
    print("\n" + "=" * 70)
    print("CREATING CASCADE FEATURES")
    print("=" * 70)

    # For each horizon h > 1:
    # - cascade_pred_lag1: Our H+h prediction from last period (shifted by 1)
    # - cascade_pred_shorter: H+(h-1) prediction for same target
    # - cascade_revision: The difference

    for h in range(1, 6):
        # Our own prediction lagged by 1
        df[f'cascade_pred_h{h}_lag1'] = df[f's1_pred_h{h}'].shift(1)

        if h > 1:
            # H+(h-1) prediction - this is for the SAME target as our lag1!
            # At time T: our lag1 was for error[T+h-1]
            # At time T: H+(h-1) predicts error[T+(h-1)] - same target!
            df[f'cascade_pred_h{h-1}_same_target'] = df[f's1_pred_h{h-1}']
            df[f'cascade_revision_h{h}'] = df[f'cascade_pred_h{h-1}_same_target'] - df[f'cascade_pred_h{h}_lag1']

        print(f"  H+{h}: cascade features created")

    # ========== STAGE 2: Train with cascade features ==========
    print("\n" + "=" * 70)
    print("STAGE 2 TRAINING (with cascade features)")
    print("=" * 70)

    models_s2 = {}
    s2_train_period = df[(df['datetime'] >= s2_train_start) & (df['datetime'] < s2_train_end)]

    for h in range(1, 6):
        target = f'target_h{h}'
        residual_col = f'residual_h{h}'

        # Create residual features
        df_s2 = create_residual_features(df.copy(), residual_col, h)

        # Base S2 features
        base_feats = [f for f in s2_features[h] if f in df_s2.columns]

        # Add cascade features for h > 1
        cascade_feats = [f'cascade_pred_h{h}_lag1']
        if h > 1:
            cascade_feats.extend([
                f'cascade_pred_h{h-1}_same_target',
                f'cascade_revision_h{h}'
            ])
        cascade_feats = [f for f in cascade_feats if f in df_s2.columns]

        all_feats = base_feats + cascade_feats

        # Training data (H2 2025 - OOF for Stage 1)
        train = df_s2[(df_s2['datetime'] >= s2_train_start) & (df_s2['datetime'] < s2_train_end)]
        train = train.dropna(subset=[residual_col] + all_feats)

        s2_lgb = {
            'n_estimators': s2_params[h]['s2_n_estimators'],
            'learning_rate': s2_params[h]['s2_learning_rate'],
            'max_depth': s2_params[h]['s2_max_depth'],
            'num_leaves': s2_params[h]['s2_num_leaves'],
            'min_child_samples': s2_params[h].get('s2_min_child_samples', 10),
            'subsample': s2_params[h].get('s2_subsample', 0.8),
            'colsample_bytree': s2_params[h].get('s2_colsample_bytree', 0.8),
            'random_state': 42,
            'verbosity': -1,
        }

        weights = get_sample_weights(train['datetime'], pd.to_datetime(s2_train_end),
                                      s2_params[h].get('s2_recency_months', 2),
                                      s2_params[h].get('s2_recency_weight', 1.5))

        model = lgb.LGBMRegressor(**s2_lgb)
        model.fit(train[all_feats], train[residual_col], sample_weight=weights)
        models_s2[h] = {'model': model, 'features': all_feats}

        print(f"  H+{h}: {len(train):,} samples, base={len(base_feats)}, cascade={len(cascade_feats)}")

    # ========== EVALUATE ON JANUARY 2026 ==========
    print("\n" + "=" * 70)
    print("EVALUATION - JANUARY 2026")
    print("=" * 70)

    test_data = df[(df['datetime'] >= test_start) & (df['datetime'] < test_end)].copy()

    results = []

    print(f"\n{'Horizon':<10} {'Baseline':<12} {'S1 MAE':<12} {'S2 Base':<12} {'S2+Cascade':<12} {'Cascade Gain':<12}")
    print("-" * 70)

    for h in range(1, 6):
        target = f'target_h{h}'
        residual_col = f'residual_h{h}'

        # Create residual features for test
        df_test = create_residual_features(test_data.copy(), residual_col, h)

        # Get S2 predictions
        all_feats = models_s2[h]['features']
        avail_feats = [f for f in all_feats if f in df_test.columns]

        test_valid = df_test.dropna(subset=[target] + avail_feats)

        if len(test_valid) < 50:
            print(f"  H+{h}: Not enough test data")
            continue

        # S2 prediction with cascade
        test_valid = test_valid.copy()
        test_valid['s2_pred'] = models_s2[h]['model'].predict(test_valid[avail_feats])
        test_valid['final_pred'] = test_valid[f's1_pred_h{h}'] + test_valid['s2_pred']

        # Metrics
        baseline_mae = np.abs(test_valid[target]).mean()
        s1_mae = np.abs(test_valid[target] - test_valid[f's1_pred_h{h}']).mean()
        s2_cascade_mae = np.abs(test_valid[target] - test_valid['final_pred']).mean()

        # Compare to S2 without cascade (train S2 without cascade features)
        base_feats_only = [f for f in s2_features[h] if f in df_test.columns]
        if len(base_feats_only) > 0:
            df_test_base = create_residual_features(test_data.copy(), residual_col, h)
            train_base = df[(df['datetime'] >= s2_train_start) & (df['datetime'] < s2_train_end)].copy()
            train_base = create_residual_features(train_base, residual_col, h)
            train_base = train_base.dropna(subset=[residual_col] + base_feats_only)

            model_base = lgb.LGBMRegressor(**s2_lgb)
            model_base.fit(train_base[base_feats_only], train_base[residual_col])

            test_base = df_test_base.dropna(subset=base_feats_only)
            test_base = test_base.copy()
            test_base['s2_pred_base'] = model_base.predict(test_base[base_feats_only])
            test_base['final_pred_base'] = test_base[f's1_pred_h{h}'] + test_base['s2_pred_base']
            s2_base_mae = np.abs(test_base[target] - test_base['final_pred_base']).mean()
        else:
            s2_base_mae = s1_mae

        cascade_gain = (s2_base_mae - s2_cascade_mae) / s2_base_mae * 100

        print(f"H+{h:<8} {baseline_mae:<12.1f} {s1_mae:<12.1f} {s2_base_mae:<12.1f} {s2_cascade_mae:<12.1f} {cascade_gain:+.1f}%")

        results.append({
            'horizon': h,
            'baseline_mae': baseline_mae,
            's1_mae': s1_mae,
            's2_base_mae': s2_base_mae,
            's2_cascade_mae': s2_cascade_mae,
            'cascade_gain': cascade_gain,
            'test_data': test_valid,
        })

    # Autocorrelation comparison
    print("\n" + "-" * 70)
    print("AUTOCORRELATION COMPARISON")
    print("-" * 70)
    print(f"\n{'Horizon':<10} {'S2 Base AC(1)':<15} {'S2+Cascade AC(1)':<18} {'AC Reduction':<15}")
    print("-" * 60)

    for r in results:
        h = r['horizon']
        test_valid = r['test_data']

        # S2 base error AC
        base_error = test_valid[f'target_h{h}'] - (test_valid[f's1_pred_h{h}'] + test_valid.get('s2_pred_base', 0))
        cascade_error = test_valid[f'target_h{h}'] - test_valid['final_pred']

        # For base, we need to compute it fresh
        if f's2_pred_base' not in test_valid.columns:
            base_ac = test_valid[f'residual_h{h}'].autocorr(lag=1)
        else:
            base_ac = base_error.autocorr(lag=1) if len(base_error.dropna()) > 10 else np.nan

        cascade_ac = cascade_error.autocorr(lag=1) if len(cascade_error.dropna()) > 10 else np.nan

        ac_reduction = (base_ac - cascade_ac) / base_ac * 100 if not np.isnan(base_ac) and base_ac != 0 else 0

        print(f"H+{h:<8} {base_ac:<15.3f} {cascade_ac:<18.3f} {ac_reduction:+.1f}%")

    # Save summary
    summary = {
        'description': 'Stage 2 with cascade features',
        'results': {r['horizon']: {
            'baseline_mae': r['baseline_mae'],
            's1_mae': r['s1_mae'],
            's2_base_mae': r['s2_base_mae'],
            's2_cascade_mae': r['s2_cascade_mae'],
            'cascade_gain': r['cascade_gain'],
        } for r in results}
    }

    with open(OUTPUT_PATH / 'cascade_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH / 'cascade_results.json'}")


if __name__ == '__main__':
    main()
