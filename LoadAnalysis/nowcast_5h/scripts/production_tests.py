"""
Production Readiness Tests
==========================

1. Post-processing strategies:
   - Bias correction (recent error mean)
   - Prediction clipping
   - Hour-specific bias correction
   - Horizon consistency smoothing

2. Training strategies:
   - Current: Train 2024, test 2025
   - Full: Train all except last 30 days
   - Fine-tuned: Train all, fine-tune on recent 3 months
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent

print("=" * 70)
print("PRODUCTION READINESS TESTS")
print("=" * 70)


def load_and_prepare_data():
    """Load and prepare all data."""
    print("\n[*] Loading data...")

    df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek

    # Create features
    for lag in range(0, 49):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    df['error_roll_mean_24h'] = df['error'].shift(1).rolling(24).mean()
    df['error_roll_std_24h'] = df['error'].shift(1).rolling(24).std()
    df['error_roll_mean_6h'] = df['error'].shift(1).rolling(6).mean()
    df['error_same_hour_yesterday'] = df['error'].shift(24)

    df['error_diff_1h'] = df['error_lag0'] - df['error_lag1']
    df['error_diff_24h'] = df['error_lag0'] - df['error_lag24']

    df['forecast_load'] = df['forecast_load_mw']
    df['forecast_diff_1h'] = df['forecast_load_mw'] - df['forecast_load_mw'].shift(1)

    df['error_lag0_abs'] = df['error_lag0'].abs()
    df['error_lag0_sign'] = np.sign(df['error_lag0'])

    for h in range(1, 6):
        df[f'target_h{h}'] = df['error'].shift(-h)

    print(f"    Total records: {len(df):,}")
    print(f"    Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def get_features():
    """Get feature list for Q0 model."""
    return [
        'error_lag0', 'error_lag1', 'error_lag2', 'error_lag3', 'error_lag24', 'error_lag48',
        'error_roll_mean_24h', 'error_roll_std_24h', 'error_roll_mean_6h',
        'error_same_hour_yesterday',
        'error_diff_1h', 'error_diff_24h',
        'forecast_load', 'forecast_diff_1h',
        'error_lag0_abs', 'error_lag0_sign',
        'hour', 'dow'
    ]


# =============================================================================
# POST-PROCESSING TESTS
# =============================================================================

def test_post_processing(df):
    """Test various post-processing strategies."""
    print("\n" + "=" * 70)
    print("TEST 1: POST-PROCESSING STRATEGIES")
    print("=" * 70)

    features = get_features()

    # Train on 2024, test on last 30 days of data
    last_date = df['datetime'].max()
    test_start = last_date - pd.Timedelta(days=30)

    train = df[df['datetime'] < test_start].dropna(subset=features + ['target_h1'])
    test = df[df['datetime'] >= test_start].dropna(subset=features + ['target_h1'])

    print(f"\n    Train: {len(train):,} ({train['datetime'].min()} to {train['datetime'].max()})")
    print(f"    Test:  {len(test):,} ({test['datetime'].min()} to {test['datetime'].max()})")

    results = []

    for h in [1, 2, 3, 5]:
        target_col = f'target_h{h}'

        # Train model
        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            verbosity=-1, random_state=42
        )
        model.fit(train[features], train[target_col])

        # Raw predictions
        pred_raw = model.predict(test[features])
        actual = test[target_col].values

        mae_raw = np.nanmean(np.abs(actual - pred_raw))

        # === POST-PROCESSING 1: Global bias correction ===
        # Use last 7 days of training data to estimate bias
        recent_train = train[train['datetime'] >= train['datetime'].max() - pd.Timedelta(days=7)]
        recent_pred = model.predict(recent_train[features])
        recent_actual = recent_train[target_col].values
        bias = np.nanmean(recent_actual - recent_pred)

        pred_bias = pred_raw + bias
        mae_bias = np.nanmean(np.abs(actual - pred_bias))

        # === POST-PROCESSING 2: Clipping to historical range ===
        error_min = train['error'].quantile(0.01)
        error_max = train['error'].quantile(0.99)
        pred_clipped = np.clip(pred_raw, error_min, error_max)
        mae_clipped = np.nanmean(np.abs(actual - pred_clipped))

        # === POST-PROCESSING 3: Hour-specific bias correction ===
        pred_hour_bias = pred_raw.copy()
        test_hours = test['hour'].values

        for hour in range(24):
            hour_mask_train = recent_train['hour'] == hour
            hour_mask_test = test_hours == hour

            if hour_mask_train.sum() > 0:
                hour_pred = model.predict(recent_train[hour_mask_train][features])
                hour_actual = recent_train[hour_mask_train][target_col].values
                hour_bias = np.nanmean(hour_actual - hour_pred)
                pred_hour_bias[hour_mask_test] += hour_bias

        mae_hour_bias = np.nanmean(np.abs(actual - pred_hour_bias))

        # === POST-PROCESSING 4: Exponential smoothing of recent bias ===
        # Use exponentially weighted recent errors
        recent_errors = recent_actual - recent_pred
        weights = np.exp(np.linspace(-2, 0, len(recent_errors)))
        weights /= weights.sum()
        ema_bias = np.sum(weights * recent_errors)

        pred_ema = pred_raw + ema_bias
        mae_ema = np.nanmean(np.abs(actual - pred_ema))

        # === POST-PROCESSING 5: Combination (hour bias + clipping) ===
        pred_combined = np.clip(pred_hour_bias, error_min, error_max)
        mae_combined = np.nanmean(np.abs(actual - pred_combined))

        results.append({
            'horizon': h,
            'mae_raw': mae_raw,
            'mae_global_bias': mae_bias,
            'mae_clipped': mae_clipped,
            'mae_hour_bias': mae_hour_bias,
            'mae_ema_bias': mae_ema,
            'mae_combined': mae_combined,
            'bias_value': bias
        })

        print(f"\n    H+{h}:")
        print(f"      Raw:           {mae_raw:.2f} MW")
        print(f"      Global bias:   {mae_bias:.2f} MW (bias={bias:+.1f})")
        print(f"      Clipped:       {mae_clipped:.2f} MW")
        print(f"      Hour bias:     {mae_hour_bias:.2f} MW")
        print(f"      EMA bias:      {mae_ema:.2f} MW")
        print(f"      Combined:      {mae_combined:.2f} MW")

    return pd.DataFrame(results)


# =============================================================================
# TRAINING STRATEGY TESTS
# =============================================================================

def test_training_strategies(df):
    """Test different training strategies."""
    print("\n" + "=" * 70)
    print("TEST 2: TRAINING STRATEGIES")
    print("=" * 70)

    features = get_features()

    # Define test set: last 30 days
    last_date = df['datetime'].max()
    test_start = last_date - pd.Timedelta(days=30)
    test = df[df['datetime'] >= test_start].dropna(subset=features + ['target_h1'])

    print(f"\n    Test set: {len(test):,} records (last 30 days)")
    print(f"    Test range: {test['datetime'].min()} to {test['datetime'].max()}")

    results = []

    for h in [1, 2, 3, 5]:
        target_col = f'target_h{h}'
        horizon_results = {'horizon': h}

        # === STRATEGY 1: Train on 2024 only ===
        train_2024 = df[df['year'] == 2024].dropna(subset=features + [target_col])
        model_2024 = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            verbosity=-1, random_state=42
        )
        model_2024.fit(train_2024[features], train_2024[target_col])
        pred_2024 = model_2024.predict(test[features])
        mae_2024 = np.nanmean(np.abs(test[target_col].values - pred_2024))
        horizon_results['mae_2024_only'] = mae_2024

        # === STRATEGY 2: Train on all data except last 30 days ===
        train_all = df[df['datetime'] < test_start].dropna(subset=features + [target_col])
        model_all = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            verbosity=-1, random_state=42
        )
        model_all.fit(train_all[features], train_all[target_col])
        pred_all = model_all.predict(test[features])
        mae_all = np.nanmean(np.abs(test[target_col].values - pred_all))
        horizon_results['mae_all_data'] = mae_all

        # === STRATEGY 3: Train on all, fine-tune on last 3 months ===
        # Use higher weight for recent 3 months data (simulates fine-tuning)
        finetune_start = test_start - pd.Timedelta(days=90)
        train_all_ft = train_all.copy()
        is_recent = train_all_ft['datetime'] >= finetune_start
        # Recent data gets 3x weight
        ft_weights = np.where(is_recent, 3.0, 1.0)

        model_finetune = lgb.LGBMRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.05,
            verbosity=-1, random_state=42
        )
        model_finetune.fit(train_all_ft[features], train_all_ft[target_col], sample_weight=ft_weights)
        pred_finetune = model_finetune.predict(test[features])
        mae_finetune = np.nanmean(np.abs(test[target_col].values - pred_finetune))
        horizon_results['mae_finetuned'] = mae_finetune

        # === STRATEGY 4: Recent data only (last 6 months before test) ===
        recent_start = test_start - pd.Timedelta(days=180)
        train_recent = df[(df['datetime'] >= recent_start) & (df['datetime'] < test_start)]
        train_recent = train_recent.dropna(subset=features + [target_col])

        model_recent = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            verbosity=-1, random_state=42
        )
        model_recent.fit(train_recent[features], train_recent[target_col])
        pred_recent = model_recent.predict(test[features])
        mae_recent = np.nanmean(np.abs(test[target_col].values - pred_recent))
        horizon_results['mae_recent_6m'] = mae_recent

        # === STRATEGY 5: Weighted training (recent data weighted higher) ===
        train_all_weighted = train_all.copy()
        days_ago = (train_all_weighted['datetime'].max() - train_all_weighted['datetime']).dt.days
        # Exponential decay: recent data gets more weight
        weights = np.exp(-days_ago / 180)  # 180-day half-life
        weights = weights / weights.mean()  # Normalize

        model_weighted = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            verbosity=-1, random_state=42
        )
        model_weighted.fit(
            train_all_weighted[features],
            train_all_weighted[target_col],
            sample_weight=weights
        )
        pred_weighted = model_weighted.predict(test[features])
        mae_weighted = np.nanmean(np.abs(test[target_col].values - pred_weighted))
        horizon_results['mae_weighted'] = mae_weighted

        results.append(horizon_results)

        print(f"\n    H+{h}:")
        print(f"      2024 only:     {mae_2024:.2f} MW")
        print(f"      All data:      {mae_all:.2f} MW")
        print(f"      Fine-tuned:    {mae_finetune:.2f} MW")
        print(f"      Recent 6m:     {mae_recent:.2f} MW")
        print(f"      Weighted:      {mae_weighted:.2f} MW")

    return pd.DataFrame(results)


# =============================================================================
# COMBINED BEST APPROACH
# =============================================================================

def test_best_combination(df):
    """Test best training + post-processing combination."""
    print("\n" + "=" * 70)
    print("TEST 3: BEST COMBINATION")
    print("=" * 70)

    features = get_features()

    last_date = df['datetime'].max()
    test_start = last_date - pd.Timedelta(days=30)
    train = df[df['datetime'] < test_start].dropna(subset=features + ['target_h1'])
    test = df[df['datetime'] >= test_start].dropna(subset=features + ['target_h1'])

    results = []

    for h in [1, 2, 3, 5]:
        target_col = f'target_h{h}'

        # Best training: all data with sample weighting
        days_ago = (train['datetime'].max() - train['datetime']).dt.days
        weights = np.exp(-days_ago / 180)
        weights = weights / weights.mean()

        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            verbosity=-1, random_state=42
        )
        model.fit(train[features], train[target_col], sample_weight=weights)

        # Raw prediction
        pred_raw = model.predict(test[features])
        actual = test[target_col].values
        mae_raw = np.nanmean(np.abs(actual - pred_raw))

        # Best post-processing: hour-specific bias from last 7 days
        recent_train = train[train['datetime'] >= train['datetime'].max() - pd.Timedelta(days=7)]
        pred_corrected = pred_raw.copy()
        test_hours = test['hour'].values

        for hour in range(24):
            hour_mask_train = recent_train['hour'] == hour
            hour_mask_test = test_hours == hour

            if hour_mask_train.sum() > 0:
                hour_pred = model.predict(recent_train[hour_mask_train][features])
                hour_actual = recent_train[hour_mask_train][target_col].values
                hour_bias = np.nanmean(hour_actual - hour_pred)
                pred_corrected[hour_mask_test] += hour_bias

        mae_corrected = np.nanmean(np.abs(actual - pred_corrected))

        results.append({
            'horizon': h,
            'mae_raw': mae_raw,
            'mae_best': mae_corrected,
            'improvement': mae_raw - mae_corrected
        })

        print(f"\n    H+{h}: {mae_raw:.2f} -> {mae_corrected:.2f} MW ({mae_raw - mae_corrected:+.2f})")

    return pd.DataFrame(results)


def main():
    df = load_and_prepare_data()

    # Test post-processing
    pp_results = test_post_processing(df)

    # Test training strategies
    train_results = test_training_strategies(df)

    # Test best combination
    best_results = test_best_combination(df)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print("\n  POST-PROCESSING:")
    avg_improvements = {
        'Global bias': (pp_results['mae_raw'] - pp_results['mae_global_bias']).mean(),
        'Clipping': (pp_results['mae_raw'] - pp_results['mae_clipped']).mean(),
        'Hour bias': (pp_results['mae_raw'] - pp_results['mae_hour_bias']).mean(),
        'EMA bias': (pp_results['mae_raw'] - pp_results['mae_ema_bias']).mean(),
        'Combined': (pp_results['mae_raw'] - pp_results['mae_combined']).mean(),
    }
    for name, imp in sorted(avg_improvements.items(), key=lambda x: -x[1]):
        print(f"    {name}: {imp:+.2f} MW avg improvement")

    print("\n  TRAINING STRATEGY:")
    baseline = train_results['mae_2024_only'].mean()
    strategies = {
        'All data': train_results['mae_all_data'].mean(),
        'Fine-tuned': train_results['mae_finetuned'].mean(),
        'Recent 6m': train_results['mae_recent_6m'].mean(),
        'Weighted': train_results['mae_weighted'].mean(),
    }
    for name, mae in sorted(strategies.items(), key=lambda x: x[1]):
        print(f"    {name}: {mae:.2f} MW ({baseline - mae:+.2f} vs 2024-only)")

    print("\n  BEST COMBINATION (weighted training + hour bias):")
    for _, row in best_results.iterrows():
        print(f"    H+{int(row['horizon'])}: {row['mae_best']:.2f} MW ({row['improvement']:+.2f} improvement)")

    return pp_results, train_results, best_results


if __name__ == "__main__":
    pp_results, train_results, best_results = main()
