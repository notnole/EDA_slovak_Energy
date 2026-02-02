"""
Two-Stage Residual Correction Model - FIXED VERSION
====================================================

Fixes applied:
1. ADDED error_lag0 (most recent error) - was missing valuable feature
2. FIXED seasonal_error computed from training data only - was leaking test info
3. FIXED Stage 2 trains on out-of-sample Stage 1 residuals via time-split

Changes from original:
- error_lag0 = error[t] added (the just-completed hour's error)
- seasonal_error computed ONLY from 2024 training data
- Stage 1 uses H1 2024 for training, generates OOS predictions for H2 2024
- Stage 2 trains on these OOS residuals (H2 2024)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent  # ipesoft_eda_data
OUTPUT_PATH = Path(__file__).parent.parent / 'models_fixed'
OUTPUT_PATH.mkdir(exist_ok=True)


def load_data():
    """Load all data sources."""
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
    try:
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
    except Exception as e:
        print(f"    [!] Could not load 3-min data: {e}")
        df['load_std_3min'] = np.nan
        df['load_trend_3min'] = np.nan

    # Regulation
    try:
        reg_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'regulation_3min.csv')
        reg_3min['datetime'] = pd.to_datetime(reg_3min['datetime'])
        reg_3min['hour_start'] = reg_3min['datetime'].dt.floor('h')
        reg_hourly = reg_3min.groupby('hour_start').agg({
            'regulation_mw': ['mean', 'std']
        }).reset_index()
        reg_hourly.columns = ['datetime', 'reg_mean', 'reg_std']
        df = df.merge(reg_hourly, on='datetime', how='left')
    except Exception as e:
        print(f"    [!] Could not load regulation data: {e}")
        df['reg_mean'] = np.nan
        df['reg_std'] = np.nan

    print(f"    Loaded {len(df):,} hourly records")
    return df


def create_stage1_features(df, seasonal_error_map=None):
    """
    Create features for Stage 1 (DAMAS error prediction).

    FIX 1: Added error_lag0 (the most recent, just-completed hour's error)
    FIX 2: seasonal_error computed from provided map (training data only)
    """
    df = df.copy()

    # FIX 1: Add error_lag0 - the just-completed hour's error (available at prediction time!)
    df['error_lag0'] = df['error']  # Current hour error (known at end of hour)

    # Error lags (lag1 onwards are definitely past)
    for lag in range(1, 9):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    # Rolling statistics (shift by 1 to not include current prediction row in mean)
    for window in [3, 6, 12, 24]:
        df[f'error_roll_mean_{window}h'] = df['error'].shift(1).rolling(window).mean()
        df[f'error_roll_std_{window}h'] = df['error'].shift(1).rolling(window).std()

    # Error trends
    df['error_trend_3h'] = df['error_lag1'] - df['error_lag3']
    df['error_trend_6h'] = df['error_lag1'] - df['error_lag6']
    df['error_momentum'] = (0.5 * (df['error_lag1'] - df['error_lag2']) +
                            0.3 * (df['error_lag2'] - df['error_lag3']) +
                            0.2 * (df['error_lag3'] - df['error_lag4']))

    # 3-min features (lagged to be safe)
    df['load_volatility_lag1'] = df['load_std_3min'].shift(1)
    df['load_trend_lag1'] = df['load_trend_3min'].shift(1)

    # Regulation features
    for lag in range(1, 4):
        df[f'reg_mean_lag{lag}'] = df['reg_mean'].shift(lag)
    df['reg_std_lag1'] = df['reg_std'].shift(1)

    # FIX 2: seasonal_error from training data only (not full dataset)
    if seasonal_error_map is not None:
        df['seasonal_error'] = df.apply(
            lambda x: seasonal_error_map.get((x['dow'], x['hour']), 0), axis=1
        )
    else:
        # Fallback: compute from current data (only use during initial training)
        df['seasonal_error'] = df.groupby(['dow', 'hour'])['error'].transform('mean')

    # Time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)

    # Targets for each horizon
    for h in range(1, 6):
        df[f'target_h{h}'] = df['error'].shift(-h)

    return df


def get_stage1_features():
    """Get Stage 1 feature list - NOW INCLUDES error_lag0!"""
    return [
        'error_lag0',  # FIX 1: Added - most recent error (just-completed hour)
        'error_lag1', 'error_lag2', 'error_lag3', 'error_lag4', 'error_lag5', 'error_lag6',
        'error_roll_mean_3h', 'error_roll_std_3h',
        'error_roll_mean_6h', 'error_roll_std_6h',
        'error_roll_mean_12h', 'error_roll_std_12h',
        'error_roll_mean_24h', 'error_roll_std_24h',
        'error_trend_3h', 'error_trend_6h', 'error_momentum',
        'load_volatility_lag1', 'load_trend_lag1',
        'reg_mean_lag1', 'reg_mean_lag2', 'reg_mean_lag3', 'reg_std_lag1',
        'seasonal_error',
        'hour', 'hour_sin', 'hour_cos', 'dow', 'is_weekend',
    ]


def create_residual_features(df, horizon=1):
    """
    Create features from OUR residuals (not DAMAS error).

    Shift formula: shift(horizon + lag - 1) ensures we only use residuals
    containing errors up to time t (the current, known error).
    """
    df = df.copy()

    for lag in range(1, 7):
        df[f'residual_lag{lag}'] = df[f'residual_h{horizon}'].shift(horizon + lag - 1)

    for window in [3, 6]:
        df[f'residual_roll_mean_{window}h'] = df[f'residual_h{horizon}'].shift(horizon).rolling(window).mean()
        df[f'residual_roll_std_{window}h'] = df[f'residual_h{horizon}'].shift(horizon).rolling(window).std()

    df['residual_trend_3h'] = df['residual_lag1'] - df['residual_lag3']

    return df


def get_stage2_features():
    """Get Stage 2 feature list (residual-based)."""
    return [
        'residual_lag1', 'residual_lag2', 'residual_lag3',
        'residual_roll_mean_3h', 'residual_roll_mean_6h',
        'residual_roll_std_3h', 'residual_roll_std_6h',
        'residual_trend_3h',
    ]


def train_two_stage_model_fixed(df, horizon=1):
    """
    Train two-stage model with all fixes applied.

    FIX 3: Stage 1 trained on H1 2024, generates OOS predictions for H2 2024
           Stage 2 then trains on these truly out-of-sample residuals
    """
    target = f'target_h{horizon}'
    stage1_features = get_stage1_features()
    stage1_avail = [f for f in stage1_features if f in df.columns]

    # Filter data with valid features
    df_model = df.dropna(subset=[target] + stage1_avail).copy()

    # ========== FIX 2: Compute seasonal_error from training data only ==========
    train_all_2024 = df_model[df_model['year'] == 2024]
    seasonal_error_map = train_all_2024.groupby(['dow', 'hour'])['error'].mean().to_dict()

    # Recompute seasonal_error using only training data
    df_model['seasonal_error'] = df_model.apply(
        lambda x: seasonal_error_map.get((x['dow'], x['hour']), 0), axis=1
    )

    # ========== FIX 3: Proper train splits to avoid in-sample residuals ==========
    # Stage 1: Train on H1 2024 (Jan-Jun), predict on H2 2024 (Jul-Dec) for OOS residuals
    train_s1 = df_model[(df_model['year'] == 2024) & (df_model['month'] <= 6)]
    oos_s1 = df_model[(df_model['year'] == 2024) & (df_model['month'] > 6)]
    test = df_model[df_model['year'] >= 2025].copy()

    print(f"      Stage 1 train: {len(train_s1):,} (H1 2024)")
    print(f"      Stage 1 OOS:   {len(oos_s1):,} (H2 2024)")
    print(f"      Test:          {len(test):,} (2025+)")

    # ========== STAGE 1 ==========
    model_s1 = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=8, num_leaves=50,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=-1
    )
    model_s1.fit(train_s1[stage1_avail], train_s1[target])

    # Generate OUT-OF-SAMPLE predictions for H2 2024 (Stage 2 training data)
    oos_s1 = oos_s1.copy()
    oos_s1['stage1_pred'] = model_s1.predict(oos_s1[stage1_avail])
    oos_s1[f'residual_h{horizon}'] = oos_s1[target] - oos_s1['stage1_pred']

    # Also get predictions for test data
    test['stage1_pred'] = model_s1.predict(test[stage1_avail])
    test[f'residual_h{horizon}'] = test[target] - test['stage1_pred']

    # ========== STAGE 2 ==========
    # Create residual features for OOS data
    oos_s1 = create_residual_features(oos_s1, horizon)
    test = create_residual_features(test, horizon)

    stage2_features = get_stage2_features()
    stage2_avail = [f for f in stage2_features if f in oos_s1.columns]

    # Train Stage 2 on OOS residuals (H2 2024)
    train_s2 = oos_s1.dropna(subset=stage2_avail).copy()
    test = test.dropna(subset=stage2_avail).copy()

    model_s2 = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5, num_leaves=30,
        min_child_samples=20, random_state=42, verbosity=-1
    )
    model_s2.fit(train_s2[stage2_avail], train_s2[f'residual_h{horizon}'])

    # Final predictions
    test['residual_pred'] = model_s2.predict(test[stage2_avail])
    test['stage2_pred'] = test['stage1_pred'] + test['residual_pred']
    test['final_residual'] = test[target] - test['stage2_pred']

    # Metrics
    baseline_mae = np.abs(test[target]).mean()
    s1_mae = np.abs(test[target] - test['stage1_pred']).mean()
    s2_mae = np.abs(test[target] - test['stage2_pred']).mean()

    s1_autocorr = test[f'residual_h{horizon}'].corr(test[f'residual_h{horizon}'].shift(1))
    s2_autocorr = test['final_residual'].corr(test['final_residual'].shift(1))

    return {
        'horizon': horizon,
        'baseline_mae': baseline_mae,
        's1_mae': s1_mae,
        's2_mae': s2_mae,
        's1_improvement': (baseline_mae - s1_mae) / baseline_mae * 100,
        's2_improvement': (baseline_mae - s2_mae) / baseline_mae * 100,
        'stage2_gain': (s1_mae - s2_mae) / s1_mae * 100,
        's1_autocorr': s1_autocorr,
        's2_autocorr': s2_autocorr,
        'model_s1': model_s1,
        'model_s2': model_s2,
        'seasonal_error_map': seasonal_error_map,
        'n_train_s1': len(train_s1),
        'n_train_s2': len(train_s2),
        'n_test': len(test),
    }


def main():
    print("=" * 70)
    print("TWO-STAGE MODEL - FIXED VERSION")
    print("=" * 70)
    print("""
Fixes applied:
  1. ADDED error_lag0 (most recent error at prediction time)
  2. FIXED seasonal_error computed from training data only
  3. FIXED Stage 2 trains on out-of-sample Stage 1 residuals
""")

    # Load and prepare
    df = load_data()

    # Create features (seasonal_error will be recomputed per-horizon with training data only)
    df = create_stage1_features(df)

    # Train for all horizons
    results = []
    models = {}

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for h in range(1, 6):
        print(f"\n  H+{h}...")
        result = train_two_stage_model_fixed(df, horizon=h)
        results.append(result)
        models[h] = {
            's1': result['model_s1'],
            's2': result['model_s2'],
            'seasonal_map': result['seasonal_error_map']
        }

        print(f"      Stage 1: {result['s1_mae']:.1f} MW ({result['s1_improvement']:+.1f}%)")
        print(f"      Stage 2: {result['s2_mae']:.1f} MW ({result['s2_improvement']:+.1f}%)")
        print(f"      Gain:    {result['stage2_gain']:+.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (FIXED MODEL)")
    print("=" * 70)
    print(f"\n  {'H':<4} {'Baseline':<10} {'Stage1':<10} {'Stage2':<10} {'S1 Improv':<11} {'S2 Improv':<11} {'Gain':<8}")
    print("  " + "-" * 75)

    for r in results:
        print(f"  H+{r['horizon']:<2} {r['baseline_mae']:<10.1f} {r['s1_mae']:<10.1f} {r['s2_mae']:<10.1f} "
              f"{r['s1_improvement']:+.1f}%      {r['s2_improvement']:+.1f}%      {r['stage2_gain']:+.1f}%")

    # Save models
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)

    for h, m in models.items():
        joblib.dump(m['s1'], OUTPUT_PATH / f'stage1_h{h}_fixed.joblib')
        joblib.dump(m['s2'], OUTPUT_PATH / f'stage2_h{h}_fixed.joblib')
        joblib.dump(m['seasonal_map'], OUTPUT_PATH / f'seasonal_map_h{h}.joblib')

    print(f"\n  Models saved to: {OUTPUT_PATH}")

    # Compare with original (if we have the numbers)
    print("\n" + "=" * 70)
    print("COMPARISON: ORIGINAL vs FIXED")
    print("=" * 70)

    # Original results from README
    original = {
        1: {'baseline': 67.0, 's1': 41.6, 's2': 34.0},
        2: {'baseline': 67.0, 's1': 50.3, 's2': 46.1},
        3: {'baseline': 67.1, 's1': 56.0, 's2': 53.4},
        4: {'baseline': 67.1, 's1': 60.0, 's2': 58.3},
        5: {'baseline': 67.2, 's1': 62.4, 's2': 61.7},
    }

    print(f"\n  {'H':<4} {'Original S2':<12} {'Fixed S2':<12} {'Diff':<10} {'Notes'}")
    print("  " + "-" * 60)

    for r in results:
        h = r['horizon']
        orig_s2 = original[h]['s2']
        fixed_s2 = r['s2_mae']
        diff = fixed_s2 - orig_s2

        if diff > 2:
            note = "Worse (leakage removed)"
        elif diff < -2:
            note = "Better (error_lag0 helped)"
        else:
            note = "Similar"

        print(f"  H+{h:<2} {orig_s2:<12.1f} {fixed_s2:<12.1f} {diff:+.1f} MW    {note}")

    return results, models


if __name__ == "__main__":
    results, models = main()
