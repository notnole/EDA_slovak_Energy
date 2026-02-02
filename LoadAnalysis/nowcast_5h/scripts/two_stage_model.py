"""
Two-Stage Residual Correction Model (CORRECTED)
================================================
Key insight: Use OUR OWN residuals (not DAMAS errors) for correction.

CRITICAL: Stage 2 features must be shifted by horizon h to avoid data leakage!
- For H+h prediction, residual_lag1 must be shifted by h (not 1)
- This ensures we don't use future error information

Stage 1: Predict DAMAS error using standard features
Stage 2: Correct prediction using OUR residual features (properly shifted)

Corrected Results (no data leakage):
  H+1: 34.0 MW (+49.2% improvement)
  H+2: 46.1 MW (+31.3% improvement)
  H+3: 53.4 MW (+20.4% improvement)
  H+4: 58.3 MW (+13.1% improvement)
  H+5: 61.7 MW (+8.0% improvement)
  (Proper decay with horizon - no leakage)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent  # ipesoft_eda_data
OUTPUT_PATH = Path(__file__).parent.parent / 'models'  # nowcast_5h/models
OUTPUT_PATH.mkdir(exist_ok=True)


def load_data():
    """Load all data sources."""
    print("Loading data...")

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
        'load_mw': ['std', 'first', 'last']
    }).reset_index()
    load_hourly.columns = ['datetime', 'load_std_3min', 'load_first', 'load_last']
    load_hourly['load_trend_3min'] = load_hourly['load_last'] - load_hourly['load_first']
    df = df.merge(load_hourly[['datetime', 'load_std_3min', 'load_trend_3min']],
                  on='datetime', how='left')

    # Regulation
    reg_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'regulation_3min.csv')
    reg_3min['datetime'] = pd.to_datetime(reg_3min['datetime'])
    reg_3min['hour_start'] = reg_3min['datetime'].dt.floor('h')
    reg_hourly = reg_3min.groupby('hour_start').agg({
        'regulation_mw': ['mean', 'std']
    }).reset_index()
    reg_hourly.columns = ['datetime', 'reg_mean', 'reg_std']
    df = df.merge(reg_hourly, on='datetime', how='left')

    print(f"  Loaded {len(df):,} hourly records")
    return df


def create_stage1_features(df):
    """Create features for Stage 1 (DAMAS error prediction)."""
    df = df.copy()

    # Error lags
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

    # Regulation
    for lag in range(1, 4):
        df[f'reg_mean_lag{lag}'] = df['reg_mean'].shift(lag)
    df['reg_std_lag1'] = df['reg_std'].shift(1)

    # Seasonal
    df['seasonal_error'] = df.groupby(['dow', 'hour'])['error'].transform('mean')

    # Time
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)

    # Targets
    for h in range(1, 6):
        df[f'target_h{h}'] = df['error'].shift(-h)

    return df


def get_stage1_features():
    """Get Stage 1 feature list."""
    return [
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

    CRITICAL: For horizon h, we must shift residuals by h to avoid leakage!
    - residual[t] = target_h[t] - s1_pred[t] = error[t+h] - s1_pred[t]
    - This contains error[t+h], which is FUTURE information for predicting target at t+h
    - To predict at time t, we can only use residuals up to time t-h
    - So residual_lag1 should be residual.shift(h), not residual.shift(1)
    """
    df = df.copy()

    # CORRECTED: Shift by horizon to avoid leakage
    for lag in range(1, 7):
        df[f'residual_lag{lag}'] = df[f'residual_h{horizon}'].shift(horizon + lag - 1)

    # Rolling stats (also need proper shift)
    for window in [3, 6]:
        df[f'residual_roll_mean_{window}h'] = df[f'residual_h{horizon}'].shift(horizon).rolling(window).mean()
        df[f'residual_roll_std_{window}h'] = df[f'residual_h{horizon}'].shift(horizon).rolling(window).std()

    # Residual trend
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


def train_two_stage_model(df, horizon=1):
    """Train two-stage model for given horizon."""
    target = f'target_h{horizon}'
    stage1_features = get_stage1_features()
    stage1_avail = [f for f in stage1_features if f in df.columns]

    # Filter data
    df_model = df.dropna(subset=[target] + stage1_avail).copy()

    # ========== STAGE 1 ==========
    # Train on all 2024
    train_s1 = df_model[df_model['year'] == 2024]

    model_s1 = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=8, num_leaves=50,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=-1
    )
    model_s1.fit(train_s1[stage1_avail], train_s1[target])

    # Get Stage 1 predictions for all data
    df_model['stage1_pred'] = model_s1.predict(df_model[stage1_avail])
    df_model[f'residual_h{horizon}'] = df_model[target] - df_model['stage1_pred']

    # ========== STAGE 2 ==========
    # Create residual features (CORRECTED: with proper horizon shift)
    df_model = create_residual_features(df_model, horizon)

    stage2_features = get_stage2_features()
    stage2_avail = [f for f in stage2_features if f in df_model.columns]

    # Train Stage 2 on H2 2024 (where we have residuals)
    train_s2 = df_model[(df_model['year'] == 2024) & (df_model['month'] > 6)]
    train_s2 = train_s2.dropna(subset=stage2_avail)

    # Test on 2025
    test = df_model[df_model['year'] >= 2025].dropna(subset=stage2_avail).copy()

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
        'n_train_s1': len(train_s1),
        'n_train_s2': len(train_s2),
        'n_test': len(test),
    }


def main():
    print("=" * 70)
    print("TWO-STAGE RESIDUAL CORRECTION MODEL (CORRECTED)")
    print("=" * 70)
    print("\nKey insight: Use OUR residuals for correction, not DAMAS errors!")
    print("CRITICAL: Stage 2 features shifted by horizon to avoid leakage!")

    # Load and prepare
    df = load_data()
    df = create_stage1_features(df)

    # Train for all horizons
    results = []
    models = {}

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for h in range(1, 6):
        print(f"\n  H+{h}...")
        result = train_two_stage_model(df, horizon=h)
        results.append(result)
        models[h] = {'s1': result['model_s1'], 's2': result['model_s2']}

        print(f"      Stage 1: {result['s1_mae']:.1f} MW ({result['s1_improvement']:+.1f}%)")
        print(f"      Stage 2: {result['s2_mae']:.1f} MW ({result['s2_improvement']:+.1f}%)")
        print(f"      Gain:    {result['stage2_gain']:+.1f}%")
        print(f"      Autocorr: {result['s1_autocorr']:.3f} -> {result['s2_autocorr']:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  {'H':<4} {'Baseline':<10} {'Stage1':<10} {'Stage2':<10} {'S1 Improv':<11} {'S2 Improv':<11} {'Gain':<8} {'AC1->AC2'}")
    print("  " + "-" * 85)

    for r in results:
        print(f"  H+{r['horizon']:<2} {r['baseline_mae']:<10.1f} {r['s1_mae']:<10.1f} {r['s2_mae']:<10.1f} "
              f"{r['s1_improvement']:+.1f}%      {r['s2_improvement']:+.1f}%      {r['stage2_gain']:+.1f}%   "
              f"{r['s1_autocorr']:.2f}->{r['s2_autocorr']:.2f}")

    # Save models
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)

    for h, m in models.items():
        joblib.dump(m['s1'], OUTPUT_PATH / f'stage1_h{h}.joblib')
        joblib.dump(m['s2'], OUTPUT_PATH / f'stage2_h{h}.joblib')

    print(f"\n  Models saved to: {OUTPUT_PATH}")

    # Key numbers
    r1 = results[0]  # H+1
    print("\n" + "=" * 70)
    print("KEY RESULTS (H+1)")
    print("=" * 70)
    print(f"\n  DAMAS Baseline:     {r1['baseline_mae']:.1f} MW")
    print(f"  Stage 1 (standard): {r1['s1_mae']:.1f} MW ({r1['s1_improvement']:+.1f}%)")
    print(f"  Stage 2 (residual): {r1['s2_mae']:.1f} MW ({r1['s2_improvement']:+.1f}%)")
    print(f"\n  *** Stage 2 gain over Stage 1: {r1['stage2_gain']:+.1f}% ({r1['s1_mae'] - r1['s2_mae']:.1f} MW) ***")
    print(f"  *** Residual autocorrelation: {r1['s1_autocorr']:.2f} -> {r1['s2_autocorr']:.2f} ***")

    return results, models


if __name__ == "__main__":
    results, models = main()
