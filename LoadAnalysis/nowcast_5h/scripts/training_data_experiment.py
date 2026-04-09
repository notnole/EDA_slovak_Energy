"""
Training Data Experiment: 2025-only vs 2024+2025
================================================
Compare models trained on:
  A) 2025 only (fine-tuned on last 90 days = Oct-Dec 2025)
  B) 2024+2025 (fine-tuned on last 90 days = Oct-Dec 2025)

Test on: January 2026

Question: Does adding 2024 data help or is 2025 alone sufficient?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent  # ipesoft_eda_data


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

    print(f"    Loaded {len(df):,} hourly records")
    print(f"    Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    return df


def create_features(df, train_data_only=None):
    """Create features for Stage 1.

    Args:
        df: Full dataframe
        train_data_only: If provided, compute seasonal_error only from this subset
    """
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

    # Seasonal error - compute from training data only to avoid leakage
    if train_data_only is not None:
        seasonal_map = train_data_only.groupby(['dow', 'hour'])['error'].mean()
        df['seasonal_error'] = df.apply(lambda r: seasonal_map.get((r['dow'], r['hour']), 0), axis=1)
    else:
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
    """Create residual features for Stage 2 with proper horizon shift."""
    df = df.copy()

    for lag in range(1, 7):
        df[f'residual_lag{lag}'] = df[f'residual_h{horizon}'].shift(horizon + lag - 1)

    for window in [3, 6]:
        df[f'residual_roll_mean_{window}h'] = df[f'residual_h{horizon}'].shift(horizon).rolling(window).mean()
        df[f'residual_roll_std_{window}h'] = df[f'residual_h{horizon}'].shift(horizon).rolling(window).std()

    df['residual_trend_3h'] = df['residual_lag1'] - df['residual_lag3']

    return df


def get_stage2_features():
    """Get Stage 2 feature list."""
    return [
        'residual_lag1', 'residual_lag2', 'residual_lag3',
        'residual_roll_mean_3h', 'residual_roll_mean_6h',
        'residual_roll_std_3h', 'residual_roll_std_6h',
        'residual_trend_3h',
    ]


def train_and_evaluate(df, train_years, test_year=2026, test_month=1, finetune_days=90):
    """
    Train two-stage model and evaluate on test period.

    Args:
        df: Full dataframe with features
        train_years: List of years to use for training (e.g., [2025] or [2024, 2025])
        test_year: Year to test on
        test_month: Month to test on
        finetune_days: Number of days before test to give 3x weight
    """
    test_start = pd.Timestamp(f'{test_year}-{test_month:02d}-01')
    finetune_start = test_start - pd.Timedelta(days=finetune_days)

    # Compute seasonal error from training data only
    train_mask = df['year'].isin(train_years)
    train_subset = df[train_mask].copy()

    # Recreate features with training-only seasonal error
    df_model = create_features(df.copy(), train_data_only=train_subset)

    stage1_features = get_stage1_features()
    stage2_features = get_stage2_features()

    results = []

    for horizon in range(1, 6):
        target = f'target_h{horizon}'

        # Filter valid data
        df_valid = df_model.dropna(subset=[target] + stage1_features).copy()

        # ========== STAGE 1 ==========
        train_s1 = df_valid[df_valid['year'].isin(train_years)].copy()

        # Apply fine-tuning weights (3x for last 90 days)
        is_recent = train_s1['datetime'] >= finetune_start
        sample_weights = np.where(is_recent, 3.0, 1.0)

        model_s1 = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=8, num_leaves=50,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=-1
        )
        model_s1.fit(train_s1[stage1_features], train_s1[target], sample_weight=sample_weights)

        # Get Stage 1 predictions for all data
        df_valid['stage1_pred'] = model_s1.predict(df_valid[stage1_features])
        df_valid[f'residual_h{horizon}'] = df_valid[target] - df_valid['stage1_pred']

        # ========== STAGE 2 ==========
        df_valid = create_residual_features(df_valid, horizon)
        df_valid = df_valid.dropna(subset=stage2_features)

        # Train Stage 2 on second half of training period (where we have residuals)
        min_train_year = min(train_years)
        train_s2 = df_valid[
            (df_valid['year'].isin(train_years)) &
            (df_valid['datetime'] >= pd.Timestamp(f'{min_train_year}-07-01'))
        ].copy()

        # Apply fine-tuning weights to Stage 2 as well
        is_recent_s2 = train_s2['datetime'] >= finetune_start
        sample_weights_s2 = np.where(is_recent_s2, 3.0, 1.0)

        model_s2 = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, num_leaves=30,
            min_child_samples=20, random_state=42, verbosity=-1
        )
        model_s2.fit(train_s2[stage2_features], train_s2[f'residual_h{horizon}'],
                     sample_weight=sample_weights_s2)

        # ========== TEST ==========
        test = df_valid[
            (df_valid['year'] == test_year) &
            (df_valid['month'] == test_month)
        ].copy()

        test['residual_pred'] = model_s2.predict(test[stage2_features])
        test['stage2_pred'] = test['stage1_pred'] + test['residual_pred']

        # Metrics
        baseline_mae = np.abs(test[target]).mean()
        s1_mae = np.abs(test[target] - test['stage1_pred']).mean()
        s2_mae = np.abs(test[target] - test['stage2_pred']).mean()

        results.append({
            'horizon': horizon,
            'baseline_mae': baseline_mae,
            's1_mae': s1_mae,
            's2_mae': s2_mae,
            's2_improvement': (baseline_mae - s2_mae) / baseline_mae * 100,
            'n_train_s1': len(train_s1),
            'n_train_s2': len(train_s2),
            'n_test': len(test),
        })

    return results


def main():
    print("=" * 70)
    print("TRAINING DATA EXPERIMENT: 2025-only vs 2024+2025")
    print("=" * 70)
    print("\nTest period: January 2026")
    print("Fine-tuning: Last 90 days (Oct-Dec 2025) get 3x weight")

    # Load data
    df = load_data()

    # Check data availability
    print("\n[*] Data availability by year:")
    for year in [2024, 2025, 2026]:
        n = len(df[df['year'] == year])
        print(f"    {year}: {n:,} hours")

    jan_2026 = df[(df['year'] == 2026) & (df['month'] == 1)]
    print(f"\n    Jan 2026 test set: {len(jan_2026):,} hours")

    # ========== EXPERIMENT A: 2025 ONLY ==========
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Train on 2025 only")
    print("=" * 70)

    results_2025 = train_and_evaluate(df, train_years=[2025], test_year=2026, test_month=1)

    print(f"\n  {'H':<4} {'Baseline':<10} {'Stage2':<10} {'Improvement':<12} {'N_train':<10}")
    print("  " + "-" * 50)
    for r in results_2025:
        print(f"  H+{r['horizon']:<2} {r['baseline_mae']:<10.1f} {r['s2_mae']:<10.1f} "
              f"{r['s2_improvement']:+.1f}%       {r['n_train_s1']:,}")

    # ========== EXPERIMENT B: 2024 + 2025 ==========
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Train on 2024 + 2025")
    print("=" * 70)

    results_both = train_and_evaluate(df, train_years=[2024, 2025], test_year=2026, test_month=1)

    print(f"\n  {'H':<4} {'Baseline':<10} {'Stage2':<10} {'Improvement':<12} {'N_train':<10}")
    print("  " + "-" * 50)
    for r in results_both:
        print(f"  H+{r['horizon']:<2} {r['baseline_mae']:<10.1f} {r['s2_mae']:<10.1f} "
              f"{r['s2_improvement']:+.1f}%       {r['n_train_s1']:,}")

    # ========== COMPARISON ==========
    print("\n" + "=" * 70)
    print("COMPARISON: 2025-only vs 2024+2025")
    print("=" * 70)

    print(f"\n  {'H':<4} {'2025-only':<12} {'2024+2025':<12} {'Diff':<10} {'Winner':<10}")
    print("  " + "-" * 55)

    total_diff = 0
    for r25, rboth in zip(results_2025, results_both):
        diff = r25['s2_mae'] - rboth['s2_mae']
        total_diff += diff
        winner = "2024+2025" if diff > 0 else "2025-only" if diff < 0 else "TIE"
        print(f"  H+{r25['horizon']:<2} {r25['s2_mae']:<12.1f} {rboth['s2_mae']:<12.1f} "
              f"{diff:+.1f} MW    {winner}")

    avg_diff = total_diff / 5
    print("  " + "-" * 55)
    print(f"  {'AVG':<4} {np.mean([r['s2_mae'] for r in results_2025]):<12.1f} "
          f"{np.mean([r['s2_mae'] for r in results_both]):<12.1f} {avg_diff:+.1f} MW")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if avg_diff > 1:
        print(f"\n  [+] Adding 2024 data HELPS: Average {avg_diff:.1f} MW improvement")
        print("      Recommendation: Use 2024+2025 for training")
    elif avg_diff < -1:
        print(f"\n  [-] Adding 2024 data HURTS: Average {-avg_diff:.1f} MW degradation")
        print("      Recommendation: Use 2025 only for training")
    else:
        print(f"\n  [=] No significant difference: {abs(avg_diff):.1f} MW")
        print("      Recommendation: Either approach is fine, but 2024+2025 has more data")

    return results_2025, results_both


if __name__ == "__main__":
    results_2025, results_both = main()
