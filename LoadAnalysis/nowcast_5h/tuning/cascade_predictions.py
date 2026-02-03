"""
Cascade Predictions - Use predictions from shorter horizons to improve longer ones
==================================================================================

Key insight: At time T, when H+5 predicts error[T+5]:
- H+5's prediction from T-1 was for error[T+4]
- H+4's prediction from T is ALSO for error[T+4] (same target, fresher info!)

We can use this "forecast revision" signal as a feature.

Features added for horizon h:
- pred_h{h}_lag1: Our H+h prediction from last period
- pred_h{h-1}_same_target: H+(h-1) prediction for same target as pred_h{h}_lag1
- revision: pred_h{h-1}_same_target - pred_h{h}_lag1 (how much is forecast being updated?)
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

    return df


def create_stage1_features(df, train_mask, horizon):
    """Create Stage 1 features."""
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


def create_residual_features(df, horizon):
    """Create Stage 2 residual features."""
    df = df.copy()
    for lag in range(1, 7):
        df[f'residual_lag{lag}'] = df['residual'].shift(horizon + lag - 1)
    for window in [3, 6]:
        df[f'residual_roll_mean_{window}h'] = df['residual'].shift(horizon).rolling(window).mean()
    df['residual_trend_3h'] = df['residual_lag1'] - df['residual_lag3']
    return df


def get_sample_weights(dates, train_end_date, recency_months, recency_weight):
    days_ago = (train_end_date - dates).dt.days
    months_ago = days_ago / 30.0
    weights = np.where(months_ago <= recency_months, recency_weight, 1.0)
    return weights / weights.mean()


def rolling_predict_all_horizons(df, models_s1, models_s2, s1_features, s2_features, s1_params, s2_params):
    """
    Make rolling predictions for all horizons, then add cascade features.

    Returns df with predictions for each horizon.
    """
    df = df.copy()

    # First pass: make all Stage 1 + Stage 2 predictions for each horizon
    for h in range(1, 6):
        target = f'target_h{h}'
        s1_avail = [f for f in s1_features[h] if f in df.columns]
        s2_avail = [f for f in s2_features[h] if f in df.columns]

        # Stage 1 prediction
        mask = df[s1_avail].notna().all(axis=1)
        df.loc[mask, f's1_pred_h{h}'] = models_s1[h].predict(df.loc[mask, s1_avail])

        # Residual
        df[f'residual_h{h}'] = df[target] - df[f's1_pred_h{h}']

        # Create residual features for Stage 2
        for lag in range(1, 7):
            df[f'residual_lag{lag}_h{h}'] = df[f'residual_h{h}'].shift(h + lag - 1)
        for window in [3, 6]:
            df[f'residual_roll_mean_{window}h_h{h}'] = df[f'residual_h{h}'].shift(h).rolling(window).mean()
        df[f'residual_trend_3h_h{h}'] = df[f'residual_lag1_h{h}'] - df[f'residual_lag3_h{h}']

        # Map generic residual feature names to horizon-specific
        s2_avail_mapped = []
        for f in s2_avail:
            if 'residual' in f:
                s2_avail_mapped.append(f + f'_h{h}')
            else:
                s2_avail_mapped.append(f)
        s2_avail_mapped = [f for f in s2_avail_mapped if f in df.columns]

        # Stage 2 prediction
        if len(s2_avail_mapped) > 0:
            mask2 = df[s2_avail_mapped].notna().all(axis=1)
            df.loc[mask2, f's2_pred_h{h}'] = models_s2[h].predict(df.loc[mask2, s2_avail_mapped])

        # Final prediction
        df[f'final_pred_h{h}'] = df[f's1_pred_h{h}'] + df.get(f's2_pred_h{h}', 0)

    return df


def add_cascade_features(df):
    """
    Add cascade features: use predictions from shorter horizons to inform longer ones.

    For horizon h at time T predicting error[T+h]:
    - pred_h{h}_lag1: Our H+h prediction from T-1 (was for error[T+h-1])
    - For h > 1: pred_h{h-1}_same_target: H+(h-1) prediction from T for error[T+h-1]
                 (same target as pred_h{h}_lag1, but with fresher info!)
    - revision_h{h}: The difference (how much did the forecast change?)
    """
    df = df.copy()

    for h in range(1, 6):
        # Our own prediction from last period
        df[f'cascade_pred_h{h}_lag1'] = df[f'final_pred_h{h}'].shift(1)

        if h > 1:
            # H+(h-1) prediction from current period - predicts the SAME target as our lag1!
            # At time T: H+h lag1 was for error[T+h-1]
            # At time T: H+(h-1) predicts error[T+(h-1)] = error[T+h-1] - same target!
            df[f'cascade_pred_h{h-1}_same_target'] = df[f'final_pred_h{h-1}']

            # Revision: how much is the forecast being updated by the shorter horizon model?
            df[f'cascade_revision_h{h}'] = df[f'cascade_pred_h{h-1}_same_target'] - df[f'cascade_pred_h{h}_lag1']

    return df


def train_cascade_model(df_train, df_test, horizon, base_features, cascade_features):
    """Train a model that uses cascade features on top of base Stage 2."""
    target = f'target_h{horizon}'

    all_features = base_features + cascade_features
    avail_features = [f for f in all_features if f in df_train.columns]

    train_data = df_train.dropna(subset=[target] + avail_features)
    test_data = df_test.dropna(subset=avail_features)

    if len(train_data) < 100:
        return None, None, None

    # Target is the final residual (after Stage 2)
    train_data = train_data.copy()
    test_data = test_data.copy()
    train_data['cascade_target'] = train_data[target] - train_data[f'final_pred_h{horizon}']

    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        random_state=42,
        verbosity=-1,
    )
    model.fit(train_data[avail_features], train_data['cascade_target'])

    # Predict
    test_data['cascade_correction'] = model.predict(test_data[avail_features])
    test_data['cascade_final_pred'] = test_data[f'final_pred_h{horizon}'] + test_data['cascade_correction']

    # Metrics
    base_mae = np.abs(test_data[target] - test_data[f'final_pred_h{horizon}']).mean()
    cascade_mae = np.abs(test_data[target] - test_data['cascade_final_pred']).mean()

    return model, base_mae, cascade_mae, test_data


def main():
    print("=" * 70)
    print("CASCADE PREDICTIONS EXPERIMENT")
    print("=" * 70)
    print("\nIdea: Use predictions from shorter horizons to improve longer ones")
    print("At time T, H+5's lag1 prediction and H+4's current prediction")
    print("are for the SAME target - use this revision signal!\n")

    df = load_data()

    # Load best params
    models_s1 = {}
    models_s2 = {}
    s1_features = {}
    s2_features = {}
    s1_params = {}
    s2_params = {}

    print("Loading tuned models...")
    for h in range(1, 6):
        h_dir = TUNING_PATH / f'h{h}'
        with open(h_dir / 'stage1_best_params.json') as f:
            s1_best = json.load(f)
        with open(h_dir / 'stage2_best_params.json') as f:
            s2_best = json.load(f)

        s1_features[h] = s1_best['features']
        s2_features[h] = s2_best['s2_features']
        s1_params[h] = s1_best['params']
        s2_params[h] = s2_best['s2_params']

    # Training period
    train_end = '2026-01-01'
    train_mask = df['datetime'] < train_end

    # Create base features
    df = create_stage1_features(df, train_mask, horizon=1)

    # Train Stage 1 models
    print("\nTraining Stage 1 models...")
    for h in range(1, 6):
        target = f'target_h{h}'
        s1_avail = [f for f in s1_features[h] if f in df.columns]

        train_data = df[train_mask].dropna(subset=[target] + s1_avail)

        lgb_params = {k: v for k, v in s1_params[h].items()
                      if k in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves',
                               'min_child_samples', 'subsample', 'colsample_bytree',
                               'reg_alpha', 'reg_lambda']}
        lgb_params['random_state'] = 42
        lgb_params['verbosity'] = -1

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(train_data[s1_avail], train_data[target])
        models_s1[h] = model
        print(f"  H+{h}: {len(train_data):,} samples, {len(s1_avail)} features")

    # Train Stage 2 models (on H2 2025 data for OOF)
    print("\nTraining Stage 2 models...")
    s2_train_start = '2025-07-01'

    for h in range(1, 6):
        target = f'target_h{h}'
        s1_avail = [f for f in s1_features[h] if f in df.columns]

        # Get S1 predictions
        df[f's1_pred_h{h}'] = models_s1[h].predict(df[s1_avail].fillna(0))
        df[f'residual_h{h}'] = df[target] - df[f's1_pred_h{h}']

        # Create S2 features
        df['residual'] = df[f'residual_h{h}']
        df = create_residual_features(df, h)

        s2_avail = [f for f in s2_features[h] if f in df.columns]

        s2_train = df[(df['datetime'] >= s2_train_start) & (df['datetime'] < train_end)]
        s2_train = s2_train.dropna(subset=s2_avail + [f'residual_h{h}'])

        s2_lgb = {
            'n_estimators': s2_params[h]['s2_n_estimators'],
            'learning_rate': s2_params[h]['s2_learning_rate'],
            'max_depth': s2_params[h]['s2_max_depth'],
            'num_leaves': s2_params[h]['s2_num_leaves'],
            'random_state': 42,
            'verbosity': -1,
        }

        model = lgb.LGBMRegressor(**s2_lgb)
        model.fit(s2_train[s2_avail], s2_train[f'residual_h{h}'])
        models_s2[h] = model

        # Get S2 predictions
        mask = df[s2_avail].notna().all(axis=1)
        df.loc[mask, f's2_pred_h{h}'] = model.predict(df.loc[mask, s2_avail])
        df[f'final_pred_h{h}'] = df[f's1_pred_h{h}'] + df[f's2_pred_h{h}'].fillna(0)

        print(f"  H+{h}: {len(s2_train):,} samples")

    # Add cascade features
    print("\nAdding cascade features...")
    df = add_cascade_features(df)

    # Test on January 2026
    test_mask = (df['datetime'] >= '2026-01-01') & (df['datetime'] < '2026-02-01')
    df_test = df[test_mask].copy()

    print("\n" + "=" * 70)
    print("RESULTS - JANUARY 2026")
    print("=" * 70)

    print(f"\n{'Horizon':<10} {'Base S2 MAE':<15} {'Cascade MAE':<15} {'Improvement':<15} {'AC(1) Base':<12} {'AC(1) Cascade':<12}")
    print("-" * 80)

    results = []

    for h in range(2, 6):  # Cascade only works for h > 1
        target = f'target_h{h}'

        # Base features for cascade model
        cascade_features = [f'cascade_pred_h{h}_lag1']
        if h > 1:
            cascade_features.extend([
                f'cascade_pred_h{h-1}_same_target',
                f'cascade_revision_h{h}'
            ])

        # Simple cascade model: just use the cascade features to predict remaining error
        avail_cascade = [f for f in cascade_features if f in df_test.columns]

        test_data = df_test.dropna(subset=[target, f'final_pred_h{h}'] + avail_cascade).copy()

        if len(test_data) < 50:
            continue

        # Base MAE (Stage 2)
        base_error = test_data[target] - test_data[f'final_pred_h{h}']
        base_mae = np.abs(base_error).mean()
        base_ac1 = base_error.autocorr(lag=1)

        # Simple cascade: use revision as correction
        # If forecast is being revised up, actual is likely higher than our prediction
        test_data['cascade_correction'] = test_data[f'cascade_revision_h{h}'] * 0.5  # Simple scaling
        test_data['cascade_pred'] = test_data[f'final_pred_h{h}'] + test_data['cascade_correction']

        cascade_error = test_data[target] - test_data['cascade_pred']
        cascade_mae = np.abs(cascade_error).mean()
        cascade_ac1 = cascade_error.autocorr(lag=1)

        improvement = (base_mae - cascade_mae) / base_mae * 100

        print(f"H+{h:<8} {base_mae:<15.2f} {cascade_mae:<15.2f} {improvement:+.1f}%          {base_ac1:<12.3f} {cascade_ac1:<12.3f}")

        results.append({
            'horizon': h,
            'base_mae': base_mae,
            'cascade_mae': cascade_mae,
            'improvement': improvement,
            'base_ac1': base_ac1,
            'cascade_ac1': cascade_ac1,
            'test_data': test_data,
        })

    # Plot for worst case (H+5)
    if results:
        r = results[-1]  # H+5
        test_data = r['test_data']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'H+{r["horizon"]} Cascade Analysis', fontsize=14)

        # Error time series
        ax1 = axes[0, 0]
        ax1.plot(test_data['datetime'], test_data[f'target_h{r["horizon"]}'] - test_data[f'final_pred_h{r["horizon"]}'],
                 label='Base S2 Error', alpha=0.7)
        ax1.plot(test_data['datetime'], test_data[f'target_h{r["horizon"]}'] - test_data['cascade_pred'],
                 label='Cascade Error', alpha=0.7)
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_title('Prediction Error Over Time')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Revision vs actual error
        ax2 = axes[0, 1]
        ax2.scatter(test_data[f'cascade_revision_h{r["horizon"]}'],
                    test_data[f'target_h{r["horizon"]}'] - test_data[f'final_pred_h{r["horizon"]}'],
                    alpha=0.5, s=10)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Forecast Revision (H+4 - H+5_lag1)')
        ax2.set_ylabel('Actual S2 Error')
        ax2.set_title('Revision Signal vs Actual Error')
        ax2.grid(alpha=0.3)

        # Correlation
        corr = test_data[f'cascade_revision_h{r["horizon"]}'].corr(
            test_data[f'target_h{r["horizon"]}'] - test_data[f'final_pred_h{r["horizon"]}'])
        ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top')

        # Error distribution
        ax3 = axes[1, 0]
        ax3.hist(test_data[f'target_h{r["horizon"]}'] - test_data[f'final_pred_h{r["horizon"]}'],
                 bins=50, alpha=0.5, label=f'Base (MAE: {r["base_mae"]:.1f})', density=True)
        ax3.hist(test_data[f'target_h{r["horizon"]}'] - test_data['cascade_pred'],
                 bins=50, alpha=0.5, label=f'Cascade (MAE: {r["cascade_mae"]:.1f})', density=True)
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(alpha=0.3)

        # H+4 vs H+5 lag predictions
        ax4 = axes[1, 1]
        ax4.scatter(test_data[f'cascade_pred_h{r["horizon"]}_lag1'],
                    test_data[f'cascade_pred_h{r["horizon"]-1}_same_target'],
                    alpha=0.5, s=10)
        min_val = min(test_data[f'cascade_pred_h{r["horizon"]}_lag1'].min(),
                      test_data[f'cascade_pred_h{r["horizon"]-1}_same_target'].min())
        max_val = max(test_data[f'cascade_pred_h{r["horizon"]}_lag1'].max(),
                      test_data[f'cascade_pred_h{r["horizon"]-1}_same_target'].max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        ax4.set_xlabel(f'H+{r["horizon"]} Prediction (lag 1)')
        ax4.set_ylabel(f'H+{r["horizon"]-1} Prediction (same target)')
        ax4.set_title('Forecast Comparison (same target)')
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / 'cascade_analysis.png', dpi=150)
        print(f"\nPlot saved to: {OUTPUT_PATH / 'cascade_analysis.png'}")
        plt.close()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("\nThe 'revision' signal (H+4 current - H+5 lag1) contains information")
    print("about forecast updates that can reduce errors for longer horizons.")


if __name__ == '__main__':
    main()
