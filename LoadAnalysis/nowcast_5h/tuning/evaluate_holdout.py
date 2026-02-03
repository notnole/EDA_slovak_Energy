"""
Evaluate Final Models on January 2026 Holdout
==============================================
Trains with best params on 2024+2025, evaluates on Jan 2026.
Creates individual error plots for each horizon.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import lightgbm as lgb
from datetime import datetime

BASE_PATH = Path(__file__).parent.parent.parent.parent
TUNING_PATH = Path(__file__).parent
OUTPUT_PATH = TUNING_PATH / 'holdout_evaluation'
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


def create_all_features(df, train_mask, horizon):
    """Create all features for Stage 1."""
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
    """Create Stage 2 residual features."""
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


def evaluate_horizon(df, horizon):
    """Evaluate a single horizon on January 2026 holdout."""
    print(f"\n{'='*60}")
    print(f"HORIZON H+{horizon}")
    print(f"{'='*60}")

    # Load best params
    h_dir = TUNING_PATH / f'h{horizon}'
    with open(h_dir / 'stage1_best_params.json') as f:
        s1_best = json.load(f)
    with open(h_dir / 'stage2_best_params.json') as f:
        s2_best = json.load(f)

    s1_features = s1_best['features']
    s1_params = s1_best['params']
    s2_features = s2_best['s2_features']
    s2_params = s2_best['s2_params']

    target = f'target_h{horizon}'

    # Training period: 2024 + 2025 (all data before Jan 2026)
    train_end = '2026-01-01'
    test_start = '2026-01-01'
    test_end = '2026-02-01'

    # For Stage 1: train on all 2024+2025
    # For Stage 2: need OOF residuals, so train S1 on 2024, get OOF on 2025
    s1_train_end = '2025-07-01'  # Train S1 on 2024 + H1 2025
    s2_train_start = '2025-07-01'  # S2 trains on H2 2025 residuals

    # Create features
    train_mask = df['datetime'] < train_end
    df_feat = create_all_features(df.copy(), train_mask, horizon)
    df_feat = df_feat.dropna(subset=[target] + [f for f in s1_features if f in df_feat.columns])

    # Stage 1 training data
    s1_train = df_feat[df_feat['datetime'] < s1_train_end]
    s1_avail = [f for f in s1_features if f in s1_train.columns]

    # Extract LightGBM params
    s1_lgb_params = {k: v for k, v in s1_params.items()
                     if k in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves',
                              'min_child_samples', 'subsample', 'colsample_bytree',
                              'reg_alpha', 'reg_lambda']}
    s1_lgb_params['random_state'] = 42
    s1_lgb_params['verbosity'] = -1

    # Sample weights for S1
    s1_weights = get_sample_weights(
        s1_train['datetime'],
        pd.to_datetime(s1_train_end),
        s1_params.get('recency_months', 3),
        s1_params.get('recency_weight', 2.0)
    )

    print(f"  Stage 1 training: {len(s1_train):,} rows")
    print(f"  Stage 1 features: {len(s1_avail)}")

    # Train Stage 1
    model_s1 = lgb.LGBMRegressor(**s1_lgb_params)
    model_s1.fit(s1_train[s1_avail], s1_train[target], sample_weight=s1_weights)

    # Get S1 predictions on S2 training period (OOF)
    s2_train_period = df_feat[(df_feat['datetime'] >= s2_train_start) &
                               (df_feat['datetime'] < train_end)].copy()
    s2_train_period['s1_pred'] = model_s1.predict(s2_train_period[s1_avail])
    s2_train_period['residual'] = s2_train_period[target] - s2_train_period['s1_pred']

    # Create residual features
    s2_train_period = create_residual_features(s2_train_period, horizon)
    s2_avail = [f for f in s2_features if f in s2_train_period.columns]
    s2_train_data = s2_train_period.dropna(subset=s2_avail)

    print(f"  Stage 2 training: {len(s2_train_data):,} rows")
    print(f"  Stage 2 features: {len(s2_avail)}")

    # Extract S2 LightGBM params
    s2_lgb_params = {
        'n_estimators': s2_params['s2_n_estimators'],
        'learning_rate': s2_params['s2_learning_rate'],
        'max_depth': s2_params['s2_max_depth'],
        'num_leaves': s2_params['s2_num_leaves'],
        'min_child_samples': s2_params['s2_min_child_samples'],
        'subsample': s2_params['s2_subsample'],
        'colsample_bytree': s2_params['s2_colsample_bytree'],
        'reg_alpha': s2_params['s2_reg_alpha'],
        'reg_lambda': s2_params['s2_reg_lambda'],
        'random_state': 42,
        'verbosity': -1,
    }

    # Sample weights for S2
    s2_weights = get_sample_weights(
        s2_train_data['datetime'],
        pd.to_datetime(train_end),
        s2_params.get('s2_recency_months', 2),
        s2_params.get('s2_recency_weight', 1.5)
    )

    # Train Stage 2
    model_s2 = lgb.LGBMRegressor(**s2_lgb_params)
    model_s2.fit(s2_train_data[s2_avail], s2_train_data['residual'], sample_weight=s2_weights)

    # ========== EVALUATE ON JANUARY 2026 ==========
    test_data = df_feat[(df_feat['datetime'] >= test_start) &
                         (df_feat['datetime'] < test_end)].copy()

    # Stage 1 predictions
    test_data['s1_pred'] = model_s1.predict(test_data[s1_avail])
    test_data['residual'] = test_data[target] - test_data['s1_pred']

    # Create residual features for test
    test_data = create_residual_features(test_data, horizon)
    test_data = test_data.dropna(subset=s2_avail)

    # Stage 2 predictions
    test_data['s2_pred'] = model_s2.predict(test_data[s2_avail])
    test_data['final_pred'] = test_data['s1_pred'] + test_data['s2_pred']

    # Errors
    test_data['s1_error'] = test_data[target] - test_data['s1_pred']
    test_data['s2_error'] = test_data[target] - test_data['final_pred']
    test_data['baseline_error'] = test_data[target]  # Predicting 0

    # Metrics
    baseline_mae = np.abs(test_data['baseline_error']).mean()
    s1_mae = np.abs(test_data['s1_error']).mean()
    s2_mae = np.abs(test_data['s2_error']).mean()

    print(f"\n  January 2026 Results ({len(test_data)} hours):")
    print(f"    Baseline (predict 0): {baseline_mae:.2f} MW")
    print(f"    Stage 1:              {s1_mae:.2f} MW ({(baseline_mae-s1_mae)/baseline_mae*100:+.1f}%)")
    print(f"    Stage 2:              {s2_mae:.2f} MW ({(baseline_mae-s2_mae)/baseline_mae*100:+.1f}%)")
    print(f"    S2 gain over S1:      {(s1_mae-s2_mae)/s1_mae*100:+.1f}%")

    return {
        'horizon': horizon,
        'test_data': test_data,
        'baseline_mae': baseline_mae,
        's1_mae': s1_mae,
        's2_mae': s2_mae,
        'n_test': len(test_data),
    }


def plot_horizon_errors(result, output_path):
    """Create detailed error analysis plot for a single horizon."""
    h = result['horizon']
    df = result['test_data'].copy()
    df = df.sort_values('datetime')

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'H+{h} Error Analysis - January 2026 Holdout', fontsize=14, fontweight='bold')

    # 1. Time series of actual vs predicted
    ax1 = axes[0, 0]
    ax1.plot(df['datetime'], df[f'target_h{h}'], label='Actual Error', alpha=0.7, linewidth=0.8)
    ax1.plot(df['datetime'], df['final_pred'], label='S2 Prediction', alpha=0.7, linewidth=0.8)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('DAMAS Error (MW)')
    ax1.set_title('Actual vs Predicted DAMAS Error')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax1.grid(alpha=0.3)

    # 2. Prediction error over time
    ax2 = axes[0, 1]
    ax2.plot(df['datetime'], df['s2_error'], alpha=0.7, linewidth=0.8, color='red')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.fill_between(df['datetime'], df['s2_error'], 0, alpha=0.3, color='red')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Prediction Error (MW)')
    ax2.set_title(f'Stage 2 Prediction Error (MAE: {result["s2_mae"]:.1f} MW)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax2.grid(alpha=0.3)

    # 3. Error distribution
    ax3 = axes[1, 0]
    ax3.hist(df['baseline_error'], bins=50, alpha=0.5, label=f'Baseline (MAE:{result["baseline_mae"]:.1f})', density=True)
    ax3.hist(df['s1_error'], bins=50, alpha=0.5, label=f'Stage 1 (MAE:{result["s1_mae"]:.1f})', density=True)
    ax3.hist(df['s2_error'], bins=50, alpha=0.5, label=f'Stage 2 (MAE:{result["s2_mae"]:.1f})', density=True)
    ax3.set_xlabel('Error (MW)')
    ax3.set_ylabel('Density')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Error by hour of day
    ax4 = axes[1, 1]
    hourly_mae = df.groupby('hour').agg({
        'baseline_error': lambda x: np.abs(x).mean(),
        's1_error': lambda x: np.abs(x).mean(),
        's2_error': lambda x: np.abs(x).mean(),
    })
    ax4.plot(hourly_mae.index, hourly_mae['baseline_error'], marker='o', label='Baseline', alpha=0.7)
    ax4.plot(hourly_mae.index, hourly_mae['s1_error'], marker='s', label='Stage 1', alpha=0.7)
    ax4.plot(hourly_mae.index, hourly_mae['s2_error'], marker='^', label='Stage 2', alpha=0.7)
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('MAE (MW)')
    ax4.set_title('MAE by Hour of Day')
    ax4.set_xticks(range(0, 24, 2))
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Error by day of week
    ax5 = axes[2, 0]
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_mae = df.groupby('dow').agg({
        'baseline_error': lambda x: np.abs(x).mean(),
        's1_error': lambda x: np.abs(x).mean(),
        's2_error': lambda x: np.abs(x).mean(),
    })
    x = np.arange(len(dow_mae))
    width = 0.25
    ax5.bar(x - width, dow_mae['baseline_error'], width, label='Baseline', alpha=0.8)
    ax5.bar(x, dow_mae['s1_error'], width, label='Stage 1', alpha=0.8)
    ax5.bar(x + width, dow_mae['s2_error'], width, label='Stage 2', alpha=0.8)
    ax5.set_xlabel('Day of Week')
    ax5.set_ylabel('MAE (MW)')
    ax5.set_title('MAE by Day of Week')
    ax5.set_xticks(x)
    ax5.set_xticklabels([dow_names[i] for i in dow_mae.index])
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    # 6. Scatter: Actual vs Predicted
    ax6 = axes[2, 1]
    ax6.scatter(df[f'target_h{h}'], df['final_pred'], alpha=0.5, s=10)
    min_val = min(df[f'target_h{h}'].min(), df['final_pred'].min())
    max_val = max(df[f'target_h{h}'].max(), df['final_pred'].max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Perfect prediction')
    ax6.set_xlabel('Actual Error (MW)')
    ax6.set_ylabel('Predicted Error (MW)')
    ax6.set_title('Actual vs Predicted')
    ax6.legend()
    ax6.grid(alpha=0.3)

    plt.tight_layout()

    # Save
    plot_path = output_path / f'h{h}_error_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {plot_path}")

    plt.close()


def main():
    print("=" * 70)
    print("JANUARY 2026 HOLDOUT EVALUATION")
    print("=" * 70)

    df = load_data()

    all_results = []

    for h in range(1, 6):
        result = evaluate_horizon(df, h)
        all_results.append(result)

        # Save predictions
        pred_path = OUTPUT_PATH / f'h{h}_predictions.parquet'
        result['test_data'].to_parquet(pred_path)
        print(f"  Predictions saved: {pred_path}")

        # Create plot
        plot_horizon_errors(result, OUTPUT_PATH)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - JANUARY 2026 HOLDOUT")
    print("=" * 70)

    print(f"\n{'Horizon':<10} {'Baseline':<12} {'S1 MAE':<12} {'S2 MAE':<12} {'S1 Improv':<12} {'S2 Improv':<12}")
    print("-" * 70)

    for r in all_results:
        s1_imp = (r['baseline_mae'] - r['s1_mae']) / r['baseline_mae'] * 100
        s2_imp = (r['baseline_mae'] - r['s2_mae']) / r['baseline_mae'] * 100
        print(f"H+{r['horizon']:<8} {r['baseline_mae']:<12.1f} {r['s1_mae']:<12.1f} {r['s2_mae']:<12.1f} "
              f"{s1_imp:+.1f}%       {s2_imp:+.1f}%")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'test_period': 'January 2026',
        'results': {r['horizon']: {
            'baseline_mae': r['baseline_mae'],
            's1_mae': r['s1_mae'],
            's2_mae': r['s2_mae'],
            'n_test': r['n_test'],
        } for r in all_results}
    }
    with open(OUTPUT_PATH / 'holdout_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
