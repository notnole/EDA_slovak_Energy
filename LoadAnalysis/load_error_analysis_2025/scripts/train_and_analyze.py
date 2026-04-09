"""
Two-Stage Model Training for H2 2025 Error Analysis
====================================================
Training setup:
- Stage 1 training: 2024 (full year)
- Stage 2 training: H1 2025 (Jan-Jun) using OOF Stage 1 residuals
- Evaluation: H2 2025 (Jul-Dec) for detailed error analysis

Stage 2 is trained on truly out-of-fold predictions since Stage 1
never sees 2025 data during training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent  # ipesoft_eda_data
OUTPUT_PATH = Path(__file__).parent.parent  # load_error_analysis_2025
MODELS_PATH = OUTPUT_PATH / 'models'
DATA_PATH = OUTPUT_PATH / 'data'
PLOTS_PATH = OUTPUT_PATH / 'plots'

for p in [MODELS_PATH, DATA_PATH, PLOTS_PATH]:
    p.mkdir(exist_ok=True)


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

    # Check 2025 data
    df_2025 = df[df['year'] == 2025]
    print(f"    2025 data: {len(df_2025):,} rows")
    if len(df_2025) > 0:
        print(f"    2025 range: {df_2025['datetime'].min()} to {df_2025['datetime'].max()}")

    return df


def create_stage1_features(df, train_mask):
    """
    Create features for Stage 1 (DAMAS error prediction).
    seasonal_error computed ONLY from training data.
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

    # seasonal_error computed ONLY from training data (2024)
    train_data = df[train_mask]
    seasonal_means = train_data.groupby(['dow', 'hour'])['error'].mean()
    df['seasonal_error'] = df.set_index(['dow', 'hour']).index.map(
        lambda x: seasonal_means.get(x, 0)
    ).values

    print(f"    seasonal_error computed from {len(train_data):,} training rows only")

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
    Create features from Stage 1 residuals.
    For horizon h, shift residuals by h to avoid leakage.
    """
    df = df.copy()

    # Shift by horizon to avoid leakage
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
    """
    Train two-stage model for H2 2025 evaluation.

    Time splits:
    - Stage 1 training: 2024 (full year)
    - Stage 2 training: H1 2025 (Jan-Jun) using OOF Stage 1 residuals
    - Evaluation: H2 2025 (Jul-Dec)
    """
    target = f'target_h{horizon}'
    stage1_features = get_stage1_features()

    # Define training mask for seasonal error (2024 only)
    train_2024_mask = df['year'] == 2024

    # Create features
    df_model = create_stage1_features(df.copy(), train_2024_mask)

    stage1_avail = [f for f in stage1_features if f in df_model.columns]
    df_model = df_model.dropna(subset=[target] + stage1_avail).copy()

    # ========== STAGE 1 ==========
    # Train on full 2024
    train_s1 = df_model[df_model['year'] == 2024]

    print(f"    Stage 1 training on 2024: {len(train_s1):,} rows")

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
    # Train on H1 2025 residuals - these are OOF since Stage 1 never saw 2025
    df_model = create_residual_features(df_model, horizon)

    stage2_features = get_stage2_features()
    stage2_avail = [f for f in stage2_features if f in df_model.columns]

    # H1 2025 for Stage 2 training (OOF residuals)
    train_s2 = df_model[(df_model['year'] == 2025) & (df_model['month'] <= 6)]
    train_s2 = train_s2.dropna(subset=stage2_avail)

    print(f"    Stage 2 training on H1 2025 (OOF residuals): {len(train_s2):,} rows")

    # H2 2025 for evaluation
    test = df_model[(df_model['year'] == 2025) & (df_model['month'] > 6)]
    test = test.dropna(subset=stage2_avail).copy()

    print(f"    Testing on H2 2025: {len(test):,} rows")

    if len(test) == 0:
        print("    [!] WARNING: No test data for H2 2025!")
        return None

    if len(train_s2) == 0:
        print("    [!] WARNING: No training data for H1 2025!")
        return None

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

    # Direction accuracy
    actual_sign = np.sign(test[target])
    pred_sign = np.sign(test['stage2_pred'])
    direction_acc = (actual_sign == pred_sign).mean() * 100

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
        'direction_acc': direction_acc,
        'model_s1': model_s1,
        'model_s2': model_s2,
        'n_train_s1': len(train_s1),
        'n_train_s2': len(train_s2),
        'n_test': len(test),
        'test_data': test,
    }


def analyze_errors(test_data, horizon):
    """Detailed error analysis for the test set."""
    df = test_data.copy()
    target = f'target_h{horizon}'

    analysis = {}

    # Overall stats
    analysis['overall'] = {
        'mae': np.abs(df[target] - df['stage2_pred']).mean(),
        'rmse': np.sqrt(np.mean((df[target] - df['stage2_pred'])**2)),
        'mape': np.abs((df[target] - df['stage2_pred']) / df[target].replace(0, np.nan)).mean() * 100,
        'bias': (df['stage2_pred'] - df[target]).mean(),
        'n_samples': len(df),
    }

    # By hour
    hourly = df.groupby('hour').apply(lambda x: pd.Series({
        'mae': np.abs(x[target] - x['stage2_pred']).mean(),
        'bias': (x['stage2_pred'] - x[target]).mean(),
        'n': len(x),
    }))
    analysis['by_hour'] = hourly.to_dict('index')

    # By day of week
    dow = df.groupby('dow').apply(lambda x: pd.Series({
        'mae': np.abs(x[target] - x['stage2_pred']).mean(),
        'bias': (x['stage2_pred'] - x[target]).mean(),
        'n': len(x),
    }))
    analysis['by_dow'] = dow.to_dict('index')

    # By month
    monthly = df.groupby('month').apply(lambda x: pd.Series({
        'mae': np.abs(x[target] - x['stage2_pred']).mean(),
        'bias': (x['stage2_pred'] - x[target]).mean(),
        'n': len(x),
    }))
    analysis['by_month'] = monthly.to_dict('index')

    # Worst days
    daily = df.groupby(df['datetime'].dt.date).apply(lambda x: pd.Series({
        'mae': np.abs(x[target] - x['stage2_pred']).mean(),
        'max_error': np.abs(x[target] - x['stage2_pred']).max(),
        'bias': (x['stage2_pred'] - x[target]).mean(),
    }))
    worst_days = daily.nlargest(10, 'mae')
    analysis['worst_days'] = worst_days.to_dict('index')

    # Error distribution
    errors = df[target] - df['stage2_pred']
    analysis['distribution'] = {
        'mean': errors.mean(),
        'std': errors.std(),
        'median': errors.median(),
        'p5': errors.quantile(0.05),
        'p25': errors.quantile(0.25),
        'p75': errors.quantile(0.75),
        'p95': errors.quantile(0.95),
        'skewness': errors.skew(),
        'kurtosis': errors.kurtosis(),
    }

    return analysis


def create_error_plots(test_data, horizon, output_path):
    """Create error analysis plots."""
    import matplotlib.pyplot as plt

    df = test_data.copy()
    target = f'target_h{horizon}'
    errors = df[target] - df['stage2_pred']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'H+{horizon} Error Analysis - H2 2025', fontsize=14, fontweight='bold')

    # 1. Error time series
    ax = axes[0, 0]
    ax.plot(df['datetime'], errors, alpha=0.5, linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Error (MW)')
    ax.set_title('Error Time Series')
    ax.tick_params(axis='x', rotation=45)

    # 2. Error distribution
    ax = axes[0, 1]
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    ax.axvline(x=errors.mean(), color='g', linestyle='--', alpha=0.7, label=f'Mean: {errors.mean():.1f}')
    ax.set_xlabel('Error (MW)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()

    # 3. By hour
    ax = axes[0, 2]
    hourly_mae = df.groupby('hour').apply(lambda x: np.abs(x[target] - x['stage2_pred']).mean())
    ax.bar(hourly_mae.index, hourly_mae.values)
    ax.set_xlabel('Hour')
    ax.set_ylabel('MAE (MW)')
    ax.set_title('MAE by Hour')
    ax.set_xticks(range(0, 24, 2))

    # 4. Actual vs Predicted
    ax = axes[1, 0]
    ax.scatter(df[target], df['stage2_pred'], alpha=0.3, s=5)
    lims = [min(df[target].min(), df['stage2_pred'].min()),
            max(df[target].max(), df['stage2_pred'].max())]
    ax.plot(lims, lims, 'r--', alpha=0.7)
    ax.set_xlabel('Actual Error (MW)')
    ax.set_ylabel('Predicted Error (MW)')
    ax.set_title('Actual vs Predicted')

    # 5. By day of week
    ax = axes[1, 1]
    dow_mae = df.groupby('dow').apply(lambda x: np.abs(x[target] - x['stage2_pred']).mean())
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.bar(range(7), dow_mae.values)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('MAE (MW)')
    ax.set_title('MAE by Day of Week')
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)

    # 6. By month
    ax = axes[1, 2]
    monthly_mae = df.groupby('month').apply(lambda x: np.abs(x[target] - x['stage2_pred']).mean())
    month_names = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    months_present = [m for m in range(7, 13) if m in monthly_mae.index]
    ax.bar(range(len(months_present)), [monthly_mae[m] for m in months_present])
    ax.set_xlabel('Month')
    ax.set_ylabel('MAE (MW)')
    ax.set_title('MAE by Month (H2 2025)')
    ax.set_xticks(range(len(months_present)))
    ax.set_xticklabels([month_names[m-7] for m in months_present])

    plt.tight_layout()
    plt.savefig(output_path / f'error_analysis_h{horizon}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    [+] Saved plot: error_analysis_h{horizon}.png")


def main():
    print("=" * 70)
    print("TWO-STAGE MODEL - H2 2025 ERROR ANALYSIS")
    print("=" * 70)
    print("\nTraining setup:")
    print("  - Stage 1 training: 2024 (full year)")
    print("  - Stage 2 training: H1 2025 (Jan-Jun) with OOF residuals")
    print("  - Evaluation: H2 2025 (Jul-Dec)")

    # Load data
    df = load_data()

    # Check data availability
    h1_2025 = df[(df['year'] == 2025) & (df['month'] <= 6)]
    h2_2025 = df[(df['year'] == 2025) & (df['month'] > 6)]
    print(f"\n    H1 2025 (Stage 2 training): {len(h1_2025):,} rows")
    print(f"    H2 2025 (Evaluation): {len(h2_2025):,} rows")

    if len(h2_2025) == 0:
        print("\n    [!] ERROR: No H2 2025 data available!")
        print("    Available months in 2025:", sorted(df[df['year'] == 2025]['month'].unique()))
        return None, None

    # Train for all horizons
    results = []
    models = {}
    all_test_data = {}

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for h in range(1, 6):
        print(f"\n  H+{h}...")
        result = train_two_stage_model(df, horizon=h)

        if result is None:
            continue

        results.append(result)
        models[h] = {'s1': result['model_s1'], 's2': result['model_s2']}
        all_test_data[h] = result['test_data']

        print(f"      Baseline:  {result['baseline_mae']:.1f} MW")
        print(f"      Stage 1:   {result['s1_mae']:.1f} MW ({result['s1_improvement']:+.1f}%)")
        print(f"      Stage 2:   {result['s2_mae']:.1f} MW ({result['s2_improvement']:+.1f}%)")
        print(f"      S2 Gain:   {result['stage2_gain']:+.1f}%")
        print(f"      Direction: {result['direction_acc']:.1f}%")

    if not results:
        print("\n  [!] No results - check data availability")
        return None, None

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY - H2 2025 EVALUATION")
    print("=" * 70)
    print(f"\n  {'H':<4} {'Baseline':<10} {'Stage1':<10} {'Stage2':<10} {'S1 Improv':<11} {'S2 Improv':<11} {'S2 Gain':<8}")
    print("  " + "-" * 75)

    for r in results:
        print(f"  H+{r['horizon']:<2} {r['baseline_mae']:<10.1f} {r['s1_mae']:<10.1f} {r['s2_mae']:<10.1f} "
              f"{r['s1_improvement']:+.1f}%      {r['s2_improvement']:+.1f}%      {r['stage2_gain']:+.1f}%")

    # Error analysis
    print("\n" + "=" * 70)
    print("DETAILED ERROR ANALYSIS")
    print("=" * 70)

    all_analysis = {}
    for h in range(1, 6):
        if h not in all_test_data:
            continue

        print(f"\n  H+{h} Error Analysis:")
        analysis = analyze_errors(all_test_data[h], h)
        all_analysis[h] = analysis

        print(f"    Overall MAE:  {analysis['overall']['mae']:.1f} MW")
        print(f"    Overall RMSE: {analysis['overall']['rmse']:.1f} MW")
        print(f"    Bias:         {analysis['overall']['bias']:+.1f} MW")

        # Worst hours
        hourly = analysis['by_hour']
        worst_hours = sorted(hourly.items(), key=lambda x: x[1]['mae'], reverse=True)[:3]
        print(f"    Worst hours:  {', '.join([f'H{h}={v['mae']:.0f}MW' for h, v in worst_hours])}")

        # Create plots
        create_error_plots(all_test_data[h], h, PLOTS_PATH)

    # Save models
    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS")
    print("=" * 70)

    for h, m in models.items():
        joblib.dump(m['s1'], MODELS_PATH / f'stage1_h{h}.joblib')
        joblib.dump(m['s2'], MODELS_PATH / f'stage2_h{h}.joblib')
        print(f"    [+] Saved models for H+{h}")

    # Save test predictions
    for h, test_df in all_test_data.items():
        cols_to_save = ['datetime', 'hour', 'dow', 'month',
                        f'target_h{h}', 'stage1_pred', 'stage2_pred', 'final_residual']
        cols_avail = [c for c in cols_to_save if c in test_df.columns]
        test_df[cols_avail].to_parquet(DATA_PATH / f'predictions_h{h}.parquet')
        print(f"    [+] Saved predictions for H+{h}")

    # Save analysis results
    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {str(k): convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    analysis_json = convert_types(all_analysis)
    with open(DATA_PATH / 'error_analysis.json', 'w') as f:
        json.dump(analysis_json, f, indent=2, default=str)
    print(f"    [+] Saved error analysis JSON")

    # Save summary
    summary = {
        'training_setup': {
            'stage1_training': '2024 (full year)',
            'stage2_training': 'H1 2025 (Jan-Jun) - OOF residuals',
            'evaluation': 'H2 2025 (Jul-Dec)',
        },
        'results': [{
            'horizon': r['horizon'],
            'baseline_mae': float(r['baseline_mae']),
            's1_mae': float(r['s1_mae']),
            's2_mae': float(r['s2_mae']),
            's1_improvement_pct': float(r['s1_improvement']),
            's2_improvement_pct': float(r['s2_improvement']),
            'stage2_gain_pct': float(r['stage2_gain']),
            'direction_acc_pct': float(r['direction_acc']),
            'n_train_s1': r['n_train_s1'],
            'n_train_s2': r['n_train_s2'],
            'n_test': r['n_test'],
        } for r in results],
    }
    with open(DATA_PATH / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"    [+] Saved training summary")

    print(f"\n    All artifacts saved to: {OUTPUT_PATH}")

    return results, all_analysis


if __name__ == "__main__":
    results, analysis = main()
