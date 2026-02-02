"""
Simple Day-Ahead Model

Prediction setup:
- At 23:00 on day D-1, predict all 24 hours of day D
- Available: all data up to hour 23 of D-1
- Target: actual load for all hours of day D

Uses everything we learned:
- Baseline forecast (strong feature)
- Error patterns from previous days
- Calendar features
- 3-min volatility features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Paths
BASE_PATH = Path(__file__).parent.parent
HOURLY_PATH = BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet'
MIN3_PATH = BASE_PATH / 'data' / 'features' / 'load_3min.csv'
PLOT_PATH = Path(__file__).parent / 'plots'


def load_and_prepare_data():
    """Load and merge all data sources."""
    print("Loading data...")

    # Hourly data with forecasts
    df = pd.read_parquet(HOURLY_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']

    # 3-min data aggregated to hourly
    df_3min = pd.read_csv(MIN3_PATH)
    df_3min['datetime'] = pd.to_datetime(df_3min['datetime'])
    df_3min = df_3min.set_index('datetime')
    df_3min['hour_start'] = df_3min.index.floor('H')

    hourly_3min = df_3min.groupby('hour_start').agg(
        load_3min_mean=('load_mw', 'mean'),
        load_3min_std=('load_mw', 'std'),
        load_3min_trend=('load_mw', lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0)
    )

    df = df.join(hourly_3min, how='left')

    print(f"Data ready: {len(df)} hourly records")
    return df


def create_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features available at 23:00 on D-1 for predicting day D.
    """
    print("\nCreating daily features...")

    # Add date column
    df = df.copy()
    df['date'] = df.index.date

    # For each day, compute features from D-1 (available at prediction time)
    daily_features = []

    dates = df['date'].unique()

    for i, target_date in enumerate(dates):
        if i == 0:
            continue  # Skip first day (no history)

        prev_date = dates[i-1]

        # Data from previous day (D-1) - available at prediction time
        prev_day = df[df['date'] == prev_date]

        if len(prev_day) < 20:
            continue

        # Previous day errors by hour
        prev_errors = prev_day.groupby('hour')['error'].first().to_dict()

        # Previous day stats
        prev_day_stats = {
            'prev_day_mean_error': prev_day['error'].mean(),
            'prev_day_std_error': prev_day['error'].std(),
            'prev_day_mean_load': prev_day['actual_load_mw'].mean(),
            'prev_day_max_error': prev_day['error'].abs().max(),
            'prev_day_bias': prev_day['error'].mean(),  # systematic over/under
        }

        # Previous day 3-min volatility
        if 'load_3min_std' in prev_day.columns:
            prev_day_stats['prev_day_volatility'] = prev_day['load_3min_std'].mean()
            prev_day_stats['prev_day_trend'] = prev_day['load_3min_trend'].mean()

        # Week ago same day (if available)
        week_ago_idx = i - 7
        if week_ago_idx >= 0:
            week_ago_date = dates[week_ago_idx]
            week_ago_day = df[df['date'] == week_ago_date]
            if len(week_ago_day) >= 20:
                week_ago_errors = week_ago_day.groupby('hour')['error'].first().to_dict()
                prev_day_stats['week_ago_mean_error'] = week_ago_day['error'].mean()
            else:
                week_ago_errors = {}
                prev_day_stats['week_ago_mean_error'] = 0
        else:
            week_ago_errors = {}
            prev_day_stats['week_ago_mean_error'] = 0

        # Target day data
        target_day = df[df['date'] == target_date]

        if len(target_day) < 20:
            continue

        # Calendar features for target day
        target_dow = pd.Timestamp(target_date).dayofweek
        target_month = pd.Timestamp(target_date).month
        target_is_weekend = 1 if target_dow >= 5 else 0

        # Create row for each hour of target day
        for _, row in target_day.iterrows():
            hour = row['hour']

            features = {
                'datetime': row.name,
                'date': target_date,
                'hour': hour,
                'target_dow': target_dow,
                'target_month': target_month,
                'target_is_weekend': target_is_weekend,

                # Baseline forecast (known day-ahead)
                'forecast': row['forecast_load_mw'],

                # Target
                'actual': row['actual_load_mw'],
                'target_error': row['error'],

                # Previous day same hour error
                'prev_day_same_hour_error': prev_errors.get(hour, 0),

                # Week ago same hour error
                'week_ago_same_hour_error': week_ago_errors.get(hour, 0),

                **prev_day_stats
            }

            daily_features.append(features)

    result = pd.DataFrame(daily_features)
    print(f"Created {len(result)} samples with daily features")
    return result


def train_simple_model(df_features: pd.DataFrame):
    """Train a simple day-ahead error correction model."""
    print("\n" + "="*70)
    print("SIMPLE DAY-AHEAD MODEL")
    print("="*70)

    # Features for prediction
    feature_cols = [
        'hour', 'target_dow', 'target_month', 'target_is_weekend',
        'prev_day_same_hour_error', 'week_ago_same_hour_error',
        'prev_day_mean_error', 'prev_day_bias',
        'prev_day_volatility', 'prev_day_trend',
        'week_ago_mean_error'
    ]

    # Filter available features
    feature_cols = [c for c in feature_cols if c in df_features.columns]

    # Remove rows with NaN
    df_valid = df_features.dropna(subset=feature_cols + ['target_error', 'forecast', 'actual'])

    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    print(f"Valid samples: {len(df_valid)}")

    # Time-based train/test split
    df_valid['date'] = pd.to_datetime(df_valid['date'])
    split_date = '2025-07-01'

    train = df_valid[df_valid['date'] < split_date]
    test = df_valid[df_valid['date'] >= split_date]

    print(f"\nTrain: {len(train)} samples ({train['date'].min()} to {train['date'].max()})")
    print(f"Test:  {len(test)} samples ({test['date'].min()} to {test['date'].max()})")

    X_train = train[feature_cols]
    y_train = train['target_error']
    X_test = test[feature_cols]
    y_test = test['target_error']

    # Train model
    print("\nTraining Gradient Boosting model...")
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    pred_error = model.predict(X_test)

    # Corrected forecast
    test = test.copy()
    test['predicted_error'] = pred_error
    test['corrected_forecast'] = test['forecast'] + test['predicted_error']

    # Metrics
    baseline_mae = (test['actual'] - test['forecast']).abs().mean()
    baseline_rmse = np.sqrt(((test['actual'] - test['forecast'])**2).mean())

    corrected_mae = (test['actual'] - test['corrected_forecast']).abs().mean()
    corrected_rmse = np.sqrt(((test['actual'] - test['corrected_forecast'])**2).mean())

    # Also try simple approaches
    # 1. Use previous day same hour error as correction
    test['simple_correction'] = test['forecast'] + test['prev_day_same_hour_error']
    simple_mae = (test['actual'] - test['simple_correction']).abs().mean()

    # 2. Use week ago same hour error
    test['weekly_correction'] = test['forecast'] + test['week_ago_same_hour_error']
    weekly_mae = (test['actual'] - test['weekly_correction']).abs().mean()

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\n{'Model':<35} {'MAE (MW)':<12} {'RMSE (MW)':<12} {'vs Baseline':<12}")
    print("-"*70)
    print(f"{'Baseline (DAMAS forecast)':<35} {baseline_mae:<12.1f} {baseline_rmse:<12.1f} {'-':<12}")
    print(f"{'+ Yesterday same hour error':<35} {simple_mae:<12.1f} {'-':<12} {(1-simple_mae/baseline_mae)*100:+.1f}%")
    print(f"{'+ Week ago same hour error':<35} {weekly_mae:<12.1f} {'-':<12} {(1-weekly_mae/baseline_mae)*100:+.1f}%")
    print(f"{'+ GB Model (all features)':<35} {corrected_mae:<12.1f} {corrected_rmse:<12.1f} {(1-corrected_mae/baseline_mae)*100:+.1f}%")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n--- Feature Importance ---")
    print(importance.to_string(index=False))

    # Analyze by hour
    print("\n--- Performance by Hour ---")
    hourly_perf = test.groupby('hour').apply(lambda x: pd.Series({
        'baseline_mae': (x['actual'] - x['forecast']).abs().mean(),
        'corrected_mae': (x['actual'] - x['corrected_forecast']).abs().mean(),
    }))
    hourly_perf['improvement'] = (1 - hourly_perf['corrected_mae'] / hourly_perf['baseline_mae']) * 100
    print(hourly_perf.round(1).to_string())

    # Analyze by day of week
    print("\n--- Performance by Day of Week ---")
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_perf = test.groupby('target_dow').apply(lambda x: pd.Series({
        'baseline_mae': (x['actual'] - x['forecast']).abs().mean(),
        'corrected_mae': (x['actual'] - x['corrected_forecast']).abs().mean(),
    }))
    dow_perf['improvement'] = (1 - dow_perf['corrected_mae'] / dow_perf['baseline_mae']) * 100
    dow_perf.index = days
    print(dow_perf.round(1).to_string())

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Model comparison
    models = ['Baseline', '+Yesterday', '+Week Ago', '+GB Model']
    maes = [baseline_mae, simple_mae, weekly_mae, corrected_mae]
    colors = ['red', 'orange', 'yellow', 'green']
    axes[0, 0].bar(models, maes, color=colors, edgecolor='black', alpha=0.7)
    axes[0, 0].set_ylabel('MAE (MW)')
    axes[0, 0].set_title('Day-Ahead Model Comparison')
    axes[0, 0].axhline(y=baseline_mae, color='red', linestyle='--', alpha=0.5)
    for i, (m, v) in enumerate(zip(models, maes)):
        axes[0, 0].text(i, v + 1, f'{v:.1f}', ha='center', fontsize=10)

    # 2. Feature importance
    axes[0, 1].barh(range(len(importance)), importance['importance'], color='steelblue', edgecolor='black')
    axes[0, 1].set_yticks(range(len(importance)))
    axes[0, 1].set_yticklabels(importance['feature'], fontsize=9)
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].set_title('Feature Importance')

    # 3. Hourly improvement
    axes[0, 2].bar(hourly_perf.index, hourly_perf['improvement'],
                   color=['green' if x > 0 else 'red' for x in hourly_perf['improvement']],
                   edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Hour')
    axes[0, 2].set_ylabel('Improvement (%)')
    axes[0, 2].set_title('Improvement by Hour')
    axes[0, 2].axhline(y=0, color='black', linestyle='-')

    # 4. Time series sample (1 week)
    sample_start = test['datetime'].min() + pd.Timedelta(days=30)
    sample_end = sample_start + pd.Timedelta(days=7)
    sample = test[(test['datetime'] >= sample_start) & (test['datetime'] < sample_end)]

    axes[1, 0].plot(sample['datetime'], sample['actual'], 'b-', label='Actual', linewidth=1.5)
    axes[1, 0].plot(sample['datetime'], sample['forecast'], 'r--', label='Baseline', linewidth=1, alpha=0.7)
    axes[1, 0].plot(sample['datetime'], sample['corrected_forecast'], 'g-', label='Corrected', linewidth=1, alpha=0.7)
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Load (MW)')
    axes[1, 0].set_title('Sample Week: Actual vs Forecasts')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. Error distribution comparison
    baseline_errors = test['actual'] - test['forecast']
    corrected_errors = test['actual'] - test['corrected_forecast']
    axes[1, 1].hist(baseline_errors, bins=50, alpha=0.5, label=f'Baseline (MAE={baseline_mae:.1f})', color='red')
    axes[1, 1].hist(corrected_errors, bins=50, alpha=0.5, label=f'Corrected (MAE={corrected_mae:.1f})', color='green')
    axes[1, 1].set_xlabel('Error (MW)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].legend()

    # 6. Scatter: predicted vs actual error
    axes[1, 2].scatter(test['predicted_error'], test['target_error'], alpha=0.2, s=5)
    axes[1, 2].plot([-200, 200], [-200, 200], 'r--', linewidth=2)
    r2 = np.corrcoef(test['predicted_error'], test['target_error'])[0, 1]**2
    axes[1, 2].set_xlabel('Predicted Error (MW)')
    axes[1, 2].set_ylabel('Actual Error (MW)')
    axes[1, 2].set_title(f'Error Prediction (R^2={r2:.3f})')
    axes[1, 2].set_xlim(-200, 200)
    axes[1, 2].set_ylim(-200, 200)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '22_simple_dayahead_model.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Saved: 22_simple_dayahead_model.png")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Simple Day-Ahead Model Results:

  Baseline DAMAS forecast:     {baseline_mae:.1f} MW MAE
  + Yesterday same hour:       {simple_mae:.1f} MW MAE ({(1-simple_mae/baseline_mae)*100:+.1f}%)
  + Week ago same hour:        {weekly_mae:.1f} MW MAE ({(1-weekly_mae/baseline_mae)*100:+.1f}%)
  + Gradient Boosting:         {corrected_mae:.1f} MW MAE ({(1-corrected_mae/baseline_mae)*100:+.1f}%)

Key features:
  1. {importance.iloc[0]['feature']}: {importance.iloc[0]['importance']:.3f}
  2. {importance.iloc[1]['feature']}: {importance.iloc[1]['importance']:.3f}
  3. {importance.iloc[2]['feature']}: {importance.iloc[2]['importance']:.3f}

Without weather/prices, we achieve ~{(1-corrected_mae/baseline_mae)*100:.0f}% improvement.
Weather and prices should add another 10-20%!
""")

    return model, test, importance


def main():
    # Load data
    df = load_and_prepare_data()

    # Create daily features
    df_features = create_daily_features(df)

    # Train model
    model, results, importance = train_simple_model(df_features)


if __name__ == '__main__':
    main()
