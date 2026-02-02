"""
Day-Ahead Model v2 - Correct Setup

At 23:00 on day D, predicting all 24 hours of day D+1:

AVAILABLE AT PREDICTION TIME:
1. Today's (D) complete data:
   - All 24 hours of actual load
   - All 24 hours of forecast
   - Therefore: ALL of today's forecast errors!

2. Tomorrow's (D+1) DAMAS forecast (known day-ahead)

3. 2024 seasonal pattern:
   - What we learned the load "should" be for each (hour, day_of_week)
   - Can compare DAMAS forecast to seasonal expectation

4. Historical error patterns from previous days
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Paths
BASE_PATH = Path(__file__).parent.parent
HOURLY_PATH = BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet'
MIN3_PATH = BASE_PATH / 'data' / 'features' / 'load_3min.csv'
PLOT_PATH = Path(__file__).parent / 'plots'


def load_data():
    """Load hourly data."""
    df = pd.read_parquet(HOURLY_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']
    df['date'] = df.index.date
    return df


def create_seasonal_pattern(df: pd.DataFrame, year: int = 2024) -> pd.DataFrame:
    """Create seasonal pattern from specified year."""
    df_year = df[df['year'] == year]

    seasonal = df_year.groupby(['day_of_week', 'hour']).agg(
        seasonal_load=('actual_load_mw', 'mean'),
        seasonal_load_std=('actual_load_mw', 'std'),
        seasonal_error=('error', 'mean'),  # systematic bias by hour/dow
    ).reset_index()

    print(f"Created seasonal pattern from {year}: {len(seasonal)} (dow, hour) combinations")
    return seasonal


def create_daily_dataset(df: pd.DataFrame, seasonal: pd.DataFrame) -> pd.DataFrame:
    """
    Create dataset where each row is a prediction for one hour of day D+1,
    made at 23:00 on day D.
    """
    print("\nCreating daily prediction dataset...")

    # Merge seasonal pattern
    df = df.reset_index().merge(seasonal, on=['day_of_week', 'hour'], how='left').set_index('datetime')

    # Deviation from seasonal
    df['forecast_vs_seasonal'] = df['forecast_load_mw'] - df['seasonal_load']
    df['actual_vs_seasonal'] = df['actual_load_mw'] - df['seasonal_load']

    dates = sorted(df['date'].unique())
    rows = []

    for i, target_date in enumerate(dates):
        if i == 0:
            continue

        today_date = dates[i-1]

        # Today's complete data (day D) - all available at prediction time!
        today = df[df['date'] == today_date].copy()
        if len(today) < 20:
            continue

        # Tomorrow's data (day D+1) - target
        tomorrow = df[df['date'] == target_date].copy()
        if len(tomorrow) < 20:
            continue

        # === TODAY'S (D) FEATURES - ALL KNOWN AT 23:00 ===

        # Today's errors by hour
        today_errors_by_hour = today.set_index('hour')['error'].to_dict()

        # Today's summary stats
        today_mean_error = today['error'].mean()
        today_std_error = today['error'].std()
        today_morning_error = today[today['hour'].between(6, 12)]['error'].mean()
        today_afternoon_error = today[today['hour'].between(13, 18)]['error'].mean()
        today_evening_error = today[today['hour'].between(19, 24)]['error'].mean()
        today_night_error = today[today['hour'].between(1, 5)]['error'].mean()

        # Today's actual vs seasonal
        today_vs_seasonal = today['actual_vs_seasonal'].mean()

        # Today's trend (was load increasing/decreasing through day?)
        today_load_trend = today['actual_load_mw'].iloc[-6:].mean() - today['actual_load_mw'].iloc[:6].mean()

        # Week ago data (D-7)
        week_ago_idx = i - 7
        if week_ago_idx >= 0:
            week_ago_date = dates[week_ago_idx]
            week_ago = df[df['date'] == week_ago_date]
            if len(week_ago) >= 20:
                week_ago_errors_by_hour = week_ago.set_index('hour')['error'].to_dict()
                week_ago_mean_error = week_ago['error'].mean()
            else:
                week_ago_errors_by_hour = {}
                week_ago_mean_error = 0
        else:
            week_ago_errors_by_hour = {}
            week_ago_mean_error = 0

        # === CREATE ROW FOR EACH HOUR OF TOMORROW (D+1) ===

        for _, row in tomorrow.iterrows():
            hour = row['hour']

            features = {
                'datetime': row.name,
                'date': target_date,
                'hour': hour,
                'dow': row['day_of_week'],
                'month': row['month'],
                'is_weekend': row['is_weekend'],

                # Target
                'actual': row['actual_load_mw'],
                'forecast': row['forecast_load_mw'],
                'target_error': row['error'],

                # Seasonal reference
                'seasonal_load': row['seasonal_load'],
                'seasonal_error': row['seasonal_error'],  # historical bias for this hour/dow
                'forecast_vs_seasonal': row['forecast_vs_seasonal'],

                # === TODAY'S (D) FEATURES ===

                # Today's same hour error
                'today_same_hour_error': today_errors_by_hour.get(hour, 0),

                # Today's adjacent hours errors (for same hour tomorrow)
                'today_prev_hour_error': today_errors_by_hour.get(hour - 1, today_errors_by_hour.get(24, 0)),
                'today_next_hour_error': today_errors_by_hour.get(hour + 1, today_errors_by_hour.get(1, 0)),

                # Today's overall stats
                'today_mean_error': today_mean_error,
                'today_std_error': today_std_error,
                'today_vs_seasonal': today_vs_seasonal,
                'today_load_trend': today_load_trend,

                # Today's period errors (morning, afternoon, evening, night)
                'today_morning_error': today_morning_error,
                'today_afternoon_error': today_afternoon_error,
                'today_evening_error': today_evening_error,
                'today_night_error': today_night_error,

                # Week ago same hour
                'week_ago_same_hour_error': week_ago_errors_by_hour.get(hour, 0),
                'week_ago_mean_error': week_ago_mean_error,
            }

            rows.append(features)

    result = pd.DataFrame(rows)
    print(f"Created {len(result)} prediction samples")
    return result


def train_model(df: pd.DataFrame):
    """Train and evaluate the day-ahead model."""
    print("\n" + "="*70)
    print("DAY-AHEAD MODEL V2 - Using Today's Complete Errors")
    print("="*70)

    feature_cols = [
        'hour', 'dow', 'month', 'is_weekend',
        'seasonal_error', 'forecast_vs_seasonal',
        'today_same_hour_error', 'today_prev_hour_error', 'today_next_hour_error',
        'today_mean_error', 'today_std_error', 'today_vs_seasonal', 'today_load_trend',
        'today_morning_error', 'today_afternoon_error', 'today_evening_error', 'today_night_error',
        'week_ago_same_hour_error', 'week_ago_mean_error',
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]
    df_valid = df.dropna(subset=feature_cols + ['target_error'])

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"Valid samples: {len(df_valid)}")

    # Split
    df_valid['date'] = pd.to_datetime(df_valid['date'])
    split_date = '2025-07-01'

    train = df_valid[df_valid['date'] < split_date]
    test = df_valid[df_valid['date'] >= split_date]

    print(f"\nTrain: {len(train)} ({train['date'].min()} to {train['date'].max()})")
    print(f"Test:  {len(test)} ({test['date'].min()} to {test['date'].max()})")

    X_train, y_train = train[feature_cols], train['target_error']
    X_test, y_test = test[feature_cols], test['target_error']

    # Train model
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    test = test.copy()
    test['pred_error'] = model.predict(X_test)
    test['corrected'] = test['forecast'] + test['pred_error']

    # Also try simple corrections
    test['seasonal_corrected'] = test['forecast'] + test['seasonal_error']
    test['today_same_hour_corrected'] = test['forecast'] + test['today_same_hour_error']

    # Results
    baseline_mae = (test['actual'] - test['forecast']).abs().mean()
    seasonal_mae = (test['actual'] - test['seasonal_corrected']).abs().mean()
    today_same_hour_mae = (test['actual'] - test['today_same_hour_corrected']).abs().mean()
    model_mae = (test['actual'] - test['corrected']).abs().mean()

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\n{'Model':<40} {'MAE (MW)':<12} {'vs Baseline':<12}")
    print("-"*65)
    print(f"{'Baseline (DAMAS forecast)':<40} {baseline_mae:<12.1f} {'-':<12}")
    print(f"{'+ Seasonal bias correction':<40} {seasonal_mae:<12.1f} {(1-seasonal_mae/baseline_mae)*100:+.1f}%")
    print(f"{'+ Today same hour error':<40} {today_same_hour_mae:<12.1f} {(1-today_same_hour_mae/baseline_mae)*100:+.1f}%")
    print(f"{'+ GB Model (all today features)':<40} {model_mae:<12.1f} {(1-model_mae/baseline_mae)*100:+.1f}%")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n--- Feature Importance ---")
    print(importance.to_string(index=False))

    # By hour
    print("\n--- MAE by Hour ---")
    hourly = test.groupby('hour').apply(lambda x: pd.Series({
        'baseline': (x['actual'] - x['forecast']).abs().mean(),
        'model': (x['actual'] - x['corrected']).abs().mean(),
    }))
    hourly['improvement'] = (1 - hourly['model'] / hourly['baseline']) * 100
    print(hourly.round(1).to_string())

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Model comparison
    models = ['Baseline', '+Seasonal', '+Today Same Hr', '+GB Model']
    maes = [baseline_mae, seasonal_mae, today_same_hour_mae, model_mae]
    colors = ['red', 'orange', 'yellow', 'green']
    axes[0, 0].bar(models, maes, color=colors, edgecolor='black')
    for i, v in enumerate(maes):
        axes[0, 0].text(i, v + 1, f'{v:.1f}', ha='center')
    axes[0, 0].set_ylabel('MAE (MW)')
    axes[0, 0].set_title('Day-Ahead Model Comparison')
    axes[0, 0].axhline(y=baseline_mae, color='red', linestyle='--', alpha=0.5)

    # 2. Feature importance
    top_n = min(12, len(importance))
    axes[0, 1].barh(range(top_n), importance['importance'].head(top_n), color='steelblue', edgecolor='black')
    axes[0, 1].set_yticks(range(top_n))
    axes[0, 1].set_yticklabels(importance['feature'].head(top_n), fontsize=9)
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].set_title('Feature Importance')

    # 3. Improvement by hour
    colors = ['green' if x > 0 else 'red' for x in hourly['improvement']]
    axes[0, 2].bar(hourly.index, hourly['improvement'], color=colors, edgecolor='black')
    axes[0, 2].set_xlabel('Hour')
    axes[0, 2].set_ylabel('Improvement (%)')
    axes[0, 2].set_title('Improvement by Hour')
    axes[0, 2].axhline(y=0, color='black')

    # 4. Sample week
    sample_start = test['datetime'].min() + pd.Timedelta(days=30)
    sample = test[(test['datetime'] >= sample_start) & (test['datetime'] < sample_start + pd.Timedelta(days=7))]
    axes[1, 0].plot(sample['datetime'], sample['actual'], 'b-', label='Actual', linewidth=1.5)
    axes[1, 0].plot(sample['datetime'], sample['forecast'], 'r--', label='Baseline', alpha=0.7)
    axes[1, 0].plot(sample['datetime'], sample['corrected'], 'g-', label='Corrected', alpha=0.7)
    axes[1, 0].legend()
    axes[1, 0].set_title('Sample Week')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. Error distribution
    axes[1, 1].hist((test['actual'] - test['forecast']), bins=50, alpha=0.5, label=f'Baseline ({baseline_mae:.1f})', color='red')
    axes[1, 1].hist((test['actual'] - test['corrected']), bins=50, alpha=0.5, label=f'Model ({model_mae:.1f})', color='green')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Error (MW)')
    axes[1, 1].set_title('Error Distribution')

    # 6. Predicted vs actual error
    r2 = np.corrcoef(test['pred_error'], test['target_error'])[0,1]**2
    axes[1, 2].scatter(test['pred_error'], test['target_error'], alpha=0.2, s=5)
    axes[1, 2].plot([-200, 200], [-200, 200], 'r--', linewidth=2)
    axes[1, 2].set_xlabel('Predicted Error')
    axes[1, 2].set_ylabel('Actual Error')
    axes[1, 2].set_title(f'Error Prediction (R²={r2:.3f})')
    axes[1, 2].set_xlim(-200, 200)
    axes[1, 2].set_ylim(-200, 200)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '23_dayahead_model_v2.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Saved: 23_dayahead_model_v2.png")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Day-Ahead Model V2 - Using Today's Complete Errors:

At 23:00 on day D, predicting day D+1 using:
- Today's (D) ALL 24 hours of forecast errors ✓
- Tomorrow's (D+1) DAMAS forecast ✓
- 2024 seasonal pattern ✓

Results:
  Baseline DAMAS:              {baseline_mae:.1f} MW MAE
  + Seasonal bias:             {seasonal_mae:.1f} MW ({(1-seasonal_mae/baseline_mae)*100:+.1f}%)
  + Today same hour error:     {today_same_hour_mae:.1f} MW ({(1-today_same_hour_mae/baseline_mae)*100:+.1f}%)
  + Full GB Model:             {model_mae:.1f} MW ({(1-model_mae/baseline_mae)*100:+.1f}%)

Top features:
  1. {importance.iloc[0]['feature']}: {importance.iloc[0]['importance']:.3f}
  2. {importance.iloc[1]['feature']}: {importance.iloc[1]['importance']:.3f}
  3. {importance.iloc[2]['feature']}: {importance.iloc[2]['importance']:.3f}
""")

    return model, test, importance


def main():
    df = load_data()
    seasonal = create_seasonal_pattern(df, year=2024)
    df_daily = create_daily_dataset(df, seasonal)
    model, results, importance = train_model(df_daily)


if __name__ == '__main__':
    main()
