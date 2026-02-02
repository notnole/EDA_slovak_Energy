"""
Final Day-Ahead Model - Clean Implementation

Key finding: Simpler is better. The core features from v2 model
perform best. Additional features (momentum, price, week stats)
add noise and hurt performance.

Core features that work:
1. forecast_vs_seasonal - How unusual is tomorrow's forecast?
2. d1_evening_error - Today's evening forecast errors
3. d1_load_vs_seasonal - Was today's load unusual?
4. seasonal_error - Historical bias for this hour/dow
5. d1_same_hour_error - Today's same hour error
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Paths
BASE_PATH = Path(__file__).parent.parent
LOAD_PATH = BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet'
PRICE_PATH = BASE_PATH / 'features' / 'DamasPrices' / 'da_prices.parquet'
PLOT_PATH = Path(__file__).parent / 'plots'


def load_data():
    """Load data."""
    df = pd.read_parquet(LOAD_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['load_error'] = df['actual_load_mw'] - df['forecast_load_mw']
    df = df.sort_values('datetime').reset_index(drop=True)

    # Optionally load prices for analysis
    try:
        df_price = pd.read_parquet(PRICE_PATH)
        df_price['datetime'] = pd.to_datetime(df_price['datetime'])
        df = df.merge(df_price[['datetime', 'price_eur_mwh', 'net_import']],
                      on='datetime', how='left')
    except:
        pass

    print(f"Loaded {len(df)} records")
    return df


def create_seasonal(df: pd.DataFrame) -> dict:
    """Create seasonal patterns from 2024."""
    df_2024 = df[df['year'] == 2024]
    return {
        'load': df_2024.groupby(['day_of_week', 'hour'])['actual_load_mw'].mean(),
        'error': df_2024.groupby(['day_of_week', 'hour'])['load_error'].mean(),
    }


def build_dataset(df: pd.DataFrame, seasonal: dict) -> pd.DataFrame:
    """Build minimal feature dataset."""
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    dates = sorted(df['date'].unique())

    rows = []

    for i, target_date in enumerate(dates):
        if i == 0:
            continue

        today_date = dates[i-1]
        today = df[df['date'] == today_date]
        tomorrow = df[df['date'] == target_date]

        if len(today) < 20 or len(tomorrow) < 20:
            continue

        # Today's data
        today_errors = today.set_index('hour')['load_error'].to_dict()
        today_mean_error = today['load_error'].mean()
        today_evening_err = today[today['hour'].between(18, 23)]['load_error'].mean()
        today_morning_err = today[today['hour'].between(6, 11)]['load_error'].mean()
        today_night_err = today[today['hour'].between(0, 5)]['load_error'].mean()

        # Today vs seasonal
        today_dow = today['day_of_week'].iloc[0]
        today_seasonal_load = np.mean([seasonal['load'].get((today_dow, h), 0) for h in range(24)])
        today_load_vs_seasonal = today['actual_load_mw'].mean() - today_seasonal_load

        for _, row in tomorrow.iterrows():
            hour = row['hour']
            dow = row['day_of_week']

            seasonal_load = seasonal['load'].get((dow, hour), row['actual_load_mw'])
            seasonal_error = seasonal['error'].get((dow, hour), 0)

            features = {
                'datetime': row['datetime'],
                'date': target_date,
                'hour': hour,
                'dow': dow,
                'is_weekend': row['is_weekend'],

                # Targets
                'actual': row['actual_load_mw'],
                'forecast': row['forecast_load_mw'],
                'target_error': row['load_error'],

                # === CORE FEATURES ===
                'seasonal_error': seasonal_error,
                'forecast_vs_seasonal': row['forecast_load_mw'] - seasonal_load,
                'd1_same_hour_error': today_errors.get(hour, 0),
                'd1_mean_error': today_mean_error,
                'd1_evening_error': today_evening_err,
                'd1_morning_error': today_morning_err,
                'd1_night_error': today_night_err,
                'd1_load_vs_seasonal': today_load_vs_seasonal,
            }

            rows.append(features)

    return pd.DataFrame(rows)


def train_and_evaluate(df: pd.DataFrame):
    """Train and evaluate model."""
    print("\n" + "=" * 70)
    print("FINAL DAY-AHEAD MODEL")
    print("=" * 70)

    features = [
        'hour', 'dow', 'is_weekend',
        'seasonal_error', 'forecast_vs_seasonal',
        'd1_same_hour_error', 'd1_mean_error',
        'd1_evening_error', 'd1_morning_error', 'd1_night_error',
        'd1_load_vs_seasonal',
    ]

    df['date'] = pd.to_datetime(df['date'])
    df_valid = df.dropna(subset=features + ['target_error'])

    split_date = '2025-07-01'
    train = df_valid[df_valid['date'] < split_date]
    test = df_valid[df_valid['date'] >= split_date]

    print(f"\nTrain: {len(train)} ({train['date'].min()} to {train['date'].max()})")
    print(f"Test:  {len(test)} ({test['date'].min()} to {test['date'].max()})")

    # Train
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(train[features], train['target_error'])

    # Predict
    test = test.copy()
    test['pred_error'] = model.predict(test[features])
    test['corrected'] = test['forecast'] + test['pred_error']

    # Results
    baseline_mae = (test['actual'] - test['forecast']).abs().mean()
    model_mae = (test['actual'] - test['corrected']).abs().mean()

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Baseline (DAMAS):  {baseline_mae:.1f} MW MAE")
    print(f"Final Model:       {model_mae:.1f} MW MAE ({(1-model_mae/baseline_mae)*100:+.1f}%)")
    print(f"{'='*50}")

    # Cross-validation
    print("\nCross-validation (5-fold time series):")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_improvements = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_valid)):
        fold_train = df_valid.iloc[train_idx]
        fold_val = df_valid.iloc[val_idx]

        fold_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, random_state=42
        )
        fold_model.fit(fold_train[features], fold_train['target_error'])

        pred = fold_model.predict(fold_val[features])
        corrected = fold_val['forecast'] + pred

        bl_mae = (fold_val['actual'] - fold_val['forecast']).abs().mean()
        md_mae = (fold_val['actual'] - corrected).abs().mean()
        improvement = (1 - md_mae / bl_mae) * 100
        cv_improvements.append(improvement)

        print(f"  Fold {fold+1}: {improvement:+.1f}%")

    print(f"  Mean: {np.mean(cv_improvements):+.1f}% +/- {np.std(cv_improvements):.1f}%")

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    for _, row in importance.iterrows():
        print(f"  {row['feature']:<25}: {row['importance']:.3f}")

    # By hour
    hourly = test.groupby('hour').apply(lambda x: pd.Series({
        'baseline': (x['actual'] - x['forecast']).abs().mean(),
        'model': (x['actual'] - x['corrected']).abs().mean(),
    }))
    hourly['improvement'] = (1 - hourly['model'] / hourly['baseline']) * 100

    print(f"\nHourly improvement:")
    print(f"  Best:  Hour {hourly['improvement'].idxmax()} ({hourly['improvement'].max():+.1f}%)")
    print(f"  Worst: Hour {hourly['improvement'].idxmin()} ({hourly['improvement'].min():+.1f}%)")
    print(f"  Mean:  {hourly['improvement'].mean():+.1f}%")

    return model, test, importance, hourly, features


def create_plots(test, importance, hourly):
    """Create final plots."""
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    baseline_mae = (test['actual'] - test['forecast']).abs().mean()
    model_mae = (test['actual'] - test['corrected']).abs().mean()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Model comparison
    models = ['Baseline\n(DAMAS)', 'Final\nModel']
    maes = [baseline_mae, model_mae]
    colors = ['#e74c3c', '#27ae60']
    bars = axes[0, 0].bar(models, maes, color=colors, edgecolor='black', linewidth=1.5)
    for bar, v in zip(bars, maes):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f}',
                        ha='center', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('MAE (MW)', fontsize=12)
    axes[0, 0].set_title(f'Model Comparison\n({(1-model_mae/baseline_mae)*100:+.1f}% improvement)', fontsize=12)
    axes[0, 0].axhline(baseline_mae, color='red', linestyle='--', alpha=0.3)

    # 2. Feature importance
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance)))[::-1]
    axes[0, 1].barh(range(len(importance)), importance['importance'], color=colors, edgecolor='black')
    axes[0, 1].set_yticks(range(len(importance)))
    axes[0, 1].set_yticklabels(importance['feature'], fontsize=10)
    axes[0, 1].set_xlabel('Importance', fontsize=12)
    axes[0, 1].set_title('Feature Importance', fontsize=12)

    # 3. Improvement by hour
    colors = ['#27ae60' if x > 0 else '#e74c3c' for x in hourly['improvement']]
    axes[0, 2].bar(hourly.index, hourly['improvement'], color=colors, edgecolor='black')
    axes[0, 2].axhline(0, color='black', linewidth=1)
    axes[0, 2].axhline(hourly['improvement'].mean(), color='blue', linestyle='--',
                       label=f"Mean: {hourly['improvement'].mean():.1f}%", linewidth=2)
    axes[0, 2].set_xlabel('Hour', fontsize=12)
    axes[0, 2].set_ylabel('Improvement (%)', fontsize=12)
    axes[0, 2].set_title('Improvement by Hour', fontsize=12)
    axes[0, 2].legend(fontsize=10)

    # 4. Sample week
    start = test['datetime'].min() + pd.Timedelta(days=30)
    sample = test[(test['datetime'] >= start) & (test['datetime'] < start + pd.Timedelta(days=7))]
    axes[1, 0].plot(sample['datetime'], sample['actual'], 'b-', label='Actual', linewidth=2)
    axes[1, 0].plot(sample['datetime'], sample['forecast'], 'r--', label='Baseline', alpha=0.7, linewidth=1.5)
    axes[1, 0].plot(sample['datetime'], sample['corrected'], 'g-', label='Model', alpha=0.8, linewidth=1.5)
    axes[1, 0].fill_between(sample['datetime'], sample['actual'], sample['corrected'],
                            alpha=0.2, color='green')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].set_title('Sample Week Forecast', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylabel('Load (MW)', fontsize=12)

    # 5. Error distribution
    baseline_err = test['actual'] - test['forecast']
    model_err = test['actual'] - test['corrected']
    axes[1, 1].hist(baseline_err, bins=50, alpha=0.6, label=f'Baseline (MAE={baseline_mae:.1f})',
                    color='red', edgecolor='darkred')
    axes[1, 1].hist(model_err, bins=50, alpha=0.6, label=f'Model (MAE={model_mae:.1f})',
                    color='green', edgecolor='darkgreen')
    axes[1, 1].axvline(0, color='black', linewidth=2)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].set_xlabel('Forecast Error (MW)', fontsize=12)
    axes[1, 1].set_title('Error Distribution', fontsize=12)

    # 6. Scatter: Predicted vs Actual Error
    r2 = np.corrcoef(test['pred_error'], test['target_error'])[0, 1] ** 2
    axes[1, 2].scatter(test['pred_error'], test['target_error'], alpha=0.15, s=8, c='steelblue')
    axes[1, 2].plot([-200, 200], [-200, 200], 'r--', linewidth=2, label='Perfect')
    axes[1, 2].set_xlabel('Predicted Error (MW)', fontsize=12)
    axes[1, 2].set_ylabel('Actual Error (MW)', fontsize=12)
    axes[1, 2].set_title(f'Error Prediction Quality (R$^2$={r2:.3f})', fontsize=12)
    axes[1, 2].set_xlim(-180, 180)
    axes[1, 2].set_ylim(-300, 300)
    axes[1, 2].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '27_final_model.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: 27_final_model.png")


def main():
    df = load_data()
    seasonal = create_seasonal(df)
    df_dataset = build_dataset(df, seasonal)

    print(f"Dataset: {len(df_dataset)} samples")

    model, test, importance, hourly, features = train_and_evaluate(df_dataset)
    create_plots(test, importance, hourly)

    # Final summary
    baseline_mae = (test['actual'] - test['forecast']).abs().mean()
    model_mae = (test['actual'] - test['corrected']).abs().mean()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"""
Day-Ahead Load Forecast Error Correction Model

Setup:
  - At 23:00 on day D, predict errors for all 24 hours of day D+1
  - Uses today's (D) complete load data and DAMAS forecast errors
  - Uses 2024 seasonal patterns as reference

Results:
  Baseline (DAMAS):  {baseline_mae:.1f} MW MAE
  Final Model:       {model_mae:.1f} MW MAE
  Improvement:       {(1-model_mae/baseline_mae)*100:+.1f}%

Key Features (ordered by importance):
  1. forecast_vs_seasonal  - Is tomorrow's forecast unusual?
  2. d1_evening_error      - Today's evening errors
  3. d1_load_vs_seasonal   - Was today's load unusual?
  4. seasonal_error        - Historical bias for hour/dow
  5. d1_same_hour_error    - Today's same-hour error

What didn't help:
  - Price features (add noise, don't improve)
  - Momentum/trend features (hurt performance)
  - Week-long statistics (too much noise)
  - Lagged errors from days 2-7 (not useful)

Conclusion:
  The DAMAS baseline is strong. Best improvements come from:
  - Comparing to seasonal expectations (2024 patterns)
  - Using today's complete error patterns
  - Keeping the model simple (fewer features = less noise)
""")


if __name__ == '__main__':
    main()
