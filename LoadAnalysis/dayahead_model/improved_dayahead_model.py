"""
Improved Day-Ahead Model

Key fix: Use HOUR-BY-HOUR features instead of daily aggregates.

The original model used d1_load_vs_seasonal as a single daily average,
losing 24x granularity. This version uses hour-specific comparisons.

Based on diagnosis from 5-hour nowcasting model results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Paths
BASE_PATH = Path(__file__).parent.parent.parent  # Go up to ipesoft_eda_data
LOAD_PATH = BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet'
PLOT_PATH = Path(__file__).parent / 'plots'


def load_data():
    """Load data."""
    df = pd.read_parquet(LOAD_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['year'] = df['datetime'].dt.year
    df['load_error'] = df['actual_load_mw'] - df['forecast_load_mw']
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"Loaded {len(df):,} records")
    return df


def create_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    """Create seasonal patterns from 2024 (hour-by-hour, not daily)."""
    df_2024 = df[df['year'] == 2024]

    seasonal = df_2024.groupby(['day_of_week', 'hour']).agg({
        'actual_load_mw': 'mean',
        'forecast_load_mw': 'mean',
        'load_error': 'mean'
    }).reset_index()
    seasonal.columns = ['day_of_week', 'hour', 'seasonal_load', 'seasonal_forecast', 'seasonal_error']

    # Merge to main df
    df = df.merge(seasonal, on=['day_of_week', 'hour'], how='left')

    # KEY FIX: Hour-specific deviation (not daily average!)
    df['actual_vs_seasonal'] = df['actual_load_mw'] - df['seasonal_load']
    df['forecast_vs_seasonal'] = df['forecast_load_mw'] - df['seasonal_forecast']

    print(f"Created seasonal from {len(df_2024):,} training records")
    return df


def build_improved_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build dataset with HOUR-SPECIFIC features (the key fix).

    Instead of daily aggregates, use:
    - Yesterday's same-hour deviation
    - Yesterday's same-hour error
    - Hour-specific rolling stats
    """
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    dates = sorted(df['date'].unique())

    rows = []

    for i, target_date in enumerate(dates):
        if i == 0:
            continue

        yesterday_date = dates[i-1]
        yesterday = df[df['date'] == yesterday_date]
        tomorrow = df[df['date'] == target_date]

        if len(yesterday) < 20 or len(tomorrow) < 20:
            continue

        # Create hour-indexed dictionaries for yesterday
        yesterday_by_hour = yesterday.set_index('hour')

        # Yesterday's hour-specific data
        yesterday_errors = yesterday_by_hour['load_error'].to_dict()
        yesterday_actual_vs_seasonal = yesterday_by_hour['actual_vs_seasonal'].to_dict()
        yesterday_load = yesterday_by_hour['actual_load_mw'].to_dict()

        # Yesterday's rolling stats by time period
        yesterday_morning = yesterday[yesterday['hour'].between(6, 11)]
        yesterday_afternoon = yesterday[yesterday['hour'].between(12, 17)]
        yesterday_evening = yesterday[yesterday['hour'].between(18, 23)]
        yesterday_night = yesterday[yesterday['hour'].between(0, 5)]

        # Get 2-day-ago data if available
        d2_errors = {}
        d2_actual_vs_seasonal = {}
        if i >= 2:
            d2_date = dates[i-2]
            d2 = df[df['date'] == d2_date]
            if len(d2) >= 20:
                d2_by_hour = d2.set_index('hour')
                d2_errors = d2_by_hour['load_error'].to_dict()
                d2_actual_vs_seasonal = d2_by_hour['actual_vs_seasonal'].to_dict()

        for _, row in tomorrow.iterrows():
            hour = row['hour']
            dow = row['day_of_week']

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

                # === KEY FIX: Hour-specific features ===
                # Yesterday's same-hour error
                'd1_same_hour_error': yesterday_errors.get(hour, 0),

                # KEY: Yesterday's same-hour deviation from seasonal (NOT daily average!)
                'd1_same_hour_vs_seasonal': yesterday_actual_vs_seasonal.get(hour, 0),

                # Adjacent hours for context
                'd1_prev_hour_error': yesterday_errors.get((hour - 1) % 24, 0),
                'd1_next_hour_error': yesterday_errors.get((hour + 1) % 24, 0),
                'd1_prev_hour_vs_seasonal': yesterday_actual_vs_seasonal.get((hour - 1) % 24, 0),

                # === Seasonal features ===
                'seasonal_error': row['seasonal_error'],
                'forecast_vs_seasonal': row['forecast_vs_seasonal'],

                # === Period-based features (for smoothing) ===
                'd1_morning_error': yesterday_morning['load_error'].mean() if len(yesterday_morning) > 0 else 0,
                'd1_afternoon_error': yesterday_afternoon['load_error'].mean() if len(yesterday_afternoon) > 0 else 0,
                'd1_evening_error': yesterday_evening['load_error'].mean() if len(yesterday_evening) > 0 else 0,
                'd1_night_error': yesterday_night['load_error'].mean() if len(yesterday_night) > 0 else 0,

                # === Error persistence (rolling stats) ===
                'd1_mean_error': yesterday['load_error'].mean(),
                'd1_std_error': yesterday['load_error'].std(),
                'd1_max_error': yesterday['load_error'].max(),
                'd1_min_error': yesterday['load_error'].min(),

                # === 2-day-ago features for longer persistence ===
                'd2_same_hour_error': d2_errors.get(hour, 0),
                'd2_same_hour_vs_seasonal': d2_actual_vs_seasonal.get(hour, 0),

                # === Trend features ===
                'd1_d2_error_diff': yesterday_errors.get(hour, 0) - d2_errors.get(hour, 0),
            }

            rows.append(features)

    return pd.DataFrame(rows)


def train_and_evaluate(df: pd.DataFrame):
    """Train and evaluate improved model."""
    print("\n" + "=" * 70)
    print("IMPROVED DAY-AHEAD MODEL (Hour-Specific Features)")
    print("=" * 70)

    # KEY: Using hour-specific features instead of daily aggregates
    features = [
        'hour', 'dow', 'is_weekend',
        # Hour-specific (THE FIX)
        'd1_same_hour_error',
        'd1_same_hour_vs_seasonal',  # KEY NEW FEATURE
        'd1_prev_hour_error',
        'd1_next_hour_error',
        'd1_prev_hour_vs_seasonal',
        # Seasonal
        'seasonal_error',
        'forecast_vs_seasonal',
        # Period-based
        'd1_morning_error',
        'd1_afternoon_error',
        'd1_evening_error',
        'd1_night_error',
        # Rolling stats
        'd1_mean_error',
        'd1_std_error',
        # 2-day patterns
        'd2_same_hour_error',
        'd2_same_hour_vs_seasonal',
        'd1_d2_error_diff',
    ]

    df['date'] = pd.to_datetime(df['date'])
    df_valid = df.dropna(subset=features + ['target_error'])

    split_date = '2025-07-01'
    train = df_valid[df_valid['date'] < split_date]
    test = df_valid[df_valid['date'] >= split_date]

    print(f"\nTrain: {len(train):,} ({train['date'].min()} to {train['date'].max()})")
    print(f"Test:  {len(test):,} ({test['date'].min()} to {test['date'].max()})")

    # Train with LightGBM (better than GradientBoosting)
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    model.fit(train[features], train['target_error'])

    # Predict
    test = test.copy()
    test['pred_error'] = model.predict(test[features])
    test['corrected'] = test['forecast'] + test['pred_error']

    # Results
    baseline_mae = (test['actual'] - test['forecast']).abs().mean()
    model_mae = (test['actual'] - test['corrected']).abs().mean()
    improvement = (1 - model_mae / baseline_mae) * 100

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Baseline (DAMAS):     {baseline_mae:.1f} MW MAE")
    print(f"Improved Model:       {model_mae:.1f} MW MAE ({improvement:+.1f}%)")
    print(f"Original Model:       62.4 MW MAE (+2.9%)")
    print(f"{'='*50}")

    # Cross-validation
    print("\nCross-validation (5-fold time series):")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_improvements = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_valid)):
        fold_train = df_valid.iloc[train_idx]
        fold_val = df_valid.iloc[val_idx]

        fold_model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            num_leaves=31, min_child_samples=30, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbose=-1
        )
        fold_model.fit(fold_train[features], fold_train['target_error'])

        pred = fold_model.predict(fold_val[features])
        corrected = fold_val['forecast'] + pred

        bl_mae = (fold_val['actual'] - fold_val['forecast']).abs().mean()
        md_mae = (fold_val['actual'] - corrected).abs().mean()
        fold_improvement = (1 - md_mae / bl_mae) * 100
        cv_improvements.append(fold_improvement)

        print(f"  Fold {fold+1}: {fold_improvement:+.1f}%")

    print(f"  Mean: {np.mean(cv_improvements):+.1f}% +/- {np.std(cv_improvements):.1f}%")

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance (Top 10):")
    for i, row in importance.head(10).iterrows():
        print(f"  {row['feature']:<30}: {row['importance']:.3f}")

    # By hour analysis
    hourly = test.groupby('hour').apply(lambda x: pd.Series({
        'baseline': (x['actual'] - x['forecast']).abs().mean(),
        'model': (x['actual'] - x['corrected']).abs().mean(),
    }))
    hourly['improvement'] = (1 - hourly['model'] / hourly['baseline']) * 100

    print(f"\nHourly improvement:")
    print(f"  Best:  Hour {hourly['improvement'].idxmax()} ({hourly['improvement'].max():+.1f}%)")
    print(f"  Worst: Hour {hourly['improvement'].idxmin()} ({hourly['improvement'].min():+.1f}%)")
    print(f"  Mean:  {hourly['improvement'].mean():+.1f}%")

    return model, test, importance, hourly, features, cv_improvements


def create_comparison_plot(test, importance, hourly):
    """Create comparison plot."""
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    baseline_mae = (test['actual'] - test['forecast']).abs().mean()
    model_mae = (test['actual'] - test['corrected']).abs().mean()
    original_model_mae = 62.4  # From original model

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Model comparison (Original vs Improved)
    models = ['DAMAS\nBaseline', 'Original\nModel', 'Improved\nModel']
    maes = [baseline_mae, original_model_mae, model_mae]
    improvements = [0, 2.9, (1 - model_mae/baseline_mae)*100]
    colors = ['#e74c3c', '#f39c12', '#27ae60']

    bars = axes[0, 0].bar(models, maes, color=colors, edgecolor='black', linewidth=1.5)
    for bar, v, imp in zip(bars, maes, improvements):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, v + 0.5,
                        f'{v:.1f}\n({imp:+.1f}%)', ha='center', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('MAE (MW)', fontsize=12)
    axes[0, 0].set_title('Model Comparison: Original vs Improved', fontsize=12, fontweight='bold')
    axes[0, 0].axhline(baseline_mae, color='red', linestyle='--', alpha=0.3)

    # 2. Feature importance
    top_features = importance.head(12)
    colors_feat = ['#27ae60' if 'same_hour' in f or 'vs_seasonal' in f else '#3498db'
                   for f in top_features['feature']]
    axes[0, 1].barh(range(len(top_features)), top_features['importance'],
                    color=colors_feat, edgecolor='black')
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features['feature'], fontsize=9)
    axes[0, 1].set_xlabel('Importance', fontsize=12)
    axes[0, 1].set_title('Feature Importance\n(Green = Hour-specific features)', fontsize=12)
    axes[0, 1].invert_yaxis()

    # 3. Improvement by hour
    colors_hour = ['#27ae60' if x > 0 else '#e74c3c' for x in hourly['improvement']]
    axes[1, 0].bar(hourly.index, hourly['improvement'], color=colors_hour, edgecolor='black')
    axes[1, 0].axhline(0, color='black', linewidth=1)
    axes[1, 0].axhline(hourly['improvement'].mean(), color='blue', linestyle='--',
                       label=f"Mean: {hourly['improvement'].mean():.1f}%", linewidth=2)
    axes[1, 0].axhline(2.9, color='orange', linestyle=':',
                       label=f"Original: +2.9%", linewidth=2)
    axes[1, 0].set_xlabel('Hour', fontsize=12)
    axes[1, 0].set_ylabel('Improvement (%)', fontsize=12)
    axes[1, 0].set_title('Improvement by Hour', fontsize=12)
    axes[1, 0].legend(fontsize=10)

    # 4. Scatter plot
    r2 = np.corrcoef(test['pred_error'], test['target_error'])[0, 1] ** 2
    axes[1, 1].scatter(test['pred_error'], test['target_error'], alpha=0.1, s=8, c='steelblue')
    axes[1, 1].plot([-200, 200], [-200, 200], 'r--', linewidth=2, label='Perfect')
    axes[1, 1].set_xlabel('Predicted Error (MW)', fontsize=12)
    axes[1, 1].set_ylabel('Actual Error (MW)', fontsize=12)
    axes[1, 1].set_title(f'Error Prediction Quality (R²={r2:.3f})', fontsize=12)
    axes[1, 1].set_xlim(-180, 180)
    axes[1, 1].set_ylim(-300, 300)
    axes[1, 1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '28_improved_dayahead_model.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: 28_improved_dayahead_model.png")


def main():
    df = load_data()
    df = create_seasonal(df)
    df_dataset = build_improved_dataset(df)

    print(f"Dataset: {len(df_dataset):,} samples")

    model, test, importance, hourly, features, cv_improvements = train_and_evaluate(df_dataset)
    create_comparison_plot(test, importance, hourly)

    baseline_mae = (test['actual'] - test['forecast']).abs().mean()
    model_mae = (test['actual'] - test['corrected']).abs().mean()
    improvement = (1 - model_mae / baseline_mae) * 100

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"""
What Was Wrong With the Original Day-Ahead Model:
=================================================
The original model used d1_load_vs_seasonal as a DAILY AVERAGE,
collapsing 24 hourly deviations into 1 number.

The Fix:
========
Use HOUR-SPECIFIC features:
- d1_same_hour_vs_seasonal (yesterday's hour H deviation)
- d1_same_hour_error (yesterday's hour H error)
- d1_prev_hour_error (for context)

Results:
========
Original Model:  +2.9% improvement (62.4 MW MAE)
Improved Model:  {improvement:+.1f}% improvement ({model_mae:.1f} MW MAE)

The hour-specific features show up as top importance:
{importance.head(5).to_string()}

Conclusion:
===========
By preserving hour-by-hour granularity instead of daily averaging,
we achieved {improvement:.1f}% improvement vs the original 2.9%.
This validates the hypothesis from the 5-hour model analysis.
""")


if __name__ == '__main__':
    main()
