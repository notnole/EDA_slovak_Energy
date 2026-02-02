"""
Test Model Variations:
1. Add raw DAMAS forecast as feature
2. Compare single model vs 24 hourly models
3. Compare different train/test splits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Paths
BASE_PATH = Path(__file__).parent.parent.parent
LOAD_PATH = BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet'
PLOT_PATH = Path(__file__).parent


def load_data():
    """Load data."""
    df = pd.read_parquet(LOAD_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['load_error'] = df['actual_load_mw'] - df['forecast_load_mw']
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    return df


def create_seasonal(df: pd.DataFrame) -> dict:
    """Create seasonal patterns from 2024."""
    df_2024 = df[df['year'] == 2024]
    return {
        'load': df_2024.groupby(['day_of_week', 'hour'])['actual_load_mw'].mean(),
        'error': df_2024.groupby(['day_of_week', 'hour'])['load_error'].mean(),
    }


def build_dataset(df: pd.DataFrame, seasonal: dict) -> pd.DataFrame:
    """Build feature dataset with DAMAS forecast included."""
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
                'actual': row['actual_load_mw'],
                'forecast': row['forecast_load_mw'],
                'target_error': row['load_error'],
                'damas_forecast': row['forecast_load_mw'],
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


def evaluate_model(train, test, features, use_hourly=False):
    """Train and evaluate model."""
    train = train.dropna(subset=features + ['target_error']).copy()
    test = test.dropna(subset=features + ['target_error']).copy()

    if use_hourly:
        # 24 separate models
        test['pred_error'] = 0.0
        for hour in range(24):
            train_h = train[train['hour'] == hour]
            test_h = test[test['hour'] == hour]
            if len(train_h) < 30:
                test.loc[test['hour'] == hour, 'pred_error'] = train_h['target_error'].mean() if len(train_h) > 0 else 0
                continue
            model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                               min_samples_leaf=10, random_state=42)
            model.fit(train_h[features], train_h['target_error'])
            test.loc[test['hour'] == hour, 'pred_error'] = model.predict(test_h[features])
        model = None
    else:
        # Single model
        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                           min_samples_leaf=20, random_state=42)
        model.fit(train[features], train['target_error'])
        test['pred_error'] = model.predict(test[features])

    test['corrected'] = test['forecast'] + test['pred_error']
    baseline_mae = (test['actual'] - test['forecast']).abs().mean()
    model_mae = (test['actual'] - test['corrected']).abs().mean()

    return baseline_mae, model_mae, test, model


def main():
    df = load_data()
    seasonal = create_seasonal(df)
    dataset = build_dataset(df, seasonal)
    dataset['date'] = pd.to_datetime(dataset['date'])

    print(f"Dataset: {len(dataset)} samples")

    # Feature sets
    features_original = [
        'hour', 'dow', 'is_weekend',
        'seasonal_error', 'forecast_vs_seasonal',
        'd1_same_hour_error', 'd1_mean_error',
        'd1_evening_error', 'd1_morning_error', 'd1_night_error',
        'd1_load_vs_seasonal',
    ]
    features_with_damas = features_original + ['damas_forecast']

    # Test different splits
    splits = [
        ('Jul-Nov 2025 test', '2025-07-01', '2025-12-01'),
        ('Dec 2025 test', '2025-12-01', '2026-01-01'),
        ('Oct-Dec 2025 test', '2025-10-01', '2026-01-01'),
    ]

    all_results = []

    for split_name, test_start, test_end in splits:
        train = dataset[dataset['date'] < test_start].copy()
        test = dataset[(dataset['date'] >= test_start) & (dataset['date'] < test_end)].copy()

        if len(test) == 0:
            continue

        print(f"\n{'='*70}")
        print(f"SPLIT: {split_name}")
        print(f"{'='*70}")
        print(f"Train: {len(train)} samples ({train['date'].min().date()} to {train['date'].max().date()})")
        print(f"Test:  {len(test)} samples ({test['date'].min().date()} to {test['date'].max().date()})")

        # Test all combinations
        configs = [
            ('Original features, single model', features_original, False),
            ('+ damas_forecast, single model', features_with_damas, False),
            ('Original features, 24 hourly models', features_original, True),
            ('+ damas_forecast, 24 hourly models', features_with_damas, True),
        ]

        print(f"\n{'Model':<45} {'Baseline':<12} {'Model':<12} {'Improvement':<12}")
        print("-" * 80)

        for config_name, features, use_hourly in configs:
            baseline_mae, model_mae, _, model = evaluate_model(train, test, features, use_hourly)
            improvement = (1 - model_mae / baseline_mae) * 100
            print(f"{config_name:<45} {baseline_mae:<12.1f} {model_mae:<12.1f} {improvement:+.1f}%")

            all_results.append({
                'split': split_name,
                'config': config_name,
                'baseline': baseline_mae,
                'model': model_mae,
                'improvement': improvement,
            })

        # Feature importance for best config
        baseline_mae, model_mae, test_result, model = evaluate_model(train, test, features_with_damas, False)
        if model is not None:
            importance = pd.DataFrame({
                'feature': features_with_damas,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\nFeature importance (with damas_forecast):")
            for _, row in importance.head(5).iterrows():
                print(f"  {row['feature']:<25}: {row['importance']:.3f}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY ACROSS ALL SPLITS")
    print(f"{'='*70}")

    results_df = pd.DataFrame(all_results)

    for split in results_df['split'].unique():
        split_results = results_df[results_df['split'] == split]
        best = split_results.loc[split_results['improvement'].idxmax()]
        print(f"\n{split}:")
        print(f"  Best config: {best['config']}")
        print(f"  Baseline: {best['baseline']:.1f} MW, Model: {best['model']:.1f} MW")
        print(f"  Improvement: {best['improvement']:+.1f}%")

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, (split_name, _, _) in enumerate(splits):
        split_results = results_df[results_df['split'] == split_name]
        if len(split_results) == 0:
            continue

        configs = split_results['config'].values
        improvements = split_results['improvement'].values
        colors = ['#27ae60' if x > 0 else '#e74c3c' for x in improvements]

        bars = axes[i].barh(range(len(configs)), improvements, color=colors, edgecolor='black')
        axes[i].axvline(0, color='black', linewidth=2)
        axes[i].set_yticks(range(len(configs)))
        axes[i].set_yticklabels([c.replace(', ', '\n') for c in configs], fontsize=9)
        axes[i].set_xlabel('Improvement vs Baseline (%)')
        axes[i].set_title(split_name)

        for bar, imp in zip(bars, improvements):
            x_pos = imp + 0.3 if imp >= 0 else imp - 0.3
            ha = 'left' if imp >= 0 else 'right'
            axes[i].text(x_pos, bar.get_y() + bar.get_height()/2, f'{imp:+.1f}%',
                        va='center', ha=ha, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '28_model_variations_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: 28_model_variations_test.png")


if __name__ == '__main__':
    main()
