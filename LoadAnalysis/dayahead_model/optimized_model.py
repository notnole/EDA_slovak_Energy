"""
Optimized Day-Ahead Model

Approach:
1. Start with best features from previous analysis
2. Add new features incrementally (test each)
3. Use cross-validation to avoid overfitting
4. Feature selection based on actual improvement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Paths
BASE_PATH = Path(__file__).parent.parent
PRICE_PATH = BASE_PATH / 'features' / 'DamasPrices' / 'da_prices.parquet'
LOAD_PATH = BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet'
PLOT_PATH = Path(__file__).parent / 'plots'


def load_and_prepare_data():
    """Load and merge data with all potential features."""
    print("Loading data...")

    df_price = pd.read_parquet(PRICE_PATH)
    df_load = pd.read_parquet(LOAD_PATH)

    df_price['datetime'] = pd.to_datetime(df_price['datetime'])
    df_load['datetime'] = pd.to_datetime(df_load['datetime'])

    df = df_load.merge(df_price, on='datetime', how='inner', suffixes=('', '_price'))
    df['load_error'] = df['actual_load_mw'] - df['forecast_load_mw']
    df = df.sort_values('datetime').reset_index(drop=True)

    # Add momentum features
    df['price_diff_1h'] = df['price_eur_mwh'].diff(1)
    df['price_diff_24h'] = df['price_eur_mwh'].diff(24)
    df['price_momentum_6h'] = df['price_diff_1h'].rolling(6).mean()
    df['load_diff_1h'] = df['actual_load_mw'].diff(1)
    df['load_momentum_6h'] = df['load_diff_1h'].rolling(6).mean()
    df['error_momentum_6h'] = df['load_error'].diff(1).rolling(6).mean()

    # Rolling stats
    df['price_roll7d_mean'] = df['price_eur_mwh'].rolling(168, min_periods=24).mean()
    df['load_roll7d_mean'] = df['actual_load_mw'].rolling(168, min_periods=24).mean()
    df['error_roll7d_mean'] = df['load_error'].rolling(168, min_periods=24).mean()
    df['error_roll7d_std'] = df['load_error'].rolling(168, min_periods=24).std()

    print(f"  Records: {len(df)}")
    return df


def create_seasonal(df: pd.DataFrame) -> dict:
    """Create seasonal patterns from 2024."""
    df_2024 = df[df['year'] == 2024]
    return {
        'load': df_2024.groupby(['day_of_week', 'hour'])['actual_load_mw'].mean(),
        'price': df_2024.groupby(['day_of_week', 'hour'])['price_eur_mwh'].mean(),
        'error': df_2024.groupby(['day_of_week', 'hour'])['load_error'].mean(),
    }


def build_dataset(df: pd.DataFrame, seasonal: dict) -> pd.DataFrame:
    """Build prediction dataset with carefully selected features."""
    print("\nBuilding dataset...")

    df['date'] = pd.to_datetime(df['datetime']).dt.date
    dates = sorted(df['date'].unique())

    rows = []

    for i, target_date in enumerate(dates):
        if i < 8:
            continue

        today_date = dates[i-1]
        today = df[df['date'] == today_date]
        tomorrow = df[df['date'] == target_date]

        if len(today) < 20 or len(tomorrow) < 20:
            continue

        # Today's data
        today_errors = today.set_index('hour')['load_error'].to_dict()
        today_prices = today.set_index('hour')['price_eur_mwh'].to_dict()
        today_loads = today.set_index('hour')['actual_load_mw'].to_dict()

        today_mean_error = today['load_error'].mean()
        today_morning_err = today[today['hour'].between(6, 11)]['load_error'].mean()
        today_evening_err = today[today['hour'].between(18, 23)]['load_error'].mean()

        # Today's end-of-day momentum
        today_end = today.iloc[-6:]
        today_load_trend = today_end['actual_load_mw'].iloc[-1] - today_end['actual_load_mw'].iloc[0] if len(today_end) >= 6 else 0
        today_price_trend = today_end['price_eur_mwh'].iloc[-1] - today_end['price_eur_mwh'].iloc[0] if len(today_end) >= 6 else 0

        # Last 7 days
        week_data = df[df['date'].isin(dates[i-8:i-1])]
        week_mean_error = week_data['load_error'].mean() if len(week_data) > 0 else 0
        week_std_error = week_data['load_error'].std() if len(week_data) > 0 else 0

        # D-2 to D-7 same hour errors
        lag_errors = {}
        for lag in [2, 3, 7]:
            if i - lag >= 0:
                lag_date = dates[i - lag]
                lag_data = df[df['date'] == lag_date]
                if len(lag_data) >= 20:
                    lag_errors[lag] = lag_data.set_index('hour')['load_error'].to_dict()
                else:
                    lag_errors[lag] = {}
            else:
                lag_errors[lag] = {}

        for _, row in tomorrow.iterrows():
            hour = row['hour']
            dow = row['day_of_week']

            # Seasonal
            seasonal_load = seasonal['load'].get((dow, hour), row['actual_load_mw'])
            seasonal_price = seasonal['price'].get((dow, hour), row['price_eur_mwh'])
            seasonal_error = seasonal['error'].get((dow, hour), 0)

            # Yesterday same hour
            d1_error = today_errors.get(hour, 0)
            d1_price = today_prices.get(hour, row['price_eur_mwh'])
            d1_load = today_loads.get(hour, row['actual_load_mw'])

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

                # === CORE FEATURES (from v2 model) ===
                'seasonal_error': seasonal_error,
                'forecast_vs_seasonal': row['forecast_load_mw'] - seasonal_load,
                'd1_same_hour_error': d1_error,
                'd1_mean_error': today_mean_error,
                'd1_evening_error': today_evening_err,
                'd1_morning_error': today_morning_err,

                # Today vs seasonal
                'd1_load_vs_seasonal': (today['actual_load_mw'].mean() -
                    seasonal['load'].get((today['day_of_week'].iloc[0], 12), today['actual_load_mw'].mean())),

                # === NEW MOMENTUM FEATURES ===
                'd1_load_trend': today_load_trend,
                'd1_price_trend': today_price_trend,

                # === PRICE FEATURES ===
                'tomorrow_price': row['price_eur_mwh'],
                'tomorrow_price_vs_seasonal': row['price_eur_mwh'] - seasonal_price,
                'tomorrow_price_vs_yesterday': row['price_eur_mwh'] - d1_price,
                'tomorrow_net_import': row['net_import'],

                # === WEEK FEATURES ===
                'week_mean_error': week_mean_error,
                'week_std_error': week_std_error,

                # === LAGGED SAME-HOUR ERRORS ===
                'd2_same_hour_error': lag_errors[2].get(hour, 0),
                'd7_same_hour_error': lag_errors[7].get(hour, 0),
            }

            rows.append(features)

    result = pd.DataFrame(rows)
    print(f"Created {len(result)} samples")
    return result


def incremental_feature_test(df: pd.DataFrame):
    """Test features incrementally to find best combination."""
    print("\n" + "=" * 70)
    print("INCREMENTAL FEATURE TESTING")
    print("=" * 70)

    df['date'] = pd.to_datetime(df['date'])
    split_date = '2025-07-01'
    train = df[df['date'] < split_date].copy()
    test = df[df['date'] >= split_date].copy()

    baseline_mae = (test['actual'] - test['forecast']).abs().mean()
    print(f"\nBaseline MAE: {baseline_mae:.1f} MW")

    # Feature groups to test
    feature_groups = {
        'Core (v2)': [
            'hour', 'dow', 'is_weekend',
            'seasonal_error', 'forecast_vs_seasonal',
            'd1_same_hour_error', 'd1_mean_error',
            'd1_evening_error', 'd1_morning_error',
            'd1_load_vs_seasonal',
        ],
        '+ Momentum': [
            'd1_load_trend', 'd1_price_trend',
        ],
        '+ Price': [
            'tomorrow_price', 'tomorrow_price_vs_seasonal',
            'tomorrow_price_vs_yesterday', 'tomorrow_net_import',
        ],
        '+ Week': [
            'week_mean_error', 'week_std_error',
        ],
        '+ Lagged': [
            'd2_same_hour_error', 'd7_same_hour_error',
        ],
    }

    results = []
    current_features = []

    for group_name, group_features in feature_groups.items():
        # Add new features
        available = [f for f in group_features if f in train.columns]
        current_features.extend(available)

        if not current_features:
            continue

        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42
        )

        train_valid = train.dropna(subset=current_features + ['target_error'])
        test_valid = test.dropna(subset=current_features + ['target_error'])

        model.fit(train_valid[current_features], train_valid['target_error'])

        pred_error = model.predict(test_valid[current_features])
        corrected = test_valid['forecast'] + pred_error
        mae = (test_valid['actual'] - corrected).abs().mean()

        improvement = (1 - mae / baseline_mae) * 100

        results.append({
            'group': group_name,
            'n_features': len(current_features),
            'mae': mae,
            'improvement': improvement,
        })

        print(f"\n{group_name}:")
        print(f"  Features: {len(current_features)}")
        print(f"  MAE: {mae:.1f} MW ({improvement:+.1f}%)")

    return pd.DataFrame(results), current_features


def cross_validate_model(df: pd.DataFrame, features: list):
    """Cross-validate to get robust estimate."""
    print("\n" + "=" * 70)
    print("TIME-SERIES CROSS-VALIDATION")
    print("=" * 70)

    df_valid = df.dropna(subset=features + ['target_error']).copy()
    df_valid = df_valid.sort_values('datetime').reset_index(drop=True)

    X = df_valid[features]
    y = df_valid['target_error']
    forecast = df_valid['forecast']
    actual = df_valid['actual']

    # Time series split (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)

    baseline_maes = []
    model_maes = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        forecast_val = forecast.iloc[val_idx]
        actual_val = actual.iloc[val_idx]

        # Train
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict
        pred_error = model.predict(X_val)
        corrected = forecast_val + pred_error

        # Metrics
        baseline_mae = (actual_val - forecast_val).abs().mean()
        model_mae = (actual_val - corrected).abs().mean()

        baseline_maes.append(baseline_mae)
        model_maes.append(model_mae)

        print(f"  Fold {fold+1}: Baseline {baseline_mae:.1f} -> Model {model_mae:.1f} ({(1-model_mae/baseline_mae)*100:+.1f}%)")

    print(f"\nCross-validation summary:")
    print(f"  Baseline: {np.mean(baseline_maes):.1f} +/- {np.std(baseline_maes):.1f} MW")
    print(f"  Model:    {np.mean(model_maes):.1f} +/- {np.std(model_maes):.1f} MW")
    print(f"  Avg improvement: {(1 - np.mean(model_maes)/np.mean(baseline_maes))*100:+.1f}%")

    return np.mean(baseline_maes), np.mean(model_maes)


def final_model(df: pd.DataFrame, features: list):
    """Train final model and analyze."""
    print("\n" + "=" * 70)
    print("FINAL MODEL")
    print("=" * 70)

    df['date'] = pd.to_datetime(df['date'])
    split_date = '2025-07-01'

    df_valid = df.dropna(subset=features + ['target_error'])
    train = df_valid[df_valid['date'] < split_date]
    test = df_valid[df_valid['date'] >= split_date]

    print(f"Train: {len(train)}, Test: {len(test)}")

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

    print(f"\nBaseline: {baseline_mae:.1f} MW")
    print(f"Model:    {model_mae:.1f} MW ({(1-model_mae/baseline_mae)*100:+.1f}%)")

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n--- Feature Importance ---")
    print(importance.to_string(index=False))

    # By hour
    hourly = test.groupby('hour').apply(lambda x: pd.Series({
        'baseline': (x['actual'] - x['forecast']).abs().mean(),
        'model': (x['actual'] - x['corrected']).abs().mean(),
    }))
    hourly['improvement'] = (1 - hourly['model'] / hourly['baseline']) * 100

    print(f"\nHourly improvement range: {hourly['improvement'].min():.1f}% to {hourly['improvement'].max():.1f}%")
    print(f"Best hours: {hourly['improvement'].nlargest(3).to_dict()}")

    return model, test, importance, hourly


def create_final_plots(test, importance, hourly, results_df):
    """Create visualization."""
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    baseline_mae = (test['actual'] - test['forecast']).abs().mean()
    model_mae = (test['actual'] - test['corrected']).abs().mean()

    # 1. Incremental feature testing
    axes[0, 0].bar(range(len(results_df)), results_df['improvement'], color='steelblue', edgecolor='black')
    axes[0, 0].set_xticks(range(len(results_df)))
    axes[0, 0].set_xticklabels(results_df['group'], rotation=45, ha='right', fontsize=9)
    axes[0, 0].set_ylabel('Improvement vs Baseline (%)')
    axes[0, 0].set_title('Incremental Feature Testing')
    axes[0, 0].axhline(0, color='black')

    # 2. Feature importance
    top_n = min(15, len(importance))
    axes[0, 1].barh(range(top_n), importance['importance'].head(top_n), color='coral', edgecolor='black')
    axes[0, 1].set_yticks(range(top_n))
    axes[0, 1].set_yticklabels(importance['feature'].head(top_n), fontsize=9)
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].set_title('Feature Importance')

    # 3. Improvement by hour
    colors = ['green' if x > 0 else 'red' for x in hourly['improvement']]
    axes[0, 2].bar(hourly.index, hourly['improvement'], color=colors, edgecolor='black')
    axes[0, 2].axhline(0, color='black')
    axes[0, 2].axhline(hourly['improvement'].mean(), color='blue', linestyle='--',
                       label=f"Avg: {hourly['improvement'].mean():.1f}%")
    axes[0, 2].set_xlabel('Hour')
    axes[0, 2].set_ylabel('Improvement (%)')
    axes[0, 2].set_title('Improvement by Hour')
    axes[0, 2].legend()

    # 4. Sample week
    start = test['datetime'].min() + pd.Timedelta(days=30)
    sample = test[(test['datetime'] >= start) & (test['datetime'] < start + pd.Timedelta(days=7))]
    axes[1, 0].plot(sample['datetime'], sample['actual'], 'b-', label='Actual', linewidth=1.5)
    axes[1, 0].plot(sample['datetime'], sample['forecast'], 'r--', label='Baseline', alpha=0.7)
    axes[1, 0].plot(sample['datetime'], sample['corrected'], 'g-', label='Model', alpha=0.7)
    axes[1, 0].legend()
    axes[1, 0].set_title('Sample Week')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. Error distribution
    baseline_err = test['actual'] - test['forecast']
    model_err = test['actual'] - test['corrected']
    axes[1, 1].hist(baseline_err, bins=50, alpha=0.5, label=f'Baseline ({baseline_mae:.1f})', color='red')
    axes[1, 1].hist(model_err, bins=50, alpha=0.5, label=f'Model ({model_mae:.1f})', color='green')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Error (MW)')
    axes[1, 1].set_title('Error Distribution')

    # 6. MAE by hour comparison
    x = np.arange(24)
    width = 0.35
    axes[1, 2].bar(x - width/2, hourly['baseline'], width, label='Baseline', color='red', alpha=0.7)
    axes[1, 2].bar(x + width/2, hourly['model'], width, label='Model', color='green', alpha=0.7)
    axes[1, 2].set_xlabel('Hour')
    axes[1, 2].set_ylabel('MAE (MW)')
    axes[1, 2].set_title('MAE by Hour')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '26_optimized_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved: 26_optimized_model.png")


def main():
    # Load data
    df = load_and_prepare_data()
    seasonal = create_seasonal(df)

    # Build dataset
    df_dataset = build_dataset(df, seasonal)

    # Test features incrementally
    results_df, best_features = incremental_feature_test(df_dataset)

    # Cross-validate
    cv_baseline, cv_model = cross_validate_model(df_dataset, best_features)

    # Final model
    model, test, importance, hourly = final_model(df_dataset, best_features)

    # Plots
    create_final_plots(test, importance, hourly, results_df)

    # Summary
    baseline_mae = (test['actual'] - test['forecast']).abs().mean()
    model_mae = (test['actual'] - test['corrected']).abs().mean()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Optimized Model Results:

Test Set:
  Baseline (DAMAS):  {baseline_mae:.1f} MW MAE
  Optimized Model:   {model_mae:.1f} MW MAE ({(1-model_mae/baseline_mae)*100:+.1f}%)

Cross-Validation (5-fold):
  Baseline:  {cv_baseline:.1f} MW MAE
  Model:     {cv_model:.1f} MW MAE ({(1-cv_model/cv_baseline)*100:+.1f}%)

Best Feature Groups:
{results_df.to_string(index=False)}

Key Features:
  1. {importance.iloc[0]['feature']}: {importance.iloc[0]['importance']:.3f}
  2. {importance.iloc[1]['feature']}: {importance.iloc[1]['importance']:.3f}
  3. {importance.iloc[2]['feature']}: {importance.iloc[2]['importance']:.3f}
""")


if __name__ == '__main__':
    main()
