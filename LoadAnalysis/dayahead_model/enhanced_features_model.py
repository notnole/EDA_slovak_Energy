"""
Enhanced Day-Ahead Model with Rich Feature Engineering

New features:
1. Price momentum (rate of change, acceleration)
2. Price differences (vs yesterday, vs week ago, vs seasonal)
3. Last week of load patterns (all 7 days)
4. Rolling statistics (7-day windows)
5. Similar day patterns (same dow last weeks)
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
PRICE_PATH = BASE_PATH / 'features' / 'DamasPrices' / 'da_prices.parquet'
LOAD_PATH = BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet'
PLOT_PATH = Path(__file__).parent / 'plots'


def load_data():
    """Load and merge price and load data."""
    print("Loading data...")

    df_price = pd.read_parquet(PRICE_PATH)
    df_load = pd.read_parquet(LOAD_PATH)

    df_price['datetime'] = pd.to_datetime(df_price['datetime'])
    df_load['datetime'] = pd.to_datetime(df_load['datetime'])

    # Merge
    df = df_load.merge(df_price, on='datetime', how='inner', suffixes=('', '_price'))
    df['load_error'] = df['actual_load_mw'] - df['forecast_load_mw']
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"  Records: {len(df)}")
    return df


def create_seasonal_patterns(df: pd.DataFrame) -> dict:
    """Create seasonal patterns from 2024."""
    df_2024 = df[df['year'] == 2024]

    return {
        'load': df_2024.groupby(['day_of_week', 'hour'])['actual_load_mw'].mean(),
        'price': df_2024.groupby(['day_of_week', 'hour'])['price_eur_mwh'].mean(),
        'error': df_2024.groupby(['day_of_week', 'hour'])['load_error'].mean(),
        'load_std': df_2024.groupby(['day_of_week', 'hour'])['actual_load_mw'].std(),
        'price_std': df_2024.groupby(['day_of_week', 'hour'])['price_eur_mwh'].std(),
    }


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price and load momentum features."""
    df = df.copy()

    # === PRICE MOMENTUM ===

    # Price change (1h, 6h, 12h, 24h)
    df['price_diff_1h'] = df['price_eur_mwh'].diff(1)
    df['price_diff_6h'] = df['price_eur_mwh'].diff(6)
    df['price_diff_12h'] = df['price_eur_mwh'].diff(12)
    df['price_diff_24h'] = df['price_eur_mwh'].diff(24)

    # Price acceleration (change of change)
    df['price_accel_1h'] = df['price_diff_1h'].diff(1)
    df['price_accel_24h'] = df['price_diff_24h'].diff(24)

    # Price momentum (rolling mean of changes)
    df['price_momentum_6h'] = df['price_diff_1h'].rolling(6).mean()
    df['price_momentum_24h'] = df['price_diff_1h'].rolling(24).mean()

    # Price trend (linear slope over last 6h, 24h)
    df['price_trend_6h'] = df['price_eur_mwh'].rolling(6).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 6 else np.nan, raw=False
    )
    df['price_trend_24h'] = df['price_eur_mwh'].rolling(24).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 24 else np.nan, raw=False
    )

    # === LOAD MOMENTUM ===

    df['load_diff_1h'] = df['actual_load_mw'].diff(1)
    df['load_diff_24h'] = df['actual_load_mw'].diff(24)
    df['load_momentum_6h'] = df['load_diff_1h'].rolling(6).mean()
    df['load_trend_6h'] = df['actual_load_mw'].rolling(6).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 6 else np.nan, raw=False
    )

    # === ERROR MOMENTUM ===

    df['error_diff_1h'] = df['load_error'].diff(1)
    df['error_diff_24h'] = df['load_error'].diff(24)
    df['error_momentum_6h'] = df['error_diff_1h'].rolling(6).mean()

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling window statistics."""
    df = df.copy()

    # === 7-DAY ROLLING STATS ===

    # Price
    df['price_roll7d_mean'] = df['price_eur_mwh'].rolling(168, min_periods=24).mean()
    df['price_roll7d_std'] = df['price_eur_mwh'].rolling(168, min_periods=24).std()
    df['price_roll7d_min'] = df['price_eur_mwh'].rolling(168, min_periods=24).min()
    df['price_roll7d_max'] = df['price_eur_mwh'].rolling(168, min_periods=24).max()
    df['price_vs_roll7d'] = df['price_eur_mwh'] - df['price_roll7d_mean']

    # Load
    df['load_roll7d_mean'] = df['actual_load_mw'].rolling(168, min_periods=24).mean()
    df['load_roll7d_std'] = df['actual_load_mw'].rolling(168, min_periods=24).std()
    df['load_vs_roll7d'] = df['actual_load_mw'] - df['load_roll7d_mean']

    # Error
    df['error_roll7d_mean'] = df['load_error'].rolling(168, min_periods=24).mean()
    df['error_roll7d_std'] = df['load_error'].rolling(168, min_periods=24).std()

    # === 24H ROLLING STATS ===

    df['price_roll24h_mean'] = df['price_eur_mwh'].rolling(24, min_periods=12).mean()
    df['price_roll24h_std'] = df['price_eur_mwh'].rolling(24, min_periods=12).std()
    df['load_roll24h_mean'] = df['actual_load_mw'].rolling(24, min_periods=12).mean()
    df['error_roll24h_mean'] = df['load_error'].rolling(24, min_periods=12).mean()

    return df


def build_enhanced_dataset(df: pd.DataFrame, seasonal: dict) -> pd.DataFrame:
    """Build dataset with enhanced features for day-ahead prediction."""
    print("\nBuilding enhanced dataset...")

    df['date'] = pd.to_datetime(df['datetime']).dt.date
    dates = sorted(df['date'].unique())

    rows = []

    for i, target_date in enumerate(dates):
        if i < 8:  # Need at least 8 days of history
            continue

        today_date = dates[i-1]

        # Get data for multiple days
        today = df[df['date'] == today_date]
        tomorrow = df[df['date'] == target_date]

        if len(today) < 20 or len(tomorrow) < 20:
            continue

        # Last 7 days of data
        week_data = df[df['date'].isin(dates[i-8:i-1])]

        # === TODAY'S (D) SUMMARY ===
        today_errors = today.set_index('hour')['load_error'].to_dict()
        today_prices = today.set_index('hour')['price_eur_mwh'].to_dict()
        today_loads = today.set_index('hour')['actual_load_mw'].to_dict()

        today_mean_error = today['load_error'].mean()
        today_std_error = today['load_error'].std()
        today_mean_price = today['price_eur_mwh'].mean()
        today_mean_load = today['actual_load_mw'].mean()

        # Today's momentum (end of day)
        today_price_momentum = today['price_momentum_6h'].iloc[-1] if 'price_momentum_6h' in today else 0
        today_load_momentum = today['load_momentum_6h'].iloc[-1] if 'load_momentum_6h' in today else 0
        today_error_momentum = today['error_momentum_6h'].iloc[-1] if 'error_momentum_6h' in today else 0

        # Today's trend
        today_price_trend = today['price_trend_6h'].iloc[-1] if 'price_trend_6h' in today else 0
        today_load_trend = today['load_trend_6h'].iloc[-1] if 'load_trend_6h' in today else 0

        # Today's periods
        today_morning_error = today[today['hour'].between(6, 11)]['load_error'].mean()
        today_afternoon_error = today[today['hour'].between(12, 17)]['load_error'].mean()
        today_evening_error = today[today['hour'].between(18, 23)]['load_error'].mean()
        today_night_error = today[today['hour'].between(0, 5)]['load_error'].mean()

        # === LAST 7 DAYS PATTERNS ===
        week_mean_error = week_data['load_error'].mean()
        week_std_error = week_data['load_error'].std()
        week_mean_price = week_data['price_eur_mwh'].mean()
        week_mean_load = week_data['actual_load_mw'].mean()

        # Same DOW from previous weeks
        target_dow = pd.to_datetime(target_date).dayofweek
        same_dow_days = [d for d in dates[max(0,i-29):i-1]
                        if pd.to_datetime(d).dayofweek == target_dow]

        if len(same_dow_days) >= 2:
            same_dow_data = df[df['date'].isin(same_dow_days[-4:])]  # Last 4 same DOWs
            same_dow_errors = same_dow_data.groupby('hour')['load_error'].mean().to_dict()
            same_dow_mean_error = same_dow_data['load_error'].mean()
        else:
            same_dow_errors = {}
            same_dow_mean_error = 0

        # === EACH HOUR OF TOMORROW ===
        for _, row in tomorrow.iterrows():
            hour = row['hour']
            dow = row['day_of_week']

            # Seasonal references
            seasonal_load = seasonal['load'].get((dow, hour), row['actual_load_mw'])
            seasonal_price = seasonal['price'].get((dow, hour), row['price_eur_mwh'])
            seasonal_error = seasonal['error'].get((dow, hour), 0)

            # Yesterday same hour
            d1_error = today_errors.get(hour, 0)
            d1_price = today_prices.get(hour, today_mean_price)
            d1_load = today_loads.get(hour, today_mean_load)

            # Days 2-7 same hour
            d2_error, d3_error, d7_error = 0, 0, 0
            for lag, var_name in [(2, 'd2_error'), (3, 'd3_error'), (7, 'd7_error')]:
                if i - lag >= 0:
                    lag_date = dates[i - lag]
                    lag_data = df[(df['date'] == lag_date) & (df['hour'] == hour)]
                    if len(lag_data) > 0:
                        if var_name == 'd2_error':
                            d2_error = lag_data['load_error'].values[0]
                        elif var_name == 'd3_error':
                            d3_error = lag_data['load_error'].values[0]
                        elif var_name == 'd7_error':
                            d7_error = lag_data['load_error'].values[0]

            features = {
                'datetime': row['datetime'],
                'date': target_date,
                'hour': hour,
                'dow': dow,
                'is_weekend': row['is_weekend'],
                'month': row['month'],

                # Targets
                'actual_load': row['actual_load_mw'],
                'forecast_load': row['forecast_load_mw'],
                'target_error': row['load_error'],

                # === TOMORROW'S PRICES (KNOWN) ===
                'tomorrow_price': row['price_eur_mwh'],
                'tomorrow_price_vs_seasonal': row['price_eur_mwh'] - seasonal_price,
                'tomorrow_price_vs_yesterday': row['price_eur_mwh'] - d1_price,
                'tomorrow_price_vs_week': row['price_eur_mwh'] - week_mean_price,
                'tomorrow_net_import': row['net_import'],
                'tomorrow_is_negative': int(row['price_eur_mwh'] < 0),

                # Price momentum features from today
                'tomorrow_price_momentum': row.get('price_momentum_6h', 0),
                'tomorrow_price_trend': row.get('price_trend_6h', 0),

                # === SEASONAL ===
                'seasonal_error': seasonal_error,
                'forecast_vs_seasonal': row['forecast_load_mw'] - seasonal_load,

                # === TODAY (D-1) ===
                'd1_same_hour_error': d1_error,
                'd1_same_hour_price': d1_price,
                'd1_mean_error': today_mean_error,
                'd1_std_error': today_std_error,
                'd1_mean_price': today_mean_price,

                # Today's momentum
                'd1_price_momentum': today_price_momentum if not pd.isna(today_price_momentum) else 0,
                'd1_load_momentum': today_load_momentum if not pd.isna(today_load_momentum) else 0,
                'd1_error_momentum': today_error_momentum if not pd.isna(today_error_momentum) else 0,
                'd1_price_trend': today_price_trend if not pd.isna(today_price_trend) else 0,
                'd1_load_trend': today_load_trend if not pd.isna(today_load_trend) else 0,

                # Today's periods
                'd1_morning_error': today_morning_error,
                'd1_afternoon_error': today_afternoon_error,
                'd1_evening_error': today_evening_error,
                'd1_night_error': today_night_error,

                # Today vs seasonal
                'd1_load_vs_seasonal': today_mean_load - seasonal['load'].get((today['day_of_week'].iloc[0], 12), today_mean_load),
                'd1_price_vs_seasonal': today_mean_price - seasonal['price'].get((today['day_of_week'].iloc[0], 12), today_mean_price),

                # === DAYS 2-7 SAME HOUR ===
                'd2_same_hour_error': d2_error,
                'd3_same_hour_error': d3_error,
                'd7_same_hour_error': d7_error,

                # === WEEK STATISTICS ===
                'week_mean_error': week_mean_error,
                'week_std_error': week_std_error,
                'week_mean_price': week_mean_price,

                # === SAME DOW HISTORY ===
                'same_dow_error': same_dow_errors.get(hour, 0),
                'same_dow_mean_error': same_dow_mean_error,

                # === PRICE DIFFERENCES ===
                'price_diff_d1_to_tomorrow': row['price_eur_mwh'] - d1_price,
                'price_diff_week_to_tomorrow': row['price_eur_mwh'] - week_mean_price,
            }

            rows.append(features)

    result = pd.DataFrame(rows)
    print(f"Created {len(result)} samples with {len(result.columns)} features")

    return result


def train_and_evaluate(df: pd.DataFrame):
    """Train and evaluate the enhanced model."""
    print("\n" + "=" * 70)
    print("TRAINING ENHANCED MODEL")
    print("=" * 70)

    # All features (excluding targets and identifiers)
    exclude_cols = ['datetime', 'date', 'actual_load', 'forecast_load', 'target_error']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    df_valid = df.dropna(subset=feature_cols + ['target_error'])
    print(f"\nValid samples: {len(df_valid)}")
    print(f"Features: {len(feature_cols)}")

    # Split
    df_valid['date'] = pd.to_datetime(df_valid['date'])
    split_date = '2025-07-01'
    train = df_valid[df_valid['date'] < split_date]
    test = df_valid[df_valid['date'] >= split_date]

    print(f"Train: {len(train)} ({train['date'].min()} to {train['date'].max()})")
    print(f"Test:  {len(test)} ({test['date'].min()} to {test['date'].max()})")

    X_train, y_train = train[feature_cols], train['target_error']
    X_test, y_test = test[feature_cols], test['target_error']

    # Train model
    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    test = test.copy()
    test['pred_error'] = model.predict(X_test)
    test['corrected'] = test['forecast_load'] + test['pred_error']

    # Results
    baseline_mae = (test['actual_load'] - test['forecast_load']).abs().mean()
    model_mae = (test['actual_load'] - test['corrected']).abs().mean()

    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"Baseline (DAMAS):     {baseline_mae:.1f} MW")
    print(f"Enhanced Model:       {model_mae:.1f} MW ({(1-model_mae/baseline_mae)*100:+.1f}%)")
    print("-" * 50)

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n--- Top 20 Features ---")
    print(importance.head(20).to_string(index=False))

    # Group features by category
    print("\n--- Importance by Category ---")
    categories = {
        'Tomorrow Price': [c for c in feature_cols if 'tomorrow' in c.lower()],
        'Day-1 (Yesterday)': [c for c in feature_cols if c.startswith('d1_')],
        'Days 2-7': [c for c in feature_cols if c.startswith(('d2_', 'd3_', 'd7_'))],
        'Week Stats': [c for c in feature_cols if 'week' in c.lower()],
        'Same DOW': [c for c in feature_cols if 'same_dow' in c.lower()],
        'Seasonal': [c for c in feature_cols if 'seasonal' in c.lower()],
        'Momentum/Trend': [c for c in feature_cols if 'momentum' in c.lower() or 'trend' in c.lower()],
        'Time': ['hour', 'dow', 'is_weekend', 'month'],
    }

    for cat_name, cat_features in categories.items():
        cat_importance = importance[importance['feature'].isin(cat_features)]['importance'].sum()
        print(f"  {cat_name:<20}: {cat_importance:.3f} ({cat_importance*100:.1f}%)")

    # By hour
    print("\n--- MAE by Hour ---")
    hourly = test.groupby('hour').apply(lambda x: pd.Series({
        'baseline': (x['actual_load'] - x['forecast_load']).abs().mean(),
        'model': (x['actual_load'] - x['corrected']).abs().mean(),
    }))
    hourly['improvement'] = (1 - hourly['model'] / hourly['baseline']) * 100

    # Summary
    print(f"\nBest hours:  {hourly['improvement'].nlargest(3).to_dict()}")
    print(f"Worst hours: {hourly['improvement'].nsmallest(3).to_dict()}")

    return model, test, importance, hourly, feature_cols


def create_plots(test: pd.DataFrame, importance: pd.DataFrame, hourly: pd.DataFrame):
    """Create visualization plots."""
    print("\n" + "=" * 70)
    print("CREATING PLOTS")
    print("=" * 70)

    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Model comparison bar
    baseline_mae = (test['actual_load'] - test['forecast_load']).abs().mean()
    model_mae = (test['actual_load'] - test['corrected']).abs().mean()

    models = ['Baseline\n(DAMAS)', 'Enhanced\nModel']
    maes = [baseline_mae, model_mae]
    colors = ['red', 'green']
    axes[0, 0].bar(models, maes, color=colors, edgecolor='black')
    for i, v in enumerate(maes):
        axes[0, 0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=12)
    axes[0, 0].set_ylabel('MAE (MW)')
    axes[0, 0].set_title(f'Model Comparison ({(1-model_mae/baseline_mae)*100:+.1f}%)')
    axes[0, 0].axhline(baseline_mae, color='red', linestyle='--', alpha=0.5)

    # 2. Feature importance (top 15)
    top_n = 15
    axes[0, 1].barh(range(top_n), importance['importance'].head(top_n), color='steelblue', edgecolor='black')
    axes[0, 1].set_yticks(range(top_n))
    axes[0, 1].set_yticklabels(importance['feature'].head(top_n), fontsize=8)
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].set_title('Top 15 Features')

    # 3. Improvement by hour
    colors = ['green' if x > 0 else 'red' for x in hourly['improvement']]
    axes[0, 2].bar(hourly.index, hourly['improvement'], color=colors, edgecolor='black')
    axes[0, 2].axhline(0, color='black')
    axes[0, 2].set_xlabel('Hour')
    axes[0, 2].set_ylabel('Improvement (%)')
    axes[0, 2].set_title('Improvement by Hour')
    axes[0, 2].axhline(hourly['improvement'].mean(), color='blue', linestyle='--',
                       label=f"Avg: {hourly['improvement'].mean():.1f}%")
    axes[0, 2].legend()

    # 4. Sample week
    start = test['datetime'].min() + pd.Timedelta(days=30)
    sample = test[(test['datetime'] >= start) & (test['datetime'] < start + pd.Timedelta(days=7))]
    axes[1, 0].plot(sample['datetime'], sample['actual_load'], 'b-', label='Actual', linewidth=1.5)
    axes[1, 0].plot(sample['datetime'], sample['forecast_load'], 'r--', label='Baseline', alpha=0.7)
    axes[1, 0].plot(sample['datetime'], sample['corrected'], 'g-', label='Enhanced', alpha=0.7)
    axes[1, 0].legend()
    axes[1, 0].set_title('Sample Week')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. Error distribution
    baseline_err = test['actual_load'] - test['forecast_load']
    model_err = test['actual_load'] - test['corrected']
    axes[1, 1].hist(baseline_err, bins=50, alpha=0.5, label=f'Baseline ({baseline_err.abs().mean():.1f})', color='red')
    axes[1, 1].hist(model_err, bins=50, alpha=0.5, label=f'Enhanced ({model_err.abs().mean():.1f})', color='green')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Error (MW)')
    axes[1, 1].set_title('Error Distribution')

    # 6. Predicted vs actual error
    r2 = np.corrcoef(test['pred_error'], test['target_error'])[0, 1] ** 2
    axes[1, 2].scatter(test['pred_error'], test['target_error'], alpha=0.2, s=5)
    axes[1, 2].plot([-200, 200], [-200, 200], 'r--', linewidth=2)
    axes[1, 2].set_xlabel('Predicted Error')
    axes[1, 2].set_ylabel('Actual Error')
    axes[1, 2].set_title(f'Error Prediction (R2={r2:.3f})')
    axes[1, 2].set_xlim(-200, 200)
    axes[1, 2].set_ylim(-200, 200)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '25_enhanced_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 25_enhanced_model.png")


def main():
    # Load data
    df = load_data()

    # Add momentum and rolling features
    print("\nAdding momentum features...")
    df = add_momentum_features(df)

    print("Adding rolling features...")
    df = add_rolling_features(df)

    # Create seasonal patterns
    seasonal = create_seasonal_patterns(df)

    # Build enhanced dataset
    df_enhanced = build_enhanced_dataset(df, seasonal)

    # Train and evaluate
    model, test, importance, hourly, features = train_and_evaluate(df_enhanced)

    # Create plots
    create_plots(test, importance, hourly)

    # Summary
    baseline_mae = (test['actual_load'] - test['forecast_load']).abs().mean()
    model_mae = (test['actual_load'] - test['corrected']).abs().mean()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Enhanced Day-Ahead Model with Rich Features:

Features added:
- Price momentum (6h, 24h trends and acceleration)
- Price differences (vs yesterday, vs week, vs seasonal)
- Last 7 days of load/error patterns
- Same day-of-week history (last 4 weeks)
- Rolling 7-day statistics
- Period-specific errors (morning, afternoon, evening, night)

Results:
  Baseline (DAMAS):  {baseline_mae:.1f} MW MAE
  Enhanced Model:    {model_mae:.1f} MW MAE ({(1-model_mae/baseline_mae)*100:+.1f}%)

Top features:
  1. {importance.iloc[0]['feature']}: {importance.iloc[0]['importance']:.3f}
  2. {importance.iloc[1]['feature']}: {importance.iloc[1]['importance']:.3f}
  3. {importance.iloc[2]['feature']}: {importance.iloc[2]['importance']:.3f}
  4. {importance.iloc[3]['feature']}: {importance.iloc[3]['importance']:.3f}
  5. {importance.iloc[4]['feature']}: {importance.iloc[4]['importance']:.3f}
""")

    return model, test, importance


if __name__ == '__main__':
    model, test, importance = main()
