"""
Price Features for Load Prediction

Key insight: Day-ahead prices are KNOWN when we predict tomorrow's load!
At 23:00 on day D, we have:
- Tomorrow's (D+1) DA prices (from market)
- Tomorrow's (D+1) DAMAS load forecast
- Today's (D) complete load and errors

Question: Do prices help predict where DAMAS will be wrong?
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

    # Load data
    df_price = pd.read_parquet(PRICE_PATH)
    df_load = pd.read_parquet(LOAD_PATH)

    df_price['datetime'] = pd.to_datetime(df_price['datetime'])
    df_load['datetime'] = pd.to_datetime(df_load['datetime'])

    # Merge
    df = df_load.merge(df_price, on='datetime', how='inner', suffixes=('', '_price'))

    # Calculate load forecast error
    df['load_error'] = df['actual_load_mw'] - df['forecast_load_mw']

    print(f"  Merged records: {len(df)}")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def create_seasonal_patterns(df: pd.DataFrame) -> dict:
    """Create seasonal patterns from 2024 data."""
    df_2024 = df[df['year'] == 2024]

    # Load seasonal pattern
    load_seasonal = df_2024.groupby(['day_of_week', 'hour'])['actual_load_mw'].mean()

    # Price seasonal pattern
    price_seasonal = df_2024.groupby(['day_of_week', 'hour'])['price_eur_mwh'].mean()

    # Error seasonal pattern (systematic bias)
    error_seasonal = df_2024.groupby(['day_of_week', 'hour'])['load_error'].mean()

    return {
        'load': load_seasonal,
        'price': price_seasonal,
        'error': error_seasonal,
    }


def analyze_price_error_correlation(df: pd.DataFrame):
    """Analyze correlation between price features and load errors."""
    print("\n" + "=" * 70)
    print("PRICE-ERROR CORRELATION ANALYSIS")
    print("=" * 70)

    # What correlates with load forecast error?
    price_features = [
        'price_eur_mwh',
        'price_change_24h',
        'price_daily_std',
        'price_daily_range',
        'net_import',
        'demand_mw',
        'supply_mw',
    ]

    print("\nCorrelation with load forecast error:")
    correlations = {}
    for feat in price_features:
        if feat in df.columns:
            corr = df[feat].corr(df['load_error'])
            correlations[feat] = corr
            print(f"  {feat:<25}: {corr:+.3f}")

    # Correlation by hour
    print("\nPrice-Error correlation by hour:")
    hourly_corr = df.groupby('hour').apply(
        lambda x: x['price_eur_mwh'].corr(x['load_error'])
    )
    print(f"  Highest: Hour {hourly_corr.idxmax()} ({hourly_corr.max():+.3f})")
    print(f"  Lowest:  Hour {hourly_corr.idxmin()} ({hourly_corr.min():+.3f})")

    return correlations, hourly_corr


def create_price_features_for_prediction(df: pd.DataFrame, seasonal: dict) -> pd.DataFrame:
    """
    Create price-based features available at prediction time.

    At 23:00 on day D predicting D+1:
    - Tomorrow's prices (KNOWN from DA market)
    - Today's prices (KNOWN)
    - Seasonal patterns from 2024 (KNOWN)
    """
    print("\n" + "=" * 70)
    print("CREATING PRICE FEATURES")
    print("=" * 70)

    df = df.copy()
    df['date'] = df['datetime'].dt.date

    # Add seasonal expectations
    df['seasonal_price'] = df.apply(
        lambda x: seasonal['price'].get((x['day_of_week'], x['hour']), np.nan), axis=1
    )
    df['seasonal_load'] = df.apply(
        lambda x: seasonal['load'].get((x['day_of_week'], x['hour']), np.nan), axis=1
    )
    df['seasonal_error'] = df.apply(
        lambda x: seasonal['error'].get((x['day_of_week'], x['hour']), np.nan), axis=1
    )

    # Price vs seasonal (is price unusually high/low?)
    df['price_vs_seasonal'] = df['price_eur_mwh'] - df['seasonal_price']
    df['price_vs_seasonal_pct'] = df['price_vs_seasonal'] / df['seasonal_price'].abs().clip(lower=1) * 100

    # Load forecast vs seasonal
    df['forecast_vs_seasonal_load'] = df['forecast_load_mw'] - df['seasonal_load']

    # Price percentile for this hour (relative ranking)
    df['price_percentile'] = df.groupby('hour')['price_eur_mwh'].transform(
        lambda x: x.rank(pct=True)
    )

    # Is it a negative price hour?
    df['is_negative_price'] = (df['price_eur_mwh'] < 0).astype(int)

    # Price volatility indicator
    df['is_high_volatility'] = (df['price_daily_std'] > df['price_daily_std'].quantile(0.75)).astype(int)

    # Net import indicator (high import = oversupply)
    df['is_high_import'] = (df['net_import'] > df['net_import'].quantile(0.75)).astype(int)

    print("Created features:")
    new_features = [
        'price_vs_seasonal', 'price_vs_seasonal_pct',
        'forecast_vs_seasonal_load', 'price_percentile',
        'is_negative_price', 'is_high_volatility', 'is_high_import'
    ]
    for f in new_features:
        print(f"  {f}")

    return df


def build_daily_prediction_dataset(df: pd.DataFrame, seasonal: dict) -> pd.DataFrame:
    """
    Build dataset for day-ahead prediction.
    Each row = prediction for one hour of D+1, made at 23:00 on day D.
    """
    print("\n" + "=" * 70)
    print("BUILDING DAILY PREDICTION DATASET")
    print("=" * 70)

    df['date'] = pd.to_datetime(df['datetime']).dt.date
    dates = sorted(df['date'].unique())

    rows = []

    for i, target_date in enumerate(dates):
        if i == 0:
            continue

        today_date = dates[i-1]

        # Today's data (day D) - all known at 23:00
        today = df[df['date'] == today_date]
        if len(today) < 20:
            continue

        # Tomorrow's data (day D+1) - prices known, load is target
        tomorrow = df[df['date'] == target_date]
        if len(tomorrow) < 20:
            continue

        # === TODAY'S FEATURES ===
        today_mean_error = today['load_error'].mean()
        today_mean_price = today['price_eur_mwh'].mean()
        today_price_std = today['price_eur_mwh'].std()
        today_vs_seasonal_load = (today['actual_load_mw'] - today['seasonal_load']).mean()
        today_vs_seasonal_price = (today['price_eur_mwh'] - today['seasonal_price']).mean()

        # Today's errors by period
        today_morning_error = today[today['hour'].between(6, 12)]['load_error'].mean()
        today_evening_error = today[today['hour'].between(18, 23)]['load_error'].mean()

        # Today's price-load relationship
        today_price_load_corr = today['price_eur_mwh'].corr(today['actual_load_mw'])

        # Week ago
        week_ago_idx = i - 7
        if week_ago_idx >= 0:
            week_ago_date = dates[week_ago_idx]
            week_ago = df[df['date'] == week_ago_date]
            week_ago_mean_error = week_ago['load_error'].mean() if len(week_ago) >= 20 else 0
        else:
            week_ago_mean_error = 0

        # === CREATE ROW FOR EACH HOUR OF TOMORROW ===
        for _, row in tomorrow.iterrows():
            hour = row['hour']

            # Get today's same hour data
            today_same_hour = today[today['hour'] == hour]
            today_same_hour_error = today_same_hour['load_error'].values[0] if len(today_same_hour) > 0 else 0
            today_same_hour_price = today_same_hour['price_eur_mwh'].values[0] if len(today_same_hour) > 0 else today_mean_price

            features = {
                'datetime': row['datetime'],
                'date': target_date,
                'hour': hour,
                'dow': row['day_of_week'],
                'is_weekend': row['is_weekend'],

                # Targets
                'actual_load': row['actual_load_mw'],
                'forecast_load': row['forecast_load_mw'],
                'target_error': row['load_error'],

                # === TOMORROW'S PRICES (KNOWN!) ===
                'tomorrow_price': row['price_eur_mwh'],
                'tomorrow_price_vs_seasonal': row['price_eur_mwh'] - seasonal['price'].get((row['day_of_week'], hour), row['price_eur_mwh']),
                'tomorrow_net_import': row['net_import'],
                'tomorrow_is_negative_price': int(row['price_eur_mwh'] < 0),

                # Tomorrow's forecast vs seasonal
                'forecast_vs_seasonal': row['forecast_load_mw'] - seasonal['load'].get((row['day_of_week'], hour), row['forecast_load_mw']),

                # Seasonal error expectation
                'seasonal_error': seasonal['error'].get((row['day_of_week'], hour), 0),

                # === TODAY'S FEATURES ===
                'today_same_hour_error': today_same_hour_error,
                'today_same_hour_price': today_same_hour_price,
                'today_mean_error': today_mean_error,
                'today_mean_price': today_mean_price,
                'today_price_std': today_price_std,
                'today_vs_seasonal_load': today_vs_seasonal_load,
                'today_vs_seasonal_price': today_vs_seasonal_price,
                'today_morning_error': today_morning_error,
                'today_evening_error': today_evening_error,

                # Price change: tomorrow vs today (same hour)
                'price_change_vs_today': row['price_eur_mwh'] - today_same_hour_price,

                # Week ago
                'week_ago_mean_error': week_ago_mean_error,
            }

            rows.append(features)

    result = pd.DataFrame(rows)
    print(f"Created {len(result)} prediction samples")

    return result


def evaluate_price_features(df: pd.DataFrame):
    """Evaluate which price features help predict load errors."""
    print("\n" + "=" * 70)
    print("EVALUATING PRICE FEATURES")
    print("=" * 70)

    # Features to test
    all_features = [
        # Time
        'hour', 'dow', 'is_weekend',
        # Seasonal
        'seasonal_error', 'forecast_vs_seasonal',
        # Today's errors
        'today_same_hour_error', 'today_mean_error',
        'today_morning_error', 'today_evening_error',
        # Today's price/load
        'today_vs_seasonal_load', 'today_vs_seasonal_price',
        # Tomorrow's prices (THE NEW FEATURES)
        'tomorrow_price', 'tomorrow_price_vs_seasonal',
        'tomorrow_net_import', 'tomorrow_is_negative_price',
        'price_change_vs_today',
        # Week ago
        'week_ago_mean_error',
    ]

    feature_cols = [f for f in all_features if f in df.columns]
    df_valid = df.dropna(subset=feature_cols + ['target_error'])

    # Split
    df_valid['date'] = pd.to_datetime(df_valid['date'])
    split_date = '2025-07-01'
    train = df_valid[df_valid['date'] < split_date]
    test = df_valid[df_valid['date'] >= split_date]

    print(f"\nTrain: {len(train)} samples")
    print(f"Test:  {len(test)} samples")

    # Model 1: Without price features
    features_no_price = [
        'hour', 'dow', 'is_weekend',
        'seasonal_error', 'forecast_vs_seasonal',
        'today_same_hour_error', 'today_mean_error',
        'today_morning_error', 'today_evening_error',
        'today_vs_seasonal_load',
        'week_ago_mean_error',
    ]
    features_no_price = [f for f in features_no_price if f in df_valid.columns]

    model_no_price = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_no_price.fit(train[features_no_price], train['target_error'])

    test_no_price = test.copy()
    test_no_price['pred_error'] = model_no_price.predict(test[features_no_price])
    test_no_price['corrected'] = test_no_price['forecast_load'] + test_no_price['pred_error']

    baseline_mae = (test['actual_load'] - test['forecast_load']).abs().mean()
    no_price_mae = (test_no_price['actual_load'] - test_no_price['corrected']).abs().mean()

    # Model 2: With price features
    features_with_price = features_no_price + [
        'tomorrow_price', 'tomorrow_price_vs_seasonal',
        'tomorrow_net_import', 'tomorrow_is_negative_price',
        'price_change_vs_today', 'today_vs_seasonal_price',
    ]
    features_with_price = [f for f in features_with_price if f in df_valid.columns]

    model_with_price = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_with_price.fit(train[features_with_price], train['target_error'])

    test_with_price = test.copy()
    test_with_price['pred_error'] = model_with_price.predict(test[features_with_price])
    test_with_price['corrected'] = test_with_price['forecast_load'] + test_with_price['pred_error']

    with_price_mae = (test_with_price['actual_load'] - test_with_price['corrected']).abs().mean()

    # Results
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"{'Model':<35} {'MAE (MW)':<12} {'vs Baseline':<12}")
    print("-" * 50)
    print(f"{'Baseline (DAMAS)':<35} {baseline_mae:<12.1f} {'-':<12}")
    print(f"{'Model WITHOUT price features':<35} {no_price_mae:<12.1f} {(1-no_price_mae/baseline_mae)*100:+.1f}%")
    print(f"{'Model WITH price features':<35} {with_price_mae:<12.1f} {(1-with_price_mae/baseline_mae)*100:+.1f}%")
    print("-" * 50)

    price_improvement = (no_price_mae - with_price_mae) / no_price_mae * 100
    print(f"\nAdditional improvement from prices: {price_improvement:+.1f}%")

    # Feature importance
    importance = pd.DataFrame({
        'feature': features_with_price,
        'importance': model_with_price.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n--- Feature Importance (with prices) ---")
    print(importance.head(15).to_string(index=False))

    # Improvement by hour
    hourly = test_with_price.groupby('hour').apply(lambda x: pd.Series({
        'baseline': (x['actual_load'] - x['forecast_load']).abs().mean(),
        'no_price': (x['actual_load'] - (x['forecast_load'] + model_no_price.predict(x[features_no_price]))).abs().mean(),
        'with_price': (x['actual_load'] - x['corrected']).abs().mean(),
    }))
    hourly['price_helps'] = (hourly['no_price'] - hourly['with_price']) / hourly['no_price'] * 100

    print("\n--- Hours where price features help most ---")
    top_hours = hourly.nlargest(5, 'price_helps')
    for hour, row in top_hours.iterrows():
        print(f"  Hour {hour}: {row['price_helps']:+.1f}% improvement from prices")

    return model_with_price, importance, test_with_price, features_with_price


def create_plots(df: pd.DataFrame, test: pd.DataFrame, importance: pd.DataFrame):
    """Create analysis plots."""
    print("\n" + "=" * 70)
    print("CREATING PLOTS")
    print("=" * 70)

    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Price vs Load Error scatter
    sample = df.sample(min(5000, len(df)), random_state=42)
    axes[0, 0].scatter(sample['price_eur_mwh'], sample['load_error'], alpha=0.2, s=5)
    axes[0, 0].axhline(0, color='red', linestyle='--')
    corr = df['price_eur_mwh'].corr(df['load_error'])
    axes[0, 0].set_xlabel('DA Price (EUR/MWh)')
    axes[0, 0].set_ylabel('Load Forecast Error (MW)')
    axes[0, 0].set_title(f'Price vs Load Error (corr={corr:.3f})')

    # 2. Price-Error correlation by hour
    hourly_corr = df.groupby('hour').apply(
        lambda x: x['price_eur_mwh'].corr(x['load_error'])
    )
    colors = ['green' if c > 0 else 'red' for c in hourly_corr]
    axes[0, 1].bar(hourly_corr.index, hourly_corr.values, color=colors, edgecolor='black')
    axes[0, 1].axhline(0, color='black')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].set_title('Price-Error Correlation by Hour')

    # 3. Feature importance
    top_n = min(12, len(importance))
    axes[0, 2].barh(range(top_n), importance['importance'].head(top_n), color='steelblue', edgecolor='black')
    axes[0, 2].set_yticks(range(top_n))
    axes[0, 2].set_yticklabels(importance['feature'].head(top_n), fontsize=9)
    axes[0, 2].set_xlabel('Importance')
    axes[0, 2].set_title('Feature Importance (with Price Features)')

    # 4. Error distribution comparison
    baseline_err = test['actual_load'] - test['forecast_load']
    model_err = test['actual_load'] - test['corrected']
    axes[1, 0].hist(baseline_err, bins=50, alpha=0.5, label=f'Baseline ({baseline_err.abs().mean():.1f})', color='red')
    axes[1, 0].hist(model_err, bins=50, alpha=0.5, label=f'With Prices ({model_err.abs().mean():.1f})', color='green')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Error (MW)')
    axes[1, 0].set_title('Error Distribution')

    # 5. Sample week
    start = test['datetime'].min() + pd.Timedelta(days=14)
    sample_week = test[(test['datetime'] >= start) & (test['datetime'] < start + pd.Timedelta(days=7))]
    axes[1, 1].plot(sample_week['datetime'], sample_week['actual_load'], 'b-', label='Actual', linewidth=1.5)
    axes[1, 1].plot(sample_week['datetime'], sample_week['forecast_load'], 'r--', label='Baseline', alpha=0.7)
    axes[1, 1].plot(sample_week['datetime'], sample_week['corrected'], 'g-', label='With Prices', alpha=0.7)
    axes[1, 1].legend()
    axes[1, 1].set_title('Sample Week')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # 6. Improvement by hour
    hourly = test.groupby('hour').apply(lambda x: pd.Series({
        'baseline': (x['actual_load'] - x['forecast_load']).abs().mean(),
        'model': (x['actual_load'] - x['corrected']).abs().mean(),
    }))
    hourly['improvement'] = (1 - hourly['model'] / hourly['baseline']) * 100
    colors = ['green' if x > 0 else 'red' for x in hourly['improvement']]
    axes[1, 2].bar(hourly.index, hourly['improvement'], color=colors, edgecolor='black')
    axes[1, 2].axhline(0, color='black')
    axes[1, 2].set_xlabel('Hour')
    axes[1, 2].set_ylabel('Improvement (%)')
    axes[1, 2].set_title('Improvement by Hour (with Price Features)')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '24_price_features_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 24_price_features_analysis.png")


def main():
    # Load and merge data
    df = load_data()

    # Create seasonal patterns
    seasonal = create_seasonal_patterns(df)

    # Analyze price-error correlation
    analyze_price_error_correlation(df)

    # Create price features
    df = create_price_features_for_prediction(df, seasonal)

    # Build daily prediction dataset
    df_daily = build_daily_prediction_dataset(df, seasonal)

    # Evaluate price features
    model, importance, test, features = evaluate_price_features(df_daily)

    # Create plots
    create_plots(df, test, importance)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: HOW TO USE PRICE FEATURES")
    print("=" * 70)
    print("""
Key price features for load prediction (available at 23:00 for D+1):

1. TOMORROW'S PRICE (from DA market):
   - tomorrow_price: Absolute price level
   - tomorrow_price_vs_seasonal: Is price unusually high/low?
   - tomorrow_is_negative_price: Binary flag for oversupply

2. TOMORROW'S NET IMPORT:
   - tomorrow_net_import: High import = oversupply expected

3. PRICE CHANGE:
   - price_change_vs_today: Tomorrow vs today same hour

4. TODAY'S PRICE CONTEXT:
   - today_vs_seasonal_price: Was today's price unusual?

WHY PRICES HELP:
- Prices reflect market expectations of demand/supply
- High prices -> expected tight supply -> possibly higher actual load
- Negative prices -> oversupply (solar/wind) -> different load patterns
- Price deviations from seasonal indicate unusual conditions
""")


if __name__ == '__main__':
    main()
