"""
3-Minute Data Features for Day-Ahead Forecasting

The 3-minute SCADA data is more granular than hourly DAMAS forecasts.
This analysis extracts sub-hourly patterns that might help predict
next-day forecast errors.

Potential signals:
1. Within-hour error variance - volatile days might predict volatile tomorrows
2. Ramp behavior - how well does DAMAS predict load changes?
3. Error autocorrelation at 3-min level - persistence patterns
4. Sub-hourly trends - systematic within-hour biases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Paths
BASE_PATH = Path(__file__).parent.parent.parent
LOAD_3MIN_PATH = BASE_PATH / 'data' / 'features' / 'load_3min.csv'
DAMAS_PATH = BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet'
OUTPUT_PATH = Path(__file__).parent / 'threeminute_analysis'
OUTPUT_PATH.mkdir(exist_ok=True)


def load_data():
    """Load both 3-minute SCADA and hourly DAMAS data."""
    print("Loading data...")

    # 3-minute SCADA load
    df_3min = pd.read_csv(LOAD_3MIN_PATH)
    df_3min['datetime'] = pd.to_datetime(df_3min['datetime'])
    df_3min = df_3min.sort_values('datetime').reset_index(drop=True)
    print(f"  3-min data: {len(df_3min):,} records")
    print(f"    Range: {df_3min['datetime'].min()} to {df_3min['datetime'].max()}")

    # Hourly DAMAS data
    df_hourly = pd.read_parquet(DAMAS_PATH)
    df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
    df_hourly['load_error'] = df_hourly['actual_load_mw'] - df_hourly['forecast_load_mw']
    print(f"  Hourly DAMAS: {len(df_hourly):,} records")

    return df_3min, df_hourly


def merge_3min_with_hourly(df_3min: pd.DataFrame, df_hourly: pd.DataFrame):
    """
    Merge 3-minute data with hourly forecasts.
    Assign each 3-min observation to its corresponding hour.
    """
    print("\nMerging 3-minute with hourly data...")

    # Add hour key to 3-min data (floor to hour)
    df_3min['hour_datetime'] = df_3min['datetime'].dt.floor('h')

    # Merge with hourly forecast
    df_merged = df_3min.merge(
        df_hourly[['datetime', 'forecast_load_mw', 'actual_load_mw', 'load_error']],
        left_on='hour_datetime',
        right_on='datetime',
        how='inner',
        suffixes=('_3min', '_hourly')
    )

    # Rename for clarity
    df_merged = df_merged.rename(columns={
        'load_mw': 'actual_3min',
        'datetime_3min': 'datetime',
        'actual_load_mw': 'actual_hourly',
        'forecast_load_mw': 'forecast_hourly',
        'load_error': 'hourly_error'
    })

    # Calculate 3-min level error (vs hourly forecast)
    df_merged['error_3min'] = df_merged['actual_3min'] - df_merged['forecast_hourly']

    # Time features
    df_merged['date'] = df_merged['datetime'].dt.date
    df_merged['hour'] = df_merged['datetime'].dt.hour
    df_merged['minute'] = df_merged['datetime'].dt.minute
    df_merged['minute_of_hour'] = df_merged['minute']  # 0, 3, 6, ..., 57
    df_merged['day_of_week'] = df_merged['datetime'].dt.dayofweek
    df_merged['month'] = df_merged['datetime'].dt.month
    df_merged['year'] = df_merged['datetime'].dt.year

    print(f"  Merged: {len(df_merged):,} records")

    return df_merged


def analyze_subhourly_patterns(df: pd.DataFrame):
    """
    Analyze error patterns within each hour.
    Does DAMAS have systematic biases at certain minutes?
    """
    print("\n" + "="*60)
    print("SUB-HOURLY ERROR PATTERNS")
    print("="*60)

    # Error by minute within hour
    minute_patterns = df.groupby('minute_of_hour').agg({
        'error_3min': ['mean', 'std', 'count'],
        'actual_3min': 'mean'
    })
    minute_patterns.columns = ['mean_error', 'std_error', 'count', 'mean_load']

    print("\nError by minute within hour:")
    print("  Minute | Mean Error | Std | Load")
    print("  " + "-"*45)
    for minute, row in minute_patterns.iterrows():
        print(f"    {minute:2d}   |  {row['mean_error']:+6.1f} MW | {row['std_error']:5.1f} | {row['mean_load']:.0f}")

    # Is there a systematic pattern?
    min_error = minute_patterns['mean_error'].min()
    max_error = minute_patterns['mean_error'].max()
    range_error = max_error - min_error

    print(f"\n  Error range across minutes: {range_error:.1f} MW")
    print(f"  Min at minute {minute_patterns['mean_error'].idxmin()}: {min_error:+.1f} MW")
    print(f"  Max at minute {minute_patterns['mean_error'].idxmax()}: {max_error:+.1f} MW")

    # Within-hour trend (early vs late in hour)
    df['early_hour'] = df['minute_of_hour'] < 30
    early_error = df[df['early_hour']]['error_3min'].mean()
    late_error = df[~df['early_hour']]['error_3min'].mean()

    print(f"\n  First half of hour (0-27 min): {early_error:+.1f} MW")
    print(f"  Second half (30-57 min): {late_error:+.1f} MW")
    print(f"  Difference: {late_error - early_error:+.1f} MW")

    return minute_patterns


def extract_3min_daily_features(df: pd.DataFrame):
    """
    Extract daily features from 3-minute data that might predict next-day errors.
    """
    print("\n" + "="*60)
    print("EXTRACTING DAILY 3-MINUTE FEATURES")
    print("="*60)

    daily_features = []

    for date, day_data in df.groupby('date'):
        if len(day_data) < 400:  # Need most of the day
            continue

        day_data = day_data.sort_values('datetime')

        features = {
            'date': date,
            'year': day_data['year'].iloc[0],
            'month': day_data['month'].iloc[0],
            'day_of_week': day_data['day_of_week'].iloc[0],
        }

        # === 3-minute error statistics ===
        features['error_3min_mean'] = day_data['error_3min'].mean()
        features['error_3min_std'] = day_data['error_3min'].std()
        features['error_3min_min'] = day_data['error_3min'].min()
        features['error_3min_max'] = day_data['error_3min'].max()
        features['error_3min_range'] = features['error_3min_max'] - features['error_3min_min']

        # === Within-hour variance (average across hours) ===
        hourly_vars = day_data.groupby('hour')['error_3min'].std()
        features['within_hour_std_mean'] = hourly_vars.mean()
        features['within_hour_std_max'] = hourly_vars.max()

        # === Ramp detection: how fast does load change? ===
        day_data = day_data.copy()
        day_data['load_diff'] = day_data['actual_3min'].diff()
        features['ramp_mean'] = day_data['load_diff'].abs().mean()
        features['ramp_max'] = day_data['load_diff'].abs().max()
        features['ramp_std'] = day_data['load_diff'].std()

        # === Error autocorrelation at 3-min level ===
        if len(day_data) > 10:
            error_series = day_data['error_3min'].values
            # Lag-1 (3 min)
            features['error_autocorr_3min'] = np.corrcoef(error_series[:-1], error_series[1:])[0, 1]
            # Lag-20 (1 hour)
            if len(error_series) > 20:
                features['error_autocorr_1h'] = np.corrcoef(error_series[:-20], error_series[20:])[0, 1]
            else:
                features['error_autocorr_1h'] = np.nan
        else:
            features['error_autocorr_3min'] = np.nan
            features['error_autocorr_1h'] = np.nan

        # === Period-specific features ===
        # Morning ramp (5-9)
        morning = day_data[day_data['hour'].between(5, 9)]
        if len(morning) > 10:
            features['morning_error_mean'] = morning['error_3min'].mean()
            features['morning_error_std'] = morning['error_3min'].std()
            features['morning_ramp'] = morning['load_diff'].mean()
        else:
            features['morning_error_mean'] = np.nan
            features['morning_error_std'] = np.nan
            features['morning_ramp'] = np.nan

        # Evening peak (17-21)
        evening = day_data[day_data['hour'].between(17, 21)]
        if len(evening) > 10:
            features['evening_error_mean'] = evening['error_3min'].mean()
            features['evening_error_std'] = evening['error_3min'].std()
        else:
            features['evening_error_mean'] = np.nan
            features['evening_error_std'] = np.nan

        # Night (0-5)
        night = day_data[day_data['hour'].between(0, 5)]
        if len(night) > 10:
            features['night_error_mean'] = night['error_3min'].mean()
            features['night_error_std'] = night['error_3min'].std()
        else:
            features['night_error_mean'] = np.nan
            features['night_error_std'] = np.nan

        # === Error trend through the day ===
        # Split into 4 quarters
        n = len(day_data)
        q1_error = day_data.iloc[:n//4]['error_3min'].mean()
        q4_error = day_data.iloc[3*n//4:]['error_3min'].mean()
        features['error_trend_q1_q4'] = q4_error - q1_error

        # === Forecast vs actual pattern matching ===
        # How well does 3-min actual follow forecast pattern?
        hourly_means = day_data.groupby('hour')['actual_3min'].mean()
        if len(hourly_means) >= 20:
            forecast_pattern = day_data.groupby('hour')['forecast_hourly'].first()
            common_hours = hourly_means.index.intersection(forecast_pattern.index)
            if len(common_hours) >= 15:
                features['pattern_corr'] = np.corrcoef(
                    hourly_means[common_hours],
                    forecast_pattern[common_hours]
                )[0, 1]
            else:
                features['pattern_corr'] = np.nan
        else:
            features['pattern_corr'] = np.nan

        daily_features.append(features)

    df_daily = pd.DataFrame(daily_features)
    df_daily['date'] = pd.to_datetime(df_daily['date'])

    print(f"  Extracted features for {len(df_daily)} days")

    # Show feature statistics
    print("\n  Feature statistics:")
    for col in ['error_3min_std', 'within_hour_std_mean', 'ramp_mean',
                'error_autocorr_3min', 'error_trend_q1_q4']:
        if col in df_daily.columns:
            print(f"    {col}: mean={df_daily[col].mean():.2f}, std={df_daily[col].std():.2f}")

    return df_daily


def test_3min_features_for_dayahead(df_3min_features: pd.DataFrame, df_hourly: pd.DataFrame):
    """
    Test if 3-minute features from day D help predict day D+1 errors.
    """
    print("\n" + "="*60)
    print("TESTING 3-MIN FEATURES FOR DAY-AHEAD PREDICTION")
    print("="*60)

    # Get daily error summary from hourly data
    df_hourly['date'] = pd.to_datetime(df_hourly['datetime']).dt.date
    daily_errors = df_hourly.groupby('date').agg({
        'load_error': ['mean', 'std', lambda x: x.abs().mean()],
        'day_of_week': 'first',
        'year': 'first'
    })
    daily_errors.columns = ['error_mean', 'error_std', 'error_mae', 'dow', 'year']
    daily_errors = daily_errors.reset_index()
    daily_errors['date'] = pd.to_datetime(daily_errors['date'])

    # Merge: features from day D, target from day D+1
    df_3min_features['next_date'] = df_3min_features['date'] + pd.Timedelta(days=1)

    merged = df_3min_features.merge(
        daily_errors[['date', 'error_mean', 'error_std', 'error_mae']],
        left_on='next_date',
        right_on='date',
        suffixes=('_d1', '_target')
    )

    print(f"\n  Merged dataset: {len(merged)} day pairs")

    # Correlation analysis: which 3-min features predict next-day error?
    print("\n  Correlation with next-day error metrics:")
    print("  Feature                    | r(MAE) | r(mean) | r(std)")
    print("  " + "-"*60)

    feature_cols = ['error_3min_mean', 'error_3min_std', 'error_3min_range',
                    'within_hour_std_mean', 'ramp_mean', 'ramp_std',
                    'error_autocorr_3min', 'error_autocorr_1h',
                    'morning_error_mean', 'evening_error_mean',
                    'error_trend_q1_q4', 'pattern_corr']

    correlations = []
    for feat in feature_cols:
        if feat in merged.columns:
            valid = merged[[feat, 'error_mae', 'error_mean', 'error_std']].dropna()
            if len(valid) > 50:
                r_mae = np.corrcoef(valid[feat], valid['error_mae'])[0, 1]
                r_mean = np.corrcoef(valid[feat], valid['error_mean'])[0, 1]
                r_std = np.corrcoef(valid[feat], valid['error_std'])[0, 1]
                print(f"  {feat:<27} | {r_mae:+.3f} | {r_mean:+.3f} | {r_std:+.3f}")
                correlations.append({'feature': feat, 'r_mae': r_mae, 'r_mean': r_mean, 'r_std': r_std})

    return merged, pd.DataFrame(correlations)


def build_enhanced_dayahead_model(merged: pd.DataFrame, df_hourly: pd.DataFrame):
    """
    Build day-ahead model enhanced with 3-minute features.
    Compare to baseline model without 3-min features.
    """
    print("\n" + "="*60)
    print("ENHANCED DAY-AHEAD MODEL WITH 3-MIN FEATURES")
    print("="*60)

    # Prepare hourly prediction dataset
    df_hourly = df_hourly.copy()
    df_hourly['date'] = pd.to_datetime(df_hourly['datetime']).dt.date
    df_hourly['date'] = pd.to_datetime(df_hourly['date'])

    # Get yesterday's 3-min features for each day
    merged['target_date'] = merged['next_date']

    # Merge 3-min features to hourly data
    df_model = df_hourly.merge(
        merged[['target_date', 'error_3min_mean', 'error_3min_std', 'error_3min_range',
                'within_hour_std_mean', 'ramp_mean', 'error_autocorr_3min',
                'morning_error_mean', 'evening_error_mean', 'error_trend_q1_q4']],
        left_on='date',
        right_on='target_date',
        how='inner'
    )

    print(f"\n  Model dataset: {len(df_model):,} hourly records")

    # Create seasonal baseline features
    df_2024 = df_model[df_model['year'] == 2024]
    seasonal = df_2024.groupby(['day_of_week', 'hour'])['load_error'].mean().to_dict()
    df_model['seasonal_error'] = df_model.apply(
        lambda x: seasonal.get((x['day_of_week'], x['hour']), 0), axis=1
    )

    # Train/test split
    train = df_model[df_model['year'] < 2025]
    test = df_model[df_model['year'] >= 2025]

    print(f"  Train: {len(train):,}, Test: {len(test):,}")

    # === Baseline model (no 3-min features) ===
    baseline_features = ['hour', 'day_of_week', 'is_weekend', 'seasonal_error']

    model_baseline = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=4,
        min_child_samples=20, random_state=42, verbose=-1
    )
    model_baseline.fit(train[baseline_features], train['load_error'])
    pred_baseline = model_baseline.predict(test[baseline_features])

    baseline_mae = np.abs(test['load_error']).mean()
    model_baseline_mae = np.abs(test['load_error'] - pred_baseline).mean()

    print(f"\n  Baseline (no 3-min features):")
    print(f"    DAMAS MAE: {baseline_mae:.1f} MW")
    print(f"    Model MAE: {model_baseline_mae:.1f} MW ({(1-model_baseline_mae/baseline_mae)*100:+.1f}%)")

    # === Enhanced model (with 3-min features) ===
    enhanced_features = baseline_features + [
        'error_3min_mean', 'error_3min_std', 'error_3min_range',
        'within_hour_std_mean', 'ramp_mean', 'error_autocorr_3min',
        'morning_error_mean', 'evening_error_mean', 'error_trend_q1_q4'
    ]

    # Drop rows with NaN in features
    train_valid = train.dropna(subset=enhanced_features)
    test_valid = test.dropna(subset=enhanced_features)

    model_enhanced = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=4,
        min_child_samples=20, random_state=42, verbose=-1
    )
    model_enhanced.fit(train_valid[enhanced_features], train_valid['load_error'])
    pred_enhanced = model_enhanced.predict(test_valid[enhanced_features])

    baseline_mae_valid = np.abs(test_valid['load_error']).mean()
    model_enhanced_mae = np.abs(test_valid['load_error'] - pred_enhanced).mean()

    print(f"\n  Enhanced (with 3-min features):")
    print(f"    DAMAS MAE: {baseline_mae_valid:.1f} MW")
    print(f"    Model MAE: {model_enhanced_mae:.1f} MW ({(1-model_enhanced_mae/baseline_mae_valid)*100:+.1f}%)")

    # Feature importance
    importance = pd.DataFrame({
        'feature': enhanced_features,
        'importance': model_enhanced.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n  Feature Importance (Enhanced Model):")
    for _, row in importance.head(10).iterrows():
        marker = " <-- 3-min" if row['feature'] not in baseline_features else ""
        print(f"    {row['feature']:<25}: {row['importance']:.0f}{marker}")

    # Calculate improvement from 3-min features
    improvement_from_3min = model_baseline_mae - model_enhanced_mae

    print(f"\n  IMPROVEMENT FROM 3-MIN FEATURES: {improvement_from_3min:.1f} MW")
    print(f"  Relative improvement: {(improvement_from_3min/model_baseline_mae)*100:+.1f}%")

    return {
        'baseline_mae': model_baseline_mae,
        'enhanced_mae': model_enhanced_mae,
        'damas_mae': baseline_mae,
        'importance': importance
    }


def create_plots(minute_patterns: pd.DataFrame, correlations: pd.DataFrame,
                 model_results: dict):
    """Create visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Sub-hourly error pattern
    axes[0, 0].bar(minute_patterns.index, minute_patterns['mean_error'],
                   color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axhline(0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Minute of Hour')
    axes[0, 0].set_ylabel('Mean Error (MW)')
    axes[0, 0].set_title('DAMAS Error by Minute Within Hour\n(Systematic sub-hourly bias?)')

    # 2. Feature correlations with next-day error
    if len(correlations) > 0:
        corr_sorted = correlations.sort_values('r_mae', key=abs, ascending=True)
        colors = ['green' if x > 0 else 'red' for x in corr_sorted['r_mae']]
        axes[0, 1].barh(range(len(corr_sorted)), corr_sorted['r_mae'],
                       color=colors, edgecolor='black', alpha=0.7)
        axes[0, 1].set_yticks(range(len(corr_sorted)))
        axes[0, 1].set_yticklabels(corr_sorted['feature'], fontsize=9)
        axes[0, 1].axvline(0, color='black', linewidth=1)
        axes[0, 1].set_xlabel('Correlation with Next-Day MAE')
        axes[0, 1].set_title('3-Min Features vs Next-Day Error')

    # 3. Model comparison
    if model_results:
        models = ['DAMAS\nBaseline', 'Calendar\nOnly', 'Calendar +\n3-Min Features']
        maes = [model_results['damas_mae'], model_results['baseline_mae'],
                model_results['enhanced_mae']]
        colors = ['gray', 'orange', 'green']
        bars = axes[1, 0].bar(models, maes, color=colors, edgecolor='black')

        for bar, mae in zip(bars, maes):
            imp = (1 - mae/maes[0]) * 100
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, mae + 1,
                           f'{mae:.1f}\n({imp:+.1f}%)', ha='center', fontsize=10)

        axes[1, 0].set_ylabel('MAE (MW)')
        axes[1, 0].set_title('Model Comparison\n(Does 3-min data help day-ahead?)')

    # 4. Feature importance
    if model_results and 'importance' in model_results:
        imp = model_results['importance'].head(10)
        colors = ['green' if 'error_3min' in f or 'ramp' in f or 'within' in f or 'autocorr' in f
                  else 'steelblue' for f in imp['feature']]
        axes[1, 1].barh(range(len(imp)), imp['importance'].values, color=colors, edgecolor='black')
        axes[1, 1].set_yticks(range(len(imp)))
        axes[1, 1].set_yticklabels(imp['feature'].values, fontsize=9)
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_title('Feature Importance\n(Green = 3-min derived)')
        axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '01_threeminute_features.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Saved: 01_threeminute_features.png")


def main():
    print("="*60)
    print("3-MINUTE DATA FEATURES FOR DAY-AHEAD FORECASTING")
    print("="*60)

    # Load data
    df_3min, df_hourly = load_data()

    # Merge 3-min with hourly
    df_merged = merge_3min_with_hourly(df_3min, df_hourly)

    # Analyze sub-hourly patterns
    minute_patterns = analyze_subhourly_patterns(df_merged)

    # Extract daily features from 3-min data
    df_3min_features = extract_3min_daily_features(df_merged)

    # Test if 3-min features help predict next-day errors
    merged, correlations = test_3min_features_for_dayahead(df_3min_features, df_hourly)

    # Build enhanced model
    model_results = build_enhanced_dayahead_model(merged, df_hourly)

    # Create plots
    print("\n" + "="*60)
    print("CREATING PLOTS")
    print("="*60)
    create_plots(minute_patterns, correlations, model_results)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"""
3-Minute Data Analysis for Day-Ahead Forecasting
=================================================

Key Questions:
1. Does DAMAS have sub-hourly biases?
2. Do 3-min patterns from day D predict day D+1 errors?
3. Can 3-min features improve day-ahead model?

Results:
- DAMAS MAE:                    {model_results['damas_mae']:.1f} MW
- Calendar-only model:          {model_results['baseline_mae']:.1f} MW ({(1-model_results['baseline_mae']/model_results['damas_mae'])*100:+.1f}%)
- Calendar + 3-min features:    {model_results['enhanced_mae']:.1f} MW ({(1-model_results['enhanced_mae']/model_results['damas_mae'])*100:+.1f}%)

Improvement from 3-min features: {model_results['baseline_mae'] - model_results['enhanced_mae']:.1f} MW
""")


if __name__ == '__main__':
    main()
