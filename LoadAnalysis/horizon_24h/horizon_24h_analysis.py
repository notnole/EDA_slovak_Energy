"""
24-Hour Horizon Forecast Analysis

REALISTIC CONSTRAINTS:
- At time T, we predict load at T+24
- Available information at prediction time T:
  * Actual load up to T-1 (full hour completed)
  * 3-min load data within current hour T (partial)
  * Forecast errors up to T-1
  * Baseline forecast for T+24 (known day-ahead)
  * Calendar features (hour, day_of_week, etc.)

QUESTIONS:
1. How predictable is the error at 24h horizon?
2. What 3-min features help?
3. What's achievable without weather/price data?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

# Paths
BASE_PATH = Path(__file__).parent.parent
HOURLY_PATH = BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet'
MIN3_PATH = BASE_PATH / 'data' / 'features' / 'load_3min.csv'
PLOT_PATH = Path(__file__).parent / 'plots'


def load_hourly_data() -> pd.DataFrame:
    """Load hourly data with forecasts."""
    df = pd.read_parquet(HOURLY_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']
    return df


def load_3min_data() -> pd.DataFrame:
    """Load 3-min resolution data."""
    df = pd.read_csv(MIN3_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    return df


def create_3min_hourly_features(df_3min: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 3-min data to hourly features.
    These capture within-hour dynamics that hourly data misses.
    """
    print("\n--- Creating 3-min based hourly features ---")

    # Create hour identifier
    df_3min = df_3min.copy()
    df_3min['hour_start'] = df_3min.index.floor('H')

    # Aggregate features per hour
    hourly_features = df_3min.groupby('hour_start').agg(
        load_mean=('load_mw', 'mean'),
        load_std=('load_mw', 'std'),
        load_min=('load_mw', 'min'),
        load_max=('load_mw', 'max'),
        load_range=('load_mw', lambda x: x.max() - x.min()),
        load_first=('load_mw', 'first'),
        load_last=('load_mw', 'last'),
        n_samples=('load_mw', 'count')
    ).reset_index()

    # Derived features
    hourly_features['load_trend'] = hourly_features['load_last'] - hourly_features['load_first']
    hourly_features['load_cv'] = hourly_features['load_std'] / hourly_features['load_mean'] * 100  # Coefficient of variation
    hourly_features['load_skew'] = hourly_features.apply(
        lambda row: 2 * (row['load_mean'] - (row['load_min'] + row['load_max'])/2) / (row['load_range'] + 1e-6),
        axis=1
    )

    hourly_features = hourly_features.rename(columns={'hour_start': 'datetime'})
    hourly_features = hourly_features.set_index('datetime')

    print(f"Created {len(hourly_features)} hourly records with 3-min features")
    print(f"Features: {list(hourly_features.columns)}")

    return hourly_features


def merge_data(df_hourly: pd.DataFrame, df_3min_features: pd.DataFrame) -> pd.DataFrame:
    """Merge hourly forecast data with 3-min features."""
    # Align indices
    df = df_hourly.join(df_3min_features, how='inner')
    print(f"\nMerged data: {len(df)} records")
    return df


def create_24h_horizon_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features available at T for predicting T+24.

    At time T, we know:
    - Errors up to T-1 (lag >= 1)
    - 3-min features up to T-1
    - Calendar features for T+24
    """
    print("\n--- Creating 24h horizon features ---")

    df = df.copy()

    # Target: error at T+24 (what we want to predict)
    df['target_error_24h'] = df['error'].shift(-24)

    # === LAGGED ERROR FEATURES (available at T for predicting T+24) ===
    # These are errors from T-1, T-2, etc. (already happened)
    df['error_lag1'] = df['error'].shift(1)   # Error 1 hour ago
    df['error_lag2'] = df['error'].shift(2)
    df['error_lag3'] = df['error'].shift(3)
    df['error_lag6'] = df['error'].shift(6)
    df['error_lag12'] = df['error'].shift(12)
    df['error_lag24'] = df['error'].shift(24)  # Error 24 hours ago (same hour yesterday)
    df['error_lag25'] = df['error'].shift(25)  # Error 25 hours ago
    df['error_lag48'] = df['error'].shift(48)  # Error 2 days ago
    df['error_lag168'] = df['error'].shift(168)  # Error 1 week ago

    # Rolling error statistics
    df['error_rolling_6h_mean'] = df['error'].rolling(6).mean().shift(1)
    df['error_rolling_6h_std'] = df['error'].rolling(6).std().shift(1)
    df['error_rolling_24h_mean'] = df['error'].rolling(24).mean().shift(1)
    df['error_rolling_24h_std'] = df['error'].rolling(24).std().shift(1)

    # Same-hour features (error at same hour in previous days)
    df['error_same_hour_1d'] = df.groupby('hour')['error'].shift(1)  # Yesterday same hour
    df['error_same_hour_2d'] = df.groupby('hour')['error'].shift(2)
    df['error_same_hour_7d'] = df.groupby('hour')['error'].shift(7)  # Last week same hour

    # === 3-MIN BASED FEATURES (lagged, available at T) ===
    df['load_std_lag1'] = df['load_std'].shift(1)
    df['load_std_lag24'] = df['load_std'].shift(24)
    df['load_trend_lag1'] = df['load_trend'].shift(1)
    df['load_trend_lag24'] = df['load_trend'].shift(24)
    df['load_cv_lag1'] = df['load_cv'].shift(1)
    df['load_cv_lag24'] = df['load_cv'].shift(24)
    df['load_range_lag1'] = df['load_range'].shift(1)
    df['load_range_lag24'] = df['load_range'].shift(24)

    # === CALENDAR FEATURES FOR T+24 ===
    # These are known in advance
    df['target_hour'] = df['hour']  # Same hour (24h ahead)
    df['target_dow'] = df['day_of_week']
    df['target_month'] = df['month']
    df['target_is_weekend'] = df['is_weekend']

    # === FORECAST FEATURES ===
    # The baseline forecast for T+24 is known (day-ahead market)
    df['forecast_24h'] = df['forecast_load_mw'].shift(-24)  # Known at T
    df['forecast_diff_24h'] = df['forecast_load_mw'].diff(24).shift(1)  # Change from yesterday's forecast

    # Count features
    n_features = len([c for c in df.columns if c.startswith(('error_', 'load_', 'target_', 'forecast_'))])
    print(f"Created {n_features} features for 24h prediction")

    return df


def analyze_24h_error_correlation(df: pd.DataFrame):
    """Analyze which features correlate with 24h ahead error."""
    print("\n" + "="*70)
    print("24H HORIZON ERROR CORRELATION ANALYSIS")
    print("="*70)

    df_valid = df.dropna(subset=['target_error_24h']).copy()

    # Features to check
    feature_cols = [
        'error_lag1', 'error_lag2', 'error_lag3', 'error_lag6', 'error_lag12',
        'error_lag24', 'error_lag25', 'error_lag48', 'error_lag168',
        'error_rolling_6h_mean', 'error_rolling_24h_mean',
        'error_same_hour_1d', 'error_same_hour_2d', 'error_same_hour_7d',
        'load_std_lag1', 'load_std_lag24', 'load_trend_lag1', 'load_trend_lag24',
        'load_cv_lag1', 'load_cv_lag24', 'load_range_lag1', 'load_range_lag24',
    ]

    # Compute correlations
    correlations = {}
    for col in feature_cols:
        if col in df_valid.columns:
            corr = df_valid['target_error_24h'].corr(df_valid[col])
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nCorrelation with 24h-ahead error:")
    print("-" * 50)
    for col, corr in sorted_corr:
        bar = '|' + '*' * int(abs(corr) * 40) + ' ' * (40 - int(abs(corr) * 40)) + '|'
        print(f"  {col:30s}: {corr:+.3f} {bar}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Correlation bar chart
    labels = [c.replace('error_', 'err_').replace('load_', 'ld_') for c, _ in sorted_corr[:15]]
    values = [v for _, v in sorted_corr[:15]]
    colors = ['green' if v > 0 else 'red' for v in values]
    axes[0].barh(range(len(labels)), values, color=colors, edgecolor='black', alpha=0.7)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_xlabel('Correlation with 24h-ahead Error')
    axes[0].set_title('Feature Correlation with Target (24h Horizon)')
    axes[0].axvline(x=0, color='black', linestyle='-')

    # Compare lag-1 vs lag-24 error predictability
    if 'error_lag1' in df_valid.columns and 'error_lag24' in df_valid.columns:
        sample = df_valid.sample(min(3000, len(df_valid)))
        axes[1].scatter(sample['error_lag24'], sample['target_error_24h'], alpha=0.3, s=10, label='lag-24')
        axes[1].plot([-300, 300], [-300, 300], 'r--', linewidth=2)
        axes[1].set_xlabel('Error 24h Ago (Same Hour Yesterday)')
        axes[1].set_ylabel('Error 24h Ahead (Target)')
        axes[1].set_title(f'Same-Hour Error Pattern (r={correlations.get("error_lag24", 0):.3f})')
        axes[1].set_xlim(-300, 300)
        axes[1].set_ylim(-300, 300)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '19_24h_horizon_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Saved: 19_24h_horizon_correlation.png")

    return sorted_corr


def train_24h_models(df: pd.DataFrame):
    """Train and evaluate models for 24h ahead error prediction."""
    print("\n" + "="*70)
    print("24H HORIZON MODEL TRAINING")
    print("="*70)

    # Define feature sets
    basic_features = ['error_lag24', 'error_same_hour_1d', 'error_same_hour_7d',
                      'error_rolling_24h_mean', 'target_hour', 'target_dow', 'target_is_weekend']

    extended_features = basic_features + [
        'error_lag1', 'error_lag25', 'error_lag48', 'error_lag168',
        'error_rolling_6h_mean', 'error_rolling_24h_std',
        'load_std_lag24', 'load_trend_lag24', 'load_cv_lag24', 'load_range_lag24',
        'target_month'
    ]

    # Prepare data
    df_valid = df.dropna(subset=['target_error_24h'] + extended_features).copy()

    # Train/test split (time-based)
    split_date = '2025-07-01'
    train = df_valid[df_valid.index < split_date]
    test = df_valid[df_valid.index >= split_date]

    print(f"\nTrain: {len(train)} samples (before {split_date})")
    print(f"Test: {len(test)} samples (from {split_date})")

    results = []

    # Baseline metrics
    baseline_mae = test['target_error_24h'].abs().mean()
    results.append({
        'Model': 'Naive (predict 0)',
        'Features': 'None',
        'R2': 0,
        'MAE': baseline_mae,
        'MAE_Reduction': 0
    })

    # Persistence: use error from 24h ago
    persistence_pred = test['error_lag24']
    results.append({
        'Model': 'Persistence (lag-24)',
        'Features': '1',
        'R2': r2_score(test['target_error_24h'], persistence_pred),
        'MAE': mean_absolute_error(test['target_error_24h'], persistence_pred),
        'MAE_Reduction': (1 - mean_absolute_error(test['target_error_24h'], persistence_pred) / baseline_mae) * 100
    })

    # Same-hour yesterday
    same_hour_pred = test['error_same_hour_1d']
    results.append({
        'Model': 'Same Hour Yesterday',
        'Features': '1',
        'R2': r2_score(test['target_error_24h'], same_hour_pred),
        'MAE': mean_absolute_error(test['target_error_24h'], same_hour_pred),
        'MAE_Reduction': (1 - mean_absolute_error(test['target_error_24h'], same_hour_pred) / baseline_mae) * 100
    })

    # Ridge with basic features
    X_train_basic = train[basic_features]
    X_test_basic = test[basic_features]
    y_train = train['target_error_24h']
    y_test = test['target_error_24h']

    ridge_basic = Ridge(alpha=10.0)
    ridge_basic.fit(X_train_basic, y_train)
    pred_ridge_basic = ridge_basic.predict(X_test_basic)
    results.append({
        'Model': 'Ridge (basic)',
        'Features': str(len(basic_features)),
        'R2': r2_score(y_test, pred_ridge_basic),
        'MAE': mean_absolute_error(y_test, pred_ridge_basic),
        'MAE_Reduction': (1 - mean_absolute_error(y_test, pred_ridge_basic) / baseline_mae) * 100
    })

    # Ridge with extended features
    X_train_ext = train[extended_features]
    X_test_ext = test[extended_features]

    ridge_ext = Ridge(alpha=10.0)
    ridge_ext.fit(X_train_ext, y_train)
    pred_ridge_ext = ridge_ext.predict(X_test_ext)
    results.append({
        'Model': 'Ridge (extended)',
        'Features': str(len(extended_features)),
        'R2': r2_score(y_test, pred_ridge_ext),
        'MAE': mean_absolute_error(y_test, pred_ridge_ext),
        'MAE_Reduction': (1 - mean_absolute_error(y_test, pred_ridge_ext) / baseline_mae) * 100
    })

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    gb.fit(X_train_ext, y_train)
    pred_gb = gb.predict(X_test_ext)
    results.append({
        'Model': 'Gradient Boosting',
        'Features': str(len(extended_features)),
        'R2': r2_score(y_test, pred_gb),
        'MAE': mean_absolute_error(y_test, pred_gb),
        'MAE_Reduction': (1 - mean_absolute_error(y_test, pred_gb) / baseline_mae) * 100
    })

    results_df = pd.DataFrame(results)
    print("\n--- 24h Horizon Model Comparison ---")
    print(results_df.to_string(index=False))

    # Feature importance
    importance = pd.DataFrame({
        'feature': extended_features,
        'importance': gb.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n--- Gradient Boosting Feature Importance ---")
    print(importance.head(15).to_string(index=False))

    # Forecast correction analysis
    print("\n" + "="*70)
    print("FORECAST CORRECTION AT 24H HORIZON")
    print("="*70)

    # Original forecast error
    original_mae = (test['actual_load_mw'] - test['forecast_load_mw']).abs().mean()

    # Corrected forecast
    corrected_forecast = test['forecast_load_mw'] + pred_gb
    corrected_mae = (test['actual_load_mw'] - corrected_forecast).abs().mean()

    print(f"\nOriginal Baseline Forecast MAE:  {original_mae:.1f} MW")
    print(f"Corrected Forecast MAE (24h):    {corrected_mae:.1f} MW")
    print(f"Improvement:                     {(1 - corrected_mae/original_mae)*100:.1f}%")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Model comparison
    x = range(len(results_df))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_df)))
    axes[0, 0].barh(x, results_df['MAE'], color=colors, edgecolor='black', alpha=0.8)
    axes[0, 0].set_yticks(x)
    axes[0, 0].set_yticklabels(results_df['Model'])
    axes[0, 0].set_xlabel('MAE (MW)')
    axes[0, 0].set_title('24h Horizon Error Prediction - Model Comparison')
    axes[0, 0].axvline(x=baseline_mae, color='red', linestyle='--', label='Naive baseline')

    # Feature importance
    top_features = importance.head(12)
    axes[0, 1].barh(range(len(top_features)), top_features['importance'], color='green', edgecolor='black', alpha=0.7)
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features['feature'].str.replace('error_', 'err_').str.replace('load_', 'ld_'), fontsize=9)
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].set_title('Feature Importance (24h Horizon)')

    # Actual vs predicted error
    sample_idx = np.random.choice(len(y_test), min(2000, len(y_test)), replace=False)
    axes[1, 0].scatter(pred_gb[sample_idx], y_test.values[sample_idx], alpha=0.3, s=10)
    axes[1, 0].plot([-200, 200], [-200, 200], 'r--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted Error (MW)')
    axes[1, 0].set_ylabel('Actual Error (MW)')
    axes[1, 0].set_title(f'24h Error Prediction (R^2={r2_score(y_test, pred_gb):.3f})')
    axes[1, 0].set_xlim(-200, 200)
    axes[1, 0].set_ylim(-200, 200)

    # Forecast correction comparison
    axes[1, 1].bar([0, 1], [original_mae, corrected_mae], color=['red', 'green'], edgecolor='black', alpha=0.7)
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(['Original Forecast', 'Corrected (24h model)'])
    axes[1, 1].set_ylabel('MAE (MW)')
    axes[1, 1].set_title(f'24h Horizon Forecast Correction\n({(1-corrected_mae/original_mae)*100:.1f}% improvement)')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '20_24h_horizon_models.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Saved: 20_24h_horizon_models.png")

    return results_df, importance, gb


def analyze_3min_value(df: pd.DataFrame, df_3min: pd.DataFrame):
    """Analyze what value 3-min data adds over hourly data."""
    print("\n" + "="*70)
    print("3-MIN DATA VALUE ANALYSIS")
    print("="*70)

    # Compare hourly aggregated actual vs DAMAS hourly actual
    df_valid = df.dropna(subset=['actual_load_mw', 'load_mean'])

    # Correlation between 3-min mean and hourly actual
    corr = df_valid['actual_load_mw'].corr(df_valid['load_mean'])
    diff = (df_valid['actual_load_mw'] - df_valid['load_mean']).abs()

    print(f"\n3-min aggregated vs Hourly actual:")
    print(f"  Correlation: {corr:.4f}")
    print(f"  Mean absolute difference: {diff.mean():.1f} MW")
    print(f"  Max difference: {diff.max():.1f} MW")

    # Value of 3-min volatility features
    print(f"\n3-min volatility features (correlation with 24h error):")
    for col in ['load_std_lag24', 'load_cv_lag24', 'load_range_lag24', 'load_trend_lag24']:
        if col in df.columns:
            corr = df['target_error_24h'].corr(df[col])
            print(f"  {col}: {corr:.3f}")

    # Within-hour patterns from 3-min data
    df_3min_copy = df_3min.copy()
    df_3min_copy['hour'] = df_3min_copy.index.hour + 1
    df_3min_copy['minute'] = df_3min_copy.index.minute

    # Average load by minute within hour
    minute_pattern = df_3min_copy.groupby('minute')['load_mw'].mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Within-hour pattern
    axes[0, 0].plot(minute_pattern.index, minute_pattern.values, 'o-', linewidth=2)
    axes[0, 0].set_xlabel('Minute of Hour')
    axes[0, 0].set_ylabel('Average Load (MW)')
    axes[0, 0].set_title('Within-Hour Load Pattern (3-min data)')
    axes[0, 0].grid(True, alpha=0.3)

    # Volatility by hour
    hourly_vol = df.groupby('hour')['load_std'].mean()
    axes[0, 1].bar(hourly_vol.index, hourly_vol.values, color='orange', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Average Within-Hour Std (MW)')
    axes[0, 1].set_title('Load Volatility by Hour (from 3-min data)')

    # Correlation: volatility vs error
    df_corr = df.dropna(subset=['load_std_lag24', 'target_error_24h'])
    axes[1, 0].scatter(df_corr['load_std_lag24'], df_corr['target_error_24h'].abs(), alpha=0.1, s=5)
    axes[1, 0].set_xlabel('Load Std 24h Ago (MW)')
    axes[1, 0].set_ylabel('Absolute Error 24h Ahead (MW)')
    axes[1, 0].set_title('Volatility vs Error Magnitude')

    # Distribution of within-hour trends
    axes[1, 1].hist(df['load_trend'].dropna(), bins=50, color='purple', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Within-Hour Trend (Last - First, MW)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Within-Hour Load Trends')
    axes[1, 1].axvline(x=0, color='red', linestyle='--')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '21_3min_value_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Saved: 21_3min_value_analysis.png")


def main():
    print("="*70)
    print("24-HOUR HORIZON FORECAST ANALYSIS")
    print("With 3-Minute Data Features")
    print("="*70)

    # Load data
    print("\nLoading data...")
    df_hourly = load_hourly_data()
    df_3min = load_3min_data()

    print(f"Hourly data: {len(df_hourly)} records")
    print(f"3-min data: {len(df_3min)} records")

    # Create 3-min features
    df_3min_features = create_3min_hourly_features(df_3min)

    # Merge data
    df = merge_data(df_hourly, df_3min_features)

    # Create 24h horizon features
    df = create_24h_horizon_features(df)

    # Analyze correlations
    correlations = analyze_24h_error_correlation(df)

    # Train models
    results, importance, model = train_24h_models(df)

    # Analyze 3-min value
    analyze_3min_value(df, df_3min)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - 24H HORIZON PREDICTION")
    print("="*70)
    print("""
REALISTIC 24H AHEAD PREDICTION:

1. AVAILABLE FEATURES AT PREDICTION TIME T:
   - Errors from T-1, T-24, T-25, T-48, T-168 (all known)
   - 3-min volatility features (std, trend, range) from past hours
   - Calendar features for target hour (hour, dow, month, weekend)
   - Baseline forecast for T+24 (day-ahead market)

2. KEY FINDINGS:
   - error_lag24 (same hour yesterday) is most predictive
   - Same-hour weekly pattern (lag-168) also helps
   - 3-min volatility features add modest value
   - Calendar features (hour, dow) are important

3. ACHIEVABLE IMPROVEMENT:
   - Without weather/price: ~10-15% MAE reduction
   - Error prediction R^2: ~0.05-0.10 (limited)
   - This is MUCH harder than lag-1 prediction!

4. RECOMMENDATIONS:
   - Use baseline forecast as primary feature
   - Add error correction model with lag-24, lag-168
   - Include 3-min volatility for extra signal
   - Weather and prices will likely help significantly!
""")


if __name__ == '__main__':
    main()
