"""
H+2 Residual Analysis: What signal is left on the table?
========================================================

Analyze H+2 prediction residuals at Q0 and Q1 to find unexploited patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent

print("=" * 70)
print("H+2 RESIDUAL ANALYSIS: Finding leftover signal")
print("=" * 70)


def load_and_prepare_data():
    """Load data and create features."""
    print("\n[*] Loading data...")

    df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)

    # Error lags
    for lag in range(0, 49):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    # Rolling statistics
    df['error_roll_mean_24h'] = df['error'].shift(1).rolling(24).mean()
    df['error_roll_std_24h'] = df['error'].shift(1).rolling(24).std()
    df['error_roll_mean_6h'] = df['error'].shift(1).rolling(6).mean()
    df['error_roll_mean_12h'] = df['error'].shift(1).rolling(12).mean()

    # Same hour patterns
    df['error_same_hour_yesterday'] = df['error'].shift(24)
    df['error_same_hour_2d_ago'] = df['error'].shift(48)
    df['error_same_hour_lastweek'] = df['error'].shift(168)

    # Error differences (momentum)
    df['error_diff_1h'] = df['error_lag0'] - df['error_lag1']
    df['error_diff_24h'] = df['error_lag0'] - df['error_lag24']

    # Forecast features
    df['forecast_load'] = df['forecast_load_mw']
    df['forecast_diff_1h'] = df['forecast_load_mw'] - df['forecast_load_mw'].shift(1)
    df['forecast_diff_24h'] = df['forecast_load_mw'] - df['forecast_load_mw'].shift(24)

    # Actual load features (lagged - we know these)
    df['actual_load_lag1'] = df['actual_load_mw'].shift(1)
    df['actual_load_diff_1h'] = df['actual_load_mw'].shift(1) - df['actual_load_mw'].shift(2)

    # Target
    df['target_h2'] = df['error'].shift(-2)

    return df


def train_q0_model(df):
    """Train Q0 H+2 model and get residuals."""
    print("\n[*] Training Q0 H+2 model...")

    features_q0 = [
        'error_lag0', 'error_lag1', 'error_lag2', 'error_lag3', 'error_lag24',
        'error_roll_mean_24h', 'error_roll_std_24h', 'error_roll_mean_6h',
        'error_same_hour_yesterday', 'hour', 'dow'
    ]

    train = df[(df['year'] == 2024)].dropna(subset=features_q0 + ['target_h2'])
    test = df[(df['year'] >= 2025)].dropna(subset=features_q0 + ['target_h2'])

    model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        verbosity=-1, random_state=42
    )
    model.fit(train[features_q0], train['target_h2'])

    test = test.copy()
    test['pred_h2'] = model.predict(test[features_q0])
    test['residual'] = test['target_h2'] - test['pred_h2']

    mae = np.abs(test['residual']).mean()
    print(f"    Q0 H+2 MAE: {mae:.1f} MW")

    return test, model, features_q0


def analyze_residual_patterns(test_df):
    """Analyze patterns in residuals."""
    print("\n" + "=" * 70)
    print("RESIDUAL PATTERN ANALYSIS")
    print("=" * 70)

    residuals = test_df['residual'].values

    # Basic stats
    print(f"\n  Basic statistics:")
    print(f"    Mean:     {np.mean(residuals):+.2f} MW (should be ~0)")
    print(f"    Std:      {np.std(residuals):.2f} MW")
    print(f"    Skewness: {stats.skew(residuals):.2f}")
    print(f"    Kurtosis: {stats.kurtosis(residuals):.2f}")

    # Autocorrelation of residuals
    print(f"\n  Residual autocorrelation (if high, we're missing patterns):")
    for lag in [1, 2, 3, 6, 12, 24]:
        autocorr = test_df['residual'].autocorr(lag=lag)
        flag = " [!]" if abs(autocorr) > 0.1 else ""
        print(f"    Lag {lag:2d}: {autocorr:+.3f}{flag}")

    # By hour of day
    print(f"\n  Residual by hour of day:")
    hourly_mae = test_df.groupby('hour')['residual'].apply(lambda x: np.abs(x).mean())
    hourly_bias = test_df.groupby('hour')['residual'].mean()

    worst_hours_mae = hourly_mae.nlargest(5)
    worst_hours_bias = hourly_bias.abs().nlargest(5)

    print(f"    Worst MAE hours: {dict(worst_hours_mae.round(1))}")
    print(f"    Worst bias hours: {dict(hourly_bias[worst_hours_bias.index].round(1))}")

    # By day of week
    print(f"\n  Residual by day of week:")
    dow_mae = test_df.groupby('dow')['residual'].apply(lambda x: np.abs(x).mean())
    dow_bias = test_df.groupby('dow')['residual'].mean()
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for d in range(7):
        print(f"    {dow_names[d]}: MAE={dow_mae[d]:.1f}, Bias={dow_bias[d]:+.1f}")

    return hourly_mae, hourly_bias


def find_correlated_features(test_df):
    """Find features correlated with residuals."""
    print("\n" + "=" * 70)
    print("FEATURE CORRELATION WITH RESIDUALS")
    print("=" * 70)

    # Candidate features we might not be using effectively
    candidate_features = [
        # More error lags
        'error_lag4', 'error_lag5', 'error_lag6', 'error_lag12', 'error_lag48',
        # Same hour patterns
        'error_same_hour_yesterday', 'error_same_hour_2d_ago', 'error_same_hour_lastweek',
        # Rolling stats
        'error_roll_mean_6h', 'error_roll_mean_12h', 'error_roll_std_24h',
        # Error momentum
        'error_diff_1h', 'error_diff_24h',
        # Forecast patterns
        'forecast_load', 'forecast_diff_1h', 'forecast_diff_24h',
        # Actual load patterns
        'actual_load_lag1', 'actual_load_diff_1h',
        # Time features
        'hour', 'dow', 'is_weekend', 'month',
    ]

    correlations = []
    for feat in candidate_features:
        if feat in test_df.columns:
            corr = test_df['residual'].corr(test_df[feat])
            if not pd.isna(corr):
                correlations.append((feat, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n  Features correlated with residuals (potential signal):")
    print("  " + "-" * 50)
    for feat, corr in correlations[:15]:
        flag = " [!] SIGNAL" if abs(corr) > 0.05 else ""
        print(f"    {feat:30s}: {corr:+.4f}{flag}")

    return correlations


def analyze_error_regime(test_df):
    """Analyze if residuals depend on error regime."""
    print("\n" + "=" * 70)
    print("ERROR REGIME ANALYSIS")
    print("=" * 70)

    # Split by error_lag0 magnitude
    test_df = test_df.copy()
    test_df['error_lag0_abs'] = test_df['error_lag0'].abs()

    # Quartiles of error magnitude
    quartiles = test_df['error_lag0_abs'].quantile([0.25, 0.5, 0.75])

    print(f"\n  Residual MAE by error_lag0 magnitude:")
    print(f"    Small errors (|e| < {quartiles[0.25]:.0f} MW):")
    small = test_df[test_df['error_lag0_abs'] < quartiles[0.25]]
    print(f"      MAE: {np.abs(small['residual']).mean():.1f} MW, Bias: {small['residual'].mean():+.1f} MW")

    print(f"    Medium errors ({quartiles[0.25]:.0f} < |e| < {quartiles[0.75]:.0f} MW):")
    medium = test_df[(test_df['error_lag0_abs'] >= quartiles[0.25]) &
                     (test_df['error_lag0_abs'] < quartiles[0.75])]
    print(f"      MAE: {np.abs(medium['residual']).mean():.1f} MW, Bias: {medium['residual'].mean():+.1f} MW")

    print(f"    Large errors (|e| > {quartiles[0.75]:.0f} MW):")
    large = test_df[test_df['error_lag0_abs'] >= quartiles[0.75]]
    print(f"      MAE: {np.abs(large['residual']).mean():.1f} MW, Bias: {large['residual'].mean():+.1f} MW")

    # Split by error sign
    print(f"\n  Residual MAE by error_lag0 sign:")
    positive = test_df[test_df['error_lag0'] > 0]
    negative = test_df[test_df['error_lag0'] < 0]
    print(f"    Positive errors: MAE={np.abs(positive['residual']).mean():.1f}, Bias={positive['residual'].mean():+.1f}")
    print(f"    Negative errors: MAE={np.abs(negative['residual']).mean():.1f}, Bias={negative['residual'].mean():+.1f}")


def analyze_sequential_patterns(test_df):
    """Analyze sequential patterns in residuals."""
    print("\n" + "=" * 70)
    print("SEQUENTIAL PATTERN ANALYSIS")
    print("=" * 70)

    test_df = test_df.copy()

    # Consecutive same-sign errors
    test_df['error_sign'] = np.sign(test_df['error_lag0'])
    test_df['error_sign_lag1'] = test_df['error_sign'].shift(1)
    test_df['error_sign_lag2'] = test_df['error_sign'].shift(2)

    # Streak of same-sign errors
    test_df['same_sign_streak'] = (
        (test_df['error_sign'] == test_df['error_sign_lag1']).astype(int) +
        (test_df['error_sign'] == test_df['error_sign_lag2']).astype(int)
    )

    print(f"\n  Residual MAE by error sign streak:")
    for streak in [0, 1, 2]:
        subset = test_df[test_df['same_sign_streak'] == streak]
        if len(subset) > 100:
            print(f"    Streak {streak}: MAE={np.abs(subset['residual']).mean():.1f} MW (n={len(subset)})")

    # Error reversal patterns
    test_df['error_reversed'] = (test_df['error_sign'] != test_df['error_sign_lag1']).astype(int)

    print(f"\n  Residual MAE by error reversal:")
    continued = test_df[test_df['error_reversed'] == 0]
    reversed_err = test_df[test_df['error_reversed'] == 1]
    print(f"    Error continued: MAE={np.abs(continued['residual']).mean():.1f} MW")
    print(f"    Error reversed:  MAE={np.abs(reversed_err['residual']).mean():.1f} MW")


def test_additional_features(df, base_features):
    """Test if additional features improve the model."""
    print("\n" + "=" * 70)
    print("TESTING ADDITIONAL FEATURES")
    print("=" * 70)

    train = df[(df['year'] == 2024)].dropna()
    test = df[(df['year'] >= 2025)].dropna()

    # Baseline
    model_base = lgb.LGBMRegressor(n_estimators=100, max_depth=6, verbosity=-1, random_state=42)
    model_base.fit(train[base_features], train['target_h2'])
    pred_base = model_base.predict(test[base_features])
    mae_base = np.abs(test['target_h2'] - pred_base).mean()

    print(f"\n  Baseline Q0 features: {mae_base:.2f} MW")

    # Test additional feature groups
    feature_groups = {
        'more_lags': ['error_lag4', 'error_lag5', 'error_lag6', 'error_lag12', 'error_lag48'],
        'same_hour': ['error_same_hour_2d_ago', 'error_same_hour_lastweek'],
        'momentum': ['error_diff_1h', 'error_diff_24h'],
        'forecast': ['forecast_load', 'forecast_diff_1h', 'forecast_diff_24h'],
        'actual_load': ['actual_load_lag1', 'actual_load_diff_1h'],
        'rolling_12h': ['error_roll_mean_12h'],
    }

    print(f"\n  Testing additional feature groups:")
    print(f"  " + "-" * 50)

    improvements = []
    for group_name, features in feature_groups.items():
        test_features = base_features + [f for f in features if f in df.columns]
        train_clean = train.dropna(subset=test_features + ['target_h2'])
        test_clean = test.dropna(subset=test_features + ['target_h2'])

        if len(train_clean) < 1000:
            continue

        model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, verbosity=-1, random_state=42)
        model.fit(train_clean[test_features], train_clean['target_h2'])
        pred = model.predict(test_clean[test_features])
        mae = np.abs(test_clean['target_h2'] - pred).mean()

        improvement = mae_base - mae
        improvements.append((group_name, mae, improvement))
        flag = " [+]" if improvement > 0.1 else ""
        print(f"    + {group_name:15s}: {mae:.2f} MW ({improvement:+.2f}){flag}")

    # Best combination
    print(f"\n  Testing best combination:")
    best_features = base_features.copy()
    for group_name, mae, improvement in improvements:
        if improvement > 0.05:
            best_features.extend([f for f in feature_groups[group_name] if f in df.columns])

    best_features = list(set(best_features))
    train_clean = train.dropna(subset=best_features + ['target_h2'])
    test_clean = test.dropna(subset=best_features + ['target_h2'])

    model_best = lgb.LGBMRegressor(n_estimators=100, max_depth=6, verbosity=-1, random_state=42)
    model_best.fit(train_clean[best_features], train_clean['target_h2'])
    pred_best = model_best.predict(test_clean[best_features])
    mae_best = np.abs(test_clean['target_h2'] - pred_best).mean()

    print(f"    Combined best features ({len(best_features)}): {mae_best:.2f} MW ({mae_base - mae_best:+.2f})")

    return improvements, best_features


def analyze_hour_specific_models(df, base_features):
    """Test if hour-specific models help."""
    print("\n" + "=" * 70)
    print("HOUR-SPECIFIC MODEL ANALYSIS")
    print("=" * 70)

    train = df[(df['year'] == 2024)].dropna(subset=base_features + ['target_h2'])
    test = df[(df['year'] >= 2025)].dropna(subset=base_features + ['target_h2'])

    # Single model baseline
    model_single = lgb.LGBMRegressor(n_estimators=100, max_depth=6, verbosity=-1, random_state=42)
    model_single.fit(train[base_features], train['target_h2'])
    pred_single = model_single.predict(test[base_features])
    mae_single = np.abs(test['target_h2'] - pred_single).mean()

    # Hour-specific models
    predictions = []
    for hour in range(24):
        train_h = train[train['hour'] == hour]
        test_h = test[test['hour'] == hour]

        if len(train_h) < 100 or len(test_h) < 50:
            # Fall back to single model
            pred_h = model_single.predict(test_h[base_features])
        else:
            model_h = lgb.LGBMRegressor(n_estimators=50, max_depth=5, verbosity=-1, random_state=42)
            model_h.fit(train_h[base_features], train_h['target_h2'])
            pred_h = model_h.predict(test_h[base_features])

        for idx, p in zip(test_h.index, pred_h):
            predictions.append((idx, p))

    pred_hourly = pd.Series(dict(predictions))
    test_aligned = test.loc[pred_hourly.index]
    mae_hourly = np.abs(test_aligned['target_h2'] - pred_hourly).mean()

    print(f"\n  Single model:       {mae_single:.2f} MW")
    print(f"  Hour-specific (24): {mae_hourly:.2f} MW ({mae_single - mae_hourly:+.2f})")


def main():
    df = load_and_prepare_data()

    # Train Q0 model and get residuals
    test_df, model, base_features = train_q0_model(df)

    # Analyze residual patterns
    analyze_residual_patterns(test_df)

    # Find correlated features
    correlations = find_correlated_features(test_df)

    # Analyze error regimes
    analyze_error_regime(test_df)

    # Analyze sequential patterns
    analyze_sequential_patterns(test_df)

    # Test additional features
    improvements, best_features = test_additional_features(df, base_features)

    # Test hour-specific models
    analyze_hour_specific_models(df, base_features)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SIGNAL LEFT ON THE TABLE")
    print("=" * 70)

    print("""
  Key findings for H+2 at Q0/Q1:

  1. RESIDUAL AUTOCORRELATION: Check if lag-1 autocorr > 0.1
     -> If yes, model is missing temporal patterns

  2. HOUR-SPECIFIC PATTERNS: Some hours have higher MAE
     -> Consider hour-specific models or hour interaction features

  3. ERROR REGIME: Large errors harder to predict
     -> Consider separate models for high-volatility periods

  4. ADDITIONAL FEATURES: Which groups help?
     -> Add the ones with positive improvement

  5. SEQUENTIAL PATTERNS: Error streaks matter
     -> Add streak/reversal features
""")


if __name__ == "__main__":
    main()
