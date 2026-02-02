"""
Quarterly Separate Models - Progressive Feature Complexity
==========================================================

4 sets of models (Q0, Q1, Q2, Q3) x 5 horizons = 20 models

Feature progression:
- Q0: Basic features only (error lags, hour, dow, rolling stats)
- Q1: + First extrapolation, estimated H+1 error
- Q2: + Trend estimations, prediction changes, momentum, our error vs DAMAS
- Q3: + Everything - full trend, std, load volatility, all intra-hour features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent

print("=" * 70)
print("QUARTERLY SEPARATE MODELS")
print("Progressive feature complexity per quarter")
print("=" * 70)


def load_data():
    """Load hourly and 3-minute data."""
    print("\n[*] Loading data...")

    df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek

    df_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'load_3min.csv')
    df_3min['datetime'] = pd.to_datetime(df_3min['datetime'])
    df_3min = df_3min.sort_values('datetime').reset_index(drop=True)
    df_3min['hour_start'] = df_3min['datetime'].dt.floor('h')

    print(f"    Hourly: {len(df):,} records")
    print(f"    3-min:  {len(df_3min):,} records")

    return df, df_3min


def create_base_features(df):
    """Create base hourly features."""
    df = df.copy()

    # Error lags
    for lag in range(0, 49):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    # Rolling statistics
    df['error_roll_mean_24h'] = df['error'].shift(1).rolling(24).mean()
    df['error_roll_std_24h'] = df['error'].shift(1).rolling(24).std()
    df['error_roll_mean_6h'] = df['error'].shift(1).rolling(6).mean()
    df['error_roll_mean_12h'] = df['error'].shift(1).rolling(12).mean()

    # Same hour yesterday/last week
    df['error_same_hour_yesterday'] = df['error'].shift(24)
    df['error_same_hour_2d_ago'] = df['error'].shift(48)
    df['error_same_hour_lastweek'] = df['error'].shift(168)

    # === NEW: Error momentum features ===
    df['error_diff_1h'] = df['error_lag0'] - df['error_lag1']
    df['error_diff_2h'] = df['error_lag0'] - df['error_lag2']
    df['error_diff_24h'] = df['error_lag0'] - df['error_lag24']

    # === NEW: Forecast context features ===
    df['forecast_load'] = df['forecast_load_mw']
    df['forecast_diff_1h'] = df['forecast_load_mw'] - df['forecast_load_mw'].shift(1)
    df['forecast_diff_24h'] = df['forecast_load_mw'] - df['forecast_load_mw'].shift(24)

    # === NEW: Error regime features ===
    df['error_lag0_abs'] = df['error_lag0'].abs()
    df['error_lag0_sign'] = np.sign(df['error_lag0'])
    df['error_lag1_sign'] = np.sign(df['error_lag1'])
    df['error_sign_same'] = (df['error_lag0_sign'] == df['error_lag1_sign']).astype(int)

    # === NEW: Hour interaction features ===
    df['hour_x_error_lag0'] = df['hour'] * df['error_lag0'] / 100  # scaled
    df['hour_x_error_sign'] = df['hour'] * df['error_lag0_sign']

    # H+1 forecast and actual (for computing extrapolation)
    df['h1_forecast'] = df['forecast_load_mw'].shift(-1)
    df['h1_actual_load'] = df['actual_load_mw'].shift(-1)
    df['h1_actual_error'] = df['error'].shift(-1)

    # DAMAS error (forecast - actual, but we use actual - forecast = -DAMAS_error)
    df['damas_error_lag0'] = df['error']  # This is our definition of error
    df['damas_error_lag1'] = df['error'].shift(1)

    # Targets
    for h in range(1, 6):
        df[f'target_h{h}'] = df['error'].shift(-h)

    return df


def get_3min_features(df_3min, hour_start, minutes_elapsed):
    """Extract features from 3-min load data."""
    hour_data = df_3min[
        (df_3min['hour_start'] == hour_start) &
        (df_3min['datetime'] < hour_start + pd.Timedelta(minutes=minutes_elapsed))
    ]

    if len(hour_data) == 0:
        return {
            'load_mean': np.nan,
            'load_std': np.nan,
            'load_min': np.nan,
            'load_max': np.nan,
            'load_range': np.nan,
            'load_trend': np.nan,
            'load_first': np.nan,
            'load_last': np.nan,
            'load_momentum': np.nan,
            'n_observations': 0
        }

    load_values = hour_data['load_mw'].values

    # Basic stats
    load_mean = np.mean(load_values)
    load_std = np.std(load_values) if len(load_values) > 1 else 0.0
    load_min = np.min(load_values)
    load_max = np.max(load_values)
    load_range = load_max - load_min

    # Trend (slope)
    if len(load_values) >= 2:
        x = np.arange(len(load_values))
        load_trend = np.polyfit(x, load_values, 1)[0]  # MW per 3-min interval
    else:
        load_trend = 0.0

    # First and last values
    load_first = load_values[0]
    load_last = load_values[-1]

    # Momentum (acceleration of trend)
    if len(load_values) >= 3:
        # Compare first half trend to second half trend
        mid = len(load_values) // 2
        first_half = load_values[:mid]
        second_half = load_values[mid:]
        trend1 = (first_half[-1] - first_half[0]) / max(len(first_half) - 1, 1)
        trend2 = (second_half[-1] - second_half[0]) / max(len(second_half) - 1, 1)
        load_momentum = trend2 - trend1
    else:
        load_momentum = 0.0

    return {
        'load_mean': load_mean,
        'load_std': load_std,
        'load_min': load_min,
        'load_max': load_max,
        'load_range': load_range,
        'load_trend': load_trend,
        'load_first': load_first,
        'load_last': load_last,
        'load_momentum': load_momentum,
        'n_observations': len(load_values)
    }


def compute_seasonal_bias(df, df_3min, train_years=[2024]):
    """Compute hour-specific extrapolation bias."""
    print("\n[*] Computing seasonal extrapolation bias...")

    bias = {15: {}, 30: {}, 45: {}}

    for minutes in [15, 30, 45]:
        for hour in range(24):
            errors = []
            train_df = df[(df['year'].isin(train_years)) & (df['hour'] == hour)]

            for _, row in train_df.iterrows():
                h1_hour = row['datetime'] + pd.Timedelta(hours=1)
                h1_actual = row['h1_actual_load']

                if pd.isna(h1_actual):
                    continue

                partial_data = df_3min[
                    (df_3min['hour_start'] == h1_hour) &
                    (df_3min['datetime'] < h1_hour + pd.Timedelta(minutes=minutes))
                ]

                if len(partial_data) > 0:
                    extrap_load = partial_data['load_mw'].mean()
                    errors.append(h1_actual - extrap_load)

            bias[minutes][hour] = np.mean(errors) if errors else 0.0

    return bias


def build_quarterly_datasets(df, df_3min, seasonal_bias):
    """Build separate datasets for each quarter with appropriate features."""
    print("\n[*] Building quarterly datasets with progressive features...")

    datasets = {0: [], 1: [], 2: [], 3: []}

    for idx, row in df.iterrows():
        hour_start = row['datetime']
        h1_hour = hour_start + pd.Timedelta(hours=1)
        h1_forecast = row['h1_forecast']
        prediction_hour = row['hour']
        h1_hod = (prediction_hour + 1) % 24

        if pd.isna(h1_forecast):
            continue

        # Get 3-min features for each quarter
        feat_15 = get_3min_features(df_3min, h1_hour, 15)
        feat_30 = get_3min_features(df_3min, h1_hour, 30)
        feat_45 = get_3min_features(df_3min, h1_hour, 45)

        # Compute extrapolated errors with bias correction
        if not pd.isna(feat_15['load_mean']):
            extrap_load_q1 = feat_15['load_mean'] + seasonal_bias[15].get(h1_hod, 0)
            extrap_error_q1 = extrap_load_q1 - h1_forecast
        else:
            extrap_error_q1 = np.nan

        if not pd.isna(feat_30['load_mean']):
            extrap_load_q2 = feat_30['load_mean'] + seasonal_bias[30].get(h1_hod, 0)
            extrap_error_q2 = extrap_load_q2 - h1_forecast
        else:
            extrap_error_q2 = np.nan

        if not pd.isna(feat_45['load_mean']):
            extrap_load_q3 = feat_45['load_mean'] + seasonal_bias[45].get(h1_hod, 0)
            extrap_error_q3 = extrap_load_q3 - h1_forecast
        else:
            extrap_error_q3 = np.nan

        # =====================================================================
        # Q0: ENHANCED BASIC FEATURES (with momentum, forecast, error regime)
        # =====================================================================
        rec_q0 = {
            'datetime': hour_start,
            'year': row['year'],
            'hour': row['hour'],
            'dow': row['dow'],
            # Core error lags
            'error_lag0': row['error_lag0'],
            'error_lag1': row['error_lag1'],
            'error_lag2': row['error_lag2'],
            'error_lag3': row['error_lag3'],
            'error_lag24': row['error_lag24'],
            'error_lag48': row['error_lag48'],
            # Rolling stats
            'error_roll_mean_24h': row['error_roll_mean_24h'],
            'error_roll_std_24h': row['error_roll_std_24h'],
            'error_roll_mean_6h': row['error_roll_mean_6h'],
            'error_roll_mean_12h': row['error_roll_mean_12h'],
            # Same hour patterns
            'error_same_hour_yesterday': row['error_same_hour_yesterday'],
            'error_same_hour_2d_ago': row['error_same_hour_2d_ago'],
            # NEW: Error momentum
            'error_diff_1h': row['error_diff_1h'],
            'error_diff_2h': row['error_diff_2h'],
            'error_diff_24h': row['error_diff_24h'],
            # NEW: Forecast context
            'forecast_load': row['forecast_load'],
            'forecast_diff_1h': row['forecast_diff_1h'],
            'forecast_diff_24h': row['forecast_diff_24h'],
            # NEW: Error regime
            'error_lag0_abs': row['error_lag0_abs'],
            'error_lag0_sign': row['error_lag0_sign'],
            'error_sign_same': row['error_sign_same'],
            # NEW: Hour interactions
            'hour_x_error_lag0': row['hour_x_error_lag0'],
            'hour_x_error_sign': row['hour_x_error_sign'],
        }
        # Add targets
        for h in range(1, 6):
            rec_q0[f'target_h{h}'] = row[f'target_h{h}']

        datasets[0].append(rec_q0)

        # =====================================================================
        # Q1: + First extrapolation, estimated H+1 error
        # =====================================================================
        rec_q1 = rec_q0.copy()
        rec_q1.update({
            # First extrapolation
            'extrap_error_q1': extrap_error_q1 if not pd.isna(extrap_error_q1) else 0.0,
            'load_mean_q1': feat_15['load_mean'] if not pd.isna(feat_15['load_mean']) else 0.0,
            'load_trend_q1': feat_15['load_trend'] if not pd.isna(feat_15['load_trend']) else 0.0,
            'load_std_q1': feat_15['load_std'] if not pd.isna(feat_15['load_std']) else 0.0,
            # Estimated H+1 error (our best estimate so far)
            'est_h1_error': extrap_error_q1 if not pd.isna(extrap_error_q1) else (row['error_lag0'] if not pd.isna(row['error_lag0']) else 0.0),
            # How different is our estimate from just using error_lag0?
            'est_vs_lag0': (extrap_error_q1 - row['error_lag0']) if (not pd.isna(extrap_error_q1) and not pd.isna(row['error_lag0'])) else 0.0,
        })
        datasets[1].append(rec_q1)

        # =====================================================================
        # Q2: + Trend estimations, prediction changes, momentum
        # =====================================================================
        rec_q2 = rec_q1.copy()

        # Q2 extrapolation
        rec_q2['extrap_error_q2'] = extrap_error_q2 if not pd.isna(extrap_error_q2) else 0.0
        rec_q2['load_mean_q2'] = feat_30['load_mean'] if not pd.isna(feat_30['load_mean']) else 0.0
        rec_q2['load_trend_q2'] = feat_30['load_trend'] if not pd.isna(feat_30['load_trend']) else 0.0
        rec_q2['load_std_q2'] = feat_30['load_std'] if not pd.isna(feat_30['load_std']) else 0.0

        # How did our estimate CHANGE from Q1 to Q2?
        if not pd.isna(extrap_error_q1) and not pd.isna(extrap_error_q2):
            rec_q2['delta_est_q1_to_q2'] = extrap_error_q2 - extrap_error_q1
        else:
            rec_q2['delta_est_q1_to_q2'] = 0.0

        # Direction/momentum of estimates
        rec_q2['est_direction'] = np.sign(rec_q2['delta_est_q1_to_q2'])

        # Our error estimate vs DAMAS baseline (DAMAS predicts error = 0)
        rec_q2['est_vs_damas'] = extrap_error_q2 if not pd.isna(extrap_error_q2) else 0.0

        # Load momentum
        rec_q2['load_momentum_q2'] = feat_30['load_momentum'] if not pd.isna(feat_30['load_momentum']) else 0.0

        # Update best estimate
        rec_q2['est_h1_error'] = extrap_error_q2 if not pd.isna(extrap_error_q2) else rec_q1['est_h1_error']

        datasets[2].append(rec_q2)

        # =====================================================================
        # Q3: EVERYTHING - full trend, std, volatility, all intra-hour features
        # =====================================================================
        rec_q3 = rec_q2.copy()

        # Q3 extrapolation
        rec_q3['extrap_error_q3'] = extrap_error_q3 if not pd.isna(extrap_error_q3) else 0.0
        rec_q3['load_mean_q3'] = feat_45['load_mean'] if not pd.isna(feat_45['load_mean']) else 0.0
        rec_q3['load_trend_q3'] = feat_45['load_trend'] if not pd.isna(feat_45['load_trend']) else 0.0
        rec_q3['load_std_q3'] = feat_45['load_std'] if not pd.isna(feat_45['load_std']) else 0.0
        rec_q3['load_min_q3'] = feat_45['load_min'] if not pd.isna(feat_45['load_min']) else 0.0
        rec_q3['load_max_q3'] = feat_45['load_max'] if not pd.isna(feat_45['load_max']) else 0.0
        rec_q3['load_range_q3'] = feat_45['load_range'] if not pd.isna(feat_45['load_range']) else 0.0
        rec_q3['load_momentum_q3'] = feat_45['load_momentum'] if not pd.isna(feat_45['load_momentum']) else 0.0

        # Full trend analysis - how did estimates evolve?
        if not pd.isna(extrap_error_q2) and not pd.isna(extrap_error_q3):
            rec_q3['delta_est_q2_to_q3'] = extrap_error_q3 - extrap_error_q2
        else:
            rec_q3['delta_est_q2_to_q3'] = 0.0

        # Acceleration of estimate changes
        rec_q3['est_acceleration'] = rec_q3['delta_est_q2_to_q3'] - rec_q2['delta_est_q1_to_q2']

        # Std of our estimates across quarters (stability of prediction)
        estimates = [e for e in [extrap_error_q1, extrap_error_q2, extrap_error_q3] if not pd.isna(e)]
        rec_q3['est_std'] = np.std(estimates) if len(estimates) > 1 else 0.0

        # Trend of our estimates
        if len(estimates) >= 2:
            rec_q3['est_trend'] = np.polyfit(range(len(estimates)), estimates, 1)[0]
        else:
            rec_q3['est_trend'] = 0.0

        # How much did load change over the 45 minutes?
        rec_q3['load_change_45min'] = (feat_45['load_last'] - feat_45['load_first']) if not pd.isna(feat_45['load_last']) else 0.0

        # Update best estimate
        rec_q3['est_h1_error'] = extrap_error_q3 if not pd.isna(extrap_error_q3) else rec_q2['est_h1_error']

        datasets[3].append(rec_q3)

    # Convert to DataFrames
    for q in range(4):
        datasets[q] = pd.DataFrame(datasets[q])
        print(f"    Q{q}: {len(datasets[q]):,} records, {len(datasets[q].columns)} features")

    return datasets


def get_feature_sets():
    """Define feature sets for each quarter."""

    # Q0: Enhanced basic features with momentum, forecast context, error regime
    features_q0 = [
        # Core error lags
        'error_lag0', 'error_lag1', 'error_lag2', 'error_lag3', 'error_lag24', 'error_lag48',
        # Rolling stats
        'error_roll_mean_24h', 'error_roll_std_24h', 'error_roll_mean_6h', 'error_roll_mean_12h',
        # Same hour patterns
        'error_same_hour_yesterday', 'error_same_hour_2d_ago',
        # NEW: Error momentum
        'error_diff_1h', 'error_diff_2h', 'error_diff_24h',
        # NEW: Forecast context
        'forecast_load', 'forecast_diff_1h', 'forecast_diff_24h',
        # NEW: Error regime
        'error_lag0_abs', 'error_lag0_sign', 'error_sign_same',
        # NEW: Hour interactions
        'hour_x_error_lag0', 'hour_x_error_sign',
        # Time features
        'hour', 'dow'
    ]

    # Q1: + First extrapolation
    features_q1 = features_q0 + [
        'extrap_error_q1', 'load_mean_q1', 'load_trend_q1', 'load_std_q1',
        'est_h1_error', 'est_vs_lag0'
    ]

    # Q2: + Trend estimations, changes, momentum
    features_q2 = features_q1 + [
        'extrap_error_q2', 'load_mean_q2', 'load_trend_q2', 'load_std_q2',
        'delta_est_q1_to_q2', 'est_direction', 'est_vs_damas', 'load_momentum_q2'
    ]

    # Q3: Everything
    features_q3 = features_q2 + [
        'extrap_error_q3', 'load_mean_q3', 'load_trend_q3', 'load_std_q3',
        'load_min_q3', 'load_max_q3', 'load_range_q3', 'load_momentum_q3',
        'delta_est_q2_to_q3', 'est_acceleration', 'est_std', 'est_trend',
        'load_change_45min'
    ]

    return {0: features_q0, 1: features_q1, 2: features_q2, 3: features_q3}


def train_models(datasets, feature_sets):
    """Train separate models for each quarter and horizon."""
    print("\n[*] Training separate models for each quarter...")

    models = {}  # models[(quarter, horizon)] = model

    for q in range(4):
        df_q = datasets[q]
        features = feature_sets[q]

        train = df_q[df_q['year'] == 2024].dropna(subset=features)

        print(f"\n  Q{q} ({len(features)} features):")

        for h in range(1, 6):
            target = f'target_h{h}'
            train_q = train.dropna(subset=[target])

            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                verbosity=-1,
                random_state=42
            )
            model.fit(train_q[features], train_q[target])
            models[(q, h)] = model

        print(f"    Trained 5 models (H+1 to H+5)")

    print(f"\n  Total: {len(models)} models trained")
    return models


def evaluate_models(datasets, feature_sets, models):
    """Evaluate all models."""
    print("\n" + "=" * 70)
    print("EVALUATION: SEPARATE MODELS PER QUARTER")
    print("=" * 70)

    results = []

    for q in range(4):
        df_q = datasets[q]
        features = feature_sets[q]

        test = df_q[df_q['year'] >= 2025].dropna(subset=features)

        for h in range(1, 6):
            target = f'target_h{h}'
            test_h = test.dropna(subset=[target])

            if len(test_h) == 0:
                continue

            model = models[(q, h)]
            pred = model.predict(test_h[features])
            actual = test_h[target].values

            mae = np.nanmean(np.abs(actual - pred))

            results.append({
                'quarter': q,
                'horizon': h,
                'mae': mae,
                'n_features': len(features),
                'n_samples': len(test_h)
            })

    return pd.DataFrame(results)


def print_results(results_df, datasets, feature_sets, models):
    """Print comprehensive results."""

    print("\n" + "=" * 70)
    print("RESULTS: SEPARATE MODELS PER QUARTER")
    print("=" * 70)

    # MAE table
    print("\n  MAE by Quarter and Horizon:")
    print("  " + "-" * 55)
    print("  Horizon |   Q0    |   Q1    |   Q2    |   Q3    |")
    print("  " + "-" * 55)

    for h in range(1, 6):
        row = f"  H+{h}     |"
        for q in range(4):
            r = results_df[(results_df['quarter'] == q) & (results_df['horizon'] == h)]
            if len(r) > 0:
                row += f"  {r['mae'].values[0]:5.1f}  |"
            else:
                row += "   N/A  |"
        print(row)

    print("  " + "-" * 55)

    # Feature counts
    print("\n  Features per quarter:")
    for q in range(4):
        print(f"    Q{q}: {len(feature_sets[q])} features")

    # Compare to baseline (Q0 for all quarters)
    print("\n" + "=" * 70)
    print("IMPROVEMENT OVER Q0 BASELINE")
    print("=" * 70)

    print("\n  Horizon | Q0 (base) |   Q1    |   Q2    |   Q3    |")
    print("  " + "-" * 55)

    for h in range(1, 6):
        q0_mae = results_df[(results_df['quarter'] == 0) & (results_df['horizon'] == h)]['mae'].values[0]
        row = f"  H+{h}     |  {q0_mae:5.1f}   |"

        for q in [1, 2, 3]:
            q_mae = results_df[(results_df['quarter'] == q) & (results_df['horizon'] == h)]['mae'].values[0]
            improvement = q0_mae - q_mae
            row += f" {q_mae:5.1f} ({improvement:+.1f}) |"
        print(row)

    # Feature importance for Q3 H+1
    print("\n" + "=" * 70)
    print("TOP FEATURES BY QUARTER (for H+1)")
    print("=" * 70)

    for q in range(4):
        model = models[(q, 1)]
        features = feature_sets[q]
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n  Q{q} Top 5 features:")
        for _, r in importance.head(5).iterrows():
            print(f"    {r['feature']}: {r['importance']}")


def compare_to_previous():
    """Print comparison to previous models."""
    print("\n" + "=" * 70)
    print("COMPARISON TO PREVIOUS SEPARATE MODELS (before new Q0 features)")
    print("=" * 70)

    print("""
  Previous Separate Models (basic Q0 features):

  Horizon | Q0    | Q1    | Q2    | Q3    |
  --------|-------|-------|-------|-------|
  H+1     | 29.6  | 18.9  | 14.3  |  9.1  |
  H+2     | 41.9  | 36.8  | 34.3  | 32.2  |
  H+3     | 50.8  | 47.0  | 44.9  | 43.3  |
  H+4     | 56.0  | 54.0  | 52.6  | 51.4  |
  H+5     | 59.9  | 58.5  | 57.6  | 57.0  |

  New features added to Q0:
  - Error momentum: error_diff_1h, error_diff_2h, error_diff_24h
  - Forecast context: forecast_load, forecast_diff_1h, forecast_diff_24h
  - Error regime: error_lag0_abs, error_lag0_sign, error_sign_same
  - Hour interactions: hour_x_error_lag0, hour_x_error_sign
  - Additional lags: error_lag48, error_roll_mean_12h, error_same_hour_2d_ago
""")


def main():
    df, df_3min = load_data()

    # Create base features
    df = create_base_features(df)

    # Compute seasonal bias
    seasonal_bias = compute_seasonal_bias(df, df_3min, train_years=[2024])

    # Build separate datasets for each quarter
    datasets = build_quarterly_datasets(df, df_3min, seasonal_bias)

    # Get feature sets
    feature_sets = get_feature_sets()

    # Train models
    models = train_models(datasets, feature_sets)

    # Evaluate
    results = evaluate_models(datasets, feature_sets, models)

    # Print results
    print_results(results, datasets, feature_sets, models)

    # Compare to previous
    compare_to_previous()

    return results, models, datasets, feature_sets


if __name__ == "__main__":
    results, models, datasets, feature_sets = main()
