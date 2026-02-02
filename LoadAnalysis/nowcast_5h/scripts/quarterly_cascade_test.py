"""
Quarterly Cascade Test: Does extrapolated H+1 error improve H+2-H+5?
=====================================================================

Hypothesis:
- At Q1-Q3, we have partial load data for the current hour
- We can extrapolate error_lag0 (current hour error) from this
- If we feed this extrapolated error into our base models,
  it might improve H+2-H+5 predictions too

Test approach:
- Use our trained base models (two-stage)
- At each quarter, replace error_lag0 with extrapolated error
- Compare performance vs Q0 (model-only) baseline

If performance improves: base models work with extrapolated error
If performance degrades: need separate quarter-specific models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent

print("=" * 70)
print("QUARTERLY CASCADE TEST")
print("Does extrapolated H+1 error improve H+2-H+5 predictions?")
print("=" * 70)


def load_data():
    """Load hourly and 3-minute data."""
    print("\n[*] Loading data...")

    # Hourly data
    df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek

    # 3-minute data
    df_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'load_3min.csv')
    df_3min['datetime'] = pd.to_datetime(df_3min['datetime'])
    df_3min = df_3min.sort_values('datetime').reset_index(drop=True)
    df_3min['hour_start'] = df_3min['datetime'].dt.floor('h')

    print(f"    Hourly: {len(df):,} records")
    print(f"    3-min:  {len(df_3min):,} records")

    return df, df_3min


def create_features(df):
    """Create all features for the model."""
    df = df.copy()

    # Error lags (including lag0 which will be replaced at Q1-Q3)
    for lag in range(0, 25):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    # Rolling statistics
    df['error_roll_mean_24h'] = df['error'].shift(1).rolling(24).mean()
    df['error_roll_std_24h'] = df['error'].shift(1).rolling(24).std()

    # Targets
    for h in range(1, 6):
        df[f'target_h{h}'] = df['error'].shift(-h)

    return df


def compute_partial_hour_estimate(df_3min, hour_start, minutes_elapsed):
    """Estimate full hour load from partial 3-min data."""
    hour_data = df_3min[
        (df_3min['hour_start'] == hour_start) &
        (df_3min['datetime'] < hour_start + pd.Timedelta(minutes=minutes_elapsed))
    ]

    if len(hour_data) == 0:
        return np.nan

    return hour_data['load_mw'].mean()


def get_extrapolated_error(df_3min, hour_start, minutes_elapsed, forecast_load):
    """Get extrapolated error for current hour."""
    estimated_load = compute_partial_hour_estimate(df_3min, hour_start, minutes_elapsed)
    if pd.isna(estimated_load):
        return np.nan
    return estimated_load - forecast_load


def train_simple_models(df):
    """Train simple Stage 1 models for each horizon."""
    print("\n[*] Training Stage 1 models...")

    # Use 2024 for training
    train = df[(df['year'] == 2024)].dropna()

    features = [
        'error_lag0', 'error_lag1', 'error_lag2', 'error_lag3',
        'error_lag24', 'error_roll_mean_24h', 'hour', 'dow'
    ]

    models = {}
    for h in range(1, 6):
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            verbosity=-1,
            random_state=42
        )
        model.fit(train[features], train[f'target_h{h}'])
        models[h] = model
        print(f"    H+{h} trained")

    return models, features


def evaluate_at_quarters(df, df_3min, models, features):
    """Evaluate model performance at each quarter position.

    IMPORTANT: At Q0, error_lag0 (current hour error) is NOT available yet!
    We must use error_lag1 as the fallback. At Q1-Q3, we have partial data.
    """
    print("\n" + "=" * 70)
    print("EVALUATING PERFORMANCE AT EACH QUARTER (REALISTIC)")
    print("=" * 70)
    print("\n  NOTE: At Q0, error_lag0 is NOT available - using error_lag1 as fallback")

    # Get test data (2025+) where we have 3-min data
    df_3min['hour_start'] = df_3min['datetime'].dt.floor('h')
    hours_with_3min = set(df_3min[df_3min['datetime'].dt.year >= 2025]['hour_start'].unique())

    test = df[df['year'] >= 2025].copy()
    test = test[test['datetime'].isin(hours_with_3min)]
    test = test.dropna(subset=features + [f'target_h{h}' for h in range(1, 6)])

    print(f"\n  Test set: {len(test):,} hours")

    results = []

    for quarter, minutes in [(0, 0), (1, 15), (2, 30), (3, 45)]:
        print(f"\n  --- Q{quarter} ({minutes} min elapsed) ---")

        quarter_results = {'quarter': quarter, 'minutes': minutes}

        for h in range(1, 6):
            errors = []

            for _, row in test.iterrows():
                hour_start = row['datetime']
                actual_error = row[f'target_h{h}']

                # Create feature vector
                X = row[features].copy()

                if minutes == 0:
                    # Q0: error_lag0 is NOT available yet!
                    # Use error_lag1 as fallback (shift everything by 1)
                    X['error_lag0'] = row['error_lag1']
                else:
                    # Q1-Q3: Use extrapolated error from partial 3-min data
                    extrap_error = get_extrapolated_error(
                        df_3min, hour_start, minutes, row['forecast_load_mw']
                    )
                    if not pd.isna(extrap_error):
                        X['error_lag0'] = extrap_error
                    else:
                        # Fallback to error_lag1 if extrapolation fails
                        X['error_lag0'] = row['error_lag1']

                # Make prediction
                pred = models[h].predict(X.values.reshape(1, -1))[0]
                errors.append(abs(actual_error - pred))

            mae = np.nanmean(errors)
            quarter_results[f'h{h}_mae'] = mae

        # Print results for this quarter
        print(f"    H+1: {quarter_results['h1_mae']:.1f} MW")
        print(f"    H+2: {quarter_results['h2_mae']:.1f} MW")
        print(f"    H+3: {quarter_results['h3_mae']:.1f} MW")
        print(f"    H+4: {quarter_results['h4_mae']:.1f} MW")
        print(f"    H+5: {quarter_results['h5_mae']:.1f} MW")

        results.append(quarter_results)

    return pd.DataFrame(results)


def analyze_results(results_df):
    """Analyze and summarize results."""
    print("\n" + "=" * 70)
    print("SUMMARY: PERFORMANCE BY QUARTER")
    print("=" * 70)

    # Get Q0 baseline
    q0 = results_df[results_df['quarter'] == 0].iloc[0]

    print("\n  Horizon | Q0 (base) | Q1 (15m) | Q2 (30m) | Q3 (45m) |")
    print("  --------|-----------|----------|----------|----------|")

    for h in range(1, 6):
        q0_mae = q0[f'h{h}_mae']
        row = f"  H+{h}     | {q0_mae:6.1f} MW |"

        for q in [1, 2, 3]:
            q_row = results_df[results_df['quarter'] == q].iloc[0]
            q_mae = q_row[f'h{h}_mae']
            diff = q_mae - q0_mae
            row += f" {q_mae:5.1f} ({diff:+.1f}) |"

        print(row)

    # Calculate average improvement
    print("\n  Average improvement over Q0:")
    for q in [1, 2, 3]:
        q_row = results_df[results_df['quarter'] == q].iloc[0]
        avg_improvement = 0
        for h in range(1, 6):
            avg_improvement += (q0[f'h{h}_mae'] - q_row[f'h{h}_mae'])
        avg_improvement /= 5
        print(f"    Q{q}: {avg_improvement:+.1f} MW average across all horizons")


def main():
    df, df_3min = load_data()

    # Create features
    df = create_features(df)

    # Train simple models
    models, features = train_simple_models(df)

    # Evaluate at each quarter
    results = evaluate_at_quarters(df, df_3min, models, features)

    # Analyze
    analyze_results(results)

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
Key question: Does extrapolated error help the base models?

If Q1-Q3 show IMPROVEMENT over Q0:
  -> Base models can use extrapolated error effectively
  -> No need for separate quarter-specific models

If Q1-Q3 show DEGRADATION:
  -> Extrapolated error has different characteristics
  -> Consider training separate models for each quarter
  -> Or use weighted combination (model pred + extrapolation)

Note: H+1 should always improve with extrapolation (direct effect)
      H+2-H+5 improvement depends on error autocorrelation
""")

    return results


if __name__ == "__main__":
    results = main()
