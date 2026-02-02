"""
Best Day-Ahead Load Forecast Error Prediction Model
====================================================
Combines all improvements discovered:
1. Hour-specific features (not daily averages)
2. 3-minute granular features from previous day
3. Optimized feature engineering

Target: Predict DAMAS forecast error for day D+1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent


def load_all_data():
    """Load hourly DAMAS data and 3-minute SCADA data."""
    # Hourly data
    hourly_path = BASE_PATH / "features" / "DamasLoad" / "load_data.parquet"
    df_hourly = pd.read_parquet(hourly_path)
    df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
    df_hourly['date'] = df_hourly['datetime'].dt.date
    df_hourly['hour'] = df_hourly['datetime'].dt.hour
    df_hourly['dow'] = df_hourly['datetime'].dt.dayofweek
    df_hourly['month'] = df_hourly['datetime'].dt.month
    df_hourly['error'] = df_hourly['actual_load_mw'] - df_hourly['forecast_load_mw']
    df_hourly['damas_load'] = df_hourly['forecast_load_mw']
    df_hourly['actual_load'] = df_hourly['actual_load_mw']

    # 3-minute data
    threeminute_path = BASE_PATH / "data" / "features" / "load_3min.csv"
    df_3min = pd.read_csv(threeminute_path)
    df_3min['datetime'] = pd.to_datetime(df_3min['datetime'])

    return df_hourly, df_3min


def compute_seasonal_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Compute seasonal error expectations by dow+hour+month."""
    seasonal = df.groupby(['dow', 'hour', 'month'])['error'].agg(['mean', 'std']).reset_index()
    seasonal.columns = ['dow', 'hour', 'month', 'seasonal_mean', 'seasonal_std']
    return seasonal


def extract_3min_daily_features(df_3min: pd.DataFrame, df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Extract daily features from 3-minute data."""
    # Rename column if needed
    if 'load_mw' in df_3min.columns and 'load' not in df_3min.columns:
        df_3min = df_3min.rename(columns={'load_mw': 'load'})

    # Merge 3-min with hourly to get errors at 3-min level
    df_3min['hour'] = df_3min['datetime'].dt.hour
    df_3min['date'] = df_3min['datetime'].dt.date

    hourly_errors = df_hourly[['date', 'hour', 'error', 'damas_load', 'actual_load']].copy()
    hourly_errors['date'] = pd.to_datetime(hourly_errors['date']).dt.date

    df_merged = df_3min.merge(hourly_errors, on=['date', 'hour'], how='inner')

    # Daily aggregation
    daily_features = []
    for date, day_data in df_merged.groupby('date'):
        if len(day_data) < 100:  # Need sufficient data
            continue

        features = {'date': date}

        # Error statistics
        features['error_mean'] = day_data['error'].mean()
        features['error_std'] = day_data['error'].std()
        features['error_range'] = day_data['error'].max() - day_data['error'].min()

        # Within-hour variance (measure of sub-hourly volatility)
        hourly_load_std = day_data.groupby('hour')['load'].std()
        features['within_hour_std'] = hourly_load_std.mean()

        # Ramp statistics
        day_data = day_data.sort_values('datetime')
        load_diff = day_data['load'].diff().abs()
        features['ramp_mean'] = load_diff.mean()
        features['ramp_max'] = load_diff.max()

        # Error autocorrelation at 3-min level
        errors = day_data['error'].values
        if len(errors) > 10:
            valid = ~np.isnan(errors[:-1]) & ~np.isnan(errors[1:])
            if valid.sum() > 10:
                features['error_autocorr'] = np.corrcoef(errors[:-1][valid], errors[1:][valid])[0, 1]
            else:
                features['error_autocorr'] = 0
        else:
            features['error_autocorr'] = 0

        # Period-specific errors
        morning = day_data[day_data['hour'].between(6, 11)]
        afternoon = day_data[day_data['hour'].between(12, 17)]
        evening = day_data[day_data['hour'].between(18, 22)]

        features['morning_error'] = morning['error'].mean() if len(morning) > 0 else 0
        features['afternoon_error'] = afternoon['error'].mean() if len(afternoon) > 0 else 0
        features['evening_error'] = evening['error'].mean() if len(evening) > 0 else 0

        # Error trend (morning to evening)
        q1 = day_data[day_data['hour'] < 6]['error'].mean()
        q4 = day_data[day_data['hour'] >= 18]['error'].mean()
        features['error_trend'] = q4 - q1 if not (np.isnan(q1) or np.isnan(q4)) else 0

        daily_features.append(features)

    return pd.DataFrame(daily_features)


def build_features(df: pd.DataFrame, seasonal: pd.DataFrame, daily_3min: pd.DataFrame) -> pd.DataFrame:
    """Build all features for day-ahead prediction."""
    df = df.copy()

    # Merge seasonal baseline
    df = df.merge(seasonal, on=['dow', 'hour', 'month'], how='left')

    # Day-ahead: we predict for tomorrow, so we need YESTERDAY's features
    # Create lagged date for joining
    df['date'] = pd.to_datetime(df['date'])
    df['prev_date'] = df['date'] - pd.Timedelta(days=1)

    # Previous day same hour error (this is our key feature)
    df_prev = df[['date', 'hour', 'error']].copy()
    df_prev.columns = ['prev_date', 'hour', 'd1_same_hour_error']
    df = df.merge(df_prev, on=['prev_date', 'hour'], how='left')

    # Deviation from seasonal
    df['d1_vs_seasonal'] = df['d1_same_hour_error'] - df['seasonal_mean']

    # 7-day ago same hour error
    df['prev_week_date'] = df['date'] - pd.Timedelta(days=7)
    df_week = df[['date', 'hour', 'error']].copy()
    df_week.columns = ['prev_week_date', 'hour', 'd7_same_hour_error']
    df = df.merge(df_week, on=['prev_week_date', 'hour'], how='left')

    # Merge 3-minute daily features (from previous day)
    daily_3min['prev_date'] = pd.to_datetime(daily_3min['date'])
    daily_3min_renamed = daily_3min.add_prefix('d1_3min_')
    daily_3min_renamed = daily_3min_renamed.rename(columns={'d1_3min_date': 'drop', 'd1_3min_prev_date': 'prev_date'})
    daily_3min_renamed = daily_3min_renamed.drop(columns=['drop'])

    df = df.merge(daily_3min_renamed, on='prev_date', how='left')

    # Hour-specific directional bias (from our analysis)
    df['is_morning_ramp'] = df['hour'].isin([5, 6, 7]).astype(int)
    df['is_evening_peak'] = df['hour'].isin([19, 20, 21]).astype(int)
    df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 23]).astype(int)

    # Day type features
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_monday'] = (df['dow'] == 0).astype(int)
    df['is_friday'] = (df['dow'] == 4).astype(int)

    return df


def train_best_model(df: pd.DataFrame):
    """Train the best day-ahead model with cross-validation."""

    # Feature columns
    feature_cols = [
        # Calendar
        'hour', 'dow', 'month', 'is_weekend', 'is_monday', 'is_friday',
        'is_morning_ramp', 'is_evening_peak', 'is_night',
        # Seasonal baseline
        'seasonal_mean', 'seasonal_std',
        # Previous day same hour
        'd1_same_hour_error', 'd1_vs_seasonal',
        # Previous week
        'd7_same_hour_error',
        # 3-minute features from previous day
        'd1_3min_error_mean', 'd1_3min_error_std', 'd1_3min_error_range',
        'd1_3min_within_hour_std', 'd1_3min_ramp_mean', 'd1_3min_ramp_max',
        'd1_3min_error_autocorr', 'd1_3min_morning_error', 'd1_3min_afternoon_error',
        'd1_3min_evening_error', 'd1_3min_error_trend'
    ]

    # Filter to available features
    available_features = [f for f in feature_cols if f in df.columns]

    # Remove rows with missing target or key features
    df_model = df.dropna(subset=['error', 'd1_same_hour_error'])
    df_model = df_model.sort_values('datetime').reset_index(drop=True)

    X = df_model[available_features].fillna(0)
    y = df_model['error']

    # Time series CV
    tscv = TimeSeriesSplit(n_splits=5)

    results = []
    feature_importance = np.zeros(len(available_features))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        damas_val = df_model.iloc[val_idx]['damas_load'].values
        actual_val = df_model.iloc[val_idx]['actual_load'].values

        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            random_state=42,
            verbosity=-1
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        # Metrics
        damas_mae = np.mean(np.abs(actual_val - damas_val))
        model_pred_load = damas_val + pred
        model_mae = np.mean(np.abs(actual_val - model_pred_load))
        improvement = (damas_mae - model_mae) / damas_mae * 100

        results.append({
            'fold': fold + 1,
            'damas_mae': damas_mae,
            'model_mae': model_mae,
            'improvement': improvement
        })

        feature_importance += model.feature_importances_

    feature_importance /= 5

    return results, available_features, feature_importance, model


def main():
    print("=" * 60)
    print("BEST DAY-AHEAD LOAD FORECAST ERROR MODEL")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df_hourly, df_3min = load_all_data()
    print(f"  Hourly records: {len(df_hourly):,}")
    print(f"  3-minute records: {len(df_3min):,}")

    # Compute seasonal baseline
    print("\nComputing seasonal baseline...")
    seasonal = compute_seasonal_baseline(df_hourly)

    # Extract 3-minute features
    print("Extracting 3-minute daily features...")
    daily_3min = extract_3min_daily_features(df_3min, df_hourly)
    print(f"  Days with 3-min features: {len(daily_3min)}")

    # Build features
    print("Building feature set...")
    df_features = build_features(df_hourly, seasonal, daily_3min)

    # Train model
    print("\nTraining model with 5-fold time series CV...")
    results, features, importance, final_model = train_best_model(df_features)

    # Results
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"\n{'Fold':<6} {'DAMAS MAE':<12} {'Model MAE':<12} {'Improvement':<12}")
    print("-" * 44)

    for r in results:
        print(f"{r['fold']:<6} {r['damas_mae']:<12.1f} {r['model_mae']:<12.1f} {r['improvement']:+.1f}%")

    avg_damas = np.mean([r['damas_mae'] for r in results])
    avg_model = np.mean([r['model_mae'] for r in results])
    avg_improvement = np.mean([r['improvement'] for r in results])

    print("-" * 44)
    print(f"{'AVG':<6} {avg_damas:<12.1f} {avg_model:<12.1f} {avg_improvement:+.1f}%")

    # Feature importance
    print("\n" + "=" * 60)
    print("TOP 15 FEATURES BY IMPORTANCE")
    print("=" * 60)

    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)

    for i, row in importance_df.head(15).iterrows():
        marker = " <-- 3min" if '3min' in row['feature'] else ""
        print(f"  {row['feature']:<30}: {row['importance']:.0f}{marker}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"""
  DAMAS Baseline MAE:     {avg_damas:.1f} MW
  Best Model MAE:         {avg_model:.1f} MW

  IMPROVEMENT:            {avg_improvement:+.1f}%
  Absolute gain:          {avg_damas - avg_model:.1f} MW

  Key contributing features:
  - Hour-specific lag-1 error (d1_same_hour_error)
  - Seasonal deviation (d1_vs_seasonal)
  - 3-min error patterns (trend, autocorr, period means)
  - Calendar features (dow, hour, weekend flags)
""")

    # Save model
    output_dir = Path(__file__).parent / "best_model"
    output_dir.mkdir(exist_ok=True)

    import joblib
    joblib.dump(final_model, output_dir / "best_dayahead_model.joblib")

    # Save feature list
    with open(output_dir / "features.txt", 'w') as f:
        for feat in features:
            f.write(f"{feat}\n")

    print(f"\n  Model saved to: {output_dir}")

    return avg_improvement


if __name__ == "__main__":
    improvement = main()
