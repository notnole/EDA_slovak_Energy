"""
Active Hours Day-Ahead Model
============================
Train only on "hard" hours (ramps, midday, peaks) where prediction is difficult.
Skip easy night hours to see if model learns more signal.
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

    threeminute_path = BASE_PATH / "data" / "features" / "load_3min.csv"
    df_3min = pd.read_csv(threeminute_path)
    df_3min['datetime'] = pd.to_datetime(df_3min['datetime'])
    if 'load_mw' in df_3min.columns:
        df_3min = df_3min.rename(columns={'load_mw': 'load'})

    return df_hourly, df_3min


def compute_seasonal_baseline(df: pd.DataFrame) -> pd.DataFrame:
    seasonal = df.groupby(['dow', 'hour', 'month'])['error'].agg(['mean', 'std']).reset_index()
    seasonal.columns = ['dow', 'hour', 'month', 'seasonal_mean', 'seasonal_std']
    return seasonal


def extract_3min_daily_features(df_3min: pd.DataFrame, df_hourly: pd.DataFrame) -> pd.DataFrame:
    df_3min['hour'] = df_3min['datetime'].dt.hour
    df_3min['date'] = df_3min['datetime'].dt.date

    hourly_errors = df_hourly[['date', 'hour', 'error', 'damas_load', 'actual_load']].copy()
    hourly_errors['date'] = pd.to_datetime(hourly_errors['date']).dt.date

    df_merged = df_3min.merge(hourly_errors, on=['date', 'hour'], how='inner')

    daily_features = []
    for date, day_data in df_merged.groupby('date'):
        if len(day_data) < 100:
            continue

        features = {'date': date}
        features['error_mean'] = day_data['error'].mean()
        features['error_std'] = day_data['error'].std()
        features['error_range'] = day_data['error'].max() - day_data['error'].min()

        hourly_load_std = day_data.groupby('hour')['load'].std()
        features['within_hour_std'] = hourly_load_std.mean()

        day_data = day_data.sort_values('datetime')
        load_diff = day_data['load'].diff().abs()
        features['ramp_mean'] = load_diff.mean()
        features['ramp_max'] = load_diff.max()

        errors = day_data['error'].values
        if len(errors) > 10:
            valid = ~np.isnan(errors[:-1]) & ~np.isnan(errors[1:])
            if valid.sum() > 10:
                features['error_autocorr'] = np.corrcoef(errors[:-1][valid], errors[1:][valid])[0, 1]
            else:
                features['error_autocorr'] = 0
        else:
            features['error_autocorr'] = 0

        morning = day_data[day_data['hour'].between(6, 11)]
        afternoon = day_data[day_data['hour'].between(12, 17)]
        evening = day_data[day_data['hour'].between(18, 22)]

        features['morning_error'] = morning['error'].mean() if len(morning) > 0 else 0
        features['afternoon_error'] = afternoon['error'].mean() if len(afternoon) > 0 else 0
        features['evening_error'] = evening['error'].mean() if len(evening) > 0 else 0

        q1 = day_data[day_data['hour'] < 6]['error'].mean()
        q4 = day_data[day_data['hour'] >= 18]['error'].mean()
        features['error_trend'] = q4 - q1 if not (np.isnan(q1) or np.isnan(q4)) else 0

        daily_features.append(features)

    return pd.DataFrame(daily_features)


def build_features(df: pd.DataFrame, seasonal: pd.DataFrame, daily_3min: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.merge(seasonal, on=['dow', 'hour', 'month'], how='left')

    df['date'] = pd.to_datetime(df['date'])
    df['prev_date'] = df['date'] - pd.Timedelta(days=1)

    df_prev = df[['date', 'hour', 'error']].copy()
    df_prev.columns = ['prev_date', 'hour', 'd1_same_hour_error']
    df = df.merge(df_prev, on=['prev_date', 'hour'], how='left')

    df['d1_vs_seasonal'] = df['d1_same_hour_error'] - df['seasonal_mean']

    df['prev_week_date'] = df['date'] - pd.Timedelta(days=7)
    df_week = df[['date', 'hour', 'error']].copy()
    df_week.columns = ['prev_week_date', 'hour', 'd7_same_hour_error']
    df = df.merge(df_week, on=['prev_week_date', 'hour'], how='left')

    daily_3min['prev_date'] = pd.to_datetime(daily_3min['date'])
    daily_3min_renamed = daily_3min.add_prefix('d1_3min_')
    daily_3min_renamed = daily_3min_renamed.rename(columns={'d1_3min_date': 'drop', 'd1_3min_prev_date': 'prev_date'})
    daily_3min_renamed = daily_3min_renamed.drop(columns=['drop'])

    df = df.merge(daily_3min_renamed, on='prev_date', how='left')

    df['is_morning_ramp'] = df['hour'].isin([5, 6, 7, 8]).astype(int)
    df['is_evening_peak'] = df['hour'].isin([18, 19, 20, 21]).astype(int)
    df['is_midday'] = df['hour'].isin([9, 10, 11, 12, 13, 14, 15, 16, 17]).astype(int)
    df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 22, 23]).astype(int)

    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_monday'] = (df['dow'] == 0).astype(int)

    return df


def train_model(df: pd.DataFrame, hour_filter=None, filter_name="All Hours"):
    """Train model with optional hour filtering."""

    feature_cols = [
        'hour', 'dow', 'month', 'is_weekend', 'is_monday',
        'is_morning_ramp', 'is_evening_peak', 'is_midday', 'is_night',
        'seasonal_mean', 'seasonal_std',
        'd1_same_hour_error', 'd1_vs_seasonal', 'd7_same_hour_error',
        'd1_3min_error_mean', 'd1_3min_error_std', 'd1_3min_error_range',
        'd1_3min_within_hour_std', 'd1_3min_ramp_mean', 'd1_3min_ramp_max',
        'd1_3min_error_autocorr', 'd1_3min_morning_error', 'd1_3min_afternoon_error',
        'd1_3min_evening_error', 'd1_3min_error_trend'
    ]

    available_features = [f for f in feature_cols if f in df.columns]

    df_model = df.dropna(subset=['error', 'd1_same_hour_error'])

    # Apply hour filter for training
    if hour_filter is not None:
        df_train_pool = df_model[df_model['hour'].isin(hour_filter)]
    else:
        df_train_pool = df_model

    df_train_pool = df_train_pool.sort_values('datetime').reset_index(drop=True)

    X = df_train_pool[available_features].fillna(0)
    y = df_train_pool['error']

    tscv = TimeSeriesSplit(n_splits=5)

    results = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        damas_val = df_train_pool.iloc[val_idx]['damas_load'].values
        actual_val = df_train_pool.iloc[val_idx]['actual_load'].values

        model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, random_state=42, verbosity=-1
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        damas_mae = np.mean(np.abs(actual_val - damas_val))
        model_pred_load = damas_val + pred
        model_mae = np.mean(np.abs(actual_val - model_pred_load))
        improvement = (damas_mae - model_mae) / damas_mae * 100

        results.append({
            'fold': fold + 1,
            'damas_mae': damas_mae,
            'model_mae': model_mae,
            'improvement': improvement,
            'n_samples': len(y_val)
        })

    return results


def main():
    print("=" * 70)
    print("ACTIVE HOURS VS ALL HOURS - DAY-AHEAD MODEL COMPARISON")
    print("=" * 70)

    # Load and prepare data
    print("\nLoading data...")
    df_hourly, df_3min = load_all_data()
    seasonal = compute_seasonal_baseline(df_hourly)
    daily_3min = extract_3min_daily_features(df_3min, df_hourly)
    df_features = build_features(df_hourly, seasonal, daily_3min)

    # Define hour groups
    night_hours = [0, 1, 2, 3, 4, 22, 23]
    morning_ramp = [5, 6, 7, 8]
    midday = [9, 10, 11, 12, 13, 14, 15, 16, 17]
    evening_peak = [18, 19, 20, 21]
    active_hours = morning_ramp + midday + evening_peak  # 5-21

    # Analyze error by hour group
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS BY HOUR GROUP")
    print("=" * 70)

    df_valid = df_features.dropna(subset=['error'])
    for name, hours in [("Night (0-4, 22-23)", night_hours),
                        ("Morning Ramp (5-8)", morning_ramp),
                        ("Midday (9-17)", midday),
                        ("Evening Peak (18-21)", evening_peak)]:
        subset = df_valid[df_valid['hour'].isin(hours)]
        mae = np.abs(subset['error']).mean()
        std = subset['error'].std()
        print(f"  {name:25}: MAE={mae:.1f} MW, Std={std:.1f} MW, n={len(subset):,}")

    # Test different training strategies
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    strategies = [
        ("All Hours (0-23)", None),
        ("Active Hours Only (5-21)", active_hours),
        ("Peak Hours Only (6-9, 17-21)", [6, 7, 8, 9, 17, 18, 19, 20, 21]),
        ("Midday + Evening (9-21)", list(range(9, 22))),
    ]

    all_results = {}
    for name, hour_filter in strategies:
        print(f"\n{name}:")
        results = train_model(df_features, hour_filter, name)

        avg_damas = np.mean([r['damas_mae'] for r in results])
        avg_model = np.mean([r['model_mae'] for r in results])
        avg_improvement = np.mean([r['improvement'] for r in results])
        avg_n = np.mean([r['n_samples'] for r in results])

        print(f"  DAMAS MAE: {avg_damas:.1f} MW")
        print(f"  Model MAE: {avg_model:.1f} MW")
        print(f"  Improvement: {avg_improvement:+.1f}%")
        print(f"  Avg samples/fold: {avg_n:.0f}")

        all_results[name] = {
            'damas_mae': avg_damas,
            'model_mae': avg_model,
            'improvement': avg_improvement,
            'results': results
        }

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Strategy':<30} {'DAMAS':<10} {'Model':<10} {'Improv':<10}")
    print("-" * 60)
    for name, res in all_results.items():
        print(f"{name:<30} {res['damas_mae']:<10.1f} {res['model_mae']:<10.1f} {res['improvement']:+.1f}%")

    # Find best
    best = max(all_results.items(), key=lambda x: x[1]['improvement'])
    print(f"\nBest strategy: {best[0]} with {best[1]['improvement']:+.1f}% improvement")


if __name__ == "__main__":
    main()
