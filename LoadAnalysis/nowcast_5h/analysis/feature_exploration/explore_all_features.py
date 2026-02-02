"""
Explore All Possibilities for 5-Hour Nowcasting Model
======================================================
Test every available data source and feature combination:
1. 3-minute SCADA data (regulation, production, export/import)
2. Day-ahead prices
3. Sub-hourly patterns
4. Extended lag features
5. Feature interactions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent.parent  # ipesoft_eda_data
OUTPUT_PATH = Path(__file__).parent / 'feature_exploration'
OUTPUT_PATH.mkdir(exist_ok=True)


def load_all_data():
    """Load all available data sources."""
    print("=" * 70)
    print("LOADING ALL AVAILABLE DATA")
    print("=" * 70)

    data = {}

    # 1. Hourly DAMAS load data
    load_path = BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet'
    df_load = pd.read_parquet(load_path)
    df_load['datetime'] = pd.to_datetime(df_load['datetime'])
    df_load = df_load.sort_values('datetime').reset_index(drop=True)
    data['load'] = df_load
    print(f"\n1. DAMAS Load: {len(df_load):,} hourly records")
    print(f"   Columns: {list(df_load.columns)}")

    # 2. Day-ahead prices
    price_path = BASE_PATH / 'features' / 'DamasPrices' / 'data' / 'da_prices.parquet'
    if price_path.exists():
        df_price = pd.read_parquet(price_path)
        df_price['datetime'] = pd.to_datetime(df_price['datetime'])
        data['price'] = df_price
        print(f"\n2. DA Prices: {len(df_price):,} records")
        print(f"   Columns: {list(df_price.columns)}")
    else:
        print("\n2. DA Prices: NOT FOUND")
        data['price'] = None

    # 3. 3-minute SCADA data
    scada_files = {
        'load_3min': 'load_3min.csv',
        'regulation_3min': 'regulation_3min.csv',
        'production_3min': 'production_3min.csv',
        'export_import_3min': 'export_import_3min.csv',
    }

    print("\n3. 3-Minute SCADA Data:")
    for name, filename in scada_files.items():
        path = BASE_PATH / 'data' / 'features' / filename
        if path.exists():
            df = pd.read_csv(path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            data[name] = df
            print(f"   {name}: {len(df):,} records, columns: {list(df.columns)}")
        else:
            print(f"   {name}: NOT FOUND")
            data[name] = None

    return data


def aggregate_3min_to_hourly(df_3min: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Aggregate 3-minute data to hourly statistics."""
    df = df_3min.copy()
    df['hour_start'] = df['datetime'].dt.floor('H')

    # Aggregations
    hourly = df.groupby('hour_start').agg({
        value_col: ['mean', 'std', 'min', 'max', 'first', 'last']
    }).reset_index()
    hourly.columns = ['datetime', f'{value_col}_mean', f'{value_col}_std',
                      f'{value_col}_min', f'{value_col}_max',
                      f'{value_col}_first', f'{value_col}_last']

    # Derived features
    hourly[f'{value_col}_range'] = hourly[f'{value_col}_max'] - hourly[f'{value_col}_min']
    hourly[f'{value_col}_trend'] = hourly[f'{value_col}_last'] - hourly[f'{value_col}_first']

    return hourly


def build_master_dataset(data: dict) -> pd.DataFrame:
    """Merge all data sources into master dataset."""
    print("\n" + "=" * 70)
    print("BUILDING MASTER DATASET")
    print("=" * 70)

    # Start with DAMAS load
    df = data['load'].copy()
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']

    print(f"\nBase load data: {len(df):,} records")

    # Add prices
    if data['price'] is not None:
        df_price = data['price'][['datetime', 'price_eur_mwh']].copy()
        df = df.merge(df_price, on='datetime', how='left')
        print(f"  + Prices merged: {df['price_eur_mwh'].notna().sum():,} matched")

    # Add 3-minute SCADA aggregations
    scada_mapping = {
        'regulation_3min': 'regulation_mw',
        'production_3min': 'production_mw',
        'export_import_3min': 'export_import_mw',
        'load_3min': 'load_mw',
    }

    for name, col in scada_mapping.items():
        if data[name] is not None:
            hourly_agg = aggregate_3min_to_hourly(data[name], col)
            before = len(df)
            df = df.merge(hourly_agg, on='datetime', how='left')
            matched = df[f'{col}_mean'].notna().sum()
            print(f"  + {name} merged: {matched:,} matched")

    print(f"\nFinal master dataset: {len(df):,} records, {len(df.columns)} columns")

    return df


def create_all_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive lag features from all available data."""
    print("\n" + "=" * 70)
    print("CREATING LAG FEATURES")
    print("=" * 70)

    df = df.copy()

    # 1. Error lags (core)
    for lag in range(1, 13):
        df[f'error_lag{lag}'] = df['error'].shift(lag)
    print("  + Error lags 1-12")

    # 2. Rolling error statistics
    for window in [3, 6, 12, 24]:
        df[f'error_roll_mean_{window}h'] = df['error'].shift(1).rolling(window).mean()
        df[f'error_roll_std_{window}h'] = df['error'].shift(1).rolling(window).std()
    print("  + Rolling error stats (3h, 6h, 12h, 24h)")

    # 3. Price features (if available)
    if 'price_eur_mwh' in df.columns:
        for lag in range(1, 6):
            df[f'price_lag{lag}'] = df['price_eur_mwh'].shift(lag)
        df['price_change_1h'] = df['price_eur_mwh'] - df['price_eur_mwh'].shift(1)
        df['price_roll_mean_6h'] = df['price_eur_mwh'].shift(1).rolling(6).mean()
        print("  + Price lags and changes")

    # 4. Regulation features (if available)
    if 'regulation_mw_mean' in df.columns:
        for lag in range(1, 6):
            df[f'reg_mean_lag{lag}'] = df['regulation_mw_mean'].shift(lag)
            df[f'reg_std_lag{lag}'] = df['regulation_mw_std'].shift(lag)
        df['reg_trend_1h'] = df['regulation_mw_trend'].shift(1)
        df['reg_range_1h'] = df['regulation_mw_range'].shift(1)
        print("  + Regulation lags and stats")

    # 5. Production features (if available)
    if 'production_mw_mean' in df.columns:
        for lag in range(1, 4):
            df[f'prod_mean_lag{lag}'] = df['production_mw_mean'].shift(lag)
        df['prod_change_1h'] = df['production_mw_mean'] - df['production_mw_mean'].shift(1)
        print("  + Production lags")

    # 6. Export/Import features (if available)
    if 'export_import_mw_mean' in df.columns:
        for lag in range(1, 4):
            df[f'expimport_mean_lag{lag}'] = df['export_import_mw_mean'].shift(lag)
        df['expimport_change_1h'] = df['export_import_mw_mean'] - df['export_import_mw_mean'].shift(1)
        print("  + Export/Import lags")

    # 7. Load 3-min features (if available)
    if 'load_mw_std' in df.columns:
        df['load_volatility_1h'] = df['load_mw_std'].shift(1)
        df['load_intra_trend'] = df['load_mw_trend'].shift(1)
        print("  + Load sub-hourly volatility")

    # 8. Seasonal baseline
    seasonal = df.groupby(['dow', 'hour'])['error'].transform('mean')
    df['seasonal_error'] = seasonal
    df['error_vs_seasonal'] = df['error'].shift(1) - df['seasonal_error']
    print("  + Seasonal baseline")

    # 9. Time interactions
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_peak'] = df['hour'].isin([8, 9, 10, 17, 18, 19, 20]).astype(int)
    print("  + Time features")

    # 10. Targets for each horizon
    for h in range(1, 6):
        df[f'target_h{h}'] = df['error'].shift(-h)
        df[f'forecast_h{h}'] = df['forecast_load_mw'].shift(-h)
        df[f'hour_h{h}'] = df['hour'].shift(-h)
    print("  + Targets for H+1 to H+5")

    return df


def get_feature_groups():
    """Define feature groups for testing."""
    return {
        'baseline': [
            'error_lag1', 'error_lag2', 'error_lag3', 'error_lag4', 'error_lag5',
            'error_roll_mean_6h', 'error_roll_std_6h',
            'hour', 'dow', 'is_weekend',
        ],
        'extended_lags': [
            'error_lag1', 'error_lag2', 'error_lag3', 'error_lag4', 'error_lag5',
            'error_lag6', 'error_lag7', 'error_lag8', 'error_lag9', 'error_lag10',
            'error_lag11', 'error_lag12',
        ],
        'rolling_stats': [
            'error_roll_mean_3h', 'error_roll_std_3h',
            'error_roll_mean_6h', 'error_roll_std_6h',
            'error_roll_mean_12h', 'error_roll_std_12h',
            'error_roll_mean_24h', 'error_roll_std_24h',
        ],
        'price': [
            'price_lag1', 'price_lag2', 'price_lag3',
            'price_change_1h', 'price_roll_mean_6h',
        ],
        'regulation': [
            'reg_mean_lag1', 'reg_mean_lag2', 'reg_mean_lag3',
            'reg_std_lag1', 'reg_std_lag2',
            'reg_trend_1h', 'reg_range_1h',
        ],
        'production': [
            'prod_mean_lag1', 'prod_mean_lag2', 'prod_mean_lag3',
            'prod_change_1h',
        ],
        'export_import': [
            'expimport_mean_lag1', 'expimport_mean_lag2', 'expimport_mean_lag3',
            'expimport_change_1h',
        ],
        'load_volatility': [
            'load_volatility_1h', 'load_intra_trend',
        ],
        'seasonal': [
            'seasonal_error', 'error_vs_seasonal',
        ],
        'time_cyclic': [
            'hour_sin', 'hour_cos', 'is_peak',
        ],
    }


def test_feature_combination(df: pd.DataFrame, features: list, horizon: int, name: str) -> dict:
    """Test a specific feature combination."""
    target = f'target_h{horizon}'

    # Filter available features
    available = [f for f in features if f in df.columns]
    if len(available) < len(features):
        missing = set(features) - set(available)
        # print(f"    Missing: {missing}")

    if len(available) < 3:
        return None

    # Prepare data
    df_model = df.dropna(subset=[target] + available).copy()
    train = df_model[df_model['year'] < 2025]
    test = df_model[df_model['year'] >= 2025]

    if len(train) < 1000 or len(test) < 100:
        return None

    X_train, y_train = train[available], train[target]
    X_test, y_test = test[available], test[target]

    # Train model
    model = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=6,
        num_leaves=31, min_child_samples=30, random_state=42, verbosity=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    pred = model.predict(X_test)
    baseline_mae = np.abs(y_test).mean()
    model_mae = np.abs(y_test - pred).mean()
    improvement = (baseline_mae - model_mae) / baseline_mae * 100

    return {
        'name': name,
        'horizon': horizon,
        'n_features': len(available),
        'baseline_mae': baseline_mae,
        'model_mae': model_mae,
        'improvement': improvement,
        'features': available,
    }


def main():
    # Load all data
    data = load_all_data()

    # Build master dataset
    df = build_master_dataset(data)

    # Create all features
    df = create_all_lag_features(df)

    print(f"\nFinal dataset: {len(df):,} records, {len(df.columns)} columns")

    # Get feature groups
    feature_groups = get_feature_groups()

    # Test each group individually
    print("\n" + "=" * 70)
    print("TESTING INDIVIDUAL FEATURE GROUPS (H+1)")
    print("=" * 70)

    individual_results = []
    for group_name, features in feature_groups.items():
        result = test_feature_combination(df, features, horizon=1, name=group_name)
        if result:
            individual_results.append(result)
            print(f"\n  {group_name:20} | {result['n_features']:2} features | "
                  f"MAE: {result['model_mae']:.1f} | Improvement: {result['improvement']:+.1f}%")

    # Test cumulative combinations
    print("\n" + "=" * 70)
    print("TESTING CUMULATIVE FEATURE COMBINATIONS (H+1)")
    print("=" * 70)

    cumulative_features = []
    cumulative_results = []

    # Order by expected importance
    group_order = ['baseline', 'extended_lags', 'rolling_stats', 'regulation',
                   'price', 'production', 'export_import', 'load_volatility',
                   'seasonal', 'time_cyclic']

    for group_name in group_order:
        if group_name in feature_groups:
            cumulative_features.extend(feature_groups[group_name])
            unique_features = list(dict.fromkeys(cumulative_features))  # Remove duplicates

            result = test_feature_combination(df, unique_features, horizon=1, name=f"+ {group_name}")
            if result:
                cumulative_results.append(result)
                print(f"\n  + {group_name:20} | {result['n_features']:3} features | "
                      f"MAE: {result['model_mae']:.1f} | Improvement: {result['improvement']:+.1f}%")

    # Best combination for all horizons
    print("\n" + "=" * 70)
    print("BEST COMBINATION ACROSS ALL HORIZONS")
    print("=" * 70)

    # Use only features that exist and have good coverage
    all_features = []
    for features in feature_groups.values():
        for f in features:
            if f in df.columns:
                # Check coverage
                coverage = df[f].notna().mean()
                if coverage > 0.5:  # At least 50% coverage
                    all_features.append(f)

    all_features = list(dict.fromkeys(all_features))
    print(f"\nFeatures with >50% coverage: {len(all_features)}")

    horizon_results = []
    for h in range(1, 6):
        result = test_feature_combination(df, all_features, horizon=h, name=f"All features H+{h}")
        if result:
            horizon_results.append(result)
            print(f"\n  H+{h}: MAE={result['model_mae']:.1f} MW | "
                  f"Baseline={result['baseline_mae']:.1f} MW | "
                  f"Improvement: {result['improvement']:+.1f}%")

    # Feature importance analysis for best model
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS (H+1 BEST MODEL)")
    print("=" * 70)

    target = 'target_h1'
    available = [f for f in all_features if f in df.columns]
    df_model = df.dropna(subset=[target] + available)
    train = df_model[df_model['year'] < 2025]

    if len(train) > 0 and len(available) > 0:
        X_train = train[available]
        y_train = train[target]

        # Check for empty dataframe
        if len(X_train) > 100:
            model = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                num_leaves=31, min_child_samples=30, random_state=42, verbosity=-1
            )
            model.fit(X_train, y_train)

            importance = pd.DataFrame({
                'feature': available,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nTop 20 Features:")
            for i, row in importance.head(20).iterrows():
                # Categorize feature
                if 'error' in row['feature']:
                    cat = 'ERROR'
                elif 'reg' in row['feature']:
                    cat = 'REGULATION'
                elif 'price' in row['feature']:
                    cat = 'PRICE'
                elif 'prod' in row['feature']:
                    cat = 'PRODUCTION'
                elif 'expimport' in row['feature']:
                    cat = 'EXP/IMP'
                elif 'load' in row['feature']:
                    cat = 'LOAD'
                else:
                    cat = 'OTHER'
                print(f"  {row['feature']:30} : {row['importance']:6.0f}  [{cat}]")
        else:
            importance = None
            print("  Not enough training data for feature importance")
    else:
        importance = None
        print("  No valid data for feature importance")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n1. Individual Feature Group Performance (H+1):")
    for r in sorted(individual_results, key=lambda x: -x['improvement']):
        print(f"   {r['name']:20} : {r['improvement']:+.1f}%")

    print("\n2. Best Model Performance by Horizon:")
    for r in horizon_results:
        print(f"   H+{r['horizon']}: {r['improvement']:+.1f}% (MAE: {r['model_mae']:.1f} MW)")

    if horizon_results:
        best_h1 = [r for r in horizon_results if r['horizon'] == 1][0]
        print(f"\n3. Best H+1 Improvement: {best_h1['improvement']:+.1f}%")
        print(f"   Baseline MAE: {best_h1['baseline_mae']:.1f} MW")
        print(f"   Model MAE:    {best_h1['model_mae']:.1f} MW")

    return df, horizon_results, importance


if __name__ == "__main__":
    df, results, importance = main()
