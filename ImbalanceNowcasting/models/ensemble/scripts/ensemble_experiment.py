"""
ENSEMBLE EXPERIMENT FOR LONGER LEAD TIMES
==========================================

Tests diverse ensemble approaches for lead 12 and 9 minute predictions.

Diversity strategies:
1. Different algorithms: LightGBM, XGBoost, CatBoost, Ridge, Random Forest
2. Different feature subsets: proxy-focused, historical, time-focused, full

Key metric: Error correlation between models
- High correlation (>0.9) = low diversity, ensemble won't help
- Low correlation (<0.7) = good diversity, ensemble likely helps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import gradient boosting libraries
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not available")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: CatBoost not available")

# Paths
FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\models\ensemble")


# =============================================================================
# DATA LOADING AND FEATURE ENGINEERING (from V4)
# =============================================================================

def load_data():
    """Load data."""
    print("Loading data...")
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
    load_df = pd.read_csv(FEATURES_DIR / 'load_3min.csv', parse_dates=['datetime'])
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
        columns={'System Imbalance (MWh)': 'imbalance'}
    )
    return reg_df, load_df, label_df


def compute_load_expected(load_df):
    """Compute expected load by time-of-day from 2024 data."""
    load_df = load_df.copy()
    load_df['hour'] = load_df['datetime'].dt.hour
    load_df['minute'] = load_df['datetime'].dt.minute
    load_df['is_weekend'] = load_df['datetime'].dt.dayofweek >= 5
    train_mask = load_df['datetime'].dt.year == 2024
    expected = load_df[train_mask].groupby(['hour', 'minute', 'is_weekend'])['load_mw'].mean()
    expected.name = 'expected_load'
    return expected


def add_historical_regulation_features(reg_df):
    """Add historical regulation features."""
    reg_df = reg_df.sort_values('datetime').copy()
    reg_df['reg_hist_mean_10'] = reg_df['regulation_mw'].shift(1).rolling(10).mean()
    reg_df['reg_hist_std_10'] = reg_df['regulation_mw'].shift(1).rolling(10).std()
    reg_df['reg_hist_min_10'] = reg_df['regulation_mw'].shift(1).rolling(10).min()
    reg_df['reg_hist_max_10'] = reg_df['regulation_mw'].shift(1).rolling(10).max()
    reg_df['reg_hist_range_10'] = reg_df['reg_hist_max_10'] - reg_df['reg_hist_min_10']
    reg_df['reg_hist_trend_10'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(10)
    reg_df['reg_hist_mean_20'] = reg_df['regulation_mw'].shift(1).rolling(20).mean()
    reg_df['reg_hist_std_20'] = reg_df['regulation_mw'].shift(1).rolling(20).std()
    reg_df['reg_momentum'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(2)
    reg_df['reg_acceleration'] = reg_df['reg_momentum'] - reg_df['reg_momentum'].shift(1)
    return reg_df


def create_base_features(reg_df, load_df, label_df, load_expected):
    """Create base feature dataframe."""
    reg_df = add_historical_regulation_features(reg_df)

    reg_df['datetime_floor'] = reg_df['datetime'].dt.floor('3min')
    reg_df['settlement_end'] = reg_df['datetime_floor'].dt.ceil('15min')
    mask = reg_df['datetime_floor'] == reg_df['settlement_end']
    reg_df.loc[mask, 'settlement_end'] = reg_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    reg_df['settlement_start'] = reg_df['settlement_end'] - pd.Timedelta(minutes=15)
    reg_df['minute_in_qh'] = (reg_df['datetime_floor'] - reg_df['settlement_start']).dt.total_seconds() / 60

    pivot_reg = reg_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='regulation_mw', aggfunc='first'
    ).reset_index()
    pivot_reg.columns = ['datetime'] + [f'reg_min{int(c)}' for c in pivot_reg.columns[1:]]

    hist_cols = ['reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_min_10', 'reg_hist_max_10',
                 'reg_hist_range_10', 'reg_hist_trend_10', 'reg_hist_mean_20', 'reg_hist_std_20',
                 'reg_momentum', 'reg_acceleration']
    reg_min0 = reg_df[reg_df['minute_in_qh'] == 0][['settlement_start'] + hist_cols].copy()
    reg_min0 = reg_min0.rename(columns={'settlement_start': 'datetime'})

    df = pd.merge(label_df, pivot_reg, on='datetime', how='inner')
    df = pd.merge(df, reg_min0, on='datetime', how='left')

    load_df = load_df.copy()
    load_df['datetime_floor'] = load_df['datetime'].dt.floor('3min')
    load_df['settlement_end'] = load_df['datetime_floor'].dt.ceil('15min')
    mask = load_df['datetime_floor'] == load_df['settlement_end']
    load_df.loc[mask, 'settlement_end'] = load_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    load_df['settlement_start'] = load_df['settlement_end'] - pd.Timedelta(minutes=15)
    load_df['minute_in_qh'] = (load_df['datetime_floor'] - load_df['settlement_start']).dt.total_seconds() / 60

    load_pivot = load_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='load_mw', aggfunc='first'
    ).reset_index()
    load_pivot.columns = ['datetime'] + [f'load_min{int(c)}' for c in load_pivot.columns[1:]]
    df = pd.merge(df, load_pivot, on='datetime', how='left')

    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def add_proxy_lag_features(df):
    """Add proxy-based lag features."""
    df = df.sort_values('datetime').copy()

    reg_cols = ['reg_min0', 'reg_min3', 'reg_min6', 'reg_min9', 'reg_min12']
    available_cols = [c for c in reg_cols if c in df.columns]
    df['period_proxy'] = -0.25 * df[available_cols].mean(axis=1)

    df['proxy_lag1'] = df['period_proxy'].shift(1)
    df['proxy_lag2'] = df['period_proxy'].shift(2)
    df['proxy_lag3'] = df['period_proxy'].shift(3)
    df['proxy_lag4'] = df['period_proxy'].shift(4)

    df['proxy_rolling_mean4'] = df['period_proxy'].shift(1).rolling(4).mean()
    df['proxy_rolling_std4'] = df['period_proxy'].shift(1).rolling(4).std()
    df['proxy_rolling_mean10'] = df['period_proxy'].shift(1).rolling(10).mean()
    df['proxy_rolling_std10'] = df['period_proxy'].shift(1).rolling(10).std()

    df['proxy_last_sign'] = np.sign(df['period_proxy'].shift(1))
    df['proxy_last_positive'] = (df['period_proxy'].shift(1) > 0).astype(int)

    signs = np.sign(df['period_proxy'])
    sign_change = (signs != signs.shift(1)).astype(int)
    sign_change.iloc[0] = 1
    groups = sign_change.cumsum()
    df['proxy_consecutive_same_sign'] = groups.groupby(groups).cumcount().shift(1)

    positive = (df['period_proxy'] > 0).astype(float)
    df['proxy_prop_positive_4'] = positive.shift(1).rolling(4).mean()
    df['proxy_prop_positive_10'] = positive.shift(1).rolling(10).mean()

    df['proxy_momentum'] = df['period_proxy'].shift(1) - df['period_proxy'].shift(2)
    df['proxy_acceleration'] = df['proxy_momentum'] - df['proxy_momentum'].shift(1)
    df['proxy_deviation_from_mean'] = df['period_proxy'].shift(1) - df['proxy_rolling_mean10']

    short_std = df['period_proxy'].shift(1).rolling(4).std()
    long_std = df['period_proxy'].shift(1).rolling(20, min_periods=5).std()
    df['proxy_volatility_ratio'] = short_std / long_std.clip(lower=0.1)

    return df


def compute_lead_features(df, lead_time, load_expected):
    """Compute lead-time specific features."""
    result = df.copy()

    available_minutes = {12: [0], 9: [0, 3], 6: [0, 3, 6], 3: [0, 3, 6, 9], 0: [0, 3, 6, 9, 12]}
    mins = available_minutes[lead_time]
    reg_cols = [f'reg_min{m}' for m in mins]
    load_cols = [f'load_min{m}' for m in mins if f'load_min{m}' in df.columns]

    result['reg_cumulative_mean'] = df[reg_cols].mean(axis=1)

    if lead_time == 12:
        result['baseline_pred'] = -0.25 * df['reg_min0']
    elif lead_time == 9:
        result['baseline_pred'] = -0.25 * (0.8 * df['reg_min3'] + 0.2 * df['reg_min0'])
    else:
        result['baseline_pred'] = -0.25 * df[reg_cols].mean(axis=1)

    if len(load_cols) > 0:
        load_mean = df[load_cols].mean(axis=1)
        temp = df[['hour', 'is_weekend']].copy()
        temp['minute'] = mins[0]
        temp = temp.merge(load_expected.reset_index(), on=['hour', 'minute', 'is_weekend'], how='left')
        result['load_deviation'] = load_mean - temp['expected_load'].values
    else:
        result['load_deviation'] = 0

    if len(reg_cols) >= 2:
        result['reg_std'] = df[reg_cols].std(axis=1)
        result['reg_range'] = df[reg_cols].max(axis=1) - df[reg_cols].min(axis=1)
        result['reg_trend'] = df[reg_cols[-1]] - df[reg_cols[0]]
    else:
        result['reg_std'] = 0
        result['reg_range'] = 0
        result['reg_trend'] = 0

    return result


# =============================================================================
# FEATURE SUBSETS FOR DIVERSITY
# =============================================================================

def get_feature_subsets(lead_time):
    """Define different feature subsets for ensemble diversity."""

    subsets = {}

    # 1. FULL: All features (like V4)
    if lead_time == 12:
        subsets['full'] = [
            'baseline_pred', 'reg_cumulative_mean',
            'reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_trend_10',
            'reg_hist_min_10', 'reg_hist_max_10', 'reg_hist_range_10',
            'reg_hist_mean_20', 'reg_hist_std_20',
            'reg_momentum', 'reg_acceleration',
            'proxy_lag1', 'proxy_lag2', 'proxy_lag3', 'proxy_lag4',
            'proxy_rolling_mean4', 'proxy_rolling_std4',
            'proxy_rolling_mean10', 'proxy_rolling_std10',
            'proxy_last_sign', 'proxy_last_positive',
            'proxy_consecutive_same_sign', 'proxy_prop_positive_4', 'proxy_prop_positive_10',
            'proxy_momentum', 'proxy_acceleration', 'proxy_deviation_from_mean',
            'proxy_volatility_ratio',
            'hour_sin', 'hour_cos', 'is_weekend', 'dow_sin', 'dow_cos',
            'load_deviation'
        ]
    else:  # lead 9
        subsets['full'] = [
            'baseline_pred', 'reg_cumulative_mean', 'reg_std', 'reg_range', 'reg_trend',
            'reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_trend_10',
            'reg_hist_min_10', 'reg_hist_max_10', 'reg_hist_range_10',
            'reg_hist_mean_20', 'reg_hist_std_20',
            'proxy_lag1', 'proxy_lag2', 'proxy_lag3',
            'proxy_rolling_mean4', 'proxy_rolling_std4',
            'proxy_rolling_mean10', 'proxy_rolling_std10',
            'proxy_last_sign', 'proxy_last_positive', 'proxy_consecutive_same_sign',
            'proxy_momentum', 'proxy_acceleration',
            'hour_sin', 'hour_cos',
            'load_deviation'
        ]

    # 2. PROXY: Focus on proxy-based features
    subsets['proxy'] = [
        'baseline_pred',
        'proxy_lag1', 'proxy_lag2', 'proxy_lag3', 'proxy_lag4',
        'proxy_rolling_mean4', 'proxy_rolling_std4',
        'proxy_rolling_mean10', 'proxy_rolling_std10',
        'proxy_last_sign', 'proxy_momentum', 'proxy_acceleration',
        'proxy_deviation_from_mean', 'proxy_volatility_ratio',
        'proxy_prop_positive_4', 'proxy_prop_positive_10'
    ]

    # 3. HISTORICAL: Focus on historical regulation stats
    subsets['historical'] = [
        'baseline_pred', 'reg_cumulative_mean',
        'reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_trend_10',
        'reg_hist_min_10', 'reg_hist_max_10', 'reg_hist_range_10',
        'reg_hist_mean_20', 'reg_hist_std_20',
        'reg_momentum', 'reg_acceleration',
        'proxy_lag1', 'proxy_rolling_mean4'
    ]

    # 4. TIME: Focus on time-based patterns
    subsets['time'] = [
        'baseline_pred', 'reg_cumulative_mean',
        'hour_sin', 'hour_cos', 'is_weekend', 'dow_sin', 'dow_cos',
        'load_deviation',
        'proxy_lag1', 'proxy_rolling_mean4',
        'proxy_prop_positive_4'
    ]

    # 5. MINIMAL: Only most important features
    subsets['minimal'] = [
        'baseline_pred', 'reg_cumulative_mean',
        'proxy_rolling_mean4', 'proxy_lag1',
        'reg_hist_mean_20', 'reg_hist_mean_10',
        'hour_cos', 'hour_sin'
    ]

    # 6. RESIDUAL: Features to predict residual from baseline
    subsets['residual'] = [
        'proxy_lag1', 'proxy_lag2', 'proxy_lag3',
        'proxy_rolling_mean4', 'proxy_rolling_std4',
        'proxy_momentum', 'proxy_deviation_from_mean',
        'reg_hist_mean_10', 'reg_hist_std_10',
        'hour_sin', 'hour_cos',
        'load_deviation'
    ]

    return subsets


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model."""
    params = {
        'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'max_depth': 6, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_data_in_leaf': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'verbose': -1, 'seed': 42,
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[val_data], callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model."""
    params = {
        'objective': 'reg:squarederror', 'eval_metric': 'mae',
        'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'seed': 42, 'verbosity': 0
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
    return model


def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost model."""
    model = cb.CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.05,
        loss_function='MAE', random_seed=42, verbose=False,
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model


def train_ridge(X_train, y_train, X_val, y_val):
    """Train Ridge regression."""
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model


def train_lasso(X_train, y_train, X_val, y_val):
    """Train Lasso regression."""
    model = Lasso(alpha=0.1, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest."""
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=50,
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def predict_model(model, X, model_type):
    """Get predictions from a model."""
    if model_type == 'xgboost':
        return model.predict(xgb.DMatrix(X))
    else:
        return model.predict(X)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_ensemble_experiment(lead_time=12):
    """Run the full ensemble experiment for a given lead time."""

    print("=" * 80)
    print(f"ENSEMBLE EXPERIMENT - LEAD {lead_time} MINUTES")
    print("=" * 80)

    # Load and prepare data
    reg_df, load_df, label_df = load_data()
    load_expected = compute_load_expected(load_df)

    print("\nCreating features...")
    df = create_base_features(reg_df, load_df, label_df, load_expected)
    df = add_proxy_lag_features(df)
    df = compute_lead_features(df, lead_time, load_expected)

    # Train/val/test split
    train_end = pd.Timestamp('2024-10-01')
    val_end = pd.Timestamp('2025-01-01')

    train_df = df[df['datetime'] < train_end].copy()
    val_df = df[(df['datetime'] >= train_end) & (df['datetime'] < val_end)].copy()
    test_df = df[df['datetime'] >= val_end].copy()

    print(f"\nData splits:")
    print(f"  Train: {len(train_df):,} samples (< {train_end.date()})")
    print(f"  Val:   {len(val_df):,} samples ({train_end.date()} to {val_end.date()})")
    print(f"  Test:  {len(test_df):,} samples (>= {val_end.date()})")

    # Get feature subsets
    feature_subsets = get_feature_subsets(lead_time)

    # Define model types to try
    model_types = []
    if HAS_LIGHTGBM:
        model_types.append(('lgb', train_lightgbm))
    if HAS_XGBOOST:
        model_types.append(('xgb', train_xgboost))
    if HAS_CATBOOST:
        model_types.append(('cat', train_catboost))
    model_types.extend([
        ('ridge', train_ridge),
        ('rf', train_random_forest),
    ])

    print(f"\nModel types: {[m[0] for m in model_types]}")
    print(f"Feature subsets: {list(feature_subsets.keys())}")

    # Train all model combinations
    results = []
    predictions = {}

    print("\n" + "-" * 80)
    print("TRAINING MODELS")
    print("-" * 80)

    for subset_name, feature_list in feature_subsets.items():
        # Filter to available features
        available_features = [f for f in feature_list if f in df.columns]

        # Prepare data
        train_clean = train_df.dropna(subset=available_features + ['imbalance'])
        val_clean = val_df.dropna(subset=available_features + ['imbalance'])
        test_clean = test_df.dropna(subset=available_features + ['imbalance'])

        X_train = train_clean[available_features].values
        y_train = train_clean['imbalance'].values
        X_val = val_clean[available_features].values
        y_val = val_clean['imbalance'].values
        X_test = test_clean[available_features].values
        y_test = test_clean['imbalance'].values
        baseline = test_clean['baseline_pred'].values

        for model_name, train_func in model_types:
            model_id = f"{model_name}_{subset_name}"

            try:
                # Train model
                model = train_func(X_train, y_train, X_val, y_val)

                # Predict on test
                y_pred = predict_model(model, X_test, model_name)

                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                dir_acc = np.mean(np.sign(y_test) == np.sign(y_pred)) * 100

                # Store results
                results.append({
                    'model_id': model_id,
                    'model_type': model_name,
                    'feature_subset': subset_name,
                    'n_features': len(available_features),
                    'mae': mae,
                    'dir_acc': dir_acc
                })

                # Store predictions for correlation analysis
                predictions[model_id] = {
                    'y_pred': y_pred,
                    'y_test': y_test,
                    'baseline': baseline,
                    'errors': y_test - y_pred
                }

                print(f"  {model_id:<25} MAE: {mae:.3f}  Dir.Acc: {dir_acc:.1f}%")

            except Exception as e:
                print(f"  {model_id:<25} FAILED: {str(e)[:50]}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results).sort_values('mae')

    # Baseline performance
    baseline_mae = mean_absolute_error(y_test, baseline)
    baseline_dir = np.mean(np.sign(y_test) == np.sign(baseline)) * 100

    print("\n" + "=" * 80)
    print("INDIVIDUAL MODEL RESULTS (sorted by MAE)")
    print("=" * 80)
    print(f"\nBaseline: MAE={baseline_mae:.3f}, Dir.Acc={baseline_dir:.1f}%")
    print(f"\n{'Model':<25} {'Type':<8} {'Subset':<12} {'Feats':<6} {'MAE':<8} {'vs Base':<10} {'Dir%':<8}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        vs_base = (1 - row['mae'] / baseline_mae) * 100
        print(f"{row['model_id']:<25} {row['model_type']:<8} {row['feature_subset']:<12} "
              f"{row['n_features']:<6} {row['mae']:<8.3f} {vs_base:>+8.1f}%  {row['dir_acc']:<8.1f}")

    # ==========================================================================
    # ERROR CORRELATION ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("ERROR CORRELATION MATRIX")
    print("=" * 80)
    print("\nLow correlation (<0.7) = good diversity for ensemble")
    print("High correlation (>0.9) = low diversity, ensemble won't help much\n")

    # Build error matrix
    model_ids = list(predictions.keys())
    n_models = len(model_ids)
    error_matrix = np.column_stack([predictions[m]['errors'] for m in model_ids])

    # Compute correlation matrix
    corr_matrix = np.corrcoef(error_matrix.T)

    # Create correlation DataFrame
    corr_df = pd.DataFrame(corr_matrix, index=model_ids, columns=model_ids)

    # Find most diverse pairs
    diverse_pairs = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            diverse_pairs.append({
                'model1': model_ids[i],
                'model2': model_ids[j],
                'error_corr': corr_matrix[i, j]
            })

    diverse_df = pd.DataFrame(diverse_pairs).sort_values('error_corr')

    print("Most DIVERSE pairs (lowest error correlation):")
    print("-" * 60)
    for _, row in diverse_df.head(10).iterrows():
        print(f"  {row['model1']:<20} + {row['model2']:<20} corr={row['error_corr']:.3f}")

    print("\nLeast diverse pairs (highest error correlation):")
    print("-" * 60)
    for _, row in diverse_df.tail(5).iterrows():
        print(f"  {row['model1']:<20} + {row['model2']:<20} corr={row['error_corr']:.3f}")

    # ==========================================================================
    # ENSEMBLE COMBINATIONS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("ENSEMBLE COMBINATIONS")
    print("=" * 80)

    ensemble_results = []

    # Get best single model
    best_single = results_df.iloc[0]
    best_single_pred = predictions[best_single['model_id']]['y_pred']

    # 1. Simple average of all models
    all_preds = np.column_stack([predictions[m]['y_pred'] for m in model_ids])
    avg_all = all_preds.mean(axis=1)
    mae_avg_all = mean_absolute_error(y_test, avg_all)
    dir_avg_all = np.mean(np.sign(y_test) == np.sign(avg_all)) * 100
    ensemble_results.append(('Average (all)', mae_avg_all, dir_avg_all, len(model_ids)))

    # 2. Average of top 5 models
    top5_ids = results_df.head(5)['model_id'].tolist()
    top5_preds = np.column_stack([predictions[m]['y_pred'] for m in top5_ids])
    avg_top5 = top5_preds.mean(axis=1)
    mae_top5 = mean_absolute_error(y_test, avg_top5)
    dir_top5 = np.mean(np.sign(y_test) == np.sign(avg_top5)) * 100
    ensemble_results.append(('Average (top 5)', mae_top5, dir_top5, 5))

    # 3. Average of top 3 models
    top3_ids = results_df.head(3)['model_id'].tolist()
    top3_preds = np.column_stack([predictions[m]['y_pred'] for m in top3_ids])
    avg_top3 = top3_preds.mean(axis=1)
    mae_top3 = mean_absolute_error(y_test, avg_top3)
    dir_top3 = np.mean(np.sign(y_test) == np.sign(avg_top3)) * 100
    ensemble_results.append(('Average (top 3)', mae_top3, dir_top3, 3))

    # 4. Weighted average by validation MAE (inverse weighting)
    weights = 1.0 / results_df['mae'].values
    weights = weights / weights.sum()
    weighted_preds = np.average(all_preds, axis=1, weights=weights)
    mae_weighted = mean_absolute_error(y_test, weighted_preds)
    dir_weighted = np.mean(np.sign(y_test) == np.sign(weighted_preds)) * 100
    ensemble_results.append(('Weighted (1/MAE)', mae_weighted, dir_weighted, len(model_ids)))

    # 5. Average of most diverse pairs (corr < 0.7)
    diverse_models = set()
    for _, row in diverse_df.iterrows():
        if row['error_corr'] < 0.7:
            diverse_models.add(row['model1'])
            diverse_models.add(row['model2'])
    diverse_models = list(diverse_models)[:6]  # Limit to 6

    if len(diverse_models) >= 2:
        diverse_preds = np.column_stack([predictions[m]['y_pred'] for m in diverse_models])
        avg_diverse = diverse_preds.mean(axis=1)
        mae_diverse = mean_absolute_error(y_test, avg_diverse)
        dir_diverse = np.mean(np.sign(y_test) == np.sign(avg_diverse)) * 100
        ensemble_results.append(('Average (diverse)', mae_diverse, dir_diverse, len(diverse_models)))

    # 6. Best model type per feature subset
    best_per_subset = results_df.groupby('feature_subset').first().reset_index()
    bps_ids = [f"{row['model_type']}_{row['feature_subset']}" for _, row in best_per_subset.iterrows()]
    bps_preds = np.column_stack([predictions[m]['y_pred'] for m in bps_ids])
    avg_bps = bps_preds.mean(axis=1)
    mae_bps = mean_absolute_error(y_test, avg_bps)
    dir_bps = np.mean(np.sign(y_test) == np.sign(avg_bps)) * 100
    ensemble_results.append(('Best per subset', mae_bps, dir_bps, len(bps_ids)))

    # 7. Different model types only (one per type, best subset)
    best_per_type = results_df.groupby('model_type').first().reset_index()
    bpt_ids = [f"{row['model_type']}_{row['feature_subset']}" for _, row in best_per_type.iterrows()]
    bpt_preds = np.column_stack([predictions[m]['y_pred'] for m in bpt_ids])
    avg_bpt = bpt_preds.mean(axis=1)
    mae_bpt = mean_absolute_error(y_test, avg_bpt)
    dir_bpt = np.mean(np.sign(y_test) == np.sign(avg_bpt)) * 100
    ensemble_results.append(('Best per model type', mae_bpt, dir_bpt, len(bpt_ids)))

    # Print ensemble results
    print(f"\n{'Ensemble':<25} {'MAE':<10} {'vs Best Single':<15} {'vs Baseline':<12} {'Dir%':<8} {'Models':<8}")
    print("-" * 85)
    print(f"{'Baseline':<25} {baseline_mae:<10.3f} {'':<15} {'':<12} {baseline_dir:<8.1f} {1:<8}")
    print(f"{'Best Single':<25} {best_single['mae']:<10.3f} {'':<15} "
          f"{(1-best_single['mae']/baseline_mae)*100:>+10.1f}%  {best_single['dir_acc']:<8.1f} {1:<8}")
    print("-" * 85)

    for name, mae, dir_acc, n_models in ensemble_results:
        vs_best = (1 - mae / best_single['mae']) * 100
        vs_base = (1 - mae / baseline_mae) * 100
        marker = " ***" if mae < best_single['mae'] else ""
        print(f"{name:<25} {mae:<10.3f} {vs_best:>+13.2f}%  {vs_base:>+10.1f}%  {dir_acc:<8.1f} {n_models:<8}{marker}")

    # ==========================================================================
    # SAVE RESULTS AND PLOTS
    # ==========================================================================

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'outputs' / f'ensemble_results_lead{lead_time}.csv', index=False)
    diverse_df.to_csv(OUTPUT_DIR / 'outputs' / f'error_correlations_lead{lead_time}.csv', index=False)

    # Plot error correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels([m.split('_')[0] + '\n' + m.split('_')[1][:4] for m in model_ids], fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels([m.split('_')[0] + '\n' + m.split('_')[1][:4] for m in model_ids], fontsize=8)
    plt.colorbar(im, ax=ax, label='Error Correlation')
    ax.set_title(f'Error Correlation Matrix - Lead {lead_time} min\n(Lower = more diverse)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plots' / f'error_correlation_lead{lead_time}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot MAE comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(results_df))
    colors = ['steelblue' if t == 'lgb' else 'orange' if t == 'xgb' else 'green' if t == 'cat'
              else 'purple' if t == 'ridge' else 'brown' for t in results_df['model_type']]
    ax.bar(x, results_df['mae'], color=colors, alpha=0.7)
    ax.axhline(baseline_mae, color='red', linestyle='--', label=f'Baseline ({baseline_mae:.3f})')
    ax.axhline(best_single['mae'], color='green', linestyle=':', label=f'Best Single ({best_single["mae"]:.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model_id'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('MAE (MWh)')
    ax.set_title(f'Model Comparison - Lead {lead_time} min', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plots' / f'model_comparison_lead{lead_time}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to: {OUTPUT_DIR / 'outputs'}")
    print(f"Plots saved to: {OUTPUT_DIR / 'plots'}")

    return results_df, diverse_df, ensemble_results


# =============================================================================
# RUN EXPERIMENT
# =============================================================================

if __name__ == '__main__':
    # Run for lead 12
    results_12, corr_12, ens_12 = run_ensemble_experiment(lead_time=12)

    print("\n\n")

    # Run for lead 9
    results_9, corr_9, ens_9 = run_ensemble_experiment(lead_time=9)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
