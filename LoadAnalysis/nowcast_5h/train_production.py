"""
Production Training: Two-Stage Load Nowcast Models
===================================================

Trains 40 models (4 quarters x 5 horizons x 2 stages) using all available data.

Architecture:
  Stage 1: Predicts DAMAS load forecast error from error history + time features
           + quarterly extrapolation features (progressive intra-hour 3-min data)
  Stage 2: Corrects Stage 1 residuals using lagged residual features

OOF approach (no data leakage):
  - Walk-forward expanding window generates out-of-fold S1 predictions
  - Stage 2 trains on OOF residuals only
  - Final Stage 1 retrained on ALL data for production inference

Uses tuned hyperparameters from tuning/h{horizon}/ directory.

Outputs (to deployment/models/):
  - s1_q{0-3}_h{1-5}.joblib  (20 Stage 1 models)
  - s2_q{0-3}_h{1-5}.joblib  (20 Stage 2 models)
  - feature_configs.json      (per-model feature lists + seasonal means)
  - seasonal_bias.json        (extrapolation bias correction)
  - training_metadata.json

Usage:
  python train_production.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent          # nowcast_5h/
BASE_PATH = SCRIPT_DIR.parent.parent                  # ipesoft_eda_data/
TUNING_PATH = SCRIPT_DIR / 'tuning'
OUTPUT_PATH = SCRIPT_DIR / 'deployment' / 'models'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 3, 4, 5]
QUARTERS = [0, 1, 2, 3]

# Walk-forward folds for OOF predictions: (train_end, pred_start, pred_end)
# Training always starts from data beginning (~2024-01)
OOF_FOLDS = [
    ('2025-01-01', '2025-01-01', '2025-07-01'),
    ('2025-07-01', '2025-07-01', '2025-10-01'),
    ('2025-10-01', '2025-10-01', '2026-01-01'),
    ('2026-01-01', '2026-01-01', '2026-02-01'),
    ('2026-02-01', '2026-02-01', '2026-03-01'),
    ('2026-03-01', '2026-03-01', '2026-04-01'),
    ('2026-04-01', '2026-04-01', '2026-05-01'),
]

# Quarter-specific extrapolation features (cumulative per quarter)
Q_EXTRAP_FEATURES = {
    0: [],
    1: ['extrap_h1_error_q1', 'trend_q1', 'vol_q1'],
    2: ['extrap_h1_error_q1', 'trend_q1', 'vol_q1',
        'extrap_h1_error_q2', 'trend_q2', 'vol_q2',
        'delta_est_q1_q2', 'trend_change_q1_q2'],
    3: ['extrap_h1_error_q1', 'trend_q1', 'vol_q1',
        'extrap_h1_error_q2', 'trend_q2', 'vol_q2',
        'delta_est_q1_q2', 'trend_change_q1_q2',
        'extrap_h1_error_q3', 'trend_q3', 'vol_q3',
        'delta_est_q2_q3', 'momentum_q'],
}


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load all data sources."""
    print("[*] Loading data...")

    # Hourly DAMAS load
    df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['year'] = df['datetime'].dt.year
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']
    print(f"[+] Hourly load: {len(df):,} rows, "
          f"{df['datetime'].min()} to {df['datetime'].max()}")

    # 3-minute load
    df_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'load_3min.csv')
    df_3min['datetime'] = pd.to_datetime(df_3min['datetime'])
    df_3min = df_3min.sort_values('datetime').reset_index(drop=True)
    df_3min['hour_start'] = df_3min['datetime'].dt.floor('h')
    print(f"[+] 3-min load: {len(df_3min):,} rows")

    # Hourly aggregates from 3-min (for load_volatility_lag1, load_trend_lag1)
    load_hourly = df_3min.groupby('hour_start').agg(
        load_std_3min=('load_mw', 'std'),
        load_first=('load_mw', 'first'),
        load_last=('load_mw', 'last'),
    ).reset_index().rename(columns={'hour_start': 'datetime'})
    load_hourly['load_trend_3min'] = load_hourly['load_last'] - load_hourly['load_first']
    df = df.merge(load_hourly[['datetime', 'load_std_3min', 'load_trend_3min']],
                  on='datetime', how='left')

    # Regulation (used by H+5)
    reg_path = BASE_PATH / 'data' / 'features' / 'regulation_3min.csv'
    if reg_path.exists():
        reg_3min = pd.read_csv(reg_path, parse_dates=['datetime'])
        reg_3min['hour_start'] = reg_3min['datetime'].dt.floor('h')
        reg_hourly = reg_3min.groupby('hour_start').agg(
            reg_mean=('regulation_mw', 'mean'),
            reg_std=('regulation_mw', 'std'),
        ).reset_index().rename(columns={'hour_start': 'datetime'})
        df = df.merge(reg_hourly, on='datetime', how='left')
        print(f"[+] Regulation: {len(reg_hourly):,} hourly periods")

    return df, df_3min


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_base_features(df):
    """Create all base Stage 1 features (everything except seasonal_error).

    These features do not depend on the train/test split, so they are
    computed once and reused across all OOF folds.
    """
    df = df.copy()

    # --- Error lags ---
    for lag in range(1, 9):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    # --- Rolling statistics ---
    for window in [3, 6, 12, 24]:
        df[f'error_roll_mean_{window}h'] = df['error'].shift(1).rolling(window).mean()
        df[f'error_roll_std_{window}h'] = df['error'].shift(1).rolling(window).std()

    # --- Trends and momentum ---
    df['error_trend_3h'] = df['error_lag1'] - df['error_lag3']
    df['error_trend_6h'] = df['error_lag1'] - df['error_lag6']
    df['error_momentum'] = (0.5 * (df['error_lag1'] - df['error_lag2']) +
                            0.3 * (df['error_lag2'] - df['error_lag3']) +
                            0.2 * (df['error_lag3'] - df['error_lag4']))

    # --- 3-min derived ---
    df['load_volatility_lag1'] = df['load_std_3min'].shift(1)
    df['load_trend_lag1'] = df['load_trend_3min'].shift(1)

    # --- Regulation (only H+5 uses these) ---
    if 'reg_mean' in df.columns:
        for lag in range(1, 4):
            df[f'reg_mean_lag{lag}'] = df['reg_mean'].shift(lag)
        df['reg_std_lag1'] = df['reg_std'].shift(1)

    # --- Time ---
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)

    # --- Targets ---
    for h in range(1, 6):
        df[f'target_h{h}'] = df['error'].shift(-h)

    return df


def compute_seasonal_means(df, mask):
    """Compute (dow, hour) -> mean error from masked rows."""
    return df[mask].groupby(['dow', 'hour'])['error'].mean().to_dict()


def apply_seasonal_error(df, seasonal_means):
    """Set the seasonal_error column from a precomputed lookup."""
    df['seasonal_error'] = [seasonal_means.get((d, h), 0)
                            for d, h in zip(df['dow'], df['hour'])]


def compute_seasonal_bias(df, df_3min, train_mask):
    """Compute hour-specific extrapolation bias (vectorized).

    For each (minutes_elapsed, hour_of_day), measures the systematic
    difference: actual_h1_load - mean(partial_3min_readings).
    """
    print("[*] Computing seasonal extrapolation bias...")

    train = df[train_mask].copy()
    train['h1_hour'] = train['datetime'] + pd.Timedelta(hours=1)

    # Lookup: hour -> actual load (dedup in case of duplicate timestamps)
    actual_lookup = df.drop_duplicates(subset='datetime', keep='last').set_index('datetime')['actual_load_mw']
    train['h1_actual'] = train['h1_hour'].map(actual_lookup)
    train = train.dropna(subset=['h1_actual'])

    # Pre-compute minutes into hour for 3-min data
    df_3min = df_3min.copy()
    df_3min['min_into_hour'] = (
        (df_3min['datetime'] - df_3min['hour_start']).dt.total_seconds() / 60
    )

    bias = {}
    for minutes in [15, 30, 45]:
        partial = df_3min[df_3min['min_into_hour'] < minutes]
        partial_means = partial.groupby('hour_start')['load_mw'].mean()

        train_with_partial = train.copy()
        train_with_partial['partial_mean'] = train_with_partial['h1_hour'].map(partial_means)
        valid = train_with_partial.dropna(subset=['partial_mean'])
        valid['bias_error'] = valid['h1_actual'] - valid['partial_mean']

        hour_bias = valid.groupby('hour')['bias_error'].mean()
        bias[str(minutes)] = {str(int(h)): float(v) for h, v in hour_bias.items()}
        print(f"    {minutes}min: {len(hour_bias)} hours, "
              f"mean bias {hour_bias.mean():+.1f} MW")

    return bias


def compute_extrapolation_features(df, df_3min, seasonal_bias):
    """Add quarterly extrapolation features (vectorized)."""
    print("[*] Computing quarterly extrapolation features...")

    df = df.copy()
    df['h1_forecast'] = df['forecast_load_mw'].shift(-1)
    df['h1_hour'] = df['datetime'] + pd.Timedelta(hours=1)
    df['h1_hod'] = (df['hour'] + 1) % 24

    # Pre-compute minutes into hour
    df_3min = df_3min.copy()
    df_3min['min_into_hour'] = (
        (df_3min['datetime'] - df_3min['hour_start']).dt.total_seconds() / 60
    )

    for q_min, q_label in [(15, 'q1'), (30, 'q2'), (45, 'q3')]:
        q_data = df_3min[df_3min['min_into_hour'] < q_min]

        # Mean, std per hour
        agg = q_data.groupby('hour_start')['load_mw'].agg(['mean', 'std'])
        agg.columns = [f'__{q_label}_mean', f'__{q_label}_std']
        agg[f'__{q_label}_std'] = agg[f'__{q_label}_std'].fillna(0)

        # Trend (polyfit slope) per hour
        def _trend(vals):
            if len(vals) < 2:
                return 0.0
            return np.polyfit(range(len(vals)), vals.values, 1)[0]

        agg[f'__{q_label}_trend'] = q_data.groupby('hour_start')['load_mw'].apply(_trend)

        # Join on h1_hour (partial data is for H+1)
        df = df.merge(agg, left_on='h1_hour', right_index=True, how='left')

        # Apply seasonal bias
        bias_map = {int(k): v for k, v in seasonal_bias[str(q_min)].items()}
        bias_values = df['h1_hod'].map(bias_map).fillna(0)

        df[f'extrap_h1_error_{q_label}'] = (
            df[f'__{q_label}_mean'] + bias_values - df['h1_forecast']
        )
        df[f'trend_{q_label}'] = df[f'__{q_label}_trend']
        df[f'vol_{q_label}'] = df[f'__{q_label}_std']

    # Delta and momentum features
    df['delta_est_q1_q2'] = df['extrap_h1_error_q2'] - df['extrap_h1_error_q1']
    df['trend_change_q1_q2'] = df['trend_q2'] - df['trend_q1']
    df['delta_est_q2_q3'] = df['extrap_h1_error_q3'] - df['extrap_h1_error_q2']
    df['momentum_q'] = df['delta_est_q2_q3'] - df['delta_est_q1_q2']

    # Drop temp columns
    for q_label in ['q1', 'q2', 'q3']:
        df.drop(columns=[f'__{q_label}_mean', f'__{q_label}_std',
                         f'__{q_label}_trend'], inplace=True, errors='ignore')
    df.drop(columns=['h1_forecast', 'h1_hour', 'h1_hod'], inplace=True, errors='ignore')

    print(f"[+] Extrapolation features added for {len(df):,} hours")
    return df


# ============================================================
# TUNED PARAMETERS
# ============================================================

def load_tuned_params(horizon):
    """Load Optuna-tuned parameters for a given horizon."""
    h_dir = TUNING_PATH / f'h{horizon}'
    with open(h_dir / 'stage1_best_params.json') as f:
        s1_best = json.load(f)
    with open(h_dir / 'stage2_best_params.json') as f:
        s2_best = json.load(f)
    return s1_best, s2_best


def extract_s1_lgb_params(params):
    """Extract LightGBM hyperparameters for Stage 1."""
    keys = ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves',
            'min_child_samples', 'subsample', 'colsample_bytree',
            'reg_alpha', 'reg_lambda']
    p = {k: params[k] for k in keys if k in params}
    p['random_state'] = 42
    p['verbosity'] = -1
    return p


def extract_s2_lgb_params(s2_params):
    """Extract LightGBM hyperparameters for Stage 2."""
    return {
        'n_estimators': s2_params.get('s2_n_estimators', 200),
        'learning_rate': s2_params.get('s2_learning_rate', 0.05),
        'max_depth': s2_params.get('s2_max_depth', 5),
        'num_leaves': s2_params.get('s2_num_leaves', 31),
        'min_child_samples': s2_params.get('s2_min_child_samples', 30),
        'subsample': s2_params.get('s2_subsample', 0.8),
        'colsample_bytree': s2_params.get('s2_colsample_bytree', 0.8),
        'reg_alpha': s2_params.get('s2_reg_alpha', 0.1),
        'reg_lambda': s2_params.get('s2_reg_lambda', 1.0),
        'random_state': 42,
        'verbosity': -1,
    }


# ============================================================
# RESIDUAL FEATURES
# ============================================================

def create_residual_features(df, horizon):
    """Create Stage 2 residual lag features on a DataFrame with 'residual' column."""
    df = df.copy()
    for lag in range(1, 7):
        df[f'residual_lag{lag}'] = df['residual'].shift(horizon + lag - 1)
    return df


# ============================================================
# SAMPLE WEIGHTS
# ============================================================

def get_sample_weights(dates, cutoff_date, recency_months, recency_weight):
    """Compute recency-biased sample weights."""
    cutoff = pd.to_datetime(cutoff_date)
    days_ago = (cutoff - dates).dt.days.clip(lower=0)
    months_ago = days_ago / 30.0
    weights = np.where(months_ago <= recency_months, recency_weight, 1.0)
    return weights / weights.mean()


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():
    print("=" * 70)
    print("PRODUCTION TRAINING: TWO-STAGE LOAD NOWCAST")
    print("=" * 70)

    # --- Load data ---
    df, df_3min = load_data()

    # --- Base features (no seasonal_error yet) ---
    print("\n[*] Creating base features...")
    df_feat = create_base_features(df)

    # --- Seasonal bias from 2024 (stable measurement pattern) ---
    bias_mask = df_feat['year'] == 2024
    seasonal_bias = compute_seasonal_bias(df_feat, df_3min, bias_mask)

    # --- Quarterly extrapolation features ---
    df_feat = compute_extrapolation_features(df_feat, df_3min, seasonal_bias)

    # --- Pre-compute seasonal means for each fold + all data ---
    print("[*] Pre-computing seasonal means per fold...")
    fold_seasonal_cache = {}
    for train_end, _, _ in OOF_FOLDS:
        mask = df_feat['datetime'] < train_end
        if mask.sum() > 0:
            fold_seasonal_cache[train_end] = compute_seasonal_means(df_feat, mask)
    all_seasonal_means = compute_seasonal_means(
        df_feat, pd.Series(True, index=df_feat.index))

    # --- Apply all-data seasonal error as default (will be swapped per fold in OOF) ---
    apply_seasonal_error(df_feat, all_seasonal_means)

    # --- Feature configs accumulator ---
    feature_configs = {}

    # --- Format seasonal means for export (dow_hour string keys) ---
    seasonal_means_export = {f"{d}_{h}": float(v)
                             for (d, h), v in all_seasonal_means.items()}

    print("\n" + "=" * 70)
    print("TRAINING 40 PRODUCTION MODELS")
    print("=" * 70)

    for horizon in HORIZONS:
        print(f"\n{'='*60}")
        print(f"HORIZON H+{horizon}")
        print(f"{'='*60}")

        s1_best, s2_best = load_tuned_params(horizon)
        s1_base_features = s1_best['features']
        s1_params = s1_best['params']
        s2_feature_names = s2_best['s2_features']
        s2_params = s2_best['s2_params']

        s1_lgb = extract_s1_lgb_params(s1_params)
        s2_lgb = extract_s2_lgb_params(s2_params)
        target = f'target_h{horizon}'

        for quarter in QUARTERS:
            s1_features = s1_base_features + Q_EXTRAP_FEATURES[quarter]

            # Verify features exist
            missing = [f for f in s1_features if f not in df_feat.columns]
            if missing:
                print(f"  [!] Q{quarter} H+{horizon}: missing {missing}")
                continue

            print(f"\n  --- Q{quarter} H+{horizon} "
                  f"({len(s1_features)} S1 / {len(s2_feature_names)} S2 features) ---")

            # ==================================================
            # WALK-FORWARD OOF FOR STAGE 1
            # ==================================================
            oof_preds = pd.Series(np.nan, index=df_feat.index)

            for train_end, pred_start, pred_end in OOF_FOLDS:
                if train_end not in fold_seasonal_cache:
                    continue

                # Apply fold-specific seasonal error
                apply_seasonal_error(df_feat, fold_seasonal_cache[train_end])

                fold_train_mask = df_feat['datetime'] < train_end
                fold_pred_mask = ((df_feat['datetime'] >= pred_start) &
                                  (df_feat['datetime'] < pred_end))

                fold_train = df_feat[fold_train_mask].dropna(
                    subset=[target] + s1_features)
                fold_pred = df_feat[fold_pred_mask].dropna(subset=s1_features)

                if len(fold_train) < 100 or len(fold_pred) == 0:
                    continue

                weights = get_sample_weights(
                    fold_train['datetime'], train_end,
                    s1_params.get('recency_months', 3),
                    s1_params.get('recency_weight', 2.0))

                model = lgb.LGBMRegressor(**s1_lgb)
                model.fit(fold_train[s1_features].values,
                          fold_train[target].values,
                          sample_weight=weights)

                oof_preds.loc[fold_pred.index] = model.predict(
                    fold_pred[s1_features].values)

            has_oof = oof_preds.notna()
            n_oof = has_oof.sum()
            print(f"    OOF: {n_oof} predictions across {len(OOF_FOLDS)} folds")

            # ==================================================
            # TRAIN STAGE 2 ON OOF RESIDUALS
            # ==================================================
            df_s2 = df_feat[has_oof].copy()
            df_s2['residual'] = df_s2[target] - oof_preds[has_oof].values
            df_s2 = create_residual_features(df_s2, horizon)

            s2_avail = [f for f in s2_feature_names if f in df_s2.columns]
            s2_train = df_s2.dropna(subset=s2_avail + [target])

            if len(s2_train) >= 50:
                s2_weights = get_sample_weights(
                    s2_train['datetime'], s2_train['datetime'].max(),
                    s2_params.get('s2_recency_months', 2),
                    s2_params.get('s2_recency_weight', 1.5))

                s2_model = lgb.LGBMRegressor(**s2_lgb)
                s2_model.fit(s2_train[s2_avail].values,
                             s2_train['residual'].values,
                             sample_weight=s2_weights)

                s2_resid_mae = np.abs(s2_train['residual']).mean()
                s2_corrected = np.abs(
                    s2_train['residual'] - s2_model.predict(
                        s2_train[s2_avail].values)).mean()
                print(f"    S2: {len(s2_train)} rows, "
                      f"residual MAE {s2_resid_mae:.1f} -> {s2_corrected:.1f}")
            else:
                print(f"    S2: only {len(s2_train)} rows, "
                      f"training minimal fallback model")
                s2_model = lgb.LGBMRegressor(
                    n_estimators=10, verbosity=-1, random_state=42)
                dummy_X = np.zeros((100, len(s2_avail)))
                s2_model.fit(dummy_X, np.zeros(100))

            # ==================================================
            # TRAIN FINAL STAGE 1 ON ALL DATA
            # ==================================================
            apply_seasonal_error(df_feat, all_seasonal_means)

            all_train = df_feat.dropna(subset=[target] + s1_features)
            weights = get_sample_weights(
                all_train['datetime'], all_train['datetime'].max(),
                s1_params.get('recency_months', 3),
                s1_params.get('recency_weight', 2.0))

            s1_model = lgb.LGBMRegressor(**s1_lgb)
            s1_model.fit(all_train[s1_features].values,
                         all_train[target].values,
                         sample_weight=weights)

            s1_mae = np.abs(
                all_train[target] - s1_model.predict(
                    all_train[s1_features].values)).mean()
            print(f"    S1 final: {len(all_train)} rows, "
                  f"in-sample MAE {s1_mae:.1f} MW")

            # ==================================================
            # SAVE
            # ==================================================
            joblib.dump(s1_model, OUTPUT_PATH / f's1_q{quarter}_h{horizon}.joblib')
            joblib.dump(s2_model, OUTPUT_PATH / f's2_q{quarter}_h{horizon}.joblib')

            feature_configs[f'q{quarter}_h{horizon}'] = {
                's1_features': s1_features,
                's2_features': s2_avail,
                'seasonal_means': seasonal_means_export,
            }

    # ==================================================
    # EXPORT CONFIGS
    # ==================================================
    print("\n" + "=" * 70)
    print("SAVING DEPLOYMENT ARTIFACTS")
    print("=" * 70)

    with open(OUTPUT_PATH / 'feature_configs.json', 'w') as f:
        json.dump(feature_configs, f, indent=2)
    print(f"[+] {OUTPUT_PATH / 'feature_configs.json'}")

    with open(OUTPUT_PATH / 'seasonal_bias.json', 'w') as f:
        json.dump(seasonal_bias, f, indent=2)
    print(f"[+] {OUTPUT_PATH / 'seasonal_bias.json'}")

    metadata = {
        'train_start': str(df_feat['datetime'].min()),
        'train_end': str(df_feat['datetime'].max()),
        'trained_at': datetime.now().isoformat(),
        'n_models': len(feature_configs) * 2,
        'quarters': QUARTERS,
        'horizons': HORIZONS,
        'oof_folds': OOF_FOLDS,
    }
    with open(OUTPUT_PATH / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] {OUTPUT_PATH / 'training_metadata.json'}")

    print(f"\n[+] Done. {len(feature_configs)} model pairs saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
