"""
2-Hour-Ahead Imbalance Predictor
================================

Predicts system imbalance (MWh) for each 15-min settlement period,
2 hours before delivery. Uses LightGBM quantile regression to output
prediction + confidence intervals.

Features available 2h before delivery:
  - Regulation proxy history (imb ~ -0.25 * regulation)
  - Load patterns and deviations
  - Solar surprise (daytime)
  - Time features (hour, day-of-week, season)
  - Regime indicators

Target: System Imbalance (MWh) per 15-min settlement period
Lead: 8 settlement periods = 2 hours

Output: Point prediction (median) + P10/P90 quantiles for confidence.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
MODEL_DIR = BASE_DIR / "models"
REPO_ROOT = SCRIPT_DIR.parents[2]

plt.rcParams.update({
    "figure.figsize": (16, 10), "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})

LEAD_PERIODS = 8  # 8 x 15min = 2 hours
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
TRAIN_END = '2025-09-30'
TEST_START = '2025-10-01'


# ============================================================
# DATA LOADING
# ============================================================

def load_regulation():
    """Load 3-min regulation data and resample to 15-min."""
    print("[*] Loading regulation data...")
    path = REPO_ROOT / "data" / "features" / "regulation_3min.csv"
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()

    # Resample to 15-min: mean regulation in each 15-min window
    reg_15 = df['regulation_mw'].resample('15min').agg(['mean', 'std', 'min', 'max', 'count'])
    reg_15.columns = ['reg_mean', 'reg_std', 'reg_min', 'reg_max', 'reg_count']
    reg_15 = reg_15[reg_15['reg_count'] >= 2]  # need at least 2 obs per 15min

    # Imbalance proxy
    reg_15['proxy'] = -0.25 * reg_15['reg_mean']

    print(f"[+] Regulation: {len(reg_15)} periods, {reg_15.index.min()} to {reg_15.index.max()}")
    return reg_15


def load_load_data():
    """Load 3-min load data and resample to 15-min."""
    print("[*] Loading load data...")
    path = REPO_ROOT / "data" / "features" / "load_3min.csv"
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()

    load_15 = df['load_mw'].resample('15min').agg(['mean', 'std'])
    load_15.columns = ['load_mean', 'load_std']

    print(f"[+] Load: {len(load_15)} periods")
    return load_15


def load_solar():
    """Load hourly solar data, broadcast to 15-min."""
    print("[*] Loading solar data...")
    path = REPO_ROOT / "data" / "clean" / "solar" / "solar_hourly.csv"
    if not path.exists():
        print("[-] Solar data not found, skipping")
        return None
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()
    # Broadcast hourly to 15-min
    solar_15 = df.resample('15min').ffill()
    print(f"[+] Solar: {len(solar_15)} periods")
    return solar_15


def load_imbalance():
    """Load 15-min imbalance labels."""
    print("[*] Loading imbalance labels...")
    path = REPO_ROOT / "data" / "master" / "master_imbalance_data.csv"
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()
    df = df.rename(columns={'System Imbalance (MWh)': 'imbalance_mwh',
                            'Imbalance Settlement Price (EUR/MWh)': 'imb_price'})
    print(f"[+] Imbalance: {len(df)} periods, {df.index.min()} to {df.index.max()}")
    return df[['imbalance_mwh', 'imb_price']]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(reg, load, solar, imb):
    """
    Build feature matrix. Each row = one 15-min settlement period.
    All features use only data available LEAD_PERIODS (8) periods before.
    """
    print("[*] Building features...")

    # Merge all 15-min data
    df = reg.join(load, how='inner')
    if solar is not None:
        df = df.join(solar, how='left')
    df = df.join(imb, how='inner')

    # ---- PROXY FEATURES (shifted by lead) ----
    # At prediction time (8 periods before), we know proxy up to period T-8
    for lag in range(LEAD_PERIODS, LEAD_PERIODS + 16):  # lags 8-23 (2h to 6h back)
        df[f'proxy_lag{lag}'] = df['proxy'].shift(lag)

    # Rolling stats from what's known at prediction time
    proxy_shifted = df['proxy'].shift(LEAD_PERIODS)
    df['proxy_rmean4'] = proxy_shifted.rolling(4).mean()    # 1h mean
    df['proxy_rmean8'] = proxy_shifted.rolling(8).mean()    # 2h mean
    df['proxy_rmean16'] = proxy_shifted.rolling(16).mean()  # 4h mean
    df['proxy_rmean32'] = proxy_shifted.rolling(32).mean()  # 8h mean
    df['proxy_rstd4'] = proxy_shifted.rolling(4).std()
    df['proxy_rstd8'] = proxy_shifted.rolling(8).std()
    df['proxy_rmin4'] = proxy_shifted.rolling(4).min()
    df['proxy_rmax4'] = proxy_shifted.rolling(4).max()
    df['proxy_range4'] = df['proxy_rmax4'] - df['proxy_rmin4']

    # Momentum and trend
    df['proxy_momentum'] = df[f'proxy_lag{LEAD_PERIODS}'] - df[f'proxy_lag{LEAD_PERIODS+1}']
    df['proxy_momentum4'] = df[f'proxy_lag{LEAD_PERIODS}'] - df[f'proxy_lag{LEAD_PERIODS+4}']
    df['proxy_acceleration'] = df['proxy_momentum'] - (df[f'proxy_lag{LEAD_PERIODS+1}'] - df[f'proxy_lag{LEAD_PERIODS+2}'])

    # Sign-asymmetric features
    df['proxy_lag8_pos'] = df[f'proxy_lag{LEAD_PERIODS}'].clip(lower=0)
    df['proxy_lag8_neg'] = df[f'proxy_lag{LEAD_PERIODS}'].clip(upper=0)

    # Regime: how many of last 8 periods were positive/negative
    for w in [4, 8, 16]:
        pos_count = pd.Series(0.0, index=df.index)
        for i in range(LEAD_PERIODS, LEAD_PERIODS + w):
            pos_count += (df['proxy'].shift(i) > 0).astype(float)
        df[f'proxy_pos_ratio_{w}'] = pos_count / w

    # Same period yesterday (96 periods ago)
    df['proxy_yesterday'] = df['proxy'].shift(96)
    df['proxy_yesterday_2'] = df['proxy'].shift(96 * 2)

    # ---- REGULATION FEATURES ----
    reg_shifted = df['reg_mean'].shift(LEAD_PERIODS)
    df['reg_rmean4'] = reg_shifted.rolling(4).mean()
    df['reg_rmean8'] = reg_shifted.rolling(8).mean()
    df['reg_rstd8'] = reg_shifted.rolling(8).std()
    df['reg_momentum'] = df['reg_mean'].shift(LEAD_PERIODS) - df['reg_mean'].shift(LEAD_PERIODS + 1)

    reg_std_shifted = df['reg_std'].shift(LEAD_PERIODS)
    df['reg_vol_rmean4'] = reg_std_shifted.rolling(4).mean()  # volatility of regulation

    # ---- LOAD FEATURES ----
    load_shifted = df['load_mean'].shift(LEAD_PERIODS)
    df['load_rmean4'] = load_shifted.rolling(4).mean()
    df['load_rmean16'] = load_shifted.rolling(16).mean()
    df['load_momentum'] = df['load_mean'].shift(LEAD_PERIODS) - df['load_mean'].shift(LEAD_PERIODS + 4)

    # Load deviation from time-of-day mean (expanding, computed properly)
    df['hour'] = df.index.hour
    df['qh'] = df.index.minute // 15
    df['hour_qh'] = df['hour'] * 4 + df['qh']
    df['dow'] = df.index.dayofweek

    # Load deviation: use only TRAIN period expanding mean to avoid leakage
    # Compute expanding mean only on data up to TRAIN_END, then freeze for test
    train_mask = df.index <= TRAIN_END
    load_train = load_shifted[train_mask]
    load_hourly_mean_train = load_train.groupby(df.loc[train_mask, 'hour_qh']).expanding().mean()
    load_hourly_mean_train = load_hourly_mean_train.droplevel(0).sort_index()
    load_hourly_mean_train = load_hourly_mean_train[~load_hourly_mean_train.index.duplicated(keep='last')]
    # For test: use the final train-period mean per hour_qh (frozen)
    frozen_load_mean = load_train.groupby(df.loc[train_mask, 'hour_qh']).mean()
    # Map: train uses expanding, test uses frozen
    load_baseline = load_hourly_mean_train.reindex(df.index)
    test_mask = df.index > TRAIN_END
    load_baseline[test_mask] = df.loc[test_mask, 'hour_qh'].map(frozen_load_mean).values
    df['load_deviation'] = load_shifted - load_baseline

    # ---- SOLAR FEATURES ----
    if solar is not None and 'solar_surprise_mw' in df.columns:
        solar_shifted = df['solar_surprise_mw'].shift(LEAD_PERIODS)
        df['solar_surprise_lag'] = solar_shifted
        df['solar_surprise_rmean4'] = solar_shifted.rolling(4).mean()
    else:
        df['solar_surprise_lag'] = 0
        df['solar_surprise_rmean4'] = 0

    # ---- TIME FEATURES ----
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['qh_in_hour'] = df['qh']  # 0-3 position within hour
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # ---- PROXY DEVIATION FROM HOURLY MEAN (leakage-safe) ----
    proxy_train = proxy_shifted[train_mask]
    proxy_hourly_mean_train = proxy_train.groupby(df.loc[train_mask, 'hour_qh']).expanding().mean()
    proxy_hourly_mean_train = proxy_hourly_mean_train.droplevel(0).sort_index()
    proxy_hourly_mean_train = proxy_hourly_mean_train[~proxy_hourly_mean_train.index.duplicated(keep='last')]
    frozen_proxy_mean = proxy_train.groupby(df.loc[train_mask, 'hour_qh']).mean()
    proxy_baseline = proxy_hourly_mean_train.reindex(df.index)
    proxy_baseline[test_mask] = df.loc[test_mask, 'hour_qh'].map(frozen_proxy_mean).values
    df['proxy_dev_from_hour'] = proxy_shifted - proxy_baseline

    # ---- TARGET ----
    df['target'] = df['imbalance_mwh']

    # Select feature columns
    feature_cols = [c for c in df.columns if c not in [
        'reg_mean', 'reg_std', 'reg_min', 'reg_max', 'reg_count',
        'proxy', 'load_mean', 'load_std',
        'solar_actual_mw', 'solar_forecast_da_mw', 'solar_surprise_mw',
        'imbalance_mwh', 'imb_price', 'target',
        'hour', 'qh', 'hour_qh', 'dow', 'month',
    ]]

    print(f"[+] Features: {len(feature_cols)} columns")
    print(f"    Sample features: {feature_cols[:10]}...")

    # Drop rows with NaN features
    valid = df[feature_cols + ['target']].dropna()
    print(f"[+] Valid samples: {len(valid)} (dropped {len(df) - len(valid)} with NaN)")

    return valid, feature_cols


# ============================================================
# TRAINING
# ============================================================

def train_quantile_models(df, feature_cols):
    """Train LightGBM models for each quantile."""
    print("\n[*] Training quantile models...")

    train = df[df.index <= TRAIN_END]
    test = df[df.index >= TEST_START]

    print(f"    Train: {len(train)} samples ({train.index.min()} to {train.index.max()})")
    print(f"    Test:  {len(test)} samples ({test.index.min()} to {test.index.max()})")

    X_train = train[feature_cols].values
    y_train = train['target'].values
    X_test = test[feature_cols].values
    y_test = test['target'].values

    models = {}
    predictions = {}

    for q in QUANTILES:
        print(f"    Training quantile {q:.2f}...")

        params = {
            'objective': 'quantile',
            'alpha': q,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_estimators': 500,
            'verbose': -1,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.log_evaluation(0)],
        )

        pred = model.predict(X_test)
        predictions[f'q{int(q*100)}'] = pred
        models[q] = model

        # Save model
        joblib.dump(model, MODEL_DIR / f"imb_2h_q{int(q*100)}.joblib")

    # Build predictions DataFrame
    pred_df = test[['target']].copy()
    for q_name, pred in predictions.items():
        pred_df[q_name] = pred

    pred_df['pred_median'] = pred_df['q50']
    pred_df['pred_direction'] = np.sign(pred_df['pred_median'])
    pred_df['actual_direction'] = np.sign(pred_df['target'])
    pred_df['confidence'] = pred_df['q90'] - pred_df['q10']  # width of 80% CI

    return models, pred_df, feature_cols


# ============================================================
# EVALUATION
# ============================================================

def evaluate(pred_df, models, feature_cols, df):
    """Evaluate model performance."""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    y_true = pred_df['target'].values
    y_pred = pred_df['pred_median'].values

    # Basic metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    print(f"\n  Point prediction (median):")
    print(f"    MAE:  {mae:.2f} MWh")
    print(f"    RMSE: {rmse:.2f} MWh")
    print(f"    R2:   {r2:.3f}")
    print(f"    Corr: {corr:.3f}")

    # Direction accuracy
    dir_correct = (pred_df['pred_direction'] == pred_df['actual_direction'])
    # Exclude zero imbalance
    nonzero = pred_df['target'].abs() > 0.1
    dir_acc = dir_correct[nonzero].mean()
    print(f"\n  Direction accuracy: {dir_acc:.1%} ({dir_correct[nonzero].sum()}/{nonzero.sum()} non-zero periods)")

    # Direction accuracy by confidence level
    print("\n  Direction accuracy by confidence (80% CI width):")
    pred_df['conf_q'] = pd.qcut(pred_df['confidence'], q=5, labels=['Q1_narrow', 'Q2', 'Q3', 'Q4', 'Q5_wide'])
    for q in ['Q1_narrow', 'Q2', 'Q3', 'Q4', 'Q5_wide']:
        mask = (pred_df['conf_q'] == q) & nonzero
        sub = pred_df[mask]
        if len(sub) > 0:
            acc = (sub['pred_direction'] == sub['actual_direction']).mean()
            avg_ci = sub['confidence'].mean()
            print(f"    {q}: {acc:.1%} accuracy, avg CI width = {avg_ci:.1f} MWh, n = {len(sub)}")

    # Direction accuracy by predicted magnitude
    print("\n  Direction accuracy by |predicted imbalance|:")
    pred_df['pred_abs'] = pred_df['pred_median'].abs()
    for lo, hi, label in [(0, 2, '0-2 MWh'), (2, 5, '2-5 MWh'), (5, 10, '5-10 MWh'),
                          (10, 20, '10-20 MWh'), (20, 999, '>20 MWh')]:
        mask = (pred_df['pred_abs'] >= lo) & (pred_df['pred_abs'] < hi) & nonzero
        sub = pred_df[mask]
        if len(sub) > 0:
            acc = (sub['pred_direction'] == sub['actual_direction']).mean()
            print(f"    |pred| {label}: {acc:.1%} ({len(sub)} periods)")

    # Quantile calibration
    print("\n  Quantile calibration (should match target %):")
    for q in QUANTILES:
        q_col = f'q{int(q*100)}'
        actual_below = (y_true < pred_df[q_col].values).mean()
        print(f"    Q{int(q*100)}: target {q:.0%}, actual {actual_below:.1%}")

    # By hour
    pred_df['hour'] = pred_df.index.hour
    print("\n  Direction accuracy by hour:")
    hourly = pred_df[nonzero].groupby('hour').apply(
        lambda g: (g['pred_direction'] == g['actual_direction']).mean()
    )
    for h, acc in hourly.items():
        bar = '#' * int(acc * 50)
        print(f"    H{h:02d}: {acc:.0%} {bar}")

    # Feature importance (from median model)
    print("\n  Top 15 features (median model):")
    imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': models[0.50].feature_importances_,
    }).sort_values('importance', ascending=False)
    imp['pct'] = imp['importance'] / imp['importance'].sum() * 100
    for _, r in imp.head(15).iterrows():
        print(f"    {r['feature']:<30} {r['pct']:>5.1f}%")

    imp.to_csv(DATA_DIR / "feature_importance.csv", index=False)
    pred_df.to_csv(DATA_DIR / "predictions_test.csv")

    return pred_df, imp


# ============================================================
# VISUALIZATION
# ============================================================

def plot_results(pred_df):
    """Generate evaluation plots."""

    # 1. Predicted vs actual scatter
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.scatter(pred_df['target'], pred_df['pred_median'], alpha=0.1, s=5, color='steelblue')
    lims = [-80, 80]
    ax.plot(lims, lims, 'r--', alpha=0.5)
    ax.set_xlabel("Actual Imbalance (MWh)")
    ax.set_ylabel("Predicted Imbalance (MWh)")
    ax.set_title("Predicted vs Actual (2h ahead)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # 2. Direction accuracy by predicted magnitude
    ax = axes[1]
    bins = np.arange(0, 35, 2)
    pred_df['pred_abs_bin'] = pd.cut(pred_df['pred_median'].abs(), bins=bins)
    nonzero = pred_df['target'].abs() > 0.1
    dir_by_mag = pred_df[nonzero].groupby('pred_abs_bin', observed=True).apply(
        lambda g: (g['pred_direction'] == g['actual_direction']).mean()
    )
    counts = pred_df[nonzero].groupby('pred_abs_bin', observed=True).size()

    x = range(len(dir_by_mag))
    ax.bar(x, dir_by_mag.values, alpha=0.7, color='steelblue')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% (coin flip)')
    ax.axhline(0.65, color='orange', linestyle='--', alpha=0.5, label='65% (target)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{b.left:.0f}' for b in dir_by_mag.index], rotation=45)
    ax.set_xlabel("|Predicted Imbalance| (MWh)")
    ax.set_ylabel("Direction Accuracy")
    ax.set_title("Direction Accuracy by Prediction Magnitude")
    ax.legend(fontsize=9)

    # Add count labels
    for i, (acc, cnt) in enumerate(zip(dir_by_mag.values, counts.values)):
        ax.text(i, acc + 0.01, f'n={cnt}', ha='center', fontsize=7)

    # 3. Sample time series
    ax = axes[2]
    sample = pred_df.iloc[:96*3]  # 3 days
    ax.fill_between(range(len(sample)), sample['q10'], sample['q90'], alpha=0.2, color='blue', label='P10-P90')
    ax.fill_between(range(len(sample)), sample['q25'], sample['q75'], alpha=0.3, color='blue', label='P25-P75')
    ax.plot(range(len(sample)), sample['pred_median'], 'b-', linewidth=1, label='Predicted (median)')
    ax.plot(range(len(sample)), sample['target'], 'k-', linewidth=1, alpha=0.7, label='Actual')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel("Settlement period")
    ax.set_ylabel("Imbalance (MWh)")
    ax.set_title("Sample: 3-Day Prediction with Confidence Bands")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_model_evaluation.png", dpi=150)
    plt.close()
    print(f"[+] Saved: {PLOT_DIR / '01_model_evaluation.png'}")

    # 4. Hourly direction accuracy
    fig, ax = plt.subplots(figsize=(12, 5))
    nonzero = pred_df['target'].abs() > 0.1
    hourly_acc = pred_df[nonzero].groupby(pred_df[nonzero].index.hour).apply(
        lambda g: (g['pred_direction'] == g['actual_direction']).mean()
    )
    colors = ['#2ecc71' if a >= 0.65 else '#f39c12' if a >= 0.55 else '#e74c3c' for a in hourly_acc]
    ax.bar(hourly_acc.index, hourly_acc.values, color=colors, alpha=0.8)
    ax.axhline(0.65, color='orange', linestyle='--', label='65% target')
    ax.axhline(0.50, color='red', linestyle='--', label='50% coin flip')
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Direction accuracy")
    ax.set_title("Imbalance Direction Accuracy by Hour (2h ahead)")
    ax.set_xticks(range(24))
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_hourly_direction_accuracy.png", dpi=150)
    plt.close()
    print(f"[+] Saved: {PLOT_DIR / '02_hourly_direction_accuracy.png'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("2-HOUR AHEAD IMBALANCE PREDICTOR")
    print(f"Lead: {LEAD_PERIODS} periods ({LEAD_PERIODS * 15} minutes)")
    print(f"Train: up to {TRAIN_END}, Test: from {TEST_START}")
    print("=" * 70)

    # Load data
    reg = load_regulation()
    load = load_load_data()
    solar = load_solar()
    imb = load_imbalance()

    # Build features
    df, feature_cols = build_features(reg, load, solar, imb)

    # Train
    models, pred_df, feature_cols = train_quantile_models(df, feature_cols)

    # Evaluate
    pred_df, imp = evaluate(pred_df, models, feature_cols, df)

    # Plot
    plot_results(pred_df)

    # Summary stats for trading
    print("\n" + "=" * 70)
    print("TRADING RELEVANCE")
    print("=" * 70)

    nonzero = pred_df['target'].abs() > 0.1
    dir_acc = (pred_df.loc[nonzero, 'pred_direction'] == pred_df.loc[nonzero, 'actual_direction']).mean()
    print(f"\n  Overall direction accuracy: {dir_acc:.1%}")

    # High confidence subset
    high_conf = pred_df['pred_abs'] > 5  # |predicted| > 5 MWh
    if high_conf.sum() > 0:
        hc_acc = (pred_df.loc[high_conf & nonzero, 'pred_direction'] == pred_df.loc[high_conf & nonzero, 'actual_direction']).mean()
        print(f"  High confidence (|pred|>5): {hc_acc:.1%} ({(high_conf & nonzero).sum()} periods, {(high_conf & nonzero).sum() / len(pred_df) * 100:.0f}% of all)")

    very_high = pred_df['pred_abs'] > 10
    if very_high.sum() > 0:
        vh_acc = (pred_df.loc[very_high & nonzero, 'pred_direction'] == pred_df.loc[very_high & nonzero, 'actual_direction']).mean()
        print(f"  Very high conf (|pred|>10): {vh_acc:.1%} ({(very_high & nonzero).sum()} periods)")

    print(f"\n  Periods per day (test): {len(pred_df) / pred_df.index.normalize().nunique():.0f}")
    print(f"  High-conf periods/day:  {high_conf.sum() / pred_df.index.normalize().nunique():.1f}")

    print("\n" + "=" * 70)
    print("[+] Done. Models saved to models/, predictions to data/")
    print("=" * 70)


if __name__ == "__main__":
    main()
