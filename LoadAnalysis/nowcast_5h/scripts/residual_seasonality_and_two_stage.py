"""
Residual Seasonality Analysis + Corrected Two-Stage Model

1. Plot residual patterns by hour, day of week, month
2. Test two-stage model with CORRECT shift (H + lag, not H + lag - 1)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
ERROR_ANALYSIS_DIR = BASE_DIR / 'error_analysis'
OUTPUT_DIR = BASE_DIR / 'residual_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions(horizon):
    """Load predictions for a horizon."""
    df = pd.read_csv(ERROR_ANALYSIS_DIR / f'h{horizon}_predictions.csv', parse_dates=['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    return df


def plot_residual_seasonality(all_dfs, output_dir):
    """Plot residual patterns by hour, day of week for all horizons."""

    fig, axes = plt.subplots(5, 3, figsize=(15, 18))

    for i, h in enumerate(range(1, 6)):
        df = all_dfs[h]
        residuals = df['prediction_error']

        # By hour
        ax1 = axes[i, 0]
        hourly = df.groupby('hour')['prediction_error'].agg(['mean', 'std'])
        ax1.bar(hourly.index, hourly['mean'], yerr=hourly['std']/5, capsize=2, alpha=0.7)
        ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Mean Residual (MW)')
        ax1.set_title(f'H+{h}: Residual by Hour')
        ax1.set_xticks(range(0, 24, 3))

        # By day of week
        ax2 = axes[i, 1]
        daily = df.groupby('dow')['prediction_error'].agg(['mean', 'std'])
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax2.bar(range(7), daily['mean'], yerr=daily['std']/5, capsize=2, alpha=0.7)
        ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Mean Residual (MW)')
        ax2.set_title(f'H+{h}: Residual by Day')
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(days)

        # By hour, colored by weekend
        ax3 = axes[i, 2]
        weekday = df[df['is_weekend'] == 0].groupby('hour')['prediction_error'].mean()
        weekend = df[df['is_weekend'] == 1].groupby('hour')['prediction_error'].mean()
        ax3.plot(weekday.index, weekday.values, 'b-', label='Weekday', linewidth=2)
        ax3.plot(weekend.index, weekend.values, 'r-', label='Weekend', linewidth=2)
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Hour')
        ax3.set_ylabel('Mean Residual (MW)')
        ax3.set_title(f'H+{h}: Weekday vs Weekend')
        ax3.legend()
        ax3.set_xticks(range(0, 24, 3))

    plt.tight_layout()
    plt.savefig(output_dir / '03_residual_seasonality.png', dpi=150)
    plt.close()
    print("[+] Saved residual seasonality plots")


def plot_residual_autocorr_decay(all_dfs, output_dir):
    """Plot how autocorrelation decays with lag for each horizon."""
    from statsmodels.tsa.stattools import acf

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.8, 5))

    for i, h in enumerate(range(1, 6)):
        df = all_dfs[h]
        residuals = df['prediction_error'].values
        acf_vals = acf(residuals, nlags=12, fft=True)

        # Mark the usable lag for this horizon
        ax.plot(range(13), acf_vals, 'o-', color=colors[i], label=f'H+{h} (usable lag={h})', linewidth=2)
        ax.axvline(h, color=colors[i], linestyle='--', alpha=0.3)

    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Residual ACF by Horizon (dashed line = usable lag)')
    ax.legend()
    ax.set_xticks(range(13))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '04_acf_decay_by_horizon.png', dpi=150)
    plt.close()
    print("[+] Saved ACF decay plot")


def test_two_stage_corrected(df, horizon):
    """
    Test two-stage model with CORRECT shift.

    Stage 2 features use shift = H + lag (not H + lag - 1).
    """
    print(f"\n{'='*60}")
    print(f"H+{horizon} Two-Stage Model (CORRECT SHIFT)")
    print(f"{'='*60}")

    residuals = df['prediction_error'].values
    actuals = df['actual_error'].values
    predictions = df['predicted_error'].values

    # Create Stage 2 features with CORRECT shift
    df_s2 = df.copy()

    # Residual lags with correct shift:
    # lag 1 = shift(H), lag 2 = shift(H+1), etc.
    # Because: at time T, most recent usable residual is at T-H
    for lag in range(1, 7):
        shift = horizon + lag - 1  # lag 1 -> shift H, lag 2 -> shift H+1
        df_s2[f'resid_lag{lag}'] = df['prediction_error'].shift(shift)

    # Rolling features on residuals (also correctly shifted)
    # Rolling mean/std starting from the most recent usable residual (shift H)
    for window in [3, 6, 12]:
        df_s2[f'resid_roll_mean_{window}'] = df['prediction_error'].shift(horizon).rolling(window).mean()
        df_s2[f'resid_roll_std_{window}'] = df['prediction_error'].shift(horizon).rolling(window).std()

    # Time features (these don't need shifting - they're for the current prediction)
    df_s2['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df_s2['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df_s2['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df_s2['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)

    # Features list
    resid_features = [f'resid_lag{i}' for i in range(1, 7)]
    roll_features = [f'resid_roll_mean_{w}' for w in [3, 6, 12]] + [f'resid_roll_std_{w}' for w in [3, 6, 12]]
    time_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'hour', 'dow', 'is_weekend']

    all_features = resid_features + roll_features + time_features

    # Target: the residual we want to predict
    df_s2['target'] = df['prediction_error']

    # Drop NaNs
    df_s2_valid = df_s2.dropna(subset=all_features + ['target'])

    # Split: 80% train, 20% test
    split_idx = int(len(df_s2_valid) * 0.8)
    train = df_s2_valid.iloc[:split_idx]
    test = df_s2_valid.iloc[split_idx:]

    X_train = train[all_features]
    y_train = train['target']
    X_test = test[all_features]
    y_test = test['target']

    # Baseline: no correction
    baseline_mae = np.abs(test['target']).mean()
    print(f"  Baseline MAE: {baseline_mae:.2f} MW")

    # Train LightGBM Stage 2
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)

    # Predict residuals
    resid_pred = model.predict(X_test)

    # Corrected predictions
    corrected_preds = test['predicted_error'].values + resid_pred
    corrected_mae = np.abs(test['actual_error'].values - corrected_preds).mean()

    improvement = baseline_mae - corrected_mae
    print(f"  LightGBM S2 MAE: {corrected_mae:.2f} MW")
    print(f"  Improvement: {improvement:+.2f} MW ({100*improvement/baseline_mae:+.1f}%)")

    # Feature importance
    importance = dict(zip(all_features, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top features: {[f[0] for f in top_features]}")

    # Compare with simple AR
    # AR(2) with correct shift
    ar_order = 2
    min_idx = horizon + ar_order
    X_ar_train = np.column_stack([train['prediction_error'].shift(horizon + k).values for k in range(1, ar_order + 1)])
    X_ar_train = X_ar_train[min_idx:]
    y_ar_train = train['target'].values[min_idx:]

    phi = np.linalg.lstsq(X_ar_train[~np.isnan(X_ar_train).any(axis=1)],
                          y_ar_train[~np.isnan(X_ar_train).any(axis=1)], rcond=None)[0]

    X_ar_test = np.column_stack([test['prediction_error'].shift(horizon + k).values for k in range(1, ar_order + 1)])
    valid_test = ~np.isnan(X_ar_test).any(axis=1)
    ar_pred = X_ar_test[valid_test] @ phi
    ar_corrected = test['predicted_error'].values[valid_test] + ar_pred
    ar_mae = np.abs(test['actual_error'].values[valid_test] - ar_corrected).mean()

    print(f"  AR(2) MAE: {ar_mae:.2f} MW (for comparison)")

    return {
        'baseline': baseline_mae,
        'lgb_s2': corrected_mae,
        'ar2': ar_mae,
        'improvement_lgb': improvement,
        'improvement_ar': baseline_mae - ar_mae,
        'top_features': top_features
    }


def main():
    print("="*60)
    print("RESIDUAL SEASONALITY & CORRECTED TWO-STAGE")
    print("="*60)

    # Load all predictions
    all_dfs = {}
    for h in range(1, 6):
        all_dfs[h] = load_predictions(h)

    # Plot seasonality
    print("\n[*] Plotting residual seasonality...")
    plot_residual_seasonality(all_dfs, OUTPUT_DIR)
    plot_residual_autocorr_decay(all_dfs, OUTPUT_DIR)

    # Test corrected two-stage
    print("\n[*] Testing corrected two-stage model...")
    results = {}
    for h in range(1, 6):
        results[h] = test_two_stage_corrected(all_dfs[h], h)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: AR vs LightGBM Stage 2 (Correct Shift)")
    print("="*60)
    print("\n| Horizon | Baseline | AR(2) | LGB S2 | AR Impr | LGB Impr |")
    print("|---------|----------|-------|--------|---------|----------|")

    for h in range(1, 6):
        r = results[h]
        print(f"| H+{h}     | {r['baseline']:.1f}    | {r['ar2']:.1f} | {r['lgb_s2']:.1f}  | "
              f"{r['improvement_ar']:+.1f} ({100*r['improvement_ar']/r['baseline']:+.1f}%) | "
              f"{r['improvement_lgb']:+.1f} ({100*r['improvement_lgb']/r['baseline']:+.1f}%) |")


if __name__ == '__main__':
    main()
