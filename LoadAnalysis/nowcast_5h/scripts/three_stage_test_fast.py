"""
Three-Stage Model Test (Fast Version)
======================================

Test whether AR residual correction can exploit remaining autocorrelation.
Simplified version without slow ARIMA rolling prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from statsmodels.tsa.ar_model import AutoReg
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent

print("=" * 70)
print("THREE-STAGE MODEL TEST (FAST)")
print("=" * 70)


def load_data():
    """Load hourly data."""
    print("\n[*] Loading data...")

    df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek

    print(f"    Records: {len(df):,}")

    return df


def create_features(df):
    """Create comprehensive feature set."""
    df = df.copy()

    # Error lags
    for lag in range(0, 49):
        df[f'error_lag{lag}'] = df['error'].shift(lag)

    # Rolling statistics
    df['error_roll_mean_24h'] = df['error'].shift(1).rolling(24).mean()
    df['error_roll_std_24h'] = df['error'].shift(1).rolling(24).std()
    df['error_roll_mean_6h'] = df['error'].shift(1).rolling(6).mean()
    df['error_roll_mean_12h'] = df['error'].shift(1).rolling(12).mean()

    # Same hour features
    df['error_same_hour_yesterday'] = df['error'].shift(24)
    df['error_same_hour_2d_ago'] = df['error'].shift(48)

    # Momentum features
    df['error_diff_1h'] = df['error_lag0'] - df['error_lag1']
    df['error_diff_2h'] = df['error_lag0'] - df['error_lag2']
    df['error_diff_24h'] = df['error_lag0'] - df['error_lag24']

    # Forecast context
    df['forecast_load'] = df['forecast_load_mw']
    df['forecast_diff_1h'] = df['forecast_load_mw'] - df['forecast_load_mw'].shift(1)
    df['forecast_diff_24h'] = df['forecast_load_mw'] - df['forecast_load_mw'].shift(24)

    # Error regime
    df['error_lag0_abs'] = df['error_lag0'].abs()
    df['error_lag0_sign'] = np.sign(df['error_lag0'])
    df['error_lag1_sign'] = np.sign(df['error_lag1'])
    df['error_sign_same'] = (df['error_lag0_sign'] == df['error_lag1_sign']).astype(int)

    # Hour interactions
    df['hour_x_error_lag0'] = df['hour'] * df['error_lag0'] / 100
    df['hour_x_error_sign'] = df['hour'] * df['error_lag0_sign']

    # Targets
    for h in range(1, 6):
        df[f'target_h{h}'] = df['error'].shift(-h)

    return df


def test_ar_correction(df, horizon=2):
    """Test AR residual correction for specific horizon."""
    print(f"\n[*] Testing H+{horizon}...")

    features = [
        'error_lag0', 'error_lag1', 'error_lag2', 'error_lag3', 'error_lag24', 'error_lag48',
        'error_roll_mean_24h', 'error_roll_std_24h', 'error_roll_mean_6h', 'error_roll_mean_12h',
        'error_same_hour_yesterday', 'error_same_hour_2d_ago',
        'error_diff_1h', 'error_diff_2h', 'error_diff_24h',
        'forecast_load', 'forecast_diff_1h', 'forecast_diff_24h',
        'error_lag0_abs', 'error_lag0_sign', 'error_sign_same',
        'hour_x_error_lag0', 'hour_x_error_sign',
        'hour', 'dow'
    ]

    target_col = f'target_h{horizon}'

    # Split data
    train = df[df['year'] == 2024].dropna(subset=features + [target_col]).copy()
    test = df[df['year'] >= 2025].dropna(subset=features + [target_col]).copy()

    print(f"    Train: {len(train):,}")
    print(f"    Test:  {len(test):,}")

    # === STAGE 1: LightGBM ===
    print("    Stage 1: LightGBM...")
    model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        verbosity=-1, random_state=42
    )
    model.fit(train[features], train[target_col])

    train_pred = model.predict(train[features])
    train_residuals = train[target_col].values - train_pred

    test_pred = model.predict(test[features])
    test_residuals = test[target_col].values - test_pred

    mae_stage1 = np.nanmean(np.abs(test_residuals))
    print(f"    Stage 1 MAE: {mae_stage1:.2f} MW")

    # === RESIDUAL AUTOCORRELATION ===
    print("    Residual autocorrelation (train):")
    resid_autocorr = {}
    for lag in [1, 2, 3, 24]:
        corr = pd.Series(train_residuals).autocorr(lag=lag)
        resid_autocorr[lag] = corr
        print(f"      Lag-{lag}: {corr:.3f}")

    # === STAGE 2: AR MODEL ===
    print("    Stage 2: AR model...")

    # Find best AR order
    best_order = 1
    best_aic = np.inf
    for order in [1, 2, 3, 4]:
        try:
            ar_model = AutoReg(train_residuals, lags=order, old_names=False)
            ar_fit = ar_model.fit()
            if ar_fit.aic < best_aic:
                best_aic = ar_fit.aic
                best_order = order
        except:
            pass

    print(f"    Best AR order: {best_order}")

    # Fit AR model
    ar_model = AutoReg(train_residuals, lags=best_order, old_names=False)
    ar_fit = ar_model.fit()
    print(f"    AR coefficients: {ar_fit.params[1:best_order+1].round(3).tolist()}")

    # === ROLLING AR PREDICTION ===
    # In practice: at time t, we know residuals up to t-1
    # We use AR to predict residual at t, then add to LightGBM prediction

    all_residuals = np.concatenate([train_residuals, test_residuals])
    n_train = len(train_residuals)

    ar_corrections = []
    for i in range(len(test_residuals)):
        # Available residuals: train + test[:i] (we don't know test[i] yet)
        # But in real application, residual at t-1 becomes known only AFTER
        # we make prediction at t-1 and observe actual at t-1+h

        # Simplification: use residual lagged by horizon
        # At time t predicting t+h, we know residual at t-h (from prediction at t-2h for t-h)
        avail_idx = n_train + i - horizon
        if avail_idx >= best_order:
            recent_resid = all_residuals[avail_idx - best_order:avail_idx]
            ar_pred = ar_fit.params[0]
            for j, coef in enumerate(ar_fit.params[1:best_order+1]):
                ar_pred += coef * recent_resid[-(j+1)]
            ar_corrections.append(ar_pred)
        else:
            ar_corrections.append(0)

    ar_corrections = np.array(ar_corrections)

    # Combined prediction
    final_pred = test_pred + ar_corrections
    final_residuals = test[target_col].values - final_pred

    mae_stage2 = np.nanmean(np.abs(final_residuals))
    print(f"    Stage 2 MAE: {mae_stage2:.2f} MW")

    improvement = mae_stage1 - mae_stage2
    print(f"    Improvement: {improvement:+.2f} MW ({improvement/mae_stage1*100:+.1f}%)")

    # Sanity check
    corr = np.corrcoef(ar_corrections, test_residuals)[0, 1]
    print(f"    Corr(AR correction, true residual): {corr:.3f}")

    return {
        'horizon': horizon,
        'mae_stage1': mae_stage1,
        'mae_stage2': mae_stage2,
        'improvement': improvement,
        'ar_order': best_order,
        'resid_autocorr_lag1': resid_autocorr[1]
    }


def test_residual_feature(df, horizon=2):
    """
    Alternative: Add lagged residuals as features to LightGBM.
    Let ML learn the AR relationship.
    """
    print(f"\n[*] Alternative (lagged residual feature) for H+{horizon}...")

    features = [
        'error_lag0', 'error_lag1', 'error_lag2', 'error_lag3', 'error_lag24', 'error_lag48',
        'error_roll_mean_24h', 'error_roll_std_24h', 'error_roll_mean_6h', 'error_roll_mean_12h',
        'error_same_hour_yesterday', 'error_same_hour_2d_ago',
        'error_diff_1h', 'error_diff_2h', 'error_diff_24h',
        'forecast_load', 'forecast_diff_1h', 'forecast_diff_24h',
        'error_lag0_abs', 'error_lag0_sign', 'error_sign_same',
        'hour_x_error_lag0', 'hour_x_error_sign',
        'hour', 'dow'
    ]

    target_col = f'target_h{horizon}'

    # First train Stage 1 model
    train_base = df[df['year'] == 2024].dropna(subset=features + [target_col]).copy()

    model_base = lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        verbosity=-1, random_state=42
    )
    model_base.fit(train_base[features], train_base[target_col])

    # Get ALL residuals
    df_valid = df.dropna(subset=features + [target_col]).copy()
    df_valid['pred_s1'] = model_base.predict(df_valid[features])
    df_valid['resid_s1'] = df_valid[target_col] - df_valid['pred_s1']

    # Create lagged residual features (lagged by horizon)
    df_valid['resid_lag_h'] = df_valid['resid_s1'].shift(horizon)
    df_valid['resid_lag_h1'] = df_valid['resid_s1'].shift(horizon + 1)
    df_valid['resid_lag_h2'] = df_valid['resid_s1'].shift(horizon + 2)

    features_enh = features + ['resid_lag_h', 'resid_lag_h1', 'resid_lag_h2']

    train = df_valid[df_valid['year'] == 2024].dropna(subset=features_enh).copy()
    test = df_valid[df_valid['year'] >= 2025].dropna(subset=features_enh).copy()

    print(f"    Train: {len(train):,}")
    print(f"    Test:  {len(test):,}")

    # Baseline
    test_pred_base = model_base.predict(test[features])
    mae_base = np.nanmean(np.abs(test[target_col].values - test_pred_base))
    print(f"    Baseline MAE: {mae_base:.2f} MW")

    # Enhanced model
    model_enh = lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        verbosity=-1, random_state=42
    )
    model_enh.fit(train[features_enh], train[target_col])

    test_pred_enh = model_enh.predict(test[features_enh])
    mae_enh = np.nanmean(np.abs(test[target_col].values - test_pred_enh))
    print(f"    Enhanced MAE: {mae_enh:.2f} MW")

    improvement = mae_base - mae_enh
    print(f"    Improvement: {improvement:+.2f} MW ({improvement/mae_base*100:+.1f}%)")

    # Check feature importance
    importance = pd.DataFrame({
        'feature': features_enh,
        'importance': model_enh.feature_importances_
    }).sort_values('importance', ascending=False)

    print("    Top features:")
    for _, row in importance.head(8).iterrows():
        print(f"      {row['feature']}: {row['importance']}")

    return {
        'mae_base': mae_base,
        'mae_enh': mae_enh,
        'improvement': improvement
    }


def main():
    df = load_data()
    df = create_features(df)

    print("\n" + "=" * 70)
    print("TEST 1: AR RESIDUAL CORRECTION")
    print("=" * 70)

    ar_results = {}
    for h in [2, 3, 4, 5]:
        ar_results[h] = test_ar_correction(df, horizon=h)

    print("\n" + "=" * 70)
    print("TEST 2: LAGGED RESIDUAL AS FEATURE")
    print("=" * 70)

    feat_results = {}
    for h in [2, 3, 4, 5]:
        feat_results[h] = test_residual_feature(df, horizon=h)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n  AR Residual Correction:")
    for h, r in ar_results.items():
        print(f"    H+{h}: {r['mae_stage1']:.1f} -> {r['mae_stage2']:.1f} MW ({r['improvement']:+.2f} MW)")

    print("\n  Lagged Residual Feature:")
    for h, r in feat_results.items():
        print(f"    H+{h}: {r['mae_base']:.1f} -> {r['mae_enh']:.1f} MW ({r['improvement']:+.2f} MW)")

    # Average improvements
    avg_ar = np.mean([r['improvement'] for r in ar_results.values()])
    avg_feat = np.mean([r['improvement'] for r in feat_results.values()])

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"\n  Average AR improvement: {avg_ar:+.2f} MW")
    print(f"  Average feature improvement: {avg_feat:+.2f} MW")

    if avg_ar > 1.0 or avg_feat > 1.0:
        print("\n  [+] Residual modeling shows meaningful improvement")
        if avg_feat > avg_ar:
            print("      RECOMMEND: Lagged residual feature (simpler, works better)")
        else:
            print("      RECOMMEND: AR correction (more gain)")
    elif avg_ar > 0.3 or avg_feat > 0.3:
        print("\n  [~] Modest improvement - consider if complexity is worth it")
    else:
        print("\n  [-] Limited improvement - additional complexity NOT justified")

    return ar_results, feat_results


if __name__ == "__main__":
    ar_results, feat_results = main()
