"""
Three-Stage Model Test: Can AR residual correction help?
========================================================

Test whether adding a statistical model (AR) on top of LightGBM residuals
can exploit the remaining autocorrelation in errors.

Stage 1: LightGBM predicts error (current approach)
Stage 2: AR model predicts Stage 1 residuals

Key question: Does the added complexity provide meaningful improvement?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent

print("=" * 70)
print("THREE-STAGE MODEL TEST")
print("Can AR residual correction exploit remaining autocorrelation?")
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


def test_three_stage(df, horizon=2):
    """Test 3-stage model for specific horizon."""
    print(f"\n[*] Testing H+{horizon} with 3-stage approach...")

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

    print(f"    Train: {len(train):,} records")
    print(f"    Test:  {len(test):,} records")

    # === STAGE 1: LightGBM ===
    print("\n  Stage 1: Training LightGBM...")
    model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        verbosity=-1, random_state=42
    )
    model.fit(train[features], train[target_col])

    # Get training predictions and residuals
    train_pred = model.predict(train[features])
    train_residuals = train[target_col].values - train_pred

    # Get test predictions and residuals
    test_pred = model.predict(test[features])
    test_residuals = test[target_col].values - test_pred

    # Stage 1 MAE
    mae_stage1 = np.nanmean(np.abs(test_residuals))
    print(f"    Stage 1 MAE: {mae_stage1:.2f} MW")

    # === ANALYZE RESIDUAL AUTOCORRELATION ===
    print("\n  Residual autocorrelation (training):")
    for lag in [1, 2, 3, 24]:
        corr = pd.Series(train_residuals).autocorr(lag=lag)
        print(f"    Lag-{lag}: {corr:.3f}")

    # === STAGE 2: AR MODEL ON RESIDUALS ===
    print("\n  Stage 2: Fitting AR model on training residuals...")

    # Try different AR orders
    best_order = 1
    best_aic = np.inf

    train_resid_series = pd.Series(train_residuals, index=train['datetime'].values)

    for order in [1, 2, 3, 4, 5]:
        try:
            ar_model = AutoReg(train_residuals, lags=order, old_names=False)
            ar_fit = ar_model.fit()
            if ar_fit.aic < best_aic:
                best_aic = ar_fit.aic
                best_order = order
        except:
            pass

    print(f"    Best AR order by AIC: {best_order}")

    # Fit final AR model
    ar_model = AutoReg(train_residuals, lags=best_order, old_names=False)
    ar_fit = ar_model.fit()

    print(f"    AR coefficients: {ar_fit.params[1:].round(3).tolist()}")

    # === ROLLING AR PREDICTION ON TEST SET ===
    print("\n  Stage 2: Rolling AR prediction on test set...")

    # We need to predict test residuals using lagged residuals
    # This simulates real-time: at time t, we have residuals up to t-1

    # Create lagged residual features
    all_residuals = np.concatenate([train_residuals, test_residuals])
    n_train = len(train_residuals)

    ar_corrections = []

    for i in range(len(test_residuals)):
        # At test point i, we have residuals up to train + i - 1
        # But we only know "realized" residuals after we see actuals
        #
        # In practice: at time T predicting T+h:
        #   - We made predictions at T-1 for (T-1)+h = T+h-1
        #   - At T, we now know actual for T+h-1, so we know residual at T-1
        #   - We can use this to correct prediction for T+h

        # Get available residuals (up to current point)
        avail_residuals = all_residuals[:n_train + i]

        if len(avail_residuals) >= best_order:
            # Use last 'best_order' residuals
            recent_resid = avail_residuals[-best_order:]

            # AR prediction: intercept + sum(coef * lagged_resid)
            ar_pred = ar_fit.params[0]  # intercept
            for j, coef in enumerate(ar_fit.params[1:]):
                ar_pred += coef * recent_resid[-(j+1)]

            ar_corrections.append(ar_pred)
        else:
            ar_corrections.append(0)

    ar_corrections = np.array(ar_corrections)

    # === COMBINED PREDICTIONS ===
    final_pred = test_pred + ar_corrections
    final_residuals = test[target_col].values - final_pred

    mae_stage2 = np.nanmean(np.abs(final_residuals))
    print(f"    Stage 2 MAE: {mae_stage2:.2f} MW")

    improvement = mae_stage1 - mae_stage2
    print(f"\n    Improvement: {improvement:+.2f} MW ({improvement/mae_stage1*100:+.1f}%)")

    # === SANITY CHECK: AR on raw actuals vs residuals ===
    print("\n  Sanity checks:")

    # Check correlation between AR correction and actual residual
    corr = np.corrcoef(ar_corrections, test_residuals)[0, 1]
    print(f"    Correlation(AR correction, actual residual): {corr:.3f}")

    # Check if we're just adding noise
    ar_var = np.var(ar_corrections)
    resid_var = np.var(test_residuals)
    print(f"    AR correction variance: {ar_var:.1f}")
    print(f"    Actual residual variance: {resid_var:.1f}")

    # === ALTERNATIVE: ARIMA ===
    print("\n  Alternative: ARIMA(1,0,1) on residuals...")
    try:
        arima_model = ARIMA(train_residuals, order=(1, 0, 1))
        arima_fit = arima_model.fit()

        # Rolling ARIMA prediction
        arima_corrections = []
        for i in range(len(test_residuals)):
            avail_residuals = all_residuals[:n_train + i]
            if len(avail_residuals) > 10:
                try:
                    # One-step forecast
                    arima_temp = ARIMA(avail_residuals[-100:], order=(1, 0, 1))
                    arima_temp_fit = arima_temp.fit()
                    fc = arima_temp_fit.forecast(steps=1)[0]
                    arima_corrections.append(fc)
                except:
                    arima_corrections.append(0)
            else:
                arima_corrections.append(0)

        arima_corrections = np.array(arima_corrections)
        arima_pred = test_pred + arima_corrections
        mae_arima = np.nanmean(np.abs(test[target_col].values - arima_pred))
        print(f"    ARIMA MAE: {mae_arima:.2f} MW")
        print(f"    ARIMA improvement: {mae_stage1 - mae_arima:+.2f} MW")
    except Exception as e:
        print(f"    ARIMA failed: {e}")

    return {
        'horizon': horizon,
        'mae_stage1': mae_stage1,
        'mae_stage2': mae_stage2,
        'improvement': improvement,
        'ar_order': best_order,
        'ar_correction_corr': corr
    }


def test_lagged_residual_feature(df, horizon=2):
    """
    Alternative approach: Add lagged residuals as FEATURES to LightGBM.
    This lets the ML model learn the AR relationship itself.
    """
    print(f"\n[*] Alternative: Lagged residual as feature for H+{horizon}...")

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

    # First, train Stage 1 to get residuals
    train_base = df[df['year'] == 2024].dropna(subset=features + [target_col]).copy()

    model_base = lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        verbosity=-1, random_state=42
    )
    model_base.fit(train_base[features], train_base[target_col])

    # Get ALL predictions and residuals (train + test)
    df_valid = df.dropna(subset=features + [target_col]).copy()
    df_valid['pred_stage1'] = model_base.predict(df_valid[features])
    df_valid['resid_stage1'] = df_valid[target_col] - df_valid['pred_stage1']

    # Create lagged residual features
    df_valid['resid_lag1'] = df_valid['resid_stage1'].shift(1)
    df_valid['resid_lag2'] = df_valid['resid_stage1'].shift(2)
    df_valid['resid_lag3'] = df_valid['resid_stage1'].shift(3)

    # Train enhanced model with residual lags
    features_enhanced = features + ['resid_lag1', 'resid_lag2', 'resid_lag3']

    train = df_valid[df_valid['year'] == 2024].dropna(subset=features_enhanced).copy()
    test = df_valid[df_valid['year'] >= 2025].dropna(subset=features_enhanced).copy()

    print(f"    Train: {len(train):,} records")
    print(f"    Test:  {len(test):,} records")

    # Stage 1 baseline on test
    test_pred_base = model_base.predict(test[features])
    mae_base = np.nanmean(np.abs(test[target_col].values - test_pred_base))
    print(f"    Baseline MAE: {mae_base:.2f} MW")

    # Train enhanced model
    model_enhanced = lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        verbosity=-1, random_state=42
    )
    model_enhanced.fit(train[features_enhanced], train[target_col])

    # Evaluate enhanced model
    test_pred_enhanced = model_enhanced.predict(test[features_enhanced])
    mae_enhanced = np.nanmean(np.abs(test[target_col].values - test_pred_enhanced))
    print(f"    Enhanced MAE: {mae_enhanced:.2f} MW")

    improvement = mae_base - mae_enhanced
    print(f"    Improvement: {improvement:+.2f} MW")

    # Feature importance
    importance = pd.DataFrame({
        'feature': features_enhanced,
        'importance': model_enhanced.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n    Top features (enhanced model):")
    for _, row in importance.head(10).iterrows():
        print(f"      {row['feature']}: {row['importance']}")

    return {
        'mae_base': mae_base,
        'mae_enhanced': mae_enhanced,
        'improvement': improvement
    }


def main():
    df = load_data()
    df = create_features(df)

    print("\n" + "=" * 70)
    print("TEST 1: THREE-STAGE (LightGBM + AR) APPROACH")
    print("=" * 70)

    results = {}
    for h in [2, 3, 5]:  # Test key horizons
        results[h] = test_three_stage(df, horizon=h)

    print("\n" + "=" * 70)
    print("TEST 2: LAGGED RESIDUAL AS LightGBM FEATURE")
    print("=" * 70)
    print("(Simpler approach: let ML learn the AR relationship)")

    alt_results = {}
    for h in [2, 3, 5]:
        alt_results[h] = test_lagged_residual_feature(df, horizon=h)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n  Three-Stage (LightGBM + AR):")
    for h, r in results.items():
        print(f"    H+{h}: {r['mae_stage1']:.1f} -> {r['mae_stage2']:.1f} ({r['improvement']:+.2f} MW)")

    print("\n  Lagged Residual Feature:")
    for h, r in alt_results.items():
        print(f"    H+{h}: {r['mae_base']:.1f} -> {r['mae_enhanced']:.1f} ({r['improvement']:+.2f} MW)")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    avg_ar_improvement = np.mean([r['improvement'] for r in results.values()])
    avg_feat_improvement = np.mean([r['improvement'] for r in alt_results.values()])

    print(f"\n  Average AR improvement: {avg_ar_improvement:+.2f} MW")
    print(f"  Average feature improvement: {avg_feat_improvement:+.2f} MW")

    if avg_ar_improvement > 1.0:
        print("\n  [+] AR approach shows meaningful improvement")
        print("      Consider implementing for production")
    elif avg_feat_improvement > 0.5:
        print("\n  [+] Lagged residual feature shows improvement")
        print("      SIMPLER approach - recommend this over 3-stage")
    else:
        print("\n  [-] Limited improvement from residual modeling")
        print("      Additional complexity not justified")

    return results, alt_results


if __name__ == "__main__":
    results, alt_results = main()
