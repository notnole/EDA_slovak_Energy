"""
Residual Autocorrelation Analysis and AR Correction

Based on forecast correction methodology:
1. Check if residuals are white noise (should be if model is good)
2. Plot ACF to identify autocorrelation structure
3. Apply AR correction if significant autocorrelation exists
4. Measure improvement from correction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
ERROR_ANALYSIS_DIR = BASE_DIR / 'error_analysis'
OUTPUT_DIR = BASE_DIR / 'residual_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions(horizon):
    """Load predictions for a horizon."""
    df = pd.read_csv(ERROR_ANALYSIS_DIR / f'h{horizon}_predictions.csv', parse_dates=['datetime'])
    return df


def analyze_residual_autocorrelation(df, horizon):
    """Analyze autocorrelation structure in residuals."""
    residuals = df['prediction_error'].values  # actual - predicted

    print(f"\n{'='*60}")
    print(f"H+{horizon} Residual Autocorrelation Analysis")
    print('='*60)

    # Basic statistics
    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals)
    print(f"\nBasic Statistics:")
    print(f"  Mean (bias): {mean_resid:.2f} MW")
    print(f"  Std: {std_resid:.2f} MW")

    # Compute ACF and PACF
    max_lag = min(168, len(residuals) // 10)  # Up to 1 week or 10% of data
    acf_vals = acf(residuals, nlags=max_lag, fft=True)
    pacf_vals = pacf(residuals, nlags=min(48, max_lag))  # PACF for shorter lags

    # Significance threshold (approximate 95% CI for white noise)
    n = len(residuals)
    sig_threshold = 1.96 / np.sqrt(n)

    # Find significant lags
    significant_acf = [(lag, acf_vals[lag]) for lag in range(1, len(acf_vals))
                       if abs(acf_vals[lag]) > sig_threshold]

    print(f"\nAutocorrelation Analysis:")
    print(f"  Significance threshold (95%): +/-{sig_threshold:.4f}")
    print(f"  Significant ACF lags: {len(significant_acf)}")

    if significant_acf:
        print(f"\n  Top 10 significant ACF lags:")
        for lag, val in sorted(significant_acf, key=lambda x: abs(x[1]), reverse=True)[:10]:
            print(f"    Lag {lag:3d}: {val:+.4f} {'***' if abs(val) > 3*sig_threshold else '**' if abs(val) > 2*sig_threshold else '*'}")

    # Ljung-Box test for white noise
    lb_test = acorr_ljungbox(residuals, lags=[24, 48, 168], return_df=True)
    print(f"\n  Ljung-Box Test (H0: white noise):")
    for lag in [24, 48, 168]:
        if lag in lb_test.index:
            stat = lb_test.loc[lag, 'lb_stat']
            pval = lb_test.loc[lag, 'lb_pvalue']
            print(f"    Lag {lag:3d}: stat={stat:.1f}, p={pval:.2e} {'(reject H0)' if pval < 0.05 else ''}")

    return {
        'mean': mean_resid,
        'std': std_resid,
        'acf': acf_vals,
        'pacf': pacf_vals,
        'significant_lags': significant_acf,
        'sig_threshold': sig_threshold
    }


def test_ar_correction(df, horizon, max_ar_order=24):
    """Test AR correction on residuals."""
    residuals = df['prediction_error'].values
    datetimes = df['datetime'].values
    actuals = df['actual_error'].values
    predictions = df['predicted_error'].values

    print(f"\n--- AR Correction Test ---")

    # Split into train (80%) and test (20%)
    split_idx = int(len(residuals) * 0.8)
    train_resid = residuals[:split_idx]
    test_resid = residuals[split_idx:]
    test_actuals = actuals[split_idx:]
    test_preds = predictions[split_idx:]

    # Baseline MAE (no correction)
    baseline_mae = np.abs(test_resid).mean()
    print(f"\n  Baseline MAE (test set): {baseline_mae:.2f} MW")

    results = []

    # Test different AR orders
    for p in [1, 2, 3, 6, 12, 24]:
        if p > max_ar_order:
            break

        # Simple AR(p) on residuals using OLS
        # r_t = sum(phi_k * r_{t-k}) + epsilon
        X_train = np.column_stack([train_resid[p-k-1:-k-1] for k in range(p)])
        y_train = train_resid[p:]

        # Fit AR coefficients
        try:
            phi = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        except:
            continue

        # Predict residuals on test set
        corrected_mae_list = []
        for i in range(p, len(test_resid)):
            # Use actual past residuals for prediction
            past_resids = test_resid[i-p:i][::-1]  # Most recent first
            resid_pred = np.dot(phi, past_resids)

            # Corrected prediction: original_pred + resid_pred
            corrected_pred = test_preds[i] + resid_pred
            corrected_error = abs(test_actuals[i] - corrected_pred)
            corrected_mae_list.append(corrected_error)

        corrected_mae = np.mean(corrected_mae_list)
        improvement = baseline_mae - corrected_mae

        results.append({
            'order': p,
            'mae': corrected_mae,
            'improvement': improvement,
            'pct_improvement': 100 * improvement / baseline_mae,
            'phi': phi
        })

        print(f"  AR({p:2d}): MAE={corrected_mae:.2f} MW, improvement={improvement:+.2f} MW ({100*improvement/baseline_mae:+.1f}%)")

    return results


def test_bias_correction(df, horizon):
    """Test simple bias correction."""
    residuals = df['prediction_error'].values
    actuals = df['actual_error'].values
    predictions = df['predicted_error'].values

    print(f"\n--- Bias Correction Test ---")

    # Split into train (80%) and test (20%)
    split_idx = int(len(residuals) * 0.8)
    train_resid = residuals[:split_idx]
    test_actuals = actuals[split_idx:]
    test_preds = predictions[split_idx:]
    test_resid = residuals[split_idx:]

    # Baseline MAE
    baseline_mae = np.abs(test_resid).mean()

    # Bias correction: add mean of training residuals
    bias = np.mean(train_resid)
    corrected_preds = test_preds + bias
    corrected_mae = np.abs(test_actuals - corrected_preds).mean()
    improvement = baseline_mae - corrected_mae

    print(f"  Training bias: {bias:.2f} MW")
    print(f"  Baseline MAE: {baseline_mae:.2f} MW")
    print(f"  Corrected MAE: {corrected_mae:.2f} MW")
    print(f"  Improvement: {improvement:+.2f} MW ({100*improvement/baseline_mae:+.1f}%)")

    return {
        'bias': bias,
        'baseline_mae': baseline_mae,
        'corrected_mae': corrected_mae,
        'improvement': improvement
    }


def test_hourly_bias_correction(df, horizon):
    """Test hour-specific bias correction."""
    print(f"\n--- Hourly Bias Correction Test ---")

    # Split into train (80%) and test (20%)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Compute hourly biases from training data
    hourly_bias = train_df.groupby('hour')['prediction_error'].mean()

    # Apply correction
    test_corrected = test_df.copy()
    test_corrected['bias'] = test_corrected['hour'].map(hourly_bias)
    test_corrected['corrected_pred'] = test_corrected['predicted_error'] + test_corrected['bias']

    baseline_mae = np.abs(test_df['prediction_error']).mean()
    corrected_mae = np.abs(test_df['actual_error'] - test_corrected['corrected_pred']).mean()
    improvement = baseline_mae - corrected_mae

    print(f"  Hourly biases (top 5 by magnitude):")
    for hour in hourly_bias.abs().nlargest(5).index:
        print(f"    Hour {hour:02d}: {hourly_bias[hour]:+.1f} MW")

    print(f"\n  Baseline MAE: {baseline_mae:.2f} MW")
    print(f"  Corrected MAE: {corrected_mae:.2f} MW")
    print(f"  Improvement: {improvement:+.2f} MW ({100*improvement/baseline_mae:+.1f}%)")

    return {
        'hourly_bias': hourly_bias,
        'baseline_mae': baseline_mae,
        'corrected_mae': corrected_mae,
        'improvement': improvement
    }


def test_combined_correction(df, horizon, ar_order=6):
    """Test combined bias + AR correction."""
    print(f"\n--- Combined Correction (Bias + AR({ar_order})) ---")

    residuals = df['prediction_error'].values
    actuals = df['actual_error'].values
    predictions = df['predicted_error'].values
    hours = df['hour'].values

    # Split
    split_idx = int(len(residuals) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_resid = residuals[:split_idx]
    test_resid = residuals[split_idx:]
    test_actuals = actuals[split_idx:]
    test_preds = predictions[split_idx:]
    test_hours = hours[split_idx:]

    # Step 1: Compute hourly biases
    hourly_bias = train_df.groupby('hour')['prediction_error'].mean().to_dict()

    # Step 2: Debias training residuals
    train_debiased = train_resid - np.array([hourly_bias[h] for h in hours[:split_idx]])

    # Step 3: Fit AR on debiased residuals
    p = ar_order
    X_train = np.column_stack([train_debiased[p-k-1:-k-1] for k in range(p)])
    y_train = train_debiased[p:]
    phi = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

    # Step 4: Apply combined correction on test set
    # Keep track of debiased residuals for AR
    test_debiased = test_resid - np.array([hourly_bias[h] for h in test_hours])

    baseline_mae = np.abs(test_resid).mean()
    corrected_errors = []

    for i in range(p, len(test_resid)):
        hour = test_hours[i]
        bias = hourly_bias[hour]

        # AR correction on debiased residuals
        past_debiased = test_debiased[i-p:i][::-1]
        ar_correction = np.dot(phi, past_debiased)

        # Total correction = hourly bias + AR
        total_correction = bias + ar_correction
        corrected_pred = test_preds[i] + total_correction
        corrected_errors.append(abs(test_actuals[i] - corrected_pred))

    corrected_mae = np.mean(corrected_errors)
    improvement = baseline_mae - corrected_mae

    print(f"  Baseline MAE: {baseline_mae:.2f} MW")
    print(f"  Corrected MAE: {corrected_mae:.2f} MW")
    print(f"  Improvement: {improvement:+.2f} MW ({100*improvement/baseline_mae:+.1f}%)")

    return {
        'baseline_mae': baseline_mae,
        'corrected_mae': corrected_mae,
        'improvement': improvement,
        'phi': phi
    }


def create_acf_plots(all_results, output_dir):
    """Create ACF/PACF plots for all horizons."""
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))

    for i, h in enumerate(range(1, 6)):
        res = all_results[h]
        acf_vals = res['acf']
        pacf_vals = res['pacf']
        sig = res['sig_threshold']

        # ACF
        ax_acf = axes[i, 0]
        lags = np.arange(len(acf_vals))
        ax_acf.bar(lags, acf_vals, width=0.3, color='steelblue', alpha=0.7)
        ax_acf.axhline(sig, color='red', linestyle='--', alpha=0.5)
        ax_acf.axhline(-sig, color='red', linestyle='--', alpha=0.5)
        ax_acf.axhline(0, color='black', linewidth=0.5)
        ax_acf.set_xlim(-1, min(50, len(acf_vals)))
        ax_acf.set_title(f'H+{h} ACF (lags 0-50)')
        ax_acf.set_xlabel('Lag (hours)')
        ax_acf.set_ylabel('ACF')

        # PACF
        ax_pacf = axes[i, 1]
        lags_p = np.arange(len(pacf_vals))
        ax_pacf.bar(lags_p, pacf_vals, width=0.3, color='darkgreen', alpha=0.7)
        ax_pacf.axhline(sig, color='red', linestyle='--', alpha=0.5)
        ax_pacf.axhline(-sig, color='red', linestyle='--', alpha=0.5)
        ax_pacf.axhline(0, color='black', linewidth=0.5)
        ax_pacf.set_title(f'H+{h} PACF')
        ax_pacf.set_xlabel('Lag (hours)')
        ax_pacf.set_ylabel('PACF')

    plt.tight_layout()
    plt.savefig(output_dir / '01_acf_pacf_all_horizons.png', dpi=150)
    plt.close()
    print(f"\n[+] Saved ACF/PACF plots")


def create_correction_comparison_plot(all_corrections, output_dir):
    """Create comparison of different correction methods."""
    horizons = list(all_corrections.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(horizons))
    width = 0.2

    baseline = [all_corrections[h]['baseline'] for h in horizons]
    bias = [all_corrections[h]['bias'] for h in horizons]
    hourly = [all_corrections[h]['hourly'] for h in horizons]
    ar = [all_corrections[h]['ar'] for h in horizons]
    combined = [all_corrections[h]['combined'] for h in horizons]

    ax.bar(x - 2*width, baseline, width, label='Baseline', color='gray')
    ax.bar(x - width, bias, width, label='Bias Correction', color='lightblue')
    ax.bar(x, hourly, width, label='Hourly Bias', color='steelblue')
    ax.bar(x + width, ar, width, label='AR(6)', color='lightgreen')
    ax.bar(x + 2*width, combined, width, label='Hourly + AR(6)', color='darkgreen')

    ax.set_xlabel('Horizon')
    ax.set_ylabel('MAE (MW)')
    ax.set_title('Residual Correction Methods Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'H+{h}' for h in horizons])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '02_correction_comparison.png', dpi=150)
    plt.close()
    print(f"[+] Saved correction comparison plot")


def main():
    print("="*60)
    print("RESIDUAL AUTOCORRELATION ANALYSIS")
    print("="*60)

    all_acf_results = {}
    all_corrections = {}

    for horizon in range(1, 6):
        print(f"\n{'#'*60}")
        print(f"# HORIZON H+{horizon}")
        print(f"{'#'*60}")

        df = load_predictions(horizon)

        # Analyze autocorrelation
        acf_results = analyze_residual_autocorrelation(df, horizon)
        all_acf_results[horizon] = acf_results

        # Test corrections
        bias_results = test_bias_correction(df, horizon)
        hourly_results = test_hourly_bias_correction(df, horizon)
        ar_results = test_ar_correction(df, horizon)
        combined_results = test_combined_correction(df, horizon, ar_order=6)

        # Store for plotting
        all_corrections[horizon] = {
            'baseline': bias_results['baseline_mae'],
            'bias': bias_results['corrected_mae'],
            'hourly': hourly_results['corrected_mae'],
            'ar': ar_results[2]['mae'] if len(ar_results) > 2 else bias_results['baseline_mae'],  # AR(6)
            'combined': combined_results['corrected_mae']
        }

    # Create plots
    create_acf_plots(all_acf_results, OUTPUT_DIR)
    create_correction_comparison_plot(all_corrections, OUTPUT_DIR)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Correction Methods Comparison")
    print("="*60)
    print("\n| Horizon | Baseline | Bias | Hourly | AR(6) | Combined | Best Gain |")
    print("|---------|----------|------|--------|-------|----------|-----------|")

    for h in range(1, 6):
        c = all_corrections[h]
        best = min(c['bias'], c['hourly'], c['ar'], c['combined'])
        gain = c['baseline'] - best
        pct = 100 * gain / c['baseline']
        print(f"| H+{h}     | {c['baseline']:.1f}    | {c['bias']:.1f} | {c['hourly']:.1f}  | {c['ar']:.1f} | {c['combined']:.1f}    | {gain:+.1f} ({pct:+.1f}%) |")

    print(f"\n[+] Results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
