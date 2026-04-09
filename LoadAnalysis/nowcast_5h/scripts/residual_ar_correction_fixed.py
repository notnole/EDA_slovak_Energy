"""
AR Correction with Correct Lag Shifting (No Data Leakage)

CORRECT UNDERSTANDING:
- At time T (end of hour T-1 to T), we know error for that hour
- For H+k model, residual at row T-k uses actual_error at (T-k)+k = T
- So shift = H (horizon), NOT H+1

Example for H+1 at T=10 (10:00 AM):
- We know error for 09:00-10:00 (just published)
- H+1 prediction at row 9 was for error at row 10 = 09:00-10:00 error
- That residual IS computable! Shift = 1.

Example for H+5 at T=10:
- H+5 prediction at row 5 was for error at row 10 = 09:00-10:00 error
- That residual IS computable! Shift = 5.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
ERROR_ANALYSIS_DIR = BASE_DIR / 'error_analysis'


def load_predictions(horizon):
    """Load predictions for a horizon."""
    df = pd.read_csv(ERROR_ANALYSIS_DIR / f'h{horizon}_predictions.csv', parse_dates=['datetime'])
    return df


def test_ar_correction_fixed(df, horizon, max_ar_order=6):
    """
    Test AR correction with CORRECT lag shifting.

    At time T (end of hour), predicting for T+H:
    - We know error for hour ending at T (just published)
    - Residual at row T-H uses actual_error at (T-H)+H = T (KNOWN!)
    - So shift = H (horizon), not H+1
    """
    residuals = df['prediction_error'].values
    actuals = df['actual_error'].values
    predictions = df['predicted_error'].values

    # Split into train (80%) and test (20%)
    split_idx = int(len(residuals) * 0.8)
    train_resid = residuals[:split_idx]
    test_resid = residuals[split_idx:]
    test_actuals = actuals[split_idx:]
    test_preds = predictions[split_idx:]

    # Baseline MAE (no correction)
    baseline_mae = np.abs(test_resid).mean()

    print(f"\n{'='*60}")
    print(f"H+{horizon} AR Correction (CORRECT SHIFT = {horizon})")
    print(f"{'='*60}")
    print(f"  Baseline MAE: {baseline_mae:.2f} MW")

    results = []

    # Minimum starting index: need H shift plus p lags
    for p in [1, 2, 3, 6]:
        if p > max_ar_order:
            break

        # At row i, we can use residuals from i-H, i-H-1, ..., i-H-p+1
        # So we need i >= H + p
        min_train_idx = horizon + p

        if min_train_idx >= len(train_resid):
            print(f"  AR({p}): Not enough training data")
            continue

        # Build training data for AR
        # We want to predict residual[i] using residual[i-H], residual[i-H-1], etc.
        X_train = []
        y_train = []

        for i in range(min_train_idx, len(train_resid)):
            target_resid = train_resid[i]
            # Most recent usable residual is at i-H, then i-H-1, etc.
            past_resids = [train_resid[i - horizon - k] for k in range(p)]
            X_train.append(past_resids)
            y_train.append(target_resid)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Fit AR coefficients
        try:
            phi = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        except:
            continue

        # Apply to test set
        min_test_idx = horizon + p
        corrected_errors = []

        for i in range(min_test_idx, len(test_resid)):
            # Past residuals with correct shift = H
            past_resids = [test_resid[i - horizon - k] for k in range(p)]
            resid_pred = np.dot(phi, past_resids)

            # Corrected prediction
            corrected_pred = test_preds[i] + resid_pred
            corrected_error = abs(test_actuals[i] - corrected_pred)
            corrected_errors.append(corrected_error)

        corrected_mae = np.mean(corrected_errors)
        improvement = baseline_mae - corrected_mae

        results.append({
            'order': p,
            'mae': corrected_mae,
            'improvement': improvement,
            'pct_improvement': 100 * improvement / baseline_mae,
            'phi': phi,
            'n_test': len(corrected_errors)
        })

        print(f"  AR({p}): MAE={corrected_mae:.2f} MW, improvement={improvement:+.2f} MW ({100*improvement/baseline_mae:+.1f}%)")

    return baseline_mae, results


def main():
    print("="*60)
    print("AR CORRECTION WITH CORRECT LAG SHIFTING")
    print("="*60)
    print("\nCorrect shift: For H+k model, shift residuals by H (not 1, not H+1)")
    print("At time T, residual at T-H uses actual at (T-H)+H = T, which is known.")

    all_results = {}

    for horizon in range(1, 6):
        df = load_predictions(horizon)
        baseline, results = test_ar_correction_fixed(df, horizon)
        all_results[horizon] = {
            'baseline': baseline,
            'ar_results': results
        }

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY: Corrected AR Results (No Data Leakage)")
    print("="*60)
    print("\n| Horizon | Baseline | AR(1) | AR(2) | AR(6) | Best Improvement |")
    print("|---------|----------|-------|-------|-------|------------------|")

    for h in range(1, 6):
        res = all_results[h]
        baseline = res['baseline']
        ar_results = {r['order']: r for r in res['ar_results']}

        ar1 = ar_results.get(1, {}).get('mae', baseline)
        ar2 = ar_results.get(2, {}).get('mae', baseline)
        ar6 = ar_results.get(6, {}).get('mae', baseline)

        best_mae = min(ar1, ar2, ar6)
        best_imp = baseline - best_mae
        best_pct = 100 * best_imp / baseline

        print(f"| H+{h}     | {baseline:.1f}    | {ar1:.1f} | {ar2:.1f} | {ar6:.1f} | {best_imp:+.1f} MW ({best_pct:+.1f}%) |")


if __name__ == '__main__':
    main()
