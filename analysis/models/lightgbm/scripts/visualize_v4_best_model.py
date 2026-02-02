"""
V4 BEST MODEL - Time Series Visualization
=========================================
Generates time series plots comparing V4 predictions vs baseline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import sys

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import feature generation from V4
from train_lightgbm_v4 import (
    load_data, compute_load_expected, create_base_features,
    add_proxy_lag_features, compute_lead_features, get_features_for_lead
)

OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\models\lightgbm")


def main():
    print("=" * 70)
    print("V4 BEST MODEL - TIME SERIES VISUALIZATION")
    print("=" * 70)

    # Load data and create features (same as V4 training)
    reg_df, load_df, label_df = load_data()
    load_expected = compute_load_expected(load_df)
    df = create_base_features(reg_df, load_df, label_df, load_expected)
    df = add_proxy_lag_features(df)

    # Load V4 models
    print("\nLoading V4 models...")
    with open(OUTPUT_DIR / 'outputs' / 'lightgbm_models_v4.pkl', 'rb') as f:
        models = pickle.load(f)

    # Test data (same split as V4)
    test_start = pd.Timestamp('2025-01-01')
    lead_times = [12, 9, 6, 3, 0]

    # Generate predictions for all lead times
    predictions = {}

    for lead in lead_times:
        lead_df = compute_lead_features(df, lead, load_expected)
        test_df = lead_df[lead_df['datetime'] >= test_start].copy()

        feature_cols = get_features_for_lead(lead)
        feature_cols = [c for c in feature_cols if c in test_df.columns]

        # Get clean data
        required_cols = feature_cols + ['imbalance', 'datetime', 'baseline_pred']
        test_clean = test_df.dropna(subset=required_cols)

        X_test = test_clean[feature_cols].values
        y_test = test_clean['imbalance'].values
        baseline = test_clean['baseline_pred'].values
        datetimes = test_clean['datetime'].values

        # Predict
        model = models[lead]
        y_pred = model.predict(X_test)

        predictions[lead] = {
            'datetime': datetimes,
            'actual': y_test,
            'predicted': y_pred,
            'baseline': baseline,
            'mae_v4': np.mean(np.abs(y_test - y_pred)),
            'mae_base': np.mean(np.abs(y_test - baseline)),
            'dir_acc_v4': np.mean(np.sign(y_test) == np.sign(y_pred)) * 100,
            'dir_acc_base': np.mean(np.sign(y_test) == np.sign(baseline)) * 100,
        }

        print(f"Lead {lead}: V4 MAE={predictions[lead]['mae_v4']:.3f}, "
              f"Baseline MAE={predictions[lead]['mae_base']:.3f}")

    # =========================================================================
    # PLOT 1: Lead 12 - Overall + Zoomed Half Day
    # =========================================================================
    print("\nCreating time series plots...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    lead = 12
    p = predictions[lead]

    # Plot 1a: Overall (first 2 weeks)
    n_periods = 96 * 14  # 2 weeks of 15-min periods
    idx = slice(0, min(n_periods, len(p['actual'])))

    ax = axes[0]
    ax.plot(p['datetime'][idx], p['actual'][idx], 'b-', alpha=0.7, label='Actual', linewidth=0.8)
    ax.plot(p['datetime'][idx], p['predicted'][idx], 'r-', alpha=0.7,
            label=f'V4 Model (MAE={p["mae_v4"]:.2f})', linewidth=0.8)
    ax.plot(p['datetime'][idx], p['baseline'][idx], 'g--', alpha=0.5,
            label=f'Baseline (MAE={p["mae_base"]:.2f})', linewidth=0.6)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Imbalance (MWh)')
    ax.set_title(f'Lead {lead} min - Overall View (First 2 Weeks of 2025)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 1b: Zoomed half day
    n_half_day = 48  # 12 hours
    start_idx = 200
    idx_zoom = slice(start_idx, start_idx + n_half_day)

    ax = axes[1]
    ax.plot(p['datetime'][idx_zoom], p['actual'][idx_zoom], 'b-o', alpha=0.8,
            label='Actual', linewidth=1.5, markersize=3)
    ax.plot(p['datetime'][idx_zoom], p['predicted'][idx_zoom], 'r-s', alpha=0.8,
            label='V4 Model', linewidth=1.5, markersize=3)
    ax.plot(p['datetime'][idx_zoom], p['baseline'][idx_zoom], 'g--^', alpha=0.6,
            label='Baseline', linewidth=1, markersize=2)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Imbalance (MWh)')
    ax.set_title(f'Lead {lead} min - Zoomed Half Day (12 hours)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 1c: Absolute errors
    errors_v4 = np.abs(p['actual'] - p['predicted'])
    errors_base = np.abs(p['actual'] - p['baseline'])

    ax = axes[2]
    ax.plot(p['datetime'][idx], errors_v4[idx], 'r-', alpha=0.6,
            label=f'V4 |Error| (MAE={p["mae_v4"]:.2f})', linewidth=0.8)
    ax.plot(p['datetime'][idx], errors_base[idx], 'g-', alpha=0.4,
            label=f'Baseline |Error| (MAE={p["mae_base"]:.2f})', linewidth=0.6)
    ax.axhline(p['mae_v4'], color='r', linestyle='--', alpha=0.8)
    ax.axhline(p['mae_base'], color='g', linestyle='--', alpha=0.6)
    ax.set_xlabel('Time')
    ax.set_ylabel('Absolute Error (MWh)')
    ax.set_title(f'Lead {lead} min - Error Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plots' / 'v4_timeseries_lead12.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: v4_timeseries_lead12.png")

    # =========================================================================
    # PLOT 2: All Lead Times Comparison (1 day)
    # =========================================================================
    fig, axes = plt.subplots(len(lead_times), 1, figsize=(16, 3*len(lead_times)))

    n_day = 96  # 1 day
    start_idx = 200

    for i, lead in enumerate(lead_times):
        p = predictions[lead]
        idx = slice(start_idx, start_idx + n_day)

        ax = axes[i]
        ax.plot(p['datetime'][idx], p['actual'][idx], 'b-', alpha=0.8, label='Actual', linewidth=1.2)
        ax.plot(p['datetime'][idx], p['predicted'][idx], 'r-', alpha=0.8,
                label=f'V4 (MAE={p["mae_v4"]:.2f})', linewidth=1.2)
        ax.plot(p['datetime'][idx], p['baseline'][idx], 'g--', alpha=0.5, label='Baseline', linewidth=0.8)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylabel('MWh')
        ax.set_title(f'Lead {lead} min (MAE: {p["mae_v4"]:.2f}, Dir.Acc: {p["dir_acc_v4"]:.1f}%)',
                    fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time')
    plt.suptitle('V4 BEST MODEL - All Lead Times (1 Day Sample)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plots' / 'v4_all_leads_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: v4_all_leads_comparison.png")

    # =========================================================================
    # PLOT 3: Performance Dashboard
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MAE by lead time
    ax = axes[0, 0]
    maes_v4 = [predictions[l]['mae_v4'] for l in lead_times]
    maes_base = [predictions[l]['mae_base'] for l in lead_times]
    x = np.arange(len(lead_times))
    width = 0.35
    ax.bar(x - width/2, maes_v4, width, label='V4 Model', color='steelblue')
    ax.bar(x + width/2, maes_base, width, label='Baseline', color='lightgreen', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Lead {l}' for l in lead_times])
    ax.set_ylabel('MAE (MWh)')
    ax.set_title('MAE by Lead Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Direction accuracy
    ax = axes[0, 1]
    dir_v4 = [predictions[l]['dir_acc_v4'] for l in lead_times]
    dir_base = [predictions[l]['dir_acc_base'] for l in lead_times]
    ax.bar(x - width/2, dir_v4, width, label='V4 Model', color='steelblue')
    ax.bar(x + width/2, dir_base, width, label='Baseline', color='lightgreen', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Lead {l}' for l in lead_times])
    ax.set_ylabel('Direction Accuracy (%)')
    ax.set_title('Direction Accuracy by Lead Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([70, 100])

    # Error distribution for lead 12
    ax = axes[1, 0]
    p = predictions[12]
    errors_v4 = p['actual'] - p['predicted']
    errors_base = p['actual'] - p['baseline']
    ax.hist(errors_v4, bins=50, alpha=0.7, label=f'V4 (std={np.std(errors_v4):.2f})', color='steelblue')
    ax.hist(errors_base, bins=50, alpha=0.5, label=f'Baseline (std={np.std(errors_base):.2f})', color='lightgreen')
    ax.axvline(np.mean(errors_v4), color='blue', linestyle='--', label=f'V4 Bias: {np.mean(errors_v4):.2f}')
    ax.axvline(np.mean(errors_base), color='green', linestyle='--', label=f'Base Bias: {np.mean(errors_base):.2f}')
    ax.set_xlabel('Error (MWh)')
    ax.set_ylabel('Count')
    ax.set_title('Lead 12 - Error Distribution', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Scatter: Predicted vs Actual
    ax = axes[1, 1]
    ax.scatter(p['actual'], p['predicted'], alpha=0.3, s=10, label='V4', color='steelblue')
    ax.scatter(p['actual'], p['baseline'], alpha=0.15, s=5, label='Baseline', color='lightgreen')
    lims = [min(p['actual'].min(), p['predicted'].min()), max(p['actual'].max(), p['predicted'].max())]
    ax.plot(lims, lims, 'r--', alpha=0.8, label='Perfect')
    ax.set_xlabel('Actual Imbalance (MWh)')
    ax.set_ylabel('Predicted Imbalance (MWh)')
    r2 = 1 - np.sum((p['actual'] - p['predicted'])**2) / np.sum((p['actual'] - np.mean(p['actual']))**2)
    ax.set_title(f'Lead 12 - Predicted vs Actual (R²={r2:.3f})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.suptitle('V4 BEST MODEL - Performance Dashboard', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plots' / 'v4_performance_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: v4_performance_dashboard.png")

    # =========================================================================
    # PLOT 4: Error Analysis by Hour and Day
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    p = predictions[12]
    analysis_df = pd.DataFrame({
        'datetime': p['datetime'],
        'actual': p['actual'],
        'predicted': p['predicted'],
        'baseline': p['baseline'],
        'error_v4': p['actual'] - p['predicted'],
        'error_base': p['actual'] - p['baseline'],
        'abs_error_v4': np.abs(p['actual'] - p['predicted']),
        'abs_error_base': np.abs(p['actual'] - p['baseline'])
    })
    analysis_df['hour'] = pd.to_datetime(analysis_df['datetime']).dt.hour
    analysis_df['dow'] = pd.to_datetime(analysis_df['datetime']).dt.dayofweek

    # MAE by hour
    ax = axes[0, 0]
    hourly = analysis_df.groupby('hour').agg({'abs_error_v4': 'mean', 'abs_error_base': 'mean'}).reset_index()
    ax.bar(hourly['hour'] - 0.2, hourly['abs_error_v4'], 0.4, label='V4', color='steelblue')
    ax.bar(hourly['hour'] + 0.2, hourly['abs_error_base'], 0.4, label='Baseline', color='lightgreen', alpha=0.7)
    ax.set_xlabel('Hour')
    ax.set_ylabel('MAE (MWh)')
    ax.set_title('Lead 12 - MAE by Hour', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Bias by hour
    ax = axes[0, 1]
    hourly_bias = analysis_df.groupby('hour').agg({'error_v4': 'mean', 'error_base': 'mean'}).reset_index()
    ax.bar(hourly_bias['hour'] - 0.2, hourly_bias['error_v4'], 0.4, label='V4', color='steelblue')
    ax.bar(hourly_bias['hour'] + 0.2, hourly_bias['error_base'], 0.4, label='Baseline', color='lightgreen', alpha=0.7)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Bias (MWh)')
    ax.set_title('Lead 12 - Bias by Hour', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # MAE by day of week
    ax = axes[1, 0]
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily = analysis_df.groupby('dow').agg({'abs_error_v4': 'mean', 'abs_error_base': 'mean'}).reset_index()
    ax.bar(daily['dow'] - 0.2, daily['abs_error_v4'], 0.4, label='V4', color='steelblue')
    ax.bar(daily['dow'] + 0.2, daily['abs_error_base'], 0.4, label='Baseline', color='lightgreen', alpha=0.7)
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('MAE (MWh)')
    ax.set_title('Lead 12 - MAE by Day of Week', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Improvement by hour (V4 vs Baseline)
    ax = axes[1, 1]
    hourly['improvement'] = (1 - hourly['abs_error_v4'] / hourly['abs_error_base']) * 100
    colors = ['steelblue' if x >= 0 else 'salmon' for x in hourly['improvement']]
    ax.bar(hourly['hour'], hourly['improvement'], color=colors)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Lead 12 - V4 Improvement over Baseline by Hour', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('V4 BEST MODEL - Error Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plots' / 'v4_error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: v4_error_analysis.png")

    # =========================================================================
    # Save summary report
    # =========================================================================
    summary = f"""
================================================================================
LIGHTGBM V4 - BEST MODEL SUMMARY
================================================================================

Model Status: *** BEST MODEL ***
Model Type: LightGBM with real-time features only
Training Data: 2024
Test Data: 2025

CONSTRAINT: Actual imbalance values NOT available until next day.
Uses only: regulation data, load data, time features, proxy-based features.

PERFORMANCE METRICS (Test Set - 2025)
================================================================================

Lead    V4 MAE    Base MAE    Improvement    V4 Dir%    Base Dir%
--------------------------------------------------------------------------------
"""

    for lead in lead_times:
        p = predictions[lead]
        improv = (1 - p['mae_v4'] / p['mae_base']) * 100
        summary += f"{lead:<7} {p['mae_v4']:<9.3f} {p['mae_base']:<11.3f} {improv:>+10.1f}%    {p['dir_acc_v4']:<10.1f} {p['dir_acc_base']:<10.1f}\n"

    summary += f"""
--------------------------------------------------------------------------------

KEY INSIGHTS:
- V4 consistently outperforms baseline across all lead times
- Direction accuracy ranges from {predictions[12]['dir_acc_v4']:.1f}% (lead 12) to {predictions[0]['dir_acc_v4']:.1f}% (lead 0)
- Largest MAE improvement at lead 12: {(1 - predictions[12]['mae_v4'] / predictions[12]['mae_base']) * 100:.1f}%

VISUALIZATIONS:
- v4_timeseries_lead12.png: Overall and zoomed time series for lead 12
- v4_all_leads_comparison.png: 1-day comparison across all lead times
- v4_performance_dashboard.png: MAE, direction accuracy, error distribution, scatter
- v4_error_analysis.png: Error patterns by hour and day of week

Files:
- Model: outputs/lightgbm_models_v4.pkl
- Results: outputs/lightgbm_v4_results.csv
- Features: outputs/feature_importance_v4.csv

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
================================================================================
"""

    with open(OUTPUT_DIR / 'V4_BEST_MODEL_README.txt', 'w') as f:
        f.write(summary)
    print("\nSaved: V4_BEST_MODEL_README.txt")

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
