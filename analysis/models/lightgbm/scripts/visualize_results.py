"""
Visualize LightGBM Model Results vs Baseline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\models\lightgbm")


def main():
    # Load results
    results = pd.read_csv(OUTPUT_DIR / 'lightgbm_results.csv')
    importance = pd.read_csv(OUTPUT_DIR / 'feature_importance.csv')

    # Baseline performance (without timestamp shift)
    baseline = {
        'lead_time': [12, 9, 6, 3, 0],
        'mae': [6.59, 4.74, 3.41, 2.52, 2.15],
        'r2': [0.307, 0.631, 0.781, 0.842, 0.860],
    }
    baseline_df = pd.DataFrame(baseline)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LightGBM vs Baseline Model Performance', fontsize=16, fontweight='bold')

    # 1. MAE Comparison
    ax = axes[0, 0]
    lead_times = results['lead_time'].values
    x = np.arange(len(lead_times))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_df['mae'], width, label='Baseline', color='tab:red', alpha=0.7)
    bars2 = ax.bar(x + width/2, results['mae'], width, label='LightGBM', color='tab:blue', alpha=0.7)

    ax.set_xlabel('Lead Time (min)')
    ax.set_ylabel('MAE (MWh)')
    ax.set_title('MAE: Baseline vs LightGBM')
    ax.set_xticks(x)
    ax.set_xticklabels(lead_times)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add improvement percentages
    for i, (b, l) in enumerate(zip(baseline_df['mae'], results['mae'])):
        improvement = (b - l) / b * 100
        ax.annotate(f'{improvement:.0f}%', xy=(x[i] + width/2, l), ha='center', va='bottom', fontsize=9, color='green')

    # 2. R² Comparison
    ax = axes[0, 1]
    bars1 = ax.bar(x - width/2, baseline_df['r2'], width, label='Baseline', color='tab:red', alpha=0.7)
    bars2 = ax.bar(x + width/2, results['r2'], width, label='LightGBM', color='tab:blue', alpha=0.7)

    ax.set_xlabel('Lead Time (min)')
    ax.set_ylabel('R²')
    ax.set_title('R²: Baseline vs LightGBM')
    ax.set_xticks(x)
    ax.set_xticklabels(lead_times)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 3. MAE by Imbalance Magnitude
    ax = axes[1, 0]
    magnitudes = ['small', 'medium', 'large', 'extreme']
    x_mag = np.arange(len(magnitudes))

    for i, lead in enumerate([12, 6, 0]):
        row = results[results['lead_time'] == lead].iloc[0]
        mae_by_mag = [row.get(f'mae_{m}', np.nan) for m in magnitudes]
        ax.plot(x_mag, mae_by_mag, 'o-', label=f'Lead {lead} min', markersize=8, linewidth=2)

    ax.set_xlabel('Imbalance Magnitude')
    ax.set_ylabel('MAE (MWh)')
    ax.set_title('MAE by Imbalance Magnitude')
    ax.set_xticks(x_mag)
    ax.set_xticklabels(['Small\n(<2 MWh)', 'Medium\n(2-5 MWh)', 'Large\n(5-10 MWh)', 'Extreme\n(>10 MWh)'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Feature Importance
    ax = axes[1, 1]
    top_features = importance.head(10)
    y_pos = np.arange(len(top_features))

    ax.barh(y_pos, top_features['importance'], color='tab:blue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Gain)')
    ax.set_title('Feature Importance (Lead 12 min model)')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lightgbm_vs_baseline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'lightgbm_vs_baseline.png'}")

    # Create second figure: Directional accuracy and hour patterns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('LightGBM Model Detailed Performance', fontsize=14, fontweight='bold')

    # 1. Directional Accuracy
    ax = axes[0]
    ax.bar(results['lead_time'], results['directional_accuracy'] * 100, color='tab:green', alpha=0.7)
    ax.set_xlabel('Lead Time (min)')
    ax.set_ylabel('Directional Accuracy (%)')
    ax.set_title('Sign Prediction Accuracy')
    ax.set_ylim(50, 100)
    ax.axhline(50, color='red', linestyle='--', label='Random')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for i, (lead, acc) in enumerate(zip(results['lead_time'], results['directional_accuracy'])):
        ax.annotate(f'{acc*100:.1f}%', xy=(lead, acc*100), ha='center', va='bottom', fontsize=10)

    # 2. MAE by Hour Group
    ax = axes[1]
    hour_groups = ['night', 'morning', 'peak', 'afternoon', 'evening']
    x_hg = np.arange(len(hour_groups))
    width = 0.15

    for i, lead in enumerate([12, 9, 6, 3, 0]):
        row = results[results['lead_time'] == lead].iloc[0]
        mae_by_hour = [row.get(f'mae_{hg}', np.nan) for hg in hour_groups]
        ax.bar(x_hg + (i - 2) * width, mae_by_hour, width, label=f'Lead {lead}', alpha=0.8)

    ax.set_xlabel('Hour Group')
    ax.set_ylabel('MAE (MWh)')
    ax.set_title('MAE by Hour Group')
    ax.set_xticks(x_hg)
    ax.set_xticklabels(['Night\n(0-5h)', 'Morning\n(6-10h)', 'Peak\n(11-14h)', 'Afternoon\n(15-20h)', 'Evening\n(21-23h)'])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lightgbm_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'lightgbm_detailed.png'}")

    # Print summary table
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    print(f"\n{'Lead':<8} {'Baseline':<12} {'LightGBM':<12} {'Improvement':<12} {'R² Gain':<12} {'Dir. Acc':<10}")
    print(f"{'Time':<8} {'MAE':<12} {'MAE':<12} {'%':<12} {'%':<12} {'%':<10}")
    print("-" * 66)

    for i, row in results.iterrows():
        lead = int(row['lead_time'])
        bl_mae = baseline_df[baseline_df['lead_time'] == lead]['mae'].values[0]
        bl_r2 = baseline_df[baseline_df['lead_time'] == lead]['r2'].values[0]
        lgb_mae = row['mae']
        lgb_r2 = row['r2']
        improvement = (bl_mae - lgb_mae) / bl_mae * 100
        r2_gain = (lgb_r2 - bl_r2) / bl_r2 * 100
        dir_acc = row['directional_accuracy'] * 100

        print(f"{lead:<8} {bl_mae:<12.2f} {lgb_mae:<12.2f} {improvement:<+12.1f} {r2_gain:<+12.1f} {dir_acc:<10.1f}")


if __name__ == '__main__':
    main()
