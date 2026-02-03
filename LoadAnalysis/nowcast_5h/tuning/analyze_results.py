"""
Analyze and Visualize Optuna Tuning Results
============================================
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TUNING_PATH = Path(__file__).parent


def load_all_results():
    """Load results for all horizons."""
    results = {}

    for h in range(1, 6):
        h_dir = TUNING_PATH / f'h{h}'
        s1_path = h_dir / 'stage1_best_params.json'
        s2_path = h_dir / 'stage2_best_params.json'

        if s1_path.exists() and s2_path.exists():
            with open(s1_path) as f:
                s1 = json.load(f)
            with open(s2_path) as f:
                s2 = json.load(f)

            results[h] = {
                's1_mae': s1['mae'],
                's1_fold_maes': s1['fold_maes'],
                's1_features': s1['features'],
                's1_params': s1['params'],
                's2_mae': s2['stage2_mae'],
                's2_fold_maes': s2['fold_maes'],
                's2_features': s2['s2_features'],
                's2_params': s2['s2_params'],
                'improvement': s2['improvement'],
            }

    return results


def print_summary(results):
    """Print summary table."""
    print("=" * 80)
    print("TUNING RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Horizon':<10} {'S1 MAE':<12} {'S2 MAE':<12} {'Improvement':<12} {'S2 Gain':<12}")
    print("-" * 58)

    for h in sorted(results.keys()):
        r = results[h]
        s2_gain = (r['s1_mae'] - r['s2_mae']) / r['s1_mae'] * 100
        print(f"H+{h:<8} {r['s1_mae']:<12.2f} {r['s2_mae']:<12.2f} {r['improvement']:+.1f}%       {s2_gain:+.1f}%")

    # Average
    avg_s1 = np.mean([r['s1_mae'] for r in results.values()])
    avg_s2 = np.mean([r['s2_mae'] for r in results.values()])
    avg_imp = np.mean([r['improvement'] for r in results.values()])
    print("-" * 58)
    print(f"{'Average':<10} {avg_s1:<12.2f} {avg_s2:<12.2f} {avg_imp:+.1f}%")


def print_feature_analysis(results):
    """Analyze features selected for each horizon."""
    print("\n" + "=" * 80)
    print("FEATURE ANALYSIS")
    print("=" * 80)

    # Stage 1 features
    print("\n--- STAGE 1 FEATURES ---")
    all_s1_features = set()
    for h, r in results.items():
        all_s1_features.update(r['s1_features'])

    # Count how often each feature is used
    feature_counts = {}
    for feat in all_s1_features:
        count = sum(1 for r in results.values() if feat in r['s1_features'])
        feature_counts[feat] = count

    print(f"\n{'Feature':<30} {'Used in':<15} {'Horizons'}")
    print("-" * 60)
    for feat, count in sorted(feature_counts.items(), key=lambda x: -x[1]):
        horizons = [f"H+{h}" for h, r in results.items() if feat in r['s1_features']]
        print(f"{feat:<30} {count}/5            {', '.join(horizons)}")

    # Stage 2 features
    print("\n--- STAGE 2 FEATURES ---")
    all_s2_features = set()
    for h, r in results.items():
        all_s2_features.update(r['s2_features'])

    feature_counts = {}
    for feat in all_s2_features:
        count = sum(1 for r in results.values() if feat in r['s2_features'])
        feature_counts[feat] = count

    print(f"\n{'Feature':<30} {'Used in':<15} {'Horizons'}")
    print("-" * 60)
    for feat, count in sorted(feature_counts.items(), key=lambda x: -x[1]):
        horizons = [f"H+{h}" for h, r in results.items() if feat in r['s2_features']]
        print(f"{feat:<30} {count}/5            {', '.join(horizons)}")


def print_hyperparameter_analysis(results):
    """Analyze hyperparameters across horizons."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER ANALYSIS")
    print("=" * 80)

    # Stage 1 key params
    print("\n--- STAGE 1 KEY HYPERPARAMETERS ---")
    print(f"\n{'Horizon':<10} {'n_est':<8} {'lr':<10} {'depth':<8} {'leaves':<8} {'recency_m':<12} {'recency_w':<10}")
    print("-" * 70)

    for h in sorted(results.keys()):
        p = results[h]['s1_params']
        print(f"H+{h:<8} {p['n_estimators']:<8} {p['learning_rate']:<10.4f} {p['max_depth']:<8} "
              f"{p['num_leaves']:<8} {p['recency_months']:<12} {p['recency_weight']:<10.2f}")

    # Stage 2 key params
    print("\n--- STAGE 2 KEY HYPERPARAMETERS ---")
    print(f"\n{'Horizon':<10} {'n_est':<8} {'lr':<10} {'depth':<8} {'leaves':<8} {'recency_m':<12} {'recency_w':<10}")
    print("-" * 70)

    for h in sorted(results.keys()):
        p = results[h]['s2_params']
        print(f"H+{h:<8} {p['s2_n_estimators']:<8} {p['s2_learning_rate']:<10.4f} {p['s2_max_depth']:<8} "
              f"{p['s2_num_leaves']:<8} {p['s2_recency_months']:<12} {p['s2_recency_weight']:<10.2f}")


def plot_results(results):
    """Create visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    horizons = sorted(results.keys())
    s1_maes = [results[h]['s1_mae'] for h in horizons]
    s2_maes = [results[h]['s2_mae'] for h in horizons]
    improvements = [results[h]['improvement'] for h in horizons]

    # Plot 1: MAE by horizon
    ax1 = axes[0, 0]
    x = np.arange(len(horizons))
    width = 0.35
    ax1.bar(x - width/2, s1_maes, width, label='Stage 1', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, s2_maes, width, label='Stage 2', color='darkorange', alpha=0.8)
    ax1.set_xlabel('Horizon')
    ax1.set_ylabel('MAE (MW)')
    ax1.set_title('MAE by Horizon')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'H+{h}' for h in horizons])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (s1, s2) in enumerate(zip(s1_maes, s2_maes)):
        ax1.text(i - width/2, s1 + 1, f'{s1:.1f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, s2 + 1, f'{s2:.1f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Stage 2 improvement
    ax2 = axes[0, 1]
    colors = ['green' if imp > 10 else 'orange' if imp > 5 else 'red' for imp in improvements]
    bars = ax2.bar(horizons, improvements, color=colors, alpha=0.8)
    ax2.set_xlabel('Horizon')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Stage 2 Improvement over Stage 1')
    ax2.set_xticks(horizons)
    ax2.set_xticklabels([f'H+{h}' for h in horizons])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='y', alpha=0.3)

    for bar, imp in zip(bars, improvements):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{imp:.1f}%', ha='center', va='bottom', fontsize=10)

    # Plot 3: Fold MAEs for Stage 2
    ax3 = axes[1, 0]
    fold_labels = ['Fold 1\n(H1 2025)', 'Fold 2\n(Q3 2025)', 'Fold 3\n(Q4 2025)']
    for h in horizons:
        ax3.plot(fold_labels, results[h]['s2_fold_maes'], marker='o', label=f'H+{h}')
    ax3.set_xlabel('Validation Fold')
    ax3.set_ylabel('MAE (MW)')
    ax3.set_title('Stage 2 MAE by CV Fold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: Feature count per horizon
    ax4 = axes[1, 1]
    s1_feat_counts = [len(results[h]['s1_features']) for h in horizons]
    s2_feat_counts = [len(results[h]['s2_features']) for h in horizons]

    x = np.arange(len(horizons))
    ax4.bar(x - width/2, s1_feat_counts, width, label='Stage 1', color='steelblue', alpha=0.8)
    ax4.bar(x + width/2, s2_feat_counts, width, label='Stage 2', color='darkorange', alpha=0.8)
    ax4.set_xlabel('Horizon')
    ax4.set_ylabel('Number of Features')
    ax4.set_title('Selected Features Count')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'H+{h}' for h in horizons])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = TUNING_PATH / 'tuning_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

    plt.show()


def main():
    print("Loading results...")
    results = load_all_results()

    if not results:
        print("No results found!")
        return

    print_summary(results)
    print_feature_analysis(results)
    print_hyperparameter_analysis(results)

    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    plot_results(results)


if __name__ == '__main__':
    main()
