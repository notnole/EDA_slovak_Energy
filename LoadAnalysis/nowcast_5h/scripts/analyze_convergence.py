"""
Analyze Optuna convergence to find optimal trial count.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Re-run with tracking
import sys
sys.path.insert(0, str(Path(__file__).parent))
from tune_q0_h2_optuna import load_data, evaluate_model, objective, get_baseline_params

OUTPUT_DIR = Path(__file__).parent.parent / 'analysis' / 'tuning_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_convergence():
    """Run study and analyze when we hit diminishing returns."""
    print("=" * 70)
    print("CONVERGENCE ANALYSIS: When do more trials stop helping?")
    print("=" * 70)

    # Load data
    df = load_data()

    # Get baseline
    baseline_params = get_baseline_params()
    baseline_mae = evaluate_model(baseline_params, df)
    print(f"\nBaseline MAE: {baseline_mae:.2f} MW")

    # Run 300 trials to see full convergence
    print("\nRunning 300 trials to analyze convergence...")

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )
    study.enqueue_trial(baseline_params)
    study.optimize(objective, n_trials=300, show_progress_bar=True)

    # Extract trial history
    trials = study.trials
    trial_numbers = []
    trial_values = []
    best_so_far = []

    current_best = float('inf')
    for t in trials:
        if t.value is not None:
            trial_numbers.append(t.number)
            trial_values.append(t.value)
            current_best = min(current_best, t.value)
            best_so_far.append(current_best)

    # Analyze convergence at different points
    print("\n" + "=" * 70)
    print("CONVERGENCE BY TRIAL COUNT")
    print("=" * 70)

    checkpoints = [25, 50, 75, 100, 150, 200, 250, 300]
    print(f"\n{'Trials':<10} {'Best MAE':<12} {'vs Baseline':<14} {'vs Final':<12}")
    print("-" * 50)

    final_best = min(best_so_far)
    for cp in checkpoints:
        if cp <= len(best_so_far):
            best_at_cp = best_so_far[cp - 1]
            vs_baseline = baseline_mae - best_at_cp
            vs_final = best_at_cp - final_best
            print(f"{cp:<10} {best_at_cp:<12.2f} {vs_baseline:+.2f} MW      {vs_final:+.2f} MW")

    # Find when 90%, 95%, 99% of improvement is reached
    total_improvement = baseline_mae - final_best
    print(f"\n\nTotal improvement: {total_improvement:.2f} MW")

    for pct in [0.9, 0.95, 0.99]:
        target = baseline_mae - (total_improvement * pct)
        for i, best in enumerate(best_so_far):
            if best <= target:
                print(f"  {pct*100:.0f}% of improvement reached at trial {i+1}")
                break

    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Best value over trials
    ax1 = axes[0]
    ax1.plot(trial_numbers, best_so_far, 'b-', linewidth=2, label='Best so far')
    ax1.axhline(baseline_mae, color='red', linestyle='--', label=f'Baseline ({baseline_mae:.1f})')
    ax1.axhline(final_best, color='green', linestyle='--', label=f'Final best ({final_best:.1f})')

    # Mark checkpoints
    for cp in [50, 100, 150, 200]:
        if cp <= len(best_so_far):
            ax1.axvline(cp, color='gray', linestyle=':', alpha=0.5)
            ax1.annotate(f'{best_so_far[cp-1]:.1f}', xy=(cp, best_so_far[cp-1]),
                        xytext=(cp+5, best_so_far[cp-1]+0.5), fontsize=9)

    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('MAE (MW)')
    ax1.set_title('Convergence: Best MAE Over Trials')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Marginal improvement
    ax2 = axes[1]
    windows = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    marginal_improvements = []

    for i, w in enumerate(windows):
        if w <= len(best_so_far):
            if i == 0:
                improvement = baseline_mae - best_so_far[w-1]
            else:
                prev_w = windows[i-1]
                improvement = best_so_far[prev_w-1] - best_so_far[w-1]
            marginal_improvements.append(improvement)
        else:
            marginal_improvements.append(0)

    ax2.bar(range(len(windows)), marginal_improvements, color='steelblue', edgecolor='black')
    ax2.set_xticks(range(len(windows)))
    ax2.set_xticklabels([f'{w}' for w in windows], rotation=45)
    ax2.set_xlabel('Trial Window')
    ax2.set_ylabel('Marginal Improvement (MW)')
    ax2.set_title('Marginal Improvement Per 25 Trials')
    ax2.axhline(0.1, color='red', linestyle='--', label='0.1 MW threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'convergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Plot saved to: {OUTPUT_DIR / 'convergence_analysis.png'}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Find where marginal improvement drops below 0.1 MW per 25 trials
    for i, (w, mi) in enumerate(zip(windows, marginal_improvements)):
        if mi < 0.1 and w > 50:
            recommended = windows[i-1] if i > 0 else 50
            print(f"\n  Recommended trials per model: {recommended}")
            print(f"  Reasoning: Marginal improvement drops below 0.1 MW after {w} trials")
            break
    else:
        print(f"\n  Recommended trials per model: 150-200")
        print(f"  Reasoning: Good balance of improvement vs time")

    return study, best_so_far


if __name__ == "__main__":
    study, convergence = analyze_convergence()
