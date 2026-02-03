"""
Master Script: Run Optuna Tuning for All Horizons
=================================================
Runs Stage 1 and Stage 2 tuning for H+1 through H+5.

Usage:
    python run_all_tuning.py --s1_trials 160 --s2_trials 80
    python run_all_tuning.py --s1_trials 2 --s2_trials 2  # Quick test
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TUNING_PATH = Path(__file__).parent


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=TUNING_PATH.parent.parent.parent.parent)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run all Optuna tuning')
    parser.add_argument('--s1_trials', type=int, default=160, help='Stage 1 trials per horizon')
    parser.add_argument('--s2_trials', type=int, default=80, help='Stage 2 trials per horizon')
    parser.add_argument('--horizons', type=str, default='1,2,3,4,5', help='Horizons to tune (comma-separated)')
    parser.add_argument('--clean', action='store_true', help='Remove existing study DBs before starting')
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(',')]

    print("=" * 70)
    print("OPTUNA TUNING - ALL HORIZONS")
    print("=" * 70)
    print(f"\nHorizons: {horizons}")
    print(f"Stage 1 trials: {args.s1_trials}")
    print(f"Stage 2 trials: {args.s2_trials}")
    print(f"Total trials: {len(horizons) * (args.s1_trials + args.s2_trials)}")

    start_time = datetime.now()
    results = {}

    for h in horizons:
        horizon_start = datetime.now()
        output_dir = TUNING_PATH / f'h{h}'
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'fold_predictions').mkdir(exist_ok=True)

        # Clean existing studies if requested
        if args.clean:
            for db_file in output_dir.glob('*.db'):
                print(f"  Removing {db_file}")
                db_file.unlink()

        # Stage 1
        s1_success = run_command(
            [sys.executable, str(TUNING_PATH / 'optuna_stage1.py'),
             '--horizon', str(h), '--n_trials', str(args.s1_trials)],
            f"H+{h} STAGE 1 ({args.s1_trials} trials)"
        )

        if not s1_success:
            print(f"\n  ERROR: Stage 1 failed for H+{h}")
            results[h] = {'status': 'FAILED', 'stage': 1}
            continue

        # Stage 2
        s2_success = run_command(
            [sys.executable, str(TUNING_PATH / 'optuna_stage2.py'),
             '--horizon', str(h), '--n_trials', str(args.s2_trials)],
            f"H+{h} STAGE 2 ({args.s2_trials} trials)"
        )

        if not s2_success:
            print(f"\n  ERROR: Stage 2 failed for H+{h}")
            results[h] = {'status': 'FAILED', 'stage': 2}
            continue

        # Load results
        s1_path = output_dir / 'stage1_best_params.json'
        s2_path = output_dir / 'stage2_best_params.json'

        if s1_path.exists() and s2_path.exists():
            with open(s1_path) as f:
                s1_results = json.load(f)
            with open(s2_path) as f:
                s2_results = json.load(f)

            results[h] = {
                'status': 'OK',
                's1_mae': s1_results['mae'],
                's2_mae': s2_results['stage2_mae'],
                'improvement': s2_results['improvement'],
                'time': (datetime.now() - horizon_start).total_seconds(),
            }
        else:
            results[h] = {'status': 'INCOMPLETE'}

    # Summary
    total_time = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  {'H':<5} {'Status':<10} {'S1 MAE':<10} {'S2 MAE':<10} {'Improv':<10} {'Time':<10}")
    print("  " + "-" * 55)

    for h in horizons:
        r = results.get(h, {})
        if r.get('status') == 'OK':
            print(f"  H+{h:<3} {'OK':<10} {r['s1_mae']:<10.2f} {r['s2_mae']:<10.2f} "
                  f"{r['improvement']:+.1f}%     {r['time']:.0f}s")
        else:
            print(f"  H+{h:<3} {r.get('status', 'UNKNOWN'):<10}")

    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save summary
    summary_path = TUNING_PATH / 'tuning_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            's1_trials': args.s1_trials,
            's2_trials': args.s2_trials,
            'total_time_seconds': total_time,
            'results': results,
        }, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
