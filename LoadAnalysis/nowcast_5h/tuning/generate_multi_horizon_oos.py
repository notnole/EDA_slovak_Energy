"""
Generate OOS predictions for H+3, H+4, H+5 horizons.
H+2 already exists. These are needed as features for the imbalance model.
"""
import json
from pathlib import Path
import sys

# Reuse everything from the existing generator
sys.path.insert(0, str(Path(__file__).parent))
from generate_oos_predictions import load_data, predict_fold, FOLDS, OUTPUT_PATH
import pandas as pd

TUNING_PATH = Path(__file__).parent


def main():
    df = load_data()

    for horizon in [3, 4, 5]:
        print(f"\n{'=' * 70}")
        print(f"GENERATING H+{horizon} OOS PREDICTIONS")
        print(f"{'=' * 70}")

        h_dir = TUNING_PATH / f'h{horizon}'
        if not h_dir.exists():
            print(f"[!] No tuning params for H+{horizon}, skipping")
            continue

        with open(h_dir / 'stage1_best_params.json') as f:
            s1_best = json.load(f)
        with open(h_dir / 'stage2_best_params.json') as f:
            s2_best = json.load(f)

        s1_features = s1_best['features']
        s1_params = s1_best['params']
        s2_features = s2_best['s2_features']
        s2_params = s2_best['s2_params']

        print(f"  S1 features: {len(s1_features)}, S2 features: {len(s2_features)}")

        all_preds = []
        for i, (train_end, pred_start, pred_end) in enumerate(FOLDS):
            print(f"  Fold {i+1}: Train < {train_end}, Predict [{pred_start}, {pred_end})")
            fold_preds = predict_fold(
                df, horizon, train_end, pred_start, pred_end,
                s1_features, s1_params, s2_features, s2_params
            )
            if fold_preds is not None and len(fold_preds) > 0:
                mae = (fold_preds['actual_error'] - fold_preds['predicted_error']).abs().mean()
                print(f"    {len(fold_preds)} hours, MAE: {mae:.1f} MW")
                all_preds.append(fold_preds)

        oos = pd.concat(all_preds, ignore_index=True)
        oos = oos.sort_values('datetime').reset_index(drop=True)
        oos = oos.drop_duplicates(subset='datetime', keep='last')

        overall_mae = (oos['actual_error'] - oos['predicted_error']).abs().mean()
        print(f"  [+] H+{horizon} OOS: {len(oos)} hours, MAE={overall_mae:.1f} MW")

        out_path = OUTPUT_PATH / f'h{horizon}_oos_predictions.csv'
        oos.to_csv(out_path, index=False)
        print(f"  [+] Saved: {out_path}")


if __name__ == "__main__":
    main()
