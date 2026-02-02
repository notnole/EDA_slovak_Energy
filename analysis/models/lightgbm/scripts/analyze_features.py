"""
Analyze features for LightGBM model optimization.

1. Compute baseline directional accuracy
2. Analyze feature importance per lead time
3. Identify features to include/exclude per lead time
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\models\lightgbm")


def load_data():
    """Load regulation and imbalance data."""
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
        columns={'System Imbalance (MWh)': 'imbalance'}
    )
    return reg_df, label_df


def create_baseline_predictions(reg_df, label_df):
    """Create baseline predictions for all lead times."""
    # Align regulation to settlement periods
    reg_df = reg_df.copy()
    reg_df['datetime_floor'] = reg_df['datetime'].dt.floor('3min')
    reg_df['settlement_end'] = reg_df['datetime_floor'].dt.ceil('15min')
    mask = reg_df['datetime_floor'] == reg_df['settlement_end']
    reg_df.loc[mask, 'settlement_end'] = reg_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    reg_df['settlement_start'] = reg_df['settlement_end'] - pd.Timedelta(minutes=15)
    reg_df['minute_in_qh'] = (reg_df['datetime_floor'] - reg_df['settlement_start']).dt.total_seconds() / 60

    # Pivot
    pivot = reg_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='regulation_mw', aggfunc='first'
    ).reset_index()
    pivot.columns = ['datetime'] + [f'reg_min{int(c)}' for c in pivot.columns[1:]]

    # Merge with labels
    df = pd.merge(label_df, pivot, on='datetime', how='inner')

    # Compute baseline predictions for each lead time
    results = []

    for lead in [12, 9, 6, 3, 0]:
        row_data = df.copy()
        row_data['lead_time'] = lead

        if lead == 12:
            row_data['baseline_pred'] = -0.25 * df['reg_min0']
        elif lead == 9:
            row_data['baseline_pred'] = -0.25 * (0.8 * df['reg_min3'] + 0.2 * df['reg_min0'])
        elif lead == 6:
            row_data['baseline_pred'] = -0.25 * (0.6 * df['reg_min6'] + 0.2 * df['reg_min3'] + 0.2 * df['reg_min0'])
        elif lead == 3:
            row_data['baseline_pred'] = -0.25 * (0.4 * df['reg_min9'] + 0.2 * df['reg_min6'] +
                                                  0.2 * df['reg_min3'] + 0.2 * df['reg_min0'])
        elif lead == 0:
            reg_cols = ['reg_min0', 'reg_min3', 'reg_min6', 'reg_min9', 'reg_min12']
            row_data['baseline_pred'] = -0.25 * df[reg_cols].mean(axis=1)

        results.append(row_data[['datetime', 'imbalance', 'baseline_pred', 'lead_time']])

    return pd.concat(results, ignore_index=True)


def compute_baseline_metrics(df, test_start='2025-10-01'):
    """Compute baseline metrics including directional accuracy."""
    test_start = pd.Timestamp(test_start)
    test_df = df[df['datetime'] >= test_start].copy()

    results = []

    for lead in [12, 9, 6, 3, 0]:
        lead_df = test_df[test_df['lead_time'] == lead].dropna()

        y_true = lead_df['imbalance'].values
        y_pred = lead_df['baseline_pred'].values

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        dir_acc = (np.sign(y_pred) == np.sign(y_true)).mean()

        results.append({
            'lead_time': lead,
            'baseline_mae': mae,
            'baseline_r2': r2,
            'baseline_dir_acc': dir_acc,
            'n_samples': len(y_true)
        })

    return pd.DataFrame(results)


def analyze_feature_importance_per_lead():
    """Load models and analyze feature importance per lead time."""
    with open(OUTPUT_DIR / 'lightgbm_models.pkl', 'rb') as f:
        models = pickle.load(f)

    feature_cols = [
        'reg_cumulative_mean', 'baseline_pred', 'load_deviation',
        'reg_std', 'reg_range', 'reg_trend',
        'hour_sin', 'hour_cos', 'is_weekend',
        'imb_proxy_lag1', 'imb_proxy_lag2',
        'imb_proxy_rolling_mean4', 'imb_proxy_rolling_std4',
    ]

    importance_data = []

    for lead in [12, 9, 6, 3, 0]:
        model = models[lead]
        importance = model.feature_importance(importance_type='gain')

        for feat, imp in zip(feature_cols, importance):
            importance_data.append({
                'lead_time': lead,
                'feature': feat,
                'importance': imp
            })

    return pd.DataFrame(importance_data)


def main():
    print("=" * 70)
    print("FEATURE ANALYSIS FOR LIGHTGBM OPTIMIZATION")
    print("=" * 70)

    # 1. Compute baseline directional accuracy
    print("\n[1] Computing baseline directional accuracy...")
    reg_df, label_df = load_data()
    baseline_df = create_baseline_predictions(reg_df, label_df)
    baseline_metrics = compute_baseline_metrics(baseline_df)

    print("\n" + "=" * 70)
    print("BASELINE vs LIGHTGBM COMPARISON (with Directional Accuracy)")
    print("=" * 70)

    # Load LightGBM results
    lgb_results = pd.read_csv(OUTPUT_DIR / 'lightgbm_results.csv')

    print(f"\n{'Lead':<8} {'Baseline':<12} {'LightGBM':<12} {'Baseline':<12} {'LightGBM':<12} {'Baseline':<12} {'LightGBM':<12}")
    print(f"{'Time':<8} {'MAE':<12} {'MAE':<12} {'R²':<12} {'R²':<12} {'Dir.Acc':<12} {'Dir.Acc':<12}")
    print("-" * 80)

    for _, bl_row in baseline_metrics.iterrows():
        lead = int(bl_row['lead_time'])
        lgb_row = lgb_results[lgb_results['lead_time'] == lead].iloc[0]

        print(f"{lead:<8} {bl_row['baseline_mae']:<12.2f} {lgb_row['mae']:<12.2f} "
              f"{bl_row['baseline_r2']:<12.3f} {lgb_row['r2']:<12.3f} "
              f"{bl_row['baseline_dir_acc']*100:<12.1f} {lgb_row['directional_accuracy']*100:<12.1f}")

    # 2. Analyze feature importance per lead time
    print("\n\n" + "=" * 70)
    print("FEATURE IMPORTANCE BY LEAD TIME")
    print("=" * 70)

    importance_df = analyze_feature_importance_per_lead()

    # Pivot to show importance across lead times
    pivot = importance_df.pivot(index='feature', columns='lead_time', values='importance')
    pivot = pivot[[12, 9, 6, 3, 0]]  # Order columns

    # Calculate total importance and sort
    pivot['total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('total', ascending=False)

    # Normalize per lead time (percentage)
    pivot_pct = pivot.copy()
    for lead in [12, 9, 6, 3, 0]:
        col_sum = pivot[lead].sum()
        if col_sum > 0:
            pivot_pct[lead] = pivot[lead] / col_sum * 100

    print("\nAbsolute Importance (Gain):")
    print("-" * 90)
    print(f"{'Feature':<30} {'Lead 12':>12} {'Lead 9':>12} {'Lead 6':>12} {'Lead 3':>12} {'Lead 0':>12}")
    print("-" * 90)

    for feat in pivot.index:
        row = pivot.loc[feat]
        print(f"{feat:<30} {row[12]:>12,.0f} {row[9]:>12,.0f} {row[6]:>12,.0f} {row[3]:>12,.0f} {row[0]:>12,.0f}")

    print("\n\nRelative Importance (%):")
    print("-" * 90)
    print(f"{'Feature':<30} {'Lead 12':>12} {'Lead 9':>12} {'Lead 6':>12} {'Lead 3':>12} {'Lead 0':>12}")
    print("-" * 90)

    for feat in pivot_pct.index:
        row = pivot_pct.loc[feat]
        print(f"{feat:<30} {row[12]:>11.1f}% {row[9]:>11.1f}% {row[6]:>11.1f}% {row[3]:>11.1f}% {row[0]:>11.1f}%")

    # 3. Feature recommendations
    print("\n\n" + "=" * 70)
    print("FEATURE RECOMMENDATIONS BY LEAD TIME")
    print("=" * 70)

    # Identify zero-importance features per lead time
    print("\nZero-importance features (candidates for removal):")
    for lead in [12, 9, 6, 3, 0]:
        zero_feats = importance_df[(importance_df['lead_time'] == lead) &
                                    (importance_df['importance'] == 0)]['feature'].tolist()
        if zero_feats:
            print(f"  Lead {lead}: {', '.join(zero_feats)}")

    # Identify low-importance features (<1% contribution)
    print("\nLow-importance features (<1% of total gain):")
    for lead in [12, 9, 6, 3, 0]:
        lead_data = importance_df[importance_df['lead_time'] == lead]
        total = lead_data['importance'].sum()
        low_feats = lead_data[lead_data['importance'] < total * 0.01]['feature'].tolist()
        # Exclude already zero features
        low_feats = [f for f in low_feats if f not in importance_df[(importance_df['lead_time'] == lead) &
                                                                     (importance_df['importance'] == 0)]['feature'].tolist()]
        if low_feats:
            print(f"  Lead {lead}: {', '.join(low_feats)}")

    # High-importance features (>5% contribution)
    print("\nHigh-importance features (>5% of total gain):")
    for lead in [12, 9, 6, 3, 0]:
        lead_data = importance_df[importance_df['lead_time'] == lead]
        total = lead_data['importance'].sum()
        high_feats = lead_data[lead_data['importance'] > total * 0.05]['feature'].tolist()
        print(f"  Lead {lead}: {', '.join(high_feats)}")

    # 4. Feature availability analysis
    print("\n\n" + "=" * 70)
    print("FEATURE AVAILABILITY BY LEAD TIME")
    print("=" * 70)

    print("""
    Feature                  | Lead 12 | Lead 9 | Lead 6 | Lead 3 | Lead 0 | Notes
    -------------------------|---------|--------|--------|--------|--------|------------------------
    reg_cumulative_mean      |    ✓    |   ✓    |   ✓    |   ✓    |   ✓    | Mean of available obs
    baseline_pred            |    ✓    |   ✓    |   ✓    |   ✓    |   ✓    | Weighted by recency
    reg_std                  |    ✗    |   ✓    |   ✓    |   ✓    |   ✓    | Needs 2+ observations
    reg_range                |    ✗    |   ✓    |   ✓    |   ✓    |   ✓    | Needs 2+ observations
    reg_trend                |    ✗    |   ✓    |   ✓    |   ✓    |   ✓    | Needs 2+ observations
    load_deviation           |    ✓    |   ✓    |   ✓    |   ✓    |   ✓    | From concurrent load
    hour_sin/cos             |    ✓    |   ✓    |   ✓    |   ✓    |   ✓    | Time encoding
    is_weekend               |    ✓    |   ✓    |   ✓    |   ✓    |   ✓    | Calendar
    imb_proxy_lag1/2         |    ✓    |   ✓    |   ✓    |   ✓    |   ✓    | Prior period proxy
    imb_proxy_rolling_*      |    ✓    |   ✓    |   ✓    |   ✓    |   ✓    | Rolling stats
    """)

    # 5. Suggested feature sets per lead time
    print("\n" + "=" * 70)
    print("SUGGESTED OPTIMAL FEATURE SETS")
    print("=" * 70)

    suggestions = {
        12: {
            'include': ['baseline_pred', 'reg_cumulative_mean', 'imb_proxy_rolling_mean4',
                       'imb_proxy_lag1', 'hour_sin', 'hour_cos', 'load_deviation'],
            'exclude': ['reg_std', 'reg_range', 'reg_trend'],  # Zero info at lead 12
            'consider': ['imb_proxy_lag2', 'imb_proxy_rolling_std4', 'is_weekend'],
        },
        9: {
            'include': ['baseline_pred', 'reg_cumulative_mean', 'imb_proxy_rolling_mean4',
                       'imb_proxy_lag1', 'reg_std', 'reg_range'],
            'exclude': [],
            'consider': ['hour_sin', 'hour_cos', 'reg_trend', 'is_weekend'],
        },
        6: {
            'include': ['baseline_pred', 'reg_cumulative_mean', 'imb_proxy_rolling_mean4',
                       'imb_proxy_lag1', 'reg_std', 'reg_range'],
            'exclude': [],
            'consider': ['hour_sin', 'hour_cos', 'load_deviation'],
        },
        3: {
            'include': ['baseline_pred', 'reg_cumulative_mean', 'imb_proxy_rolling_mean4',
                       'reg_std', 'reg_range', 'imb_proxy_lag1'],
            'exclude': [],
            'consider': ['hour_sin', 'hour_cos', 'reg_trend'],
        },
        0: {
            'include': ['baseline_pred', 'reg_cumulative_mean', 'imb_proxy_rolling_mean4',
                       'reg_std', 'reg_range', 'imb_proxy_lag1'],
            'exclude': [],
            'consider': ['reg_trend', 'hour_sin', 'hour_cos'],
        },
    }

    for lead in [12, 9, 6, 3, 0]:
        print(f"\nLead {lead} min:")
        print(f"  INCLUDE: {', '.join(suggestions[lead]['include'])}")
        if suggestions[lead]['exclude']:
            print(f"  EXCLUDE: {', '.join(suggestions[lead]['exclude'])}")
        print(f"  CONSIDER: {', '.join(suggestions[lead]['consider'])}")

    # Save importance data
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance_all_leads.csv', index=False)
    baseline_metrics.to_csv(OUTPUT_DIR / 'baseline_metrics.csv', index=False)

    print(f"\n\nSaved: feature_importance_all_leads.csv, baseline_metrics.csv")


if __name__ == '__main__':
    main()
