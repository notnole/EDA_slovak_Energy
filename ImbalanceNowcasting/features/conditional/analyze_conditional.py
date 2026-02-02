"""
Conditional Analysis

Analyze how feature-label relationships change under different conditions:
1. High vs low load periods
2. Weekday vs weekend
3. Different hours of day
4. Different months/seasons
5. Ramp vs stable periods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\features\conditional")
DATA_DIR = OUTPUT_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load features and label."""
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
    load_df = pd.read_csv(FEATURES_DIR / 'load_3min.csv', parse_dates=['datetime'])
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
        columns={'System Imbalance (MWh)': 'imbalance'}
    )
    return reg_df, load_df, label_df


def align_features_to_label(reg_df, load_df, label_df):
    """Align 3-min features to 15-min labels."""
    # For each label, get the regulation values within the settlement period
    label_df = label_df.copy()
    label_df['period_start'] = label_df['datetime'] - pd.Timedelta(minutes=15)

    # Get mean regulation per settlement period
    merged = pd.merge_asof(
        label_df.sort_values('datetime'),
        reg_df.sort_values('datetime'),
        on='datetime',
        direction='backward',
        tolerance=pd.Timedelta(minutes=15)
    )

    # Get load at settlement time
    merged = pd.merge_asof(
        merged.sort_values('datetime'),
        load_df.sort_values('datetime'),
        on='datetime',
        direction='backward',
        tolerance=pd.Timedelta(minutes=15)
    )

    # Add time features
    merged['hour'] = merged['datetime'].dt.hour
    merged['dow'] = merged['datetime'].dt.dayofweek
    merged['is_weekend'] = merged['dow'] >= 5
    merged['month'] = merged['datetime'].dt.month
    merged['date'] = merged['datetime'].dt.date

    # Compute load deviation using ToD means from the data
    merged['tod'] = merged['hour'] * 4 + merged['datetime'].dt.minute // 15
    tod_means = merged.groupby(['tod', 'is_weekend'])['load_mw'].transform('mean')
    merged['load_deviation'] = merged['load_mw'] - tod_means

    return merged.dropna(subset=['regulation_mw', 'imbalance'])


def compute_correlation(df, feature_col, label_col='imbalance'):
    """Compute correlation with significance test."""
    valid = df[[feature_col, label_col]].dropna()
    if len(valid) < 30:
        return np.nan, np.nan, len(valid)
    corr, pval = stats.pearsonr(valid[feature_col], valid[label_col])
    return corr, pval, len(valid)


def analyze_by_condition(df, condition_col, feature_col='regulation_mw'):
    """Analyze correlation by condition groups."""
    results = []
    for group_val in df[condition_col].unique():
        subset = df[df[condition_col] == group_val]
        corr, pval, n = compute_correlation(subset, feature_col)
        results.append({
            'condition': condition_col,
            'group': group_val,
            'correlation': corr,
            'p_value': pval,
            'n_samples': n
        })
    return pd.DataFrame(results).sort_values('group')


def analyze_by_quantile(df, split_col, feature_col='regulation_mw', n_quantiles=4):
    """Analyze correlation by quantiles of another variable."""
    df = df.copy()
    df['quantile'] = pd.qcut(df[split_col], n_quantiles, labels=False, duplicates='drop')
    quantile_labels = pd.qcut(df[split_col], n_quantiles, duplicates='drop').unique()

    results = []
    for q in range(n_quantiles):
        subset = df[df['quantile'] == q]
        corr, pval, n = compute_correlation(subset, feature_col)
        q_range = quantile_labels[q] if q < len(quantile_labels) else f"Q{q}"
        results.append({
            'quantile': q,
            'range': str(q_range),
            'correlation': corr,
            'p_value': pval,
            'n_samples': n,
            'mean_value': subset[split_col].mean()
        })
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("CONDITIONAL ANALYSIS")
    print("=" * 70)
    print("\nAnalyzing how feature-label relationships change under different conditions")

    reg_df, load_df, label_df = load_data()
    df = align_features_to_label(reg_df, load_df, label_df)
    print(f"\nAligned dataset: {len(df):,} rows")

    # Overall baseline
    corr_base, _, n_base = compute_correlation(df, 'regulation_mw')
    print(f"Overall regulation correlation: r = {corr_base:.4f} (n={n_base:,})")

    all_results = []

    # =================================================================
    # 1. WEEKDAY VS WEEKEND
    # =================================================================
    print("\n" + "=" * 70)
    print("1. WEEKDAY VS WEEKEND")
    print("=" * 70)

    weekend_results = analyze_by_condition(df, 'is_weekend')
    weekend_results['condition_label'] = weekend_results['group'].map({False: 'Weekday', True: 'Weekend'})
    print(weekend_results[['condition_label', 'correlation', 'n_samples']].to_string(index=False))
    weekend_results.to_csv(DATA_DIR / '01_weekend_analysis.csv', index=False)
    all_results.append(('Weekend', weekend_results))

    # =================================================================
    # 2. HOUR OF DAY
    # =================================================================
    print("\n" + "=" * 70)
    print("2. HOUR OF DAY")
    print("=" * 70)

    hour_results = analyze_by_condition(df, 'hour')
    print("Hour | Correlation | N")
    print("-" * 30)
    for _, row in hour_results.iterrows():
        print(f"  {int(row['group']):2d}  |   {row['correlation']:.3f}    | {int(row['n_samples']):,}")
    hour_results.to_csv(DATA_DIR / '02_hour_analysis.csv', index=False)
    all_results.append(('Hour', hour_results))

    # =================================================================
    # 3. MONTH/SEASON
    # =================================================================
    print("\n" + "=" * 70)
    print("3. MONTH")
    print("=" * 70)

    month_results = analyze_by_condition(df, 'month')
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    month_results['month_name'] = month_results['group'].map(month_names)
    print(month_results[['month_name', 'correlation', 'n_samples']].to_string(index=False))
    month_results.to_csv(DATA_DIR / '03_month_analysis.csv', index=False)
    all_results.append(('Month', month_results))

    # =================================================================
    # 4. LOAD LEVEL (QUANTILES)
    # =================================================================
    print("\n" + "=" * 70)
    print("4. LOAD LEVEL (QUARTILES)")
    print("=" * 70)

    load_results = analyze_by_quantile(df, 'load_mw', n_quantiles=4)
    load_results['label'] = ['Low Load', 'Med-Low', 'Med-High', 'High Load']
    print(load_results[['label', 'correlation', 'n_samples', 'mean_value']].to_string(index=False))
    load_results.to_csv(DATA_DIR / '04_load_level_analysis.csv', index=False)
    all_results.append(('Load Level', load_results))

    # =================================================================
    # 5. IMBALANCE MAGNITUDE (ARE EXTREME IMBALANCES DIFFERENT?)
    # =================================================================
    print("\n" + "=" * 70)
    print("5. IMBALANCE MAGNITUDE")
    print("=" * 70)

    df['imb_abs'] = df['imbalance'].abs()
    imb_results = analyze_by_quantile(df, 'imb_abs', n_quantiles=4)
    imb_results['label'] = ['Small |Imb|', 'Medium-Small', 'Medium-Large', 'Large |Imb|']
    print(imb_results[['label', 'correlation', 'n_samples', 'mean_value']].to_string(index=False))
    imb_results.to_csv(DATA_DIR / '05_imbalance_magnitude_analysis.csv', index=False)
    all_results.append(('Imbalance Magnitude', imb_results))

    # =================================================================
    # 6. LOAD RAMP (RATE OF CHANGE)
    # =================================================================
    print("\n" + "=" * 70)
    print("6. LOAD RAMP RATE")
    print("=" * 70)

    # Compute load change from previous period
    df = df.sort_values('datetime')
    df['load_prev'] = df['load_mw'].shift(1)
    df['load_ramp'] = df['load_mw'] - df['load_prev']
    df_ramp = df.dropna(subset=['load_ramp'])

    ramp_results = analyze_by_quantile(df_ramp, 'load_ramp', n_quantiles=4)
    ramp_results['label'] = ['Falling Fast', 'Falling Slow', 'Rising Slow', 'Rising Fast']
    print(ramp_results[['label', 'correlation', 'n_samples', 'mean_value']].to_string(index=False))
    ramp_results.to_csv(DATA_DIR / '06_load_ramp_analysis.csv', index=False)
    all_results.append(('Load Ramp', ramp_results))

    # =================================================================
    # 7. REGULATION SIGN (POSITIVE VS NEGATIVE)
    # =================================================================
    print("\n" + "=" * 70)
    print("7. REGULATION SIGN")
    print("=" * 70)

    df['reg_sign'] = np.where(df['regulation_mw'] > 0, 'Positive (Oversupply)', 'Negative (Undersupply)')
    sign_results = analyze_by_condition(df, 'reg_sign')
    print(sign_results[['group', 'correlation', 'n_samples']].to_string(index=False))
    sign_results.to_csv(DATA_DIR / '07_regulation_sign_analysis.csv', index=False)
    all_results.append(('Regulation Sign', sign_results))

    # =================================================================
    # VISUALIZATION
    # =================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Conditional Analysis: Regulation-Imbalance Correlation by Condition',
                 fontsize=14, fontweight='bold')

    # Plot 1: Weekend
    ax = axes[0, 0]
    weekend_plot = weekend_results.copy()
    ax.bar(['Weekday', 'Weekend'], weekend_plot['correlation'].values, color=['tab:blue', 'tab:orange'])
    ax.axhline(corr_base, color='red', linestyle='--', label=f'Overall: {corr_base:.3f}')
    ax.set_ylabel('Correlation')
    ax.set_title('By Day Type')
    ax.legend()
    for i, v in enumerate(weekend_plot['correlation'].values):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')

    # Plot 2: Hour of Day
    ax = axes[0, 1]
    hours = hour_results['group'].astype(int)
    corrs = hour_results['correlation']
    ax.bar(hours, corrs, color='tab:green', alpha=0.7)
    ax.axhline(corr_base, color='red', linestyle='--', label=f'Overall: {corr_base:.3f}')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Correlation')
    ax.set_title('By Hour of Day')
    ax.set_xticks(range(0, 24, 3))
    ax.legend()

    # Plot 3: Month
    ax = axes[0, 2]
    months = month_results['group'].astype(int)
    corrs = month_results['correlation']
    ax.bar(months, corrs, color='tab:purple', alpha=0.7)
    ax.axhline(corr_base, color='red', linestyle='--', label=f'Overall: {corr_base:.3f}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Correlation')
    ax.set_title('By Month')
    ax.set_xticks(range(1, 13))
    ax.legend()

    # Plot 4: Load Level
    ax = axes[1, 0]
    labels = load_results['label']
    corrs = load_results['correlation']
    ax.barh(labels, corrs, color='tab:cyan')
    ax.axvline(corr_base, color='red', linestyle='--', label=f'Overall: {corr_base:.3f}')
    ax.set_xlabel('Correlation')
    ax.set_title('By Load Level')
    ax.legend()
    for i, v in enumerate(corrs):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')

    # Plot 5: Imbalance Magnitude
    ax = axes[1, 1]
    labels = imb_results['label']
    corrs = imb_results['correlation']
    ax.barh(labels, corrs, color='tab:red', alpha=0.7)
    ax.axvline(corr_base, color='red', linestyle='--', label=f'Overall: {corr_base:.3f}')
    ax.set_xlabel('Correlation')
    ax.set_title('By Imbalance Magnitude')
    ax.legend()
    for i, v in enumerate(corrs):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')

    # Plot 6: Load Ramp
    ax = axes[1, 2]
    labels = ramp_results['label']
    corrs = ramp_results['correlation']
    ax.barh(labels, corrs, color='tab:brown', alpha=0.7)
    ax.axvline(corr_base, color='red', linestyle='--', label=f'Overall: {corr_base:.3f}')
    ax.set_xlabel('Correlation')
    ax.set_title('By Load Ramp Rate')
    ax.legend()
    for i, v in enumerate(corrs):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_conditional_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_conditional_correlations.png")

    # =================================================================
    # SCATTER PLOTS FOR KEY CONDITIONS
    # =================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Regulation vs Imbalance: Scatter by Condition', fontsize=14, fontweight='bold')

    # Sample for plotting (too many points otherwise)
    df_sample = df.sample(min(5000, len(df)), random_state=42)

    # Weekday vs Weekend
    ax = axes[0, 0]
    for is_wknd, label, color in [(False, 'Weekday', 'tab:blue'), (True, 'Weekend', 'tab:orange')]:
        subset = df_sample[df_sample['is_weekend'] == is_wknd]
        ax.scatter(subset['regulation_mw'], subset['imbalance'], alpha=0.3, s=10, label=label, c=color)
    ax.set_xlabel('Regulation (MW)')
    ax.set_ylabel('Imbalance (MWh)')
    ax.set_title('Weekday vs Weekend')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Peak vs Off-peak
    ax = axes[0, 1]
    df_sample['is_peak'] = df_sample['hour'].between(8, 20)
    for is_peak, label, color in [(True, 'Peak (8-20)', 'tab:red'), (False, 'Off-peak', 'tab:green')]:
        subset = df_sample[df_sample['is_peak'] == is_peak]
        ax.scatter(subset['regulation_mw'], subset['imbalance'], alpha=0.3, s=10, label=label, c=color)
    ax.set_xlabel('Regulation (MW)')
    ax.set_ylabel('Imbalance (MWh)')
    ax.set_title('Peak vs Off-Peak Hours')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # High vs Low Load
    ax = axes[1, 0]
    load_median = df_sample['load_mw'].median()
    df_sample['is_high_load'] = df_sample['load_mw'] > load_median
    for is_high, label, color in [(True, 'High Load', 'tab:purple'), (False, 'Low Load', 'tab:cyan')]:
        subset = df_sample[df_sample['is_high_load'] == is_high]
        ax.scatter(subset['regulation_mw'], subset['imbalance'], alpha=0.3, s=10, label=label, c=color)
    ax.set_xlabel('Regulation (MW)')
    ax.set_ylabel('Imbalance (MWh)')
    ax.set_title('High vs Low Load')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Positive vs Negative Regulation
    ax = axes[1, 1]
    for is_pos, label, color in [(True, 'Positive Reg', 'tab:green'), (False, 'Negative Reg', 'tab:red')]:
        subset = df_sample[df_sample['regulation_mw'] > 0] if is_pos else df_sample[df_sample['regulation_mw'] <= 0]
        ax.scatter(subset['regulation_mw'], subset['imbalance'], alpha=0.3, s=10, label=label, c=color)
    ax.set_xlabel('Regulation (MW)')
    ax.set_ylabel('Imbalance (MWh)')
    ax.set_title('By Regulation Sign')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_conditional_scatters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_conditional_scatters.png")

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nBaseline correlation: r = {corr_base:.4f}")
    print("\nCorrelation stability across conditions:")

    summary_data = []
    for name, result_df in all_results:
        corr_min = result_df['correlation'].min()
        corr_max = result_df['correlation'].max()
        corr_range = corr_max - corr_min
        summary_data.append({
            'Condition': name,
            'Min r': corr_min,
            'Max r': corr_max,
            'Range': corr_range,
            'Stable?': 'YES' if corr_range < 0.1 else 'NO'
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(DATA_DIR / 'conditional_summary.csv', index=False)

    # Write report
    report_lines = [
        "CONDITIONAL ANALYSIS REPORT",
        "=" * 50,
        "",
        f"Baseline regulation-imbalance correlation: r = {corr_base:.4f}",
        f"Total samples: {len(df):,}",
        "",
        "CORRELATION STABILITY BY CONDITION",
        "-" * 50,
        ""
    ]

    for _, row in summary_df.iterrows():
        report_lines.append(f"{row['Condition']:20s}: {row['Min r']:.3f} to {row['Max r']:.3f} (range: {row['Range']:.3f}) - {row['Stable?']}")

    report_lines.extend([
        "",
        "KEY FINDINGS",
        "-" * 50,
        ""
    ])

    # Find most variable conditions
    most_variable = summary_df.loc[summary_df['Range'].idxmax()]
    most_stable = summary_df.loc[summary_df['Range'].idxmin()]

    report_lines.append(f"Most variable: {most_variable['Condition']} (range: {most_variable['Range']:.3f})")
    report_lines.append(f"Most stable: {most_stable['Condition']} (range: {most_stable['Range']:.3f})")

    with open(DATA_DIR / 'conditional_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
