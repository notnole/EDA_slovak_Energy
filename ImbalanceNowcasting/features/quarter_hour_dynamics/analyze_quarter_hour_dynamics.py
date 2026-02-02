"""
Within Quarter-Hour Dynamics Analysis

Analyze how features evolve during the 5 observations (0, 3, 6, 9, 12 min)
within each 15-minute settlement period, and how early observations
predict the final 15-min imbalance.

Key questions:
1. How do features evolve within the quarter-hour?
2. Does minute 0-3 regulation predict minute 15 imbalance?
3. Does prediction accuracy improve as we get more observations?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\features\quarter_hour_dynamics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load features and labels."""
    # Load regulation (main feature)
    reg = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])

    # Load load with deviation
    load = pd.read_csv(FEATURES_DIR / 'load_3min_with_deviation.csv', parse_dates=['datetime'])
    load = load[['datetime', 'load_mw', 'load_mw_deviation']]

    # Load labels
    label = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label = label[['datetime', 'System Imbalance (MWh)']].rename(columns={'System Imbalance (MWh)': 'imbalance'})

    return reg, load, label

def assign_settlement_period(df):
    """Assign each row to its 15-min settlement period and minute within period."""
    df = df.copy()

    # Floor datetime to nearest 3-minute mark to handle irregular timestamps
    df['datetime_floored'] = df['datetime'].dt.floor('3min')

    df['date'] = df['datetime_floored'].dt.date
    df['hour'] = df['datetime_floored'].dt.hour
    df['minute'] = df['datetime_floored'].dt.minute

    # Settlement period (1-96 per day)
    df['settlement_term'] = (df['hour'] * 4) + (df['minute'] // 15) + 1

    # Minute within settlement (0, 3, 6, 9, 12)
    df['minute_in_settlement'] = df['minute'] % 15

    # Create settlement period key
    df['settlement_key'] = df['date'].astype(str) + '_' + df['settlement_term'].astype(str)

    return df

def analyze_feature_evolution(reg_df, output_dir):
    """Analyze how regulation evolves within quarter-hour."""
    print("\n[1] Analyzing feature evolution within quarter-hour...")

    reg_df = assign_settlement_period(reg_df)

    # Group by minute_in_settlement
    evolution = reg_df.groupby('minute_in_settlement')['regulation_mw'].agg(['mean', 'std', 'count'])
    evolution.columns = ['mean', 'std', 'count']

    # Also compute percentiles
    for p in [5, 25, 50, 75, 95]:
        evolution[f'p{p}'] = reg_df.groupby('minute_in_settlement')['regulation_mw'].quantile(p/100)

    # Save data
    evolution.to_csv(output_dir / '01_regulation_evolution_stats.csv')
    print(f"  Saved: 01_regulation_evolution_stats.csv")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Regulation Evolution Within Quarter-Hour', fontsize=14, fontweight='bold')

    # Mean with std
    ax = axes[0]
    minutes = evolution.index
    ax.plot(minutes, evolution['mean'], 'o-', linewidth=2, markersize=8)
    ax.fill_between(minutes, evolution['mean'] - evolution['std'],
                   evolution['mean'] + evolution['std'], alpha=0.3)
    ax.set_xlabel('Minute in Settlement Period')
    ax.set_ylabel('Regulation (MW)')
    ax.set_title('Mean ± Std')
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)

    # Percentile bands
    ax = axes[1]
    ax.fill_between(minutes, evolution['p5'], evolution['p95'], alpha=0.2, label='P5-P95')
    ax.fill_between(minutes, evolution['p25'], evolution['p75'], alpha=0.4, label='P25-P75')
    ax.plot(minutes, evolution['p50'], 'o-', linewidth=2, markersize=8, label='Median')
    ax.set_xlabel('Minute in Settlement Period')
    ax.set_ylabel('Regulation (MW)')
    ax.set_title('Percentile Bands')
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / '01_regulation_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 01_regulation_evolution.png")

    return evolution

def analyze_predictive_power_by_minute(reg_df, label_df, output_dir):
    """Analyze how prediction accuracy improves with more observations."""
    print("\n[2] Analyzing predictive power by minute in settlement...")

    reg_df = assign_settlement_period(reg_df)
    label_df = assign_settlement_period(label_df)

    # For each settlement period, get regulation at each minute
    pivot_reg = reg_df.pivot_table(
        index='settlement_key',
        columns='minute_in_settlement',
        values='regulation_mw',
        aggfunc='first'
    )
    pivot_reg.columns = [f'reg_min{int(c)}' for c in pivot_reg.columns]

    # Get imbalance for each settlement
    label_agg = label_df.groupby('settlement_key')['imbalance'].first()

    # Merge
    merged = pivot_reg.join(label_agg, how='inner')
    merged = merged.dropna()

    print(f"  Matched settlement periods: {len(merged):,}")

    # Calculate correlation at each minute
    results = []
    for min_col in [c for c in merged.columns if c.startswith('reg_min')]:
        minute = int(min_col.replace('reg_min', ''))
        corr, pval = stats.pearsonr(merged[min_col], merged['imbalance'])
        results.append({
            'minute': minute,
            'correlation': corr,
            'r_squared': corr**2,
            'n': len(merged)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / '02_correlation_by_minute.csv', index=False)
    print(f"  Saved: 02_correlation_by_minute.csv")

    # Also compute cumulative features (mean of observations so far)
    cumulative_results = []
    for end_min in [0, 3, 6, 9, 12]:
        cols = [f'reg_min{m}' for m in range(0, end_min+1, 3) if f'reg_min{m}' in merged.columns]
        if cols:
            merged[f'cumul_mean_{end_min}'] = merged[cols].mean(axis=1)
            corr, _ = stats.pearsonr(merged[f'cumul_mean_{end_min}'], merged['imbalance'])
            cumulative_results.append({
                'minutes_available': end_min + 3,  # +3 because we have data through minute X
                'features_used': len(cols),
                'correlation': corr,
                'r_squared': corr**2
            })

    cumul_df = pd.DataFrame(cumulative_results)
    cumul_df.to_csv(output_dir / '02_cumulative_correlation.csv', index=False)
    print(f"  Saved: 02_cumulative_correlation.csv")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Predictive Power by Minute in Settlement', fontsize=14, fontweight='bold')

    # Single minute correlation
    ax = axes[0]
    ax.bar(results_df['minute'], results_df['correlation'].abs(), color='tab:blue', alpha=0.7)
    ax.set_xlabel('Minute in Settlement')
    ax.set_ylabel('|Correlation| with Imbalance')
    ax.set_title('Single Observation Correlation')
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.grid(True, alpha=0.3, axis='y')
    for i, row in results_df.iterrows():
        ax.text(row['minute'], abs(row['correlation']) + 0.01, f"{row['correlation']:.3f}",
               ha='center', fontsize=9)

    # Cumulative correlation
    ax = axes[1]
    ax.plot(cumul_df['minutes_available'], cumul_df['correlation'].abs(), 'o-',
           linewidth=2, markersize=10, color='tab:green')
    ax.set_xlabel('Minutes of Data Available')
    ax.set_ylabel('|Correlation| with Imbalance')
    ax.set_title('Cumulative Mean Correlation')
    ax.set_xticks([3, 6, 9, 12, 15])
    ax.grid(True, alpha=0.3)
    for _, row in cumul_df.iterrows():
        ax.text(row['minutes_available'], abs(row['correlation']) + 0.01,
               f"{row['correlation']:.3f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / '02_predictive_power_by_minute.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 02_predictive_power_by_minute.png")

    return results_df, cumul_df, merged

def analyze_within_qh_variability(merged, output_dir):
    """Analyze variability of regulation within quarter-hour."""
    print("\n[3] Analyzing within quarter-hour variability...")

    # Get all regulation columns
    reg_cols = [c for c in merged.columns if c.startswith('reg_min')]

    # Compute within-QH statistics
    merged['qh_mean'] = merged[reg_cols].mean(axis=1)
    merged['qh_std'] = merged[reg_cols].std(axis=1)
    merged['qh_range'] = merged[reg_cols].max(axis=1) - merged[reg_cols].min(axis=1)
    merged['qh_trend'] = merged[reg_cols[-1]] - merged[reg_cols[0]] if len(reg_cols) > 1 else 0

    # Correlation of these with imbalance
    var_results = []
    for col in ['qh_mean', 'qh_std', 'qh_range', 'qh_trend']:
        valid = merged[[col, 'imbalance']].dropna()
        corr, _ = stats.pearsonr(valid[col], valid['imbalance'])
        var_results.append({
            'feature': col,
            'correlation': corr,
            'r_squared': corr**2,
            'mean': merged[col].mean(),
            'std': merged[col].std()
        })

    var_df = pd.DataFrame(var_results)
    var_df.to_csv(output_dir / '03_within_qh_variability.csv', index=False)
    print(f"  Saved: 03_within_qh_variability.csv")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Within Quarter-Hour Variability Features', fontsize=14, fontweight='bold')

    for ax, (col, label) in zip(axes.flat, [
        ('qh_mean', 'QH Mean Regulation'),
        ('qh_std', 'QH Std Regulation'),
        ('qh_range', 'QH Range (max-min)'),
        ('qh_trend', 'QH Trend (last-first)')
    ]):
        valid = merged[[col, 'imbalance']].dropna()
        corr, _ = stats.pearsonr(valid[col], valid['imbalance'])

        ax.hexbin(valid[col], valid['imbalance'], gridsize=40, cmap='Blues', mincnt=1)
        ax.set_xlabel(f'{label} (MW)')
        ax.set_ylabel('Imbalance (MWh)')
        ax.set_title(f'r = {corr:.3f}')

        # Regression line
        z = np.polyfit(valid[col], valid['imbalance'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[col].min(), valid[col].max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2)

    plt.tight_layout()
    plt.savefig(output_dir / '03_within_qh_variability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 03_within_qh_variability.png")

    return var_df

def analyze_early_warning(merged, output_dir):
    """Analyze if early minutes provide early warning of extreme imbalances."""
    print("\n[4] Analyzing early warning capability...")

    # Define extreme imbalance thresholds
    p10 = merged['imbalance'].quantile(0.10)
    p90 = merged['imbalance'].quantile(0.90)

    merged['extreme_negative'] = merged['imbalance'] < p10
    merged['extreme_positive'] = merged['imbalance'] > p90
    merged['extreme'] = merged['extreme_negative'] | merged['extreme_positive']

    # For each minute, compute mean regulation for extreme vs normal periods
    results = []
    for min_col in [c for c in merged.columns if c.startswith('reg_min')]:
        minute = int(min_col.replace('reg_min', ''))

        extreme_neg_mean = merged[merged['extreme_negative']][min_col].mean()
        extreme_pos_mean = merged[merged['extreme_positive']][min_col].mean()
        normal_mean = merged[~merged['extreme']][min_col].mean()

        results.append({
            'minute': minute,
            'normal_mean': normal_mean,
            'extreme_negative_mean': extreme_neg_mean,
            'extreme_positive_mean': extreme_pos_mean,
            'separation_neg': extreme_neg_mean - normal_mean,
            'separation_pos': extreme_pos_mean - normal_mean
        })

    early_df = pd.DataFrame(results)
    early_df.to_csv(output_dir / '04_early_warning.csv', index=False)
    print(f"  Saved: 04_early_warning.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    minutes = early_df['minute']
    width = 0.8

    ax.bar(minutes - width/3, early_df['extreme_negative_mean'], width/3,
          label=f'Extreme Negative (P<10, imb<{p10:.1f})', color='tab:red', alpha=0.7)
    ax.bar(minutes, early_df['normal_mean'], width/3,
          label='Normal', color='tab:gray', alpha=0.7)
    ax.bar(minutes + width/3, early_df['extreme_positive_mean'], width/3,
          label=f'Extreme Positive (P>90, imb>{p90:.1f})', color='tab:green', alpha=0.7)

    ax.set_xlabel('Minute in Settlement')
    ax.set_ylabel('Mean Regulation (MW)')
    ax.set_title('Early Warning: Regulation by Minute for Extreme vs Normal Imbalances')
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / '04_early_warning.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 04_early_warning.png")

    return early_df

def generate_summary_report(evolution, results_df, cumul_df, var_df, early_df, output_dir):
    """Generate summary report."""
    report = []
    report.append("=" * 70)
    report.append("WITHIN QUARTER-HOUR DYNAMICS ANALYSIS")
    report.append("=" * 70)
    report.append("")

    report.append("1. REGULATION EVOLUTION")
    report.append("-" * 40)
    report.append("Mean regulation at each minute in settlement period:")
    for idx, row in evolution.iterrows():
        report.append(f"  Minute {int(idx):2d}: mean={row['mean']:+6.2f} MW, std={row['std']:5.2f} MW")
    report.append("")

    report.append("2. PREDICTIVE POWER BY MINUTE")
    report.append("-" * 40)
    report.append("Correlation with final 15-min imbalance:")
    for _, row in results_df.iterrows():
        report.append(f"  Minute {int(row['minute']):2d}: r = {row['correlation']:.4f} (R² = {row['r_squared']*100:.1f}%)")
    report.append("")
    report.append("Cumulative mean correlation:")
    for _, row in cumul_df.iterrows():
        report.append(f"  After {int(row['minutes_available']):2d} min: r = {row['correlation']:.4f} (R² = {row['r_squared']*100:.1f}%)")
    report.append("")

    report.append("3. WITHIN-QH VARIABILITY FEATURES")
    report.append("-" * 40)
    for _, row in var_df.iterrows():
        report.append(f"  {row['feature']:12s}: r = {row['correlation']:.4f}")
    report.append("")

    report.append("4. EARLY WARNING CAPABILITY")
    report.append("-" * 40)
    report.append("Separation of regulation for extreme vs normal imbalances:")
    for _, row in early_df.iterrows():
        report.append(f"  Minute {int(row['minute']):2d}: neg_sep={row['separation_neg']:+6.2f}, pos_sep={row['separation_pos']:+6.2f}")
    report.append("")

    report.append("=" * 70)
    report.append("KEY FINDINGS")
    report.append("=" * 70)

    # Best single minute
    best_single = results_df.loc[results_df['correlation'].abs().idxmax()]
    report.append(f"  Best single minute: {int(best_single['minute'])} (r = {best_single['correlation']:.4f})")

    # Improvement from more data
    first_corr = cumul_df.iloc[0]['correlation']
    last_corr = cumul_df.iloc[-1]['correlation']
    improvement = (abs(last_corr) - abs(first_corr)) / abs(first_corr) * 100
    report.append(f"  Improvement from 3min to 15min data: {improvement:+.1f}%")

    # Best variability feature
    best_var = var_df.loc[var_df['correlation'].abs().idxmax()]
    report.append(f"  Best variability feature: {best_var['feature']} (r = {best_var['correlation']:.4f})")

    report.append("")
    report.append("=" * 70)

    with open(output_dir / 'quarter_hour_dynamics_report.txt', 'w') as f:
        f.write('\n'.join(report))

    print('\n'.join(report))

def main():
    print("=" * 70)
    print("WITHIN QUARTER-HOUR DYNAMICS ANALYSIS")
    print("=" * 70)

    reg_df, load_df, label_df = load_data()
    print(f"Regulation: {len(reg_df):,} rows")
    print(f"Labels: {len(label_df):,} rows")

    # 1. Feature evolution
    evolution = analyze_feature_evolution(reg_df, OUTPUT_DIR)

    # 2. Predictive power by minute
    results_df, cumul_df, merged = analyze_predictive_power_by_minute(reg_df, label_df, OUTPUT_DIR)

    # 3. Within-QH variability
    var_df = analyze_within_qh_variability(merged, OUTPUT_DIR)

    # 4. Early warning
    early_df = analyze_early_warning(merged, OUTPUT_DIR)

    # Generate report
    print("\n[5] Generating summary report...")
    generate_summary_report(evolution, results_df, cumul_df, var_df, early_df, OUTPUT_DIR)
    print(f"  Saved: quarter_hour_dynamics_report.txt")

    print(f"\nOutput: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
