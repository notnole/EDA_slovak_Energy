"""
Feature-Label Correlation Analysis

Analyze correlation between features and imbalance label:
1. Raw features vs label
2. Deviation features (feature - ToD mean) vs label
3. Scatter plots
4. Correlation matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\features\correlation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load label and features with deviations."""
    # Load label
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(columns={'System Imbalance (MWh)': 'imbalance'})

    # Load features with deviations
    features = {}

    # Regulation
    reg = pd.read_csv(FEATURES_DIR / 'regulation_3min_with_deviation.csv', parse_dates=['datetime'])
    features['regulation'] = reg[['datetime', 'regulation_mw', 'regulation_mw_deviation']]

    # Load
    load = pd.read_csv(FEATURES_DIR / 'load_3min_with_deviation.csv', parse_dates=['datetime'])
    features['load'] = load[['datetime', 'load_mw', 'load_mw_deviation']]

    # Production (short period)
    prod = pd.read_csv(FEATURES_DIR / 'production_3min_with_deviation.csv', parse_dates=['datetime'])
    features['production'] = prod[['datetime', 'production_mw', 'production_mw_deviation']]

    # Export/Import (short period)
    exp = pd.read_csv(FEATURES_DIR / 'export_import_3min_with_deviation.csv', parse_dates=['datetime'])
    features['export_import'] = exp[['datetime', 'export_import_mw', 'export_import_mw_deviation']]

    return label_df, features

def align_feature_to_label(label_df, feature_df):
    """
    Align 3-min features to 15-min labels.
    For each 15-min label, get the most recent feature value available.
    """
    # For each label timestamp, find the latest feature before it
    merged = pd.merge_asof(
        label_df.sort_values('datetime'),
        feature_df.sort_values('datetime'),
        on='datetime',
        direction='backward',
        tolerance=pd.Timedelta(minutes=15)
    )
    return merged

def plot_scatter_raw(merged_df, feature_col, label_col, name, output_path):
    """Scatter plot of raw feature vs label."""
    fig, ax = plt.subplots(figsize=(10, 8))

    x = merged_df[feature_col].dropna()
    y = merged_df.loc[x.index, label_col]

    # Remove any remaining NaN
    mask = ~(x.isna() | y.isna())
    x, y = x[mask], y[mask]

    # Correlation
    corr, pval = stats.pearsonr(x, y)

    # Scatter with density coloring
    ax.hexbin(x, y, gridsize=50, cmap='Blues', mincnt=1)
    ax.set_xlabel(f'{name} (MW)')
    ax.set_ylabel('Imbalance (MWh)')
    ax.set_title(f'{name} vs Imbalance\nr = {corr:.3f} (p < 0.001)')

    # Add regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'y = {z[0]:.3f}x + {z[1]:.2f}')
    ax.legend()

    plt.colorbar(ax.collections[0], label='Count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return corr

def plot_correlation_matrix(merged_full, output_path):
    """Plot correlation matrix of all features vs label."""
    # Select numeric columns
    cols = ['imbalance', 'regulation_mw', 'regulation_mw_deviation',
            'load_mw', 'load_mw_deviation']

    # Only include columns that exist
    cols = [c for c in cols if c in merged_full.columns]

    corr_matrix = merged_full[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title('Correlation Matrix: Features vs Imbalance', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return corr_matrix

def plot_multi_scatter(merged_full, output_path):
    """Plot multiple scatter plots in grid."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature vs Imbalance Correlations', fontsize=14, fontweight='bold')

    configs = [
        ('regulation_mw', 'Regulation (MW)'),
        ('regulation_mw_deviation', 'Regulation Deviation (MW)'),
        ('load_mw', 'Load (MW)'),
        ('load_mw_deviation', 'Load Deviation (MW)'),
    ]

    for ax, (col, label) in zip(axes.flat, configs):
        if col not in merged_full.columns:
            ax.text(0.5, 0.5, f'{col} not available', ha='center', va='center', transform=ax.transAxes)
            continue

        x = merged_full[col].dropna()
        y = merged_full.loc[x.index, 'imbalance']
        mask = ~(x.isna() | y.isna())
        x, y = x[mask], y[mask]

        if len(x) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            continue

        corr, _ = stats.pearsonr(x, y)

        ax.hexbin(x, y, gridsize=30, cmap='Blues', mincnt=1)
        ax.set_xlabel(label)
        ax.set_ylabel('Imbalance (MWh)')
        ax.set_title(f'r = {corr:.3f}')

        # Regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2)

    # Hide unused axes
    for ax in axes.flat[len(configs):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_report(correlations, output_path):
    """Generate correlation report."""
    report = []
    report.append("=" * 60)
    report.append("FEATURE-LABEL CORRELATION REPORT")
    report.append("=" * 60)
    report.append("")
    report.append("Correlation with Imbalance (MWh):")
    report.append("-" * 40)

    for name, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        report.append(f"  {name:30s}: r = {corr:+.4f}")

    report.append("")
    report.append("INTERPRETATION:")
    report.append("-" * 40)
    report.append("  - Regulation is the inverse of imbalance (strong negative correlation expected)")
    report.append("  - Deviation features capture 'surprise' vs expected")
    report.append("  - Higher |r| = more predictive power")
    report.append("")
    report.append("FEATURE RANKING (by |correlation|):")
    report.append("-" * 40)

    for i, (name, corr) in enumerate(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True), 1):
        report.append(f"  {i}. {name}: |r| = {abs(corr):.4f}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print('\n'.join(report))

def main():
    print("=" * 60)
    print("FEATURE-LABEL CORRELATION ANALYSIS")
    print("=" * 60)

    label_df, features = load_data()
    print(f"Label: {len(label_df):,} rows")

    correlations = {}

    # Merge all features to label
    merged_full = label_df.copy()

    for name, feat_df in features.items():
        print(f"\nProcessing {name}...")
        merged = align_feature_to_label(label_df, feat_df)

        # Get column names
        raw_col = [c for c in feat_df.columns if c.endswith('_mw') and not c.endswith('_deviation')][0]
        dev_col = [c for c in feat_df.columns if c.endswith('_deviation')][0]

        # Add to full merged
        merged_full = align_feature_to_label(merged_full, feat_df)

        # Calculate correlations
        for col in [raw_col, dev_col]:
            if col in merged.columns:
                valid = merged[[col, 'imbalance']].dropna()
                if len(valid) > 100:
                    corr, _ = stats.pearsonr(valid[col], valid['imbalance'])
                    correlations[col] = corr
                    print(f"  {col}: r = {corr:.4f} (n={len(valid):,})")

    # Plots
    print("\nGenerating plots...")

    # 1. Multi-scatter
    plot_multi_scatter(merged_full, OUTPUT_DIR / '01_feature_scatter_grid.png')
    print("  Saved: 01_feature_scatter_grid.png")

    # 2. Correlation matrix
    plot_correlation_matrix(merged_full, OUTPUT_DIR / '02_correlation_matrix.png')
    print("  Saved: 02_correlation_matrix.png")

    # 3. Individual scatter for regulation (most important)
    if 'regulation_mw' in merged_full.columns:
        plot_scatter_raw(merged_full, 'regulation_mw', 'imbalance', 'Regulation',
                        OUTPUT_DIR / '03_regulation_vs_imbalance.png')
        print("  Saved: 03_regulation_vs_imbalance.png")

    # 4. Report
    generate_report(correlations, OUTPUT_DIR / 'correlation_report.txt')
    print("  Saved: correlation_report.txt")

    print(f"\nOutput: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
