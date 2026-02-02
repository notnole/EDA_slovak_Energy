"""
Cross-Correlation with Lags

Analyze if lagged regulation values add predictive power beyond current regulation.
Useful for nowcasting where we predict before period ends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\features\lag_correlation")
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


def create_qh_aggregates(reg_df, label_df):
    """Aggregate regulation to quarter-hour level and merge with labels."""
    # Round regulation timestamps to their settlement period
    reg_df = reg_df.copy()
    reg_df['settlement_end'] = reg_df['datetime'].dt.ceil('15min')

    # Aggregate regulation per settlement period
    qh_reg = reg_df.groupby('settlement_end').agg(
        reg_mean=('regulation_mw', 'mean'),
        reg_min=('regulation_mw', 'min'),
        reg_max=('regulation_mw', 'max'),
        reg_std=('regulation_mw', 'std'),
        reg_last=('regulation_mw', 'last'),
        reg_first=('regulation_mw', 'first'),
        n_obs=('regulation_mw', 'count')
    ).reset_index()
    qh_reg = qh_reg.rename(columns={'settlement_end': 'datetime'})

    # Merge with labels
    merged = pd.merge(label_df, qh_reg, on='datetime', how='inner')
    merged = merged.sort_values('datetime').reset_index(drop=True)

    return merged


def add_lags(df, cols, max_lag=8):
    """Add lagged versions of columns (in settlement periods, i.e., 15-min intervals)."""
    df = df.copy()
    for col in cols:
        for lag in range(1, max_lag + 1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df


def compute_correlation(x, y):
    """Compute correlation, handling NaN."""
    valid = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(valid) < 30:
        return np.nan, np.nan
    corr, pval = stats.pearsonr(valid['x'], valid['y'])
    return corr, pval


def main():
    print("=" * 70)
    print("CROSS-CORRELATION WITH LAGS")
    print("=" * 70)

    reg_df, load_df, label_df = load_data()

    # Create quarter-hour aggregates
    df = create_qh_aggregates(reg_df, label_df)
    print(f"\nQuarter-hour dataset: {len(df):,} rows")

    # Filter to complete observations (5 obs per QH)
    df = df[df['n_obs'] >= 4].copy()
    print(f"After filtering (>=4 obs per QH): {len(df):,} rows")

    # =================================================================
    # 1. REGULATION LAGS
    # =================================================================
    print("\n" + "=" * 70)
    print("1. REGULATION LAG CORRELATIONS")
    print("=" * 70)

    # Add lags
    df = add_lags(df, ['reg_mean', 'imbalance'], max_lag=12)

    # Compute correlations for regulation lags
    reg_lag_results = []
    for lag in range(0, 13):
        col = 'reg_mean' if lag == 0 else f'reg_mean_lag{lag}'
        if col in df.columns:
            corr, pval = compute_correlation(df[col], df['imbalance'])
            reg_lag_results.append({
                'lag': lag,
                'lag_minutes': lag * 15,
                'correlation': corr,
                'p_value': pval
            })
            print(f"  Lag {lag:2d} ({lag*15:3d} min): r = {corr:.4f}")

    reg_lag_df = pd.DataFrame(reg_lag_results)
    reg_lag_df.to_csv(DATA_DIR / 'regulation_lag_correlation.csv', index=False)

    # =================================================================
    # 2. IMBALANCE AUTOCORRELATION (using estimated imbalance from regulation)
    # =================================================================
    print("\n" + "=" * 70)
    print("2. IMBALANCE AUTOCORRELATION")
    print("=" * 70)
    print("(Note: In production, actual imbalance available 1 day later)")
    print("(But can estimate from regulation: imb ~ -reg_mean * 0.25)")

    # Estimated imbalance from regulation
    df['imb_estimated'] = -df['reg_mean'] * 0.25

    imb_lag_results = []
    for lag in range(1, 13):
        col = f'imbalance_lag{lag}'
        if col in df.columns:
            # Actual imbalance autocorrelation
            corr_actual, _ = compute_correlation(df[col], df['imbalance'])
            # Estimated imbalance (what we'd have in real-time)
            df[f'imb_estimated_lag{lag}'] = df['imb_estimated'].shift(lag)
            corr_estimated, _ = compute_correlation(df[f'imb_estimated_lag{lag}'], df['imbalance'])

            imb_lag_results.append({
                'lag': lag,
                'lag_minutes': lag * 15,
                'corr_actual_imbalance': corr_actual,
                'corr_estimated_imbalance': corr_estimated
            })
            print(f"  Lag {lag:2d}: Actual r = {corr_actual:.4f}, Estimated r = {corr_estimated:.4f}")

    imb_lag_df = pd.DataFrame(imb_lag_results)
    imb_lag_df.to_csv(DATA_DIR / 'imbalance_autocorrelation.csv', index=False)

    # =================================================================
    # 3. INCREMENTAL VALUE OF LAGS
    # =================================================================
    print("\n" + "=" * 70)
    print("3. INCREMENTAL VALUE OF LAGS (Multiple Regression R²)")
    print("=" * 70)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # Prepare features
    feature_sets = [
        ('reg_mean only', ['reg_mean']),
        ('+ lag1', ['reg_mean', 'reg_mean_lag1']),
        ('+ lag1-2', ['reg_mean', 'reg_mean_lag1', 'reg_mean_lag2']),
        ('+ lag1-4', ['reg_mean', 'reg_mean_lag1', 'reg_mean_lag2', 'reg_mean_lag3', 'reg_mean_lag4']),
        ('+ imb_est_lag1', ['reg_mean', 'imb_estimated_lag1']),
        ('+ imb_est_lag1-2', ['reg_mean', 'imb_estimated_lag1', 'imb_estimated_lag2']),
    ]

    incremental_results = []
    for name, features in feature_sets:
        valid_df = df[['imbalance'] + features].dropna()
        if len(valid_df) < 100:
            continue

        X = valid_df[features].values
        y = valid_df['imbalance'].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        incremental_results.append({
            'features': name,
            'n_features': len(features),
            'r2': r2,
            'n_samples': len(valid_df)
        })
        print(f"  {name:20s}: R² = {r2:.4f} (n={len(valid_df):,})")

    incremental_df = pd.DataFrame(incremental_results)
    incremental_df.to_csv(DATA_DIR / 'incremental_lag_value.csv', index=False)

    # =================================================================
    # 4. CROSS-CORRELATION FUNCTION PLOT
    # =================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Cross-Correlation Analysis with Lags', fontsize=14, fontweight='bold')

    # Plot 1: Regulation lag correlation
    ax = axes[0, 0]
    ax.bar(reg_lag_df['lag'], reg_lag_df['correlation'], color='tab:blue', alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Lag (settlement periods, 15 min each)')
    ax.set_ylabel('Correlation with Imbalance')
    ax.set_title('Regulation Lag Correlation')
    ax.set_xticks(range(0, 13))
    for i, row in reg_lag_df.iterrows():
        ax.text(row['lag'], row['correlation'] + 0.02, f"{row['correlation']:.2f}",
                ha='center', fontsize=8)

    # Plot 2: Imbalance autocorrelation
    ax = axes[0, 1]
    x = np.arange(len(imb_lag_df))
    width = 0.35
    ax.bar(x - width/2, imb_lag_df['corr_actual_imbalance'], width,
           label='Actual Imbalance', color='tab:green', alpha=0.7)
    ax.bar(x + width/2, imb_lag_df['corr_estimated_imbalance'], width,
           label='Estimated (-reg×0.25)', color='tab:orange', alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Lag (settlement periods)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Imbalance Autocorrelation')
    ax.set_xticks(x)
    ax.set_xticklabels(imb_lag_df['lag'])
    ax.legend()

    # Plot 3: Incremental R² from lags
    ax = axes[1, 0]
    ax.barh(incremental_df['features'], incremental_df['r2'], color='tab:purple', alpha=0.7)
    ax.set_xlabel('R²')
    ax.set_title('Incremental Value of Lag Features')
    for i, row in incremental_df.iterrows():
        ax.text(row['r2'] + 0.01, i, f"{row['r2']:.3f}", va='center')
    ax.set_xlim(0, 1)

    # Plot 4: Correlation decay curve
    ax = axes[1, 1]
    ax.plot(reg_lag_df['lag_minutes'], reg_lag_df['correlation'].abs(),
            'o-', color='tab:blue', label='|Regulation Corr|', markersize=8)
    ax.plot(imb_lag_df['lag_minutes'], imb_lag_df['corr_actual_imbalance'].abs(),
            's-', color='tab:green', label='|Imbalance Autocorr|', markersize=8)
    ax.set_xlabel('Lag (minutes)')
    ax.set_ylabel('|Correlation|')
    ax.set_title('Correlation Decay with Lag')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lag_correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: lag_correlation_analysis.png")

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline_r2 = incremental_df[incremental_df['features'] == 'reg_mean only']['r2'].values[0]
    best_r2 = incremental_df['r2'].max()
    best_features = incremental_df.loc[incremental_df['r2'].idxmax(), 'features']

    print(f"\nBaseline (reg_mean only): R² = {baseline_r2:.4f}")
    print(f"Best with lags ({best_features}): R² = {best_r2:.4f}")
    print(f"Improvement: +{(best_r2 - baseline_r2)*100:.1f}% R²")

    # Key finding
    lag1_corr = reg_lag_df[reg_lag_df['lag'] == 1]['correlation'].values[0]
    lag0_corr = reg_lag_df[reg_lag_df['lag'] == 0]['correlation'].values[0]

    print(f"\nRegulation correlation decay:")
    print(f"  Lag 0 (current): r = {lag0_corr:.4f}")
    print(f"  Lag 1 (15 min ago): r = {lag1_corr:.4f}")
    print(f"  Decay: {(1 - abs(lag1_corr)/abs(lag0_corr))*100:.1f}%")

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
