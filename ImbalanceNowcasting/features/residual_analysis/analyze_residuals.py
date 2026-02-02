"""
Residual Analysis

After accounting for regulation, what patterns remain in the residuals?
Are residuals random or is there exploitable structure?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from pathlib import Path

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\features\residual_analysis")
DATA_DIR = OUTPUT_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """Load features and label, align to quarter-hour."""
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
    load_df = pd.read_csv(FEATURES_DIR / 'load_3min.csv', parse_dates=['datetime'])
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
        columns={'System Imbalance (MWh)': 'imbalance'}
    )

    # Aggregate regulation to QH
    reg_df['settlement_end'] = reg_df['datetime'].dt.ceil('15min')
    qh_reg = reg_df.groupby('settlement_end').agg(
        reg_mean=('regulation_mw', 'mean')
    ).reset_index().rename(columns={'settlement_end': 'datetime'})

    # Aggregate load to QH
    load_df['settlement_end'] = load_df['datetime'].dt.ceil('15min')
    qh_load = load_df.groupby('settlement_end').agg(
        load_mean=('load_mw', 'mean')
    ).reset_index().rename(columns={'settlement_end': 'datetime'})

    # Merge all
    df = pd.merge(label_df, qh_reg, on='datetime', how='inner')
    df = pd.merge(df, qh_load, on='datetime', how='left')

    # Add time features
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['dow'] >= 5
    df['month'] = df['datetime'].dt.month
    df['date'] = df['datetime'].dt.date

    # Compute load deviation
    df['tod'] = df['hour'] * 4 + df['datetime'].dt.minute // 15
    tod_load_means = df.groupby(['tod', 'is_weekend'])['load_mean'].transform('mean')
    df['load_deviation'] = df['load_mean'] - tod_load_means

    return df.dropna(subset=['reg_mean', 'imbalance'])


def fit_simple_model(df):
    """Fit simple linear regression: imbalance ~ regulation."""
    X = df[['reg_mean']].values
    y = df['imbalance'].values

    model = LinearRegression()
    model.fit(X, y)

    df = df.copy()
    df['predicted'] = model.predict(X)
    df['residual'] = df['imbalance'] - df['predicted']

    r2 = model.score(X, y)
    coef = model.coef_[0]
    intercept = model.intercept_

    return df, model, r2, coef, intercept


def analyze_residual_patterns(df):
    """Analyze patterns in residuals."""
    results = {}

    # By hour
    hour_residuals = df.groupby('hour')['residual'].agg(['mean', 'std', 'count'])
    hour_residuals.columns = ['mean_residual', 'std_residual', 'count']
    results['by_hour'] = hour_residuals.reset_index()

    # By day of week
    dow_residuals = df.groupby('dow')['residual'].agg(['mean', 'std', 'count'])
    dow_residuals.columns = ['mean_residual', 'std_residual', 'count']
    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    dow_residuals = dow_residuals.reset_index()
    dow_residuals['day_name'] = dow_residuals['dow'].map(dow_names)
    results['by_dow'] = dow_residuals

    # By month
    month_residuals = df.groupby('month')['residual'].agg(['mean', 'std', 'count'])
    month_residuals.columns = ['mean_residual', 'std_residual', 'count']
    results['by_month'] = month_residuals.reset_index()

    # By load level
    df['load_quartile'] = pd.qcut(df['load_mean'], 4, labels=['Q1-Low', 'Q2', 'Q3', 'Q4-High'])
    load_residuals = df.groupby('load_quartile')['residual'].agg(['mean', 'std', 'count'])
    load_residuals.columns = ['mean_residual', 'std_residual', 'count']
    results['by_load'] = load_residuals.reset_index()

    # By imbalance magnitude (predicted)
    df['pred_quartile'] = pd.qcut(df['predicted'].abs(), 4, labels=['Q1-Small', 'Q2', 'Q3', 'Q4-Large'])
    pred_residuals = df.groupby('pred_quartile')['residual'].agg(['mean', 'std', 'count'])
    pred_residuals.columns = ['mean_residual', 'std_residual', 'count']
    results['by_predicted'] = pred_residuals.reset_index()

    return results


def test_residual_correlations(df):
    """Test if residuals correlate with other features."""
    correlations = {}

    # With load deviation
    valid = df[['residual', 'load_deviation']].dropna()
    if len(valid) > 100:
        corr, pval = stats.pearsonr(valid['residual'], valid['load_deviation'])
        correlations['load_deviation'] = {'correlation': corr, 'p_value': pval}

    # With hour (circular, so use sin/cos)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    corr_sin, _ = stats.pearsonr(df['residual'], df['hour_sin'])
    corr_cos, _ = stats.pearsonr(df['residual'], df['hour_cos'])
    correlations['hour_sin'] = {'correlation': corr_sin}
    correlations['hour_cos'] = {'correlation': corr_cos}

    # With lagged residual (autocorrelation)
    df['residual_lag1'] = df['residual'].shift(1)
    valid = df[['residual', 'residual_lag1']].dropna()
    corr, pval = stats.pearsonr(valid['residual'], valid['residual_lag1'])
    correlations['residual_lag1'] = {'correlation': corr, 'p_value': pval}

    df['residual_lag2'] = df['residual'].shift(2)
    valid = df[['residual', 'residual_lag2']].dropna()
    corr, pval = stats.pearsonr(valid['residual'], valid['residual_lag2'])
    correlations['residual_lag2'] = {'correlation': corr, 'p_value': pval}

    return correlations, df


def main():
    print("=" * 70)
    print("RESIDUAL ANALYSIS")
    print("=" * 70)
    print("\nAnalyzing what patterns remain after accounting for regulation")

    df = load_and_prepare_data()
    print(f"\nDataset: {len(df):,} rows")

    # =================================================================
    # 1. FIT SIMPLE MODEL
    # =================================================================
    print("\n" + "=" * 70)
    print("1. SIMPLE LINEAR MODEL: Imbalance ~ Regulation")
    print("=" * 70)

    df, model, r2, coef, intercept = fit_simple_model(df)

    print(f"\n  Model: imbalance = {coef:.4f} × regulation + {intercept:.4f}")
    print(f"  R² = {r2:.4f}")
    print(f"  Residual std = {df['residual'].std():.2f} MWh")
    print(f"  Residual mean = {df['residual'].mean():.4f} MWh (should be ~0)")

    # =================================================================
    # 2. RESIDUAL DISTRIBUTION
    # =================================================================
    print("\n" + "=" * 70)
    print("2. RESIDUAL DISTRIBUTION")
    print("=" * 70)

    residual_stats = {
        'mean': df['residual'].mean(),
        'std': df['residual'].std(),
        'skewness': stats.skew(df['residual']),
        'kurtosis': stats.kurtosis(df['residual']),
        'min': df['residual'].min(),
        'max': df['residual'].max(),
        'q25': df['residual'].quantile(0.25),
        'q75': df['residual'].quantile(0.75)
    }

    print(f"  Mean: {residual_stats['mean']:.3f}")
    print(f"  Std: {residual_stats['std']:.3f}")
    print(f"  Skewness: {residual_stats['skewness']:.3f} (0 = symmetric)")
    print(f"  Kurtosis: {residual_stats['kurtosis']:.3f} (0 = normal)")
    print(f"  Range: [{residual_stats['min']:.1f}, {residual_stats['max']:.1f}]")
    print(f"  IQR: [{residual_stats['q25']:.1f}, {residual_stats['q75']:.1f}]")

    # Normality test
    _, shapiro_p = stats.shapiro(df['residual'].sample(min(5000, len(df)), random_state=42))
    print(f"  Shapiro-Wilk p-value: {shapiro_p:.4f} ({'Normal' if shapiro_p > 0.05 else 'Not normal'})")

    # =================================================================
    # 3. RESIDUAL PATTERNS BY CONDITION
    # =================================================================
    print("\n" + "=" * 70)
    print("3. RESIDUAL PATTERNS BY CONDITION")
    print("=" * 70)

    patterns = analyze_residual_patterns(df)

    # Save all pattern data
    for name, pattern_df in patterns.items():
        pattern_df.to_csv(DATA_DIR / f'residual_{name}.csv', index=False)

    # Report significant patterns
    print("\n  By Hour (mean residual):")
    hour_df = patterns['by_hour']
    significant_hours = hour_df[hour_df['mean_residual'].abs() > 1.0]
    if len(significant_hours) > 0:
        for _, row in significant_hours.iterrows():
            print(f"    Hour {int(row['hour']):2d}: {row['mean_residual']:+.2f} MWh")
    else:
        print("    No hours with |mean residual| > 1 MWh")

    print("\n  By Day of Week (mean residual):")
    dow_df = patterns['by_dow']
    for _, row in dow_df.iterrows():
        marker = "*" if abs(row['mean_residual']) > 0.5 else " "
        print(f"    {row['day_name']}: {row['mean_residual']:+.2f} MWh {marker}")

    # =================================================================
    # 4. RESIDUAL CORRELATIONS
    # =================================================================
    print("\n" + "=" * 70)
    print("4. RESIDUAL CORRELATIONS (exploitable structure?)")
    print("=" * 70)

    correlations, df = test_residual_correlations(df)

    corr_results = []
    for feature, vals in correlations.items():
        corr = vals['correlation']
        pval = vals.get('p_value', np.nan)
        print(f"  {feature:20s}: r = {corr:+.4f}")
        corr_results.append({'feature': feature, 'correlation': corr, 'p_value': pval})

    corr_df = pd.DataFrame(corr_results)
    corr_df.to_csv(DATA_DIR / 'residual_correlations.csv', index=False)

    # =================================================================
    # 5. CAN OTHER FEATURES HELP?
    # =================================================================
    print("\n" + "=" * 70)
    print("5. CAN OTHER FEATURES REDUCE RESIDUALS?")
    print("=" * 70)

    from sklearn.metrics import r2_score, mean_absolute_error

    # Test adding features to the model
    feature_tests = [
        ('regulation only', ['reg_mean']),
        ('+ load_deviation', ['reg_mean', 'load_deviation']),
        ('+ hour (sin/cos)', ['reg_mean', 'hour_sin', 'hour_cos']),
        ('+ load_dev + hour', ['reg_mean', 'load_deviation', 'hour_sin', 'hour_cos']),
        ('+ residual_lag1', ['reg_mean', 'residual_lag1']),
        ('full model', ['reg_mean', 'load_deviation', 'hour_sin', 'hour_cos', 'residual_lag1']),
    ]

    improvement_results = []
    for name, features in feature_tests:
        valid = df[['imbalance'] + features].dropna()
        if len(valid) < 100:
            continue

        X = valid[features].values
        y = valid['imbalance'].values

        model_test = LinearRegression()
        model_test.fit(X, y)
        y_pred = model_test.predict(X)

        r2_test = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        residual_std = (y - y_pred).std()

        improvement_results.append({
            'model': name,
            'r2': r2_test,
            'mae': mae,
            'residual_std': residual_std,
            'n_samples': len(valid)
        })
        print(f"  {name:25s}: R² = {r2_test:.4f}, MAE = {mae:.2f}, Std = {residual_std:.2f}")

    improvement_df = pd.DataFrame(improvement_results)
    improvement_df.to_csv(DATA_DIR / 'model_improvements.csv', index=False)

    # =================================================================
    # 6. VISUALIZATION
    # =================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Residual Analysis: What Remains After Regulation?', fontsize=14, fontweight='bold')

    # Plot 1: Residual distribution
    ax = axes[0, 0]
    ax.hist(df['residual'], bins=50, density=True, alpha=0.7, color='tab:blue', edgecolor='black')
    # Overlay normal distribution
    x = np.linspace(df['residual'].min(), df['residual'].max(), 100)
    ax.plot(x, stats.norm.pdf(x, df['residual'].mean(), df['residual'].std()),
            'r-', linewidth=2, label='Normal fit')
    ax.set_xlabel('Residual (MWh)')
    ax.set_ylabel('Density')
    ax.set_title(f'Residual Distribution (std={df["residual"].std():.2f})')
    ax.legend()

    # Plot 2: Residual vs Predicted (heteroscedasticity check)
    ax = axes[0, 1]
    sample = df.sample(min(3000, len(df)), random_state=42)
    ax.scatter(sample['predicted'], sample['residual'], alpha=0.3, s=10, c='tab:blue')
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Predicted Imbalance (MWh)')
    ax.set_ylabel('Residual (MWh)')
    ax.set_title('Residual vs Predicted')
    ax.grid(True, alpha=0.3)

    # Plot 3: Residual by hour
    ax = axes[0, 2]
    hour_df = patterns['by_hour']
    ax.bar(hour_df['hour'], hour_df['mean_residual'], color='tab:green', alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.fill_between(hour_df['hour'],
                    hour_df['mean_residual'] - hour_df['std_residual']/np.sqrt(hour_df['count']),
                    hour_df['mean_residual'] + hour_df['std_residual']/np.sqrt(hour_df['count']),
                    alpha=0.3, color='tab:green')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Mean Residual (MWh)')
    ax.set_title('Residual Pattern by Hour')
    ax.set_xticks(range(0, 24, 3))

    # Plot 4: Residual autocorrelation
    ax = axes[1, 0]
    lags = range(1, 13)
    autocorrs = []
    for lag in lags:
        df[f'res_lag{lag}'] = df['residual'].shift(lag)
        valid = df[['residual', f'res_lag{lag}']].dropna()
        corr, _ = stats.pearsonr(valid['residual'], valid[f'res_lag{lag}'])
        autocorrs.append(corr)
    ax.bar(lags, autocorrs, color='tab:orange', alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(0.05, color='red', linestyle='--', alpha=0.5)
    ax.axhline(-0.05, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag (settlement periods)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Residual Autocorrelation')

    # Plot 5: Q-Q plot
    ax = axes[1, 1]
    stats.probplot(df['residual'], dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)')

    # Plot 6: Model improvement
    ax = axes[1, 2]
    improvement_df_plot = improvement_df.copy()
    ax.barh(improvement_df_plot['model'], improvement_df_plot['r2'], color='tab:purple', alpha=0.7)
    ax.set_xlabel('R²')
    ax.set_title('Model Improvement with Features')
    for i, row in improvement_df_plot.iterrows():
        ax.text(row['r2'] + 0.01, i, f"{row['r2']:.3f}", va='center')
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'residual_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: residual_analysis.png")

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline_r2 = improvement_df[improvement_df['model'] == 'regulation only']['r2'].values[0]
    best_r2 = improvement_df['r2'].max()
    best_model = improvement_df.loc[improvement_df['r2'].idxmax(), 'model']

    print(f"\nBaseline (regulation only): R² = {baseline_r2:.4f}")
    print(f"Best model ({best_model}): R² = {best_r2:.4f}")
    print(f"Potential improvement: +{(best_r2 - baseline_r2)*100:.1f}% R²")

    print(f"\nResidual characteristics:")
    print(f"  - Std: {df['residual'].std():.2f} MWh (unexplained variation)")
    print(f"  - Autocorr(lag1): {autocorrs[0]:.3f} (temporal structure)")
    print(f"  - Load deviation corr: {correlations['load_deviation']['correlation']:.3f}")

    if abs(correlations['load_deviation']['correlation']) > 0.05:
        print("\n  -> Load deviation has weak but exploitable signal")
    if abs(autocorrs[0]) > 0.1:
        print("  -> Residuals show temporal autocorrelation - lags help")

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
