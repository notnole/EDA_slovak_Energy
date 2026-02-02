"""
Residual (Model Error) Analysis
================================
Analyze correlations in our model's prediction errors to find:
1. Are residuals autocorrelated? (exploitable signal remaining)
2. Are residuals correlated across horizons?
3. What patterns remain in the errors?
4. Can we build a "correction" model on top?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent.parent  # ipesoft_eda_data
OUTPUT_PATH = Path(__file__).parent / 'residual_analysis'
OUTPUT_PATH.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')


def load_and_prepare():
    """Load and prepare data with all features."""
    df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['year'] = df['datetime'].dt.year
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']

    # 3-min load
    load_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'load_3min.csv')
    load_3min['datetime'] = pd.to_datetime(load_3min['datetime'])
    load_3min['hour_start'] = load_3min['datetime'].dt.floor('H')
    load_hourly = load_3min.groupby('hour_start').agg({'load_mw': ['std', 'first', 'last']}).reset_index()
    load_hourly.columns = ['datetime', 'load_std_3min', 'load_first', 'load_last']
    load_hourly['load_trend_3min'] = load_hourly['load_last'] - load_hourly['load_first']
    df = df.merge(load_hourly[['datetime', 'load_std_3min', 'load_trend_3min']], on='datetime', how='left')

    # Regulation
    reg_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'regulation_3min.csv')
    reg_3min['datetime'] = pd.to_datetime(reg_3min['datetime'])
    reg_3min['hour_start'] = reg_3min['datetime'].dt.floor('H')
    reg_hourly = reg_3min.groupby('hour_start').agg({'regulation_mw': ['mean', 'std']}).reset_index()
    reg_hourly.columns = ['datetime', 'reg_mean', 'reg_std']
    df = df.merge(reg_hourly, on='datetime', how='left')

    # Features
    for lag in range(1, 13):
        df[f'error_lag{lag}'] = df['error'].shift(lag)
    for window in [3, 6, 12, 24]:
        df[f'error_roll_mean_{window}h'] = df['error'].shift(1).rolling(window).mean()
        df[f'error_roll_std_{window}h'] = df['error'].shift(1).rolling(window).std()

    # Error trends
    df['error_trend_3h'] = df['error_lag1'] - df['error_lag3']
    df['error_trend_6h'] = df['error_lag1'] - df['error_lag6']
    df['error_momentum'] = 0.5 * (df['error_lag1'] - df['error_lag2']) + \
                           0.3 * (df['error_lag2'] - df['error_lag3']) + \
                           0.2 * (df['error_lag3'] - df['error_lag4'])
    df['error_accel'] = (df['error_lag1'] - df['error_lag2']) - (df['error_lag2'] - df['error_lag3'])

    df['load_volatility_lag1'] = df['load_std_3min'].shift(1)
    df['load_trend_lag1'] = df['load_trend_3min'].shift(1)
    for lag in range(1, 4):
        df[f'reg_mean_lag{lag}'] = df['reg_mean'].shift(lag)

    seasonal = df.groupby(['dow', 'hour'])['error'].transform('mean')
    df['seasonal_error'] = seasonal
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)

    # Targets
    for h in range(1, 6):
        df[f'target_h{h}'] = df['error'].shift(-h)

    return df


def get_features():
    return [
        'error_lag1', 'error_lag2', 'error_lag3', 'error_lag4', 'error_lag5',
        'error_lag6', 'error_lag7', 'error_lag8',
        'error_roll_mean_3h', 'error_roll_std_3h',
        'error_roll_mean_6h', 'error_roll_std_6h',
        'error_roll_mean_12h', 'error_roll_std_12h',
        'error_roll_mean_24h', 'error_roll_std_24h',
        'error_trend_3h', 'error_trend_6h', 'error_momentum', 'error_accel',
        'load_volatility_lag1', 'load_trend_lag1',
        'reg_mean_lag1', 'reg_mean_lag2', 'reg_mean_lag3',
        'seasonal_error',
        'hour', 'hour_sin', 'hour_cos', 'dow', 'is_weekend',
    ]


def train_and_get_residuals(df, features):
    """Train models and get residuals for analysis."""
    results = {}

    for h in range(1, 6):
        target = f'target_h{h}'
        available = [f for f in features if f in df.columns]

        df_model = df.dropna(subset=[target] + available).copy()
        train = df_model[df_model['year'] < 2025]
        test = df_model[df_model['year'] >= 2025].copy()

        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=8, num_leaves=50,
            min_child_samples=30, random_state=42, verbosity=-1
        )
        model.fit(train[available], train[target])

        test['pred'] = model.predict(test[available])
        test['residual'] = test[target] - test['pred']  # actual - predicted

        results[h] = test[['datetime', 'hour', 'dow', target, 'pred', 'residual']].copy()

    return results


def analyze_residual_autocorrelation(results):
    """Analyze autocorrelation of residuals."""
    print("\n" + "=" * 70)
    print("1. RESIDUAL AUTOCORRELATION")
    print("=" * 70)
    print("\n   Are our prediction errors correlated over time?")
    print("   If yes, we could predict and correct them.\n")

    for h in [1, 3, 5]:
        residuals = results[h]['residual'].values

        print(f"   H+{h} Residual Autocorrelation:")
        autocorrs = []
        for lag in [1, 2, 3, 6, 12, 24]:
            if len(residuals) > lag:
                r = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                autocorrs.append((lag, r))
                print(f"      Lag {lag:2}h: r = {r:+.3f}")
        print()

    return autocorrs


def analyze_cross_horizon_correlation(results):
    """Analyze correlation between residuals at different horizons."""
    print("\n" + "=" * 70)
    print("2. CROSS-HORIZON RESIDUAL CORRELATION")
    print("=" * 70)
    print("\n   Are errors at H+1 correlated with H+2, H+3, etc.?")
    print("   If yes, we could use H+1 residual to correct H+2-5.\n")

    # Align residuals by datetime
    merged = results[1][['datetime', 'residual']].rename(columns={'residual': 'res_h1'})
    for h in range(2, 6):
        merged = merged.merge(
            results[h][['datetime', 'residual']].rename(columns={'residual': f'res_h{h}'}),
            on='datetime', how='inner'
        )

    print("   Correlation Matrix:")
    print("         H+1    H+2    H+3    H+4    H+5")
    for h1 in range(1, 6):
        row = f"   H+{h1}  "
        for h2 in range(1, 6):
            r = merged[f'res_h{h1}'].corr(merged[f'res_h{h2}'])
            row += f"{r:+.2f}  "
        print(row)

    return merged


def analyze_residual_patterns(results):
    """Analyze patterns in residuals by hour/dow."""
    print("\n" + "=" * 70)
    print("3. RESIDUAL PATTERNS BY HOUR AND DAY")
    print("=" * 70)
    print("\n   Do residuals have systematic biases we missed?\n")

    test = results[1]

    # By hour
    by_hour = test.groupby('hour')['residual'].agg(['mean', 'std', 'count'])
    print("   Mean Residual by Hour (H+1):")
    print("   Hour | Mean    | Std     | Significant?")
    print("   " + "-" * 45)

    for hour, row in by_hour.iterrows():
        # Is mean significantly different from 0? (rough t-test)
        se = row['std'] / np.sqrt(row['count'])
        t = abs(row['mean'] / se) if se > 0 else 0
        sig = "YES" if t > 2 else "no"
        print(f"   {hour:4} | {row['mean']:+6.1f} | {row['std']:6.1f} | {sig}")

    # By dow
    print("\n   Mean Residual by Day of Week (H+1):")
    by_dow = test.groupby('dow')['residual'].agg(['mean', 'std', 'count'])
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for dow, row in by_dow.iterrows():
        se = row['std'] / np.sqrt(row['count'])
        t = abs(row['mean'] / se) if se > 0 else 0
        sig = "YES" if t > 2 else "no"
        print(f"   {days[dow]:3} | {row['mean']:+6.1f} | {row['std']:6.1f} | {sig}")

    return by_hour, by_dow


def test_residual_correction(results):
    """Test if we can predict and correct residuals."""
    print("\n" + "=" * 70)
    print("4. RESIDUAL CORRECTION MODEL")
    print("=" * 70)
    print("\n   Can we predict our own errors and correct them?\n")

    test = results[1].copy()

    # Create lagged residuals (would need to be from actual past predictions)
    # For simulation, we use test set residuals
    test = test.sort_values('datetime').reset_index(drop=True)
    test['res_lag1'] = test['residual'].shift(1)
    test['res_lag2'] = test['residual'].shift(2)
    test['res_lag3'] = test['residual'].shift(3)
    test['res_roll_mean'] = test['residual'].shift(1).rolling(3).mean()

    # Can we predict current residual from past residuals?
    test_valid = test.dropna(subset=['res_lag1', 'res_lag2', 'res_lag3'])

    # Split for fair evaluation
    n = len(test_valid)
    train_res = test_valid.iloc[:n//2]
    test_res = test_valid.iloc[n//2:]

    # Train residual prediction model
    res_features = ['res_lag1', 'res_lag2', 'res_lag3', 'res_roll_mean', 'hour', 'dow']
    model_res = lgb.LGBMRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=-1)
    model_res.fit(train_res[res_features], train_res['residual'])

    # Predict residuals
    test_res = test_res.copy()
    test_res['pred_residual'] = model_res.predict(test_res[res_features])

    # Correct the original prediction
    test_res['corrected_pred'] = test_res['pred'] + test_res['pred_residual']
    test_res['corrected_residual'] = test_res['target_h1'] - test_res['corrected_pred']

    # Compare
    original_mae = np.abs(test_res['residual']).mean()
    corrected_mae = np.abs(test_res['corrected_residual']).mean()
    improvement = (original_mae - corrected_mae) / original_mae * 100

    print(f"   Original Model MAE:     {original_mae:.2f} MW")
    print(f"   After Correction MAE:   {corrected_mae:.2f} MW")
    print(f"   Improvement:            {improvement:+.1f}%")

    if improvement > 0:
        print(f"\n   Residual correction helps by {original_mae - corrected_mae:.2f} MW!")
    else:
        print(f"\n   Residual correction doesn't help (residuals are unpredictable)")

    return test_res, improvement


def plot_residual_analysis(results, merged):
    """Create visualization of residual analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Residual autocorrelation
    residuals = results[1]['residual'].values
    lags = range(1, 25)
    autocorrs = [np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1] for lag in lags]

    axes[0, 0].bar(lags, autocorrs, color='steelblue', alpha=0.7)
    axes[0, 0].axhline(y=0, color='black', linestyle='-')
    axes[0, 0].axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Lag (hours)')
    axes[0, 0].set_ylabel('Autocorrelation')
    axes[0, 0].set_title('H+1 Residual Autocorrelation')

    # 2. Cross-horizon correlation heatmap
    corr_matrix = merged[[f'res_h{h}' for h in range(1, 6)]].corr()
    im = axes[0, 1].imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 1].set_xticks(range(5))
    axes[0, 1].set_yticks(range(5))
    axes[0, 1].set_xticklabels([f'H+{h}' for h in range(1, 6)])
    axes[0, 1].set_yticklabels([f'H+{h}' for h in range(1, 6)])
    axes[0, 1].set_title('Cross-Horizon Residual Correlation')
    for i in range(5):
        for j in range(5):
            axes[0, 1].text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                           ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=axes[0, 1])

    # 3. Residual by hour
    by_hour = results[1].groupby('hour')['residual'].mean()
    colors = ['green' if x < 0 else 'red' for x in by_hour.values]
    axes[1, 0].bar(by_hour.index, by_hour.values, color=colors, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Mean Residual (MW)')
    axes[1, 0].set_title('H+1 Mean Residual by Hour (bias)')

    # 4. Residual distribution
    axes[1, 1].hist(results[1]['residual'], bins=50, density=True, alpha=0.7, color='steelblue')
    axes[1, 1].axvline(x=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Residual (MW)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title(f'H+1 Residual Distribution (std={results[1]["residual"].std():.1f} MW)')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'residual_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Saved: {OUTPUT_PATH / 'residual_analysis.png'}")


def main():
    print("=" * 70)
    print("RESIDUAL (MODEL ERROR) ANALYSIS")
    print("=" * 70)

    # Load data and train models
    print("\nLoading data and training models...")
    df = load_and_prepare()
    features = get_features()
    results = train_and_get_residuals(df, features)

    print(f"\n   Test set size: {len(results[1]):,} hours")
    print(f"   H+1 Model MAE: {np.abs(results[1]['residual']).mean():.2f} MW")

    # Analysis
    autocorrs = analyze_residual_autocorrelation(results)
    merged = analyze_cross_horizon_correlation(results)
    by_hour, by_dow = analyze_residual_patterns(results)
    test_res, improvement = test_residual_correction(results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: WHAT CAN WE DO WITH RESIDUAL CORRELATIONS?")
    print("=" * 70)

    h1_lag1_corr = np.corrcoef(results[1]['residual'].values[:-1],
                               results[1]['residual'].values[1:])[0, 1]

    print(f"""
   1. RESIDUAL AUTOCORRELATION
      - Lag-1h correlation: r = {h1_lag1_corr:.3f}
      - This is {'WEAK' if abs(h1_lag1_corr) < 0.2 else 'MODERATE' if abs(h1_lag1_corr) < 0.5 else 'STRONG'}
      - {'Little exploitable signal' if abs(h1_lag1_corr) < 0.2 else 'Some signal to exploit!'}

   2. CROSS-HORIZON CORRELATION
      - H+1 vs H+2 residuals: r = {merged['res_h1'].corr(merged['res_h2']):.3f}
      - H+1 vs H+5 residuals: r = {merged['res_h1'].corr(merged['res_h5']):.3f}
      - {'Can use H+1 errors to help H+2-5' if merged['res_h1'].corr(merged['res_h2']) > 0.3 else 'Limited cross-horizon benefit'}

   3. RESIDUAL CORRECTION MODEL
      - Improvement from predicting residuals: {improvement:+.1f}%
      - {'Worth implementing!' if improvement > 2 else 'Not worth the complexity'}

   4. REMAINING BIASES
      - Hour biases: {'Significant' if by_hour['mean'].abs().max() > 5 else 'Small'}
      - DoW biases: {'Significant' if by_dow['mean'].abs().max() > 5 else 'Small'}
""")

    # Generate plots
    plot_residual_analysis(results, merged)

    return results, merged


if __name__ == "__main__":
    results, merged = main()
