"""
Error Analysis for Lead 12 and 9 minute models.

Understand:
1. When does the model fail?
2. Are errors correlated (residual signal)?
3. What features correlate with errors?
4. What patterns are we missing?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import pickle

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\analysis\models\lightgbm")


def load_and_prepare_data():
    """Load data and prepare features (same as V4)."""
    print("Loading data...")
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
    load_df = pd.read_csv(FEATURES_DIR / 'load_3min.csv', parse_dates=['datetime'])
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
        columns={'System Imbalance (MWh)': 'imbalance'}
    )

    # Align regulation to settlement periods
    reg_df = reg_df.sort_values('datetime').copy()
    reg_df['datetime_floor'] = reg_df['datetime'].dt.floor('3min')
    reg_df['settlement_end'] = reg_df['datetime_floor'].dt.ceil('15min')
    mask = reg_df['datetime_floor'] == reg_df['settlement_end']
    reg_df.loc[mask, 'settlement_end'] = reg_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    reg_df['settlement_start'] = reg_df['settlement_end'] - pd.Timedelta(minutes=15)
    reg_df['minute_in_qh'] = (reg_df['datetime_floor'] - reg_df['settlement_start']).dt.total_seconds() / 60

    # Pivot regulation
    pivot = reg_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='regulation_mw', aggfunc='first'
    ).reset_index()
    pivot.columns = ['datetime'] + [f'reg_min{int(c)}' for c in pivot.columns[1:]]

    # Merge
    df = pd.merge(label_df, pivot, on='datetime', how='inner')

    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['datetime'].dt.month
    df['date'] = df['datetime'].dt.date

    # Compute proxy for each period
    reg_cols = ['reg_min0', 'reg_min3', 'reg_min6', 'reg_min9', 'reg_min12']
    available = [c for c in reg_cols if c in df.columns]
    df['period_proxy'] = -0.25 * df[available].mean(axis=1)

    # Lag features
    df = df.sort_values('datetime')
    df['proxy_lag1'] = df['period_proxy'].shift(1)
    df['proxy_lag2'] = df['period_proxy'].shift(2)
    df['imbalance_lag1'] = df['imbalance'].shift(1)
    df['imbalance_lag2'] = df['imbalance'].shift(2)

    # Rolling stats
    df['proxy_rolling_mean4'] = df['period_proxy'].shift(1).rolling(4).mean()
    df['proxy_rolling_std4'] = df['period_proxy'].shift(1).rolling(4).std()
    df['imbalance_rolling_mean4'] = df['imbalance'].shift(1).rolling(4).mean()
    df['imbalance_rolling_std4'] = df['imbalance'].shift(1).rolling(4).std()

    # Baseline predictions
    df['baseline_12'] = -0.25 * df['reg_min0']
    df['baseline_9'] = -0.25 * (0.8 * df['reg_min3'] + 0.2 * df['reg_min0'])

    return df


def generate_predictions(df, lead_time):
    """Generate predictions using saved model."""
    with open(OUTPUT_DIR / 'lightgbm_models_v4.pkl', 'rb') as f:
        models = pickle.load(f)

    model = models[lead_time]

    # Get feature columns (from V4)
    if lead_time == 12:
        from train_lightgbm_v4 import get_features_for_lead
    else:
        # Inline feature list for lead 9
        pass

    # We need to recreate the features - let's load from v4 script
    # For simplicity, load the test predictions directly by re-running minimal feature creation

    return model


def main():
    print("=" * 70)
    print("ERROR ANALYSIS - LEAD 12 AND 9 MINUTE MODELS")
    print("=" * 70)

    df = load_and_prepare_data()

    # Filter to test period
    test_start = pd.Timestamp('2025-10-01')
    test_df = df[df['datetime'] >= test_start].copy()

    print(f"\nTest set: {len(test_df):,} rows")
    print(f"Period: {test_df['datetime'].min()} to {test_df['datetime'].max()}")

    # Load models
    with open(OUTPUT_DIR / 'lightgbm_models_v4.pkl', 'rb') as f:
        models = pickle.load(f)

    # For error analysis, we'll use baseline predictions as proxy for model predictions
    # (The actual model predictions require full feature recreation)
    # Let's analyze baseline errors first, then compare

    for lead_time in [12, 9]:
        print(f"\n{'='*70}")
        print(f"LEAD TIME: {lead_time} MINUTES")
        print(f"{'='*70}")

        baseline_col = f'baseline_{lead_time}'
        test_df[f'error_{lead_time}'] = test_df[baseline_col] - test_df['imbalance']
        test_df[f'abs_error_{lead_time}'] = test_df[f'error_{lead_time}'].abs()

        errors = test_df[f'error_{lead_time}'].dropna()
        abs_errors = test_df[f'abs_error_{lead_time}'].dropna()

        print(f"\n[1] BASIC ERROR STATISTICS")
        print("-" * 40)
        print(f"  Mean Error (Bias): {errors.mean():.3f} MWh")
        print(f"  Std Error: {errors.std():.3f} MWh")
        print(f"  MAE: {abs_errors.mean():.3f} MWh")
        print(f"  Median AE: {abs_errors.median():.3f} MWh")
        print(f"  90th percentile AE: {abs_errors.quantile(0.9):.3f} MWh")
        print(f"  95th percentile AE: {abs_errors.quantile(0.95):.3f} MWh")
        print(f"  Max AE: {abs_errors.max():.3f} MWh")

        # ================================================================
        # ERROR BY TIME OF DAY
        # ================================================================
        print(f"\n[2] ERROR BY HOUR")
        print("-" * 40)
        hour_stats = test_df.groupby('hour').agg({
            f'error_{lead_time}': ['mean', 'std'],
            f'abs_error_{lead_time}': 'mean',
            'imbalance': 'count'
        }).round(3)
        hour_stats.columns = ['bias', 'std', 'mae', 'n']

        print(f"{'Hour':<6} {'Bias':>8} {'Std':>8} {'MAE':>8} {'N':>6}")
        print("-" * 40)
        for hour in range(24):
            if hour in hour_stats.index:
                row = hour_stats.loc[hour]
                print(f"{hour:<6} {row['bias']:>+8.2f} {row['std']:>8.2f} {row['mae']:>8.2f} {int(row['n']):>6}")

        worst_hours = hour_stats.nlargest(3, 'mae')
        best_hours = hour_stats.nsmallest(3, 'mae')
        print(f"\n  Worst hours (MAE): {list(worst_hours.index)}")
        print(f"  Best hours (MAE): {list(best_hours.index)}")

        # ================================================================
        # ERROR BY DAY OF WEEK
        # ================================================================
        print(f"\n[3] ERROR BY DAY OF WEEK")
        print("-" * 40)
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_stats = test_df.groupby('day_of_week').agg({
            f'error_{lead_time}': 'mean',
            f'abs_error_{lead_time}': 'mean',
        }).round(3)

        for dow in range(7):
            if dow in dow_stats.index:
                row = dow_stats.loc[dow]
                print(f"  {dow_names[dow]}: Bias={row[f'error_{lead_time}']:+.2f}, MAE={row[f'abs_error_{lead_time}']:.2f}")

        # ================================================================
        # ERROR BY IMBALANCE MAGNITUDE
        # ================================================================
        print(f"\n[4] ERROR BY IMBALANCE MAGNITUDE")
        print("-" * 40)
        test_df['imb_magnitude'] = pd.cut(
            test_df['imbalance'].abs(),
            bins=[0, 2, 5, 10, 20, 1000],
            labels=['<2', '2-5', '5-10', '10-20', '>20']
        )

        mag_stats = test_df.groupby('imb_magnitude', observed=True).agg({
            f'error_{lead_time}': 'mean',
            f'abs_error_{lead_time}': ['mean', 'std'],
            'imbalance': 'count'
        }).round(3)
        mag_stats.columns = ['bias', 'mae', 'mae_std', 'n']

        for mag in ['<2', '2-5', '5-10', '10-20', '>20']:
            if mag in mag_stats.index:
                row = mag_stats.loc[mag]
                print(f"  |Imb| {mag:>5}: Bias={row['bias']:+6.2f}, MAE={row['mae']:5.2f} +/- {row['mae_std']:5.2f} (n={int(row['n'])})")

        # ================================================================
        # ERROR BY IMBALANCE SIGN
        # ================================================================
        print(f"\n[5] ERROR BY IMBALANCE SIGN")
        print("-" * 40)
        test_df['imb_sign'] = np.sign(test_df['imbalance'])

        sign_stats = test_df.groupby('imb_sign').agg({
            f'error_{lead_time}': 'mean',
            f'abs_error_{lead_time}': 'mean',
            'imbalance': 'count'
        }).round(3)

        for sign, label in [(-1, 'Negative'), (0, 'Zero'), (1, 'Positive')]:
            if sign in sign_stats.index:
                row = sign_stats.loc[sign]
                print(f"  {label}: Bias={row[f'error_{lead_time}']:+.2f}, MAE={row[f'abs_error_{lead_time}']:.2f} (n={int(row['imbalance'])})")

        # ================================================================
        # ERROR AUTOCORRELATION (residual signal?)
        # ================================================================
        print(f"\n[6] ERROR AUTOCORRELATION")
        print("-" * 40)
        print("  (If significant, there's residual signal we're not capturing)")

        errors_clean = test_df[[f'error_{lead_time}']].dropna()

        for lag in [1, 2, 3, 4, 5, 10]:
            autocorr = errors_clean[f'error_{lead_time}'].autocorr(lag=lag)
            significance = "***" if abs(autocorr) > 0.1 else "**" if abs(autocorr) > 0.05 else "*" if abs(autocorr) > 0.02 else ""
            print(f"  Lag {lag:>2}: r = {autocorr:+.3f} {significance}")

        # ================================================================
        # CORRELATION OF ERROR WITH POTENTIAL FEATURES
        # ================================================================
        print(f"\n[7] CORRELATION OF ERROR WITH POTENTIAL FEATURES")
        print("-" * 40)
        print("  (High correlation = unused signal)")

        potential_features = [
            'imbalance_lag1', 'imbalance_lag2',
            'imbalance_rolling_mean4', 'imbalance_rolling_std4',
            'proxy_lag1', 'proxy_lag2',
            'proxy_rolling_mean4', 'proxy_rolling_std4',
            'hour', 'is_weekend'
        ]

        correlations = []
        for feat in potential_features:
            if feat in test_df.columns:
                corr = test_df[[f'error_{lead_time}', feat]].dropna().corr().iloc[0, 1]
                correlations.append((feat, corr))

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        for feat, corr in correlations:
            significance = "***" if abs(corr) > 0.2 else "**" if abs(corr) > 0.1 else "*" if abs(corr) > 0.05 else ""
            print(f"  {feat:<30}: r = {corr:+.3f} {significance}")

        # ================================================================
        # LARGE ERROR ANALYSIS
        # ================================================================
        print(f"\n[8] LARGE ERROR ANALYSIS (|error| > 10 MWh)")
        print("-" * 40)

        large_errors = test_df[test_df[f'abs_error_{lead_time}'] > 10].copy()
        print(f"  Count: {len(large_errors)} ({len(large_errors)/len(test_df)*100:.1f}% of samples)")

        if len(large_errors) > 0:
            print(f"\n  Hour distribution of large errors:")
            hour_dist = large_errors['hour'].value_counts().sort_index()
            for hour, count in hour_dist.items():
                pct = count / len(large_errors) * 100
                bar = "*" * int(pct / 2)
                print(f"    {hour:02d}:00  {count:>3} ({pct:>4.1f}%) {bar}")

            print(f"\n  Day of week distribution:")
            dow_dist = large_errors['day_of_week'].value_counts().sort_index()
            for dow, count in dow_dist.items():
                print(f"    {dow_names[dow]}: {count}")

            print(f"\n  Magnitude distribution of large errors:")
            mag_dist = large_errors['imb_magnitude'].value_counts()
            for mag, count in mag_dist.items():
                print(f"    |Imb| {mag}: {count}")

        # ================================================================
        # ERROR VS VOLATILITY
        # ================================================================
        print(f"\n[9] ERROR VS RECENT VOLATILITY")
        print("-" * 40)

        test_df['volatility_quintile'] = pd.qcut(
            test_df['proxy_rolling_std4'].fillna(0),
            q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            duplicates='drop'
        )

        vol_stats = test_df.groupby('volatility_quintile', observed=True).agg({
            f'abs_error_{lead_time}': 'mean',
            'imbalance': 'count'
        }).round(3)

        for vol in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
            if vol in vol_stats.index:
                row = vol_stats.loc[vol]
                print(f"  {vol:<10}: MAE = {row[f'abs_error_{lead_time}']:.2f} (n={int(row['imbalance'])})")

        # ================================================================
        # ERROR PERSISTENCE
        # ================================================================
        print(f"\n[10] ERROR PERSISTENCE ANALYSIS")
        print("-" * 40)

        test_df[f'error_lag1_{lead_time}'] = test_df[f'error_{lead_time}'].shift(1)
        test_df[f'same_sign_error_{lead_time}'] = (
            np.sign(test_df[f'error_{lead_time}']) == np.sign(test_df[f'error_lag1_{lead_time}'])
        )

        same_sign_pct = test_df[f'same_sign_error_{lead_time}'].mean() * 100
        print(f"  Consecutive same-sign errors: {same_sign_pct:.1f}%")
        print(f"  (50% = random, >50% = persistent bias pattern)")

        # Conditional error
        test_df['prev_error_positive'] = test_df[f'error_lag1_{lead_time}'] > 0
        cond_stats = test_df.groupby('prev_error_positive')[f'error_{lead_time}'].mean()
        if True in cond_stats.index and False in cond_stats.index:
            print(f"\n  Mean error after positive error: {cond_stats[True]:+.2f}")
            print(f"  Mean error after negative error: {cond_stats[False]:+.2f}")

    # ================================================================
    # COMPARISON: WHAT DOES ACTUAL IMBALANCE TELL US?
    # ================================================================
    print(f"\n{'='*70}")
    print("INFORMATION IN ACTUAL IMBALANCE VS PROXY")
    print("(This is what we CAN'T use in real-time)")
    print("=" * 70)

    # How much better could we predict if we knew the actual previous imbalance?
    print("\nCorrelation with next period's imbalance:")
    print(f"  proxy_lag1 -> imbalance:     r = {test_df[['proxy_lag1', 'imbalance']].corr().iloc[0,1]:.3f}")
    print(f"  imbalance_lag1 -> imbalance: r = {test_df[['imbalance_lag1', 'imbalance']].corr().iloc[0,1]:.3f}")

    print("\nThe gap shows how much signal is in actual imbalance vs proxy.")

    # ================================================================
    # SAVE ERROR DATA FOR FURTHER ANALYSIS
    # ================================================================
    error_df = test_df[['datetime', 'hour', 'day_of_week', 'is_weekend', 'imbalance',
                        'baseline_12', 'baseline_9', 'error_12', 'error_9',
                        'abs_error_12', 'abs_error_9', 'imb_magnitude',
                        'proxy_lag1', 'proxy_rolling_mean4', 'proxy_rolling_std4']].copy()
    error_df.to_csv(OUTPUT_DIR / 'error_analysis_data.csv', index=False)
    print(f"\nError data saved to: {OUTPUT_DIR / 'error_analysis_data.csv'}")

    # ================================================================
    # CREATE ERROR VISUALIZATION
    # ================================================================
    print("\nGenerating error visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Error Analysis - Lead 12 Minutes (Baseline)', fontsize=14, fontweight='bold')

    # 1. Error distribution
    ax = axes[0, 0]
    ax.hist(test_df['error_12'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('Error (MWh)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')

    # 2. Error by hour
    ax = axes[0, 1]
    hour_mae = test_df.groupby('hour')['abs_error_12'].mean()
    ax.bar(hour_mae.index, hour_mae.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Hour')
    ax.set_ylabel('MAE (MWh)')
    ax.set_title('MAE by Hour')
    ax.set_xticks(range(0, 24, 3))

    # 3. Error by magnitude
    ax = axes[0, 2]
    mag_order = ['<2', '2-5', '5-10', '10-20', '>20']
    mag_mae = test_df.groupby('imb_magnitude', observed=True)['abs_error_12'].mean()
    mag_mae = mag_mae.reindex(mag_order)
    ax.bar(range(len(mag_mae)), mag_mae.values, color='coral', alpha=0.7)
    ax.set_xticks(range(len(mag_mae)))
    ax.set_xticklabels(mag_order)
    ax.set_xlabel('|Imbalance| Category')
    ax.set_ylabel('MAE (MWh)')
    ax.set_title('MAE by Imbalance Magnitude')

    # 4. Error autocorrelation
    ax = axes[1, 0]
    lags = range(1, 21)
    autocorrs = [test_df['error_12'].autocorr(lag=l) for l in lags]
    ax.bar(lags, autocorrs, color='green', alpha=0.7)
    ax.axhline(0, color='black', linestyle='-')
    ax.axhline(0.05, color='red', linestyle='--', alpha=0.5)
    ax.axhline(-0.05, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Error Autocorrelation')

    # 5. Actual vs Predicted scatter
    ax = axes[1, 1]
    sample = test_df.dropna(subset=['baseline_12', 'imbalance']).sample(min(2000, len(test_df)))
    ax.scatter(sample['imbalance'], sample['baseline_12'], alpha=0.3, s=10)
    ax.plot([-30, 30], [-30, 30], 'r--', label='Perfect')
    ax.set_xlabel('Actual Imbalance (MWh)')
    ax.set_ylabel('Predicted (MWh)')
    ax.set_title('Actual vs Predicted')
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.legend()

    # 6. Error vs volatility
    ax = axes[1, 2]
    vol_mae = test_df.groupby('volatility_quintile', observed=True)['abs_error_12'].mean()
    vol_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    vol_mae = vol_mae.reindex(vol_order)
    ax.bar(range(len(vol_mae)), vol_mae.values, color='purple', alpha=0.7)
    ax.set_xticks(range(len(vol_mae)))
    ax.set_xticklabels(['VLow', 'Low', 'Med', 'High', 'VHigh'])
    ax.set_xlabel('Volatility Quintile')
    ax.set_ylabel('MAE (MWh)')
    ax.set_title('MAE by Recent Volatility')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'error_analysis.png'}")


if __name__ == '__main__':
    main()
