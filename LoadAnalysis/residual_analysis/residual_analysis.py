"""
Residual Analysis - Testing the user's hypotheses:

1. Peak hours have LOWER autocorrelation (more chaotic)?
2. MAPE by hour is more comparable than MAE?
3. If we learn seasonal pattern from 2024 and remove it from 2025,
   how well does the baseline forecast capture the residuals?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import acf
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

# Paths
BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet'
PLOT_PATH = Path(__file__).parent / 'plots'


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    return df


# =============================================================================
# 1. AUTOCORRELATION BY HOUR
# =============================================================================
def analyze_autocorrelation_by_hour(df: pd.DataFrame):
    """Check if peak hours have lower autocorrelation (more chaotic)."""
    print("\n" + "="*70)
    print("1. AUTOCORRELATION BY HOUR - Are peak hours more chaotic?")
    print("="*70)

    results = []

    for hour in range(1, 25):
        hour_data = df[df['hour'] == hour]['actual_load_mw'].dropna()

        if len(hour_data) >= 50:
            # Compute ACF at key lags
            acf_vals = acf(hour_data, nlags=7, fft=True)  # lag 1-7 days (same hour)

            # Also compute lag-1 on the full series for this hour
            lag1_corr = hour_data.autocorr(lag=1)  # Next day same hour

            results.append({
                'hour': hour,
                'acf_lag1': acf_vals[1],  # 1 day later same hour
                'acf_lag7': acf_vals[7],  # 1 week later same hour
                'mean_load': hour_data.mean(),
                'std_load': hour_data.std(),
                'cv': hour_data.std() / hour_data.mean() * 100  # Coefficient of variation
            })

    results_df = pd.DataFrame(results)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1a. ACF lag-1 by hour
    colors = ['red' if 9 <= h <= 14 else 'steelblue' for h in results_df['hour']]
    axes[0, 0].bar(results_df['hour'], results_df['acf_lag1'], color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('ACF (lag=1 day)')
    axes[0, 0].set_title('Day-to-Day Autocorrelation by Hour\n(Red = Peak hours 9-14)')
    axes[0, 0].set_xticks(range(1, 25))

    # 1b. ACF lag-7 by hour (weekly)
    axes[0, 1].bar(results_df['hour'], results_df['acf_lag7'], color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('ACF (lag=7 days)')
    axes[0, 1].set_title('Week-to-Week Autocorrelation by Hour\n(Red = Peak hours 9-14)')
    axes[0, 1].set_xticks(range(1, 25))

    # 1c. Coefficient of Variation by hour
    axes[1, 0].bar(results_df['hour'], results_df['cv'], color=colors, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('CV (%)')
    axes[1, 0].set_title('Coefficient of Variation by Hour\n(Higher = More Variable/Chaotic)')
    axes[1, 0].set_xticks(range(1, 25))

    # 1d. Scatter: Mean load vs ACF
    axes[1, 1].scatter(results_df['mean_load'], results_df['acf_lag1'], s=100, alpha=0.7)
    for i, row in results_df.iterrows():
        axes[1, 1].annotate(f"H{int(row['hour'])}", (row['mean_load'], row['acf_lag1']),
                           fontsize=8, ha='center')
    axes[1, 1].set_xlabel('Mean Load (MW)')
    axes[1, 1].set_ylabel('ACF (lag=1 day)')
    axes[1, 1].set_title('Mean Load vs Autocorrelation')

    # Correlation
    corr = results_df['mean_load'].corr(results_df['acf_lag1'])
    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '09_autocorrelation_by_hour.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print findings
    print("\nAutocorrelation by Hour (lag=1 day, same hour next day):")
    print(results_df[['hour', 'acf_lag1', 'acf_lag7', 'cv', 'mean_load']].to_string(index=False))

    # Statistics
    peak_hours = results_df[results_df['hour'].between(9, 14)]
    off_peak = results_df[~results_df['hour'].between(9, 14)]

    print(f"\n--- Peak Hours (9-14) vs Off-Peak ---")
    print(f"Peak ACF lag-1:     {peak_hours['acf_lag1'].mean():.3f}")
    print(f"Off-Peak ACF lag-1: {off_peak['acf_lag1'].mean():.3f}")
    print(f"Peak CV:            {peak_hours['cv'].mean():.2f}%")
    print(f"Off-Peak CV:        {off_peak['cv'].mean():.2f}%")

    # Statistical test
    t_stat, p_val = stats.ttest_ind(peak_hours['acf_lag1'], off_peak['acf_lag1'])
    print(f"\nT-test (ACF difference): t={t_stat:.2f}, p={p_val:.4f}")
    print(f"Conclusion: {'Peak hours ARE more chaotic (lower ACF)' if p_val < 0.05 and t_stat < 0 else 'No significant difference'}")

    print("\n  Saved: 09_autocorrelation_by_hour.png")

    return results_df


# =============================================================================
# 2. MAPE BY HOUR
# =============================================================================
def analyze_mape_by_hour(df: pd.DataFrame):
    """Compare MAE vs MAPE by hour."""
    print("\n" + "="*70)
    print("2. MAE vs MAPE BY HOUR - Fair comparison")
    print("="*70)

    df_valid = df.dropna(subset=['forecast_error_mw', 'actual_load_mw']).copy()
    df_valid['abs_error'] = df_valid['forecast_error_mw'].abs()
    df_valid['ape'] = (df_valid['abs_error'] / df_valid['actual_load_mw'] * 100)

    hourly = df_valid.groupby('hour').agg({
        'abs_error': 'mean',  # MAE
        'ape': 'mean',  # MAPE
        'actual_load_mw': 'mean'
    }).reset_index()
    hourly.columns = ['hour', 'MAE', 'MAPE', 'mean_load']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # MAE by hour
    colors = ['red' if 9 <= h <= 14 else 'steelblue' for h in hourly['hour']]
    axes[0].bar(hourly['hour'], hourly['MAE'], color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('MAE (MW)')
    axes[0].set_title('Mean Absolute Error by Hour')
    axes[0].set_xticks(range(1, 25))

    # MAPE by hour
    axes[1].bar(hourly['hour'], hourly['MAPE'], color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('MAPE (%)')
    axes[1].set_title('Mean Absolute Percentage Error by Hour')
    axes[1].set_xticks(range(1, 25))

    # Comparison
    ax2 = axes[2].twinx()
    axes[2].bar(hourly['hour'], hourly['MAE'], alpha=0.5, color='blue', label='MAE (MW)')
    ax2.plot(hourly['hour'], hourly['MAPE'], 'ro-', linewidth=2, markersize=6, label='MAPE (%)')
    axes[2].set_xlabel('Hour of Day')
    axes[2].set_ylabel('MAE (MW)', color='blue')
    ax2.set_ylabel('MAPE (%)', color='red')
    axes[2].set_title('MAE vs MAPE Comparison')
    axes[2].set_xticks(range(1, 25))
    axes[2].legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '10_mae_vs_mape_by_hour.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nMAE vs MAPE by Hour:")
    print(hourly.to_string(index=False))

    print(f"\n--- Key Observations ---")
    print(f"MAE Range:  {hourly['MAE'].min():.1f} - {hourly['MAE'].max():.1f} MW (ratio: {hourly['MAE'].max()/hourly['MAE'].min():.2f}x)")
    print(f"MAPE Range: {hourly['MAPE'].min():.2f} - {hourly['MAPE'].max():.2f}% (ratio: {hourly['MAPE'].max()/hourly['MAPE'].min():.2f}x)")

    # Correlation with load
    mae_load_corr = hourly['MAE'].corr(hourly['mean_load'])
    mape_load_corr = hourly['MAPE'].corr(hourly['mean_load'])
    print(f"\nCorrelation with mean load:")
    print(f"  MAE-Load:  {mae_load_corr:.3f} (MAE scales with load)")
    print(f"  MAPE-Load: {mape_load_corr:.3f} (MAPE should be load-independent)")

    print("\n  Saved: 10_mae_vs_mape_by_hour.png")

    return hourly


# =============================================================================
# 3. SEASONAL DECOMPOSITION: 2024 -> 2025 RESIDUALS
# =============================================================================
def analyze_residual_forecast_skill(df: pd.DataFrame):
    """
    Learn seasonal pattern from 2024, apply to 2025, and check how well
    the baseline forecast captures the residuals.
    """
    print("\n" + "="*70)
    print("3. RESIDUAL FORECAST SKILL")
    print("   Learn 2024 seasonality, remove from 2025, test forecast on residuals")
    print("="*70)

    # Split by year
    df_2024 = df[df['year'] == 2024].copy()
    df_2025 = df[df['year'] == 2025].copy()

    print(f"\n2024 data: {len(df_2024)} hours")
    print(f"2025 data: {len(df_2025)} hours")

    # Learn seasonal pattern from 2024: average by (day_of_week, hour)
    seasonal_2024 = df_2024.groupby(['day_of_week', 'hour']).agg({
        'actual_load_mw': 'mean',
        'forecast_load_mw': 'mean'
    }).reset_index()
    seasonal_2024.columns = ['day_of_week', 'hour', 'seasonal_actual', 'seasonal_forecast']

    print(f"\nSeasonal pattern: {len(seasonal_2024)} unique (day_of_week, hour) combinations")

    # Apply 2024 seasonal to 2025
    df_2025 = df_2025.reset_index().merge(seasonal_2024, on=['day_of_week', 'hour'], how='left').set_index('datetime')

    # Compute residuals
    df_2025['actual_residual'] = df_2025['actual_load_mw'] - df_2025['seasonal_actual']
    df_2025['forecast_residual'] = df_2025['forecast_load_mw'] - df_2025['seasonal_forecast']

    # Forecast errors on raw vs residual
    df_2025['raw_error'] = df_2025['actual_load_mw'] - df_2025['forecast_load_mw']
    df_2025['residual_error'] = df_2025['actual_residual'] - df_2025['forecast_residual']

    # Statistics
    df_valid = df_2025.dropna()

    raw_mae = df_valid['raw_error'].abs().mean()
    raw_rmse = np.sqrt((df_valid['raw_error']**2).mean())
    raw_mape = (df_valid['raw_error'].abs() / df_valid['actual_load_mw'] * 100).mean()

    resid_mae = df_valid['residual_error'].abs().mean()
    resid_rmse = np.sqrt((df_valid['residual_error']**2).mean())
    # MAPE on residuals doesn't make sense (residuals can be near zero)

    print(f"\n--- 2025 Forecast Performance ---")
    print(f"{'Metric':<15} {'Raw Data':<15} {'Residuals':<15} {'Change':<15}")
    print("-" * 60)
    print(f"{'MAE (MW)':<15} {raw_mae:<15.1f} {resid_mae:<15.1f} {(resid_mae/raw_mae-1)*100:+.1f}%")
    print(f"{'RMSE (MW)':<15} {raw_rmse:<15.1f} {resid_rmse:<15.1f} {(resid_rmse/raw_rmse-1)*100:+.1f}%")

    # What does this mean?
    print(f"\n--- Interpretation ---")
    if resid_mae < raw_mae * 0.9:
        print("[OK] Residual MAE is LOWER - Baseline adds value beyond seasonal!")
    elif resid_mae > raw_mae * 1.1:
        print("[X] Residual MAE is HIGHER - Baseline struggles with non-seasonal variation!")
    else:
        print("[~] Similar MAE - Baseline mainly captures the seasonal pattern")

    # Variance decomposition
    actual_var = df_valid['actual_load_mw'].var()
    seasonal_var = df_valid['seasonal_actual'].var()
    residual_var = df_valid['actual_residual'].var()

    print(f"\n--- Variance Decomposition (2025 Actual) ---")
    print(f"Total variance:    {actual_var:.0f} MW^2")
    print(f"Seasonal variance: {seasonal_var:.0f} MW^2 ({seasonal_var/actual_var*100:.1f}%)")
    print(f"Residual variance: {residual_var:.0f} MW^2 ({residual_var/actual_var*100:.1f}%)")

    # How much of residual does forecast explain?
    forecast_resid_var = df_valid['forecast_residual'].var()
    residual_error_var = df_valid['residual_error'].var()

    # R² for residuals
    ss_tot = ((df_valid['actual_residual'] - df_valid['actual_residual'].mean())**2).sum()
    ss_res = ((df_valid['actual_residual'] - df_valid['forecast_residual'])**2).sum()
    r2_residual = 1 - ss_res / ss_tot

    print(f"\n--- Forecast Skill on Residuals ---")
    print(f"R^2 on residuals: {r2_residual:.3f}")
    print(f"{'Good' if r2_residual > 0.3 else 'Poor'} - Baseline {'does' if r2_residual > 0.3 else 'does NOT'} capture residual patterns")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 3a. Seasonal pattern (2024)
    pivot = seasonal_2024.pivot(index='hour', columns='day_of_week', values='seasonal_actual')
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, day in enumerate(days):
        axes[0, 0].plot(pivot.index, pivot[i], label=day, linewidth=2)
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Load (MW)')
    axes[0, 0].set_title('Seasonal Pattern from 2024')
    axes[0, 0].legend(loc='upper right', fontsize=8)

    # 3b. 2025 Actual vs Seasonal
    sample = df_valid.iloc[:168*2]  # 2 weeks
    axes[0, 1].plot(sample.index, sample['actual_load_mw'], label='Actual 2025', alpha=0.8)
    axes[0, 1].plot(sample.index, sample['seasonal_actual'], label='2024 Seasonal', alpha=0.8)
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Load (MW)')
    axes[0, 1].set_title('2025 Actual vs 2024 Seasonal Pattern (2 weeks)')
    axes[0, 1].legend()

    # 3c. Residuals
    axes[0, 2].plot(sample.index, sample['actual_residual'], label='Actual Residual', alpha=0.8)
    axes[0, 2].plot(sample.index, sample['forecast_residual'], label='Forecast Residual', alpha=0.8)
    axes[0, 2].axhline(y=0, color='black', linestyle='--')
    axes[0, 2].set_xlabel('Date')
    axes[0, 2].set_ylabel('Residual (MW)')
    axes[0, 2].set_title('Residuals: Actual vs Forecast (2 weeks)')
    axes[0, 2].legend()

    # 3d. Error distribution comparison
    axes[1, 0].hist(df_valid['raw_error'], bins=50, alpha=0.5, label=f'Raw (MAE={raw_mae:.1f})', density=True)
    axes[1, 0].hist(df_valid['residual_error'], bins=50, alpha=0.5, label=f'Residual (MAE={resid_mae:.1f})', density=True)
    axes[1, 0].set_xlabel('Error (MW)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Error Distribution: Raw vs Residual')
    axes[1, 0].legend()

    # 3e. Residual error by hour
    resid_error_by_hour = df_valid.groupby('hour')['residual_error'].apply(lambda x: x.abs().mean())
    raw_error_by_hour = df_valid.groupby('hour')['raw_error'].apply(lambda x: x.abs().mean())

    x = np.arange(1, 25)
    width = 0.35
    axes[1, 1].bar(x - width/2, raw_error_by_hour.values, width, label='Raw MAE', alpha=0.7)
    axes[1, 1].bar(x + width/2, resid_error_by_hour.values, width, label='Residual MAE', alpha=0.7)
    axes[1, 1].set_xlabel('Hour')
    axes[1, 1].set_ylabel('MAE (MW)')
    axes[1, 1].set_title('MAE by Hour: Raw vs Residual')
    axes[1, 1].set_xticks(x)
    axes[1, 1].legend()

    # 3f. Scatter: Actual residual vs Forecast residual
    axes[1, 2].scatter(df_valid['forecast_residual'], df_valid['actual_residual'], alpha=0.1, s=1)
    axes[1, 2].plot([-500, 500], [-500, 500], 'r--', linewidth=2, label='Perfect')
    axes[1, 2].set_xlabel('Forecast Residual (MW)')
    axes[1, 2].set_ylabel('Actual Residual (MW)')
    axes[1, 2].set_title(f'Residual Correlation (R^2={r2_residual:.3f})')
    axes[1, 2].legend()
    axes[1, 2].set_xlim(-500, 500)
    axes[1, 2].set_ylim(-500, 500)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '11_residual_forecast_skill.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  Saved: 11_residual_forecast_skill.png")

    # Return data for further analysis
    return df_2025, seasonal_2024


# =============================================================================
# 4. ADDITIONAL: Skill by hour on residuals
# =============================================================================
def analyze_residual_skill_by_hour(df_2025: pd.DataFrame):
    """Check if forecast skill on residuals varies by hour."""
    print("\n" + "="*70)
    print("4. RESIDUAL FORECAST SKILL BY HOUR")
    print("="*70)

    df_valid = df_2025.dropna()

    results = []
    for hour in range(1, 25):
        hour_data = df_valid[df_valid['hour'] == hour]

        if len(hour_data) > 10:
            # R² for this hour
            ss_tot = ((hour_data['actual_residual'] - hour_data['actual_residual'].mean())**2).sum()
            ss_res = ((hour_data['actual_residual'] - hour_data['forecast_residual'])**2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # Correlation
            corr = hour_data['actual_residual'].corr(hour_data['forecast_residual'])

            results.append({
                'hour': hour,
                'r2_residual': r2,
                'correlation': corr,
                'raw_mae': hour_data['raw_error'].abs().mean(),
                'residual_mae': hour_data['residual_error'].abs().mean(),
                'mae_improvement': (hour_data['raw_error'].abs().mean() - hour_data['residual_error'].abs().mean())
            })

    results_df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # R² by hour
    colors = ['green' if r > 0.3 else 'red' for r in results_df['r2_residual']]
    axes[0].bar(results_df['hour'], results_df['r2_residual'], color=colors, alpha=0.7, edgecolor='black')
    axes[0].axhline(y=0.3, color='orange', linestyle='--', label='R^2=0.3 threshold')
    axes[0].set_xlabel('Hour')
    axes[0].set_ylabel('R² on Residuals')
    axes[0].set_title('Forecast Skill on Residuals by Hour\n(Green = Good, Red = Poor)')
    axes[0].set_xticks(range(1, 25))
    axes[0].legend()

    # Correlation by hour
    axes[1].bar(results_df['hour'], results_df['correlation'], color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Hour')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title('Actual-Forecast Residual Correlation by Hour')
    axes[1].set_xticks(range(1, 25))

    # MAE improvement by hour
    colors = ['green' if x > 0 else 'red' for x in results_df['mae_improvement']]
    axes[2].bar(results_df['hour'], results_df['mae_improvement'], color=colors, alpha=0.7, edgecolor='black')
    axes[2].axhline(y=0, color='black', linestyle='-')
    axes[2].set_xlabel('Hour')
    axes[2].set_ylabel('MAE Improvement (MW)')
    axes[2].set_title('Raw MAE - Residual MAE by Hour\n(Positive = Seasonal removal helped)')
    axes[2].set_xticks(range(1, 25))

    plt.tight_layout()
    plt.savefig(PLOT_PATH / '12_residual_skill_by_hour.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nResidual Forecast Skill by Hour:")
    print(results_df.to_string(index=False))

    # Summary
    poor_hours = results_df[results_df['r2_residual'] < 0.3]['hour'].tolist()
    good_hours = results_df[results_df['r2_residual'] >= 0.3]['hour'].tolist()

    print(f"\n--- Summary ---")
    print(f"Hours with GOOD residual skill (R^2>=0.3): {good_hours}")
    print(f"Hours with POOR residual skill (R^2<0.3): {poor_hours}")

    print("\n  Saved: 12_residual_skill_by_hour.png")

    return results_df


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70)
    print("RESIDUAL ANALYSIS - Testing Hypotheses")
    print("="*70)

    df = load_data()

    # 1. Autocorrelation by hour
    acf_results = analyze_autocorrelation_by_hour(df)

    # 2. MAE vs MAPE by hour
    error_results = analyze_mape_by_hour(df)

    # 3. Residual forecast skill
    df_2025, seasonal = analyze_residual_forecast_skill(df)

    # 4. Skill by hour
    skill_results = analyze_residual_skill_by_hour(df_2025)

    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
    1. AUTOCORRELATION BY HOUR:
       Check if peak hours (9-14) have lower day-to-day correlation.

    2. MAPE vs MAE:
       MAPE provides fairer comparison across hours.
       If MAPE is more uniform than MAE, peak hour errors are proportional.

    3. RESIDUAL FORECAST SKILL:
       - If R² on residuals is low: Baseline mainly captures obvious seasonality
       - If R² on residuals is high: Baseline adds real value
       - This tells us how much room there is for improvement!

    4. OPPORTUNITY:
       - Hours with low residual R² are where ML can add most value
       - Focus features/modeling on capturing residual patterns
    """)


if __name__ == '__main__':
    main()
