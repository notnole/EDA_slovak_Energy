"""
Load Signal Analysis for DA-Imbalance Spread Trading
Investigates whether DAMAS load prediction errors help predict the spread.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data"

# =============================================================================
# 1. LOAD AND MERGE DATA
# =============================================================================
print("=" * 80)
print("[*] LOADING DATA")
print("=" * 80)

# Load data
load_df = pd.read_csv(f"{BASE}/features/DamasLoad/load_data.csv", parse_dates=['datetime'])
print(f"[+] Load data: {len(load_df)} rows, {load_df['datetime'].min()} to {load_df['datetime'].max()}")

# Market data
market_df = pd.read_csv(f"{BASE}/MarketPriceGap/data/processed/hourly_market_prices.csv")
# Check the datetime column name
if 'timestamp_hour' in market_df.columns:
    market_df.rename(columns={'timestamp_hour': 'datetime'}, inplace=True)
market_df['datetime'] = pd.to_datetime(market_df['datetime'])
print(f"[+] Market data: {len(market_df)} rows, {market_df['datetime'].min()} to {market_df['datetime'].max()}")

# Temperature data - custom parser
print("[*] Parsing temperature data...")
temp_rows = []
with open(f"{BASE}/RawData/ActualTemp2024-2026.csv", 'r', encoding='utf-8-sig') as f:
    header = f.readline()  # skip header
    for line in f:
        line = line.strip()
        if not line or '(invalid)' in line:
            continue
        parts = line.split(',')
        if len(parts) < 4:
            continue
        try:
            dt_str = parts[0]
            int_part = parts[1].strip() if len(parts) > 1 else ''
            # The format is: datetime,000,integer_part,decimal_part,
            # Actually looking at the data: 01/02/2024 00:00:00,000,-0,29,
            # parts[0] = datetime, parts[1] = '000', parts[2] = integer_part, parts[3] = decimal_part
            int_part = parts[2].strip()
            dec_part = parts[3].strip()
            temp = float(int_part + '.' + dec_part)
            dt = pd.to_datetime(dt_str, format='%d/%m/%Y %H:%M:%S')
            temp_rows.append({'datetime': dt, 'temperature': temp})
        except (ValueError, IndexError):
            continue

temp_df = pd.DataFrame(temp_rows)
print(f"[+] Temperature data: {len(temp_df)} rows, {temp_df['datetime'].min()} to {temp_df['datetime'].max()}")

# Merge all on datetime
df = load_df[['datetime', 'actual_load_mw', 'forecast_load_mw', 'forecast_error_mw',
              'day_of_week', 'is_weekend']].copy()
df = df.merge(market_df[['datetime', 'da_price', 'imb_settlement_price', 'spread_da_imb', 'imbalance_mwh']],
              on='datetime', how='inner')
df = df.merge(temp_df, on='datetime', how='left')

df['date'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# Season
def get_season(month):
    if month in [12, 1, 2]:
        return 'DJF'
    elif month in [3, 4, 5]:
        return 'MAM'
    elif month in [6, 7, 8]:
        return 'JJA'
    else:
        return 'SON'

df['season'] = df['month'].apply(get_season)

# Drop rows with missing spread
df = df.dropna(subset=['spread_da_imb', 'forecast_error_mw'])
print(f"[+] Merged data: {len(df)} rows, {df['datetime'].min()} to {df['datetime'].max()}")
print(f"[+] Temperature coverage: {df['temperature'].notna().sum()} / {len(df)} rows")

# =============================================================================
# QUESTION 1: Does the load prediction residual indicate anything?
# =============================================================================
print("\n" + "=" * 80)
print("[*] QUESTION 1: Load Prediction Residual vs Spread")
print("=" * 80)

# Daily aggregation
daily = df.groupby('date').agg(
    mean_load_error=('forecast_error_mw', 'mean'),
    mean_spread=('spread_da_imb', 'mean'),
    mean_temp=('temperature', 'mean'),
    n_hours=('spread_da_imb', 'count')
).reset_index()
daily = daily[daily['n_hours'] >= 20]  # require mostly complete days
daily['date'] = pd.to_datetime(daily['date'])
daily['year'] = daily['date'].dt.year

print(f"\n[*] Daily data: {len(daily)} days")
print(f"[*] Mean daily load error: {daily['mean_load_error'].mean():.1f} MW (std={daily['mean_load_error'].std():.1f})")
print(f"[*] Mean daily spread: {daily['mean_spread'].mean():.2f} EUR/MWh (std={daily['mean_spread'].std():.2f})")

# 1c) Daily correlation
r_daily, p_daily = stats.pearsonr(daily['mean_load_error'], daily['mean_spread'])
rho_daily, p_rho_daily = stats.spearmanr(daily['mean_load_error'], daily['mean_spread'])
print(f"\n--- Daily Correlation ---")
print(f"[*] Pearson r  = {r_daily:.4f}, p = {p_daily:.2e}")
print(f"[*] Spearman rho = {rho_daily:.4f}, p = {p_rho_daily:.2e}")

# 1d) Quintile analysis
daily['load_error_quintile'] = pd.qcut(daily['mean_load_error'], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'])
quintile_stats = daily.groupby('load_error_quintile', observed=True).agg(
    mean_spread=('mean_spread', 'mean'),
    std_spread=('mean_spread', 'std'),
    n=('mean_spread', 'count'),
    mean_load_error=('mean_load_error', 'mean')
).reset_index()
quintile_stats['se'] = quintile_stats['std_spread'] / np.sqrt(quintile_stats['n'])
quintile_stats['ci95'] = 1.96 * quintile_stats['se']

print(f"\n--- Spread by Load Error Quintile ---")
print(quintile_stats[['load_error_quintile', 'mean_load_error', 'mean_spread', 'ci95', 'n']].to_string(index=False))

# Q1-Q5 difference
q1_spread = daily[daily['load_error_quintile'] == 'Q1(low)']['mean_spread']
q5_spread = daily[daily['load_error_quintile'] == 'Q5(high)']['mean_spread']
t_q, p_q = stats.ttest_ind(q1_spread, q5_spread)
print(f"\n[*] Q5-Q1 spread difference: {q5_spread.mean() - q1_spread.mean():.2f} EUR/MWh")
print(f"[*] Two-sample t-test: t={t_q:.2f}, p={p_q:.4f}")

# 1e) Hourly correlation
r_hourly, p_hourly = stats.pearsonr(df['forecast_error_mw'], df['spread_da_imb'])
print(f"\n--- Hourly Correlation ---")
print(f"[*] Pearson r  = {r_hourly:.4f}, p = {p_hourly:.2e}")

# Hourly correlation by hour of day
print(f"\n--- Correlation by Hour of Day ---")
hour_corrs = []
for h in range(24):
    hdf = df[df['hour'] == h]
    if len(hdf) > 30:
        r_h, p_h = stats.pearsonr(hdf['forecast_error_mw'], hdf['spread_da_imb'])
        hour_corrs.append({'hour': h, 'r': r_h, 'p': p_h, 'n': len(hdf)})
hour_corr_df = pd.DataFrame(hour_corrs)
print(hour_corr_df.to_string(index=False))

# =============================================================================
# QUESTION 2: Deviation from similar day
# =============================================================================
print("\n" + "=" * 80)
print("[*] QUESTION 2: Deviation from Similar Day")
print("=" * 80)

# Compute similar day baseline
df['dow'] = df['datetime'].dt.dayofweek
similar_day_baseline = df.groupby(['dow', 'season', 'hour']).agg(
    baseline_actual_load=('actual_load_mw', 'mean'),
    baseline_forecast_load=('forecast_load_mw', 'mean')
).reset_index()

df = df.merge(similar_day_baseline, on=['dow', 'season', 'hour'], how='left')
df['actual_deviation'] = df['actual_load_mw'] - df['baseline_actual_load']
df['forecast_deviation'] = df['forecast_load_mw'] - df['baseline_forecast_load']  # knowable D-1!

# Daily deviations
daily2 = df.groupby('date').agg(
    mean_actual_dev=('actual_deviation', 'mean'),
    mean_forecast_dev=('forecast_deviation', 'mean'),
    mean_spread=('spread_da_imb', 'mean'),
    mean_load_error=('forecast_error_mw', 'mean'),
    mean_temp=('temperature', 'mean'),
    n_hours=('spread_da_imb', 'count')
).reset_index()
daily2 = daily2[daily2['n_hours'] >= 20]
daily2['date'] = pd.to_datetime(daily2['date'])
daily2['year'] = daily2['date'].dt.year

# Correlation: actual deviation vs spread
r_actdev, p_actdev = stats.pearsonr(daily2['mean_actual_dev'].dropna(),
                                      daily2.loc[daily2['mean_actual_dev'].notna(), 'mean_spread'])
print(f"\n--- Actual Load Deviation from Similar Day vs Spread ---")
print(f"[*] Pearson r = {r_actdev:.4f}, p = {p_actdev:.2e}")

# Correlation: forecast deviation vs spread (knowable D-1!)
r_fdev, p_fdev = stats.pearsonr(daily2['mean_forecast_dev'].dropna(),
                                  daily2.loc[daily2['mean_forecast_dev'].notna(), 'mean_spread'])
print(f"\n--- Forecast Deviation from Similar Day vs Spread (D-1 knowable!) ---")
print(f"[*] Pearson r = {r_fdev:.4f}, p = {p_fdev:.2e}")

# Quintile analysis for actual deviation
daily2['actual_dev_quintile'] = pd.qcut(daily2['mean_actual_dev'], 5,
                                         labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'])
q_dev_stats = daily2.groupby('actual_dev_quintile', observed=True).agg(
    mean_spread=('mean_spread', 'mean'),
    std_spread=('mean_spread', 'std'),
    n=('mean_spread', 'count'),
    mean_dev=('mean_actual_dev', 'mean')
).reset_index()
q_dev_stats['ci95'] = 1.96 * q_dev_stats['std_spread'] / np.sqrt(q_dev_stats['n'])

print(f"\n--- Spread by Actual Load Deviation Quintile ---")
print(q_dev_stats[['actual_dev_quintile', 'mean_dev', 'mean_spread', 'ci95', 'n']].to_string(index=False))

# Quintile analysis for forecast deviation (D-1 knowable)
daily2['forecast_dev_quintile'] = pd.qcut(daily2['mean_forecast_dev'], 5,
                                            labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'])
q_fdev_stats = daily2.groupby('forecast_dev_quintile', observed=True).agg(
    mean_spread=('mean_spread', 'mean'),
    std_spread=('mean_spread', 'std'),
    n=('mean_spread', 'count'),
    mean_dev=('mean_forecast_dev', 'mean')
).reset_index()
q_fdev_stats['ci95'] = 1.96 * q_fdev_stats['std_spread'] / np.sqrt(q_fdev_stats['n'])

print(f"\n--- Spread by Forecast Deviation Quintile (D-1 knowable) ---")
print(q_fdev_stats[['forecast_dev_quintile', 'mean_dev', 'mean_spread', 'ci95', 'n']].to_string(index=False))

# =============================================================================
# QUESTION 3: Combined temperature + load signal strategy
# =============================================================================
print("\n" + "=" * 80)
print("[*] QUESTION 3: Combined Temperature + Load Strategy")
print("=" * 80)

# 3a) Daily regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

daily_model = daily2.dropna(subset=['mean_temp', 'mean_load_error', 'mean_actual_dev']).copy()
print(f"\n[*] Days with all features: {len(daily_model)}")

X_cols = ['mean_temp', 'mean_load_error', 'mean_actual_dev']
X = daily_model[X_cols].values
y = daily_model['mean_spread'].values

reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
r2 = r2_score(y, y_pred)
print(f"\n--- Linear Model: spread ~ temp + load_error + load_deviation ---")
print(f"[*] R-squared = {r2:.4f}")
for name, coef in zip(X_cols, reg.coef_):
    print(f"    {name}: {coef:.4f}")
print(f"    intercept: {reg.intercept_:.4f}")

# 3b/c) Strategy backtests
# Baseline: always sell DA, buy imbalance (= collect spread_da_imb)
# Temperature filter: reverse when temp < 0 or > 25
# Combined: temperature filter + load signal scaling

daily_strat = daily2.dropna(subset=['mean_temp']).copy()
daily_strat = daily_strat.sort_values('date').reset_index(drop=True)

# Baseline: always long spread (sell DA, buy imbalance)
daily_strat['pnl_baseline'] = daily_strat['mean_spread']

# Temperature filter
daily_strat['temp_signal'] = 1.0  # default: sell DA buy imb
daily_strat.loc[daily_strat['mean_temp'] < 0, 'temp_signal'] = -1.0
daily_strat.loc[daily_strat['mean_temp'] > 25, 'temp_signal'] = -1.0
daily_strat['pnl_temp'] = daily_strat['temp_signal'] * daily_strat['mean_spread']

# Combined: temp filter + load error scaling within normal temp range
# Use forecast deviation (D-1 knowable) as the load signal
daily_strat['combined_signal'] = daily_strat['temp_signal'].copy()
normal_temp = (daily_strat['mean_temp'] >= 0) & (daily_strat['mean_temp'] <= 25)

# Within normal temp, scale by forecast deviation signal
# Hypothesis: high forecast deviation -> higher actual load -> deficit -> negative spread
# So when forecast_dev is high, we might want to flip to buy DA
fdev_median = daily_strat.loc[normal_temp, 'mean_forecast_dev'].median()
fdev_std = daily_strat.loc[normal_temp, 'mean_forecast_dev'].std()

# Scale: reduce position when forecast deviation is extreme
# If forecast_dev > median + 1*std, reduce to 0.5
# If forecast_dev < median - 1*std, increase to 1.5
scaling = np.ones(len(daily_strat))
high_dev = normal_temp & (daily_strat['mean_forecast_dev'] > fdev_median + fdev_std)
low_dev = normal_temp & (daily_strat['mean_forecast_dev'] < fdev_median - fdev_std)
scaling[high_dev] = 0.5
scaling[low_dev] = 1.5
daily_strat['combined_signal'] = daily_strat['combined_signal'] * scaling
daily_strat['pnl_combined'] = daily_strat['combined_signal'] * daily_strat['mean_spread']

# Also test: use forecast deviation to FLIP the signal (more aggressive)
daily_strat['combined_flip_signal'] = daily_strat['temp_signal'].copy()
# When forecast_dev > +1 std, flip from sell DA to buy DA (expect deficit/high settlement)
daily_strat.loc[high_dev, 'combined_flip_signal'] = -1.0 * daily_strat.loc[high_dev, 'temp_signal']
daily_strat['pnl_combined_flip'] = daily_strat['combined_flip_signal'] * daily_strat['mean_spread']

# Strategy performance
for name, col in [('Baseline', 'pnl_baseline'), ('Temp Only', 'pnl_temp'),
                   ('Temp+Load Scale', 'pnl_combined'), ('Temp+Load Flip', 'pnl_combined_flip')]:
    total = daily_strat[col].sum()
    mean = daily_strat[col].mean()
    std = daily_strat[col].std()
    sharpe = mean / std * np.sqrt(252) if std > 0 else 0
    win_rate = (daily_strat[col] > 0).mean()
    print(f"\n--- {name} ---")
    print(f"  Total P&L:  {total:.1f} EUR/MWh")
    print(f"  Daily mean: {mean:.2f} EUR/MWh")
    print(f"  Daily std:  {std:.2f} EUR/MWh")
    print(f"  Sharpe:     {sharpe:.2f}")
    print(f"  Win rate:   {win_rate:.1%}")

# Improvement test: is combined significantly better than temp-only?
diff_pnl = daily_strat['pnl_combined'].values - daily_strat['pnl_temp'].values
t_imp, p_imp = stats.ttest_1samp(diff_pnl, 0)
print(f"\n--- Improvement Test (Combined Scale vs Temp Only) ---")
print(f"[*] Mean daily improvement: {diff_pnl.mean():.4f} EUR/MWh")
print(f"[*] t-stat = {t_imp:.3f}, p = {p_imp:.4f}")

diff_pnl_flip = daily_strat['pnl_combined_flip'].values - daily_strat['pnl_temp'].values
t_imp2, p_imp2 = stats.ttest_1samp(diff_pnl_flip, 0)
print(f"\n--- Improvement Test (Combined Flip vs Temp Only) ---")
print(f"[*] Mean daily improvement: {diff_pnl_flip.mean():.4f} EUR/MWh")
print(f"[*] t-stat = {t_imp2:.3f}, p = {p_imp2:.4f}")

# =============================================================================
# QUESTION 4: Statistical Significance
# =============================================================================
print("\n" + "=" * 80)
print("[*] QUESTION 4: Statistical Significance Tests")
print("=" * 80)

N_PERM = 10000
N_BOOT = 10000
np.random.seed(42)

# Define all signals to test
signals = {
    'daily_load_error_vs_spread': {
        'x': daily['mean_load_error'].values,
        'y': daily['mean_spread'].values,
        'observed_r': r_daily
    },
    'hourly_load_error_vs_spread': {
        'x': df['forecast_error_mw'].values,
        'y': df['spread_da_imb'].values,
        'observed_r': r_hourly
    },
    'actual_deviation_vs_spread': {
        'x': daily2['mean_actual_dev'].dropna().values,
        'y': daily2.loc[daily2['mean_actual_dev'].notna(), 'mean_spread'].values,
        'observed_r': r_actdev
    },
    'forecast_deviation_vs_spread': {
        'x': daily2['mean_forecast_dev'].dropna().values,
        'y': daily2.loc[daily2['mean_forecast_dev'].notna(), 'mean_spread'].values,
        'observed_r': r_fdev
    }
}

n_tests = len(signals)
bonferroni_alpha = 0.01 / n_tests
print(f"\n[*] Number of tests: {n_tests}")
print(f"[*] Bonferroni-corrected alpha: {bonferroni_alpha:.4f}")

perm_results = {}
boot_results = {}

for name, sig in signals.items():
    x, y = sig['x'], sig['y']
    obs_r = sig['observed_r']
    n = len(x)

    # 4a) Permutation test
    perm_rs = np.zeros(N_PERM)
    for i in range(N_PERM):
        y_perm = np.random.permutation(y)
        perm_rs[i] = np.corrcoef(x, y_perm)[0, 1]

    p_perm = np.mean(np.abs(perm_rs) >= np.abs(obs_r))
    perm_results[name] = {'observed_r': obs_r, 'p_perm': p_perm, 'null_dist': perm_rs}

    # 4b) Bootstrap CI
    boot_rs = np.zeros(N_BOOT)
    for i in range(N_BOOT):
        idx = np.random.randint(0, n, n)
        boot_rs[i] = np.corrcoef(x[idx], y[idx])[0, 1]

    ci_low, ci_high = np.percentile(boot_rs, [2.5, 97.5])
    boot_results[name] = {'ci_low': ci_low, 'ci_high': ci_high, 'boot_dist': boot_rs}

    sig_bonf = "YES" if p_perm < bonferroni_alpha else "NO"

    print(f"\n--- {name} ---")
    print(f"  Observed r:     {obs_r:.4f}")
    print(f"  Permutation p:  {p_perm:.4f}")
    print(f"  Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Significant after Bonferroni (alpha={bonferroni_alpha:.4f}): {sig_bonf}")

# 4c) Out-of-sample test
print(f"\n--- Out-of-Sample: Train 2024, Test 2025+ ---")

for name, data_src, x_col, y_col in [
    ('daily_load_error', daily, 'mean_load_error', 'mean_spread'),
    ('actual_deviation', daily2, 'mean_actual_dev', 'mean_spread'),
    ('forecast_deviation', daily2, 'mean_forecast_dev', 'mean_spread')
]:
    d = data_src.dropna(subset=[x_col, y_col]).copy()
    train = d[d['year'] == 2024]
    test = d[d['year'] >= 2025]

    if len(train) > 10 and len(test) > 10:
        r_train, p_train = stats.pearsonr(train[x_col], train[y_col])
        r_test, p_test = stats.pearsonr(test[x_col], test[y_col])
        print(f"\n  {name}:")
        print(f"    Train (2024): r={r_train:.4f}, p={p_train:.4f}, n={len(train)}")
        print(f"    Test (2025+): r={r_test:.4f}, p={p_test:.4f}, n={len(test)}")

        # Also test strategy OOS
        if name == 'forecast_deviation':
            # Simple strategy: sell DA when forecast_dev < median, buy DA when > median
            med_train = train[x_col].median()
            test_signal = np.where(test[x_col] < med_train, 1.0, -1.0)
            test_pnl_signal = test_signal * test[y_col].values
            test_pnl_baseline = test[y_col].values

            print(f"    OOS Strategy (signal): total={test_pnl_signal.sum():.1f}, mean={test_pnl_signal.mean():.2f}")
            print(f"    OOS Baseline:          total={test_pnl_baseline.sum():.1f}, mean={test_pnl_baseline.mean():.2f}")
    else:
        print(f"\n  {name}: Insufficient data (train={len(train)}, test={len(test)})")

# 4d) Strategy OOS test
print(f"\n--- Out-of-Sample Strategy Test ---")
daily_strat['year'] = pd.to_datetime(daily_strat['date']).dt.year
train_strat = daily_strat[daily_strat['year'] == 2024]
test_strat = daily_strat[daily_strat['year'] >= 2025]

oos_results = {}
for name, col in [('Baseline', 'pnl_baseline'), ('Temp Only', 'pnl_temp'),
                   ('Temp+Load Scale', 'pnl_combined'), ('Temp+Load Flip', 'pnl_combined_flip')]:
    train_total = train_strat[col].sum()
    test_total = test_strat[col].sum()
    train_mean = train_strat[col].mean()
    test_mean = test_strat[col].mean()
    train_sharpe = train_mean / train_strat[col].std() * np.sqrt(252) if train_strat[col].std() > 0 else 0
    test_sharpe = test_mean / test_strat[col].std() * np.sqrt(252) if test_strat[col].std() > 0 else 0

    oos_results[name] = {
        'train_total': train_total, 'test_total': test_total,
        'train_sharpe': train_sharpe, 'test_sharpe': test_sharpe,
        'train_n': len(train_strat), 'test_n': len(test_strat)
    }

    print(f"\n  {name}:")
    print(f"    Train 2024: total={train_total:.1f}, sharpe={train_sharpe:.2f}, n={len(train_strat)}")
    print(f"    Test 2025+: total={test_total:.1f}, sharpe={test_sharpe:.2f}, n={len(test_strat)}")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n" + "=" * 80)
print("[*] GENERATING PLOTS")
print("=" * 80)

fig, axes = plt.subplots(3, 2, figsize=(16, 16))
fig.suptitle('Load Signal Analysis for DA-Imbalance Spread', fontsize=14, fontweight='bold')

# Panel 1: Scatter of daily load forecast error vs daily spread
ax = axes[0, 0]
ax.scatter(daily['mean_load_error'], daily['mean_spread'], alpha=0.3, s=15, c='steelblue')
# Regression line
z = np.polyfit(daily['mean_load_error'], daily['mean_spread'], 1)
p = np.poly1d(z)
x_range = np.linspace(daily['mean_load_error'].min(), daily['mean_load_error'].max(), 100)
ax.plot(x_range, p(x_range), 'r-', linewidth=2)
ax.set_xlabel('Daily Mean Load Forecast Error (MW)')
ax.set_ylabel('Daily Mean Spread DA-Imb (EUR/MWh)')
ax.set_title(f'Q1: Load Error vs Spread\nr={r_daily:.3f}, p={p_daily:.2e}')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

# Panel 2: Bar chart of mean spread by load error quintile
ax = axes[0, 1]
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 5))
bars = ax.bar(range(5), quintile_stats['mean_spread'],
              yerr=quintile_stats['ci95'], capsize=5, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(5))
ax.set_xticklabels([f'{q}\n({le:.0f} MW)' for q, le in
                     zip(quintile_stats['load_error_quintile'], quintile_stats['mean_load_error'])],
                    fontsize=8)
ax.set_xlabel('Load Error Quintile (mean error)')
ax.set_ylabel('Mean Spread DA-Imb (EUR/MWh)')
ax.set_title(f'Q1: Spread by Load Error Quintile\nQ5-Q1 = {q5_spread.mean()-q1_spread.mean():.2f}, p={p_q:.4f}')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: Similar-day deviation vs spread (both actual and forecast)
ax = axes[1, 0]
ax.scatter(daily2['mean_actual_dev'], daily2['mean_spread'], alpha=0.3, s=15, c='steelblue', label=f'Actual dev (r={r_actdev:.3f})')
ax.scatter(daily2['mean_forecast_dev'], daily2['mean_spread'], alpha=0.3, s=15, c='darkorange', label=f'Forecast dev (r={r_fdev:.3f})')
# Regression lines
mask_a = daily2['mean_actual_dev'].notna()
z_a = np.polyfit(daily2.loc[mask_a, 'mean_actual_dev'], daily2.loc[mask_a, 'mean_spread'], 1)
x_a = np.linspace(daily2['mean_actual_dev'].min(), daily2['mean_actual_dev'].max(), 100)
ax.plot(x_a, np.poly1d(z_a)(x_a), 'steelblue', linewidth=2)
mask_f = daily2['mean_forecast_dev'].notna()
z_f = np.polyfit(daily2.loc[mask_f, 'mean_forecast_dev'], daily2.loc[mask_f, 'mean_spread'], 1)
x_f = np.linspace(daily2['mean_forecast_dev'].min(), daily2['mean_forecast_dev'].max(), 100)
ax.plot(x_f, np.poly1d(z_f)(x_f), 'darkorange', linewidth=2)
ax.set_xlabel('Deviation from Similar Day Mean (MW)')
ax.set_ylabel('Daily Mean Spread DA-Imb (EUR/MWh)')
ax.set_title('Q2: Similar-Day Deviation vs Spread')
ax.legend(fontsize=8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

# Panel 4: Cumulative P&L comparison
ax = axes[1, 1]
cum_baseline = daily_strat['pnl_baseline'].cumsum()
cum_temp = daily_strat['pnl_temp'].cumsum()
cum_combined = daily_strat['pnl_combined'].cumsum()
cum_flip = daily_strat['pnl_combined_flip'].cumsum()

dates = pd.to_datetime(daily_strat['date'])
ax.plot(dates, cum_baseline, label=f'Baseline ({cum_baseline.iloc[-1]:.0f})', alpha=0.7, linewidth=1)
ax.plot(dates, cum_temp, label=f'Temp Only ({cum_temp.iloc[-1]:.0f})', alpha=0.7, linewidth=1.5)
ax.plot(dates, cum_combined, label=f'Temp+Load Scale ({cum_combined.iloc[-1]:.0f})', alpha=0.7, linewidth=1.5)
ax.plot(dates, cum_flip, label=f'Temp+Load Flip ({cum_flip.iloc[-1]:.0f})', alpha=0.7, linewidth=1.5, linestyle='--')

# Add vertical line for OOS split
oos_date = pd.Timestamp('2025-01-01')
ax.axvline(x=oos_date, color='red', linestyle=':', alpha=0.7, label='Train/Test split')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative P&L (EUR/MWh)')
ax.set_title('Q3: Cumulative Strategy P&L')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)

# Panel 5: Permutation test histogram
ax = axes[2, 0]
# Use daily load error correlation permutation
null_dist = perm_results['daily_load_error_vs_spread']['null_dist']
obs = perm_results['daily_load_error_vs_spread']['observed_r']
p_val = perm_results['daily_load_error_vs_spread']['p_perm']

ax.hist(null_dist, bins=80, color='lightgray', edgecolor='gray', density=True, label='Null distribution')
ax.axvline(x=obs, color='red', linewidth=2, label=f'Observed r={obs:.4f}')
ax.axvline(x=-obs, color='red', linewidth=2, linestyle='--', alpha=0.5)
ax.set_xlabel('Correlation (r)')
ax.set_ylabel('Density')
ax.set_title(f'Q4: Permutation Test - Daily Load Error\np-value = {p_val:.4f} (n={N_PERM} permutations)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 6: OOS performance comparison
ax = axes[2, 1]
strat_names = list(oos_results.keys())
train_sharpes = [oos_results[s]['train_sharpe'] for s in strat_names]
test_sharpes = [oos_results[s]['test_sharpe'] for s in strat_names]

x_pos = np.arange(len(strat_names))
width = 0.35
bars1 = ax.bar(x_pos - width/2, train_sharpes, width, label='Train (2024)', color='steelblue', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, test_sharpes, width, label='Test (2025+)', color='darkorange', alpha=0.7)

ax.set_xlabel('Strategy')
ax.set_ylabel('Annualized Sharpe Ratio')
ax.set_title('Q4: Out-of-Sample Performance')
ax.set_xticks(x_pos)
ax.set_xticklabels(strat_names, fontsize=8, rotation=15)
ax.legend()
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.2f}', ha='center', va='bottom', fontsize=7)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.2f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = f"{BASE}/MarketPriceGap/strategies/load_signal/load_signal_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n[+] Saved figure to: {out_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("[*] FINAL SUMMARY")
print("=" * 80)

print("""
QUESTION 1: Load Prediction Residual
  - Daily load forecast error has a correlation of r={r_d:.4f} with daily spread
  - Hourly correlation is r={r_h:.4f}
  - Q5-Q1 quintile spread difference: {q5q1:.2f} EUR/MWh

QUESTION 2: Similar-Day Deviation
  - Actual load deviation: r={r_ad:.4f}
  - Forecast deviation (D-1 knowable): r={r_fd:.4f}

QUESTION 3: Combined Strategy
  - Temperature-only vs combined strategies show marginal improvement
  - Strategy improvement p-value: {p_imp:.4f}

QUESTION 4: Statistical Significance
  - After Bonferroni correction (alpha={bonf:.4f}):
""".format(
    r_d=r_daily, r_h=r_hourly,
    q5q1=q5_spread.mean()-q1_spread.mean(),
    r_ad=r_actdev, r_fd=r_fdev,
    p_imp=p_imp, bonf=bonferroni_alpha
))

for name in signals:
    pr = perm_results[name]
    br = boot_results[name]
    sig_str = "SIGNIFICANT" if pr['p_perm'] < bonferroni_alpha else "NOT significant"
    print(f"  {name}: r={pr['observed_r']:.4f}, p_perm={pr['p_perm']:.4f}, "
          f"CI=[{br['ci_low']:.4f}, {br['ci_high']:.4f}] -> {sig_str}")

print("\n[+] Analysis complete.")
