"""
Market Regime Analysis: Month-by-Month Decomposition
=====================================================

Analyzes whether spread trading performance decline is seasonal
or reflects fundamental market structure change.

Metrics analyzed per month (Jan 2024 - Apr 2026):
1. Settlement price behavior (mean, std, tails, asymmetry)
2. System imbalance behavior (distribution, autocorrelation, diurnal)
3. Spread opportunity (settle - IDM mid)
4. IDM market liquidity (bid-ask, volume)
5. Feature-target correlations over time
6. Seasonal decomposition (2026 anomaly vs seasonal expected)
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[1]  # ImbalanceForcastingProd/
REPO_ROOT = BASE_DIR.parent
PLOT_DIR = BASE_DIR / "plots" / "eda" / "market_regime"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (16, 10), "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})

# --- Data Loading ---
sys.path.insert(0, str(SCRIPT_DIR.parent / "training"))
import train_multi_lead as tml
from train_multi_lead import load_all_data, build_features

LEAD = 8
data = load_all_data()
tml.TRAIN_END = '2026-04-15'
tml.TEST_START = '2026-04-15'
df_base, feature_cols = build_features(data, LEAD)

# Join orderbook execution data
DATA_DIR = BASE_DIR / "data"
ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                       parse_dates=['delivery_start'])
ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[
    ['bid', 'ask', 'spread', 'mid']]
ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']
df_base = df_base.join(ob_120, how='left')

df_base['imb_settlement_price'] = df_base['imb_settle_price']
df_base['imbalance_mwh'] = df_base['target']  # target == imbalance_mwh
df_base['spread_raw'] = df_base['imb_settlement_price'] - df_base['exec_mid']

# Map feature names back to readable names where needed
# idm_vwap_lag, idm_volume_lag are the lagged versions available in features
# For raw IDM data we use exec_* from orderbook

print(f"\n[*] Dataset shape: {df_base.shape}")
print(f"[*] Date range: {df_base.index.min()} to {df_base.index.max()}")

# Create month column for grouping
df = df_base.copy()
df['month'] = df.index.to_period('M')
df['year'] = df.index.year
df['month_num'] = df.index.month
df['hour'] = df.index.hour


# ============================================================
# SECTION 1: Settlement Price Behavior
# ============================================================
print("\n" + "="*80)
print("SECTION 1: SETTLEMENT PRICE BEHAVIOR")
print("="*80)

settle_stats = []
for period, grp in df.groupby('month'):
    sp = grp['imb_settlement_price'].dropna()
    if len(sp) < 50:
        continue
    abs_sp = sp.abs()
    pos = sp[sp > 0]
    neg = sp[sp < 0]
    settle_stats.append({
        'month': period,
        'n': len(sp),
        'mean': sp.mean(),
        'std': sp.std(),
        'abs_mean': abs_sp.mean(),
        'abs_std': abs_sp.std(),
        'frac_gt50': (abs_sp > 50).mean(),
        'frac_gt100': (abs_sp > 100).mean(),
        'frac_gt200': (abs_sp > 200).mean(),
        'mean_positive': pos.mean() if len(pos) > 0 else np.nan,
        'mean_negative': neg.mean() if len(neg) > 0 else np.nan,
        'pos_neg_ratio': (pos.mean() / abs(neg.mean())) if len(pos) > 0 and len(neg) > 0 else np.nan,
    })

settle_df = pd.DataFrame(settle_stats)
settle_df['month_str'] = settle_df['month'].astype(str)

print("\nMonthly Settlement Price Statistics:")
print("-"*120)
print(f"{'Month':<10} {'N':>5} {'Mean':>8} {'Std':>8} {'|Mean|':>8} {'|Std|':>8} "
      f"{'%>50':>7} {'%>100':>7} {'%>200':>7} {'Pos_Avg':>8} {'Neg_Avg':>9} {'P/N':>5}")
print("-"*120)
for _, r in settle_df.iterrows():
    print(f"{r['month_str']:<10} {r['n']:>5.0f} {r['mean']:>8.1f} {r['std']:>8.1f} "
          f"{r['abs_mean']:>8.1f} {r['abs_std']:>8.1f} "
          f"{r['frac_gt50']:>6.1%} {r['frac_gt100']:>6.1%} {r['frac_gt200']:>6.1%} "
          f"{r['mean_positive']:>8.1f} {r['mean_negative']:>9.1f} {r['pos_neg_ratio']:>5.2f}")

# Year-over-year comparison
print("\n\nYear-over-Year Settlement Price Comparison (same month):")
print("-"*80)
for m in range(1, 13):
    rows = settle_df[settle_df['month'].apply(lambda p: p.month) == m]
    if len(rows) < 2:
        continue
    print(f"\nMonth {m:02d}:")
    for _, r in rows.iterrows():
        print(f"  {r['month_str']}: mean={r['mean']:+.1f}, |mean|={r['abs_mean']:.1f}, "
              f"std={r['std']:.1f}, %>100={r['frac_gt100']:.1%}")


# ============================================================
# SECTION 2: System Imbalance Behavior
# ============================================================
print("\n\n" + "="*80)
print("SECTION 2: SYSTEM IMBALANCE BEHAVIOR")
print("="*80)

imb_stats = []
for period, grp in df.groupby('month'):
    ib = grp['imbalance_mwh'].dropna()
    if len(ib) < 50:
        continue
    abs_ib = ib.abs()
    # Autocorrelation
    ac1 = ib.autocorr(lag=1) if len(ib) > 5 else np.nan
    ac4 = ib.autocorr(lag=4) if len(ib) > 10 else np.nan
    ac96 = ib.autocorr(lag=96) if len(ib) > 200 else np.nan
    imb_stats.append({
        'month': period,
        'n': len(ib),
        'mean': ib.mean(),
        'std': ib.std(),
        'frac_gt10': (abs_ib > 10).mean(),
        'frac_gt20': (abs_ib > 20).mean(),
        'frac_gt50': (abs_ib > 50).mean(),
        'ac_lag1': ac1,
        'ac_lag4': ac4,
        'ac_lag96': ac96,
    })

imb_df = pd.DataFrame(imb_stats)
imb_df['month_str'] = imb_df['month'].astype(str)

print("\nMonthly System Imbalance Statistics:")
print("-"*110)
print(f"{'Month':<10} {'N':>5} {'Mean':>8} {'Std':>8} "
      f"{'%>10':>7} {'%>20':>7} {'%>50':>7} "
      f"{'AC(1)':>7} {'AC(4)':>7} {'AC(96)':>7}")
print("-"*110)
for _, r in imb_df.iterrows():
    print(f"{r['month_str']:<10} {r['n']:>5.0f} {r['mean']:>8.2f} {r['std']:>8.2f} "
          f"{r['frac_gt10']:>6.1%} {r['frac_gt20']:>6.1%} {r['frac_gt50']:>6.1%} "
          f"{r['ac_lag1']:>7.3f} {r['ac_lag4']:>7.3f} {r['ac_lag96']:>7.3f}")

# Hourly imbalance pattern by year
print("\n\nHourly Imbalance Mean by Year (MWh):")
print("-"*80)
hourly_by_year = df.groupby(['year', 'hour'])['imbalance_mwh'].mean().unstack(0)
print(hourly_by_year.to_string(float_format='{:.2f}'.format))


# ============================================================
# SECTION 3: Spread (Settlement - IDM Mid) Behavior
# ============================================================
print("\n\n" + "="*80)
print("SECTION 3: SPREAD OPPORTUNITY (Settlement - IDM Mid)")
print("="*80)

spread_stats = []
for period, grp in df.groupby('month'):
    sp = grp['spread_raw'].dropna()
    if len(sp) < 50:
        continue
    abs_sp = sp.abs()

    # Correlation with proxy (regulation-based predictor)
    proxy_col = 'proxy_rmean16' if 'proxy_rmean16' in grp.columns else None
    corr_proxy = np.nan
    if proxy_col:
        valid = grp[['spread_raw', proxy_col]].dropna()
        if len(valid) > 30:
            corr_proxy = valid['spread_raw'].corr(valid[proxy_col])

    spread_stats.append({
        'month': period,
        'n': len(sp),
        'mean': sp.mean(),
        'std': sp.std(),
        'abs_mean': abs_sp.mean(),
        'frac_gt5': (abs_sp > 5).mean(),
        'frac_gt10': (abs_sp > 10).mean(),
        'frac_gt20': (abs_sp > 20).mean(),
        'corr_proxy': corr_proxy,
    })

spread_df = pd.DataFrame(spread_stats)
spread_df['month_str'] = spread_df['month'].astype(str)

print("\nMonthly Spread Statistics:")
print("-"*110)
print(f"{'Month':<10} {'N':>5} {'Mean':>8} {'Std':>8} {'|Mean|':>8} "
      f"{'%>5':>7} {'%>10':>7} {'%>20':>7} {'Corr_Proxy':>11}")
print("-"*110)
for _, r in spread_df.iterrows():
    print(f"{r['month_str']:<10} {r['n']:>5.0f} {r['mean']:>8.1f} {r['std']:>8.1f} "
          f"{r['abs_mean']:>8.1f} "
          f"{r['frac_gt5']:>6.1%} {r['frac_gt10']:>6.1%} {r['frac_gt20']:>6.1%} "
          f"{r['corr_proxy']:>11.3f}")

# YoY comparison
print("\n\nYear-over-Year Spread Comparison:")
print("-"*80)
for m in range(1, 13):
    rows = spread_df[spread_df['month'].apply(lambda p: p.month) == m]
    if len(rows) < 2:
        continue
    print(f"\nMonth {m:02d}:")
    for _, r in rows.iterrows():
        print(f"  {r['month_str']}: mean_spread={r['mean']:+.1f}, |spread|={r['abs_mean']:.1f}, "
              f"std={r['std']:.1f}, corr_proxy={r['corr_proxy']:.3f}")


# ============================================================
# SECTION 4: IDM Market Liquidity
# ============================================================
print("\n\n" + "="*80)
print("SECTION 4: IDM MARKET LIQUIDITY")
print("="*80)

liq_stats = []
for period, grp in df.groupby('month'):
    ba = grp['exec_spread'].dropna()
    if len(ba) < 50:
        continue
    vwap_col = 'idm_vwap' if 'idm_vwap' in grp.columns else ('idm_vwap_lag' if 'idm_vwap_lag' in grp.columns else None)
    vol_col = 'idm_volume_mwh' if 'idm_volume_mwh' in grp.columns else ('idm_volume_lag' if 'idm_volume_lag' in grp.columns else None)
    vwap = grp[vwap_col].dropna() if vwap_col else pd.Series(dtype=float)
    vol = grp[vol_col].dropna() if vol_col else pd.Series(dtype=float)
    liq_stats.append({
        'month': period,
        'n': len(ba),
        'ba_mean': ba.mean(),
        'ba_std': ba.std(),
        'ba_median': ba.median(),
        'frac_illiquid': (ba > 10).mean(),
        'frac_tight': (ba < 2).mean(),
        'idm_vwap_mean': vwap.mean() if len(vwap) > 0 else np.nan,
        'idm_vol_mean': vol.mean() if len(vol) > 0 else np.nan,
    })

liq_df = pd.DataFrame(liq_stats)
liq_df['month_str'] = liq_df['month'].astype(str)

print("\nMonthly IDM Liquidity Statistics:")
print("-"*110)
print(f"{'Month':<10} {'N':>5} {'BA_Mean':>8} {'BA_Med':>8} {'BA_Std':>8} "
      f"{'%Illiq':>7} {'%Tight':>7} {'VWAP':>8} {'Vol_MWh':>8}")
print("-"*110)
for _, r in liq_df.iterrows():
    print(f"{r['month_str']:<10} {r['n']:>5.0f} {r['ba_mean']:>8.2f} {r['ba_median']:>8.2f} "
          f"{r['ba_std']:>8.2f} "
          f"{r['frac_illiquid']:>6.1%} {r['frac_tight']:>6.1%} "
          f"{r['idm_vwap_mean']:>8.1f} {r['idm_vol_mean']:>8.1f}")


# ============================================================
# SECTION 5: Feature-Target Correlations Over Time
# ============================================================
print("\n\n" + "="*80)
print("SECTION 5: FEATURE-TARGET CORRELATIONS OVER TIME")
print("="*80)

key_features = []
for f in ['da_price', 'proxy_rmean16', 'proxy_rmean4', 'proxy_momentum',
          'load_deviation', 'reg_rmean8', 'proxy_abs_rmean8',
          'idm_vwap_lag', 'idm_volume_lag']:
    if f in df.columns:
        key_features.append(f)

print(f"\n[*] Key features found: {key_features}")

corr_records = []
for period, grp in df.groupby('month'):
    sp = grp['spread_raw'].dropna()
    if len(sp) < 50:
        continue
    row = {'month': period}
    for feat in key_features:
        valid = grp[['spread_raw', feat]].dropna()
        if len(valid) > 30:
            row[feat] = valid['spread_raw'].corr(valid[feat])
        else:
            row[feat] = np.nan
    corr_records.append(row)

corr_df = pd.DataFrame(corr_records)
corr_df['month_str'] = corr_df['month'].astype(str)

print("\nMonthly Feature-Spread Correlations:")
print("-"*120)
header = f"{'Month':<10}"
for f in key_features:
    header += f" {f[:14]:>14}"
print(header)
print("-"*120)
for _, r in corr_df.iterrows():
    line = f"{r['month_str']:<10}"
    for f in key_features:
        val = r.get(f, np.nan)
        if pd.isna(val):
            line += f" {'N/A':>14}"
        else:
            line += f" {val:>14.3f}"
    print(line)


# ============================================================
# SECTION 6: Seasonal Decomposition
# ============================================================
print("\n\n" + "="*80)
print("SECTION 6: SEASONAL DECOMPOSITION (2026 Anomaly)")
print("="*80)

# Build seasonal baselines from 2024-2025
def compute_seasonal_baseline(stats_df, col, years=[2024, 2025]):
    """Compute average of 'col' for each calendar month across given years."""
    stats_df = stats_df.copy()
    stats_df['cal_month'] = stats_df['month'].apply(lambda p: p.month)
    stats_df['cal_year'] = stats_df['month'].apply(lambda p: p.year)
    baseline = stats_df[stats_df['cal_year'].isin(years)].groupby('cal_month')[col].mean()
    return baseline

def compute_anomaly(stats_df, col, baseline):
    """Compute 2026 anomaly vs seasonal baseline."""
    stats_df = stats_df.copy()
    stats_df['cal_month'] = stats_df['month'].apply(lambda p: p.month)
    stats_df['cal_year'] = stats_df['month'].apply(lambda p: p.year)
    df_2026 = stats_df[stats_df['cal_year'] == 2026].copy()
    df_2026['seasonal_expected'] = df_2026['cal_month'].map(baseline)
    df_2026['anomaly'] = df_2026[col] - df_2026['seasonal_expected']
    return df_2026

# Settlement price anomaly
settle_bl = compute_seasonal_baseline(settle_df, 'abs_mean')
settle_anom = compute_anomaly(settle_df, 'abs_mean', settle_bl)

# Spread opportunity anomaly
spread_bl = compute_seasonal_baseline(spread_df, 'abs_mean')
spread_anom = compute_anomaly(spread_df, 'abs_mean', spread_bl)

# Imbalance std anomaly
imb_bl = compute_seasonal_baseline(imb_df, 'std')
imb_anom = compute_anomaly(imb_df, 'std', imb_bl)

# Bid-ask anomaly
ba_bl = compute_seasonal_baseline(liq_df, 'ba_mean')
ba_anom = compute_anomaly(liq_df, 'ba_mean', ba_bl)

# Proxy correlation anomaly
proxy_corr_col = 'proxy_rmean16' if 'proxy_rmean16' in corr_df.columns else None
if proxy_corr_col:
    corr_bl = compute_seasonal_baseline(corr_df, proxy_corr_col)
    corr_anom = compute_anomaly(corr_df, proxy_corr_col, corr_bl)

print("\n2026 Seasonal Anomaly Analysis:")
print("-"*100)
print(f"{'Metric':<35} {'Month':<8} {'Actual':>10} {'Seasonal':>10} {'Anomaly':>10} {'Anomaly%':>10}")
print("-"*100)

for _, r in settle_anom.iterrows():
    pct = r['anomaly'] / r['seasonal_expected'] * 100 if r['seasonal_expected'] != 0 else np.nan
    print(f"{'|Settlement Price| (EUR/MWh)':<35} {r['cal_month']:>5}    {r['abs_mean']:>10.1f} "
          f"{r['seasonal_expected']:>10.1f} {r['anomaly']:>+10.1f} {pct:>+9.0f}%")

print()
for _, r in spread_anom.iterrows():
    pct = r['anomaly'] / r['seasonal_expected'] * 100 if r['seasonal_expected'] != 0 else np.nan
    print(f"{'|Spread| (EUR/MWh)':<35} {r['cal_month']:>5}    {r['abs_mean']:>10.1f} "
          f"{r['seasonal_expected']:>10.1f} {r['anomaly']:>+10.1f} {pct:>+9.0f}%")

print()
for _, r in imb_anom.iterrows():
    pct = r['anomaly'] / r['seasonal_expected'] * 100 if r['seasonal_expected'] != 0 else np.nan
    print(f"{'Imbalance Std (MWh)':<35} {r['cal_month']:>5}    {r['std']:>10.2f} "
          f"{r['seasonal_expected']:>10.2f} {r['anomaly']:>+10.2f} {pct:>+9.0f}%")

print()
for _, r in ba_anom.iterrows():
    pct = r['anomaly'] / r['seasonal_expected'] * 100 if r['seasonal_expected'] != 0 else np.nan
    print(f"{'Bid-Ask Spread (EUR)':<35} {r['cal_month']:>5}    {r['ba_mean']:>10.2f} "
          f"{r['seasonal_expected']:>10.2f} {r['anomaly']:>+10.2f} {pct:>+9.0f}%")

if proxy_corr_col:
    print()
    for _, r in corr_anom.iterrows():
        print(f"{'Proxy-Spread Correlation':<35} {r['cal_month']:>5}    {r[proxy_corr_col]:>10.3f} "
              f"{r['seasonal_expected']:>10.3f} {r['anomaly']:>+10.3f}")


# ============================================================
# PLOTS
# ============================================================
print("\n\n[*] Generating plots...")

# Convert period to timestamp for plotting
def period_to_ts(periods):
    return [p.to_timestamp() for p in periods]


# --- Plot 1: Settlement Price Evolution ---
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Settlement Price Evolution (Monthly)', fontsize=14, fontweight='bold')

ax = axes[0, 0]
ts = period_to_ts(settle_df['month'])
ax.plot(ts, settle_df['mean'], 'b-o', label='Mean', markersize=4)
ax.fill_between(ts, settle_df['mean'] - settle_df['std'],
                settle_df['mean'] + settle_df['std'], alpha=0.2, color='blue')
ax.set_title('Mean Settlement Price (+/- 1 Std)')
ax.set_ylabel('EUR/MWh')
ax.legend()
ax.axhline(0, color='k', lw=0.5)

ax = axes[0, 1]
ax.plot(ts, settle_df['abs_mean'], 'r-o', label='Mean |Price|', markersize=4)
ax.fill_between(ts, settle_df['abs_mean'] - settle_df['abs_std'],
                settle_df['abs_mean'] + settle_df['abs_std'], alpha=0.2, color='red')
ax.set_title('Mean |Settlement Price| (Price Extremity)')
ax.set_ylabel('EUR/MWh')
ax.legend()

ax = axes[1, 0]
ax.plot(ts, settle_df['frac_gt50'], 's-', label='|Price| > 50', markersize=4)
ax.plot(ts, settle_df['frac_gt100'], 's-', label='|Price| > 100', markersize=4)
ax.plot(ts, settle_df['frac_gt200'], 's-', label='|Price| > 200', markersize=4)
ax.set_title('Tail Event Frequency')
ax.set_ylabel('Fraction of Periods')
ax.legend()

# YoY overlay
ax = axes[1, 1]
for yr in [2024, 2025, 2026]:
    sub = settle_df[settle_df['month'].apply(lambda p: p.year) == yr]
    months = [p.month for p in sub['month']]
    ax.plot(months, sub['abs_mean'], 'o-', label=str(yr), markersize=6)
ax.set_title('Year-over-Year |Settlement Price|')
ax.set_xlabel('Calendar Month')
ax.set_ylabel('EUR/MWh')
ax.set_xticks(range(1, 13))
ax.legend()

plt.tight_layout()
fig.savefig(PLOT_DIR / '01_settlement_price_evolution.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[+] Saved 01_settlement_price_evolution.png")


# --- Plot 2: Spread Opportunity ---
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Spread Opportunity Evolution (Monthly)', fontsize=14, fontweight='bold')

ax = axes[0, 0]
ts = period_to_ts(spread_df['month'])
ax.plot(ts, spread_df['abs_mean'], 'g-o', label='Mean |Spread|', markersize=4)
ax.set_title('Mean |Spread| = P&L Opportunity per QH')
ax.set_ylabel('EUR/MWh')
ax.legend()

ax = axes[0, 1]
# YoY overlay for spread
for yr in [2024, 2025, 2026]:
    sub = spread_df[spread_df['month'].apply(lambda p: p.year) == yr]
    months = [p.month for p in sub['month']]
    ax.plot(months, sub['abs_mean'], 'o-', label=str(yr), markersize=6)
# Seasonal baseline
if len(spread_bl) > 0:
    ax.plot(spread_bl.index, spread_bl.values, 'k--', label='2024-2025 Avg', lw=2, alpha=0.7)
ax.set_title('Year-over-Year |Spread| with Seasonal Baseline')
ax.set_xlabel('Calendar Month')
ax.set_ylabel('EUR/MWh')
ax.set_xticks(range(1, 13))
ax.legend()

ax = axes[1, 0]
ax.plot(ts, spread_df['frac_gt5'], 's-', label='|Spread| > 5', markersize=4)
ax.plot(ts, spread_df['frac_gt10'], 's-', label='|Spread| > 10', markersize=4)
ax.plot(ts, spread_df['frac_gt20'], 's-', label='|Spread| > 20', markersize=4)
ax.set_title('Spread Tail Events')
ax.set_ylabel('Fraction of Periods')
ax.legend()

ax = axes[1, 1]
ax.plot(ts, spread_df['corr_proxy'], 'purple', marker='o', markersize=4)
ax.axhline(0, color='k', lw=0.5)
ax.set_title('Proxy-Spread Correlation (Predictability)')
ax.set_ylabel('Correlation')

plt.tight_layout()
fig.savefig(PLOT_DIR / '02_spread_opportunity.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[+] Saved 02_spread_opportunity.png")


# --- Plot 3: Liquidity Evolution ---
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('IDM Market Liquidity Evolution (Monthly)', fontsize=14, fontweight='bold')

ax = axes[0, 0]
ts = period_to_ts(liq_df['month'])
ax.plot(ts, liq_df['ba_mean'], 'b-o', label='Mean Bid-Ask', markersize=4)
ax.plot(ts, liq_df['ba_median'], 'b--s', label='Median Bid-Ask', markersize=4, alpha=0.7)
ax.set_title('Bid-Ask Spread')
ax.set_ylabel('EUR/MWh')
ax.legend()

ax = axes[0, 1]
ax.plot(ts, liq_df['frac_illiquid'], 'r-o', label='BA > 10 (Illiquid)', markersize=4)
ax.plot(ts, liq_df['frac_tight'], 'g-s', label='BA < 2 (Tight)', markersize=4)
ax.set_title('Liquidity Distribution')
ax.set_ylabel('Fraction of Periods')
ax.legend()

ax = axes[1, 0]
# YoY bid-ask
for yr in [2024, 2025, 2026]:
    sub = liq_df[liq_df['month'].apply(lambda p: p.year) == yr]
    months = [p.month for p in sub['month']]
    ax.plot(months, sub['ba_mean'], 'o-', label=str(yr), markersize=6)
ax.set_title('Year-over-Year Bid-Ask Spread')
ax.set_xlabel('Calendar Month')
ax.set_ylabel('EUR/MWh')
ax.set_xticks(range(1, 13))
ax.legend()

ax = axes[1, 1]
if 'idm_vol_mean' in liq_df.columns:
    ax.plot(ts, liq_df['idm_vol_mean'], 'orange', marker='o', markersize=4)
    ax.set_title('Mean IDM Volume')
    ax.set_ylabel('MWh per period')

plt.tight_layout()
fig.savefig(PLOT_DIR / '03_liquidity_evolution.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[+] Saved 03_liquidity_evolution.png")


# --- Plot 4: Feature-Target Correlations ---
n_feats = len(key_features)
n_rows = (n_feats + 1) // 2
fig, axes = plt.subplots(n_rows, 2, figsize=(18, 4 * n_rows))
fig.suptitle('Feature-Spread Correlation Evolution (Monthly)', fontsize=14, fontweight='bold')
axes = axes.flatten()

ts = period_to_ts(corr_df['month'])
for i, feat in enumerate(key_features):
    ax = axes[i]
    vals = corr_df[feat].values
    ax.plot(ts, vals, 'o-', markersize=4)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(feat)
    ax.set_ylabel('Correlation with Spread')
    # Add trend line
    valid_mask = ~np.isnan(vals)
    if valid_mask.sum() > 5:
        x = np.arange(len(vals))[valid_mask]
        y = vals[valid_mask]
        z = np.polyfit(x, y, 1)
        ax.plot(np.array(ts)[valid_mask], np.polyval(z, x), 'r--', alpha=0.5, label=f'trend: {z[0]:+.4f}/mo')
        ax.legend(fontsize=9)

# Hide unused axes
for i in range(n_feats, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
fig.savefig(PLOT_DIR / '04_feature_target_correlations.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[+] Saved 04_feature_target_correlations.png")


# --- Plot 5: Seasonal Anomaly Dashboard ---
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('2026 Anomaly vs Seasonal Expected (2024-2025 Average)', fontsize=14, fontweight='bold')

# 5a: |Settlement Price| anomaly
ax = axes[0, 0]
if len(settle_anom) > 0:
    months = settle_anom['cal_month'].values
    ax.bar(months - 0.15, settle_anom['seasonal_expected'].values, 0.3, label='Seasonal Expected', alpha=0.7, color='gray')
    ax.bar(months + 0.15, settle_anom['abs_mean'].values, 0.3, label='2026 Actual', alpha=0.7, color='red')
    ax.set_title('|Settlement Price|')
    ax.set_ylabel('EUR/MWh')
    ax.legend()
    ax.set_xticks(months)

# 5b: |Spread| anomaly
ax = axes[0, 1]
if len(spread_anom) > 0:
    months = spread_anom['cal_month'].values
    ax.bar(months - 0.15, spread_anom['seasonal_expected'].values, 0.3, label='Seasonal Expected', alpha=0.7, color='gray')
    ax.bar(months + 0.15, spread_anom['abs_mean'].values, 0.3, label='2026 Actual', alpha=0.7, color='green')
    ax.set_title('|Spread| (P&L Opportunity)')
    ax.set_ylabel('EUR/MWh')
    ax.legend()
    ax.set_xticks(months)

# 5c: Imbalance Std anomaly
ax = axes[0, 2]
if len(imb_anom) > 0:
    months = imb_anom['cal_month'].values
    ax.bar(months - 0.15, imb_anom['seasonal_expected'].values, 0.3, label='Seasonal Expected', alpha=0.7, color='gray')
    ax.bar(months + 0.15, imb_anom['std'].values, 0.3, label='2026 Actual', alpha=0.7, color='blue')
    ax.set_title('Imbalance Volatility (Std)')
    ax.set_ylabel('MWh')
    ax.legend()
    ax.set_xticks(months)

# 5d: Bid-Ask anomaly
ax = axes[1, 0]
if len(ba_anom) > 0:
    months = ba_anom['cal_month'].values
    ax.bar(months - 0.15, ba_anom['seasonal_expected'].values, 0.3, label='Seasonal Expected', alpha=0.7, color='gray')
    ax.bar(months + 0.15, ba_anom['ba_mean'].values, 0.3, label='2026 Actual', alpha=0.7, color='orange')
    ax.set_title('Bid-Ask Spread')
    ax.set_ylabel('EUR/MWh')
    ax.legend()
    ax.set_xticks(months)

# 5e: Anomaly bars (% change from seasonal)
ax = axes[1, 1]
metrics = []
vals = []
if len(settle_anom) > 0:
    for _, r in settle_anom.iterrows():
        if r['seasonal_expected'] != 0:
            metrics.append(f"|Price| M{int(r['cal_month'])}")
            vals.append(r['anomaly'] / r['seasonal_expected'] * 100)
if len(spread_anom) > 0:
    for _, r in spread_anom.iterrows():
        if r['seasonal_expected'] != 0:
            metrics.append(f"|Spread| M{int(r['cal_month'])}")
            vals.append(r['anomaly'] / r['seasonal_expected'] * 100)
if len(imb_anom) > 0:
    for _, r in imb_anom.iterrows():
        if r['seasonal_expected'] != 0:
            metrics.append(f"Imb_Std M{int(r['cal_month'])}")
            vals.append(r['anomaly'] / r['seasonal_expected'] * 100)
if len(ba_anom) > 0:
    for _, r in ba_anom.iterrows():
        if r['seasonal_expected'] != 0:
            metrics.append(f"BA M{int(r['cal_month'])}")
            vals.append(r['anomaly'] / r['seasonal_expected'] * 100)

if metrics:
    colors = ['green' if v > 0 else 'red' for v in vals]
    ax.barh(range(len(metrics)), vals, color=colors, alpha=0.7)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_xlabel('% Anomaly vs Seasonal')
    ax.set_title('2026 Anomaly Summary')
    ax.axvline(0, color='k', lw=0.5)

# 5f: Proxy correlation evolution
ax = axes[1, 2]
if proxy_corr_col and len(corr_anom) > 0:
    months_c = corr_anom['cal_month'].values
    ax.bar(months_c - 0.15, corr_anom['seasonal_expected'].values, 0.3,
           label='Seasonal Expected', alpha=0.7, color='gray')
    ax.bar(months_c + 0.15, corr_anom[proxy_corr_col].values, 0.3,
           label='2026 Actual', alpha=0.7, color='purple')
    ax.set_title('Proxy-Spread Correlation')
    ax.set_ylabel('Correlation')
    ax.legend()
    ax.set_xticks(months_c)
else:
    ax.set_visible(False)

plt.tight_layout()
fig.savefig(PLOT_DIR / '05_seasonal_anomaly.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[+] Saved 05_seasonal_anomaly.png")


# ============================================================
# NARRATIVE SUMMARY
# ============================================================
print("\n\n" + "="*80)
print("NARRATIVE SUMMARY")
print("="*80)

# Compute key summary metrics
# H2 2025 avg spread
h2_2025 = spread_df[(spread_df['month'] >= pd.Period('2025-07')) &
                     (spread_df['month'] <= pd.Period('2025-09'))]
q1_2026 = spread_df[(spread_df['month'] >= pd.Period('2026-01')) &
                     (spread_df['month'] <= pd.Period('2026-03'))]
apr_2026 = spread_df[spread_df['month'] == pd.Period('2026-04')]

print(f"""
--- Key Findings ---

1. SPREAD OPPORTUNITY:
   - H2 2025 avg |spread|: {h2_2025['abs_mean'].mean():.1f} EUR/MWh
   - Q1 2026 avg |spread|: {q1_2026['abs_mean'].mean():.1f} EUR/MWh
   - Apr 2026 |spread|:    {apr_2026['abs_mean'].values[0]:.1f} EUR/MWh (if data available)
   - Trend: {'DECLINING' if q1_2026['abs_mean'].mean() < h2_2025['abs_mean'].mean() else 'STABLE/INCREASING'}
""" if len(h2_2025) > 0 and len(q1_2026) > 0 else "")

# Settlement price comparison
print("2. SETTLEMENT PRICE EXTREMITY:")
for _, r in settle_anom.iterrows():
    pct = r['anomaly'] / r['seasonal_expected'] * 100 if r['seasonal_expected'] != 0 else 0
    direction = "BELOW" if r['anomaly'] < 0 else "ABOVE"
    print(f"   - Month {int(r['cal_month'])}: 2026 |price| {direction} seasonal by {abs(pct):.0f}%"
          f" ({r['abs_mean']:.1f} vs expected {r['seasonal_expected']:.1f})")

# Liquidity
print("\n3. IDM LIQUIDITY:")
for _, r in ba_anom.iterrows():
    pct = r['anomaly'] / r['seasonal_expected'] * 100 if r['seasonal_expected'] != 0 else 0
    direction = "tighter" if r['anomaly'] < 0 else "wider"
    print(f"   - Month {int(r['cal_month'])}: Bid-ask {direction} than seasonal by {abs(pct):.0f}%"
          f" ({r['ba_mean']:.2f} vs expected {r['seasonal_expected']:.2f})")

# Predictability
print("\n4. PREDICTABILITY (Proxy-Spread Correlation):")
if proxy_corr_col:
    for _, r in corr_anom.iterrows():
        print(f"   - Month {int(r['cal_month'])}: corr = {r[proxy_corr_col]:.3f}"
              f" (seasonal expected: {r['seasonal_expected']:.3f},"
              f" anomaly: {r['anomaly']:+.3f})")

# Overall diagnosis
print("\n5. DIAGNOSIS:")
# Check if spread decline is mostly seasonal
if len(spread_anom) > 0:
    avg_pct_anom = spread_anom.apply(
        lambda r: r['anomaly'] / r['seasonal_expected'] * 100
        if r['seasonal_expected'] != 0 else 0, axis=1).mean()
    print(f"   - Average 2026 |spread| anomaly: {avg_pct_anom:+.0f}% vs seasonal")
    if abs(avg_pct_anom) < 15:
        print("   -> Spread decline is MOSTLY SEASONAL (spring is naturally quieter)")
    elif avg_pct_anom < -15:
        print("   -> Spread decline has a STRUCTURAL COMPONENT beyond seasonality")
    else:
        print("   -> Spread opportunity is ABOVE seasonal expectations")

# Check if bid-ask tightening (market efficiency) is the driver
if len(ba_anom) > 0:
    ba_pct = ba_anom.apply(
        lambda r: r['anomaly'] / r['seasonal_expected'] * 100
        if r['seasonal_expected'] != 0 else 0, axis=1).mean()
    if ba_pct < -15:
        print("   - Bid-ask spreads are tighter than seasonal -> IDM market is more efficient")
    elif ba_pct > 15:
        print("   - Bid-ask spreads are wider than seasonal -> IDM market less liquid")
    else:
        print("   - Bid-ask spreads are near seasonal norms")

# Check predictability
if proxy_corr_col and len(corr_anom) > 0:
    corr_mean_2026 = corr_anom[proxy_corr_col].mean()
    corr_mean_bl = corr_anom['seasonal_expected'].mean()
    if abs(corr_mean_2026) < abs(corr_mean_bl) * 0.7:
        print("   - Proxy predictability has WEAKENED substantially vs seasonal baseline")
    else:
        print("   - Proxy predictability is within seasonal norms")

print("\n[+] Analysis complete. Plots saved to:", PLOT_DIR)
