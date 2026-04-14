"""
00_label: Spread Target (imb_settlement_price - exec_mid) Analysis
==================================================================
Generates:
  01_spread_distribution.png
  02_spread_seasonality.png
  03_spread_autocorrelation.png
  04_spread_regime_change.png
"""
import sys, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import acf, pacf
import calendar

# --- Data setup (inline, same pattern as 03/04) ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

LEAD = 8
OUT_DIR = Path(__file__).resolve().parent

YEAR_COLORS = {2024: 'steelblue', 2025: 'forestgreen', 2026: 'indianred'}

plt.rcParams.update({
    "figure.figsize": (16, 10), "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})

print("[*] Loading data...")
data = load_all_data()
tml.TRAIN_END = '2026-04-15'
tml.TEST_START = '2026-04-15'
df_base, feature_cols = build_features(data, LEAD)

ob = pd.read_csv(
    Path(__file__).resolve().parents[2] / "data" / "features" / "orderbook_qh_features.csv",
    parse_dates=['delivery_start'],
)
ob_120 = ob[ob['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'mid', 'spread']]
ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
ob_120.columns = ['exec_bid', 'exec_ask', 'exec_mid', 'exec_spread']
df_base = df_base.join(ob_120, how='left')
df_base['imb_settlement_price'] = df_base['imb_settle_price']

# Hourly-smoothed spread target (same as investigate_decay.py)
df_base['hour_ts'] = df_base.index.floor('h')
df_base['settle_hourly'] = df_base.groupby('hour_ts')['imb_settlement_price'].transform('mean')
df_base['mid_hourly'] = df_base.groupby('hour_ts')['exec_mid'].transform('mean')
df_base['spread_target'] = df_base['settle_hourly'] - df_base['mid_hourly']

sp = df_base.dropna(subset=['spread_target']).copy()
sp['year'] = sp.index.year
sp['abs_spread'] = sp['spread_target'].abs()
sp['hour'] = sp.index.hour
sp['dow'] = sp.index.dayofweek
sp['month'] = sp.index.month
sp['month_label'] = sp.index.to_period('M').astype(str)
sp['quarter_label'] = sp.index.year.astype(str) + '-Q' + sp.index.quarter.astype(str)

s = sp['spread_target']
print(f"[+] Dataset ready: {len(sp)} rows, spread_target mean={s.mean():.2f}, std={s.std():.2f}")

# ================================================================
# 01_spread_distribution.png
# ================================================================
print("[*] Creating 01_spread_distribution.png ...")
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# (a) Histogram full period
ax = fig.add_subplot(gs[0, 0])
vals = s.clip(-100, 100)
ax.hist(vals, bins=120, color='steelblue', alpha=0.7, edgecolor='none', density=True)
ax.axvline(vals.mean(), color='red', ls='--', lw=1.5, label=f'Mean={vals.mean():.2f}')
ax.axvline(vals.median(), color='orange', ls='--', lw=1.5, label=f'Median={vals.median():.2f}')
ax.set_xlabel('Spread Target (EUR/MWh)')
ax.set_ylabel('Density')
ax.set_title('Spread Distribution (Full Period, clipped [-100, 100])')
ax.legend(fontsize=9)

# (b) By year overlay
ax = fig.add_subplot(gs[0, 1])
for yr in sorted(sp['year'].unique()):
    subset = sp[sp['year'] == yr]['spread_target'].clip(-100, 100)
    ax.hist(subset, bins=80, alpha=0.45, color=YEAR_COLORS.get(yr, 'gray'),
            label=f'{yr} (n={len(subset)}, mean={subset.mean():.1f})', density=True, edgecolor='none')
ax.set_xlabel('Spread Target (EUR/MWh)')
ax.set_ylabel('Density')
ax.set_title('Spread Distribution by Year')
ax.legend(fontsize=9)

# (c) Box plot by quarter
ax = fig.add_subplot(gs[1, 0])
quarters = sorted(sp['quarter_label'].unique())
box_data = [sp[sp['quarter_label'] == q]['spread_target'].clip(-80, 80).values for q in quarters]
bp = ax.boxplot(box_data, labels=quarters, patch_artist=True, showfliers=False,
                medianprops=dict(color='black', lw=1.5))
for i, patch in enumerate(bp['boxes']):
    yr = int(quarters[i][:4])
    patch.set_facecolor(YEAR_COLORS.get(yr, 'gray'))
    patch.set_alpha(0.5)
ax.set_xticklabels(quarters, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Spread Target (EUR/MWh)')
ax.set_title('Spread by Quarter')

# (d) Stats table
ax = fig.add_subplot(gs[1, 1])
ax.axis('off')
stat_rows = [
    ['Mean', f'{s.mean():.2f}'],
    ['Median', f'{s.median():.2f}'],
    ['Std', f'{s.std():.2f}'],
    ['Skewness', f'{s.skew():.2f}'],
    ['Kurtosis', f'{s.kurtosis():.2f}'],
    ['P5', f'{s.quantile(0.05):.2f}'],
    ['P10', f'{s.quantile(0.10):.2f}'],
    ['P25', f'{s.quantile(0.25):.2f}'],
    ['P75', f'{s.quantile(0.75):.2f}'],
    ['P90', f'{s.quantile(0.90):.2f}'],
    ['P95', f'{s.quantile(0.95):.2f}'],
    ['IQR', f'{s.quantile(0.75) - s.quantile(0.25):.2f}'],
    ['Frac |s|>5', f'{(s.abs() > 5).mean():.1%}'],
    ['Frac |s|>10', f'{(s.abs() > 10).mean():.1%}'],
    ['Frac |s|>20', f'{(s.abs() > 20).mean():.1%}'],
    ['Frac |s|>50', f'{(s.abs() > 50).mean():.1%}'],
]
for yr in sorted(sp['year'].unique()):
    sub = sp[sp['year'] == yr]['spread_target']
    stat_rows.append([f'--- {yr} ---', ''])
    stat_rows.append([f'  Mean', f'{sub.mean():.2f}'])
    stat_rows.append([f'  Std', f'{sub.std():.2f}'])
    stat_rows.append([f'  |mean|', f'{sub.abs().mean():.2f}'])
    stat_rows.append([f'  P5/P95', f'{sub.quantile(0.05):.1f} / {sub.quantile(0.95):.1f}'])

table = ax.table(cellText=stat_rows, colLabels=['Statistic', 'Value'],
                 loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 0.65)
ax.set_title('Spread Statistics', fontsize=12, pad=10)

fig.suptitle('Spread Target Analysis: Distribution', fontsize=14, fontweight='bold', y=0.98)
fig.savefig(OUT_DIR / '01_spread_distribution.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("[+] Saved 01_spread_distribution.png")

# Print key stats
print(f"    Mean={s.mean():.2f}, Std={s.std():.2f}, Skew={s.skew():.2f}, Kurt={s.kurtosis():.2f}")
print(f"    P5={s.quantile(0.05):.2f}, P95={s.quantile(0.95):.2f}")
for yr in sorted(sp['year'].unique()):
    sub = sp[sp['year'] == yr]['spread_target']
    print(f"    {yr}: mean={sub.mean():.2f}, std={sub.std():.2f}, |mean|={sub.abs().mean():.2f}")

# ================================================================
# 02_spread_seasonality.png
# ================================================================
print("[*] Creating 02_spread_seasonality.png ...")
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# (a) Mean |spread| by hour, by year
ax = fig.add_subplot(gs[0, 0])
for yr in sorted(sp['year'].unique()):
    sub = sp[sp['year'] == yr]
    hourly = sub.groupby('hour')['abs_spread'].mean()
    ax.plot(hourly.index, hourly.values, '.-', color=YEAR_COLORS.get(yr, 'gray'),
            label=f'{yr}', lw=2)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Mean |Spread| (EUR/MWh)')
ax.set_title('Hourly Pattern of |Spread| by Year')
ax.legend()
ax.set_xticks(range(24))

# (b) Mean |spread| by day-of-week
ax = fig.add_subplot(gs[0, 1])
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for yr in sorted(sp['year'].unique()):
    sub = sp[sp['year'] == yr]
    dow_data = sub.groupby('dow')['abs_spread'].mean()
    ax.plot(dow_data.index, dow_data.values, '.-', color=YEAR_COLORS.get(yr, 'gray'),
            label=f'{yr}', lw=2)
ax.set_xticks(range(7))
ax.set_xticklabels(dow_names)
ax.set_xlabel('Day of Week')
ax.set_ylabel('Mean |Spread| (EUR/MWh)')
ax.set_title('Weekly Pattern of |Spread| by Year')
ax.legend()

# (c) Mean |spread| by month (bar chart, color by year)
ax = fig.add_subplot(gs[1, 0])
monthly = sp.groupby(['year', 'month']).agg(
    abs_spread_mean=('abs_spread', 'mean'),
    n=('abs_spread', 'count')
).reset_index()
years = sorted(monthly['year'].unique())
width = 0.8 / len(years)
for i, yr in enumerate(years):
    sub = monthly[monthly['year'] == yr]
    offsets = sub['month'] - 0.4 + i * width + width / 2
    ax.bar(offsets, sub['abs_spread_mean'], width=width,
           color=YEAR_COLORS.get(yr, 'gray'), alpha=0.7, label=f'{yr}')
ax.set_xticks(range(1, 13))
ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
ax.set_xlabel('Month')
ax.set_ylabel('Mean |Spread| (EUR/MWh)')
ax.set_title('Monthly Mean |Spread| by Year')
ax.legend()

# (d) Heatmap: hour x month of mean |spread|
ax = fig.add_subplot(gs[1, 1])
pivot = sp.pivot_table(values='abs_spread', index='hour', columns='month', aggfunc='mean')
im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_yticks(range(24))
ax.set_yticklabels(range(24), fontsize=8)
ax.set_xticks(range(12))
ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)], fontsize=8)
ax.set_ylabel('Hour of Day')
ax.set_xlabel('Month')
ax.set_title('Mean |Spread| Heatmap: Hour x Month')
plt.colorbar(im, ax=ax, shrink=0.7, label='EUR/MWh')

fig.suptitle('Spread Target Analysis: Seasonality', fontsize=14, fontweight='bold', y=0.98)
fig.savefig(OUT_DIR / '02_spread_seasonality.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("[+] Saved 02_spread_seasonality.png")

# ================================================================
# 03_spread_autocorrelation.png
# ================================================================
print("[*] Creating 03_spread_autocorrelation.png ...")

# Use hourly mean spread for ACF (avoids within-hour repetition of QH values)
hourly_spread = sp.groupby(sp.index.floor('h'))['spread_target'].mean().dropna()

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

nlags = 96  # 96 hours = 4 days

# (a) ACF
ax = fig.add_subplot(gs[0, 0])
acf_vals = acf(hourly_spread.values, nlags=nlags, fft=True)
ax.bar(range(nlags + 1), acf_vals, color='steelblue', alpha=0.7, width=1.0)
ci = 1.96 / np.sqrt(len(hourly_spread))
ax.axhline(ci, ls='--', color='red', alpha=0.5, label=f'95% CI (+/-{ci:.3f})')
ax.axhline(-ci, ls='--', color='red', alpha=0.5)
ax.set_xlabel('Lag (hours)')
ax.set_ylabel('ACF')
ax.set_title('Autocorrelation Function (hourly spread)')
ax.legend(fontsize=9)

# (b) PACF
ax = fig.add_subplot(gs[0, 1])
pacf_vals = pacf(hourly_spread.values, nlags=min(nlags, len(hourly_spread) // 2 - 1))
ax.bar(range(len(pacf_vals)), pacf_vals, color='darkorange', alpha=0.7, width=1.0)
ax.axhline(ci, ls='--', color='red', alpha=0.5)
ax.axhline(-ci, ls='--', color='red', alpha=0.5)
ax.set_xlabel('Lag (hours)')
ax.set_ylabel('PACF')
ax.set_title('Partial Autocorrelation Function (hourly spread)')

# (c) Rolling 30-day autocorrelation at lag 1, lag 4, lag 24
ax = fig.add_subplot(gs[1, :])
window_days = 30
window_h = window_days * 24

rolling_acf1 = hourly_spread.rolling(window_h, min_periods=window_h // 2).apply(
    lambda x: pd.Series(x).autocorr(lag=1), raw=False)
rolling_acf4 = hourly_spread.rolling(window_h, min_periods=window_h // 2).apply(
    lambda x: pd.Series(x).autocorr(lag=4), raw=False)
rolling_acf24 = hourly_spread.rolling(window_h, min_periods=window_h // 2).apply(
    lambda x: pd.Series(x).autocorr(lag=24), raw=False)

ax.plot(rolling_acf1.index, rolling_acf1.values, '-', color='steelblue',
        label='Lag 1h', lw=1.5, alpha=0.8)
ax.plot(rolling_acf4.index, rolling_acf4.values, '-', color='darkorange',
        label='Lag 4h', lw=1.5, alpha=0.8)
ax.plot(rolling_acf24.index, rolling_acf24.values, '-', color='green',
        label='Lag 24h', lw=1.5, alpha=0.8)
ax.axhline(0, color='gray', ls='-', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Rolling 30-day Autocorrelation')
ax.set_title('Rolling 30-day Autocorrelation of Hourly Spread')
ax.legend()

for yr_start in pd.to_datetime(['2025-01-01', '2026-01-01']):
    if yr_start >= rolling_acf1.index.min() and yr_start <= rolling_acf1.index.max():
        ax.axvline(yr_start, color='gray', ls='--', alpha=0.5)

fig.suptitle('Spread Target Analysis: Autocorrelation', fontsize=14, fontweight='bold', y=0.98)
fig.savefig(OUT_DIR / '03_spread_autocorrelation.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("[+] Saved 03_spread_autocorrelation.png")

# ================================================================
# 04_spread_regime_change.png
# ================================================================
print("[*] Creating 04_spread_regime_change.png ...")

# Compute daily statistics
daily = sp.groupby(sp.index.date).agg(
    spread_mean=('spread_target', 'mean'),
    spread_abs_mean=('abs_spread', 'mean'),
    spread_std=('spread_target', 'std'),
    frac_gt10=('abs_spread', lambda x: (x > 10).mean()),
    n=('spread_target', 'count'),
)
daily.index = pd.to_datetime(daily.index)

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# (a) Rolling 30-day mean |spread|
ax = fig.add_subplot(gs[0, :])
roll_abs = daily['spread_abs_mean'].rolling(30, min_periods=15).mean()
ax.plot(roll_abs.index, roll_abs.values, '-', color='steelblue', lw=2)
ax.fill_between(roll_abs.index, 0, roll_abs.values, alpha=0.15, color='steelblue')
ax.set_ylabel('30-day Rolling Mean |Spread| (EUR/MWh)')
ax.set_title('Mean |Spread| Over Time (30-day Rolling)')
for yr_start in pd.to_datetime(['2025-01-01', '2026-01-01']):
    ax.axvline(yr_start, color='gray', ls='--', alpha=0.5, label=str(yr_start.year))
ax.legend()

# (b) Rolling 30-day std
ax = fig.add_subplot(gs[1, 0])
roll_std = daily['spread_std'].rolling(30, min_periods=15).mean()
ax.plot(roll_std.index, roll_std.values, '-', color='darkorange', lw=2)
ax.set_ylabel('30-day Rolling Std of Spread (EUR/MWh)')
ax.set_title('Spread Volatility Over Time')
for yr_start in pd.to_datetime(['2025-01-01', '2026-01-01']):
    ax.axvline(yr_start, color='gray', ls='--', alpha=0.5)

# (c) Rolling 30-day fraction |spread|>10
ax = fig.add_subplot(gs[1, 1])
roll_frac = daily['frac_gt10'].rolling(30, min_periods=15).mean()
ax.plot(roll_frac.index, roll_frac.values, '-', color='red', lw=2)
ax.set_ylabel('Fraction |Spread| > 10 EUR')
ax.set_title('Tail Frequency Over Time (30-day Rolling)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
for yr_start in pd.to_datetime(['2025-01-01', '2026-01-01']):
    ax.axvline(yr_start, color='gray', ls='--', alpha=0.5)

# (d) Year-over-year overlay: 2025 vs 2026 (align by day-of-year)
ax = fig.add_subplot(gs[2, :])
for yr, color in [(2025, YEAR_COLORS[2025]), (2026, YEAR_COLORS[2026])]:
    yr_data = daily[daily.index.year == yr].copy()
    if len(yr_data) == 0:
        continue
    yr_data['doy'] = yr_data.index.dayofyear
    roll = yr_data.set_index('doy')['spread_abs_mean'].rolling(14, min_periods=7).mean()
    ax.plot(roll.index, roll.values, '-', color=color, lw=2, label=f'{yr}')
ax.set_xlabel('Day of Year')
ax.set_ylabel('14-day Rolling Mean |Spread| (EUR/MWh)')
ax.set_title('Year-over-Year Comparison: |Spread| by Day of Year')
ax.legend()
month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
ax.set_xticks(month_starts)
ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)], fontsize=9)

fig.suptitle('Spread Target Analysis: Regime Change', fontsize=14, fontweight='bold', y=0.98)
fig.savefig(OUT_DIR / '04_spread_regime_change.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("[+] Saved 04_spread_regime_change.png")

# ================================================================
# Print summary data for summary.md
# ================================================================
print("\n" + "=" * 60)
print("SUMMARY DATA FOR 00_label")
print("=" * 60)

hourly_abs = sp.groupby('hour')['abs_spread'].mean()
top_hours = hourly_abs.sort_values(ascending=False).head(5)
print(f"\nTop 5 hours by mean |spread|:")
for h, v in top_hours.items():
    print(f"  Hour {h:2d}: {v:.2f} EUR/MWh")

bottom_hours = hourly_abs.sort_values().head(3)
print(f"\nBottom 3 hours by mean |spread|:")
for h, v in bottom_hours.items():
    print(f"  Hour {h:2d}: {v:.2f} EUR/MWh")

print(f"\nACF lag 1h: {acf_vals[1]:.3f}")
print(f"ACF lag 4h: {acf_vals[4]:.3f}")
print(f"ACF lag 24h: {acf_vals[24]:.3f}")

for yr in sorted(sp['year'].unique()):
    sub = sp[sp['year'] == yr]['spread_target']
    print(f"\n{yr}: mean |spread|={sub.abs().mean():.2f}, std={sub.std():.2f}, "
          f"frac>10={((sub.abs() > 10).mean()):.1%}")

print("\n[+] 00_label complete.")
