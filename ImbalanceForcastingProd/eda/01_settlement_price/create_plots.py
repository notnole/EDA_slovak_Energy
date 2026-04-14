"""
01_settlement_price: Settlement Price Deep Dive
================================================
Generates:
  01_distribution_and_extremes.png
  02_hourly_pattern.png
  03_monthly_evolution.png
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
import calendar

# --- Data setup (inline) ---
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
df_base, _ = build_features(data, LEAD)

ob = pd.read_csv(
    Path(__file__).resolve().parents[2] / "data" / "features" / "orderbook_qh_features.csv",
    parse_dates=['delivery_start'],
)
ob_120 = ob[ob['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'mid', 'spread']]
ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
ob_120.columns = ['exec_bid', 'exec_ask', 'exec_mid', 'exec_spread']
df_base = df_base.join(ob_120, how='left')
df_base['imb_settlement_price'] = df_base['imb_settle_price']

sp = df_base[df_base['imb_settlement_price'].notna()].copy()
sp['year'] = sp.index.year
sp['hour'] = sp.index.hour
sp['month'] = sp.index.month
sp['month_label'] = sp.index.to_period('M').astype(str)
price = sp['imb_settlement_price']

print(f"[+] Dataset ready: {len(sp)} rows, settlement price mean={price.mean():.2f}")

# ================================================================
# 01_distribution_and_extremes.png
# ================================================================
print("[*] Creating 01_distribution_and_extremes.png ...")
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# (a) Histogram by year
ax = fig.add_subplot(gs[0, 0])
for yr in sorted(sp['year'].unique()):
    vals = sp[sp['year'] == yr]['imb_settlement_price'].clip(-200, 500)
    ax.hist(vals, bins=100, alpha=0.45, color=YEAR_COLORS.get(yr, 'gray'),
            label=f'{yr} (n={len(vals)}, mean={vals.mean():.1f})', density=True, edgecolor='none')
ax.set_xlabel('Settlement Price (EUR/MWh)')
ax.set_ylabel('Density')
ax.set_title('Settlement Price Distribution by Year')
ax.legend(fontsize=9)

# (b) Tail analysis
ax = fig.add_subplot(gs[0, 1])
thresholds = [50, 100, 200, 500, 1000]
years = sorted(sp['year'].unique())
width = 0.8 / len(years)
for i, yr in enumerate(years):
    sub = sp[sp['year'] == yr]['imb_settlement_price']
    fracs = [(sub.abs() > t).mean() * 100 for t in thresholds]
    x = np.arange(len(thresholds)) + i * width - 0.4 + width / 2
    ax.bar(x, fracs, width=width, color=YEAR_COLORS.get(yr, 'gray'),
           alpha=0.7, label=f'{yr}')
ax.set_xticks(range(len(thresholds)))
ax.set_xticklabels([f'|p|>{t}' for t in thresholds])
ax.set_ylabel('% of Periods')
ax.set_title('Extreme Price Frequency by Year')
ax.legend(fontsize=9)

# (c) Positive vs negative distribution
ax = fig.add_subplot(gs[1, 0])
pos = price[price > 0]
neg = price[price < 0]
ax.hist(pos.clip(0, 500), bins=80, alpha=0.6, color='green', label=f'Positive (n={len(pos)})',
        density=True, edgecolor='none')
ax.hist(neg.clip(-200, 0), bins=40, alpha=0.6, color='red', label=f'Negative (n={len(neg)})',
        density=True, edgecolor='none')
ax.set_xlabel('Settlement Price (EUR/MWh)')
ax.set_ylabel('Density')
ax.set_title('Positive vs Negative Price Distribution')
ax.legend()

# (d) Stats table
ax = fig.add_subplot(gs[1, 1])
ax.axis('off')
rows = [
    ['Overall Mean', f'{price.mean():.2f}'],
    ['Overall Std', f'{price.std():.2f}'],
    ['Min', f'{price.min():.2f}'],
    ['Max', f'{price.max():.2f}'],
    ['Frac Negative', f'{(price < 0).mean():.1%}'],
    ['Frac > 200', f'{(price > 200).mean():.1%}'],
    ['Frac < -50', f'{(price < -50).mean():.1%}'],
]
for yr in sorted(sp['year'].unique()):
    sub = sp[sp['year'] == yr]['imb_settlement_price']
    rows.append([f'--- {yr} ---', ''])
    rows.append([f'  Mean', f'{sub.mean():.2f}'])
    rows.append([f'  Std', f'{sub.std():.2f}'])
    rows.append([f'  P5/P95', f'{sub.quantile(0.05):.1f} / {sub.quantile(0.95):.1f}'])

table = ax.table(cellText=rows, colLabels=['Statistic', 'Value'],
                 loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 0.75)
ax.set_title('Settlement Price Statistics', fontsize=12, pad=10)

fig.suptitle('Settlement Price Deep Dive: Distribution & Extremes',
             fontsize=14, fontweight='bold', y=0.98)
fig.savefig(OUT_DIR / '01_distribution_and_extremes.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("[+] Saved 01_distribution_and_extremes.png")

# ================================================================
# 02_hourly_pattern.png
# ================================================================
print("[*] Creating 02_hourly_pattern.png ...")
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# (a) Mean settlement by hour per year
ax = fig.add_subplot(gs[0, 0])
for yr in sorted(sp['year'].unique()):
    sub = sp[sp['year'] == yr]
    hourly = sub.groupby('hour')['imb_settlement_price'].mean()
    ax.plot(hourly.index, hourly.values, '.-', color=YEAR_COLORS.get(yr, 'gray'),
            label=f'{yr}', lw=2)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Mean Settlement Price (EUR/MWh)')
ax.set_title('Mean Settlement Price by Hour')
ax.legend()
ax.set_xticks(range(24))

# (b) Std of settlement by hour
ax = fig.add_subplot(gs[0, 1])
for yr in sorted(sp['year'].unique()):
    sub = sp[sp['year'] == yr]
    hourly_std = sub.groupby('hour')['imb_settlement_price'].std()
    ax.plot(hourly_std.index, hourly_std.values, '.-', color=YEAR_COLORS.get(yr, 'gray'),
            label=f'{yr}', lw=2)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Std Settlement Price (EUR/MWh)')
ax.set_title('Settlement Price Volatility by Hour')
ax.legend()
ax.set_xticks(range(24))

# (c) Fraction extreme by hour
ax = fig.add_subplot(gs[1, 0])
for yr in sorted(sp['year'].unique()):
    sub = sp[sp['year'] == yr]
    hourly_ext = sub.groupby('hour').apply(
        lambda x: (x['imb_settlement_price'].abs() > 100).mean())
    ax.plot(hourly_ext.index, hourly_ext.values * 100, '.-',
            color=YEAR_COLORS.get(yr, 'gray'), label=f'{yr}', lw=2)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('% Periods with |Price| > 100')
ax.set_title('Extreme Price Frequency by Hour')
ax.legend()
ax.set_xticks(range(24))

# (d) Settlement volatility heatmap: hour x month
ax = fig.add_subplot(gs[1, 1])
pivot = sp.pivot_table(values='imb_settlement_price', index='hour',
                       columns='month', aggfunc='std')
im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_yticks(range(24))
ax.set_xticks(range(12))
ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)], fontsize=8)
ax.set_ylabel('Hour of Day')
ax.set_title('Settlement Volatility Heatmap: Hour x Month')
plt.colorbar(im, ax=ax, shrink=0.7, label='Std EUR/MWh')

fig.suptitle('Settlement Price Deep Dive: Hourly Patterns',
             fontsize=14, fontweight='bold', y=0.98)
fig.savefig(OUT_DIR / '02_hourly_pattern.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("[+] Saved 02_hourly_pattern.png")

# ================================================================
# 03_monthly_evolution.png
# ================================================================
print("[*] Creating 03_monthly_evolution.png ...")
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

monthly = sp.groupby('month_label').agg(
    mean=('imb_settlement_price', 'mean'),
    std=('imb_settlement_price', 'std'),
    p10=('imb_settlement_price', lambda x: x.quantile(0.10)),
    p50=('imb_settlement_price', 'median'),
    p90=('imb_settlement_price', lambda x: x.quantile(0.90)),
    n=('imb_settlement_price', 'count'),
)

# (a) Monthly mean and std
ax = fig.add_subplot(gs[0, :])
x = range(len(monthly))
ax.plot(x, monthly['mean'], 'b.-', lw=2, label='Mean')
ax.fill_between(x, monthly['mean'] - monthly['std'],
                monthly['mean'] + monthly['std'], alpha=0.15, color='blue')
ax.set_xticks(x)
ax.set_xticklabels(monthly.index, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Settlement Price (EUR/MWh)')
ax.set_title('Monthly Mean +/- 1 Std')
ax.legend()
for i, label in enumerate(monthly.index):
    if label.endswith('-01'):
        ax.axvline(i, color='gray', ls='--', alpha=0.4)

# (b) P10/P50/P90 bands
ax = fig.add_subplot(gs[1, 0])
ax.fill_between(x, monthly['p10'], monthly['p90'], alpha=0.2, color='steelblue', label='P10-P90')
ax.plot(x, monthly['p50'], 'b.-', lw=2, label='Median')
ax.set_xticks(x)
ax.set_xticklabels(monthly.index, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Settlement Price (EUR/MWh)')
ax.set_title('Monthly P10 / P50 / P90')
ax.legend()

# (c) YoY volatility comparison
ax = fig.add_subplot(gs[1, 1])
sp['ym'] = sp.index.month
for yr in sorted(sp['year'].unique()):
    sub = sp[sp['year'] == yr]
    monthly_yr = sub.groupby('ym')['imb_settlement_price'].std()
    ax.plot(monthly_yr.index, monthly_yr.values, '.-', color=YEAR_COLORS.get(yr, 'gray'),
            label=f'{yr}', lw=2)
ax.set_xticks(range(1, 13))
ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
ax.set_ylabel('Std Settlement Price (EUR/MWh)')
ax.set_title('Settlement Volatility: Year-over-Year by Month')
ax.legend()

fig.suptitle('Settlement Price Deep Dive: Monthly Evolution',
             fontsize=14, fontweight='bold', y=0.98)
fig.savefig(OUT_DIR / '03_monthly_evolution.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("[+] Saved 03_monthly_evolution.png")

# Print summary data
print("\n" + "=" * 60)
print("SUMMARY DATA FOR 01_settlement_price")
print("=" * 60)
for yr in sorted(sp['year'].unique()):
    sub = sp[sp['year'] == yr]['imb_settlement_price']
    print(f"\n{yr}: mean={sub.mean():.2f}, std={sub.std():.2f}, "
          f"neg%={((sub < 0).mean()):.1%}, >200={((sub > 200).mean()):.1%}")

print("\n[+] 01_settlement_price complete.")
