"""
02_idm_market: IDM Execution Environment Analysis
===================================================
Generates:
  01_idm_mid_vs_settlement.png
  02_bid_ask_spread.png
  03_idm_convergence_path.png
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

# --- Data setup (inline) ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

LEAD = 8
OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

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

ob_path = DATA_DIR / "features" / "orderbook_qh_features.csv"
ob = pd.read_csv(ob_path, parse_dates=['delivery_start'])

ob_120 = ob[ob['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'mid', 'spread']]
ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
ob_120.columns = ['exec_bid', 'exec_ask', 'exec_mid', 'exec_spread']
df_base = df_base.join(ob_120, how='left')
df_base['imb_settlement_price'] = df_base['imb_settle_price']

sp = df_base[df_base['exec_mid'].notna() & df_base['imb_settlement_price'].notna()].copy()
sp['year'] = sp.index.year
sp['hour'] = sp.index.hour
sp['month_label'] = sp.index.to_period('M').astype(str)

print(f"[+] Dataset ready: {len(sp)} rows with exec_mid + settlement price")

# ================================================================
# 01_idm_mid_vs_settlement.png
# ================================================================
print("[*] Creating 01_idm_mid_vs_settlement.png ...")
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# (a) Scatter: IDM mid vs settlement, color by year
ax = fig.add_subplot(gs[0, :])
for yr in sorted(sp['year'].unique()):
    sub = sp[sp['year'] == yr]
    if len(sub) > 5000:
        plot_sub = sub.sample(5000, random_state=42)
    else:
        plot_sub = sub
    ax.scatter(plot_sub['exec_mid'], plot_sub['imb_settlement_price'],
               alpha=0.12, s=6, color=YEAR_COLORS.get(yr, 'gray'), label=f'{yr}')
lims = [-200, 500]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='y=x')
ax.set_xlim(-200, 500)
ax.set_ylim(-300, 600)
ax.set_xlabel('IDM Mid Price at T-120min (EUR/MWh)')
ax.set_ylabel('Settlement Price (EUR/MWh)')
ax.set_title('IDM Mid (T-120min) vs Settlement Price')
ax.legend(fontsize=10)

# (b) R-squared per year
ax = fig.add_subplot(gs[1, 0])
r2_data = []
for yr in sorted(sp['year'].unique()):
    sub = sp[sp['year'] == yr].dropna(subset=['exec_mid', 'imb_settlement_price'])
    sub_clip = sub[(sub['imb_settlement_price'].abs() < 2000) & (sub['exec_mid'].abs() < 1000)]
    if len(sub_clip) > 10:
        r, p = sp_stats.pearsonr(sub_clip['exec_mid'], sub_clip['imb_settlement_price'])
        r2_data.append({'year': yr, 'r': r, 'r2': r ** 2, 'n': len(sub_clip)})
r2_df = pd.DataFrame(r2_data)
colors = [YEAR_COLORS.get(yr, 'gray') for yr in r2_df['year']]
ax.bar(range(len(r2_df)), r2_df['r2'], color=colors, alpha=0.7)
ax.set_xticks(range(len(r2_df)))
ax.set_xticklabels([str(yr) for yr in r2_df['year']])
for i, row in r2_df.iterrows():
    ax.text(i, row['r2'] + 0.01, f"r={row['r']:.3f}\nR2={row['r2']:.3f}",
            ha='center', fontsize=9)
ax.set_ylabel('R-squared')
ax.set_title('Correlation Strength: IDM Mid vs Settlement by Year')

# (c) Monthly R-squared evolution
ax = fig.add_subplot(gs[1, 1])
monthly_r = []
for month_label, grp in sp.groupby('month_label'):
    grp_clean = grp[(grp['imb_settlement_price'].abs() < 2000) & (grp['exec_mid'].abs() < 1000)]
    if len(grp_clean) > 20:
        r, _ = sp_stats.pearsonr(grp_clean['exec_mid'], grp_clean['imb_settlement_price'])
        monthly_r.append({'month': month_label, 'r': r, 'r2': r ** 2})
mr_df = pd.DataFrame(monthly_r)
ax.plot(range(len(mr_df)), mr_df['r2'], 'b.-', lw=2)
ax.set_xticks(range(len(mr_df)))
ax.set_xticklabels(mr_df['month'], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('R-squared')
ax.set_title('Monthly R-squared: IDM Mid vs Settlement')
for i, label in enumerate(mr_df['month']):
    if label.endswith('-01'):
        ax.axvline(i, color='gray', ls='--', alpha=0.4)

fig.suptitle('IDM Market: Mid Price vs Settlement',
             fontsize=14, fontweight='bold', y=0.98)
fig.savefig(OUT_DIR / '01_idm_mid_vs_settlement.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("[+] Saved 01_idm_mid_vs_settlement.png")

for _, row in r2_df.iterrows():
    print(f"  {int(row['year'])}: r={row['r']:.3f}, R2={row['r2']:.3f}")

# ================================================================
# 02_bid_ask_spread.png
# ================================================================
print("[*] Creating 02_bid_ask_spread.png ...")
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

ba = sp[sp['exec_spread'].notna()].copy()

# (a) Monthly median bid-ask spread
ax = fig.add_subplot(gs[0, :])
monthly_ba = ba.groupby('month_label')['exec_spread'].agg(['median', 'mean', 'count'])
x = range(len(monthly_ba))
ax.bar(x, monthly_ba['median'], color='steelblue', alpha=0.7, label='Median BA Spread')
ax.plot(x, monthly_ba['mean'], 'r.-', lw=1.5, label='Mean BA Spread')
ax.set_xticks(x)
ax.set_xticklabels(monthly_ba.index, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Bid-Ask Spread (EUR/MWh)')
ax.set_title('Monthly Bid-Ask Spread Evolution (T-120min)')
ax.legend()
for i, label in enumerate(monthly_ba.index):
    if label.endswith('-01'):
        ax.axvline(i, color='gray', ls='--', alpha=0.4)

# (b) Fraction liquid
ax = fig.add_subplot(gs[1, 0])
monthly_liq = ba.groupby('month_label').apply(
    lambda x: pd.Series({
        'BA<5': (x['exec_spread'] < 5).mean(),
        'BA<10': (x['exec_spread'] < 10).mean(),
        'BA<15': (x['exec_spread'] < 15).mean(),
    })
)
x = range(len(monthly_liq))
for col, color in [('BA<5', 'green'), ('BA<10', 'blue'), ('BA<15', 'orange')]:
    ax.plot(x, monthly_liq[col] * 100, '-', color=color, lw=2, label=col, marker='.')
ax.set_xticks(x)
ax.set_xticklabels(monthly_liq.index, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('% of Periods')
ax.set_title('Liquidity: Fraction of Periods with Tight Spreads')
ax.legend()

# (c) Bid-ask by hour
ax = fig.add_subplot(gs[1, 1])
for yr in sorted(ba['year'].unique()):
    sub = ba[ba['year'] == yr]
    hourly = sub.groupby('hour')['exec_spread'].median()
    ax.plot(hourly.index, hourly.values, '.-', color=YEAR_COLORS.get(yr, 'gray'),
            label=f'{yr}', lw=2)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Median Bid-Ask Spread (EUR/MWh)')
ax.set_title('Bid-Ask Spread by Hour of Day')
ax.legend()
ax.set_xticks(range(24))

fig.suptitle('IDM Market: Bid-Ask Spread Analysis',
             fontsize=14, fontweight='bold', y=0.98)
fig.savefig(OUT_DIR / '02_bid_ask_spread.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("[+] Saved 02_bid_ask_spread.png")

# ================================================================
# 03_idm_convergence_path.png
# ================================================================
print("[*] Creating 03_idm_convergence_path.png ...")

leads = sorted(ob['lead_minutes'].unique())
imb_settle = df_base[['imb_settlement_price']].dropna()

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# Build convergence data
convergence_data = []
for lead_min in leads:
    ob_lead = ob[ob['lead_minutes'] == lead_min].set_index('delivery_start')[['mid']]
    ob_lead = ob_lead[~ob_lead.index.duplicated(keep='last')]
    merged = ob_lead.join(imb_settle, how='inner')
    merged['abs_diff'] = (merged['imb_settlement_price'] - merged['mid']).abs()
    merged['year'] = merged.index.year
    for yr in sorted(merged['year'].unique()):
        sub = merged[merged['year'] == yr]
        if len(sub) > 50:
            convergence_data.append({
                'lead_minutes': lead_min, 'year': yr,
                'mean_abs_diff': sub['abs_diff'].mean(),
                'median_abs_diff': sub['abs_diff'].median(),
                'n': len(sub),
            })

conv_df = pd.DataFrame(convergence_data)

# (a) Mean |settle - mid| by lead
ax = fig.add_subplot(gs[0, :])
for yr in sorted(conv_df['year'].unique()):
    sub = conv_df[conv_df['year'] == yr].sort_values('lead_minutes')
    ax.plot(sub['lead_minutes'], sub['mean_abs_diff'], '.-',
            color=YEAR_COLORS.get(yr, 'gray'), label=f'{yr}', lw=2, markersize=10)
ax.set_xlabel('Lead Time (minutes before delivery)')
ax.set_ylabel('Mean |Settlement - IDM Mid| (EUR/MWh)')
ax.set_title('IDM Price Convergence Path: How Close is IDM Mid to Settlement?')
ax.legend()
ax.invert_xaxis()

# (b) Median convergence
ax = fig.add_subplot(gs[1, 0])
for yr in sorted(conv_df['year'].unique()):
    sub = conv_df[conv_df['year'] == yr].sort_values('lead_minutes')
    ax.plot(sub['lead_minutes'], sub['median_abs_diff'], '.-',
            color=YEAR_COLORS.get(yr, 'gray'), label=f'{yr}', lw=2, markersize=10)
ax.set_xlabel('Lead Time (minutes before delivery)')
ax.set_ylabel('Median |Settlement - IDM Mid| (EUR/MWh)')
ax.set_title('Median Convergence Path')
ax.legend()
ax.invert_xaxis()

# (c) Convergence improvement relative to T-120
ax = fig.add_subplot(gs[1, 1])
for yr in sorted(conv_df['year'].unique()):
    sub = conv_df[conv_df['year'] == yr].sort_values('lead_minutes')
    base = sub[sub['lead_minutes'] == 120]['mean_abs_diff'].values
    if len(base) > 0:
        sub = sub.copy()
        sub['improvement'] = (base[0] - sub['mean_abs_diff']) / base[0] * 100
        ax.plot(sub['lead_minutes'], sub['improvement'], '.-',
                color=YEAR_COLORS.get(yr, 'gray'), label=f'{yr}', lw=2, markersize=10)
ax.set_xlabel('Lead Time (minutes before delivery)')
ax.set_ylabel('Improvement vs T-120 (%)')
ax.set_title('Relative Convergence: % Closer to Settlement vs T-120')
ax.legend()
ax.invert_xaxis()
ax.axhline(0, color='gray', ls='--', alpha=0.5)

fig.suptitle('IDM Market: Price Convergence Path',
             fontsize=14, fontweight='bold', y=0.98)
fig.savefig(OUT_DIR / '03_idm_convergence_path.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("[+] Saved 03_idm_convergence_path.png")

# Summary data
print("\n" + "=" * 60)
print("SUMMARY DATA FOR 02_idm_market")
print("=" * 60)
print("\nConvergence (mean |settle - mid|):")
for _, row in conv_df.iterrows():
    print(f"  Lead={int(row['lead_minutes'])}min, {int(row['year'])}: "
          f"mean={row['mean_abs_diff']:.2f}, median={row['median_abs_diff']:.2f}")

ba_overall = ba['exec_spread'].describe()
print(f"\nBid-Ask spread (T-120): median={ba_overall['50%']:.2f}, mean={ba_overall['mean']:.2f}")

print("\n[+] 02_idm_market complete.")
