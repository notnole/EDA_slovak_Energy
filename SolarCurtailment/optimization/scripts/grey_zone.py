"""Grey zone analysis: periods that PRODUCE despite low DA, after all rules applied."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
mkt = pd.read_csv(ROOT / 'MarketPriceGap/data/processed/qh_market_prices.csv', parse_dates=['timestamp_qh'])

df = mkt[(mkt['timestamp_qh'].dt.year == 2025) & (mkt['timestamp_qh'].dt.month.between(3, 8))].copy()
df = df.dropna(subset=['da_price', 'imb_settlement_price', 'imbalance_mwh'])

np.random.seed(42)
df['pred'] = df['imbalance_mwh'] + np.random.normal(0, 9.0, len(df))

# Rules: X=34, A=10, Y=5, F=-3 (user's floor)
F = -3
X = 34
A = 10
Y = 5

# What each rule curtails
r1 = df['pred'] > X
r2_raw = (df['da_price'] < A) & (df['pred'] > Y) & ~r1
floor_save = (r1 | r2_raw) & (df['pred'] <= F)
curtail = (r1 | r2_raw) & (df['pred'] > F)

print(f"Rule 1 curtails:  {r1.sum()} periods (pred > {X})")
print(f"Rule 2 curtails:  {r2_raw.sum()} periods (DA < {A} AND pred > {Y})")
print(f"Floor saves:      {floor_save.sum()} periods (pred <= {F})")
print(f"Total curtailed:  {curtail.sum()} periods")
print()

# Grey zone: DA < 20, pred between F and Y -- these PRODUCE
grey = (df['da_price'] < 20) & (df['pred'] > F) & (df['pred'] <= Y)
grey_df = df[grey]

print(f"Grey zone (DA<20, pred {F} to {Y}): {grey.sum()} periods")
print(f"  Avg settlement:  {grey_df['imb_settlement_price'].mean():.1f} EUR/MWh")
print(f"  Median:          {grey_df['imb_settlement_price'].median():.1f} EUR/MWh")
print(f"  Pct negative:    {(grey_df['imb_settlement_price'] < 0).mean()*100:.1f}%")

plt.rcParams.update({'figure.figsize': (14, 7), 'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# --- Panel 1: Scatter pred vs settlement, grey zone highlighted ---
ax = axes[0, 0]
bg = df[(df['da_price'] < 30) & (df['pred'] > -20) & (df['pred'] < 50)]
ax.scatter(bg['pred'], bg['imb_settlement_price'], c='#eeeeee', s=8, alpha=0.3, zorder=1)

# R2 curtailed periods (for context)
r2_curtailed = df[curtail & ~r1]
ax.scatter(r2_curtailed['pred'], r2_curtailed['imb_settlement_price'],
           c='#e74c3c', s=15, alpha=0.5, label=f'R2 curtailed ({len(r2_curtailed)})', zorder=2, marker='x')

# Grey zone
sc = ax.scatter(grey_df['pred'], grey_df['imb_settlement_price'],
                c=grey_df['da_price'], cmap='coolwarm_r', s=25, alpha=0.7,
                vmin=-10, vmax=20, edgecolors='k', linewidth=0.3, zorder=3)
plt.colorbar(sc, ax=ax, label='DA Price (EUR/MWh)', shrink=0.8)

ax.axhline(0, color='black', lw=1.5)
ax.axvline(Y, color='red', lw=2, ls='--', label=f'Y={Y} (R2 threshold)')
ax.axvline(F, color='green', lw=2, ls='--', label=f'F={F} (Floor)')
ax.axvspan(F, Y, alpha=0.08, color='yellow', label=f'Grey zone (pred {F} to {Y})')
ax.set_xlabel('Predicted Imbalance (MWh)')
ax.set_ylabel('Actual Settlement Price (EUR/MWh)')
ax.set_title(f'Grey Zone After Rules: DA<20, pred {F} to {Y}\nThese periods PRODUCE (not curtailed)')
ax.legend(fontsize=9)
ax.set_xlim(-15, 45)
ax.set_ylim(-250, 250)

# --- Panel 2: Distribution of settlement prices in grey zone ---
ax = axes[0, 1]
prices = grey_df['imb_settlement_price']
bins = np.arange(-20, 21, 1)
pos_p = prices[prices >= 0]
neg_p = prices[prices < 0]
clipped = prices[(prices >= -20) & (prices <= 20)]
neg_clip = clipped[clipped < 0]
pos_clip = clipped[clipped >= 0]
n_below = (prices < -20).sum()
n_above = (prices > 20).sum()
ax.hist(neg_clip, bins=bins, color='#e74c3c', alpha=0.7,
        label=f'Negative ({len(neg_p)}, {len(neg_p)/len(prices)*100:.0f}%)')
ax.hist(pos_clip, bins=bins, color='#27ae60', alpha=0.7,
        label=f'Positive ({len(pos_p)}, {len(pos_p)/len(prices)*100:.0f}%)')
ax.axvline(prices.mean(), color='black', lw=2, ls='--', label=f'Mean = {prices.mean():.1f} EUR/MWh')
ax.axvline(0, color='black', lw=1)
if n_below > 0:
    ax.annotate(f'{n_below} periods\n< -20', xy=(-19, ax.get_ylim()[1]*0.8),
                fontsize=9, color='#e74c3c', fontweight='bold')
if n_above > 0:
    ax.annotate(f'{n_above} periods\n> 20', xy=(14, ax.get_ylim()[1]*0.8),
                fontsize=9, color='#27ae60', fontweight='bold')
ax.set_xlim(-21, 21)
ax.set_xlabel('Settlement Price (EUR/MWh)')
ax.set_ylabel('Count')
ax.set_title(f'Settlement Price Distribution (zoomed -20 to +20)\n'
             f'{len(prices)} periods, mean={prices.mean():.1f}, median={prices.median():.1f}')
ax.legend(fontsize=10)

# --- Panel 3: Avg settlement by pred bin within grey zone ---
ax = axes[1, 0]
gz = grey_df.copy()
pred_bins = np.arange(F, Y + 1, 1)
gz['pred_bin'] = pd.cut(gz['pred'], bins=pred_bins)
bin_stats = gz.groupby('pred_bin', observed=True).agg(
    avg_price=('imb_settlement_price', 'mean'),
    pct_neg=('imb_settlement_price', lambda x: (x < 0).mean() * 100),
    n=('imb_settlement_price', 'count')
).reset_index()
bin_stats['mid'] = bin_stats['pred_bin'].apply(lambda x: x.mid)

colors_bar = ['#27ae60' if v >= 0 else '#e74c3c' for v in bin_stats['avg_price']]
ax.bar(bin_stats['mid'], bin_stats['avg_price'], width=0.8, color=colors_bar,
       edgecolor='black', linewidth=0.5, alpha=0.8)
ax.axhline(0, color='black', lw=1.5)
for _, row in bin_stats.iterrows():
    offset = 3 if row['avg_price'] >= 0 else -8
    ax.text(row['mid'], row['avg_price'] + offset,
            f"n={row['n']:.0f}\n{row['pct_neg']:.0f}%neg", ha='center', fontsize=8)
ax.set_xlabel('Predicted Imbalance Bin (MWh)')
ax.set_ylabel('Avg Settlement Price (EUR/MWh)')
ax.set_title(f'Avg Settlement by Prediction Level\nGrey Zone (pred {F} to {Y}, DA<20)')

# --- Panel 4: Avg settlement by DA price bin within grey zone ---
ax = axes[1, 1]
gz['da_bin'] = pd.cut(gz['da_price'], bins=[-50, -5, 0, 5, 10, 15, 20])
da_stats = gz.groupby('da_bin', observed=True).agg(
    avg_price=('imb_settlement_price', 'mean'),
    pct_neg=('imb_settlement_price', lambda x: (x < 0).mean() * 100),
    n=('imb_settlement_price', 'count')
).reset_index()
da_stats['label'] = da_stats['da_bin'].astype(str)

colors_da = ['#27ae60' if v >= 0 else '#e74c3c' for v in da_stats['avg_price']]
bars = ax.bar(range(len(da_stats)), da_stats['avg_price'], color=colors_da,
              edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_xticks(range(len(da_stats)))
ax.set_xticklabels(da_stats['label'], fontsize=9)
ax.axhline(0, color='black', lw=1.5)
for i, (_, row) in enumerate(da_stats.iterrows()):
    offset = 3 if row['avg_price'] >= 0 else -8
    ax.text(i, row['avg_price'] + offset,
            f"n={row['n']:.0f}\n{row['pct_neg']:.0f}%neg", ha='center', fontsize=8)
ax.set_xlabel('DA Price Bin (EUR/MWh)')
ax.set_ylabel('Avg Settlement Price (EUR/MWh)')
ax.set_title(f'Avg Settlement by DA Price Level\nGrey Zone (pred {F} to {Y})')

plt.suptitle(f'Grey Zone After Rules: pred {F} to {Y}, DA < 20 -- Periods that PRODUCE',
             fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(OUT / '05_grey_zone_after_rules.png', dpi=150, bbox_inches='tight')
plt.close()
print('[+] 05_grey_zone_after_rules.png')
