"""
Solar Curtailment Optimization - Grid Search
=============================================

Optimizes two curtailment rules for a 1MW E-W solar panel that settles
at imbalance price. Uses real XGBoost predictions from the production model.

Rules:
  Rule 1: Curtail if imbalance_pred > X  (always, regardless of DA)
  Rule 2: Curtail if DA_price < A  AND  imbalance_pred > Y  (Y < X)

Combined: curtail if (pred > X) OR (DA < A AND pred > Y)
Otherwise: produce and receive imb_settlement_price * solar_mwh

Data:
  - Predictions: ImbalanceForcastingProd/data/predictions/predictions_test_v2.csv
  - Market: MarketPriceGap/data/processed/qh_market_prices.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parents[1]

PRED_PATH = ROOT / "ImbalanceForcastingProd" / "data" / "predictions" / "predictions_test_v2.csv"
MKT_PATH = ROOT / "MarketPriceGap" / "data" / "processed" / "qh_market_prices.csv"

plt.rcParams.update({
    'figure.figsize': (14, 7), 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.3
})

# ============================================================
# SOLAR MODEL (PVGIS-calibrated 1MW E-W, same as existing)
# ============================================================
PVGIS = {1:24.9, 2:41.0, 3:78.9, 4:107.8, 5:123.6, 6:131.1,
         7:133.3, 8:114.2, 9:87.0, 10:55.2, 11:27.4, 12:19.2}
DAYS  = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

def _shape(hour, month):
    dl = {1:8.5,2:10,3:11.5,4:13.5,5:15,6:16,7:15.5,8:14,9:12.5,10:11,11:9,12:8}[month]
    sr, ss = 12 - dl/2, 12 + dl/2
    if hour < sr or hour >= ss:
        return 0.0
    sp = (hour - sr) / (ss - sr)
    irr = np.sin(sp * np.pi)
    ef = np.exp(-((sp - 0.30)**2) / (2*0.12**2))
    wf = np.exp(-((sp - 0.70)**2) / (2*0.12**2))
    return 0.5*irr*(0.4 + 0.6*ef) + 0.5*irr*(0.4 + 0.6*wf)

def solar_mwh(hour, month):
    rd = sum(_shape(h, month) for h in range(24))
    if rd <= 0:
        return 0.0
    return _shape(hour, month) * PVGIS[month] / (rd * DAYS[month])


# ============================================================
# LOAD & MERGE DATA
# ============================================================
print("[*] Loading predictions...")
pred = pd.read_csv(PRED_PATH, parse_dates=['datetime'])
pred = pred.rename(columns={'datetime': 'timestamp_qh'})
print(f"    Predictions: {len(pred)} rows, {pred['timestamp_qh'].min()} to {pred['timestamp_qh'].max()}")

print("[*] Loading market data...")
mkt = pd.read_csv(MKT_PATH, parse_dates=['timestamp_qh'])
print(f"    Market: {len(mkt)} rows, {mkt['timestamp_qh'].min()} to {mkt['timestamp_qh'].max()}")

# Merge on quarter-hour timestamp
df = pred.merge(mkt[['timestamp_qh', 'da_price', 'imb_settlement_price', 'imbalance_mwh']],
                on='timestamp_qh', how='inner')
print(f"    Merged: {len(df)} rows, {df['timestamp_qh'].min()} to {df['timestamp_qh'].max()}")

# Add solar production
df['month'] = df['timestamp_qh'].dt.month
df['hour'] = df['timestamp_qh'].dt.hour
df['solar_mwh'] = df.apply(lambda r: solar_mwh(r['hour'], r['month']), axis=1)

# Filter to solar hours only
df = df[df['solar_mwh'] > 0.001].copy()
df = df.dropna(subset=['da_price', 'imb_settlement_price', 'pred_median'])

total_solar = df['solar_mwh'].sum()
n_periods = len(df)
print(f"    Solar periods: {n_periods}, total production: {total_solar:.1f} MWh")
print(f"    Date range: {df['timestamp_qh'].min()} to {df['timestamp_qh'].max()}")
print(f"    Months covered: {sorted(df['month'].unique())}")

# ============================================================
# BASELINE: no curtailment (always produce)
# ============================================================
baseline_rev = (df['imb_settlement_price'] * df['solar_mwh']).sum()
baseline_per_mwh = baseline_rev / total_solar
print(f"\n[*] Baseline (always produce): {baseline_rev:.0f} EUR total, {baseline_per_mwh:.1f} EUR/MWh")

# Current rule: curtail if pred > 15 MWh
current_mask = df['pred_median'] > 15
current_rev = (df.loc[~current_mask, 'imb_settlement_price'] * df.loc[~current_mask, 'solar_mwh']).sum()
current_per_mwh = current_rev / total_solar
n_curtailed_current = current_mask.sum()
print(f"[*] Current rule (pred>15): {current_rev:.0f} EUR, {current_per_mwh:.1f} EUR/MWh, "
      f"curtailed {n_curtailed_current}/{n_periods} periods ({n_curtailed_current/n_periods*100:.1f}%)")


# ============================================================
# GRID SEARCH
# ============================================================
print("\n[*] Running grid search...")

# Precompute arrays for speed
pred_vals = df['pred_median'].values
da_vals = df['da_price'].values
settle_vals = df['imb_settlement_price'].values
sol_vals = df['solar_mwh'].values

# Rule 1: curtail if pred > X  (sweep X from 5 to 60 MWh, step 1)
# Rule 2: curtail if DA < A AND pred > Y  (A: 0-80 EUR step 5, Y: -10 to X step 2)
# Combined: curtail = (pred > X) | (DA < A & pred > Y)

rule1_grid = np.arange(5, 61, 2)       # X: imbalance threshold (always)
da_grid = np.arange(0, 85, 5)          # A: DA price threshold
rule2_grid = np.arange(-10, 51, 2)     # Y: imbalance threshold when DA is low

results = []

for X in rule1_grid:
    # Rule 1 only (no DA condition)
    curtail_r1 = pred_vals > X
    produce_r1 = ~curtail_r1
    rev_r1 = (settle_vals[produce_r1] * sol_vals[produce_r1]).sum()

    results.append({
        'X': X, 'A': np.nan, 'Y': np.nan,
        'rule': 'rule1_only',
        'total_rev': rev_r1,
        'rev_per_mwh': rev_r1 / total_solar,
        'n_curtailed': curtail_r1.sum(),
        'pct_curtailed': curtail_r1.mean() * 100,
        'avoided_loss': (settle_vals[curtail_r1] * sol_vals[curtail_r1]).sum(),
    })

    # Rule 1 + Rule 2 combined
    for A in da_grid:
        for Y in rule2_grid:
            if Y >= X:
                continue  # Rule 2 threshold must be below Rule 1

            curtail = (pred_vals > X) | ((da_vals < A) & (pred_vals > Y))
            produce = ~curtail

            rev = (settle_vals[produce] * sol_vals[produce]).sum()
            n_curt = curtail.sum()

            results.append({
                'X': X, 'A': A, 'Y': Y,
                'rule': 'combined',
                'total_rev': rev,
                'rev_per_mwh': rev / total_solar,
                'n_curtailed': n_curt,
                'pct_curtailed': n_curt / n_periods * 100,
                'avoided_loss': (settle_vals[curtail] * sol_vals[curtail]).sum(),
            })

grid = pd.DataFrame(results)
print(f"[+] Evaluated {len(grid)} combinations")

# Best rule1-only
best_r1 = grid[grid['rule'] == 'rule1_only'].sort_values('rev_per_mwh', ascending=False).iloc[0]
# Best combined
best_comb = grid[grid['rule'] == 'combined'].sort_values('rev_per_mwh', ascending=False).iloc[0]
# Overall best
best = grid.sort_values('rev_per_mwh', ascending=False).iloc[0]

print(f"\n--- RESULTS ---")
print(f"  Baseline (no curtailment):    {baseline_per_mwh:.1f} EUR/MWh")
print(f"  Current rule (pred>15):       {current_per_mwh:.1f} EUR/MWh")
print(f"\n  Best Rule 1 only:")
print(f"    X = {best_r1['X']:.0f} MWh (curtail if pred > {best_r1['X']:.0f})")
print(f"    Revenue: {best_r1['rev_per_mwh']:.1f} EUR/MWh  ({best_r1['total_rev']:.0f} EUR)")
print(f"    Curtailed: {best_r1['n_curtailed']:.0f} periods ({best_r1['pct_curtailed']:.1f}%)")
print(f"    Uplift vs baseline: {best_r1['rev_per_mwh'] - baseline_per_mwh:+.1f} EUR/MWh")
print(f"    Uplift vs current:  {best_r1['rev_per_mwh'] - current_per_mwh:+.1f} EUR/MWh")

print(f"\n  Best Combined (Rule 1 + Rule 2):")
print(f"    X = {best_comb['X']:.0f} MWh (curtail if pred > {best_comb['X']:.0f})")
print(f"    A = {best_comb['A']:.0f} EUR  (DA threshold)")
print(f"    Y = {best_comb['Y']:.0f} MWh  (curtail if DA < {best_comb['A']:.0f} AND pred > {best_comb['Y']:.0f})")
print(f"    Revenue: {best_comb['rev_per_mwh']:.1f} EUR/MWh  ({best_comb['total_rev']:.0f} EUR)")
print(f"    Curtailed: {best_comb['n_curtailed']:.0f} periods ({best_comb['pct_curtailed']:.1f}%)")
print(f"    Uplift vs baseline: {best_comb['rev_per_mwh'] - baseline_per_mwh:+.1f} EUR/MWh")
print(f"    Uplift vs current:  {best_comb['rev_per_mwh'] - current_per_mwh:+.1f} EUR/MWh")
print(f"    Uplift vs best R1:  {best_comb['rev_per_mwh'] - best_r1['rev_per_mwh']:+.1f} EUR/MWh")


# ============================================================
# FIGURE 1: Rule 1 sweep - revenue vs threshold X
# ============================================================
r1 = grid[grid['rule'] == 'rule1_only'].copy()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.plot(r1['X'], r1['rev_per_mwh'], 'k-o', linewidth=2, markersize=5, label='Rule 1: pred > X')
ax.axhline(baseline_per_mwh, color='gray', ls=':', lw=1.5, label=f'No curtailment = {baseline_per_mwh:.1f}')
ax.axhline(current_per_mwh, color='blue', ls='--', lw=1.5, label=f'Current (pred>15) = {current_per_mwh:.1f}')
ax.scatter([best_r1['X']], [best_r1['rev_per_mwh']], color='red', s=150, zorder=5, edgecolors='k', lw=2)
ax.annotate(f"Best: X={best_r1['X']:.0f}, {best_r1['rev_per_mwh']:.1f} EUR/MWh",
            xy=(best_r1['X'], best_r1['rev_per_mwh']),
            xytext=(best_r1['X']+5, best_r1['rev_per_mwh']-1.5),
            fontsize=10, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
ax.set_xlabel('Imbalance Prediction Threshold X (MWh)', fontsize=12)
ax.set_ylabel('Revenue (EUR/MWh)', fontsize=12)
ax.set_title('Rule 1: Curtail if pred > X', fontsize=13)
ax.legend(fontsize=10)

ax = axes[1]
ax.plot(r1['X'], r1['pct_curtailed'], 'k-o', linewidth=2, markersize=5)
ax.axvline(best_r1['X'], color='red', ls='--', lw=1.5, label=f'Optimal X={best_r1["X"]:.0f}')
ax.axvline(15, color='blue', ls='--', lw=1.5, label='Current X=15')
ax.set_xlabel('Imbalance Prediction Threshold X (MWh)', fontsize=12)
ax.set_ylabel('% Periods Curtailed', fontsize=12)
ax.set_title('Curtailment Rate vs Threshold', fontsize=13)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / '01_rule1_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 01_rule1_sweep.png")


# ============================================================
# FIGURE 2: Heatmap - best combined at optimal X, sweep A vs Y
# ============================================================
comb = grid[grid['rule'] == 'combined'].copy()

# Fix X at the best combined X, show A vs Y heatmap
sub = comb[comb['X'] == best_comb['X']].copy()
pivot = sub.pivot_table(index='Y', columns='A', values='rev_per_mwh')

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ax = axes[0]
im = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap='RdYlGn',
               extent=[pivot.columns.min(), pivot.columns.max(),
                       pivot.index.min(), pivot.index.max()])
ax.set_xlabel('DA Price Threshold A (EUR/MWh)', fontsize=12)
ax.set_ylabel('Imbalance Pred Threshold Y (MWh)', fontsize=12)
ax.set_title(f'Revenue Heatmap at X={best_comb["X"]:.0f} MWh\n(EUR/MWh)', fontsize=13)
ax.plot(best_comb['A'], best_comb['Y'], 'k*', markersize=18)
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 2: fix A at optimal, sweep X vs Y
sub2 = comb[comb['A'] == best_comb['A']].copy()
pivot2 = sub2.pivot_table(index='Y', columns='X', values='rev_per_mwh')

ax = axes[1]
im = ax.imshow(pivot2.values, aspect='auto', origin='lower', cmap='RdYlGn',
               extent=[pivot2.columns.min(), pivot2.columns.max(),
                       pivot2.index.min(), pivot2.index.max()])
ax.set_xlabel('Imbalance Pred Threshold X (MWh) - Rule 1', fontsize=12)
ax.set_ylabel('Imbalance Pred Threshold Y (MWh) - Rule 2', fontsize=12)
ax.set_title(f'Revenue Heatmap at A={best_comb["A"]:.0f} EUR\n(EUR/MWh)', fontsize=13)
ax.plot(best_comb['X'], best_comb['Y'], 'k*', markersize=18)
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig(OUT_DIR / '02_combined_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 02_combined_heatmap.png")


# ============================================================
# FIGURE 3: Comparison bar chart + cumulative P&L
# ============================================================
# Build strategies for comparison
strategies = {
    'No curtailment': np.zeros(n_periods, dtype=bool),
    'Current (pred>15)': pred_vals > 15,
    f'Best R1 (pred>{best_r1["X"]:.0f})': pred_vals > best_r1['X'],
    f'Best Combined': (pred_vals > best_comb['X']) | ((da_vals < best_comb['A']) & (pred_vals > best_comb['Y'])),
    'Oracle (curtail neg price)': settle_vals < 0,
}

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Panel 1: Bar comparison
ax = axes[0]
names = []
revenues = []
for name, mask in strategies.items():
    produce = ~mask
    rev = (settle_vals[produce] * sol_vals[produce]).sum() / total_solar
    names.append(name)
    revenues.append(rev)

colors = ['#95a5a6', '#3498db', '#e67e22', '#e74c3c', '#27ae60']
bars = ax.barh(names, revenues, color=colors, edgecolor='black', linewidth=0.5)
for bar, rev in zip(bars, revenues):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{rev:.1f}', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Revenue (EUR/MWh)', fontsize=12)
ax.set_title('Strategy Comparison', fontsize=13)
ax.set_xlim(0, max(revenues) * 1.15)

# Panel 2: Cumulative P&L over time
ax = axes[1]
df_sorted = df.sort_values('timestamp_qh').copy()
settle_sorted = df_sorted['imb_settlement_price'].values
sol_sorted = df_sorted['solar_mwh'].values
pred_sorted = df_sorted['pred_median'].values
da_sorted = df_sorted['da_price'].values
dates = df_sorted['timestamp_qh'].values

for name, mask_orig in strategies.items():
    # Recompute mask on sorted data
    if name == 'No curtailment':
        mask = np.zeros(n_periods, dtype=bool)
    elif name == 'Current (pred>15)':
        mask = pred_sorted > 15
    elif name.startswith('Best R1'):
        mask = pred_sorted > best_r1['X']
    elif name == 'Best Combined':
        mask = (pred_sorted > best_comb['X']) | ((da_sorted < best_comb['A']) & (pred_sorted > best_comb['Y']))
    else:
        mask = settle_sorted < 0

    period_pnl = np.where(mask, 0, settle_sorted * sol_sorted)
    ax.plot(dates, np.cumsum(period_pnl), label=name, linewidth=1.5)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Revenue (EUR)', fontsize=12)
ax.set_title('Cumulative Revenue Over Time', fontsize=13)
ax.legend(fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig(OUT_DIR / '03_strategy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 03_strategy_comparison.png")


# ============================================================
# FIGURE 4: When does Rule 2 help? Scatter of DA vs pred
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Color by settlement price
scatter = ax.scatter(df['pred_median'], df['da_price'],
                     c=df['imb_settlement_price'], cmap='RdYlGn',
                     s=8, alpha=0.6, vmin=-100, vmax=200)
plt.colorbar(scatter, ax=ax, label='Imbalance Settlement Price (EUR/MWh)')

# Draw rule boundaries
ax.axvline(best_comb['X'], color='red', lw=2, ls='--',
           label=f'Rule 1: pred > {best_comb["X"]:.0f} (always curtail)')
ax.axhline(best_comb['A'], color='orange', lw=2, ls=':',
           label=f'DA < {best_comb["A"]:.0f} EUR zone')
ax.axvline(best_comb['Y'], color='orange', lw=2, ls='--',
           label=f'Rule 2: pred > {best_comb["Y"]:.0f} (when DA < {best_comb["A"]:.0f})')

# Shade curtailment zones
ax.axvspan(best_comb['X'], ax.get_xlim()[1], alpha=0.15, color='red', label='_')
# Rule 2 zone: DA < A and pred > Y and pred <= X
ylim = ax.get_ylim()
ax.fill_between([best_comb['Y'], best_comb['X']], [0, 0], [best_comb['A'], best_comb['A']],
                alpha=0.15, color='orange', label='Rule 2 curtailment zone')

ax.set_xlabel('Predicted Imbalance (MWh)', fontsize=12)
ax.set_ylabel('DA Price (EUR/MWh)', fontsize=12)
ax.set_title('Curtailment Decision Space\n(color = actual settlement price)', fontsize=13)
ax.legend(fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig(OUT_DIR / '04_decision_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 04_decision_space.png")


# ============================================================
# SAVE RESULTS
# ============================================================
# Top 20 combined
top20 = grid[grid['rule'] == 'combined'].nlargest(20, 'rev_per_mwh')
print("\n--- Top 20 Combined Rules ---")
print(f"{'X':>5} | {'A':>5} | {'Y':>5} | {'EUR/MWh':>8} | {'Curt%':>6} | {'Uplift vs Current':>18}")
print("-" * 60)
for _, r in top20.iterrows():
    print(f"{r['X']:>5.0f} | {r['A']:>5.0f} | {r['Y']:>5.0f} | {r['rev_per_mwh']:>8.1f} | "
          f"{r['pct_curtailed']:>5.1f}% | {r['rev_per_mwh'] - current_per_mwh:>+17.1f}")

grid.to_csv(OUT_DIR / 'data' / 'grid_search_full.csv', index=False)
print(f"\n[+] Saved full grid ({len(grid)} rows) to data/grid_search_full.csv")

# Summary
summary = pd.DataFrame([
    {'strategy': 'No curtailment', 'rev_per_mwh': baseline_per_mwh, 'X': None, 'A': None, 'Y': None},
    {'strategy': 'Current (pred>15)', 'rev_per_mwh': current_per_mwh, 'X': 15, 'A': None, 'Y': None},
    {'strategy': 'Best Rule 1', 'rev_per_mwh': best_r1['rev_per_mwh'], 'X': best_r1['X'], 'A': None, 'Y': None},
    {'strategy': 'Best Combined', 'rev_per_mwh': best_comb['rev_per_mwh'],
     'X': best_comb['X'], 'A': best_comb['A'], 'Y': best_comb['Y']},
])
summary.to_csv(OUT_DIR / 'data' / 'summary.csv', index=False)
print("[+] Saved summary to data/summary.csv")

print("\n[+] Done.")
