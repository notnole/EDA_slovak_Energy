"""
Solar DA Commit + No-Produce Strategy v2
=========================================

Two-step analysis:
  1. Find the imbalance volume (MWh) at which settlement prices turn negative
     on average. This becomes the produce/no-produce threshold.
  2. Sweep DA bid thresholds 0-80 EUR (step 5) and find the optimal.

Strategy per hour:
  2a) Sold on DA:
        imbalance_mwh < V* -> PRODUCE   -> revenue = da_price
        imbalance_mwh >= V* -> NO-PRODUCE -> revenue = da_price - imb_settle
  2b) Not sold on DA:
        imbalance_mwh < V* -> PRODUCE   -> revenue = imb_settle
        imbalance_mwh >= V* -> NO-PRODUCE -> revenue = 0

Uses actual imbalance_mwh as the signal (no forecasting model yet).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR  = Path(__file__).resolve().parent
DATA_PATH = OUT_DIR.parent.parent / "data" / "processed" / "hourly_market_prices.csv"

plt.rcParams.update({'figure.figsize': (12, 6), 'font.size': 11,
                     'axes.grid': True, 'grid.alpha': 0.3})

# ============================================================
# SOLAR MODEL (same as before)
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
# LOAD DATA
# ============================================================
print("[*] Loading data...")
mkt = pd.read_csv(DATA_PATH, parse_dates=['timestamp_hour'])

sol = mkt[
    (mkt['timestamp_hour'].dt.year == 2025) &
    (mkt['month'].between(3, 8))
].copy()
sol = sol.dropna(subset=['da_price', 'imb_settlement_price', 'imbalance_mwh'])
sol['solar_mwh'] = sol.apply(lambda r: solar_mwh(r['hour'], r['month']), axis=1)
sol = sol[sol['solar_mwh'] > 0.001].copy()
tv  = sol['solar_mwh'].sum()
print(f"[+] {len(sol)} solar hours, {tv:.0f} MWh total")

# ============================================================
# STEP 1: Find the imbalance volume threshold where
# average settlement price crosses zero
# ============================================================
print("[*] Step 1: Finding volume threshold where avg price turns negative...")

# Bin by imbalance_mwh and compute mean settlement price per bin
bins = np.arange(-800, 801, 50)
sol['vol_bin'] = pd.cut(sol['imbalance_mwh'], bins=bins)
bin_stats = sol.groupby('vol_bin', observed=True).agg(
    avg_price=('imb_settlement_price', 'mean'),
    n_hours=('imb_settlement_price', 'count'),
    pct_negative=('imb_settlement_price', lambda x: (x < 0).mean() * 100),
).reset_index()
bin_stats['bin_mid'] = bin_stats['vol_bin'].apply(lambda x: x.mid)
bin_stats = bin_stats.dropna(subset=['bin_mid'])

# Find crossing point: first bin mid where avg_price < 0
crossing = bin_stats[bin_stats['avg_price'] < 0]
if len(crossing) > 0:
    vol_threshold = crossing['bin_mid'].iloc[0]
else:
    vol_threshold = bin_stats.loc[bin_stats['avg_price'].idxmin(), 'bin_mid']

print(f"[+] Volume threshold (avg price < 0): {vol_threshold:.0f} MWh")
print(f"    Hours above threshold: {(sol['imbalance_mwh'] >= vol_threshold).sum()} "
      f"({(sol['imbalance_mwh'] >= vol_threshold).mean()*100:.1f}%)")
print(f"    Avg settlement price above threshold: "
      f"{sol.loc[sol['imbalance_mwh'] >= vol_threshold, 'imb_settlement_price'].mean():.1f} EUR/MWh")
print(f"    Avg settlement price below threshold: "
      f"{sol.loc[sol['imbalance_mwh'] < vol_threshold, 'imb_settlement_price'].mean():.1f} EUR/MWh")

# ============================================================
# FIGURE 1: Avg settlement price vs imbalance volume
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
has_data = bin_stats['n_hours'] >= 5
neg_mask = bin_stats['avg_price'] < 0
ax.bar(bin_stats.loc[has_data & ~neg_mask, 'bin_mid'],
       bin_stats.loc[has_data & ~neg_mask, 'avg_price'],
       width=45, color='#27ae60', alpha=0.8, label='Avg price > 0 (produce)')
ax.bar(bin_stats.loc[has_data & neg_mask, 'bin_mid'],
       bin_stats.loc[has_data & neg_mask, 'avg_price'],
       width=45, color='#e74c3c', alpha=0.8, label='Avg price < 0 (no-produce)')
ax.axhline(0, color='black', linewidth=1.5)
ax.axvline(vol_threshold, color='black', linewidth=2, linestyle='--',
           label=f'Threshold = {vol_threshold:.0f} MWh')
ax.set_xlabel('System Imbalance Volume (MWh, + = surplus)', fontsize=12)
ax.set_ylabel('Avg Settlement Price (EUR/MWh)', fontsize=12)
ax.set_title('Avg Imbalance Settlement Price by System Volume\n2025 Solar Hours (Mar-Aug)', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-800, 800)

ax = axes[1]
ax.bar(bin_stats.loc[has_data, 'bin_mid'],
       bin_stats.loc[has_data, 'pct_negative'],
       width=45, color='#2980b9', alpha=0.8)
ax.axvline(vol_threshold, color='black', linewidth=2, linestyle='--',
           label=f'Threshold = {vol_threshold:.0f} MWh')
ax.axhline(50, color='red', linewidth=1, linestyle=':', label='50% negative')
ax.set_xlabel('System Imbalance Volume (MWh)', fontsize=12)
ax.set_ylabel('% Hours with Negative Settlement Price', fontsize=12)
ax.set_title('Probability of Negative Price by Volume Bin\n2025 Solar Hours', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-800, 800)

plt.tight_layout()
plt.savefig(OUT_DIR / 'v2_01_volume_threshold.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] v2_01_volume_threshold.png")

# ============================================================
# STEP 2: Simulate strategy for each DA bid threshold
# ============================================================
print(f"\n[*] Step 2: Sweeping DA bid thresholds 0-80 EUR (step 5)...")
print(f"    Produce/no-produce threshold: imbalance_mwh >= {vol_threshold:.0f} MWh")

surplus = sol['imbalance_mwh'] >= vol_threshold   # True = no-produce signal

rows = []
for bid in range(0, 81, 5):
    sold_da = sol['da_price'] >= bid

    # 2a: sold on DA
    da_produce   = sold_da & ~surplus   # produce: deficit signal
    da_noproduce = sold_da &  surplus   # no-produce: surplus signal

    # 2b: not sold on DA
    imb_produce  = ~sold_da & ~surplus  # produce at imbalance: deficit signal
    # ~sold_da & surplus -> no-produce, revenue = 0

    rev_da_prod   = (sol.loc[da_produce,   'da_price']           * sol.loc[da_produce,   'solar_mwh']).sum()
    rev_da_noprod = ((sol.loc[da_noproduce, 'da_price'] - sol.loc[da_noproduce, 'imb_settlement_price'])
                     * sol.loc[da_noproduce, 'solar_mwh']).sum()
    rev_imb_prod  = (sol.loc[imb_produce,  'imb_settlement_price'] * sol.loc[imb_produce,  'solar_mwh']).sum()

    total_rev = rev_da_prod + rev_da_noprod + rev_imb_prod

    rows.append({
        'bid':           bid,
        'total_rev':     total_rev,
        'rev_per_mwh':   total_rev / tv,
        'rev_da_prod':   rev_da_prod,
        'rev_da_noprod': rev_da_noprod,
        'rev_imb_prod':  rev_imb_prod,
        'h_da_prod':     da_produce.sum(),
        'h_da_noprod':   da_noproduce.sum(),
        'h_imb_prod':    imb_produce.sum(),
        'h_curtailed':   (~sold_da & surplus).sum(),
    })

results = pd.DataFrame(rows)
opt = results.loc[results['rev_per_mwh'].idxmax()]

# Reference benchmarks
pure_da_rev  = (sol['da_price'] * sol['solar_mwh']).sum() / tv
old_best_rev = 62.2  # from previous analysis

print("\n--- Results by DA bid threshold ---")
print(f"{'Bid':>5} | {'EUR/MWh':>8} | {'DA+prod':>8} | {'DA+noprod':>10} | {'Imb+prod':>9} | {'Curtail':>8}")
print("-" * 65)
for _, r in results.iterrows():
    marker = " <-- OPTIMAL" if r['bid'] == opt['bid'] else ""
    print(f"{r['bid']:>5.0f} | {r['rev_per_mwh']:>8.1f} | {r['h_da_prod']:>8.0f} | "
          f"{r['h_da_noprod']:>10.0f} | {r['h_imb_prod']:>9.0f} | {r['h_curtailed']:>8.0f}{marker}")

print(f"\n  Optimal bid:       {opt['bid']:.0f} EUR/MWh")
print(f"  Optimal revenue:   {opt['rev_per_mwh']:.1f} EUR/MWh")
print(f"  Pure DA (bid=0):   {pure_da_rev:.1f} EUR/MWh")
print(f"  Old best (bid=58): {old_best_rev:.1f} EUR/MWh")
print(f"  Uplift vs pure DA: +{opt['rev_per_mwh'] - pure_da_rev:.1f} EUR/MWh")
print(f"  Uplift vs old:     +{opt['rev_per_mwh'] - old_best_rev:.1f} EUR/MWh")

# ============================================================
# FIGURE 2: Revenue vs DA bid threshold + hour breakdown
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.plot(results['bid'], results['rev_per_mwh'], 'k-o', linewidth=2.5,
        markersize=8, label='New strategy (DA commit + no-produce)')
ax.axhline(pure_da_rev, color='#95a5a6', linewidth=1.5, linestyle=':',
           label=f'Pure DA = {pure_da_rev:.1f} EUR/MWh')
ax.axhline(old_best_rev, color='#8e44ad', linewidth=1.5, linestyle='--',
           label=f'Old best (curtail) = {old_best_rev:.1f} EUR/MWh')

# Mark optimum
ax.scatter([opt['bid']], [opt['rev_per_mwh']], color='#e74c3c', s=180,
           zorder=5, edgecolors='black', linewidth=2)
ax.annotate(f"Optimal: {opt['rev_per_mwh']:.1f} EUR/MWh\nbid = {opt['bid']:.0f} EUR",
            xy=(opt['bid'], opt['rev_per_mwh']),
            xytext=(opt['bid'] + 12, opt['rev_per_mwh'] - 2),
            fontsize=11, fontweight='bold', color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

ax.set_xlabel('DA Bid Threshold (EUR/MWh)', fontsize=12)
ax.set_ylabel('Revenue per MWh (EUR/MWh)', fontsize=12)
ax.set_title(f'Revenue vs DA Bid Threshold\n(produce threshold = {vol_threshold:.0f} MWh surplus)', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-2, 82)

# Panel 2: Hour breakdown stacked bar at each bid
ax = axes[1]
x = results['bid'].values
w = 3.5
ax.bar(x, results['h_da_prod'],   width=w, label='DA + produce',    color='#27ae60', alpha=0.85)
ax.bar(x, results['h_da_noprod'], width=w, bottom=results['h_da_prod'],
       label='DA + no-produce', color='#2ecc71', alpha=0.85)
ax.bar(x, results['h_imb_prod'],  width=w,
       bottom=results['h_da_prod'] + results['h_da_noprod'],
       label='Imb + produce',   color='#2980b9', alpha=0.85)
ax.bar(x, results['h_curtailed'], width=w,
       bottom=results['h_da_prod'] + results['h_da_noprod'] + results['h_imb_prod'],
       label='Curtailed (no DA, surplus)', color='#e74c3c', alpha=0.7)

ax.axvline(opt['bid'], color='black', linewidth=2, linestyle='--', label=f"Optimal bid = {opt['bid']:.0f}")
ax.set_xlabel('DA Bid Threshold (EUR/MWh)', fontsize=12)
ax.set_ylabel('Number of Hours', fontsize=12)
ax.set_title('Hour Breakdown by Category\nat Each Bid Threshold', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(-2, 82)

plt.tight_layout()
plt.savefig(OUT_DIR / 'v2_02_bid_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] v2_02_bid_sweep.png")

results.to_csv(OUT_DIR / 'v2_bid_sweep_results.csv', index=False)
print("[+] v2_bid_sweep_results.csv")


# ============================================================
# STEP 3: Compare with vol_curtail top-50% for non-DA hours
# ============================================================
# This isolates whether the gap vs old strategy comes from the
# curtailment approach (125 MWh vs vol_curtail) or from
# the DA no-produce feature.

def vol_curtail(df, pct):
    """Old approach: curtail during the largest-surplus hours,
    up to `pct` fraction of total volume (by solar_mwh).
    Only curtails surplus hours (imbalance_mwh > 0)."""
    if len(df) == 0 or pct <= 0:
        return pd.Series(False, index=df.index)
    budget = df['solar_mwh'].sum() * pct
    s = df.sort_values('imbalance_mwh', ascending=False)
    s['is_sur'] = s['imbalance_mwh'] > 0
    s['cv'] = (s['solar_mwh'] * s['is_sur']).cumsum()
    s['c'] = s['is_sur'] & (s['cv'] <= budget)
    return s['c'].reindex(df.index)


# ============================================================
# STEP 3: 3D grid search
#   P1: DA bid threshold (0-80, step 5)
#   P2: vol threshold for case 2b - curtail if NOT sold on DA
#       (imbalance_mwh >= V2 -> curtail, revenue=0)
#   P3: vol threshold for case 2a - no-produce if SOLD on DA
#       (imbalance_mwh >= V3 -> no-produce, revenue=da-imb)
# ============================================================
print("\n[*] Step 3: 3D grid search...")

bid_grid  = list(range(0, 81, 5))         # 17 values
vol_grid  = list(range(-100, 501, 25))     # 25 values
# total: 17 x 25 x 25 = 10,625 combos

imb_vals = sol['imbalance_mwh'].values
da_vals  = sol['da_price'].values
imb_p    = sol['imb_settlement_price'].values
sol_vals = sol['solar_mwh'].values

rows_grid = []
for bid in bid_grid:
    sold_da = da_vals >= bid

    for v_noda in vol_grid:       # case 2b threshold
        # non-DA hours: produce if imb < v_noda, curtail if imb >= v_noda
        noda_produce = ~sold_da & (imb_vals < v_noda)
        rev_imb = (imb_p[noda_produce] * sol_vals[noda_produce]).sum()
        n_imb = noda_produce.sum()
        n_curt = (~sold_da & ~noda_produce).sum()

        for v_da in vol_grid:     # case 2a threshold
            # DA hours: produce if imb < v_da, no-produce if imb >= v_da
            da_prod   = sold_da & (imb_vals < v_da)
            da_noprod = sold_da & (imb_vals >= v_da)

            rev_da_p = (da_vals[da_prod] * sol_vals[da_prod]).sum()
            rev_da_n = ((da_vals[da_noprod] - imb_p[da_noprod]) * sol_vals[da_noprod]).sum()

            total = rev_da_p + rev_da_n + rev_imb

            rows_grid.append({
                'bid': bid, 'v_noda': v_noda, 'v_da': v_da,
                'rev_per_mwh': total / tv,
                'h_da_prod': da_prod.sum(), 'h_da_noprod': da_noprod.sum(),
                'h_imb': n_imb, 'h_curt': n_curt,
            })

grid = pd.DataFrame(rows_grid)
best = grid.loc[grid['rev_per_mwh'].idxmax()]

print(f"[+] Grid: {len(bid_grid)} x {len(vol_grid)} x {len(vol_grid)} = {len(grid)} combos")
print(f"\n--- BEST COMBINATION ---")
print(f"  DA bid threshold:        {best['bid']:.0f} EUR/MWh")
print(f"  Vol thresh (no DA sell):  {best['v_noda']:.0f} MWh  (curtail when surplus >= this)")
print(f"  Vol thresh (DA sell):     {best['v_da']:.0f} MWh  (no-produce when surplus >= this)")
print(f"  Revenue:                  {best['rev_per_mwh']:.1f} EUR/MWh")
print(f"  Hours: DA+prod={best['h_da_prod']:.0f}, DA+noprod={best['h_da_noprod']:.0f}, "
      f"Imb+prod={best['h_imb']:.0f}, Curtail={best['h_curt']:.0f}")
print(f"\n  vs Pure DA (50.8):   +{best['rev_per_mwh'] - pure_da_rev:.1f} EUR/MWh")
print(f"  vs Old best (62.2):  +{best['rev_per_mwh'] - old_best_rev:.1f} EUR/MWh")

# Top 10 combos
print("\n--- Top 10 combinations ---")
print(f"{'Bid':>5} | {'V_noDA':>7} | {'V_DA':>6} | {'EUR/MWh':>8} | {'DA+p':>5} | {'DA+np':>6} | {'Imb':>5} | {'Curt':>5}")
print("-" * 65)
for _, r in grid.nlargest(10, 'rev_per_mwh').iterrows():
    print(f"{r['bid']:>5.0f} | {r['v_noda']:>7.0f} | {r['v_da']:>6.0f} | {r['rev_per_mwh']:>8.1f} | "
          f"{r['h_da_prod']:>5.0f} | {r['h_da_noprod']:>6.0f} | {r['h_imb']:>5.0f} | {r['h_curt']:>5.0f}")

# ============================================================
# FIGURE 3: Heatmaps - fix one param at its optimum, show other two
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

# Panel 1: fix DA bid at optimum, show V_noDA vs V_DA
ax = axes[0]
sub = grid[grid['bid'] == best['bid']].pivot_table(
    index='v_da', columns='v_noda', values='rev_per_mwh')
im = ax.imshow(sub.values, aspect='auto', origin='lower', cmap='RdYlGn',
               extent=[sub.columns.min(), sub.columns.max(),
                       sub.index.min(), sub.index.max()])
ax.set_xlabel('V_noDA (curtail thresh, no DA)', fontsize=11)
ax.set_ylabel('V_DA (no-produce thresh, DA sold)', fontsize=11)
ax.set_title(f'Revenue at bid={best["bid"]:.0f} EUR\n(EUR/MWh)', fontsize=12)
ax.plot(best['v_noda'], best['v_da'], 'k*', markersize=15)
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 2: fix V_DA at optimum, show bid vs V_noDA
ax = axes[1]
sub = grid[grid['v_da'] == best['v_da']].pivot_table(
    index='bid', columns='v_noda', values='rev_per_mwh')
im = ax.imshow(sub.values, aspect='auto', origin='lower', cmap='RdYlGn',
               extent=[sub.columns.min(), sub.columns.max(),
                       sub.index.min(), sub.index.max()])
ax.set_xlabel('V_noDA (curtail thresh, no DA)', fontsize=11)
ax.set_ylabel('DA bid threshold (EUR/MWh)', fontsize=11)
ax.set_title(f'Revenue at V_DA={best["v_da"]:.0f} MWh\n(EUR/MWh)', fontsize=12)
ax.plot(best['v_noda'], best['bid'], 'k*', markersize=15)
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 3: fix V_noDA at optimum, show bid vs V_DA
ax = axes[2]
sub = grid[grid['v_noda'] == best['v_noda']].pivot_table(
    index='bid', columns='v_da', values='rev_per_mwh')
im = ax.imshow(sub.values, aspect='auto', origin='lower', cmap='RdYlGn',
               extent=[sub.columns.min(), sub.columns.max(),
                       sub.index.min(), sub.index.max()])
ax.set_xlabel('V_DA (no-produce thresh, DA sold)', fontsize=11)
ax.set_ylabel('DA bid threshold (EUR/MWh)', fontsize=11)
ax.set_title(f'Revenue at V_noDA={best["v_noda"]:.0f} MWh\n(EUR/MWh)', fontsize=12)
ax.plot(best['v_da'], best['bid'], 'k*', markersize=15)
plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle('3D Grid Search: Revenue Heatmaps (black star = optimum)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / 'v2_03_grid_search.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] v2_03_grid_search.png")

grid.to_csv(OUT_DIR / 'v2_grid_search_full.csv', index=False)
print("[+] v2_grid_search_full.csv")
print("\n[+] Done. Output in:", OUT_DIR)
