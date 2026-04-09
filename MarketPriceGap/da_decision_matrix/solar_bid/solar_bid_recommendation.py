"""
Solar Farm DA Bid Strategy - Final Recommendation
===================================================

1MW East-West solar farm in Slovakia (Bratislava, 48.15N).
PVGIS-calibrated production. Volume-based imbalance predictor.

CURTAILMENT RULE explained:
  We have a predictor that gives us predicted system imbalance in MWh.
  Positive imbalance_mwh = system surplus = too much generation = prices crash.

  "Curtail top 50% surplus volume" means:
    - Take all hours where DA rejected our bid (DA price < our bid threshold)
    - Rank those hours by predicted surplus volume (largest imbalance_mwh first)
    - Curtail (turn off) during the top 50% of that volume (biggest surplus hours)
    - Produce at imbalance price during the remaining hours

  The 50% refers to volume (MWh), not hours. We budget curtailing up to
  half of the production that would have gone to imbalance.

Strategies compared:
  A) Pure DA (bid=0): sell everything on DA market
  B) DA or nothing: bid on DA, curtail if DA rejects
  C) Hybrid: DA above threshold; below threshold, check predictor:
     - large surplus predicted -> curtail
     - deficit or small surplus -> produce at imbalance price
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = OUT_DIR.parent.parent / "data" / "processed" / "hourly_market_prices.csv"

plt.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ============================================================
# SOLAR MODEL (PVGIS v5.3, Bratislava, 0.5MW East + 0.5MW West, 35 deg)
# ============================================================
PVGIS = {1:24.9, 2:41.0, 3:78.9, 4:107.8, 5:123.6, 6:131.1,
         7:133.3, 8:114.2, 9:87.0, 10:55.2, 11:27.4, 12:19.2}
DAYS = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

def _shape(hour, month):
    dl = {1:8.5,2:10,3:11.5,4:13.5,5:15,6:16,7:15.5,8:14,9:12.5,10:11,11:9,12:8}[month]
    sr, ss = 12 - dl/2, 12 + dl/2
    if hour < sr or hour >= ss:
        return 0.0
    sp = (hour - sr) / (ss - sr)
    irr = np.sin(sp * np.pi)
    ef = np.exp(-((sp - 0.30)**2) / (2 * 0.12**2))
    wf = np.exp(-((sp - 0.70)**2) / (2 * 0.12**2))
    return 0.5 * irr * (0.4 + 0.6 * ef) + 0.5 * irr * (0.4 + 0.6 * wf)

def solar_mwh(hour, month):
    rd = sum(_shape(h, month) for h in range(24))
    if rd <= 0:
        return 0.0
    return _shape(hour, month) * PVGIS[month] / (rd * DAYS[month])

def vol_curtail(df, pct):
    """Curtail during the largest-surplus hours, up to `pct` of volume.

    Sorts imbalance hours by predicted surplus (imbalance_mwh, descending).
    Marks hours as curtailed until the cumulative solar production of
    curtailed hours reaches `pct` fraction of total imbalance volume.
    Only curtails during actual surplus hours (imbalance_mwh > 0).
    """
    if len(df) == 0 or pct <= 0:
        return pd.Series(False, index=df.index)
    budget = df['solar_mwh'].sum() * pct
    s = df.sort_values('imbalance_mwh', ascending=False)
    s['is_sur'] = s['imbalance_mwh'] > 0
    s['cv'] = (s['solar_mwh'] * s['is_sur']).cumsum()
    s['c'] = s['is_sur'] & (s['cv'] <= budget)
    return s['c'].reindex(df.index)


def sweep_strategies(sol):
    """Sweep bid thresholds for each turn-off level. Returns DataFrame."""
    rows = []
    for tp in [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]:
        for bid in np.arange(-50, 251, 1):
            da_mask = sol['da_price'] >= bid
            da_r = (sol.loc[da_mask, 'da_price'] * sol.loc[da_mask, 'solar_mwh']).sum()
            da_vol = sol.loc[da_mask, 'solar_mwh'].sum()
            imb_h = sol[~da_mask].copy()
            if len(imb_h) > 0 and tp < 1.0:
                imb_h['c'] = vol_curtail(imb_h, tp)
                imb_r = (imb_h.loc[~imb_h['c'], 'imb_settlement_price'] *
                         imb_h.loc[~imb_h['c'], 'solar_mwh']).sum()
                n_curt = imb_h['c'].sum()
                imb_vol = imb_h.loc[~imb_h['c'], 'solar_mwh'].sum()
                curt_vol = imb_h.loc[imb_h['c'], 'solar_mwh'].sum()
            elif len(imb_h) > 0:
                imb_r = 0; n_curt = len(imb_h)
                imb_vol = 0; curt_vol = imb_h['solar_mwh'].sum()
            else:
                imb_r = 0; n_curt = 0; imb_vol = 0; curt_vol = 0
            rows.append({
                'curtail_pct': tp, 'bid': bid,
                'total_rev': da_r + imb_r, 'da_rev': da_r, 'imb_rev': imb_r,
                'da_hours': da_mask.sum(), 'curt_hours': n_curt,
                'da_vol': da_vol, 'imb_vol': imb_vol, 'curt_vol': curt_vol,
                'prod_mwh': da_vol + imb_vol,
            })
    return pd.DataFrame(rows)


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
print("[*] Loading market data...")
mkt = pd.read_csv(DATA_PATH, parse_dates=['timestamp_hour'])

def prepare(mkt, year, months):
    d = mkt[(mkt['timestamp_hour'].dt.year == year) & (mkt['month'].isin(months))].copy()
    d = d.dropna(subset=['da_price', 'imb_settlement_price', 'imbalance_mwh'])
    d['solar_mwh'] = d.apply(lambda r: solar_mwh(r['hour'], r['month']), axis=1)
    return d[d['solar_mwh'] > 0.001].copy()

sol_25ss = prepare(mkt, 2025, [3,4,5,6,7,8])   # 2025 spring-summer (reference)
sol_25fm = prepare(mkt, 2025, [2,3])             # 2025 Feb-Mar (YoY comparison)
sol_26fm = prepare(mkt, 2026, [2,3])             # 2026 Feb-Mar (current)

print(f"[+] 2025 Mar-Aug: {len(sol_25ss)} hours, {sol_25ss['solar_mwh'].sum():.0f} MWh")
print(f"[+] 2025 Feb-Mar: {len(sol_25fm)} hours, {sol_25fm['solar_mwh'].sum():.0f} MWh")
print(f"[+] 2026 Feb-Mar: {len(sol_26fm)} hours, {sol_26fm['solar_mwh'].sum():.0f} MWh")

# Main sweep on 2025 spring-summer
print("[*] Sweeping strategies on 2025 Mar-Aug...")
sweep = sweep_strategies(sol_25ss)

optima = sweep.loc[sweep.groupby('curtail_pct')['total_rev'].idxmax()]
tv_25ss = sol_25ss['solar_mwh'].sum()

print("[*] Generating figures...")


# ============================================================
# FIGURE 1: Revenue & Volume Composition by Market
# Shows WHERE the energy goes and WHERE the money comes from
# as a function of the DA bid threshold.
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sub50 = sweep[sweep['curtail_pct'] == 0.50]
sub0 = sweep[sweep['curtail_pct'] == 0.0]
bids = sub50['bid'].values

# --- Left panel: Where does the energy go? ---
ax = axes[0]

da_pct = sub50['da_vol'].values / tv_25ss * 100
imb_pct = sub50['imb_vol'].values / tv_25ss * 100
off_pct = sub50['curt_vol'].values / tv_25ss * 100

ax.fill_between(bids, 0, da_pct,
                color='#27ae60', alpha=0.6, label='Sold on DA market')
ax.fill_between(bids, da_pct, da_pct + imb_pct,
                color='#2980b9', alpha=0.6, label='Sold at imbalance price')
ax.fill_between(bids, da_pct + imb_pct, da_pct + imb_pct + off_pct,
                color='#e74c3c', alpha=0.4, label='Curtailed (surplus hours)')

ax.axvline(58, color='black', linestyle=':', alpha=0.7, lw=1.5)
ax.text(60, 50, 'Bid = 58', fontsize=11, fontweight='bold', rotation=90,
        va='center', ha='left')

ax.set_xlabel('DA Bid Threshold (EUR/MWh)', fontsize=12)
ax.set_ylabel('Share of Total Solar Production (%)', fontsize=12)
ax.set_title('Where Does the Energy Go?\n'
             '(curtail top 50% surplus volume)', fontsize=13)
ax.legend(fontsize=10, loc='center left')
ax.set_xlim(-50, 200)
ax.set_ylim(0, 105)

# --- Right panel: Where does the money come from? ---
ax = axes[1]

da_r = sub50['da_rev'].values / tv_25ss
imb_r = sub50['imb_rev'].values / tv_25ss
total_r = sub50['total_rev'].values / tv_25ss
total_no_off = sub0['total_rev'].values / tv_25ss

# Stacked: DA revenue from 0, Imbalance stacked on top
ax.fill_between(bids, 0, da_r,
                color='#27ae60', alpha=0.5, label='DA market revenue')
ax.fill_between(bids, da_r, da_r + imb_r,
                color='#2980b9', alpha=0.5, label='Imbalance market revenue')

# Total lines
ax.plot(bids, total_r, 'k-', lw=2.5, label='Total (with curtailment)', zorder=4)
ax.plot(bids, total_no_off, color='gray', linestyle='--', lw=1.5,
        label='Total (no curtailment)')

# Highlight curtailment savings
savings = total_r - total_no_off
ax.fill_between(bids, total_no_off, total_r,
                where=(savings > 0),
                color='#e74c3c', alpha=0.15, label='Savings from curtailment')

# Mark optimum
opt_idx = int(np.argmax(total_r))
ax.scatter([bids[opt_idx]], [total_r[opt_idx]], color='black', s=120, zorder=5,
           edgecolors='white', linewidth=2)
ax.annotate(f'Best: {total_r[opt_idx]:.1f} EUR/MWh\nbid = {bids[opt_idx]:.0f} EUR',
            xy=(bids[opt_idx], total_r[opt_idx]),
            xytext=(bids[opt_idx] + 40, total_r[opt_idx] - 6),
            fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.axhline(0, color='gray', alpha=0.3)
ax.set_xlabel('DA Bid Threshold (EUR/MWh)', fontsize=12)
ax.set_ylabel('Revenue per MWh Produced (EUR/MWh)', fontsize=12)
ax.set_title('Where Does the Money Come From?\n'
             '(curtail top 50% surplus volume)', fontsize=13)
ax.legend(fontsize=9, loc='lower left')
ax.set_xlim(-50, 200)

plt.tight_layout()
plt.savefig(OUT_DIR / '01_revenue_composition.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 01_revenue_composition.png")


# ============================================================
# FIGURE 1b: Zoomed bid effect on strategy split (0-80 EUR)
# For the report: shows how the three buckets shift as bid rises
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

mask = (sub50['bid'] >= 0) & (sub50['bid'] <= 80)
zb = sub50[mask]['bid'].values
z_da = sub50[mask]['da_vol'].values / tv_25ss * 100
z_imb = sub50[mask]['imb_vol'].values / tv_25ss * 100
z_curt = sub50[mask]['curt_vol'].values / tv_25ss * 100

ax.fill_between(zb, 0, z_da, color='#27ae60', alpha=0.6, label='Sold on DA')
ax.fill_between(zb, z_da, z_da + z_imb, color='#2980b9', alpha=0.6,
                label='Produced at imbalance')
ax.fill_between(zb, z_da + z_imb, z_da + z_imb + z_curt, color='#e74c3c',
                alpha=0.4, label='Curtailed (top surplus hours)')

ax.axvline(58, color='black', linestyle='--', alpha=0.8, lw=2)
ax.text(59, 50, 'Recommended bid = 58 EUR', fontsize=11, fontweight='bold',
        rotation=90, va='center', ha='left')

# Add annotations at a few bid levels
for bid_val in [0, 20, 40, 58]:
    row = sub50[sub50['bid'] == bid_val].iloc[0]
    da_p = row['da_vol'] / tv_25ss * 100
    im_p = row['imb_vol'] / tv_25ss * 100
    cu_p = row['curt_vol'] / tv_25ss * 100
    ax.annotate(f'DA {da_p:.0f}% | Imb {im_p:.0f}% | Curt {cu_p:.0f}%',
                xy=(bid_val, 102), fontsize=8, ha='center', va='bottom',
                color='#333333')

ax.set_xlabel('DA Bid Threshold (EUR/MWh)', fontsize=12)
ax.set_ylabel('Share of Production (%)', fontsize=12)
ax.set_title('How the DA Bid Splits Production Between Markets\n'
             '(curtail top 50% surplus volume, 2025 Mar-Aug)', fontsize=13)
ax.legend(fontsize=10, loc='center left')
ax.set_xlim(0, 80)
ax.set_ylim(0, 112)

plt.tight_layout()
plt.savefig(OUT_DIR / '07_bid_effect_zoomed.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 07_bid_effect_zoomed.png")


# ============================================================
# FIGURE 7b: Simple revenue vs bid (single line, 50% curtailment)
# For the report: clean view of how revenue depends on bid
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

rev_line = sub50['total_rev'].values / tv_25ss

ax.plot(bids, rev_line, color='#2980b9', linewidth=2.5)
ax.fill_between(bids, rev_line, alpha=0.08, color='#2980b9')

# Mark optimum
opt_idx = int(np.argmax(rev_line))
ax.scatter([bids[opt_idx]], [rev_line[opt_idx]], color='#2980b9', s=120,
           zorder=5, edgecolors='white', linewidth=2)
ax.annotate(f'{rev_line[opt_idx]:.1f} EUR/MWh at bid = {bids[opt_idx]:.0f} EUR',
            xy=(bids[opt_idx], rev_line[opt_idx]),
            xytext=(bids[opt_idx] + 35, rev_line[opt_idx] + 1),
            fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.5),
            color='#2980b9')

# Reference: pure DA (bid=0)
pure_da = rev_line[bids == 0][0]
ax.axhline(pure_da, color='#27ae60', linestyle='--', alpha=0.6, lw=1.5)
ax.text(160, pure_da + 0.5, f'Pure DA: {pure_da:.1f} EUR/MWh',
        fontsize=10, color='#27ae60')

ax.set_xlabel('DA Bid Threshold (EUR/MWh)', fontsize=12)
ax.set_ylabel('Revenue per MWh Produced (EUR/MWh)', fontsize=12)
ax.set_title('Revenue vs DA Bid Threshold\n'
             '(hybrid strategy, curtail top 50% surplus volume)', fontsize=13)
ax.set_xlim(-50, 200)
ax.set_ylim(40, 65)

plt.tight_layout()
plt.savefig(OUT_DIR / '07_revenue_vs_bid.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 07_revenue_vs_bid.png")


# ============================================================
# FIGURE 2: Strategy Comparison - All curtailment levels
# ============================================================
fig, ax = plt.subplots(figsize=(13, 7))

labels = {
    0.0:  'No curtailment (all imbalance)',
    0.10: 'Curtail top 10% surplus vol',
    0.25: 'Curtail top 25% surplus vol',
    0.50: 'Curtail top 50% surplus vol (recommended)',
    0.75: 'Curtail top 75% surplus vol',
    1.0:  'DA or nothing (curtail 100%)',
}
colors = {0.0: '#bdc3c7', 0.10: '#e67e22', 0.25: '#27ae60',
          0.50: '#2980b9', 0.75: '#8e44ad', 1.0: '#e74c3c'}
lwidths = {0.0: 1, 0.10: 1.5, 0.25: 1.5, 0.50: 3, 0.75: 1.5, 1.0: 1.5}

for tp in [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]:
    sub = sweep[sweep['curtail_pct'] == tp]
    opt = optima[optima['curtail_pct'] == tp].iloc[0]
    ax.plot(sub['bid'], sub['total_rev'] / tv_25ss, color=colors[tp],
            linewidth=lwidths[tp],
            label=f'{labels[tp]} (bid={opt["bid"]:.0f}, {opt["total_rev"]/tv_25ss:.1f} EUR/MWh)',
            alpha=0.9 if tp == 0.50 else 0.6)
    ax.scatter([opt['bid']], [opt['total_rev'] / tv_25ss], color=colors[tp],
               s=100, zorder=5, edgecolors='black', linewidth=0.5)

ax.set_xlabel('DA Bid Threshold (EUR/MWh)', fontsize=12)
ax.set_ylabel('Revenue per MWh Produced (EUR/MWh)', fontsize=12)
ax.set_title('Strategy Comparison: Revenue vs DA Bid Threshold\n'
             '1MW Solar E-W, 2025 Spring-Summer', fontsize=13)
ax.legend(fontsize=9, loc='lower left')
ax.set_xlim(-50, 200)
ax.set_ylim(40, 65)
ax.axvline(58, color='#2980b9', linestyle=':', alpha=0.4)
ax.annotate('Recommended:\nbid 58 EUR, curtail\ntop 50% surplus vol',
            xy=(58, 62.2), xytext=(105, 56),
            fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.5),
            color='#2980b9')

plt.tight_layout()
plt.savefig(OUT_DIR / '02_strategy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 02_strategy_comparison.png")


# ============================================================
# FIGURE 3: Seasonal Strategy Map
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
month_names = {2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug'}

# Panel 1: Monthly optimal bid (turn off top 50% surplus)
ax = axes[0]
monthly_bids = []
for m in range(2, 9):
    sol_m = prepare(mkt, 2025, [m])
    if len(sol_m) < 20:
        continue
    tv_m = sol_m['solar_mwh'].sum()
    best = {'bid': 0, 'rev': -1e9}
    for bid in np.arange(-50, 201, 1):
        da_mask = sol_m['da_price'] >= bid
        da_r = (sol_m.loc[da_mask, 'da_price'] * sol_m.loc[da_mask, 'solar_mwh']).sum()
        imb_h = sol_m[~da_mask].copy()
        if len(imb_h) > 0:
            imb_h['c'] = vol_curtail(imb_h, 0.50)
            imb_r = (imb_h.loc[~imb_h['c'], 'imb_settlement_price'] *
                     imb_h.loc[~imb_h['c'], 'solar_mwh']).sum()
        else:
            imb_r = 0
        total = da_r + imb_r
        if total > best['rev']:
            best = {'bid': bid, 'rev': total}
    monthly_bids.append({'month': m, 'name': month_names[m], 'bid': best['bid'],
                         'rev_per_mwh': best['rev'] / tv_m, 'prod_mwh': tv_m})

mb = pd.DataFrame(monthly_bids)

bar_colors = []
for _, r in mb.iterrows():
    if r['bid'] <= 10:
        bar_colors.append('#27ae60')
    elif r['bid'] <= 60:
        bar_colors.append('#f39c12')
    else:
        bar_colors.append('#2980b9')

bars = ax.bar(mb['name'], mb['bid'], color=bar_colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, mb['bid']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Optimal DA Bid (EUR/MWh)', fontsize=12)
ax.set_title('Monthly Optimal DA Bid\n(curtail top 50% surplus volume, 2025)', fontsize=12)
ax.axhline(0, color='gray', alpha=0.5)
legend_elements = [
    Patch(facecolor='#27ae60', edgecolor='black', label='Pure DA (bid ~0)'),
    Patch(facecolor='#f39c12', edgecolor='black', label='Transition (bid 20-60)'),
    Patch(facecolor='#2980b9', edgecolor='black', label='Selective DA (bid 60+)'),
]
ax.legend(handles=legend_elements, fontsize=9)

# Panel 2: Monthly revenue comparison
ax = axes[1]
rev_da_only = []
rev_hybrid = []
for _, r in mb.iterrows():
    sol_m = prepare(mkt, 2025, [r['month']])
    tv_m = sol_m['solar_mwh'].sum()
    rev_da_only.append((sol_m['da_price'] * sol_m['solar_mwh']).sum() / tv_m)
    rev_hybrid.append(r['rev_per_mwh'])

x = np.arange(len(mb))
width = 0.35
ax.bar(x - width/2, rev_da_only, width, label='Pure DA', color='#27ae60', alpha=0.7)
ax.bar(x + width/2, rev_hybrid, width, label='Hybrid (curtail surplus)', color='#2980b9', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(mb['name'])
ax.set_ylabel('Revenue per MWh (EUR/MWh)', fontsize=12)
ax.set_title('Monthly Revenue: Pure DA vs Hybrid\n(2025 data)', fontsize=12)
ax.legend(fontsize=10)

for i, (da, hyb) in enumerate(zip(rev_da_only, rev_hybrid)):
    diff = hyb - da
    if diff > 1:
        ax.annotate(f'+{diff:.0f}', xy=(i + width/2, hyb),
                    xytext=(i + width/2 + 0.1, hyb + 3),
                    fontsize=9, color='#2980b9', fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / '03_seasonal_strategy.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 03_seasonal_strategy.png")


# ============================================================
# FIGURE 4: 2026 vs 2025 - Current Season Context
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
for sol, color, label, marker in [(sol_26fm, '#2980b9', '2026 Feb-Mar', 'o'),
                                   (sol_25fm, '#e67e22', '2025 Feb-Mar', 's')]:
    ax.scatter(sol['da_price'], sol['imb_settlement_price'],
               color=color, alpha=0.35, s=15, marker=marker, label=label)
ax.plot([-200, 400], [-200, 400], 'k--', alpha=0.3, label='DA = Imbalance')
ax.axhline(0, color='gray', alpha=0.2)
ax.axvline(0, color='gray', alpha=0.2)
ax.set_xlabel('DA Price (EUR/MWh)', fontsize=12)
ax.set_ylabel('Imbalance Settlement Price (EUR/MWh)', fontsize=12)
ax.set_title('DA vs Imbalance Prices: 2026 vs 2025\nFeb-Mar solar hours', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-150, 250)
ax.set_ylim(-400, 400)

ax = axes[1]
for sol, color, label in [(sol_26fm, '#2980b9', '2026'), (sol_25fm, '#e67e22', '2025')]:
    h = sol.groupby('hour').agg(
        da=('da_price', 'mean'),
        imb=('imb_settlement_price', 'mean'),
    ).reset_index()
    ax.plot(h['hour'], h['da'], '-o', color=color, linewidth=2, markersize=5,
            label=f'DA {label}')
    ax.plot(h['hour'], h['imb'], '--s', color=color, linewidth=1.5, markersize=4,
            alpha=0.6, label=f'Imb {label}')
ax.axhline(0, color='gray', alpha=0.3)
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Mean Price (EUR/MWh)', fontsize=12)
ax.set_title('Hourly Price Profiles: 2026 vs 2025\nFeb-Mar solar hours', fontsize=12)
ax.legend(fontsize=9)
ax.set_xticks(range(7, 19))

plt.tight_layout()
plt.savefig(OUT_DIR / '04_current_season.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 04_current_season.png")


# ============================================================
# FIGURE 5: Volume Predictor Effectiveness
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ref_bid = 58
imb_hours = sol_25ss[sol_25ss['da_price'] < ref_bid].copy()

# Panel 1: Revenue vs curtailment level (oracle vs volume predictor)
ax = axes[0]
curtail_levels = np.arange(0, 1.01, 0.02)
rev_vol = []
rev_oracle = []
imb_tv = imb_hours['solar_mwh'].sum()

for tp in curtail_levels:
    if tp > 0:
        vc = vol_curtail(imb_hours, tp)
        kept = imb_hours[~vc]
    else:
        kept = imb_hours
    rev_vol.append((kept['imb_settlement_price'] * kept['solar_mwh']).sum())

    if tp > 0:
        ss = imb_hours.sort_values('imb_settlement_price')
        ss['cv'] = ss['solar_mwh'].cumsum()
        oracle_off = ss['cv'] <= imb_tv * tp
        kept_o = imb_hours[~oracle_off.reindex(imb_hours.index)]
    else:
        kept_o = imb_hours
    rev_oracle.append((kept_o['imb_settlement_price'] * kept_o['solar_mwh']).sum())

ax.plot([t*100 for t in curtail_levels], [r/imb_tv for r in rev_oracle],
        'r-', linewidth=2, label='Oracle (perfect price foresight)')
ax.plot([t*100 for t in curtail_levels], [r/imb_tv for r in rev_vol],
        'b-', linewidth=2, label='Volume predictor (realistic)')
ax.fill_between([t*100 for t in curtail_levels],
                [r/imb_tv for r in rev_vol], [r/imb_tv for r in rev_oracle],
                alpha=0.1, color='red', label='Prediction gap')
ax.axvline(50, color='#2980b9', linestyle=':', alpha=0.6, label='50% (recommended)')
ax.axvline(10, color='#e67e22', linestyle=':', alpha=0.6, label='10%')
ax.set_xlabel('Curtailment budget (% of imbalance volume, ranked by surplus)', fontsize=12)
ax.set_ylabel('Revenue per MWh on imbalance (EUR/MWh)', fontsize=12)
ax.set_title('Curtailment Value: Volume Predictor vs Oracle\n'
             'Hours below DA bid of 58 EUR, 2025 Mar-Aug', fontsize=12)
ax.legend(fontsize=9)

# Panel 2: Scatter showing produce vs curtail decision at 50%
ax = axes[1]
imb_hours['curt_50'] = vol_curtail(imb_hours, 0.50)
kept = imb_hours[~imb_hours['curt_50']]
curtailed = imb_hours[imb_hours['curt_50']]

ax.scatter(kept['imbalance_mwh'], kept['imb_settlement_price'],
           alpha=0.3, s=12, color='#27ae60',
           label=f'Produce ({len(kept)}h, avg {kept["imb_settlement_price"].mean():.0f} EUR)')
if len(curtailed) > 0:
    ax.scatter(curtailed['imbalance_mwh'], curtailed['imb_settlement_price'],
               alpha=0.5, s=18, color='#e74c3c', marker='x',
               label=f'Curtail ({len(curtailed)}h, avg {curtailed["imb_settlement_price"].mean():.0f} EUR)')

ax.axhline(0, color='black', linestyle=':', alpha=0.3)
ax.axvline(0, color='black', linestyle=':', alpha=0.3)
ax.set_xlabel('Predicted System Surplus (imbalance_mwh, + = surplus)', fontsize=12)
ax.set_ylabel('Imbalance Settlement Price (EUR/MWh)', fontsize=12)
ax.set_title('Produce vs Curtail at 50% Budget\n'
             'Red = largest surplus hours (curtailed by predictor)', fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / '05_volume_predictor.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 05_volume_predictor.png")


# ============================================================
# FIGURE 6: E-W Solar Profile & Production Economics
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
for m, color, label in [(3, '#27ae60', 'March (now)'), (5, '#f39c12', 'May'),
                         (6, '#e74c3c', 'June'), (8, '#2980b9', 'August')]:
    hours = list(range(4, 22))
    prod = [solar_mwh(h, m) * 1000 for h in hours]
    ax.plot(hours, prod, '-o', color=color, label=label, markersize=4, linewidth=2)
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Production (kW per 1MW installed)', fontsize=12)
ax.set_title('E-W Solar Profile by Month\nPVGIS-calibrated, Bratislava (48.15N)', fontsize=12)
ax.legend(fontsize=10)
ax.set_xticks(range(5, 21))

ax = axes[1]
months_all = list(range(2, 9))
prod_monthly = []
rev_da = []
rev_hyb = []

for m in months_all:
    sol_m = prepare(mkt, 2025, [m])
    if len(sol_m) < 10:
        prod_monthly.append(0); rev_da.append(0); rev_hyb.append(0)
        continue
    tv_m = sol_m['solar_mwh'].sum()
    prod_monthly.append(tv_m)
    rev_da.append((sol_m['da_price'] * sol_m['solar_mwh']).sum())
    best_r = -1e9
    for bid in np.arange(-50, 201, 2):
        da_mask = sol_m['da_price'] >= bid
        r = (sol_m.loc[da_mask, 'da_price'] * sol_m.loc[da_mask, 'solar_mwh']).sum()
        imb_h = sol_m[~da_mask].copy()
        if len(imb_h) > 0:
            imb_h['c'] = vol_curtail(imb_h, 0.50)
            r += (imb_h.loc[~imb_h['c'], 'imb_settlement_price'] *
                  imb_h.loc[~imb_h['c'], 'solar_mwh']).sum()
        if r > best_r:
            best_r = r
    rev_hyb.append(best_r)

x = np.arange(len(months_all))
width = 0.35
ax.bar(x - width/2, [r/1000 for r in rev_da], width,
       label='Pure DA', color='#27ae60', alpha=0.7)
ax.bar(x + width/2, [r/1000 for r in rev_hyb], width,
       label='Hybrid (curtail surplus)', color='#2980b9', alpha=0.7)

ax2 = ax.twinx()
ax2.plot(x, prod_monthly, 'k-o', linewidth=2, markersize=6, label='Production (MWh)')
ax2.set_ylabel('Production (MWh)', fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels([month_names.get(m, str(m)) for m in months_all])
ax.set_ylabel('Revenue (kEUR)', fontsize=12)
ax.set_title('Monthly Revenue & Production\n1MW E-W Solar, 2025 data', fontsize=12)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

for i, m in enumerate(months_all):
    diff = rev_hyb[i] - rev_da[i]
    if diff > 500:
        ax.annotate(f'+{diff/1000:.1f}k', xy=(i + width/2, rev_hyb[i]/1000),
                    xytext=(i + 0.4, rev_hyb[i]/1000 + 0.3),
                    fontsize=8, color='#2980b9', fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / '06_production_economics.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 06_production_economics.png")


# ============================================================
# PRINT FINAL RECOMMENDATION
# ============================================================
print("\n" + "=" * 70)
print("RECOMMENDATION - Solar Farm DA Bidding Strategy")
print("=" * 70)

print("""
PLANT: 1MW East-West solar, Bratislava, Slovakia
DATE: March 2026 (entering solar season)
DATA: 2025 full year + 2026 Jan-Mar

STRATEGY: Hybrid with volume-predictor curtailment

  How it works:
    1. Set a DA bid threshold (e.g., 58 EUR/MWh)
    2. Hours where DA price >= threshold: SELL on DA (committed)
    3. Hours where DA < threshold: check the volume predictor
       - Predictor says large surplus -> CURTAIL (avoid negative prices)
       - Predictor says deficit or small surplus -> PRODUCE at imbalance
    4. Curtailment budget: top 50% of surplus volume
       (rank hours by predicted surplus, curtail the biggest half)

  Seasonal thresholds (based on 2025 data):

  WINTER (Nov-Feb):     Bid 0 EUR/MWh (pure DA, no turn-off needed)
  MARCH-APRIL (now):    Bid 40-55 EUR/MWh
  MAY-JUNE:             Bid 45-58 EUR/MWh
  JULY-AUGUST:          Bid 40-55 EUR/MWh

EXPECTED PERFORMANCE (2025 spring-summer backtest):
  Hybrid (curtail top 50%):  62.2 EUR/MWh  (42,581 EUR for 685 MWh)
  Pure DA:                     50.8 EUR/MWh  (34,762 EUR) -- baseline
  DA or nothing:               53.1 EUR/MWh  (36,369 EUR)
  Improvement:                +22.5% vs pure DA, +11.4 EUR/MWh

2026 FEB-MAR (actual):
  DA dominates: 96.3 EUR/MWh. No benefit to curtailment.
  Consistent with winter rule (bid 0, pure DA).
""")

# Save summary data
summary = pd.DataFrame([
    {'period': '2025 Mar-Aug', 'strategy': 'Hybrid curtail top-50%', 'opt_bid': 58, 'rev_per_mwh': 62.2},
    {'period': '2025 Mar-Aug', 'strategy': 'Pure DA', 'opt_bid': 0, 'rev_per_mwh': 50.8},
    {'period': '2025 Mar-Aug', 'strategy': 'DA or nothing', 'opt_bid': 0, 'rev_per_mwh': 53.1},
    {'period': '2026 Feb-Mar', 'strategy': 'Pure DA', 'opt_bid': 0, 'rev_per_mwh': 96.3},
])
summary.to_csv(OUT_DIR / 'recommendation_summary.csv', index=False)

print("[+] All files saved to", OUT_DIR)
