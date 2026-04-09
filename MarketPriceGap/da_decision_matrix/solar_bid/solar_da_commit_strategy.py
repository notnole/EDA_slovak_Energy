"""
Solar Farm: DA Commit + No-Produce Strategy
============================================

New strategy: sell on DA at a low bid threshold, then CHOOSE whether to
produce based on realised/predicted imbalance settlement price.

Key insight:
  If you sell on DA and do NOT produce, your imbalance position is SHORT
  (you under-delivered). Revenue = (da_price - imb_settlement_price) * Q
  When imb_settlement_price < 0 (system surplus), this ADDS to DA revenue.
  This is strictly better than curtailment (revenue=0) AND better than
  producing during negative-price hours.

Decision tree each solar hour:
  1. DA bid >= bid_threshold -> SOLD on DA
       - imb_settlement_price < no_produce_thresh -> DON'T PRODUCE
           revenue = (da_price - imb_settlement_price) * Q  [double benefit]
       - imb_settlement_price >= no_produce_thresh -> PRODUCE
           revenue = da_price * Q                            [normal DA]
  2. DA bid < bid_threshold -> NOT on DA, go to imbalance
       - imb_settlement_price >= 0 -> PRODUCE at imbalance
           revenue = imb_settlement_price * Q
       - imb_settlement_price < 0  -> CURTAIL
           revenue = 0

Analysis:
  A) Oracle: perfect knowledge of imbalance price at decision time
  B) Volume predictor: use system imbalance_mwh as surplus proxy
  C) Sweep bid thresholds to find optimal
  D) Compare to old strategy (curtail only, no DA commitment trick)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
DATA_PATH = OUT_DIR.parent.parent / "data" / "processed" / "hourly_market_prices.csv"

plt.rcParams.update({'figure.figsize': (14, 8), 'font.size': 11,
                     'axes.grid': True, 'grid.alpha': 0.3})

# ============================================================
# SOLAR MODEL (PVGIS v5.3, Bratislava, 0.5MW East + 0.5MW West)
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
print("[*] Loading market data...")
mkt = pd.read_csv(DATA_PATH, parse_dates=['timestamp_hour'])

def prepare(df, year, months):
    d = df[(df['timestamp_hour'].dt.year == year) & df['month'].isin(months)].copy()
    d = d.dropna(subset=['da_price', 'imb_settlement_price', 'imbalance_mwh'])
    d['solar_mwh'] = d.apply(lambda r: solar_mwh(r['hour'], r['month']), axis=1)
    return d[d['solar_mwh'] > 0.001].copy()

sol = prepare(mkt, 2025, [3,4,5,6,7,8])
tv  = sol['solar_mwh'].sum()
print(f"[+] 2025 Mar-Aug: {len(sol)} hours, {tv:.0f} MWh")

# ============================================================
# VOLUME PREDICTOR: predict surplus/deficit from imbalance_mwh
# Use the actual imbalance_mwh as a proxy (positive = surplus = price risk)
# In reality you'd have a nowcast; here we rank by this signal.
# ============================================================

def predict_no_produce(df, surplus_thresh_mwh):
    """Return boolean mask: True = DON'T PRODUCE (predicted surplus)."""
    return df['imbalance_mwh'] >= surplus_thresh_mwh

# ============================================================
# STRATEGY SIMULATION
# ============================================================

def simulate(df, bid_thresh, no_produce_thresh_mwh, use_oracle=False):
    """
    Parameters
    ----------
    bid_thresh : float
        Minimum DA price to sell on DA market.
    no_produce_thresh_mwh : float
        Volume-predictor threshold: if imbalance_mwh >= this, don't produce
        (only applies to hours sold on DA). Used when use_oracle=False.
    use_oracle : bool
        If True, use actual imb_settlement_price < 0 as the no-produce signal.
    """
    df = df.copy()

    # DA sale flag
    df['sold_da'] = df['da_price'] >= bid_thresh

    # No-produce decision for DA-sold hours
    if use_oracle:
        df['no_produce'] = df['sold_da'] & (df['imb_settlement_price'] < 0)
    else:
        df['no_produce'] = df['sold_da'] & predict_no_produce(df, no_produce_thresh_mwh)

    # Revenue per hour
    # Case 1: sold on DA + produce normally
    mask_da_prod = df['sold_da'] & ~df['no_produce']
    # Case 2: sold on DA + no produce (earn DA - imbalance, since you're short)
    mask_da_noprod = df['sold_da'] & df['no_produce']
    # Case 3: not on DA, imbalance positive -> produce
    mask_imb_prod = ~df['sold_da'] & (df['imb_settlement_price'] >= 0)
    # Case 4: not on DA, imbalance negative -> curtail
    # (no revenue)

    df['revenue'] = 0.0
    df.loc[mask_da_prod,   'revenue'] = df.loc[mask_da_prod,   'da_price'] * df.loc[mask_da_prod,   'solar_mwh']
    df.loc[mask_da_noprod, 'revenue'] = (df.loc[mask_da_noprod, 'da_price'] - df.loc[mask_da_noprod, 'imb_settlement_price']) * df.loc[mask_da_noprod, 'solar_mwh']
    df.loc[mask_imb_prod,  'revenue'] = df.loc[mask_imb_prod,  'imb_settlement_price'] * df.loc[mask_imb_prod,  'solar_mwh']

    total_rev = df['revenue'].sum()
    n_da_prod   = mask_da_prod.sum()
    n_da_noprod = mask_da_noprod.sum()
    n_imb_prod  = mask_imb_prod.sum()
    n_curtail   = (~df['sold_da'] & (df['imb_settlement_price'] < 0)).sum()

    return {
        'total_rev': total_rev,
        'rev_per_mwh': total_rev / tv,
        'n_da_prod': n_da_prod,
        'n_da_noprod': n_da_noprod,
        'n_imb_prod': n_imb_prod,
        'n_curtail': n_curtail,
        'pct_da': df.loc[df['sold_da'], 'solar_mwh'].sum() / tv,
        'avg_noprod_imb': df.loc[mask_da_noprod, 'imb_settlement_price'].mean() if mask_da_noprod.any() else np.nan,
    }

# ============================================================
# FIGURE 1: How often are imbalance prices negative in solar hours?
# ============================================================
print("[*] Figure 1: Negative imbalance price frequency...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Distribution of imb_settlement_price during solar hours
ax = axes[0]
ax.hist(sol['imb_settlement_price'], bins=80, color='#2980b9', alpha=0.7, edgecolor='white')
ax.axvline(0, color='red', linewidth=2, label='Zero (surplus/deficit boundary)')
neg_pct = (sol['imb_settlement_price'] < 0).mean() * 100
ax.set_xlabel('Imbalance Settlement Price (EUR/MWh)', fontsize=12)
ax.set_ylabel('Hours', fontsize=12)
ax.set_title(f'Distribution of Imbalance Settlement Price\nSolar Hours (Mar-Aug 2025)\n{neg_pct:.1f}% of hours < 0', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-300, 400)

# Panel 2: Negative price frequency by month
ax = axes[1]
months = [3,4,5,6,7,8]
mnames = ['Mar','Apr','May','Jun','Jul','Aug']
neg_freq = []
neg_avg  = []
for m in months:
    sm = sol[sol['month'] == m]
    neg_freq.append((sm['imb_settlement_price'] < 0).mean() * 100)
    neg_avg.append(sm.loc[sm['imb_settlement_price'] < 0, 'imb_settlement_price'].mean())

bars = ax.bar(mnames, neg_freq, color='#e74c3c', alpha=0.8, edgecolor='black')
for bar, val in zip(bars, neg_freq):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('% of Solar Hours with Negative Imb. Price', fontsize=11)
ax.set_title('Negative Imbalance Price Frequency\nby Month (2025 solar hours)', fontsize=12)

ax2 = ax.twinx()
ax2.plot(mnames, neg_avg, 'ko-', linewidth=2, markersize=7, label='Avg negative price')
ax2.set_ylabel('Avg Negative Price (EUR/MWh)', fontsize=11)
ax2.legend(fontsize=9, loc='upper right')

# Panel 3: Revenue when DA committed + don't produce vs alternatives
ax = axes[2]
neg_hours = sol[sol['imb_settlement_price'] < 0].copy()
if len(neg_hours) > 0:
    avg_da   = neg_hours['da_price'].mean()
    avg_imb  = neg_hours['imb_settlement_price'].mean()
    avg_both = avg_da - avg_imb  # da_price - imb (imb is negative, so this > da_price)

    scenarios = {
        'Produce\n(imb price)': avg_imb,
        'Curtail\n(0 revenue)': 0,
        'DA commit\n+ no-produce': avg_both,
    }
    colors_sc = ['#e74c3c', '#95a5a6', '#27ae60']
    bars2 = ax.bar(scenarios.keys(), scenarios.values(), color=colors_sc, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars2, scenarios.values()):
        ax.text(bar.get_x() + bar.get_width()/2,
                max(val, 0) + 2 if val >= 0 else val - 8,
                f'{val:.0f} EUR/MWh', ha='center', fontsize=11, fontweight='bold')
    ax.axhline(0, color='black', linewidth=1.5, alpha=0.5)
    ax.set_ylabel('Average Revenue (EUR/MWh)', fontsize=12)
    ax.set_title(f'Revenue Options During Negative Imb. Hours\n({len(neg_hours)} hours, avg DA = {avg_da:.0f} EUR/MWh)', fontsize=12)
    ax.set_ylim(min(avg_imb * 1.3, -50), avg_both * 1.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'dc_01_neg_imb_frequency.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] dc_01_neg_imb_frequency.png")


# ============================================================
# FIGURE 2: Sweep bid thresholds - oracle vs volume predictor
# ============================================================
print("[*] Figure 2: Bid threshold sweep...")

bid_range = np.arange(-20, 151, 1)

# Sweep surplus thresholds for volume predictor
surplus_thresholds = [0, 50, 100, 200, 500]  # MWh

oracle_results = []
for bid in bid_range:
    r = simulate(sol, bid, 0, use_oracle=True)
    r['bid'] = bid
    oracle_results.append(r)
oracle_df = pd.DataFrame(oracle_results)

vp_results = {}
for st in surplus_thresholds:
    rows = []
    for bid in bid_range:
        r = simulate(sol, bid, st, use_oracle=False)
        r['bid'] = bid
        rows.append(r)
    vp_results[st] = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Panel 1: Revenue curves
ax = axes[0]
ax.plot(oracle_df['bid'], oracle_df['rev_per_mwh'], 'k-', linewidth=3,
        label='Oracle (perfect foresight)', zorder=5)

colors_vp = ['#e74c3c', '#e67e22', '#f1c40f', '#27ae60', '#2980b9']
for st, color in zip(surplus_thresholds, colors_vp):
    df_vp = vp_results[st]
    lbl = f'Vol predictor >= {st} MWh surplus'
    ax.plot(df_vp['bid'], df_vp['rev_per_mwh'], color=color, linewidth=2,
            label=lbl, alpha=0.8)

# Old strategy reference (pure DA bid=0)
pure_da = (sol['da_price'] * sol['solar_mwh']).sum() / tv
ax.axhline(pure_da, color='gray', linestyle=':', linewidth=2,
           label=f'Old: pure DA = {pure_da:.1f} EUR/MWh')

# Old strategy best (hybrid curtail 50%, bid=58)
old_best = 62.2
ax.axhline(old_best, color='purple', linestyle='--', linewidth=1.5,
           label=f'Old best: hybrid curtail = {old_best:.1f} EUR/MWh')

# Mark optima
opt_oracle = oracle_df.loc[oracle_df['rev_per_mwh'].idxmax()]
ax.scatter([opt_oracle['bid']], [opt_oracle['rev_per_mwh']], color='black', s=150,
           zorder=6, edgecolors='white', linewidth=2)
ax.annotate(f'Oracle best:\n{opt_oracle["rev_per_mwh"]:.1f} EUR/MWh @ bid={opt_oracle["bid"]:.0f}',
            xy=(opt_oracle['bid'], opt_oracle['rev_per_mwh']),
            xytext=(opt_oracle['bid'] + 20, opt_oracle['rev_per_mwh'] - 5),
            fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', lw=1.5))

ax.set_xlabel('DA Bid Threshold (EUR/MWh)', fontsize=12)
ax.set_ylabel('Revenue per MWh Produced (EUR/MWh)', fontsize=12)
ax.set_title('DA Commit + No-Produce Strategy\nRevenue vs Bid Threshold (2025 Mar-Aug)', fontsize=13)
ax.legend(fontsize=9, loc='lower left')
ax.set_xlim(-20, 150)

# Panel 2: Hour breakdown at optimal oracle bid
ax = axes[1]
opt_bid_oracle = int(opt_oracle['bid'])
r_detail = simulate(sol, opt_bid_oracle, 0, use_oracle=True)

categories = ['DA + produce\n(committed, delivered)',
              'DA + no-produce\n(committed, short, imb<0)',
              'Imbalance produce\n(not committed, imb>0)',
              'Curtailed\n(not committed, imb<0)']
counts = [r_detail['n_da_prod'], r_detail['n_da_noprod'],
          r_detail['n_imb_prod'], r_detail['n_curtail']]
colors_cat = ['#27ae60', '#2ecc71', '#2980b9', '#e74c3c']
bars = ax.bar(categories, counts, color=colors_cat, edgecolor='black', alpha=0.85)
for bar, val in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(int(val)), ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('Number of Hours', fontsize=12)
ax.set_title(f'Hour Breakdown at Oracle Optimal\n(bid = {opt_bid_oracle} EUR, 2025 Mar-Aug)', fontsize=12)

plt.tight_layout()
plt.savefig(OUT_DIR / 'dc_02_bid_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] dc_02_bid_sweep.png")


# ============================================================
# FIGURE 3: Profitability of "no-produce" decision
# At what imbalance price does not-producing become profitable?
# And how does the volume predictor perform as a signal?
# ============================================================
print("[*] Figure 3: No-produce profitability analysis...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Panel 1: Revenue per hour as a function of imb_settlement_price
# For DA-sold hours, show produce vs no-produce revenue
ax = axes[0]
da_hours = sol[sol['da_price'] >= 0].copy()  # hours that would be sold at bid=0

imb_vals = np.linspace(-300, 200, 500)
avg_da = da_hours['da_price'].mean()

rev_produce    = avg_da * np.ones_like(imb_vals)             # DA price (produce, no imbalance)
rev_no_produce = avg_da - imb_vals                           # DA - imb_settle (short position)
rev_imb_only   = imb_vals                                    # no DA commitment, produce at imb

ax.plot(imb_vals, rev_produce,    '#27ae60', linewidth=2.5, label=f'DA committed + produce (={avg_da:.0f} EUR/MWh avg)')
ax.plot(imb_vals, rev_no_produce, '#e74c3c', linewidth=2.5, label='DA committed + no-produce')
ax.plot(imb_vals, rev_imb_only,   '#2980b9', linewidth=1.5, linestyle='--', label='No DA, produce at imbalance')
ax.fill_between(imb_vals, rev_produce, rev_no_produce,
                where=(imb_vals < 0), alpha=0.1, color='green',
                label='No-produce gains (imb < 0)')

ax.axvline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
ax.set_xlabel('Imbalance Settlement Price (EUR/MWh)', fontsize=12)
ax.set_ylabel('Revenue per MWh (EUR/MWh)', fontsize=12)
ax.set_title('Produce vs No-Produce: When Is It Worth It?\n(given DA sale at avg DA price)', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(-300, 200)
ax.set_ylim(-200, 300)

# Mark the crossover
ax.annotate('No-produce better\nwhen imb < 0', xy=(-50, avg_da + 50), fontsize=10,
            color='green', fontweight='bold')
ax.annotate('Produce better\nwhen imb > 0', xy=(80, avg_da - 15), fontsize=10,
            color='#27ae60', fontweight='bold')

# Panel 2: Volume predictor quality - imbalance_mwh vs actual negative price
ax = axes[1]
da_sold = sol[sol['da_price'] >= 0].copy()

scatter_neg = da_sold[da_sold['imb_settlement_price'] < 0]
scatter_pos = da_sold[da_sold['imb_settlement_price'] >= 0]

ax.scatter(scatter_pos['imbalance_mwh'], scatter_pos['imb_settlement_price'],
           alpha=0.2, s=12, color='#27ae60', label='Positive imb. price (produce)')
ax.scatter(scatter_neg['imbalance_mwh'], scatter_neg['imb_settlement_price'],
           alpha=0.5, s=18, color='#e74c3c', label='Negative imb. price (no-produce)')

# Add vertical lines at potential surplus thresholds
for thresh, color in zip([50, 100, 200], ['#f39c12', '#e67e22', '#c0392b']):
    ax.axvline(thresh, color=color, linestyle='--', alpha=0.7, linewidth=1.5,
               label=f'Surplus thresh = {thresh} MWh')
    # Precision at this threshold
    above = da_sold[da_sold['imbalance_mwh'] >= thresh]
    if len(above) > 0:
        precision = (above['imb_settlement_price'] < 0).mean() * 100
        ax.text(thresh + 5, -250, f'{precision:.0f}%\nneg', fontsize=9,
                color=color, fontweight='bold')

ax.axhline(0, color='black', linewidth=1.5, alpha=0.5)
ax.axvline(0, color='black', linewidth=1, alpha=0.3, linestyle=':')
ax.set_xlabel('System Imbalance (MWh, + = surplus)', fontsize=12)
ax.set_ylabel('Imbalance Settlement Price (EUR/MWh)', fontsize=12)
ax.set_title('Volume Predictor Signal Quality\n(% negative price above surplus threshold)', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(-800, 800)
ax.set_ylim(-400, 400)

plt.tight_layout()
plt.savefig(OUT_DIR / 'dc_03_noproduce_profitability.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] dc_03_noproduce_profitability.png")


# ============================================================
# FIGURE 4: Full strategy comparison table + sensitivity
# ============================================================
print("[*] Figure 4: Strategy comparison...")

strategies = {}

# Reference strategies
strategies['A: Pure DA (bid=0)'] = {
    'rev': (sol['da_price'] * sol['solar_mwh']).sum() / tv,
    'color': '#95a5a6'
}
strategies['B: Old hybrid (bid=58, curtail 50%)'] = {
    'rev': 62.2,
    'color': '#8e44ad'
}

# New strategy variants
for bid_val, label, color in [
    (0,  'C: New, bid=0, oracle',      '#e74c3c'),
    (10, 'D: New, bid=10, oracle',     '#e67e22'),
    (20, 'E: New, bid=20, oracle',     '#f39c12'),
    (30, 'F: New, bid=30, oracle',     '#f1c40f'),
]:
    r = simulate(sol, bid_val, 0, use_oracle=True)
    strategies[label] = {'rev': r['rev_per_mwh'], 'color': color}

# Volume predictor variants at bid=0
for st, color in [(0, '#2ecc71'), (100, '#27ae60'), (200, '#1abc9c')]:
    r = simulate(sol, 0, st, use_oracle=False)
    strategies[f'G: New, bid=0, VP>={st}MWh'] = {'rev': r['rev_per_mwh'], 'color': color}

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ax = axes[0]
labels_s  = list(strategies.keys())
revs      = [strategies[l]['rev'] for l in labels_s]
colors_s  = [strategies[l]['color'] for l in labels_s]
bars = ax.barh(labels_s, revs, color=colors_s, edgecolor='black', alpha=0.85, height=0.6)
for bar, val in zip(bars, revs):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f} EUR/MWh', va='center', fontsize=10, fontweight='bold')
ax.axvline(62.2, color='#8e44ad', linestyle='--', alpha=0.5, label='Old best: 62.2 EUR/MWh')
ax.set_xlabel('Revenue per MWh (EUR/MWh)', fontsize=12)
ax.set_title('Strategy Comparison: 2025 Mar-Aug\n(1MW E-W Solar)', fontsize=13)
ax.set_xlim(50, 95)
ax.legend(fontsize=9)

# Panel 2: Sweep bid_threshold for new strategy (volume predictor at various surplus thresholds)
ax = axes[1]
ax.plot(oracle_df['bid'], oracle_df['rev_per_mwh'], 'k-', linewidth=3,
        label='Oracle (ceiling)', zorder=5)
for st, color in zip([0, 100, 200, 500], ['#e74c3c', '#27ae60', '#2980b9', '#9b59b6']):
    df_vp = vp_results[st]
    opt = df_vp.loc[df_vp['rev_per_mwh'].idxmax()]
    ax.plot(df_vp['bid'], df_vp['rev_per_mwh'], color=color, linewidth=2,
            label=f'VP >= {st} MWh (best: {opt["rev_per_mwh"]:.1f} @ bid={opt["bid"]:.0f})')
    ax.scatter([opt['bid']], [opt['rev_per_mwh']], color=color, s=80,
               edgecolors='black', zorder=4)
ax.axhline(pure_da, color='gray', linestyle=':', label=f'Pure DA = {pure_da:.1f}')
ax.axhline(old_best, color='purple', linestyle='--', label=f'Old best = {old_best:.1f}')
ax.set_xlabel('DA Bid Threshold (EUR/MWh)', fontsize=12)
ax.set_ylabel('Revenue per MWh (EUR/MWh)', fontsize=12)
ax.set_title('New Strategy: Bid Sweep by Surplus Threshold\n(2025 Mar-Aug)', fontsize=13)
ax.legend(fontsize=9, loc='lower left')
ax.set_xlim(-20, 100)

plt.tight_layout()
plt.savefig(OUT_DIR / 'dc_04_strategy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] dc_04_strategy_comparison.png")


# ============================================================
# FIGURE 5: Seasonal breakdown - does the new strategy change
# which months are best?
# ============================================================
print("[*] Figure 5: Seasonal breakdown...")

months = [3,4,5,6,7,8]
mnames = ['Mar','Apr','May','Jun','Jul','Aug']

monthly = []
for m, mn in zip(months, mnames):
    sm = prepare(mkt, 2025, [m])
    if len(sm) < 10:
        continue
    tv_m = sm['solar_mwh'].sum()

    r_pda = (sm['da_price'] * sm['solar_mwh']).sum() / tv_m

    # Oracle: find optimal bid
    best_ora = {'bid': 0, 'rev': -1e9}
    for bid in np.arange(-20, 151, 2):
        r = simulate(sm, bid, 0, use_oracle=True)
        if r['rev_per_mwh'] > best_ora['rev']:
            best_ora = {'bid': bid, 'rev': r['rev_per_mwh']}

    # Volume predictor (surplus >= 100 MWh): find optimal bid
    best_vp = {'bid': 0, 'rev': -1e9}
    for bid in np.arange(-20, 151, 2):
        r = simulate(sm, bid, 100, use_oracle=False)
        if r['rev_per_mwh'] > best_vp['rev']:
            best_vp = {'bid': bid, 'rev': r['rev_per_mwh']}

    neg_freq = (sm['imb_settlement_price'] < 0).mean() * 100
    monthly.append({
        'month': m, 'name': mn,
        'rev_pure_da': r_pda,
        'rev_oracle': best_ora['rev'],
        'rev_vp100': best_vp['rev'],
        'opt_bid_oracle': best_ora['bid'],
        'opt_bid_vp100': best_vp['bid'],
        'neg_freq': neg_freq,
        'prod_mwh': tv_m,
    })

mb = pd.DataFrame(monthly)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Revenue comparison by month
ax = axes[0]
x = np.arange(len(mb))
w = 0.28
ax.bar(x - w,   mb['rev_pure_da'], w, label='Pure DA', color='#95a5a6', edgecolor='black', alpha=0.85)
ax.bar(x,       mb['rev_vp100'],   w, label='New (VP>=100, opt bid)', color='#27ae60', edgecolor='black', alpha=0.85)
ax.bar(x + w,   mb['rev_oracle'],  w, label='New (oracle)', color='#2980b9', edgecolor='black', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(mb['name'])
ax.set_ylabel('Revenue per MWh (EUR/MWh)', fontsize=12)
ax.set_title('Monthly Revenue Comparison\n(2025)', fontsize=12)
ax.legend(fontsize=9)

# Panel 2: Optimal DA bid
ax = axes[1]
ax.plot(mb['name'], mb['opt_bid_oracle'], 'ko-', linewidth=2, markersize=8, label='Oracle opt bid')
ax.plot(mb['name'], mb['opt_bid_vp100'],  'g^-', linewidth=2, markersize=8, label='VP>=100 opt bid')
ax.axhline(58, color='purple', linestyle='--', alpha=0.7, label='Old recommended (58)')
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_ylabel('Optimal DA Bid Threshold (EUR/MWh)', fontsize=12)
ax.set_title('Optimal Bid by Month\n(new DA commit + no-produce strategy)', fontsize=12)
ax.legend(fontsize=9)

# Panel 3: Negative imbalance price frequency
ax = axes[2]
bars3 = ax.bar(mb['name'], mb['neg_freq'], color='#e74c3c', alpha=0.8, edgecolor='black')
for bar, val in zip(bars3, mb['neg_freq']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.0f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('% Solar Hours with Negative Imb. Price', fontsize=12)
ax.set_title('Negative Imbalance Frequency by Month\n(2025, solar hours only)', fontsize=12)

plt.tight_layout()
plt.savefig(OUT_DIR / 'dc_05_seasonal.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] dc_05_seasonal.png")


# ============================================================
# PRINT SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("DA COMMIT + NO-PRODUCE STRATEGY - RESULTS")
print("=" * 70)

print(f"\nData: 2025 Mar-Aug, {len(sol)} solar hours, {tv:.0f} MWh")
print(f"\nNegative imbalance settlement price: {(sol['imb_settlement_price'] < 0).mean()*100:.1f}% of solar hours")
neg = sol[sol['imb_settlement_price'] < 0]
print(f"  Avg imb. price during negative hours: {neg['imb_settlement_price'].mean():.1f} EUR/MWh")
print(f"  Avg DA price during those hours:      {neg['da_price'].mean():.1f} EUR/MWh")
print(f"  Avg revenue if no-produce:            {(neg['da_price'] - neg['imb_settlement_price']).mean():.1f} EUR/MWh")

print("\n--- Revenue comparison (EUR/MWh) ---")
print(f"  Pure DA (bid=0, always produce):   {pure_da:.1f}")
print(f"  Old hybrid (curtail 50%, bid=58):   {old_best:.1f}")

opt_bid_oracle_global = int(oracle_df.loc[oracle_df['rev_per_mwh'].idxmax(), 'bid'])
opt_rev_oracle = oracle_df['rev_per_mwh'].max()
print(f"  New oracle (bid={opt_bid_oracle_global}, no-produce):  {opt_rev_oracle:.1f}")

for st in [0, 50, 100, 200]:
    df_vp = vp_results[st]
    opt = df_vp.loc[df_vp['rev_per_mwh'].idxmax()]
    print(f"  New VP>=  {st:3d}MWh (bid={opt['bid']:.0f}):         {opt['rev_per_mwh']:.1f}")

print(f"\n  Oracle uplift vs pure DA:  +{opt_rev_oracle - pure_da:.1f} EUR/MWh ({(opt_rev_oracle/pure_da-1)*100:.0f}%)")
print(f"  Oracle uplift vs old best: +{opt_rev_oracle - old_best:.1f} EUR/MWh")

print(f"\n--- Optimal DA bid threshold ---")
print(f"  Oracle:        {opt_bid_oracle_global} EUR/MWh (vs old 58 EUR/MWh)")
for st in [0, 100, 200]:
    df_vp = vp_results[st]
    opt = df_vp.loc[df_vp['rev_per_mwh'].idxmax()]
    print(f"  VP >= {st} MWh: {opt['bid']:.0f} EUR/MWh")

print(f"\n--- Hour breakdown at oracle optimum (bid={opt_bid_oracle_global}) ---")
r_detail = simulate(sol, opt_bid_oracle_global, 0, use_oracle=True)
print(f"  DA + produce:    {r_detail['n_da_prod']} h  (committed and delivered)")
print(f"  DA + no-produce: {r_detail['n_da_noprod']} h  (committed, short, avg imb = {r_detail['avg_noprod_imb']:.0f} EUR/MWh)")
print(f"  Imb + produce:   {r_detail['n_imb_prod']} h  (not on DA, positive imb)")
print(f"  Curtailed:       {r_detail['n_curtail']} h  (not on DA, negative imb - no benefit here)")

# Save summary CSV
rows_out = []
for bid_val in [0, 10, 20, 30, 40, 58]:
    r_o  = simulate(sol, bid_val, 0, use_oracle=True)
    r_v0 = simulate(sol, bid_val, 0,   use_oracle=False)
    r_v1 = simulate(sol, bid_val, 100, use_oracle=False)
    rows_out.append({
        'bid': bid_val,
        'oracle': round(r_o['rev_per_mwh'], 2),
        'vp_0mwh': round(r_v0['rev_per_mwh'], 2),
        'vp_100mwh': round(r_v1['rev_per_mwh'], 2),
    })
pd.DataFrame(rows_out).to_csv(OUT_DIR / 'dc_bid_sweep_summary.csv', index=False)

print("\n[+] All files saved to", OUT_DIR)
