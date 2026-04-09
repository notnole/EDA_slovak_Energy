"""
Visualize why the new DA-commit + no-produce strategy can't beat the old one.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent

mkt = pd.read_csv(OUT.parent.parent / "data" / "processed" / "hourly_market_prices.csv",
                   parse_dates=['timestamp_hour'])
sol = mkt[(mkt['timestamp_hour'].dt.year == 2025) & (mkt['month'].between(3, 8))].copy()
sol = sol.dropna(subset=['da_price', 'imb_settlement_price', 'imbalance_mwh'])

PVGIS = {1:24.9, 2:41.0, 3:78.9, 4:107.8, 5:123.6, 6:131.1,
         7:133.3, 8:114.2, 9:87.0, 10:55.2, 11:27.4, 12:19.2}
DAYS  = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
def _shape(hour, month):
    dl = {1:8.5,2:10,3:11.5,4:13.5,5:15,6:16,7:15.5,8:14,9:12.5,10:11,11:9,12:8}[month]
    sr, ss = 12 - dl/2, 12 + dl/2
    if hour < sr or hour >= ss: return 0.0
    sp = (hour - sr) / (ss - sr)
    irr = np.sin(sp * np.pi)
    ef = np.exp(-((sp - 0.30)**2) / (2*0.12**2))
    wf = np.exp(-((sp - 0.70)**2) / (2*0.12**2))
    return 0.5*irr*(0.4 + 0.6*ef) + 0.5*irr*(0.4 + 0.6*wf)
def solar_mwh(hour, month):
    rd = sum(_shape(h, month) for h in range(24))
    if rd <= 0: return 0.0
    return _shape(hour, month) * PVGIS[month] / (rd * DAYS[month])

sol['solar_mwh'] = sol.apply(lambda r: solar_mwh(r['hour'], r['month']), axis=1)
sol = sol[sol['solar_mwh'] > 0.001].copy()
tv = sol['solar_mwh'].sum()

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})

# ============================================================
# FIGURE 1: The correlation problem
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ax = axes[0]
sc = ax.scatter(sol['da_price'], sol['imbalance_mwh'],
                c=sol['imb_settlement_price'], cmap='RdYlGn_r',
                vmin=-150, vmax=150, s=8, alpha=0.6)
plt.colorbar(sc, ax=ax, label='Imb Settlement Price (EUR/MWh)', shrink=0.8)

ax.axhline(75, color='black', linewidth=2, linestyle='--', alpha=0.7)
ax.axvline(58, color='black', linewidth=2, linestyle='--', alpha=0.7)

# Count each quadrant
for x_center, y_center, x_cond, y_cond, label, color in [
    (150, 350, 'da_price >= 58', 'imbalance_mwh >= 75',
     'DA sold + surplus\n(NO-PRODUCE zone)', '#e74c3c'),
    (150, -350, 'da_price >= 58', 'imbalance_mwh < 75',
     'DA sold + deficit\n(normal produce)', '#27ae60'),
    (-60, 350, 'da_price < 58', 'imbalance_mwh >= 75',
     'Not on DA + surplus\n(curtailed)', '#f39c12'),
    (-60, -350, 'da_price < 58', 'imbalance_mwh < 75',
     'Not on DA + deficit\n(imb produce)', '#2980b9'),
]:
    n = len(sol.query(f'{x_cond} and {y_cond}'))
    ax.text(x_center, y_center, f'{label}\n({n} hours)', fontsize=10, fontweight='bold',
            color=color, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

ax.set_xlabel('DA Price (EUR/MWh)', fontsize=12)
ax.set_ylabel('System Imbalance Volume (MWh, + = surplus)', fontsize=12)
ax.set_title('The Correlation Problem\nDA price vs surplus volume, colored by imb settlement price',
             fontsize=12)
ax.set_xlim(-100, 250)
ax.set_ylim(-700, 700)

# Panel 2: DA price distribution in surplus vs deficit hours
ax = axes[1]
surplus_hours = sol[sol['imbalance_mwh'] >= 75]
deficit_hours = sol[sol['imbalance_mwh'] < 75]

ax.hist(deficit_hours['da_price'], bins=50, alpha=0.6, color='#2980b9',
        label=f'Deficit hours (imb_vol < 75): {len(deficit_hours)}', density=True)
ax.hist(surplus_hours['da_price'], bins=30, alpha=0.7, color='#e74c3c',
        label=f'Surplus hours (imb_vol >= 75): {len(surplus_hours)}', density=True)
ax.axvline(58, color='black', linewidth=2, linestyle='--', label='Old bid = 58 EUR')
ax.axvline(20, color='gray', linewidth=2, linestyle=':', label='Low bid = 20 EUR')

pct_above_58 = (surplus_hours['da_price'] >= 58).mean() * 100
pct_above_20 = (surplus_hours['da_price'] >= 20).mean() * 100
ax.text(0.98, 0.95,
        f'Surplus hours with DA >= 58: {pct_above_58:.0f}%\n'
        f'Surplus hours with DA >= 20: {pct_above_20:.0f}%',
        transform=ax.transAxes, fontsize=11, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('DA Price (EUR/MWh)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('DA Price Distribution: Surplus vs Deficit Hours\n'
             'Where are the surplus hours?', fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT / 'v2_04_correlation_problem.png', dpi=150, bbox_inches='tight')
plt.close()
print('[+] v2_04_correlation_problem.png')


# ============================================================
# FIGURE 2: What happens in the contested hours (DA 20-58)?
# ============================================================
contested = sol[(sol['da_price'] >= 20) & (sol['da_price'] < 58)].copy()

# Old strategy: not on DA (da < 58). Curtail if imb_vol >= 25, else produce at imbalance.
contested['rev_old'] = np.where(
    contested['imbalance_mwh'] >= 25, 0, contested['imb_settlement_price'])

# New strategy: on DA (da >= 20). No-produce if imb_vol >= 75, else produce at da_price.
contested['rev_new'] = np.where(
    contested['imbalance_mwh'] >= 75,
    contested['da_price'] - contested['imb_settlement_price'],
    contested['da_price'])

contested['delta'] = contested['rev_new'] - contested['rev_old']

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Panel 1: scatter rev_old vs rev_new
ax = axes[0, 0]
wins_new = contested['delta'] > 0
wins_old = contested['delta'] <= 0
ax.scatter(contested.loc[wins_old, 'rev_old'], contested.loc[wins_old, 'rev_new'],
           c='#e74c3c', s=15, alpha=0.5, label=f'Old wins ({wins_old.sum()} hours)')
ax.scatter(contested.loc[wins_new, 'rev_old'], contested.loc[wins_new, 'rev_new'],
           c='#27ae60', s=15, alpha=0.5, label=f'New wins ({wins_new.sum()} hours)')
ax.plot([-500, 500], [-500, 500], 'k--', alpha=0.3, label='Equal')
ax.set_xlabel('Revenue OLD strategy (EUR/MWh)', fontsize=12)
ax.set_ylabel('Revenue NEW strategy (EUR/MWh)', fontsize=12)
ax.set_title('Per-Hour Revenue: Old vs New\n(contested hours, DA price 20-58 EUR)', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)

# Panel 2: histogram of delta weighted by solar_mwh
ax = axes[0, 1]
ax.hist(contested['delta'], bins=60, weights=contested['solar_mwh'],
        color='#2980b9', alpha=0.7, edgecolor='white')
ax.axvline(0, color='black', linewidth=2)

total_gain = (contested.loc[wins_new, 'delta'] * contested.loc[wins_new, 'solar_mwh']).sum()
total_loss = (contested.loc[wins_old, 'delta'] * contested.loc[wins_old, 'solar_mwh']).sum()
net = total_gain + total_loss
ax.text(0.02, 0.95,
        f'New wins: +{total_gain:.0f} EUR\n'
        f'Old wins:  {total_loss:.0f} EUR\n'
        f'Net:       {net:+.0f} EUR',
        transform=ax.transAxes, fontsize=12, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Revenue difference NEW - OLD (EUR/MWh)', fontsize=12)
ax.set_ylabel('Solar MWh in bin', fontsize=12)
ax.set_title('Distribution of Gains/Losses\n(weighted by production)', fontsize=12)

# Panel 3: decomposition by surplus/deficit category
ax = axes[1, 0]
cats = {
    'Big surplus\n(vol >= 75)': contested['imbalance_mwh'] >= 75,
    'Moderate surplus\n(25 <= vol < 75)': (contested['imbalance_mwh'] >= 25) & (contested['imbalance_mwh'] < 75),
    'Small surplus\n(0 <= vol < 25)': (contested['imbalance_mwh'] >= 0) & (contested['imbalance_mwh'] < 25),
    'Deficit\n(vol < 0)': contested['imbalance_mwh'] < 0,
}
cat_names = list(cats.keys())
cat_gains = []
cat_counts = []
for name, mask in cats.items():
    sub = contested[mask]
    cat_gains.append((sub['delta'] * sub['solar_mwh']).sum())
    cat_counts.append(len(sub))

colors_bar = ['#27ae60' if g > 0 else '#e74c3c' for g in cat_gains]
bars = ax.barh(cat_names, cat_gains, color=colors_bar, alpha=0.8, edgecolor='black')
for bar, val, n in zip(bars, cat_gains, cat_counts):
    x_pos = val + (80 if val >= 0 else -80)
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f'{val:+.0f} EUR ({n}h)', va='center', fontsize=11, fontweight='bold')
ax.axvline(0, color='black', linewidth=1.5)
ax.set_xlabel('Net EUR gained by NEW strategy (negative = old wins)', fontsize=12)
ax.set_title('Where Does Each Strategy Win?\n(contested hours, DA 20-58 EUR)', fontsize=12)

# Panel 4: revenue profile sorted by imbalance settlement price
ax = axes[1, 1]
cs = contested.sort_values('imb_settlement_price').reset_index(drop=True)

ax.fill_between(range(len(cs)), 0, cs['rev_old'].values, alpha=0.3, color='#8e44ad',
                label='Old strategy')
ax.fill_between(range(len(cs)), 0, cs['rev_new'].values, alpha=0.3, color='#2980b9',
                label='New strategy')
ax.plot(range(len(cs)), cs['rev_old'].values, color='#8e44ad', linewidth=1, alpha=0.7)
ax.plot(range(len(cs)), cs['rev_new'].values, color='#2980b9', linewidth=1, alpha=0.7)
ax.axhline(0, color='black', linewidth=1)

ax.set_xlabel('Hours sorted by imbalance settlement price (low to high) -->', fontsize=12)
ax.set_ylabel('Revenue (EUR/MWh)', fontsize=12)
ax.set_title('Revenue Profile: Each Contested Hour\n(sorted by imbalance settlement price)',
             fontsize=12)
ax.legend(fontsize=10, loc='upper left')
ax.set_ylim(-250, 300)

plt.tight_layout()
plt.savefig(OUT / 'v2_05_why_new_loses.png', dpi=150, bbox_inches='tight')
plt.close()
print('[+] v2_05_why_new_loses.png')


# ============================================================
# FIGURE 3: Cumulative revenue across ALL hours
# ============================================================
# OLD: bid=58, curtail if imb_vol >= 25 for non-DA hours
sol['rev_old'] = np.where(
    sol['da_price'] >= 58,
    sol['da_price'],
    np.where(sol['imbalance_mwh'] >= 25, 0, sol['imb_settlement_price'])
)

# NEW: bid=20, DA no-produce at V_DA=75, non-DA curtail at V_noDA=25
sol['rev_new'] = np.where(
    sol['da_price'] >= 20,
    np.where(sol['imbalance_mwh'] >= 75,
             sol['da_price'] - sol['imb_settlement_price'],
             sol['da_price']),
    np.where(sol['imbalance_mwh'] >= 25, 0, sol['imb_settlement_price'])
)

fig, ax = plt.subplots(figsize=(16, 7))

ss = sol.sort_values('da_price').reset_index(drop=True)
ss['cum_old'] = (ss['rev_old'] * ss['solar_mwh']).cumsum()
ss['cum_new'] = (ss['rev_new'] * ss['solar_mwh']).cumsum()

ax.plot(ss['da_price'], ss['cum_old'], color='#8e44ad', linewidth=2.5,
        label=f'Old (bid=58, curtail V>=25): {ss["cum_old"].iloc[-1]/tv:.1f} EUR/MWh')
ax.plot(ss['da_price'], ss['cum_new'], color='#2980b9', linewidth=2.5,
        label=f'New (bid=20, DA no-prod V>=75): {ss["cum_new"].iloc[-1]/tv:.1f} EUR/MWh')

ax.fill_between(ss['da_price'], ss['cum_old'], ss['cum_new'],
                where=(ss['cum_new'] > ss['cum_old']),
                alpha=0.15, color='#2980b9', label='New ahead')
ax.fill_between(ss['da_price'], ss['cum_old'], ss['cum_new'],
                where=(ss['cum_new'] <= ss['cum_old']),
                alpha=0.15, color='#8e44ad', label='Old ahead')

ax.axvline(20, color='gray', linewidth=1.5, linestyle=':', alpha=0.7, label='New bid (20)')
ax.axvline(58, color='black', linewidth=1.5, linestyle='--', alpha=0.7, label='Old bid (58)')

ax.set_xlabel('DA Price (EUR/MWh) -- hours sorted low to high', fontsize=12)
ax.set_ylabel('Cumulative Revenue (EUR)', fontsize=12)
ax.set_title('Cumulative Revenue: Old vs New Strategy\n'
             '(each hour added left to right by DA price)', fontsize=13)
ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig(OUT / 'v2_06_cumulative_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('[+] v2_06_cumulative_comparison.png')


# ============================================================
# Summary numbers
# ============================================================
print()
print('=== CONTESTED HOURS (DA price 20-58 EUR) ===')
print(f'Total: {len(contested)} hours')
print()

deficit_h = contested[contested['imbalance_mwh'] < 0]
surplus_big = contested[contested['imbalance_mwh'] >= 75]
surplus_mod = contested[(contested['imbalance_mwh'] >= 25) & (contested['imbalance_mwh'] < 75)]

print('The story:')
print(f'  Deficit hours ({len(deficit_h)}):')
print(f'    Old earns avg {deficit_h["rev_old"].mean():.0f} EUR/MWh (imbalance settlement)')
print(f'    New earns avg {deficit_h["rev_new"].mean():.0f} EUR/MWh (DA price)')
print(f'    Loss per hour: {deficit_h["rev_new"].mean() - deficit_h["rev_old"].mean():.0f} EUR/MWh')
print()
print(f'  Big surplus hours ({len(surplus_big)}):')
print(f'    Old earns avg {surplus_big["rev_old"].mean():.0f} EUR/MWh (curtailed = 0)')
print(f'    New earns avg {surplus_big["rev_new"].mean():.0f} EUR/MWh (DA + |imb|)')
print(f'    Gain per hour: +{surplus_big["rev_new"].mean() - surplus_big["rev_old"].mean():.0f} EUR/MWh')
print()
print(f'  Moderate surplus ({len(surplus_mod)}):')
print(f'    Old earns avg {surplus_mod["rev_old"].mean():.0f} EUR/MWh (curtailed = 0)')
print(f'    New earns avg {surplus_mod["rev_new"].mean():.0f} EUR/MWh (DA price, produces)')
print(f'    Gain per hour: +{surplus_mod["rev_new"].mean() - surplus_mod["rev_old"].mean():.0f} EUR/MWh')
print()
print(f'  BUT: {len(deficit_h)} deficit hours x {abs(deficit_h["rev_new"].mean() - deficit_h["rev_old"].mean()):.0f} EUR loss')
print(f'    > {len(surplus_big)} surplus hours x {surplus_big["rev_new"].mean() - surplus_big["rev_old"].mean():.0f} EUR gain')
print(f'    The deficit losses outweigh the surplus gains.')
