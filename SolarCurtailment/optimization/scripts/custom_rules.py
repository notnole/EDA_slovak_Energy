"""
Backtest custom curtailment rules.

Rules:
  1. pred > 30         -> CURTAIL (always, extreme surplus)
  2. DA < 3            -> CURTAIL unless pred < 0 (deficit)
  3. DA 3-20, pred < 8 -> PRODUCE
  4. DA >= 20          -> PRODUCE (always, good DA price)
  5. DA 3-20, pred >= 8 -> CURTAIL (implied by rules above)

Simplified logic:
  PRODUCE if:
    - pred <= 0 (deficit, override everything)
    - DA >= 20 AND pred <= 30
    - DA >= 3 AND pred < 8 AND pred <= 30
  CURTAIL otherwise (i.e. pred > 30, or DA < 3 with pred >= 0, or DA 3-20 with pred >= 8)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]

# Solar model
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
    if rd <= 0: return 0.0
    return _shape(hour, month) * PVGIS[month] / (rd * DAYS[month])

# Load data
mkt = pd.read_csv(ROOT / 'MarketPriceGap/data/processed/qh_market_prices.csv', parse_dates=['timestamp_qh'])
df = mkt[(mkt['timestamp_qh'].dt.year == 2025) & (mkt['timestamp_qh'].dt.month.between(3, 8))].copy()
df = df.dropna(subset=['da_price', 'imb_settlement_price', 'imbalance_mwh'])
df['month'] = df['timestamp_qh'].dt.month
df['hour'] = df['timestamp_qh'].dt.hour
df['solar_mwh'] = df.apply(lambda r: solar_mwh(r['hour'], r['month']), axis=1)
df = df[df['solar_mwh'] > 0.001].copy()

np.random.seed(42)
df['pred'] = df['imbalance_mwh'] + np.random.normal(0, 9.0, len(df))

total_solar = df['solar_mwh'].sum()
n = len(df)
settle = df['imb_settlement_price'].values
sol = df['solar_mwh'].values
pred = df['pred'].values
da = df['da_price'].values

print(f"[*] Data: {n} solar periods, {total_solar:.0f} MWh")

# ============================================================
# DEFINE STRATEGIES
# ============================================================

# Custom rules
custom_produce = (
    (pred <= 0) |                              # deficit -> always produce
    ((da >= 20) & (pred <= 30)) |              # good DA, not extreme surplus
    ((da >= 3) & (pred < 8) & (pred <= 30))    # moderate DA, low surplus pred
)
custom_curtail = ~custom_produce

# Grid search best (R1+R2+Floor): X=34, A=10, Y=5, F=-6
grid_raw = (pred > 34) | ((da < 10) & (pred > 5))
grid_curtail = grid_raw & (pred > -6)

# Current
current_curtail = pred > 15

# Oracle
oracle_curtail = settle < 0

strategies = {
    'No curtailment':   np.zeros(n, dtype=bool),
    'Current (pred>15)': current_curtail,
    'Grid search best':  grid_curtail,
    'Custom rules':      custom_curtail,
    'Oracle':            oracle_curtail,
}

print(f"\n{'Strategy':<25} {'EUR/MWh':>8} {'Curtail%':>8} {'Uplift':>8}")
print("-" * 55)
for name, mask in strategies.items():
    produce = ~mask
    rev = (settle[produce] * sol[produce]).sum() / total_solar
    pct = mask.mean() * 100
    print(f"{name:<25} {rev:>8.1f} {pct:>7.1f}% {rev - 50.5:>+8.1f}")

# ============================================================
# DETAILED BREAKDOWN OF CUSTOM RULES
# ============================================================
print("\n--- Custom Rules Breakdown ---")

# Categorize each period
cats = np.full(n, '', dtype=object)
cats[pred > 30] = 'R1: pred>30 CURTAIL'
cats[(cats == '') & (da < 3) & (pred >= 0)] = 'R2: DA<3 surplus CURTAIL'
cats[(cats == '') & (da < 3) & (pred < 0)] = 'R2: DA<3 deficit PRODUCE'
cats[(cats == '') & (da >= 3) & (da < 20) & (pred < 8)] = 'R3: DA 3-20 pred<8 PRODUCE'
cats[(cats == '') & (da >= 3) & (da < 20) & (pred >= 8)] = 'R3: DA 3-20 pred>=8 CURTAIL'
cats[(cats == '') & (da >= 20)] = 'R4: DA>=20 PRODUCE'

df['category'] = cats
for cat in sorted(df['category'].unique()):
    sub = df[df['category'] == cat]
    avg_settle = sub['imb_settlement_price'].mean()
    pct_neg = (sub['imb_settlement_price'] < 0).mean() * 100
    action = 'CURTAIL' if 'CURTAIL' in cat else 'PRODUCE'
    avoided = sub['imb_settlement_price'].mean() if action == 'CURTAIL' else None
    print(f"  {cat}: {len(sub)} periods, avg settle={avg_settle:+.1f}, {pct_neg:.0f}% neg"
          f"{f', avoided={avoided:+.1f}' if avoided is not None else ''}")

# ============================================================
# PLOTS
# ============================================================
plt.rcParams.update({'figure.figsize': (14, 7), 'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Panel 1: Decision space
ax = axes[0]
sc = ax.scatter(pred, da, c=settle, cmap='RdYlGn', s=8, alpha=0.5, vmin=-100, vmax=200)
plt.colorbar(sc, ax=ax, label='Settlement Price (EUR/MWh)', shrink=0.8)

# Draw rule boundaries
ax.axvline(30, color='red', lw=2.5, ls='--', label='pred>30: CURTAIL')
ax.axhline(3, color='purple', lw=2, ls='--', label='DA=3 boundary')
ax.axhline(20, color='blue', lw=2, ls='--', label='DA=20 boundary')
ax.axvline(8, color='orange', lw=2, ls='--', label='pred=8 (DA 3-20)')
ax.axvline(0, color='green', lw=2, ls='--', label='pred=0 (deficit line)')

# Shade curtailment zones
# R1: pred > 30
ax.axvspan(30, pred.max() + 5, alpha=0.15, color='red')
# R2: DA < 3, pred >= 0
ax.fill_between([0, 30], [0, 0], [3, 3], alpha=0.15, color='purple', label='DA<3 surplus: CURTAIL')
# R3: DA 3-20, pred >= 8
ax.fill_between([8, 30], [3, 3], [20, 20], alpha=0.15, color='orange', label='DA 3-20 pred>=8: CURTAIL')
# Produce zones
ax.fill_between([pred.min()-5, 0], [0, 0], [3, 3], alpha=0.10, color='green', label='DA<3 deficit: PRODUCE')

ax.set_xlabel('Predicted Imbalance (MWh)', fontsize=12)
ax.set_ylabel('DA Price (EUR/MWh)', fontsize=12)
ax.set_title('Custom Rules Decision Space\n(color = actual settlement price)', fontsize=13)
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(-80, 70)
ax.set_ylim(-50, 250)

# Panel 2: Strategy comparison bar chart
ax = axes[1]
names = []
revs = []
for name, mask in strategies.items():
    produce = ~mask
    rev = (settle[produce] * sol[produce]).sum() / total_solar
    names.append(name)
    revs.append(rev)

colors = ['#95a5a6', '#3498db', '#e74c3c', '#8e44ad', '#27ae60']
bars = ax.barh(names, revs, color=colors, edgecolor='black', linewidth=0.5)
for bar, rev in zip(bars, revs):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{rev:.1f}', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Revenue (EUR/MWh)', fontsize=12)
ax.set_title('Strategy Comparison\n2025 Mar-Aug, Simulated Predictions', fontsize=13)
ax.set_xlim(0, max(revs) * 1.15)

plt.tight_layout()
plt.savefig(OUT / '06_custom_rules.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[+] 06_custom_rules.png")

# Cumulative PnL
fig, ax = plt.subplots(figsize=(14, 7))
df_s = df.sort_values('timestamp_qh')
s_s = df_s['imb_settlement_price'].values
p_s = df_s['pred'].values
d_s = df_s['da_price'].values
sol_s = df_s['solar_mwh'].values
dates = df_s['timestamp_qh'].values

custom_p = (p_s <= 0) | ((d_s >= 20) & (p_s <= 30)) | ((d_s >= 3) & (p_s < 8) & (p_s <= 30))
strats_cum = {
    'No curtailment': np.ones(len(df_s), dtype=bool),
    'Current (pred>15)': p_s <= 15,
    'Grid search': ~(((p_s > 34) | ((d_s < 10) & (p_s > 5))) & (p_s > -6)),
    'Custom rules': custom_p,
    'Oracle': s_s >= 0,
}
colors_cum = ['#95a5a6', '#3498db', '#e74c3c', '#8e44ad', '#27ae60']
for (name, produce), color in zip(strats_cum.items(), colors_cum):
    pnl = np.where(produce, s_s * sol_s, 0)
    ax.plot(dates, np.cumsum(pnl), label=name, linewidth=1.8, color=color)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Revenue (EUR)', fontsize=12)
ax.set_title('Cumulative Revenue: Custom Rules vs Others', fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / '07_custom_cumulative.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 07_custom_cumulative.png")
