"""
Jan 2026 Specific Direction Prediction

Focus ONLY on Jan 2026 where direction actually flips day-to-day.
Can we predict it?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data")
OUT_DIR = BASE / "MarketPriceGap" / "features" / "regulation_regime"

print("[*] Loading daily data...")
daily = pd.read_csv(OUT_DIR / 'data' / 'daily_direction.csv')
daily['date'] = pd.to_datetime(daily['date'])

# Focus on Jan 2026
jan26 = daily[daily['period'] == 'Jan 2026'].copy().reset_index(drop=True)
print(f"[+] Jan 2026: {len(jan26)} days")

# Also get last week of Dec 2025 for lag features
dec25 = daily[(daily['date'] >= '2025-12-25') & (daily['date'] < '2026-01-01')].copy()
combined = pd.concat([dec25, jan26], ignore_index=True).sort_values('date').reset_index(drop=True)

# Re-compute features for Jan 2026 only
combined['direction_yesterday'] = combined['direction'].shift(1)
combined['spread_yesterday'] = combined['spread_mean'].shift(1)
combined['reg_vol_yesterday'] = combined['reg_vol'].shift(1)

# Only keep Jan 2026 rows
jan26_pred = combined[combined['date'] >= '2026-01-01'].dropna(subset=['direction_yesterday']).copy()

print(f"[+] Jan 2026 prediction dataset: {len(jan26_pred)} days")

# ===== BASIC STATS =====
print("\n" + "="*60)
print("JAN 2026 DAILY DIRECTION ANALYSIS")
print("="*60)

print(f"\nDirection split:")
print(f"  Positive days: {jan26_pred['direction'].sum()} ({jan26_pred['direction'].mean()*100:.1f}%)")
print(f"  Negative days: {len(jan26_pred) - jan26_pred['direction'].sum()} ({(1-jan26_pred['direction'].mean())*100:.1f}%)")

# ===== TRANSITION PROBABILITIES =====
print("\n--- Jan 2026 Day-to-Day Transitions ---")

trans = pd.crosstab(
    jan26_pred['direction_yesterday'].map({0: 'Yesterday NEG', 1: 'Yesterday POS'}),
    jan26_pred['direction'].map({0: 'Today NEG', 1: 'Today POS'}),
    margins=True
)
print(trans)

trans_pct = pd.crosstab(
    jan26_pred['direction_yesterday'].map({0: 'Yesterday NEG', 1: 'Yesterday POS'}),
    jan26_pred['direction'].map({0: 'Today NEG', 1: 'Today POS'}),
    normalize='index'
) * 100
print("\nProbabilities:")
print(trans_pct.round(1))

# ===== PREDICTION RULES =====
print("\n--- Jan 2026 Prediction Rules ---")

rules = {}

# Rule 1: Follow yesterday
jan26_pred['pred_follow'] = jan26_pred['direction_yesterday']
rules['Follow yesterday'] = (jan26_pred['pred_follow'] == jan26_pred['direction']).mean()

# Rule 2: Contrarian
jan26_pred['pred_contr'] = 1 - jan26_pred['direction_yesterday']
rules['Contrarian'] = (jan26_pred['pred_contr'] == jan26_pred['direction']).mean()

# Rule 3: Always positive
rules['Always SELL IDM'] = jan26_pred['direction'].mean()

# Rule 4: Always negative
rules['Always BUY IDM'] = 1 - jan26_pred['direction'].mean()

# Rule 5: High reg volatility yesterday -> positive
median_reg = jan26_pred['reg_vol_yesterday'].median()
jan26_pred['pred_high_reg'] = (jan26_pred['reg_vol_yesterday'] > median_reg).astype(int)
rules['High reg vol -> positive'] = (jan26_pred['pred_high_reg'] == jan26_pred['direction']).mean()

# Rule 6: Low reg volatility -> negative
jan26_pred['pred_low_reg'] = (jan26_pred['reg_vol_yesterday'] <= median_reg).astype(int)
rules['Low reg vol -> positive'] = (jan26_pred['pred_low_reg'] == jan26_pred['direction']).mean()

# Rule 7: Large spread magnitude yesterday -> same direction
median_mag = jan26_pred['spread_yesterday'].abs().median()
jan26_pred['large_spread'] = jan26_pred['spread_yesterday'].abs() > median_mag
jan26_pred['pred_momentum'] = jan26_pred.apply(
    lambda x: x['direction_yesterday'] if x['large_spread'] else 1 - x['direction_yesterday'], axis=1
)
rules['Momentum (large -> follow)'] = (jan26_pred['pred_momentum'] == jan26_pred['direction']).mean()

# Rule 8: Weekday pattern (Mon-Fri positive, Sat-Sun negative)
jan26_pred['pred_weekday'] = (jan26_pred['dow'] < 5).astype(int)
rules['Weekday -> positive'] = (jan26_pred['pred_weekday'] == jan26_pred['direction']).mean()

for rule, acc in sorted(rules.items(), key=lambda x: -x[1]):
    status = "[+]" if acc > 0.55 else "[-]" if acc < 0.45 else "[~]"
    print(f"  {status} {rule}: {acc*100:.1f}%")

# ===== DAY BY DAY =====
print("\n--- Day-by-Day Results in Jan 2026 ---")
print("Date        | Spread  | Direction | Yesterday | Follow? | Reg Vol")
print("-" * 70)

for _, row in jan26_pred.iterrows():
    date_str = row['date'].strftime('%Y-%m-%d')
    spread = row['spread_mean']
    direction = 'POS' if row['direction'] == 1 else 'NEG'
    yesterday = 'POS' if row['direction_yesterday'] == 1 else 'NEG'
    follow = 'YES' if row['direction'] == row['direction_yesterday'] else 'NO'
    reg_vol = row['reg_vol']
    print(f"{date_str} | {spread:+7.1f} | {direction:^9} | {yesterday:^9} | {follow:^7} | {reg_vol:.1f}")

# ===== STREAK ANALYSIS =====
print("\n--- Streak Patterns in Jan 2026 ---")

# Count consecutive same-direction days
jan26_pred['same_as_yesterday'] = jan26_pred['direction'] == jan26_pred['direction_yesterday']
streaks = jan26_pred['same_as_yesterday'].value_counts()
print(f"  Days that followed yesterday: {streaks.get(True, 0)} ({streaks.get(True, 0)/len(jan26_pred)*100:.1f}%)")
print(f"  Days that reversed: {streaks.get(False, 0)} ({streaks.get(False, 0)/len(jan26_pred)*100:.1f}%)")

# ===== VISUALIZATION =====
print("\n[*] Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Daily spread with direction
ax1 = axes[0, 0]
colors = ['green' if d == 1 else 'red' for d in jan26_pred['direction']]
ax1.bar(range(len(jan26_pred)), jan26_pred['spread_mean'], color=colors, alpha=0.7)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax1.set_xticks(range(0, len(jan26_pred), 2))
ax1.set_xticklabels([jan26_pred.iloc[i]['date'].strftime('%d') for i in range(0, len(jan26_pred), 2)], rotation=45)
ax1.set_xlabel('Day of January 2026')
ax1.set_ylabel('Daily Avg Spread (EUR/MWh)')
ax1.set_title('Jan 2026: Daily Spread Direction (Green=Pos, Red=Neg)')

# Plot 2: Transition heatmap
ax2 = axes[0, 1]
trans_data = trans_pct.values[:2, :2]
im = ax2.imshow(trans_data, cmap='RdYlGn', vmin=0, vmax=100)
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Negative', 'Positive'])
ax2.set_yticklabels(['Negative', 'Positive'])
ax2.set_xlabel('Today')
ax2.set_ylabel('Yesterday')
ax2.set_title('Jan 2026: Transition Probabilities')
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f'{trans_data[i][j]:.0f}%', ha='center', va='center', fontsize=16, fontweight='bold')
plt.colorbar(im, ax=ax2)

# Plot 3: Prediction rules
ax3 = axes[1, 0]
rules_sorted = sorted(rules.items(), key=lambda x: -x[1])
rule_names = [r[0] for r in rules_sorted]
rule_accs = [r[1]*100 for r in rules_sorted]
colors = ['green' if a > 55 else 'red' if a < 45 else 'gray' for a in rule_accs]
ax3.barh(rule_names, rule_accs, color=colors, alpha=0.7)
ax3.axvline(x=50, color='black', linestyle='--', alpha=0.7, label='Random (50%)')
ax3.set_xlabel('Accuracy (%)')
ax3.set_title('Jan 2026: Prediction Rule Performance')
ax3.set_xlim(30, 70)
ax3.legend()

# Plot 4: Cumulative P&L for different strategies
ax4 = axes[1, 1]

# Strategy 1: Always Sell IDM
jan26_pred['pnl_always_sell'] = jan26_pred['spread_mean']
jan26_pred['cum_pnl_always_sell'] = jan26_pred['pnl_always_sell'].cumsum()

# Strategy 2: Follow yesterday
jan26_pred['pnl_follow'] = jan26_pred.apply(
    lambda x: x['spread_mean'] if x['direction_yesterday'] == 1 else -x['spread_mean'], axis=1
)
jan26_pred['cum_pnl_follow'] = jan26_pred['pnl_follow'].cumsum()

# Strategy 3: Contrarian
jan26_pred['pnl_contr'] = jan26_pred.apply(
    lambda x: x['spread_mean'] if x['direction_yesterday'] == 0 else -x['spread_mean'], axis=1
)
jan26_pred['cum_pnl_contr'] = jan26_pred['pnl_contr'].cumsum()

ax4.plot(range(len(jan26_pred)), jan26_pred['cum_pnl_always_sell'], 'b-', linewidth=2, label=f'Always Sell ({jan26_pred["cum_pnl_always_sell"].iloc[-1]:+.0f} EUR)')
ax4.plot(range(len(jan26_pred)), jan26_pred['cum_pnl_follow'], 'g-', linewidth=2, label=f'Follow Yesterday ({jan26_pred["cum_pnl_follow"].iloc[-1]:+.0f} EUR)')
ax4.plot(range(len(jan26_pred)), jan26_pred['cum_pnl_contr'], 'r-', linewidth=2, label=f'Contrarian ({jan26_pred["cum_pnl_contr"].iloc[-1]:+.0f} EUR)')
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax4.set_xlabel('Day')
ax4.set_ylabel('Cumulative P&L (EUR/MWh)')
ax4.set_title('Jan 2026: Strategy Comparison (per 1 MWh traded daily)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / '04_jan2026_direction.png', dpi=150, bbox_inches='tight')
print(f"[+] Saved: 04_jan2026_direction.png")

# ===== SUMMARY =====
print("\n" + "="*60)
print("SUMMARY: Jan 2026 Direction Prediction")
print("="*60)

best_rule = max(rules.items(), key=lambda x: x[1])
print(f"\nBest rule: '{best_rule[0]}' with {best_rule[1]*100:.1f}% accuracy")

print("\n--- Key Finding ---")
if best_rule[1] > 0.55:
    print(f"[+] There IS a predictable pattern!")
    print(f"    Using '{best_rule[0]}': {best_rule[1]*100:.1f}% accuracy")
else:
    print("[-] Direction is essentially UNPREDICTABLE in Jan 2026")
    print("    Best rule barely beats random (50%)")

print("\n--- P&L Comparison (per 1 MWh traded daily) ---")
print(f"  Always Sell IDM: {jan26_pred['cum_pnl_always_sell'].iloc[-1]:+.1f} EUR")
print(f"  Follow Yesterday: {jan26_pred['cum_pnl_follow'].iloc[-1]:+.1f} EUR")
print(f"  Contrarian: {jan26_pred['cum_pnl_contr'].iloc[-1]:+.1f} EUR")

winner = max([
    ('Always Sell', jan26_pred['cum_pnl_always_sell'].iloc[-1]),
    ('Follow Yesterday', jan26_pred['cum_pnl_follow'].iloc[-1]),
    ('Contrarian', jan26_pred['cum_pnl_contr'].iloc[-1])
], key=lambda x: x[1])

print(f"\n  Best strategy: {winner[0]} with {winner[1]:+.1f} EUR")

print("\n[+] Analysis complete!")
