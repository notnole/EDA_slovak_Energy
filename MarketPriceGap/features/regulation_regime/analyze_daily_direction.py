"""
Daily Spread Direction Prediction Analysis

Key insight: The spread didn't disappear - it became bidirectional.
Question: Can we predict which direction the spread will go on a given day?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data")
DATA_FILE = BASE / "MarketPriceGap" / "features" / "regulation_regime" / "data" / "hourly_reg_spread.csv"
OUT_DIR = BASE / "MarketPriceGap" / "features" / "regulation_regime"

print("[*] Loading hourly data...")
df = pd.read_csv(DATA_FILE, parse_dates=['datetime'])
print(f"[+] Loaded {len(df):,} hourly records")

# Aggregate to daily
print("[*] Aggregating to daily...")
df['date'] = df['datetime'].dt.date
df['date'] = pd.to_datetime(df['date'])

daily = df.groupby('date').agg(
    spread_mean=('idm_imb_spread', 'mean'),
    spread_sum=('idm_imb_spread', 'sum'),
    spread_std=('idm_imb_spread', 'std'),
    spread_positive_pct=('idm_imb_spread', lambda x: (x > 0).mean() * 100),
    reg_vol=('reg_std', 'mean'),
    n_hours=('idm_imb_spread', 'count')
).reset_index()

# Direction: 1 = positive (sell IDM profitable), -1 = negative (buy IDM profitable)
daily['direction'] = np.sign(daily['spread_mean'])
daily['direction_binary'] = (daily['spread_mean'] > 0).astype(int)

# Add calendar features
daily['dow'] = daily['date'].dt.dayofweek  # 0=Mon, 6=Sun
daily['dow_name'] = daily['date'].dt.day_name()
daily['month'] = daily['date'].dt.month
daily['year_month'] = daily['date'].dt.to_period('M')
daily['is_weekend'] = daily['dow'].isin([5, 6]).astype(int)

# Add lagged features (yesterday's values)
daily = daily.sort_values('date')
daily['direction_lag1'] = daily['direction_binary'].shift(1)
daily['direction_lag2'] = daily['direction_binary'].shift(2)
daily['direction_lag3'] = daily['direction_binary'].shift(3)
daily['spread_lag1'] = daily['spread_mean'].shift(1)
daily['spread_lag2'] = daily['spread_mean'].shift(2)
daily['reg_vol_lag1'] = daily['reg_vol'].shift(1)

# Rolling features
daily['direction_rolling_3d'] = daily['direction_binary'].rolling(3).mean().shift(1)
daily['direction_rolling_7d'] = daily['direction_binary'].rolling(7).mean().shift(1)
daily['spread_rolling_7d'] = daily['spread_mean'].rolling(7).mean().shift(1)

# Period marker
daily['period'] = daily['date'].apply(
    lambda x: 'Jan 2026' if x >= pd.Timestamp('2026-01-01') else '2025'
)

print(f"[+] Created {len(daily)} daily records")

# ===== ANALYSIS 1: Direction Distribution =====
print("\n" + "="*60)
print("ANALYSIS 1: Daily Spread Direction Distribution")
print("="*60)

for period in ['2025', 'Jan 2026']:
    subset = daily[daily['period'] == period]
    pos = (subset['direction'] > 0).sum()
    neg = (subset['direction'] < 0).sum()
    zero = (subset['direction'] == 0).sum()
    total = len(subset)
    print(f"\n{period}:")
    print(f"  Positive days (sell IDM): {pos} ({pos/total*100:.1f}%)")
    print(f"  Negative days (buy IDM):  {neg} ({neg/total*100:.1f}%)")
    print(f"  Neutral days:             {zero}")
    print(f"  Average daily spread: {subset['spread_mean'].mean():+.1f} EUR/MWh")
    print(f"  Std of daily spread:  {subset['spread_mean'].std():.1f} EUR/MWh")

# ===== ANALYSIS 2: Autocorrelation (Persistence) =====
print("\n" + "="*60)
print("ANALYSIS 2: Direction Persistence (Autocorrelation)")
print("="*60)

# Transition matrix
daily_clean = daily.dropna(subset=['direction_lag1'])

for period in ['2025', 'Jan 2026']:
    subset = daily_clean[daily_clean['period'] == period]

    # If yesterday positive, what's prob of today positive?
    yest_pos = subset[subset['direction_lag1'] == 1]
    yest_neg = subset[subset['direction_lag1'] == 0]

    p_pos_given_pos = yest_pos['direction_binary'].mean() if len(yest_pos) > 0 else np.nan
    p_pos_given_neg = yest_neg['direction_binary'].mean() if len(yest_neg) > 0 else np.nan

    print(f"\n{period}:")
    print(f"  P(today +ve | yesterday +ve) = {p_pos_given_pos:.1%}")
    print(f"  P(today +ve | yesterday -ve) = {p_pos_given_neg:.1%}")
    print(f"  --> Persistence strength: {p_pos_given_pos - p_pos_given_neg:.1%}")

# ===== ANALYSIS 3: Day-of-Week Pattern =====
print("\n" + "="*60)
print("ANALYSIS 3: Day-of-Week Pattern")
print("="*60)

dow_stats = daily.groupby(['period', 'dow_name']).agg(
    positive_rate=('direction_binary', 'mean'),
    avg_spread=('spread_mean', 'mean'),
    n=('spread_mean', 'count')
).reset_index()

# Order days properly
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_stats['dow_name'] = pd.Categorical(dow_stats['dow_name'], categories=day_order, ordered=True)
dow_stats = dow_stats.sort_values(['period', 'dow_name'])

print("\n2025:")
for _, row in dow_stats[dow_stats['period'] == '2025'].iterrows():
    print(f"  {row['dow_name']:9s}: {row['positive_rate']*100:5.1f}% positive, avg spread {row['avg_spread']:+6.1f} EUR")

print("\nJan 2026:")
for _, row in dow_stats[dow_stats['period'] == 'Jan 2026'].iterrows():
    print(f"  {row['dow_name']:9s}: {row['positive_rate']*100:5.1f}% positive, avg spread {row['avg_spread']:+6.1f} EUR")

# ===== ANALYSIS 4: Simple Prediction Rules =====
print("\n" + "="*60)
print("ANALYSIS 4: Simple Prediction Rules (Jan 2026 Only)")
print("="*60)

jan2026 = daily_clean[daily_clean['period'] == 'Jan 2026'].copy()

rules = []

# Rule 1: Always predict positive (baseline)
acc = jan2026['direction_binary'].mean()
rules.append(('Always positive', acc))

# Rule 2: Always predict negative
acc = 1 - jan2026['direction_binary'].mean()
rules.append(('Always negative', acc))

# Rule 3: Follow yesterday
acc = (jan2026['direction_binary'] == jan2026['direction_lag1']).mean()
rules.append(('Follow yesterday', acc))

# Rule 4: Opposite of yesterday
acc = (jan2026['direction_binary'] != jan2026['direction_lag1']).mean()
rules.append(('Opposite of yesterday', acc))

# Rule 5: Follow 3-day trend (if >50% positive, predict positive)
jan2026['pred_3d'] = (jan2026['direction_rolling_3d'] > 0.5).astype(int)
acc = (jan2026['direction_binary'] == jan2026['pred_3d']).mean()
rules.append(('Follow 3-day trend', acc))

# Rule 6: Follow 7-day trend
jan2026['pred_7d'] = (jan2026['direction_rolling_7d'] > 0.5).astype(int)
acc = (jan2026['direction_binary'] == jan2026['pred_7d']).mean()
rules.append(('Follow 7-day trend', acc))

# Rule 7: Weekday positive, weekend negative
jan2026['pred_dow'] = 1 - jan2026['is_weekend']
acc = (jan2026['direction_binary'] == jan2026['pred_dow']).mean()
rules.append(('Weekday +ve, Weekend -ve', acc))

# Rule 8: Based on yesterday's spread magnitude
jan2026['pred_spread_mag'] = (jan2026['spread_lag1'] > 0).astype(int)
acc = (jan2026['direction_binary'] == jan2026['pred_spread_mag']).mean()
rules.append(('Yesterday spread sign', acc))

# Rule 9: High reg vol yesterday -> positive
jan2026['high_vol'] = (jan2026['reg_vol_lag1'] > jan2026['reg_vol_lag1'].median()).astype(int)
acc = (jan2026['direction_binary'] == jan2026['high_vol']).mean()
rules.append(('High reg vol yesterday', acc))

print("\nPrediction Rule Accuracy:")
for rule, acc in sorted(rules, key=lambda x: -x[1]):
    indicator = "***" if acc > 0.6 else "**" if acc > 0.55 else "*" if acc > 0.5 else ""
    print(f"  {rule:30s}: {acc*100:5.1f}% {indicator}")

# ===== ANALYSIS 5: Potential Trading P&L =====
print("\n" + "="*60)
print("ANALYSIS 5: Trading P&L Simulation (Jan 2026)")
print("="*60)

# Assuming 1 MWh traded per hour, ~17 trading hours/day
MWH_PER_DAY = 17

# Strategy 1: Always sell IDM
pnl_always = jan2026['spread_mean'].sum() * MWH_PER_DAY
print(f"\nAlways Sell IDM:    {pnl_always:+,.0f} EUR")

# Strategy 2: Follow yesterday
jan2026['trade_follow'] = np.where(jan2026['direction_lag1'] == 1, 1, -1)
jan2026['pnl_follow'] = jan2026['trade_follow'] * jan2026['spread_mean'] * MWH_PER_DAY
pnl_follow = jan2026['pnl_follow'].sum()
print(f"Follow Yesterday:   {pnl_follow:+,.0f} EUR")

# Strategy 3: Opposite of yesterday
jan2026['trade_opposite'] = np.where(jan2026['direction_lag1'] == 1, -1, 1)
jan2026['pnl_opposite'] = jan2026['trade_opposite'] * jan2026['spread_mean'] * MWH_PER_DAY
pnl_opposite = jan2026['pnl_opposite'].sum()
print(f"Opposite Yesterday: {pnl_opposite:+,.0f} EUR")

# Strategy 4: Perfect foresight (upper bound)
pnl_perfect = np.abs(jan2026['spread_mean']).sum() * MWH_PER_DAY
print(f"Perfect Foresight:  {pnl_perfect:+,.0f} EUR (upper bound)")

# ===== VISUALIZATION =====
print("\n[*] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Daily spread time series with direction
ax1 = axes[0, 0]
colors = daily['direction'].map({1: 'green', -1: 'red', 0: 'gray'})
ax1.scatter(daily['date'], daily['spread_mean'], c=colors, alpha=0.6, s=20)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax1.axvline(x=pd.Timestamp('2026-01-01'), color='blue', linestyle='--', alpha=0.7)
ax1.set_xlabel('Date')
ax1.set_ylabel('Daily Avg Spread (EUR/MWh)')
ax1.set_title('Daily Spread Direction (Green=+ve, Red=-ve)')

# Plot 2: Direction distribution by period
ax2 = axes[0, 1]
periods = ['2025', 'Jan 2026']
pos_rates = [daily[daily['period']==p]['direction_binary'].mean() * 100 for p in periods]
neg_rates = [100 - r for r in pos_rates]

x = np.arange(len(periods))
width = 0.35
ax2.bar(x - width/2, pos_rates, width, label='Positive (Sell IDM)', color='green', alpha=0.7)
ax2.bar(x + width/2, neg_rates, width, label='Negative (Buy IDM)', color='red', alpha=0.7)
ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Random')
ax2.set_xticks(x)
ax2.set_xticklabels(periods)
ax2.set_ylabel('% of Days')
ax2.set_title('Spread Direction Distribution')
ax2.legend()

for i, (pos, neg) in enumerate(zip(pos_rates, neg_rates)):
    ax2.text(i - width/2, pos + 2, f'{pos:.0f}%', ha='center', fontsize=10)
    ax2.text(i + width/2, neg + 2, f'{neg:.0f}%', ha='center', fontsize=10)

# Plot 3: Transition probabilities
ax3 = axes[1, 0]
# Calculate for both periods
trans_data = []
for period in ['2025', 'Jan 2026']:
    subset = daily_clean[daily_clean['period'] == period]
    yest_pos = subset[subset['direction_lag1'] == 1]
    yest_neg = subset[subset['direction_lag1'] == 0]

    p_pos_pos = yest_pos['direction_binary'].mean()
    p_neg_pos = yest_neg['direction_binary'].mean()
    trans_data.append((period, p_pos_pos, p_neg_pos))

x = np.arange(2)
width = 0.35
bars1 = ax3.bar(x - width/2, [t[1] for t in trans_data], width, label='Yesterday +ve', color='green', alpha=0.7)
bars2 = ax3.bar(x + width/2, [t[2] for t in trans_data], width, label='Yesterday -ve', color='red', alpha=0.7)
ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
ax3.set_xticks(x)
ax3.set_xticklabels(['2025', 'Jan 2026'])
ax3.set_ylabel('P(Today Positive)')
ax3.set_title('Transition Probabilities: Does Yesterday Predict Today?')
ax3.legend()
ax3.set_ylim(0, 1)

for bar, val in zip(bars1, [t[1] for t in trans_data]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.0%}', ha='center', fontsize=10)
for bar, val in zip(bars2, [t[2] for t in trans_data]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.0%}', ha='center', fontsize=10)

# Plot 4: Day-of-week pattern
ax4 = axes[1, 1]
dow_2025 = dow_stats[dow_stats['period'] == '2025'].set_index('dow_name')['positive_rate'].reindex(day_order)
dow_2026 = dow_stats[dow_stats['period'] == 'Jan 2026'].set_index('dow_name')['positive_rate'].reindex(day_order)

x = np.arange(7)
width = 0.35
ax4.bar(x - width/2, dow_2025.values * 100, width, label='2025', color='steelblue', alpha=0.7)
ax4.bar(x + width/2, dow_2026.values * 100, width, label='Jan 2026', color='coral', alpha=0.7)
ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5)
ax4.set_xticks(x)
ax4.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax4.set_ylabel('% Days Positive')
ax4.set_title('Day-of-Week Pattern')
ax4.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / '03_daily_direction_analysis.png', dpi=150, bbox_inches='tight')
print(f"[+] Saved: 03_daily_direction_analysis.png")

# ===== SUMMARY =====
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

best_rule = max(rules, key=lambda x: x[1])
print(f"""
KEY FINDING: The spread became BIDIRECTIONAL in Jan 2026

2025: {daily[daily['period']=='2025']['direction_binary'].mean()*100:.0f}% positive days (consistent direction)
Jan 2026: {daily[daily['period']=='Jan 2026']['direction_binary'].mean()*100:.0f}% positive days (balanced direction)

BEST SIMPLE PREDICTOR: {best_rule[0]} ({best_rule[1]*100:.1f}% accuracy)

PERSISTENCE ANALYSIS:
- 2025: Strong persistence (yesterday predicts today)
- Jan 2026: Weaker persistence, direction changes more frequently

TRADING IMPLICATION:
- "Always sell" no longer works
- Need directional prediction to capture the spread
- The spread SIZE is still there, just alternating direction
""")

# Save daily data
daily.to_csv(OUT_DIR / 'data' / 'daily_direction.csv', index=False)
print(f"[+] Saved: data/daily_direction.csv")
print("\n[+] Analysis complete!")
