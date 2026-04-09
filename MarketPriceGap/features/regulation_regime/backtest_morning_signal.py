"""
Backtest: Morning Signal Strategy for Jan 2026

Strategy:
- At 6:00 AM, check if morning spread (hours 0-5) was positive on average
- If positive: SELL IDM for rest of day (hours 7-23)
- If negative: BUY IDM for rest of day
- Trade 1 MW per quarter hour
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data")
OUT_DIR = BASE / "MarketPriceGap" / "features" / "regulation_regime"

print("[*] Loading hourly data...")
df = pd.read_csv(OUT_DIR / 'data' / 'hourly_reg_spread.csv', parse_dates=['datetime'])

# Filter to Jan 2026
df = df[df['datetime'] >= '2026-01-01'].copy()
print(f"[+] Jan 2026 data: {len(df)} hourly records")

# Add time features
df['date'] = df['datetime'].dt.date
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['datetime'].dt.hour

# Calculate morning signal for each day (hours 0-5)
morning = df[df['hour'].between(0, 5)].groupby('date')['idm_imb_spread'].mean()
morning = morning.rename('morning_spread_avg')

# Merge signal back
df = df.merge(morning.reset_index(), on='date', how='left')

# Trading hours: after morning ramp (hours 7-23)
# In 15-min resolution, this is 4 QH per hour
TRADING_START_HOUR = 7
TRADING_END_HOUR = 23  # inclusive

trading_hours = df[df['hour'].between(TRADING_START_HOUR, TRADING_END_HOUR)].copy()
print(f"[+] Trading hours (7-23): {len(trading_hours)} hourly records")

# Strategy:
# If morning_spread_avg > 0: SELL IDM (expect positive spread, profit = spread)
# If morning_spread_avg < 0: BUY IDM (expect negative spread, profit = -spread)

trading_hours['signal'] = np.where(trading_hours['morning_spread_avg'] > 0, 1, -1)
trading_hours['position'] = trading_hours['signal']  # 1 = sell IDM, -1 = buy IDM

# P&L per hour: position * spread * volume
# With hourly data: 4 QH per hour, 1 MW each = 4 MWh per hour
# Actually we have hourly data, spread is in EUR/MWh
# P&L = position * spread * 4 (for 4 QH at 1MW each)
QH_PER_HOUR = 4
MW_PER_QH = 1

trading_hours['pnl_strategy'] = trading_hours['position'] * trading_hours['idm_imb_spread'] * QH_PER_HOUR * MW_PER_QH

# Baseline: always sell IDM
trading_hours['pnl_always_sell'] = trading_hours['idm_imb_spread'] * QH_PER_HOUR * MW_PER_QH

# Perfect foresight
trading_hours['pnl_perfect'] = np.abs(trading_hours['idm_imb_spread']) * QH_PER_HOUR * MW_PER_QH

# ===== RESULTS =====
print("\n" + "="*60)
print("BACKTEST RESULTS: Jan 2026")
print("="*60)

total_strategy = trading_hours['pnl_strategy'].sum()
total_always_sell = trading_hours['pnl_always_sell'].sum()
total_perfect = trading_hours['pnl_perfect'].sum()

print(f"\nTotal P&L (hours 7-23, 1 MW per QH = 4 MWh/hour):")
print(f"  Morning Signal Strategy: {total_strategy:+,.0f} EUR")
print(f"  Always Sell IDM:         {total_always_sell:+,.0f} EUR")
print(f"  Perfect Foresight:       {total_perfect:+,.0f} EUR (upper bound)")
print(f"\n  Strategy improvement over Always Sell: {total_strategy - total_always_sell:+,.0f} EUR")

# Daily breakdown
daily_pnl = trading_hours.groupby('date').agg(
    pnl_strategy=('pnl_strategy', 'sum'),
    pnl_always_sell=('pnl_always_sell', 'sum'),
    pnl_perfect=('pnl_perfect', 'sum'),
    morning_signal=('morning_spread_avg', 'first'),
    n_hours=('pnl_strategy', 'count')
).reset_index()

daily_pnl['strategy_correct'] = ((daily_pnl['morning_signal'] > 0) == (daily_pnl['pnl_always_sell'] > 0))

print(f"\n--- Daily Breakdown ---")
print(f"Days where strategy was correct: {daily_pnl['strategy_correct'].sum()} / {len(daily_pnl)}")
print(f"Winning days (strategy P&L > 0): {(daily_pnl['pnl_strategy'] > 0).sum()} / {len(daily_pnl)}")
print(f"Avg daily P&L (strategy): {daily_pnl['pnl_strategy'].mean():+,.0f} EUR")
print(f"Avg daily P&L (always sell): {daily_pnl['pnl_always_sell'].mean():+,.0f} EUR")

print(f"\n--- Statistics ---")
print(f"Trading hours per day: {daily_pnl['n_hours'].mean():.1f}")
print(f"Total trading hours: {len(trading_hours)}")
print(f"Total MWh traded: {len(trading_hours) * QH_PER_HOUR * MW_PER_QH:,}")

# Win rate by hour
print(f"\n--- Hourly Win Rate (Strategy) ---")
hourly_wr = trading_hours.groupby('hour').apply(lambda x: (x['pnl_strategy'] > 0).mean() * 100)
for hour, wr in hourly_wr.items():
    bar = '#' * int(wr / 5)
    print(f"  Hour {hour:2d}: {wr:5.1f}% {bar}")

# ===== VISUALIZATION =====
print("\n[*] Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cumulative P&L
ax1 = axes[0, 0]
trading_sorted = trading_hours.sort_values('datetime')
trading_sorted['cum_strategy'] = trading_sorted['pnl_strategy'].cumsum()
trading_sorted['cum_always'] = trading_sorted['pnl_always_sell'].cumsum()

ax1.plot(trading_sorted['datetime'], trading_sorted['cum_strategy'], label='Morning Signal Strategy', linewidth=2)
ax1.plot(trading_sorted['datetime'], trading_sorted['cum_always'], label='Always Sell IDM', linewidth=2, alpha=0.7)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative P&L (EUR)')
ax1.set_title('Cumulative P&L: Morning Signal vs Always Sell')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Daily P&L comparison
ax2 = axes[0, 1]
x = range(len(daily_pnl))
width = 0.35
ax2.bar([i - width/2 for i in x], daily_pnl['pnl_strategy'], width, label='Morning Signal', color='steelblue', alpha=0.8)
ax2.bar([i + width/2 for i in x], daily_pnl['pnl_always_sell'], width, label='Always Sell', color='coral', alpha=0.8)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax2.set_xlabel('Day (Jan 2026)')
ax2.set_ylabel('Daily P&L (EUR)')
ax2.set_title('Daily P&L: Strategy vs Always Sell')
ax2.legend()

# Plot 3: Morning signal vs day outcome
ax3 = axes[1, 0]
colors = daily_pnl['strategy_correct'].map({True: 'green', False: 'red'})
ax3.scatter(daily_pnl['morning_signal'], daily_pnl['pnl_always_sell'], c=colors, s=100, alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax3.set_xlabel('Morning Signal (Avg Spread hours 0-5)')
ax3.set_ylabel('Day P&L if Always Sell (EUR)')
ax3.set_title('Morning Signal vs Day Outcome (Green=Correct, Red=Wrong)')

# Quadrant labels
ax3.text(30, 1500, 'Signal: SELL\nDay: Profitable', fontsize=9, ha='center')
ax3.text(-30, 1500, 'Signal: BUY\nDay: Loss', fontsize=9, ha='center')
ax3.text(30, -1500, 'Signal: SELL\nDay: Loss', fontsize=9, ha='center')
ax3.text(-30, -1500, 'Signal: BUY\nDay: Profitable', fontsize=9, ha='center')

# Plot 4: Summary stats
ax4 = axes[1, 1]
strategies = ['Morning\nSignal', 'Always\nSell', 'Perfect\nForesight']
pnls = [total_strategy, total_always_sell, total_perfect]
colors = ['steelblue', 'coral', 'green']
bars = ax4.bar(strategies, pnls, color=colors, alpha=0.8)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax4.set_ylabel('Total P&L (EUR)')
ax4.set_title('Total P&L Comparison (Jan 2026)')

for bar, pnl in zip(bars, pnls):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'{pnl:+,.0f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / '05_backtest_results.png', dpi=150, bbox_inches='tight')
print(f"[+] Saved: 05_backtest_results.png")

# ===== TRADE LOG =====
print("\n--- Sample Trade Log (first 5 days) ---")
for _, row in daily_pnl.head(5).iterrows():
    signal = "SELL" if row['morning_signal'] > 0 else "BUY"
    correct = "OK" if row['strategy_correct'] else "WRONG"
    print(f"  {row['date'].strftime('%Y-%m-%d')}: Morning={row['morning_signal']:+.1f} -> {signal} -> P&L={row['pnl_strategy']:+,.0f} EUR ({correct})")

print("\n[+] Backtest complete!")
