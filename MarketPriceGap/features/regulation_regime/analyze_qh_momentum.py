"""
QH-Level Momentum Strategy Analysis

Trading setup (matching existing analysis):
- 1 MWh per QH
- Hours 5-21 (17 hours)
- QH 1 & 2 only (34 QH per day)
- P&L in EUR total
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data")
OUT_DIR = BASE / "MarketPriceGap" / "features" / "regulation_regime"

print("[*] Loading hourly data...")
df = pd.read_csv(OUT_DIR / 'data' / 'hourly_reg_spread.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour

# Filter to hours 5-21 (matching existing analysis)
df = df[(df['hour'] >= 5) & (df['hour'] <= 21)].copy()
print(f"[+] Filtered to hours 5-21: {len(df):,} records")

# Add period
df['period'] = df['datetime'].apply(
    lambda x: 'Jan 2026' if x >= pd.Timestamp('2026-01-01') else
              'Sep-Dec 2025' if x >= pd.Timestamp('2025-09-01') else '2025 (early)'
)

# Sort and add lag features
df = df.sort_values('datetime').reset_index(drop=True)
df['spread_lag1'] = df['idm_imb_spread'].shift(1)
df['direction_lag1'] = (df['idm_imb_spread'] > 0).shift(1).astype(float)

# Drop first row (no lag available)
df = df.dropna(subset=['spread_lag1'])

print(f"[+] Dataset with lags: {len(df):,} hourly records")

# ===== STRATEGIES =====
# Note: Each hourly record represents 2 QH (QH1 and QH2), so multiply P&L by 2

# Strategy 1: Always Sell IDM (baseline)
df['pnl_always_sell'] = df['idm_imb_spread'] * 2  # 2 QH per hour

# Strategy 2: Follow previous hour's direction
df['pnl_follow'] = df.apply(
    lambda x: x['idm_imb_spread'] * 2 if x['direction_lag1'] == 1 else -x['idm_imb_spread'] * 2, axis=1
)

# Strategy 3: Momentum rule (large spread -> follow, small -> contrarian)
median_spread = df['spread_lag1'].abs().median()
df['large_spread'] = df['spread_lag1'].abs() > median_spread
df['pnl_momentum'] = df.apply(
    lambda x: (x['idm_imb_spread'] * 2 if x['direction_lag1'] == 1 else -x['idm_imb_spread'] * 2)
              if x['large_spread']
              else (-x['idm_imb_spread'] * 2 if x['direction_lag1'] == 1 else x['idm_imb_spread'] * 2),
    axis=1
)

# Strategy 4: Skip small spread hours (only trade when previous hour had large spread)
df['pnl_selective'] = df.apply(
    lambda x: (x['idm_imb_spread'] * 2 if x['direction_lag1'] == 1 else -x['idm_imb_spread'] * 2)
              if x['large_spread'] else 0, axis=1
)

print(f"\n[*] Median |spread| threshold: {median_spread:.1f} EUR/MWh")

# ===== RESULTS BY PERIOD =====
print("\n" + "="*70)
print("QH-LEVEL MOMENTUM STRATEGY RESULTS")
print("="*70)

for period in ['Sep-Dec 2025', 'Jan 2026']:
    subset = df[df['period'] == period]
    n_hours = len(subset)
    n_qh = n_hours * 2

    print(f"\n--- {period} ({n_hours} hours, {n_qh} QH) ---")

    # Calculate cumulative P&L
    pnl_sell = subset['pnl_always_sell'].sum()
    pnl_follow = subset['pnl_follow'].sum()
    pnl_momentum = subset['pnl_momentum'].sum()
    pnl_selective = subset['pnl_selective'].sum()
    n_selective = subset['large_spread'].sum() * 2  # QH traded

    # Win rates
    wr_sell = (subset['idm_imb_spread'] > 0).mean() * 100
    wr_follow = ((subset['pnl_follow'] > 0) | ((subset['pnl_follow'] == 0) & (subset['idm_imb_spread'] == 0))).mean() * 100
    wr_momentum = (subset['pnl_momentum'] > 0).mean() * 100

    print(f"  {'Strategy':<25} {'P&L (EUR)':>12} {'Win Rate':>10} {'Per QH':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Always Sell':<25} {pnl_sell:>+12,.0f} {wr_sell:>9.0f}% {pnl_sell/n_qh:>+10.1f}")
    print(f"  {'Follow Prev Hour':<25} {pnl_follow:>+12,.0f} {wr_follow:>9.0f}% {pnl_follow/n_qh:>+10.1f}")
    print(f"  {'Momentum Rule':<25} {pnl_momentum:>+12,.0f} {wr_momentum:>9.0f}% {pnl_momentum/n_qh:>+10.1f}")
    print(f"  {'Selective (large only)':<25} {pnl_selective:>+12,.0f} {'--':>10} {pnl_selective/n_selective:>+10.1f}")

# ===== DAILY AGGREGATION FOR COMPARISON =====
print("\n" + "="*70)
print("DAILY P&L FOR COMPARISON (matching your chart format)")
print("="*70)

daily = df.groupby(['date', 'period']).agg(
    pnl_always_sell=('pnl_always_sell', 'sum'),
    pnl_follow=('pnl_follow', 'sum'),
    pnl_momentum=('pnl_momentum', 'sum'),
    pnl_selective=('pnl_selective', 'sum'),
    n_hours=('idm_imb_spread', 'count')
).reset_index()

daily['date'] = pd.to_datetime(daily['date'])

# Jan 2026 daily stats
jan26_daily = daily[daily['period'] == 'Jan 2026'].copy()
print(f"\nJan 2026: {len(jan26_daily)} days")
print(f"  Always Sell total:  {jan26_daily['pnl_always_sell'].sum():>+,.0f} EUR")
print(f"  Follow Prev Hour:   {jan26_daily['pnl_follow'].sum():>+,.0f} EUR")
print(f"  Momentum Rule:      {jan26_daily['pnl_momentum'].sum():>+,.0f} EUR")
print(f"  Selective (large):  {jan26_daily['pnl_selective'].sum():>+,.0f} EUR")

# ===== VISUALIZATION (matching your format) =====
print("\n[*] Creating visualization...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Get Dec 2025 + Jan 2026 for comparison
plot_data = daily[(daily['date'] >= '2025-12-01') & (daily['date'] <= '2026-01-31')].copy()
plot_data = plot_data.sort_values('date')

# Panel 1: Daily Profit Comparison
ax1 = axes[0]
x = range(len(plot_data))
width = 0.35

ax1.bar([i - width/2 for i in x], plot_data['pnl_always_sell'], width,
        label='Always Sell', color='steelblue', alpha=0.7)
ax1.bar([i + width/2 for i in x], plot_data['pnl_momentum'], width,
        label='Momentum Rule', color='green', alpha=0.7)

# Jan 1 line
jan1_idx = plot_data[plot_data['date'] >= '2026-01-01'].index[0] - plot_data.index[0]
ax1.axvline(x=jan1_idx - 0.5, color='red', linestyle='--', linewidth=2, label='Jan 1, 2026')
ax1.axhline(y=0, color='black', linewidth=0.5)

ax1.set_xticks(x[::3])
ax1.set_xticklabels([plot_data.iloc[i]['date'].strftime('%b %d') for i in range(0, len(plot_data), 3)], rotation=45)
ax1.set_ylabel('Daily Profit (EUR)')
ax1.set_title('Daily Profit: Always Sell vs Momentum Rule (1 MWh per QH, hours 5-21)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Panel 2: Cumulative P&L
ax2 = axes[1]
plot_data['cum_sell'] = plot_data['pnl_always_sell'].cumsum()
plot_data['cum_momentum'] = plot_data['pnl_momentum'].cumsum()
plot_data['cum_follow'] = plot_data['pnl_follow'].cumsum()

ax2.plot(x, plot_data['cum_sell'], 'b-', linewidth=2,
         label=f"Always Sell ({plot_data['cum_sell'].iloc[-1]:+,.0f} EUR)")
ax2.plot(x, plot_data['cum_momentum'], 'g-', linewidth=2,
         label=f"Momentum ({plot_data['cum_momentum'].iloc[-1]:+,.0f} EUR)")
ax2.plot(x, plot_data['cum_follow'], 'orange', linewidth=2, linestyle='--',
         label=f"Follow Prev ({plot_data['cum_follow'].iloc[-1]:+,.0f} EUR)")

ax2.axvline(x=jan1_idx - 0.5, color='red', linestyle='--', linewidth=2)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_xticks(x[::3])
ax2.set_xticklabels([plot_data.iloc[i]['date'].strftime('%b %d') for i in range(0, len(plot_data), 3)], rotation=45)
ax2.set_ylabel('Cumulative P&L (EUR)')
ax2.set_title('Cumulative P&L: Dec 2025 - Jan 2026')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Panel 3: Strategy comparison bar chart (Jan 2026 only)
ax3 = axes[2]
jan26_totals = {
    'Always Sell': jan26_daily['pnl_always_sell'].sum(),
    'Follow Prev Hour': jan26_daily['pnl_follow'].sum(),
    'Momentum Rule': jan26_daily['pnl_momentum'].sum(),
    'Selective': jan26_daily['pnl_selective'].sum()
}

colors = ['steelblue', 'orange', 'green', 'purple']
bars = ax3.bar(jan26_totals.keys(), jan26_totals.values(), color=colors, alpha=0.7)
ax3.axhline(y=0, color='black', linewidth=0.5)
ax3.set_ylabel('Total P&L (EUR)')
ax3.set_title('Jan 2026 Total P&L by Strategy (1 MWh per QH, hours 5-21, QH 1&2)')

for bar, val in zip(bars, jan26_totals.values()):
    ax3.text(bar.get_x() + bar.get_width()/2, val + (100 if val > 0 else -200),
             f'{val:+,.0f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / '05_qh_momentum_strategy.png', dpi=150, bbox_inches='tight')
print(f"[+] Saved: 05_qh_momentum_strategy.png")

# ===== FINAL SUMMARY =====
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

jan_sell = jan26_daily['pnl_always_sell'].sum()
jan_momentum = jan26_daily['pnl_momentum'].sum()
improvement = jan_momentum - jan_sell

print(f"\nJan 2026 (trading 1 MWh per QH, hours 5-21, QH 1&2):")
print(f"  Always Sell:   {jan_sell:>+8,.0f} EUR")
print(f"  Momentum Rule: {jan_momentum:>+8,.0f} EUR")
print(f"  Improvement:   {improvement:>+8,.0f} EUR ({improvement/abs(jan_sell)*100 if jan_sell != 0 else 0:+.0f}%)")

print("\n[+] Analysis complete!")
