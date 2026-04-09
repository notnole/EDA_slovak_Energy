"""
DA vs Imbalance Settlement Price Comparison
Reproduces two charts:
  1) da_vs_imbalance_2026.png - January 2026 DA vs imbalance price profiles
  2) da_vs_imbalance_sep25_vs_jan26.png - Sep-Dec 2025 vs Jan 2026 comparison
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
BASE = Path(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data')
MKT_PATH = BASE / 'MarketPriceGap' / 'data' / 'processed' / 'hourly_market_prices.csv'
OUT_DIR = BASE / 'MarketPriceGap' / 'strategies' / 'weather_impact'

# ============================================================
# 1. LOAD DATA
# ============================================================
print("[*] Loading market price data...")
df = pd.read_csv(MKT_PATH, parse_dates=['timestamp_hour'])
df = df.rename(columns={'timestamp_hour': 'datetime'})
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek  # 0=Mon, 6=Sun
df['date'] = df['datetime'].dt.date
print(f"[+] Loaded {len(df)} hourly records")
print(f"    Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# Compute spread (DA - Imbalance Settlement)
df['spread'] = df['da_price'] - df['imb_settlement_price']

# Define periods
jan26 = df[(df['datetime'] >= '2026-01-01') & (df['datetime'] < '2026-02-01')].copy()
sep_dec_25 = df[(df['datetime'] >= '2025-09-01') & (df['datetime'] < '2026-01-01')].copy()

print(f"[+] Jan 2026: {len(jan26)} hours")
print(f"[+] Sep-Dec 2025: {len(sep_dec_25)} hours")

# ============================================================
# 2. CHART 1: DA vs Imbalance Settlement Price - January 2026
# ============================================================
print("\n[*] Creating Chart 1: DA vs Imbalance - January 2026...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('DA vs Imbalance Settlement Price - January 2026', fontsize=14, fontweight='bold')

# --- Top-left: Hourly price profiles ---
ax = axes[0, 0]
hourly_da = jan26.groupby('hour')['da_price'].mean()
hourly_imb = jan26.groupby('hour')['imb_settlement_price'].mean()

ax.plot(hourly_da.index, hourly_da.values, 'b-o', markersize=4, linewidth=1.5, label='DA Price')
ax.plot(hourly_imb.index, hourly_imb.values, 'r-s', markersize=4, linewidth=1.5, label='Imbalance Settlement')

# Fill between where DA > Imb (green) and DA < Imb (red)
hours = hourly_da.index.values
da_vals = hourly_da.values
imb_vals = hourly_imb.reindex(hours).values
ax.fill_between(hours, da_vals, imb_vals, where=(da_vals >= imb_vals),
                alpha=0.15, color='green', label='DA > Imb')
ax.fill_between(hours, da_vals, imb_vals, where=(da_vals < imb_vals),
                alpha=0.15, color='red')

ax.set_xlabel('Hour of Day')
ax.set_ylabel('EUR/MWh')
ax.set_title('Average Price by Hour')
ax.set_xticks(range(0, 24))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Top-right: Hourly spread bars ---
ax = axes[0, 1]
hourly_spread = jan26.groupby('hour')['spread'].mean()
mean_spread = jan26['spread'].mean()
colors_h = ['green' if v >= 0 else 'indianred' for v in hourly_spread.values]
ax.bar(hourly_spread.index, hourly_spread.values, color=colors_h, alpha=0.7, edgecolor='black', linewidth=0.3)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axhline(y=mean_spread, color='navy', linewidth=1.0, linestyle='--',
           label=f'Mean: {mean_spread:.1f} EUR')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('DA - Imbalance Settlement (EUR/MWh)')
ax.set_title('Spread by Hour (DA - Imbalance)')
ax.set_xticks(range(0, 24))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Bottom-left: DOW price bars ---
ax = axes[1, 0]
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_da = jan26.groupby('dayofweek')['da_price'].mean()
dow_imb = jan26.groupby('dayofweek')['imb_settlement_price'].mean()
x = np.arange(7)
width = 0.35
ax.bar(x - width/2, dow_da.reindex(range(7)).values, width, color='steelblue', label='DA Price')
ax.bar(x + width/2, dow_imb.reindex(range(7)).values, width, color='indianred', label='Imbalance Settlement')
ax.set_xlabel('Day of Week')
ax.set_ylabel('EUR/MWh')
ax.set_title('Average Price by Day of Week')
ax.set_xticks(x)
ax.set_xticklabels(dow_labels)
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# --- Bottom-right: DOW spread bars ---
ax = axes[1, 1]
dow_spread = jan26.groupby('dayofweek')['spread'].mean()
colors_dow = ['green' if v >= 0 else 'indianred' for v in dow_spread.reindex(range(7)).values]
bars = ax.bar(x, dow_spread.reindex(range(7)).values, color=colors_dow, alpha=0.7,
              edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linewidth=0.5)

# Annotate each bar with value
for i, bar in enumerate(bars):
    val = dow_spread.reindex(range(7)).values[i]
    ypos = bar.get_height() if val >= 0 else bar.get_height()
    va = 'bottom' if val >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.1f}',
            ha='center', va=va, fontsize=8, fontweight='bold',
            color='green' if val >= 0 else 'indianred')

ax.set_xlabel('Day of Week')
ax.set_ylabel('DA - Imbalance Settlement (EUR/MWh)')
ax.set_title('Spread by Day of Week')
ax.set_xticks(x)
ax.set_xticklabels(dow_labels)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out1 = OUT_DIR / 'da_vs_imbalance_2026.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[+] Saved: {out1}")

# ============================================================
# 3. CHART 2: Sep-Dec 2025 vs Jan 2026 Comparison
# ============================================================
print("\n[*] Creating Chart 2: Sep-Dec 2025 vs Jan 2026 comparison...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('DA vs Imbalance: Sep-Dec 2025 vs Jan 2026', fontsize=14, fontweight='bold')

# Compute mean spreads for annotation
mean_spread_sep = sep_dec_25['spread'].mean()
mean_spread_jan = jan26['spread'].mean()

# --- Top-left: Hourly profiles (4 lines) ---
ax = axes[0, 0]
h_da_sep = sep_dec_25.groupby('hour')['da_price'].mean()
h_imb_sep = sep_dec_25.groupby('hour')['imb_settlement_price'].mean()
h_da_jan = jan26.groupby('hour')['da_price'].mean()
h_imb_jan = jan26.groupby('hour')['imb_settlement_price'].mean()

ax.plot(h_da_sep.index, h_da_sep.values, 'b-o', markersize=3, linewidth=1.5, label='DA Sep-Dec 2025')
ax.plot(h_imb_sep.index, h_imb_sep.values, 'r-o', markersize=3, linewidth=1.5, label='Imb Sep-Dec 2025')
ax.plot(h_da_jan.index, h_da_jan.values, 'b--s', markersize=3, linewidth=1.5, label='DA Jan 2026')
ax.plot(h_imb_jan.index, h_imb_jan.values, 'r--s', markersize=3, linewidth=1.5, label='Imb Jan 2026')

ax.set_xlabel('Hour of Day')
ax.set_ylabel('EUR/MWh')
ax.set_title('Average Price by Hour')
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# --- Top-right: Hourly spread bars by period ---
ax = axes[0, 1]
h_spread_sep = sep_dec_25.groupby('hour')['spread'].mean()
h_spread_jan = jan26.groupby('hour')['spread'].mean()
x_h = np.arange(24)
width_h = 0.4
ax.bar(x_h - width_h/2, h_spread_sep.reindex(range(24)).values, width_h,
       color='steelblue', alpha=0.7, label=f'Sep-Dec 2025 (mean={mean_spread_sep:.1f})')
ax.bar(x_h + width_h/2, h_spread_jan.reindex(range(24)).values, width_h,
       color='indianred', alpha=0.7, label=f'Jan 2026 (mean={mean_spread_jan:.1f})')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('DA - Imbalance Settlement (EUR/MWh)')
ax.set_title('Spread by Hour: Period Comparison')
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

# --- Bottom-left: DOW price bars (4 series) ---
ax = axes[1, 0]
dow_da_sep = sep_dec_25.groupby('dayofweek')['da_price'].mean()
dow_imb_sep = sep_dec_25.groupby('dayofweek')['imb_settlement_price'].mean()
dow_da_jan = jan26.groupby('dayofweek')['da_price'].mean()
dow_imb_jan = jan26.groupby('dayofweek')['imb_settlement_price'].mean()

x_d = np.arange(7)
w = 0.2
ax.bar(x_d - 1.5*w, dow_da_sep.reindex(range(7)).values, w,
       color='steelblue', alpha=0.8, label='DA Sep-Dec 2025', hatch='')
ax.bar(x_d - 0.5*w, dow_imb_sep.reindex(range(7)).values, w,
       color='steelblue', alpha=0.4, label='Imb Sep-Dec 2025', hatch='//')
ax.bar(x_d + 0.5*w, dow_da_jan.reindex(range(7)).values, w,
       color='indianred', alpha=0.8, label='DA Jan 2026')
ax.bar(x_d + 1.5*w, dow_imb_jan.reindex(range(7)).values, w,
       color='indianred', alpha=0.4, label='Imb Jan 2026', hatch='//')

ax.set_xlabel('Day of Week')
ax.set_ylabel('EUR/MWh')
ax.set_title('Average Price by Day of Week')
ax.set_xticks(x_d)
ax.set_xticklabels(dow_labels)
ax.set_ylim(bottom=0)
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3, axis='y')

# --- Bottom-right: DOW spread bars by period ---
ax = axes[1, 1]
dow_spread_sep = sep_dec_25.groupby('dayofweek')['spread'].mean()
dow_spread_jan = jan26.groupby('dayofweek')['spread'].mean()

bars_sep = ax.bar(x_d - w/2, dow_spread_sep.reindex(range(7)).values, w,
                  color='steelblue', alpha=0.7,
                  label=f'Sep-Dec 2025 (mean={mean_spread_sep:.1f})')
bars_jan = ax.bar(x_d + w/2, dow_spread_jan.reindex(range(7)).values, w,
                  color='indianred', alpha=0.7,
                  label=f'Jan 2026 (mean={mean_spread_jan:.1f})')
ax.axhline(y=0, color='black', linewidth=0.5)

# Annotate each bar with rounded value
for bars, vals in [(bars_sep, dow_spread_sep.reindex(range(7)).values),
                   (bars_jan, dow_spread_jan.reindex(range(7)).values)]:
    for bar, val in zip(bars, vals):
        if np.isnan(val):
            continue
        ypos = bar.get_height()
        va = 'bottom' if val >= 0 else 'top'
        color = 'steelblue' if bars is bars_sep else 'indianred'
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.0f}',
                ha='center', va=va, fontsize=7, fontweight='bold', color=color)

ax.set_xlabel('Day of Week')
ax.set_ylabel('DA - Imbalance Settlement (EUR/MWh)')
ax.set_title('Spread by Day of Week: Period Comparison')
ax.set_xticks(x_d)
ax.set_xticklabels(dow_labels)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out2 = OUT_DIR / 'da_vs_imbalance_sep25_vs_jan26.png'
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[+] Saved: {out2}")

print("\n[+] Market comparison analysis complete.")
