"""
Regulation Volatility as IDM-Imbalance Spread Regime Indicator

Hypothesis: High regulation volatility = system stress = larger spreads
            Low regulation volatility = stable system = converged markets

This could explain why the arbitrage closed in Jan 2026.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data")
REG_FILE = BASE / "RawData" / "3MIN_REG.csv"
MASTER_FILE = BASE / "data" / "master" / "master_imbalance_data.csv"
OUT_DIR = BASE / "MarketPriceGap" / "features" / "regulation_regime"

print("[*] Loading regulation data...")
# Load regulation data (European decimal format - comma is both delimiter AND decimal separator)
# Format: datetime, milliseconds, integer_part, decimal_part
reg_df = pd.read_csv(REG_FILE, skiprows=1, header=None, names=['datetime', 'ms', 'int_part', 'dec_part', 'extra'],
                     encoding='utf-8-sig')

# Parse datetime (Slovak format: DD/MM/YYYY HH:MM:SS)
reg_df['datetime'] = pd.to_datetime(reg_df['datetime'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
reg_df = reg_df.dropna(subset=['datetime'])

# Reconstruct value from integer and decimal parts
# Value = int_part.dec_part (e.g., -12.788)
# Handle invalid values
reg_df['int_part'] = pd.to_numeric(reg_df['int_part'], errors='coerce')
reg_df['dec_part'] = pd.to_numeric(reg_df['dec_part'], errors='coerce')
reg_df = reg_df.dropna(subset=['int_part'])

reg_df['value'] = reg_df['int_part'] + reg_df['dec_part'].fillna(0) / 1000
# Handle negative values - decimal part should have same sign as integer part
mask_neg = reg_df['int_part'] < 0
reg_df.loc[mask_neg, 'value'] = reg_df.loc[mask_neg, 'int_part'] - reg_df.loc[mask_neg, 'dec_part'].fillna(0) / 1000
reg_df = reg_df.dropna(subset=['value'])

print(f"[+] Loaded {len(reg_df):,} regulation records")
print(f"    Date range: {reg_df['datetime'].min()} to {reg_df['datetime'].max()}")

# Aggregate to hourly statistics
print("[*] Aggregating to hourly volatility metrics...")
reg_df['hour'] = reg_df['datetime'].dt.floor('h')

hourly_reg = reg_df.groupby('hour').agg(
    reg_mean=('value', 'mean'),
    reg_std=('value', 'std'),
    reg_range=('value', lambda x: x.max() - x.min()),
    reg_abs_mean=('value', lambda x: np.abs(x).mean()),
    reg_count=('value', 'count')
).reset_index()

hourly_reg = hourly_reg.rename(columns={'hour': 'datetime'})
print(f"[+] Created {len(hourly_reg):,} hourly records")

# Load master imbalance data
print("[*] Loading master imbalance data...")
master_df = pd.read_csv(MASTER_FILE)
master_df['datetime'] = pd.to_datetime(master_df['datetime'])

# Aggregate to hourly (take mean of 4 quarters)
master_df['hour'] = master_df['datetime'].dt.floor('h')
hourly_imb = master_df.groupby('hour').agg(
    imbalance=('System Imbalance (MWh)', 'sum'),  # Sum for hourly total
    imb_price=('Imbalance Settlement Price (EUR/MWh)', 'mean')
).reset_index()
hourly_imb = hourly_imb.rename(columns={'hour': 'datetime'})

print(f"[+] Loaded {len(hourly_imb):,} hourly imbalance records")

# Load IDM data from multiple folders
print("[*] Loading IDM price data...")
IDM_BASE = BASE / "RawData" / "IDM_MarketData"
idm_folders = list(IDM_BASE.glob("IDM_total_results_*"))

idm_dfs = []
for folder in idm_folders:
    idm_file = folder / "15 min.csv"
    if idm_file.exists():
        try:
            df = pd.read_csv(idm_file, encoding='utf-8-sig', sep=';')
            idm_dfs.append(df)
        except Exception as e:
            print(f"    [-] Error loading {folder.name}: {e}")

idm_df = pd.concat(idm_dfs, ignore_index=True)

# Find the VWAP column (contains 'average' or 'VWAP')
vwap_col = [c for c in idm_df.columns if 'average' in c.lower() or 'vwap' in c.lower()][0]
date_col = 'Delivery day'
period_col = 'Period number'

# Parse date and period to datetime
idm_df['date'] = pd.to_datetime(idm_df[date_col], format='%d.%m.%Y', errors='coerce')
idm_df['period'] = pd.to_numeric(idm_df[period_col], errors='coerce')
idm_df['datetime'] = idm_df['date'] + pd.to_timedelta((idm_df['period'] - 1) * 15, unit='m')

idm_df['idm_price'] = pd.to_numeric(idm_df[vwap_col].astype(str).str.replace(',', '.'), errors='coerce')

# Aggregate to hourly (first 2 QH only for consistency)
idm_df['hour'] = idm_df['datetime'].dt.floor('h')
idm_df['qh'] = idm_df['period'] % 4  # 1,2,3,4 -> 1,2,3,0
idm_df['qh'] = idm_df['qh'].replace(0, 4)  # Map back to 1-4
idm_hourly = idm_df[idm_df['qh'].isin([1, 2])].groupby('hour')['idm_price'].mean().reset_index()
idm_hourly = idm_hourly.rename(columns={'hour': 'datetime'})

print(f"[+] Loaded {len(idm_hourly):,} hourly IDM records from {len(idm_folders)} folders")

# Merge all data
print("[*] Merging datasets...")
merged = hourly_reg.merge(hourly_imb, on='datetime', how='inner')
merged = merged.merge(idm_hourly, on='datetime', how='inner')

# Calculate spread
merged['idm_imb_spread'] = merged['idm_price'] - merged['imb_price']

# Filter valid data
merged = merged[merged['imb_price'].notna() & merged['idm_price'].notna()]
merged = merged[np.abs(merged['imbalance']) <= 300]  # Remove outliers

print(f"[+] Merged dataset: {len(merged):,} hourly records")
print(f"    Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")

# Add time features
merged['month'] = merged['datetime'].dt.to_period('M')
merged['year_month'] = merged['datetime'].dt.strftime('%Y-%m')

# ===== ANALYSIS 1: Monthly Regulation Volatility vs Spread =====
print("\n[*] Analyzing monthly patterns...")

monthly = merged.groupby('year_month').agg(
    reg_vol=('reg_std', 'mean'),
    reg_range=('reg_range', 'mean'),
    spread_mean=('idm_imb_spread', 'mean'),
    spread_std=('idm_imb_spread', 'std'),
    win_rate=('idm_imb_spread', lambda x: (x > 0).mean() * 100),
    n_obs=('idm_imb_spread', 'count')
).reset_index()

print("\n--- Monthly Regulation Volatility vs IDM-Imbalance Spread ---")
print(monthly.to_string(index=False))

# ===== ANALYSIS 2: Quartile Analysis =====
print("\n[*] Analyzing by regulation volatility quartiles...")

merged['reg_vol_quartile'] = pd.qcut(merged['reg_std'].rank(method='first'), 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

quartile_stats = merged.groupby('reg_vol_quartile').agg(
    avg_spread=('idm_imb_spread', 'mean'),
    win_rate=('idm_imb_spread', lambda x: (x > 0).mean() * 100),
    n_obs=('idm_imb_spread', 'count')
).reset_index()

print("\n--- Spread by Regulation Volatility Quartile ---")
print(quartile_stats.to_string(index=False))

# ===== ANALYSIS 3: 2025 vs Jan 2026 Comparison =====
print("\n[*] Comparing 2025 vs Jan 2026...")

merged['period'] = merged['datetime'].apply(
    lambda x: 'Jan 2026' if x >= pd.Timestamp('2026-01-01') else '2025'
)

period_stats = merged.groupby('period').agg(
    reg_vol_mean=('reg_std', 'mean'),
    reg_range_mean=('reg_range', 'mean'),
    spread_mean=('idm_imb_spread', 'mean'),
    win_rate=('idm_imb_spread', lambda x: (x > 0).mean() * 100),
    n_obs=('idm_imb_spread', 'count')
).reset_index()

print("\n--- 2025 vs Jan 2026 Comparison ---")
print(period_stats.to_string(index=False))

# ===== VISUALIZATIONS =====
print("\n[*] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Monthly regulation volatility and spread
ax1 = axes[0, 0]
ax1_twin = ax1.twinx()
months = range(len(monthly))
ax1.bar(months, monthly['reg_vol'], alpha=0.7, color='steelblue', label='Reg Volatility (MW)')
ax1_twin.plot(months, monthly['spread_mean'], 'ro-', linewidth=2, markersize=6, label='Avg Spread (EUR/MWh)')
ax1.set_xticks(months)
ax1.set_xticklabels(monthly['year_month'], rotation=45, ha='right')
ax1.set_ylabel('Regulation Volatility (MW)', color='steelblue')
ax1_twin.set_ylabel('IDM-Imbalance Spread (EUR/MWh)', color='red')
ax1.set_title('Monthly: Regulation Volatility vs IDM-Imbalance Spread')
ax1.axvline(x=len(monthly)-1.5, color='black', linestyle='--', alpha=0.5)
ax1.text(len(monthly)-1, ax1.get_ylim()[1]*0.9, 'Jan 2026', fontsize=9)

# Plot 2: Spread by volatility quartile
ax2 = axes[0, 1]
colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
bars = ax2.bar(quartile_stats['reg_vol_quartile'], quartile_stats['avg_spread'], color=colors)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_xlabel('Regulation Volatility Quartile')
ax2.set_ylabel('Average IDM-Imbalance Spread (EUR/MWh)')
ax2.set_title('Spread by Regulation Volatility Level')
for bar, wr in zip(bars, quartile_stats['win_rate']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'WR: {wr:.0f}%', ha='center', fontsize=9)

# Plot 3: Scatter - Regulation volatility vs Spread
ax3 = axes[1, 0]
colors_scatter = merged['period'].map({'2025': 'steelblue', 'Jan 2026': 'red'})
ax3.scatter(merged['reg_std'], merged['idm_imb_spread'], alpha=0.1, c=colors_scatter, s=5)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax3.set_xlabel('Hourly Regulation Volatility (MW)')
ax3.set_ylabel('IDM-Imbalance Spread (EUR/MWh)')
ax3.set_title('Regulation Volatility vs Spread (Blue=2025, Red=Jan2026)')

# Add correlation
corr = merged['reg_std'].corr(merged['idm_imb_spread'])
ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Period comparison
ax4 = axes[1, 1]
x = [0, 1]
width = 0.35
p2025 = period_stats[period_stats['period'] == '2025'].iloc[0]
p2026 = period_stats[period_stats['period'] == 'Jan 2026'].iloc[0]

ax4.bar(x[0] - width/2, p2025['reg_vol_mean'], width, label='Reg Vol (MW)', color='steelblue')
ax4.bar(x[1] - width/2, p2026['reg_vol_mean'], width, color='steelblue')
ax4_twin = ax4.twinx()
ax4_twin.bar(x[0] + width/2, p2025['spread_mean'], width, label='Spread (EUR)', color='coral')
ax4_twin.bar(x[1] + width/2, p2026['spread_mean'], width, color='coral')

ax4.set_xticks(x)
ax4.set_xticklabels(['2025', 'Jan 2026'])
ax4.set_ylabel('Regulation Volatility (MW)', color='steelblue')
ax4_twin.set_ylabel('Avg Spread (EUR/MWh)', color='coral')
ax4.set_title('2025 vs Jan 2026: Did Regulation Volatility Drop?')

# Add annotations
ax4.text(x[0], p2025['reg_vol_mean'] + 1, f"{p2025['reg_vol_mean']:.1f}", ha='center', fontsize=9)
ax4.text(x[1], p2026['reg_vol_mean'] + 1, f"{p2026['reg_vol_mean']:.1f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / '01_regulation_spread_analysis.png', dpi=150, bbox_inches='tight')
print(f"[+] Saved: 01_regulation_spread_analysis.png")

# ===== CORRELATION ANALYSIS =====
print("\n[*] Correlation analysis...")

# Rolling correlation
merged_sorted = merged.sort_values('datetime')
merged_sorted['rolling_corr'] = merged_sorted['reg_std'].rolling(168).corr(merged_sorted['idm_imb_spread'])  # 7-day window

fig2, ax = plt.subplots(figsize=(12, 5))
ax.plot(merged_sorted['datetime'], merged_sorted['rolling_corr'], alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.axvline(x=pd.Timestamp('2026-01-01'), color='red', linestyle='--', alpha=0.7, label='Jan 2026')
ax.set_xlabel('Date')
ax.set_ylabel('Rolling 7-day Correlation')
ax.set_title('Rolling Correlation: Regulation Volatility vs IDM-Imbalance Spread')
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / '02_rolling_correlation.png', dpi=150, bbox_inches='tight')
print(f"[+] Saved: 02_rolling_correlation.png")

# ===== SUMMARY STATISTICS =====
print("\n" + "="*60)
print("SUMMARY: Regulation Volatility as Spread Regime Indicator")
print("="*60)

print(f"\nOverall correlation (Reg Vol vs Spread): {corr:.3f}")

print("\n--- By Volatility Quartile ---")
for _, row in quartile_stats.iterrows():
    print(f"  {row['reg_vol_quartile']}: Spread = {row['avg_spread']:+.1f} EUR, Win Rate = {row['win_rate']:.0f}%")

print("\n--- 2025 vs Jan 2026 ---")
print(f"  2025:     Reg Vol = {p2025['reg_vol_mean']:.1f} MW, Spread = {p2025['spread_mean']:+.1f} EUR, WR = {p2025['win_rate']:.0f}%")
print(f"  Jan 2026: Reg Vol = {p2026['reg_vol_mean']:.1f} MW, Spread = {p2026['spread_mean']:+.1f} EUR, WR = {p2026['win_rate']:.0f}%")

vol_change = (p2026['reg_vol_mean'] - p2025['reg_vol_mean']) / p2025['reg_vol_mean'] * 100
spread_change = p2026['spread_mean'] - p2025['spread_mean']

print(f"\n  Reg Volatility Change: {vol_change:+.1f}%")
print(f"  Spread Change: {spread_change:+.1f} EUR/MWh")

if vol_change < -10:
    print("\n[!] FINDING: Regulation volatility DECREASED in Jan 2026")
    print("    This suggests a calmer grid = converged markets = closed arbitrage")
elif vol_change > 10:
    print("\n[!] FINDING: Regulation volatility INCREASED in Jan 2026")
    print("    The arbitrage closure is NOT explained by regulation stability")
else:
    print("\n[!] FINDING: Regulation volatility was SIMILAR in Jan 2026")
    print("    Other factors must explain the arbitrage closure")

# Save data for further analysis
data_dir = OUT_DIR / 'data'
data_dir.mkdir(exist_ok=True)
merged.to_csv(data_dir / 'hourly_reg_spread.csv', index=False)
print(f"[+] Saved: data/hourly_reg_spread.csv")
print("\n[+] Analysis complete!")
