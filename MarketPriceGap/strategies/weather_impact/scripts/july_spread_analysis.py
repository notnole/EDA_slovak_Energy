"""
July DA-Imbalance Spread Anomaly Investigation
Investigates why July shows more negative DA-Imb spread in both 2024 and 2025.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

BASE = r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data"

# =============================================================================
# 1. Load market data
# =============================================================================
print("[*] Loading market data...")
mkt = pd.read_csv(f"{BASE}/MarketPriceGap/data/processed/hourly_market_prices.csv",
                   parse_dates=['timestamp_hour'])
mkt = mkt.rename(columns={'timestamp_hour': 'datetime'})
mkt['date'] = mkt['datetime'].dt.date
mkt['hour'] = mkt['datetime'].dt.hour
mkt['month'] = mkt['datetime'].dt.month
mkt['year'] = mkt['datetime'].dt.year
mkt['year_month'] = mkt['datetime'].dt.to_period('M')
# deficit = imbalance < 0 (system short)
mkt['is_deficit'] = mkt['imbalance_mwh'] < 0
print(f"[+] Market data: {len(mkt)} rows, {mkt['datetime'].min()} to {mkt['datetime'].max()}")

# =============================================================================
# 2. Load temperature data
# =============================================================================
print("[*] Loading temperature data...")
temp_rows = []
with open(f"{BASE}/RawData/ActualTemp2024-2026.csv", 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()

for line in lines[1:]:  # skip header
    line = line.strip()
    if not line:
        continue
    if 'invalid' in line.lower():
        continue
    parts = line.split(',')
    # Format: DD/MM/YYYY HH:MM:SS,000,integer_part,decimal_part,
    if len(parts) < 4:
        continue
    dt_str = parts[0]
    # parts[1] = '000'
    int_part = parts[2]
    dec_part = parts[3]
    try:
        # Handle negative: int_part might be like "-0"
        sign = -1 if int_part.strip().startswith('-') else 1
        abs_int = abs(int(int_part.strip()))
        dec_str = dec_part.strip()
        dec_val = int(dec_str)
        # Dynamic divisor based on number of digits (2024: 2-digit, 2025-Mar+: 4-digit)
        divisor = 10 ** len(dec_str)
        temp = sign * (abs_int + dec_val / divisor)
        dt = pd.to_datetime(dt_str, format='%d/%m/%Y %H:%M:%S')
        temp_rows.append({'datetime': dt, 'temperature': temp})
    except (ValueError, IndexError):
        continue

temp = pd.DataFrame(temp_rows)
temp['date'] = temp['datetime'].dt.date
temp['hour'] = temp['datetime'].dt.hour
print(f"[+] Temperature data: {len(temp)} rows, {temp['datetime'].min()} to {temp['datetime'].max()}")

# =============================================================================
# 3. Merge market + temperature on (date, hour)
# =============================================================================
print("[*] Merging datasets...")
df = mkt.merge(temp[['date', 'hour', 'temperature']], on=['date', 'hour'], how='left')
print(f"[+] Merged: {len(df)} rows, temperature coverage: {df['temperature'].notna().sum()} ({df['temperature'].notna().mean()*100:.1f}%)")

# =============================================================================
# 4. Monthly summary table
# =============================================================================
print("\n" + "="*100)
print("MONTHLY SUMMARY TABLE")
print("="*100)

monthly = df.groupby('year_month').agg(
    mean_spread_da_imb=('spread_da_imb', 'mean'),
    mean_da_price=('da_price', 'mean'),
    mean_imb_price=('imb_settlement_price', 'mean'),
    pct_deficit=('is_deficit', 'mean'),
    mean_temperature=('temperature', 'mean'),
    count=('spread_da_imb', 'count')
).reset_index()

monthly['pct_deficit'] = monthly['pct_deficit'] * 100

print(f"\n{'Month':<10} {'Spread':>8} {'DA_Price':>9} {'Imb_Price':>10} {'%Deficit':>9} {'Temp_C':>7} {'N':>5}")
print("-" * 65)
for _, r in monthly.iterrows():
    temp_str = f"{r['mean_temperature']:.1f}" if pd.notna(r['mean_temperature']) else "N/A"
    print(f"{str(r['year_month']):<10} {r['mean_spread_da_imb']:>8.2f} {r['mean_da_price']:>9.2f} "
          f"{r['mean_imb_price']:>10.2f} {r['pct_deficit']:>8.1f}% {temp_str:>7} {r['count']:>5}")

# =============================================================================
# 5. July-specific deep dive
# =============================================================================
print("\n" + "="*100)
print("JULY DEEP DIVE: 2024 vs 2025")
print("="*100)

for year in [2024, 2025]:
    jul = df[(df['year'] == year) & (df['month'] == 7)]
    if len(jul) == 0:
        print(f"\n[!] No data for July {year}")
        continue

    print(f"\n--- July {year} ---")
    print(f"  Mean spread_da_imb: {jul['spread_da_imb'].mean():.2f} EUR/MWh")
    print(f"  Mean DA price: {jul['da_price'].mean():.2f} EUR/MWh")
    print(f"  Mean Imb price: {jul['imb_settlement_price'].mean():.2f} EUR/MWh")
    print(f"  Deficit %: {jul['is_deficit'].mean()*100:.1f}%")
    print(f"  Mean temperature: {jul['temperature'].mean():.1f} C")

    # Hourly breakdown
    hourly = jul.groupby('hour').agg(
        spread=('spread_da_imb', 'mean'),
        da=('da_price', 'mean'),
        imb=('imb_settlement_price', 'mean'),
        deficit_pct=('is_deficit', 'mean')
    )
    hourly['deficit_pct'] *= 100

    print(f"\n  Hourly spread profile (top 5 most negative hours):")
    worst = hourly.nsmallest(5, 'spread')
    for h, r in worst.iterrows():
        print(f"    Hour {h:02d}: spread={r['spread']:>7.2f}, DA={r['da']:>6.2f}, Imb={r['imb']:>7.2f}, deficit={r['deficit_pct']:.0f}%")

    print(f"\n  Hourly spread profile (top 5 most positive hours):")
    best = hourly.nlargest(5, 'spread')
    for h, r in best.iterrows():
        print(f"    Hour {h:02d}: spread={r['spread']:>7.2f}, DA={r['da']:>6.2f}, Imb={r['imb']:>7.2f}, deficit={r['deficit_pct']:.0f}%")

# =============================================================================
# 6. July vs neighboring months (June, August)
# =============================================================================
print("\n" + "="*100)
print("JULY vs NEIGHBORS (June, July, August) -- Deficit/Surplus Frequency")
print("="*100)

for year in [2024, 2025]:
    print(f"\n--- {year} ---")
    for m, name in [(6, 'June'), (7, 'July'), (8, 'August')]:
        sub = df[(df['year'] == year) & (df['month'] == m)]
        if len(sub) == 0:
            print(f"  {name}: no data")
            continue
        n_def = sub['is_deficit'].sum()
        n_sur = (~sub['is_deficit']).sum()
        print(f"  {name:<8}: spread={sub['spread_da_imb'].mean():>7.2f}, "
              f"deficit={n_def}/{len(sub)} ({sub['is_deficit'].mean()*100:.1f}%), "
              f"surplus={n_sur}/{len(sub)} ({(~sub['is_deficit']).mean()*100:.1f}%), "
              f"temp={sub['temperature'].mean():.1f}C, "
              f"DA={sub['da_price'].mean():.1f}, Imb={sub['imb_settlement_price'].mean():.1f}")

# =============================================================================
# 7. Heat wave analysis: hot days (>25C mean) vs spread
# =============================================================================
print("\n" + "="*100)
print("HEAT WAVE ANALYSIS: Hot days (daily mean >25C) vs Normal days")
print("="*100)

daily = df.groupby('date').agg(
    mean_spread=('spread_da_imb', 'mean'),
    mean_temp=('temperature', 'mean'),
    mean_da=('da_price', 'mean'),
    mean_imb=('imb_settlement_price', 'mean'),
    deficit_pct=('is_deficit', 'mean'),
    month=('month', 'first'),
    year=('year', 'first')
).reset_index()

daily['is_hot'] = daily['mean_temp'] > 25

# Overall
hot = daily[daily['is_hot']]
cool = daily[~daily['is_hot'] & daily['mean_temp'].notna()]
print(f"\nOverall dataset:")
print(f"  Hot days (>25C):    N={len(hot)}, mean spread={hot['mean_spread'].mean():.2f}, deficit%={hot['deficit_pct'].mean()*100:.1f}%")
print(f"  Normal days (<=25C): N={len(cool)}, mean spread={cool['mean_spread'].mean():.2f}, deficit%={cool['deficit_pct'].mean()*100:.1f}%")

# Summer months only (June-Aug)
for year in [2024, 2025]:
    summer = daily[(daily['year'] == year) & (daily['month'].isin([6, 7, 8]))]
    s_hot = summer[summer['is_hot']]
    s_cool = summer[~summer['is_hot'] & summer['mean_temp'].notna()]
    print(f"\nSummer {year} (Jun-Aug):")
    print(f"  Hot days (>25C):    N={len(s_hot)}, mean spread={s_hot['mean_spread'].mean():.2f}")
    print(f"  Normal days (<=25C): N={len(s_cool)}, mean spread={s_cool['mean_spread'].mean():.2f}")

# July specifically
for year in [2024, 2025]:
    jul_daily = daily[(daily['year'] == year) & (daily['month'] == 7)]
    j_hot = jul_daily[jul_daily['is_hot']]
    j_cool = jul_daily[~jul_daily['is_hot'] & jul_daily['mean_temp'].notna()]
    print(f"\nJuly {year}:")
    print(f"  Hot days (>25C):    N={len(j_hot)}, mean spread={j_hot['mean_spread'].mean():.2f}, mean temp={j_hot['mean_temp'].mean():.1f}C")
    print(f"  Normal days (<=25C): N={len(j_cool)}, mean spread={j_cool['mean_spread'].mean():.2f}, mean temp={j_cool['mean_temp'].mean():.1f}C")

# =============================================================================
# 8. Correlation: daily mean temperature vs daily mean spread
# =============================================================================
print("\n" + "="*100)
print("CORRELATION: Daily Mean Temperature vs Daily Mean Spread")
print("="*100)

valid = daily.dropna(subset=['mean_temp', 'mean_spread'])
corr = valid['mean_temp'].corr(valid['mean_spread'])
print(f"\nFull dataset: r = {corr:.4f} (N={len(valid)} days)")

# By summer
summer_valid = valid[valid['month'].isin([6, 7, 8])]
corr_summer = summer_valid['mean_temp'].corr(summer_valid['mean_spread'])
print(f"Summer only (Jun-Aug): r = {corr_summer:.4f} (N={len(summer_valid)} days)")

# By year
for year in [2024, 2025]:
    yr_valid = valid[valid['year'] == year]
    c = yr_valid['mean_temp'].corr(yr_valid['mean_spread'])
    print(f"  {year}: r = {c:.4f} (N={len(yr_valid)} days)")

# =============================================================================
# 9. Solar hypothesis: hourly pattern analysis
# =============================================================================
print("\n" + "="*100)
print("SOLAR HYPOTHESIS: Midday (10-16h) vs Evening (17-21h) spread in July")
print("="*100)

for year in [2024, 2025]:
    jul = df[(df['year'] == year) & (df['month'] == 7)]
    midday = jul[jul['hour'].between(10, 16)]
    evening = jul[jul['hour'].between(17, 21)]
    night = jul[jul['hour'].isin([0,1,2,3,4,5,22,23])]

    print(f"\nJuly {year}:")
    print(f"  Midday (10-16h):  spread={midday['spread_da_imb'].mean():.2f}, DA={midday['da_price'].mean():.1f}, Imb={midday['imb_settlement_price'].mean():.1f}, deficit%={midday['is_deficit'].mean()*100:.1f}%")
    print(f"  Evening (17-21h): spread={evening['spread_da_imb'].mean():.2f}, DA={evening['da_price'].mean():.1f}, Imb={evening['imb_settlement_price'].mean():.1f}, deficit%={evening['is_deficit'].mean()*100:.1f}%")
    print(f"  Night (22-05h):   spread={night['spread_da_imb'].mean():.2f}, DA={night['da_price'].mean():.1f}, Imb={night['imb_settlement_price'].mean():.1f}, deficit%={night['is_deficit'].mean()*100:.1f}%")

# =============================================================================
# 10. Create the 3-panel chart
# =============================================================================
print("\n[*] Creating chart...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('July DA-Imbalance Spread Anomaly Investigation', fontsize=14, fontweight='bold', y=0.98)

# --- Panel 1: Monthly mean spread as bar chart ---
ax1 = axes[0]
monthly_plot = monthly.copy()
monthly_plot['month_str'] = monthly_plot['year_month'].astype(str)
monthly_plot['is_july'] = monthly_plot['year_month'].apply(lambda x: x.month == 7)

colors = ['#d62728' if is_jul else '#1f77b4' for is_jul in monthly_plot['is_july']]
bars = ax1.bar(range(len(monthly_plot)), monthly_plot['mean_spread_da_imb'], color=colors, edgecolor='white', linewidth=0.5)
ax1.set_xticks(range(len(monthly_plot)))
ax1.set_xticklabels(monthly_plot['month_str'], rotation=45, ha='right', fontsize=7)
ax1.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax1.set_ylabel('Mean Spread DA-Imb (EUR/MWh)')
ax1.set_title('Panel 1: Monthly Mean DA-Imbalance Spread (July highlighted in red)')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on July bars
for i, (is_jul, val) in enumerate(zip(monthly_plot['is_july'], monthly_plot['mean_spread_da_imb'])):
    if is_jul:
        ax1.annotate(f'{val:.1f}', (i, val), textcoords="offset points",
                     xytext=(0, -12 if val < 0 else 5), ha='center', fontsize=8, fontweight='bold', color='#d62728')

# --- Panel 2: Scatter of daily temp vs daily spread ---
ax2 = axes[1]
daily_plot = daily.dropna(subset=['mean_temp', 'mean_spread'])
is_july = daily_plot['month'] == 7
non_july = daily_plot[~is_july]
july_days = daily_plot[is_july]

ax2.scatter(non_july['mean_temp'], non_july['mean_spread'], alpha=0.3, s=10, color='#1f77b4', label='Other months', zorder=2)
ax2.scatter(july_days['mean_temp'], july_days['mean_spread'], alpha=0.7, s=25, color='#d62728', label='July', zorder=3, edgecolors='black', linewidths=0.3)

# Add regression line for full dataset
from numpy.polynomial.polynomial import polyfit
mask = daily_plot['mean_temp'].notna() & daily_plot['mean_spread'].notna()
x_fit = daily_plot.loc[mask, 'mean_temp'].values
y_fit = daily_plot.loc[mask, 'mean_spread'].values
z = np.polyfit(x_fit, y_fit, 1)
p = np.poly1d(z)
x_range = np.linspace(x_fit.min(), x_fit.max(), 100)
ax2.plot(x_range, p(x_range), '--', color='gray', linewidth=1.5, label=f'Trend (r={corr:.3f})', zorder=1)

ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax2.set_xlabel('Daily Mean Temperature (C)')
ax2.set_ylabel('Daily Mean Spread DA-Imb (EUR/MWh)')
ax2.set_title('Panel 2: Daily Temperature vs DA-Imbalance Spread (July highlighted)')
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(alpha=0.3)

# --- Panel 3: Hourly spread profile for July 2024, July 2025, and yearly averages ---
ax3 = axes[2]

hours = range(24)
# Yearly averages
for year, ls, color in [(2024, '-', '#aec7e8'), (2025, '-', '#c7c7c7')]:
    yr = df[df['year'] == year]
    if len(yr) == 0:
        continue
    hourly_avg = yr.groupby('hour')['spread_da_imb'].mean()
    ax3.plot(hours, [hourly_avg.get(h, np.nan) for h in hours], ls,
             color=color, linewidth=2, label=f'{year} avg (all months)', alpha=0.7)

# July specific
for year, marker, color in [(2024, 'o', '#d62728'), (2025, 's', '#ff7f0e')]:
    jul = df[(df['year'] == year) & (df['month'] == 7)]
    if len(jul) == 0:
        continue
    hourly_jul = jul.groupby('hour')['spread_da_imb'].mean()
    ax3.plot(hours, [hourly_jul.get(h, np.nan) for h in hours], '-',
             color=color, linewidth=2.5, marker=marker, markersize=4,
             label=f'July {year}')

ax3.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax3.set_xlabel('Hour of Day')
ax3.set_ylabel('Mean Spread DA-Imb (EUR/MWh)')
ax3.set_title('Panel 3: Hourly DA-Imbalance Spread Profile -- July vs Yearly Average')
ax3.set_xticks(range(24))
ax3.legend(fontsize=8, loc='lower right')
ax3.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = f"{BASE}/MarketPriceGap/strategies/weather_impact/july_spread_anomaly.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"[+] Chart saved to {out_path}")
plt.close()

print("\n[+] Analysis complete.")
