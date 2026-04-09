"""
Cold Snap Day-Decay Analysis
Analyzes whether "sell DA, buy imbalance" strategy losses during cold snaps
are concentrated in the first few days, with the market adapting afterwards.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

print('[*] Loading market data...')
mkt = pd.read_csv(
    r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data\MarketPriceGap\data\processed\hourly_market_prices.csv',
    parse_dates=['timestamp_hour']
)
mkt['date'] = mkt['timestamp_hour'].dt.date

print('[*] Loading temperature data...')
# Parse temperature from raw CSV line-by-line to handle European decimal comma format
# Format: DD/MM/YYYY HH:MM:SS,000,integer_part,decimal_part,
# Decimal part varies: 2-digit (29 => .29) or 4-digit (0711 => .0711)
temp_records = []
with open(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data\RawData\ActualTemp2024-2026.csv', 'r') as f:
    next(f)  # skip header
    for line in f:
        line = line.strip().rstrip(',')
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 4:
            continue
        dt_str = parts[0]
        int_str = parts[2].strip()
        if int_str == '(invalid)' or int_str == '':
            continue
        dec_str = parts[3].strip()
        if dec_str == '' or dec_str == '(invalid)':
            continue
        # Build temperature: "integer_part.decimal_part" as string, then parse
        is_neg = int_str.startswith('-')
        abs_int = int_str.lstrip('-')
        temp_str = abs_int + '.' + dec_str
        try:
            temp_val = float(temp_str)
        except ValueError:
            continue
        if is_neg:
            temp_val = -temp_val
        try:
            dt = datetime.strptime(dt_str.strip()[:19], '%d/%m/%Y %H:%M:%S')
        except Exception:
            continue
        temp_records.append({'datetime': dt, 'temp_c': temp_val})

tmp = pd.DataFrame(temp_records)
tmp['date'] = tmp['datetime'].dt.date

print(f'[+] Temperature: {len(tmp)} hours, range {tmp["temp_c"].min():.1f} to {tmp["temp_c"].max():.1f} C')

# Define cold snap periods
cold_snaps = [
    ('2024-12-29', '2025-01-05', '2024-12-29 to 2025-01-05'),
    ('2025-01-14', '2025-01-16', '2025-01-14 to 2025-01-16'),
    ('2025-01-18', '2025-01-22', '2025-01-18 to 2025-01-22'),
    ('2025-02-11', '2025-02-13', '2025-02-11 to 2025-02-13'),
    ('2025-02-15', '2025-02-22', '2025-02-15 to 2025-02-22'),
    ('2025-12-29', '2026-01-01', '2025-12-29 to 2026-01-01'),
    ('2026-01-03', '2026-01-15', '2026-01-03 to 2026-01-15'),
    ('2026-01-18', '2026-01-23', '2026-01-18 to 2026-01-23'),
]

print('[*] Computing per-day metrics for each cold snap...')

all_rows = []
snap_trajectories = {}

for i, (start, end, label) in enumerate(cold_snaps):
    dates = pd.date_range(start, end, freq='D')
    snap_daily = []

    for day_num, dt in enumerate(dates, 1):
        d = dt.date()
        day_mkt = mkt[mkt['date'] == d]
        day_tmp = tmp[tmp['date'] == d]

        if len(day_mkt) == 0:
            continue

        valid_spread = day_mkt['spread_da_imb'].dropna()
        valid_imb = day_mkt['imbalance_mwh'].dropna()

        if len(valid_spread) == 0:
            continue

        mean_spread = valid_spread.mean()
        mean_imb = valid_imb.mean() if len(valid_imb) > 0 else np.nan
        pct_deficit = (valid_imb < 0).mean() * 100 if len(valid_imb) > 0 else np.nan
        mean_temp = day_tmp['temp_c'].mean() if len(day_tmp) > 0 else np.nan
        n_hours = len(valid_spread)
        daily_pnl = valid_spread.sum()  # EUR for 1 MW position

        row = {
            'snap_id': i + 1,
            'snap_label': label,
            'date': d,
            'day_num': day_num,
            'mean_spread': mean_spread,
            'mean_imb': mean_imb,
            'pct_deficit': pct_deficit,
            'mean_temp': mean_temp,
            'n_hours': n_hours,
            'daily_pnl': daily_pnl,
        }
        all_rows.append(row)
        snap_daily.append(row)

    if snap_daily:
        cum_pnl = np.cumsum([r['daily_pnl'] for r in snap_daily])
        day_nums = [r['day_num'] for r in snap_daily]
        snap_trajectories[label] = (day_nums, cum_pnl, len(dates))

df = pd.DataFrame(all_rows)
print(f'[+] Total day-snap observations: {len(df)}')

# Per-snap summary
print()
print('=== Per Cold Snap Summary ===')
for snap_id in df['snap_id'].unique():
    sub = df[df['snap_id'] == snap_id]
    label = sub['snap_label'].iloc[0]
    total_pnl = sub['daily_pnl'].sum()
    print(f'  Snap {snap_id} ({label}): {len(sub)}d, total P&L = {total_pnl:+.0f} EUR, '
          f'mean spread = {sub["mean_spread"].mean():+.1f} EUR/MWh, '
          f'mean temp = {sub["mean_temp"].mean():.1f} C')

# Bucketed aggregation (day 8+)
df['day_label'] = df['day_num'].apply(lambda x: str(x) if x < 8 else '8+')
df['day_sort'] = df['day_num'].clip(upper=8)

agg_bucketed = df.groupby(['day_sort', 'day_label']).agg(
    mean_spread=('mean_spread', 'mean'),
    std_spread=('mean_spread', 'std'),
    mean_daily_pnl=('daily_pnl', 'mean'),
    n_snaps=('snap_id', 'count'),
    mean_temp=('mean_temp', 'mean'),
    pct_deficit=('pct_deficit', 'mean'),
).reset_index().sort_values('day_sort')

print()
print('=== Aggregated by Day Within Cold Snap ===')
for _, r in agg_bucketed.iterrows():
    print(f'  Day {r["day_label"]:>3s}: spread = {r["mean_spread"]:+6.1f} EUR/MWh, '
          f'P&L = {r["mean_daily_pnl"]:+7.0f} EUR, n = {int(r["n_snaps"])}, '
          f'deficit = {r["pct_deficit"]:.0f}%, temp = {r["mean_temp"]:.1f} C')

# Raw day-level
print()
print('=== Raw Day-Level (Not Bucketed) ===')
agg_raw = df.groupby('day_num').agg(
    mean_spread=('mean_spread', 'mean'),
    n_snaps=('snap_id', 'count'),
    mean_daily_pnl=('daily_pnl', 'mean'),
    pct_deficit=('pct_deficit', 'mean'),
    mean_temp=('mean_temp', 'mean'),
).reset_index()
for _, r in agg_raw.iterrows():
    print(f'  Day {int(r["day_num"]):>2d}: spread = {r["mean_spread"]:+6.1f} EUR/MWh, '
          f'P&L = {r["mean_daily_pnl"]:+7.0f} EUR, n = {int(r["n_snaps"])}, '
          f'deficit = {r["pct_deficit"]:.0f}%, temp = {r["mean_temp"]:.1f} C')

# ==================== PLOTTING ====================
print()
print('[*] Creating chart...')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'hspace': 0.35})

# --- Panel 1: Bar chart of mean spread by day ---
colors = ['#d62728' if v < 0 else '#2ca02c' for v in agg_bucketed['mean_spread']]
bars = ax1.bar(
    range(len(agg_bucketed)),
    agg_bucketed['mean_spread'].values,
    color=colors, edgecolor='black', linewidth=0.5, alpha=0.85
)

for idx, (bar, n) in enumerate(zip(bars, agg_bucketed['n_snaps'].values)):
    yval = bar.get_height()
    offset = -3 if yval < 0 else 3
    va = 'top' if yval < 0 else 'bottom'
    ax1.annotate(
        f'n={int(n)}', (bar.get_x() + bar.get_width() / 2, yval),
        ha='center', va=va, fontsize=9, fontweight='bold',
        xytext=(0, offset), textcoords='offset points'
    )

ax1.set_xticks(range(len(agg_bucketed)))
ax1.set_xticklabels(agg_bucketed['day_label'].values, fontsize=11)
ax1.set_xlabel('Day Within Cold Snap', fontsize=11)
ax1.set_ylabel('Mean DA-Imbalance Spread (EUR/MWh)', fontsize=11)
ax1.set_title(
    '"Sell DA, Buy Imbalance" Spread by Day Within Cold Snap\n'
    '(Aggregated Across All 8 Cold Snaps)',
    fontsize=12, fontweight='bold'
)
ax1.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax1.grid(axis='y', alpha=0.3)

# --- Panel 2: Cumulative P&L trajectories ---
cmap = plt.cm.tab10
for idx, (label, (day_nums, cum_pnl, n_days)) in enumerate(snap_trajectories.items()):
    short_label = label.replace(' to ', ' - ')
    ax2.plot(
        day_nums, cum_pnl, marker='o', markersize=4, linewidth=1.8,
        color=cmap(idx % 10), label=short_label, alpha=0.85
    )

ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax2.set_xlabel('Day Within Cold Snap', fontsize=11)
ax2.set_ylabel('Cumulative P&L (EUR, 1MW position)', fontsize=11)
ax2.set_title('Cumulative Strategy P&L Trajectory Per Cold Snap', fontsize=12, fontweight='bold')
ax2.legend(fontsize=7.5, loc='lower left', ncol=2, framealpha=0.9)
ax2.grid(alpha=0.3)

max_day = max(max(v[0]) for v in snap_trajectories.values())
ax2.set_xticks(range(1, max_day + 1))

out_path = r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data\MarketPriceGap\strategies\weather_impact\cold_snap_day_decay.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'[+] Chart saved to {out_path}')

# ==================== KEY FINDINGS ====================
print()
print('=== KEY FINDINGS ===')
day1_spread = agg_bucketed[agg_bucketed['day_sort'] == 1]['mean_spread'].values[0]
day2_spread = agg_bucketed[agg_bucketed['day_sort'] == 2]['mean_spread'].values[0]
day3_spread = agg_bucketed[agg_bucketed['day_sort'] == 3]['mean_spread'].values[0]
late_days = agg_bucketed[agg_bucketed['day_sort'] >= 5]
late_spread = late_days['mean_spread'].mean() if len(late_days) > 0 else 0

print(f'  Day 1 mean spread: {day1_spread:+.1f} EUR/MWh')
print(f'  Day 2 mean spread: {day2_spread:+.1f} EUR/MWh')
print(f'  Day 3 mean spread: {day3_spread:+.1f} EUR/MWh')
print(f'  Day 5+ mean spread: {late_spread:+.1f} EUR/MWh')

early_pnl = agg_bucketed[agg_bucketed['day_sort'] <= 3]['mean_daily_pnl'].sum()
late_pnl = agg_bucketed[agg_bucketed['day_sort'] > 3]['mean_daily_pnl'].sum()
total_pnl = agg_bucketed['mean_daily_pnl'].sum()
print(f'  Mean P&L days 1-3: {early_pnl:+.0f} EUR')
print(f'  Mean P&L days 4+:  {late_pnl:+.0f} EUR')
print(f'  Mean total:        {total_pnl:+.0f} EUR')

# Check adaptation hypothesis
if abs(late_spread) < abs(day1_spread) * 0.5:
    print('[+] ADAPTATION DETECTED: Late-snap losses are <50% of Day 1')
else:
    print('[-] NO CLEAR ADAPTATION: Losses persist throughout cold snaps')

print()
print('[+] Analysis complete.')
