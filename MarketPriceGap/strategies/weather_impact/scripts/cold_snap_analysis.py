"""
Cold Snap Market Impact Analysis
Identifies all cold snap periods (daily mean temp < 0C for 3+ consecutive days)
and cross-references with DA/imbalance market data.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ============================================================
# 1. PARSE TEMPERATURE DATA
# ============================================================
print("[*] Parsing temperature data...")

temp_path = r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data\RawData\ActualTemp2024-2026.csv'
lines = open(temp_path, 'r', encoding='utf-8-sig').readlines()

records = []
for line in lines[1:]:  # skip header
    parts = line.strip().rstrip(',').split(',')
    # Valid rows have at least 4 parts: datetime, '000', int_part, dec_part
    # Invalid rows have: datetime, '000', '(invalid)'
    if len(parts) >= 4 and '(invalid)' not in line:
        dt_str = parts[0]  # DD/MM/YYYY HH:MM:SS
        # parts[1] = '000' (milliseconds, ignore)
        # parts[2] + '.' + parts[3] = temperature with European decimal
        try:
            temp_str = parts[2] + '.' + parts[3]
            temp = float(temp_str)
            dt = pd.to_datetime(dt_str, format='%d/%m/%Y %H:%M:%S')
            records.append({'datetime': dt, 'temp': temp})
        except Exception:
            pass

df_temp = pd.DataFrame(records)
df_temp = df_temp.sort_values('datetime').reset_index(drop=True)
print(f"[+] Parsed {len(df_temp)} hourly temperature records")
print(f"    Date range: {df_temp['datetime'].min()} to {df_temp['datetime'].max()}")
print(f"    Temp range: {df_temp['temp'].min():.1f} to {df_temp['temp'].max():.1f} C")

# ============================================================
# 2. DAILY TEMPERATURE STATS
# ============================================================
print("\n[*] Computing daily temperature statistics...")

df_temp['date'] = df_temp['datetime'].dt.date
daily = df_temp.groupby('date').agg(
    temp_mean=('temp', 'mean'),
    temp_min=('temp', 'min'),
    temp_max=('temp', 'max'),
    n_obs=('temp', 'count')
).reset_index()
daily['date'] = pd.to_datetime(daily['date'])

# Only keep days with at least 20 hourly observations
daily = daily[daily['n_obs'] >= 20].reset_index(drop=True)
print(f"[+] {len(daily)} complete days of temperature data")

# ============================================================
# 3. FIND ALL COLD SNAP PERIODS
# ============================================================
print("\n[*] Identifying cold snap periods (daily mean < 0C, 3+ consecutive days)...")

daily['below_zero'] = daily['temp_mean'] < 0.0

# Find consecutive groups of below-zero days
daily['group'] = (daily['below_zero'] != daily['below_zero'].shift()).cumsum()
cold_groups = daily[daily['below_zero']].groupby('group').agg(
    start=('date', 'min'),
    end=('date', 'max'),
    duration=('date', 'count'),
    mean_temp=('temp_mean', 'mean'),
    min_temp=('temp_min', 'min'),
    max_temp=('temp_max', 'max'),
    coldest_day_mean=('temp_mean', 'min')
).reset_index(drop=True)

# Filter: 3+ consecutive days
cold_snaps = cold_groups[cold_groups['duration'] >= 3].reset_index(drop=True)
print(f"[+] Found {len(cold_snaps)} cold snap periods\n")

# Print comprehensive table
print("=" * 110)
print(f"{'#':>3} | {'Start':>12} | {'End':>12} | {'Days':>4} | {'Mean T':>7} | {'Min T':>7} | {'Max T':>7} | {'Coldest Day':>12} | {'Coldest Mean':>12}")
print("-" * 110)
for i, row in cold_snaps.iterrows():
    coldest_idx = daily[(daily['date'] >= row['start']) & (daily['date'] <= row['end'])]['temp_mean'].idxmin()
    coldest_date = daily.loc[coldest_idx, 'date']
    print(f"{i+1:>3} | {row['start'].strftime('%Y-%m-%d'):>12} | {row['end'].strftime('%Y-%m-%d'):>12} | "
          f"{row['duration']:>4} | {row['mean_temp']:>6.1f}C | {row['min_temp']:>6.1f}C | {row['max_temp']:>6.1f}C | "
          f"{coldest_date.strftime('%Y-%m-%d'):>12} | {row['coldest_day_mean']:>10.1f}C")
print("=" * 110)

# ============================================================
# 4. LOAD MARKET DATA
# ============================================================
print("\n[*] Loading market price data...")

mkt_path = r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data\MarketPriceGap\data\processed\hourly_market_prices.csv'
df_mkt = pd.read_csv(mkt_path, parse_dates=['timestamp_hour'])
df_mkt = df_mkt.rename(columns={'timestamp_hour': 'datetime'})
df_mkt['date'] = df_mkt['datetime'].dt.date
df_mkt['date'] = pd.to_datetime(df_mkt['date'])
print(f"[+] Loaded {len(df_mkt)} hourly market records")
print(f"    Date range: {df_mkt['datetime'].min()} to {df_mkt['datetime'].max()}")

# Check available columns
print(f"[*] Available market columns: {list(df_mkt.columns)}")

# ============================================================
# 5. COLD SNAP MARKET ANALYSIS
# ============================================================
print("\n[*] Analyzing market behavior during cold snaps...\n")

# Tag each market hour with cold_snap membership
df_mkt['in_cold_snap'] = False
df_mkt['cold_snap_id'] = np.nan

for i, snap in cold_snaps.iterrows():
    mask = (df_mkt['date'] >= snap['start']) & (df_mkt['date'] <= snap['end'])
    df_mkt.loc[mask, 'in_cold_snap'] = True
    df_mkt.loc[mask, 'cold_snap_id'] = i

# Per cold snap analysis
print("=" * 140)
print(f"{'#':>3} | {'Period':>25} | {'Days':>4} | {'Mean DA':>8} | {'Mean Imb Stl':>12} | {'Spread DA-Imb':>13} | {'% Deficit':>9} | {'Strategy PnL':>12} | {'PnL/MWh/h':>10}")
print("-" * 140)

snap_results = []
for i, snap in cold_snaps.iterrows():
    mask = (df_mkt['date'] >= snap['start']) & (df_mkt['date'] <= snap['end'])
    sub = df_mkt[mask].copy()

    n_hours = len(sub)
    if n_hours == 0:
        snap_results.append({
            'snap_id': i, 'start': snap['start'], 'end': snap['end'],
            'duration': snap['duration'], 'mean_temp': snap['mean_temp'],
            'mean_da': np.nan, 'mean_imb_stl': np.nan, 'mean_spread': np.nan,
            'pct_deficit': np.nan, 'total_pnl': np.nan, 'pnl_per_hour': np.nan,
            'n_hours': 0, 'n_hours_with_data': 0
        })
        print(f"{i+1:>3} | {snap['start'].strftime('%Y-%m-%d')} to {snap['end'].strftime('%Y-%m-%d'):>10} | {snap['duration']:>4} | {'NO MARKET DATA':>60}")
        continue

    # Compute metrics only on rows that have data
    has_da = sub['da_price'].notna()
    has_imb_stl = sub['imb_settlement_price'].notna()
    has_spread = sub['spread_da_imb'].notna()
    has_imb_mwh = sub['imbalance_mwh'].notna()

    mean_da = sub.loc[has_da, 'da_price'].mean() if has_da.any() else np.nan
    mean_imb_stl = sub.loc[has_imb_stl, 'imb_settlement_price'].mean() if has_imb_stl.any() else np.nan
    mean_spread = sub.loc[has_spread, 'spread_da_imb'].mean() if has_spread.any() else np.nan

    # % hours in deficit (imbalance_mwh < 0)
    if has_imb_mwh.any():
        pct_deficit = (sub.loc[has_imb_mwh, 'imbalance_mwh'] < 0).mean() * 100
    else:
        pct_deficit = np.nan

    # Strategy P&L: "sell DA, buy imbalance" = da_price - imb_settlement_price per MWh per hour
    # This is essentially the spread_da_imb column
    if has_spread.any():
        total_pnl = sub.loc[has_spread, 'spread_da_imb'].sum()
        pnl_per_hour = sub.loc[has_spread, 'spread_da_imb'].mean()
        n_hours_data = has_spread.sum()
    else:
        total_pnl = np.nan
        pnl_per_hour = np.nan
        n_hours_data = 0

    snap_results.append({
        'snap_id': i, 'start': snap['start'], 'end': snap['end'],
        'duration': snap['duration'], 'mean_temp': snap['mean_temp'],
        'mean_da': mean_da, 'mean_imb_stl': mean_imb_stl, 'mean_spread': mean_spread,
        'pct_deficit': pct_deficit, 'total_pnl': total_pnl, 'pnl_per_hour': pnl_per_hour,
        'n_hours': n_hours, 'n_hours_with_data': int(n_hours_data)
    })

    period_str = f"{snap['start'].strftime('%Y-%m-%d')} to {snap['end'].strftime('%Y-%m-%d')}"
    da_str = f"{mean_da:.1f}" if not np.isnan(mean_da) else "N/A"
    imb_str = f"{mean_imb_stl:.1f}" if not np.isnan(mean_imb_stl) else "N/A"
    spread_str = f"{mean_spread:.1f}" if not np.isnan(mean_spread) else "N/A"
    deficit_str = f"{pct_deficit:.0f}%" if not np.isnan(pct_deficit) else "N/A"
    pnl_str = f"{total_pnl:.0f}" if not np.isnan(total_pnl) else "N/A"
    pnl_h_str = f"{pnl_per_hour:.2f}" if not np.isnan(pnl_per_hour) else "N/A"

    print(f"{i+1:>3} | {period_str:>25} | {snap['duration']:>4} | {da_str:>8} | {imb_str:>12} | {spread_str:>13} | {deficit_str:>9} | {pnl_str:>12} | {pnl_h_str:>10}")

print("=" * 140)

snap_df = pd.DataFrame(snap_results)

# ============================================================
# 6. COLD SNAP vs NON-COLD-SNAP MONTHLY COMPARISON
# ============================================================
print("\n[*] Monthly comparison: cold snap vs non-cold-snap periods...")

# Daily market aggregation
daily_mkt = df_mkt.groupby('date').agg(
    mean_da=('da_price', 'mean'),
    mean_imb_stl=('imb_settlement_price', 'mean'),
    mean_spread=('spread_da_imb', 'mean'),
    pct_deficit=('imbalance_mwh', lambda x: (x.dropna() < 0).mean() * 100 if x.notna().any() else np.nan),
    total_spread=('spread_da_imb', 'sum'),
    n_hours_spread=('spread_da_imb', lambda x: x.notna().sum())
).reset_index()

daily_mkt['month'] = daily_mkt['date'].dt.to_period('M')
daily_mkt['in_cold_snap'] = False
for _, snap in cold_snaps.iterrows():
    mask = (daily_mkt['date'] >= snap['start']) & (daily_mkt['date'] <= snap['end'])
    daily_mkt.loc[mask, 'in_cold_snap'] = True

# Monthly summary
months_with_cold = daily_mkt[daily_mkt['in_cold_snap']].groupby('month').agg(
    cold_days=('date', 'count'),
    cold_mean_da=('mean_da', 'mean'),
    cold_mean_spread=('mean_spread', 'mean'),
    cold_total_pnl=('total_spread', 'sum')
).reset_index()

months_all = daily_mkt.groupby('month').agg(
    total_days=('date', 'count'),
    overall_mean_da=('mean_da', 'mean'),
    overall_mean_spread=('mean_spread', 'mean'),
    overall_total_pnl=('total_spread', 'sum')
).reset_index()

months_non_cold = daily_mkt[~daily_mkt['in_cold_snap']].groupby('month').agg(
    non_cold_days=('date', 'count'),
    non_cold_mean_da=('mean_da', 'mean'),
    non_cold_mean_spread=('mean_spread', 'mean'),
    non_cold_total_pnl=('total_spread', 'sum')
).reset_index()

monthly = months_all.merge(months_with_cold, on='month', how='left').merge(months_non_cold, on='month', how='left')
monthly = monthly.fillna(0)

print(f"\n{'Month':>8} | {'Tot Days':>8} | {'Cold Days':>9} | {'All DA':>7} | {'Cold DA':>8} | {'Non-C DA':>8} | {'All Spread':>10} | {'Cold Spread':>11} | {'Non-C Spread':>12}")
print("-" * 110)
for _, row in monthly.iterrows():
    cold_da_str = f"{row['cold_mean_da']:.1f}" if row['cold_days'] > 0 else "-"
    cold_spr_str = f"{row['cold_mean_spread']:.1f}" if row['cold_days'] > 0 else "-"
    print(f"{str(row['month']):>8} | {row['total_days']:>8.0f} | {row['cold_days']:>9.0f} | "
          f"{row['overall_mean_da']:>7.1f} | {cold_da_str:>8} | {row['non_cold_mean_da']:>8.1f} | "
          f"{row['overall_mean_spread']:>10.1f} | {cold_spr_str:>11} | {row['non_cold_mean_spread']:>12.1f}")
print("-" * 110)

# Summary stats
cold_hours = df_mkt[df_mkt['in_cold_snap'] & df_mkt['spread_da_imb'].notna()]
non_cold_hours = df_mkt[~df_mkt['in_cold_snap'] & df_mkt['spread_da_imb'].notna()]

print(f"\n--- AGGREGATE COMPARISON ---")
if len(cold_hours) > 0:
    print(f"[+] Cold snap hours:     {len(cold_hours):>6} | Mean DA: {cold_hours['da_price'].mean():>7.1f} | Mean Spread: {cold_hours['spread_da_imb'].mean():>7.2f} EUR/MWh | Total PnL: {cold_hours['spread_da_imb'].sum():>10.0f} EUR")
else:
    print("[-] No cold snap hours with market data")
if len(non_cold_hours) > 0:
    print(f"[+] Non-cold-snap hours: {len(non_cold_hours):>6} | Mean DA: {non_cold_hours['da_price'].mean():>7.1f} | Mean Spread: {non_cold_hours['spread_da_imb'].mean():>7.2f} EUR/MWh | Total PnL: {non_cold_hours['spread_da_imb'].sum():>10.0f} EUR")

# ============================================================
# 7. VISUALIZATION
# ============================================================
print("\n[*] Creating visualization...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1, 0.8]})

# --- Panel 1: Daily mean temperature with cold snap highlighting ---
ax1 = axes[0]
ax1.plot(daily['date'], daily['temp_mean'], color='#333333', linewidth=0.6, alpha=0.8, label='Daily mean temp')
ax1.axhline(y=0, color='red', linewidth=0.8, linestyle='--', alpha=0.5, label='0 C threshold')

for i, snap in cold_snaps.iterrows():
    ax1.axvspan(snap['start'], snap['end'] + pd.Timedelta(days=1),
                alpha=0.25, color='steelblue', zorder=0,
                label='Cold snap' if i == 0 else None)

ax1.set_ylabel('Temperature (C)')
ax1.set_title('Daily Mean Temperature with Cold Snap Periods (mean < 0C, 3+ days)', fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', fontsize=8)
ax1.set_xlim(daily['date'].min(), daily['date'].max())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=30)

# --- Panel 2: Daily spread with cold snap highlighting ---
ax2 = axes[1]

# Compute daily spread
daily_spread = df_mkt.groupby(df_mkt['datetime'].dt.date).agg(
    mean_spread=('spread_da_imb', 'mean')
).reset_index()
daily_spread.columns = ['date', 'mean_spread']
daily_spread['date'] = pd.to_datetime(daily_spread['date'])
daily_spread = daily_spread.dropna(subset=['mean_spread']).sort_values('date')

# 7-day rolling average
daily_spread['rolling_7d'] = daily_spread['mean_spread'].rolling(7, min_periods=1).mean()

ax2.bar(daily_spread['date'], daily_spread['mean_spread'], color='gray', alpha=0.3, width=1.0, label='Daily spread')
ax2.plot(daily_spread['date'], daily_spread['rolling_7d'], color='darkblue', linewidth=1.5, label='7-day rolling avg')
ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='-')

for i, snap in cold_snaps.iterrows():
    ax2.axvspan(snap['start'], snap['end'] + pd.Timedelta(days=1),
                alpha=0.20, color='steelblue', zorder=0,
                label='Cold snap' if i == 0 else None)

ax2.set_ylabel('Spread DA-Imb (EUR/MWh)')
ax2.set_title('DA minus Imbalance Settlement Price Spread with Cold Snap Periods', fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', fontsize=8)
ax2.set_xlim(daily_spread['date'].min(), daily_spread['date'].max())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=30)

# --- Panel 3: Bar chart - strategy PnL per cold snap vs monthly ---
ax3 = axes[2]

# Cold snap PnL bars
valid_snaps = snap_df[snap_df['total_pnl'].notna()].copy()
if len(valid_snaps) > 0:
    snap_labels = [f"CS{int(r['snap_id'])+1}\n{r['start'].strftime('%b%y')}\n({int(r['duration'])}d)"
                   for _, r in valid_snaps.iterrows()]
    snap_pnls = valid_snaps['pnl_per_hour'].values

    # Monthly averages for comparison
    monthly_pnl = df_mkt.groupby(df_mkt['datetime'].dt.to_period('M')).agg(
        mean_spread=('spread_da_imb', 'mean')
    ).reset_index()
    monthly_pnl.columns = ['month', 'mean_spread']
    monthly_pnl = monthly_pnl.dropna()
    month_labels = [str(m) for m in monthly_pnl['month']]
    month_pnls = monthly_pnl['mean_spread'].values

    n_snaps = len(snap_labels)
    n_months = len(month_labels)

    # Plot cold snaps first, then monthly
    x_snaps = np.arange(n_snaps)
    x_months = np.arange(n_snaps + 1, n_snaps + 1 + n_months)

    colors_snap = ['steelblue' if p >= 0 else 'indianred' for p in snap_pnls]
    colors_month = ['#888888' if p >= 0 else '#cc6666' for p in month_pnls]

    bars1 = ax3.bar(x_snaps, snap_pnls, color=colors_snap, edgecolor='black', linewidth=0.5, label='Cold snap avg spread')
    bars2 = ax3.bar(x_months, month_pnls, color=colors_month, edgecolor='black', linewidth=0.3, alpha=0.7, label='Monthly avg spread')

    ax3.set_xticks(np.concatenate([x_snaps, x_months]))
    all_labels = list(snap_labels) + list(month_labels)
    ax3.set_xticklabels(all_labels, fontsize=6, rotation=45, ha='right')

    # Add divider line
    ax3.axvline(x=n_snaps + 0.5, color='black', linewidth=1.0, linestyle='--', alpha=0.5)
    ax3.text(n_snaps / 2, ax3.get_ylim()[1] * 0.9 if ax3.get_ylim()[1] > 0 else 5,
             'Cold Snaps', ha='center', fontsize=9, fontweight='bold', color='steelblue')
    ax3.text(n_snaps + 1 + n_months / 2, ax3.get_ylim()[1] * 0.9 if ax3.get_ylim()[1] > 0 else 5,
             'Monthly', ha='center', fontsize=9, fontweight='bold', color='gray')

ax3.axhline(y=0, color='black', linewidth=0.5)
ax3.set_ylabel('Avg Spread (EUR/MWh/h)')
ax3.set_title('Strategy P&L: Sell DA / Buy Imbalance -- Cold Snap Periods vs Monthly Averages', fontsize=11, fontweight='bold')
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

out_path = r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data\MarketPriceGap\strategies\weather_impact\cold_snap_market_impact.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[+] Saved visualization to: {out_path}")

print("\n[+] Cold snap market impact analysis complete.")
