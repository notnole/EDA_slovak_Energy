"""
Deficit Root Cause Analysis
Reproduces: deficit_root_cause_analysis.png

Investigates why the Jan 2026 imbalance market shifted toward deficit,
examining deficit frequency, temperature, load forecast bias, and
load error -> imbalance transmission.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ============================================================
# PATHS
# ============================================================
BASE = Path(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data')
MKT_PATH = BASE / 'MarketPriceGap' / 'data' / 'processed' / 'hourly_market_prices.csv'
TEMP_PATH = BASE / 'RawData' / 'ActualTemp2024-2026.csv'
LOAD_PATH = BASE / 'features' / 'DamasLoad' / 'load_data.csv'
OUT_DIR = BASE / 'MarketPriceGap' / 'strategies' / 'weather_impact'

# ============================================================
# 1. LOAD MARKET DATA
# ============================================================
print("[*] Loading market data...")
df_mkt = pd.read_csv(MKT_PATH, parse_dates=['timestamp_hour'])
df_mkt = df_mkt.rename(columns={'timestamp_hour': 'datetime'})
df_mkt['hour'] = df_mkt['datetime'].dt.hour
df_mkt['date'] = df_mkt['datetime'].dt.date
print(f"[+] Market data: {len(df_mkt)} records, {df_mkt['datetime'].min()} to {df_mkt['datetime'].max()}")

# ============================================================
# 2. PARSE TEMPERATURE DATA
# ============================================================
print("[*] Parsing temperature data...")

lines = open(TEMP_PATH, 'r', encoding='utf-8-sig').readlines()

records = []
for line in lines[1:]:  # skip header
    parts = line.strip().rstrip(',').split(',')
    # Valid rows: datetime, '000', int_part, dec_part
    # Invalid rows: datetime, '000', '(invalid)'
    if len(parts) >= 4 and '(invalid)' not in line:
        dt_str = parts[0]
        try:
            temp_str = parts[2] + '.' + parts[3]
            temp = float(temp_str)
            dt = pd.to_datetime(dt_str, format='%d/%m/%Y %H:%M:%S')
            records.append({'datetime': dt, 'temp': temp})
        except Exception:
            pass

df_temp = pd.DataFrame(records)
df_temp = df_temp.sort_values('datetime').reset_index(drop=True)
df_temp['date'] = df_temp['datetime'].dt.date
print(f"[+] Temperature: {len(df_temp)} hourly records")
print(f"    Range: {df_temp['temp'].min():.1f} to {df_temp['temp'].max():.1f} C")

# Daily mean temperature
daily_temp = df_temp.groupby('date').agg(
    temp_mean=('temp', 'mean')
).reset_index()
daily_temp['date'] = pd.to_datetime(daily_temp['date']).dt.date

# ============================================================
# 3. LOAD FORECAST DATA
# ============================================================
print("[*] Loading load forecast data...")
df_load = pd.read_csv(LOAD_PATH, parse_dates=['datetime'])

# Load data uses hour 1-24, we need 0-23 to match market data
# Derive hour from datetime instead
df_load['hour_0'] = df_load['datetime'].dt.hour
df_load['date'] = df_load['datetime'].dt.date

# forecast_error_mw = actual - forecast (positive = under-forecast)
print(f"[+] Load data: {len(df_load)} records")
print(f"    Columns: {list(df_load.columns)}")

# ============================================================
# 4. DEFINE PERIODS
# ============================================================
print("[*] Defining analysis periods...")

def in_sep_dec_25(dt_or_date):
    """Check if date falls in Sep-Dec 2025"""
    if hasattr(dt_or_date, 'date'):
        d = dt_or_date.date()
    else:
        d = dt_or_date
    return pd.Timestamp('2025-09-01').date() <= d < pd.Timestamp('2026-01-01').date()

def in_jan_26(dt_or_date):
    """Check if date falls in Jan 2026"""
    if hasattr(dt_or_date, 'date'):
        d = dt_or_date.date()
    else:
        d = dt_or_date
    return pd.Timestamp('2026-01-01').date() <= d < pd.Timestamp('2026-02-01').date()

# Tag market data
mkt_sep = df_mkt[df_mkt['datetime'].apply(in_sep_dec_25)].copy()
mkt_jan = df_mkt[df_mkt['datetime'].apply(in_jan_26)].copy()
print(f"[+] Market Sep-Dec 2025: {len(mkt_sep)} hours")
print(f"[+] Market Jan 2026: {len(mkt_jan)} hours")

# Tag load data
load_sep = df_load[df_load['datetime'].apply(in_sep_dec_25)].copy()
load_jan = df_load[df_load['datetime'].apply(in_jan_26)].copy()
print(f"[+] Load Sep-Dec 2025: {len(load_sep)} hours")
print(f"[+] Load Jan 2026: {len(load_jan)} hours")

# Tag temperature data
daily_temp_sep = daily_temp[daily_temp['date'].apply(in_sep_dec_25)].copy()
daily_temp_jan = daily_temp[daily_temp['date'].apply(in_jan_26)].copy()

# ============================================================
# 5. MERGE LOAD + MARKET FOR SCATTER (Panel 4)
# ============================================================
print("[*] Merging load and market data for scatter analysis...")

# Match on datetime (both are hourly)
merged_sep = pd.merge(
    mkt_sep[['datetime', 'imbalance_mwh']],
    load_sep[['datetime', 'forecast_error_mw']],
    on='datetime', how='inner'
).dropna(subset=['imbalance_mwh', 'forecast_error_mw'])

merged_jan = pd.merge(
    mkt_jan[['datetime', 'imbalance_mwh']],
    load_jan[['datetime', 'forecast_error_mw']],
    on='datetime', how='inner'
).dropna(subset=['imbalance_mwh', 'forecast_error_mw'])

print(f"[+] Merged Sep-Dec 2025: {len(merged_sep)} points")
print(f"[+] Merged Jan 2026: {len(merged_jan)} points")

# Compute correlations
if len(merged_sep) > 2:
    r_sep = merged_sep['forecast_error_mw'].corr(merged_sep['imbalance_mwh'])
else:
    r_sep = np.nan
if len(merged_jan) > 2:
    r_jan = merged_jan['forecast_error_mw'].corr(merged_jan['imbalance_mwh'])
else:
    r_jan = np.nan
print(f"[+] Correlation load_error vs imbalance: Sep-Dec r={r_sep:.2f}, Jan r={r_jan:.2f}")

# ============================================================
# 6. DAILY IMBALANCE FOR TEMPERATURE SCATTER (Panel 2)
# ============================================================
daily_imb = df_mkt.groupby('date').agg(
    mean_imb=('imbalance_mwh', 'mean')
).reset_index()

# Merge with daily temperature
daily_merged = pd.merge(
    pd.DataFrame({'date': daily_temp['date'], 'temp_mean': daily_temp['temp_mean']}),
    daily_imb,
    on='date', how='inner'
).dropna()

daily_merged_sep = daily_merged[daily_merged['date'].apply(in_sep_dec_25)].copy()
daily_merged_jan = daily_merged[daily_merged['date'].apply(in_jan_26)].copy()

# Trend line using only Sep-Dec 2025 + Jan 2026 data (the analysis periods)
daily_merged_period = pd.concat([daily_merged_sep, daily_merged_jan], ignore_index=True)
all_temp = daily_merged_period['temp_mean'].values
all_imb = daily_merged_period['mean_imb'].values
valid = ~(np.isnan(all_temp) | np.isnan(all_imb))
if valid.sum() > 2:
    slope, intercept, _, _, _ = stats.linregress(all_temp[valid], all_imb[valid])
else:
    slope, intercept = 0, 0
print(f"[+] Temperature-imbalance trend: {slope:.1f} MWh/C")

# ============================================================
# 7. CREATE VISUALIZATION
# ============================================================
print("\n[*] Creating root cause analysis chart...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Why Jan 2026 Imbalance Market Changed: Root Cause Analysis',
             fontsize=14, fontweight='bold')

# --- Panel 1 (top-left): Deficit Frequency by Hour ---
ax = axes[0, 0]

def deficit_pct_by_hour(data):
    """Compute % of hours where imbalance_mwh < 0, by hour"""
    result = {}
    for h in range(24):
        sub = data[data['hour'] == h]['imbalance_mwh'].dropna()
        if len(sub) > 0:
            result[h] = (sub < 0).mean() * 100
        else:
            result[h] = np.nan
    return pd.Series(result)

def_sep = deficit_pct_by_hour(mkt_sep)
def_jan = deficit_pct_by_hour(mkt_jan)
avg_def_sep = mkt_sep['imbalance_mwh'].dropna().pipe(lambda s: (s < 0).mean() * 100)
avg_def_jan = mkt_jan['imbalance_mwh'].dropna().pipe(lambda s: (s < 0).mean() * 100)

x_h = np.arange(24)
width = 0.4
ax.bar(x_h - width/2, def_sep.reindex(range(24)).values, width,
       color='steelblue', alpha=0.7, label=f'Sep-Dec 2025 (avg {avg_def_sep:.0f}%)')
ax.bar(x_h + width/2, def_jan.reindex(range(24)).values, width,
       color='indianred', alpha=0.7, label=f'Jan 2026 (avg {avg_def_jan:.0f}%)')
ax.axhline(y=50, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('% Hours in Deficit')
ax.set_title('Q1: Deficit Frequency by Hour')
ax.set_xticks(range(0, 24))
ax.set_ylim(20, 80)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# --- Panel 2 (top-right): Temperature vs System Imbalance ---
ax = axes[0, 1]

ax.scatter(daily_merged_sep['temp_mean'], daily_merged_sep['mean_imb'],
           color='steelblue', alpha=0.5, s=30, label='Sep-Dec 2025')
ax.scatter(daily_merged_jan['temp_mean'], daily_merged_jan['mean_imb'],
           color='indianred', alpha=0.5, s=30, label='Jan 2026')

# Trend line across full range
x_line = np.linspace(daily_merged['temp_mean'].min() - 1,
                     daily_merged['temp_mean'].max() + 1, 100)
ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1.5,
        label=f'Trend: {slope:.1f} MWh/C')
ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Mean Daily Imbalance (MWh)')
ax.set_title('Q3: Temperature vs System Imbalance')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Annotate Jan 2026 label
if len(daily_merged_jan) > 0:
    jan_x = daily_merged_jan['temp_mean'].mean()
    jan_y = daily_merged_jan['mean_imb'].min()
    ax.annotate('Jan 2026', xy=(jan_x, jan_y), fontsize=9, color='indianred',
                fontweight='bold')

# Annotate Sep-Dec 2025 label
if len(daily_merged_sep) > 0:
    sep_x = daily_merged_sep['temp_mean'].quantile(0.75)
    sep_y = daily_merged_sep['mean_imb'].quantile(0.75)
    ax.annotate('Sep-Dec 2025', xy=(sep_x, sep_y), fontsize=9, color='steelblue',
                fontweight='bold')

# --- Panel 3 (bottom-left): Load Forecast Bias by Hour ---
ax = axes[1, 0]

bias_sep = load_sep.groupby('hour_0')['forecast_error_mw'].mean()
bias_jan = load_jan.groupby('hour_0')['forecast_error_mw'].mean()

ax.bar(x_h - width/2, bias_sep.reindex(range(24)).fillna(0).values, width,
       color='steelblue', alpha=0.7, label='Sep-Dec 2025')
ax.bar(x_h + width/2, bias_jan.reindex(range(24)).fillna(0).values, width,
       color='indianred', alpha=0.7, label='Jan 2026')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Load Forecast Error (MW, +ve = under-forecast)')
ax.set_title('Q4: Load Forecast Bias by Hour')
ax.set_xticks(range(0, 24))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# --- Panel 4 (bottom-right): Load Error -> Imbalance Transmission ---
ax = axes[1, 1]

ax.scatter(merged_sep['forecast_error_mw'], merged_sep['imbalance_mwh'],
           color='steelblue', alpha=0.15, s=8,
           label=f'Sep-Dec 2025 (r={r_sep:.2f})')
ax.scatter(merged_jan['forecast_error_mw'], merged_jan['imbalance_mwh'],
           color='indianred', alpha=0.15, s=8,
           label=f'Jan 2026 (r={r_jan:.2f})')
ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
ax.set_xlim(-400, 400)
ax.set_ylim(-300, 300)
ax.set_xlabel('Load Forecast Error (MW)')
ax.set_ylabel('System Imbalance (MWh)')
ax.set_title('Q4: Load Error -> Imbalance Transmission')
ax.legend(fontsize=8, markerscale=3)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / 'deficit_root_cause_analysis.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[+] Saved: {out_path}")

print("\n[+] Root cause analysis complete.")
