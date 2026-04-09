"""
Temperature Forecast Analysis: GFS vs Actual
Effect on electricity load and day-ahead price.

Data: Ipesoft EDA export (Slovakia), Sept 2025 - Feb 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

BASE = Path(__file__).parent.parent.parent.parent
OUT = BASE / 'LoadAnalysis' / 'temperature_analysis'
OUT.mkdir(parents=True, exist_ok=True)

# --- Load temperature data ---
temp = pd.read_csv(BASE / 'data/clean/weather/temperature_15min.csv', parse_dates=['datetime'])
temp['temp_forecast_error'] = temp['temp_forecast_gfs'] - temp['temp_actual']
temp['hour'] = temp['datetime'].dt.floor('h')
temp_hourly = temp.groupby('hour').agg(
    temp_forecast=('temp_forecast_gfs', 'mean'),
    temp_actual=('temp_actual', 'first'),
    temp_error=('temp_forecast_error', 'mean')
).reset_index().rename(columns={'hour': 'datetime'})

print(f'[*] Temperature hourly: {len(temp_hourly)} rows')

# --- Load DAMAS Load data ---
load = pd.read_csv(BASE / 'features/DamasLoad/load_data.csv', parse_dates=['datetime'])
load = load[['datetime', 'actual_load_mw', 'forecast_load_mw', 'forecast_error_mw', 'forecast_error_pct']].copy()
print(f'[*] Load data: {len(load)} rows')

# --- Load DA Price data ---
da_raw = pd.read_csv(BASE / 'data/clean/market/day_ahead/da_auction_results.csv', sep=';')

def parse_eu_float(val):
    if pd.isna(val) or str(val).strip() == '':
        return np.nan
    return float(str(val).replace(',', '.'))

da_raw['price_eur_mwh'] = da_raw['price_sk_eur_mwh'].apply(parse_eu_float)
da_raw['trading_dt'] = pd.to_datetime(da_raw['trading_day'])
period_col = [c for c in da_raw.columns if 'slo per' in c.lower() or 'perió' in c.lower()][0]
da_raw['period_num'] = da_raw[period_col].astype(int)
period_mins = da_raw['Perióda (min)'].mode().values[0]

if period_mins == 15:
    da_raw['hour_offset'] = ((da_raw['period_num'] - 1) * 15) // 60
else:
    da_raw['hour_offset'] = da_raw['period_num'] - 1

da_raw['datetime'] = da_raw['trading_dt'] + pd.to_timedelta(da_raw['hour_offset'], unit='h')
da_hourly = da_raw.groupby('datetime')['price_eur_mwh'].mean().reset_index()
print(f'[*] DA price hourly: {len(da_hourly)} rows')

# --- Merge all ---
merged = temp_hourly.merge(load, on='datetime', how='inner')
merged = merged.merge(da_hourly, on='datetime', how='left')
merged['date'] = merged['datetime'].dt.date
merged['hour_of_day'] = merged['datetime'].dt.hour

valid = merged.dropna(subset=['temp_actual', 'actual_load_mw', 'price_eur_mwh', 'temp_error'])
print(f'[*] Merged dataset: {len(merged)} rows, valid for analysis: {len(valid)} rows')
print(f'    Period: {merged.datetime.min().date()} to {merged.datetime.max().date()}')

# =============================================
# PLOT 1: Temperature time series (forecast vs actual)
# =============================================
daily = merged.groupby('date').agg(
    temp_actual=('temp_actual', 'mean'),
    temp_forecast=('temp_forecast', 'mean'),
    temp_error=('temp_error', 'mean'),
    load=('actual_load_mw', 'mean'),
    load_error=('forecast_error_mw', 'mean'),
    price=('price_eur_mwh', 'mean')
).reset_index()
daily['date'] = pd.to_datetime(daily['date'])

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

ax = axes[0]
ax.plot(daily['date'], daily['temp_actual'], label='Actual', color='#e74c3c', linewidth=1.5)
ax.plot(daily['date'], daily['temp_forecast'], label='GFS Forecast', color='#3498db', linewidth=1.0, alpha=0.8)
ax.fill_between(daily['date'], daily['temp_actual'], daily['temp_forecast'], alpha=0.15, color='orange')
ax.set_ylabel('Temperature (C)')
ax.set_title('Daily Average Temperature: GFS Forecast vs Actual (Slovakia)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

ax = axes[1]
colors = np.where(daily['temp_error'] > 0, '#e74c3c', '#3498db')
ax.bar(daily['date'], daily['temp_error'], width=1, color=colors, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
mean_bias = daily['temp_error'].mean()
ax.axhline(mean_bias, color='red', linewidth=1, linestyle='--',
           label=f'Mean bias: {mean_bias:.2f} C')
ax.set_ylabel('Forecast Error (C)')
ax.set_title('GFS Temperature Forecast Error (Forecast - Actual)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(daily['date'], daily['load'], color='#2ecc71', linewidth=1.5)
ax.set_ylabel('Load (MW)')
ax.set_title('Daily Average Electricity Load (MW)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / '01_temperature_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print('[+] Saved 01_temperature_timeseries.png')

# =============================================
# PLOT 2: Temperature vs Load and Price scatter
# =============================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 2a: Temperature vs Load
ax = axes[0]
sc = ax.scatter(valid['temp_actual'], valid['actual_load_mw'], c=valid['hour_of_day'],
                cmap='twilight_shifted', alpha=0.3, s=10, edgecolors='none')
slope1, intercept1, r1, p1, se1 = stats.linregress(valid['temp_actual'], valid['actual_load_mw'])
x_line = np.linspace(valid['temp_actual'].min(), valid['temp_actual'].max(), 100)
ax.plot(x_line, slope1 * x_line + intercept1, 'r-', linewidth=2,
        label=f'r={r1:.3f}, slope={slope1:.1f} MW/C')
ax.set_xlabel('Actual Temperature (C)')
ax.set_ylabel('Actual Load (MW)')
ax.set_title('Temperature vs Electricity Load')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label='Hour of day')

# 2b: Temperature vs DA Price
ax = axes[1]
sc = ax.scatter(valid['temp_actual'], valid['price_eur_mwh'], c=valid['hour_of_day'],
                cmap='twilight_shifted', alpha=0.3, s=10, edgecolors='none')
slope2, intercept2, r2, p2, se2 = stats.linregress(valid['temp_actual'], valid['price_eur_mwh'])
ax.plot(x_line, slope2 * x_line + intercept2, 'r-', linewidth=2,
        label=f'r={r2:.3f}, slope={slope2:.1f} EUR/C')
ax.set_xlabel('Actual Temperature (C)')
ax.set_ylabel('DA Price (EUR/MWh)')
ax.set_title('Temperature vs DA Price')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label='Hour of day')

# 2c: Temp forecast error vs Load forecast error
ax = axes[2]
sc = ax.scatter(valid['temp_error'], valid['forecast_error_mw'], c=valid['hour_of_day'],
                cmap='twilight_shifted', alpha=0.3, s=10, edgecolors='none')
slope3, intercept3, r3, p3, se3 = stats.linregress(valid['temp_error'], valid['forecast_error_mw'])
x3 = np.linspace(valid['temp_error'].min(), valid['temp_error'].max(), 100)
ax.plot(x3, slope3 * x3 + intercept3, 'r-', linewidth=2,
        label=f'r={r3:.3f}, slope={slope3:.1f} MW/C')
ax.set_xlabel('Temp Forecast Error (C)')
ax.set_ylabel('Load Forecast Error (MW)')
ax.set_title('Temp Error vs Load Error')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label='Hour of day')

plt.tight_layout()
plt.savefig(OUT / '02_temperature_scatter_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print('[+] Saved 02_temperature_scatter_analysis.png')

# =============================================
# PLOT 3: Error distribution + sensitivity
# =============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3a: Histogram
ax = axes[0, 0]
ax.hist(valid['temp_error'], bins=60, color='#3498db', alpha=0.7, edgecolor='white')
mean_err = valid['temp_error'].mean()
median_err = valid['temp_error'].median()
ax.axvline(mean_err, color='red', linewidth=2, linestyle='--', label=f'Mean: {mean_err:.2f} C')
ax.axvline(median_err, color='orange', linewidth=2, linestyle='--', label=f'Median: {median_err:.2f} C')
ax.set_xlabel('GFS Forecast Error (C)')
ax.set_ylabel('Count')
ax.set_title('Temperature Forecast Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 3b: Hourly profile
ax = axes[0, 1]
hourly_stats = valid.groupby('hour_of_day')['temp_error'].agg(['mean', 'std']).reset_index()
ax.bar(hourly_stats['hour_of_day'], hourly_stats['mean'], color='#3498db', alpha=0.7, edgecolor='white')
ax.errorbar(hourly_stats['hour_of_day'], hourly_stats['mean'], yerr=hourly_stats['std'],
            fmt='none', color='black', capsize=3, alpha=0.5)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Mean Forecast Error (C)')
ax.set_title('GFS Temp Error by Hour of Day')
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 24))

# 3c: Load sensitivity by temp bin
ax = axes[1, 0]
valid_binned = valid.copy()
valid_binned['temp_bin'] = pd.cut(valid_binned['temp_actual'], bins=np.arange(-12, 30, 2))
temp_load = valid_binned.groupby('temp_bin', observed=True).agg(
    mean_load=('actual_load_mw', 'mean'),
    std_load=('actual_load_mw', 'std'),
    count=('actual_load_mw', 'count')
).reset_index()
temp_load = temp_load[temp_load['count'] > 10]
bin_labels = [f'{int(x.left)}' for x in temp_load['temp_bin']]
ax.bar(range(len(bin_labels)), temp_load['mean_load'], color='#2ecc71', alpha=0.7, edgecolor='white')
ax.errorbar(range(len(bin_labels)), temp_load['mean_load'], yerr=temp_load['std_load'],
            fmt='none', color='black', capsize=3, alpha=0.5)
ax.set_xticks(range(len(bin_labels)))
ax.set_xticklabels(bin_labels, rotation=45)
ax.set_xlabel('Temperature Bin (C)')
ax.set_ylabel('Mean Load (MW)')
ax.set_title('Load Sensitivity to Temperature')
ax.grid(True, alpha=0.3)

# 3d: Price sensitivity by temp bin
ax = axes[1, 1]
temp_price = valid_binned.groupby('temp_bin', observed=True).agg(
    mean_price=('price_eur_mwh', 'mean'),
    std_price=('price_eur_mwh', 'std'),
    count=('price_eur_mwh', 'count')
).reset_index()
temp_price = temp_price[temp_price['count'] > 10]
bin_labels2 = [f'{int(x.left)}' for x in temp_price['temp_bin']]
ax.bar(range(len(bin_labels2)), temp_price['mean_price'], color='#e67e22', alpha=0.7, edgecolor='white')
ax.errorbar(range(len(bin_labels2)), temp_price['mean_price'], yerr=temp_price['std_price'],
            fmt='none', color='black', capsize=3, alpha=0.5)
ax.set_xticks(range(len(bin_labels2)))
ax.set_xticklabels(bin_labels2, rotation=45)
ax.set_xlabel('Temperature Bin (C)')
ax.set_ylabel('Mean DA Price (EUR/MWh)')
ax.set_title('DA Price Sensitivity to Temperature')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / '03_forecast_error_and_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print('[+] Saved 03_forecast_error_and_sensitivity.png')

# =============================================
# PLOT 4: Rolling correlation + cold spell
# =============================================
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# 4a: Rolling 14-day correlation
ax = axes[0]
window = 14
roll_temp_load = daily['temp_actual'].rolling(window).corr(daily['load'])
roll_temperr_loaderr = daily['temp_error'].rolling(window).corr(daily['load_error'])

ax.plot(daily['date'], roll_temp_load, label='Temp vs Load (r)', color='#2ecc71', linewidth=2)
ax.plot(daily['date'], roll_temperr_loaderr, label='Temp Error vs Load Error (r)',
        color='#e74c3c', linewidth=1.5, linestyle='--')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Rolling Correlation (14-day)')
ax.set_title('Rolling Correlation: Temperature-Load and Error-Error')
ax.legend()
ax.grid(True, alpha=0.3)

# 4b: Cold spell impact
ax = axes[1]
ax2 = ax.twinx()
ax.plot(daily['date'], daily['temp_actual'], color='#3498db', linewidth=1.5, label='Temperature')
ax2.plot(daily['date'], daily['price'], color='#e67e22', linewidth=1.0, alpha=0.7, label='DA Price')
cold = daily[daily['temp_actual'] < 0]
for _, row in cold.iterrows():
    ax.axvspan(row['date'], row['date'] + pd.Timedelta(days=1), alpha=0.1, color='blue')
ax.set_ylabel('Temperature (C)', color='#3498db')
ax2.set_ylabel('DA Price (EUR/MWh)', color='#e67e22')
ax.set_title('Cold Spell Impact on DA Price (blue shading = below 0C)')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / '04_rolling_correlation_cold_spells.png', dpi=150, bbox_inches='tight')
plt.close()
print('[+] Saved 04_rolling_correlation_cold_spells.png')

# --- Save merged dataset ---
data_dir = OUT / 'data'
data_dir.mkdir(exist_ok=True)
merged.to_csv(data_dir / 'temp_load_price_merged.csv', index=False)
print(f'[+] Saved merged dataset: {len(merged)} rows')

# --- Print key findings ---
print()
print('=' * 60)
print('KEY FINDINGS')
print('=' * 60)
print(f'Period: {merged.datetime.min().date()} to {merged.datetime.max().date()}')
print(f'GFS forecast bias: {valid["temp_error"].mean():.2f} C (systematic cold bias)')
print(f'GFS forecast MAE:  {valid["temp_error"].abs().mean():.2f} C')
print(f'GFS forecast RMSE: {np.sqrt((valid["temp_error"]**2).mean()):.2f} C')
print()
print(f'Temperature-Load correlation: r = {r1:.3f}')
print(f'  -> {abs(slope1):.1f} MW per 1C temperature change')
print(f'Temperature-Price correlation: r = {r2:.3f}')
print(f'  -> {abs(slope2):.1f} EUR/MWh per 1C temperature change')
print(f'Temp Error-Load Error correlation: r = {r3:.3f}')
print(f'  -> {abs(slope3):.1f} MW load error per 1C temp error')
print()
cold_mask = valid['temp_actual'] < 0
warm_mask = valid['temp_actual'] >= 10
cold_load = valid.loc[cold_mask, 'actual_load_mw'].mean()
warm_load = valid.loc[warm_mask, 'actual_load_mw'].mean()
cold_price = valid.loc[cold_mask, 'price_eur_mwh'].mean()
warm_price = valid.loc[warm_mask, 'price_eur_mwh'].mean()
print(f'Cold (<0C):  mean load = {cold_load:.0f} MW, mean price = {cold_price:.1f} EUR/MWh')
print(f'Warm (>10C): mean load = {warm_load:.0f} MW, mean price = {warm_price:.1f} EUR/MWh')
print(f'Delta: {cold_load - warm_load:.0f} MW, {cold_price - warm_price:.1f} EUR/MWh')
