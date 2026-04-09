# -*- coding: utf-8 -*-
"""
Solar Decomposition Overlay - Actual vs Forecast side by side
"""

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

print("[*] Solar Decomposition Overlay Analysis")
print("=" * 50)

# =============================================================================
# 1. Load and parse data
# =============================================================================
print("\n[*] Loading production data...")

with open('RawData/ProductionPerType.csv', 'r', encoding='utf-8-sig') as f:
    header_line = f.readline().strip()
    data_lines = f.readlines()

header_parts = header_line.split(',')
columns = []
for i in range(0, len(header_parts), 2):
    if header_parts[i].strip():
        columns.append(header_parts[i].strip())

solar_actual_idx = columns.index('Sk.A.Solar')
solar_forecast_idx = columns.index('Sk.F.Solar')

ts_pattern = re.compile(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}')

def parse_row(parts, col_idx):
    current_col = 0
    i = 0
    while i < len(parts) and current_col <= col_idx:
        if ts_pattern.match(parts[i]):
            if current_col == col_idx:
                ts_str = parts[i]
                try:
                    ts = pd.to_datetime(ts_str, format='%d/%m/%Y %H:%M:%S')
                except:
                    return None, np.nan
                i += 1
                if i >= len(parts):
                    return ts, np.nan
                i += 1
                if i >= len(parts):
                    return ts, np.nan
                val_or_invalid = parts[i]
                if val_or_invalid == '(invalid)':
                    return ts, np.nan
                else:
                    int_part = val_or_invalid
                    i += 1
                    if i < len(parts) and not ts_pattern.match(parts[i]) and parts[i] != '(invalid)':
                        dec_part = parts[i]
                        try:
                            value = float(int_part) + float(dec_part) / (10 ** len(dec_part))
                        except:
                            value = float(int_part) if int_part else np.nan
                    else:
                        try:
                            value = float(int_part)
                        except:
                            value = np.nan
                    return ts, value
            current_col += 1
        i += 1
    return None, np.nan

data_actual = []
data_forecast = []

for line in data_lines:
    parts = line.strip().split(',')
    ts_a, val_a = parse_row(parts, solar_actual_idx)
    ts_f, val_f = parse_row(parts, solar_forecast_idx)
    if ts_a is not None:
        data_actual.append({'timestamp': ts_a, 'actual': val_a})
    if ts_f is not None:
        data_forecast.append({'timestamp': ts_f, 'forecast': val_f})

df_actual = pd.DataFrame(data_actual)
df_forecast = pd.DataFrame(data_forecast)

df = df_actual.merge(df_forecast, on='timestamp', how='inner')
df = df.dropna().sort_values('timestamp').reset_index(drop=True)
df['error'] = df['forecast'] - df['actual']

print(f"[*] Valid records: {len(df)}")
print(f"[*] Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# =============================================================================
# 2. Prepare hourly data
# =============================================================================
df_ts = df.set_index('timestamp')
df_hourly = df_ts.resample('h').mean()
df_hourly = df_hourly.interpolate(method='linear', limit=3).dropna()

print(f"[*] Hourly records: {len(df_hourly)}")

# =============================================================================
# 3. STL Decomposition
# =============================================================================
print("\n[*] Running STL decomposition...")

stl_actual = STL(df_hourly['actual'], period=24, robust=True).fit()
stl_forecast = STL(df_hourly['forecast'], period=24, robust=True).fit()
stl_error = STL(df_hourly['error'], period=24, robust=True).fit()

os.makedirs('analysis/solar_forecast_error', exist_ok=True)

# =============================================================================
# 4. OVERLAY PLOT - All components
# =============================================================================
print("\n[*] Creating overlay plots...")

fig, axes = plt.subplots(4, 1, figsize=(16, 14))
fig.suptitle('STL Decomposition Overlay: Actual vs Forecast Solar Production',
             fontsize=14, fontweight='bold')

# Original series
ax1 = axes[0]
ax1.plot(df_hourly.index, df_hourly['actual'], 'b-', linewidth=0.6, label='Actual', alpha=0.8)
ax1.plot(df_hourly.index, df_hourly['forecast'], 'r-', linewidth=0.6, label='Forecast', alpha=0.8)
ax1.set_ylabel('Power (MW)')
ax1.set_title('Original Series')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Trend
ax2 = axes[1]
ax2.plot(df_hourly.index, stl_actual.trend, 'b-', linewidth=1.5, label='Actual Trend')
ax2.plot(df_hourly.index, stl_forecast.trend, 'r--', linewidth=1.5, label='Forecast Trend')
ax2.set_ylabel('Trend (MW)')
ax2.set_title('Trend Component')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Seasonal
ax3 = axes[2]
ax3.plot(df_hourly.index, stl_actual.seasonal, 'b-', linewidth=0.6, label='Actual Seasonal', alpha=0.8)
ax3.plot(df_hourly.index, stl_forecast.seasonal, 'r-', linewidth=0.6, label='Forecast Seasonal', alpha=0.8)
ax3.set_ylabel('Seasonal (MW)')
ax3.set_title('Seasonal Component (24h period)')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Residual
ax4 = axes[3]
ax4.plot(df_hourly.index, stl_actual.resid, 'b-', linewidth=0.5, label='Actual Residual', alpha=0.7)
ax4.plot(df_hourly.index, stl_forecast.resid, 'r-', linewidth=0.5, label='Forecast Residual', alpha=0.7)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax4.set_ylabel('Residual (MW)')
ax4.set_title('Residual Component')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

for ax in axes:
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('analysis/solar_forecast_error/06_decomposition_overlay.png', dpi=150)
plt.close()
print("[+] Saved: analysis/solar_forecast_error/06_decomposition_overlay.png")

# =============================================================================
# 5. Daily Seasonal Pattern Comparison (single day)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Get one day of seasonal pattern
seasonal_actual_day = stl_actual.seasonal[:24].values
seasonal_forecast_day = stl_forecast.seasonal[:24].values
seasonal_error_day = stl_error.seasonal[:24].values
hours = range(24)

# Plot 1: Seasonal overlay
ax1 = axes[0, 0]
ax1.plot(hours, seasonal_actual_day, 'b-', linewidth=2, label='Actual', marker='o', markersize=5)
ax1.plot(hours, seasonal_forecast_day, 'r--', linewidth=2, label='Forecast', marker='s', markersize=5)
ax1.fill_between(hours, seasonal_actual_day, seasonal_forecast_day, alpha=0.3, color='gray')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Seasonal Component (MW)')
ax1.set_title('Daily Seasonal Pattern: Actual vs Forecast')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, 24, 2))

# Plot 2: Seasonal difference
ax2 = axes[0, 1]
seasonal_diff = seasonal_forecast_day - seasonal_actual_day
colors = ['green' if v >= 0 else 'red' for v in seasonal_diff]
ax2.bar(hours, seasonal_diff, color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Forecast - Actual (MW)')
ax2.set_title('Seasonal Difference (green=overforecast, red=underforecast)')
ax2.set_xticks(range(0, 24, 2))
ax2.grid(True, alpha=0.3)

# Plot 3: Trend difference over time
ax3 = axes[1, 0]
trend_diff = stl_forecast.trend - stl_actual.trend
ax3.plot(df_hourly.index, trend_diff, 'purple', linewidth=1)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax3.fill_between(df_hourly.index, trend_diff, 0,
                 where=trend_diff >= 0, color='green', alpha=0.3, label='Forecast > Actual')
ax3.fill_between(df_hourly.index, trend_diff, 0,
                 where=trend_diff < 0, color='red', alpha=0.3, label='Forecast < Actual')
ax3.set_xlabel('Date')
ax3.set_ylabel('Trend Difference (MW)')
ax3.set_title('Trend Difference: Forecast - Actual')
ax3.legend()
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)

# Plot 4: Residual correlation
ax4 = axes[1, 1]
ax4.scatter(stl_actual.resid, stl_forecast.resid, alpha=0.3, s=10, c='purple')
max_res = max(abs(stl_actual.resid).max(), abs(stl_forecast.resid).max())
ax4.plot([-max_res, max_res], [-max_res, max_res], 'r--', label='1:1 line')
ax4.set_xlabel('Actual Residual (MW)')
ax4.set_ylabel('Forecast Residual (MW)')
ax4.set_title('Residual Correlation')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-max_res*1.1, max_res*1.1)
ax4.set_ylim(-max_res*1.1, max_res*1.1)

# Calculate correlation
corr = np.corrcoef(stl_actual.resid, stl_forecast.resid)[0, 1]
ax4.annotate(f'Correlation: {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=11, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('analysis/solar_forecast_error/07_decomposition_differences.png', dpi=150)
plt.close()
print("[+] Saved: analysis/solar_forecast_error/07_decomposition_differences.png")

# =============================================================================
# 6. Summary Statistics
# =============================================================================
print("\n" + "=" * 60)
print("DECOMPOSITION COMPARISON SUMMARY")
print("=" * 60)

print("\n[+] Seasonal Amplitude:")
print(f"    Actual:   {stl_actual.seasonal.max() - stl_actual.seasonal.min():.1f} MW")
print(f"    Forecast: {stl_forecast.seasonal.max() - stl_forecast.seasonal.min():.1f} MW")
print(f"    Difference: {(stl_forecast.seasonal.max() - stl_forecast.seasonal.min()) - (stl_actual.seasonal.max() - stl_actual.seasonal.min()):.1f} MW")

print("\n[+] Trend Range:")
print(f"    Actual:   {stl_actual.trend.min():.1f} to {stl_actual.trend.max():.1f} MW")
print(f"    Forecast: {stl_forecast.trend.min():.1f} to {stl_forecast.trend.max():.1f} MW")

print("\n[+] Residual Std Dev:")
print(f"    Actual:   {stl_actual.resid.std():.2f} MW")
print(f"    Forecast: {stl_forecast.resid.std():.2f} MW")

print(f"\n[+] Residual Correlation: {corr:.3f}")

print("\n[+] Peak Hour Comparison (seasonal component):")
actual_peak_hour = np.argmax(seasonal_actual_day)
forecast_peak_hour = np.argmax(seasonal_forecast_day)
print(f"    Actual peak hour:   {actual_peak_hour}:00 ({seasonal_actual_day[actual_peak_hour]:.1f} MW)")
print(f"    Forecast peak hour: {forecast_peak_hour}:00 ({seasonal_forecast_day[forecast_peak_hour]:.1f} MW)")

print("\n[+] Analysis complete!")
