# -*- coding: utf-8 -*-
"""
Solar Production Decomposition Analysis
STL decomposition of: actual, forecast, and error
"""

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

print("[*] Solar Decomposition Analysis")
print("=" * 50)

# =============================================================================
# 1. Load and parse data (reuse parsing logic)
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

solar_actual_idx = columns.index('Sk.A.Solar') if 'Sk.A.Solar' in columns else None
solar_forecast_idx = columns.index('Sk.F.Solar') if 'Sk.F.Solar' in columns else None

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
    if solar_actual_idx is not None:
        ts, val = parse_row(parts, solar_actual_idx)
        if ts is not None:
            data_actual.append({'timestamp': ts, 'actual': val})
    if solar_forecast_idx is not None:
        ts, val = parse_row(parts, solar_forecast_idx)
        if ts is not None:
            data_forecast.append({'timestamp': ts, 'forecast': val})

df_actual = pd.DataFrame(data_actual)
df_forecast = pd.DataFrame(data_forecast)

print(f"[*] Loaded {len(df_actual)} actual records, {len(df_forecast)} forecast records")

# Merge
df = df_actual.merge(df_forecast, on='timestamp', how='inner')
df = df.dropna()
df = df.sort_values('timestamp').reset_index(drop=True)
df['error'] = df['forecast'] - df['actual']

print(f"[*] Merged valid records: {len(df)}")
print(f"[*] Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# =============================================================================
# 2. Prepare hourly data for decomposition
# =============================================================================
print("\n[*] Preparing data for STL decomposition...")

# Set timestamp as index
df_ts = df.set_index('timestamp')

# Resample to ensure regular hourly frequency (fill gaps if any)
df_hourly = df_ts.resample('h').mean()

# Fill small gaps with interpolation
df_hourly = df_hourly.interpolate(method='linear', limit=3)

# Drop rows with NaN (larger gaps)
df_hourly = df_hourly.dropna()

print(f"[*] Hourly records for decomposition: {len(df_hourly)}")

# For STL, we need sufficient data - use period=24 for daily seasonality
# Need at least 2 full periods, ideally more

if len(df_hourly) < 48:
    print("[!] Not enough data for decomposition (need at least 48 hours)")
else:
    os.makedirs('analysis/solar_forecast_error', exist_ok=True)

    # =============================================================================
    # 3. Decomposition of ACTUAL solar production
    # =============================================================================
    print("\n[*] Decomposing ACTUAL solar production...")

    # STL decomposition - period=24 for daily cycle
    stl_actual = STL(df_hourly['actual'], period=24, robust=True)
    result_actual = stl_actual.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('STL Decomposition: ACTUAL Solar Production', fontsize=14, fontweight='bold')

    axes[0].plot(df_hourly.index, df_hourly['actual'], 'b-', linewidth=0.5)
    axes[0].set_ylabel('Actual (MW)')
    axes[0].set_title('Original Series')

    axes[1].plot(df_hourly.index, result_actual.trend, 'g-', linewidth=1)
    axes[1].set_ylabel('Trend (MW)')
    axes[1].set_title('Trend Component')

    axes[2].plot(df_hourly.index, result_actual.seasonal, 'orange', linewidth=0.5)
    axes[2].set_ylabel('Seasonal (MW)')
    axes[2].set_title('Seasonal Component (24h period)')

    axes[3].plot(df_hourly.index, result_actual.resid, 'r-', linewidth=0.5, alpha=0.7)
    axes[3].set_ylabel('Residual (MW)')
    axes[3].set_title('Residual Component')
    axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('analysis/solar_forecast_error/02_decomposition_actual.png', dpi=150)
    plt.close()
    print("[+] Saved: analysis/solar_forecast_error/02_decomposition_actual.png")

    # Stats for actual
    print(f"\n    Actual - Trend range: {result_actual.trend.min():.1f} to {result_actual.trend.max():.1f} MW")
    print(f"    Actual - Seasonal amplitude: {result_actual.seasonal.max() - result_actual.seasonal.min():.1f} MW")
    print(f"    Actual - Residual std: {result_actual.resid.std():.2f} MW")

    # =============================================================================
    # 4. Decomposition of FORECAST solar production
    # =============================================================================
    print("\n[*] Decomposing FORECAST solar production...")

    stl_forecast = STL(df_hourly['forecast'], period=24, robust=True)
    result_forecast = stl_forecast.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('STL Decomposition: FORECAST Solar Production', fontsize=14, fontweight='bold')

    axes[0].plot(df_hourly.index, df_hourly['forecast'], 'b-', linewidth=0.5)
    axes[0].set_ylabel('Forecast (MW)')
    axes[0].set_title('Original Series')

    axes[1].plot(df_hourly.index, result_forecast.trend, 'g-', linewidth=1)
    axes[1].set_ylabel('Trend (MW)')
    axes[1].set_title('Trend Component')

    axes[2].plot(df_hourly.index, result_forecast.seasonal, 'orange', linewidth=0.5)
    axes[2].set_ylabel('Seasonal (MW)')
    axes[2].set_title('Seasonal Component (24h period)')

    axes[3].plot(df_hourly.index, result_forecast.resid, 'r-', linewidth=0.5, alpha=0.7)
    axes[3].set_ylabel('Residual (MW)')
    axes[3].set_title('Residual Component')
    axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('analysis/solar_forecast_error/03_decomposition_forecast.png', dpi=150)
    plt.close()
    print("[+] Saved: analysis/solar_forecast_error/03_decomposition_forecast.png")

    print(f"\n    Forecast - Trend range: {result_forecast.trend.min():.1f} to {result_forecast.trend.max():.1f} MW")
    print(f"    Forecast - Seasonal amplitude: {result_forecast.seasonal.max() - result_forecast.seasonal.min():.1f} MW")
    print(f"    Forecast - Residual std: {result_forecast.resid.std():.2f} MW")

    # =============================================================================
    # 5. Decomposition of FORECAST ERROR
    # =============================================================================
    print("\n[*] Decomposing FORECAST ERROR...")

    stl_error = STL(df_hourly['error'], period=24, robust=True)
    result_error = stl_error.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('STL Decomposition: FORECAST ERROR (Forecast - Actual)', fontsize=14, fontweight='bold')

    axes[0].plot(df_hourly.index, df_hourly['error'], 'b-', linewidth=0.5)
    axes[0].set_ylabel('Error (MW)')
    axes[0].set_title('Original Error Series')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    axes[1].plot(df_hourly.index, result_error.trend, 'g-', linewidth=1)
    axes[1].set_ylabel('Trend (MW)')
    axes[1].set_title('Trend Component (systematic bias over time)')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    axes[2].plot(df_hourly.index, result_error.seasonal, 'orange', linewidth=0.5)
    axes[2].set_ylabel('Seasonal (MW)')
    axes[2].set_title('Seasonal Component (daily error pattern)')
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    axes[3].plot(df_hourly.index, result_error.resid, 'r-', linewidth=0.5, alpha=0.7)
    axes[3].set_ylabel('Residual (MW)')
    axes[3].set_title('Residual Component (random error)')
    axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('analysis/solar_forecast_error/04_decomposition_error.png', dpi=150)
    plt.close()
    print("[+] Saved: analysis/solar_forecast_error/04_decomposition_error.png")

    print(f"\n    Error - Trend range: {result_error.trend.min():.1f} to {result_error.trend.max():.1f} MW")
    print(f"    Error - Seasonal amplitude: {result_error.seasonal.max() - result_error.seasonal.min():.1f} MW")
    print(f"    Error - Residual std: {result_error.resid.std():.2f} MW")

    # =============================================================================
    # 6. Comparison plot: seasonal patterns
    # =============================================================================
    print("\n[*] Creating comparison plots...")

    # Extract one typical day of seasonal pattern
    seasonal_actual_day = result_actual.seasonal[:24].values
    seasonal_forecast_day = result_forecast.seasonal[:24].values
    seasonal_error_day = result_error.seasonal[:24].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Daily seasonal patterns comparison
    ax1 = axes[0, 0]
    hours = range(24)
    ax1.plot(hours, seasonal_actual_day, 'b-', linewidth=2, label='Actual', marker='o', markersize=4)
    ax1.plot(hours, seasonal_forecast_day, 'r--', linewidth=2, label='Forecast', marker='s', markersize=4)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Seasonal Component (MW)')
    ax1.set_title('Daily Seasonal Pattern: Actual vs Forecast')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))

    # Plot 2: Error seasonal pattern
    ax2 = axes[0, 1]
    colors = ['green' if v >= 0 else 'red' for v in seasonal_error_day]
    ax2.bar(hours, seasonal_error_day, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Error Seasonal (MW)')
    ax2.set_title('Daily Error Pattern (green=overforecast, red=underforecast)')
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, alpha=0.3)

    # Plot 3: Trend comparison
    ax3 = axes[1, 0]
    ax3.plot(df_hourly.index, result_actual.trend, 'b-', linewidth=1, label='Actual Trend', alpha=0.8)
    ax3.plot(df_hourly.index, result_forecast.trend, 'r--', linewidth=1, label='Forecast Trend', alpha=0.8)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Trend (MW)')
    ax3.set_title('Trend Component: Actual vs Forecast')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Error trend
    ax4 = axes[1, 1]
    ax4.plot(df_hourly.index, result_error.trend, 'purple', linewidth=1)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.fill_between(df_hourly.index, result_error.trend, 0,
                     where=result_error.trend >= 0, color='green', alpha=0.3, label='Overforecast')
    ax4.fill_between(df_hourly.index, result_error.trend, 0,
                     where=result_error.trend < 0, color='red', alpha=0.3, label='Underforecast')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Error Trend (MW)')
    ax4.set_title('Error Trend Over Time (Systematic Bias)')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('analysis/solar_forecast_error/05_decomposition_comparison.png', dpi=150)
    plt.close()
    print("[+] Saved: analysis/solar_forecast_error/05_decomposition_comparison.png")

    # =============================================================================
    # 7. Summary statistics
    # =============================================================================
    print("\n" + "=" * 60)
    print("DECOMPOSITION SUMMARY")
    print("=" * 60)

    print("\n[+] ACTUAL Solar Production:")
    print(f"    Trend range:        {result_actual.trend.min():.1f} to {result_actual.trend.max():.1f} MW")
    print(f"    Seasonal amplitude: {result_actual.seasonal.max() - result_actual.seasonal.min():.1f} MW")
    print(f"    Residual std:       {result_actual.resid.std():.2f} MW")

    print("\n[+] FORECAST Solar Production:")
    print(f"    Trend range:        {result_forecast.trend.min():.1f} to {result_forecast.trend.max():.1f} MW")
    print(f"    Seasonal amplitude: {result_forecast.seasonal.max() - result_forecast.seasonal.min():.1f} MW")
    print(f"    Residual std:       {result_forecast.resid.std():.2f} MW")

    print("\n[+] FORECAST ERROR:")
    print(f"    Trend range:        {result_error.trend.min():.1f} to {result_error.trend.max():.1f} MW")
    print(f"    Seasonal amplitude: {result_error.seasonal.max() - result_error.seasonal.min():.1f} MW")
    print(f"    Residual std:       {result_error.resid.std():.2f} MW")

    # Variance decomposition
    var_actual = df_hourly['actual'].var()
    var_trend_a = result_actual.trend.var()
    var_seasonal_a = result_actual.seasonal.var()
    var_resid_a = result_actual.resid.var()

    var_forecast = df_hourly['forecast'].var()
    var_trend_f = result_forecast.trend.var()
    var_seasonal_f = result_forecast.seasonal.var()
    var_resid_f = result_forecast.resid.var()

    var_error = df_hourly['error'].var()
    var_trend_e = result_error.trend.var()
    var_seasonal_e = result_error.seasonal.var()
    var_resid_e = result_error.resid.var()

    print("\n[+] Variance Decomposition (% of total):")
    print(f"    ACTUAL:   Trend {100*var_trend_a/var_actual:.1f}%, Seasonal {100*var_seasonal_a/var_actual:.1f}%, Residual {100*var_resid_a/var_actual:.1f}%")
    print(f"    FORECAST: Trend {100*var_trend_f/var_forecast:.1f}%, Seasonal {100*var_seasonal_f/var_forecast:.1f}%, Residual {100*var_resid_f/var_forecast:.1f}%")
    print(f"    ERROR:    Trend {100*var_trend_e/var_error:.1f}%, Seasonal {100*var_seasonal_e/var_error:.1f}%, Residual {100*var_resid_e/var_error:.1f}%")

print("\n[+] Decomposition analysis complete!")
