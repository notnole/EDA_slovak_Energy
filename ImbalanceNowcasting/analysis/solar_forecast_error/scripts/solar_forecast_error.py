# -*- coding: utf-8 -*-
"""
Solar Forecast Error Analysis
Compares TSO solar prediction with actual solar production
"""

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt

print("[*] Solar Forecast Error Analysis")
print("=" * 50)

# =============================================================================
# 1. Parse ProductionPerType.csv with variable-width columns
# =============================================================================
# Structure is tricky: invalid values have 3 parts (ts, ms, (invalid))
#                      valid values have 4 parts (ts, ms, int, dec)
# Need to parse by finding timestamp patterns

print("\n[*] Loading production data...")

with open('RawData/ProductionPerType.csv', 'r', encoding='utf-8-sig') as f:
    header_line = f.readline().strip()
    data_lines = f.readlines()

# Parse header - column names at even positions
header_parts = header_line.split(',')
columns = []
for i in range(0, len(header_parts), 2):
    if header_parts[i].strip():
        columns.append(header_parts[i].strip())

print(f"[*] Found {len(columns)} columns: {columns}")

# Find Solar column indices
solar_actual_idx = columns.index('Sk.A.Solar') if 'Sk.A.Solar' in columns else None
solar_forecast_idx = columns.index('Sk.F.Solar') if 'Sk.F.Solar' in columns else None

print(f"[*] Sk.A.Solar is column index {solar_actual_idx}")
print(f"[*] Sk.F.Solar is column index {solar_forecast_idx}")

# Timestamp regex pattern
ts_pattern = re.compile(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}')

def parse_row(parts, col_idx):
    """Parse row to extract value for specific column index"""
    # Walk through parts to find the col_idx-th timestamp
    current_col = 0
    i = 0

    while i < len(parts) and current_col <= col_idx:
        # Check if this looks like a timestamp
        if ts_pattern.match(parts[i]):
            if current_col == col_idx:
                # Found our column
                ts_str = parts[i]
                try:
                    ts = pd.to_datetime(ts_str, format='%d/%m/%Y %H:%M:%S')
                except:
                    return None, np.nan

                # Next part should be ms (000)
                i += 1
                if i >= len(parts):
                    return ts, np.nan

                # Next is value or (invalid)
                i += 1
                if i >= len(parts):
                    return ts, np.nan

                val_or_invalid = parts[i]

                if val_or_invalid == '(invalid)':
                    return ts, np.nan
                else:
                    # Integer part
                    int_part = val_or_invalid
                    # Check for decimal part
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

# Parse all data
data_actual = []
data_forecast = []

for line in data_lines:
    parts = line.strip().split(',')

    if solar_actual_idx is not None:
        ts, val = parse_row(parts, solar_actual_idx)
        if ts is not None:
            data_actual.append({'timestamp': ts, 'solar_actual_mw': val})

    if solar_forecast_idx is not None:
        ts, val = parse_row(parts, solar_forecast_idx)
        if ts is not None:
            data_forecast.append({'timestamp': ts, 'solar_forecast_mw': val})

df_actual = pd.DataFrame(data_actual)
df_forecast = pd.DataFrame(data_forecast) if data_forecast else pd.DataFrame()

print(f"\n[*] Parsed {len(df_actual)} actual solar records")
print(f"[*] Parsed {len(df_forecast)} forecast solar records")

if len(df_actual) > 0:
    print(f"[*] Actual date range: {df_actual['timestamp'].min()} to {df_actual['timestamp'].max()}")
    non_zero = df_actual[df_actual['solar_actual_mw'] > 0]['solar_actual_mw'].head(10).tolist()
    print(f"[*] Sample actual values (first 10 non-zero): {non_zero}")

if len(df_forecast) > 0:
    print(f"[*] Forecast date range: {df_forecast['timestamp'].min()} to {df_forecast['timestamp'].max()}")
    non_zero = df_forecast[df_forecast['solar_forecast_mw'] > 0]['solar_forecast_mw'].head(10).tolist()
    print(f"[*] Sample forecast values (first 10 non-zero): {non_zero}")

# =============================================================================
# 2. Merge and calculate errors
# =============================================================================
print("\n[*] Merging actual and forecast data...")

if len(df_forecast) == 0:
    print("[!] No forecast data found in ProductionPerType.csv")
    print("[*] The Sk.F.Solar column may contain only (invalid) values")
    print("[*] Using only actual solar data for summary")

    # Just show actual solar statistics
    df_valid = df_actual[df_actual['solar_actual_mw'] > 0].copy()
    print(f"\n[*] Actual solar daylight periods: {len(df_valid)}")

    if len(df_valid) > 0:
        print("\n" + "=" * 50)
        print("ACTUAL SOLAR PRODUCTION STATISTICS")
        print("=" * 50)
        print(f"\nMean solar (daylight): {df_valid['solar_actual_mw'].mean():.1f} MW")
        print(f"Max solar: {df_valid['solar_actual_mw'].max():.1f} MW")
        print(f"Min solar (daylight): {df_valid['solar_actual_mw'].min():.1f} MW")

        # Monthly stats
        df_valid['month'] = df_valid['timestamp'].dt.month
        monthly = df_valid.groupby('month')['solar_actual_mw'].agg(['mean', 'max', 'count']).round(1)
        print(f"\n[+] Monthly actual solar (MW):")
        print(monthly.to_string())

else:
    df_merged = df_actual.merge(df_forecast, on='timestamp', how='inner')
    print(f"[*] Merged dataset: {len(df_merged)} records")

    df_merged = df_merged.dropna()
    print(f"[*] After removing NaN: {len(df_merged)} records")

    df_merged['error_mw'] = df_merged['solar_forecast_mw'] - df_merged['solar_actual_mw']
    df_merged['abs_error_mw'] = df_merged['error_mw'].abs()

    df_daylight = df_merged[df_merged['solar_actual_mw'] > 0].copy()
    df_daylight['pct_error'] = 100 * df_daylight['error_mw'] / df_daylight['solar_actual_mw']

    print(f"[*] Daylight periods (actual > 0): {len(df_daylight)} records")

    # Error Statistics
    print("\n" + "=" * 50)
    print("SOLAR FORECAST ERROR STATISTICS")
    print("=" * 50)

    if len(df_daylight) > 0:
        mae = df_daylight['abs_error_mw'].mean()
        rmse = np.sqrt((df_daylight['error_mw'] ** 2).mean())
        mbe = df_daylight['error_mw'].mean()
        mape = df_daylight['pct_error'].abs().mean()

        print(f"\n[+] All daylight periods (solar > 0 MW):")
        print(f"    MAE  (Mean Absolute Error): {mae:.2f} MW")
        print(f"    RMSE (Root Mean Square Error): {rmse:.2f} MW")
        print(f"    MBE  (Mean Bias Error): {mbe:.2f} MW")
        print(f"    MAPE (Mean Abs % Error): {mape:.2f}%")

        # Hourly stats
        df_daylight['hour'] = df_daylight['timestamp'].dt.hour
        hourly_stats = df_daylight.groupby('hour').agg({
            'abs_error_mw': 'mean',
            'error_mw': 'mean',
            'solar_actual_mw': 'mean',
            'solar_forecast_mw': 'mean'
        }).round(2)

        print(f"\n[+] Error by hour of day:")
        print(hourly_stats.to_string())

        # Monthly stats
        df_daylight['month'] = df_daylight['timestamp'].dt.month
        monthly_stats = df_daylight.groupby('month').agg({
            'abs_error_mw': 'mean',
            'error_mw': 'mean',
            'solar_actual_mw': 'mean',
            'solar_forecast_mw': 'mean'
        }).round(2)

        print(f"\n[+] Error by month:")
        print(monthly_stats.to_string())

        # =============================================================================
        # Visualization
        # =============================================================================
        print("\n[*] Creating visualizations...")

        os.makedirs('analysis/solar_forecast_error', exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Time series sample
        ax1 = axes[0, 0]
        summer = df_daylight[(df_daylight['timestamp'].dt.month >= 5) &
                              (df_daylight['timestamp'].dt.month <= 8)]
        sample = summer.head(24*14) if len(summer) > 0 else df_daylight.head(24*14)

        ax1.plot(sample['timestamp'], sample['solar_actual_mw'], label='Actual', alpha=0.8, lw=0.8)
        ax1.plot(sample['timestamp'], sample['solar_forecast_mw'], label='Forecast', alpha=0.8, lw=0.8)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Solar Power (MW)')
        ax1.set_title('Actual vs Forecast Solar (sample)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Error distribution
        ax2 = axes[0, 1]
        ax2.hist(df_daylight['error_mw'], bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', label='Zero')
        ax2.axvline(x=mbe, color='g', linestyle='--', label=f'Bias: {mbe:.1f}')
        ax2.set_xlabel('Forecast Error (MW)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.legend()

        # Plot 3: MAE by hour
        ax3 = axes[1, 0]
        ax3.bar(hourly_stats.index, hourly_stats['abs_error_mw'], color='steelblue', alpha=0.7)
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('MAE (MW)')
        ax3.set_title('MAE by Hour')

        # Plot 4: Scatter
        ax4 = axes[1, 1]
        ax4.scatter(df_daylight['solar_actual_mw'], df_daylight['solar_forecast_mw'], alpha=0.3, s=5)
        max_val = max(df_daylight['solar_actual_mw'].max(), df_daylight['solar_forecast_mw'].max())
        ax4.plot([0, max_val], [0, max_val], 'r--', label='Perfect')
        ax4.set_xlabel('Actual (MW)')
        ax4.set_ylabel('Forecast (MW)')
        ax4.set_title('Actual vs Forecast')
        ax4.legend()

        plt.tight_layout()
        plt.savefig('analysis/solar_forecast_error/01_error_analysis.png', dpi=150)
        plt.close()
        print("[+] Saved: analysis/solar_forecast_error/01_error_analysis.png")

        # Summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Data period: {df_daylight['timestamp'].min().date()} to {df_daylight['timestamp'].max().date()}")
        print(f"Daylight observations: {len(df_daylight)}")
        print(f"\nError Metrics:")
        print(f"  MAE:  {mae:.2f} MW")
        print(f"  RMSE: {rmse:.2f} MW")
        print(f"  MBE:  {mbe:.2f} MW")
        print(f"  MAPE: {mape:.2f}%")

print("\n[+] Analysis complete!")
