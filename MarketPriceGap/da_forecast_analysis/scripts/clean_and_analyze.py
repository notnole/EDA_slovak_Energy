"""
DA Price Forecast Analysis
- Clean Ipesoft EDA export (5 series, European decimal format)
- Merge with actual OKTE DA prices
- Forecast error decomposition (STL)
- Residual correlation: forecast residual vs actual price residual

Series in raw file:
  1. DE Forecast1 Spot 15    (DE spot forecast)
  2. Forecast M SK spot 15min (ALWAYS INVALID - dropped)
  3. Sk Forecast1 Spot 15     (SK forecast model 1)
  4. SK Forecast2 Spot 15     (SK forecast model 2)
  5. SK.F.M.Merged.Spot       (SK merged forecast, only from 18/01/2026)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent.parent.parent
RAW_PATH = BASE / 'RawData' / 'DA_Price_Forcasts.csv'
CLEAN_DIR = BASE / 'data' / 'clean' / 'market' / 'da_forecasts'
OUT_DIR = BASE / 'MarketPriceGap' / 'da_forecast_analysis'

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / 'data').mkdir(exist_ok=True)

SERIES_NAMES = [
    'de_forecast1_spot',
    'forecast_m_sk',       # always invalid, will be dropped
    'sk_forecast1_spot',
    'sk_forecast2_spot',
    'sk_merged_spot',
]

# ================================================================
# STEP 1: Parse the EDA export
# ================================================================
print('[*] Parsing DA price forecast CSV...')

rows = []
with open(RAW_PATH, 'r', encoding='utf-8-sig') as f:
    header = f.readline()
    for line_num, line in enumerate(f, 2):
        line = line.strip()
        if not line:
            continue

        fields = line.split(',')
        pos = 0
        dt = None
        values = {}

        for s_idx, s_name in enumerate(SERIES_NAMES):
            if pos >= len(fields):
                values[s_name] = np.nan
                continue

            field_val = fields[pos].strip() if pos < len(fields) else ''

            # Check for valid datetime pattern
            if not re.match(r'\d{2}/\d{2}/\d{4}', field_val):
                # Empty field - skip
                values[s_name] = np.nan
                while pos < len(fields) and not re.match(r'\d{2}/\d{2}/\d{4}', fields[pos].strip()):
                    pos += 1
                continue

            # Parse datetime (take from first series only)
            if dt is None:
                try:
                    dt = pd.to_datetime(field_val, format='%d/%m/%Y %H:%M:%S')
                except Exception:
                    pass
            pos += 1  # skip ms field ('000')
            pos += 1  # now at value field

            if pos >= len(fields):
                values[s_name] = np.nan
                continue

            val_field = fields[pos].strip()
            if val_field == '(invalid)':
                values[s_name] = np.nan
                pos += 1
            else:
                try:
                    dec_part = fields[pos + 1].strip() if (pos + 1) < len(fields) else '0'
                    # Check if dec_part looks like a date (next series start)
                    if re.match(r'\d{2}/\d{2}/\d{4}', dec_part):
                        values[s_name] = float(val_field)
                        pos += 1
                    else:
                        values[s_name] = float(val_field + '.' + dec_part)
                        pos += 2
                except (ValueError, IndexError):
                    values[s_name] = np.nan
                    pos += 1

        if dt is not None:
            row = {'datetime': dt}
            row.update(values)
            rows.append(row)

df = pd.DataFrame(rows)

# Drop the always-invalid column
df.drop(columns=['forecast_m_sk'], inplace=True, errors='ignore')

print(f'[+] Parsed {len(df)} rows')
print(f'    Date range: {df.datetime.min()} to {df.datetime.max()}')
for col in [c for c in df.columns if c != 'datetime']:
    valid = df[col].notna().sum()
    print(f'    {col}: {valid}/{len(df)} valid ({100*valid/len(df):.1f}%)')

# Save clean data
clean_path = CLEAN_DIR / 'da_price_forecasts_15min.csv'
df.to_csv(clean_path, index=False)
print(f'[+] Saved clean data: {clean_path}')

# ================================================================
# STEP 2: Load actual DA prices and merge
# ================================================================
print('\n[*] Loading actual DA prices...')

# Use the processed hourly file which correctly handles mixed resolutions
# (2024-2025 hourly + 2026 15-min aggregated to hourly)
PROCESSED_PATH = BASE / 'MarketPriceGap' / 'data' / 'processed' / 'hourly_market_prices.csv'
da_hourly = pd.read_csv(PROCESSED_PATH, parse_dates=['timestamp_hour'])
actual_prices = da_hourly[['timestamp_hour', 'da_price']].rename(
    columns={'timestamp_hour': 'datetime', 'da_price': 'price_actual'}
).copy()
actual_prices = actual_prices.dropna(subset=['price_actual'])
actual_prices = actual_prices.sort_values('datetime').drop_duplicates('datetime')

print(f'[*] Actual prices: {len(actual_prices)} rows, {actual_prices.datetime.min()} to {actual_prices.datetime.max()}')

# Aggregate forecasts to match actual price resolution
# Actual might be hourly (periods 1-24) or 15-min (periods 1-96)
# Forecasts are 15-min. Aggregate forecasts to hourly mean if actuals are hourly.
forecast_cols = ['de_forecast1_spot', 'sk_forecast1_spot', 'sk_forecast2_spot', 'sk_merged_spot']

# Check if actuals are hourly
time_diffs = actual_prices['datetime'].diff().dropna()
median_diff = time_diffs.median()
print(f'[*] Actual price resolution: {median_diff}')

if median_diff >= pd.Timedelta(minutes=30):
    # Actuals are hourly - aggregate forecasts to hourly
    print('[*] Aggregating forecasts to hourly to match actual prices...')
    df['hour'] = df['datetime'].dt.floor('h')
    df_hourly = df.groupby('hour')[forecast_cols].mean().reset_index()
    df_hourly.rename(columns={'hour': 'datetime'}, inplace=True)
    merged = df_hourly.merge(actual_prices, on='datetime', how='inner')
else:
    # Both 15-min
    merged = df.merge(actual_prices, on='datetime', how='inner')

# Compute errors
for col in forecast_cols:
    merged[f'{col}_error'] = merged[col] - merged['price_actual']

merged['date'] = merged['datetime'].dt.date
merged['hour'] = merged['datetime'].dt.hour

print(f'[+] Merged dataset: {len(merged)} rows')
print(f'    Period: {merged.datetime.min().date()} to {merged.datetime.max().date()}')

# Save merged
merged.to_csv(OUT_DIR / 'data' / 'forecast_vs_actual_merged.csv', index=False)

# ================================================================
# STEP 3: Forecast error statistics
# ================================================================
print('\n[*] Forecast error statistics:')
print('-' * 70)
print(f'{"Model":<25} {"Bias":>8} {"MAE":>8} {"RMSE":>8} {"r":>8} {"N":>6}')
print('-' * 70)

error_stats = {}
for col in forecast_cols:
    err_col = f'{col}_error'
    valid = merged.dropna(subset=[col, 'price_actual'])
    if len(valid) < 10:
        continue
    err = valid[err_col]
    bias = err.mean()
    mae = err.abs().mean()
    rmse = np.sqrt((err**2).mean())
    r, p = stats.pearsonr(valid[col], valid['price_actual'])
    error_stats[col] = {'bias': bias, 'mae': mae, 'rmse': rmse, 'r': r, 'n': len(valid)}
    print(f'{col:<25} {bias:>8.2f} {mae:>8.2f} {rmse:>8.2f} {r:>8.3f} {len(valid):>6}')
print('-' * 70)

# ================================================================
# PLOT 1: Time series of forecasts vs actual
# ================================================================
print('\n[*] Generating plots...')

# Use SK Forecast1 as the main model (longest coverage)
main_model = 'sk_forecast1_spot'

daily = merged.groupby('date').agg(
    price_actual=('price_actual', 'mean'),
    **{col: (col, 'mean') for col in forecast_cols}
).reset_index()
daily['date'] = pd.to_datetime(daily['date'])

fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)

ax = axes[0]
ax.plot(daily['date'], daily['price_actual'], color='black', linewidth=2, label='Actual DA Price', zorder=5)
colors = {'de_forecast1_spot': '#3498db', 'sk_forecast1_spot': '#e74c3c',
          'sk_forecast2_spot': '#2ecc71', 'sk_merged_spot': '#9b59b6'}
labels = {'de_forecast1_spot': 'DE Forecast1', 'sk_forecast1_spot': 'SK Forecast1',
          'sk_forecast2_spot': 'SK Forecast2', 'sk_merged_spot': 'SK Merged'}
for col in forecast_cols:
    d = daily.dropna(subset=[col])
    if len(d) > 5:
        ax.plot(d['date'], d[col], color=colors[col], linewidth=1, alpha=0.8, label=labels[col])
ax.set_ylabel('Price (EUR/MWh)')
ax.set_title('DA Price: Forecasts vs Actual (Daily Average)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Forecast errors
ax = axes[1]
for col in forecast_cols:
    err_col = f'{col}_error'
    d = daily.copy()
    d[err_col] = d[col] - d['price_actual']
    d = d.dropna(subset=[err_col])
    if len(d) > 5:
        ax.plot(d['date'], d[err_col], color=colors[col], linewidth=1, alpha=0.8, label=labels[col])
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Forecast Error (EUR/MWh)')
ax.set_title('DA Price Forecast Error (Forecast - Actual)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Absolute error comparison
ax = axes[2]
for col in forecast_cols:
    err_col = f'{col}_error'
    d = daily.copy()
    d[err_col] = (d[col] - d['price_actual']).abs()
    d = d.dropna(subset=[err_col])
    if len(d) > 5:
        ax.plot(d['date'], d[err_col].rolling(7).mean(), color=colors[col], linewidth=1.5,
                alpha=0.8, label=f'{labels[col]} (7d rolling MAE)')
ax.set_ylabel('Absolute Error (EUR/MWh)')
ax.set_title('Forecast Absolute Error (7-day Rolling)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / '01_forecast_vs_actual_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print('[+] Saved 01_forecast_vs_actual_timeseries.png')

# ================================================================
# PLOT 2: Scatter and error distribution
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

active_models = [c for c in forecast_cols if c in error_stats]

for idx, col in enumerate(active_models[:3]):
    valid = merged.dropna(subset=[col, 'price_actual'])
    if len(valid) < 10:
        continue
    s = error_stats[col]

    # Scatter
    ax = axes[0, idx]
    ax.scatter(valid['price_actual'], valid[col], alpha=0.15, s=5, color=colors[col], edgecolors='none')
    lims = [min(valid['price_actual'].min(), valid[col].min()),
            max(valid['price_actual'].max(), valid[col].max())]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='Perfect forecast')
    ax.set_xlabel('Actual Price (EUR/MWh)')
    ax.set_ylabel('Forecast (EUR/MWh)')
    ax.set_title(f'{labels[col]} (r={s["r"]:.3f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Error histogram
    ax = axes[1, idx]
    err = valid[f'{col}_error']
    ax.hist(err, bins=60, color=colors[col], alpha=0.7, edgecolor='white')
    ax.axvline(s['bias'], color='red', linewidth=2, linestyle='--',
               label=f'Bias: {s["bias"]:.1f}')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Forecast Error (EUR/MWh)')
    ax.set_ylabel('Count')
    ax.set_title(f'{labels[col]} Error (MAE={s["mae"]:.1f}, RMSE={s["rmse"]:.1f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / '02_scatter_and_error_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('[+] Saved 02_scatter_and_error_distribution.png')

# ================================================================
# PLOT 3: STL Decomposition of forecast error
# ================================================================
try:
    from statsmodels.tsa.seasonal import STL

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f'STL Decomposition of SK Forecast1 Error', fontsize=14, y=0.98)

    valid = merged.dropna(subset=[main_model, 'price_actual']).copy()
    valid = valid.set_index('datetime').sort_index()
    valid = valid[~valid.index.duplicated(keep='first')]
    err_series = valid[f'{main_model}_error']

    # Resample to regular hourly for STL
    err_hourly = err_series.resample('h').mean().dropna()

    if len(err_hourly) > 48:
        stl = STL(err_hourly, period=24, robust=True)
        result = stl.fit()

        ax = axes[0]
        ax.plot(result.observed.index, result.observed, color='black', linewidth=0.8)
        ax.set_ylabel('Error (EUR/MWh)')
        ax.set_title('Observed Forecast Error')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(result.trend.index, result.trend, color='#e74c3c', linewidth=1.5)
        ax.set_ylabel('Trend')
        ax.set_title('Trend Component')
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(result.seasonal.index, result.seasonal, color='#3498db', linewidth=0.8)
        ax.set_ylabel('Seasonal')
        ax.set_title('Seasonal Component (24h period)')
        ax.grid(True, alpha=0.3)

        ax = axes[3]
        ax.plot(result.resid.index, result.resid, color='#7f8c8d', linewidth=0.5, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel('Residual')
        ax.set_title('Residual Component')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUT_DIR / '03_stl_decomposition_forecast_error.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('[+] Saved 03_stl_decomposition_forecast_error.png')

        # Also decompose actual price for residual comparison
        actual_hourly = valid['price_actual'].resample('h').mean().dropna()
        common_idx = result.resid.index.intersection(actual_hourly.index)

        if len(common_idx) > 48:
            stl_actual = STL(actual_hourly.loc[common_idx], period=24, robust=True)
            result_actual = stl_actual.fit()

            # Save decomposition data
            decomp = pd.DataFrame({
                'forecast_error_observed': result.observed,
                'forecast_error_trend': result.trend,
                'forecast_error_seasonal': result.seasonal,
                'forecast_error_residual': result.resid,
            })
            decomp_actual = pd.DataFrame({
                'actual_price_observed': result_actual.observed,
                'actual_price_trend': result_actual.trend,
                'actual_price_seasonal': result_actual.seasonal,
                'actual_price_residual': result_actual.resid,
            })
            decomp_merged = decomp.join(decomp_actual, how='inner').dropna()
            decomp_merged.to_csv(OUT_DIR / 'data' / 'decomposition_residuals.csv')

            # Store for PLOT 4
            resid_forecast_error = decomp_merged['forecast_error_residual']
            resid_actual_price = decomp_merged['actual_price_residual']
        else:
            resid_forecast_error = None
            resid_actual_price = None
    else:
        print('[-] Not enough data for STL decomposition')
        resid_forecast_error = None
        resid_actual_price = None

except ImportError:
    print('[-] statsmodels not available, skipping STL decomposition')
    resid_forecast_error = None
    resid_actual_price = None

# ================================================================
# PLOT 4: Residual correlation analysis
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 4a: Hourly error profile
ax = axes[0, 0]
valid_h = merged.dropna(subset=[main_model, 'price_actual'])
hourly_err = valid_h.groupby('hour')[f'{main_model}_error'].agg(['mean', 'std']).reset_index()
ax.bar(hourly_err['hour'], hourly_err['mean'], color='#e74c3c', alpha=0.7, edgecolor='white')
ax.errorbar(hourly_err['hour'], hourly_err['mean'], yerr=hourly_err['std'],
            fmt='none', color='black', capsize=3, alpha=0.5)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Mean Forecast Error (EUR/MWh)')
ax.set_title('SK Forecast1 Error by Hour of Day')
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)

# 4b: Error vs price level
ax = axes[0, 1]
valid_h2 = merged.dropna(subset=[main_model, 'price_actual']).copy()
valid_h2['price_bin'] = pd.cut(valid_h2['price_actual'],
                                bins=np.arange(0, 300, 20))
pb = valid_h2.groupby('price_bin', observed=True)[f'{main_model}_error'].agg(['mean', 'std', 'count']).reset_index()
pb = pb[pb['count'] > 10]
bin_labels = [f'{int(x.left)}' for x in pb['price_bin']]
ax.bar(range(len(bin_labels)), pb['mean'], color='#3498db', alpha=0.7, edgecolor='white')
ax.errorbar(range(len(bin_labels)), pb['mean'], yerr=pb['std'],
            fmt='none', color='black', capsize=3, alpha=0.5)
ax.set_xticks(range(len(bin_labels)))
ax.set_xticklabels(bin_labels, rotation=45)
ax.set_xlabel('Actual Price Bin (EUR/MWh)')
ax.set_ylabel('Mean Forecast Error')
ax.set_title('Forecast Error vs Price Level')
ax.axhline(0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3)

# 4c: Residual correlation scatter
ax = axes[1, 0]
if resid_forecast_error is not None and resid_actual_price is not None:
    r_resid, p_resid = stats.pearsonr(resid_forecast_error, resid_actual_price)
    ax.scatter(resid_actual_price, resid_forecast_error, alpha=0.2, s=8, color='#9b59b6', edgecolors='none')
    slope_r, intercept_r, _, _, _ = stats.linregress(resid_actual_price, resid_forecast_error)
    x_r = np.linspace(resid_actual_price.min(), resid_actual_price.max(), 100)
    ax.plot(x_r, slope_r * x_r + intercept_r, 'r-', linewidth=2,
            label=f'r={r_resid:.3f}, slope={slope_r:.3f}')
    ax.set_xlabel('Actual Price Residual (EUR/MWh)')
    ax.set_ylabel('Forecast Error Residual (EUR/MWh)')
    ax.set_title('STL Residual: Forecast Error vs Actual Price')
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
else:
    ax.text(0.5, 0.5, 'STL decomposition\nnot available', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    r_resid = np.nan
    ax.set_title('STL Residual Correlation (N/A)')
ax.grid(True, alpha=0.3)

# 4d: Rolling correlation of raw forecast vs actual
ax = axes[1, 1]
daily2 = merged.dropna(subset=[main_model, 'price_actual']).groupby('date').agg(
    fc=( main_model, 'mean'),
    actual=('price_actual', 'mean'),
    err=(f'{main_model}_error', 'mean')
).reset_index()
daily2['date'] = pd.to_datetime(daily2['date'])
roll_r = daily2['fc'].rolling(14).corr(daily2['actual'])
roll_err_abs = daily2['err'].abs().rolling(7).mean()

ax.plot(daily2['date'], roll_r, color='#2ecc71', linewidth=2, label='Forecast-Actual corr (14d)')
ax2 = ax.twinx()
ax2.plot(daily2['date'], roll_err_abs, color='#e74c3c', linewidth=1, alpha=0.7,
         label='|Error| 7d rolling')
ax.set_ylabel('Rolling Correlation', color='#2ecc71')
ax2.set_ylabel('Rolling |Error| (EUR/MWh)', color='#e74c3c')
ax.set_title('Rolling Forecast Quality')
ax.legend(loc='lower left', fontsize=8)
ax2.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / '04_residual_and_error_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print('[+] Saved 04_residual_and_error_analysis.png')

# ================================================================
# PRINT KEY FINDINGS
# ================================================================
print('\n' + '=' * 65)
print('KEY FINDINGS')
print('=' * 65)
print(f'Period: {merged.datetime.min().date()} to {merged.datetime.max().date()}')
print(f'Resolution: hourly (forecasts aggregated from 15-min)')
print()

for col in active_models:
    s = error_stats[col]
    print(f'{labels[col]}:')
    print(f'  Bias={s["bias"]:+.1f}, MAE={s["mae"]:.1f}, RMSE={s["rmse"]:.1f}, r={s["r"]:.3f} (n={s["n"]})')

if resid_forecast_error is not None:
    print(f'\nSTL Residual correlation (forecast error vs actual price):')
    print(f'  r = {r_resid:.3f} (p={p_resid:.4f})')
    var_explained = r_resid**2 * 100
    print(f'  Residual R-squared: {var_explained:.1f}%')
print()

# Hourly pattern
best_hours = hourly_err.nsmallest(3, 'mean')[['hour', 'mean']]
worst_hours = hourly_err.nlargest(3, 'mean')[['hour', 'mean']]
print('Lowest bias hours:', ', '.join(f'H{int(r.hour)}({r["mean"]:+.1f})' for _, r in best_hours.iterrows()))
print('Highest bias hours:', ', '.join(f'H{int(r.hour)}({r["mean"]:+.1f})' for _, r in worst_hours.iterrows()))
