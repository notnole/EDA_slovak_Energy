"""
Structural Regression: Imbalance = f(Load Surprise, Cross-Border IDM Flow)

Tests the hypothesis:
  System Imbalance = Load Surprise - Net Cross-Border IDM Correction

Uses H2H capacity reduction (first-last snapshot) as a proxy for
actual cross-border IDM flow into/out of Slovakia.

Data is at 15-minute resolution (matching OKTE imbalance periods).
Load surprise is hourly (broadcast to all 4 QH periods within each hour).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent.parent
OUT = Path(__file__).resolve().parent
DATA_OUT = OUT / "data"

# =========================================================================
# 1. Load all data
# =========================================================================
print("[*] Loading data sources...")

# --- H2H cross-border flow (15-min, pivoted) ---
xborder = pd.read_csv(DATA_OUT / "h2h_xborder_flow.csv")
xborder['timestamp'] = pd.to_datetime(xborder['timestamp'])
print(f"  [+] H2H cross-border: {len(xborder)} rows, "
      f"{xborder['timestamp'].min()} to {xborder['timestamp'].max()}")

# --- H2H per-border detail ---
h2h_detail = pd.read_csv(DATA_OUT / "h2h_capacity_reduction.csv")
h2h_detail['periodfrom'] = pd.to_datetime(h2h_detail['periodfrom'])

# --- Load surprise (hourly) ---
load_df = pd.read_csv(BASE / "features" / "DamasLoad" / "load_data.csv",
                       parse_dates=["datetime"])
load_df = load_df.rename(columns={"datetime": "timestamp_hour"})
print(f"  [+] Load data: {len(load_df)} rows")

# --- OKTE imbalance (15-min) ---
imb_df = pd.read_csv(BASE / "data" / "master" / "master_imbalance_data.csv")
# The master file has various date formats - parse carefully
if 'timestamp' in imb_df.columns:
    imb_df['timestamp'] = pd.to_datetime(imb_df['timestamp'])
elif 'period_start' in imb_df.columns:
    imb_df['timestamp'] = pd.to_datetime(imb_df['period_start'])
else:
    # Check columns
    print(f"  [!] Imbalance columns: {list(imb_df.columns[:10])}")

print(f"  [+] Imbalance: {len(imb_df)} rows")
print(f"  Columns: {list(imb_df.columns[:15])}")

# --- Hourly market data (has hourly imbalance sum) ---
mkt_df = pd.read_csv(BASE / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                      parse_dates=["timestamp_hour"])
print(f"  [+] Market hourly: {len(mkt_df)} rows")

# =========================================================================
# 2. Prepare H2H at hourly resolution
# =========================================================================
print("\n[*] Aggregating H2H to hourly...")

xborder['hour'] = xborder['timestamp'].dt.floor('h')

# Sum capacity used across 4 QH periods within each hour
# cap_used is in MW; for energy balance, multiply by 0.25 to get MWh
xborder_hourly = xborder.groupby('hour').agg({
    'net_xborder_import': 'mean',  # Average MW flow during the hour
    'total_xborder_import_used': 'mean',
    'total_xborder_export_used': 'mean',
}).reset_index()

# Convert MW to MWh (1 hour period)
xborder_hourly['net_xborder_import_mwh'] = xborder_hourly['net_xborder_import']
# Keep MW for the regression since load surprise is also in MW

# Also get per-border hourly averages
for border in ['CEPS', 'APG', 'TTG', '50HzT', 'MAVIR', 'PSE']:
    in_col = f"{border}_cap_used_in"
    out_col = f"{border}_cap_used_out"
    if in_col in xborder.columns and out_col in xborder.columns:
        border_hourly = xborder.groupby('hour').agg({in_col: 'mean', out_col: 'mean'}).reset_index()
        border_hourly[f'{border}_net_import'] = border_hourly[in_col] - border_hourly[out_col]
        xborder_hourly = xborder_hourly.merge(
            border_hourly[['hour', f'{border}_net_import']], on='hour', how='left')

xborder_hourly = xborder_hourly.rename(columns={'hour': 'timestamp_hour'})
print(f"  [+] Hourly H2H: {len(xborder_hourly)} rows")
print(f"  Columns: {list(xborder_hourly.columns)}")

# =========================================================================
# 3. Merge all datasets at hourly resolution
# =========================================================================
print("\n[*] Merging datasets...")

# Core: load surprise + hourly market data (has imbalance_mwh)
df = load_df[['timestamp_hour', 'forecast_error_mw', 'actual_load_mw',
              'forecast_load_mw', 'hour']].merge(
    mkt_df[['timestamp_hour', 'imbalance_mwh', 'da_price', 'imb_settlement_price']],
    on='timestamp_hour', how='inner'
)
print(f"  Load + Market: {len(df)} rows")

# Add H2H cross-border flow
df = df.merge(xborder_hourly, on='timestamp_hour', how='inner')
print(f"  + H2H: {len(df)} rows ({df['timestamp_hour'].min()} to {df['timestamp_hour'].max()})")

# Rename for clarity
df['load_surprise_mw'] = df['forecast_error_mw']

# =========================================================================
# 4. Feature engineering
# =========================================================================
print("\n[*] Engineering features...")

# Structural residual: Load surprise not absorbed by cross-border flow
# If load_surprise > 0 (more demand), net_import should increase (positive)
# Residual = load_surprise - net_import_adjustment
df['structural_residual'] = df['load_surprise_mw'] - df['net_xborder_import']

# Also try: only the import side (cross-border counter-trading)
df['import_correction'] = df['total_xborder_import_used']
df['export_correction'] = df['total_xborder_export_used']

print(f"  Load surprise std:       {df['load_surprise_mw'].std():.1f} MW")
print(f"  Net xborder import std:  {df['net_xborder_import'].std():.1f} MW")
print(f"  Structural residual std: {df['structural_residual'].std():.1f} MW")
print(f"  Imbalance std:           {df['imbalance_mwh'].std():.1f} MWh")

# =========================================================================
# 5. Correlation analysis
# =========================================================================
print("\n[*] Correlation analysis...")

key_cols = ['imbalance_mwh', 'load_surprise_mw', 'net_xborder_import',
            'structural_residual', 'total_xborder_import_used', 'total_xborder_export_used']

# Add per-border if available
border_cols = [c for c in df.columns if c.endswith('_net_import') and c != 'net_xborder_import']
key_cols.extend(border_cols)

corr = df[key_cols].corr()
print("\nCorrelation with imbalance_mwh:")
for col in key_cols[1:]:
    r = corr.loc['imbalance_mwh', col]
    print(f"  {col:35s} r = {r:+.4f}")

print("\nCorrelation: load_surprise vs net_xborder_import:")
r_ls_xb = df[['load_surprise_mw', 'net_xborder_import']].corr().iloc[0, 1]
print(f"  r = {r_ls_xb:+.4f}")
if r_ls_xb > 0:
    print("  -> When load > forecast, cross-border import increases (expected correction)")
else:
    print("  -> When load > forecast, cross-border import DECREASES (counter-intuitive!)")

# =========================================================================
# 6. Regression models
# =========================================================================
print("\n[*] Running structural regressions...")
print("=" * 80)

results = []

def run_model(name, features, data=df, target="imbalance_mwh"):
    mask = data[features + [target]].dropna().index
    X = data.loc[mask, features].values
    y = data.loc[mask, target].values
    n = len(y)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    r = np.sign(np.corrcoef(y, y_pred)[0, 1]) * np.sqrt(abs(r2))
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    print(f"\n--- {name} ---")
    print(f"  N={n}, R2={r2:.4f}, r={r:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    for feat, coef in zip(features, model.coef_):
        print(f"    {feat:35s} coef={coef:+.6f}")
    print(f"    {'intercept':35s} = {model.intercept_:+.4f}")

    results.append({"model": name, "n": n, "r2": r2, "r": r, "mae": mae, "rmse": rmse})
    for feat, coef in zip(features, model.coef_):
        results[-1][f"coef_{feat}"] = coef
    return model


# --- Baseline: Load surprise only ---
run_model("S1: Load Surprise Only", ["load_surprise_mw"])

# --- S2: Load + Net Cross-Border Import ---
run_model("S2: Load + Net XBorder Import", ["load_surprise_mw", "net_xborder_import"])

# --- S3: Load + Import/Export separately ---
run_model("S3: Load + Import/Export Split",
          ["load_surprise_mw", "total_xborder_import_used", "total_xborder_export_used"])

# --- S4: Structural residual only ---
run_model("S4: Structural Residual", ["structural_residual"])

# --- S5: Load + per-border net imports ---
border_feats = ["load_surprise_mw"] + border_cols
if len(border_cols) > 0:
    run_model("S5: Load + Per-Border Net Import", border_feats)

# --- S6: Full model ---
full_feats = ["load_surprise_mw", "net_xborder_import",
              "total_xborder_import_used", "total_xborder_export_used"]
run_model("S6: Full Features", full_feats)

# =========================================================================
# 7. Hourly breakdown
# =========================================================================
print("\n" + "=" * 80)
print("[*] R-squared by hour of day...")

hourly_r2 = []
for h in range(24):
    sub = df[df['hour'] == h].dropna(subset=['load_surprise_mw', 'net_xborder_import', 'imbalance_mwh'])
    if len(sub) < 20:
        continue
    # Load only
    m1 = LinearRegression().fit(sub[['load_surprise_mw']], sub['imbalance_mwh'])
    r2_load = r2_score(sub['imbalance_mwh'], m1.predict(sub[['load_surprise_mw']]))
    # Load + xborder
    X2 = sub[['load_surprise_mw', 'net_xborder_import']]
    m2 = LinearRegression().fit(X2, sub['imbalance_mwh'])
    r2_xborder = r2_score(sub['imbalance_mwh'], m2.predict(X2))
    # Structural residual only
    m3 = LinearRegression().fit(sub[['structural_residual']], sub['imbalance_mwh'])
    r2_residual = r2_score(sub['imbalance_mwh'], m3.predict(sub[['structural_residual']]))

    hourly_r2.append({'hour': h, 'r2_load': r2_load, 'r2_load_xborder': r2_xborder,
                       'r2_residual': r2_residual, 'gain': r2_xborder - r2_load, 'n': len(sub)})
    print(f"  H{h:02d}: R2_load={r2_load:.3f}  R2_load+xborder={r2_xborder:.3f}  "
          f"R2_residual={r2_residual:.3f}  gain={r2_xborder - r2_load:+.3f}  (n={len(sub)})")

hourly_df = pd.DataFrame(hourly_r2)

# =========================================================================
# 8. Save results
# =========================================================================
pd.DataFrame(results).to_csv(DATA_OUT / "structural_regression_results.csv", index=False)
hourly_df.to_csv(DATA_OUT / "structural_hourly_r2.csv", index=False)
df[key_cols].corr().to_csv(DATA_OUT / "structural_correlation_matrix.csv")

# =========================================================================
# 9. Visualization
# =========================================================================
print("\n[*] Creating plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Structural Decomposition: Imbalance = Load Surprise - Cross-Border IDM Flow",
             fontsize=14, fontweight="bold")

# --- Panel 1: Model comparison ---
ax = axes[0, 0]
res_df = pd.DataFrame(results)
labels = res_df['model'].str.replace(r'S\d+: ', '', regex=True)
ax.barh(range(len(res_df)), res_df['r2'], color='#4477AA', edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(res_df)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('R-squared')
ax.set_title('Structural Model Comparison')
for i, r2 in enumerate(res_df['r2']):
    ax.text(r2 + 0.002, i, f'{r2:.4f}', va='center', fontsize=8)
ax.invert_yaxis()

# --- Panel 2: Load surprise vs cross-border import ---
ax = axes[0, 1]
sample = df.sample(min(2000, len(df)), random_state=42)
ax.scatter(sample['load_surprise_mw'], sample['net_xborder_import'],
           alpha=0.2, s=8, c='#228833')
r_val = df[['load_surprise_mw', 'net_xborder_import']].corr().iloc[0, 1]
ax.set_xlabel('Load Surprise (MW)')
ax.set_ylabel('Net Cross-Border Import (MW)')
ax.set_title(f'Load Surprise vs XBorder Import (r={r_val:.3f})')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# --- Panel 3: Structural residual vs imbalance ---
ax = axes[0, 2]
ax.scatter(sample['structural_residual'], sample['imbalance_mwh'],
           alpha=0.2, s=8, c='#EE6677')
r_res = df[['structural_residual', 'imbalance_mwh']].corr().iloc[0, 1]
ax.set_xlabel('Structural Residual (Load Surprise - XBorder Import) (MW)')
ax.set_ylabel('Imbalance (MWh)')
ax.set_title(f'Structural Residual vs Imbalance (r={r_res:.3f})')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# --- Panel 4: Per-border contribution scatter ---
ax = axes[1, 0]
if border_cols:
    colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#AA3377', '#66CCEE']
    for i, col in enumerate(border_cols[:6]):
        border_name = col.replace('_net_import', '')
        r_b = df[[col, 'imbalance_mwh']].corr().iloc[0, 1]
        ax.bar(i, r_b, color=colors[i % len(colors)], label=f'{border_name} (r={r_b:.3f})')
    ax.set_xticks(range(len(border_cols[:6])))
    ax.set_xticklabels([c.replace('_net_import', '') for c in border_cols[:6]], fontsize=9)
    ax.set_ylabel('Correlation with Imbalance')
    ax.set_title('Per-Border Correlation')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend(fontsize=7)

# --- Panel 5: Hourly R2 comparison ---
ax = axes[1, 1]
if len(hourly_df) > 0:
    w = 0.25
    ax.bar(hourly_df['hour'] - w, hourly_df['r2_load'], width=w,
           label='Load only', color='#4477AA', edgecolor='black', linewidth=0.5)
    ax.bar(hourly_df['hour'], hourly_df['r2_load_xborder'], width=w,
           label='Load + XBorder', color='#EE6677', edgecolor='black', linewidth=0.5)
    ax.bar(hourly_df['hour'] + w, hourly_df['r2_residual'], width=w,
           label='Structural Residual', color='#228833', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('R-squared')
    ax.set_title('Explanatory Power by Hour')
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, 24, 3))

# --- Panel 6: Time series of structural residual vs imbalance ---
ax = axes[1, 2]
# Weekly average to smooth
df_ts = df.set_index('timestamp_hour').sort_index()
weekly = df_ts[['load_surprise_mw', 'net_xborder_import', 'structural_residual',
                'imbalance_mwh']].resample('W').mean()
ax.plot(weekly.index, weekly['load_surprise_mw'], label='Load Surprise', alpha=0.7)
ax.plot(weekly.index, weekly['net_xborder_import'] / 50, label='XBorder Import /50', alpha=0.7)
ax.plot(weekly.index, weekly['imbalance_mwh'], label='Imbalance (MWh)', alpha=0.7, linewidth=2)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Weekly Averages')
ax.legend(fontsize=8)
ax.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(OUT / "02_structural_decomposition.png", dpi=150, bbox_inches="tight")
print(f"  [+] {OUT / '02_structural_decomposition.png'}")

# =========================================================================
# 10. Summary
# =========================================================================
print("\n" + "=" * 80)
print("STRUCTURAL REGRESSION SUMMARY")
print("=" * 80)

s1_r2 = res_df[res_df['model'].str.contains('S1')]['r2'].values[0]
s2_r2 = res_df[res_df['model'].str.contains('S2')]['r2'].values[0]
s4_r2 = res_df[res_df['model'].str.contains('S4')]['r2'].values[0]

print(f"\nHypothesis: Imbalance = Load Surprise - Net Cross-Border IDM Flow")
print(f"\n  Load surprise alone:              R2 = {s1_r2:.4f}")
print(f"  + Net cross-border import:         R2 = {s2_r2:.4f} (gain: {s2_r2 - s1_r2:+.4f})")
print(f"  Structural residual only:          R2 = {s4_r2:.4f}")

if s2_r2 > s1_r2 + 0.01:
    print(f"\n  [+] Cross-border flow IMPROVES the model by {(s2_r2 - s1_r2) * 100:.1f} percentage points")
elif s4_r2 > s1_r2:
    print(f"\n  [+] Structural residual is better than load alone: +{(s4_r2 - s1_r2) * 100:.1f} pp")
else:
    print(f"\n  [-] Cross-border flow does NOT improve the model")
    print(f"      Possible reasons:")
    print(f"      - H2H capacity reduction reflects ALL xborder changes (not just IDM counter-trading)")
    print(f"      - Scale mismatch: xborder flows are ~{df['net_xborder_import'].std():.0f} MW vs "
          f"imbalance ~{df['imbalance_mwh'].std():.0f} MWh")
    print(f"      - Need finer decomposition: separate DA-scheduled vs IDM-traded xborder")

print(f"\n  Key correlation: load_surprise <-> xborder_import: r = {r_ls_xb:+.4f}")
print(f"  Key correlation: structural_residual <-> imbalance: r = {r_res:+.4f}")

# Best and worst hours
if len(hourly_df) > 0:
    best = hourly_df.loc[hourly_df['gain'].idxmax()]
    worst = hourly_df.loc[hourly_df['gain'].idxmin()]
    print(f"\n  Best hour for XBorder signal: H{int(best['hour']):02d} (gain: {best['gain']:+.3f})")
    print(f"  Worst hour:                   H{int(worst['hour']):02d} (gain: {worst['gain']:+.3f})")

print("\n[+] Analysis complete.")
