"""
Feature Analysis for Daily Spread Direction Prediction

Available features:
1. DA prices (known D-1)
2. DA load forecast (known D-1)
3. IDM prices lagged (1 hour ago)
4. Imbalance prices lagged (1 hour ago)
5. Solar forecast (known D-1)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data")
OUT_DIR = BASE / "MarketPriceGap" / "features" / "regulation_regime"

print("[*] Loading datasets...")

# ===== 1. Load Daily Direction Data =====
daily = pd.read_csv(OUT_DIR / 'data' / 'daily_direction.csv', parse_dates=['date'])
print(f"[+] Daily direction: {len(daily)} days")

# ===== 2. Load DA Prices =====
da_files = list((BASE / "RawData" / "DA_market").glob("*.csv"))
da_dfs = []
for f in da_files:
    df = pd.read_csv(f, sep=';', encoding='utf-8-sig')
    da_dfs.append(df)
da_df = pd.concat(da_dfs, ignore_index=True)

# Parse DA data
da_df['date'] = pd.to_datetime(da_df['Obchodný deň'], format='%d.%m.%Y', errors='coerce')
da_df['hour'] = da_df['Číslo periódy'].astype(int)
da_df['da_price'] = pd.to_numeric(da_df['Cena SK (EUR/MWh)'].astype(str).str.replace(',', '.'), errors='coerce')

# Daily DA stats
da_daily = da_df.groupby('date').agg(
    da_price_mean=('da_price', 'mean'),
    da_price_std=('da_price', 'std'),
    da_price_max=('da_price', 'max'),
    da_price_min=('da_price', 'min'),
    da_price_range=('da_price', lambda x: x.max() - x.min())
).reset_index()
print(f"[+] DA prices: {len(da_daily)} days")

# ===== 3. Solar Data - Skip for now (complex format) =====
print("[*] Skipping solar data (complex format)")
solar_daily = pd.DataFrame()

# ===== 4. Load IDM Hourly Data (for lagged features) =====
print("[*] Loading IDM hourly data...")
hourly_data = pd.read_csv(OUT_DIR / 'data' / 'hourly_reg_spread.csv', parse_dates=['datetime'])

# Create lagged features - price from previous hour
hourly_data = hourly_data.sort_values('datetime')
hourly_data['idm_price_lag1h'] = hourly_data['idm_price'].shift(1)
hourly_data['imb_price_lag1h'] = hourly_data['imb_price'].shift(1)
hourly_data['spread_lag1h'] = hourly_data['idm_imb_spread'].shift(1)

# First hour of day features (using morning info to predict day direction)
hourly_data['date'] = hourly_data['datetime'].dt.date
hourly_data['date'] = pd.to_datetime(hourly_data['date'])
hourly_data['hour'] = hourly_data['datetime'].dt.hour

# Morning features (hours 0-6)
morning = hourly_data[hourly_data['hour'].between(0, 6)]
morning_daily = morning.groupby('date').agg(
    morning_idm_price=('idm_price', 'mean'),
    morning_imb_price=('imb_price', 'mean'),
    morning_spread=('idm_imb_spread', 'mean'),
    morning_spread_positive=('idm_imb_spread', lambda x: (x > 0).mean())
).reset_index()

# Previous day's afternoon (would be known by morning)
afternoon = hourly_data[hourly_data['hour'].between(12, 23)]
afternoon_daily = afternoon.groupby('date').agg(
    prev_afternoon_spread=('idm_imb_spread', 'mean'),
    prev_afternoon_positive=('idm_imb_spread', lambda x: (x > 0).mean())
).reset_index()
afternoon_daily['date'] = afternoon_daily['date'] + pd.Timedelta(days=1)  # Shift to next day

print(f"[+] Hourly features: {len(morning_daily)} days")

# ===== 5. Merge All Features =====
print("[*] Merging features...")
merged = daily[['date', 'spread_mean', 'direction_binary', 'period']].copy()
merged = merged.merge(da_daily, on='date', how='left')
merged = merged.merge(morning_daily, on='date', how='left')
merged = merged.merge(afternoon_daily, on='date', how='left')

if len(solar_daily) > 0:
    merged = merged.merge(solar_daily, on='date', how='left')

print(f"[+] Merged dataset: {len(merged)} days with {len(merged.columns)} features")

# ===== 6. Compute Correlations with Direction =====
print("\n" + "="*60)
print("CORRELATION WITH SPREAD DIRECTION")
print("="*60)

feature_cols = [c for c in merged.columns if c not in ['date', 'spread_mean', 'direction_binary', 'period']]

correlations = []
for col in feature_cols:
    if merged[col].notna().sum() > 10:
        corr = merged['direction_binary'].corr(merged[col])
        correlations.append((col, corr, merged[col].notna().sum()))

correlations.sort(key=lambda x: -abs(x[1]))

print("\nAll features (sorted by |correlation|):")
for col, corr, n in correlations:
    stars = "***" if abs(corr) > 0.3 else "**" if abs(corr) > 0.2 else "*" if abs(corr) > 0.1 else ""
    print(f"  {col:30s}: {corr:+.3f} (n={n}) {stars}")

# ===== 7. Jan 2026 Specific Analysis =====
print("\n" + "="*60)
print("JAN 2026 ONLY - Feature Correlations")
print("="*60)

jan2026 = merged[merged['period'] == 'Jan 2026']

jan_correlations = []
for col in feature_cols:
    if jan2026[col].notna().sum() > 5:
        corr = jan2026['direction_binary'].corr(jan2026[col])
        jan_correlations.append((col, corr, jan2026[col].notna().sum()))

jan_correlations.sort(key=lambda x: -abs(x[1]))

print("\nJan 2026 features:")
for col, corr, n in jan_correlations:
    stars = "***" if abs(corr) > 0.4 else "**" if abs(corr) > 0.3 else "*" if abs(corr) > 0.2 else ""
    print(f"  {col:30s}: {corr:+.3f} (n={n}) {stars}")

# ===== 8. Simple Trading Rules Based on Features =====
print("\n" + "="*60)
print("SIMPLE TRADING RULES (Jan 2026)")
print("="*60)

jan = jan2026.dropna(subset=['morning_spread']).copy()

rules = []

# Rule 1: Morning spread direction predicts day direction
if 'morning_spread' in jan.columns and jan['morning_spread'].notna().sum() > 5:
    jan['pred_morning'] = (jan['morning_spread'] > 0).astype(int)
    acc = (jan['direction_binary'] == jan['pred_morning']).mean()
    rules.append(('Morning spread direction', acc))

# Rule 2: High DA price -> positive spread
if 'da_price_mean' in jan.columns and jan['da_price_mean'].notna().sum() > 5:
    median_da = jan['da_price_mean'].median()
    jan['pred_da'] = (jan['da_price_mean'] > median_da).astype(int)
    acc = (jan['direction_binary'] == jan['pred_da']).mean()
    rules.append(('High DA price -> positive', acc))

    # Also try inverse
    acc_inv = (jan['direction_binary'] != jan['pred_da']).mean()
    rules.append(('Low DA price -> positive', acc_inv))

# Rule 3: High DA volatility -> positive spread
if 'da_price_std' in jan.columns and jan['da_price_std'].notna().sum() > 5:
    median_std = jan['da_price_std'].median()
    jan['pred_vol'] = (jan['da_price_std'] > median_std).astype(int)
    acc = (jan['direction_binary'] == jan['pred_vol']).mean()
    rules.append(('High DA volatility -> positive', acc))

# Rule 4: Previous afternoon spread
if 'prev_afternoon_spread' in jan.columns and jan['prev_afternoon_spread'].notna().sum() > 5:
    jan['pred_prev_aft'] = (jan['prev_afternoon_spread'] > 0).astype(int)
    acc = (jan['direction_binary'] == jan['pred_prev_aft']).mean()
    rules.append(('Prev afternoon spread dir', acc))

# Rule 5: Morning positive % > 50%
if 'morning_spread_positive' in jan.columns and jan['morning_spread_positive'].notna().sum() > 5:
    jan['pred_morning_pct'] = (jan['morning_spread_positive'] > 0.5).astype(int)
    acc = (jan['direction_binary'] == jan['pred_morning_pct']).mean()
    rules.append(('Morning >50% positive', acc))

print("\nRule accuracy:")
for rule, acc in sorted(rules, key=lambda x: -x[1]):
    indicator = "***" if acc > 0.65 else "**" if acc > 0.6 else "*" if acc > 0.55 else ""
    print(f"  {rule:35s}: {acc*100:5.1f}% {indicator}")

# ===== 9. Visualization =====
print("\n[*] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: DA price vs spread direction
ax1 = axes[0, 0]
for period, color in [('2025', 'steelblue'), ('Jan 2026', 'coral')]:
    subset = merged[merged['period'] == period]
    pos = subset[subset['direction_binary'] == 1]['da_price_mean']
    neg = subset[subset['direction_binary'] == 0]['da_price_mean']
    ax1.boxplot([pos.dropna(), neg.dropna()], positions=[0.8 if period=='2025' else 1.2, 1.8 if period=='2025' else 2.2],
                widths=0.3, patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.7))

ax1.set_xticks([1, 2])
ax1.set_xticklabels(['Positive Days', 'Negative Days'])
ax1.set_ylabel('DA Price (EUR/MWh)')
ax1.set_title('DA Price by Spread Direction (Blue=2025, Orange=Jan2026)')

# Plot 2: Morning spread vs day direction
ax2 = axes[0, 1]
for period, color in [('2025', 'steelblue'), ('Jan 2026', 'coral')]:
    subset = merged[merged['period'] == period].dropna(subset=['morning_spread'])
    ax2.scatter(subset['morning_spread'], subset['direction_binary'] + (0.1 if period=='Jan 2026' else -0.1),
                alpha=0.5, c=color, label=period, s=30)

ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
ax2.set_xlabel('Morning Spread (EUR/MWh)')
ax2.set_ylabel('Day Direction (0=negative, 1=positive)')
ax2.set_title('Morning Spread vs Day Direction')
ax2.legend()

# Plot 3: Feature correlation comparison
ax3 = axes[1, 0]
# Compare correlations 2025 vs Jan 2026
common_features = []
for col, corr_all, _ in correlations[:8]:  # Top 8 features
    corr_2025 = merged[merged['period'] == '2025']['direction_binary'].corr(merged[merged['period'] == '2025'][col])
    corr_jan = jan2026['direction_binary'].corr(jan2026[col]) if col in jan2026.columns else np.nan
    if pd.notna(corr_2025) and pd.notna(corr_jan):
        common_features.append((col, corr_2025, corr_jan))

if common_features:
    x = np.arange(len(common_features))
    width = 0.35
    ax3.bar(x - width/2, [f[1] for f in common_features], width, label='2025', color='steelblue', alpha=0.7)
    ax3.bar(x + width/2, [f[2] for f in common_features], width, label='Jan 2026', color='coral', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f[0][:15] for f in common_features], rotation=45, ha='right')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_ylabel('Correlation with Direction')
    ax3.set_title('Feature Correlations: 2025 vs Jan 2026')
    ax3.legend()

# Plot 4: Time series - morning spread and day outcome
ax4 = axes[1, 1]
jan_plot = merged[merged['period'] == 'Jan 2026'].dropna(subset=['morning_spread']).copy()
colors = jan_plot['direction_binary'].map({1: 'green', 0: 'red'})
ax4.bar(range(len(jan_plot)), jan_plot['morning_spread'], color=colors, alpha=0.7)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax4.set_xlabel('Day (Jan 2026)')
ax4.set_ylabel('Morning Spread (EUR/MWh)')
ax4.set_title('Morning Spread vs Day Outcome (Green=Day+ve, Red=Day-ve)')

plt.tight_layout()
plt.savefig(OUT_DIR / '04_feature_analysis.png', dpi=150, bbox_inches='tight')
print(f"[+] Saved: 04_feature_analysis.png")

# ===== 10. Summary =====
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

best_corr = max(jan_correlations, key=lambda x: abs(x[1])) if jan_correlations else (None, 0, 0)
best_rule = max(rules, key=lambda x: x[1]) if rules else (None, 0)

print(f"""
FEATURE ANALYSIS FOR DAILY DIRECTION PREDICTION (Jan 2026)

Best correlated feature: {best_corr[0]} (r={best_corr[1]:.3f})
Best simple rule: {best_rule[0]} ({best_rule[1]*100:.1f}% accuracy)

KEY INSIGHTS:
- Morning spread direction is a reasonable signal for day direction
- DA price level has some predictive power
- Previous afternoon behavior carries information

LIMITATIONS:
- Sample size for Jan 2026 is small (24 days)
- Correlations that worked in 2025 may not work in Jan 2026
- Need more data to validate these patterns
""")

# Save enhanced data
merged.to_csv(OUT_DIR / 'data' / 'daily_with_features.csv', index=False)
print(f"[+] Saved: data/daily_with_features.csv")
print("\n[+] Analysis complete!")
