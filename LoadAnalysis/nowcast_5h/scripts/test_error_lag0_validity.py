"""
Test whether error_lag0 is valid or leakage.

If error_lag0 is leakage, we'd expect:
1. Unusually high correlation with target (too good to be true)
2. Model with ONLY error_lag0 would perform suspiciously well
3. Shuffling error_lag0 would destroy most of the model's power

Let's test these hypotheses.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent

print("=" * 70)
print("TESTING: Is error_lag0 valid or leakage?")
print("=" * 70)

# Load data
df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
df['year'] = df['datetime'].dt.year
df['error'] = df['actual_load_mw'] - df['forecast_load_mw']

# Create features
df['error_lag0'] = df['error']  # Current hour (potentially problematic)
df['error_lag1'] = df['error'].shift(1)  # Previous hour (definitely valid)
df['error_lag2'] = df['error'].shift(2)
df['target_h1'] = df['error'].shift(-1)  # Next hour (what we predict)

df = df.dropna()

# Split
train = df[df['year'] == 2024]
test = df[df['year'] >= 2025]

print(f"\nData: {len(train):,} train, {len(test):,} test")

# =============================================================================
# TEST 1: Correlation analysis
# =============================================================================
print("\n" + "=" * 70)
print("TEST 1: Correlation with target")
print("=" * 70)

print(f"""
  error_lag0 (current hour):   r = {df['error_lag0'].corr(df['target_h1']):.4f}
  error_lag1 (previous hour):  r = {df['error_lag1'].corr(df['target_h1']):.4f}
  error_lag2 (2 hours ago):    r = {df['error_lag2'].corr(df['target_h1']):.4f}
""")

# The correlation decay should be smooth
# If lag0 is leakage, we'd see a jump (lag0 >> lag1)
corr_lag0 = df['error_lag0'].corr(df['target_h1'])
corr_lag1 = df['error_lag1'].corr(df['target_h1'])
corr_lag2 = df['error_lag2'].corr(df['target_h1'])

decay_0_to_1 = corr_lag0 - corr_lag1
decay_1_to_2 = corr_lag1 - corr_lag2

print(f"  Correlation decay lag0->lag1: {decay_0_to_1:.4f}")
print(f"  Correlation decay lag1->lag2: {decay_1_to_2:.4f}")

if decay_0_to_1 > decay_1_to_2 * 2:
    print("\n  [!] WARNING: Decay from lag0 to lag1 is unusually large!")
    print("      This COULD indicate lag0 contains future information.")
else:
    print("\n  [+] Decay pattern is consistent - lag0 appears legitimate.")

# =============================================================================
# TEST 2: Model with ONLY error_lag0
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: Model using ONLY error_lag0")
print("=" * 70)

# If lag0 is valid, a model using ONLY lag0 should do reasonably well
# If lag0 is leakage, it would do suspiciously well (near-perfect)

model_lag0_only = lgb.LGBMRegressor(n_estimators=50, max_depth=3, verbosity=-1)
model_lag0_only.fit(train[['error_lag0']], train['target_h1'])
pred_lag0 = model_lag0_only.predict(test[['error_lag0']])

model_lag1_only = lgb.LGBMRegressor(n_estimators=50, max_depth=3, verbosity=-1)
model_lag1_only.fit(train[['error_lag1']], train['target_h1'])
pred_lag1 = model_lag1_only.predict(test[['error_lag1']])

baseline_mae = test['target_h1'].abs().mean()
mae_lag0_only = np.abs(test['target_h1'] - pred_lag0).mean()
mae_lag1_only = np.abs(test['target_h1'] - pred_lag1).mean()

print(f"  Baseline (predict 0):  {baseline_mae:.1f} MW")
print(f"  Model with lag0 only:  {mae_lag0_only:.1f} MW ({100*(baseline_mae-mae_lag0_only)/baseline_mae:+.1f}%)")
print(f"  Model with lag1 only:  {mae_lag1_only:.1f} MW ({100*(baseline_mae-mae_lag1_only)/baseline_mae:+.1f}%)")

improvement_ratio = (baseline_mae - mae_lag0_only) / (baseline_mae - mae_lag1_only)
print(f"\n  Improvement ratio (lag0/lag1): {improvement_ratio:.2f}x")

if mae_lag0_only < 10:  # Suspiciously low
    print("\n  [!] CRITICAL: lag0-only model is too good - likely leakage!")
elif improvement_ratio > 2.0:
    print("\n  [!] WARNING: lag0 provides disproportionately large improvement.")
else:
    print("\n  [+] lag0 improvement is reasonable - appears legitimate.")

# =============================================================================
# TEST 3: What happens if we shuffle error_lag0?
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: Model robustness to shuffled lag0")
print("=" * 70)

# If lag0 is leakage, shuffling it would destroy model performance
# If lag0 is just a good feature, model should still work with lag1

features_all = ['error_lag0', 'error_lag1', 'error_lag2']

# Model with all features
model_all = lgb.LGBMRegressor(n_estimators=100, max_depth=5, verbosity=-1)
model_all.fit(train[features_all], train['target_h1'])
pred_all = model_all.predict(test[features_all])
mae_all = np.abs(test['target_h1'] - pred_all).mean()

# Model without lag0
features_no_lag0 = ['error_lag1', 'error_lag2']
model_no_lag0 = lgb.LGBMRegressor(n_estimators=100, max_depth=5, verbosity=-1)
model_no_lag0.fit(train[features_no_lag0], train['target_h1'])
pred_no_lag0 = model_no_lag0.predict(test[features_no_lag0])
mae_no_lag0 = np.abs(test['target_h1'] - pred_no_lag0).mean()

print(f"  Model with lag0,1,2:   {mae_all:.1f} MW")
print(f"  Model with lag1,2:     {mae_no_lag0:.1f} MW")
print(f"  Contribution of lag0:  {mae_no_lag0 - mae_all:.1f} MW improvement")

# =============================================================================
# TEST 4: Operational timing check
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: Operational timing analysis")
print("=" * 70)

print("""
QUESTION: At what point is error[t] actually known?

Timeline for hour t (e.g., 10:00-11:00):
  10:00 - Hour starts, SCADA begins measuring
  10:03 - First 3-min measurement available
  ...
  10:57 - Last 3-min measurement available
  11:00 - Hour ends, final actual_load[t] computed

error[t] = actual_load[t] - forecast_load[t]
         = (SCADA measurement sum) - (day-ahead forecast)

=> error[t] is KNOWN at 11:00 (hour boundary)
=> If predictions are made AT 11:00 for hour 11:00-12:00, error[t] is valid

HOWEVER:
- If predictions must be made BEFORE 11:00 (e.g., 10:45), error[t] is NOT known
- If there's a data processing delay (e.g., SCADA data arrives 5 min late), adjust accordingly
""")

# =============================================================================
# CONCLUSION
# =============================================================================
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

print(f"""
Based on the tests:

1. CORRELATION: Decay from lag0->lag1 ({decay_0_to_1:.3f}) vs lag1->lag2 ({decay_1_to_2:.3f})
   {'[!] Suspicious' if decay_0_to_1 > decay_1_to_2 * 2 else '[+] Consistent'}

2. SINGLE FEATURE: lag0-only model achieves {mae_lag0_only:.1f} MW MAE
   {'[!] Suspicious (too good)' if mae_lag0_only < 20 else '[+] Reasonable'}

3. CONTRIBUTION: Adding lag0 improves by {mae_no_lag0 - mae_all:.1f} MW
   {'[!] Suspicious (too much)' if (mae_no_lag0 - mae_all) > 15 else '[+] Reasonable'}

VERDICT: error_lag0 appears to be {'VALID' if decay_0_to_1 <= decay_1_to_2 * 2 else 'SUSPICIOUS'}
         IF predictions are made at hour boundaries.

RECOMMENDATION:
- If your operational setup makes predictions AT hour boundaries -> USE error_lag0
- If predictions must be made BEFORE hour ends -> DO NOT use error_lag0
- When in doubt, use the conservative approach (error_lag1 only)
""")
