"""Comprehensive leakage check for the spread prediction model."""

import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

REPO_ROOT = Path(__file__).parent.parent.parent

data = load_all_data()
tml.TRAIN_END = '2026-01-31'
tml.TEST_START = '2026-02-01'

lead = 8
df, feature_cols = build_features(data, lead)

ob_exec = pd.read_csv(Path(__file__).parent.parent / "data" / "orderbook_qh_features.csv",
                       parse_dates=['delivery_start'])
ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'spread', 'mid']]
ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']

mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                   parse_dates=['timestamp_hour'], index_col='timestamp_hour')
mkt = mkt[~mkt.index.duplicated(keep='last')]

df['hour_ts'] = df.index.floor('h')
df = df.join(mkt[['imb_settlement_price']], on='hour_ts', how='left')
df = df.join(ob_120, how='left')
df['spread_target'] = df['imb_settlement_price'] - df['exec_mid']

valid = df.dropna(subset=['target', 'proxy_lag9', 'spread_target', 'exec_mid'])
valid = valid[valid['imb_settlement_price'].abs() <= 5000]

train = valid[valid.index <= tml.TRAIN_END]
test = valid[valid.index >= tml.TEST_START]

base = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
            reg_lambda=1.0, n_estimators=600, verbose=-1)

print("=" * 70)
print("SPREAD MODEL LEAKAGE INVESTIGATION")
print("=" * 70)

# CHECK 1: Features correlated with exec_mid (the IDM price at prediction time)
print("\n--- CHECK 1: Features correlated with exec_mid ---")
suspect = []
for col in feature_cols:
    c = valid[[col, 'exec_mid']].dropna()
    if len(c) > 100:
        r = c[col].corr(c['exec_mid'])
        if abs(r) > 0.3:
            suspect.append((col, r))
suspect.sort(key=lambda x: abs(x[1]), reverse=True)
if suspect:
    print("  Features with |corr| > 0.3 with exec_mid:")
    for col, r in suspect[:15]:
        print(f"    {col:<35s}: {r:+.3f}")
else:
    print("  None found")

# CHECK 2: Spread model feature importance
print("\n--- CHECK 2: Spread model top features ---")
m_sp = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **base)
m_sp.fit(train[feature_cols].values, train['spread_target'].values)
pred_sp = m_sp.predict(test[feature_cols].values)

imp = pd.DataFrame({'feature': feature_cols, 'importance': m_sp.feature_importances_})
imp['pct'] = imp['importance'] / imp['importance'].sum() * 100
imp = imp.sort_values('importance', ascending=False)
for _, row in imp.head(20).iterrows():
    print(f"  {row['feature']:<35s} {row['pct']:>5.2f}%")

# CHECK 3: Is the model just learning a constant or time-of-day pattern?
print("\n--- CHECK 3: Prediction distribution ---")
print(f"  Pred: mean={pred_sp.mean():+.1f}, std={pred_sp.std():.1f}, "
      f"min={pred_sp.min():+.1f}, max={pred_sp.max():+.1f}")
print(f"  Unique rounded values: {len(np.unique(np.round(pred_sp, 0)))}")

# CHECK 4: Null model — constant prediction
print("\n--- CHECK 4: Null model (always predict training median) ---")
median_spread = train['spread_target'].median()
print(f"  Training median spread: {median_spread:+.1f} EUR/MWh")
te = test[test['exec_spread'].notna() & (test['exec_spread'] <= 10)].copy()
# Always buy (deficit) if median > 0, always sell if median < 0
te['pnl'] = 0.0
if median_spread > 0:
    te['pnl'] = (te['imb_settlement_price'] - te['exec_ask']) * 5.0 / 4
else:
    te['pnl'] = (te['exec_bid'] - te['imb_settlement_price']) * 5.0 / 4
nd = te.index.normalize().nunique()
wr = (te['pnl'] > 0).mean()
print(f"  Always {'buy' if median_spread > 0 else 'sell'}: "
      f"{len(te)} trades, win={wr:.0%}, P&L={te['pnl'].sum():>+,.0f} ({te['pnl'].sum()/nd:>+,.0f}/day)")

# CHECK 5: Walk-forward monthly — does it hold every month?
print("\n--- CHECK 5: Walk-forward monthly P&L ---")
monthly_configs = [
    ('2025-09-30', '2025-10-01', '2025-11-01', 'Oct 2025'),
    ('2025-10-31', '2025-11-01', '2025-12-01', 'Nov 2025'),
    ('2025-11-30', '2025-12-01', '2026-01-01', 'Dec 2025'),
    ('2025-12-31', '2026-01-01', '2026-02-01', 'Jan 2026'),
    ('2026-01-31', '2026-02-01', '2026-03-01', 'Feb 2026'),
    ('2026-02-28', '2026-03-01', '2026-04-01', 'Mar 2026'),
]

for train_end, test_start, test_end, label in monthly_configs:
    tr = valid[valid.index <= train_end]
    te = valid[(valid.index >= test_start) & (valid.index < test_end)]
    te = te[te['exec_spread'].notna() & (te['exec_spread'] <= 10)]
    if len(tr) < 100 or len(te) < 100:
        print(f"  {label}: insufficient data")
        continue

    m = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **base)
    m.fit(tr[feature_cols].values, tr['spread_target'].values)
    te = te.copy()
    te['pred'] = m.predict(te[feature_cols].values)

    # Trade when |pred| >= 3
    mask = te['pred'].abs() >= 3
    sub = te[mask].copy()
    sub['size'] = sub['pred'].abs().clip(upper=5)
    surplus = sub['pred'] < 0
    deficit = sub['pred'] > 0
    sub['pnl'] = 0.0
    sub.loc[surplus, 'pnl'] = (sub.loc[surplus, 'exec_bid'] - sub.loc[surplus, 'imb_settlement_price']) * sub.loc[surplus, 'size'] / 4
    sub.loc[deficit, 'pnl'] = (sub.loc[deficit, 'imb_settlement_price'] - sub.loc[deficit, 'exec_ask']) * sub.loc[deficit, 'size'] / 4
    nd = sub.index.normalize().nunique() if len(sub) > 0 else 1
    total = sub['pnl'].sum()
    wr = (sub['pnl'] > 0).mean() if len(sub) > 0 else 0
    nz = sub['spread_target'].abs() > 0.1
    da = (np.sign(sub['pred']) == np.sign(sub['spread_target']))[nz].mean() if nz.sum() > 0 else 0
    print(f"  {label}: {len(sub):4d} trades, dir={da:.0%}, win={wr:.0%}, "
          f"P&L={total:>+7,.0f} ({total/nd:>+4.0f}/day)")

# CHECK 6: Does the spread model P&L come from a few outlier settlements?
print("\n--- CHECK 6: P&L concentration (spread MAE, |pred|>=3) ---")
te_full = test[test['exec_spread'].notna() & (test['exec_spread'] <= 10)].copy()
te_full['pred'] = m_sp.predict(te_full[feature_cols].values)
mask = te_full['pred'].abs() >= 3
sub = te_full[mask].copy()
sub['size'] = sub['pred'].abs().clip(upper=5)
surplus = sub['pred'] < 0
deficit = sub['pred'] > 0
sub['pnl'] = 0.0
sub.loc[surplus, 'pnl'] = (sub.loc[surplus, 'exec_bid'] - sub.loc[surplus, 'imb_settlement_price']) * sub.loc[surplus, 'size'] / 4
sub.loc[deficit, 'pnl'] = (sub.loc[deficit, 'imb_settlement_price'] - sub.loc[deficit, 'exec_ask']) * sub.loc[deficit, 'size'] / 4

sorted_pnl = sub['pnl'].sort_values(ascending=False)
total = sorted_pnl.sum()
print(f"  Total P&L: {total:+,.0f} EUR")
for n in [5, 10, 20, 50]:
    top_n = sorted_pnl.head(n).sum()
    print(f"  Top {n:3d} trades: {top_n:>+8,.0f} ({top_n/total*100:.0f}% of total)")
without_top10 = sorted_pnl.iloc[10:].sum()
print(f"  Without top 10: {without_top10:>+8,.0f} ({without_top10/total*100:.0f}%)")

# CHECK 7: Actual spread vs imbalance — why do they disagree?
print("\n--- CHECK 7: Spread target analysis ---")
te_all = test.copy()
print(f"  Test period spread: mean={te_all['spread_target'].mean():+.1f}, median={te_all['spread_target'].median():+.1f}")
print(f"  Test period imbalance: mean={te_all['target'].mean():+.1f}, median={te_all['target'].median():+.1f}")

# When imbalance is positive (surplus), is spread negative?
surplus_imb = te_all['target'] > 0
deficit_imb = te_all['target'] < 0
print(f"  When surplus (imb>0): mean spread={te_all.loc[surplus_imb, 'spread_target'].mean():+.1f}")
print(f"  When deficit (imb<0): mean spread={te_all.loc[deficit_imb, 'spread_target'].mean():+.1f}")
print(f"  So surplus -> IDM settles {'higher' if te_all.loc[surplus_imb, 'spread_target'].mean() > 0 else 'lower'} than imb")
print(f"  And deficit -> IDM settles {'higher' if te_all.loc[deficit_imb, 'spread_target'].mean() > 0 else 'lower'} than imb")
