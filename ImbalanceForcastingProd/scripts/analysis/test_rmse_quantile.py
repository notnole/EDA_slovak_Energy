"""Test RMSE loss + quantile confidence bands + asymmetric thresholds."""

import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

REPO_ROOT = Path(__file__).parent.parent.parent

data = load_all_data()
tml.TRAIN_END = '2026-01-31'
tml.TEST_START = '2026-02-01'

lead = 8
df, feature_cols = build_features(data, lead)
train = df[df.index <= tml.TRAIN_END].dropna(subset=['target', 'proxy_lag9'])
test = df[df.index >= tml.TEST_START].dropna(subset=['target', 'proxy_lag9'])

X_train, y_train = train[feature_cols].values, train['target'].values
X_test, y_test = test[feature_cols].values, test['target'].values

print(f"Train: {len(train)}, Test: {len(test)}")

base = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
            reg_lambda=1.0, n_estimators=600, verbose=-1)

# Train MAE median
m_mae = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **base)
m_mae.fit(X_train, y_train)
pred_mae = m_mae.predict(X_test)

# Train RMSE
m_rmse = lgb.LGBMRegressor(objective='regression', **base)
m_rmse.fit(X_train, y_train)
pred_rmse = m_rmse.predict(X_test)

# Quantile bands (MAE-based)
bands = {}
for q in [0.10, 0.25, 0.75, 0.90]:
    m = lgb.LGBMRegressor(objective='quantile', alpha=q, **base)
    m.fit(X_train, y_train)
    bands[f'q{int(q*100)}_mae'] = m.predict(X_test)

# Quantile bands (RMSE residual-based)
resid = y_train - m_rmse.predict(X_train)
for q in [0.10, 0.25, 0.75, 0.90]:
    m = lgb.LGBMRegressor(objective='quantile', alpha=q, **base)
    m.fit(X_train, resid)
    bands[f'q{int(q*100)}_rmse'] = pred_rmse + m.predict(X_test)

# ============================================================
print("\n" + "=" * 70)
print("PREDICTION COMPARISON: MAE vs RMSE")
print("=" * 70)
nz = np.abs(y_test) > 0.1

print(f"\n  {'':>10s}  {'mean':>7s}  {'std':>6s}  {'|pred|':>7s}  {'dir acc':>8s}  {'R2':>6s}  {'corr':>6s}")
for name, pred in [('MAE', pred_mae), ('RMSE', pred_rmse), ('Actual', y_test)]:
    r2 = 1 - np.sum((y_test - pred)**2) / np.sum((y_test - y_test.mean())**2) if name != 'Actual' else 1.0
    corr = np.corrcoef(y_test, pred)[0,1] if name != 'Actual' else 1.0
    da = (np.sign(pred) == np.sign(y_test))[nz].mean() if name != 'Actual' else 1.0
    print(f"  {name:>10s}  {pred.mean():>+7.2f}  {pred.std():>6.2f}  {np.abs(pred).mean():>7.2f}  {da:>7.1%}  {r2:>6.3f}  {corr:>6.3f}")

# ============================================================
# Load execution prices
mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                   parse_dates=['timestamp_hour'], index_col='timestamp_hour')
mkt = mkt[~mkt.index.duplicated(keep='last')]

ob_exec = pd.read_csv(Path(__file__).parent.parent / "data" / "orderbook_qh_features.csv",
                       parse_dates=['delivery_start'])
ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'spread']]
ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread']

t = test[['target']].copy()
t['pred_mae'] = pred_mae
t['pred_rmse'] = pred_rmse
for k, v in bands.items():
    t[k] = v
t['hour_ts'] = t.index.floor('h')
t = t.join(mkt[['imb_settlement_price']], on='hour_ts', how='left')
t = t.join(ob_120, how='left')
t = t.dropna(subset=['imb_settlement_price'])
t = t[t['imb_settlement_price'].abs() <= 5000]

def bt(t, pred_col, mask, label):
    sub = t[mask & t['exec_spread'].notna() & (t['exec_spread'] <= 10)].copy()
    if len(sub) < 30:
        print(f"  {label:<55s}  too few trades ({mask.sum()})")
        return
    sub['size'] = sub[pred_col].abs().clip(upper=5)
    surplus = sub[pred_col] > 0
    deficit = sub[pred_col] < 0
    sub['pnl'] = 0.0
    sub.loc[surplus, 'pnl'] = (sub.loc[surplus, 'exec_bid'] - sub.loc[surplus, 'imb_settlement_price']) * sub.loc[surplus, 'size'] / 4
    sub.loc[deficit, 'pnl'] = (sub.loc[deficit, 'imb_settlement_price'] - sub.loc[deficit, 'exec_ask']) * sub.loc[deficit, 'size'] / 4
    nd = sub.index.normalize().nunique()
    total = sub['pnl'].sum()
    wr = (sub['pnl'] > 0).mean()
    daily = sub.groupby(sub.index.date)['pnl'].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    print(f"  {label:<55s} {len(sub):4d}t, win={wr:.0%}, {total:>+7,.0f} ({total/nd:>+4.0f}/d) Sharpe={sharpe:.1f}")

print("\n" + "=" * 70)
print("TRADING BACKTEST (bid/ask at T-120min, spread<=10)")
print("=" * 70)

print("\n--- MAE model: magnitude threshold (baseline) ---")
bt(t, 'pred_mae', t['pred_mae'].abs() >= 3, 'MAE |pred|>=3')
bt(t, 'pred_mae', t['pred_mae'].abs() >= 5, 'MAE |pred|>=5')

print("\n--- MAE model: quantile confidence ---")
bt(t, 'pred_mae', (t['q10_mae'] > 0) | (t['q90_mae'] < 0), 'MAE 90% confident (Q10>0 or Q90<0)')
bt(t, 'pred_mae', (t['q25_mae'] > 0) | (t['q75_mae'] < 0), 'MAE 75% confident (Q25>0 or Q75<0)')

print("\n--- RMSE model: magnitude threshold ---")
bt(t, 'pred_rmse', t['pred_rmse'].abs() >= 3, 'RMSE |pred|>=3')
bt(t, 'pred_rmse', t['pred_rmse'].abs() >= 5, 'RMSE |pred|>=5')

print("\n--- RMSE model: quantile confidence ---")
bt(t, 'pred_rmse', (t['q10_rmse'] > 0) | (t['q90_rmse'] < 0), 'RMSE 90% confident (Q10>0 or Q90<0)')
bt(t, 'pred_rmse', (t['q25_rmse'] > 0) | (t['q75_rmse'] < 0), 'RMSE 75% confident (Q25>0 or Q75<0)')

print("\n--- Asymmetric MAE: tighter surplus, looser deficit ---")
bt(t, 'pred_mae',
   (t['q25_mae'] > 0) | (t['q75_mae'] < 0),
   'Sym: Q25>0 / Q75<0')
bt(t, 'pred_mae',
   ((t['q10_mae'] > 0) & (t['pred_mae'] > 0)) | ((t['q75_mae'] < 0) & (t['pred_mae'] < 0)),
   'Asym: surplus Q10>0, deficit Q75<0')
bt(t, 'pred_mae',
   ((t['q25_mae'] > 0) & (t['pred_mae'] > 0)) | ((t['pred_mae'] < -3)),
   'Asym: surplus Q25>0, deficit |pred|>3')
bt(t, 'pred_mae',
   ((t['pred_mae'] > 5)) | ((t['q75_mae'] < 0) & (t['pred_mae'] < 0)),
   'Asym: surplus |pred|>5, deficit Q75<0')

print("\n--- Asymmetric RMSE ---")
bt(t, 'pred_rmse',
   ((t['q10_rmse'] > 0) & (t['pred_rmse'] > 0)) | ((t['q75_rmse'] < 0) & (t['pred_rmse'] < 0)),
   'Asym RMSE: surplus Q10>0, deficit Q75<0')
bt(t, 'pred_rmse',
   ((t['q25_rmse'] > 0) & (t['pred_rmse'] > 0)) | ((t['pred_rmse'] < -3)),
   'Asym RMSE: surplus Q25>0, deficit |pred|>3')
bt(t, 'pred_rmse',
   ((t['pred_rmse'] > 5)) | ((t['q75_rmse'] < 0) & (t['pred_rmse'] < 0)),
   'Asym RMSE: surplus |pred|>5, deficit Q75<0')
bt(t, 'pred_rmse',
   ((t['pred_rmse'] > 5)) | ((t['pred_rmse'] < -3)),
   'Asym RMSE: surplus |pred|>5, deficit |pred|>3')
