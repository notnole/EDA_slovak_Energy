"""Test predicting the IDM-to-settlement spread directly instead of imbalance."""

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

# Load execution prices and settlement
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

# New target: spread = imb_settlement_price - IDM_mid
# Positive = deficit profitable (buy IDM cheap, imbalance settles higher)
# Negative = surplus profitable (sell IDM high, imbalance settles lower)
df['spread_target'] = df['imb_settlement_price'] - df['exec_mid']

valid = df.dropna(subset=['target', 'proxy_lag9', 'spread_target', 'exec_mid'])
valid = valid[valid['imb_settlement_price'].abs() <= 5000]

train = valid[valid.index <= tml.TRAIN_END]
test = valid[valid.index >= tml.TEST_START]

X_train = train[feature_cols].values
X_test = test[feature_cols].values

print(f"Train: {len(train)}, Test: {len(test)}")
print(f"Spread target: mean={train['spread_target'].mean():+.1f}, std={train['spread_target'].std():.0f}, "
      f"median={train['spread_target'].median():+.1f}")

base = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
            reg_lambda=1.0, n_estimators=600, verbose=-1)

# Model A: predict imbalance (original)
m_imb = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **base)
m_imb.fit(X_train, train['target'].values)
pred_imb = m_imb.predict(X_test)

# Model B: predict spread (RMSE)
m_sp_rmse = lgb.LGBMRegressor(objective='regression', **base)
m_sp_rmse.fit(X_train, train['spread_target'].values)
pred_sp_rmse = m_sp_rmse.predict(X_test)

# Model C: predict spread (MAE)
m_sp_mae = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **base)
m_sp_mae.fit(X_train, train['spread_target'].values)
pred_sp_mae = m_sp_mae.predict(X_test)

# Quantile bands for spread
sp_bands = {}
for q in [0.10, 0.25, 0.75, 0.90]:
    m = lgb.LGBMRegressor(objective='quantile', alpha=q, **base)
    m.fit(X_train, train['spread_target'].values)
    sp_bands[f'q{int(q*100)}'] = m.predict(X_test)

# ============================================================
y_imb = test['target'].values
y_sp = test['spread_target'].values
nz_imb = np.abs(y_imb) > 0.1
nz_sp = np.abs(y_sp) > 0.1

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for name, pred, actual, nz in [
    ('Imbalance (MAE)', pred_imb, y_imb, nz_imb),
    ('Spread (RMSE)', pred_sp_rmse, y_sp, nz_sp),
    ('Spread (MAE)', pred_sp_mae, y_sp, nz_sp),
]:
    r2 = 1 - np.sum((actual - pred)**2) / np.sum((actual - actual.mean())**2)
    corr = np.corrcoef(actual, pred)[0, 1]
    da = (np.sign(pred) == np.sign(actual))[nz].mean()
    print(f"  {name:<20s}: R2={r2:.3f}, corr={corr:.3f}, dir_acc={da:.1%}, "
          f"pred_std={pred.std():.1f}, actual_std={actual.std():.0f}")

# How well does imbalance direction match spread direction?
both_nz = nz_imb & nz_sp
agree = (np.sign(y_imb) == np.sign(y_sp))[both_nz].mean()
print(f"\n  Imbalance direction = Spread direction: {agree:.1%}")
print(f"  Imbalance-Spread corr: {np.corrcoef(y_imb, y_sp)[0,1]:.3f}")

# ============================================================
# Trading backtest
# ============================================================
t = test[['target', 'spread_target', 'exec_bid', 'exec_ask', 'exec_spread',
          'exec_mid', 'imb_settlement_price']].copy()
t['pred_imb'] = pred_imb
t['pred_sp_rmse'] = pred_sp_rmse
t['pred_sp_mae'] = pred_sp_mae
for k, v in sp_bands.items():
    t[k] = v
t = t[t['exec_spread'].notna() & (t['exec_spread'] <= 10)]


def bt(sub_t, label, surplus_mask, deficit_mask, size_col):
    sub = sub_t[surplus_mask | deficit_mask].copy()
    if len(sub) < 30:
        print(f"  {label:<55s}  ({(surplus_mask | deficit_mask).sum()} periods)")
        return
    sub['size'] = sub[size_col].abs().clip(upper=5)
    s = surplus_mask.reindex(sub.index, fill_value=False)
    d = deficit_mask.reindex(sub.index, fill_value=False)
    sub['pnl'] = 0.0
    sub.loc[s, 'pnl'] = (sub.loc[s, 'exec_bid'] - sub.loc[s, 'imb_settlement_price']) * sub.loc[s, 'size'] / 4
    sub.loc[d, 'pnl'] = (sub.loc[d, 'imb_settlement_price'] - sub.loc[d, 'exec_ask']) * sub.loc[d, 'size'] / 4
    nd = sub.index.normalize().nunique()
    total = sub['pnl'].sum()
    wr = (sub['pnl'] > 0).mean()
    daily = sub.groupby(sub.index.date)['pnl'].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    print(f"  {label:<55s} {len(sub):4d}t, win={wr:.0%}, {total:>+7,.0f} ({total/nd:>+4.0f}/d) Sharpe={sharpe:.1f}")


print("\n" + "=" * 70)
print("TRADING (bid/ask at T-120min, spread<=10)")
print("=" * 70)

# Baseline imbalance model
print("\n--- A: Imbalance model (current best) ---")
bt(t, "Imb |pred|>=3",
   t['pred_imb'] >= 3, t['pred_imb'] <= -3, 'pred_imb')
bt(t, "Imb asym: surplus>=5, deficit>=3",
   t['pred_imb'] >= 5, t['pred_imb'] <= -3, 'pred_imb')

# Spread RMSE model
# surplus = sell IDM = expect spread < 0 (imb < IDM)
# deficit = buy IDM = expect spread > 0 (imb > IDM)
print("\n--- B: Spread RMSE model (predict EUR spread directly) ---")
bt(t, "Spread RMSE |pred|>=3",
   t['pred_sp_rmse'] <= -3, t['pred_sp_rmse'] >= 3, 'pred_sp_rmse')
bt(t, "Spread RMSE |pred|>=5",
   t['pred_sp_rmse'] <= -5, t['pred_sp_rmse'] >= 5, 'pred_sp_rmse')
bt(t, "Spread RMSE |pred|>=10",
   t['pred_sp_rmse'] <= -10, t['pred_sp_rmse'] >= 10, 'pred_sp_rmse')

# Spread MAE model
print("\n--- C: Spread MAE model ---")
bt(t, "Spread MAE |pred|>=3",
   t['pred_sp_mae'] <= -3, t['pred_sp_mae'] >= 3, 'pred_sp_mae')
bt(t, "Spread MAE |pred|>=5",
   t['pred_sp_mae'] <= -5, t['pred_sp_mae'] >= 5, 'pred_sp_mae')

# Spread with quantile confidence
print("\n--- D: Spread quantile confidence ---")
bt(t, "Spread Q75<0 (sell) or Q25>0 (buy)",
   t['q75'] < 0, t['q25'] > 0, 'pred_sp_mae')
bt(t, "Spread Q90<0 (sell) or Q10>0 (buy)",
   t['q90'] < 0, t['q10'] > 0, 'pred_sp_mae')

# Ensemble: trade when BOTH imbalance AND spread models agree
print("\n--- E: Ensemble (both models agree) ---")
bt(t, "Both agree: imb>=3 AND spread_rmse same dir",
   (t['pred_imb'] >= 3) & (t['pred_sp_rmse'] < 0),
   (t['pred_imb'] <= -3) & (t['pred_sp_rmse'] > 0),
   'pred_imb')
bt(t, "Both agree: imb>=3 AND spread_rmse>=3",
   (t['pred_imb'] >= 3) & (t['pred_sp_rmse'] <= -3),
   (t['pred_imb'] <= -3) & (t['pred_sp_rmse'] >= 3),
   'pred_imb')
bt(t, "Both agree: imb asym AND spread confirms",
   (t['pred_imb'] >= 5) & (t['pred_sp_rmse'] < 0),
   (t['pred_imb'] <= -3) & (t['pred_sp_rmse'] > 0),
   'pred_imb')
bt(t, "Both agree: imb>=3 AND spread Q25>0/Q75<0",
   (t['pred_imb'] >= 3) & (t['q75'] < 0),
   (t['pred_imb'] <= -3) & (t['q25'] > 0),
   'pred_imb')
