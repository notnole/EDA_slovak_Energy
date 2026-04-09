"""Plot trading backtest results for Feb-Mar 2026 weekly retrain strategy."""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

print("[*] Loading data and running weekly retrain...")
data = load_all_data()

weeks = pd.date_range('2026-02-02', '2026-03-30', freq='W-MON')
weeks = list(weeks) + [pd.Timestamp('2026-03-30')]

lead = 4
all_preds = []

for i in range(len(weeks) - 1):
    week_start = weeks[i].strftime('%Y-%m-%d')
    week_end = weeks[i + 1].strftime('%Y-%m-%d')
    train_end = (weeks[i] - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    tml.TRAIN_END = train_end
    tml.TEST_START = week_start
    df, feature_cols = build_features(data, lead)
    train = df[df.index <= train_end]
    test = df[(df.index >= week_start) & (df.index < week_end)]
    if len(test) == 0:
        continue
    model = lgb.LGBMRegressor(
        objective='quantile', alpha=0.50, learning_rate=0.05,
        num_leaves=63, min_child_samples=50, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        n_estimators=600, verbose=-1)
    model.fit(train[feature_cols].values, train['target'].values)
    week_df = test[['target']].copy()
    week_df['pred_median'] = model.predict(test[feature_cols].values)
    all_preds.append(week_df)
    n = len(test)
    nonzero = week_df['target'].abs() > 0.1
    da = ((np.sign(week_df['pred_median']) == np.sign(week_df['target']))[nonzero]).mean()
    print(f"  Week {week_start}: train<={train_end}, test={n}, dir={da:.0%}")

pred_all = pd.concat(all_preds)
print(f"[+] {len(pred_all)} predictions")

# Market prices
REPO = Path(__file__).parent.parent.parent
mkt = pd.read_csv(REPO / 'MarketPriceGap' / 'data' / 'processed' / 'hourly_market_prices.csv',
                   parse_dates=['timestamp_hour'], index_col='timestamp_hour')
mkt = mkt[~mkt.index.duplicated(keep='last')]

pred_all['hour_ts'] = pred_all.index.floor('h')
m = pred_all.join(mkt[['idm_vwap', 'imb_settlement_price']], on='hour_ts', how='left')
m = m.dropna(subset=['idm_vwap', 'imb_settlement_price'])
m = m[m['imb_settlement_price'].abs() <= 5000]

# Trades
mask = m['pred_median'].abs() > 2
t = m[mask].copy()
t['size'] = t['pred_median'].abs().clip(upper=5)
surplus = t['pred_median'] > 0
deficit = t['pred_median'] < 0
t['pnl'] = 0.0
t.loc[surplus, 'pnl'] = (t.loc[surplus, 'idm_vwap'] - t.loc[surplus, 'imb_settlement_price']) * t.loc[surplus, 'size'] / 4
t.loc[deficit, 'pnl'] = (t.loc[deficit, 'imb_settlement_price'] - t.loc[deficit, 'idm_vwap']) * t.loc[deficit, 'size'] / 4
t['cum_pnl'] = t['pnl'].cumsum()
t['direction_correct'] = np.sign(t['pred_median']) == np.sign(t['target'])

# Daily stats
daily = t.groupby(t.index.date).agg(
    pnl=('pnl', 'sum'),
    n_trades=('pnl', 'count'),
    win_rate=('pnl', lambda x: (x > 0).mean()),
    dir_acc=('direction_correct', 'mean'),
    avg_size=('size', 'mean'),
).reset_index()
daily.columns = ['date', 'pnl', 'n_trades', 'win_rate', 'dir_acc', 'avg_size']
daily['date'] = pd.to_datetime(daily['date'])
daily['cum_pnl'] = daily['pnl'].cumsum()
daily['rolling_sharpe'] = daily['pnl'].rolling(10, min_periods=5).apply(
    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0)

# =============================================
# PLOT
# =============================================
fig = plt.figure(figsize=(18, 22))
gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)

# 1. Cumulative P&L
ax1 = fig.add_subplot(gs[0, :])
ax1.fill_between(daily['date'], 0, daily['cum_pnl'], alpha=0.3, color='#2ecc71')
ax1.plot(daily['date'], daily['cum_pnl'], color='#27ae60', linewidth=2)
ax1.axhline(0, color='gray', linewidth=0.5)
ax1.set_ylabel('Cumulative P&L (EUR)')
total_pnl = daily['cum_pnl'].iloc[-1]
ax1.set_title(f'Cumulative P&L - Weekly Retrain, Lead 4 (1h ahead), Feb-Mar 2026 OOS\n'
              f'Total: +{total_pnl:,.0f} EUR | {total_pnl/len(daily):,.0f}/day | '
              f'Sharpe {daily["pnl"].mean()/daily["pnl"].std()*np.sqrt(252):.1f}',
              fontsize=13, fontweight='bold')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))

# 2. Daily P&L bars
ax2 = fig.add_subplot(gs[1, :])
colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in daily['pnl']]
ax2.bar(daily['date'], daily['pnl'], color=colors, alpha=0.8, width=0.8)
ax2.axhline(0, color='gray', linewidth=0.5)
ax2.set_ylabel('Daily P&L (EUR)')
ax2.set_title('Daily P&L')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
prof = (daily['pnl'] > 0).sum()
ax2.text(0.02, 0.95, f'{prof}/{len(daily)} days profitable ({prof/len(daily):.0%})',
         transform=ax2.transAxes, fontsize=10, va='top')

# 3. Direction accuracy (rolling 7-day)
ax3 = fig.add_subplot(gs[2, 0])
rolling_dir = daily['dir_acc'].rolling(7, min_periods=3).mean()
ax3.plot(daily['date'], rolling_dir * 100, color='#3498db', linewidth=2)
ax3.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% (coin flip)')
ax3.axhline(65, color='orange', linestyle='--', alpha=0.5, label='65% target')
ax3.set_ylabel('Direction Accuracy (%)')
ax3.set_title('7-Day Rolling Direction Accuracy')
ax3.legend(fontsize=8)
ax3.set_ylim(40, 90)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# 4. Win rate (rolling 7-day)
ax4 = fig.add_subplot(gs[2, 1])
rolling_wr = daily['win_rate'].rolling(7, min_periods=3).mean()
ax4.plot(daily['date'], rolling_wr * 100, color='#9b59b6', linewidth=2)
ax4.axhline(50, color='red', linestyle='--', alpha=0.5)
ax4.set_ylabel('Win Rate (%)')
ax4.set_title('7-Day Rolling Win Rate')
ax4.set_ylim(40, 90)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# 5. Prediction distribution
ax5 = fig.add_subplot(gs[3, 0])
bins = np.arange(-20, 21, 1)
ax5.hist(t['pred_median'], bins=bins, alpha=0.7, color='#3498db', edgecolor='white', linewidth=0.5)
ax5.axvline(0, color='gray', linewidth=1)
ax5.axvline(2, color='orange', linestyle='--', alpha=0.7, label='|threshold| = 2')
ax5.axvline(-2, color='orange', linestyle='--', alpha=0.7)
ax5.set_xlabel('Predicted Imbalance (MWh)')
ax5.set_ylabel('Count')
ax5.set_title('Prediction Distribution (traded periods)')
ax5.legend(fontsize=8)

# 6. Win rate by confidence bucket
ax6 = fig.add_subplot(gs[3, 1])
t['pred_bin'] = pd.cut(t['pred_median'].abs(), bins=[0, 2, 5, 10, 50],
                        labels=['0-2', '2-5', '5-10', '10+'])
bin_stats = t.groupby('pred_bin', observed=True).agg(
    wr=('pnl', lambda x: (x > 0).mean()),
    n=('pnl', 'count'),
    da=('direction_correct', 'mean')).reset_index()
x = range(len(bin_stats))
ax6.bar(x, bin_stats['wr'] * 100, alpha=0.7, color='#2ecc71', label='Win Rate')
ax6.bar(x, bin_stats['da'] * 100, alpha=0.3, color='#3498db', label='Dir Accuracy')
ax6.axhline(50, color='red', linestyle='--', alpha=0.5)
ax6.set_xticks(x)
ax6.set_xticklabels(bin_stats['pred_bin'])
ax6.set_xlabel('|Predicted Imbalance| (MWh)')
ax6.set_ylabel('%')
ax6.set_title('Accuracy by Prediction Magnitude')
ax6.legend(fontsize=8)
for i, row in bin_stats.iterrows():
    ax6.text(i, row['wr'] * 100 + 1.5, f'n={row["n"]:,}', ha='center', fontsize=9)

# 7. Surplus vs Deficit cumulative P&L
ax7 = fig.add_subplot(gs[4, 0])
for ttype, color, label in [(surplus, '#e67e22', 'Surplus (Sell IDM)'),
                             (deficit, '#2980b9', 'Deficit (Buy IDM)')]:
    sub = t[ttype]
    daily_type = sub.groupby(sub.index.date)['pnl'].sum().cumsum()
    ax7.plot(pd.to_datetime(daily_type.index), daily_type.values,
             label=f'{label}: +{daily_type.iloc[-1]:,.0f} EUR', linewidth=2, color=color)
ax7.axhline(0, color='gray', linewidth=0.5)
ax7.set_ylabel('Cumulative P&L (EUR)')
ax7.set_title('P&L by Trade Direction')
ax7.legend(fontsize=9)
ax7.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# 8. Rolling Sharpe
ax8 = fig.add_subplot(gs[4, 1])
ax8.plot(daily['date'], daily['rolling_sharpe'], color='#e74c3c', linewidth=2)
ax8.axhline(0, color='gray', linewidth=0.5)
ax8.axhline(10, color='green', linestyle='--', alpha=0.5, label='Sharpe = 10')
ax8.set_ylabel('Annualized Sharpe')
ax8.set_title('10-Day Rolling Sharpe Ratio')
ax8.legend(fontsize=8)
ax8.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

OUT = Path(__file__).parent.parent / 'plots' / 'trading_backtest_feb_mar_2026.png'
plt.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close()
print(f'[+] Saved: {OUT}')
