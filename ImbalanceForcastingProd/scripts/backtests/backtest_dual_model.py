"""
Dual-Model Production Backtest
===============================

Two models working together:
  1. Imbalance model (MAE): predicts system imbalance direction
  2. Spread model (MAE): predicts IDM-to-settlement spread (the actual P&L)

Trading strategies tested:
  A: Imbalance model only (baseline)
  B: Spread model only
  C: Ensemble — trade when both agree on direction
  D: Spread model with imbalance confirmation (spread decides, imbalance filters)
  E: Imbalance model with spread sizing (imbalance decides, spread sets size)

Weekly retrain + P&L calibration on all variants.
Execution: bid/ask at T-120min, spread <= 10 filter.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path
from collections import deque

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml
from backtest_production import pull_ob_day, get_ob_at_time, Calibrator, DB_CONN

import psycopg2

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPO_ROOT = BASE_DIR.parent

MAX_SPREAD = 10
LEAD = 8


def main():
    print("=" * 70)
    print("DUAL-MODEL PRODUCTION BACKTEST")
    print("=" * 70)

    data = load_all_data()

    mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                       parse_dates=['timestamp_hour'], index_col='timestamp_hour')
    mkt = mkt[~mkt.index.duplicated(keep='last')]

    # Load order book execution prices
    ob_exec = pd.read_csv(DATA_DIR / "orderbook_qh_features.csv", parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'spread', 'mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']
    print(f"[+] Order book execution prices: {len(ob_120)} rows")

    base_params = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
                       subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                       reg_lambda=1.0, n_estimators=600, verbose=-1)

    weeks = pd.date_range('2026-02-02', '2026-03-30', freq='W-MON')
    weeks = list(weeks) + [pd.Timestamp('2026-03-30')]

    # Track trades for each strategy
    strategies = ['A_imb_only', 'B_spread_only', 'C_both_agree',
                  'D_spread_imb_filter', 'E_imb_spread_size']
    all_trades = {s: [] for s in strategies}

    for wi in range(len(weeks) - 1):
        week_start = weeks[wi]
        week_end = weeks[wi + 1]
        train_end = (week_start - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        print(f"\n--- Week {week_start.strftime('%Y-%m-%d')}: retrain <= {train_end} ---")

        tml.TRAIN_END = train_end
        tml.TEST_START = week_start.strftime('%Y-%m-%d')
        df, feature_cols = build_features(data, LEAD)

        # Join settlement + execution prices for spread target
        df['hour_ts'] = df.index.floor('h')
        df = df.join(mkt[['imb_settlement_price']], on='hour_ts', how='left')
        df = df.join(ob_120, how='left')
        df['spread_target'] = df['imb_settlement_price'] - df['exec_mid']

        train = df[df.index <= train_end].dropna(subset=['target', f'proxy_lag{LEAD+1}', 'spread_target'])
        train = train[train['imb_settlement_price'].abs() <= 5000]

        week_data = df[(df.index >= week_start.strftime('%Y-%m-%d')) &
                       (df.index < week_end.strftime('%Y-%m-%d'))]
        week_data = week_data.dropna(subset=[f'proxy_lag{LEAD+1}'])
        if len(week_data) == 0 or len(train) == 0:
            continue

        X_train = train[feature_cols].values

        # Train imbalance model
        m_imb = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **base_params)
        m_imb.fit(X_train, train['target'].values)

        # Train spread model
        m_spread = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **base_params)
        m_spread.fit(X_train, train['spread_target'].values)

        # Train spread quantile bands
        m_sp_q25 = lgb.LGBMRegressor(objective='quantile', alpha=0.25, **base_params)
        m_sp_q25.fit(X_train, train['spread_target'].values)
        m_sp_q75 = lgb.LGBMRegressor(objective='quantile', alpha=0.75, **base_params)
        m_sp_q75.fit(X_train, train['spread_target'].values)

        # Predict
        X_week = week_data[feature_cols].values
        week_data = week_data.copy()
        week_data['pred_imb'] = m_imb.predict(X_week)
        week_data['pred_spread'] = m_spread.predict(X_week)
        week_data['pred_sp_q25'] = m_sp_q25.predict(X_week)
        week_data['pred_sp_q75'] = m_sp_q75.predict(X_week)

        # Process each trade
        for idx, row in week_data.iterrows():
            if pd.isna(row.get('exec_spread')) or row['exec_spread'] > MAX_SPREAD:
                continue
            if pd.isna(row.get('imb_settlement_price')) or abs(row['imb_settlement_price']) > 5000:
                continue
            if pd.isna(row.get('exec_bid')) and pd.isna(row.get('exec_ask')):
                continue

            imb_price = row['imb_settlement_price']
            pred_imb = row['pred_imb']
            pred_sp = row['pred_spread']
            pred_sp_q25 = row['pred_sp_q25']
            pred_sp_q75 = row['pred_sp_q75']

            # Imbalance model: surplus = pred > 0 -> sell; deficit = pred < 0 -> buy
            imb_surplus = pred_imb > 0
            imb_deficit = pred_imb < 0
            # Spread model: positive = imb > IDM = buy profitable; negative = sell profitable
            sp_buy = pred_sp > 0
            sp_sell = pred_sp < 0

            def make_trade(direction, size, strategy):
                pnl = None
                if direction == 'surplus' and pd.notna(row['exec_bid']):
                    pnl = (row['exec_bid'] - imb_price) * size / 4
                elif direction == 'deficit' and pd.notna(row['exec_ask']):
                    pnl = (imb_price - row['exec_ask']) * size / 4
                if pnl is not None:
                    all_trades[strategy].append({
                        'datetime': idx, 'direction': direction,
                        'pred_imb': pred_imb, 'pred_spread': pred_sp,
                        'target': row['target'], 'size': size, 'pnl': pnl,
                    })

            # A: Imbalance only (asymmetric: surplus Q25-like via |pred|>=5, deficit |pred|>=3)
            if pred_imb >= 5:
                make_trade('surplus', min(abs(pred_imb), 5), 'A_imb_only')
            elif pred_imb <= -3:
                make_trade('deficit', min(abs(pred_imb), 5), 'A_imb_only')

            # B: Spread only (|pred_spread| >= 3)
            if pred_sp <= -3:
                make_trade('surplus', min(abs(pred_sp), 5), 'B_spread_only')
            elif pred_sp >= 3:
                make_trade('deficit', min(abs(pred_sp), 5), 'B_spread_only')

            # C: Both agree on direction
            if imb_surplus and sp_sell and abs(pred_imb) >= 3 and abs(pred_sp) >= 3:
                make_trade('surplus', min(abs(pred_sp), 5), 'C_both_agree')
            elif imb_deficit and sp_buy and abs(pred_imb) >= 3 and abs(pred_sp) >= 3:
                make_trade('deficit', min(abs(pred_sp), 5), 'C_both_agree')

            # D: Spread decides, imbalance must confirm direction
            if sp_sell and abs(pred_sp) >= 3 and pred_imb > 0:
                make_trade('surplus', min(abs(pred_sp), 5), 'D_spread_imb_filter')
            elif sp_buy and abs(pred_sp) >= 3 and pred_imb < 0:
                make_trade('deficit', min(abs(pred_sp), 5), 'D_spread_imb_filter')

            # E: Imbalance decides direction, spread sets size
            # Only trade when imbalance confident, size by spread magnitude
            if pred_imb >= 5 and pred_sp < 0:
                make_trade('surplus', min(abs(pred_sp), 5), 'E_imb_spread_size')
            elif pred_imb <= -3 and pred_sp > 0:
                make_trade('deficit', min(abs(pred_sp), 5), 'E_imb_spread_size')

        n_str = ", ".join(f"{s.split('_')[0]}:{len(all_trades[s])}" for s in strategies)
        print(f"  Trades: {n_str}")

    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS (Feb-Mar 2026, bid/ask execution, spread<=10)")
    print("=" * 70)

    labels = {
        'A_imb_only': 'A: Imbalance only (asym: S>=5, D>=3)',
        'B_spread_only': 'B: Spread only (|pred|>=3)',
        'C_both_agree': 'C: Both agree (|pred|>=3 each)',
        'D_spread_imb_filter': 'D: Spread decides + imb confirms dir',
        'E_imb_spread_size': 'E: Imb decides + spread sizes',
    }

    for strategy in strategies:
        trades = all_trades[strategy]
        if not trades:
            print(f"\n--- {labels[strategy]}: No trades ---")
            continue

        tf = pd.DataFrame(trades).set_index('datetime')
        nd = tf.index.normalize().nunique()
        total = tf['pnl'].sum()
        wr = (tf['pnl'] > 0).mean()
        daily = tf.groupby(tf.index.date)['pnl'].sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        prof = (daily > 0).sum()

        print(f"\n--- {labels[strategy]} ---")
        print(f"  Trades: {len(tf)} ({len(tf)/nd:.0f}/day), avg size: {tf['size'].mean():.1f} MWh")
        print(f"  Win: {wr:.1%}, P&L: {total:>+,.0f} EUR ({total/nd:>+,.0f}/day), Sharpe: {sharpe:.1f}")
        print(f"  Days: {prof}/{len(daily)} ({prof/len(daily):.0%}), worst: {daily.min():>+,.0f}")

        # Monthly
        tf['month'] = tf.index.to_period('M')
        for period, grp in tf.groupby('month'):
            mnd = grp.index.normalize().nunique()
            mpnl = grp['pnl'].sum()
            mwr = (grp['pnl'] > 0).mean()
            print(f"    {period}: {len(grp)} trades, win={mwr:.0%}, "
                  f"P&L={mpnl:>+,.0f} ({mpnl/mnd:>+,.0f}/day)")

        # Surplus vs deficit
        for d in ['surplus', 'deficit']:
            sub = tf[tf['direction'] == d]
            if len(sub) > 0:
                print(f"    {d}: {len(sub)} trades, win={(sub['pnl']>0).mean():.0%}, "
                      f"P&L={sub['pnl'].sum():>+,.0f}")


if __name__ == "__main__":
    main()
