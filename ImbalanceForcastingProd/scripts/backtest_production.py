"""
Production Simulation Backtest
==============================

Full simulation of a production trading system:

1. Weekly retrain: Every Monday, retrain Lead 8 on all data up to that point
2. Live calibration: After every N trades, recalibrate confidence thresholds
   based on recent surplus vs deficit performance
3. Morning calibration: At start of each day, set thresholds from last 7 days
4. Two execution modes tested in parallel:
   A: Market taker (hit bid/ask at T-120min, spread filter)
   B: Market maker (limit order at top of book, wait for fill until T-65min)

Calibration logic:
- Track rolling win rate for surplus and deficit separately
- If surplus win rate drops below 55% in recent N trades -> raise surplus threshold
- If deficit win rate drops below 55% -> raise deficit threshold
- Reset to base thresholds each morning using 7-day lookback
"""

import psycopg2
import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path
from collections import deque

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPO_ROOT = BASE_DIR.parent

DB_CONN = dict(host='localhost', port=5432, dbname='DB_EMS',
               user='postgres', password='Kapitan1', connect_timeout=10)

# Calibration parameters
BASE_THRESHOLD = 3.0       # minimum |pred| to trade
RECALIB_EVERY_N = 20       # recalibrate after this many trades per side
WIN_RATE_TARGET = 0.55     # if rolling win rate drops below this, raise threshold
THRESHOLD_STEP = 1.0       # how much to raise threshold per calibration
MAX_THRESHOLD = 8.0        # never go above this
LOOKBACK_DAYS = 7          # morning calibration lookback
MAX_SPREAD_TAKER = 10      # spread filter for market taker
MAX_SPREAD_MAKER = 10      # entry spread filter for limit orders


def pull_ob_day(cur, trade_date):
    """Pull all QH order book snapshots for one day (65-125 min window)."""
    cur.execute("""
        SELECT periodfrom, lastupdate,
               MAX(CASE WHEN tradetype = 'N' THEN price END) as bid,
               MAX(CASE WHEN tradetype = 'P' THEN price END) as ask
        FROM db_ems.vdt_isot_knihaobjednavok_best
        WHERE tradeday = %s AND deliverydur = 15 AND id_depth = 1
        GROUP BY periodfrom, lastupdate
    """, (trade_date,))
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['periodfrom', 'lastupdate', 'bid', 'ask'])
    df['bid'] = df['bid'].astype(float)
    df['ask'] = df['ask'].astype(float)
    df['lastupdate'] = pd.to_datetime(df['lastupdate'])
    trade_dt = pd.Timestamp(trade_date)
    df['delivery_start'] = trade_dt + pd.to_timedelta(df['periodfrom'] * 15, unit='m')
    df['minutes_before'] = (df['delivery_start'] - df['lastupdate']).dt.total_seconds() / 60
    df = df[(df['minutes_before'] >= 60) & (df['minutes_before'] <= 130)]
    return df


def get_ob_at_time(period_data, minutes_before_target):
    """Get latest snapshot at least minutes_before_target before delivery."""
    mask = period_data['minutes_before'] >= minutes_before_target
    if not mask.any():
        return None
    return period_data[mask].iloc[-1]


def simulate_limit_fill(period_data, direction, entry_snap):
    """Check if limit order would fill between entry and T-65min."""
    after_entry = period_data[period_data['lastupdate'] > entry_snap['lastupdate']]
    if direction == 'surplus':
        limit_price = entry_snap['ask']
        if pd.isna(limit_price):
            return False, np.nan
        max_bid = after_entry['bid'].max()
        filled = pd.notna(max_bid) and max_bid >= limit_price
        return filled, limit_price
    else:
        limit_price = entry_snap['bid']
        if pd.isna(limit_price):
            return False, np.nan
        min_ask = after_entry['ask'].min()
        filled = pd.notna(min_ask) and min_ask <= limit_price
        return filled, limit_price


class Calibrator:
    """Tracks recent trade outcomes and adjusts thresholds."""

    def __init__(self):
        self.surplus_trades = deque(maxlen=RECALIB_EVERY_N * 2)
        self.deficit_trades = deque(maxlen=RECALIB_EVERY_N * 2)
        self.surplus_threshold = BASE_THRESHOLD
        self.deficit_threshold = BASE_THRESHOLD
        self.n_since_calib_surplus = 0
        self.n_since_calib_deficit = 0

    def morning_reset(self, recent_trades):
        """Morning calibration from last 7 days of actual P&L."""
        if len(recent_trades) == 0:
            self.surplus_threshold = BASE_THRESHOLD
            self.deficit_threshold = BASE_THRESHOLD
            return

        for direction, attr in [('surplus', 'surplus_threshold'), ('deficit', 'deficit_threshold')]:
            side = [t for t in recent_trades if t['direction'] == direction]
            if len(side) >= 10:
                wr = np.mean([t['pnl'] > 0 for t in side])
                if wr < WIN_RATE_TARGET:
                    new_thresh = min(BASE_THRESHOLD + THRESHOLD_STEP * 2, MAX_THRESHOLD)
                elif wr > 0.65:
                    new_thresh = BASE_THRESHOLD
                else:
                    new_thresh = BASE_THRESHOLD + THRESHOLD_STEP
                setattr(self, attr, new_thresh)
            else:
                setattr(self, attr, BASE_THRESHOLD)

        self.n_since_calib_surplus = 0
        self.n_since_calib_deficit = 0

    def record_trade(self, direction, pnl):
        """Record a trade outcome and recalibrate if needed."""
        if direction == 'surplus':
            self.surplus_trades.append(pnl > 0)
            self.n_since_calib_surplus += 1
            if self.n_since_calib_surplus >= RECALIB_EVERY_N and len(self.surplus_trades) >= RECALIB_EVERY_N:
                recent = list(self.surplus_trades)[-RECALIB_EVERY_N:]
                wr = np.mean(recent)
                if wr < WIN_RATE_TARGET:
                    self.surplus_threshold = min(self.surplus_threshold + THRESHOLD_STEP, MAX_THRESHOLD)
                elif wr > 0.65 and self.surplus_threshold > BASE_THRESHOLD:
                    self.surplus_threshold = max(self.surplus_threshold - THRESHOLD_STEP, BASE_THRESHOLD)
                self.n_since_calib_surplus = 0
        else:
            self.deficit_trades.append(pnl > 0)
            self.n_since_calib_deficit += 1
            if self.n_since_calib_deficit >= RECALIB_EVERY_N and len(self.deficit_trades) >= RECALIB_EVERY_N:
                recent = list(self.deficit_trades)[-RECALIB_EVERY_N:]
                wr = np.mean(recent)
                if wr < WIN_RATE_TARGET:
                    self.deficit_threshold = min(self.deficit_threshold + THRESHOLD_STEP, MAX_THRESHOLD)
                elif wr > 0.65 and self.deficit_threshold > BASE_THRESHOLD:
                    self.deficit_threshold = max(self.deficit_threshold - THRESHOLD_STEP, BASE_THRESHOLD)
                self.n_since_calib_deficit = 0

    def should_trade(self, direction, pred_abs):
        """Check if prediction exceeds current calibrated threshold."""
        if direction == 'surplus':
            return pred_abs >= self.surplus_threshold
        else:
            return pred_abs >= self.deficit_threshold


def main():
    print("=" * 70)
    print("PRODUCTION SIMULATION BACKTEST")
    print("Weekly retrain + live calibration + dual execution")
    print("=" * 70)

    data = load_all_data()

    # Load market prices for settlement
    mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                       parse_dates=['timestamp_hour'], index_col='timestamp_hour')
    mkt = mkt[~mkt.index.duplicated(keep='last')]

    # DB connection for order book
    conn = psycopg2.connect(**DB_CONN)
    cur = conn.cursor()

    # Weekly boundaries
    weeks = pd.date_range('2026-02-02', '2026-03-30', freq='W-MON')
    weeks = list(weeks) + [pd.Timestamp('2026-03-30')]

    lead = 8
    calib_taker = Calibrator()
    calib_maker = Calibrator()
    all_taker_trades = []
    all_maker_trades = []
    recent_taker_history = deque(maxlen=500)
    recent_maker_history = deque(maxlen=500)

    for wi in range(len(weeks) - 1):
        week_start = weeks[wi]
        week_end = weeks[wi + 1]
        train_end = (week_start - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        print(f"\n--- Week {week_start.strftime('%Y-%m-%d')}: retrain on data <= {train_end} ---")

        # Retrain
        tml.TRAIN_END = train_end
        tml.TEST_START = week_start.strftime('%Y-%m-%d')
        df, feature_cols = build_features(data, lead)
        train = df[df.index <= train_end]
        model = lgb.LGBMRegressor(
            objective='quantile', alpha=0.50, learning_rate=0.05,
            num_leaves=63, min_child_samples=50, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            n_estimators=600, verbose=-1)
        model.fit(train[feature_cols].values, train['target'].values)

        # Predict the week
        week_data = df[(df.index >= week_start.strftime('%Y-%m-%d')) &
                       (df.index < week_end.strftime('%Y-%m-%d'))]
        if len(week_data) == 0:
            continue
        week_data = week_data.copy()
        week_data['pred'] = model.predict(week_data[feature_cols].values)

        # Process day by day
        for day in pd.date_range(week_start, week_end - pd.Timedelta(days=1), freq='D'):
            day_str = day.strftime('%Y-%m-%d')
            day_data = week_data[week_data.index.date == day.date()]
            if len(day_data) == 0:
                continue

            # Morning calibration
            recent_taker_list = [t for t in recent_taker_history
                                 if t['datetime'] >= day - pd.Timedelta(days=LOOKBACK_DAYS)]
            recent_maker_list = [t for t in recent_maker_history
                                 if t['datetime'] >= day - pd.Timedelta(days=LOOKBACK_DAYS)]
            calib_taker.morning_reset(recent_taker_list)
            calib_maker.morning_reset(recent_maker_list)

            # Pull order book for this day
            ob_day = pull_ob_day(cur, day_str)

            for idx, row in day_data.iterrows():
                pred_val = row['pred']
                pred_abs = abs(pred_val)
                direction = 'surplus' if pred_val > 0 else 'deficit'
                size = min(pred_abs, 5.0)
                period = idx.hour * 4 + idx.minute // 15

                # Get settlement price
                hour_ts = idx.floor('h')
                if hour_ts not in mkt.index:
                    continue
                imb_price = mkt.loc[hour_ts, 'imb_settlement_price']
                if pd.isna(imb_price) or abs(imb_price) > 5000:
                    continue

                # Get order book state
                period_ob = ob_day[ob_day['periodfrom'] == period].sort_values('lastupdate')
                entry_snap = get_ob_at_time(period_ob, 120) if len(period_ob) > 0 else None
                exit_snap = get_ob_at_time(period_ob, 65) if len(period_ob) > 0 else None

                # === MARKET TAKER ===
                if calib_taker.should_trade(direction, pred_abs):
                    if entry_snap is not None:
                        spread = entry_snap['ask'] - entry_snap['bid'] if pd.notna(entry_snap['ask']) and pd.notna(entry_snap['bid']) else np.nan
                        if pd.notna(spread) and spread <= MAX_SPREAD_TAKER:
                            if direction == 'surplus' and pd.notna(entry_snap['bid']):
                                pnl = (entry_snap['bid'] - imb_price) * size / 4
                            elif direction == 'deficit' and pd.notna(entry_snap['ask']):
                                pnl = (imb_price - entry_snap['ask']) * size / 4
                            else:
                                pnl = None

                            if pnl is not None:
                                trade = {'datetime': idx, 'direction': direction, 'pred': pred_val,
                                         'target': row['target'], 'size': size, 'pnl': pnl,
                                         'spread': spread,
                                         'thresh_s': calib_taker.surplus_threshold,
                                         'thresh_d': calib_taker.deficit_threshold}
                                all_taker_trades.append(trade)
                                recent_taker_history.append(trade)
                                calib_taker.record_trade(direction, pnl)

                # === MARKET MAKER (limit order) ===
                if calib_maker.should_trade(direction, pred_abs):
                    if entry_snap is not None and len(period_ob) > 1:
                        entry_spread = entry_snap['ask'] - entry_snap['bid'] if pd.notna(entry_snap['ask']) and pd.notna(entry_snap['bid']) else np.nan
                        if pd.notna(entry_spread) and entry_spread <= MAX_SPREAD_MAKER:
                            filled, limit_price = simulate_limit_fill(period_ob, direction, entry_snap)
                            if filled and pd.notna(limit_price):
                                if direction == 'surplus':
                                    pnl = (limit_price - imb_price) * size / 4
                                else:
                                    pnl = (imb_price - limit_price) * size / 4

                                trade = {'datetime': idx, 'direction': direction, 'pred': pred_val,
                                         'target': row['target'], 'size': size, 'pnl': pnl,
                                         'spread': entry_spread, 'limit_price': limit_price,
                                         'thresh_s': calib_maker.surplus_threshold,
                                         'thresh_d': calib_maker.deficit_threshold}
                                all_maker_trades.append(trade)
                                recent_maker_history.append(trade)
                                calib_maker.record_trade(direction, pnl)

        n_t = len(all_taker_trades)
        n_m = len(all_maker_trades)
        print(f"  Taker: {n_t} trades, Maker: {n_m} trades")

    conn.close()

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 70)
    print("PRODUCTION SIMULATION RESULTS (Feb-Mar 2026)")
    print("=" * 70)

    for name, trades_list in [("MARKET TAKER (hit bid/ask)", all_taker_trades),
                               ("MARKET MAKER (limit orders)", all_maker_trades)]:
        if not trades_list:
            print(f"\n--- {name}: No trades ---")
            continue

        tf = pd.DataFrame(trades_list)
        tf['datetime'] = pd.to_datetime(tf['datetime'])
        tf = tf.set_index('datetime')
        n_days = tf.index.normalize().nunique()

        total = tf['pnl'].sum()
        wr = (tf['pnl'] > 0).mean()
        daily = tf.groupby(tf.index.date)['pnl'].sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        prof = (daily > 0).sum()

        print(f"\n--- {name} ---")
        print(f"  Trades: {len(tf)} ({len(tf)/n_days:.0f}/day)")
        print(f"  Win rate: {wr:.1%}")
        print(f"  P&L: {total:>+,.0f} EUR ({total/n_days:>+,.0f}/day), Sharpe: {sharpe:.1f}")
        print(f"  Days: {prof}/{len(daily)} ({prof/len(daily):.0%}), worst: {daily.min():>+,.0f}")

        # Monthly
        tf['month'] = tf.index.to_period('M')
        for period, grp in tf.groupby('month'):
            nd = grp.index.normalize().nunique()
            mpnl = grp['pnl'].sum()
            mwr = (grp['pnl'] > 0).mean()
            print(f"    {period}: {len(grp)} trades, win={mwr:.0%}, P&L={mpnl:>+,.0f} ({mpnl/nd:>+,.0f}/day)")

        # Surplus vs deficit
        for d in ['surplus', 'deficit']:
            sub = tf[tf['direction'] == d]
            if len(sub) > 0:
                print(f"    {d}: {len(sub)} trades, win={(sub['pnl']>0).mean():.0%}, P&L={sub['pnl'].sum():>+,.0f}")

        # Threshold evolution
        print(f"  Threshold evolution:")
        tf['week'] = tf.index.to_period('W')
        for week, grp in tf.groupby('week'):
            ts = grp['thresh_s'].iloc[-1]
            td = grp['thresh_d'].iloc[-1]
            print(f"    {week}: surplus_thresh={ts:.1f}, deficit_thresh={td:.1f}")

        # Save
        tf.to_csv(DATA_DIR / f"backtest_production_{name.split('(')[0].strip().lower().replace(' ', '_')}.csv")

    print(f"\n[+] Done!")


if __name__ == "__main__":
    main()
