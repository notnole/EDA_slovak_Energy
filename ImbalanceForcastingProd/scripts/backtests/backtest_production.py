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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPO_ROOT = BASE_DIR.parent

DB_CONN = dict(host='localhost', port=5432, dbname='DB_EMS',
               user='postgres', password='Kapitan1', connect_timeout=10)

# Calibration parameters
BASE_THRESHOLD = 3.0       # minimum |pred| to trade
BASE_SIZE_SCALE = 1.0      # base position size multiplier (1.0 = normal)
RECALIB_EVERY_N = 15       # recalibrate after this many trades per side
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
    """P&L-based calibrator.

    Tracks actual EUR profit per trade (not just win/loss) and adjusts:
    - Prediction threshold: raise when avg P&L/trade is negative, lower when profitable
    - Position size scale: scale down when losing, scale up when profitable
    - Independently for surplus and deficit sides

    The key insight: a side can have 55% win rate but still lose money if the
    average loss is bigger than the average win. P&L per trade captures this.
    """

    def __init__(self):
        self.surplus_pnls = deque(maxlen=60)
        self.deficit_pnls = deque(maxlen=60)
        self.surplus_threshold = BASE_THRESHOLD
        self.deficit_threshold = BASE_THRESHOLD
        self.surplus_size_scale = BASE_SIZE_SCALE
        self.deficit_size_scale = BASE_SIZE_SCALE
        self.n_since_calib_surplus = 0
        self.n_since_calib_deficit = 0

    def _calibrate_side(self, pnls, current_thresh, current_scale):
        """Calibrate one side based on recent P&L per trade."""
        if len(pnls) < 8:
            return current_thresh, current_scale

        recent = list(pnls)[-RECALIB_EVERY_N:] if len(pnls) >= RECALIB_EVERY_N else list(pnls)
        avg_pnl = np.mean(recent)
        total_pnl = np.sum(recent)

        # Threshold: based on whether the side is profitable
        if avg_pnl < -2:
            # Losing money — raise threshold significantly
            new_thresh = min(current_thresh + 1.5, 8.0)
        elif avg_pnl < 0:
            # Slightly negative — nudge up
            new_thresh = min(current_thresh + 0.5, 8.0)
        elif avg_pnl > 5:
            # Very profitable — can afford lower threshold
            new_thresh = max(current_thresh - 1.0, BASE_THRESHOLD)
        elif avg_pnl > 2:
            # Solidly profitable — ease back slightly
            new_thresh = max(current_thresh - 0.5, BASE_THRESHOLD)
        else:
            # Marginal — hold steady
            new_thresh = current_thresh

        # Size scaling: proportional to profitability
        if avg_pnl > 3:
            new_scale = min(current_scale + 0.2, 1.5)
        elif avg_pnl > 0:
            new_scale = min(current_scale + 0.1, 1.3)
        elif avg_pnl > -2:
            new_scale = max(current_scale - 0.1, 0.5)
        else:
            new_scale = max(current_scale - 0.3, 0.3)

        return new_thresh, new_scale

    def morning_reset(self, recent_trades):
        """Morning calibration from last N days of actual P&L."""
        for direction in ['surplus', 'deficit']:
            side = [t for t in recent_trades if t['direction'] == direction]
            if len(side) >= 10:
                avg_pnl = np.mean([t['pnl'] for t in side])
                # Set threshold based on recent avg P&L
                if avg_pnl < -1:
                    thresh = min(BASE_THRESHOLD + 2.0, 8.0)
                    scale = 0.5
                elif avg_pnl < 1:
                    thresh = BASE_THRESHOLD + 1.0
                    scale = 0.8
                elif avg_pnl > 5:
                    thresh = BASE_THRESHOLD
                    scale = 1.3
                else:
                    thresh = BASE_THRESHOLD
                    scale = 1.0
            else:
                thresh = BASE_THRESHOLD
                scale = BASE_SIZE_SCALE

            if direction == 'surplus':
                self.surplus_threshold = thresh
                self.surplus_size_scale = scale
            else:
                self.deficit_threshold = thresh
                self.deficit_size_scale = scale

        self.n_since_calib_surplus = 0
        self.n_since_calib_deficit = 0

    def record_trade(self, direction, pnl):
        """Record a trade's actual P&L and recalibrate if enough trades."""
        if direction == 'surplus':
            self.surplus_pnls.append(pnl)
            self.n_since_calib_surplus += 1
            if self.n_since_calib_surplus >= RECALIB_EVERY_N:
                self.surplus_threshold, self.surplus_size_scale = self._calibrate_side(
                    self.surplus_pnls, self.surplus_threshold, self.surplus_size_scale)
                self.n_since_calib_surplus = 0
        else:
            self.deficit_pnls.append(pnl)
            self.n_since_calib_deficit += 1
            if self.n_since_calib_deficit >= RECALIB_EVERY_N:
                self.deficit_threshold, self.deficit_size_scale = self._calibrate_side(
                    self.deficit_pnls, self.deficit_threshold, self.deficit_size_scale)
                self.n_since_calib_deficit = 0

    def should_trade(self, direction, pred_abs):
        """Check if prediction exceeds current calibrated threshold."""
        if direction == 'surplus':
            return pred_abs >= self.surplus_threshold
        else:
            return pred_abs >= self.deficit_threshold

    def get_size_scale(self, direction):
        """Get current position size multiplier."""
        if direction == 'surplus':
            return self.surplus_size_scale
        else:
            return self.deficit_size_scale


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

    # Load intraday order book features
    ob_path = DATA_DIR / "intraday_ob_features.csv"
    ob_features = None
    ob_cols = []
    if ob_path.exists():
        ob_features = pd.read_csv(ob_path, parse_dates=['delivery_start'])
        ob_features = ob_features.set_index('delivery_start')
        ob_features = ob_features[~ob_features.index.duplicated(keep='last')]
        ob_cols = [c for c in ob_features.columns if c.startswith('ob_')]
        print(f"[+] OB features: {len(ob_features)} rows, {len(ob_cols)} features")
    else:
        print("[-] No OB features found, running without")

    # DB connection for order book execution prices
    conn = psycopg2.connect(**DB_CONN)
    cur = conn.cursor()

    # Weekly boundaries (set RETRAIN_WEEKLY=False to train once and just calibrate)
    RETRAIN_WEEKLY = False
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

        print(f"\n--- Week {week_start.strftime('%Y-%m-%d')}: {'retrain' if RETRAIN_WEEKLY or wi == 0 else 'reuse model'}, data <= {train_end} ---")

        # Build features (always needed for the week's predictions)
        tml.TRAIN_END = train_end
        tml.TEST_START = week_start.strftime('%Y-%m-%d')

        if RETRAIN_WEEKLY or wi == 0:
            df, feature_cols = build_features(data, lead)

            # Join OB features
            if ob_features is not None:
                df = df.join(ob_features[ob_cols], how='left')
                all_feature_cols = feature_cols + ob_cols
            else:
                all_feature_cols = feature_cols

            train = df[df.index <= train_end]
            model = lgb.LGBMRegressor(
                objective='quantile', alpha=0.50, learning_rate=0.05,
                num_leaves=63, min_child_samples=50, subsample=0.8,
                colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                n_estimators=600, verbose=-1)
            model.fit(train[all_feature_cols].values, train['target'].values)
            # Cache df for subsequent weeks
            cached_df = df
            cached_feature_cols = all_feature_cols
        else:
            df = cached_df

        # Predict the week
        week_data = df[(df.index >= week_start.strftime('%Y-%m-%d')) &
                       (df.index < week_end.strftime('%Y-%m-%d'))]
        if len(week_data) == 0:
            continue
        week_data = week_data.copy()
        week_data['pred'] = model.predict(week_data[cached_feature_cols].values)

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

                # === MARKET TAKER ===
                if calib_taker.should_trade(direction, pred_abs):
                    if entry_snap is not None:
                        spread = entry_snap['ask'] - entry_snap['bid'] if pd.notna(entry_snap['ask']) and pd.notna(entry_snap['bid']) else np.nan
                        if pd.notna(spread) and spread <= MAX_SPREAD_TAKER:
                            # Size = prediction magnitude * calibrated scale, capped at 5
                            scale = calib_taker.get_size_scale(direction)
                            size = min(pred_abs * scale, 5.0)

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
                                         'thresh_d': calib_taker.deficit_threshold,
                                         'scale_s': calib_taker.surplus_size_scale,
                                         'scale_d': calib_taker.deficit_size_scale}
                                all_taker_trades.append(trade)
                                recent_taker_history.append(trade)
                                calib_taker.record_trade(direction, pnl)

                # === MARKET MAKER (limit order) ===
                if calib_maker.should_trade(direction, pred_abs):
                    if entry_snap is not None and len(period_ob) > 1:
                        entry_spread = entry_snap['ask'] - entry_snap['bid'] if pd.notna(entry_snap['ask']) and pd.notna(entry_snap['bid']) else np.nan
                        if pd.notna(entry_spread) and entry_spread <= MAX_SPREAD_MAKER:
                            scale = calib_maker.get_size_scale(direction)
                            size = min(pred_abs * scale, 5.0)

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
                                         'thresh_d': calib_maker.deficit_threshold,
                                         'scale_s': calib_maker.surplus_size_scale,
                                         'scale_d': calib_maker.deficit_size_scale}
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

        # Calibration evolution
        print(f"  Calibration evolution:")
        tf['week'] = tf.index.to_period('W')
        for week, grp in tf.groupby('week'):
            ts = grp['thresh_s'].iloc[-1]
            td = grp['thresh_d'].iloc[-1]
            ss = grp['scale_s'].iloc[-1] if 'scale_s' in grp.columns else 1.0
            sd = grp['scale_d'].iloc[-1] if 'scale_d' in grp.columns else 1.0
            avg_pnl = grp['pnl'].mean()
            print(f"    {week}: thresh S={ts:.1f}/D={td:.1f}, scale S={ss:.1f}/D={sd:.1f}, avg_pnl={avg_pnl:+.1f}")

        # Avg P&L per trade (the metric calibration is based on)
        print(f"  Avg P&L per trade: {tf['pnl'].mean():+.1f} EUR")
        print(f"  Avg size: {tf['size'].mean():.1f} MWh")

        # Save
        tf.to_csv(DATA_DIR / f"backtest_production_{name.split('(')[0].strip().lower().replace(' ', '_')}.csv")

    print(f"\n[+] Done!")


if __name__ == "__main__":
    main()
