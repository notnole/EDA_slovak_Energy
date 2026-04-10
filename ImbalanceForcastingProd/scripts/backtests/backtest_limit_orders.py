"""
Limit Order Backtest — Place at Top of Book, Wait for Fill
==========================================================

Strategy:
  1. At T-120min: model predicts imbalance direction
  2. If surplus -> place SELL limit at current best ask (top of sell side)
     If deficit -> place BUY limit at current best bid (top of buy side)
  3. Monitor order book from T-120min to T-65min (5min before gate closure)
  4. Fill: bid ever >= our ask (sell filled) or ask ever <= our bid (buy filled)
  5. If filled: P&L = limit_price - imb_settlement_price (sells), reversed for buys
  6. If not filled: no trade (also test fallback to market order at T-65min)

Data pull: one bulk query per day from DB_EMS, then process in Python.
Uses 15-min products (deliverydur=15) and Lead 8 (120min) predictions.
"""

import psycopg2
import pandas as pd
import numpy as np
from pathlib import Path
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPO_ROOT = BASE_DIR.parent

DB_CONN = dict(host='localhost', port=5432, dbname='DB_EMS',
               user='postgres', password='Kapitan1', connect_timeout=10)

# Execution window: place order at T-120min, monitor until T-65min
ENTRY_MINUTES = 120   # when we place the limit order
EXIT_MINUTES = 65     # when we cancel if not filled (5min before gate)


def pull_day(cur, trade_date):
    """Pull all QH order book snapshots for one day in the execution window.

    Returns DataFrame with columns: periodfrom, lastupdate, bid, ask,
    and minutes_before delivery.
    """
    cur.execute("""
        SELECT periodfrom, lastupdate,
               MAX(CASE WHEN tradetype = 'N' THEN price END) as bid,
               MAX(CASE WHEN tradetype = 'P' THEN price END) as ask
        FROM db_ems.vdt_isot_knihaobjednavok_best
        WHERE tradeday = %s
          AND deliverydur = 15
          AND id_depth = 1
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

    # Keep only snapshots in our execution window (65-125 min before delivery)
    # Slightly wider than ENTRY to catch the entry snapshot
    df = df[(df['minutes_before'] >= EXIT_MINUTES) & (df['minutes_before'] <= ENTRY_MINUTES + 5)]

    return df


def simulate_period(period_data, direction):
    """Simulate limit order for one delivery period.

    Args:
        period_data: DataFrame of order book snapshots, sorted by lastupdate
        direction: 'surplus' (sell) or 'deficit' (buy)

    Returns:
        dict with fill info, or None if can't place order
    """
    if len(period_data) == 0:
        return None

    # Entry: latest snapshot at or before T-ENTRY_MINUTES
    entry_mask = period_data['minutes_before'] >= ENTRY_MINUTES
    if not entry_mask.any():
        return None
    entry_snap = period_data[entry_mask].iloc[-1]

    if direction == 'surplus':
        # Place SELL limit at current best ask
        limit_price = entry_snap['ask']
        if pd.isna(limit_price):
            # No ask at entry — use mid or skip
            if pd.notna(entry_snap['bid']):
                limit_price = entry_snap['bid'] + 1.0  # bid + minimum tick
            else:
                return None

        # Check: did bid ever reach our limit price after entry?
        after_entry = period_data[period_data['lastupdate'] > entry_snap['lastupdate']]
        max_bid = after_entry['bid'].max()
        filled = pd.notna(max_bid) and max_bid >= limit_price

        # Fallback: market order at T-65min (hit best bid)
        exit_mask = period_data['minutes_before'] <= EXIT_MINUTES + 5
        exit_snap = period_data[exit_mask].iloc[0] if exit_mask.any() else None
        fallback_price = exit_snap['bid'] if exit_snap is not None and pd.notna(exit_snap['bid']) else np.nan

    else:  # deficit -> buy
        # Place BUY limit at current best bid
        limit_price = entry_snap['bid']
        if pd.isna(limit_price):
            if pd.notna(entry_snap['ask']):
                limit_price = entry_snap['ask'] - 1.0
            else:
                return None

        # Check: did ask ever drop to our limit price after entry?
        after_entry = period_data[period_data['lastupdate'] > entry_snap['lastupdate']]
        min_ask = after_entry['ask'].min()
        filled = pd.notna(min_ask) and min_ask <= limit_price

        exit_mask = period_data['minutes_before'] <= EXIT_MINUTES + 5
        exit_snap = period_data[exit_mask].iloc[0] if exit_mask.any() else None
        fallback_price = exit_snap['ask'] if exit_snap is not None and pd.notna(exit_snap['ask']) else np.nan

    return {
        'limit_price': limit_price,
        'filled': filled,
        'entry_bid': entry_snap['bid'],
        'entry_ask': entry_snap['ask'],
        'entry_spread': entry_snap['ask'] - entry_snap['bid'] if pd.notna(entry_snap['ask']) and pd.notna(entry_snap['bid']) else np.nan,
        'n_snapshots': len(after_entry),
        'fallback_price': fallback_price,
    }


def main():
    print("=" * 70)
    print("LIMIT ORDER BACKTEST")
    print(f"Entry: T-{ENTRY_MINUTES}min, Monitor until T-{EXIT_MINUTES}min")
    print("=" * 70)

    # Load predictions (Lead 8 = 120min)
    pred = pd.read_csv(DATA_DIR / "predictions_lead8.csv",
                       parse_dates=['datetime'], index_col='datetime')
    pred = pred[pred['pred_median'].abs() >= 3]
    pred['direction'] = np.where(pred['pred_median'] > 0, 'surplus', 'deficit')
    pred['size'] = pred['pred_median'].abs().clip(upper=5)
    print(f"[+] Predictions (Lead 8, |pred|>=3): {len(pred)}")

    # Load imbalance settlement prices
    mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                       parse_dates=['timestamp_hour'], index_col='timestamp_hour')
    mkt = mkt[~mkt.index.duplicated(keep='last')]

    # Pull order book data day by day
    conn = psycopg2.connect(**DB_CONN)
    cur = conn.cursor()

    test_dates = pd.date_range('2026-02-01', '2026-03-29', freq='D')
    all_results = []

    for i, date in enumerate(test_dates):
        date_str = date.strftime('%Y-%m-%d')
        ob_day = pull_day(cur, date_str)

        if len(ob_day) == 0:
            continue

        # Get predictions for this day
        day_pred = pred[pred.index.date == date.date()]

        for idx, row in day_pred.iterrows():
            period = idx.hour * 4 + idx.minute // 15
            period_data = ob_day[ob_day['periodfrom'] == period].sort_values('lastupdate')

            result = simulate_period(period_data, row['direction'])
            if result is None:
                continue

            # Get settlement price
            hour_ts = idx.floor('h')
            if hour_ts in mkt.index:
                imb_price = mkt.loc[hour_ts, 'imb_settlement_price']
            else:
                continue

            if pd.isna(imb_price) or abs(imb_price) > 5000:
                continue

            result['datetime'] = idx
            result['target'] = row['target']
            result['pred_median'] = row['pred_median']
            result['direction'] = row['direction']
            result['size'] = row['size']
            result['imb_settlement_price'] = imb_price

            # P&L if filled at limit price
            if row['direction'] == 'surplus':
                result['pnl_limit'] = (result['limit_price'] - imb_price) * row['size'] / 4
                result['pnl_fallback'] = (result['fallback_price'] - imb_price) * row['size'] / 4 if pd.notna(result['fallback_price']) else np.nan
            else:
                result['pnl_limit'] = (imb_price - result['limit_price']) * row['size'] / 4
                result['pnl_fallback'] = (imb_price - result['fallback_price']) * row['size'] / 4 if pd.notna(result['fallback_price']) else np.nan

            all_results.append(result)

        if (i + 1) % 7 == 0:
            print(f"  {date_str}: {i+1}/{len(test_dates)} days, {len(all_results):,} trades so far")

    conn.close()

    results = pd.DataFrame(all_results)
    results = results.set_index('datetime').sort_index()
    print(f"\n[+] Total simulated orders: {len(results)}")

    # Save raw results
    results.to_csv(DATA_DIR / "backtest_limit_orders.csv")
    print(f"[+] Saved: {DATA_DIR / 'backtest_limit_orders.csv'}")

    # ============================================================
    # ANALYSIS
    # ============================================================
    n_days = results.index.normalize().nunique()
    filled = results[results['filled']]
    not_filled = results[~results['filled']]

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    print(f"\n  Orders placed: {len(results)} ({len(results)/n_days:.0f}/day)")
    print(f"  Filled: {len(filled)} ({len(filled)/len(results):.0%})")
    print(f"  Not filled: {len(not_filled)} ({len(not_filled)/len(results):.0%})")

    # Scenario 1: Only limit orders (no fallback)
    print(f"\n--- Scenario 1: Limit orders only (unfilled = no trade) ---")
    if len(filled) > 0:
        total = filled['pnl_limit'].sum()
        wr = (filled['pnl_limit'] > 0).mean()
        daily = filled.groupby(filled.index.date)['pnl_limit'].sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        prof = (daily > 0).sum()
        print(f"  Trades: {len(filled)} ({len(filled)/n_days:.0f}/day)")
        print(f"  Win rate: {wr:.1%}")
        print(f"  P&L: {total:>+,.0f} EUR ({total/n_days:>+,.0f}/day), Sharpe: {sharpe:.1f}")
        print(f"  Days: {prof}/{len(daily)} ({prof/len(daily):.0%}), worst: {daily.min():>+,.0f}")

        filled_m = filled.copy()
        filled_m['month'] = filled_m.index.to_period('M')
        for period, grp in filled_m.groupby('month'):
            nd = grp.index.normalize().nunique()
            mpnl = grp['pnl_limit'].sum()
            mwr = (grp['pnl_limit'] > 0).mean()
            print(f"    {period}: {len(grp)} fills, win={mwr:.0%}, P&L={mpnl:>+,.0f} ({mpnl/nd:>+,.0f}/day)")

    # Scenario 2: Limit + fallback market order at T-65min for unfilled
    print(f"\n--- Scenario 2: Limit orders + market fallback at T-65min ---")
    results['pnl_combined'] = np.where(results['filled'], results['pnl_limit'], results['pnl_fallback'])
    valid = results.dropna(subset=['pnl_combined'])
    if len(valid) > 0:
        total = valid['pnl_combined'].sum()
        wr = (valid['pnl_combined'] > 0).mean()
        daily = valid.groupby(valid.index.date)['pnl_combined'].sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        prof = (daily > 0).sum()
        n_filled = valid['filled'].sum()
        print(f"  Trades: {len(valid)} ({n_filled} limit fills + {len(valid)-n_filled} market fallbacks)")
        print(f"  Win rate: {wr:.1%}")
        print(f"  P&L: {total:>+,.0f} EUR ({total/n_days:>+,.0f}/day), Sharpe: {sharpe:.1f}")
        print(f"  Days: {prof}/{len(daily)} ({prof/len(daily):.0%}), worst: {daily.min():>+,.0f}")

        valid_m = valid.copy()
        valid_m['month'] = valid_m.index.to_period('M')
        for period, grp in valid_m.groupby('month'):
            nd = grp.index.normalize().nunique()
            mpnl = grp['pnl_combined'].sum()
            print(f"    {period}: P&L={mpnl:>+,.0f} ({mpnl/nd:>+,.0f}/day)")

    # Scenario 3: Limit only, but with spread filter at entry
    print(f"\n--- Scenario 3: Limit orders, entry spread <= 10 EUR/MWh ---")
    tight = filled[filled['entry_spread'].notna() & (filled['entry_spread'] <= 10)]
    if len(tight) > 0:
        total = tight['pnl_limit'].sum()
        wr = (tight['pnl_limit'] > 0).mean()
        daily = tight.groupby(tight.index.date)['pnl_limit'].sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        prof = (daily > 0).sum()
        print(f"  Trades: {len(tight)} ({len(tight)/n_days:.0f}/day)")
        print(f"  Win rate: {wr:.1%}")
        print(f"  P&L: {total:>+,.0f} EUR ({total/n_days:>+,.0f}/day), Sharpe: {sharpe:.1f}")
        print(f"  Days: {prof}/{len(daily)} ({prof/len(daily):.0%}), worst: {daily.min():>+,.0f}")

    # Fill rate analysis
    print(f"\n--- Fill Rate Analysis ---")
    print(f"  Overall fill rate: {results['filled'].mean():.1%}")
    for dir_name in ['surplus', 'deficit']:
        sub = results[results['direction'] == dir_name]
        print(f"  {dir_name}: {sub['filled'].mean():.1%} fill rate ({sub['filled'].sum()}/{len(sub)})")

    # Entry spread vs fill rate
    print(f"\n  Fill rate by entry spread:")
    results['spread_bin'] = pd.cut(results['entry_spread'], bins=[0, 2, 5, 10, 20, 50, 10000],
                                    labels=['0-2', '2-5', '5-10', '10-20', '20-50', '>50'])
    for bin_name, grp in results.groupby('spread_bin', observed=True):
        if len(grp) > 10:
            print(f"    spread {bin_name}: fill={grp['filled'].mean():.0%}, n={len(grp)}")


if __name__ == "__main__":
    main()
