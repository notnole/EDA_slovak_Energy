"""
Extract Intraday Order Book Features — Monthly Chunks
======================================================

Pulls data in monthly chunks to avoid 99M-row single query.
Uses server-side time filtering to keep memory reasonable.
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
               user='postgres', password='Kapitan1', connect_timeout=30)

START_DATE = '2024-03-01'
END_DATE = '2026-04-01'
LIQUID_SPREAD_THRESHOLD = 15.0


def pull_month(conn, month_start, month_end):
    """Pull one month of QH snapshots — NO GROUP BY, pivot in pandas.

    Following the BESS intra-trading pattern: raw SELECT + ORDER BY is
    10-100x faster than GROUP BY with CASE pivots. The DB does a simple
    index scan, pandas does the pivot in-memory.
    """
    query = """
        SELECT tradeday, periodfrom, lastupdate, tradetype, price
        FROM db_ems.vdt_isot_knihaobjednavok_best
        WHERE tradeday >= %s AND tradeday < %s
          AND deliverydur = 15 AND id_depth = 1
        ORDER BY tradeday, lastupdate, periodfrom
    """
    cur = conn.cursor()
    cur.execute(query, (month_start, month_end))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return pd.DataFrame()

    df_raw = pd.DataFrame(rows, columns=['tradeday', 'periodfrom', 'lastupdate', 'tradetype', 'price'])
    df_raw['price'] = df_raw['price'].astype(float)
    df_raw['tradeday'] = pd.to_datetime(df_raw['tradeday'])
    df_raw['lastupdate'] = pd.to_datetime(df_raw['lastupdate'])

    # Pivot bid/ask in pandas (fast vectorized operation)
    df = df_raw.pivot_table(
        index=['tradeday', 'periodfrom', 'lastupdate'],
        columns='tradetype',
        values='price',
        aggfunc='first'
    ).reset_index()
    df.columns.name = None
    df = df.rename(columns={'N': 'bid', 'P': 'ask'})

    df['delivery_start'] = df['tradeday'] + pd.to_timedelta(df['periodfrom'] * 15, unit='m')
    df['minutes_before'] = (df['delivery_start'] - df['lastupdate']).dt.total_seconds() / 60

    # Keep 60-600min window
    df = df[(df['minutes_before'] >= 60) & (df['minutes_before'] <= 600)]

    df['mid'] = df[['bid', 'ask']].mean(axis=1)
    df.loc[df['bid'].isna() & df['ask'].notna(), 'mid'] = df['ask']
    df.loc[df['ask'].isna() & df['bid'].notna(), 'mid'] = df['bid']
    df['spread'] = df['ask'] - df['bid']

    return df


def compute_pressure_by_timeslot(day_data, da_hourly, trade_day):
    """Precompute market-wide pressure for each 15-min timeslot in one day.

    Instead of recomputing for each delivery period (O(n^2)), compute once
    per timeslot and broadcast. There are only ~96 timeslots per day.
    Returns dict: timeslot_start -> pressure features.
    """
    trade_dt = pd.Timestamp(trade_day)
    pressure_cache = {}

    # Generate 15-min timeslots covering the trading day
    for slot_idx in range(96):
        slot_time = trade_dt + pd.Timedelta(minutes=slot_idx * 15)
        window_start = slot_time - pd.Timedelta(minutes=15)

        window = day_data[(day_data['lastupdate'] >= window_start) &
                          (day_data['lastupdate'] <= slot_time)]
        if len(window) == 0:
            continue

        bid_up = 0
        bid_dn = 0
        ask_dn = 0
        ask_up = 0
        da_spreads = []
        act_spreads = []
        n_act = 0

        # Vectorized: get first and last per period in one pass
        for p, grp in window.groupby('periodfrom'):
            if len(grp) < 2:
                continue
            grp = grp.sort_values('lastupdate')
            n_act += 1
            f_bid, l_bid = grp['bid'].iloc[0], grp['bid'].iloc[-1]
            f_ask, l_ask = grp['ask'].iloc[0], grp['ask'].iloc[-1]

            if pd.notna(f_bid) and pd.notna(l_bid):
                if l_bid > f_bid: bid_up += 1
                elif l_bid < f_bid: bid_dn += 1
            if pd.notna(f_ask) and pd.notna(l_ask):
                if l_ask < f_ask: ask_dn += 1
                elif l_ask > f_ask: ask_up += 1

            l_mid = grp['mid'].iloc[-1]
            p_hour = trade_dt + pd.Timedelta(hours=int(p) // 4)
            da_p = da_hourly.get(p_hour, np.nan)
            if pd.notna(l_mid) and pd.notna(da_p):
                da_spreads.append(l_mid - da_p)
            l_spread = grp['spread'].iloc[-1]
            if pd.notna(l_spread) and l_spread > 0:
                act_spreads.append(l_spread)

        if n_act > 0:
            pressure_cache[slot_time] = {
                'ob_pct_bid_rising': bid_up / n_act,
                'ob_pct_ask_falling': ask_dn / n_act,
                'ob_net_pressure': (bid_up - bid_dn) / n_act,
                'ob_n_active_periods': n_act,
                'ob_mean_da_spread': np.mean(da_spreads) if da_spreads else np.nan,
                'ob_mean_spread': np.mean(act_spreads) if act_spreads else np.nan,
            }

    return pressure_cache


def process_month(df, da_hourly):
    """Process all delivery periods from one month of data.

    Step 1: Precompute market pressure per (day, timeslot) — O(days * 96)
    Step 2: Compute target features per (day, period) — O(periods)
    Step 3: Join pressure onto target features by prediction_time timeslot
    """
    if len(df) == 0:
        return pd.DataFrame()

    # Step 1: Precompute pressure per day
    pressure_by_day = {}
    for trade_day, day_data in df.groupby('tradeday'):
        pressure_by_day[trade_day] = compute_pressure_by_timeslot(
            day_data, da_hourly, trade_day)

    # Step 2: Target features per period
    results = []
    for (trade_day, period), period_data in df.groupby(['tradeday', 'periodfrom']):
        period_data = period_data.sort_values('lastupdate')
        delivery_start = pd.Timestamp(trade_day) + pd.Timedelta(minutes=int(period) * 15)
        prediction_time = delivery_start - pd.Timedelta(minutes=120)

        at_pred = period_data[period_data['minutes_before'] >= 120]
        if len(at_pred) == 0:
            continue
        current = at_pred.iloc[-1]

        hour_ts = delivery_start.floor('h')
        da_price = da_hourly.get(hour_ts, np.nan)

        row = {
            'delivery_start': delivery_start,
            'ob_target_mid': current['mid'],
            'ob_target_spread': current['spread'],
            'ob_target_da_spread': (current['mid'] - da_price) if pd.notna(current['mid']) and pd.notna(da_price) else np.nan,
        }

        # 15min and 1h changes
        snap_15 = period_data[period_data['minutes_before'] >= 135]
        snap_60 = period_data[period_data['minutes_before'] >= 180]
        row['ob_target_mid_change_15m'] = (current['mid'] - snap_15.iloc[-1]['mid']) if len(snap_15) > 0 and pd.notna(snap_15.iloc[-1]['mid']) and pd.notna(current['mid']) else np.nan
        row['ob_target_mid_change_1h'] = (current['mid'] - snap_60.iloc[-1]['mid']) if len(snap_60) > 0 and pd.notna(snap_60.iloc[-1]['mid']) and pd.notna(current['mid']) else np.nan

        # Since liquid
        liquid = period_data[(period_data['spread'] > 0) & (period_data['spread'] < LIQUID_SPREAD_THRESHOLD)]
        if len(liquid) > 0:
            liq = liquid.iloc[0]
            row['ob_target_mid_since_liquid'] = (current['mid'] - liq['mid']) if pd.notna(liq['mid']) and pd.notna(current['mid']) else np.nan
            row['ob_target_bid_since_liquid'] = (current['bid'] - liq['bid']) if pd.notna(liq['bid']) and pd.notna(current['bid']) else np.nan
            row['ob_target_ask_since_liquid'] = (current['ask'] - liq['ask']) if pd.notna(liq['ask']) and pd.notna(current['ask']) else np.nan
            row['ob_target_time_liquid_min'] = (current['lastupdate'] - liq['lastupdate']).total_seconds() / 60
        else:
            row['ob_target_mid_since_liquid'] = np.nan
            row['ob_target_bid_since_liquid'] = np.nan
            row['ob_target_ask_since_liquid'] = np.nan
            row['ob_target_time_liquid_min'] = 0.0

        # Step 3: Look up precomputed pressure for nearest timeslot
        pred_slot = prediction_time.floor('15min')
        day_pressure = pressure_by_day.get(trade_day, {})
        pressure = day_pressure.get(pred_slot)
        if pressure:
            row.update(pressure)

        results.append(row)

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("EXTRACTING INTRADAY ORDER BOOK FEATURES (monthly chunks)")
    print(f"Period: {START_DATE} to {END_DATE}")
    print("=" * 70)

    da = pd.read_csv(REPO_ROOT / 'features' / 'DamasPrices' / 'data' / 'da_prices.csv',
                     parse_dates=['datetime'], index_col='datetime')
    da_hourly = da['price_eur_mwh']
    da_hourly = da_hourly[~da_hourly.index.duplicated(keep='last')]
    print(f"[+] DA prices: {len(da_hourly)} hours")

    conn = psycopg2.connect(**DB_CONN)

    months = pd.date_range(START_DATE, END_DATE, freq='MS')
    all_results = []

    for i in range(len(months)):
        m_start = months[i].strftime('%Y-%m-%d')
        m_end = (months[i] + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        if pd.Timestamp(m_end) > pd.Timestamp(END_DATE):
            m_end = END_DATE

        print(f"  [{i+1}/{len(months)}] {m_start}...", end=' ', flush=True)
        df = pull_month(conn, m_start, m_end)
        print(f"{len(df):,} rows ->", end=' ', flush=True)

        if len(df) > 0:
            result = process_month(df, da_hourly)
            all_results.append(result)
            print(f"{len(result):,} periods")
        else:
            print("no data")

    conn.close()

    final = pd.concat(all_results, ignore_index=True)
    final = final.sort_values('delivery_start').reset_index(drop=True)

    print(f"\n[+] Total: {len(final):,} rows")
    print(f"    Range: {final['delivery_start'].min()} to {final['delivery_start'].max()}")

    print(f"\n  Feature coverage:")
    for col in sorted([c for c in final.columns if c.startswith('ob_')]):
        pct = final[col].notna().mean() * 100
        if pct > 0:
            print(f"    {col:<35s}: {pct:5.1f}%")

    out_path = DATA_DIR / "intraday_ob_features.csv"
    final.to_csv(out_path, index=False)
    print(f"\n[+] Saved: {out_path}")


if __name__ == "__main__":
    main()
