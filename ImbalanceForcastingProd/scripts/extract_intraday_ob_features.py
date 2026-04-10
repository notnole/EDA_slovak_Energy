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
    """Pull one month of QH snapshots, pre-grouped in SQL."""
    query = """
        SELECT tradeday, periodfrom, lastupdate,
               MAX(CASE WHEN tradetype = 'N' THEN price END) as bid,
               MAX(CASE WHEN tradetype = 'P' THEN price END) as ask
        FROM db_ems.vdt_isot_knihaobjednavok_best
        WHERE tradeday >= %s AND tradeday < %s
          AND deliverydur = 15 AND id_depth = 1
        GROUP BY tradeday, periodfrom, lastupdate
    """
    cur = conn.cursor()
    cur.execute(query, (month_start, month_end))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=['tradeday', 'periodfrom', 'lastupdate', 'bid', 'ask'])
    df['bid'] = df['bid'].astype(float)
    df['ask'] = df['ask'].astype(float)
    df['tradeday'] = pd.to_datetime(df['tradeday'])
    df['lastupdate'] = pd.to_datetime(df['lastupdate'])
    df['delivery_start'] = df['tradeday'] + pd.to_timedelta(df['periodfrom'] * 15, unit='m')
    df['minutes_before'] = (df['delivery_start'] - df['lastupdate']).dt.total_seconds() / 60

    # Keep 60-600min window
    df = df[(df['minutes_before'] >= 60) & (df['minutes_before'] <= 600)]

    df['mid'] = df[['bid', 'ask']].mean(axis=1)
    df.loc[df['bid'].isna() & df['ask'].notna(), 'mid'] = df['ask']
    df.loc[df['ask'].isna() & df['bid'].notna(), 'mid'] = df['bid']
    df['spread'] = df['ask'] - df['bid']

    return df


def process_month(df, da_hourly):
    """Process all delivery periods from one month of data."""
    if len(df) == 0:
        return pd.DataFrame()

    results = []

    for (trade_day, period), period_data in df.groupby(['tradeday', 'periodfrom']):
        period_data = period_data.sort_values('lastupdate')
        delivery_start = pd.Timestamp(trade_day) + pd.Timedelta(minutes=int(period) * 15)
        prediction_time = delivery_start - pd.Timedelta(minutes=120)

        # Current state at T-120min
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

        # Market-wide pressure: all periods with updates in 15min before prediction_time
        same_day = df[df['tradeday'] == trade_day]
        window = same_day[(same_day['lastupdate'] >= prediction_time - pd.Timedelta(minutes=15)) &
                          (same_day['lastupdate'] <= prediction_time)]

        if len(window) > 0:
            bid_up = 0
            bid_dn = 0
            ask_up = 0
            ask_dn = 0
            da_spreads = []
            act_spreads = []
            n_act = 0

            for p, grp in window.groupby('periodfrom'):
                grp = grp.sort_values('lastupdate')
                if len(grp) < 2:
                    continue
                n_act += 1
                f, l = grp.iloc[0], grp.iloc[-1]
                if pd.notna(f['bid']) and pd.notna(l['bid']):
                    if l['bid'] > f['bid']: bid_up += 1
                    elif l['bid'] < f['bid']: bid_dn += 1
                if pd.notna(f['ask']) and pd.notna(l['ask']):
                    if l['ask'] < f['ask']: ask_dn += 1
                    elif l['ask'] > f['ask']: ask_up += 1
                p_hour = pd.Timestamp(trade_day) + pd.Timedelta(hours=int(p) // 4)
                da_p = da_hourly.get(p_hour, np.nan)
                if pd.notna(l['mid']) and pd.notna(da_p):
                    da_spreads.append(l['mid'] - da_p)
                if pd.notna(l['spread']) and l['spread'] > 0:
                    act_spreads.append(l['spread'])

            if n_act > 0:
                row['ob_pct_bid_rising'] = bid_up / n_act
                row['ob_pct_ask_falling'] = ask_dn / n_act
                row['ob_net_pressure'] = (bid_up - bid_dn) / n_act
                row['ob_n_active_periods'] = n_act
                row['ob_mean_da_spread'] = np.mean(da_spreads) if da_spreads else np.nan
                row['ob_mean_spread'] = np.mean(act_spreads) if act_spreads else np.nan

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
