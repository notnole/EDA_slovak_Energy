"""
Extract IDM Order Book at Extended Lead Times (180, 240, 300 min)
=================================================================

Same logic as extract_orderbook_qh.py but for longer lead times
to test whether trading EARLIER captures a wider IDM-settlement spread.

Lead times extracted:
  - 180min (3h before delivery)
  - 240min (4h before delivery)
  - 300min (5h before delivery)

Output: data/features/orderbook_qh_extended_leads.csv
"""

import psycopg2
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parents[2]  # ImbalanceForcastingProd/
DATA_DIR = BASE_DIR / "data" / "features"

DB_CONN = dict(host='localhost', port=5432, dbname='DB_EMS',
               user='postgres', password='Kapitan1', connect_timeout=10)

START_DATE = '2024-03-01'
END_DATE = '2026-04-13'

# Extended lead times (minutes before delivery start)
LEAD_MINUTES = [180, 240, 300]


def extract_day(cur, trade_date):
    """Extract QH order book snapshots for all 15-min periods on one day."""
    cur.execute("""
        SELECT lastupdate, periodfrom,
               MAX(CASE WHEN tradetype = 'N' THEN price END) as bid,
               MAX(CASE WHEN tradetype = 'P' THEN price END) as ask
        FROM db_ems.vdt_isot_knihaobjednavok_best
        WHERE tradeday = %s
          AND deliverydur = 15
          AND id_depth = 1
        GROUP BY lastupdate, periodfrom
        ORDER BY periodfrom, lastupdate
    """, (trade_date,))

    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=['lastupdate', 'periodfrom', 'bid', 'ask'])
    df['lastupdate'] = pd.to_datetime(df['lastupdate'])
    df['bid'] = df['bid'].astype(float)
    df['ask'] = df['ask'].astype(float)

    # Delivery start = tradeday + periodfrom * 15min
    trade_dt = pd.Timestamp(trade_date)
    df['delivery_start'] = trade_dt + pd.to_timedelta(df['periodfrom'] * 15, unit='m')

    # Mid and spread
    df['mid'] = df[['bid', 'ask']].mean(axis=1)
    df.loc[df['bid'].isna() & df['ask'].notna(), 'mid'] = df['ask']
    df.loc[df['ask'].isna() & df['bid'].notna(), 'mid'] = df['bid']
    df['spread'] = df['ask'] - df['bid']

    # Minutes before delivery
    df['minutes_before'] = (df['delivery_start'] - df['lastupdate']).dt.total_seconds() / 60
    df = df[df['minutes_before'] > 0]

    results = []

    for period in range(96):  # 96 QH periods per day
        period_data = df[df['periodfrom'] == period].sort_values('lastupdate')
        if len(period_data) == 0:
            continue

        delivery_start = trade_dt + timedelta(minutes=period * 15)

        for lead_min in LEAD_MINUTES:
            # Find latest snapshot at least lead_min before delivery
            mask = period_data['minutes_before'] >= lead_min
            if not mask.any():
                continue
            snap = period_data[mask].iloc[-1]

            row = {
                'delivery_start': delivery_start,
                'lead_minutes': lead_min,
                'snapshot_time': snap['lastupdate'],
                'bid': snap['bid'],
                'ask': snap['ask'],
                'mid': snap['mid'],
                'spread': snap['spread'],
            }
            results.append(row)

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("EXTRACTING QH ORDER BOOK AT EXTENDED LEAD TIMES")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Leads: {LEAD_MINUTES} minutes before delivery")
    print("=" * 70)

    conn = psycopg2.connect(**DB_CONN)
    cur = conn.cursor()

    dates = pd.date_range(START_DATE, END_DATE, freq='D')
    all_dfs = []

    for i, date in enumerate(dates):
        date_str = date.strftime('%Y-%m-%d')
        df = extract_day(cur, date_str)
        if len(df) > 0:
            all_dfs.append(df)
        if (i + 1) % 30 == 0:
            n_rows = sum(len(d) for d in all_dfs)
            print(f"  {date_str}: {i+1}/{len(dates)} days, {n_rows:,} rows total")

    conn.close()

    if not all_dfs:
        print("[!] No data extracted")
        return

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values(['delivery_start', 'lead_minutes']).reset_index(drop=True)

    print(f"\n[+] Extracted {len(result):,} rows")
    print(f"    Date range: {result['delivery_start'].min()} to {result['delivery_start'].max()}")
    print(f"    Lead times: {sorted(result['lead_minutes'].unique())}")

    out_path = DATA_DIR / "orderbook_qh_extended_leads.csv"
    result.to_csv(out_path, index=False)
    print(f"[+] Saved: {out_path}")

    # Summary stats per lead
    for lead in LEAD_MINUTES:
        sub = result[result['lead_minutes'] == lead]
        both = sub['spread'].notna()
        print(f"\n  Lead {lead}min: {len(sub):,} rows, {both.mean():.0%} have spread")
        if both.any():
            s = sub.loc[both, 'spread']
            print(f"    Spread: median={s.median():.1f}, mean={s.mean():.1f}, P90={s.quantile(0.9):.1f}")
            m = sub.loc[both, 'mid']
            print(f"    Mid: mean={m.mean():.1f}, std={m.std():.1f}")

    print("\n[+] Done!")


if __name__ == "__main__":
    main()
