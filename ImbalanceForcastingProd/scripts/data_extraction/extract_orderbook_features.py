"""
Extract IDM Order Book Features from DB_EMS
============================================

For each 15-min settlement period, extracts the order book state at
multiple time points before delivery to compute:
  - Current best bid, ask, mid price, spread
  - Mid price movement from first quote (session start)
  - Mid price change in last 15min and last 1h
  - Spread change (liquidity dynamics)
  - Distance from DA price

Uses 60-min IDM products (more liquid) and maps to 15-min periods.

Output: data/orderbook_features.csv
"""

import psycopg2
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
REPO_ROOT = SCRIPT_DIR.parents[1]

DB_CONN = dict(host='localhost', port=5432, dbname='DB_EMS',
               user='postgres', password='Kapitan1', connect_timeout=10)

# Extract for the training + test period
START_DATE = '2024-03-01'  # order book data starts Feb 2024
END_DATE = '2026-04-01'


def extract_day(cur, trade_date):
    """Extract order book snapshots for all delivery hours on one day.

    For each delivery hour, gets bid/ask at multiple time points:
    the latest snapshot within each 15-min window before delivery.
    """
    # Get all snapshots for this day's 60-min products
    cur.execute("""
        SELECT lastupdate, periodfrom,
               MAX(CASE WHEN tradetype = 'N' THEN price END) as bid,
               MAX(CASE WHEN tradetype = 'P' THEN price END) as ask
        FROM db_ems.vdt_isot_knihaobjednavok_best
        WHERE tradeday = %s
          AND deliverydur = 60
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

    # Delivery hour = periodfrom // 4
    df['delivery_hour'] = df['periodfrom'] // 4

    # Compute mid price (use available side if one is missing)
    df['mid'] = df[['bid', 'ask']].mean(axis=1)
    df.loc[df['bid'].isna() & df['ask'].notna(), 'mid'] = df['ask']
    df.loc[df['ask'].isna() & df['bid'].notna(), 'mid'] = df['bid']
    df['spread'] = df['ask'] - df['bid']

    # Delivery start time
    trade_dt = pd.Timestamp(trade_date)
    df['delivery_start'] = trade_dt + pd.to_timedelta(df['delivery_hour'], unit='h')

    # Time before delivery (in minutes)
    df['minutes_before'] = (df['delivery_start'] - df['lastupdate']).dt.total_seconds() / 60

    # Filter: only snapshots BEFORE delivery (positive minutes_before)
    # and at least 30 min before (realistic — can't trade in last 30min on some products)
    df = df[df['minutes_before'] > 0]

    # For each delivery hour, compute features at key time windows
    results = []

    for hour in range(24):
        hour_data = df[df['delivery_hour'] == hour].sort_values('lastupdate')
        if len(hour_data) == 0:
            continue

        delivery_start = trade_dt + timedelta(hours=hour)

        # Get snapshots at key times before delivery
        def get_latest_before(minutes_before_target, tolerance=15):
            """Get the latest snapshot that's at least minutes_before_target before delivery."""
            mask = hour_data['minutes_before'] >= minutes_before_target
            if mask.any():
                return hour_data[mask].iloc[-1]
            return None

        # Current state at various lead times
        for lead_minutes in [60, 75, 90, 105, 120]:
            snap = get_latest_before(lead_minutes)
            if snap is None:
                continue

            # First snapshot of the session for this period
            first = hour_data.iloc[0]

            # Snapshot from 15min earlier
            snap_15min_ago = get_latest_before(lead_minutes + 15)
            # Snapshot from 1h earlier
            snap_1h_ago = get_latest_before(lead_minutes + 60)

            row = {
                'delivery_start': delivery_start,
                'lead_minutes': lead_minutes,
                'snapshot_time': snap['lastupdate'],
                'bid': snap['bid'],
                'ask': snap['ask'],
                'mid': snap['mid'],
                'spread': snap['spread'],
                # Movement from session start
                'mid_move_from_start': (snap['mid'] - first['mid']) if pd.notna(first['mid']) and pd.notna(snap['mid']) else np.nan,
                # Movement in last 15 min
                'mid_change_15min': (snap['mid'] - snap_15min_ago['mid']) if snap_15min_ago is not None and pd.notna(snap_15min_ago['mid']) and pd.notna(snap['mid']) else np.nan,
                # Movement in last 1h
                'mid_change_1h': (snap['mid'] - snap_1h_ago['mid']) if snap_1h_ago is not None and pd.notna(snap_1h_ago['mid']) and pd.notna(snap['mid']) else np.nan,
                # Spread change from 15min ago
                'spread_change_15min': (snap['spread'] - snap_15min_ago['spread']) if snap_15min_ago is not None and pd.notna(snap_15min_ago['spread']) and pd.notna(snap['spread']) else np.nan,
                # How many snapshots (activity/liquidity proxy)
                'n_updates_session': len(hour_data),
                # Session start mid
                'session_start_mid': first['mid'],
            }
            results.append(row)

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("EXTRACTING IDM ORDER BOOK FEATURES")
    print(f"Period: {START_DATE} to {END_DATE}")
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
            print(f"  {date_str}: {i+1}/{len(dates)} days processed, {sum(len(d) for d in all_dfs):,} rows total")

    conn.close()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values(['delivery_start', 'lead_minutes']).reset_index(drop=True)

    print(f"\n[+] Extracted {len(result):,} rows")
    print(f"    Date range: {result['delivery_start'].min()} to {result['delivery_start'].max()}")
    print(f"    Lead times: {sorted(result['lead_minutes'].unique())}")

    # Pivot: one row per (delivery_start, lead_minutes) with all features
    out_path = DATA_DIR / "orderbook_features.csv"
    result.to_csv(out_path, index=False)
    print(f"[+] Saved: {out_path}")

    # Also create a simplified version: for each delivery hour, features at lead=60 and lead=120
    # This is what the model will primarily use
    for lead in [60, 120]:
        sub = result[result['lead_minutes'] == lead].copy()
        print(f"\n  Lead {lead}min: {len(sub)} rows, {sub['mid'].notna().mean():.1%} have mid price")
        print(f"    Mid price: mean={sub['mid'].dropna().mean():.1f}, std={sub['mid'].dropna().std():.1f}")
        print(f"    Spread: mean={sub['spread'].dropna().mean():.1f}, median={sub['spread'].dropna().median():.1f}")
        print(f"    Mid move from start: mean={sub['mid_move_from_start'].dropna().mean():.2f}")

    print("\n[+] Done!")


if __name__ == "__main__":
    main()
