"""
Buy-only vs Sell-only analysis.
Does one direction consistently make money while the other loses?
"""

import pandas as pd
import numpy as np
import psycopg2
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DB = {"host": "localhost", "port": 5432, "dbname": "DB_EMS", "user": "postgres", "password": "Kapitan1"}


def main():
    conn = psycopg2.connect(**DB)
    ob = pd.read_sql("""
        WITH ob AS (
            SELECT tradeday::date AS tradeday, periodfrom AS hour,
                   tradetype, price,
                   tradeday::date + periodfrom * INTERVAL '1 hour' AS delivery_start,
                   lastupdate
            FROM db_ems.vdt_isot_knihaobjednavok_best
            WHERE deliverydur = 60 AND id_depth = 1 AND price > 0 AND price < 1000
        ),
        with_ttd AS (
            SELECT *, EXTRACT(EPOCH FROM (delivery_start - lastupdate)) / 3600.0 AS htd
            FROM ob WHERE lastupdate < delivery_start
        )
        SELECT tradeday, hour, tradetype, AVG(price) AS price
        FROM with_ttd WHERE htd >= 0.75 AND htd < 2
        GROUP BY tradeday, hour, tradetype
    """, conn)
    conn.close()

    bids = ob[ob['tradetype'] == 'N'][['tradeday', 'hour', 'price']].rename(columns={'price': 'bid'})
    asks = ob[ob['tradetype'] == 'P'][['tradeday', 'hour', 'price']].rename(columns={'price': 'ask'})
    idm = bids.merge(asks, on=['tradeday', 'hour'], how='inner')
    idm['tradeday'] = pd.to_datetime(idm['tradeday'])

    mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv")
    mkt['timestamp_hour'] = pd.to_datetime(mkt['timestamp_hour'])
    mkt['tradeday'] = pd.to_datetime(mkt['timestamp_hour'].dt.date)
    if 'hour' not in mkt.columns:
        mkt['hour'] = mkt['timestamp_hour'].dt.hour

    m = idm.merge(mkt[['tradeday', 'hour', 'imb_settlement_price', 'imbalance_mwh']].drop_duplicates(),
                  on=['tradeday', 'hour'], how='inner')
    m = m.dropna(subset=['bid', 'ask', 'imb_settlement_price'])
    m['year'] = m['tradeday'].dt.year
    m['is_deficit'] = (m['imbalance_mwh'] < 0).astype(int)

    # P&L for each direction (no prediction, just always do it)
    m['pnl_buy'] = m['imb_settlement_price'] - m['ask']    # buy IDM, settle at imb
    m['pnl_sell'] = m['bid'] - m['imb_settlement_price']   # sell IDM, settle at imb

    # ================================================================
    print("=" * 90)
    print("NO PREDICTION: Always buy vs always sell (every hour, 1 MWh)")
    print("=" * 90)

    for direction, col in [('ALWAYS BUY IDM', 'pnl_buy'), ('ALWAYS SELL IDM', 'pnl_sell')]:
        print(f"\n  {direction}:")
        print(f"  {'Year':<8} {'Hours':>7} {'Avg/trade':>10} {'Median':>8} {'P5':>8} {'P95':>8} {'Daily':>8} {'Win%':>6} {'Worst day':>10}")
        print(f"  {'-'*75}")
        for year in [2024, 2025, 2026]:
            sub = m[m['year'] == year]
            if len(sub) == 0:
                continue
            pnl = sub[col]
            daily = sub.groupby('tradeday')[col].sum()
            worst = daily.min()
            print(f"  {year:<8} {len(sub):>7} {pnl.mean():>+10.1f} {pnl.median():>+8.1f} "
                  f"{pnl.quantile(0.05):>+8.0f} {pnl.quantile(0.95):>+8.0f} "
                  f"{daily.mean():>+8.0f} {(pnl > 0).mean():>5.0%} {worst:>+10.0f}")

    # ================================================================
    print("\n\n" + "=" * 90)
    print("WITH 65% DIRECTION ACCURACY (500 MC sims, 1 MWh)")
    print("Only trade in one direction, skip the other entirely.")
    print("=" * 90)

    rng = np.random.default_rng(42)

    for direction in ['BUY only', 'SELL only']:
        print(f"\n  {direction} (predict when to enter, skip opposite):")
        print(f"  {'Year':<8} {'Trades/day':>10} {'Med daily':>10} {'P10':>8} {'P90':>8} {'Worst day':>10}")
        print(f"  {'-'*60}")

        for year in [2024, 2025, 2026]:
            sub = m[m['year'] == year].copy()
            if len(sub) < 50:
                continue

            n_days = sub['tradeday'].nunique()
            n_deficit = (sub['is_deficit'] == 1).sum()
            n_surplus = (sub['is_deficit'] == 0).sum()

            deficit_rows = sub[sub['is_deficit'] == 1]
            surplus_rows = sub[sub['is_deficit'] == 0]

            dailies = []
            worst_days = []

            for _ in range(500):
                if direction == 'BUY only':
                    # We predict deficit -> buy IDM
                    # True positives: 65% of actual deficit hours (we correctly buy)
                    tp_idx = deficit_rows.index[rng.random(n_deficit) < 0.65]
                    # False positives: 35% of surplus hours (we wrongly buy)
                    fp_idx = surplus_rows.index[rng.random(n_surplus) < 0.35]
                    traded = sub.loc[tp_idx.union(fp_idx)]
                    pnl = traded['pnl_buy']
                else:
                    # We predict surplus -> sell IDM
                    # True positives: 65% of actual surplus hours (we correctly sell)
                    tp_idx = surplus_rows.index[rng.random(n_surplus) < 0.65]
                    # False positives: 35% of deficit hours (we wrongly sell)
                    fp_idx = deficit_rows.index[rng.random(n_deficit) < 0.35]
                    traded = sub.loc[tp_idx.union(fp_idx)]
                    pnl = traded['pnl_sell']

                df_tmp = pd.DataFrame({'date': traded['tradeday'].values, 'pnl': pnl.values})
                daily = df_tmp.groupby('date')['pnl'].sum()
                all_days = pd.date_range(sub['tradeday'].min(), sub['tradeday'].max(), freq='D')
                daily = daily.reindex(all_days, fill_value=0)
                dailies.append(daily.mean())
                worst_days.append(daily.min())

            trades_day = (n_deficit * 0.65 + n_surplus * 0.35) / n_days if direction == 'BUY only' else (n_surplus * 0.65 + n_deficit * 0.35) / n_days

            print(f"  {year:<8} {trades_day:>10.1f} {np.median(dailies):>+10.0f} "
                  f"{np.percentile(dailies, 10):>+8.0f} {np.percentile(dailies, 90):>+8.0f} "
                  f"{np.median(worst_days):>+10.0f}")

    # ================================================================
    print("\n\n" + "=" * 90)
    print("WHY? Average P&L when correct vs incorrect, by direction and year")
    print("=" * 90)

    for year in [2024, 2025, 2026]:
        sub = m[m['year'] == year]
        deficit = sub[sub['is_deficit'] == 1]
        surplus = sub[sub['is_deficit'] == 0]

        print(f"\n  {year}:")
        print(f"    Deficit hours: {len(deficit)} ({len(deficit)/len(sub)*100:.0f}%)")
        print(f"    Surplus hours: {len(surplus)} ({len(surplus)/len(sub)*100:.0f}%)")
        print(f"    Avg imb price:  deficit={deficit['imb_settlement_price'].mean():.0f}, surplus={surplus['imb_settlement_price'].mean():.0f}")
        print(f"    Avg IDM bid:    deficit={deficit['bid'].mean():.0f}, surplus={surplus['bid'].mean():.0f}")
        print(f"    Avg IDM ask:    deficit={deficit['ask'].mean():.0f}, surplus={surplus['ask'].mean():.0f}")
        print()
        print(f"    BUY in deficit (correct):   avg P&L = {deficit['pnl_buy'].mean():+.1f} (imb_price - ask)")
        print(f"    BUY in surplus (wrong):     avg P&L = {surplus['pnl_buy'].mean():+.1f} (imb_price - ask)")
        print(f"    SELL in surplus (correct):  avg P&L = {surplus['pnl_sell'].mean():+.1f} (bid - imb_price)")
        print(f"    SELL in deficit (wrong):    avg P&L = {deficit['pnl_sell'].mean():+.1f} (bid - imb_price)")

    print("\n[+] Done.")


if __name__ == "__main__":
    main()
