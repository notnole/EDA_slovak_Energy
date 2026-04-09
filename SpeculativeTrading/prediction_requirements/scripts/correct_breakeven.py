"""
Correct Breakeven Accuracy Analysis
====================================

Previous analysis was wrong because it used sorted (optimistic) assignment.
This uses proper random Monte Carlo AND tests each year separately
AND accounts for the asymmetric loss structure.

For each direction (buy/sell/both), at each accuracy level,
run 500 random simulations and report median P&L.
"""

import pandas as pd
import numpy as np
import psycopg2
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB = {"host": "localhost", "port": 5432, "dbname": "DB_EMS", "user": "postgres", "password": "Kapitan1"}

ACCURACY_LEVELS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
N_SIMS = 500


def load_data():
    print("[*] Loading data...")
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
    m['pnl_buy'] = m['imb_settlement_price'] - m['ask']
    m['pnl_sell'] = m['bid'] - m['imb_settlement_price']

    print(f"[+] {len(m)} hours, {m['tradeday'].nunique()} days")
    return m


def simulate_both_directions(sub, accuracy, rng):
    """
    Trade both directions: for each hour, predict deficit or surplus.
    At X% accuracy, X% of hours get the correct direction.
    Correct direction for deficit = buy IDM. Correct for surplus = sell IDM.
    Wrong direction for deficit = sell IDM. Wrong for surplus = buy IDM.
    """
    n = len(sub)
    is_correct = rng.random(n) < accuracy
    is_deficit = sub['is_deficit'].values.astype(bool)

    pnl = np.where(
        is_correct,
        np.where(is_deficit, sub['pnl_buy'].values, sub['pnl_sell'].values),     # correct
        np.where(is_deficit, sub['pnl_sell'].values, sub['pnl_buy'].values),      # wrong
    )
    return pnl, sub['tradeday'].values


def simulate_buy_only(sub, accuracy, rng):
    """
    Only buy direction. Predict deficit at X% accuracy.
    True positive: correctly predict deficit -> buy -> get pnl_buy on deficit hour.
    False positive: wrongly predict deficit on surplus hour -> buy -> get pnl_buy on surplus hour.
    True negative / false negative: don't trade.
    """
    deficit = sub[sub['is_deficit'] == 1]
    surplus = sub[sub['is_deficit'] == 0]

    tp_mask = rng.random(len(deficit)) < accuracy       # correctly identify deficit
    fp_mask = rng.random(len(surplus)) < (1 - accuracy) # wrongly flag surplus as deficit

    pnl_tp = deficit['pnl_buy'].values[tp_mask]
    pnl_fp = surplus['pnl_buy'].values[fp_mask]
    dates_tp = deficit['tradeday'].values[tp_mask]
    dates_fp = surplus['tradeday'].values[fp_mask]

    return np.concatenate([pnl_tp, pnl_fp]), np.concatenate([dates_tp, dates_fp])


def simulate_sell_only(sub, accuracy, rng):
    """
    Only sell direction. Predict surplus at X% accuracy.
    """
    deficit = sub[sub['is_deficit'] == 1]
    surplus = sub[sub['is_deficit'] == 0]

    tp_mask = rng.random(len(surplus)) < accuracy
    fp_mask = rng.random(len(deficit)) < (1 - accuracy)

    pnl_tp = surplus['pnl_sell'].values[tp_mask]
    pnl_fp = deficit['pnl_sell'].values[fp_mask]
    dates_tp = surplus['tradeday'].values[tp_mask]
    dates_fp = deficit['tradeday'].values[fp_mask]

    return np.concatenate([pnl_tp, pnl_fp]), np.concatenate([dates_tp, dates_fp])


def run_sweep(sub, sim_fn, label, year):
    """Run accuracy sweep with MC simulation."""
    rng = np.random.default_rng(42)
    n_days = sub['tradeday'].nunique()
    all_days = pd.date_range(sub['tradeday'].min(), sub['tradeday'].max(), freq='D')

    results = []
    for acc in ACCURACY_LEVELS:
        daily_means = []
        daily_worsts = []
        daily_win_rates = []

        for _ in range(N_SIMS):
            pnl, dates = sim_fn(sub, acc, rng)
            if len(pnl) == 0:
                continue
            df_tmp = pd.DataFrame({'date': dates, 'pnl': pnl})
            daily = df_tmp.groupby('date')['pnl'].sum().reindex(all_days, fill_value=0)
            daily_means.append(daily.mean())
            daily_worsts.append(daily.min())
            daily_win_rates.append((daily > 0).mean())

        if len(daily_means) == 0:
            continue

        med_daily = np.median(daily_means)
        results.append({
            'accuracy': acc,
            'med_daily': med_daily,
            'p10_daily': np.percentile(daily_means, 10),
            'p90_daily': np.percentile(daily_means, 90),
            'med_worst_day': np.median(daily_worsts),
            'med_win_rate': np.median(daily_win_rates),
            'mwh_for_200': 200.0 / med_daily if med_daily > 0 else float('inf'),
        })

    return pd.DataFrame(results)


def find_breakeven(df):
    """Find accuracy where med_daily crosses zero."""
    for i in range(len(df) - 1):
        if df.iloc[i]['med_daily'] <= 0 < df.iloc[i + 1]['med_daily']:
            a1, a2 = df.iloc[i]['accuracy'], df.iloc[i + 1]['accuracy']
            p1, p2 = df.iloc[i]['med_daily'], df.iloc[i + 1]['med_daily']
            return a1 + (a2 - a1) * (-p1) / (p2 - p1)
    if len(df) > 0 and df.iloc[0]['med_daily'] > 0:
        return df.iloc[0]['accuracy'] - 0.05  # below tested range
    return 0.95


def main():
    m = load_data()

    strategies = [
        ('BOTH directions', simulate_both_directions),
        ('BUY only (predict deficit)', simulate_buy_only),
        ('SELL only (predict surplus)', simulate_sell_only),
    ]

    years = [2024, 2025, 2026]

    all_results = []

    for strat_name, sim_fn in strategies:
        print(f"\n{'='*90}")
        print(f"  {strat_name}")
        print(f"{'='*90}")

        for year in years:
            sub = m[m['year'] == year]
            if len(sub) < 100:
                continue

            sweep = run_sweep(sub, sim_fn, strat_name, year)
            be = find_breakeven(sweep)

            print(f"\n  {year} ({len(sub)} hours, {sub['tradeday'].nunique()} days)  |  Breakeven: {be:.0%}")
            print(f"  {'Acc':>5} {'Med daily':>10} {'P10':>8} {'P90':>8} {'Worst day':>10} {'WinRate':>8} {'MWh@200':>8}")
            print(f"  {'-'*65}")

            for _, r in sweep.iterrows():
                mwh = f"{r['mwh_for_200']:.1f}" if not np.isinf(r['mwh_for_200']) and r['mwh_for_200'] > 0 else "N/A"
                print(f"  {r['accuracy']:>4.0%} {r['med_daily']:>+10.0f} {r['p10_daily']:>+8.0f} "
                      f"{r['p90_daily']:>+8.0f} {r['med_worst_day']:>+10.0f} "
                      f"{r['med_win_rate']:>7.0%} {mwh:>8}")

            for _, r in sweep.iterrows():
                all_results.append({
                    'strategy': strat_name,
                    'year': year,
                    'breakeven': be,
                    **r.to_dict(),
                })

    # Save
    pd.DataFrame(all_results).to_csv(DATA_DIR / "correct_breakeven.csv", index=False)

    # ================================================================
    print("\n\n" + "=" * 90)
    print("SUMMARY: Breakeven accuracy by strategy and year")
    print("=" * 90)
    print(f"\n  {'Strategy':<30} {'2024':>8} {'2025':>8} {'2026':>8}")
    print(f"  {'-'*56}")
    for strat_name, _ in strategies:
        line = f"  {strat_name:<30}"
        for year in years:
            res = [r for r in all_results if r['strategy'] == strat_name and r['year'] == year]
            if res:
                line += f" {res[0]['breakeven']:>7.0%}"
            else:
                line += f" {'N/A':>7}"
        print(line)

    # MWh needed at 75% accuracy
    print(f"\n  {'Strategy':<30} {'2024':>8} {'2025':>8} {'2026':>8}   (MWh for EUR 200/day @ 75% accuracy)")
    print(f"  {'-'*70}")
    for strat_name, _ in strategies:
        line = f"  {strat_name:<30}"
        for year in years:
            res = [r for r in all_results if r['strategy'] == strat_name and r['year'] == year and r['accuracy'] == 0.75]
            if res:
                mwh = res[0]['mwh_for_200']
                line += f" {mwh:>7.1f}" if not np.isinf(mwh) and mwh > 0 else f" {'N/A':>7}"
            else:
                line += f" {'N/A':>7}"
        print(line)

    print("\n[+] Done.")


if __name__ == "__main__":
    main()
