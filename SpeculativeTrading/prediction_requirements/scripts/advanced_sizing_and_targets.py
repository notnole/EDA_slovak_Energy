"""
Advanced Analysis: Confidence-Based Sizing & Creative Prediction Targets
========================================================================

Two questions:
1. What if we size positions by prediction confidence instead of flat sizing?
2. What alternative/creative prediction targets could unlock more edge?

Creative targets explored:
  A. Predict imbalance MAGNITUDE (not just direction) -- big imbalances = big price swings
  B. Predict IDM session VOLATILITY -- high vol = bigger round-trip opportunity
  C. Predict WHICH HOURS will have largest spreads -- concentrate trading
  D. Combine predictions: imbalance direction + DA-IDM spread as filter
  E. Predict imbalance SIGN + trade only extreme confidence hours
"""

import pandas as pd
import numpy as np
import psycopg2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
REPO_ROOT = SCRIPT_DIR.parents[2]

DB = {
    "host": "localhost", "port": 5432,
    "dbname": "DB_EMS", "user": "postgres", "password": "Kapitan1",
}

TARGET_DAILY_EUR = 200.0

plt.rcParams.update({
    "figure.figsize": (16, 10), "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})


def get_conn():
    return psycopg2.connect(**DB)


def max_drawdown(eq):
    peak = eq.cummax()
    return abs((eq - peak).min())


# ============================================================
# DATA LOADING
# ============================================================

def load_all_data():
    """Load order book + market prices, merge into one analysis DataFrame."""
    print("[*] Loading data...")

    with get_conn() as conn:
        ob = pd.read_sql("""
            WITH ob AS (
                SELECT tradeday::date AS tradeday, periodfrom AS hour,
                       tradetype, price, lastupdate,
                       tradeday::date + periodfrom * INTERVAL '1 hour' AS delivery_start
                FROM db_ems.vdt_isot_knihaobjednavok_best
                WHERE deliverydur = 60 AND id_depth = 1 AND price > 0 AND price < 1000
            ),
            with_ttd AS (
                SELECT *,
                    EXTRACT(EPOCH FROM (delivery_start - lastupdate)) / 3600.0 AS htd
                FROM ob WHERE lastupdate < delivery_start
            ),
            bucketed AS (
                SELECT *, CASE
                    WHEN htd >= 2 AND htd < 4.5 THEN 3
                    WHEN htd >= 0.75 AND htd < 2 THEN 1
                    ELSE NULL END AS ttd
                FROM with_ttd
            )
            SELECT tradeday, hour, ttd, tradetype,
                   AVG(price) AS avg_price, MIN(price) AS min_price,
                   MAX(price) AS max_price, STDDEV(price) AS std_price,
                   COUNT(*) AS n
            FROM bucketed WHERE ttd IS NOT NULL
            GROUP BY tradeday, hour, ttd, tradetype
        """, conn)

    print(f"[+] OB: {len(ob)} rows")

    # Pivot to bid/ask
    bids = ob[ob['tradetype'] == 'N'].copy()
    asks = ob[ob['tradetype'] == 'P'].copy()

    def pivot_ttd(df, prefix, ttd_val):
        sub = df[df['ttd'] == ttd_val][['tradeday', 'hour', 'avg_price', 'min_price', 'max_price', 'std_price']].copy()
        sub.columns = ['tradeday', 'hour', f'{prefix}_avg', f'{prefix}_min', f'{prefix}_max', f'{prefix}_std']
        return sub

    bid3 = pivot_ttd(bids, 'bid3', 3)
    ask3 = pivot_ttd(asks, 'ask3', 3)
    bid1 = pivot_ttd(bids, 'bid1', 1)
    ask1 = pivot_ttd(asks, 'ask1', 1)

    m = bid3.merge(ask3, on=['tradeday', 'hour'], how='outer')
    m = m.merge(bid1, on=['tradeday', 'hour'], how='outer')
    m = m.merge(ask1, on=['tradeday', 'hour'], how='outer')

    # Load market prices
    path = REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv"
    mkt = pd.read_csv(path)
    tc = 'datetime' if 'datetime' in mkt.columns else 'timestamp_hour'
    mkt[tc] = pd.to_datetime(mkt[tc])
    mkt['tradeday'] = mkt[tc].dt.date
    if 'hour' not in mkt.columns:
        mkt['hour'] = mkt[tc].dt.hour

    m['tradeday'] = pd.to_datetime(m['tradeday']).dt.date
    mkt['tradeday'] = pd.to_datetime(mkt['tradeday']).dt.date

    df = m.merge(mkt[['tradeday', 'hour', 'da_price', 'imb_settlement_price', 'imbalance_mwh']],
                 on=['tradeday', 'hour'], how='inner')

    df = df.dropna(subset=['bid1_avg', 'ask1_avg', 'imb_settlement_price', 'da_price'])
    df['year'] = pd.to_datetime(df['tradeday']).dt.year
    df['date'] = pd.to_datetime(df['tradeday'])

    # Derived
    df['imb_abs'] = df['imbalance_mwh'].abs()
    df['is_deficit'] = (df['imbalance_mwh'] < 0).astype(int)
    df['idm_vol_1h'] = df['bid1_std'].fillna(0) + df['ask1_std'].fillna(0)
    df['da_idm_spread'] = df['bid1_avg'] - df['da_price']
    df['imb_idm_spread'] = df['imb_settlement_price'] - df['ask1_avg']

    # P&L for imbalance direction trade
    df['pnl_correct_dir'] = np.where(
        df['is_deficit'],
        df['imb_settlement_price'] - df['ask1_avg'],  # deficit: buy IDM, settle at higher imb
        df['bid1_avg'] - df['imb_settlement_price'],   # surplus: sell IDM, settle at lower imb
    )
    df['pnl_incorrect_dir'] = np.where(
        df['is_deficit'],
        df['bid1_avg'] - df['imb_settlement_price'],
        df['imb_settlement_price'] - df['ask1_avg'],
    )

    print(f"[+] Merged: {len(df)} rows, {df['tradeday'].nunique()} days")
    return df


# ============================================================
# ANALYSIS 1: CONFIDENCE-BASED POSITION SIZING
# ============================================================

def analysis_confidence_sizing(df):
    """What if we size by prediction confidence?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Confidence-Based Position Sizing")
    print("=" * 70)

    # Simulate: model predicts imbalance direction with some accuracy.
    # High-confidence trades are the ones where edge is largest.
    # Strategy: rank all trades by |pnl_correct - pnl_incorrect| (edge),
    # assume the model's confidence correlates with actual edge.

    edge = (df['pnl_correct_dir'] - df['pnl_incorrect_dir']).values
    pnl_c = df['pnl_correct_dir'].values
    pnl_i = df['pnl_incorrect_dir'].values
    dates = df['date'].values

    # Sort by edge magnitude (proxy for confidence)
    order = np.argsort(np.abs(edge))[::-1]  # highest edge first

    accuracy = 0.65  # realistic accuracy

    results = []
    # Test: trade only top N% of hours (highest confidence)
    for top_pct in [1.0, 0.75, 0.50, 0.25, 0.10]:
        n_trade = int(top_pct * len(order))
        idx = order[:n_trade]

        n_correct = int(accuracy * n_trade)
        pnl = np.empty(n_trade)
        pnl[:n_correct] = pnl_c[idx[:n_correct]]
        pnl[n_correct:] = pnl_i[idx[n_correct:]]

        # Daily P&L
        df_tmp = pd.DataFrame({'date': dates[idx], 'pnl': pnl})
        daily = df_tmp.groupby('date')['pnl'].sum()

        avg = pnl.mean()
        hours_per_day = n_trade / df['tradeday'].nunique()
        daily_1mwh = avg * hours_per_day

        if daily_1mwh > 0:
            mwh = TARGET_DAILY_EUR / daily_1mwh
        else:
            mwh = float('inf')

        sharpe = (daily.mean() / daily.std()) * np.sqrt(365) if daily.std() > 0 else 0
        md = max_drawdown(daily.cumsum())

        results.append({
            'top_pct': top_pct,
            'n_trades': n_trade,
            'hours_per_day': hours_per_day,
            'avg_pnl_per_mwh': avg,
            'daily_1mwh': daily_1mwh,
            'mwh_for_200': mwh,
            'sharpe': sharpe,
            'max_dd_1mwh': md,
            'win_rate': (daily > 0).mean(),
        })

    res = pd.DataFrame(results)
    print("\n  Trade only the highest-confidence hours (65% base accuracy):")
    print(f"  {'Top%':>6} {'Hrs/day':>8} {'EUR/MWh':>9} {'EUR/day@1MW':>12} {'MWh@200':>8} {'Sharpe':>7} {'WinRate':>8} {'MaxDD':>8}")
    for _, r in res.iterrows():
        mwh_str = f"{r['mwh_for_200']:.1f}" if not np.isinf(r['mwh_for_200']) else ">999"
        print(f"  {r['top_pct']:>5.0%} {r['hours_per_day']:>8.1f} {r['avg_pnl_per_mwh']:>+9.1f} {r['daily_1mwh']:>+12.1f} {mwh_str:>8} {r['sharpe']:>7.1f} {r['win_rate']:>7.0%} {r['max_dd_1mwh']:>8.0f}")

    res.to_csv(DATA_DIR / "confidence_sizing.csv", index=False)

    # Now test: variable sizing proportional to confidence
    print("\n  Variable sizing (Kelly-inspired): size proportional to edge")
    for acc in [0.60, 0.65, 0.70, 0.75]:
        n_correct = int(acc * len(order))
        pnl = np.empty(len(order))
        pnl[:n_correct] = pnl_c[order[:n_correct]]
        pnl[n_correct:] = pnl_i[order[n_correct:]]

        # Size = edge / max_edge (normalized 0-1)
        edge_abs = np.abs(edge[order])
        max_edge = np.percentile(edge_abs, 95)  # clip at 95th to avoid outliers
        sizes = np.clip(edge_abs / max_edge, 0.05, 1.0)  # min 5% of max position

        weighted_pnl = pnl * sizes
        df_tmp = pd.DataFrame({'date': dates[order], 'pnl': weighted_pnl})
        daily = df_tmp.groupby('date')['pnl'].sum()

        avg_daily = daily.mean()
        sharpe = (daily.mean() / daily.std()) * np.sqrt(365) if daily.std() > 0 else 0

        # What base position for 200/day?
        if avg_daily > 0:
            base_mwh = TARGET_DAILY_EUR / avg_daily
        else:
            base_mwh = float('inf')

        base_str = f"{base_mwh:.1f}" if not np.isinf(base_mwh) else ">999"
        print(f"    Accuracy {acc:.0%}: avg daily {avg_daily:+.1f} EUR @1MWh base, need {base_str} MWh base, Sharpe {sharpe:.1f}")

    return res


# ============================================================
# ANALYSIS 2: PREDICT IMBALANCE MAGNITUDE
# ============================================================

def analysis_magnitude(df):
    """What if we can predict imbalance magnitude (large vs small)?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Trading Only Large Imbalance Hours")
    print("=" * 70)

    # Hypothesis: large imbalances have bigger price impact,
    # so trading only when |imbalance| is predicted to be large gives better P&L per trade

    print("\n  P&L by imbalance magnitude (at 65% direction accuracy):")
    print(f"  {'Bucket':<20} {'N':>6} {'Avg correct':>12} {'Avg incorrect':>14} {'Edge':>8} {'Pct of hours':>12}")

    thresholds = [0, 5, 10, 20, 50]
    bucket_results = []

    for i in range(len(thresholds)):
        lo = thresholds[i]
        hi = thresholds[i + 1] if i + 1 < len(thresholds) else 999

        mask = (df['imb_abs'] >= lo) & (df['imb_abs'] < hi)
        sub = df[mask]
        if len(sub) == 0:
            continue

        avg_c = sub['pnl_correct_dir'].mean()
        avg_i = sub['pnl_incorrect_dir'].mean()
        edge = avg_c - avg_i
        pct = len(sub) / len(df) * 100

        label = f"|imb| {lo}-{hi} MWh"
        print(f"  {label:<20} {len(sub):>6} {avg_c:>+12.1f} {avg_i:>+14.1f} {edge:>+8.1f} {pct:>11.1f}%")

        bucket_results.append({
            'bucket': label, 'n': len(sub), 'avg_correct': avg_c,
            'avg_incorrect': avg_i, 'edge': edge, 'pct_hours': pct,
        })

    # Also: what if we ONLY trade hours with |imbalance| > X?
    print("\n  Strategy: only trade when predicted |imbalance| > threshold")
    print(f"  {'Threshold':>10} {'N trades':>9} {'Hrs/day':>8} {'Edge':>8} {'EUR/day@1MW':>12} {'MWh@200':>8} {'Sharpe':>7}")

    for thresh in [0, 5, 10, 15, 20, 30]:
        mask = df['imb_abs'] >= thresh
        sub = df[mask].copy()
        if len(sub) < 50:
            continue

        acc = 0.65
        pnl_c = sub['pnl_correct_dir'].values
        pnl_i = sub['pnl_incorrect_dir'].values
        edge_arr = pnl_c - pnl_i
        order = np.argsort(edge_arr)[::-1]
        n_correct = int(acc * len(order))
        pnl = np.empty(len(order))
        pnl[:n_correct] = pnl_c[order[:n_correct]]
        pnl[n_correct:] = pnl_i[order[n_correct:]]

        df_tmp = pd.DataFrame({'date': sub['date'].values, 'pnl': pnl})
        daily = df_tmp.groupby('date')['pnl'].sum()

        hours_per_day = len(sub) / df['tradeday'].nunique()
        avg_daily = daily.mean()
        sharpe = (daily.mean() / daily.std()) * np.sqrt(365) if daily.std() > 0 else 0
        mwh = TARGET_DAILY_EUR / avg_daily if avg_daily > 0 else float('inf')
        mwh_str = f"{mwh:.1f}" if not np.isinf(mwh) else ">999"

        print(f"  {thresh:>8} MW {len(sub):>9} {hours_per_day:>8.1f} {(pnl_c - pnl_i).mean():>+8.1f} {avg_daily:>+12.1f} {mwh_str:>8} {sharpe:>7.1f}")

    pd.DataFrame(bucket_results).to_csv(DATA_DIR / "magnitude_buckets.csv", index=False)


# ============================================================
# ANALYSIS 3: PREDICT IDM VOLATILITY -> ROUND-TRIP
# ============================================================

def analysis_volatility(df):
    """Can we predict which hours will have high IDM volatility?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: IDM Volatility as Trading Filter")
    print("=" * 70)

    # idm_vol_1h = bid std + ask std at TTD=1h
    df_vol = df.dropna(subset=['idm_vol_1h']).copy()
    df_vol['vol_q'] = pd.qcut(df_vol['idm_vol_1h'], q=5, labels=['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high'])

    print("\n  Edge by IDM volatility quintile (imbalance direction trade @65%):")
    print(f"  {'Quintile':<10} {'N':>6} {'Avg vol':>8} {'Correct':>10} {'Incorrect':>12} {'Edge':>8}")

    for q in ['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high']:
        sub = df_vol[df_vol['vol_q'] == q]
        if len(sub) == 0:
            continue
        print(f"  {q:<10} {len(sub):>6} {sub['idm_vol_1h'].mean():>8.1f} {sub['pnl_correct_dir'].mean():>+10.1f} {sub['pnl_incorrect_dir'].mean():>+12.1f} {(sub['pnl_correct_dir'] - sub['pnl_incorrect_dir']).mean():>+8.1f}")


# ============================================================
# ANALYSIS 4: COMBINE PREDICTIONS
# ============================================================

def analysis_combinations(df):
    """What if we combine multiple prediction signals?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Combining Prediction Signals")
    print("=" * 70)

    # Strategy: only trade imbalance direction when DA-IDM spread AGREES
    # i.e., if we predict deficit (buy IDM) AND IDM is already above DA (confirming),
    # that's a stronger signal

    df_c = df.dropna(subset=['bid3_avg', 'ask3_avg']).copy()

    # DA-IDM spread direction
    df_c['da_idm_dir'] = np.where(df_c['bid3_avg'] > df_c['da_price'], 1, -1)  # 1 = IDM > DA

    # Imbalance direction expected effect
    df_c['imb_dir_signal'] = np.where(df_c['is_deficit'], 1, -1)  # deficit -> expect higher imb -> buy IDM

    # Agreement = both signals say the same direction
    df_c['agree'] = (df_c['da_idm_dir'] == df_c['imb_dir_signal']).astype(int)

    agree = df_c[df_c['agree'] == 1]
    disagree = df_c[df_c['agree'] == 0]

    print(f"\n  Signals agree:    {len(agree)} hours ({len(agree)/len(df_c)*100:.0f}%)")
    print(f"  Signals disagree: {len(disagree)} hours ({len(disagree)/len(df_c)*100:.0f}%)")

    for label, sub in [("AGREE", agree), ("DISAGREE", disagree), ("ALL", df_c)]:
        if len(sub) < 50:
            continue
        acc = 0.65
        pnl_c = sub['pnl_correct_dir'].values
        pnl_i = sub['pnl_incorrect_dir'].values
        edge_arr = pnl_c - pnl_i
        order = np.argsort(edge_arr)[::-1]
        n_correct = int(acc * len(order))
        pnl = np.empty(len(order))
        pnl[:n_correct] = pnl_c[order[:n_correct]]
        pnl[n_correct:] = pnl_i[order[n_correct:]]

        df_tmp = pd.DataFrame({'date': sub['date'].values, 'pnl': pnl})
        daily = df_tmp.groupby('date')['pnl'].sum()
        sharpe = (daily.mean() / daily.std()) * np.sqrt(365) if daily.std() > 0 else 0
        hrs = len(sub) / df_c['tradeday'].nunique()
        avg_daily = daily.mean()
        mwh = TARGET_DAILY_EUR / avg_daily if avg_daily > 0 else float('inf')
        mwh_str = f"{mwh:.1f}" if not np.isinf(mwh) else ">999"

        print(f"    {label:<10}: edge={pnl.mean():+.1f}/MWh, {hrs:.1f} hrs/day, daily@1MW={avg_daily:+.1f}, MWh@200={mwh_str}, Sharpe={sharpe:.1f}")

    # Test: imbalance direction + magnitude filter + DA-IDM agreement
    print("\n  Triple filter: Imb direction + |imb|>10 + DA-IDM agrees:")
    triple = df_c[(df_c['agree'] == 1) & (df_c['imb_abs'] >= 10)]
    if len(triple) > 50:
        pnl_c = triple['pnl_correct_dir'].values
        pnl_i = triple['pnl_incorrect_dir'].values
        edge_arr = pnl_c - pnl_i
        order = np.argsort(edge_arr)[::-1]
        n_correct = int(0.65 * len(order))
        pnl = np.empty(len(order))
        pnl[:n_correct] = pnl_c[order[:n_correct]]
        pnl[n_correct:] = pnl_i[order[n_correct:]]

        df_tmp = pd.DataFrame({'date': triple['date'].values, 'pnl': pnl})
        daily = df_tmp.groupby('date')['pnl'].sum()
        sharpe = (daily.mean() / daily.std()) * np.sqrt(365) if daily.std() > 0 else 0
        hrs = len(triple) / df_c['tradeday'].nunique()
        avg_daily = daily.mean()
        mwh = TARGET_DAILY_EUR / avg_daily if avg_daily > 0 else float('inf')
        mwh_str = f"{mwh:.1f}" if not np.isinf(mwh) else ">999"

        print(f"    N={len(triple)}, {hrs:.1f} hrs/day, edge={pnl.mean():+.1f}/MWh, daily@1MW={avg_daily:+.1f}, MWh@200={mwh_str}, Sharpe={sharpe:.1f}")


# ============================================================
# ANALYSIS 5: HOUR SELECTION -- BEST N HOURS PER DAY
# ============================================================

def analysis_hour_selection(df):
    """What if we only trade the N best hours per day?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Selective Trading -- Best N Hours Per Day")
    print("=" * 70)

    # If we could perfectly identify the best N hours each day to trade,
    # how much better would we do?

    acc = 0.65
    df_s = df.copy()
    df_s['edge'] = df_s['pnl_correct_dir'] - df_s['pnl_incorrect_dir']

    print(f"\n  (65% direction accuracy, selecting top-edge hours per day)")
    print(f"  {'Hours/day':>10} {'Avg edge':>10} {'EUR/day@1MW':>12} {'MWh@200':>8} {'Sharpe':>7}")

    for n_hours in [24, 16, 12, 8, 6, 4, 2, 1]:
        # Select top N hours by edge each day
        top = df_s.groupby('tradeday').apply(
            lambda g: g.nlargest(min(n_hours, len(g)), 'edge'), include_groups=False
        ).reset_index(drop=True)

        pnl_c = top['pnl_correct_dir'].values
        pnl_i = top['pnl_incorrect_dir'].values
        edge_arr = pnl_c - pnl_i
        order = np.argsort(edge_arr)[::-1]
        n_correct = int(acc * len(order))
        pnl = np.empty(len(order))
        pnl[:n_correct] = pnl_c[order[:n_correct]]
        pnl[n_correct:] = pnl_i[order[n_correct:]]

        df_tmp = pd.DataFrame({'date': top['date'].values, 'pnl': pnl})
        daily = df_tmp.groupby('date')['pnl'].sum()
        sharpe = (daily.mean() / daily.std()) * np.sqrt(365) if daily.std() > 0 else 0
        avg_daily = daily.mean()
        mwh = TARGET_DAILY_EUR / avg_daily if avg_daily > 0 else float('inf')
        mwh_str = f"{mwh:.1f}" if not np.isinf(mwh) else ">999"
        avg_edge = top['edge'].mean()

        print(f"  {n_hours:>10} {avg_edge:>+10.1f} {avg_daily:>+12.1f} {mwh_str:>8} {sharpe:>7.1f}")


# ============================================================
# ANALYSIS 6: BY YEAR BREAKDOWN OF BEST STRATEGIES
# ============================================================

def analysis_by_year(df):
    """Show the best strategies broken down by year."""
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Best Strategies by Year")
    print("=" * 70)

    acc = 0.65

    strategies = {
        'Flat 24h/day': lambda d: d,
        'Top 8 hrs/day': lambda d: d.groupby('tradeday').apply(
            lambda g: g.nlargest(min(8, len(g)), 'edge'), include_groups=False).reset_index(drop=True),
        '|imb| > 10 MWh': lambda d: d[d['imb_abs'] >= 10],
        'Top 4 hrs/day': lambda d: d.groupby('tradeday').apply(
            lambda g: g.nlargest(min(4, len(g)), 'edge'), include_groups=False).reset_index(drop=True),
    }

    df['edge'] = df['pnl_correct_dir'] - df['pnl_incorrect_dir']

    print(f"\n  {'Strategy':<20} {'Year':>5} {'Hrs/day':>8} {'EUR/MWh':>9} {'Daily@1MW':>10} {'MWh@200':>8} {'Sharpe':>7}")
    print(f"  {'-'*80}")

    for strat_name, filter_fn in strategies.items():
        for year in [2024, 2025, 2026]:
            sub_year = df[df['year'] == year].copy()
            if len(sub_year) < 50:
                continue

            filtered = filter_fn(sub_year)
            if len(filtered) < 20:
                continue

            pnl_c = filtered['pnl_correct_dir'].values
            pnl_i = filtered['pnl_incorrect_dir'].values
            edge_arr = pnl_c - pnl_i
            order = np.argsort(edge_arr)[::-1]
            n_correct = int(acc * len(order))
            pnl = np.empty(len(order))
            pnl[:n_correct] = pnl_c[order[:n_correct]]
            pnl[n_correct:] = pnl_i[order[n_correct:]]

            df_tmp = pd.DataFrame({'date': filtered['date'].values, 'pnl': pnl})
            daily = df_tmp.groupby('date')['pnl'].sum()
            sharpe = (daily.mean() / daily.std()) * np.sqrt(365) if daily.std() > 0 else 0
            hrs = len(filtered) / sub_year['tradeday'].nunique()
            avg_daily = daily.mean()
            mwh = TARGET_DAILY_EUR / avg_daily if avg_daily > 0 else float('inf')
            mwh_str = f"{mwh:.1f}" if not np.isinf(mwh) else ">999"

            print(f"  {strat_name:<20} {year:>5} {hrs:>8.1f} {pnl.mean():>+9.1f} {avg_daily:>+10.1f} {mwh_str:>8} {sharpe:>7.1f}")
        print()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ADVANCED SIZING & CREATIVE PREDICTION TARGETS")
    print("=" * 70)

    df = load_all_data()

    analysis_confidence_sizing(df)
    analysis_magnitude(df)
    analysis_volatility(df)
    analysis_combinations(df)
    analysis_hour_selection(df)
    analysis_by_year(df)

    print("\n" + "=" * 70)
    print("[+] Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
