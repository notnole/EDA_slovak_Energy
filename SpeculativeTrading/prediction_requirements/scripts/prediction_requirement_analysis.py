"""
Prediction Requirement Analysis for Speculative Electricity Trading
===================================================================

Goal: EUR 200/day from the Slovak electricity market (OKTE).
Question: What do we need to predict, and how accurately?

Works backward from P&L target through actual bid/ask order book data
to produce a decision matrix showing which prediction targets are worth pursuing.

7 Prediction Targets:
  1. DA->IDM spread direction
  2. IDM price movement between TTD windows
  3. IDM binary direction
  4. IDM round-trip (session max/min timing)
  5. Imbalance price -> IDM position
  6. Imbalance direction -> IDM position
  7. DA-IDM spread direction (improve existing)
"""

import pandas as pd
import numpy as np
import psycopg2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# --- Config ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
REPO_ROOT = SCRIPT_DIR.parents[2]

DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "DB_EMS",
    "user": "postgres",
    "password": "Kapitan1",
}

TARGET_DAILY_EUR = 200.0
ACCURACY_LEVELS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
TRADING_HOURS_PER_DAY = 24

plt.rcParams.update({
    "figure.figsize": (14, 8),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ============================================================
# SHARED INFRASTRUCTURE
# ============================================================

def get_conn():
    return psycopg2.connect(**DB)


def max_drawdown(equity_curve):
    """Return max drawdown (positive number) and duration in days."""
    peak = equity_curve.cummax()
    dd = equity_curve - peak
    max_dd = dd.min()
    if max_dd == 0:
        return 0.0, 0
    is_in_dd = dd < 0
    groups = (~is_in_dd).cumsum()
    if is_in_dd.any():
        dd_lengths = is_in_dd.groupby(groups).sum()
        max_dur = int(dd_lengths.max())
    else:
        max_dur = 0
    return abs(max_dd), max_dur


def compute_risk_metrics(daily_pnl):
    """Compute risk metrics for a daily P&L series."""
    if len(daily_pnl) == 0 or daily_pnl.std() == 0:
        return {
            'total_pnl': 0, 'mean_daily': 0, 'std_daily': 0,
            'sharpe': 0, 'max_dd': 0, 'max_dd_days': 0,
            'worst_week': 0, 'win_rate': 0, 'profit_factor': 0,
        }

    equity = daily_pnl.cumsum()
    dd, dd_days = max_drawdown(equity)

    wins = daily_pnl[daily_pnl > 0]
    losses = daily_pnl[daily_pnl < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    # Worst rolling 7-day P&L
    if len(daily_pnl) >= 7:
        worst_week = daily_pnl.rolling(7).sum().min()
    else:
        worst_week = daily_pnl.sum()

    return {
        'total_pnl': daily_pnl.sum(),
        'mean_daily': daily_pnl.mean(),
        'std_daily': daily_pnl.std(),
        'sharpe': (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(365),
        'max_dd': dd,
        'max_dd_days': dd_days,
        'worst_week': worst_week,
        'win_rate': (daily_pnl > 0).mean(),
        'profit_factor': pf,
    }


def simulate_accuracy(pnl_correct, pnl_incorrect, accuracy):
    """
    Given P&L arrays for correct and incorrect predictions,
    simulate blended P&L at a given accuracy level.

    Deterministic: sort trades by profitability, assign the best `accuracy` fraction
    as correct calls and worst fraction as incorrect.

    Returns: array of per-trade P&L at this accuracy level.
    """
    n = len(pnl_correct)
    if n == 0:
        return np.array([])

    # The spread between correct and incorrect call
    edge = pnl_correct - pnl_incorrect  # positive means correct call is better

    # Sort by edge (best opportunities first)
    order = np.argsort(edge)[::-1]
    n_correct = int(accuracy * n)

    result = np.empty(n)
    result[order[:n_correct]] = pnl_correct[order[:n_correct]]
    result[order[n_correct:]] = pnl_incorrect[order[n_correct:]]

    return result


def accuracy_sweep(pnl_correct, pnl_incorrect, dates, hours):
    """Run accuracy sweep and return results DataFrame."""
    rows = []
    for acc in ACCURACY_LEVELS:
        trade_pnl = simulate_accuracy(pnl_correct, pnl_incorrect, acc)

        # Aggregate to daily
        df_tmp = pd.DataFrame({
            'date': dates, 'hour': hours, 'pnl': trade_pnl
        })
        daily = df_tmp.groupby('date')['pnl'].sum()

        metrics = compute_risk_metrics(daily)
        avg_per_mwh = trade_pnl.mean()
        n_hours = TRADING_HOURS_PER_DAY

        if avg_per_mwh > 0:
            mwh_needed = TARGET_DAILY_EUR / (avg_per_mwh * n_hours)
        else:
            mwh_needed = float('inf')

        rows.append({
            'accuracy': acc,
            'avg_pnl_per_mwh': avg_per_mwh,
            'mwh_for_200': mwh_needed,
            'daily_pnl_1mwh': metrics['mean_daily'],
            'sharpe': metrics['sharpe'],
            'max_dd_1mwh': metrics['max_dd'],
            'worst_week_1mwh': metrics['worst_week'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'n_trades': len(trade_pnl),
        })

    return pd.DataFrame(rows)


def find_breakeven(sweep_df):
    """Find the accuracy where avg_pnl_per_mwh crosses zero."""
    for i in range(len(sweep_df) - 1):
        if sweep_df.iloc[i]['avg_pnl_per_mwh'] <= 0 and sweep_df.iloc[i + 1]['avg_pnl_per_mwh'] > 0:
            # Linear interpolation
            a1 = sweep_df.iloc[i]['accuracy']
            a2 = sweep_df.iloc[i + 1]['accuracy']
            p1 = sweep_df.iloc[i]['avg_pnl_per_mwh']
            p2 = sweep_df.iloc[i + 1]['avg_pnl_per_mwh']
            return a1 + (a2 - a1) * (-p1) / (p2 - p1)
    if sweep_df.iloc[0]['avg_pnl_per_mwh'] > 0:
        return 0.50  # Already profitable at 50%
    return 0.95  # Never profitable in range


# ============================================================
# DATA LOADING
# ============================================================

def load_orderbook_snapshots():
    """Load order book bid/ask at various TTD windows from PostgreSQL."""
    print("[*] Loading order book snapshots from DB...")
    print("    (this queries 92M rows, may take a few minutes)")

    with get_conn() as conn:
        # Get the last bid and last ask before each TTD cutoff
        # For each (tradeday, hour, ttd_cutoff), find the most recent snapshot
        df = pd.read_sql("""
            WITH ob AS (
                SELECT
                    tradeday::date AS tradeday,
                    periodfrom AS hour,
                    tradetype,
                    price,
                    lastupdate,
                    tradeday::date + periodfrom * INTERVAL '1 hour' AS delivery_start
                FROM db_ems.vdt_isot_knihaobjednavok_best
                WHERE deliverydur = 60
                AND id_depth = 1
            ),
            with_ttd AS (
                SELECT *,
                    EXTRACT(EPOCH FROM (delivery_start - lastupdate)) / 3600.0 AS hours_to_delivery
                FROM ob
                WHERE lastupdate < delivery_start
                AND price > 0
                AND price < 1000
            ),
            bucketed AS (
                SELECT *,
                    CASE
                        WHEN hours_to_delivery >= 20 THEN 24
                        WHEN hours_to_delivery >= 9 THEN 12
                        WHEN hours_to_delivery >= 4.5 THEN 6
                        WHEN hours_to_delivery >= 2 THEN 3
                        WHEN hours_to_delivery >= 0.75 THEN 1
                        ELSE 0.5
                    END AS ttd_bucket
                FROM with_ttd
            )
            SELECT
                tradeday,
                hour,
                ttd_bucket,
                tradetype,
                AVG(price) AS avg_price,
                MIN(price) AS min_price,
                MAX(price) AS max_price,
                COUNT(*) AS n_snapshots
            FROM bucketed
            GROUP BY tradeday, hour, ttd_bucket, tradetype
            ORDER BY tradeday, hour, ttd_bucket
        """, conn)

    print(f"[+] Order book: {len(df)} rows loaded")
    return df


def pivot_bid_ask(ob_df):
    """Pivot order book into bid/ask columns per (tradeday, hour, ttd_bucket)."""
    bids = ob_df[ob_df['tradetype'] == 'N'].rename(columns={
        'avg_price': 'bid', 'min_price': 'bid_min', 'max_price': 'bid_max',
        'n_snapshots': 'n_bid'
    }).drop(columns=['tradetype'])

    asks = ob_df[ob_df['tradetype'] == 'P'].rename(columns={
        'avg_price': 'ask', 'min_price': 'ask_min', 'max_price': 'ask_max',
        'n_snapshots': 'n_ask'
    }).drop(columns=['tradetype'])

    merged = bids.merge(asks, on=['tradeday', 'hour', 'ttd_bucket'], how='outer')
    merged['mid'] = (merged['bid'].fillna(merged['ask']) + merged['ask'].fillna(merged['bid'])) / 2
    merged['spread'] = merged['ask'] - merged['bid']

    return merged


def load_market_prices():
    """Load DA + IDM VWAP + Imbalance settlement prices."""
    print("[*] Loading market prices...")
    path = REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv"
    df = pd.read_csv(path)

    time_col = 'datetime' if 'datetime' in df.columns else 'timestamp_hour'
    df[time_col] = pd.to_datetime(df[time_col])
    df['tradeday'] = df[time_col].dt.date
    if 'hour' not in df.columns:
        df['hour'] = df[time_col].dt.hour

    print(f"[+] Market prices: {len(df)} rows, {df['tradeday'].nunique()} days")
    return df


def load_round_trip_data():
    """Load session min ask and max bid per delivery period."""
    print("[*] Loading round-trip (session min/max) data...")

    with get_conn() as conn:
        df = pd.read_sql("""
            WITH ob AS (
                SELECT
                    tradeday::date AS tradeday,
                    periodfrom AS hour,
                    tradetype,
                    price
                FROM db_ems.vdt_isot_knihaobjednavok_best
                WHERE deliverydur = 60
                AND id_depth = 1
                AND price > 0
                AND price < 1000
            )
            SELECT
                tradeday,
                hour,
                tradetype,
                MIN(price) AS min_price,
                MAX(price) AS max_price,
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY price) AS p10,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) AS p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY price) AS p50,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) AS p75,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY price) AS p90,
                COUNT(*) AS n
            FROM ob
            GROUP BY tradeday, hour, tradetype
            ORDER BY tradeday, hour
        """, conn)

    print(f"[+] Round-trip data: {len(df)} rows")
    return df


# ============================================================
# TARGET ANALYSES
# ============================================================

def target1_da_to_idm(ob_pivot, market_df):
    """Target 1: Predict DA-IDM spread direction. Buy DA, sell IDM (or vice versa)."""
    print("\n--- Target 1: DA -> IDM spread direction ---")

    # Use TTD=3h as exit point on IDM (liquid, tight spread)
    idm = ob_pivot[ob_pivot['ttd_bucket'] == 3.0][['tradeday', 'hour', 'bid', 'ask']].copy()
    idm.columns = ['tradeday', 'hour', 'idm_bid', 'idm_ask']

    da = market_df[['tradeday', 'hour', 'da_price']].drop_duplicates()
    da['tradeday'] = pd.to_datetime(da['tradeday']).dt.date

    m = idm.merge(da, on=['tradeday', 'hour'], how='inner').dropna(subset=['idm_bid', 'idm_ask', 'da_price'])

    # Correct call: buy DA, sell IDM when IDM bid > DA (profit = idm_bid - da_price)
    # OR: sell DA, buy IDM when DA > IDM ask (profit = da_price - idm_ask)
    pnl_long = (m['idm_bid'] - m['da_price']).values   # buy DA, sell IDM
    pnl_short = (m['da_price'] - m['idm_ask']).values   # sell DA, buy IDM

    # For each hour, correct direction = whichever is positive (or more positive)
    pnl_correct = np.maximum(pnl_long, pnl_short)
    pnl_incorrect = np.minimum(pnl_long, pnl_short)

    dates = pd.to_datetime(m['tradeday']).values
    hours = m['hour'].values

    sweep = accuracy_sweep(pnl_correct, pnl_incorrect, dates, hours)
    sweep['target'] = 'T1: DA->IDM'
    print(f"    Avg correct: {pnl_correct.mean():+.2f}, Avg incorrect: {pnl_incorrect.mean():+.2f}")
    print(f"    Breakeven accuracy: {find_breakeven(sweep):.1%}")
    return sweep


def target2_idm_ttd_movement(ob_pivot):
    """Target 2: Predict IDM price movement between TTD windows."""
    print("\n--- Target 2: IDM price movement (6h -> 1h) ---")

    entry_ttd = 6.0
    exit_ttd = 1.0

    entry = ob_pivot[ob_pivot['ttd_bucket'] == entry_ttd][['tradeday', 'hour', 'bid', 'ask', 'mid']].copy()
    entry.columns = ['tradeday', 'hour', 'entry_bid', 'entry_ask', 'entry_mid']

    exit_ = ob_pivot[ob_pivot['ttd_bucket'] == exit_ttd][['tradeday', 'hour', 'bid', 'ask', 'mid']].copy()
    exit_.columns = ['tradeday', 'hour', 'exit_bid', 'exit_ask', 'exit_mid']

    m = entry.merge(exit_, on=['tradeday', 'hour'], how='inner')
    m = m.dropna(subset=['entry_bid', 'entry_ask', 'exit_bid', 'exit_ask'])

    # Long: buy at entry ask, sell at exit bid
    pnl_long = (m['exit_bid'] - m['entry_ask']).values
    # Short: sell at entry bid, buy at exit ask
    pnl_short = (m['entry_bid'] - m['exit_ask']).values

    pnl_correct = np.maximum(pnl_long, pnl_short)
    pnl_incorrect = np.minimum(pnl_long, pnl_short)

    dates = pd.to_datetime(m['tradeday']).values
    hours = m['hour'].values

    sweep = accuracy_sweep(pnl_correct, pnl_incorrect, dates, hours)
    sweep['target'] = 'T2: IDM 6h->1h'
    print(f"    Avg correct: {pnl_correct.mean():+.2f}, Avg incorrect: {pnl_incorrect.mean():+.2f}")
    print(f"    Breakeven accuracy: {find_breakeven(sweep):.1%}")
    return sweep


def target3_idm_direction(ob_pivot):
    """Target 3: Binary IDM direction prediction (3h -> 1h)."""
    print("\n--- Target 3: IDM direction (3h -> 1h) ---")

    entry_ttd = 3.0
    exit_ttd = 1.0

    entry = ob_pivot[ob_pivot['ttd_bucket'] == entry_ttd][['tradeday', 'hour', 'bid', 'ask', 'mid']].copy()
    entry.columns = ['tradeday', 'hour', 'entry_bid', 'entry_ask', 'entry_mid']

    exit_ = ob_pivot[ob_pivot['ttd_bucket'] == exit_ttd][['tradeday', 'hour', 'bid', 'ask', 'mid']].copy()
    exit_.columns = ['tradeday', 'hour', 'exit_bid', 'exit_ask', 'exit_mid']

    m = entry.merge(exit_, on=['tradeday', 'hour'], how='inner')
    m = m.dropna(subset=['entry_bid', 'entry_ask', 'exit_bid', 'exit_ask'])

    pnl_long = (m['exit_bid'] - m['entry_ask']).values
    pnl_short = (m['entry_bid'] - m['exit_ask']).values

    pnl_correct = np.maximum(pnl_long, pnl_short)
    pnl_incorrect = np.minimum(pnl_long, pnl_short)

    dates = pd.to_datetime(m['tradeday']).values
    hours = m['hour'].values

    sweep = accuracy_sweep(pnl_correct, pnl_incorrect, dates, hours)
    sweep['target'] = 'T3: IDM dir 3h->1h'
    print(f"    Avg correct: {pnl_correct.mean():+.2f}, Avg incorrect: {pnl_incorrect.mean():+.2f}")
    print(f"    Breakeven accuracy: {find_breakeven(sweep):.1%}")
    return sweep


def target4_idm_round_trip(rt_df):
    """Target 4: IDM round-trip - buy at session min ask, sell at session max bid."""
    print("\n--- Target 4: IDM round-trip (session max/min timing) ---")

    bids = rt_df[rt_df['tradetype'] == 'N'][['tradeday', 'hour', 'min_price', 'max_price',
                                              'p10', 'p25', 'p50', 'p75', 'p90']].copy()
    bids.columns = ['tradeday', 'hour', 'bid_min', 'bid_max', 'bid_p10', 'bid_p25',
                    'bid_p50', 'bid_p75', 'bid_p90']

    asks = rt_df[rt_df['tradetype'] == 'P'][['tradeday', 'hour', 'min_price', 'max_price',
                                              'p10', 'p25', 'p50', 'p75', 'p90']].copy()
    asks.columns = ['tradeday', 'hour', 'ask_min', 'ask_max', 'ask_p10', 'ask_p25',
                    'ask_p50', 'ask_p75', 'ask_p90']

    m = bids.merge(asks, on=['tradeday', 'hour'], how='inner')

    # At various timing accuracies:
    # 100% = buy at ask_min, sell at bid_max (perfect)
    # 90% = buy at ask_p10, sell at bid_p90
    # 75% = buy at ask_p25, sell at bid_p75
    # 50% = buy at ask_p50, sell at bid_p50
    accuracy_map = {
        0.50: ('ask_p50', 'bid_p50'),
        0.55: ('ask_p50', 'bid_p50'),  # approximate
        0.60: ('ask_p50', 'bid_p50'),
        0.65: ('ask_p25', 'bid_p75'),  # approximate
        0.70: ('ask_p25', 'bid_p75'),
        0.75: ('ask_p25', 'bid_p75'),
        0.80: ('ask_p10', 'bid_p90'),
        0.85: ('ask_p10', 'bid_p90'),
        0.90: ('ask_p10', 'bid_p90'),
    }

    rows = []
    dates = pd.to_datetime(m['tradeday'])
    for acc in ACCURACY_LEVELS:
        buy_col, sell_col = accuracy_map[acc]
        trade_pnl = (m[sell_col] - m[buy_col]).values

        df_tmp = pd.DataFrame({'date': dates.values, 'pnl': trade_pnl})
        daily = df_tmp.groupby('date')['pnl'].sum()

        metrics = compute_risk_metrics(daily)
        avg_per_mwh = trade_pnl.mean()

        if avg_per_mwh > 0:
            mwh_needed = TARGET_DAILY_EUR / (avg_per_mwh * TRADING_HOURS_PER_DAY)
        else:
            mwh_needed = float('inf')

        rows.append({
            'accuracy': acc,
            'avg_pnl_per_mwh': avg_per_mwh,
            'mwh_for_200': mwh_needed,
            'daily_pnl_1mwh': metrics['mean_daily'],
            'sharpe': metrics['sharpe'],
            'max_dd_1mwh': metrics['max_dd'],
            'worst_week_1mwh': metrics['worst_week'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'n_trades': len(trade_pnl),
        })

    sweep = pd.DataFrame(rows)
    sweep['target'] = 'T4: IDM round-trip'

    perfect = (m['bid_max'] - m['ask_min']).mean()
    p75 = (m['bid_p75'] - m['ask_p25']).mean()
    print(f"    Perfect round-trip: {perfect:+.2f} EUR/MWh")
    print(f"    P75/P25 round-trip: {p75:+.2f} EUR/MWh")
    return sweep


def target5_imbalance_price(ob_pivot, market_df):
    """Target 5: Predict imbalance price, trade IDM."""
    print("\n--- Target 5: Imbalance price -> IDM position ---")

    idm = ob_pivot[ob_pivot['ttd_bucket'] == 1.0][['tradeday', 'hour', 'bid', 'ask']].copy()
    idm.columns = ['tradeday', 'hour', 'idm_bid', 'idm_ask']

    mkt = market_df[['tradeday', 'hour', 'imb_settlement_price']].drop_duplicates()
    mkt['tradeday'] = pd.to_datetime(mkt['tradeday']).dt.date

    m = idm.merge(mkt, on=['tradeday', 'hour'], how='inner')
    m = m.dropna(subset=['idm_bid', 'idm_ask', 'imb_settlement_price'])

    # Correct: sell IDM if imb < IDM (pocket the premium)
    # OR: buy IDM if imb > IDM (pocket the imbalance premium)
    pnl_sell = (m['idm_bid'] - m['imb_settlement_price']).values  # sell IDM, settle at imb
    pnl_buy = (m['imb_settlement_price'] - m['idm_ask']).values   # buy IDM, settle at imb

    pnl_correct = np.maximum(pnl_sell, pnl_buy)
    pnl_incorrect = np.minimum(pnl_sell, pnl_buy)

    dates = pd.to_datetime(m['tradeday']).values
    hours = m['hour'].values

    sweep = accuracy_sweep(pnl_correct, pnl_incorrect, dates, hours)
    sweep['target'] = 'T5: Imb price->IDM'
    print(f"    Avg correct: {pnl_correct.mean():+.2f}, Avg incorrect: {pnl_incorrect.mean():+.2f}")
    print(f"    Breakeven accuracy: {find_breakeven(sweep):.1%}")
    return sweep


def target6_imbalance_direction(ob_pivot, market_df):
    """Target 6: Predict imbalance direction (surplus/deficit)."""
    print("\n--- Target 6: Imbalance direction -> IDM position ---")

    idm = ob_pivot[ob_pivot['ttd_bucket'] == 1.0][['tradeday', 'hour', 'bid', 'ask']].copy()
    idm.columns = ['tradeday', 'hour', 'idm_bid', 'idm_ask']

    mkt = market_df[['tradeday', 'hour', 'imb_settlement_price', 'imbalance_mwh']].drop_duplicates()
    mkt['tradeday'] = pd.to_datetime(mkt['tradeday']).dt.date

    m = idm.merge(mkt, on=['tradeday', 'hour'], how='inner')
    m = m.dropna(subset=['idm_bid', 'idm_ask', 'imb_settlement_price', 'imbalance_mwh'])

    # In deficit (imbalance < 0): imb price tends high -> sell IDM (profit if imb_price > idm_bid... wait)
    # Actually: in deficit, system needs power, imb price rises.
    # Sell IDM at bid, settle at (higher) imbalance -> loss
    # Buy IDM at ask, settle at (higher) imbalance -> profit
    # In surplus: system has excess, imb price drops.
    # Buy IDM at ask, settle at (lower) imbalance -> loss
    # Sell IDM at bid, settle at (lower but could be negative) imbalance -> depends

    # Correct action when deficit (imbalance < 0): BUY IDM (imb price will be high)
    # Correct action when surplus (imbalance > 0): SELL IDM (imb price will be low)
    is_deficit = m['imbalance_mwh'].values < 0

    pnl_correct_deficit = (m['imb_settlement_price'] - m['idm_ask']).values  # buy IDM
    pnl_correct_surplus = (m['idm_bid'] - m['imb_settlement_price']).values  # sell IDM

    pnl_correct = np.where(is_deficit, pnl_correct_deficit, pnl_correct_surplus)
    pnl_incorrect = np.where(is_deficit, -pnl_correct_surplus, -pnl_correct_deficit)
    # Wrong direction: you did the opposite action
    pnl_incorrect = np.where(is_deficit,
                             (m['idm_bid'] - m['imb_settlement_price']).values,  # sold when should buy
                             (m['imb_settlement_price'] - m['idm_ask']).values)  # bought when should sell

    dates = pd.to_datetime(m['tradeday']).values
    hours = m['hour'].values

    sweep = accuracy_sweep(pnl_correct, pnl_incorrect, dates, hours)
    sweep['target'] = 'T6: Imb direction'
    print(f"    Avg correct: {pnl_correct.mean():+.2f}, Avg incorrect: {pnl_incorrect.mean():+.2f}")
    print(f"    Breakeven accuracy: {find_breakeven(sweep):.1%}")
    return sweep


def target7_da_idm_spread(ob_pivot, market_df):
    """Target 7: DA-IDM spread direction (improve existing strategy)."""
    print("\n--- Target 7: DA-IDM spread direction ---")

    # Use TTD=3h as IDM execution
    idm = ob_pivot[ob_pivot['ttd_bucket'] == 3.0][['tradeday', 'hour', 'bid', 'ask']].copy()
    idm.columns = ['tradeday', 'hour', 'idm_bid', 'idm_ask']

    da = market_df[['tradeday', 'hour', 'da_price']].drop_duplicates()
    da['tradeday'] = pd.to_datetime(da['tradeday']).dt.date

    m = idm.merge(da, on=['tradeday', 'hour'], how='inner').dropna(subset=['idm_bid', 'idm_ask', 'da_price'])

    # Spread = IDM - DA
    # Predict spread > 0: buy DA, sell IDM at bid -> P&L = idm_bid - da_price
    # Predict spread < 0: sell DA, buy IDM at ask -> P&L = da_price - idm_ask
    actual_spread = m['idm_bid'] - m['da_price']

    # When spread actually > 0: correct = buy DA sell IDM, incorrect = sell DA buy IDM
    # When spread actually < 0: correct = sell DA buy IDM, incorrect = buy DA sell IDM
    pnl_long = (m['idm_bid'] - m['da_price']).values
    pnl_short = (m['da_price'] - m['idm_ask']).values

    pnl_correct = np.where(actual_spread.values >= 0, pnl_long, pnl_short)
    pnl_incorrect = np.where(actual_spread.values >= 0, pnl_short, pnl_long)

    dates = pd.to_datetime(m['tradeday']).values
    hours = m['hour'].values

    sweep = accuracy_sweep(pnl_correct, pnl_incorrect, dates, hours)
    sweep['target'] = 'T7: DA-IDM spread dir'
    print(f"    Avg correct: {pnl_correct.mean():+.2f}, Avg incorrect: {pnl_incorrect.mean():+.2f}")
    print(f"    Breakeven accuracy: {find_breakeven(sweep):.1%}")
    return sweep


# ============================================================
# DECISION MATRIX & VISUALIZATION
# ============================================================

def build_decision_matrix(all_sweeps):
    """Build the final decision matrix from all target sweeps."""
    print("\n" + "=" * 70)
    print("DECISION MATRIX")
    print("=" * 70)

    rows = []
    for sweep in all_sweeps:
        target_name = sweep['target'].iloc[0]
        be = find_breakeven(sweep)

        row = {'target': target_name, 'breakeven_accuracy': be}

        for acc in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            s = sweep[sweep['accuracy'] == acc]
            if len(s) > 0:
                s = s.iloc[0]
                row[f'eur_mwh_{int(acc*100)}'] = s['avg_pnl_per_mwh']
                row[f'mwh_{int(acc*100)}'] = s['mwh_for_200']
                row[f'sharpe_{int(acc*100)}'] = s['sharpe']
                row[f'maxdd_{int(acc*100)}'] = s['max_dd_1mwh']

        rows.append(row)

    matrix = pd.DataFrame(rows)
    return matrix


def plot_results(all_sweeps, matrix):
    """Generate the 4 summary plots."""

    # --- Plot 1: Breakeven accuracy ---
    fig, ax = plt.subplots(figsize=(12, 6))
    targets = matrix['target'].values
    breakevens = matrix['breakeven_accuracy'].values
    colors = ['#2ecc71' if b < 0.55 else '#f39c12' if b < 0.65 else '#e74c3c' for b in breakevens]
    bars = ax.barh(range(len(targets)), breakevens, color=colors, alpha=0.8)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets)
    ax.set_xlabel("Breakeven Accuracy")
    ax.set_title("Minimum Accuracy Needed for Profitability (after bid-ask)")
    ax.axvline(0.50, color='gray', linestyle='--', alpha=0.5, label='50% (coin flip)')
    ax.axvline(0.55, color='orange', linestyle='--', alpha=0.5, label='55% (existing)')
    for i, (b, t) in enumerate(zip(breakevens, targets)):
        ax.text(b + 0.005, i, f'{b:.0%}', va='center', fontsize=10)
    ax.legend()
    ax.set_xlim(0.40, 1.0)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_breakeven_accuracy.png", dpi=150)
    plt.close()
    print(f"[+] Saved: {PLOT_DIR / '01_breakeven_accuracy.png'}")

    # --- Plot 2: P&L sensitivity ---
    fig, ax = plt.subplots(figsize=(14, 7))
    colors_line = plt.cm.tab10(np.linspace(0, 1, len(all_sweeps)))
    for i, sweep in enumerate(all_sweeps):
        name = sweep['target'].iloc[0]
        ax.plot(sweep['accuracy'] * 100, sweep['avg_pnl_per_mwh'],
                'o-', label=name, color=colors_line[i], linewidth=2, markersize=5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel("Prediction Accuracy (%)")
    ax.set_ylabel("Average P&L per MWh (EUR)")
    ax.set_title("P&L Sensitivity to Prediction Accuracy")
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xticks(range(50, 95, 5))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_pnl_sensitivity.png", dpi=150)
    plt.close()
    print(f"[+] Saved: {PLOT_DIR / '02_pnl_sensitivity.png'}")

    # --- Plot 3: Position size heatmap ---
    fig, ax = plt.subplots(figsize=(14, 6))
    acc_cols = [c for c in matrix.columns if c.startswith('mwh_')]
    acc_labels = [c.replace('mwh_', '') + '%' for c in acc_cols]
    heatmap_data = matrix[acc_cols].values.copy()
    heatmap_data = np.clip(heatmap_data, 0, 200)  # cap for visualization
    heatmap_data[np.isinf(heatmap_data)] = 200

    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=200)
    ax.set_xticks(range(len(acc_labels)))
    ax.set_xticklabels(acc_labels)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets)
    ax.set_xlabel("Prediction Accuracy")
    ax.set_title("MWh Position Needed for EUR 200/day (green=feasible, red=large)")

    # Annotate cells
    for i in range(len(targets)):
        for j in range(len(acc_cols)):
            val = matrix[acc_cols[j]].iloc[i]
            if np.isinf(val) or val > 200:
                txt = ">200"
            else:
                txt = f"{val:.0f}"
            ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                    color='white' if heatmap_data[i, j] > 100 else 'black')

    plt.colorbar(im, ax=ax, label="MWh position size")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_position_size_heatmap.png", dpi=150)
    plt.close()
    print(f"[+] Saved: {PLOT_DIR / '03_position_size_heatmap.png'}")

    # --- Plot 4: Risk-return scatter at 70% accuracy ---
    fig, ax = plt.subplots(figsize=(10, 8))
    sharpe_col = 'sharpe_70'
    mwh_col = 'mwh_70'
    if sharpe_col in matrix.columns and mwh_col in matrix.columns:
        valid = matrix[~np.isinf(matrix[mwh_col])].copy()
        valid = valid[valid[mwh_col] < 500]

        for i, row in valid.iterrows():
            ax.scatter(row[sharpe_col], row[mwh_col],
                      s=max(50, 300 - row['breakeven_accuracy'] * 300),
                      alpha=0.7, zorder=5)
            ax.annotate(row['target'], (row[sharpe_col], row[mwh_col]),
                       textcoords="offset points", xytext=(10, 5), fontsize=9)

        ax.set_xlabel("Sharpe Ratio (at 70% accuracy)")
        ax.set_ylabel("MWh Position Needed for EUR 200/day")
        ax.set_title("Risk-Return at 70% Accuracy\n(lower position + higher Sharpe = better)")
        ax.axhline(50, color='green', linestyle='--', alpha=0.3, label='50 MWh threshold')
        ax.axhline(100, color='orange', linestyle='--', alpha=0.3, label='100 MWh threshold')
        ax.legend()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_risk_return_scatter.png", dpi=150)
    plt.close()
    print(f"[+] Saved: {PLOT_DIR / '04_risk_return_scatter.png'}")


def generate_summary(matrix, all_sweeps):
    """Generate summary.md with findings."""
    lines = [
        "# Prediction Requirement Analysis: Decision Matrix",
        "",
        "## Goal: EUR 200/day from speculative electricity trading",
        "",
        "## Key Question: What accuracy do we need for each prediction target?",
        "",
        "## Decision Matrix",
        "",
    ]

    # Format matrix as markdown table
    lines.append("| Target | Breakeven | EUR/MWh @60% | EUR/MWh @70% | MWh @70% | Sharpe @70% |")
    lines.append("|--------|-----------|-------------|-------------|----------|------------|")

    for _, row in matrix.iterrows():
        be = f"{row['breakeven_accuracy']:.0%}"
        e60 = f"{row.get('eur_mwh_60', float('nan')):+.1f}" if not pd.isna(row.get('eur_mwh_60')) else "N/A"
        e70 = f"{row.get('eur_mwh_70', float('nan')):+.1f}" if not pd.isna(row.get('eur_mwh_70')) else "N/A"
        m70 = row.get('mwh_70', float('inf'))
        m70_str = f"{m70:.0f}" if not np.isinf(m70) and m70 < 999 else ">999"
        s70 = f"{row.get('sharpe_70', 0):.1f}"
        lines.append(f"| {row['target']} | {be} | {e60} | {e70} | {m70_str} | {s70} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Breakeven accuracy**: Minimum prediction accuracy needed to be profitable after bid-ask costs",
        "- **EUR/MWh @X%**: Average profit per MWh traded at X% accuracy",
        "- **MWh @70%**: Position size (MWh per hour) needed for EUR 200/day at 70% accuracy",
        "- **Sharpe @70%**: Annualized Sharpe ratio at that position size",
        "",
        "## Recommendations",
        "",
        "Targets are ranked by feasibility (low breakeven + low position size + high Sharpe):",
        "",
    ])

    # Rank targets
    rankable = matrix.copy()
    rankable['score'] = (1 - rankable['breakeven_accuracy']) * 100  # lower breakeven = better
    if 'mwh_70' in rankable.columns:
        rankable['score'] -= rankable['mwh_70'].clip(upper=200) * 0.1  # lower position = better
    if 'sharpe_70' in rankable.columns:
        rankable['score'] += rankable['sharpe_70'] * 5  # higher Sharpe = better
    rankable = rankable.sort_values('score', ascending=False)

    for i, (_, row) in enumerate(rankable.iterrows()):
        m70 = row.get('mwh_70', float('inf'))
        feasible = "FEASIBLE" if not np.isinf(m70) and m70 < 100 else "LARGE POSITION" if not np.isinf(m70) else "UNPROFITABLE"
        lines.append(f"{i+1}. **{row['target']}** -- Breakeven: {row['breakeven_accuracy']:.0%}, {feasible}")

    summary_text = "\n".join(lines)

    with open(BASE_DIR / "summary.md", 'w') as f:
        f.write(summary_text)
    print(f"\n[+] Saved: {BASE_DIR / 'summary.md'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PREDICTION REQUIREMENT ANALYSIS")
    print(f"Target: EUR {TARGET_DAILY_EUR:.0f}/day")
    print("=" * 70)

    # Load data
    ob_raw = load_orderbook_snapshots()
    ob_pivot = pivot_bid_ask(ob_raw)
    market_df = load_market_prices()
    rt_df = load_round_trip_data()

    # Run all targets
    all_sweeps = []
    all_sweeps.append(target1_da_to_idm(ob_pivot, market_df))
    all_sweeps.append(target2_idm_ttd_movement(ob_pivot))
    all_sweeps.append(target3_idm_direction(ob_pivot))
    all_sweeps.append(target4_idm_round_trip(rt_df))
    all_sweeps.append(target5_imbalance_price(ob_pivot, market_df))
    all_sweeps.append(target6_imbalance_direction(ob_pivot, market_df))
    all_sweeps.append(target7_da_idm_spread(ob_pivot, market_df))

    # Save individual sweeps
    for sweep in all_sweeps:
        name = sweep['target'].iloc[0].replace(' ', '_').replace(':', '').replace('->', '_')
        sweep.to_csv(DATA_DIR / f"sweep_{name}.csv", index=False)

    # Build decision matrix
    matrix = build_decision_matrix(all_sweeps)
    matrix.to_csv(DATA_DIR / "decision_matrix.csv", index=False)
    print(f"\n[+] Saved: {DATA_DIR / 'decision_matrix.csv'}")

    # Print matrix
    print("\n" + matrix.to_string(index=False))

    # Plots
    plot_results(all_sweeps, matrix)

    # Summary
    generate_summary(matrix, all_sweeps)

    print("\n" + "=" * 70)
    print("[+] Analysis complete. Check plots/ and data/ for outputs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
