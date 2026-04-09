"""
Backtest: Apply imbalance predictions to actual IDM/imbalance prices.

Strategies:
  1. Flat 5 MWh on confident predictions (|pred| > threshold)
  2. Magnitude-based sizing: size = |pred| / 5, capped 0.5-10 MWh
  3. CI-width sizing: narrow CI = bigger position
  4. Combined: magnitude sizing + skip wide CI hours
"""

import pandas as pd
import numpy as np
import psycopg2
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
REPO_ROOT = SCRIPT_DIR.parents[2]

DB = {
    "host": "localhost", "port": 5432,
    "dbname": "DB_EMS", "user": "postgres", "password": "Kapitan1",
}


def load_idm_prices():
    """Load IDM bid/ask at TTD=1h."""
    print("[*] Loading IDM prices...")
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
        FROM with_ttd
        WHERE htd >= 0.75 AND htd < 2
        GROUP BY tradeday, hour, tradetype
    """, conn)
    conn.close()

    bids = ob[ob['tradetype'] == 'N'][['tradeday', 'hour', 'price']].rename(columns={'price': 'bid'})
    asks = ob[ob['tradetype'] == 'P'][['tradeday', 'hour', 'price']].rename(columns={'price': 'ask'})
    idm = bids.merge(asks, on=['tradeday', 'hour'], how='inner')
    idm['tradeday'] = pd.to_datetime(idm['tradeday'])
    print(f"[+] IDM: {len(idm)} hours")
    return idm


def load_predictions():
    """Load model predictions and aggregate to hourly."""
    print("[*] Loading predictions...")
    pred = pd.read_csv(DATA_DIR / "predictions_test.csv", index_col=0, parse_dates=True)
    pred['tradeday'] = pred.index.normalize()
    pred['hour'] = pred.index.hour

    hourly = pred.groupby(['tradeday', 'hour']).agg(
        pred_median=('q50', 'mean'),
        q10=('q10', 'mean'),
        q25=('q25', 'mean'),
        q75=('q75', 'mean'),
        q90=('q90', 'mean'),
        actual_imb=('target', 'mean'),
    ).reset_index()

    print(f"[+] Predictions: {len(hourly)} hours")
    return hourly


def load_imbalance_prices():
    """Load imbalance settlement prices."""
    print("[*] Loading imbalance prices...")
    path = REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv"
    mkt = pd.read_csv(path)
    tc = 'datetime' if 'datetime' in mkt.columns else 'timestamp_hour'
    mkt[tc] = pd.to_datetime(mkt[tc])
    mkt['tradeday'] = pd.to_datetime(mkt[tc].dt.date)
    if 'hour' not in mkt.columns:
        mkt['hour'] = mkt[tc].dt.hour
    print(f"[+] Market: {len(mkt)} rows")
    return mkt[['tradeday', 'hour', 'imb_settlement_price']].drop_duplicates()


def compute_pnl(sub):
    """Compute per-trade P&L for a filtered subset."""
    pnl = np.where(
        sub['pred_median'] > 0,
        sub['imb_settlement_price'] - sub['ask'],   # buy IDM, settle at imb
        sub['bid'] - sub['imb_settlement_price'],    # sell IDM, settle at imb
    )
    return pnl


def report(daily, sub, label):
    """Print strategy report."""
    total = daily.sum()
    avg = daily.mean()
    std = daily.std()
    sharpe = (avg / std) * np.sqrt(365) if std > 0 else 0
    worst = daily.min()
    best = daily.max()
    win = (daily > 0).mean()
    eq = daily.cumsum()
    max_dd = abs((eq - eq.cummax()).min())
    n_days = len(daily)
    hrs_day = len(sub) / n_days if n_days > 0 else 0

    correct = (np.sign(sub['pred_median'].values) == np.sign(sub['actual_imb'].values))
    nonzero = sub['actual_imb'].abs() > 0.1
    dir_acc = correct[nonzero.values].mean() if nonzero.sum() > 0 else 0

    print(f"\n  {label}")
    print(f"    Trades:       {len(sub)} hours ({hrs_day:.1f}/day)")
    print(f"    Direction:    {dir_acc:.1%}")
    if 'size' in sub.columns:
        print(f"    Avg position: {sub['size'].mean():.1f} MWh")
        print(f"    Max position: {sub['size'].max():.1f} MWh")
    print(f"    Total P&L:    {total:+,.0f} EUR over {n_days} days")
    print(f"    Avg daily:    {avg:+,.0f} EUR")
    print(f"    Std daily:    {std:,.0f} EUR")
    print(f"    Sharpe (ann): {sharpe:.1f}")
    print(f"    Win rate:     {win:.0%} of days")
    print(f"    Best day:     {best:+,.0f} EUR")
    print(f"    Worst day:    {worst:+,.0f} EUR")
    print(f"    Max drawdown: {max_dd:,.0f} EUR")


def main():
    print("=" * 70)
    print("BACKTEST: Imbalance Prediction -> IDM Trading")
    print("=" * 70)

    idm = load_idm_prices()
    hourly = load_predictions()
    mkt = load_imbalance_prices()

    # Merge all
    m = hourly.merge(idm, on=['tradeday', 'hour'], how='inner')
    m = m.merge(mkt, on=['tradeday', 'hour'], how='inner')
    m = m.dropna(subset=['bid', 'ask', 'imb_settlement_price'])
    m['date'] = m['tradeday']
    m['ci_width'] = m['q90'] - m['q10']

    all_days = pd.date_range(m['tradeday'].min(), m['tradeday'].max(), freq='D')

    print(f"\n[+] Merged: {len(m)} hours, {m['tradeday'].nunique()} days")
    print(f"    Period: {m['tradeday'].min().date()} to {m['tradeday'].max().date()}")

    # ================================================================
    # STRATEGY 1: Flat 5 MWh at various thresholds
    # ================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 1: Flat 5 MWh position")
    print("=" * 70)

    for thresh in [2, 5, 10]:
        mask = m['pred_median'].abs() > thresh
        sub = m[mask].copy()
        sub['pnl_per_mwh'] = compute_pnl(sub)
        sub['size'] = 5.0
        sub['pnl'] = sub['pnl_per_mwh'] * 5.0

        daily = sub.groupby('date')['pnl'].sum().reindex(all_days, fill_value=0)
        report(daily, sub, f"|pred| > {thresh} MWh, flat 5 MWh")

    # ================================================================
    # STRATEGY 2: Magnitude-based sizing
    # ================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 2: Size = |pred| / 5, capped [0.5, 10] MWh")
    print("=" * 70)

    mask = m['pred_median'].abs() > 2
    sub = m[mask].copy()
    sub['pnl_per_mwh'] = compute_pnl(sub)
    sub['size'] = (sub['pred_median'].abs() / 5).clip(0.5, 10.0)
    sub['pnl'] = sub['pnl_per_mwh'] * sub['size']

    daily = sub.groupby('date')['pnl'].sum().reindex(all_days, fill_value=0)
    report(daily, sub, "Magnitude sizing, |pred| > 2 MWh")

    # ================================================================
    # STRATEGY 3: Inverse CI sizing
    # ================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 3: Narrow CI = bigger position")
    print("=" * 70)

    mask = m['pred_median'].abs() > 2
    sub = m[mask].copy()
    sub['pnl_per_mwh'] = compute_pnl(sub)
    median_ci = sub['ci_width'].median()
    sub['size'] = ((median_ci / sub['ci_width']) * 3).clip(0.5, 10.0)
    sub['pnl'] = sub['pnl_per_mwh'] * sub['size']

    daily = sub.groupby('date')['pnl'].sum().reindex(all_days, fill_value=0)
    report(daily, sub, "CI-width sizing, |pred| > 2 MWh")

    # ================================================================
    # STRATEGY 4: Magnitude sizing + tight CI filter
    # ================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 4: Magnitude sizing + only narrow CI hours")
    print("=" * 70)

    mask = m['pred_median'].abs() > 2
    sub = m[mask].copy()
    ci_median = sub['ci_width'].median()
    sub = sub[sub['ci_width'] <= ci_median].copy()
    sub['pnl_per_mwh'] = compute_pnl(sub)
    sub['size'] = (sub['pred_median'].abs() / 5).clip(0.5, 10.0)
    sub['pnl'] = sub['pnl_per_mwh'] * sub['size']

    daily = sub.groupby('date')['pnl'].sum().reindex(all_days, fill_value=0)
    report(daily, sub, "Magnitude + tight CI, |pred| > 2 MWh")

    print("\n" + "=" * 70)
    print("[+] Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
