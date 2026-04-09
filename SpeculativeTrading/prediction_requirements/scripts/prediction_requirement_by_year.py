"""
Prediction Requirement Analysis -- SPLIT BY YEAR
=================================================

Runs the same 7-target analysis but separately for 2024, 2025, and 2026
to detect regime changes (especially IDM-Imbalance spread collapse in 2026).
"""

import pandas as pd
import numpy as np
import psycopg2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

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
ACCURACY_LEVELS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TRADING_HOURS_PER_DAY = 24

plt.rcParams.update({
    "figure.figsize": (16, 8),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ============================================================
# SHARED (same as main script)
# ============================================================

def get_conn():
    return psycopg2.connect(**DB)


def max_drawdown(equity_curve):
    peak = equity_curve.cummax()
    dd = equity_curve - peak
    return abs(dd.min()), 0


def compute_risk_metrics(daily_pnl):
    if len(daily_pnl) == 0 or daily_pnl.std() == 0:
        return {'mean_daily': 0, 'sharpe': 0, 'max_dd': 0, 'win_rate': 0, 'profit_factor': 0}
    equity = daily_pnl.cumsum()
    dd, _ = max_drawdown(equity)
    wins = daily_pnl[daily_pnl > 0]
    losses = daily_pnl[daily_pnl < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    return {
        'mean_daily': daily_pnl.mean(),
        'sharpe': (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(365) if daily_pnl.std() > 0 else 0,
        'max_dd': dd,
        'win_rate': (daily_pnl > 0).mean(),
        'profit_factor': pf,
    }


def simulate_accuracy(pnl_correct, pnl_incorrect, accuracy):
    n = len(pnl_correct)
    if n == 0:
        return np.array([])
    edge = pnl_correct - pnl_incorrect
    order = np.argsort(edge)[::-1]
    n_correct = int(accuracy * n)
    result = np.empty(n)
    result[order[:n_correct]] = pnl_correct[order[:n_correct]]
    result[order[n_correct:]] = pnl_incorrect[order[n_correct:]]
    return result


def sweep_one(pnl_correct, pnl_incorrect, dates, hours, target_name):
    rows = []
    for acc in ACCURACY_LEVELS:
        trade_pnl = simulate_accuracy(pnl_correct, pnl_incorrect, acc)
        if len(trade_pnl) == 0:
            continue

        df_tmp = pd.DataFrame({'date': dates, 'pnl': trade_pnl})
        daily = df_tmp.groupby('date')['pnl'].sum()
        metrics = compute_risk_metrics(daily)

        avg = trade_pnl.mean()
        mwh = TARGET_DAILY_EUR / (avg * TRADING_HOURS_PER_DAY) if avg > 0 else float('inf')

        rows.append({
            'accuracy': acc,
            'avg_pnl_per_mwh': avg,
            'mwh_for_200': mwh,
            'sharpe': metrics['sharpe'],
            'max_dd_1mwh': metrics['max_dd'],
            'win_rate': metrics['win_rate'],
            'n_trades': len(trade_pnl),
            'target': target_name,
        })
    return pd.DataFrame(rows)


def find_breakeven(sweep_df):
    for i in range(len(sweep_df) - 1):
        if sweep_df.iloc[i]['avg_pnl_per_mwh'] <= 0 < sweep_df.iloc[i + 1]['avg_pnl_per_mwh']:
            a1, a2 = sweep_df.iloc[i]['accuracy'], sweep_df.iloc[i + 1]['accuracy']
            p1, p2 = sweep_df.iloc[i]['avg_pnl_per_mwh'], sweep_df.iloc[i + 1]['avg_pnl_per_mwh']
            return a1 + (a2 - a1) * (-p1) / (p2 - p1)
    if len(sweep_df) > 0 and sweep_df.iloc[0]['avg_pnl_per_mwh'] > 0:
        return 0.50
    return 0.95


# ============================================================
# DATA LOADING
# ============================================================

def load_orderbook_snapshots():
    print("[*] Loading order book snapshots...")
    with get_conn() as conn:
        df = pd.read_sql("""
            WITH ob AS (
                SELECT
                    tradeday::date AS tradeday,
                    periodfrom AS hour,
                    tradetype, price, lastupdate,
                    tradeday::date + periodfrom * INTERVAL '1 hour' AS delivery_start
                FROM db_ems.vdt_isot_knihaobjednavok_best
                WHERE deliverydur = 60 AND id_depth = 1 AND price > 0 AND price < 1000
            ),
            with_ttd AS (
                SELECT *,
                    EXTRACT(EPOCH FROM (delivery_start - lastupdate)) / 3600.0 AS hours_to_delivery
                FROM ob WHERE lastupdate < delivery_start
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
            SELECT tradeday, hour, ttd_bucket, tradetype,
                   AVG(price) AS avg_price, COUNT(*) AS n_snapshots
            FROM bucketed
            GROUP BY tradeday, hour, ttd_bucket, tradetype
        """, conn)
    print(f"[+] {len(df)} rows")
    return df


def pivot_bid_ask(ob_df):
    bids = ob_df[ob_df['tradetype'] == 'N'][['tradeday', 'hour', 'ttd_bucket', 'avg_price']].copy()
    bids.columns = ['tradeday', 'hour', 'ttd_bucket', 'bid']
    asks = ob_df[ob_df['tradetype'] == 'P'][['tradeday', 'hour', 'ttd_bucket', 'avg_price']].copy()
    asks.columns = ['tradeday', 'hour', 'ttd_bucket', 'ask']
    m = bids.merge(asks, on=['tradeday', 'hour', 'ttd_bucket'], how='outer')
    m['mid'] = (m['bid'].fillna(m['ask']) + m['ask'].fillna(m['bid'])) / 2
    return m


def load_market_prices():
    print("[*] Loading market prices...")
    path = REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv"
    df = pd.read_csv(path)
    tc = 'datetime' if 'datetime' in df.columns else 'timestamp_hour'
    df[tc] = pd.to_datetime(df[tc])
    df['tradeday'] = df[tc].dt.date
    if 'hour' not in df.columns:
        df['hour'] = df[tc].dt.hour
    print(f"[+] {len(df)} rows")
    return df


def load_round_trip_data():
    print("[*] Loading round-trip data...")
    with get_conn() as conn:
        df = pd.read_sql("""
            WITH ob AS (
                SELECT tradeday::date AS tradeday, periodfrom AS hour, tradetype, price
                FROM db_ems.vdt_isot_knihaobjednavok_best
                WHERE deliverydur = 60 AND id_depth = 1 AND price > 0 AND price < 1000
            )
            SELECT tradeday, hour, tradetype,
                   MIN(price) AS min_price, MAX(price) AS max_price,
                   PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) AS p25,
                   PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY price) AS p50,
                   PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) AS p75
            FROM ob GROUP BY tradeday, hour, tradetype
        """, conn)
    print(f"[+] {len(df)} rows")
    return df


# ============================================================
# RUN FOR ONE YEAR
# ============================================================

def run_targets_for_period(ob_pivot, market_df, rt_df, period_name, year_filter):
    """Run all targets for a specific time period."""
    print(f"\n{'='*60}")
    print(f"  Period: {period_name}")
    print(f"{'='*60}")

    # Filter order book
    ob = ob_pivot[ob_pivot['tradeday'].apply(lambda d: year_filter(d))].copy()
    mkt = market_df[market_df['tradeday'].apply(lambda d: year_filter(d))].copy()
    rt = rt_df[rt_df['tradeday'].apply(lambda d: year_filter(d))].copy()

    n_ob = len(ob)
    n_mkt = len(mkt)
    print(f"  OB rows: {n_ob}, Market rows: {n_mkt}")

    if n_ob == 0 or n_mkt == 0:
        print("  [!] No data for this period, skipping")
        return []

    results = []

    # --- T1: DA -> IDM ---
    idm = ob[ob['ttd_bucket'] == 3.0][['tradeday', 'hour', 'bid', 'ask']].copy()
    idm.columns = ['tradeday', 'hour', 'idm_bid', 'idm_ask']
    da = mkt[['tradeday', 'hour', 'da_price']].drop_duplicates()
    m = idm.merge(da, on=['tradeday', 'hour'], how='inner').dropna(subset=['idm_bid', 'idm_ask', 'da_price'])
    if len(m) > 0:
        pnl_c = np.maximum((m['idm_bid'] - m['da_price']).values, (m['da_price'] - m['idm_ask']).values)
        pnl_i = np.minimum((m['idm_bid'] - m['da_price']).values, (m['da_price'] - m['idm_ask']).values)
        s = sweep_one(pnl_c, pnl_i, pd.to_datetime(m['tradeday']).values, m['hour'].values, 'T1: DA->IDM')
        results.append(s)
        print(f"  T1: {len(m)} trades, correct={pnl_c.mean():+.1f}, incorrect={pnl_i.mean():+.1f}")

    # --- T2: IDM 6h->1h ---
    entry = ob[ob['ttd_bucket'] == 6.0][['tradeday', 'hour', 'bid', 'ask']].copy()
    entry.columns = ['tradeday', 'hour', 'e_bid', 'e_ask']
    exit_ = ob[ob['ttd_bucket'] == 1.0][['tradeday', 'hour', 'bid', 'ask']].copy()
    exit_.columns = ['tradeday', 'hour', 'x_bid', 'x_ask']
    m2 = entry.merge(exit_, on=['tradeday', 'hour']).dropna()
    if len(m2) > 0:
        pnl_l = (m2['x_bid'] - m2['e_ask']).values
        pnl_s = (m2['e_bid'] - m2['x_ask']).values
        pnl_c = np.maximum(pnl_l, pnl_s)
        pnl_i = np.minimum(pnl_l, pnl_s)
        s = sweep_one(pnl_c, pnl_i, pd.to_datetime(m2['tradeday']).values, m2['hour'].values, 'T2: IDM 6h->1h')
        results.append(s)
        print(f"  T2: {len(m2)} trades, correct={pnl_c.mean():+.1f}, incorrect={pnl_i.mean():+.1f}")

    # --- T3: IDM 3h->1h ---
    entry3 = ob[ob['ttd_bucket'] == 3.0][['tradeday', 'hour', 'bid', 'ask']].copy()
    entry3.columns = ['tradeday', 'hour', 'e_bid', 'e_ask']
    m3 = entry3.merge(exit_, on=['tradeday', 'hour']).dropna()
    if len(m3) > 0:
        pnl_l = (m3['x_bid'] - m3['e_ask']).values
        pnl_s = (m3['e_bid'] - m3['x_ask']).values
        pnl_c = np.maximum(pnl_l, pnl_s)
        pnl_i = np.minimum(pnl_l, pnl_s)
        s = sweep_one(pnl_c, pnl_i, pd.to_datetime(m3['tradeday']).values, m3['hour'].values, 'T3: IDM 3h->1h')
        results.append(s)
        print(f"  T3: {len(m3)} trades, correct={pnl_c.mean():+.1f}, incorrect={pnl_i.mean():+.1f}")

    # --- T4: Round-trip ---
    bids_rt = rt[rt['tradetype'] == 'N'][['tradeday', 'hour', 'p25', 'p50', 'p75']].copy()
    bids_rt.columns = ['tradeday', 'hour', 'bid_p25', 'bid_p50', 'bid_p75']
    asks_rt = rt[rt['tradetype'] == 'P'][['tradeday', 'hour', 'p25', 'p50', 'p75']].copy()
    asks_rt.columns = ['tradeday', 'hour', 'ask_p25', 'ask_p50', 'ask_p75']
    m4 = bids_rt.merge(asks_rt, on=['tradeday', 'hour']).dropna()
    if len(m4) > 0:
        rt_rows = []
        for acc in ACCURACY_LEVELS:
            if acc < 0.65:
                buy_c, sell_c = 'ask_p50', 'bid_p50'
            elif acc < 0.80:
                buy_c, sell_c = 'ask_p25', 'bid_p75'
            else:
                buy_c, sell_c = 'ask_p25', 'bid_p75'  # best we can do with percentiles
            tp = (m4[sell_c] - m4[buy_c]).values
            df_t = pd.DataFrame({'date': pd.to_datetime(m4['tradeday']).values, 'pnl': tp})
            daily = df_t.groupby('date')['pnl'].sum()
            met = compute_risk_metrics(daily)
            avg = tp.mean()
            rt_rows.append({
                'accuracy': acc, 'avg_pnl_per_mwh': avg,
                'mwh_for_200': TARGET_DAILY_EUR / (avg * 24) if avg > 0 else float('inf'),
                'sharpe': met['sharpe'], 'max_dd_1mwh': met['max_dd'],
                'win_rate': met['win_rate'], 'n_trades': len(tp), 'target': 'T4: Round-trip',
            })
        results.append(pd.DataFrame(rt_rows))
        print(f"  T4: {len(m4)} delivery periods")

    # --- T5: Imbalance price ---
    idm1 = ob[ob['ttd_bucket'] == 1.0][['tradeday', 'hour', 'bid', 'ask']].copy()
    idm1.columns = ['tradeday', 'hour', 'idm_bid', 'idm_ask']
    imb = mkt[['tradeday', 'hour', 'imb_settlement_price']].drop_duplicates()
    m5 = idm1.merge(imb, on=['tradeday', 'hour']).dropna(subset=['idm_bid', 'idm_ask', 'imb_settlement_price'])
    if len(m5) > 0:
        pnl_sell = (m5['idm_bid'] - m5['imb_settlement_price']).values
        pnl_buy = (m5['imb_settlement_price'] - m5['idm_ask']).values
        pnl_c = np.maximum(pnl_sell, pnl_buy)
        pnl_i = np.minimum(pnl_sell, pnl_buy)
        s = sweep_one(pnl_c, pnl_i, pd.to_datetime(m5['tradeday']).values, m5['hour'].values, 'T5: Imb price')
        results.append(s)
        print(f"  T5: {len(m5)} trades, correct={pnl_c.mean():+.1f}, incorrect={pnl_i.mean():+.1f}")

    # --- T6: Imbalance direction ---
    imb_dir = mkt[['tradeday', 'hour', 'imb_settlement_price', 'imbalance_mwh']].drop_duplicates()
    m6 = idm1.merge(imb_dir, on=['tradeday', 'hour']).dropna(subset=['idm_bid', 'idm_ask', 'imb_settlement_price', 'imbalance_mwh'])
    if len(m6) > 0:
        is_def = m6['imbalance_mwh'].values < 0
        pnl_c = np.where(is_def,
                         (m6['imb_settlement_price'] - m6['idm_ask']).values,
                         (m6['idm_bid'] - m6['imb_settlement_price']).values)
        pnl_i = np.where(is_def,
                         (m6['idm_bid'] - m6['imb_settlement_price']).values,
                         (m6['imb_settlement_price'] - m6['idm_ask']).values)
        s = sweep_one(pnl_c, pnl_i, pd.to_datetime(m6['tradeday']).values, m6['hour'].values, 'T6: Imb direction')
        results.append(s)
        print(f"  T6: {len(m6)} trades, correct={pnl_c.mean():+.1f}, incorrect={pnl_i.mean():+.1f}")

    # --- T7: DA-IDM spread direction ---
    idm7 = ob[ob['ttd_bucket'] == 3.0][['tradeday', 'hour', 'bid', 'ask']].copy()
    idm7.columns = ['tradeday', 'hour', 'idm_bid', 'idm_ask']
    da7 = mkt[['tradeday', 'hour', 'da_price']].drop_duplicates()
    m7 = idm7.merge(da7, on=['tradeday', 'hour']).dropna(subset=['idm_bid', 'idm_ask', 'da_price'])
    if len(m7) > 0:
        actual = m7['idm_bid'].values - m7['da_price'].values
        pnl_l = (m7['idm_bid'] - m7['da_price']).values
        pnl_s = (m7['da_price'] - m7['idm_ask']).values
        pnl_c = np.where(actual >= 0, pnl_l, pnl_s)
        pnl_i = np.where(actual >= 0, pnl_s, pnl_l)
        s = sweep_one(pnl_c, pnl_i, pd.to_datetime(m7['tradeday']).values, m7['hour'].values, 'T7: DA-IDM spread')
        results.append(s)
        print(f"  T7: {len(m7)} trades, correct={pnl_c.mean():+.1f}, incorrect={pnl_i.mean():+.1f}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PREDICTION REQUIREMENT ANALYSIS -- BY YEAR")
    print("=" * 70)

    ob_raw = load_orderbook_snapshots()
    ob_pivot = pivot_bid_ask(ob_raw)
    market_df = load_market_prices()
    rt_df = load_round_trip_data()

    # Ensure date types match
    ob_pivot['tradeday'] = pd.to_datetime(ob_pivot['tradeday']).dt.date
    market_df['tradeday'] = pd.to_datetime(market_df['tradeday']).dt.date
    rt_df['tradeday'] = pd.to_datetime(rt_df['tradeday']).dt.date

    periods = {
        '2024': lambda d: d.year == 2024,
        '2025': lambda d: d.year == 2025,
        '2026': lambda d: d.year == 2026,
    }

    all_results = {}
    for period_name, year_filter in periods.items():
        results = run_targets_for_period(ob_pivot, market_df, rt_df, period_name, year_filter)
        all_results[period_name] = results

    # ---- Build comparison table ----
    print("\n\n" + "=" * 70)
    print("YEAR-BY-YEAR COMPARISON")
    print("=" * 70)

    # For each target, show key metrics at 60% and 70% accuracy by year
    target_names = ['T1: DA->IDM', 'T2: IDM 6h->1h', 'T3: IDM 3h->1h', 'T4: Round-trip',
                    'T5: Imb price', 'T6: Imb direction', 'T7: DA-IDM spread']

    comparison_rows = []
    for period_name, results in all_results.items():
        for sweep_df in results:
            if len(sweep_df) == 0:
                continue
            tname = sweep_df['target'].iloc[0]
            be = find_breakeven(sweep_df)

            for acc in [0.55, 0.60, 0.70]:
                s = sweep_df[sweep_df['accuracy'] == acc]
                if len(s) > 0:
                    s = s.iloc[0]
                    comparison_rows.append({
                        'period': period_name,
                        'target': tname,
                        'accuracy': acc,
                        'breakeven': be,
                        'eur_per_mwh': s['avg_pnl_per_mwh'],
                        'mwh_for_200': s['mwh_for_200'],
                        'sharpe': s['sharpe'],
                        'n_trades': s['n_trades'],
                    })

    comp = pd.DataFrame(comparison_rows)
    comp.to_csv(DATA_DIR / "comparison_by_year.csv", index=False)

    # Print formatted
    for tname in target_names:
        sub = comp[comp['target'] == tname]
        if len(sub) == 0:
            continue
        print(f"\n  {tname}")
        print(f"  {'Period':<8} {'BE':>6} | {'EUR/MWh@55%':>12} {'MWh@55%':>8} | {'EUR/MWh@60%':>12} {'MWh@60%':>8} | {'EUR/MWh@70%':>12} {'MWh@70%':>8} {'Sharpe':>7}")
        print(f"  {'-'*100}")
        for period in ['2024', '2025', '2026']:
            sp = sub[sub['period'] == period]
            if len(sp) == 0:
                print(f"  {period:<8}  -- no data --")
                continue
            be = sp.iloc[0]['breakeven']
            vals = {}
            for acc in [0.55, 0.60, 0.70]:
                row = sp[sp['accuracy'] == acc]
                if len(row) > 0:
                    vals[acc] = row.iloc[0]
                else:
                    vals[acc] = None

            def fmt(v, key, width=8):
                if v is None:
                    return "N/A".rjust(width)
                val = v[key]
                if np.isinf(val):
                    return ">999".rjust(width)
                return f"{val:+.1f}".rjust(width) if key == 'eur_per_mwh' else f"{val:.1f}".rjust(width)

            be_str = f"{be:.0%}".rjust(6)
            line = f"  {period:<8} {be_str} |"
            for acc in [0.55, 0.60, 0.70]:
                v = vals.get(acc)
                line += f" {fmt(v, 'eur_per_mwh', 12)} {fmt(v, 'mwh_for_200', 8)} |"
            if vals.get(0.70) is not None:
                line += f" {fmt(vals[0.70], 'sharpe', 7)}"
            print(line)

    # ---- Plot: EUR/MWh at 60% accuracy by year for each target ----
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    colors_year = {'2024': '#3498db', '2025': '#2ecc71', '2026': '#e74c3c'}

    for i, tname in enumerate(target_names):
        ax = axes[i]
        for period in ['2024', '2025', '2026']:
            sub = comp[(comp['target'] == tname) & (comp['period'] == period)]
            if len(sub) > 0:
                ax.plot(sub['accuracy'] * 100, sub['eur_per_mwh'],
                       'o-', color=colors_year[period], label=period, linewidth=2, markersize=5)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(tname, fontsize=10)
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("EUR/MWh")
        ax.legend(fontsize=8)

    # Hide unused subplot
    axes[7].set_visible(False)

    fig.suptitle("P&L per MWh by Year and Prediction Accuracy", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_pnl_by_year.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Saved: {PLOT_DIR / '05_pnl_by_year.png'}")

    # ---- Plot: Position size at 60% accuracy by year ----
    fig, ax = plt.subplots(figsize=(14, 7))
    acc_focus = 0.60
    sub60 = comp[comp['accuracy'] == acc_focus].copy()
    sub60['mwh_clipped'] = sub60['mwh_for_200'].clip(upper=200)

    x_positions = np.arange(len(target_names))
    width = 0.25
    for j, (period, color) in enumerate(colors_year.items()):
        vals = []
        for tname in target_names:
            row = sub60[(sub60['target'] == tname) & (sub60['period'] == period)]
            if len(row) > 0:
                v = row.iloc[0]['mwh_clipped']
                vals.append(v if not np.isinf(v) else 200)
            else:
                vals.append(0)
        ax.bar(x_positions + j * width, vals, width, label=period, color=color, alpha=0.8)

    ax.set_xticks(x_positions + width)
    ax.set_xticklabels(target_names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel("MWh position for EUR 200/day")
    ax.set_title(f"Required Position Size at {acc_focus:.0%} Accuracy -- by Year")
    ax.axhline(50, color='orange', linestyle='--', alpha=0.5, label='50 MWh cap')
    ax.legend()
    ax.set_ylim(0, 210)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_position_by_year.png", dpi=150)
    plt.close()
    print(f"[+] Saved: {PLOT_DIR / '06_position_by_year.png'}")

    print("\n[+] Done.")


if __name__ == "__main__":
    main()
