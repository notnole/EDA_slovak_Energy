"""
Realistic Simulation: Random Accuracy + Tail Risk Analysis
==========================================================

Fixes the methodological flaw in the previous analysis:
  - Previous: sorted trades by edge, assigned top X% as correct (OPTIMISTIC)
  - Now: RANDOMLY assign correct/incorrect at the given accuracy rate (REALISTIC)
  - Run 500 Monte Carlo simulations to get confidence intervals

Also: tail risk comparison between concentrated vs spread-out trading.
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
N_SIMULATIONS = 500

plt.rcParams.update({
    "figure.figsize": (16, 10), "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})


def get_conn():
    return psycopg2.connect(**DB)


def max_drawdown(eq):
    peak = eq.cummax()
    return abs((eq - peak).min())


def load_data():
    """Same data loading as advanced script."""
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
            )
            SELECT tradeday, hour, tradetype,
                   AVG(CASE WHEN htd >= 0.75 AND htd < 2 THEN price END) AS price_1h
            FROM with_ttd
            GROUP BY tradeday, hour, tradetype
        """, conn)

    bids = ob[ob['tradetype'] == 'N'][['tradeday', 'hour', 'price_1h']].copy()
    bids.columns = ['tradeday', 'hour', 'bid']
    asks = ob[ob['tradetype'] == 'P'][['tradeday', 'hour', 'price_1h']].copy()
    asks.columns = ['tradeday', 'hour', 'ask']

    m = bids.merge(asks, on=['tradeday', 'hour'], how='inner')

    mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv")
    tc = 'datetime' if 'datetime' in mkt.columns else 'timestamp_hour'
    mkt[tc] = pd.to_datetime(mkt[tc])
    mkt['tradeday'] = mkt[tc].dt.date
    if 'hour' not in mkt.columns:
        mkt['hour'] = mkt[tc].dt.hour

    m['tradeday'] = pd.to_datetime(m['tradeday']).dt.date

    df = m.merge(mkt[['tradeday', 'hour', 'da_price', 'imb_settlement_price', 'imbalance_mwh']],
                 on=['tradeday', 'hour'], how='inner')
    df = df.dropna(subset=['bid', 'ask', 'imb_settlement_price'])
    df['date'] = pd.to_datetime(df['tradeday'])
    df['year'] = df['date'].dt.year
    df['imb_abs'] = df['imbalance_mwh'].abs()
    df['is_deficit'] = (df['imbalance_mwh'] < 0).astype(int)

    df['pnl_correct'] = np.where(
        df['is_deficit'],
        df['imb_settlement_price'] - df['ask'],
        df['bid'] - df['imb_settlement_price'],
    )
    df['pnl_incorrect'] = np.where(
        df['is_deficit'],
        df['bid'] - df['imb_settlement_price'],
        df['imb_settlement_price'] - df['ask'],
    )

    print(f"[+] {len(df)} hours, {df['tradeday'].nunique()} days")
    return df


# ============================================================
# REALISTIC MONTE CARLO SIMULATION
# ============================================================

def monte_carlo_accuracy(pnl_correct, pnl_incorrect, dates, accuracy, n_sims=N_SIMULATIONS):
    """
    Randomly assign correct/incorrect at the given accuracy rate.
    Returns distribution of daily P&L stats across simulations.
    """
    rng = np.random.default_rng(42)
    n = len(pnl_correct)

    daily_means = []
    daily_sharpes = []
    daily_max_dds = []
    daily_worst_days = []
    daily_worst_weeks = []

    for _ in range(n_sims):
        # Randomly decide which trades are correct
        is_correct = rng.random(n) < accuracy
        trade_pnl = np.where(is_correct, pnl_correct, pnl_incorrect)

        df_tmp = pd.DataFrame({'date': dates, 'pnl': trade_pnl})
        daily = df_tmp.groupby('date')['pnl'].sum()

        daily_means.append(daily.mean())
        if daily.std() > 0:
            daily_sharpes.append((daily.mean() / daily.std()) * np.sqrt(365))
        else:
            daily_sharpes.append(0)
        eq = daily.cumsum()
        daily_max_dds.append(max_drawdown(eq))
        daily_worst_days.append(daily.min())
        if len(daily) >= 7:
            daily_worst_weeks.append(daily.rolling(7).sum().min())
        else:
            daily_worst_weeks.append(daily.sum())

    return {
        'mean_daily': np.array(daily_means),
        'sharpe': np.array(daily_sharpes),
        'max_dd': np.array(daily_max_dds),
        'worst_day': np.array(daily_worst_days),
        'worst_week': np.array(daily_worst_weeks),
    }


def analysis_realistic_vs_optimistic(df):
    """Compare optimistic (sorted) vs realistic (random) accuracy simulation."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Realistic vs Optimistic Accuracy Simulation")
    print("=" * 70)

    pnl_c = df['pnl_correct'].values
    pnl_i = df['pnl_incorrect'].values
    dates = df['date'].values

    print(f"\n  {'Accuracy':>8} | {'--- OPTIMISTIC (sorted) ---':>35} | {'--- REALISTIC (random, median) ---':>42} | {'Realistic P10':>14}")
    print(f"  {'':>8} | {'EUR/MWh':>10} {'Daily@1MW':>10} {'MWh@200':>8} | {'EUR/MWh':>10} {'Daily@1MW':>10} {'MWh@200':>8} {'Sharpe':>7} | {'Daily@1MW':>14}")
    print(f"  {'-'*115}")

    rows = []
    for acc in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        # Optimistic: sorted
        edge = pnl_c - pnl_i
        order = np.argsort(edge)[::-1]
        n_correct = int(acc * len(order))
        pnl_opt = np.empty(len(order))
        pnl_opt[:n_correct] = pnl_c[order[:n_correct]]
        pnl_opt[n_correct:] = pnl_i[order[n_correct:]]

        df_opt = pd.DataFrame({'date': dates, 'pnl': pnl_opt})
        daily_opt = df_opt.groupby('date')['pnl'].sum()
        avg_opt = pnl_opt.mean()
        daily_opt_mean = daily_opt.mean()
        mwh_opt = TARGET_DAILY_EUR / daily_opt_mean if daily_opt_mean > 0 else float('inf')

        # Realistic: Monte Carlo
        mc = monte_carlo_accuracy(pnl_c, pnl_i, dates, acc)
        avg_real = np.median(mc['mean_daily'])
        sharpe_real = np.median(mc['sharpe'])
        p10_daily = np.percentile(mc['mean_daily'], 10)

        # Compute realistic EUR/MWh from daily mean
        hours_per_day = len(df) / df['tradeday'].nunique()
        eur_mwh_real = avg_real / hours_per_day
        mwh_real = TARGET_DAILY_EUR / avg_real if avg_real > 0 else float('inf')
        mwh_p10 = TARGET_DAILY_EUR / p10_daily if p10_daily > 0 else float('inf')

        mwh_opt_s = f"{mwh_opt:.1f}" if not np.isinf(mwh_opt) else ">999"
        mwh_real_s = f"{mwh_real:.1f}" if not np.isinf(mwh_real) else ">999"

        print(f"  {acc:>7.0%} | {avg_opt:>+10.1f} {daily_opt_mean:>+10.1f} {mwh_opt_s:>8} | {eur_mwh_real:>+10.1f} {avg_real:>+10.1f} {mwh_real_s:>8} {sharpe_real:>7.1f} | {p10_daily:>+14.1f}")

        rows.append({
            'accuracy': acc,
            'optimistic_eur_mwh': avg_opt,
            'optimistic_daily': daily_opt_mean,
            'optimistic_mwh': mwh_opt,
            'realistic_eur_mwh': eur_mwh_real,
            'realistic_daily': avg_real,
            'realistic_mwh': mwh_real,
            'realistic_sharpe': sharpe_real,
            'realistic_p10_daily': p10_daily,
        })

    pd.DataFrame(rows).to_csv(DATA_DIR / "realistic_vs_optimistic.csv", index=False)


# ============================================================
# TAIL RISK: CONCENTRATED VS SPREAD
# ============================================================

def analysis_tail_risk(df):
    """Compare tail risk between 4 trades/day vs 24 trades/day."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Tail Risk -- Concentrated vs Spread")
    print("=" * 70)

    pnl_c = df['pnl_correct'].values
    pnl_i = df['pnl_incorrect'].values
    dates = df['date'].values

    acc = 0.65
    rng = np.random.default_rng(42)

    # Sort by imbalance magnitude for selecting top-N hours
    df_sorted = df.copy()
    df_sorted['edge'] = df_sorted['pnl_correct'] - df_sorted['pnl_incorrect']

    scenarios = {
        '24h/day, 0.4 MWh': {'n_hours': 24, 'mwh': 0.4},
        '8h/day, 1.0 MWh':  {'n_hours': 8, 'mwh': 1.0},
        '4h/day, 2.0 MWh':  {'n_hours': 4, 'mwh': 2.0},
        '2h/day, 4.0 MWh':  {'n_hours': 2, 'mwh': 4.0},
    }

    print(f"\n  All targeting ~EUR 200/day, 65% accuracy, 500 MC simulations")
    print(f"  {'Scenario':<22} {'Med daily':>10} {'P10 daily':>10} {'Worst day':>10} {'Worst week':>11} {'MaxDD':>8} {'Sharpe':>7} {'P(day<-200)':>12}")
    print(f"  {'-'*95}")

    all_scenario_results = {}

    for name, params in scenarios.items():
        n_hours = params['n_hours']
        mwh = params['mwh']

        # Select top N hours per day by edge
        if n_hours < 24:
            selected = df_sorted.groupby('tradeday').apply(
                lambda g: g.nlargest(min(n_hours, len(g)), 'edge'), include_groups=False
            ).reset_index(drop=True)
        else:
            selected = df_sorted.copy()

        pc = selected['pnl_correct'].values
        pi = selected['pnl_incorrect'].values
        d = selected['date'].values

        daily_results = []
        worst_days_all = []
        worst_weeks_all = []
        max_dds_all = []
        prob_bad_day = []

        for _ in range(N_SIMULATIONS):
            is_correct = rng.random(len(pc)) < acc
            trade_pnl = np.where(is_correct, pc, pi) * mwh

            df_tmp = pd.DataFrame({'date': d, 'pnl': trade_pnl})
            daily = df_tmp.groupby('date')['pnl'].sum()

            daily_results.append(daily.mean())
            worst_days_all.append(daily.min())
            if len(daily) >= 7:
                worst_weeks_all.append(daily.rolling(7).sum().min())
            max_dds_all.append(max_drawdown(daily.cumsum()))
            prob_bad_day.append((daily < -200).mean())

        med_daily = np.median(daily_results)
        p10_daily = np.percentile(daily_results, 10)
        med_worst_day = np.median(worst_days_all)
        med_worst_week = np.median(worst_weeks_all) if worst_weeks_all else 0
        med_max_dd = np.median(max_dds_all)
        sharpe_med = np.median([(d / np.std(daily_results)) * np.sqrt(365) for d in daily_results]) if np.std(daily_results) > 0 else 0
        # Simpler sharpe
        sharpe = (np.median(daily_results) / np.std(daily_results)) * np.sqrt(365) if np.std(daily_results) > 0 else 0
        prob_bad = np.median(prob_bad_day)

        print(f"  {name:<22} {med_daily:>+10.0f} {p10_daily:>+10.0f} {med_worst_day:>+10.0f} {med_worst_week:>+11.0f} {med_max_dd:>8.0f} {sharpe:>7.1f} {prob_bad:>11.1%}")

        all_scenario_results[name] = {
            'daily_results': daily_results,
            'worst_days': worst_days_all,
            'worst_weeks': worst_weeks_all,
            'max_dds': max_dds_all,
        }

    # --- Plot: Distribution of worst days by scenario ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for i, (name, data) in enumerate(all_scenario_results.items()):
        ax = axes[i // 2][i % 2]
        ax.hist(data['worst_days'], bins=50, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(np.median(data['worst_days']), color='red', linestyle='--',
                   label=f'Median: {np.median(data["worst_days"]):.0f} EUR')
        ax.axvline(np.percentile(data['worst_days'], 5), color='orange', linestyle='--',
                   label=f'P5: {np.percentile(data["worst_days"], 5):.0f} EUR')
        ax.set_xlabel("Worst single day P&L (EUR)")
        ax.set_ylabel("Count (of 500 simulations)")
        ax.set_title(f"{name}")
        ax.legend(fontsize=9)

    fig.suptitle("Distribution of Worst Single Day by Strategy (500 MC sims, 65% accuracy)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_tail_risk_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Saved: {PLOT_DIR / '07_tail_risk_comparison.png'}")

    # --- Plot: Daily P&L distribution by scenario ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for i, (name, params) in enumerate(scenarios.items()):
        ax = axes[i // 2][i % 2]
        n_hours = params['n_hours']
        mwh = params['mwh']

        if n_hours < 24:
            selected = df_sorted.groupby('tradeday').apply(
                lambda g: g.nlargest(min(n_hours, len(g)), 'edge'), include_groups=False
            ).reset_index(drop=True)
        else:
            selected = df_sorted.copy()

        # One representative simulation
        rng2 = np.random.default_rng(123)
        is_correct = rng2.random(len(selected)) < acc
        trade_pnl = np.where(is_correct, selected['pnl_correct'].values, selected['pnl_incorrect'].values) * mwh
        df_tmp = pd.DataFrame({'date': selected['date'].values, 'pnl': trade_pnl})
        daily = df_tmp.groupby('date')['pnl'].sum()

        ax.hist(daily.values, bins=80, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(daily.mean(), color='red', linestyle='--', label=f'Mean: {daily.mean():.0f} EUR')
        ax.axvline(daily.quantile(0.05), color='orange', linestyle='--', label=f'P5: {daily.quantile(0.05):.0f} EUR')
        ax.axvline(-200, color='darkred', linestyle=':', label='-200 EUR')
        ax.set_xlabel("Daily P&L (EUR)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}")
        ax.legend(fontsize=9)

    fig.suptitle("Daily P&L Distribution (one representative sim, 65% accuracy)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "08_daily_pnl_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {PLOT_DIR / '08_daily_pnl_distributions.png'}")


# ============================================================
# REALISTIC YEAR-BY-YEAR
# ============================================================

def analysis_realistic_by_year(df):
    """Realistic MC simulation by year."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Realistic Simulation by Year")
    print("=" * 70)

    print(f"\n  65% accuracy, 500 MC simulations, median results")
    print(f"  {'Strategy':<22} {'Year':>5} {'Med daily':>10} {'MWh@200':>8} {'Sharpe':>7} {'Med worst day':>14} {'Med MaxDD':>10}")
    print(f"  {'-'*80}")

    df['edge'] = df['pnl_correct'] - df['pnl_incorrect']
    acc = 0.65
    rng = np.random.default_rng(42)

    for n_hours, mwh, name in [(24, 0.4, 'Flat 24h, 0.4MW'), (8, 1.0, 'Top 8h, 1.0MW'), (4, 2.0, 'Top 4h, 2.0MW')]:
        for year in [2024, 2025, 2026]:
            sub = df[df['year'] == year].copy()
            if len(sub) < 50:
                continue

            if n_hours < 24:
                selected = sub.groupby('tradeday').apply(
                    lambda g: g.nlargest(min(n_hours, len(g)), 'edge'), include_groups=False
                ).reset_index(drop=True)
            else:
                selected = sub.copy()

            pc = selected['pnl_correct'].values
            pi = selected['pnl_incorrect'].values
            d = selected['date'].values

            daily_means = []
            worst_days = []
            max_dds = []

            for _ in range(N_SIMULATIONS):
                is_correct = rng.random(len(pc)) < acc
                trade_pnl = np.where(is_correct, pc, pi) * mwh
                df_tmp = pd.DataFrame({'date': d, 'pnl': trade_pnl})
                daily = df_tmp.groupby('date')['pnl'].sum()
                daily_means.append(daily.mean())
                worst_days.append(daily.min())
                max_dds.append(max_drawdown(daily.cumsum()))

            med = np.median(daily_means)
            mwh_need = TARGET_DAILY_EUR / med if med > 0 else float('inf')
            mwh_s = f"{mwh_need:.1f}" if not np.isinf(mwh_need) else ">999"
            sharpe = (np.median(daily_means) / np.std(daily_means)) * np.sqrt(365) if np.std(daily_means) > 0 else 0

            print(f"  {name:<22} {year:>5} {med:>+10.0f} {mwh_s:>8} {sharpe:>7.1f} {np.median(worst_days):>+14.0f} {np.median(max_dds):>10.0f}")
        print()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("REALISTIC SIMULATION + TAIL RISK ANALYSIS")
    print("=" * 70)

    df = load_data()

    analysis_realistic_vs_optimistic(df)
    analysis_tail_risk(df)
    analysis_realistic_by_year(df)

    print("\n" + "=" * 70)
    print("[+] Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
