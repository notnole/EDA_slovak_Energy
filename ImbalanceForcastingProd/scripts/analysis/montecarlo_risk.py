"""
Monte Carlo risk analysis for the standalone spread model (5 MW).
Bootstraps from observed trade P&L distribution to estimate VaR, CVaR,
drawdowns, and annual P&L confidence intervals.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parents[2]
PLOT_DIR = BASE / "plots" / "eda" / "montecarlo"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (18, 12), "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130,
})

SIZE_MW = 5.0
QH = 0.25
ENERGY = SIZE_MW * QH
N_SIMS = 50_000
N_DAYS = 252

np.random.seed(42)


def main():
    print("=" * 70)
    print("MONTE CARLO RISK ANALYSIS — Standalone Spread Model (5 MW)")
    print("=" * 70)

    # Load observed trades
    stk = pd.read_csv(BASE / "data" / "predictions" / "stacked_test_predictions.csv",
                       parse_dates=['datetime'], index_col='datetime')
    pred = stk['standalone_spread_pred']
    trades = stk[(pred <= -3) | (pred >= 3)].copy()
    trades['pnl'] = np.where(
        trades['standalone_spread_pred'] > 0,
        (trades['imb_settlement_price'] - trades['exec_ask']) * ENERGY,
        (trades['exec_bid'] - trades['imb_settlement_price']) * ENERGY)
    trades = trades[trades['pnl'].notna()]
    trade_pnl = trades['pnl'].values
    tpd = trades.groupby(trades.index.date).size().values
    avg_tpd = int(tpd.mean())

    observed_daily = trades.groupby(trades.index.date)['pnl'].sum()

    print(f"[+] Observed: {len(trade_pnl)} trades, {len(tpd)} days")
    print(f"[+] Trade P&L: mean={trade_pnl.mean():+.1f}, std={trade_pnl.std():.1f}")
    print(f"[+] Trades/day: mean={tpd.mean():.0f}, used={avg_tpd}")
    print(f"[+] Simulating {N_SIMS:,} years x {N_DAYS} days x {avg_tpd} trades/day")

    # Vectorized simulation
    samples = np.random.choice(trade_pnl, size=(N_SIMS, N_DAYS, avg_tpd), replace=True)
    daily = samples.sum(axis=2)
    yearly = daily.sum(axis=1)
    worst_day = daily.min(axis=1)
    cumsum = daily.cumsum(axis=1)
    running_max = np.maximum.accumulate(cumsum, axis=1)
    drawdowns = (cumsum - running_max).min(axis=1)
    flat_daily = daily.flatten()

    var95 = np.percentile(flat_daily, 5)
    var99 = np.percentile(flat_daily, 1)
    cvar95 = flat_daily[flat_daily <= var95].mean()
    cvar99 = flat_daily[flat_daily <= var99].mean()

    # ===== PLOT =====
    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Observed trade P&L distribution
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(trade_pnl, bins=80, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.3)
    ax.axvline(trade_pnl.mean(), color='red', ls='--', label=f'Mean={trade_pnl.mean():+.1f}')
    ax.axvline(0, color='gray', ls='-', alpha=0.5)
    ax.set_xlabel('Trade P&L (EUR)')
    ax.set_ylabel('Count')
    ax.set_title(f'Observed Trade P&L Distribution (n={len(trade_pnl)})')
    ax.legend()

    # 2. Simulated daily P&L distribution
    ax = fig.add_subplot(gs[0, 1])
    daily_sample = flat_daily[np.random.choice(len(flat_daily), 50000, replace=False)]
    ax.hist(daily_sample, bins=100, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.2)
    ax.axvline(var95, color='orange', ls='--', lw=2, label=f'VaR 95% = {var95:+.0f}')
    ax.axvline(var99, color='red', ls='--', lw=2, label=f'VaR 99% = {var99:+.0f}')
    ax.axvline(np.median(flat_daily), color='green', ls='--', label=f'Median = {np.median(flat_daily):+.0f}')
    ax.set_xlabel('Daily P&L (EUR)')
    ax.set_ylabel('Count')
    ax.set_title('Simulated Daily P&L Distribution')
    ax.legend(fontsize=9)

    # 3. Annual P&L distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(yearly / 1000, bins=80, alpha=0.7, color='coral', edgecolor='black', linewidth=0.3)
    p5 = np.percentile(yearly, 5) / 1000
    p50 = np.percentile(yearly, 50) / 1000
    p95 = np.percentile(yearly, 95) / 1000
    ax.axvline(p5, color='orange', ls='--', lw=2, label=f'P05 = {p5:+.0f}k')
    ax.axvline(p50, color='green', ls='--', lw=2, label=f'P50 = {p50:+.0f}k')
    ax.axvline(p95, color='blue', ls='--', lw=2, label=f'P95 = {p95:+.0f}k')
    ax.set_xlabel('Annual P&L (kEUR)')
    ax.set_ylabel('Count')
    ax.set_title(f'Simulated Annual P&L ({N_SIMS:,} sims)')
    ax.legend(fontsize=9)

    # 4. Worst day distribution
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(worst_day, bins=80, alpha=0.7, color='indianred', edgecolor='black', linewidth=0.3)
    for p, c, ls in [(1, 'red', '-'), (5, 'orange', '--'), (50, 'green', '--')]:
        v = np.percentile(worst_day, p)
        ax.axvline(v, color=c, ls=ls, lw=2, label=f'P{p:02d} = {v:+.0f}')
    ax.set_xlabel('Worst Single Day P&L (EUR)')
    ax.set_ylabel('Count')
    ax.set_title('Worst Day Distribution (per simulated year)')
    ax.legend(fontsize=9)

    # 5. Max drawdown distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(drawdowns, bins=80, alpha=0.7, color='indianred', edgecolor='black', linewidth=0.3)
    for p, c, ls in [(1, 'red', '-'), (5, 'orange', '--'), (50, 'green', '--')]:
        v = np.percentile(drawdowns, p)
        ax.axvline(v, color=c, ls=ls, lw=2, label=f'P{p:02d} = {v:+.0f}')
    ax.set_xlabel('Max Annual Drawdown (EUR)')
    ax.set_ylabel('Count')
    ax.set_title('Max Drawdown Distribution (per simulated year)')
    ax.legend(fontsize=9)

    # 6. Sample equity curves (20 random years)
    ax = fig.add_subplot(gs[1, 2])
    for i in range(20):
        eq = daily[i].cumsum()
        ax.plot(range(N_DAYS), eq, alpha=0.3, linewidth=0.8, color='steelblue')
    # Median path
    median_eq = np.median(cumsum, axis=0)
    p5_eq = np.percentile(cumsum, 5, axis=0)
    p95_eq = np.percentile(cumsum, 95, axis=0)
    ax.plot(range(N_DAYS), median_eq, color='red', lw=2, label='Median')
    ax.fill_between(range(N_DAYS), p5_eq, p95_eq, alpha=0.15, color='red', label='P05-P95')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Cumulative P&L (EUR)')
    ax.set_title('Simulated Equity Curves (20 paths + P05/P50/P95)')
    ax.legend(fontsize=9)

    # 7. Observed vs simulated daily P&L comparison
    ax = fig.add_subplot(gs[2, 0])
    obs_sorted = np.sort(observed_daily.values)
    n_obs = len(obs_sorted)
    sim_quantiles = np.percentile(flat_daily, np.linspace(0, 100, n_obs))
    ax.scatter(sim_quantiles, obs_sorted, s=15, alpha=0.6, c='steelblue')
    lims = [min(sim_quantiles.min(), obs_sorted.min()), max(sim_quantiles.max(), obs_sorted.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5)
    ax.set_xlabel('Simulated Quantiles (EUR)')
    ax.set_ylabel('Observed Quantiles (EUR)')
    ax.set_title('QQ: Observed vs Simulated Daily P&L')

    # 8. CVaR at various confidence levels
    ax = fig.add_subplot(gs[2, 1])
    conf_levels = [90, 95, 97.5, 99, 99.5, 99.9]
    vars_list = []
    cvars_list = []
    for cl in conf_levels:
        alpha = 100 - cl
        v = np.percentile(flat_daily, alpha)
        cv = flat_daily[flat_daily <= v].mean()
        vars_list.append(v)
        cvars_list.append(cv)
    x = range(len(conf_levels))
    ax.bar([i - 0.17 for i in x], [-v for v in vars_list], width=0.34, alpha=0.7,
           color='steelblue', label='VaR')
    ax.bar([i + 0.17 for i in x], [-cv for cv in cvars_list], width=0.34, alpha=0.7,
           color='indianred', label='CVaR')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{cl}%' for cl in conf_levels], fontsize=9)
    ax.set_xlabel('Confidence Level')
    ax.set_ylabel('Loss (EUR, positive = loss)')
    ax.set_title('Daily VaR and CVaR')
    ax.legend()

    # 9. Recovery time from drawdowns
    ax = fig.add_subplot(gs[2, 2])
    recovery_days = []
    for i in range(min(5000, N_SIMS)):
        eq = cumsum[i]
        dd = eq - running_max[i]
        if dd.min() < -100:
            trough_idx = dd.argmin()
            # Find recovery (first day after trough where eq >= running_max at trough)
            peak_val = running_max[i, trough_idx]
            recovered = np.where(eq[trough_idx:] >= peak_val)[0]
            if len(recovered) > 0:
                recovery_days.append(recovered[0])
    if recovery_days:
        ax.hist(recovery_days, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.3)
        ax.axvline(np.median(recovery_days), color='red', ls='--',
                   label=f'Median = {np.median(recovery_days):.0f} days')
        ax.axvline(np.percentile(recovery_days, 95), color='orange', ls='--',
                   label=f'P95 = {np.percentile(recovery_days, 95):.0f} days')
        ax.set_xlabel('Recovery Days')
        ax.set_ylabel('Count')
        ax.set_title(f'Recovery Time from >100 EUR Drawdowns (n={len(recovery_days)})')
        ax.legend(fontsize=9)

    fig.savefig(PLOT_DIR / "01_montecarlo_risk.png", bbox_inches='tight')
    plt.close(fig)
    print(f"\n[+] Saved: {PLOT_DIR / '01_montecarlo_risk.png'}")

    # Print summary stats
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Strategy: Standalone spread, 5 MW, threshold |pred|>=3")
    print(f"Based on: {len(trade_pnl)} observed trades over {len(tpd)} days")
    print(f"Simulation: {N_SIMS:,} years x {N_DAYS} days x {avg_tpd} trades/day")
    print()
    print(f"Annual P&L:  P05={np.percentile(yearly,5):+,.0f}  "
          f"P50={np.percentile(yearly,50):+,.0f}  "
          f"P95={np.percentile(yearly,95):+,.0f} EUR")
    print(f"P(annual loss): {(yearly < 0).mean():.4%}")
    print()
    print(f"Daily VaR 95%:   {var95:+,.0f} EUR")
    print(f"Daily VaR 99%:   {var99:+,.0f} EUR")
    print(f"Daily CVaR 95%:  {cvar95:+,.0f} EUR")
    print(f"Daily CVaR 99%:  {cvar99:+,.0f} EUR")
    print()
    print(f"Worst day:   P01={np.percentile(worst_day,1):+,.0f}  "
          f"P05={np.percentile(worst_day,5):+,.0f}  "
          f"P50={np.percentile(worst_day,50):+,.0f} EUR")
    print(f"Max DD:      P01={np.percentile(drawdowns,1):+,.0f}  "
          f"P05={np.percentile(drawdowns,5):+,.0f}  "
          f"P50={np.percentile(drawdowns,50):+,.0f} EUR")
    if recovery_days:
        print(f"Recovery:    Median={np.median(recovery_days):.0f}  "
              f"P95={np.percentile(recovery_days,95):.0f} days")


if __name__ == "__main__":
    main()
