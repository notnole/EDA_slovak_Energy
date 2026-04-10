"""
Walk-Forward Monte Carlo Risk Analysis
========================================

Proper out-of-sample evaluation:
  1. Expanding window: train on all data up to month N
  2. Trade month N+1 with bid/ask execution
  3. Collect OOS trades across ALL months
  4. Bootstrap MC from the combined OOS distribution

This avoids the circularity of testing on the same data we selected the strategy from.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
REPO_ROOT = BASE_DIR.parent
PLOT_DIR = BASE_DIR / "plots" / "eda" / "montecarlo"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LEAD = 8
SIZE_MW = 5.0
QH = 0.25
ENERGY = SIZE_MW * QH

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                  reg_lambda=1.0, n_estimators=600, verbose=-1)

# Walk-forward folds: train on everything before, trade the month
# Start trading from Oct 2024 (need 6+ months of training)
FOLDS = [
    ('2024-10-01', '2024-10-01', '2024-11-01'),
    ('2024-11-01', '2024-11-01', '2024-12-01'),
    ('2024-12-01', '2024-12-01', '2025-01-01'),
    ('2025-01-01', '2025-01-01', '2025-02-01'),
    ('2025-02-01', '2025-02-01', '2025-03-01'),
    ('2025-03-01', '2025-03-01', '2025-04-01'),
    ('2025-04-01', '2025-04-01', '2025-05-01'),
    ('2025-05-01', '2025-05-01', '2025-06-01'),
    ('2025-06-01', '2025-06-01', '2025-07-01'),
    ('2025-07-01', '2025-07-01', '2025-08-01'),
    ('2025-08-01', '2025-08-01', '2025-09-01'),
    ('2025-09-01', '2025-09-01', '2025-10-01'),
    # Gap: Oct-Nov 2025 has sparse OB data, skip
    ('2025-12-01', '2025-12-01', '2026-01-01'),
    ('2026-01-01', '2026-01-01', '2026-02-01'),
    ('2026-02-01', '2026-02-01', '2026-03-01'),
    ('2026-03-01', '2026-03-01', '2026-04-01'),
]


def main():
    print("=" * 70)
    print("WALK-FORWARD MONTE CARLO RISK ANALYSIS")
    print("=" * 70)

    # Load all data once
    data = load_all_data()
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, feature_cols = build_features(data, LEAD)

    # Join execution + settlement
    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                          parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'spread', 'mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']

    mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                       parse_dates=['timestamp_hour'], index_col='timestamp_hour')
    mkt = mkt[~mkt.index.duplicated(keep='last')]

    df_base['hour_ts'] = df_base.index.floor('h')
    df_base = df_base.join(mkt[['imb_settlement_price']], on='hour_ts', how='left')
    df_base = df_base.join(ob_120, how='left')
    df_base['spread_target'] = df_base['imb_settlement_price'] - df_base['exec_mid']

    print(f"[+] Base data: {len(df_base)} rows, {len(feature_cols)} features")

    # ===== WALK-FORWARD =====
    all_oos_trades = []

    for train_end, test_start, test_end in FOLDS:
        print(f"\n--- Train < {train_end}, Trade [{test_start}, {test_end}) ---")

        train = df_base[df_base.index < train_end].dropna(
            subset=['spread_target', f'proxy_lag{LEAD+1}'])
        train = train[train['imb_settlement_price'].abs() <= 5000]

        test = df_base[(df_base.index >= test_start) & (df_base.index < test_end)].copy()
        test = test.dropna(subset=[f'proxy_lag{LEAD+1}'])
        test = test[test['spread_target'].notna()]
        test = test[test['exec_bid'].notna() & test['exec_ask'].notna() & (test['exec_spread'] <= 10)]

        if len(train) < 1000 or len(test) < 50:
            print(f"  Skipped: train={len(train)}, test={len(test)}")
            continue

        print(f"  Train: {len(train)}, Test: {len(test)}")

        # Train spread model
        model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        model.fit(train[feature_cols].values, train['spread_target'].values)

        # Predict
        test['pred'] = model.predict(test[feature_cols].values)

        # Trade with threshold=3
        surplus = test['pred'] <= -3
        deficit = test['pred'] >= 3
        active = test[surplus | deficit].copy()

        if len(active) < 10:
            print(f"  Too few trades: {len(active)}")
            continue

        s = surplus.reindex(active.index, fill_value=False)
        d = deficit.reindex(active.index, fill_value=False)
        active['pnl'] = 0.0
        active.loc[s, 'pnl'] = (active.loc[s, 'exec_bid'] - active.loc[s, 'imb_settlement_price']) * ENERGY
        active.loc[d, 'pnl'] = (active.loc[d, 'imb_settlement_price'] - active.loc[d, 'exec_ask']) * ENERGY
        active['direction'] = np.where(active['pred'] > 0, 'deficit', 'surplus')

        nd = active.index.normalize().nunique()
        total = active['pnl'].sum()
        wr = (active['pnl'] > 0).mean()
        daily = active.groupby(active.index.date)['pnl'].sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0

        print(f"  Trades: {len(active)}, {total:+,.0f} EUR ({total/nd:+.0f}/day), "
              f"Sharpe={sharpe:.1f}, Win={wr:.0%}")

        all_oos_trades.append(active[['pnl', 'pred', 'direction', 'spread_target']])

    # ===== COMBINE ALL OOS =====
    if not all_oos_trades:
        print("[!] No OOS trades collected")
        return

    oos = pd.concat(all_oos_trades)
    oos_daily = oos.groupby(oos.index.date)['pnl'].sum()
    tpd = oos.groupby(oos.index.date).size().values

    print(f"\n{'='*70}")
    print("COMBINED WALK-FORWARD OOS RESULTS")
    print(f"{'='*70}")
    print(f"Total OOS trades: {len(oos)}")
    print(f"Total OOS days:   {len(oos_daily)}")
    print(f"Date range:       {oos.index.min()} to {oos.index.max()}")
    print(f"Total P&L:        {oos['pnl'].sum():+,.0f} EUR")
    print(f"Per day:          {oos_daily.mean():+,.0f} EUR")
    sharpe_oos = oos_daily.mean() / oos_daily.std() * np.sqrt(252) if oos_daily.std() > 0 else 0
    print(f"Sharpe:           {sharpe_oos:.1f}")
    print(f"Win rate (trade): {(oos['pnl'] > 0).mean():.0%}")
    print(f"Win rate (daily): {(oos_daily > 0).mean():.0%}")
    print(f"Worst day:        {oos_daily.min():+,.0f} EUR")
    print(f"Trade P&L:        mean={oos['pnl'].mean():+.1f}, std={oos['pnl'].std():.1f}")
    print(f"Trades/day:       mean={tpd.mean():.0f}")

    # Monthly breakdown
    print(f"\n--- Monthly OOS P&L ---")
    oos['month'] = oos.index.to_period('M')
    monthly = oos.groupby('month')['pnl'].agg(['sum', 'count'])
    monthly['days'] = oos.groupby('month').apply(lambda x: x.index.normalize().nunique())
    monthly['per_day'] = monthly['sum'] / monthly['days']
    for m, r in monthly.iterrows():
        print(f"  {m}: {r['sum']:+8,.0f} EUR ({r['per_day']:+.0f}/day), {int(r['count'])} trades, {int(r['days'])} days")

    # ===== MONTE CARLO from walk-forward OOS =====
    print(f"\n{'='*70}")
    print("MONTE CARLO FROM WALK-FORWARD OOS")
    print(f"{'='*70}")

    trade_pnl = oos['pnl'].values
    avg_tpd = int(tpd.mean())
    N_SIMS = 50_000
    N_DAYS = 252

    np.random.seed(42)
    samples = np.random.choice(trade_pnl, size=(N_SIMS, N_DAYS, avg_tpd), replace=True)
    daily_sim = samples.sum(axis=2)
    yearly = daily_sim.sum(axis=1)
    worst_day_sim = daily_sim.min(axis=1)
    cumsum = daily_sim.cumsum(axis=1)
    running_max = np.maximum.accumulate(cumsum, axis=1)
    drawdowns = (cumsum - running_max).min(axis=1)
    flat_daily = daily_sim.flatten()

    var95 = np.percentile(flat_daily, 5)
    var99 = np.percentile(flat_daily, 1)
    cvar95 = flat_daily[flat_daily <= var95].mean()
    cvar99 = flat_daily[flat_daily <= var99].mean()

    print(f"\nSimulation: {N_SIMS:,} years x {N_DAYS} days x {avg_tpd} trades/day")
    print(f"\n--- Annual P&L ---")
    for p in [5, 25, 50, 75, 95]:
        print(f"  P{p:02d}: {np.percentile(yearly, p):+10,.0f} EUR")
    print(f"  Mean: {yearly.mean():+10,.0f} EUR ({yearly.mean()/N_DAYS:+.0f}/day)")
    print(f"  P(annual loss): {(yearly < 0).mean():.4%}")

    print(f"\n--- Daily VaR / CVaR ---")
    print(f"  VaR 95%:  {var95:+,.0f} EUR")
    print(f"  VaR 99%:  {var99:+,.0f} EUR")
    print(f"  CVaR 95%: {cvar95:+,.0f} EUR")
    print(f"  CVaR 99%: {cvar99:+,.0f} EUR")

    print(f"\n--- Max Drawdown ---")
    for p in [1, 5, 10, 50]:
        print(f"  P{p:02d}: {np.percentile(drawdowns, p):+,.0f} EUR")

    # ===== PLOTS =====
    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Walk-forward monthly P&L
    ax = fig.add_subplot(gs[0, 0])
    months = monthly.index.astype(str)
    colors = ['green' if v > 0 else 'red' for v in monthly['per_day']]
    ax.bar(range(len(months)), monthly['per_day'], color=colors, alpha=0.7)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, fontsize=7)
    ax.set_ylabel('EUR/day')
    ax.set_title('Walk-Forward OOS: Monthly P&L/day')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.axhline(oos_daily.mean(), color='blue', ls='--', alpha=0.5,
               label=f'Mean={oos_daily.mean():+.0f}/day')
    ax.legend(fontsize=9)

    # 2. OOS daily P&L distribution
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(oos_daily.values, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.3)
    ax.axvline(oos_daily.mean(), color='red', ls='--', label=f'Mean={oos_daily.mean():+.0f}')
    ax.axvline(0, color='gray', ls='-', alpha=0.5)
    ax.set_xlabel('Daily P&L (EUR)')
    ax.set_title(f'OOS Daily P&L (n={len(oos_daily)} days)')
    ax.legend()

    # 3. OOS equity curve
    ax = fig.add_subplot(gs[0, 2])
    eq = oos_daily.cumsum()
    ax.plot(range(len(eq)), eq.values, color='steelblue', lw=1.5)
    ax.fill_between(range(len(eq)), 0, eq.values, alpha=0.1, color='steelblue')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Cumulative P&L (EUR)')
    ax.set_title(f'Walk-Forward OOS Equity Curve (Sharpe={sharpe_oos:.1f})')

    # 4. Simulated daily P&L
    ax = fig.add_subplot(gs[1, 0])
    daily_sample = flat_daily[np.random.choice(len(flat_daily), 50000, replace=False)]
    ax.hist(daily_sample, bins=100, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.2)
    ax.axvline(var95, color='orange', ls='--', lw=2, label=f'VaR 95% = {var95:+.0f}')
    ax.axvline(var99, color='red', ls='--', lw=2, label=f'VaR 99% = {var99:+.0f}')
    ax.set_xlabel('Simulated Daily P&L (EUR)')
    ax.set_title('MC Daily P&L (from WF-OOS trades)')
    ax.legend(fontsize=9)

    # 5. Simulated annual P&L
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(yearly / 1000, bins=80, alpha=0.7, color='coral', edgecolor='black', linewidth=0.3)
    for p, c, ls in [(5, 'orange', '--'), (50, 'green', '--'), (95, 'blue', '--')]:
        v = np.percentile(yearly, p) / 1000
        ax.axvline(v, color=c, ls=ls, lw=2, label=f'P{p:02d} = {v:+.0f}k')
    ax.axvline(0, color='gray', ls='-', alpha=0.5)
    ax.set_xlabel('Annual P&L (kEUR)')
    ax.set_title(f'MC Annual P&L | P(loss)={((yearly<0).mean()*100):.2f}%')
    ax.legend(fontsize=9)

    # 6. Simulated equity curves
    ax = fig.add_subplot(gs[1, 2])
    for i in range(30):
        ax.plot(range(N_DAYS), daily_sim[i].cumsum(), alpha=0.2, lw=0.7, color='steelblue')
    median_eq = np.median(cumsum, axis=0)
    p5_eq = np.percentile(cumsum, 5, axis=0)
    p95_eq = np.percentile(cumsum, 95, axis=0)
    ax.plot(range(N_DAYS), median_eq, color='red', lw=2, label='P50')
    ax.fill_between(range(N_DAYS), p5_eq, p95_eq, alpha=0.15, color='red', label='P05-P95')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Cumulative P&L (EUR)')
    ax.set_title('MC Equity Curves (30 paths + P05/P50/P95)')
    ax.legend(fontsize=9)

    # 7. Max drawdown distribution
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(drawdowns, bins=80, alpha=0.7, color='indianred', edgecolor='black', linewidth=0.3)
    for p, c, ls in [(1, 'red', '-'), (5, 'orange', '--'), (50, 'green', '--')]:
        v = np.percentile(drawdowns, p)
        ax.axvline(v, color=c, ls=ls, lw=2, label=f'P{p:02d} = {v:+.0f}')
    ax.set_xlabel('Max Annual Drawdown (EUR)')
    ax.set_title('MC Max Drawdown Distribution')
    ax.legend(fontsize=9)

    # 8. VaR/CVaR ladder
    ax = fig.add_subplot(gs[2, 1])
    conf_levels = [90, 95, 97.5, 99, 99.5]
    vars_l, cvars_l = [], []
    for cl in conf_levels:
        v = np.percentile(flat_daily, 100 - cl)
        cv = flat_daily[flat_daily <= v].mean()
        vars_l.append(v)
        cvars_l.append(cv)
    x = range(len(conf_levels))
    ax.bar([i - 0.17 for i in x], [-v for v in vars_l], width=0.34, alpha=0.7,
           color='steelblue', label='VaR')
    ax.bar([i + 0.17 for i in x], [-cv for cv in cvars_l], width=0.34, alpha=0.7,
           color='indianred', label='CVaR')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{cl}%' for cl in conf_levels], fontsize=9)
    ax.set_xlabel('Confidence Level')
    ax.set_ylabel('Loss (EUR)')
    ax.set_title('Daily VaR and CVaR')
    ax.legend()

    # 9. Comparison: in-sample MC vs walk-forward MC
    ax = fig.add_subplot(gs[2, 2])
    # Load the in-sample result for comparison
    stk = pd.read_csv(BASE_DIR / "data" / "predictions" / "stacked_test_predictions.csv",
                       parse_dates=['datetime'], index_col='datetime')
    pred_is = stk['standalone_spread_pred']
    trades_is = stk[(pred_is <= -3) | (pred_is >= 3)].copy()
    trades_is['pnl'] = np.where(
        trades_is['standalone_spread_pred'] > 0,
        (trades_is['imb_settlement_price'] - trades_is['exec_ask']) * ENERGY,
        (trades_is['exec_bid'] - trades_is['imb_settlement_price']) * ENERGY)
    trades_is = trades_is[trades_is['pnl'].notna()]
    is_daily = trades_is.groupby(trades_is.index.date)['pnl'].sum()

    ax.hist(is_daily.values, bins=30, alpha=0.5, color='coral', label='In-sample (Feb-Mar)', density=True)
    ax.hist(oos_daily.values, bins=40, alpha=0.5, color='steelblue', label='Walk-forward OOS', density=True)
    ax.axvline(is_daily.mean(), color='coral', ls='--')
    ax.axvline(oos_daily.mean(), color='steelblue', ls='--')
    ax.set_xlabel('Daily P&L (EUR)')
    ax.set_title('In-Sample vs Walk-Forward OOS Daily P&L')
    ax.legend()

    fig.savefig(PLOT_DIR / "02_walkforward_montecarlo.png", bbox_inches='tight')
    plt.close(fig)
    print(f"\n[+] Saved: {PLOT_DIR / '02_walkforward_montecarlo.png'}")


if __name__ == "__main__":
    main()
