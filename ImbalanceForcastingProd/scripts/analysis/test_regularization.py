"""
Regularization Hyperparameter Grid Test
========================================

Tests 10 LightGBM configurations on the full 16-month walk-forward
to find if stronger regularization helps recent (2026) performance
without destroying historical (2024-2025) edge.

All configs use 52 selected features, hourly-smoothed spread target,
quantile regression alpha=0.50, threshold=3 EUR.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
import time
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
REPO_ROOT = BASE_DIR.parent
PLOT_DIR = BASE_DIR / "plots" / "eda" / "regularization"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LEAD = 8
SIZE_MW = 5.0
QH = 0.25
ENERGY = SIZE_MW * QH

SELECTED_FEATURES = [
    'da_price', 'cloudcover', 'hour_cos', 'idm_vwap_lag', 'da_supply',
    'da_price_change24h', 'proxy_rmax4', 'temp_forecast_da', 'temp_national_spread',
    'temp_bratislava', 'load_rmean16', 'nowcast_momentum_h2h3', 'temp_national_change6h',
    'da_demand', 'temp_surprise_lag', 'proxy_rmean16', 'proxy_range8', 'hour_sin',
    'spread_da_imb_lag', 'prod_momentum', 'nowcast_pred_rmean4', 'nowcast_momentum_h4h5',
    'da_flow_cz', 'load_momentum', 'xborder_momentum', 'nowcast_h3', 'radiation_national',
    'da_net_import', 'proxy_rmean32', 'nowcast_trend_h2_h5', 'dow_sin', 'imb_price_rmean4',
    'reg_rmean8', 'reg_vol_rmean4', 'proxy_dev_from_hour', 'proxy_yesterday', 'prod_rmean8',
    'dow_cos', 'solar_surprise_lag', 'nowcast_h5', 'proxy_rmin4', 'nowcast_convergence',
    'reg_rmean4', 'is_weekend', 'proxy_yesterday_2', 'temp_rmean24h', 'proxy_range4',
    'proxy_lag12', 'proxy_pos_ratio_4', 'proxy_lag21', 'proxy_lag18', 'damas_fe_rmean4',
]

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
    ('2025-12-01', '2025-12-01', '2026-01-01'),
    ('2026-01-01', '2026-01-01', '2026-02-01'),
    ('2026-02-01', '2026-02-01', '2026-03-01'),
    ('2026-03-01', '2026-03-01', '2026-04-01'),
]

# 10 configurations to test
CONFIGS = {
    "1_baseline": dict(
        num_leaves=63, n_estimators=600, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
        reg_lambda=1.0, min_child_samples=50, verbose=-1),
    "2_fewer_trees": dict(
        num_leaves=63, n_estimators=300, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
        reg_lambda=1.0, min_child_samples=50, verbose=-1),
    "3_fewer_trees_low_lr": dict(
        num_leaves=63, n_estimators=300, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
        reg_lambda=1.0, min_child_samples=50, verbose=-1),
    "4_shallow_31": dict(
        num_leaves=31, n_estimators=600, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
        reg_lambda=1.0, min_child_samples=50, verbose=-1),
    "5_very_shallow_15": dict(
        num_leaves=15, n_estimators=600, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
        reg_lambda=1.0, min_child_samples=50, verbose=-1),
    "6_heavy_L1": dict(
        num_leaves=63, n_estimators=600, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0,
        reg_lambda=1.0, min_child_samples=50, verbose=-1),
    "7_heavy_L2": dict(
        num_leaves=63, n_estimators=600, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
        reg_lambda=10.0, min_child_samples=50, verbose=-1),
    "8_aggr_subsample": dict(
        num_leaves=63, n_estimators=600, learning_rate=0.05,
        subsample=0.5, colsample_bytree=0.5, reg_alpha=0.1,
        reg_lambda=1.0, min_child_samples=50, verbose=-1),
    "9_kitchen_sink": dict(
        num_leaves=31, n_estimators=400, learning_rate=0.03,
        subsample=0.6, colsample_bytree=0.5, reg_alpha=0.5,
        reg_lambda=5.0, min_child_samples=100, verbose=-1),
    "10_very_simple": dict(
        num_leaves=15, n_estimators=200, learning_rate=0.03,
        subsample=0.5, colsample_bytree=0.5, reg_alpha=1.0,
        reg_lambda=10.0, min_child_samples=200, verbose=-1),
}


def run_walkforward(df_base, spread_features, lgb_params):
    """Run full walk-forward and return all OOS trades."""
    all_trades = []

    for train_end, test_start, test_end in FOLDS:
        train = df_base[df_base.index < train_end].dropna(
            subset=['spread_target', f'proxy_lag{LEAD+1}'])
        train = train[train['imb_settlement_price'].abs() <= 5000]

        test = df_base[(df_base.index >= test_start) & (df_base.index < test_end)].copy()
        test = test.dropna(subset=[f'proxy_lag{LEAD+1}'])
        test = test[test['spread_target'].notna()]
        test = test[test['exec_bid'].notna() & test['exec_ask'].notna() & (test['exec_spread'] <= 10)]

        if len(train) < 1000 or len(test) < 50:
            continue

        model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **lgb_params)
        model.fit(train[spread_features].values, train['spread_target'].values)

        test['pred'] = model.predict(test[spread_features].values)

        surplus = test['pred'] <= -3
        deficit = test['pred'] >= 3
        active = test[surplus | deficit].copy()

        if len(active) < 10:
            continue

        s = surplus.reindex(active.index, fill_value=False)
        d = deficit.reindex(active.index, fill_value=False)
        active['pnl'] = 0.0
        active.loc[s, 'pnl'] = (active.loc[s, 'exec_bid'] - active.loc[s, 'imb_settlement_price']) * ENERGY
        active.loc[d, 'pnl'] = (active.loc[d, 'imb_settlement_price'] - active.loc[d, 'exec_ask']) * ENERGY

        all_trades.append(active[['pnl', 'pred']])

    if not all_trades:
        return None
    return pd.concat(all_trades)


def compute_stats(oos):
    """Compute summary stats from OOS trades."""
    if oos is None or len(oos) == 0:
        return None

    daily = oos.groupby(oos.index.date)['pnl'].sum()
    monthly = oos.groupby(oos.index.to_period('M'))['pnl'].sum()
    monthly_days = oos.groupby(oos.index.to_period('M')).apply(
        lambda x: x.index.normalize().nunique())
    monthly_ppd = monthly / monthly_days

    n_days = len(daily)
    total = oos['pnl'].sum()
    eur_day = daily.mean()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    win_rate_trade = (oos['pnl'] > 0).mean()
    win_rate_day = (daily > 0).mean()
    worst_month_ppd = monthly_ppd.min()
    worst_month_label = str(monthly_ppd.idxmin())
    n_losing_months = (monthly_ppd < 0).sum()
    n_months = len(monthly_ppd)

    # Split 2024-2025 vs 2026
    oos_early = oos[oos.index < '2026-01-01']
    oos_late = oos[oos.index >= '2026-01-01']

    early_daily = oos_early.groupby(oos_early.index.date)['pnl'].sum() if len(oos_early) > 0 else pd.Series(dtype=float)
    late_daily = oos_late.groupby(oos_late.index.date)['pnl'].sum() if len(oos_late) > 0 else pd.Series(dtype=float)

    early_ppd = early_daily.mean() if len(early_daily) > 0 else 0
    late_ppd = late_daily.mean() if len(late_daily) > 0 else 0
    early_sharpe = early_daily.mean() / early_daily.std() * np.sqrt(252) if len(early_daily) > 1 and early_daily.std() > 0 else 0
    late_sharpe = late_daily.mean() / late_daily.std() * np.sqrt(252) if len(late_daily) > 1 and late_daily.std() > 0 else 0

    return {
        'n_trades': len(oos),
        'n_days': n_days,
        'total_eur': total,
        'eur_day': eur_day,
        'sharpe': sharpe,
        'win_trade': win_rate_trade,
        'win_day': win_rate_day,
        'worst_month_ppd': worst_month_ppd,
        'worst_month': worst_month_label,
        'losing_months': n_losing_months,
        'total_months': n_months,
        'early_ppd': early_ppd,
        'early_sharpe': early_sharpe,
        'late_ppd': late_ppd,
        'late_sharpe': late_sharpe,
        'monthly_ppd': monthly_ppd,
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("REGULARIZATION HYPERPARAMETER GRID TEST")
    print("10 configs x 16-month walk-forward")
    print("=" * 80)

    # Load data once
    data = load_all_data()
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, feature_cols = build_features(data, LEAD)

    spread_features = [f for f in SELECTED_FEATURES if f in feature_cols]
    missing = [f for f in SELECTED_FEATURES if f not in feature_cols]
    if missing:
        print(f"[!] Warning: {len(missing)} features not found: {missing}")
    print(f"[+] Using {len(spread_features)} features")

    # Join execution prices
    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                          parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'spread', 'mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']

    df_base = df_base.join(ob_120, how='left')
    df_base['imb_settlement_price'] = df_base['imb_settle_price']

    # Hourly-smoothed spread target
    df_base['hour_ts'] = df_base.index.floor('h')
    df_base['settle_hourly'] = df_base.groupby('hour_ts')['imb_settlement_price'].transform('mean')
    df_base['mid_hourly'] = df_base.groupby('hour_ts')['exec_mid'].transform('mean')
    df_base['spread_target'] = df_base['settle_hourly'] - df_base['mid_hourly']

    print(f"[+] Data ready: {len(df_base)} rows")
    print()

    # Run all configs
    results = {}
    for name, params in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"[*] Config: {name}")
        key_diffs = []
        baseline = CONFIGS["1_baseline"]
        for k in ['num_leaves', 'n_estimators', 'learning_rate', 'subsample',
                   'colsample_bytree', 'reg_alpha', 'reg_lambda', 'min_child_samples']:
            if params.get(k) != baseline.get(k):
                key_diffs.append(f"{k}={params.get(k)}")
        if key_diffs:
            print(f"    Changes vs baseline: {', '.join(key_diffs)}")
        else:
            print(f"    (baseline reference)")
        print(f"{'='*60}")

        t1 = time.time()
        oos = run_walkforward(df_base, spread_features, params)
        elapsed = time.time() - t1

        stats = compute_stats(oos)
        if stats is None:
            print(f"  [!] No trades produced")
            continue

        results[name] = stats
        print(f"  Trades: {stats['n_trades']}, Days: {stats['n_days']}")
        print(f"  Total:  {stats['total_eur']:+,.0f} EUR | EUR/day: {stats['eur_day']:+,.0f}")
        print(f"  Sharpe: {stats['sharpe']:.2f} | Win(trade): {stats['win_trade']:.0%} | Win(day): {stats['win_day']:.0%}")
        print(f"  Worst month: {stats['worst_month']} ({stats['worst_month_ppd']:+,.0f}/day)")
        print(f"  Losing months: {stats['losing_months']}/{stats['total_months']}")
        print(f"  --- Period split ---")
        print(f"  2024-2025: {stats['early_ppd']:+,.0f}/day (Sharpe={stats['early_sharpe']:.2f})")
        print(f"  2026:      {stats['late_ppd']:+,.0f}/day (Sharpe={stats['late_sharpe']:.2f})")
        print(f"  Time: {elapsed:.0f}s")

    # ===== SUMMARY TABLE =====
    print(f"\n\n{'='*80}")
    print("SUMMARY: ALL CONFIGS")
    print(f"{'='*80}")
    print(f"{'Config':<28} {'EUR/d':>7} {'Sharpe':>7} {'Win%':>6} {'Worst':>8} "
          f"{'Lose':>5} {'24-25':>8} {'2026':>8} {'26Sh':>6}")
    print("-" * 95)
    for name, s in results.items():
        print(f"{name:<28} {s['eur_day']:>+7.0f} {s['sharpe']:>7.2f} {s['win_trade']*100:>5.0f}% "
              f"{s['worst_month_ppd']:>+8.0f} {s['losing_months']:>3}/{s['total_months']:<2} "
              f"{s['early_ppd']:>+8.0f} {s['late_ppd']:>+8.0f} {s['late_sharpe']:>6.2f}")

    # Find best for 2026
    if results:
        best_2026 = max(results.items(), key=lambda x: x[1]['late_ppd'])
        best_overall = max(results.items(), key=lambda x: x[1]['eur_day'])
        best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe'])
        print(f"\n--- BEST ---")
        print(f"  Best overall EUR/day: {best_overall[0]} ({best_overall[1]['eur_day']:+,.0f}/day)")
        print(f"  Best Sharpe:          {best_sharpe[0]} ({best_sharpe[1]['sharpe']:.2f})")
        print(f"  Best 2026 EUR/day:    {best_2026[0]} ({best_2026[1]['late_ppd']:+,.0f}/day)")

    total_time = time.time() - t0
    print(f"\n[+] Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)")

    # ===== PLOT =====
    if len(results) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        names = list(results.keys())
        short_names = [n.split('_', 1)[1] for n in names]

        # 1. EUR/day comparison
        ax = axes[0, 0]
        vals = [results[n]['eur_day'] for n in names]
        colors = ['green' if v > 0 else 'red' for v in vals]
        ax.barh(range(len(names)), vals, color=colors, alpha=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(short_names, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color='gray', ls='-', alpha=0.5)
        ax.set_xlabel('EUR/day')
        ax.set_title('Overall EUR/day (16-month WF)')

        # 2. 2024-2025 vs 2026 split
        ax = axes[0, 1]
        early = [results[n]['early_ppd'] for n in names]
        late = [results[n]['late_ppd'] for n in names]
        y = np.arange(len(names))
        ax.barh(y - 0.17, early, height=0.34, alpha=0.7, color='steelblue', label='2024-2025')
        ax.barh(y + 0.17, late, height=0.34, alpha=0.7, color='coral', label='2026')
        ax.set_yticks(y)
        ax.set_yticklabels(short_names, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color='gray', ls='-', alpha=0.5)
        ax.set_xlabel('EUR/day')
        ax.set_title('Period Split: 2024-2025 vs 2026')
        ax.legend()

        # 3. Sharpe comparison
        ax = axes[1, 0]
        sharpes = [results[n]['sharpe'] for n in names]
        colors_s = ['green' if v > 0 else 'red' for v in sharpes]
        ax.barh(range(len(names)), sharpes, color=colors_s, alpha=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(short_names, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color='gray', ls='-', alpha=0.5)
        ax.set_xlabel('Sharpe Ratio (annualized)')
        ax.set_title('Sharpe Ratio')

        # 4. Monthly P&L heatmap
        ax = axes[1, 1]
        # Show monthly P&L/day for each config
        all_months = sorted(set().union(*[set(results[n]['monthly_ppd'].index) for n in names]))
        data_matrix = []
        for n in names:
            row = []
            for m in all_months:
                if m in results[n]['monthly_ppd'].index:
                    row.append(results[n]['monthly_ppd'][m])
                else:
                    row.append(np.nan)
            data_matrix.append(row)
        data_matrix = np.array(data_matrix)

        im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn', vmin=-1500, vmax=1500)
        ax.set_xticks(range(len(all_months)))
        ax.set_xticklabels([str(m) for m in all_months], rotation=45, fontsize=7, ha='right')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(short_names, fontsize=9)
        ax.set_title('Monthly EUR/day by Config')
        plt.colorbar(im, ax=ax, label='EUR/day')

        fig.suptitle('Regularization Grid Test: 10 Configs x 16-Month Walk-Forward', fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "regularization_grid.png", bbox_inches='tight', dpi=120)
        plt.close(fig)
        print(f"\n[+] Saved plot: {PLOT_DIR / 'regularization_grid.png'}")


if __name__ == "__main__":
    main()
