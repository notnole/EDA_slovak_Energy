"""
Combined Improvements Test: Feature Reduction + Regularization
================================================================

Tests whether two independent improvements STACK:
  1. Fewer features (top30, top20, 8-feature market+time)
  2. Stronger regularization (simple, kitchen_sink, shallow)

14 configurations on full 16-month walk-forward.
Confidence sizing: size = |pred|.clip(upper=5), energy = size * 0.25.
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
PLOT_DIR = BASE_DIR / "plots" / "eda" / "combined_improvements"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LEAD = 8
QH = 0.25
THRESHOLD = 3

# ============================================================
# FEATURE SETS (permutation importance order)
# ============================================================

ALL_52 = [
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
TOP_30 = ALL_52[:30]
TOP_20 = ALL_52[:20]
MKT_TIME = ['idm_vwap_lag', 'spread_da_imb_lag', 'imb_price_rmean4',
            'hour_cos', 'hour_sin', 'dow_sin', 'dow_cos', 'is_weekend']

FEATURE_SETS = {
    'all52': ALL_52,
    'top30': TOP_30,
    'top20': TOP_20,
    'mkt_time': MKT_TIME,
}

# ============================================================
# PARAMETER SETS
# ============================================================

BASELINE_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
                       subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                       reg_lambda=1.0, n_estimators=600, verbose=-1)

SIMPLE_PARAMS = dict(learning_rate=0.03, num_leaves=15, min_child_samples=200,
                     subsample=0.5, colsample_bytree=0.5, reg_alpha=1.0,
                     reg_lambda=10.0, n_estimators=200, verbose=-1)

KITCHEN_PARAMS = dict(learning_rate=0.03, num_leaves=31, min_child_samples=100,
                      subsample=0.6, colsample_bytree=0.5, reg_alpha=0.5,
                      reg_lambda=5.0, n_estimators=400, verbose=-1)

SHALLOW_PARAMS = dict(learning_rate=0.05, num_leaves=31, min_child_samples=50,
                      subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                      reg_lambda=1.0, n_estimators=600, verbose=-1)

# ============================================================
# CONFIGURATIONS
# ============================================================

CONFIGS = {
    # Baseline
    '52f_baseline':  {'features': 'all52',    'params': BASELINE_PARAMS},
    # Feature reduction only (baseline params)
    '30f_baseline':  {'features': 'top30',    'params': BASELINE_PARAMS},
    '8f_baseline':   {'features': 'mkt_time', 'params': BASELINE_PARAMS},
    # Regularization only (52 features)
    '52f_simple':    {'features': 'all52',    'params': SIMPLE_PARAMS},
    '52f_kitchen':   {'features': 'all52',    'params': KITCHEN_PARAMS},
    '52f_shallow':   {'features': 'all52',    'params': SHALLOW_PARAMS},
    # COMBINED: fewer features + stronger regularization
    '30f_simple':    {'features': 'top30',    'params': SIMPLE_PARAMS},
    '30f_kitchen':   {'features': 'top30',    'params': KITCHEN_PARAMS},
    '30f_shallow':   {'features': 'top30',    'params': SHALLOW_PARAMS},
    '8f_simple':     {'features': 'mkt_time', 'params': SIMPLE_PARAMS},
    '8f_kitchen':    {'features': 'mkt_time', 'params': KITCHEN_PARAMS},
    '8f_shallow':    {'features': 'mkt_time', 'params': SHALLOW_PARAMS},
    # Also test top 20 + regularization
    '20f_simple':    {'features': 'top20',    'params': SIMPLE_PARAMS},
    '20f_kitchen':   {'features': 'top20',    'params': KITCHEN_PARAMS},
}

# ============================================================
# WALK-FORWARD FOLDS
# ============================================================

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


def run_walkforward(df_base, feature_list, lgb_params, feature_cols):
    """Run full 16-month walk-forward. Returns DataFrame of all OOS trades."""
    feats = [f for f in feature_list if f in feature_cols]
    if len(feats) < 3:
        return None

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
        model.fit(train[feats].values, train['spread_target'].values)

        test['pred'] = model.predict(test[feats].values)

        # Threshold filter
        surplus = test['pred'] <= -THRESHOLD
        deficit = test['pred'] >= THRESHOLD
        active = test[surplus | deficit].copy()

        if len(active) < 5:
            continue

        s = surplus.reindex(active.index, fill_value=False)
        d = deficit.reindex(active.index, fill_value=False)

        # Confidence sizing: size = |pred|.clip(upper=5), energy = size * 0.25
        active['size_mw'] = active['pred'].abs().clip(upper=5.0)
        active['energy'] = active['size_mw'] * QH

        active['pnl'] = 0.0
        active.loc[s, 'pnl'] = (active.loc[s, 'exec_bid'] - active.loc[s, 'imb_settlement_price']) * active.loc[s, 'energy']
        active.loc[d, 'pnl'] = (active.loc[d, 'imb_settlement_price'] - active.loc[d, 'exec_ask']) * active.loc[d, 'energy']

        all_trades.append(active[['pnl', 'pred']])

    if not all_trades:
        return None
    return pd.concat(all_trades)


def compute_stats(oos):
    """Compute summary stats from OOS trades DataFrame."""
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
        'late_ppd': late_ppd,
        'monthly_ppd': monthly_ppd,
        'daily_pnl': daily,
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("COMBINED IMPROVEMENTS TEST: Feature Reduction + Regularization")
    print("14 configs x 16-month walk-forward, confidence sizing")
    print("=" * 80)

    # --- Load data ---
    print("\n[*] Loading data...")
    data = load_all_data()
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, feature_cols = build_features(data, LEAD)

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

    print(f"[+] Data ready: {len(df_base)} rows, {len(feature_cols)} total features")

    # Validate feature sets
    for fs_name, fs_list in FEATURE_SETS.items():
        avail = [f for f in fs_list if f in feature_cols]
        missing = [f for f in fs_list if f not in feature_cols]
        print(f"  {fs_name}: {len(avail)}/{len(fs_list)} available", end='')
        if missing:
            print(f" (missing: {missing})", end='')
        print()

    # --- Run all configs ---
    results = {}
    for name, cfg in CONFIGS.items():
        feat_list = FEATURE_SETS[cfg['features']]
        params = cfg['params']

        print(f"\n{'='*70}")
        print(f"[*] Config: {name} ({len(feat_list)}f, "
              f"lr={params['learning_rate']}, leaves={params['num_leaves']}, "
              f"n_est={params['n_estimators']})")
        print(f"{'='*70}")

        t1 = time.time()
        oos = run_walkforward(df_base, feat_list, params, feature_cols)
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
        print(f"  2024-2025: {stats['early_ppd']:+,.0f}/day | 2026: {stats['late_ppd']:+,.0f}/day")
        print(f"  Time: {elapsed:.0f}s")

    if not results:
        print("[!] No configs produced results")
        return

    # ===== SUMMARY TABLE (sorted by Sharpe) =====
    print(f"\n\n{'='*120}")
    print("SUMMARY TABLE -- SORTED BY SHARPE DESCENDING")
    print(f"{'='*120}")
    print(f"{'Config':<18} {'Total EUR':>10} {'EUR/day':>8} {'Sharpe':>7} {'Win%d':>6} "
          f"{'LoseM':>6} {'Worst/d':>8} {'24-25/d':>8} {'2026/d':>8} {'Trades':>7}")
    print("-" * 120)

    sorted_configs = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    for name, s in sorted_configs:
        print(f"{name:<18} {s['total_eur']:>+10,.0f} {s['eur_day']:>+8.0f} {s['sharpe']:>7.2f} "
              f"{s['win_day']*100:>5.0f}% {s['losing_months']:>3}/{s['total_months']:<2} "
              f"{s['worst_month_ppd']:>+8.0f} {s['early_ppd']:>+8.0f} {s['late_ppd']:>+8.0f} "
              f"{s['n_trades']:>7}")

    # ===== INTERACTION ANALYSIS =====
    print(f"\n\n{'='*80}")
    print("INTERACTION ANALYSIS: Do improvements stack?")
    print(f"{'='*80}")

    # Compare baseline, feature-only, reg-only, and combined
    combos = [
        ('Feature reduction (30f)', '52f_baseline', '30f_baseline'),
        ('Feature reduction (8f)',  '52f_baseline', '8f_baseline'),
        ('Regularization (simple)', '52f_baseline', '52f_simple'),
        ('Regularization (kitchen)','52f_baseline', '52f_kitchen'),
        ('30f + simple',            '52f_baseline', '30f_simple'),
        ('30f + kitchen',           '52f_baseline', '30f_kitchen'),
        ('30f + shallow',           '52f_baseline', '30f_shallow'),
        ('20f + simple',            '52f_baseline', '20f_simple'),
        ('20f + kitchen',           '52f_baseline', '20f_kitchen'),
        ('8f + simple',             '52f_baseline', '8f_simple'),
        ('8f + kitchen',            '52f_baseline', '8f_kitchen'),
    ]

    bl_ppd = results.get('52f_baseline', {}).get('eur_day', 0)
    bl_sharpe = results.get('52f_baseline', {}).get('sharpe', 0)

    print(f"\n  Baseline: {bl_ppd:+.0f}/day, Sharpe={bl_sharpe:.2f}")
    print(f"\n  {'Improvement':<25} {'EUR/day':>8} {'Delta':>8} {'Sharpe':>7} {'dSharpe':>8}")
    print("  " + "-" * 60)
    for label, _, cfg_name in combos:
        if cfg_name in results:
            s = results[cfg_name]
            delta = s['eur_day'] - bl_ppd
            d_sharpe = s['sharpe'] - bl_sharpe
            print(f"  {label:<25} {s['eur_day']:>+8.0f} {delta:>+8.0f} {s['sharpe']:>7.2f} {d_sharpe:>+8.2f}")
        else:
            print(f"  {label:<25} {'N/A':>8}")

    # Additivity check
    print(f"\n--- Additivity check ---")
    print("  If improvements stack linearly, combined delta ~ sum of individual deltas")
    additivity_checks = [
        ('30f+simple', '30f_baseline', '52f_simple', '30f_simple'),
        ('30f+kitchen', '30f_baseline', '52f_kitchen', '30f_kitchen'),
        ('8f+simple',  '8f_baseline',  '52f_simple',  '8f_simple'),
        ('8f+kitchen', '8f_baseline',  '52f_kitchen',  '8f_kitchen'),
        ('20f+simple', None, '52f_simple', '20f_simple'),
        ('20f+kitchen', None, '52f_kitchen', '20f_kitchen'),
    ]

    print(f"\n  {'Combo':<16} {'Feat dlt':>9} {'Reg dlt':>9} {'Sum':>9} {'Actual':>9} {'Super?':>7}")
    print("  " + "-" * 65)
    for label, feat_cfg, reg_cfg, combo_cfg in additivity_checks:
        if combo_cfg not in results:
            continue
        feat_delta = results[feat_cfg]['eur_day'] - bl_ppd if feat_cfg and feat_cfg in results else 0
        reg_delta = results[reg_cfg]['eur_day'] - bl_ppd if reg_cfg in results else 0
        expected_sum = feat_delta + reg_delta
        actual = results[combo_cfg]['eur_day'] - bl_ppd
        is_super = "YES" if actual > expected_sum else "no"
        print(f"  {label:<16} {feat_delta:>+9.0f} {reg_delta:>+9.0f} {expected_sum:>+9.0f} "
              f"{actual:>+9.0f} {is_super:>7}")

    # ===== MONTHLY BREAKDOWN for top 5 =====
    print(f"\n\n{'='*100}")
    print("MONTHLY EUR/DAY -- TOP 5 BY SHARPE")
    print(f"{'='*100}")

    top5 = sorted_configs[:5]
    all_months = sorted(set().union(*[set(results[n]['monthly_ppd'].index) for n, _ in top5]))

    header = f"{'Month':<10}"
    for name, _ in top5:
        header += f" {name:>14}"
    print(header)
    print("-" * (10 + 15 * len(top5)))

    for m in all_months:
        row = f"{str(m):<10}"
        for name, s in top5:
            if m in s['monthly_ppd'].index:
                row += f" {s['monthly_ppd'][m]:>+14.0f}"
            else:
                row += f" {'---':>14}"
        print(row)

    # ===== BEST CONFIG =====
    best_name, best_stats = sorted_configs[0]
    print(f"\n\n{'='*80}")
    print(f"BEST CONFIG: {best_name}")
    print(f"{'='*80}")
    print(f"  Total P&L:     {best_stats['total_eur']:+,.0f} EUR")
    print(f"  EUR/day:       {best_stats['eur_day']:+,.0f}")
    print(f"  Sharpe:        {best_stats['sharpe']:.2f}")
    print(f"  Win rate(day): {best_stats['win_day']:.0%}")
    print(f"  Losing months: {best_stats['losing_months']}/{best_stats['total_months']}")
    print(f"  Worst month:   {best_stats['worst_month']} ({best_stats['worst_month_ppd']:+,.0f}/day)")
    print(f"  2024-2025:     {best_stats['early_ppd']:+,.0f}/day")
    print(f"  2026:          {best_stats['late_ppd']:+,.0f}/day")

    # ===== PLOTS =====
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Combined Improvements: Feature Reduction + Regularization\n'
                 '14 configs, 16-month walk-forward, confidence sizing',
                 fontsize=14, fontweight='bold')

    names_sorted = [n for n, _ in sorted_configs]
    short_names = names_sorted

    # 1. EUR/day bars
    ax = axes[0, 0]
    vals = [results[n]['eur_day'] for n in names_sorted]
    colors = []
    for n in names_sorted:
        if 'baseline' in n and '52f' in n:
            colors.append('#2196f3')  # blue = baseline
        elif 'baseline' in n:
            colors.append('#4caf50')  # green = feat only
        elif '52f' in n:
            colors.append('#ff9800')  # orange = reg only
        else:
            colors.append('#e91e63')  # pink = combined
    ax.barh(range(len(names_sorted)), vals, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', ls='-', alpha=0.5)
    ax.set_xlabel('EUR/day')
    ax.set_title('EUR/day (sorted by Sharpe)')
    for i, v in enumerate(vals):
        ax.text(v + 10, i, f'{v:+.0f}', va='center', fontsize=8)

    # 2. 2024-2025 vs 2026 split
    ax = axes[0, 1]
    early = [results[n]['early_ppd'] for n in names_sorted]
    late = [results[n]['late_ppd'] for n in names_sorted]
    y = np.arange(len(names_sorted))
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
    sharpes = [results[n]['sharpe'] for n in names_sorted]
    ax.barh(range(len(names_sorted)), sharpes, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', ls='-', alpha=0.5)
    ax.set_xlabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio (sorted)')
    for i, v in enumerate(sharpes):
        ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=8)

    # 4. Equity curves for top 5
    ax = axes[1, 1]
    cmap = plt.cm.tab10
    for i, (name, s) in enumerate(sorted_configs[:5]):
        eq = s['daily_pnl'].sort_index().cumsum()
        ax.plot(pd.to_datetime(eq.index), eq.values,
                label=f"{name} (Sh={s['sharpe']:.2f})",
                color=cmap(i), lw=1.5, alpha=0.9)
    ax.axhline(0, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative P&L (EUR)')
    ax.set_title('Equity Curves -- Top 5 by Sharpe')
    ax.legend(fontsize=8, loc='best')
    ax.tick_params(axis='x', rotation=45)

    fig.tight_layout()
    plot_path = PLOT_DIR / "01_combined_improvements.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[+] Plot saved: {plot_path}")

    total_time = time.time() - t0
    print(f"\n[+] Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("[+] Done.")


if __name__ == "__main__":
    main()
