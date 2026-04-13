"""
Weekly Retrain Walk-Forward Backtest with Adaptive Feature Selection
=====================================================================

Tests whether weekly retraining + per-window feature selection improves
spread-model P&L vs monthly retraining with a fixed feature set.

Configs:
  1. Monthly retrain, fixed 50 features (baseline = current production)
  2. Weekly retrain, fixed 50 features
  3. Weekly retrain + adaptive features (importance > 1%)
  4. Weekly retrain + adaptive features (importance > 2%)

Walk-forward: expanding window from Oct 2024 to Apr 2026.
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
PLOT_DIR = BASE_DIR / "plots" / "eda" / "weekly_retrain"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LEAD = 8
QH = 0.25
THRESHOLD = 3

LGB_PARAMS = dict(learning_rate=0.03, num_leaves=15, min_child_samples=200,
                  subsample=0.5, colsample_bytree=0.5, reg_alpha=1.0,
                  reg_lambda=10.0, n_estimators=200, verbose=-1)

SELECTED_50 = [
    'da_price', 'cloudcover', 'hour_cos', 'idm_vwap_lag', 'da_supply',
    'da_price_change24h', 'proxy_rmax4', 'temp_forecast_da', 'temp_national_spread',
    'temp_bratislava', 'load_rmean16', 'nowcast_momentum_h2h3', 'temp_national_change6h',
    'da_demand', 'temp_surprise_lag', 'proxy_rmean16', 'proxy_range8', 'hour_sin',
    'prod_momentum', 'nowcast_pred_rmean4', 'nowcast_momentum_h4h5',
    'da_flow_cz', 'load_momentum', 'xborder_momentum', 'nowcast_h3', 'radiation_national',
    'da_net_import', 'proxy_rmean32', 'nowcast_trend_h2_h5', 'dow_sin',
    'reg_rmean8', 'reg_vol_rmean4', 'proxy_dev_from_hour', 'proxy_yesterday', 'prod_rmean8',
    'dow_cos', 'solar_surprise_lag', 'nowcast_h5', 'proxy_rmin4', 'nowcast_convergence',
    'reg_rmean4', 'is_weekend', 'proxy_yesterday_2', 'temp_rmean24h', 'proxy_range4',
    'proxy_lag12', 'proxy_pos_ratio_4', 'proxy_lag21', 'proxy_lag18', 'damas_fe_rmean4',
]


def generate_weekly_folds(start, end):
    """Generate weekly folds (Mon-Sun rolling)."""
    folds = []
    current = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while current < end_ts:
        next_week = current + pd.Timedelta(days=7)
        if next_week > end_ts:
            next_week = end_ts
        folds.append((current.strftime('%Y-%m-%d'), next_week.strftime('%Y-%m-%d')))
        current = next_week
    return folds


def generate_monthly_folds(start, end):
    """Generate monthly folds."""
    folds = []
    current = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while current < end_ts:
        # Next month start
        if current.month == 12:
            next_month = pd.Timestamp(f'{current.year + 1}-01-01')
        else:
            next_month = pd.Timestamp(f'{current.year}-{current.month + 1:02d}-01')
        if next_month > end_ts:
            next_month = end_ts
        folds.append((current.strftime('%Y-%m-%d'), next_month.strftime('%Y-%m-%d')))
        current = next_month
    return folds


def select_features_by_importance(model, feature_names, threshold_pct):
    """Select features with importance > threshold_pct of total."""
    imp = model.feature_importances_
    total = imp.sum()
    if total == 0:
        return feature_names, imp
    pct = imp / total * 100
    mask = pct >= threshold_pct
    selected = [f for f, m in zip(feature_names, mask) if m]
    if len(selected) < 5:
        # Fallback: keep top 10 if too aggressive
        top_idx = np.argsort(imp)[::-1][:10]
        selected = [feature_names[i] for i in top_idx]
    return selected, pct


def run_walkforward(df_base, features_list, folds, config_name,
                    adaptive=False, importance_threshold=1.0):
    """Run walk-forward backtest for one configuration.

    Returns: (all_trades_df, feature_selection_log)
    """
    all_trades = []
    feat_log = []  # (fold_idx, n_selected, selected_features)
    n_folds = len(folds)

    for i, (test_start, test_end) in enumerate(folds):
        # Train on everything before test_start
        train = df_base[df_base.index < test_start].dropna(
            subset=['spread_target', f'proxy_lag{LEAD+1}'])
        train = train[train['imb_settlement_price'].abs() <= 5000]

        test = df_base[(df_base.index >= test_start) & (df_base.index < test_end)].copy()
        test = test.dropna(subset=[f'proxy_lag{LEAD+1}'])
        test = test[test['spread_target'].notna()]
        test = test[test['exec_bid'].notna() & test['exec_ask'].notna() & (test['exec_spread'] <= 10)]

        if len(train) < 1000 or len(test) < 5:
            continue

        use_features = features_list

        if adaptive:
            # Stage 1: train on all features, get importance
            model_full = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
            model_full.fit(train[features_list].values, train['spread_target'].values)
            selected, pcts = select_features_by_importance(
                model_full, features_list, importance_threshold)
            use_features = selected
            feat_log.append((i, len(selected), selected))

        # Stage 2 (or only stage if not adaptive): train on selected features
        model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        model.fit(train[use_features].values, train['spread_target'].values)

        # Predict
        test['pred'] = model.predict(test[use_features].values)

        # Trade with confidence sizing: size = |pred|.clip(upper=5)
        surplus = test['pred'] <= -THRESHOLD
        deficit = test['pred'] >= THRESHOLD
        active = test[surplus | deficit].copy()

        if len(active) == 0:
            continue

        s = surplus.reindex(active.index, fill_value=False)
        d = deficit.reindex(active.index, fill_value=False)

        # Size = |pred| clipped at 5 MW
        active['size_mw'] = active['pred'].abs().clip(upper=5)
        active['energy'] = active['size_mw'] * QH

        active['pnl'] = 0.0
        active.loc[s, 'pnl'] = (
            (active.loc[s, 'exec_bid'] - active.loc[s, 'imb_settlement_price'])
            * active.loc[s, 'energy']
        )
        active.loc[d, 'pnl'] = (
            (active.loc[d, 'imb_settlement_price'] - active.loc[d, 'exec_ask'])
            * active.loc[d, 'energy']
        )

        all_trades.append(active[['pnl', 'pred', 'spread_target']])

        if (i + 1) % 10 == 0 or i == n_folds - 1:
            cum_pnl = sum(t['pnl'].sum() for t in all_trades)
            n_selected = len(use_features)
            print(f"  [{config_name}] Fold {i+1}/{n_folds}: "
                  f"test={test_start}..{test_end}, "
                  f"train={len(train)}, test={len(active)}/{len(test)}, "
                  f"feats={n_selected}, cum_pnl={cum_pnl:+,.0f}")

    if not all_trades:
        return pd.DataFrame(), feat_log

    result = pd.concat(all_trades)
    return result, feat_log


def compute_stats(trades_df, label):
    """Compute summary stats for a config."""
    if trades_df.empty:
        return {}
    daily = trades_df.groupby(trades_df.index.date)['pnl'].sum()
    weekly = trades_df.groupby(pd.Grouper(freq='W'))['pnl'].sum()
    n_days = len(daily)
    total = trades_df['pnl'].sum()
    stats = {
        'label': label,
        'total_eur': total,
        'eur_per_day': total / n_days if n_days > 0 else 0,
        'sharpe': (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0,
        'win_rate': (trades_df['pnl'] > 0).mean(),
        'n_trades': len(trades_df),
        'n_days': n_days,
        'losing_weeks': (weekly < 0).sum(),
        'total_weeks': len(weekly),
        'worst_week': weekly.min(),
        'daily_series': daily,
    }
    return stats


def print_period_breakdown(trades_df, label):
    """Print P&L split by period."""
    if trades_df.empty:
        print(f"  {label}: No trades")
        return
    periods = {
        '2024-Q4': ('2024-10-01', '2025-01-01'),
        '2025-H1': ('2025-01-01', '2025-07-01'),
        '2025-H2': ('2025-07-01', '2026-01-01'),
        '2026-Q1': ('2026-01-01', '2026-04-01'),
        '2026-Apr': ('2026-04-01', '2026-04-14'),
    }
    print(f"\n  --- {label} Period Breakdown ---")
    for pname, (pstart, pend) in periods.items():
        mask = (trades_df.index >= pstart) & (trades_df.index < pend)
        sub = trades_df[mask]
        if len(sub) == 0:
            print(f"    {pname}: no data")
            continue
        daily = sub.groupby(sub.index.date)['pnl'].sum()
        nd = len(daily)
        total = sub['pnl'].sum()
        spd = total / nd if nd > 0 else 0
        wr = (sub['pnl'] > 0).mean()
        sh = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        print(f"    {pname}: {total:+8,.0f} EUR ({spd:+.0f}/day), "
              f"Sharpe={sh:.1f}, WR={wr:.0%}, {len(sub)} trades, {nd} days")


def main():
    t0 = time.time()
    print("=" * 70)
    print("WEEKLY RETRAIN WALK-FORWARD WITH ADAPTIVE FEATURE SELECTION")
    print("=" * 70)

    # --- Load data ---
    data = load_all_data()
    tml.TRAIN_END = '2026-04-13'
    tml.TEST_START = '2026-04-13'
    df_base, feature_cols = build_features(data, LEAD)

    # Validate selected features
    spread_features = [f for f in SELECTED_50 if f in feature_cols]
    missing = [f for f in SELECTED_50 if f not in feature_cols]
    if missing:
        print(f"[!] Warning: {len(missing)} selected features not found: {missing}")
    print(f"[+] Using {len(spread_features)} of {len(SELECTED_50)} selected features")

    # Join execution prices
    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                          parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[
        ['bid', 'ask', 'spread', 'mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']

    df_base = df_base.join(ob_120, how='left')
    df_base['imb_settlement_price'] = df_base['imb_settle_price']

    # Raw 15-min spread target
    df_base['spread_target'] = df_base['imb_settlement_price'] - df_base['exec_mid']

    print(f"[+] Base data: {len(df_base)} rows, {len(spread_features)} features")
    print(f"[+] Date range: {df_base.index.min()} to {df_base.index.max()}")

    # --- Generate folds ---
    START = '2024-10-01'
    END = '2026-04-13'
    weekly_folds = generate_weekly_folds(START, END)
    monthly_folds = generate_monthly_folds(START, END)
    print(f"[+] Weekly folds: {len(weekly_folds)}, Monthly folds: {len(monthly_folds)}")

    # --- Run all 4 configs ---
    configs = {}

    # Config 1: Monthly retrain, fixed 50 features
    print(f"\n{'='*70}")
    print("CONFIG 1: Monthly retrain, fixed 50 features")
    print(f"{'='*70}")
    trades1, _ = run_walkforward(df_base, spread_features, monthly_folds,
                                  "Monthly-Fixed50", adaptive=False)
    configs['Monthly-Fixed50'] = (trades1, _)

    # Config 2: Weekly retrain, fixed 50 features
    print(f"\n{'='*70}")
    print("CONFIG 2: Weekly retrain, fixed 50 features")
    print(f"{'='*70}")
    trades2, _ = run_walkforward(df_base, spread_features, weekly_folds,
                                  "Weekly-Fixed50", adaptive=False)
    configs['Weekly-Fixed50'] = (trades2, _)

    # Config 3: Weekly retrain + adaptive features (1% threshold)
    print(f"\n{'='*70}")
    print("CONFIG 3: Weekly retrain + adaptive features (>1%)")
    print(f"{'='*70}")
    trades3, flog3 = run_walkforward(df_base, spread_features, weekly_folds,
                                      "Weekly-Adapt1pct", adaptive=True,
                                      importance_threshold=1.0)
    configs['Weekly-Adapt1pct'] = (trades3, flog3)

    # Config 4: Weekly retrain + adaptive features (2% threshold)
    print(f"\n{'='*70}")
    print("CONFIG 4: Weekly retrain + adaptive features (>2%)")
    print(f"{'='*70}")
    trades4, flog4 = run_walkforward(df_base, spread_features, weekly_folds,
                                      "Weekly-Adapt2pct", adaptive=True,
                                      importance_threshold=2.0)
    configs['Weekly-Adapt2pct'] = (trades4, flog4)

    elapsed = time.time() - t0
    print(f"\n[+] All configs done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY: ALL CONFIGURATIONS")
    print(f"{'='*70}")

    all_stats = {}
    for name, (trades, flog) in configs.items():
        stats = compute_stats(trades, name)
        all_stats[name] = stats
        if not stats:
            print(f"\n{name}: No trades")
            continue
        print(f"\n{name}:")
        print(f"  Total P&L:     {stats['total_eur']:+10,.0f} EUR")
        print(f"  EUR/day:       {stats['eur_per_day']:+10,.0f}")
        print(f"  Sharpe:        {stats['sharpe']:.2f}")
        print(f"  Win rate:      {stats['win_rate']:.0%}")
        print(f"  Trades:        {stats['n_trades']}")
        print(f"  Trading days:  {stats['n_days']}")
        print(f"  Losing weeks:  {stats['losing_weeks']} / {stats['total_weeks']}")
        print(f"  Worst week:    {stats['worst_week']:+,.0f} EUR")

        print_period_breakdown(trades, name)

    # --- Adaptive feature selection stats ---
    for name, threshold in [('Weekly-Adapt1pct', 1.0), ('Weekly-Adapt2pct', 2.0)]:
        trades, flog = configs[name]
        if not flog:
            continue
        n_selected = [x[1] for x in flog]
        print(f"\n--- Feature Selection Stats: {name} (>{threshold}%) ---")
        print(f"  Features per window: mean={np.mean(n_selected):.1f}, "
              f"min={np.min(n_selected)}, max={np.max(n_selected)}")

        # Track feature survival
        from collections import Counter
        survival = Counter()
        for _, _, feats in flog:
            for f in feats:
                survival[f] += 1
        n_windows = len(flog)

        always = [f for f, c in survival.items() if c == n_windows]
        never = [f for f in spread_features if f not in survival]
        rarely = [f for f, c in survival.most_common() if c < n_windows * 0.25]

        print(f"  Always selected ({len(always)}): {always[:15]}")
        if len(always) > 15:
            print(f"    ... and {len(always)-15} more")
        print(f"  Never selected ({len(never)}): {never}")
        print(f"  Rarely selected (<25% of windows, {len(rarely)}): "
              f"{[f for f,c in survival.most_common()[::-1] if c < n_windows*0.25][:10]}")

        # Top 10 most stable
        print(f"  Top 10 most selected:")
        for feat, cnt in survival.most_common(10):
            print(f"    {feat}: {cnt}/{n_windows} ({cnt/n_windows:.0%})")

        # Bottom 10
        print(f"  Bottom 10 least selected:")
        for feat, cnt in survival.most_common()[-10:]:
            print(f"    {feat}: {cnt}/{n_windows} ({cnt/n_windows:.0%})")

    # --- Plot: Cumulative P&L comparison ---
    print(f"\n[*] Generating comparison plot...")

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    colors = {
        'Monthly-Fixed50': '#2196F3',
        'Weekly-Fixed50': '#FF9800',
        'Weekly-Adapt1pct': '#4CAF50',
        'Weekly-Adapt2pct': '#E91E63',
    }

    # Panel 1: Cumulative P&L curves
    ax = axes[0]
    for name in configs:
        stats = all_stats.get(name)
        if not stats or 'daily_series' not in stats:
            continue
        daily = stats['daily_series']
        cum = daily.cumsum()
        dates = pd.to_datetime(cum.index)
        label = (f"{name} ({stats['total_eur']:+,.0f} EUR, "
                 f"Sharpe={stats['sharpe']:.1f})")
        ax.plot(dates, cum.values, color=colors.get(name, 'gray'),
                lw=2, label=label, alpha=0.9)

    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_ylabel('Cumulative P&L (EUR)')
    ax.set_title('Weekly vs Monthly Retrain: Cumulative P&L Comparison (OOS Walk-Forward)')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 2: Rolling 4-week P&L
    ax2 = axes[1]
    for name in configs:
        stats = all_stats.get(name)
        if not stats or 'daily_series' not in stats:
            continue
        daily = stats['daily_series']
        rolling = daily.rolling(28, min_periods=7).sum()
        dates = pd.to_datetime(rolling.index)
        ax2.plot(dates, rolling.values, color=colors.get(name, 'gray'),
                 lw=1.5, label=name, alpha=0.8)

    ax2.axhline(0, color='gray', ls='--', alpha=0.5)
    ax2.set_ylabel('Rolling 4-Week P&L (EUR)')
    ax2.set_xlabel('Date')
    ax2.set_title('Rolling 4-Week P&L')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = PLOT_DIR / "01_weekly_retrain_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Plot saved: {plot_path}")

    total_elapsed = time.time() - t0
    print(f"\n[+] Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print("[+] Done.")


if __name__ == '__main__':
    main()
