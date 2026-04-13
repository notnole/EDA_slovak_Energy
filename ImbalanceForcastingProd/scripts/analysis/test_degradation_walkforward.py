"""
Walk-Forward Signal Degradation Analysis
==========================================

Combines the walk-forward methodology (16 monthly folds, Oct 2024 - Mar 2026)
with signal degradation testing to get stable feature importance estimates
across the FULL out-of-sample period -- not just Feb-Mar 2026.

For each fold:
  1. Train on all data before the month (clean)
  2. Predict with full features (baseline)
  3. Predict with each feature group replaced by random noise
  4. Collect P&L across all 16 months

This avoids the short-sample bias of testing degradation on only 2 months.
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
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots" / "signal_degradation"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (16, 8), "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})

# ============================================================
# CONFIG
# ============================================================

LEAD = 8
SIZE_MW = 5.0
QH = 0.25
ENERGY = SIZE_MW * QH
THRESHOLD = 3
N_SEEDS = 3  # seeds per fold per group (3 x 16 folds = 48 samples)

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                  reg_lambda=1.0, n_estimators=600, verbose=-1)

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

FEATURE_GROUPS = {
    'proxy_regulation': ['proxy_rmax4', 'proxy_rmean16', 'proxy_range8', 'proxy_rmean32',
                         'proxy_dev_from_hour', 'proxy_yesterday', 'proxy_rmin4', 'proxy_range4',
                         'proxy_lag12', 'proxy_pos_ratio_4', 'proxy_lag21', 'proxy_lag18',
                         'proxy_yesterday_2', 'reg_rmean8', 'reg_vol_rmean4', 'reg_rmean4'],
    'weather': ['cloudcover', 'temp_forecast_da', 'temp_national_spread', 'temp_bratislava',
                'temp_national_change6h', 'temp_surprise_lag', 'radiation_national', 'temp_rmean24h'],
    'da_prices': ['da_price', 'da_supply', 'da_price_change24h', 'da_demand', 'da_flow_cz', 'da_net_import'],
    'load_nowcast': ['nowcast_momentum_h2h3', 'nowcast_pred_rmean4', 'nowcast_momentum_h4h5',
                     'nowcast_h3', 'nowcast_trend_h2_h5', 'nowcast_h5', 'nowcast_convergence'],
    'time': ['hour_cos', 'hour_sin', 'dow_sin', 'dow_cos', 'is_weekend'],
    'market_idm': ['idm_vwap_lag', 'spread_da_imb_lag', 'imb_price_rmean4'],
    'load_scada': ['load_rmean16', 'load_momentum'],
    'other': ['prod_momentum', 'xborder_momentum', 'prod_rmean8', 'solar_surprise_lag', 'damas_fe_rmean4'],
}

# Walk-forward folds (same as walkforward_montecarlo.py)
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
    # Gap: Oct-Nov 2025 has sparse OB data
    ('2025-12-01', '2025-12-01', '2026-01-01'),
    ('2026-01-01', '2026-01-01', '2026-02-01'),
    ('2026-02-01', '2026-02-01', '2026-03-01'),
    ('2026-03-01', '2026-03-01', '2026-04-01'),
]


# ============================================================
# HELPERS
# ============================================================

def degrade_full_replace(test_X, group_features, rng):
    """Replace group features with random noise matching mean/std."""
    degraded = test_X.copy()
    for feat in group_features:
        if feat not in degraded.columns:
            continue
        col = degraded[feat]
        mu, sigma = col.mean(), col.std()
        if sigma == 0 or np.isnan(sigma):
            sigma = 1.0
        degraded[feat] = rng.normal(mu, sigma, size=len(col))
    return degraded


def calc_trades(test_df, predictions):
    """Return per-trade P&L series for a set of predictions."""
    t = test_df.copy()
    t['pred'] = predictions
    surplus = t['pred'] <= -THRESHOLD
    deficit = t['pred'] >= THRESHOLD
    active = t[surplus | deficit].copy()
    if len(active) < 5:
        return pd.Series(dtype=float)
    s = surplus.reindex(active.index, fill_value=False)
    d = deficit.reindex(active.index, fill_value=False)
    active['pnl'] = 0.0
    active.loc[s, 'pnl'] = (active.loc[s, 'exec_bid'] - active.loc[s, 'imb_settlement_price']) * ENERGY
    active.loc[d, 'pnl'] = (active.loc[d, 'imb_settlement_price'] - active.loc[d, 'exec_ask']) * ENERGY
    return active['pnl']


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("WALK-FORWARD SIGNAL DEGRADATION ANALYSIS")
    print("16 monthly folds, Oct 2024 - Mar 2026")
    print("=" * 70)

    # --- Load data ---
    print("\n[*] Loading data...")
    data = load_all_data()
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, feature_cols = build_features(data, LEAD)

    # Join execution prices
    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                          parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[
        ['bid', 'ask', 'spread', 'mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']
    df_base = df_base.join(ob_120, how='left')
    df_base['imb_settlement_price'] = df_base['imb_settle_price']

    # Hourly-smoothed spread target
    df_base['hour_ts'] = df_base.index.floor('h')
    df_base['settle_hourly'] = df_base.groupby('hour_ts')['imb_settlement_price'].transform('mean')
    df_base['mid_hourly'] = df_base.groupby('hour_ts')['exec_mid'].transform('mean')
    df_base['spread_target'] = df_base['settle_hourly'] - df_base['mid_hourly']

    # Validate features
    spread_features = [f for f in SELECTED_FEATURES if f in feature_cols]
    missing = [f for f in SELECTED_FEATURES if f not in feature_cols]
    if missing:
        print(f"[-] Warning: {len(missing)} features missing: {missing}")
    print(f"[+] Using {len(spread_features)} of {len(SELECTED_FEATURES)} features")

    # ============================================================
    # WALK-FORWARD: baseline + all degradation scenarios
    # ============================================================

    # Collect trades per scenario per fold
    # scenario -> list of pnl Series
    results = {'baseline': []}
    for gname in FEATURE_GROUPS:
        results[gname] = []

    # Also collect per-fold monthly summaries for detailed view
    fold_summaries = []

    for fi, (train_end, test_start, test_end) in enumerate(FOLDS):
        fold_label = f"{test_start[:7]}"
        print(f"\n--- Fold {fi+1}/{len(FOLDS)}: trade {fold_label} ---")

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

        # Train model on CLEAN data
        model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        model.fit(train[spread_features].values, train['spread_target'].values)

        X_test = test[spread_features]

        # Baseline prediction
        baseline_pred = model.predict(X_test.values)
        baseline_trades = calc_trades(test, baseline_pred)
        results['baseline'].append(baseline_trades)

        base_total = baseline_trades.sum() if len(baseline_trades) > 0 else 0
        base_days = baseline_trades.index.normalize().nunique() if len(baseline_trades) > 0 else 1
        base_ppd = base_total / base_days if base_days > 0 else 0

        fold_info = {'fold': fold_label, 'baseline_pnl': base_total,
                     'baseline_ppd': base_ppd, 'n_trades': len(baseline_trades)}

        # Degradation for each group
        group_strs = []
        for gname, gfeats in FEATURE_GROUPS.items():
            active = [f for f in gfeats if f in spread_features]
            if not active:
                results[gname].append(pd.Series(dtype=float))
                continue

            # Average over N_SEEDS
            seed_pnls = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed + fi * 100 + 42)
                X_degraded = degrade_full_replace(X_test, active, rng)
                pred = model.predict(X_degraded.values)
                trades = calc_trades(test, pred)
                seed_pnls.append(trades.sum() if len(trades) > 0 else 0)

            # Use the middle seed's actual trades for detailed analysis
            rng = np.random.default_rng(1 + fi * 100 + 42)
            X_degraded = degrade_full_replace(X_test, active, rng)
            pred = model.predict(X_degraded.values)
            trades = calc_trades(test, pred)
            results[gname].append(trades)

            mean_pnl = np.mean(seed_pnls)
            fold_info[f'{gname}_pnl'] = mean_pnl

            delta = mean_pnl - base_total
            group_strs.append(f"{gname}={delta:+.0f}")

        fold_summaries.append(fold_info)
        print(f"  Train={len(train)}, Test={len(test)}, "
              f"Baseline: {len(baseline_trades)} trades, {base_total:+,.0f} EUR ({base_ppd:+.0f}/day)")
        # Print compact group deltas
        if group_strs:
            print(f"  Deltas: {', '.join(group_strs)}")

    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS: 16-month walk-forward degradation")
    print("=" * 70)

    # Compute total P&L and daily stats for each scenario
    summary = {}
    for scenario, trade_lists in results.items():
        all_trades = pd.concat(trade_lists) if trade_lists else pd.Series(dtype=float)
        if len(all_trades) == 0:
            summary[scenario] = {'total_pnl': 0, 'n_trades': 0, 'n_days': 0,
                                 'pnl_per_day': 0, 'sharpe': 0, 'win_rate': 0}
            continue
        daily = all_trades.groupby(all_trades.index.date).sum()
        nd = len(daily)
        total = all_trades.sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        wr = (all_trades > 0).mean()
        summary[scenario] = {
            'total_pnl': total,
            'n_trades': len(all_trades),
            'n_days': nd,
            'pnl_per_day': total / nd if nd > 0 else 0,
            'sharpe': sharpe,
            'win_rate': wr,
        }

    # Print baseline
    bl = summary['baseline']
    print(f"\nBaseline (full model):")
    print(f"  Total P&L:   {bl['total_pnl']:+,.0f} EUR")
    print(f"  Per day:     {bl['pnl_per_day']:+,.0f} EUR")
    print(f"  Trades:      {bl['n_trades']:,}")
    print(f"  Days:        {bl['n_days']}")
    print(f"  Sharpe:      {bl['sharpe']:.2f}")
    print(f"  Win rate:    {bl['win_rate']:.0%}")

    # Ranking table
    print(f"\n{'='*70}")
    print("SIGNAL IMPORTANCE RANKING (by P&L impact when group killed)")
    print(f"{'='*70}")
    print(f"\n{'Group':20s} {'Feats':>5s} {'Total PnL':>10s} {'EUR/day':>8s} "
          f"{'Delta':>8s} {'Impact%':>8s} {'Sharpe':>7s} {'WinR':>5s}")
    print("-" * 75)

    group_impacts = []
    for gname in FEATURE_GROUPS:
        s = summary[gname]
        active = [f for f in FEATURE_GROUPS[gname] if f in spread_features]
        delta_ppd = s['pnl_per_day'] - bl['pnl_per_day']
        pct = (s['pnl_per_day'] - bl['pnl_per_day']) / abs(bl['pnl_per_day']) * 100 \
            if bl['pnl_per_day'] != 0 else 0
        group_impacts.append((gname, len(active), s, delta_ppd, pct))

    # Sort by impact (most negative = most important)
    group_impacts.sort(key=lambda x: x[4])

    for gname, nf, s, delta_ppd, pct in group_impacts:
        print(f"{gname:20s} {nf:5d} {s['total_pnl']:+10,.0f} {s['pnl_per_day']:+8.0f} "
              f"{delta_ppd:+8.0f} {pct:+7.1f}% {s['sharpe']:+6.2f} {s['win_rate']:4.0%}")

    print("-" * 75)
    print(f"{'BASELINE':20s} {len(spread_features):5d} {bl['total_pnl']:+10,.0f} {bl['pnl_per_day']:+8.0f} "
          f"{'---':>8s} {'0.0%':>8s} {bl['sharpe']:+6.2f} {bl['win_rate']:4.0%}")

    # Risk categories
    print("\n[*] Risk Assessment:")
    critical = [(n, p) for n, nf, s, d, p in group_impacts if p < -30]
    important = [(n, p) for n, nf, s, d, p in group_impacts if -30 <= p < -10]
    minor = [(n, p) for n, nf, s, d, p in group_impacts if -10 <= p < 0]
    helpful_noise = [(n, p) for n, nf, s, d, p in group_impacts if p >= 0]

    if critical:
        print(f"  [!] CRITICAL (>30% loss): {', '.join(f'{n} ({p:+.1f}%)' for n, p in critical)}")
    if important:
        print(f"  [-] IMPORTANT (10-30% loss): {', '.join(f'{n} ({p:+.1f}%)' for n, p in important)}")
    if minor:
        print(f"  [+] MINOR (<10% loss): {', '.join(f'{n} ({p:+.1f}%)' for n, p in minor)}")
    if helpful_noise:
        print(f"  [+] NOISE HELPS (removing improves): {', '.join(f'{n} ({p:+.1f}%)' for n, p in helpful_noise)}")

    # ============================================================
    # MONTHLY BREAKDOWN TABLE
    # ============================================================
    print(f"\n{'='*70}")
    print("MONTHLY FOLD BREAKDOWN (baseline vs degraded, EUR total)")
    print(f"{'='*70}")

    if fold_summaries:
        df_folds = pd.DataFrame(fold_summaries).set_index('fold')
        # Show baseline and delta for each group
        print(f"\n{'Month':>8s} {'Base':>7s}", end='')
        for gname in FEATURE_GROUPS:
            print(f" {gname[:8]:>8s}", end='')
        print()
        print("-" * (16 + 9 * len(FEATURE_GROUPS)))

        for _, row in df_folds.iterrows():
            base = row['baseline_pnl']
            print(f"{row.name:>8s} {base:+7.0f}", end='')
            for gname in FEATURE_GROUPS:
                col = f'{gname}_pnl'
                if col in row and not pd.isna(row[col]):
                    delta = row[col] - base
                    print(f" {delta:+8.0f}", end='')
                else:
                    print(f" {'n/a':>8s}", end='')
            print()

    # ============================================================
    # PLOT
    # ============================================================
    print("\n[*] Generating plots...")

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Panel 1: Bar chart of full-period impact
    ax = axes[0]
    names = [n for n, nf, s, d, p in group_impacts]
    pcts = [p for n, nf, s, d, p in group_impacts]
    colors = ['#d32f2f' if v < -30 else '#f57c00' if v < -10 else '#388e3c' if v < 0 else '#1565c0'
              for v in pcts]
    bars = ax.barh(names, pcts, color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('P&L Change from Baseline (%)')
    ax.set_title('Feature Group Importance\n(16-month walk-forward, signal death)')
    ax.invert_yaxis()
    for bar, val in zip(bars, pcts):
        x_pos = val - 2 if val < 0 else val + 1
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{val:+.1f}%',
                va='center', ha='right' if val < 0 else 'left', fontsize=9, fontweight='bold')

    # Panel 2: Per-month delta heatmap as grouped bars
    ax = axes[1]
    if fold_summaries:
        df_folds = pd.DataFrame(fold_summaries).set_index('fold')
        months = df_folds.index.tolist()
        n_groups = len(FEATURE_GROUPS)
        x = np.arange(len(months))
        width = 0.8 / n_groups
        cmap = plt.cm.tab10

        for gi, gname in enumerate(FEATURE_GROUPS):
            col = f'{gname}_pnl'
            if col not in df_folds.columns:
                continue
            deltas = df_folds[col] - df_folds['baseline_pnl']
            ax.bar(x + gi * width - 0.4, deltas.values, width, label=gname[:10],
                   color=cmap(gi), alpha=0.7, edgecolor='black', linewidth=0.2)

        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, fontsize=7)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('P&L Delta (EUR)')
        ax.set_title('Monthly Impact by Feature Group\n(negative = group is valuable)')
        ax.legend(fontsize=6, ncol=2, loc='lower left')

    # Panel 3: EUR/day comparison
    ax = axes[2]
    scenario_names = ['BASELINE'] + [n for n, nf, s, d, p in group_impacts]
    ppd_values = [bl['pnl_per_day']] + [s['pnl_per_day'] for n, nf, s, d, p in group_impacts]
    bar_colors = ['#2196f3'] + ['#d32f2f' if p < -30 else '#f57c00' if p < -10
                                 else '#388e3c' if p < 0 else '#1565c0'
                                 for n, nf, s, d, p in group_impacts]
    bars = ax.barh(scenario_names, ppd_values, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(bl['pnl_per_day'], color='blue', linestyle='--', alpha=0.3,
               label=f'Baseline={bl["pnl_per_day"]:+.0f}/day')
    ax.set_xlabel('EUR/day')
    ax.set_title('P&L/day by Scenario\n(with group replaced by noise)')
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    for bar, val in zip(bars, ppd_values):
        ax.text(val + 10 if val >= 0 else val - 10, bar.get_y() + bar.get_height() / 2,
                f'{val:+.0f}', va='center', ha='left' if val >= 0 else 'right',
                fontsize=8, fontweight='bold')

    fig.suptitle('Walk-Forward Signal Degradation Analysis\n'
                 '52-feature spread model, 16 monthly OOS folds, 3 seeds/fold',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()

    plot_path = PLOT_DIR / "02_degradation_walkforward.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Plot saved: {plot_path}")

    print("\n[+] Done.")


if __name__ == "__main__":
    main()
