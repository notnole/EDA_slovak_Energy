"""
Pruned Feature Walk-Forward Test
=================================

Based on the 16-month walk-forward signal degradation results, several feature
groups hurt or add nothing. This script tests progressive pruning:

  1. Full 52 features (baseline)
  2. Drop load_scada (50 features)
  3. Drop load_scada + time (45 features)
  4. Drop load_scada + time + load_nowcast (38 features)
  5. Drop all <5% groups -> keep only da_prices + weather (14 features)

Each config runs 16 monthly walk-forward folds with real bid/ask execution.
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
PLOT_DIR = BASE_DIR / "plots" / "pruned_features"
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

# Groups to drop progressively
DROP_LOAD_SCADA = ['load_rmean16', 'load_momentum']
DROP_TIME = ['hour_cos', 'hour_sin', 'dow_sin', 'dow_cos', 'is_weekend']
DROP_LOAD_NOWCAST = ['nowcast_momentum_h2h3', 'nowcast_pred_rmean4', 'nowcast_momentum_h4h5',
                     'nowcast_h3', 'nowcast_trend_h2_h5', 'nowcast_h5', 'nowcast_convergence']

# For config 5: keep only da_prices + weather
KEEP_DA_PRICES = ['da_price', 'da_supply', 'da_price_change24h', 'da_demand', 'da_flow_cz', 'da_net_import']
KEEP_WEATHER = ['cloudcover', 'temp_forecast_da', 'temp_national_spread', 'temp_bratislava',
                'temp_national_change6h', 'temp_surprise_lag', 'radiation_national', 'temp_rmean24h']

# Build configs
def build_configs(available_features):
    """Build pruning configs from features that are actually available."""
    full = [f for f in SELECTED_FEATURES if f in available_features]

    drop1 = [f for f in full if f not in DROP_LOAD_SCADA]
    drop2 = [f for f in drop1 if f not in DROP_TIME]
    drop3 = [f for f in drop2 if f not in DROP_LOAD_NOWCAST]
    minimal = [f for f in KEEP_DA_PRICES + KEEP_WEATHER if f in available_features]

    configs = {
        'Full 52': full,
        'Drop load_scada (50)': drop1,
        'Drop +time (45)': drop2,
        'Drop +time+nowcast (38)': drop3,
        'DA+weather only (14)': minimal,
    }
    return configs


# Walk-forward folds
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


def compute_stats(trade_series_list):
    """Compute aggregate stats from a list of per-fold trade Series."""
    all_trades = pd.concat(trade_series_list) if trade_series_list else pd.Series(dtype=float)
    all_trades = all_trades.dropna()
    if len(all_trades) == 0:
        return {'total_pnl': 0, 'n_trades': 0, 'n_days': 0,
                'pnl_per_day': 0, 'sharpe': 0, 'win_rate': 0,
                'losing_months': 0, 'monthly_pnl': pd.Series(dtype=float)}

    daily = all_trades.groupby(all_trades.index.date).sum()
    nd = len(daily)
    total = all_trades.sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    wr = (all_trades > 0).mean()

    # Monthly P&L for losing months count
    monthly = daily.groupby(pd.to_datetime(daily.index).to_period('M')).sum()

    return {
        'total_pnl': total,
        'n_trades': len(all_trades),
        'n_days': nd,
        'pnl_per_day': total / nd if nd > 0 else 0,
        'sharpe': sharpe,
        'win_rate': wr,
        'losing_months': (monthly < 0).sum(),
        'monthly_pnl': monthly,
    }


def compute_period_stats(trade_series_list, period_start, period_end):
    """Compute stats for a sub-period."""
    all_trades = pd.concat(trade_series_list) if trade_series_list else pd.Series(dtype=float)
    all_trades = all_trades.dropna()
    if len(all_trades) == 0:
        return {'pnl_per_day': 0, 'sharpe': 0, 'n_days': 0}

    mask = (pd.to_datetime(all_trades.index.date) >= pd.Timestamp(period_start)) & \
           (pd.to_datetime(all_trades.index.date) < pd.Timestamp(period_end))
    sub = all_trades[mask]
    if len(sub) == 0:
        return {'pnl_per_day': 0, 'sharpe': 0, 'n_days': 0}

    daily = sub.groupby(sub.index.date).sum()
    nd = len(daily)
    total = sub.sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    return {'pnl_per_day': total / nd if nd > 0 else 0, 'sharpe': sharpe, 'n_days': nd}


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PRUNED FEATURE WALK-FORWARD TEST")
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

    # Build configs
    configs = build_configs(feature_cols)
    for name, feats in configs.items():
        print(f"  {name}: {len(feats)} features")

    # ============================================================
    # WALK-FORWARD FOR EACH CONFIG
    # ============================================================

    # config_name -> list of per-fold trade Series
    all_results = {name: [] for name in configs}
    # config_name -> list of (fold_label, fold_pnl, fold_days) tuples
    fold_details = {name: [] for name in configs}

    for fi, (train_end, test_start, test_end) in enumerate(FOLDS):
        fold_label = f"{test_start[:7]}"
        print(f"\n--- Fold {fi+1}/{len(FOLDS)}: {fold_label} ---")

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

        results_line = []
        for name, feats in configs.items():
            # Ensure features exist in this fold's data
            avail = [f for f in feats if f in train.columns and f in test.columns]

            model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
            model.fit(train[avail].values, train['spread_target'].values)

            pred = model.predict(test[avail].values)
            trades = calc_trades(test, pred)
            all_results[name].append(trades)

            total = trades.sum() if len(trades) > 0 else 0
            days = trades.index.normalize().nunique() if len(trades) > 0 else 1
            ppd = total / days if days > 0 else 0
            fold_details[name].append((fold_label, total, days, len(trades)))
            results_line.append(f"{name.split('(')[0].strip()[:12]}={ppd:+.0f}")

        print(f"  Train={len(train)}, Test={len(test)}")
        print(f"  {', '.join(results_line)}")

    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS: 16-month walk-forward, pruned features")
    print("=" * 70)

    summaries = {}
    for name in configs:
        summaries[name] = compute_stats(all_results[name])

    bl = summaries['Full 52']

    print(f"\n{'Config':35s} {'Feats':>5s} {'EUR/day':>8s} {'Sharpe':>7s} "
          f"{'Trades':>7s} {'WinR':>5s} {'LoseM':>6s} {'vs Base':>8s}")
    print("-" * 85)

    for name in configs:
        s = summaries[name]
        nf = len(configs[name])
        delta_pct = (s['pnl_per_day'] - bl['pnl_per_day']) / abs(bl['pnl_per_day']) * 100 \
            if bl['pnl_per_day'] != 0 else 0
        delta_str = f"{delta_pct:+.1f}%" if name != 'Full 52' else "---"
        print(f"{name:35s} {nf:5d} {s['pnl_per_day']:+8.0f} {s['sharpe']:+7.2f} "
              f"{s['n_trades']:7d} {s['win_rate']:4.0%} {s['losing_months']:6d} {delta_str:>8s}")

    # ============================================================
    # PERIOD SPLIT: 2024-2025 vs 2026
    # ============================================================
    print(f"\n{'='*70}")
    print("PERIOD SPLIT: Oct 2024 - Sep 2025 vs Dec 2025 - Mar 2026")
    print(f"{'='*70}")

    print(f"\n--- 2024-2025 period (12 folds) ---")
    print(f"{'Config':35s} {'EUR/day':>8s} {'Sharpe':>7s} {'Days':>5s}")
    print("-" * 60)
    for name in configs:
        s = compute_period_stats(all_results[name], '2024-10-01', '2025-10-01')
        print(f"{name:35s} {s['pnl_per_day']:+8.0f} {s['sharpe']:+7.2f} {s['n_days']:5d}")

    print(f"\n--- 2026 period (4 folds) ---")
    print(f"{'Config':35s} {'EUR/day':>8s} {'Sharpe':>7s} {'Days':>5s}")
    print("-" * 60)
    for name in configs:
        s = compute_period_stats(all_results[name], '2025-12-01', '2026-04-01')
        print(f"{name:35s} {s['pnl_per_day']:+8.0f} {s['sharpe']:+7.2f} {s['n_days']:5d}")

    # ============================================================
    # MONTHLY BREAKDOWN
    # ============================================================
    print(f"\n{'='*70}")
    print("MONTHLY P&L BREAKDOWN (EUR total per month)")
    print(f"{'='*70}")

    short_names = []
    for name in configs:
        short = name.split('(')[0].strip()
        if len(short) > 12:
            short = short[:12]
        short_names.append(short)

    print(f"\n{'Month':>8s}", end='')
    for sn in short_names:
        print(f" {sn:>12s}", end='')
    print()
    print("-" * (8 + 13 * len(configs)))

    # Get monthly series for each config
    for fi in range(len(FOLDS)):
        fold_label = FOLDS[fi][1][:7]
        vals = []
        for name in configs:
            if fi < len(fold_details[name]):
                _, total, _, _ = fold_details[name][fi]
                vals.append(total)
            else:
                vals.append(float('nan'))

        print(f"{fold_label:>8s}", end='')
        for v in vals:
            if np.isnan(v):
                print(f" {'n/a':>12s}", end='')
            else:
                print(f" {v:+12.0f}", end='')
        print()

    # Totals
    print("-" * (8 + 13 * len(configs)))
    print(f"{'TOTAL':>8s}", end='')
    for name in configs:
        s = summaries[name]
        print(f" {s['total_pnl']:+12.0f}", end='')
    print()

    # ============================================================
    # PLOT
    # ============================================================
    print("\n[*] Generating plots...")

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Panel 1: EUR/day comparison bar chart
    ax = axes[0]
    config_names = list(configs.keys())
    ppd_values = [summaries[n]['pnl_per_day'] for n in config_names]
    colors = ['#2196f3', '#4caf50', '#4caf50', '#ff9800', '#f44336']
    bars = ax.barh(config_names, ppd_values, color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    bl_ppd = summaries['Full 52']['pnl_per_day']
    ax.axvline(bl_ppd, color='blue', linestyle='--', alpha=0.3,
               label=f'Full baseline={bl_ppd:+.0f}')
    ax.set_xlabel('EUR/day')
    ax.set_title('P&L per Day by Feature Config')
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    for bar, val in zip(bars, ppd_values):
        x_pos = val + 10 if val >= 0 else val - 10
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{val:+.0f}',
                va='center', ha='left' if val >= 0 else 'right',
                fontsize=9, fontweight='bold')

    # Panel 2: Sharpe comparison
    ax = axes[1]
    sharpe_values = [summaries[n]['sharpe'] for n in config_names]
    bars = ax.barh(config_names, sharpe_values, color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Annualized Sharpe')
    ax.set_title('Sharpe Ratio by Feature Config')
    ax.invert_yaxis()
    for bar, val in zip(bars, sharpe_values):
        x_pos = val + 0.05 if val >= 0 else val - 0.05
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{val:.2f}',
                va='center', ha='left' if val >= 0 else 'right',
                fontsize=9, fontweight='bold')

    # Panel 3: Cumulative P&L curves
    ax = axes[2]
    cmap_lines = plt.cm.tab10
    for ci, name in enumerate(config_names):
        all_trades = pd.concat(all_results[name]) if all_results[name] else pd.Series(dtype=float)
        all_trades = all_trades.dropna().sort_index()
        if len(all_trades) == 0:
            continue
        daily = all_trades.groupby(all_trades.index.date).sum()
        cum = daily.cumsum()
        ax.plot(pd.to_datetime(cum.index), cum.values, label=name,
                color=cmap_lines(ci), linewidth=1.5)

    ax.axhline(0, color='red', linestyle=':', alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative P&L (EUR)')
    ax.set_title('Cumulative P&L Curves')
    ax.legend(fontsize=7, loc='best')
    ax.tick_params(axis='x', rotation=45)

    fig.suptitle('Pruned Feature Walk-Forward Test\n'
                 '16 monthly OOS folds, LGB quantile 0.50, 5 MW / threshold 3',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()

    plot_path = PLOT_DIR / "01_pruned_features_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Plot saved: {plot_path}")

    print("\n[+] Done.")


if __name__ == "__main__":
    main()
