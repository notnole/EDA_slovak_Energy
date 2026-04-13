"""
Signal Degradation Robustness Analysis
========================================

Tests how the 52-feature spread model degrades when individual data source
signals fail or become noisy -- simulating real production scenarios like
feed outages, stale data, or sensor drift.

Two test modes:
  A. Full replacement (feed death): replace group with random walks
  B. Noise injection at 25/50/75/100%: mix real signal with noise

Evaluation: train on CLEAN data, predict on DEGRADED test data,
measure P&L with bid/ask execution on real 15-min settlement.
Each degradation run 5x with different seeds for stability.
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
REPO_ROOT = BASE_DIR.parent
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
ENERGY = SIZE_MW * 0.25
THRESHOLD = 3
N_SEEDS = 5

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

NOISE_LEVELS = [0.25, 0.50, 0.75, 1.00]


# ============================================================
# P&L CALCULATION
# ============================================================

def calc_pnl(test_df, predictions):
    """Calculate trading P&L with bid/ask execution."""
    t = test_df.copy()
    t['pred'] = predictions
    surplus = t['pred'] <= -THRESHOLD
    deficit = t['pred'] >= THRESHOLD
    sub = t[surplus | deficit].copy()
    if len(sub) < 10:
        return {'pnl_per_day': 0, 'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'total_pnl': 0}
    s = surplus.reindex(sub.index, fill_value=False)
    d = deficit.reindex(sub.index, fill_value=False)
    sub['pnl'] = 0.0
    sub.loc[s, 'pnl'] = (sub.loc[s, 'exec_bid'] - sub.loc[s, 'imb_settlement_price']) * ENERGY
    sub.loc[d, 'pnl'] = (sub.loc[d, 'imb_settlement_price'] - sub.loc[d, 'exec_ask']) * ENERGY
    daily = sub.groupby(sub.index.date)['pnl'].sum()
    nd = len(daily)
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    return {'pnl_per_day': sub['pnl'].sum() / nd, 'sharpe': sharpe,
            'n_trades': len(sub), 'win_rate': (sub['pnl'] > 0).mean(),
            'total_pnl': sub['pnl'].sum()}


# ============================================================
# DEGRADATION FUNCTIONS
# ============================================================

def degrade_full_replace(test_X, group_features, rng):
    """Replace group features with random walks matching mean/std."""
    degraded = test_X.copy()
    for feat in group_features:
        if feat not in degraded.columns:
            continue
        col = degraded[feat]
        mu, sigma = col.mean(), col.std()
        if sigma == 0 or np.isnan(sigma):
            sigma = 1.0
        # Random walk with matching distribution
        noise = rng.normal(mu, sigma, size=len(col))
        degraded[feat] = noise
    return degraded


def degrade_noise_inject(test_X, group_features, noise_level, rng):
    """Mix real signal with noise: (1-alpha)*real + alpha*noise."""
    degraded = test_X.copy()
    for feat in group_features:
        if feat not in degraded.columns:
            continue
        col = degraded[feat].values.copy()
        mu, sigma = np.nanmean(col), np.nanstd(col)
        if sigma == 0 or np.isnan(sigma):
            sigma = 1.0
        noise = rng.normal(mu, sigma, size=len(col))
        degraded[feat] = (1 - noise_level) * col + noise_level * noise
    return degraded


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("Signal Degradation Robustness Analysis")
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

    # Hourly-smoothed target
    df_base['hour_ts'] = df_base.index.floor('h')
    df_base['settle_hourly'] = df_base.groupby('hour_ts')['imb_settlement_price'].transform('mean')
    df_base['mid_hourly'] = df_base.groupby('hour_ts')['exec_mid'].transform('mean')
    df_base['spread_target'] = df_base['settle_hourly'] - df_base['mid_hourly']

    # Filter features to selected
    spread_features = [f for f in SELECTED_FEATURES if f in feature_cols]
    missing = [f for f in SELECTED_FEATURES if f not in feature_cols]
    if missing:
        print(f"[-] Warning: {len(missing)} features missing: {missing}")
    print(f"[+] Using {len(spread_features)} of {len(SELECTED_FEATURES)} features")

    # Split
    train = df_base[df_base.index < '2026-02-01'].dropna(
        subset=['spread_target', f'proxy_lag{LEAD+1}'])
    train = train[train['imb_settlement_price'].abs() <= 5000]
    test = df_base[(df_base.index >= '2026-02-01') & (df_base.index < '2026-04-01')].copy()
    test = test[test['exec_bid'].notna() & test['exec_ask'].notna() & (test['exec_spread'] <= 10)]

    print(f"[+] Train: {len(train)} rows, Test: {len(test)} rows")

    # Prepare train/test arrays
    X_train = train[spread_features]
    y_train = train['spread_target']
    X_test = test[spread_features]

    # --- Train model on CLEAN data ---
    print("\n[*] Training model on clean data...")
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train)
    print("[+] Model trained")

    # --- Baseline (no degradation) ---
    print("\n[*] Computing baseline (no degradation)...")
    baseline_pred = model.predict(X_test)
    baseline = calc_pnl(test, baseline_pred)
    print(f"[+] Baseline: EUR/day={baseline['pnl_per_day']:.0f}, "
          f"Sharpe={baseline['sharpe']:.2f}, "
          f"Trades={baseline['n_trades']}, "
          f"Win={baseline['win_rate']:.1%}")

    # --- Full replacement test ---
    print("\n" + "=" * 70)
    print("TEST A: Full Signal Replacement (Feed Death)")
    print("=" * 70)

    full_replace_results = {}

    for group_name, group_feats in FEATURE_GROUPS.items():
        active = [f for f in group_feats if f in spread_features]
        if not active:
            print(f"[-] {group_name}: no active features, skipping")
            continue

        pnls = []
        sharpes = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed + 42)
            X_degraded = degrade_full_replace(X_test, active, rng)
            pred = model.predict(X_degraded)
            result = calc_pnl(test, pred)
            pnls.append(result['pnl_per_day'])
            sharpes.append(result['sharpe'])

        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        mean_sharpe = np.mean(sharpes)
        pct_change = (mean_pnl - baseline['pnl_per_day']) / abs(baseline['pnl_per_day']) * 100 \
            if baseline['pnl_per_day'] != 0 else 0

        full_replace_results[group_name] = {
            'pnl_per_day': mean_pnl,
            'pnl_std': std_pnl,
            'sharpe': mean_sharpe,
            'pct_change': pct_change,
            'n_features': len(active),
        }
        print(f"  {group_name:20s} ({len(active):2d} feats): "
              f"EUR/day={mean_pnl:+7.0f} +/-{std_pnl:4.0f}, "
              f"Sharpe={mean_sharpe:+5.2f}, "
              f"Change={pct_change:+6.1f}%")

    # --- Noise injection test ---
    print("\n" + "=" * 70)
    print("TEST B: Noise Injection Curves (25/50/75/100%)")
    print("=" * 70)

    noise_results = {}  # {group: {level: mean_pnl}}

    for group_name, group_feats in FEATURE_GROUPS.items():
        active = [f for f in group_feats if f in spread_features]
        if not active:
            continue

        noise_results[group_name] = {}
        row_str = f"  {group_name:20s}: "

        for level in NOISE_LEVELS:
            pnls = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed + 100)
                X_degraded = degrade_noise_inject(X_test, active, level, rng)
                pred = model.predict(X_degraded)
                result = calc_pnl(test, pred)
                pnls.append(result['pnl_per_day'])

            mean_pnl = np.mean(pnls)
            noise_results[group_name][level] = mean_pnl
            row_str += f"{int(level*100):3d}%={mean_pnl:+6.0f}  "

        print(row_str)

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY: Signal Importance Ranking (by P&L impact)")
    print("=" * 70)
    print(f"\n{'Group':20s} {'Feats':>5s} {'Clean':>8s} {'Dead':>8s} {'Impact':>8s} {'Sharpe':>7s}")
    print("-" * 60)

    sorted_groups = sorted(full_replace_results.items(),
                           key=lambda x: x[1]['pct_change'])

    for group_name, res in sorted_groups:
        print(f"{group_name:20s} {res['n_features']:5d} "
              f"{baseline['pnl_per_day']:+8.0f} "
              f"{res['pnl_per_day']:+8.0f} "
              f"{res['pct_change']:+7.1f}% "
              f"{res['sharpe']:+6.2f}")

    print("-" * 60)
    print(f"{'Baseline':20s} {len(spread_features):5d} "
          f"{baseline['pnl_per_day']:+8.0f} "
          f"{'---':>8s} "
          f"{'0.0%':>8s} "
          f"{baseline['sharpe']:+6.2f}")

    # --- Risk assessment ---
    print("\n[*] Risk Assessment:")
    critical = [(n, r) for n, r in sorted_groups if r['pct_change'] < -30]
    important = [(n, r) for n, r in sorted_groups if -30 <= r['pct_change'] < -10]
    minor = [(n, r) for n, r in sorted_groups if r['pct_change'] >= -10]

    if critical:
        print(f"  [!] CRITICAL feeds (>30% P&L loss if dead): "
              f"{', '.join(n for n, _ in critical)}")
    if important:
        print(f"  [-] IMPORTANT feeds (10-30% P&L loss): "
              f"{', '.join(n for n, _ in important)}")
    if minor:
        print(f"  [+] MINOR feeds (<10% P&L loss): "
              f"{', '.join(n for n, _ in minor)}")

    # ============================================================
    # PLOT
    # ============================================================
    print("\n[*] Generating plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Left panel: bar chart of full replacement impact
    groups = [n for n, _ in sorted_groups]
    impacts = [r['pct_change'] for _, r in sorted_groups]
    colors = ['#d32f2f' if v < -30 else '#f57c00' if v < -10 else '#388e3c' for v in impacts]

    bars = ax1.barh(groups, impacts, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.set_xlabel('P&L Change from Baseline (%)')
    ax1.set_title('Full Signal Replacement Impact\n(feed death simulation)')
    ax1.invert_yaxis()

    # Add value labels on bars
    for bar, val in zip(bars, impacts):
        x_pos = val - 2 if val < 0 else val + 1
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.1f}%',
                 va='center', ha='right' if val < 0 else 'left', fontsize=9, fontweight='bold')

    # Right panel: noise injection curves
    cmap = plt.cm.tab10
    x_levels = [0] + [int(l * 100) for l in NOISE_LEVELS]

    for idx, (group_name, levels_dict) in enumerate(noise_results.items()):
        y_vals = [baseline['pnl_per_day']]
        for level in NOISE_LEVELS:
            y_vals.append(levels_dict[level])
        ax2.plot(x_levels, y_vals, 'o-', label=group_name,
                 color=cmap(idx), linewidth=2, markersize=6)

    ax2.axhline(baseline['pnl_per_day'], color='black', linestyle='--',
                alpha=0.5, label=f'Baseline ({baseline["pnl_per_day"]:.0f})')
    ax2.axhline(0, color='red', linestyle=':', alpha=0.3)
    ax2.set_xlabel('Noise Level (%)')
    ax2.set_ylabel('EUR/day')
    ax2.set_title('P&L Degradation Curves\n(noise injection 0-100%)')
    ax2.set_xticks(x_levels)
    ax2.legend(fontsize=8, loc='best')

    fig.suptitle('Signal Degradation Robustness Analysis\n'
                 '52-feature spread model, Feb-Mar 2026 OOS, 5 seeds per test',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()

    plot_path = PLOT_DIR / "01_degradation_curves.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Plot saved: {plot_path}")
    print("\n[+] Done.")


if __name__ == "__main__":
    main()
