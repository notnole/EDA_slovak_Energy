"""
Leakage Fix Test: Walk-Forward Comparison
==========================================

Two features use OKTE imbalance settlement prices shifted by hourly_shift:
  - imb_price_rmean4: rolling mean of imb_settlement_price
  - spread_da_imb_lag: DA price minus imb_settlement_price

Problem: OKTE imbalance prices are published D+1, NOT in real-time.
The hourly_shift (lead+4 = 12 periods = 3h) is NOT enough -- these
prices are simply not available for live trading.

This script tests 6 configurations across the full 16-month walk-forward:
  1. Current 52 features (baseline, WITH leakage)
  2. 50 features (remove the 2 leaky features)
  3. 6 clean minimal features (IDM + time only)
  4. 52 features with leaky ones REPLACED by IDM-based equivalents
  5. 50 features + strong regularization
  6. 6 clean features + strong regularization

P&L: confidence sizing (size=|pred|.clip(upper=5)), bid/ask execution,
     threshold=3, exec_spread<=10, hourly-smoothed spread target.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots" / "eda" / "leakage_fix"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LEAD = 8
QH = 0.25
THRESHOLD = 3
HOURLY_SHIFT = LEAD + 4  # = 12

BASELINE_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
                       subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                       reg_lambda=1.0, n_estimators=600, verbose=-1)
SIMPLE_PARAMS = dict(learning_rate=0.03, num_leaves=15, min_child_samples=200,
                     subsample=0.5, colsample_bytree=0.5, reg_alpha=1.0,
                     reg_lambda=10.0, n_estimators=200, verbose=-1)

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

LEAKY_FEATURES = ['imb_price_rmean4', 'spread_da_imb_lag']

CLEAN_50 = [f for f in SELECTED_FEATURES if f not in LEAKY_FEATURES]

CLEAN_6 = ['idm_vwap_lag', 'hour_cos', 'hour_sin', 'dow_sin', 'dow_cos', 'is_weekend']

# Config 4: replace leaky with IDM-based
REPLACED_52 = CLEAN_50 + ['idm_vwap_rmean4', 'spread_da_idm_lag_fixed']

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


def run_walkforward(df_base, feature_list, feature_cols, lgb_params, config_name):
    """Run walk-forward for a single config with confidence sizing."""
    feats = [f for f in feature_list if f in feature_cols]
    missing = [f for f in feature_list if f not in feature_cols]
    if missing:
        print(f"  [!] {len(missing)} features missing: {missing}")
    if len(feats) < 3:
        print(f"  [!] Too few features ({len(feats)}), skipping")
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

    oos = pd.concat(all_trades)
    daily = oos.groupby(oos.index.date)['pnl'].sum()
    monthly = oos.groupby(oos.index.to_period('M'))['pnl'].sum()
    monthly_days = oos.groupby(oos.index.to_period('M')).apply(
        lambda x: x.index.normalize().nunique())
    monthly_ppd = monthly / monthly_days

    n_days = len(daily)
    total = oos['pnl'].sum()
    eur_day = daily.mean()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    win_rate = (oos['pnl'] > 0).mean()
    losing_months = int((monthly < 0).sum())
    worst_month = monthly.min()
    worst_month_name = str(monthly.idxmin()) if len(monthly) > 0 else 'N/A'

    # 2024-2025 vs 2026 split
    daily_s = pd.Series(daily.values, index=pd.DatetimeIndex(daily.index))
    pre_2026 = daily_s[daily_s.index < '2026-01-01']
    post_2026 = daily_s[daily_s.index >= '2026-01-01']

    pre_eur_day = pre_2026.mean() if len(pre_2026) > 0 else 0
    post_eur_day = post_2026.mean() if len(post_2026) > 0 else 0
    pre_sharpe = pre_2026.mean() / pre_2026.std() * np.sqrt(252) if len(pre_2026) > 1 and pre_2026.std() > 0 else 0
    post_sharpe = post_2026.mean() / post_2026.std() * np.sqrt(252) if len(post_2026) > 1 and post_2026.std() > 0 else 0

    return {
        'config': config_name,
        'n_features': len(feats),
        'total_pnl': total,
        'eur_day': eur_day,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'n_trades': len(oos),
        'n_days': n_days,
        'losing_months': losing_months,
        'total_months': len(monthly),
        'worst_month': worst_month,
        'worst_month_name': worst_month_name,
        'monthly_pnl': monthly,
        'monthly_ppd': monthly_ppd,
        'daily_pnl': daily,
        'pre2026_eur_day': pre_eur_day,
        'post2026_eur_day': post_eur_day,
        'pre2026_sharpe': pre_sharpe,
        'post2026_sharpe': post_sharpe,
        'pre2026_days': len(pre_2026),
        'post2026_days': len(post_2026),
    }


def main():
    t0 = time.time()
    print("=" * 75)
    print("LEAKAGE FIX TEST -- WALK-FORWARD COMPARISON")
    print("=" * 75)
    print("")
    print("[*] Leaky features identified:")
    print("    - imb_price_rmean4: rolling mean of OKTE settlement price (D+1)")
    print("    - spread_da_imb_lag: DA price minus OKTE settlement price (D+1)")
    print("")

    # --- Load data ---
    print("[*] Loading data...")
    data = load_all_data()
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, feature_cols = build_features(data, LEAD)

    # --- Add replacement features for config 4 ---
    # idm_vwap_rmean4: rolling 4-period mean of IDM VWAP (shifted by hourly_shift)
    if 'idm_vwap' in data['mkt'].columns if data['mkt'] is not None else False:
        # data['mkt'] was already joined into df via build_features; we need the raw df
        # Rebuild from the base df that build_features produced
        pass

    # We need the raw 'idm_vwap' from the joined df. build_features already joined
    # mkt data, but we need to access it via the data dict.
    # The build_features function joins data['mkt'] which contains 'idm_vwap'.
    # After joining, the raw 'idm_vwap' is in the intermediate df but NOT in feat_df.
    # We need to recompute from the raw data source.

    # Reload the raw mkt data and join manually
    import os
    mkt_path = BASE_DIR.parent / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv"
    if mkt_path.exists():
        mkt_raw = pd.read_csv(mkt_path, parse_dates=['timestamp_hour']).set_index('timestamp_hour').sort_index()
        mkt_raw = mkt_raw[~mkt_raw.index.duplicated(keep='last')]
        idm_vwap_15 = mkt_raw['idm_vwap'].resample('15min').ffill()

        # idm_vwap_rmean4: rolling 4 of idm_vwap shifted by HOURLY_SHIFT
        idm_shifted = idm_vwap_15.reindex(df_base.index).shift(HOURLY_SHIFT)
        df_base['idm_vwap_rmean4'] = idm_shifted.rolling(4).mean()

        # spread_da_idm_lag_fixed: DA price - IDM VWAP, both available real-time
        da_price_15 = None
        da_path = BASE_DIR.parent / "features" / "DamasPrices" / "data" / "da_prices.csv"
        if da_path.exists():
            da_raw = pd.read_csv(da_path, parse_dates=['datetime']).set_index('datetime').sort_index()
            da_raw = da_raw[~da_raw.index.duplicated(keep='last')]
            da_price_15 = da_raw['price_eur_mwh'].resample('15min').ffill()

        if da_price_15 is not None:
            da_reindexed = da_price_15.reindex(df_base.index)
            df_base['spread_da_idm_lag_fixed'] = da_reindexed - idm_shifted
        else:
            # Fallback: use da_price feature already in df_base
            df_base['spread_da_idm_lag_fixed'] = df_base['da_price'] - idm_shifted

        feature_cols_extended = feature_cols + ['idm_vwap_rmean4', 'spread_da_idm_lag_fixed']
        print(f"[+] Added replacement features: idm_vwap_rmean4, spread_da_idm_lag_fixed")
    else:
        feature_cols_extended = feature_cols
        print("[!] Could not load market data for replacement features")

    # Join OB execution prices
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

    # --- Define configs ---
    CONFIGS = {
        '1_baseline_52_leaky':     (SELECTED_FEATURES, BASELINE_PARAMS),
        '2_clean_50':              (CLEAN_50, BASELINE_PARAMS),
        '3_clean_6_minimal':       (CLEAN_6, BASELINE_PARAMS),
        '4_replaced_52':           (REPLACED_52, BASELINE_PARAMS),
        '5_clean_50_regularized':  (CLEAN_50, SIMPLE_PARAMS),
        '6_clean_6_regularized':   (CLEAN_6, SIMPLE_PARAMS),
    }

    # --- Run all configs ---
    results = {}
    for name, (feat_list, params) in CONFIGS.items():
        print(f"\n{'='*75}")
        print(f"[*] Config: {name} ({len(feat_list)} features)")
        if params is SIMPLE_PARAMS:
            print(f"    Regularization: num_leaves=15, n_est=200, lr=0.03, "
                  f"subsample=0.5, colsample=0.5, reg_alpha=1.0, reg_lambda=10.0, min_child=200")
        print(f"{'='*75}")

        res = run_walkforward(df_base, feat_list, feature_cols_extended, params, name)
        if res:
            results[name] = res
            print(f"\n  [+] RESULT: {res['total_pnl']:+,.0f} EUR total | "
                  f"{res['eur_day']:+.0f} EUR/day | Sharpe={res['sharpe']:.2f} | "
                  f"Win={res['win_rate']:.0%}")
            print(f"      Losing months: {res['losing_months']}/{res['total_months']} | "
                  f"Worst month: {res['worst_month']:+,.0f} ({res['worst_month_name']})")
            print(f"      2024-25: {res['pre2026_eur_day']:+.0f}/day (Sharpe={res['pre2026_sharpe']:.2f}, "
                  f"{res['pre2026_days']} days) | "
                  f"2026: {res['post2026_eur_day']:+.0f}/day (Sharpe={res['post2026_sharpe']:.2f}, "
                  f"{res['post2026_days']} days)")
        else:
            print(f"  [-] No results for {name}")

    if not results:
        print("[!] No configs produced results")
        return

    # ===== SUMMARY TABLE =====
    print(f"\n\n{'='*100}")
    print("SUMMARY: LEAKAGE FIX COMPARISON")
    print(f"{'='*100}")
    print(f"\n{'Config':<30s} {'Feats':>5s} {'EUR/day':>8s} {'Sharpe':>7s} "
          f"{'Win%':>5s} {'Losing':>7s} {'Worst Mo':>10s} "
          f"{'24-25/d':>8s} {'2026/d':>8s}")
    print("-" * 100)

    for name, res in results.items():
        print(f"{res['config']:<30s} {res['n_features']:>5d} "
              f"{res['eur_day']:>+8.0f} {res['sharpe']:>7.2f} "
              f"{res['win_rate']:>5.0%} "
              f"{res['losing_months']:>3d}/{res['total_months']:<3d} "
              f"{res['worst_month']:>+10,.0f} "
              f"{res['pre2026_eur_day']:>+8.0f} {res['post2026_eur_day']:>+8.0f}")

    # --- Leakage impact ---
    if '1_baseline_52_leaky' in results and '2_clean_50' in results:
        b = results['1_baseline_52_leaky']
        c = results['2_clean_50']
        diff = c['eur_day'] - b['eur_day']
        pct = (diff / abs(b['eur_day']) * 100) if abs(b['eur_day']) > 0 else 0
        print(f"\n--- Leakage Impact ---")
        print(f"  Removing 2 leaky features: {diff:+.0f} EUR/day ({pct:+.1f}%)")
        print(f"  Baseline Sharpe: {b['sharpe']:.2f} -> Clean: {c['sharpe']:.2f}")

    if '1_baseline_52_leaky' in results and '4_replaced_52' in results:
        b = results['1_baseline_52_leaky']
        r = results['4_replaced_52']
        diff = r['eur_day'] - b['eur_day']
        pct = (diff / abs(b['eur_day']) * 100) if abs(b['eur_day']) > 0 else 0
        print(f"\n--- Replacement Features ---")
        print(f"  Replacing leaky with IDM-based: {diff:+.0f} EUR/day ({pct:+.1f}%)")
        print(f"  Baseline Sharpe: {b['sharpe']:.2f} -> Replaced: {r['sharpe']:.2f}")

    # --- Monthly P&L table ---
    print(f"\n\n--- Monthly EUR/day by Config ---")
    all_months = set()
    for res in results.values():
        all_months.update(res['monthly_ppd'].index)
    all_months = sorted(all_months)

    header = f"{'Month':<12s}"
    for name in results:
        short = name.split('_', 1)[1][:12]
        header += f" {short:>12s}"
    print(header)
    print("-" * (12 + 13 * len(results)))

    for m in all_months:
        row = f"{str(m):<12s}"
        for name, res in results.items():
            if m in res['monthly_ppd'].index:
                val = res['monthly_ppd'][m]
                row += f" {val:>+12.0f}"
            else:
                row += f" {'N/A':>12s}"
        print(row)

    # ===== PLOT =====
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Cumulative P&L
    ax = axes[0]
    for name, res in results.items():
        daily = res['daily_pnl']
        daily_s = pd.Series(daily.values, index=pd.DatetimeIndex(daily.index))
        cum = daily_s.sort_index().cumsum()
        label = f"{name} ({res['eur_day']:+.0f}/d, S={res['sharpe']:.2f})"
        ax.plot(cum.index, cum.values, label=label, linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_title('Cumulative P&L -- Leakage Fix Comparison (Walk-Forward OOS)')
    ax.set_ylabel('Cumulative P&L (EUR)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Monthly P&L bars
    ax = axes[1]
    n_configs = len(results)
    width = 0.8 / n_configs
    x = np.arange(len(all_months))
    for i, (name, res) in enumerate(results.items()):
        vals = [res['monthly_pnl'].get(m, 0) for m in all_months]
        short = name.split('_', 1)[1][:15]
        ax.bar(x + i * width, vals, width, label=short, alpha=0.8)
    ax.set_xticks(x + width * n_configs / 2)
    ax.set_xticklabels([str(m) for m in all_months], rotation=45, ha='right', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('Monthly P&L by Config')
    ax.set_ylabel('Monthly P&L (EUR)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = PLOT_DIR / "leakage_fix_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Plot saved: {plot_path}")

    elapsed = time.time() - t0
    print(f"\n[+] Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
