"""
Minimal Feature Set Comparison
===============================

Tests whether simpler models are more robust across the full 16-month
walk-forward period (Oct 2024 - Mar 2026).

Configurations tested:
  1. Full model (52 features) -- baseline
  2. Market + Time (8 features)
  3. Market + Time + DA prices (14 features)
  4. Market + Time + DA + Weather (22 features)
  5. Top 30 by permutation importance
  6. Top 20
  7. Top 10

Same setup as walkforward_montecarlo.py: quantile LGB (alpha=0.50),
hourly-smoothed spread target, 5 MW, bid/ask execution, threshold=3.
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
PLOT_DIR = BASE_DIR / "plots" / "eda" / "minimal_features"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LEAD = 8
SIZE_MW = 5.0
QH = 0.25
ENERGY = SIZE_MW * QH

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                  reg_lambda=1.0, n_estimators=600, verbose=-1)

# 52 selected features in permutation importance order
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

# --- Feature set configurations ---
MARKET_TIME = [
    'idm_vwap_lag', 'spread_da_imb_lag', 'imb_price_rmean4',
    'hour_cos', 'hour_sin', 'dow_sin', 'dow_cos', 'is_weekend',
]

DA_PRICES = ['da_price', 'da_supply', 'da_price_change24h', 'da_demand', 'da_flow_cz', 'da_net_import']

WEATHER = [
    'cloudcover', 'temp_forecast_da', 'temp_national_spread', 'temp_bratislava',
    'temp_national_change6h', 'temp_surprise_lag', 'radiation_national', 'temp_rmean24h',
]

CONFIGS = {
    '1_full_52':            SELECTED_FEATURES,
    '2_market_time_8':      MARKET_TIME,
    '3_mkt_time_da_14':     MARKET_TIME + DA_PRICES,
    '4_mkt_time_da_wx_22':  MARKET_TIME + DA_PRICES + WEATHER,
    '5_top30':              SELECTED_FEATURES[:30],
    '6_top20':              SELECTED_FEATURES[:20],
    '7_top10':              SELECTED_FEATURES[:10],
}


def run_walkforward(df_base, feature_list, feature_cols, config_name):
    """Run walk-forward for a single feature config. Returns dict of results."""
    # Validate features exist
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

        model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        model.fit(train[feats].values, train['spread_target'].values)

        test['pred'] = model.predict(test[feats].values)

        surplus = test['pred'] <= -3
        deficit = test['pred'] >= 3
        active = test[surplus | deficit].copy()

        if len(active) < 5:
            continue

        s = surplus.reindex(active.index, fill_value=False)
        d = deficit.reindex(active.index, fill_value=False)
        active['pnl'] = 0.0
        active.loc[s, 'pnl'] = (active.loc[s, 'exec_bid'] - active.loc[s, 'imb_settlement_price']) * ENERGY
        active.loc[d, 'pnl'] = (active.loc[d, 'imb_settlement_price'] - active.loc[d, 'exec_ask']) * ENERGY

        all_trades.append(active[['pnl', 'pred']])

    if not all_trades:
        return None

    oos = pd.concat(all_trades)
    oos_daily = oos.groupby(oos.index.date)['pnl'].sum()

    # Monthly breakdown
    oos_m = oos.copy()
    oos_m['month'] = oos_m.index.to_period('M')
    monthly_pnl = oos_m.groupby('month')['pnl'].sum()
    monthly_days = oos_m.groupby('month').apply(lambda x: x.index.normalize().nunique())
    monthly_ppd = monthly_pnl / monthly_days

    total = oos['pnl'].sum()
    n_days = len(oos_daily)
    eur_day = oos_daily.mean()
    sharpe = oos_daily.mean() / oos_daily.std() * np.sqrt(252) if oos_daily.std() > 0 else 0
    win_rate = (oos['pnl'] > 0).mean()
    losing_months = (monthly_pnl < 0).sum()
    worst_month = monthly_pnl.min()
    worst_month_name = str(monthly_pnl.idxmin()) if len(monthly_pnl) > 0 else 'N/A'

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
        'total_months': len(monthly_pnl),
        'worst_month': worst_month,
        'worst_month_name': worst_month_name,
        'monthly_pnl': monthly_pnl,
        'monthly_ppd': monthly_ppd,
        'daily_pnl': oos_daily,
    }


def main():
    t0 = time.time()
    print("=" * 75)
    print("MINIMAL FEATURE SET COMPARISON -- WALK-FORWARD")
    print("=" * 75)
    print(f"[*] Testing {len(CONFIGS)} configurations across {len(FOLDS)} monthly folds")

    # --- Load data ---
    print("\n[*] Loading data...")
    data = load_all_data()
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, feature_cols = build_features(data, LEAD)

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

    print(f"[+] Data ready: {len(df_base)} rows, {len(feature_cols)} total features")

    # --- Run all configs ---
    results = {}
    for name, feat_list in CONFIGS.items():
        print(f"\n{'='*75}")
        print(f"[*] Config: {name} ({len(feat_list)} features)")
        print(f"{'='*75}")
        res = run_walkforward(df_base, feat_list, feature_cols, name)
        if res:
            results[name] = res
            print(f"  [+] Total P&L: {res['total_pnl']:+,.0f} EUR | "
                  f"{res['eur_day']:+.0f} EUR/day | Sharpe={res['sharpe']:.2f} | "
                  f"Win={res['win_rate']:.0%} | "
                  f"Losing months: {res['losing_months']}/{res['total_months']} | "
                  f"Worst month: {res['worst_month']:+,.0f} ({res['worst_month_name']})")
        else:
            print(f"  [-] No results for {name}")

    if not results:
        print("[!] No configs produced results")
        return

    # --- Summary table ---
    print(f"\n\n{'='*100}")
    print("SUMMARY TABLE -- ALL CONFIGURATIONS")
    print(f"{'='*100}")
    header = f"{'Config':<25s} {'#Feat':>5s} {'Total EUR':>12s} {'EUR/day':>9s} " \
             f"{'Sharpe':>7s} {'Win%':>6s} {'Trades':>7s} " \
             f"{'Lose Mo':>8s} {'Worst Mo':>10s}"
    print(header)
    print("-" * 100)

    sorted_configs = sorted(results.values(), key=lambda x: x['sharpe'], reverse=True)
    for r in sorted_configs:
        print(f"{r['config']:<25s} {r['n_features']:>5d} {r['total_pnl']:>+12,.0f} "
              f"{r['eur_day']:>+9.0f} {r['sharpe']:>7.2f} {r['win_rate']:>6.0%} "
              f"{r['n_trades']:>7d} {r['losing_months']:>4d}/{r['total_months']:<3d} "
              f"{r['worst_month']:>+10,.0f}")

    # --- Monthly breakdown for top 3 ---
    top3 = sorted_configs[:3]
    print(f"\n\n{'='*100}")
    print("MONTHLY BREAKDOWN -- TOP 3 CONFIGS BY SHARPE")
    print(f"{'='*100}")

    # Collect all months
    all_months = set()
    for r in top3:
        all_months.update(r['monthly_ppd'].index)
    all_months = sorted(all_months)

    header_m = f"{'Month':<12s}"
    for r in top3:
        header_m += f"  {r['config']:>20s}"
    print(header_m)
    print("-" * (12 + 22 * len(top3)))

    for m in all_months:
        row = f"{str(m):<12s}"
        for r in top3:
            if m in r['monthly_ppd'].index:
                v = r['monthly_ppd'][m]
                row += f"  {v:>+18.0f}/d"
            else:
                row += f"  {'---':>20s}"
        print(row)

    # Totals
    row = f"{'TOTAL':<12s}"
    for r in top3:
        row += f"  {r['eur_day']:>+18.0f}/d"
    print("-" * (12 + 22 * len(top3)))
    print(row)

    # --- Feb-Mar 2026 focus ---
    print(f"\n\n{'='*100}")
    print("SIGNAL DEGRADATION CHECK: Feb-Mar 2026 vs Earlier")
    print(f"{'='*100}")

    for r in sorted_configs:
        early = r['monthly_ppd'][[m for m in r['monthly_ppd'].index if m < pd.Period('2026-02', 'M')]]
        late = r['monthly_ppd'][[m for m in r['monthly_ppd'].index if m >= pd.Period('2026-02', 'M')]]
        if len(early) > 0 and len(late) > 0:
            print(f"  {r['config']:<25s} | Pre-Feb26: {early.mean():>+7.0f}/d | "
                  f"Feb-Mar26: {late.mean():>+7.0f}/d | "
                  f"Delta: {late.mean() - early.mean():>+7.0f}/d")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Minimal Feature Set Comparison -- Walk-Forward OOS', fontsize=14, fontweight='bold')

    # 1. Total P&L bar chart
    ax = axes[0, 0]
    names = [r['config'] for r in sorted_configs]
    totals = [r['total_pnl'] for r in sorted_configs]
    colors = ['green' if v > 0 else 'red' for v in totals]
    bars = ax.barh(range(len(names)), totals, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', ls='-', alpha=0.5)
    ax.set_xlabel('Total P&L (EUR)')
    ax.set_title('Total Walk-Forward P&L (sorted by Sharpe)')

    # 2. Sharpe comparison
    ax = axes[0, 1]
    sharpes = [r['sharpe'] for r in sorted_configs]
    colors_s = ['darkgreen' if v > 2 else 'green' if v > 1 else 'orange' if v > 0 else 'red' for v in sharpes]
    ax.barh(range(len(names)), sharpes, color=colors_s, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', ls='-', alpha=0.5)
    ax.axvline(1, color='blue', ls='--', alpha=0.3, label='Sharpe=1')
    ax.set_xlabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio by Config')
    ax.legend()

    # 3. Equity curves for top configs
    ax = axes[1, 0]
    cmap = plt.cm.tab10
    for i, r in enumerate(sorted_configs):
        eq = r['daily_pnl'].cumsum()
        ax.plot(range(len(eq)), eq.values, label=f"{r['config']} ({r['n_features']}f)",
                color=cmap(i), lw=1.5, alpha=0.8)
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Cumulative P&L (EUR)')
    ax.set_title('Walk-Forward Equity Curves')
    ax.legend(fontsize=7, loc='best')

    # 4. Monthly heatmap for top 3
    ax = axes[1, 1]
    width = 0.25
    x = np.arange(len(all_months))
    for i, r in enumerate(top3):
        vals = [r['monthly_ppd'].get(m, 0) for m in all_months]
        ax.bar(x + i * width, vals, width=width, alpha=0.7,
               label=f"{r['config']} ({r['n_features']}f)", color=cmap(i))
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(m) for m in all_months], rotation=45, fontsize=7)
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.set_ylabel('EUR/day')
    ax.set_title('Monthly P&L/day -- Top 3 Configs')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "01_minimal_features_comparison.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"\n[+] Saved: {PLOT_DIR / '01_minimal_features_comparison.png'}")

    elapsed = time.time() - t0
    print(f"\n[+] Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
