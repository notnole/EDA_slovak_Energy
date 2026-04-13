"""
Test Target Variants: Walk-Forward Comparison
===============================================

The current hourly target uses groupby('hour_ts').transform('mean') which
leaks future QH within the same hour. This script tests 4 target variants
on the full 16-month walk-forward to quantify the impact.

Targets:
  1. Current hourly (look-ahead): hourly_mean(settle) - hourly_mean(mid)
  2. Previous-hour:  shift(4) of hourly_mean(settle) - current exec_mid
  3. Raw 15-min (no smoothing): settle - exec_mid
  4. Lagged hourly: shift(4) of hourly_mean(settle) - shift(4) of hourly_mean(mid)

P&L always evaluated on real 15-min settlement with bid/ask execution.
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
PLOT_DIR = BASE_DIR / "plots" / "eda" / "target_variants"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LEAD = 8
SIZE_MW = 5.0
QH = 0.25
ENERGY = SIZE_MW * QH

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

THRESHOLD = 3
MAX_SPREAD = 10


def build_targets(df):
    """Build all 4 target variants."""
    df['imb_settlement_price'] = df['imb_settle_price']
    df['hour_ts'] = df.index.floor('h')

    # Hourly means (these use look-ahead within hour)
    settle_hourly = df.groupby('hour_ts')['imb_settlement_price'].transform('mean')
    mid_hourly = df.groupby('hour_ts')['exec_mid'].transform('mean')

    # Target 1: Current hourly (look-ahead within hour)
    df['target_hourly'] = settle_hourly - mid_hourly

    # Target 2: Previous-hour settle minus current exec_mid
    # shift(4) moves to previous hour since we have 4 QH per hour
    df['target_prev_hour'] = settle_hourly.shift(4) - df['exec_mid']

    # Target 3: Raw 15-min (no smoothing, "correct" target)
    df['target_raw'] = df['imb_settlement_price'] - df['exec_mid']

    # Target 4: Lagged hourly (previous hour's full spread)
    df['target_lagged_hourly'] = settle_hourly.shift(4) - mid_hourly.shift(4)

    return df


def run_walkforward(df_base, spread_features, target_col, label):
    """Run full walk-forward for a given target, return OOS trades."""
    all_oos = []

    for train_end, test_start, test_end in FOLDS:
        train = df_base[df_base.index < train_end].dropna(
            subset=[target_col, f'proxy_lag{LEAD+1}'])
        train = train[train['imb_settlement_price'].abs() <= 5000]
        train = train[train[target_col].notna()]

        test = df_base[(df_base.index >= test_start) & (df_base.index < test_end)].copy()
        test = test.dropna(subset=[f'proxy_lag{LEAD+1}'])
        test = test[test['exec_bid'].notna() & test['exec_ask'].notna()]
        test = test[test['exec_spread'] <= MAX_SPREAD]
        test = test[test['imb_settlement_price'].notna()]

        if len(train) < 1000 or len(test) < 50:
            continue

        model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        model.fit(train[spread_features].values, train[target_col].values)

        test['pred'] = model.predict(test[spread_features].values)

        # Trade with threshold
        surplus = test['pred'] <= -THRESHOLD
        deficit = test['pred'] >= THRESHOLD
        active = test[surplus | deficit].copy()

        if len(active) < 10:
            continue

        s = surplus.reindex(active.index, fill_value=False)
        d = deficit.reindex(active.index, fill_value=False)
        active['pnl'] = 0.0
        # P&L always on real 15-min settlement
        active.loc[s, 'pnl'] = (active.loc[s, 'exec_bid'] - active.loc[s, 'imb_settlement_price']) * ENERGY
        active.loc[d, 'pnl'] = (active.loc[d, 'imb_settlement_price'] - active.loc[d, 'exec_ask']) * ENERGY
        active['variant'] = label

        all_oos.append(active[['pnl', 'pred', 'variant']])

    if all_oos:
        return pd.concat(all_oos)
    return pd.DataFrame()


def print_summary(oos, label):
    """Print summary stats for one variant."""
    if oos.empty:
        print(f"  {label:30s} | NO TRADES")
        return {}

    daily = oos.groupby(oos.index.date)['pnl'].sum()
    nd = len(daily)
    total = oos['pnl'].sum()
    wr_trade = (oos['pnl'] > 0).mean()
    wr_daily = (daily > 0).mean()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0

    # Monthly breakdown
    oos_m = oos.copy()
    oos_m['month'] = oos_m.index.to_period('M')
    monthly_pnl = oos_m.groupby('month')['pnl'].sum()
    worst_month = monthly_pnl.min()
    best_month = monthly_pnl.max()
    loss_months = (monthly_pnl < 0).sum()

    print(f"  {label:30s} | {total:+8,.0f} EUR | {total/nd:+6.0f}/d | "
          f"Sh={sharpe:5.1f} | W(t)={wr_trade:.0%} | W(d)={wr_daily:.0%} | "
          f"{len(oos)} trades | worst_m={worst_month:+,.0f} | loss_m={loss_months}/{len(monthly_pnl)}")

    return dict(label=label, total=total, per_day=total/nd, sharpe=sharpe,
                wr_trade=wr_trade, wr_daily=wr_daily, n_trades=len(oos),
                n_days=nd, worst_month=worst_month, loss_months=loss_months,
                total_months=len(monthly_pnl))


def main():
    print("=" * 80)
    print("TARGET VARIANT COMPARISON - 16-Month Walk-Forward")
    print("=" * 80)
    print(f"[*] Lead={LEAD}, Size={SIZE_MW}MW, Threshold={THRESHOLD}, MaxSpread={MAX_SPREAD}")
    print(f"[*] Features: {len(SELECTED_FEATURES)}, Folds: {len(FOLDS)}")
    print()

    # Load data
    print("[*] Loading data...")
    data = load_all_data()
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, feature_cols = build_features(data, LEAD)

    # Validate features
    spread_features = [f for f in SELECTED_FEATURES if f in feature_cols]
    missing = [f for f in SELECTED_FEATURES if f not in feature_cols]
    if missing:
        print(f"[!] Missing features: {missing}")
    print(f"[+] Using {len(spread_features)} features")

    # Join execution prices
    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                          parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'spread', 'mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']

    df_base = df_base.join(ob_120, how='left')

    # Build all target variants
    df_base = build_targets(df_base)

    # Check target stats
    print()
    print("--- Target statistics (non-NaN rows in full dataset) ---")
    for tgt in ['target_hourly', 'target_prev_hour', 'target_raw', 'target_lagged_hourly']:
        valid = df_base[tgt].notna().sum()
        if valid > 0:
            s = df_base[tgt].dropna()
            print(f"  {tgt:25s}: n={valid:,}, mean={s.mean():+.2f}, std={s.std():.2f}, "
                  f"median={s.median():+.2f}")
    print()

    # Run walk-forward for each variant
    TARGET_CONFIGS = [
        ('target_hourly',        '1. Hourly (look-ahead)'),
        ('target_prev_hour',     '2. Previous-hour settle'),
        ('target_raw',           '3. Raw 15-min (correct)'),
        ('target_lagged_hourly', '4. Lagged hourly spread'),
    ]

    results = {}
    for target_col, label in TARGET_CONFIGS:
        print(f"\n{'='*60}")
        print(f"[*] Running: {label} -> {target_col}")
        print(f"{'='*60}")

        oos = run_walkforward(df_base, spread_features, target_col, label)
        results[label] = oos

        if not oos.empty:
            # Per-fold summary
            oos_m = oos.copy()
            oos_m['month'] = oos_m.index.to_period('M')
            for m, grp in oos_m.groupby('month'):
                d = grp.groupby(grp.index.date)['pnl'].sum()
                print(f"  {m}: {grp['pnl'].sum():+8,.0f} EUR ({grp['pnl'].sum()/len(d):+.0f}/d), "
                      f"{len(grp)} trades, {len(d)} days")

    # Summary comparison
    print()
    print("=" * 120)
    print("SUMMARY COMPARISON")
    print("=" * 120)
    print(f"  {'Variant':30s} | {'Total':>10s} | {'$/day':>7s} | "
          f"{'Sharpe':>6s} | {'W(t)':>5s} | {'W(d)':>5s} | "
          f"{'Trades':>6s} | {'Worst Mo':>10s} | {'Loss Mo':>8s}")
    print("-" * 120)

    summaries = []
    for target_col, label in TARGET_CONFIGS:
        oos = results[label]
        s = print_summary(oos, label)
        if s:
            summaries.append(s)

    print("-" * 120)

    # Correlation between variant predictions
    print()
    print("--- Direction agreement between variants (on overlapping test periods) ---")
    for i, (_, l1) in enumerate(TARGET_CONFIGS):
        for j, (_, l2) in enumerate(TARGET_CONFIGS):
            if j <= i:
                continue
            if results[l1].empty or results[l2].empty:
                continue
            common = results[l1].index.intersection(results[l2].index)
            if len(common) > 100:
                agree = (np.sign(results[l1].loc[common, 'pred']) ==
                         np.sign(results[l2].loc[common, 'pred'])).mean()
                print(f"  {l1} vs {l2}: {agree:.1%} same direction ({len(common)} common)")

    # Plot cumulative P&L
    print()
    print("[*] Generating comparison plot...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Cumulative daily P&L
    ax = axes[0]
    for target_col, label in TARGET_CONFIGS:
        oos = results[label]
        if oos.empty:
            continue
        daily = oos.groupby(oos.index.date)['pnl'].sum()
        daily_s = pd.Series(daily.values, index=pd.to_datetime(daily.index))
        cum = daily_s.cumsum()
        ax.plot(cum.index, cum.values, label=label, linewidth=1.5)

    ax.set_title('Cumulative P&L by Target Variant (Walk-Forward OOS)', fontsize=13)
    ax.set_ylabel('Cumulative P&L (EUR)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

    # Monthly P&L comparison
    ax = axes[1]
    monthly_data = {}
    all_months = set()
    for target_col, label in TARGET_CONFIGS:
        oos = results[label]
        if oos.empty:
            continue
        oos_m = oos.copy()
        oos_m['month'] = oos_m.index.to_period('M')
        mpnl = oos_m.groupby('month')['pnl'].sum()
        monthly_data[label] = mpnl
        all_months.update(mpnl.index)

    if monthly_data:
        months_sorted = sorted(all_months)
        x = np.arange(len(months_sorted))
        width = 0.2
        for i, (label, mpnl) in enumerate(monthly_data.items()):
            vals = [mpnl.get(m, 0) for m in months_sorted]
            ax.bar(x + i * width - 0.3, vals, width, label=label, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in months_sorted], rotation=45, ha='right', fontsize=8)
        ax.set_title('Monthly P&L by Target Variant', fontsize=13)
        ax.set_ylabel('Monthly P&L (EUR)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plot_path = PLOT_DIR / "target_variant_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Plot saved: {plot_path}")

    print()
    print("[+] Done.")


if __name__ == "__main__":
    main()
