"""
Threshold Strategy Analysis for Spread Model
=============================================

Tests adaptive and alternative threshold strategies on the full 16-month
walk-forward to understand how the fixed |pred|>=3 threshold performs vs
alternatives, especially as the IDM-settlement spread narrows in 2026.

Strategies tested:
  1. Fixed threshold sweep: |pred| >= 1, 2, 3, 5, 8, 10
  2. Flat position (5 MW flat) vs confidence-scaled sizing
  3. Percentile-based adaptive threshold (25th/50th on trailing 30 days)
  4. Volatility-scaled threshold (rolling 30d spread vol)
  5. Direction-only model (trade ALL periods, flat 5 MW)
  6. Split analysis: 2024-2025 vs 2026 for each strategy
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
PLOT_DIR = BASE_DIR / "plots" / "eda" / "thresholds"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LEAD = 8
SIZE_MW = 5.0
QH = 0.25
ENERGY = SIZE_MW * QH  # 1.25 MWh

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

FIXED_THRESHOLDS = [1, 2, 3, 5, 8, 10]


def compute_pnl_for_trades(test_df, threshold, sizing='confidence'):
    """
    Compute P&L for a given threshold and sizing strategy.

    sizing='confidence': size = |pred|.clip(upper=5), energy = size * 0.25
    sizing='flat':       energy = 5 * 0.25 = 1.25 for all trades
    """
    surplus = test_df['pred'] <= -threshold
    deficit = test_df['pred'] >= threshold

    active = test_df[surplus | deficit].copy()
    if len(active) == 0:
        return active

    s = surplus.reindex(active.index, fill_value=False)
    d = deficit.reindex(active.index, fill_value=False)

    if sizing == 'confidence':
        active['size_mw'] = active['pred'].abs().clip(upper=5)
        active['energy'] = active['size_mw'] * QH
    else:  # flat
        active['size_mw'] = SIZE_MW
        active['energy'] = ENERGY

    active['pnl'] = 0.0
    if s.sum() > 0:
        active.loc[s, 'pnl'] = (active.loc[s, 'exec_bid'] - active.loc[s, 'imb_settlement_price']) * active.loc[s, 'energy']
    if d.sum() > 0:
        active.loc[d, 'pnl'] = (active.loc[d, 'imb_settlement_price'] - active.loc[d, 'exec_ask']) * active.loc[d, 'energy']

    return active


def compute_direction_only_pnl(test_df):
    """Trade ALL periods with flat 5 MW in predicted direction. No threshold."""
    active = test_df.copy()
    active['energy'] = ENERGY

    surplus = active['pred'] < 0
    deficit = active['pred'] >= 0

    active['pnl'] = 0.0
    if surplus.sum() > 0:
        active.loc[surplus, 'pnl'] = (active.loc[surplus, 'exec_bid'] - active.loc[surplus, 'imb_settlement_price']) * ENERGY
    if deficit.sum() > 0:
        active.loc[deficit, 'pnl'] = (active.loc[deficit, 'imb_settlement_price'] - active.loc[deficit, 'exec_ask']) * ENERGY

    return active


def summarize_trades(trades, label=''):
    """Return a dict with summary stats for a set of trades."""
    if len(trades) == 0:
        return {'label': label, 'n_trades': 0, 'total_pnl': 0, 'daily_pnl': 0,
                'sharpe': 0, 'win_rate': 0, 'n_days': 0}

    nd = trades.index.normalize().nunique()
    total = trades['pnl'].sum()
    daily = trades.groupby(trades.index.date)['pnl'].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    wr = (trades['pnl'] > 0).mean()

    return {
        'label': label,
        'n_trades': len(trades),
        'n_days': nd,
        'total_pnl': total,
        'daily_pnl': total / nd if nd > 0 else 0,
        'sharpe': sharpe,
        'win_rate': wr,
        'trades_per_day': len(trades) / nd if nd > 0 else 0,
        'avg_pnl': trades['pnl'].mean(),
    }


def split_period(trades):
    """Split trades into 2024-2025 and 2026 subsets."""
    if len(trades) == 0:
        return pd.DataFrame(), pd.DataFrame()
    mask_2026 = trades.index >= '2026-01-01'
    return trades[~mask_2026], trades[mask_2026]


def main():
    print("=" * 70)
    print("THRESHOLD STRATEGY ANALYSIS")
    print("=" * 70)

    # --- Load data ---
    data = load_all_data()
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, feature_cols = build_features(data, LEAD)

    spread_features = [f for f in SELECTED_FEATURES if f in feature_cols]
    missing = [f for f in SELECTED_FEATURES if f not in feature_cols]
    if missing:
        print(f"[!] Warning: {len(missing)} selected features not found: {missing}")

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

    print(f"[+] Base data: {len(df_base)} rows, {len(spread_features)} selected features")

    # ===================================================================
    # WALK-FORWARD: collect predictions for all folds
    # ===================================================================
    all_test_rows = []

    for train_end, test_start, test_end in FOLDS:
        fold_label = pd.Timestamp(test_start).strftime('%Y-%m')
        print(f"\n--- Fold: {fold_label} (train < {train_end}, test [{test_start}, {test_end})) ---")

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
        model.fit(train[spread_features].values, train['spread_target'].values)

        # Predict
        test['pred'] = model.predict(test[spread_features].values)

        # Compute adaptive thresholds for this fold
        # Percentile-based: use last 30 days of training predictions
        train_recent = train[train.index >= (pd.Timestamp(train_end) - pd.Timedelta(days=30))]
        if len(train_recent) > 100:
            train_preds_recent = model.predict(train_recent[spread_features].values)
            test['thresh_p25'] = np.percentile(np.abs(train_preds_recent), 25)
            test['thresh_p50'] = np.percentile(np.abs(train_preds_recent), 50)
        else:
            test['thresh_p25'] = 1.0
            test['thresh_p50'] = 2.0

        # Volatility-scaled threshold
        # Rolling 30-day std of spread target from training
        spread_30d_std = train[train.index >= (pd.Timestamp(train_end) - pd.Timedelta(days=30))]['spread_target'].std()
        # Historical vol: overall training std
        spread_hist_std = train['spread_target'].std()
        vol_ratio = spread_30d_std / spread_hist_std if spread_hist_std > 0 else 1.0
        test['vol_ratio'] = vol_ratio
        test['thresh_vol3'] = 3.0 * vol_ratio  # base=3
        test['thresh_vol2'] = 2.0 * vol_ratio  # base=2

        print(f"  Pred |mean|={test['pred'].abs().mean():.2f}, "
              f"thresh_p25={test['thresh_p25'].iloc[0]:.2f}, "
              f"thresh_p50={test['thresh_p50'].iloc[0]:.2f}, "
              f"vol_ratio={vol_ratio:.3f}, "
              f"thresh_vol3={test['thresh_vol3'].iloc[0]:.2f}")

        all_test_rows.append(test)

    if not all_test_rows:
        print("[!] No test data collected")
        return

    oos = pd.concat(all_test_rows)
    print(f"\n[+] Total OOS rows: {len(oos)}, date range: {oos.index.min()} to {oos.index.max()}")

    # ===================================================================
    # STRATEGY 1: Fixed threshold sweep
    # ===================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 1: FIXED THRESHOLD SWEEP (confidence sizing)")
    print("=" * 70)

    fixed_results = []
    for thr in FIXED_THRESHOLDS:
        trades = compute_pnl_for_trades(oos, thr, sizing='confidence')
        s = summarize_trades(trades, f'fixed_{thr}')
        s['threshold'] = thr
        fixed_results.append(s)

        trades_25, trades_26 = split_period(trades)
        s25 = summarize_trades(trades_25, f'fixed_{thr}_2025')
        s26 = summarize_trades(trades_26, f'fixed_{thr}_2026')

        print(f"\n  Threshold |pred| >= {thr}:")
        print(f"    ALL:       {s['n_trades']:5d} trades, {s['total_pnl']:+10,.0f} EUR, "
              f"{s['daily_pnl']:+7.0f}/day, Sharpe={s['sharpe']:.1f}, "
              f"Win={s['win_rate']:.0%}, {s['trades_per_day']:.1f} tpd")
        print(f"    2024-2025: {s25['n_trades']:5d} trades, {s25['total_pnl']:+10,.0f} EUR, "
              f"{s25['daily_pnl']:+7.0f}/day, Sharpe={s25['sharpe']:.1f}, "
              f"Win={s25['win_rate']:.0%}")
        print(f"    2026:      {s26['n_trades']:5d} trades, {s26['total_pnl']:+10,.0f} EUR, "
              f"{s26['daily_pnl']:+7.0f}/day, Sharpe={s26['sharpe']:.1f}, "
              f"Win={s26['win_rate']:.0%}")

    # ===================================================================
    # STRATEGY 2: Flat position (no confidence scaling)
    # ===================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 2: FLAT 5 MW SIZING (vs confidence sizing)")
    print("=" * 70)

    flat_results = []
    for thr in FIXED_THRESHOLDS:
        trades_conf = compute_pnl_for_trades(oos, thr, sizing='confidence')
        trades_flat = compute_pnl_for_trades(oos, thr, sizing='flat')
        sc = summarize_trades(trades_conf, f'conf_{thr}')
        sf = summarize_trades(trades_flat, f'flat_{thr}')
        sf['threshold'] = thr
        flat_results.append(sf)

        # Split for flat
        tf25, tf26 = split_period(trades_flat)
        sf25 = summarize_trades(tf25, f'flat_{thr}_2025')
        sf26 = summarize_trades(tf26, f'flat_{thr}_2026')

        print(f"\n  Threshold >= {thr}:")
        print(f"    Confidence: {sc['total_pnl']:+10,.0f} EUR, {sc['daily_pnl']:+7.0f}/day, "
              f"Sharpe={sc['sharpe']:.1f}")
        print(f"    Flat 5 MW:  {sf['total_pnl']:+10,.0f} EUR, {sf['daily_pnl']:+7.0f}/day, "
              f"Sharpe={sf['sharpe']:.1f}")
        print(f"    Flat 2025:  {sf25['total_pnl']:+10,.0f} EUR, {sf25['daily_pnl']:+7.0f}/day, "
              f"Sharpe={sf25['sharpe']:.1f}")
        print(f"    Flat 2026:  {sf26['total_pnl']:+10,.0f} EUR, {sf26['daily_pnl']:+7.0f}/day, "
              f"Sharpe={sf26['sharpe']:.1f}")

    # ===================================================================
    # STRATEGY 3: Percentile-based adaptive threshold
    # ===================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 3: PERCENTILE-BASED ADAPTIVE THRESHOLD")
    print("=" * 70)

    for pctl_col, pctl_label in [('thresh_p25', 'P25'), ('thresh_p50', 'P50')]:
        # Apply per-row adaptive threshold
        surplus = oos['pred'] <= -oos[pctl_col]
        deficit = oos['pred'] >= oos[pctl_col]
        active = oos[surplus | deficit].copy()
        s_mask = surplus.reindex(active.index, fill_value=False)
        d_mask = deficit.reindex(active.index, fill_value=False)

        # Confidence sizing
        active['size_mw'] = active['pred'].abs().clip(upper=5)
        active['energy'] = active['size_mw'] * QH
        active['pnl'] = 0.0
        if s_mask.sum() > 0:
            active.loc[s_mask, 'pnl'] = (active.loc[s_mask, 'exec_bid'] - active.loc[s_mask, 'imb_settlement_price']) * active.loc[s_mask, 'energy']
        if d_mask.sum() > 0:
            active.loc[d_mask, 'pnl'] = (active.loc[d_mask, 'imb_settlement_price'] - active.loc[d_mask, 'exec_ask']) * active.loc[d_mask, 'energy']

        sa = summarize_trades(active, f'adaptive_{pctl_label}_conf')
        a25, a26 = split_period(active)
        sa25 = summarize_trades(a25, f'adaptive_{pctl_label}_2025')
        sa26 = summarize_trades(a26, f'adaptive_{pctl_label}_2026')

        # Show the actual thresholds used
        thresh_values = oos.groupby(oos.index.to_period('M'))[pctl_col].first()
        thresh_str = ", ".join([f"{m}:{v:.2f}" for m, v in thresh_values.items()])

        print(f"\n  Adaptive {pctl_label} (confidence sizing):")
        print(f"    Thresholds by month: {thresh_str}")
        print(f"    ALL:       {sa['n_trades']:5d} trades, {sa['total_pnl']:+10,.0f} EUR, "
              f"{sa['daily_pnl']:+7.0f}/day, Sharpe={sa['sharpe']:.1f}, "
              f"Win={sa['win_rate']:.0%}, {sa['trades_per_day']:.1f} tpd")
        print(f"    2024-2025: {sa25['n_trades']:5d} trades, {sa25['total_pnl']:+10,.0f} EUR, "
              f"{sa25['daily_pnl']:+7.0f}/day, Sharpe={sa25['sharpe']:.1f}, "
              f"Win={sa25['win_rate']:.0%}")
        print(f"    2026:      {sa26['n_trades']:5d} trades, {sa26['total_pnl']:+10,.0f} EUR, "
              f"{sa26['daily_pnl']:+7.0f}/day, Sharpe={sa26['sharpe']:.1f}, "
              f"Win={sa26['win_rate']:.0%}")

        # Also flat sizing
        active_flat = oos[surplus | deficit].copy()
        active_flat['energy'] = ENERGY
        active_flat['pnl'] = 0.0
        s_mask_f = surplus.reindex(active_flat.index, fill_value=False)
        d_mask_f = deficit.reindex(active_flat.index, fill_value=False)
        if s_mask_f.sum() > 0:
            active_flat.loc[s_mask_f, 'pnl'] = (active_flat.loc[s_mask_f, 'exec_bid'] - active_flat.loc[s_mask_f, 'imb_settlement_price']) * ENERGY
        if d_mask_f.sum() > 0:
            active_flat.loc[d_mask_f, 'pnl'] = (active_flat.loc[d_mask_f, 'imb_settlement_price'] - active_flat.loc[d_mask_f, 'exec_ask']) * ENERGY

        sf_a = summarize_trades(active_flat, f'adaptive_{pctl_label}_flat')
        print(f"    Flat 5MW:  {sf_a['n_trades']:5d} trades, {sf_a['total_pnl']:+10,.0f} EUR, "
              f"{sf_a['daily_pnl']:+7.0f}/day, Sharpe={sf_a['sharpe']:.1f}")

    # ===================================================================
    # STRATEGY 4: Volatility-scaled threshold
    # ===================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 4: VOLATILITY-SCALED THRESHOLD")
    print("=" * 70)

    for vol_col, base, vol_label in [('thresh_vol3', 3, 'vol_base3'), ('thresh_vol2', 2, 'vol_base2')]:
        surplus = oos['pred'] <= -oos[vol_col]
        deficit = oos['pred'] >= oos[vol_col]
        active = oos[surplus | deficit].copy()
        s_mask = surplus.reindex(active.index, fill_value=False)
        d_mask = deficit.reindex(active.index, fill_value=False)

        active['size_mw'] = active['pred'].abs().clip(upper=5)
        active['energy'] = active['size_mw'] * QH
        active['pnl'] = 0.0
        if s_mask.sum() > 0:
            active.loc[s_mask, 'pnl'] = (active.loc[s_mask, 'exec_bid'] - active.loc[s_mask, 'imb_settlement_price']) * active.loc[s_mask, 'energy']
        if d_mask.sum() > 0:
            active.loc[d_mask, 'pnl'] = (active.loc[d_mask, 'imb_settlement_price'] - active.loc[d_mask, 'exec_ask']) * active.loc[d_mask, 'energy']

        sv = summarize_trades(active, vol_label)
        v25, v26 = split_period(active)
        sv25 = summarize_trades(v25, f'{vol_label}_2025')
        sv26 = summarize_trades(v26, f'{vol_label}_2026')

        # Show effective thresholds
        thresh_values = oos.groupby(oos.index.to_period('M'))[vol_col].first()
        thresh_str = ", ".join([f"{m}:{v:.2f}" for m, v in thresh_values.items()])
        vol_ratios = oos.groupby(oos.index.to_period('M'))['vol_ratio'].first()
        vol_str = ", ".join([f"{m}:{v:.3f}" for m, v in vol_ratios.items()])

        print(f"\n  Volatility-scaled base={base}:")
        print(f"    Vol ratios:  {vol_str}")
        print(f"    Eff. thresh: {thresh_str}")
        print(f"    ALL:       {sv['n_trades']:5d} trades, {sv['total_pnl']:+10,.0f} EUR, "
              f"{sv['daily_pnl']:+7.0f}/day, Sharpe={sv['sharpe']:.1f}, "
              f"Win={sv['win_rate']:.0%}, {sv['trades_per_day']:.1f} tpd")
        print(f"    2024-2025: {sv25['n_trades']:5d} trades, {sv25['total_pnl']:+10,.0f} EUR, "
              f"{sv25['daily_pnl']:+7.0f}/day, Sharpe={sv25['sharpe']:.1f}, "
              f"Win={sv25['win_rate']:.0%}")
        print(f"    2026:      {sv26['n_trades']:5d} trades, {sv26['total_pnl']:+10,.0f} EUR, "
              f"{sv26['daily_pnl']:+7.0f}/day, Sharpe={sv26['sharpe']:.1f}, "
              f"Win={sv26['win_rate']:.0%}")

    # ===================================================================
    # STRATEGY 5: Direction-only model (trade ALL, flat 5 MW)
    # ===================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 5: DIRECTION-ONLY (trade ALL, flat 5 MW)")
    print("=" * 70)

    dir_trades = compute_direction_only_pnl(oos)
    sd = summarize_trades(dir_trades, 'direction_only')
    d25, d26 = split_period(dir_trades)
    sd25 = summarize_trades(d25, 'dir_2025')
    sd26 = summarize_trades(d26, 'dir_2026')

    dir_acc_all = (np.sign(oos['pred']) == np.sign(oos['spread_target'])).mean()
    dir_acc_25 = np.nan
    dir_acc_26 = np.nan
    oos_25 = oos[oos.index < '2026-01-01']
    oos_26 = oos[oos.index >= '2026-01-01']
    if len(oos_25) > 0:
        dir_acc_25 = (np.sign(oos_25['pred']) == np.sign(oos_25['spread_target'])).mean()
    if len(oos_26) > 0:
        dir_acc_26 = (np.sign(oos_26['pred']) == np.sign(oos_26['spread_target'])).mean()

    print(f"\n  Direction accuracy: ALL={dir_acc_all:.1%}, 2024-2025={dir_acc_25:.1%}, 2026={dir_acc_26:.1%}")
    print(f"  ALL:       {sd['n_trades']:5d} trades, {sd['total_pnl']:+10,.0f} EUR, "
          f"{sd['daily_pnl']:+7.0f}/day, Sharpe={sd['sharpe']:.1f}, "
          f"Win={sd['win_rate']:.0%}")
    print(f"  2024-2025: {sd25['n_trades']:5d} trades, {sd25['total_pnl']:+10,.0f} EUR, "
          f"{sd25['daily_pnl']:+7.0f}/day, Sharpe={sd25['sharpe']:.1f}, "
          f"Win={sd25['win_rate']:.0%}")
    print(f"  2026:      {sd26['n_trades']:5d} trades, {sd26['total_pnl']:+10,.0f} EUR, "
          f"{sd26['daily_pnl']:+7.0f}/day, Sharpe={sd26['sharpe']:.1f}, "
          f"Win={sd26['win_rate']:.0%}")

    # ===================================================================
    # SUMMARY COMPARISON TABLE
    # ===================================================================
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 70)

    print(f"\n{'Strategy':<35s} {'Total EUR':>10s} {'EUR/day':>8s} {'Sharpe':>7s} "
          f"{'Trades':>7s} {'TPD':>5s} {'Win%':>5s} "
          f"| {'2025 EUR/d':>10s} {'2026 EUR/d':>10s}")
    print("-" * 110)

    # Collect all strategies for the summary
    all_strategies = []

    # Fixed thresholds (confidence)
    for thr in FIXED_THRESHOLDS:
        trades = compute_pnl_for_trades(oos, thr, sizing='confidence')
        s = summarize_trades(trades)
        t25, t26 = split_period(trades)
        s25 = summarize_trades(t25)
        s26 = summarize_trades(t26)
        row = {
            'name': f'Fixed>={thr} (conf)',
            'total': s['total_pnl'], 'daily': s['daily_pnl'],
            'sharpe': s['sharpe'], 'trades': s['n_trades'],
            'tpd': s['trades_per_day'], 'wr': s['win_rate'],
            'd25': s25['daily_pnl'], 'd26': s26['daily_pnl'],
        }
        all_strategies.append(row)

    # Fixed thresholds (flat)
    for thr in [2, 3, 5]:
        trades = compute_pnl_for_trades(oos, thr, sizing='flat')
        s = summarize_trades(trades)
        t25, t26 = split_period(trades)
        s25 = summarize_trades(t25)
        s26 = summarize_trades(t26)
        row = {
            'name': f'Fixed>={thr} (flat 5MW)',
            'total': s['total_pnl'], 'daily': s['daily_pnl'],
            'sharpe': s['sharpe'], 'trades': s['n_trades'],
            'tpd': s['trades_per_day'], 'wr': s['win_rate'],
            'd25': s25['daily_pnl'], 'd26': s26['daily_pnl'],
        }
        all_strategies.append(row)

    # Adaptive percentile
    for pctl_col, pctl_label in [('thresh_p25', 'Adpt P25'), ('thresh_p50', 'Adpt P50')]:
        surplus = oos['pred'] <= -oos[pctl_col]
        deficit = oos['pred'] >= oos[pctl_col]
        active = oos[surplus | deficit].copy()
        s_m = surplus.reindex(active.index, fill_value=False)
        d_m = deficit.reindex(active.index, fill_value=False)
        active['energy'] = active['pred'].abs().clip(upper=5) * QH
        active['pnl'] = 0.0
        if s_m.sum() > 0:
            active.loc[s_m, 'pnl'] = (active.loc[s_m, 'exec_bid'] - active.loc[s_m, 'imb_settlement_price']) * active.loc[s_m, 'energy']
        if d_m.sum() > 0:
            active.loc[d_m, 'pnl'] = (active.loc[d_m, 'imb_settlement_price'] - active.loc[d_m, 'exec_ask']) * active.loc[d_m, 'energy']

        s = summarize_trades(active)
        a25, a26 = split_period(active)
        s25 = summarize_trades(a25)
        s26 = summarize_trades(a26)
        all_strategies.append({
            'name': f'{pctl_label} (conf)',
            'total': s['total_pnl'], 'daily': s['daily_pnl'],
            'sharpe': s['sharpe'], 'trades': s['n_trades'],
            'tpd': s['trades_per_day'], 'wr': s['win_rate'],
            'd25': s25['daily_pnl'], 'd26': s26['daily_pnl'],
        })

    # Volatility-scaled
    for vol_col, base in [('thresh_vol3', 3), ('thresh_vol2', 2)]:
        surplus = oos['pred'] <= -oos[vol_col]
        deficit = oos['pred'] >= oos[vol_col]
        active = oos[surplus | deficit].copy()
        s_m = surplus.reindex(active.index, fill_value=False)
        d_m = deficit.reindex(active.index, fill_value=False)
        active['energy'] = active['pred'].abs().clip(upper=5) * QH
        active['pnl'] = 0.0
        if s_m.sum() > 0:
            active.loc[s_m, 'pnl'] = (active.loc[s_m, 'exec_bid'] - active.loc[s_m, 'imb_settlement_price']) * active.loc[s_m, 'energy']
        if d_m.sum() > 0:
            active.loc[d_m, 'pnl'] = (active.loc[d_m, 'imb_settlement_price'] - active.loc[d_m, 'exec_ask']) * active.loc[d_m, 'energy']

        s = summarize_trades(active)
        a25, a26 = split_period(active)
        s25 = summarize_trades(a25)
        s26 = summarize_trades(a26)
        all_strategies.append({
            'name': f'Vol-scaled base={base} (conf)',
            'total': s['total_pnl'], 'daily': s['daily_pnl'],
            'sharpe': s['sharpe'], 'trades': s['n_trades'],
            'tpd': s['trades_per_day'], 'wr': s['win_rate'],
            'd25': s25['daily_pnl'], 'd26': s26['daily_pnl'],
        })

    # Direction-only
    all_strategies.append({
        'name': 'Direction-only (flat 5MW)',
        'total': sd['total_pnl'], 'daily': sd['daily_pnl'],
        'sharpe': sd['sharpe'], 'trades': sd['n_trades'],
        'tpd': sd['trades_per_day'], 'wr': sd['win_rate'],
        'd25': sd25['daily_pnl'], 'd26': sd26['daily_pnl'],
    })

    # Print sorted by daily PnL
    all_strategies.sort(key=lambda x: x['daily'], reverse=True)
    for r in all_strategies:
        print(f"  {r['name']:<33s} {r['total']:+10,.0f} {r['daily']:+8.0f} "
              f"{r['sharpe']:7.1f} {r['trades']:7d} {r['tpd']:5.1f} {r['wr']:5.0%} "
              f"| {r['d25']:+10.0f} {r['d26']:+10.0f}")

    # ===================================================================
    # KEY FINDINGS
    # ===================================================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Best overall
    best = max(all_strategies, key=lambda x: x['sharpe'])
    print(f"\n  Best Sharpe: {best['name']} (Sharpe={best['sharpe']:.1f}, {best['daily']:+.0f}/day)")

    best_daily = max(all_strategies, key=lambda x: x['daily'])
    print(f"  Best EUR/day: {best_daily['name']} ({best_daily['daily']:+.0f}/day, Sharpe={best_daily['sharpe']:.1f})")

    # Best in 2026
    best_26 = max(all_strategies, key=lambda x: x['d26'])
    print(f"  Best 2026: {best_26['name']} ({best_26['d26']:+.0f}/day in 2026)")

    # Confidence vs flat at threshold=3
    conf3 = next((r for r in all_strategies if r['name'] == 'Fixed>=3 (conf)'), None)
    flat3 = next((r for r in all_strategies if r['name'] == 'Fixed>=3 (flat 5MW)'), None)
    if conf3 and flat3:
        print(f"\n  Confidence vs Flat at threshold=3:")
        print(f"    Confidence: {conf3['daily']:+.0f}/day, Sharpe={conf3['sharpe']:.1f}")
        print(f"    Flat:       {flat3['daily']:+.0f}/day, Sharpe={flat3['sharpe']:.1f}")
        diff = conf3['daily'] - flat3['daily']
        print(f"    Confidence sizing {'helps' if diff > 0 else 'hurts'}: {diff:+.0f}/day difference")

    # ===================================================================
    # PLOTS
    # ===================================================================
    print("\n[*] Generating plots...")

    fig = plt.figure(figsize=(22, 20))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Plot 1: Fixed threshold sweep - daily PnL (all, 2025, 2026)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(FIXED_THRESHOLDS))
    daily_all = []
    daily_2025 = []
    daily_2026 = []
    for thr in FIXED_THRESHOLDS:
        trades = compute_pnl_for_trades(oos, thr, sizing='confidence')
        s = summarize_trades(trades)
        t25, t26 = split_period(trades)
        s25 = summarize_trades(t25)
        s26 = summarize_trades(t26)
        daily_all.append(s['daily_pnl'])
        daily_2025.append(s25['daily_pnl'])
        daily_2026.append(s26['daily_pnl'])

    w = 0.25
    ax.bar(x - w, daily_2025, w, label='2024-2025', color='steelblue', alpha=0.8)
    ax.bar(x, daily_2026, w, label='2026', color='coral', alpha=0.8)
    ax.bar(x + w, daily_all, w, label='All', color='gray', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in FIXED_THRESHOLDS])
    ax.set_xlabel('Threshold |pred| >=')
    ax.set_ylabel('EUR/day')
    ax.set_title('Fixed Threshold Sweep: EUR/day')
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)

    # Plot 2: Fixed threshold sweep - Sharpe
    ax = fig.add_subplot(gs[0, 1])
    sharpe_all = []
    sharpe_2025 = []
    sharpe_2026 = []
    for thr in FIXED_THRESHOLDS:
        trades = compute_pnl_for_trades(oos, thr, sizing='confidence')
        s = summarize_trades(trades)
        t25, t26 = split_period(trades)
        s25 = summarize_trades(t25)
        s26 = summarize_trades(t26)
        sharpe_all.append(s['sharpe'])
        sharpe_2025.append(s25['sharpe'])
        sharpe_2026.append(s26['sharpe'])

    ax.plot(FIXED_THRESHOLDS, sharpe_2025, 'b.-', label='2024-2025', markersize=8)
    ax.plot(FIXED_THRESHOLDS, sharpe_2026, 'r.-', label='2026', markersize=8)
    ax.plot(FIXED_THRESHOLDS, sharpe_all, 'k.--', label='All', markersize=8)
    ax.set_xlabel('Threshold |pred| >=')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Fixed Threshold Sweep: Sharpe')
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)

    # Plot 3: Confidence vs Flat comparison
    ax = fig.add_subplot(gs[0, 2])
    conf_daily = []
    flat_daily_vals = []
    for thr in FIXED_THRESHOLDS:
        tc = compute_pnl_for_trades(oos, thr, sizing='confidence')
        tf = compute_pnl_for_trades(oos, thr, sizing='flat')
        sc = summarize_trades(tc)
        sf = summarize_trades(tf)
        conf_daily.append(sc['daily_pnl'])
        flat_daily_vals.append(sf['daily_pnl'])

    ax.bar(x - 0.15, conf_daily, 0.3, label='Confidence', color='steelblue', alpha=0.8)
    ax.bar(x + 0.15, flat_daily_vals, 0.3, label='Flat 5MW', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in FIXED_THRESHOLDS])
    ax.set_xlabel('Threshold |pred| >=')
    ax.set_ylabel('EUR/day')
    ax.set_title('Confidence vs Flat Sizing')
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)

    # Plot 4: Trades per day vs threshold
    ax = fig.add_subplot(gs[1, 0])
    tpd_all = []
    wr_all = []
    for thr in FIXED_THRESHOLDS:
        trades = compute_pnl_for_trades(oos, thr, sizing='confidence')
        s = summarize_trades(trades)
        tpd_all.append(s['trades_per_day'])
        wr_all.append(s['win_rate'] * 100)

    ax.bar(x, tpd_all, color='steelblue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in FIXED_THRESHOLDS])
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Trades/day')
    ax.set_title('Trade Frequency vs Threshold')
    ax2 = ax.twinx()
    ax2.plot(x, wr_all, 'r.-', markersize=8, label='Win Rate %')
    ax2.set_ylabel('Win Rate (%)')
    ax2.legend(fontsize=8, loc='center right')

    # Plot 5: Strategy comparison bar chart (top 12 by Sharpe)
    ax = fig.add_subplot(gs[1, 1:])
    sorted_strats = sorted(all_strategies, key=lambda x: x['sharpe'], reverse=True)[:12]
    names = [r['name'] for r in sorted_strats]
    sharpes = [r['sharpe'] for r in sorted_strats]
    dailies = [r['daily'] for r in sorted_strats]

    colors_bar = ['green' if s > 0 else 'red' for s in sharpes]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, sharpes, color=colors_bar, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Sharpe Ratio')
    ax.set_title('Strategy Comparison by Sharpe (top 12)')
    for i, (sh, dl) in enumerate(zip(sharpes, dailies)):
        ax.annotate(f'{dl:+.0f}/d', xy=(sh, i), fontsize=7, va='center')

    # Plot 6: Monthly equity curves for key strategies
    ax = fig.add_subplot(gs[2, 0:2])

    key_strategies_config = [
        ('Fixed>=3 (conf)', 3, 'confidence', 'steelblue'),
        ('Fixed>=2 (conf)', 2, 'confidence', 'green'),
        ('Fixed>=5 (conf)', 5, 'confidence', 'orange'),
        ('Fixed>=3 (flat)', 3, 'flat', 'red'),
    ]

    for label, thr, sizing, color in key_strategies_config:
        trades = compute_pnl_for_trades(oos, thr, sizing=sizing)
        if len(trades) == 0:
            continue
        daily_ts = trades.groupby(trades.index.date)['pnl'].sum().cumsum()
        ax.plot(range(len(daily_ts)), daily_ts.values, label=label, color=color, lw=1.2, alpha=0.8)

    # Add direction-only
    dir_daily = dir_trades.groupby(dir_trades.index.date)['pnl'].sum().cumsum()
    ax.plot(range(len(dir_daily)), dir_daily.values, label='Direction-only', color='purple', lw=1.2, ls='--', alpha=0.8)

    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Cumulative P&L (EUR)')
    ax.set_title('Equity Curves: Key Strategies')
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0, color='gray', ls='--', alpha=0.3)

    # Plot 7: 2026 performance specifically
    ax = fig.add_subplot(gs[2, 2])
    strats_26 = sorted(all_strategies, key=lambda x: x['d26'], reverse=True)[:10]
    names_26 = [r['name'] for r in strats_26]
    daily_26 = [r['d26'] for r in strats_26]
    colors_26 = ['green' if d > 0 else 'red' for d in daily_26]
    y_pos_26 = np.arange(len(names_26))
    ax.barh(y_pos_26, daily_26, color=colors_26, alpha=0.7)
    ax.set_yticks(y_pos_26)
    ax.set_yticklabels(names_26, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('EUR/day')
    ax.set_title('2026 Performance (top 10)')
    ax.axvline(0, color='gray', ls='--', alpha=0.5)

    fig.savefig(PLOT_DIR / "01_threshold_analysis.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"\n[+] Saved: {PLOT_DIR / '01_threshold_analysis.png'}")

    print("\n[+] Threshold analysis complete.")


if __name__ == "__main__":
    main()
