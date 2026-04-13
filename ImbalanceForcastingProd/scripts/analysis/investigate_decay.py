"""
Investigate Temporal Performance Decay of 52-Feature Spread Model
=================================================================

The walk-forward model averages +757/day over 16 months, but degrades
from ~+1000/day in 2025 to ~+400/day in 2026. This script investigates:

1. Monthly feature importance stability (overfitting? signal fading?)
2. Prediction calibration drift (confidence, direction accuracy)
3. Feature distribution shift (KS tests train vs test)
4. Market regime analysis (spread distribution, volatility changes)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
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
PLOT_DIR = BASE_DIR / "plots" / "eda" / "decay_investigation"
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

KEY_FEATURES = ['da_price', 'idm_vwap_lag', 'proxy_rmean16', 'temp_forecast_da',
                'nowcast_momentum_h2h3']

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


def main():
    print("=" * 70)
    print("INVESTIGATION: TEMPORAL PERFORMANCE DECAY")
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

    print(f"[+] Base data: {len(df_base)} rows, {len(spread_features)} features")

    # ===================================================================
    # WALK-FORWARD: collect per-fold diagnostics
    # ===================================================================
    importance_records = []
    calibration_records = []
    dist_shift_records = []

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

        # Train model
        model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        model.fit(train[spread_features].values, train['spread_target'].values)

        # --- 1. Feature importance ---
        raw_imp = model.feature_importances_
        imp_pct = 100.0 * raw_imp / raw_imp.sum()
        imp_dict = {'fold': fold_label}
        for fname, val in zip(spread_features, imp_pct):
            imp_dict[fname] = val
        importance_records.append(imp_dict)

        # --- 2. Prediction calibration ---
        test['pred'] = model.predict(test[spread_features].values)
        surplus = test['pred'] <= -3
        deficit = test['pred'] >= 3
        active = test[surplus | deficit].copy()

        s = surplus.reindex(active.index, fill_value=False)
        d = deficit.reindex(active.index, fill_value=False)
        active['pnl'] = 0.0
        if s.sum() > 0:
            active.loc[s, 'pnl'] = (active.loc[s, 'exec_bid'] - active.loc[s, 'imb_settlement_price']) * ENERGY
        if d.sum() > 0:
            active.loc[d, 'pnl'] = (active.loc[d, 'imb_settlement_price'] - active.loc[d, 'exec_ask']) * ENERGY

        nd = test.index.normalize().nunique()
        dir_acc = (np.sign(test['pred']) == np.sign(test['spread_target'])).mean()
        dir_acc_active = np.nan
        if len(active) > 0:
            dir_acc_active = (np.sign(active['pred']) == np.sign(active['spread_target'])).mean()

        total_pnl = active['pnl'].sum() if len(active) > 0 else 0
        daily_pnl = total_pnl / nd if nd > 0 else 0

        cal = {
            'fold': fold_label,
            'n_test': len(test),
            'n_trades': len(active),
            'trades_per_day': len(active) / nd if nd > 0 else 0,
            'pred_mean': test['pred'].mean(),
            'pred_std': test['pred'].std(),
            'pred_abs_mean': test['pred'].abs().mean(),
            'frac_trades': (test['pred'].abs() > 3).mean(),
            'dir_accuracy_all': dir_acc,
            'dir_accuracy_active': dir_acc_active,
            'spread_target_mean': test['spread_target'].mean(),
            'spread_target_std': test['spread_target'].std(),
            'spread_target_abs_mean': test['spread_target'].abs().mean(),
            'total_pnl': total_pnl,
            'daily_pnl': daily_pnl,
            'win_rate': (active['pnl'] > 0).mean() if len(active) > 0 else np.nan,
        }
        calibration_records.append(cal)

        print(f"  Pred: mean={cal['pred_mean']:+.2f}, std={cal['pred_std']:.2f}, "
              f"frac_active={cal['frac_trades']:.1%}")
        print(f"  Dir accuracy: all={cal['dir_accuracy_all']:.1%}, "
              f"active={cal['dir_accuracy_active']:.1%}")
        print(f"  P&L: {total_pnl:+,.0f} EUR ({daily_pnl:+.0f}/day), "
              f"win={cal['win_rate']:.0%}" if not np.isnan(cal['win_rate']) else "  No trades")

        # --- 3. Feature distribution shift ---
        for feat in KEY_FEATURES:
            if feat not in spread_features:
                continue
            tr_vals = train[feat].dropna().values
            te_vals = test[feat].dropna().values
            if len(tr_vals) < 10 or len(te_vals) < 10:
                continue
            ks_stat, ks_pval = stats.ks_2samp(tr_vals, te_vals)
            dist_shift_records.append({
                'fold': fold_label,
                'feature': feat,
                'train_mean': tr_vals.mean(),
                'train_std': tr_vals.std(),
                'test_mean': te_vals.mean(),
                'test_std': te_vals.std(),
                'mean_shift': te_vals.mean() - tr_vals.mean(),
                'std_ratio': te_vals.std() / tr_vals.std() if tr_vals.std() > 0 else np.nan,
                'ks_stat': ks_stat,
                'ks_pval': ks_pval,
                'significant': ks_pval < 0.01,
            })

    # ===================================================================
    # ANALYSIS 1: Feature Importance Stability
    # ===================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: FEATURE IMPORTANCE STABILITY")
    print("=" * 70)

    imp_df = pd.DataFrame(importance_records).set_index('fold')
    imp_mean = imp_df.mean().sort_values(ascending=False)
    imp_std = imp_df.std()
    imp_cv = (imp_std / imp_mean).replace([np.inf, -np.inf], np.nan)

    # Split into 2025 and 2026 folds
    folds_2025 = [f for f in imp_df.index if f.startswith('2024') or f.startswith('2025')]
    folds_2026 = [f for f in imp_df.index if f.startswith('2026')]

    imp_2025 = imp_df.loc[folds_2025].mean()
    imp_2026 = imp_df.loc[folds_2026].mean()
    imp_change = imp_2026 - imp_2025

    print("\n--- Top 15 features by average importance ---")
    for feat in imp_mean.head(15).index:
        cv_val = imp_cv.get(feat, np.nan)
        chg = imp_change.get(feat, np.nan)
        print(f"  {feat:35s} mean={imp_mean[feat]:5.2f}%  std={imp_std[feat]:5.2f}%  "
              f"CV={cv_val:.2f}  2025={imp_2025[feat]:5.2f}%  2026={imp_2026[feat]:5.2f}%  "
              f"shift={chg:+.2f}%")

    print("\n--- Features with BIGGEST increase in 2026 (potential overfitting to noise) ---")
    for feat in imp_change.sort_values(ascending=False).head(10).index:
        print(f"  {feat:35s} 2025={imp_2025[feat]:5.2f}%  2026={imp_2026[feat]:5.2f}%  "
              f"shift={imp_change[feat]:+.2f}%")

    print("\n--- Features with BIGGEST decrease in 2026 (signal fading?) ---")
    for feat in imp_change.sort_values(ascending=True).head(10).index:
        print(f"  {feat:35s} 2025={imp_2025[feat]:5.2f}%  2026={imp_2026[feat]:5.2f}%  "
              f"shift={imp_change[feat]:+.2f}%")

    print("\n--- Most VOLATILE features (high CV = unstable importance) ---")
    for feat in imp_cv.sort_values(ascending=False).head(10).index:
        print(f"  {feat:35s} mean={imp_mean[feat]:5.2f}%  CV={imp_cv[feat]:.2f}")

    # ===================================================================
    # ANALYSIS 2: Prediction Calibration Drift
    # ===================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: PREDICTION CALIBRATION DRIFT")
    print("=" * 70)

    cal_df = pd.DataFrame(calibration_records).set_index('fold')
    print("\n--- Monthly calibration table ---")
    print(f"{'Fold':>10s} {'PredMean':>9s} {'PredStd':>8s} {'PredAbs':>8s} "
          f"{'FracAct':>8s} {'DirAll':>7s} {'DirAct':>7s} "
          f"{'SprdMean':>9s} {'SprdStd':>8s} {'SprdAbs':>8s} "
          f"{'EUR/day':>8s} {'WinRate':>8s}")

    for fold, r in cal_df.iterrows():
        print(f"{fold:>10s} {r['pred_mean']:+9.2f} {r['pred_std']:8.2f} {r['pred_abs_mean']:8.2f} "
              f"{r['frac_trades']:8.1%} {r['dir_accuracy_all']:7.1%} "
              f"{r['dir_accuracy_active']:7.1%} "
              f"{r['spread_target_mean']:+9.2f} {r['spread_target_std']:8.2f} "
              f"{r['spread_target_abs_mean']:8.2f} "
              f"{r['daily_pnl']:+8.0f} {r['win_rate']:8.1%}")

    # Period summaries
    cal_2025 = cal_df.loc[[f for f in cal_df.index if f.startswith('2024') or f.startswith('2025')]]
    cal_2026 = cal_df.loc[[f for f in cal_df.index if f.startswith('2026')]]

    print("\n--- Period comparison ---")
    for label, sub in [('2024-2025', cal_2025), ('2026', cal_2026)]:
        if len(sub) == 0:
            continue
        print(f"\n  {label} ({len(sub)} folds):")
        print(f"    Avg prediction magnitude: {sub['pred_abs_mean'].mean():.2f}")
        print(f"    Avg prediction std:       {sub['pred_std'].mean():.2f}")
        print(f"    Avg fraction active:      {sub['frac_trades'].mean():.1%}")
        print(f"    Avg direction accuracy:   {sub['dir_accuracy_all'].mean():.1%} (all), "
              f"{sub['dir_accuracy_active'].mean():.1%} (active)")
        print(f"    Avg spread target |mean|: {sub['spread_target_abs_mean'].mean():.2f}")
        print(f"    Avg spread target std:    {sub['spread_target_std'].mean():.2f}")
        print(f"    Avg daily P&L:            {sub['daily_pnl'].mean():+.0f} EUR")
        print(f"    Avg win rate:             {sub['win_rate'].mean():.1%}")

    # ===================================================================
    # ANALYSIS 3: Feature Distribution Shift
    # ===================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: FEATURE DISTRIBUTION SHIFT (KS TESTS)")
    print("=" * 70)

    dist_df = pd.DataFrame(dist_shift_records)
    if len(dist_df) > 0:
        for feat in KEY_FEATURES:
            sub = dist_df[dist_df['feature'] == feat].copy()
            if len(sub) == 0:
                continue
            print(f"\n  Feature: {feat}")
            print(f"  {'Fold':>10s} {'TrainMean':>10s} {'TestMean':>10s} {'Shift':>8s} "
                  f"{'StdRatio':>9s} {'KS-stat':>8s} {'KS-pval':>10s} {'Sig?':>5s}")
            for _, r in sub.iterrows():
                sig = "***" if r['ks_pval'] < 0.001 else ("**" if r['ks_pval'] < 0.01 else
                       ("*" if r['ks_pval'] < 0.05 else ""))
                print(f"  {r['fold']:>10s} {r['train_mean']:10.2f} {r['test_mean']:10.2f} "
                      f"{r['mean_shift']:+8.2f} {r['std_ratio']:9.3f} "
                      f"{r['ks_stat']:8.3f} {r['ks_pval']:10.4f} {sig:>5s}")

        # Summarize: which features shifted most in 2026?
        dist_2026 = dist_df[dist_df['fold'].str.startswith('2026')]
        if len(dist_2026) > 0:
            print("\n--- 2026 folds: average KS stat per feature ---")
            ks_summary = dist_2026.groupby('feature')['ks_stat'].mean().sort_values(ascending=False)
            for feat, ks in ks_summary.items():
                n_sig = (dist_2026[dist_2026['feature'] == feat]['significant']).sum()
                n_tot = len(dist_2026[dist_2026['feature'] == feat])
                print(f"    {feat:35s} avg KS={ks:.3f}  significant: {n_sig}/{n_tot} folds")

    # ===================================================================
    # ANALYSIS 4: Market Regime Analysis
    # ===================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: MARKET REGIME ANALYSIS")
    print("=" * 70)

    # Monthly spread target statistics
    df_valid = df_base[df_base['spread_target'].notna() &
                       df_base['exec_bid'].notna()].copy()
    df_valid['month'] = df_valid.index.to_period('M')

    print("\n--- Monthly spread target statistics ---")
    print(f"{'Month':>10s} {'Mean':>8s} {'Std':>8s} {'AbsMean':>8s} {'Median':>8s} "
          f"{'Pct>3':>7s} {'Pct>5':>7s} {'Settle_Std':>11s} {'N':>6s}")

    monthly_regime = df_valid.groupby('month').agg(
        spread_mean=('spread_target', 'mean'),
        spread_std=('spread_target', 'std'),
        spread_abs_mean=('spread_target', lambda x: x.abs().mean()),
        spread_median=('spread_target', 'median'),
        frac_gt3=('spread_target', lambda x: (x.abs() > 3).mean()),
        frac_gt5=('spread_target', lambda x: (x.abs() > 5).mean()),
        settle_std=('imb_settlement_price', 'std'),
        n=('spread_target', 'count'),
    )

    for month, r in monthly_regime.iterrows():
        print(f"  {str(month):>10s} {r['spread_mean']:+8.2f} {r['spread_std']:8.2f} "
              f"{r['spread_abs_mean']:8.2f} {r['spread_median']:+8.2f} "
              f"{r['frac_gt3']:7.1%} {r['frac_gt5']:7.1%} "
              f"{r['settle_std']:11.1f} {int(r['n']):6d}")

    # Period comparison
    months_2025 = [m for m in monthly_regime.index if m.year <= 2025]
    months_2026 = [m for m in monthly_regime.index if m.year == 2026]

    if len(months_2025) > 0 and len(months_2026) > 0:
        r25 = monthly_regime.loc[months_2025]
        r26 = monthly_regime.loc[months_2026]
        print("\n--- Period comparison ---")
        for label, sub in [('2024-2025', r25), ('2026', r26)]:
            wn = sub['n']  # weight by count
            w = wn / wn.sum()
            print(f"\n  {label}:")
            print(f"    Weighted mean spread:   {(sub['spread_mean'] * w).sum():+.2f}")
            print(f"    Weighted std spread:    {(sub['spread_std'] * w).sum():.2f}")
            print(f"    Weighted |spread|:      {(sub['spread_abs_mean'] * w).sum():.2f}")
            print(f"    Weighted frac |spr|>3:  {(sub['frac_gt3'] * w).sum():.1%}")
            print(f"    Weighted frac |spr|>5:  {(sub['frac_gt5'] * w).sum():.1%}")
            print(f"    Weighted settle vol:    {(sub['settle_std'] * w).sum():.1f}")

    # ===================================================================
    # PLOTS
    # ===================================================================
    print("\n[*] Generating plots...")

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)

    # --- Plot 1: Feature importance heatmap (top 20) ---
    ax = fig.add_subplot(gs[0, :])
    top20 = imp_mean.head(20).index.tolist()
    imp_top = imp_df[top20]
    im = ax.imshow(imp_top.T.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(range(len(imp_top.index)))
    ax.set_xticklabels(imp_top.index, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20, fontsize=8)
    ax.set_title('Feature Importance (%) by Walk-Forward Fold - Top 20', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.6, label='Importance %')

    # --- Plot 2: Importance shift 2025 vs 2026 ---
    ax = fig.add_subplot(gs[1, 0])
    top_shift = imp_change.abs().sort_values(ascending=False).head(15)
    colors = ['red' if imp_change[f] > 0 else 'blue' for f in top_shift.index]
    ax.barh(range(len(top_shift)), [imp_change[f] for f in top_shift.index], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_shift)))
    ax.set_yticklabels(top_shift.index, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', ls='-', alpha=0.5)
    ax.set_xlabel('Importance change (2026 - 2025) in %pts')
    ax.set_title('Feature Importance Shift: 2025 -> 2026')

    # --- Plot 3: Direction accuracy + daily PnL ---
    ax = fig.add_subplot(gs[1, 1])
    folds = cal_df.index.tolist()
    ax.bar(range(len(folds)), cal_df['daily_pnl'], color=['green' if v > 0 else 'red'
           for v in cal_df['daily_pnl']], alpha=0.7)
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels(folds, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('EUR/day')
    ax.set_title('Daily P&L by Fold')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(range(len(folds)), cal_df['dir_accuracy_all'] * 100, 'b.-', label='Dir Acc (all)', alpha=0.7)
    ax2.plot(range(len(folds)), cal_df['dir_accuracy_active'] * 100, 'r.-', label='Dir Acc (active)', alpha=0.7)
    ax2.set_ylabel('Direction Accuracy (%)')
    ax2.legend(fontsize=7, loc='lower right')

    # --- Plot 4: Prediction confidence vs spread magnitude ---
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(range(len(folds)), cal_df['pred_abs_mean'], 'b.-', label='|Pred| mean')
    ax.plot(range(len(folds)), cal_df['spread_target_abs_mean'], 'r.-', label='|Spread| mean')
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels(folds, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=8)
    ax.set_ylabel('EUR/MWh')
    ax.set_title('Model Confidence vs Actual Spread Magnitude')

    # --- Plot 5: Fraction of active trades ---
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(range(len(folds)), cal_df['frac_trades'] * 100, 'g.-', label='Frac |pred|>3')
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels(folds, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('% of hours traded')
    ax.set_title('Trade Frequency Over Time')
    ax.legend(fontsize=8)

    # --- Plot 6: Spread target distribution over time ---
    ax = fig.add_subplot(gs[2, 1])
    valid_months = monthly_regime.index.astype(str).tolist()
    ax.plot(range(len(valid_months)), monthly_regime['spread_std'], 'b.-', label='Spread Std')
    ax.plot(range(len(valid_months)), monthly_regime['spread_abs_mean'], 'r.-', label='|Spread| Mean')
    ax.set_xticks(range(len(valid_months)))
    ax.set_xticklabels(valid_months, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('EUR/MWh')
    ax.set_title('Spread Target Distribution Over Time')
    ax.legend(fontsize=8)

    # --- Plot 7: Settlement price volatility ---
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(range(len(valid_months)), monthly_regime['settle_std'], 'k.-')
    ax.set_xticks(range(len(valid_months)))
    ax.set_xticklabels(valid_months, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Std (EUR/MWh)')
    ax.set_title('Settlement Price Volatility by Month')

    # --- Plot 8-9: KS test results for key features ---
    if len(dist_df) > 0:
        for i, feat in enumerate(KEY_FEATURES[:2]):
            ax = fig.add_subplot(gs[3, i])
            sub = dist_df[dist_df['feature'] == feat]
            if len(sub) == 0:
                continue
            ax.bar(range(len(sub)), sub['ks_stat'], color=['red' if s else 'steelblue'
                   for s in sub['significant']], alpha=0.7)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub['fold'].tolist(), rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('KS Statistic')
            ax.set_title(f'Distribution Shift: {feat}')
            ax.axhline(0.1, color='orange', ls='--', alpha=0.5, label='KS=0.1')
            ax.legend(fontsize=7)

        # Plot 10: Mean shift of key features
        ax = fig.add_subplot(gs[3, 2])
        max_len = 0
        for feat in KEY_FEATURES:
            sub = dist_df[dist_df['feature'] == feat]
            if len(sub) == 0:
                continue
            ax.plot(range(len(sub)), sub['mean_shift'].values, '.-', label=feat, alpha=0.7)
            max_len = max(max_len, len(sub))
        fold_labels = [pd.Timestamp(f[1]).strftime('%Y-%m') for f in FOLDS]
        ax.set_xticks(range(max_len))
        ax.set_xticklabels(fold_labels[:max_len], rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Test Mean - Train Mean')
        ax.set_title('Feature Mean Shift (Test vs Train)')
        ax.legend(fontsize=6, ncol=2)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)

    fig.savefig(PLOT_DIR / "01_decay_investigation.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"\n[+] Saved: {PLOT_DIR / '01_decay_investigation.png'}")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)

    # Most important diagnostic: has spread target narrowed?
    if len(months_2025) > 0 and len(months_2026) > 0:
        abs_25 = (r25['spread_abs_mean'] * r25['n']).sum() / r25['n'].sum()
        abs_26 = (r26['spread_abs_mean'] * r26['n']).sum() / r26['n'].sum()
        std_25 = (r25['spread_std'] * r25['n']).sum() / r25['n'].sum()
        std_26 = (r26['spread_std'] * r26['n']).sum() / r26['n'].sum()
        print(f"\n  1. Spread target magnitude: 2025={abs_25:.2f}, 2026={abs_26:.2f} "
              f"({100*(abs_26/abs_25 - 1):+.0f}%)")
        print(f"     Spread target volatility: 2025={std_25:.2f}, 2026={std_26:.2f} "
              f"({100*(std_26/std_25 - 1):+.0f}%)")

    if len(cal_2025) > 0 and len(cal_2026) > 0:
        print(f"\n  2. Direction accuracy: 2025={cal_2025['dir_accuracy_all'].mean():.1%}, "
              f"2026={cal_2026['dir_accuracy_all'].mean():.1%}")
        print(f"     Prediction magnitude: 2025={cal_2025['pred_abs_mean'].mean():.2f}, "
              f"2026={cal_2026['pred_abs_mean'].mean():.2f}")
        print(f"     Trade frequency: 2025={cal_2025['frac_trades'].mean():.1%}, "
              f"2026={cal_2026['frac_trades'].mean():.1%}")

    # Feature stability
    biggest_increase = imp_change.sort_values(ascending=False).head(3)
    biggest_decrease = imp_change.sort_values(ascending=True).head(3)
    print(f"\n  3. Features gaining importance in 2026: "
          f"{', '.join(biggest_increase.index.tolist())}")
    print(f"     Features losing importance in 2026: "
          f"{', '.join(biggest_decrease.index.tolist())}")

    print("\n[+] Investigation complete.")


if __name__ == "__main__":
    main()
