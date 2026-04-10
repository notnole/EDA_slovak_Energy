"""
Validate Selected Features: Full 3-Stage Stacked Pipeline
==========================================================

Runs the complete stacked model pipeline (Stage 1 load nowcast OOF ->
Stage 2 imbalance -> Stage 3 spread) with BOTH feature sets:

  A) Full feature set (113 features) — baseline
  B) Selected feature set (66 features) — from feature_selection_spread.py

Compares trading metrics side-by-side on the same test period (Feb-Mar 2026).
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent  # ImbalanceForcastingProd/
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
REPO_ROOT = BASE_DIR.parent

sys.path.insert(0, str(BASE_DIR / "scripts" / "training"))
import train_multi_lead as tml
# Fix path resolution after restructure
tml.REPO_ROOT = REPO_ROOT
tml.DATA_DIR = DATA_DIR
tml.PLOT_DIR = PLOT_DIR
tml.MODEL_DIR = BASE_DIR / "models"
from train_multi_lead import load_all_data, build_features

# Load selected features
sys.path.insert(0, str(DATA_DIR / "feature_selection"))
from recommended_features import SELECTED_FEATURES

plt.rcParams.update({
    "figure.figsize": (16, 10), "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})

LEAD = 8

FOLDS = [
    # (train_end, pred_start, pred_end)
    ('2025-07-01', '2025-07-01', '2025-10-01'),
    ('2025-10-01', '2025-10-01', '2026-01-01'),
    ('2026-01-01', '2026-01-01', '2026-02-01'),
    ('2026-02-01', '2026-02-01', '2026-03-01'),   # test
    ('2026-03-01', '2026-03-01', '2026-04-01'),   # test
]

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                  reg_lambda=1.0, n_estimators=600, verbose=-1)

PRED_THRESHOLD = 3


def run_stacked_pipeline(df_base, feature_cols, label="full"):
    """Run complete 3-stage stacked pipeline. Returns test predictions DataFrame."""
    print(f"\n  --- Running stacked pipeline: {label} ({len(feature_cols)} features) ---")

    all_stage2_oof = []
    all_stage3_oof = []

    for fi, (train_end, pred_start, pred_end) in enumerate(FOLDS):
        df = df_base.copy()

        # Join accumulated Stage 2 OOF from prior folds
        if all_stage2_oof:
            prior_s2 = pd.concat(all_stage2_oof)
            prior_s2 = prior_s2[~prior_s2.index.duplicated(keep='last')]
            df = df.join(prior_s2[['stk_imb_pred', 'stk_imb_pred_abs', 'stk_imb_direction']], how='left')
        else:
            df['stk_imb_pred'] = np.nan
            df['stk_imb_pred_abs'] = np.nan
            df['stk_imb_direction'] = np.nan

        train_mask = df.index < train_end
        pred_mask = (df.index >= pred_start) & (df.index < pred_end)
        train = df[train_mask].dropna(subset=['target', f'proxy_lag{LEAD+1}'])
        pred_data = df[pred_mask].dropna(subset=[f'proxy_lag{LEAD+1}'])

        if len(train) == 0 or len(pred_data) == 0:
            continue

        # Stage 2: imbalance model (always uses base feature_cols)
        m_imb = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        m_imb.fit(train[feature_cols].values, train['target'].values)

        pred_data = pred_data.copy()
        pred_data['stk_imb_pred'] = m_imb.predict(pred_data[feature_cols].values)
        pred_data['stk_imb_pred_abs'] = pred_data['stk_imb_pred'].abs()
        pred_data['stk_imb_direction'] = np.sign(pred_data['stk_imb_pred'])

        s2_oof = pred_data[['stk_imb_pred', 'stk_imb_pred_abs', 'stk_imb_direction']].copy()
        all_stage2_oof.append(s2_oof)

        # Stage 3: spread model with stacking features
        s3_features = feature_cols + ['stk_imb_pred', 'stk_imb_pred_abs', 'stk_imb_direction']

        train_with_s2 = train.copy()
        s3_train = train_with_s2.dropna(subset=['spread_target'])
        s3_train = s3_train[s3_train['imb_settlement_price'].abs() <= 5000]

        if len(s3_train) < 100:
            # Fallback without stacking
            m_sp = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
            m_sp.fit(s3_train[feature_cols].values, s3_train['spread_target'].values)
            pred_data['stk_spread_pred'] = m_sp.predict(pred_data[feature_cols].values)
        else:
            m_sp = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
            m_sp.fit(s3_train[s3_features].values, s3_train['spread_target'].values)
            pred_data['stk_spread_pred'] = m_sp.predict(pred_data[s3_features].values)

        s3_oof = pred_data[['stk_spread_pred', 'stk_imb_pred', 'target', 'spread_target',
                             'exec_bid', 'exec_ask', 'exec_spread', 'imb_settlement_price']].copy()
        all_stage3_oof.append(s3_oof)

        if fi >= 3:
            nz = pred_data['spread_target'].abs() > 0.1
            valid = pred_data['spread_target'].notna() & nz
            if valid.sum() > 0:
                dir_acc = (np.sign(pred_data.loc[valid, 'stk_spread_pred']) ==
                           np.sign(pred_data.loc[valid, 'spread_target'])).mean()
                print(f"    Fold {fi+1} ({pred_start[:7]}): dir_acc={dir_acc:.1%}, "
                      f"{len(pred_data)} periods")

    # Collect test predictions (folds 4-5)
    test_folds = [s3 for fi, s3 in enumerate(all_stage3_oof) if fi >= 3]
    if not test_folds:
        print("  [!] No test fold predictions")
        return pd.DataFrame()

    test_df = pd.concat(test_folds)
    test_df = test_df[test_df['exec_spread'].notna() & (test_df['exec_spread'] <= 10)]
    return test_df


def backtest(test_df, pred_col, label, threshold=PRED_THRESHOLD):
    """Run trading backtest, return metrics dict."""
    t = test_df.copy()
    surplus = t[pred_col] <= -threshold
    deficit = t[pred_col] >= threshold
    sub = t[surplus | deficit].copy()

    if len(sub) < 30:
        return None

    sub['size'] = sub[pred_col].abs().clip(upper=5)
    s = surplus.reindex(sub.index, fill_value=False)
    d = deficit.reindex(sub.index, fill_value=False)
    sub['pnl'] = 0.0
    sub.loc[s, 'pnl'] = (sub.loc[s, 'exec_bid'] - sub.loc[s, 'imb_settlement_price']) * sub.loc[s, 'size'] / 4
    sub.loc[d, 'pnl'] = (sub.loc[d, 'imb_settlement_price'] - sub.loc[d, 'exec_ask']) * sub.loc[d, 'size'] / 4

    daily = sub.groupby(sub.index.date)['pnl'].sum()
    nd = len(daily)
    total = sub['pnl'].sum()
    wr = (sub['pnl'] > 0).mean()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0

    # Max drawdown
    cumulative = daily.cumsum()
    running_max = cumulative.cummax()
    max_dd = (cumulative - running_max).min()

    # Profitable days
    prof_days = (daily > 0).mean()

    # Weekly
    daily_dt = daily.copy()
    daily_dt.index = pd.to_datetime(daily_dt.index)
    weekly = daily_dt.resample('W').sum()
    losing_weeks = (weekly < 0).sum()
    worst_week = weekly.min()

    return {
        'label': label,
        'total_pnl': total,
        'pnl_per_day': total / max(nd, 1),
        'sharpe': sharpe,
        'n_trades': len(sub),
        'win_rate': wr,
        'prof_days': prof_days,
        'max_dd': max_dd,
        'n_days': nd,
        'losing_weeks': losing_weeks,
        'worst_week': worst_week,
        'daily_pnl': daily,
    }


def main():
    print("=" * 70)
    print("STACKED MODEL VALIDATION: Full vs Selected Features")
    print("=" * 70)

    # --- Load data ---
    print("\n[*] Loading data...")
    data = load_all_data()

    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, all_feature_cols = build_features(data, LEAD)

    # Join Stage 1 OOF (load nowcast)
    stage1_oof = pd.read_csv(
        REPO_ROOT / "LoadAnalysis" / "nowcast_5h" / "tuning" / "oos_predictions" / "h2_oos_predictions.csv",
        parse_dates=['datetime'], index_col='datetime')
    stage1_oof = stage1_oof[~stage1_oof.index.duplicated(keep='last')]
    stage1_oof = stage1_oof[['predicted_error']].rename(columns={'predicted_error': 'stk_load_nowcast'})
    stage1_oof_15 = stage1_oof.resample('15min').ffill()

    # Execution prices
    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv", parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'spread', 'mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']

    # Settlement prices
    mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                       parse_dates=['timestamp_hour'], index_col='timestamp_hour')
    mkt = mkt[~mkt.index.duplicated(keep='last')]

    # Assemble
    df_base = df_base.join(stage1_oof_15, how='left')
    df_base = df_base.join(ob_120, how='left')
    df_base['hour_ts'] = df_base.index.floor('h')
    df_base = df_base.join(mkt[['imb_settlement_price']], on='hour_ts', how='left')
    df_base['spread_target'] = df_base['imb_settlement_price'] - df_base['exec_mid']

    print(f"[+] Base: {len(all_feature_cols)} features, {len(df_base)} rows")

    # Verify selected features are all present
    selected = [f for f in SELECTED_FEATURES if f in all_feature_cols]
    missing = [f for f in SELECTED_FEATURES if f not in all_feature_cols]
    if missing:
        print(f"[!] Warning: {len(missing)} selected features not found: {missing}")
    print(f"[+] Selected features: {len(selected)} (of {len(SELECTED_FEATURES)} requested)")

    # --- Run both pipelines ---
    print("\n" + "=" * 70)
    print("RUNNING STACKED PIPELINES")
    print("=" * 70)

    test_full = run_stacked_pipeline(df_base, all_feature_cols, label=f"full ({len(all_feature_cols)})")
    test_sel = run_stacked_pipeline(df_base, selected, label=f"selected ({len(selected)})")

    # --- Compare ---
    print("\n" + "=" * 70)
    print("TRADING BACKTEST COMPARISON (Feb-Mar 2026)")
    print("=" * 70)

    results = []
    for test_df, label in [(test_full, 'Full (113 feat)'), (test_sel, f'Selected ({len(selected)} feat)')]:
        if len(test_df) == 0:
            continue

        print(f"\n  --- {label} ---")
        for thresh in [3, 5]:
            r = backtest(test_df, 'stk_spread_pred', f'{label} |pred|>={thresh}', threshold=thresh)
            if r:
                results.append(r)
                print(f"    |pred|>={thresh}: {r['n_trades']:4d} trades, "
                      f"win={r['win_rate']:.0%}, "
                      f"P&L={r['total_pnl']:>+8,.0f} ({r['pnl_per_day']:>+4.0f}/d), "
                      f"Sharpe={r['sharpe']:.1f}, "
                      f"maxDD={r['max_dd']:>+,.0f}, "
                      f"prof_days={r['prof_days']:.0%}")

    # --- Detailed comparison table ---
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON (|pred| >= 3)")
    print("=" * 70)

    r_full = backtest(test_full, 'stk_spread_pred', 'full', threshold=3)
    r_sel = backtest(test_sel, 'stk_spread_pred', 'selected', threshold=3)

    if r_full and r_sel:
        metrics = [
            ('Total P&L (EUR)', 'total_pnl', '+,.0f'),
            ('P&L per day (EUR)', 'pnl_per_day', '+,.0f'),
            ('Sharpe ratio', 'sharpe', '.1f'),
            ('Trades', 'n_trades', ',d'),
            ('Win rate', 'win_rate', '.1%'),
            ('Profitable days', 'prof_days', '.0%'),
            ('Max drawdown (EUR)', 'max_dd', '+,.0f'),
            ('Losing weeks', 'losing_weeks', 'd'),
            ('Worst week (EUR)', 'worst_week', '+,.0f'),
        ]

        print(f"\n  {'Metric':<25s}  {'Full (113)':>14s}  {'Selected ('+str(len(selected))+')':>14s}  {'Delta':>10s}")
        print(f"  {'-'*25}  {'-'*14}  {'-'*14}  {'-'*10}")

        for name, key, fmt in metrics:
            v1 = r_full[key]
            v2 = r_sel[key]
            s1 = format(v1, fmt)
            s2 = format(v2, fmt)
            if isinstance(v1, (int, float)) and abs(v1) > 0:
                delta = (v2 - v1) / abs(v1) * 100
                ds = f"{delta:+.1f}%"
            else:
                ds = ""
            print(f"  {name:<25s}  {s1:>14s}  {s2:>14s}  {ds:>10s}")

        # Monthly breakdown
        print(f"\n  --- Monthly P&L breakdown ---")
        for label_name, r in [('Full', r_full), ('Selected', r_sel)]:
            daily = r['daily_pnl'].copy()
            daily.index = pd.to_datetime(daily.index)
            monthly = daily.resample('ME').agg(['sum', 'mean', 'std', 'count'])
            print(f"\n  {label_name}:")
            for idx, row in monthly.iterrows():
                m_sharpe = row['mean'] / row['std'] * np.sqrt(252) if row['std'] > 0 else 0
                print(f"    {idx.strftime('%Y-%m')}: "
                      f"total={row['sum']:>+8,.0f}, "
                      f"avg={row['mean']:>+6.0f}/d, "
                      f"Sharpe={m_sharpe:.1f}, "
                      f"days={int(row['count'])}")

    # --- Plot comparison ---
    if r_full and r_sel:
        _plot_comparison(r_full, r_sel, len(selected))

    # --- Save predictions ---
    out_full = test_full[['target', 'spread_target', 'stk_imb_pred', 'stk_spread_pred',
                           'exec_bid', 'exec_ask', 'imb_settlement_price']].copy()
    out_full.columns = [f'{c}_full' if c == 'stk_spread_pred' else c for c in out_full.columns]
    out_sel = test_sel[['stk_spread_pred']].copy()
    out_sel.columns = ['stk_spread_pred_selected']
    combined = out_full.join(out_sel, how='outer')
    out_path = DATA_DIR / "feature_selection" / "stacked_validation_predictions.csv"
    combined.to_csv(out_path)
    print(f"\n[+] Saved predictions: {out_path}")
    print("\n[+] Validation complete!")


def _plot_comparison(r_full, r_sel, n_selected):
    """Plot cumulative P&L comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Cumulative P&L
    ax = axes[0]
    cum_full = r_full['daily_pnl'].cumsum()
    cum_sel = r_sel['daily_pnl'].cumsum()

    ax.plot(cum_full.index, cum_full.values, 'b-', linewidth=2, label=f'Full (113 features)')
    ax.plot(cum_sel.index, cum_sel.values, 'r-', linewidth=2, label=f'Selected ({n_selected} features)')
    ax.fill_between(cum_full.index, cum_full.values, cum_sel.values, alpha=0.1, color='gray')
    ax.set_ylabel('Cumulative P&L (EUR)')
    ax.set_title('Stacked Model: Full vs Selected Features — Cumulative P&L')
    ax.legend(fontsize=12)

    # Daily P&L bars
    ax = axes[1]
    x = np.arange(len(r_full['daily_pnl']))
    width = 0.4
    ax.bar(x - width/2, r_full['daily_pnl'].values, width, label='Full', alpha=0.7, color='steelblue')
    ax.bar(x + width/2, r_sel['daily_pnl'].values, width, label='Selected', alpha=0.7, color='indianred')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Daily P&L (EUR)')
    ax.set_title('Daily P&L Comparison')

    # Only show every 5th date label
    dates = r_full['daily_pnl'].index
    tick_positions = range(0, len(dates), 5)
    ax.set_xticks(list(tick_positions))
    ax.set_xticklabels([str(dates[i]) for i in tick_positions], rotation=45, ha='right', fontsize=8)
    ax.legend()

    plt.tight_layout()
    out_path = PLOT_DIR / "feature_selection" / "04_stacked_validation.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


if __name__ == "__main__":
    main()
