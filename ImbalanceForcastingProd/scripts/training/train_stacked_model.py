"""
2-Stage Spread Model + Imbalance Diagnostic
=============================================

Pipeline:
  Stage 1: Load nowcast (H+2) → OOF predictions (already generated)
  Spread model: base features only → hourly-smoothed spread target
  Imbalance model: trained per fold for diagnostics only (NOT a stage)

Hourly smoothing: the spread target averages settlement and execution
prices to hourly resolution before differencing. This removes
unpredictable QH noise and lets the model focus on the learnable
hourly signal. P&L is always evaluated on real 15-min settlement.

Folds 1-3 build up OOF history. Folds 4-5 are the test period.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

BASE_DIR = Path(__file__).resolve().parents[2]  # ImbalanceForcastingProd/
DATA_DIR = BASE_DIR / "data"
REPO_ROOT = BASE_DIR.parent  # repo root

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


def main():
    print("=" * 70)
    print("2-STAGE SPREAD MODEL + IMBALANCE DIAGNOSTIC")
    print("=" * 70)

    # Load all data
    data = load_all_data()

    # Load execution prices + settlement for spread target
    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv", parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'spread', 'mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']

    # Build base features once (they don't change per fold — only the train/test split does)
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, feature_cols = build_features(data, LEAD)

    # Join execution prices, settlement
    df_base = df_base.join(ob_120, how='left')
    df_base['imb_settlement_price'] = df_base['imb_settle_price']

    # Hourly-smoothed spread target: average both settlement and exec_mid to
    # hourly resolution, then difference. Removes unpredictable QH noise —
    # the model learns the hourly signal it can actually capture.
    df_base['hour_ts'] = df_base.index.floor('h')
    df_base['settle_hourly'] = df_base.groupby('hour_ts')['imb_settlement_price'].transform('mean')
    df_base['mid_hourly'] = df_base.groupby('hour_ts')['exec_mid'].transform('mean')
    df_base['spread_target'] = df_base['settle_hourly'] - df_base['mid_hourly']

    # Raw 15-min spread (for P&L evaluation only — never trained on)
    df_base['spread_15m'] = df_base['imb_settlement_price'] - df_base['exec_mid']

    print(f"[+] Base features: {len(feature_cols)} columns, {len(df_base)} rows")

    # ============================================================
    # WALK-FORWARD FOLD PROCESSING
    # ============================================================
    all_spread_oof = []   # spread model OOF predictions
    all_imb_oof = []      # imbalance model OOF (diagnostic only)

    for fi, (train_end, pred_start, pred_end) in enumerate(FOLDS):
        print(f"\n{'='*60}")
        print(f"FOLD {fi+1}/{len(FOLDS)}: train < {train_end}, predict [{pred_start}, {pred_end})")
        print(f"{'='*60}")

        # --- Prepare data for this fold ---
        df = df_base.copy()

        # Split
        train_mask = df.index < train_end
        pred_mask = (df.index >= pred_start) & (df.index < pred_end)
        train = df[train_mask].dropna(subset=['target', f'proxy_lag{LEAD+1}'])
        pred_data = df[pred_mask].dropna(subset=[f'proxy_lag{LEAD+1}'])

        if len(train) == 0 or len(pred_data) == 0:
            print(f"  [!] Insufficient data, skipping")
            continue

        print(f"  Train: {len(train)}, Predict: {len(pred_data)}")

        # --- IMBALANCE MODEL (diagnostic only — NOT fed into spread model) ---
        print(f"  Imbalance (diagnostic): {len(feature_cols)} features")
        m_imb = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        m_imb.fit(train[feature_cols].values, train['target'].values)

        pred_data = pred_data.copy()
        pred_data['imb_pred'] = m_imb.predict(pred_data[feature_cols].values)

        nz = pred_data['target'].abs() > 0.1
        imb_dir = (np.sign(pred_data['imb_pred']) == np.sign(pred_data['target']))[nz].mean()
        print(f"    Imbalance OOF: dir_acc={imb_dir:.1%}")

        imb_oof = pred_data[['imb_pred', 'target']].copy()
        all_imb_oof.append(imb_oof)

        # --- SPREAD MODEL (main model — base features, hourly-smoothed target) ---
        sp_train = train.dropna(subset=['spread_target'])
        sp_train = sp_train[sp_train['imb_settlement_price'].abs() <= 5000]

        print(f"  Spread model: {len(feature_cols)} features, {len(sp_train)} train rows")

        m_sp = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        m_sp.fit(sp_train[feature_cols].values, sp_train['spread_target'].values)

        pred_data['spread_pred'] = m_sp.predict(pred_data[feature_cols].values)

        if pred_data['spread_15m'].notna().sum() > 0:
            sp_nz = pred_data['spread_15m'].abs() > 0.1
            sp_dir = (np.sign(pred_data['spread_pred']) == np.sign(pred_data['spread_15m']))[sp_nz].mean()
            print(f"    Spread OOF: dir_acc={sp_dir:.1%} (vs real 15-min spread)")

        # Feature importance (last fold only to avoid spam)
        if fi == len(FOLDS) - 1:
            imp = pd.DataFrame({'feature': feature_cols, 'importance': m_sp.feature_importances_})
            imp['pct'] = imp['importance'] / imp['importance'].sum() * 100
            imp = imp.sort_values('pct', ascending=False)
            print(f"\n    Top 15 features (last fold):")
            for _, row in imp.head(15).iterrows():
                print(f"      {row['feature']:<35s} {row['pct']:.1f}%")

        # Save OOF
        oof = pred_data[['spread_pred', 'imb_pred', 'target', 'spread_target', 'spread_15m',
                          'exec_bid', 'exec_ask', 'exec_spread', 'imb_settlement_price']].copy()
        all_spread_oof.append(oof)

    # ============================================================
    # TRADING BACKTEST on test folds (4-5: Feb-Mar 2026)
    # ============================================================
    print("\n" + "=" * 70)
    print("TRADING BACKTEST (test period: Feb-Mar 2026)")
    print("P&L on real 15-min settlement with bid/ask execution")
    print("=" * 70)

    # Collect test predictions
    test_folds = [s for fi, s in enumerate(all_spread_oof) if fi >= 3]  # folds 4-5
    if not test_folds:
        print("[!] No test fold predictions")
        return

    test_df = pd.concat(test_folds)
    test_df = test_df[test_df['exec_spread'].notna() & (test_df['exec_spread'] <= 10)]

    n_days = test_df.index.normalize().nunique()

    def bt(t, pred_col, label, threshold=3):
        surplus_mask = t[pred_col] <= -threshold
        deficit_mask = t[pred_col] >= threshold
        sub = t[surplus_mask | deficit_mask].copy()
        if len(sub) < 30:
            print(f"  {label:<50s}  too few trades")
            return
        sub['size'] = sub[pred_col].abs().clip(upper=5)
        s = surplus_mask.reindex(sub.index, fill_value=False)
        d = deficit_mask.reindex(sub.index, fill_value=False)
        sub['pnl'] = 0.0
        # P&L always on real 15-min settlement price
        sub.loc[s, 'pnl'] = (sub.loc[s, 'exec_bid'] - sub.loc[s, 'imb_settlement_price']) * sub.loc[s, 'size'] / 4
        sub.loc[d, 'pnl'] = (sub.loc[d, 'imb_settlement_price'] - sub.loc[d, 'exec_ask']) * sub.loc[d, 'size'] / 4
        nd = sub.index.normalize().nunique()
        total = sub['pnl'].sum()
        wr = (sub['pnl'] > 0).mean()
        daily = sub.groupby(sub.index.date)['pnl'].sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        dd = (daily.cumsum() - daily.cumsum().cummax()).min()
        print(f"  {label:<50s} {len(sub):4d}t, win={wr:.0%}, {total:>+7,.0f} ({total/nd:>+4.0f}/d) "
              f"Sharpe={sharpe:.1f} DD={dd:+.0f}")

    print(f"\nTest: {len(test_df)} periods, {n_days} days")

    print("\n--- Spread model (hourly target) ---")
    for thresh in [3, 5, 8]:
        bt(test_df, 'spread_pred', f'Spread |pred|>={thresh}', threshold=thresh)

    # Monthly breakdown
    print("\n--- Monthly (|pred|>=3) ---")
    test_df['month'] = test_df.index.to_period('M')
    for period in test_df['month'].unique():
        sub = test_df[test_df['month'] == period]
        bt(sub, 'spread_pred', f'  {period}')

    # Imbalance diagnostic summary
    print("\n--- Imbalance model (diagnostic) ---")
    imb_df = pd.concat(all_imb_oof)
    imb_test = imb_df[imb_df.index >= '2026-02-01']
    if len(imb_test) > 0:
        nz = imb_test['target'].abs() > 0.1
        dir_acc = (np.sign(imb_test['imb_pred']) == np.sign(imb_test['target']))[nz].mean()
        mae = (imb_test['imb_pred'] - imb_test['target']).abs().mean()
        print(f"  Test dir_acc={dir_acc:.1%}, MAE={mae:.2f} MWh, n={len(imb_test)}")

    # Save predictions
    out = test_df[['spread_pred', 'imb_pred', 'target', 'spread_target', 'spread_15m',
                    'exec_bid', 'exec_ask', 'exec_spread', 'imb_settlement_price']].copy()
    out_path = DATA_DIR / "predictions" / "spread_model_predictions.csv"
    out.to_csv(out_path)
    print(f"\n[+] Saved: {out_path}")


if __name__ == "__main__":
    main()
