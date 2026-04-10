"""
3-Stage Stacked Model with Proper OOF Predictions
===================================================

Pipeline:
  Stage 1: Load nowcast (H+2) → OOF predictions (already generated)
  Stage 2: Imbalance model (uses Stage 1 OOF) → OOF predictions
  Stage 3: Spread model (uses Stage 1 + Stage 2 OOF) → final trading signal

Walk-forward folds — each stage's OOF predictions from prior folds
are used as features in the next stage's training.

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
    print("3-STAGE STACKED MODEL")
    print("=" * 70)

    # Load all data
    data = load_all_data()

    # Load Stage 1 OOF (load nowcast — already generated)
    stage1_oof = pd.read_csv(
        REPO_ROOT / "LoadAnalysis" / "nowcast_5h" / "tuning" / "oos_predictions" / "h2_oos_predictions.csv",
        parse_dates=['datetime'], index_col='datetime')
    stage1_oof = stage1_oof[~stage1_oof.index.duplicated(keep='last')]
    stage1_oof = stage1_oof[['predicted_error']].rename(columns={'predicted_error': 'stk_load_nowcast'})
    # Broadcast hourly to 15-min
    stage1_oof_15 = stage1_oof.resample('15min').ffill()
    print(f"[+] Stage 1 OOF (load nowcast): {len(stage1_oof_15)} periods, "
          f"{stage1_oof_15.index.min()} to {stage1_oof_15.index.max()}")

    # Load execution prices + settlement for spread target
    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv", parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'spread', 'mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']

    mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                       parse_dates=['timestamp_hour'], index_col='timestamp_hour')
    mkt = mkt[~mkt.index.duplicated(keep='last')]

    # Build base features once (they don't change per fold — only the train/test split does)
    # Use the widest possible train_end so all data is present
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, feature_cols = build_features(data, LEAD)

    # Join Stage 1 OOF, execution prices, settlement
    df_base = df_base.join(stage1_oof_15, how='left')
    df_base['hour_ts'] = df_base.index.floor('h')
    df_base = df_base.join(mkt[['imb_settlement_price']], on='hour_ts', how='left')
    df_base = df_base.join(ob_120, how='left')
    df_base['spread_target'] = df_base['imb_settlement_price'] - df_base['exec_mid']

    print(f"[+] Base features: {len(feature_cols)} columns, {len(df_base)} rows")

    # ============================================================
    # WALK-FORWARD FOLD PROCESSING
    # ============================================================
    # Accumulate OOF predictions across folds
    all_stage2_oof = []  # imbalance OOF predictions
    all_stage3_oof = []  # spread OOF predictions

    for fi, (train_end, pred_start, pred_end) in enumerate(FOLDS):
        print(f"\n{'='*60}")
        print(f"FOLD {fi+1}/{len(FOLDS)}: train < {train_end}, predict [{pred_start}, {pred_end})")
        print(f"{'='*60}")

        # --- Prepare data for this fold ---
        df = df_base.copy()

        # Join accumulated Stage 2 OOF from PRIOR folds (not this fold!)
        if all_stage2_oof:
            prior_s2 = pd.concat(all_stage2_oof)
            prior_s2 = prior_s2[~prior_s2.index.duplicated(keep='last')]
            df = df.join(prior_s2[['stk_imb_pred', 'stk_imb_pred_abs', 'stk_imb_direction']], how='left')
        else:
            # No prior OOF yet — these features will be NaN (LightGBM handles it)
            df['stk_imb_pred'] = np.nan
            df['stk_imb_pred_abs'] = np.nan
            df['stk_imb_direction'] = np.nan

        # Split
        train_mask = df.index < train_end
        pred_mask = (df.index >= pred_start) & (df.index < pred_end)
        train = df[train_mask].dropna(subset=['target', f'proxy_lag{LEAD+1}'])
        pred_data = df[pred_mask].dropna(subset=[f'proxy_lag{LEAD+1}'])

        if len(train) == 0 or len(pred_data) == 0:
            print(f"  [!] Insufficient data, skipping")
            continue

        print(f"  Train: {len(train)}, Predict: {len(pred_data)}")

        # --- STAGE 2: Train imbalance model ---
        # Features: base + Stage 1 OOF (stk_load_nowcast is already in base via join)
        s2_features = feature_cols  # stk_load_nowcast is in df but not in feature_cols from build_features
        # We need to check if stk_load_nowcast is already captured by the nowcast features in build_features
        # It IS — build_features already joins nowcast_pred_error. So feature_cols already includes it.

        print(f"  Stage 2 (imbalance): {len(s2_features)} features")
        m_imb = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        m_imb.fit(train[s2_features].values, train['target'].values)

        pred_data = pred_data.copy()
        pred_data['stk_imb_pred'] = m_imb.predict(pred_data[s2_features].values)
        pred_data['stk_imb_pred_abs'] = pred_data['stk_imb_pred'].abs()
        pred_data['stk_imb_direction'] = np.sign(pred_data['stk_imb_pred'])

        # Save Stage 2 OOF for this fold
        s2_oof = pred_data[['stk_imb_pred', 'stk_imb_pred_abs', 'stk_imb_direction']].copy()
        all_stage2_oof.append(s2_oof)

        nz = pred_data['target'].abs() > 0.1
        s2_dir = (np.sign(pred_data['stk_imb_pred']) == np.sign(pred_data['target']))[nz].mean()
        print(f"    Imbalance OOF: dir_acc={s2_dir:.1%}")

        # --- STAGE 3: Train spread model with Stage 2 OOF as feature ---
        # Now the spread model gets stk_imb_pred from PRIOR folds' OOF (in training)
        # and from this fold's Stage 2 prediction (in prediction)
        # Need to update the training data's stk_imb_pred from the accumulated OOF
        train_with_s2 = train.copy()
        # train already has stk_imb_pred from prior folds' OOF (joined at top of loop)
        # For rows in train that DON'T have Stage 2 OOF (very early data), it's NaN — fine for LightGBM

        s3_features = feature_cols + ['stk_imb_pred', 'stk_imb_pred_abs', 'stk_imb_direction']

        # Filter train to rows with valid spread target
        s3_train = train_with_s2.dropna(subset=['spread_target'])
        s3_train = s3_train[s3_train['imb_settlement_price'].abs() <= 5000]

        if len(s3_train) < 100:
            print(f"  Stage 3: insufficient spread training data ({len(s3_train)}), using spread-only")
            # Fallback: spread model without stacking features
            s3_features_fallback = feature_cols
            m_sp = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
            m_sp.fit(s3_train[s3_features_fallback].values, s3_train['spread_target'].values)
            pred_data['stk_spread_pred'] = m_sp.predict(pred_data[s3_features_fallback].values)
        else:
            print(f"  Stage 3 (spread): {len(s3_features)} features, {len(s3_train)} train rows")
            print(f"    stk_imb_pred coverage in train: {s3_train['stk_imb_pred'].notna().mean():.0%}")

            m_sp = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
            m_sp.fit(s3_train[s3_features].values, s3_train['spread_target'].values)

            # Predict — pred_data has stk_imb_pred from Stage 2 (this fold)
            pred_data['stk_spread_pred'] = m_sp.predict(pred_data[s3_features].values)

            # Feature importance for stacking features
            imp = pd.DataFrame({'feature': s3_features, 'importance': m_sp.feature_importances_})
            imp['pct'] = imp['importance'] / imp['importance'].sum() * 100
            stk_imp = imp[imp['feature'].str.startswith('stk_')]
            print(f"    Stacking feature importance: {stk_imp['pct'].sum():.1f}%")
            for _, row in stk_imp.iterrows():
                print(f"      {row['feature']:<30s} {row['pct']:.2f}%")

        # Save Stage 3 OOF
        s3_oof = pred_data[['stk_spread_pred', 'stk_imb_pred', 'target', 'spread_target',
                             'exec_bid', 'exec_ask', 'exec_spread', 'imb_settlement_price']].copy()
        all_stage3_oof.append(s3_oof)

        if pred_data['spread_target'].notna().sum() > 0:
            sp_nz = pred_data['spread_target'].abs() > 0.1
            s3_dir = (np.sign(pred_data['stk_spread_pred']) == np.sign(pred_data['spread_target']))[sp_nz].mean()
            print(f"    Spread OOF: dir_acc={s3_dir:.1%}")

    # ============================================================
    # TRADING BACKTEST on test folds (4-5: Feb-Mar 2026)
    # ============================================================
    print("\n" + "=" * 70)
    print("TRADING BACKTEST (test period: Feb-Mar 2026)")
    print("=" * 70)

    # Also need a standalone spread model (no stacking) for comparison
    # Train on all data < 2026-02-01
    standalone_train = df_base[df_base.index < '2026-02-01'].dropna(subset=['spread_target', f'proxy_lag{LEAD+1}'])
    standalone_train = standalone_train[standalone_train['imb_settlement_price'].abs() <= 5000]
    m_standalone = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
    m_standalone.fit(standalone_train[feature_cols].values, standalone_train['spread_target'].values)

    # Collect test predictions
    test_folds = [s3 for fi, s3 in enumerate(all_stage3_oof) if fi >= 3]  # folds 4-5
    if not test_folds:
        print("[!] No test fold predictions")
        return

    test_df = pd.concat(test_folds)
    test_df = test_df[test_df['exec_spread'].notna() & (test_df['exec_spread'] <= 10)]

    # Add standalone prediction
    test_base = df_base.loc[test_df.index].dropna(subset=[f'proxy_lag{LEAD+1}'])
    test_df['standalone_spread_pred'] = m_standalone.predict(test_base[feature_cols].values)

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
        sub.loc[s, 'pnl'] = (sub.loc[s, 'exec_bid'] - sub.loc[s, 'imb_settlement_price']) * sub.loc[s, 'size'] / 4
        sub.loc[d, 'pnl'] = (sub.loc[d, 'imb_settlement_price'] - sub.loc[d, 'exec_ask']) * sub.loc[d, 'size'] / 4
        nd = sub.index.normalize().nunique()
        total = sub['pnl'].sum()
        wr = (sub['pnl'] > 0).mean()
        daily = sub.groupby(sub.index.date)['pnl'].sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        print(f"  {label:<50s} {len(sub):4d}t, win={wr:.0%}, {total:>+7,.0f} ({total/nd:>+4.0f}/d) Sharpe={sharpe:.1f}")

    print(f"\nTest: {len(test_df)} periods, {n_days} days")

    print("\n--- Spread model comparison ---")
    bt(test_df, 'standalone_spread_pred', 'Standalone spread (no stacking)')
    bt(test_df, 'stk_spread_pred', 'Stacked spread (with imb OOF feature)')

    print("\n--- Threshold sweep (stacked) ---")
    for thresh in [3, 5, 8]:
        bt(test_df, 'stk_spread_pred', f'Stacked |pred|>={thresh}', threshold=thresh)

    # Monthly breakdown
    print("\n--- Monthly (stacked, |pred|>=3) ---")
    test_df['month'] = test_df.index.to_period('M')
    for period in test_df['month'].unique():
        sub = test_df[test_df['month'] == period]
        bt(sub, 'stk_spread_pred', f'  {period}')

    # Save
    out = test_df[['target', 'spread_target', 'stk_imb_pred', 'stk_spread_pred',
                    'standalone_spread_pred', 'exec_bid', 'exec_ask', 'exec_spread',
                    'imb_settlement_price']].copy()
    out.to_csv(DATA_DIR / "stacked_test_predictions.csv")
    print(f"\n[+] Saved: {DATA_DIR / 'stacked_test_predictions.csv'}")


if __name__ == "__main__":
    main()
