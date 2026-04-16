"""
Train Production Spread Model
==============================

Trains a single spread model on ALL available data and saves it for live use.
No walk-forward folds, no test holdout — uses every row for maximum model quality.

The model and feature list are saved to:
  models/spread_production.joblib
  models/spread_production_metadata.json

Usage:
    python ImbalanceForcastingProd/scripts/training/train_production_model.py

For backtesting/evaluation, use train_stacked_model.py instead.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import sys
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

BASE_DIR = Path(__file__).resolve().parents[2]  # ImbalanceForcastingProd/
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPO_ROOT = BASE_DIR.parent

LEAD = 8

# Must match train_stacked_model.py and predict.py exactly
SELECTED_FEATURES = [
    'da_price_qh', 'temp_national_change6h', 'da_demand', 'nowcast_momentum_h2h3',
    'nowcast_trend_h2_h5', 'idm_vwap_lag', 'da_flow_cz', 'prod_rmean8',
    'spread_da_idm_lag', 'nowcast_h5', 'damas_fe_rmean4', 'reg_rmean8',
    'nowcast_momentum_h3h4', 'da_price_qh_dev_hourly', 'proxy_lag9',
    'proxy_lag96_diff', 'da_price_qh_diff_next', 'dow_sin', 'wind_national',
    'temp_surprise_lag', 'proxy_lag15', 'prod_momentum', 'nowcast_pred_rmean4',
    'hour_sin', 'da_net_import', 'is_weekend', 'temp_bratislava', 'da_supply',
    'xborder_vol', 'xborder_deviation',
]

LGB_PARAMS = dict(learning_rate=0.03, num_leaves=15, min_child_samples=200,
                  subsample=0.5, colsample_bytree=0.5, reg_alpha=1.0,
                  reg_lambda=10.0, n_estimators=200, verbose=-1)


def main():
    print("=" * 70)
    print("TRAIN PRODUCTION SPREAD MODEL (all data)")
    print("=" * 70)

    # Load all data
    data = load_all_data()

    # Load execution prices for spread target
    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                           parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[
        ['bid', 'ask', 'spread', 'mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']

    # Build features using all data
    tml.TRAIN_END = '2099-01-01'   # no test split — everything is training
    tml.TEST_START = '2099-01-01'
    df_base, feature_cols = build_features(data, LEAD)

    df_base = df_base.join(ob_120, how='left')
    df_base['imb_settlement_price'] = df_base['imb_settle_price']
    df_base['spread_target'] = df_base['imb_settlement_price'] - df_base['exec_mid']

    # Validate features
    spread_features = [f for f in SELECTED_FEATURES if f in feature_cols]
    missing = [f for f in SELECTED_FEATURES if f not in feature_cols]
    if missing:
        print(f"[!] Warning: {len(missing)} selected features not found: {missing}")

    # Prepare training data
    train = df_base.dropna(subset=['spread_target', f'proxy_lag{LEAD+1}'])
    train = train[train['imb_settlement_price'].abs() <= 5000]

    print(f"[+] Features: {len(spread_features)}")
    print(f"[+] Training rows: {len(train):,}")
    print(f"[+] Date range: {train.index.min().date()} to {train.index.max().date()}")
    print(f"[+] Spread target: mean={train['spread_target'].mean():.1f}, "
          f"std={train['spread_target'].std():.1f}")

    # Train
    print(f"\n[*] Training LightGBM (quantile median)...")
    model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
    model.fit(train[spread_features].values, train['spread_target'].values)

    # Feature importance
    imp = pd.DataFrame({
        'feature': spread_features,
        'importance': model.feature_importances_
    })
    imp['pct'] = imp['importance'] / imp['importance'].sum() * 100
    imp = imp.sort_values('pct', ascending=False)

    print(f"\n  Top 15 features:")
    for _, row in imp.head(15).iterrows():
        print(f"    {row['feature']:<35s} {row['pct']:.1f}%")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "spread_production.joblib"
    joblib.dump(model, model_path)
    print(f"\n[+] Model saved: {model_path}")

    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'train_rows': len(train),
        'train_start': str(train.index.min().date()),
        'train_end': str(train.index.max().date()),
        'n_features': len(spread_features),
        'features': spread_features,
        'lgb_params': LGB_PARAMS,
        'lead': LEAD,
        'target': 'imb_settlement_price - exec_mid (raw 15-min)',
        'spread_target_mean': float(train['spread_target'].mean()),
        'spread_target_std': float(train['spread_target'].std()),
    }
    meta_path = MODEL_DIR / "spread_production_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Metadata saved: {meta_path}")

    # Quick sanity check: predict on last week of training data
    last_week = train[train.index >= train.index.max() - pd.Timedelta(days=7)]
    preds = model.predict(last_week[spread_features].values)
    print(f"\n[*] Sanity check (last week of training):")
    print(f"    Predictions: mean={preds.mean():.1f}, std={preds.std():.1f}")
    print(f"    Range: [{preds.min():.1f}, {preds.max():.1f}]")
    print(f"    % surplus (pred<=-3): {(preds <= -3).mean():.0%}")
    print(f"    % deficit (pred>=3): {(preds >= 3).mean():.0%}")
    print(f"    % no_trade: {((preds > -3) & (preds < 3)).mean():.0%}")

    print(f"\n{'='*70}")
    print(f"Production model ready.")
    print(f"  Model:    {model_path}")
    print(f"  Metadata: {meta_path}")
    print(f"  Use with: predict.py SpreadPredictor('{model_path}')")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
