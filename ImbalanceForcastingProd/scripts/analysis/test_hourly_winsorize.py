"""
Test: hourly-smoothed target with and without Winsorize +/-30.
All P&L evaluated on real 15-min settlement with bid/ask execution.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
LEAD = 8
SIZE_MW = 5.0
ENERGY = SIZE_MW * 0.25

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                  reg_lambda=1.0, n_estimators=600, verbose=-1)


def run_bt(label, test_df, pred_col, thresholds=(3, 5)):
    """Backtest on real 15-min settlement."""
    for thresh in thresholds:
        surplus = test_df[pred_col] <= -thresh
        deficit = test_df[pred_col] >= thresh
        sub = test_df[surplus | deficit].copy()
        if len(sub) < 30:
            print(f"  {label:45s} |p|>={thresh}: too few trades ({len(sub)})")
            continue

        s = surplus.reindex(sub.index, fill_value=False)
        d = deficit.reindex(sub.index, fill_value=False)
        sub['pnl'] = 0.0
        sub.loc[s, 'pnl'] = (sub.loc[s, 'exec_bid'] - sub.loc[s, 'settle_15m']) * ENERGY
        sub.loc[d, 'pnl'] = (sub.loc[d, 'settle_15m'] - sub.loc[d, 'exec_ask']) * ENERGY

        daily = sub.groupby(sub.index.date)['pnl'].sum()
        nd = len(daily)
        total = sub['pnl'].sum()
        wr = (sub['pnl'] > 0).mean()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        dd = (daily.cumsum() - daily.cumsum().cummax()).min()

        # Direction accuracy on real 15-min spread
        nz = test_df['spread_15m'].abs() > 0.1
        dacc = (np.sign(test_df.loc[nz, pred_col]) == np.sign(test_df.loc[nz, 'spread_15m'])).mean()

        print(f"  {label:45s} |p|>={thresh}: {len(sub):4d}t "
              f"{total/nd:+7.0f}/d Sh={sharpe:5.1f} W={wr:.0%} DD={dd:+.0f} DirAcc={dacc:.1%}")


def main():
    print("=" * 80)
    print("HOURLY SMOOTHED TARGET: WINSORIZE +/-30 vs NO WINSORIZE")
    print("=" * 80)

    data = load_all_data()
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df, feature_cols = build_features(data, LEAD)

    # Execution prices
    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                          parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid','ask','spread','mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']
    df = df.join(ob_120, how='left')

    # Settlement at 15-min and hourly
    df['settle_15m'] = df['imb_settle_price']
    df['hour_ts'] = df.index.floor('h')
    df['settle_hourly_avg'] = df.groupby('hour_ts')['settle_15m'].transform('mean')
    df['mid_hourly_avg'] = df.groupby('hour_ts')['exec_mid'].transform('mean')

    # Targets
    df['spread_15m'] = df['settle_15m'] - df['exec_mid']
    df['spread_hourly'] = df['settle_hourly_avg'] - df['mid_hourly_avg']
    df['spread_hourly_w30'] = df['spread_hourly'].clip(-30, 30)

    # Also test: 15-min raw and 15-min winsorized for reference
    df['spread_15m_w30'] = df['spread_15m'].clip(-30, 30)

    # Train/test split
    train = df[df.index < '2026-02-01'].copy()
    test = df[(df.index >= '2026-02-01') & (df.index < '2026-04-01')].copy()
    test = test[test['exec_bid'].notna() & test['exec_ask'].notna() & (test['exec_spread'] <= 10)]

    # Filter train
    train = train[train['settle_15m'].notna() & (train['settle_15m'].abs() <= 5000)]

    print(f"\nTrain: {len(train)}, Test: {len(test)}")
    print(f"Train spread_hourly stats: mean={train['spread_hourly'].mean():.1f}, "
          f"std={train['spread_hourly'].std():.1f}, "
          f"min={train['spread_hourly'].min():.0f}, max={train['spread_hourly'].max():.0f}")
    print(f"Train rows clipped by W30: {(train['spread_hourly'].abs() > 30).sum()} "
          f"({(train['spread_hourly'].abs() > 30).mean():.1%})")
    print()

    # --- Train models ---
    variants = [
        ("A. 15-min raw (reference)",       'spread_15m',        'pred_15m'),
        ("B. 15-min + Winsorize30",         'spread_15m_w30',    'pred_15m_w30'),
        ("C. Hourly smoothed (no winsor)",  'spread_hourly',     'pred_hourly'),
        ("D. Hourly smoothed + Winsorize30", 'spread_hourly_w30', 'pred_hourly_w30'),
    ]

    print("=== TRAINING ===")
    for label, target_col, pred_col in variants:
        t = train.dropna(subset=[target_col, f'proxy_lag{LEAD+1}'])
        model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        model.fit(t[feature_cols].values, t[target_col].values)
        test[pred_col] = model.predict(test[feature_cols].values)
        print(f"  {label}: trained on {len(t)} rows, "
              f"target range [{t[target_col].min():.0f}, {t[target_col].max():.0f}]")

    # --- Evaluate ---
    print()
    print("=== RESULTS (P&L on real 15-min settlement, bid/ask execution) ===")
    print()
    for label, target_col, pred_col in variants:
        run_bt(label, test, pred_col)
        print()

    # --- Prediction distribution comparison ---
    print("=== PREDICTION DISTRIBUTIONS ===")
    for label, target_col, pred_col in variants:
        p = test[pred_col]
        print(f"  {label:45s} mean={p.mean():+5.1f} std={p.std():5.1f} "
              f"[{p.min():+6.1f}, {p.max():+6.1f}]  "
              f"|p|>3: {(p.abs()>3).sum()} ({(p.abs()>3).mean():.0%})  "
              f"|p|>5: {(p.abs()>5).sum()} ({(p.abs()>5).mean():.0%})")


if __name__ == "__main__":
    main()
