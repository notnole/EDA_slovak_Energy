"""Quick test: train on hourly-avg smoothed spread, evaluate on real 15-min P&L."""
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
REPO_ROOT = BASE_DIR.parent
LEAD = 8
SIZE_MW = 5.0
ENERGY = SIZE_MW * 0.25

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                  reg_lambda=1.0, n_estimators=600, verbose=-1)


def main():
    data = load_all_data()
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df, feature_cols = build_features(data, LEAD)

    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                          parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid','ask','spread','mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid','exec_ask','exec_spread','exec_mid']

    df = df.join(ob_120, how='left')
    df['settle_15m'] = df['imb_settle_price']
    df['hour_ts'] = df.index.floor('h')
    df['settle_hourly_avg'] = df.groupby('hour_ts')['settle_15m'].transform('mean')
    df['mid_hourly_avg'] = df.groupby('hour_ts')['exec_mid'].transform('mean')

    # Three targets
    df['spread_15m'] = df['settle_15m'] - df['exec_mid']
    df['spread_hourly'] = df['settle_hourly_avg'] - df['mid_hourly_avg']
    df['spread_mixed'] = df['settle_hourly_avg'] - df['exec_mid']

    train = df[df.index < '2026-02-01'].copy()
    test = df[(df.index >= '2026-02-01') & (df.index < '2026-04-01')].copy()
    test = test[test['exec_bid'].notna() & test['exec_ask'].notna() & (test['exec_spread'] <= 10)]

    print(f"Train: {len(train)}, Test: {len(test)}")
    print()

    def run_model(label, train_df, test_df, target_col):
        t = train_df.dropna(subset=[target_col, f'proxy_lag{LEAD+1}'])
        t = t[t['settle_15m'].notna() & (t['settle_15m'].abs() <= 5000)]

        model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        model.fit(t[feature_cols].values, t[target_col].values)

        test_df = test_df.copy()
        test_df['pred'] = model.predict(test_df[feature_cols].values)

        # Direction accuracy on real 15-min spread
        nz = test_df['spread_15m'].abs() > 0.1
        all_dacc = (np.sign(test_df.loc[nz, 'pred']) == np.sign(test_df.loc[nz, 'spread_15m'])).mean()

        # P&L always on real 15-min settlement
        for thresh in [3, 5]:
            surplus = test_df['pred'] <= -thresh
            deficit = test_df['pred'] >= thresh
            sub = test_df[surplus | deficit].copy()
            if len(sub) < 30:
                continue

            s = surplus.reindex(sub.index, fill_value=False)
            d = deficit.reindex(sub.index, fill_value=False)
            sub['pnl'] = 0.0
            sub.loc[s, 'pnl'] = (sub.loc[s, 'exec_bid'] - sub.loc[s, 'settle_15m']) * ENERGY
            sub.loc[d, 'pnl'] = (sub.loc[d, 'settle_15m'] - sub.loc[d, 'exec_ask']) * ENERGY

            nd = sub.index.normalize().nunique()
            total = sub['pnl'].sum()
            wr = (sub['pnl'] > 0).mean()
            daily = sub.groupby(sub.index.date)['pnl'].sum()
            sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0

            print(f"  {label:40s} |p|>={thresh}: {len(sub):4d}t "
                  f"{total/nd:+7.0f}/d Sh={sharpe:5.1f} W={wr:.0%} DirAcc={all_dacc:.1%}")

    print("=== TRAIN TARGET COMPARISON (P&L always on real 15-min settlement) ===")
    print()
    run_model("A. 15-min spread (correct)", train, test, 'spread_15m')
    run_model("B. Hourly-avg spread (smoothed)", train, test, 'spread_hourly')
    run_model("C. Hourly settle - QH mid (mixed)", train, test, 'spread_mixed')


if __name__ == "__main__":
    main()
