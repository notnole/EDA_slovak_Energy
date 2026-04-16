"""Compare feature set configurations in the walk-forward pipeline."""
import pandas as pd, numpy as np, lightgbm as lgb, sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
LEAD = 8

LGB_PARAMS = dict(learning_rate=0.03, num_leaves=15, min_child_samples=200,
                  subsample=0.5, colsample_bytree=0.5, reg_alpha=1.0,
                  reg_lambda=10.0, n_estimators=200, verbose=-1)

FOLDS = [
    ('2025-04-01', '2025-01-01', '2025-04-01'),
    ('2025-07-01', '2025-04-01', '2025-07-01'),
    ('2025-10-01', '2025-07-01', '2025-10-01'),
    ('2026-01-01', '2025-10-01', '2026-01-01'),
    ('2026-02-01', '2026-01-01', '2026-02-01'),
    ('2026-03-01', '2026-02-01', '2026-03-01'),   # test
    ('2026-04-10', '2026-03-01', '2026-04-10'),   # test
]

# Current 47 features
CURRENT_47 = [
    'da_price', 'cloudcover', 'hour_cos', 'idm_vwap_lag', 'da_supply',
    'da_price_change24h', 'proxy_rmax4', 'temp_forecast_da', 'temp_national_spread',
    'temp_bratislava', 'load_rmean16', 'nowcast_momentum_h2h3', 'temp_national_change6h',
    'da_demand', 'temp_surprise_lag', 'proxy_rmean16', 'proxy_range8', 'hour_sin',
    'nowcast_pred_rmean4', 'nowcast_momentum_h4h5',
    'da_flow_cz', 'load_momentum', 'nowcast_h3', 'radiation_national',
    'da_net_import', 'proxy_rmean32', 'nowcast_trend_h2_h5', 'dow_sin',
    'reg_rmean8', 'reg_vol_rmean4', 'proxy_dev_from_hour', 'proxy_yesterday',
    'dow_cos', 'solar_surprise_lag', 'nowcast_h5', 'proxy_rmin4', 'nowcast_convergence',
    'reg_rmean4', 'is_weekend', 'proxy_yesterday_2', 'temp_rmean24h', 'proxy_range4',
    'proxy_lag12', 'proxy_pos_ratio_4', 'proxy_lag21', 'proxy_lag18', 'damas_fe_rmean4',
]

# Recommended 71 from feature selection
REC_71 = [
    'da_price_qh', 'temp_national_change6h', 'da_demand', 'nowcast_momentum_h2h3',
    'nowcast_trend_h2_h5', 'idm_vwap_lag', 'da_flow_cz', 'prod_rmean8',
    'spread_da_idm_lag', 'nowcast_h5', 'damas_fe_rmean4', 'reg_rmean8',
    'nowcast_momentum_h3h4', 'da_price_qh_dev_hourly', 'proxy_lag9',
    'proxy_lag96_diff', 'da_price_qh_diff_next', 'dow_sin', 'wind_national',
    'temp_surprise_lag', 'proxy_lag15', 'prod_momentum', 'nowcast_pred_rmean4',
    'hour_sin', 'da_net_import', 'is_weekend', 'temp_bratislava', 'da_supply',
    'xborder_vol', 'xborder_deviation', 'cloudcover', 'proxy_rstd8',
    'temp_deviation', 'proxy_pos_ratio_8', 'load_rmean16', 'xborder_rmean4',
    'nowcast_momentum_h4h5', 'temp_forecast_da', 'proxy_lag18', 'proxy_pos_ratio_4',
    'xborder_momentum', 'proxy_abs_rmean8', 'proxy_lag10', 'proxy_lag11',
    'temp_change6h', 'load_rmean4', 'is_peak', 'proxy_lag22', 'proxy_rmean32',
    'proxy_acceleration', 'proxy_range4', 'load_yesterday', 'proxy_momentum',
    'load_ramp4', 'proxy_lag20', 'proxy_lag21', 'proxy_lag19', 'proxy_lag13',
    'month_sin', 'da_price_qh_rank', 'month_cos', 'proxy_lag23', 'proxy_lag12',
    'proxy_lag16', 'proxy_yesterday_2', 'proxy_lag24', 'radiation_national',
    'proxy_momentum4', 'dow_cos', 'proxy_zero_cross4', 'load_rstd4',
]

# Top 30 from permutation importance (no leaky, with corr pruning already applied)
TOP_30 = [
    'da_price_qh', 'temp_national_change6h', 'da_demand', 'nowcast_momentum_h2h3',
    'nowcast_trend_h2_h5', 'idm_vwap_lag', 'da_flow_cz', 'prod_rmean8',
    'spread_da_idm_lag', 'nowcast_h5', 'damas_fe_rmean4', 'reg_rmean8',
    'nowcast_momentum_h3h4', 'da_price_qh_dev_hourly', 'proxy_lag9',
    'proxy_lag96_diff', 'da_price_qh_diff_next', 'dow_sin', 'wind_national',
    'temp_surprise_lag', 'proxy_lag15', 'prod_momentum', 'nowcast_pred_rmean4',
    'hour_sin', 'da_net_import', 'is_weekend', 'temp_bratislava', 'da_supply',
    'xborder_vol', 'xborder_deviation',
]

# Top 30 minus short-coverage (no prod/xborder that start Oct 2025)
TOP_26_CLEAN = [f for f in TOP_30
                if not f.startswith('prod_') and not f.startswith('xborder_')]

# Top 20 only
TOP_20 = TOP_30[:20]

configs = {
    'Current 47': CURRENT_47,
    'Recommended 71': REC_71,
    'Top 30 (perm)': TOP_30,
    'Top 26 (no prod/xb)': TOP_26_CLEAN,
    'Top 20': TOP_20,
}


def main():
    print("[*] Loading data...")
    data = load_all_data()

    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                           parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[
        ['bid', 'ask', 'spread', 'mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']

    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, all_feature_cols = build_features(data, LEAD)
    df_base = df_base.join(ob_120, how='left')
    df_base['imb_settlement_price'] = df_base['imb_settle_price']
    df_base['spread_target'] = df_base['imb_settlement_price'] - df_base['exec_mid']

    print(f"[+] Data ready: {len(df_base)} rows, {len(all_feature_cols)} features\n")

    for name, feat_list in configs.items():
        feats = [f for f in feat_list if f in all_feature_cols]

        all_oof = []
        for fi, (train_end, pred_start, pred_end) in enumerate(FOLDS):
            train = df_base[df_base.index < train_end].dropna(
                subset=['target', f'proxy_lag{LEAD+1}'])
            pred_data = df_base[
                (df_base.index >= pred_start) & (df_base.index < pred_end)
            ].dropna(subset=[f'proxy_lag{LEAD+1}'])

            sp_train = train.dropna(subset=['spread_target'])
            sp_train = sp_train[sp_train['imb_settlement_price'].abs() <= 5000]

            m = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
            m.fit(sp_train[feats].values, sp_train['spread_target'].values)
            pred_data = pred_data.copy()
            pred_data['spread_pred'] = m.predict(pred_data[feats].values)
            all_oof.append(pred_data)

        # Test on last 2 folds
        test = pd.concat(all_oof[-2:])
        test = test[test['exec_spread'].notna() & (test['exec_spread'] <= 10)]

        surplus = test['spread_pred'] <= -3
        deficit = test['spread_pred'] >= 3
        trades = test[surplus | deficit].copy()
        trades['size'] = trades['spread_pred'].abs().clip(upper=5)
        s = surplus.reindex(trades.index, fill_value=False)
        d = deficit.reindex(trades.index, fill_value=False)
        trades['pnl'] = 0.0
        trades.loc[s, 'pnl'] = (
            (trades.loc[s, 'exec_bid'] - trades.loc[s, 'imb_settlement_price'])
            * trades.loc[s, 'size'] / 4
        )
        trades.loc[d, 'pnl'] = (
            (trades.loc[d, 'imb_settlement_price'] - trades.loc[d, 'exec_ask'])
            * trades.loc[d, 'size'] / 4
        )

        nd = trades.index.normalize().nunique()
        daily = trades.groupby(trades.index.date)['pnl'].sum()
        sharpe = (daily.mean() / daily.std() * np.sqrt(252)
                  if daily.std() > 0 else 0)
        dd = (daily.cumsum() - daily.cumsum().cummax()).min()
        weekly = daily.groupby(
            pd.Series(daily.index).apply(
                lambda x: pd.Timestamp(x).isocalendar()[1]).values
        ).sum()

        print(f"=== {name} ({len(feats)} features) ===")
        print(f"  {trades.pnl.sum()/nd:>+.0f}/d  Sharpe={sharpe:.1f}  "
              f"win={(trades.pnl > 0).mean():.0%}  DD={dd:+.0f}  "
              f"losing_wk={(weekly < 0).sum()}/{len(weekly)}")

        trades['month'] = trades.index.to_period('M')
        for mo, g in trades.groupby('month'):
            mnd = g.index.normalize().nunique()
            print(f"    {mo}: {g.pnl.sum()/mnd:>+.0f}/d  "
                  f"win={(g.pnl > 0).mean():.0%}")
        print()


if __name__ == "__main__":
    main()
