"""Test QH DA price features with Oct 2025+ training (native 15-min data)."""
import sys, pandas as pd, numpy as np, lightgbm as lgb
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LEAD = 8
ENERGY = 5.0 * 0.25

SELECTED_50 = [
    'da_price', 'cloudcover', 'hour_cos', 'idm_vwap_lag', 'da_supply',
    'da_price_change24h', 'proxy_rmax4', 'temp_forecast_da', 'temp_national_spread',
    'temp_bratislava', 'load_rmean16', 'nowcast_momentum_h2h3', 'temp_national_change6h',
    'da_demand', 'temp_surprise_lag', 'proxy_rmean16', 'proxy_range8', 'hour_sin',
    'prod_momentum', 'nowcast_pred_rmean4', 'nowcast_momentum_h4h5',
    'da_flow_cz', 'load_momentum', 'xborder_momentum', 'nowcast_h3', 'radiation_national',
    'da_net_import', 'proxy_rmean32', 'nowcast_trend_h2_h5', 'dow_sin',
    'reg_rmean8', 'reg_vol_rmean4', 'proxy_dev_from_hour', 'proxy_yesterday', 'prod_rmean8',
    'dow_cos', 'solar_surprise_lag', 'nowcast_h5', 'proxy_rmin4', 'nowcast_convergence',
    'reg_rmean4', 'is_weekend', 'proxy_yesterday_2', 'temp_rmean24h', 'proxy_range4',
    'proxy_lag12', 'proxy_pos_ratio_4', 'proxy_lag21', 'proxy_lag18', 'damas_fe_rmean4',
]
QH_FEATS = ['da_price_qh', 'da_price_qh_diff_prev', 'da_price_qh_diff_next',
            'da_price_qh_dev_hourly', 'da_price_qh_rank']
SELECTED_55 = SELECTED_50 + QH_FEATS

LP = dict(learning_rate=0.03, num_leaves=15, min_child_samples=200,
          subsample=0.5, colsample_bytree=0.5, reg_alpha=1.0,
          reg_lambda=10.0, n_estimators=200, verbose=-1)

FOLDS = [
    ('2026-01-01', '2026-01-01', '2026-02-01'),
    ('2026-02-01', '2026-02-01', '2026-03-01'),
    ('2026-03-01', '2026-03-01', '2026-04-01'),
    ('2026-04-01', '2026-04-01', '2026-04-13'),
]


def run_config(df, fc, label, feats, train_start='2025-10-01'):
    sf = [f for f in feats if f in fc]
    all_trades = []
    last_model = None

    for te, ts, tend in FOLDS:
        tr = df[(df.index >= train_start) & (df.index < te)].dropna(
            subset=['spread_target', 'proxy_lag9'])
        tr = tr[tr['imb_settlement_price'].abs() <= 5000]
        tt = df[(df.index >= ts) & (df.index < tend)].copy()
        tt = tt.dropna(subset=['proxy_lag9'])
        tt = tt[tt['exec_bid'].notna() & tt['exec_ask'].notna() & (tt['exec_spread'] <= 10)]
        if len(tr) < 500 or len(tt) < 30:
            continue

        m = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LP)
        m.fit(tr[sf].values, tr['spread_target'].values)
        last_model = m
        tt['pred'] = m.predict(tt[sf].values)

        sur = tt['pred'] <= -3
        defi = tt['pred'] >= 3
        act = tt[sur | defi].copy()
        if len(act) < 5:
            continue
        s = sur.reindex(act.index, fill_value=False)
        d = defi.reindex(act.index, fill_value=False)
        act['pnl'] = 0.0
        act.loc[s, 'pnl'] = (act.loc[s, 'exec_bid'] - act.loc[s, 'imb_settlement_price']) * ENERGY
        act.loc[d, 'pnl'] = (act.loc[d, 'imb_settlement_price'] - act.loc[d, 'exec_ask']) * ENERGY
        all_trades.append(act[['pnl']])

    if not all_trades:
        print(f'  {label}: no trades')
        return

    trades = pd.concat(all_trades)
    daily = trades.groupby(trades.index.date)['pnl'].sum()
    sh = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0

    print(f'\n  {label} ({len(sf)} feats, train>={train_start}):')
    print(f'    {trades["pnl"].sum():+,.0f} EUR total, {daily.mean():+,.0f}/day, '
          f'Sharpe={sh:.1f}, Win={((daily > 0).mean()):.0%}')

    # Monthly
    trades['mo'] = trades.index.to_period('M')
    for mo in sorted(trades['mo'].unique()):
        sub = trades[trades['mo'] == mo]
        d = sub.groupby(sub.index.date)['pnl'].sum()
        print(f'      {mo}: {sub["pnl"].sum():>+8,.0f} ({sub["pnl"].sum() / len(d):>+4.0f}/d) '
              f'[{len(sub)}t, {len(d)}d]')

    # QH feature importance
    if last_model and 'da_price_qh' in sf:
        imp = pd.DataFrame({'feature': sf, 'importance': last_model.feature_importances_})
        imp['pct'] = imp['importance'] / imp['importance'].sum() * 100
        qh_imp = imp[imp['feature'].str.contains('da_price_qh')].sort_values('pct', ascending=False)
        print(f'    QH importance: {qh_imp["pct"].sum():.1f}%')
        for _, r in qh_imp.iterrows():
            print(f'      {r["feature"]:<30s} {r["pct"]:.1f}%')


def main():
    data = load_all_data()
    tml.TRAIN_END = '2026-04-15'
    tml.TEST_START = '2026-04-15'
    df, fc = build_features(data, LEAD)

    ob = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                     parse_dates=['delivery_start'])
    ob_120 = ob[ob['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'mid', 'spread']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']
    df = df.join(ob_120, how='left')
    df['imb_settlement_price'] = df['imb_settle_price']
    df['spread_target'] = df['imb_settlement_price'] - df['exec_mid']

    print("=" * 70)
    print("QH DA PRICE FEATURES TEST")
    print("Test: Jan-Apr 2026 | Training from Oct 2025 (native QH data)")
    print("=" * 70)

    # Test 1: baseline vs +QH, training from Oct 2025
    run_config(df, fc, "50 feat baseline", SELECTED_50, '2025-10-01')
    run_config(df, fc, "55 feat +QH", SELECTED_55, '2025-10-01')

    # Test 2: same but training from all data
    print(f'\n{"=" * 70}')
    print("COMPARISON: Training from Oct 2025 vs All data")
    print(f'{"=" * 70}')
    run_config(df, fc, "55 feat +QH (Oct+)", SELECTED_55, '2025-10-01')
    run_config(df, fc, "55 feat +QH (All)", SELECTED_55, '2024-01-01')
    run_config(df, fc, "50 feat baseline (Oct+)", SELECTED_50, '2025-10-01')
    run_config(df, fc, "50 feat baseline (All)", SELECTED_50, '2024-01-01')

    # Test 3: QH features only (DA price + QH diffs)
    QH_ONLY = ['da_price_qh', 'da_price_qh_diff_prev', 'da_price_qh_diff_next',
               'da_price_qh_dev_hourly', 'da_price_qh_rank',
               'hour_cos', 'hour_sin', 'dow_sin', 'dow_cos']
    print(f'\n{"=" * 70}')
    print("MINIMAL: QH price features + time only")
    print(f'{"=" * 70}')
    run_config(df, fc, "9 feat (QH+time)", QH_ONLY, '2025-10-01')


if __name__ == "__main__":
    main()
