"""
Test QH DA price features with relaxed OB filter.
Instead of requiring both bid+ask with tight spread, only require
the execution price for our trade direction exists.
"""
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

LP = dict(learning_rate=0.03, num_leaves=15, min_child_samples=200,
          subsample=0.5, colsample_bytree=0.5, reg_alpha=1.0,
          reg_lambda=10.0, n_estimators=200, verbose=-1)

FOLDS = [
    ('2025-12-01', '2025-12-01', '2026-01-01'),
    ('2026-01-01', '2026-01-01', '2026-02-01'),
    ('2026-02-01', '2026-02-01', '2026-03-01'),
    ('2026-03-01', '2026-03-01', '2026-04-01'),
    ('2026-04-01', '2026-04-01', '2026-04-13'),
]


def relaxed_backtest(test_df, pred_col, threshold=3):
    """P&L with relaxed filter: only need exec price for our direction."""
    t = test_df.copy()
    surplus = t[pred_col] <= -threshold
    deficit = t[pred_col] >= threshold

    # For surplus (sell): need bid. For deficit (buy): need ask.
    can_sell = surplus & t['exec_bid'].notna()
    can_buy = deficit & t['exec_ask'].notna()
    active = t[can_sell | can_buy].copy()

    if len(active) < 5:
        return None

    s = can_sell.reindex(active.index, fill_value=False)
    d = can_buy.reindex(active.index, fill_value=False)
    active['pnl'] = 0.0
    active.loc[s, 'pnl'] = (active.loc[s, 'exec_bid'] - active.loc[s, 'imb_settlement_price']) * ENERGY
    active.loc[d, 'pnl'] = (active.loc[d, 'imb_settlement_price'] - active.loc[d, 'exec_ask']) * ENERGY

    daily = active.groupby(active.index.date)['pnl'].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    return {
        'total': active['pnl'].sum(),
        'per_day': daily.mean(),
        'sharpe': sharpe,
        'win_daily': (daily > 0).mean(),
        'n_trades': len(active),
        'n_days': len(daily),
        'daily': daily,
    }


def strict_backtest(test_df, pred_col, threshold=3):
    """Original strict filter: need both bid+ask with tight spread."""
    t = test_df.copy()
    t = t[t['exec_bid'].notna() & t['exec_ask'].notna() & (t['exec_spread'] <= 15)]
    surplus = t[pred_col] <= -threshold
    deficit = t[pred_col] >= threshold
    active = t[surplus | deficit].copy()

    if len(active) < 5:
        return None

    s = surplus.reindex(active.index, fill_value=False)
    d = deficit.reindex(active.index, fill_value=False)
    active['pnl'] = 0.0
    active.loc[s, 'pnl'] = (active.loc[s, 'exec_bid'] - active.loc[s, 'imb_settlement_price']) * ENERGY
    active.loc[d, 'pnl'] = (active.loc[d, 'imb_settlement_price'] - active.loc[d, 'exec_ask']) * ENERGY

    daily = active.groupby(active.index.date)['pnl'].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    return {
        'total': active['pnl'].sum(),
        'per_day': daily.mean(),
        'sharpe': sharpe,
        'win_daily': (daily > 0).mean(),
        'n_trades': len(active),
        'n_days': len(daily),
        'daily': daily,
    }


def run_config(df, fc, label, feats, bt_fn, train_start='2025-10-01'):
    sf = [f for f in feats if f in fc]
    all_trades = []

    for te, ts, tend in FOLDS:
        tr = df[(df.index >= train_start) & (df.index < te)].dropna(
            subset=['spread_target', 'proxy_lag9'])
        tr = tr[tr['imb_settlement_price'].abs() <= 5000]
        tt = df[(df.index >= ts) & (df.index < tend)].copy()
        tt = tt.dropna(subset=['proxy_lag9'])
        # Only require settlement price exists (for target/P&L)
        tt = tt[tt['imb_settlement_price'].notna()]

        if len(tr) < 500 or len(tt) < 30:
            continue

        m = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LP)
        m.fit(tr[sf].values, tr['spread_target'].values)
        tt['pred'] = m.predict(tt[sf].values)

        result = bt_fn(tt, 'pred')
        if result:
            print(f'    {ts[:7]}: {result["total"]:>+8,.0f} ({result["per_day"]:>+4.0f}/d) '
                  f'[{result["n_trades"]}t, {result["n_days"]}d]')
            all_trades.append(result['daily'])

    if not all_trades:
        print(f'  {label}: no trades')
        return

    combined = pd.concat(all_trades)
    sh = combined.mean() / combined.std() * np.sqrt(252) if combined.std() > 0 else 0
    print(f'  --> {label}: {combined.mean():+,.0f}/day, Sharpe={sh:.1f}, '
          f'Win={((combined > 0).mean()):.0%}, {len(combined)} days\n')


def main():
    data = load_all_data()
    tml.TRAIN_END = '2026-04-15'
    tml.TEST_START = '2026-04-15'
    df, fc = build_features(data, LEAD)

    ob = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                     parse_dates=['delivery_start'])
    ob120 = ob[ob['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'mid', 'spread']]
    ob120 = ob120[~ob120.index.duplicated(keep='last')]
    ob120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']
    df = df.join(ob120, how='left')
    df['imb_settlement_price'] = df['imb_settle_price']
    # Use mid for target even with one-sided book
    df['spread_target'] = df['imb_settlement_price'] - df['exec_mid']

    print("=" * 70)
    print("QH FEATURES + RELAXED OB FILTER")
    print("Train: expanding from Oct 2025 | Test: Dec 2025 - Apr 2026")
    print("=" * 70)

    # Data coverage check
    for start, end in [('2025-10', '2025-11'), ('2025-11', '2025-12'), ('2025-12', '2026-01'),
                       ('2026-01', '2026-02'), ('2026-02', '2026-03'), ('2026-03', '2026-04'), ('2026-04', '2026-05')]:
        sub = df[(df.index >= start + '-01') & (df.index < end + '-01')]
        has_mid = sub['exec_mid'].notna().sum()
        has_bid = sub['exec_bid'].notna().sum()
        has_ask = sub['exec_ask'].notna().sum()
        has_settle = sub['imb_settlement_price'].notna().sum()
        print(f'  {start}: {len(sub)} rows, mid={has_mid}, bid={has_bid}, ask={has_ask}, settle={has_settle}')

    print(f'\n--- STRICT FILTER (both bid+ask, spread<15) ---')
    run_config(df, fc, '50 baseline (strict)', SELECTED_50, strict_backtest)
    run_config(df, fc, '55 +QH (strict)', SELECTED_50 + QH_FEATS, strict_backtest)

    print(f'--- RELAXED FILTER (only need exec price for direction) ---')
    run_config(df, fc, '50 baseline (relaxed)', SELECTED_50, relaxed_backtest)
    run_config(df, fc, '55 +QH (relaxed)', SELECTED_50 + QH_FEATS, relaxed_backtest)

    # Also test with all training data
    print(f'--- RELAXED + ALL TRAINING DATA ---')
    run_config(df, fc, '50 baseline (relaxed, all)', SELECTED_50, relaxed_backtest, '2024-01-01')
    run_config(df, fc, '55 +QH (relaxed, all)', SELECTED_50 + QH_FEATS, relaxed_backtest, '2024-01-01')


if __name__ == "__main__":
    main()
