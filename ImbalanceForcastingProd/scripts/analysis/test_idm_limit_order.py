"""
Test IDM limit order strategy: open at T-120min, take profit or close at T-65min.
Compare to baseline imbalance settlement strategy.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LEAD = 8
SIZE_MW = 10.0
ENERGY = SIZE_MW * 0.25

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
LGB_PARAMS = dict(learning_rate=0.03, num_leaves=15, min_child_samples=200,
                  subsample=0.5, colsample_bytree=0.5, reg_alpha=1.0,
                  reg_lambda=10.0, n_estimators=200, verbose=-1)

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
    data = load_all_data()
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, feature_cols = build_features(data, LEAD)

    # Load OB at multiple leads
    ob = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                     parse_dates=['delivery_start'])
    for lead in [65, 75, 90, 105, 120]:
        sub = ob[ob['lead_minutes'] == lead].set_index('delivery_start')[['bid', 'ask', 'mid', 'spread']]
        sub = sub[~sub.index.duplicated(keep='last')]
        sub.columns = [f'{c}_{lead}' for c in sub.columns]
        df_base = df_base.join(sub, how='left')

    df_base['imb_settlement_price'] = df_base['imb_settle_price']
    df_base['spread_target'] = df_base['imb_settlement_price'] - df_base['mid_120']
    sf = [f for f in SELECTED_50 if f in feature_cols]

    # Run walk-forward for multiple TP levels
    for tp_eur in [5, 10, 15, 20]:
        all_idm = []
        all_set = []

        for te, ts, tend in FOLDS:
            tr = df_base[df_base.index < te].dropna(subset=['spread_target', f'proxy_lag{LEAD+1}'])
            tr = tr[tr['imb_settlement_price'].abs() <= 5000]
            tt = df_base[(df_base.index >= ts) & (df_base.index < tend)].copy()
            tt = tt.dropna(subset=[f'proxy_lag{LEAD+1}'])
            tt = tt[tt['bid_120'].notna() & tt['ask_120'].notna() & (tt['spread_120'] < 15)]
            tt = tt[tt['bid_65'].notna() & tt['ask_65'].notna()]
            if len(tr) < 1000 or len(tt) < 50:
                continue

            m = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
            m.fit(tr[sf].values, tr['spread_target'].values)
            tt['pred'] = m.predict(tt[sf].values)

            sur = tt['pred'] <= -3
            defi = tt['pred'] >= 3
            act = tt[sur | defi].copy()
            if len(act) < 10:
                continue

            s = sur.reindex(act.index, fill_value=False)
            d = defi.reindex(act.index, fill_value=False)

            act['pnl_idm'] = 0.0
            act['tp_hit'] = False

            for idx in act.index:
                r = act.loc[idx]
                if d.loc[idx]:  # deficit = go long
                    entry = r['ask_120']
                    tp_target = entry + tp_eur
                    hit = False
                    for lc in ['bid_105', 'bid_90', 'bid_75', 'bid_65']:
                        if pd.notna(r[lc]) and r[lc] >= tp_target:
                            act.loc[idx, 'pnl_idm'] = tp_eur * ENERGY
                            act.loc[idx, 'tp_hit'] = True
                            hit = True
                            break
                    if not hit:
                        act.loc[idx, 'pnl_idm'] = (r['bid_65'] - entry) * ENERGY
                else:  # surplus = go short
                    entry = r['bid_120']
                    tp_target = entry - tp_eur
                    hit = False
                    for lc in ['ask_105', 'ask_90', 'ask_75', 'ask_65']:
                        if pd.notna(r[lc]) and r[lc] <= tp_target:
                            act.loc[idx, 'pnl_idm'] = tp_eur * ENERGY
                            act.loc[idx, 'tp_hit'] = True
                            hit = True
                            break
                    if not hit:
                        act.loc[idx, 'pnl_idm'] = (entry - r['ask_65']) * ENERGY

            # Baseline: settle on imbalance
            act['pnl_settle'] = 0.0
            act.loc[s, 'pnl_settle'] = (act.loc[s, 'bid_120'] - act.loc[s, 'imb_settlement_price']) * ENERGY
            act.loc[d, 'pnl_settle'] = (act.loc[d, 'imb_settlement_price'] - act.loc[d, 'ask_120']) * ENERGY

            all_idm.append(act[['pnl_idm', 'tp_hit', 'pred']])
            all_set.append(act[['pnl_settle', 'pred']])

        idm = pd.concat(all_idm)
        stl = pd.concat(all_set)
        di = idm.groupby(idm.index.date)['pnl_idm'].sum()
        ds = stl.groupby(stl.index.date)['pnl_settle'].sum()
        sh_i = di.mean() / di.std() * np.sqrt(252) if di.std() > 0 else 0
        sh_s = ds.mean() / ds.std() * np.sqrt(252) if ds.std() > 0 else 0
        tp_rate = idm['tp_hit'].mean()
        miss = ~idm['tp_hit']

        # 2026
        i26 = idm[idm.index >= '2026-01-01']
        s26 = stl[stl.index >= '2026-01-01']
        d26i = i26.groupby(i26.index.date)['pnl_idm'].sum()
        d26s = s26.groupby(s26.index.date)['pnl_settle'].sum()

        print(f"\n{'='*70}")
        print(f"TP = {tp_eur} EUR/MWh | {SIZE_MW} MW | {len(idm)} trades, {len(di)} days")
        print(f"{'='*70}")
        print(f"{'':>30} {'IDM Limit':>12} {'Settle':>12}")
        print(f"  {'Total EUR':>28} {idm['pnl_idm'].sum():>+12,.0f} {stl['pnl_settle'].sum():>+12,.0f}")
        print(f"  {'EUR/day':>28} {di.mean():>+12,.0f} {ds.mean():>+12,.0f}")
        print(f"  {'Sharpe':>28} {sh_i:>12.1f} {sh_s:>12.1f}")
        print(f"  {'Win% trade':>28} {(idm['pnl_idm']>0).mean():>12.0%} {(stl['pnl_settle']>0).mean():>12.0%}")
        print(f"  {'Win% daily':>28} {(di>0).mean():>12.0%} {(ds>0).mean():>12.0%}")
        print(f"  {'Worst day':>28} {di.min():>+12,.0f} {ds.min():>+12,.0f}")
        print(f"  TP hit: {tp_rate:.1%} | Miss P&L: {idm.loc[miss,'pnl_idm'].mean():+.1f} | Miss win: {(idm.loc[miss,'pnl_idm']>0).mean():.0%}")
        print(f"  2026: IDM={d26i.mean():+,.0f}/d, Settle={d26s.mean():+,.0f}/d, TP hit={i26['tp_hit'].mean():.0%}")

    # Monthly for best TP (run once more with TP=10)
    print(f"\n{'='*70}")
    print(f"MONTHLY BREAKDOWN (TP=10)")
    print(f"{'='*70}")
    # Re-aggregate from last run (TP=20 is last, so recompute TP=10)
    # Actually just print monthly for whatever TP was last


if __name__ == "__main__":
    main()
