"""
Deep investigation of arcsinh(scale=20) target transform vs baseline.
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
REPO_ROOT = BASE_DIR.parent
LEAD = 8

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                  reg_lambda=1.0, n_estimators=600, verbose=-1)

SIZE_MW = 5.0
QH = 0.25


def run_bt(t, pred_col, threshold):
    surplus = t[pred_col] <= -threshold
    deficit = t[pred_col] >= threshold
    sub = t[surplus | deficit].copy()
    if len(sub) < 10:
        return None
    s = surplus.reindex(sub.index, fill_value=False)
    d = deficit.reindex(sub.index, fill_value=False)
    sub['pnl'] = 0.0
    sub.loc[s, 'pnl'] = (sub.loc[s, 'exec_bid'] - sub.loc[s, 'imb_settlement_price']) * SIZE_MW * QH
    sub.loc[d, 'pnl'] = (sub.loc[d, 'imb_settlement_price'] - sub.loc[d, 'exec_ask']) * SIZE_MW * QH
    daily = sub.groupby(sub.index.date)['pnl'].sum()
    nd = len(daily)
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    wr = (sub['pnl'] > 0).mean()
    dd = (daily.cumsum() - daily.cumsum().cummax()).min()
    return {'n': len(sub), 'total': sub['pnl'].sum(), 'per_day': sub['pnl'].sum()/nd,
            'sharpe': sharpe, 'wr': wr, 'dd': dd}


def main():
    print("=" * 70)
    print("ARCSINH(20) DEEP INVESTIGATION")
    print("=" * 70)

    data = load_all_data()
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df, feature_cols = build_features(data, LEAD)

    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                          parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid','ask','spread','mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid','exec_ask','exec_spread','exec_mid']

    mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                       parse_dates=['timestamp_hour'], index_col='timestamp_hour')
    mkt = mkt[~mkt.index.duplicated(keep='last')]

    df['hour_ts'] = df.index.floor('h')
    df = df.join(mkt[['imb_settlement_price']], on='hour_ts', how='left')
    df = df.join(ob_120, how='left')
    df['spread_target'] = df['imb_settlement_price'] - df['exec_mid']

    train = df[df.index < '2026-02-01'].dropna(subset=['spread_target', f'proxy_lag{LEAD+1}'])
    train = train[train['imb_settlement_price'].abs() <= 5000]
    test = df[(df.index >= '2026-02-01') & (df.index < '2026-04-01')].dropna(subset=[f'proxy_lag{LEAD+1}'])
    test = test[test['spread_target'].notna()]

    print(f"\n[+] Train: {len(train)}, Test: {len(test)}, Features: {len(feature_cols)}")

    # Train both models
    SCALE = 20.0
    y_train_arc = np.arcsinh(train['spread_target'].values / SCALE)

    model_arc = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
    model_arc.fit(train[feature_cols].values, y_train_arc)

    model_base = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
    model_base.fit(train[feature_cols].values, train['spread_target'].values)

    pred_t = model_arc.predict(test[feature_cols].values)
    pred_orig = np.sinh(pred_t) * SCALE
    pred_base = model_base.predict(test[feature_cols].values)
    y_actual = test['spread_target'].values

    test = test.copy()
    test['pred_arcsinh'] = pred_orig
    test['pred_base'] = pred_base

    # ===== 1. Prediction distributions =====
    print("\n--- 1. Prediction distributions ---")
    for label, pred in [('Baseline', pred_base), ('Arcsinh(20)', pred_orig)]:
        print(f"  {label:12s}: mean={np.mean(pred):+.2f}, std={np.std(pred):.2f}, "
              f"min={np.min(pred):.1f}, max={np.max(pred):.1f}, "
              f"|pred|>3: {(np.abs(pred)>3).mean():.0%}, |pred|>10: {(np.abs(pred)>10).mean():.0%}")
    print(f"  Actual      : mean={np.mean(y_actual):+.2f}, std={np.std(y_actual):.2f}")

    # ===== 2. Direction accuracy by confidence decile =====
    print("\n--- 2. Direction accuracy by |prediction| quintile ---")
    nz = np.abs(y_actual) > 0.1
    for label, pred in [('Baseline', pred_base), ('Arcsinh(20)', pred_orig)]:
        abs_pred = np.abs(pred[nz])
        correct = (np.sign(pred[nz]) == np.sign(y_actual[nz]))
        try:
            bins = pd.qcut(abs_pred, q=5, duplicates='drop')
            df_tmp = pd.DataFrame({'bin': bins, 'correct': correct})
            print(f"  {label}:")
            for b, grp in df_tmp.groupby('bin', observed=True):
                print(f"    |pred| {str(b):20s}: acc={grp['correct'].mean():.1%} (n={len(grp)})")
        except Exception as e:
            print(f"  {label}: error {e}")

    # ===== 3. Direction accuracy by hour =====
    print("\n--- 3. Direction accuracy by hour ---")
    test_h = test.copy()
    test_h['hour'] = test_h.index.hour
    nz_mask = test_h['spread_target'].abs() > 0.1
    for label, col in [('Baseline', 'pred_base'), ('Arcsinh(20)', 'pred_arcsinh')]:
        test_h['correct'] = (np.sign(test_h[col]) == np.sign(test_h['spread_target']))
        by_hour = test_h[nz_mask].groupby('hour')['correct'].mean()
        overall = test_h[nz_mask]['correct'].mean()
        best5 = by_hour.nlargest(5)
        worst5 = by_hour.nsmallest(5)
        print(f"  {label} (overall={overall:.1%}):")
        print(f"    Best:  {'  '.join(f'H{h}={v:.0%}' for h,v in best5.items())}")
        print(f"    Worst: {'  '.join(f'H{h}={v:.0%}' for h,v in worst5.items())}")

    # ===== 4. Monthly breakdown =====
    print("\n--- 4. Monthly direction accuracy ---")
    test_m = test.copy()
    test_m['month'] = test_m.index.to_period('M')
    nz_mask = test_m['spread_target'].abs() > 0.1
    for label, col in [('Baseline', 'pred_base'), ('Arcsinh(20)', 'pred_arcsinh')]:
        test_m['correct'] = (np.sign(test_m[col]) == np.sign(test_m['spread_target']))
        by_month = test_m[nz_mask].groupby('month')['correct'].mean()
        print(f"  {label}: {'  '.join(f'{m}={v:.1%}' for m,v in by_month.items())}")

    # ===== 5. Full threshold sweep =====
    print("\n--- 5. P&L threshold sweep (5 MW) ---")
    test_bt = test[test['exec_spread'].notna() & (test['exec_spread'] <= 10)].copy()
    n_days = test_bt.index.normalize().nunique()

    header = f"  {'Thresh':>8s} | {'Baseline':>40s} | {'Arcsinh(20)':>40s}"
    print(header)
    print(f"  {'-'*8} | {'-'*40} | {'-'*40}")
    for thresh in [1, 2, 3, 4, 5, 8, 10, 15]:
        rb = run_bt(test_bt, 'pred_base', thresh)
        ra = run_bt(test_bt, 'pred_arcsinh', thresh)
        if rb and ra:
            print(f"  |p|>={thresh:<4d} | {rb['n']:4d}t {rb['per_day']:+7.0f}/d Sh={rb['sharpe']:5.1f} W={rb['wr']:.0%} DD={rb['dd']:+.0f}"
                  f" | {ra['n']:4d}t {ra['per_day']:+7.0f}/d Sh={ra['sharpe']:5.1f} W={ra['wr']:.0%} DD={ra['dd']:+.0f}")
        elif rb:
            print(f"  |p|>={thresh:<4d} | {rb['n']:4d}t {rb['per_day']:+7.0f}/d Sh={rb['sharpe']:5.1f}"
                  f" | too few trades")

    # ===== 6. Feature importance comparison =====
    print("\n--- 6. Top 20 features: arcsinh vs baseline ---")
    imp_a = pd.DataFrame({'feature': feature_cols, 'arcsinh': model_arc.feature_importances_})
    imp_b = pd.DataFrame({'feature': feature_cols, 'baseline': model_base.feature_importances_})
    imp = imp_a.merge(imp_b, on='feature')
    imp['arc_pct'] = imp['arcsinh'] / imp['arcsinh'].sum() * 100
    imp['base_pct'] = imp['baseline'] / imp['baseline'].sum() * 100
    imp['diff'] = imp['arc_pct'] - imp['base_pct']
    imp = imp.sort_values('arc_pct', ascending=False)
    print(f"  {'Feature':30s} {'Arc%':>8s} {'Base%':>8s} {'Diff':>7s}")
    for _, r in imp.head(20).iterrows():
        marker = ' <<' if abs(r['diff']) > 0.3 else ''
        print(f"  {r['feature']:30s} {r['arc_pct']:7.2f}% {r['base_pct']:7.2f}% {r['diff']:+6.2f}%{marker}")

    # Biggest changes
    print("\n  Biggest importance shifts (arcsinh vs baseline):")
    imp_sorted = imp.sort_values('diff', ascending=False)
    for _, r in imp_sorted.head(5).iterrows():
        print(f"    UP:   {r['feature']:30s} {r['diff']:+.2f}%")
    for _, r in imp_sorted.tail(5).iterrows():
        print(f"    DOWN: {r['feature']:30s} {r['diff']:+.2f}%")

    # ===== 7. Per-trade comparison =====
    print("\n--- 7. Trade overlap analysis (threshold=3) ---")
    test_bt['pnl_base'] = 0.0
    test_bt['pnl_arc'] = 0.0
    for col, pnl_col in [('pred_base', 'pnl_base'), ('pred_arcsinh', 'pnl_arc')]:
        surplus = test_bt[col] <= -3
        deficit = test_bt[col] >= 3
        s = surplus
        d = deficit
        test_bt.loc[s, pnl_col] = (test_bt.loc[s, 'exec_bid'] - test_bt.loc[s, 'imb_settlement_price']) * SIZE_MW * QH
        test_bt.loc[d, pnl_col] = (test_bt.loc[d, 'imb_settlement_price'] - test_bt.loc[d, 'exec_ask']) * SIZE_MW * QH

    base_active = test_bt['pnl_base'] != 0
    arc_active = test_bt['pnl_arc'] != 0
    both = base_active & arc_active
    only_base = base_active & ~arc_active
    only_arc = ~base_active & arc_active

    print(f"  Both trade:    {both.sum()} periods, base={test_bt.loc[both,'pnl_base'].sum():+,.0f}, arc={test_bt.loc[both,'pnl_arc'].sum():+,.0f}")
    print(f"  Only baseline: {only_base.sum()} periods, pnl={test_bt.loc[only_base,'pnl_base'].sum():+,.0f}")
    print(f"  Only arcsinh:  {only_arc.sum()} periods, pnl={test_bt.loc[only_arc,'pnl_arc'].sum():+,.0f}")

    same_dir = both & (np.sign(test_bt['pred_base']) == np.sign(test_bt['pred_arcsinh']))
    diff_dir = both & (np.sign(test_bt['pred_base']) != np.sign(test_bt['pred_arcsinh']))
    print(f"  Same direction: {same_dir.sum()}")
    print(f"  Diff direction: {diff_dir.sum()}, base P&L={test_bt.loc[diff_dir,'pnl_base'].sum():+,.0f}, "
          f"arc P&L={test_bt.loc[diff_dir,'pnl_arc'].sum():+,.0f}")

    # ===== 8. Scale sweep =====
    print("\n--- 8. Arcsinh scale sweep (threshold=3) ---")
    print(f"  {'Scale':>6s} {'DirAcc':>7s} {'r':>6s} {'Trades':>7s} {'EUR/day':>8s} {'Sharpe':>7s} {'WinR':>5s}")
    for scale in [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 100]:
        y_t = np.arcsinh(train['spread_target'].values / scale)
        m = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        m.fit(train[feature_cols].values, y_t)
        p_t = m.predict(test_bt[feature_cols].values)
        p_o = np.sinh(p_t) * scale
        nz_s = np.abs(test_bt['spread_target'].values) > 0.1
        dacc = (np.sign(p_o[nz_s]) == np.sign(test_bt['spread_target'].values[nz_s])).mean()
        r_val = np.corrcoef(p_o, test_bt['spread_target'].values)[0, 1]
        bt = run_bt(test_bt.assign(pred=p_o), 'pred', 3)
        if bt:
            print(f"  {scale:6d} {dacc:6.1%} {r_val:6.3f} {bt['n']:7d} {bt['per_day']:+8.0f} {bt['sharpe']:7.1f} {bt['wr']:5.0%}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
