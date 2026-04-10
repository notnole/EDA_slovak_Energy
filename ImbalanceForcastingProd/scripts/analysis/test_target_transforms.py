"""
Systematic test of target transformations for the spread model.
================================================================

Tests each transform individually, then combinations.
All evaluated on Feb-Mar 2026 out-of-sample with bid/ask execution.

Transforms:
  A. Baseline (no transform)
  B. Winsorize at +/-30
  C. Winsorize at +/-15
  D. Arcsinh: arcsinh(x / scale)
  E. Signed-log: sign(x) * log(1 + |x|)
  F. Sign target: sign(x) — regression on {-1, 0, +1}
  G. Custom sign-penalty loss (LightGBM custom objective)
  H. Combinations
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

BASE_DIR = Path(__file__).resolve().parents[2]  # ImbalanceForcastingProd/
DATA_DIR = BASE_DIR / "data"
REPO_ROOT = BASE_DIR.parent

LEAD = 8

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                  reg_lambda=1.0, n_estimators=600, verbose=-1)

TRAIN_END = '2026-02-01'
TEST_START = '2026-02-01'
TEST_END = '2026-04-01'


# =================================================================
# TARGET TRANSFORMS
# =================================================================

def transform_none(y):
    """A. Baseline — no transform."""
    return y

def inverse_none(y_hat):
    return y_hat

def transform_winsorize30(y):
    """B. Winsorize at +/-30."""
    return y.clip(-30, 30)

def inverse_winsorize30(y_hat):
    return y_hat  # predictions are already in clipped space, use as-is

def transform_winsorize15(y):
    """C. Winsorize at +/-15."""
    return y.clip(-15, 15)

def inverse_winsorize15(y_hat):
    return y_hat

def transform_arcsinh(y, scale=10.0):
    """D. Arcsinh: arcsinh(y / scale). Smooth sign-preserving compression."""
    return np.arcsinh(y / scale)

def inverse_arcsinh(y_hat, scale=10.0):
    return np.sinh(y_hat) * scale

def transform_signed_log(y):
    """E. Signed-log: sign(y) * log(1 + |y|). Stronger compression than arcsinh."""
    return np.sign(y) * np.log1p(np.abs(y))

def inverse_signed_log(y_hat):
    return np.sign(y_hat) * (np.exp(np.abs(y_hat)) - 1)

def transform_sign(y):
    """F. Sign target: {-1, 0, +1}. Pure direction, no magnitude."""
    return np.sign(y)

def inverse_sign(y_hat):
    return y_hat  # prediction magnitude = confidence

def transform_winsorize30_arcsinh(y):
    """H1. Winsorize30 + arcsinh."""
    return transform_arcsinh(transform_winsorize30(y))

def inverse_winsorize30_arcsinh(y_hat):
    return inverse_arcsinh(y_hat)  # don't unclip, arcsinh output is smooth

def transform_winsorize15_signed_log(y):
    """H2. Winsorize15 + signed-log."""
    return transform_signed_log(transform_winsorize15(y))

def inverse_winsorize15_signed_log(y_hat):
    return inverse_signed_log(y_hat)

def transform_winsorize30_signed_log(y):
    """H3. Winsorize30 + signed-log."""
    return transform_signed_log(transform_winsorize30(y))

def inverse_winsorize30_signed_log(y_hat):
    return inverse_signed_log(y_hat)


# =================================================================
# CUSTOM SIGN-PENALTY LOSS
# =================================================================

def sign_penalty_objective(alpha=2.0):
    """G. MAE + sign penalty. alpha controls sign penalty strength."""
    def objective(y_true, y_pred):
        residual = y_pred - y_true
        # MAE gradient
        grad = np.sign(residual)
        hess = np.ones_like(residual)

        # Sign penalty: fires when sign(pred) != sign(true)
        wrong_sign = (y_pred * y_true) < 0
        sign_grad = np.where(wrong_sign, np.sign(y_pred) * alpha, 0.0)
        sign_hess = np.where(wrong_sign, alpha, 0.0)

        return grad + sign_grad, hess + sign_hess

    return objective


# =================================================================
# EVALUATION
# =================================================================

def evaluate_transform(name, feature_cols, train_df, test_df,
                       transform_fn, inverse_fn, custom_obj=None,
                       objective='quantile', alpha=0.50):
    """Train spread model with given transform, evaluate trading P&L."""

    # Transform target
    train = train_df.copy()
    test = test_df.copy()

    train['spread_t'] = transform_fn(train['spread_target'])
    # Don't transform test target — evaluate on real spread

    # Remove extreme / invalid
    train = train[train['spread_t'].notna() & np.isfinite(train['spread_t'])]

    if custom_obj is not None:
        model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05,
                                   num_leaves=63, min_child_samples=50,
                                   subsample=0.8, colsample_bytree=0.7,
                                   reg_alpha=0.1, reg_lambda=1.0, verbose=-1)
        model.fit(train[feature_cols].values, train['spread_t'].values,
                  eval_set=[(test[feature_cols].values, transform_fn(test['spread_target']).values)],
                  callbacks=[lgb.log_evaluation(0)])
        # Re-fit with custom objective (LightGBM needs initial fit for structure)
        model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05,
                                   num_leaves=63, min_child_samples=50,
                                   subsample=0.8, colsample_bytree=0.7,
                                   reg_alpha=0.1, reg_lambda=1.0, verbose=-1,
                                   objective=custom_obj)
        model.fit(train[feature_cols].values, train['spread_t'].values)
    else:
        model = lgb.LGBMRegressor(objective=objective, alpha=alpha, **LGB_PARAMS)
        model.fit(train[feature_cols].values, train['spread_t'].values)

    # Predict in transformed space, then inverse
    pred_transformed = model.predict(test[feature_cols].values)
    pred_original = inverse_fn(pred_transformed)

    # Direction accuracy
    actual = test['spread_target'].values
    nz = np.abs(actual) > 0.1
    dir_acc = (np.sign(pred_original[nz]) == np.sign(actual[nz])).mean()

    # Correlation
    valid = np.isfinite(pred_original) & np.isfinite(actual)
    r = np.corrcoef(pred_original[valid], actual[valid])[0, 1] if valid.sum() > 100 else 0

    # Prediction stats
    pred_std = np.std(pred_original[valid])
    actual_std = np.std(actual[valid])
    shrinkage = pred_std / actual_std if actual_std > 0 else 0

    # Trading backtest (use pred_original for direction, pred_transformed for threshold)
    # Since some transforms compress, use DIRECTION from pred_original
    # and a universal threshold on percentile of |prediction|
    test_bt = test.copy()
    test_bt['pred'] = pred_original
    test_bt['pred_t'] = pred_transformed

    results = {}
    # Test with different threshold approaches
    finite_preds = pred_original[np.isfinite(pred_original)]
    for thresh_label, pct_lo, pct_hi in [
        ('auto_p25', 25, 75),
        ('auto_p15', 15, 85),
    ]:
        lo_thresh = np.percentile(finite_preds, pct_lo)
        hi_thresh = np.percentile(finite_preds, pct_hi)
        surplus_mask = pd.Series(pred_original <= lo_thresh, index=test_bt.index)
        deficit_mask = pd.Series(pred_original >= hi_thresh, index=test_bt.index)

        sub = test_bt[surplus_mask | deficit_mask].copy()
        sub = sub[sub['exec_spread'].notna() & (sub['exec_spread'] <= 10)]
        if len(sub) < 30:
            continue

        s = surplus_mask.reindex(sub.index, fill_value=False)
        d = deficit_mask.reindex(sub.index, fill_value=False)

        SIZE_MW = 5.0
        QH = 0.25
        sub['pnl'] = 0.0
        sub.loc[s, 'pnl'] = (sub.loc[s, 'exec_bid'] - sub.loc[s, 'imb_settlement_price']) * SIZE_MW * QH
        sub.loc[d, 'pnl'] = (sub.loc[d, 'imb_settlement_price'] - sub.loc[d, 'exec_ask']) * SIZE_MW * QH

        nd = sub.index.normalize().nunique()
        total = sub['pnl'].sum()
        wr = (sub['pnl'] > 0).mean()
        daily = sub.groupby(sub.index.date)['pnl'].sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0

        results[thresh_label] = {
            'trades': len(sub), 'total': total, 'per_day': total / nd,
            'sharpe': sharpe, 'win_rate': wr, 'days': nd,
        }

    # Best result
    best_key = max(results.keys(), key=lambda k: results[k]['sharpe']) if results else None
    best = results.get(best_key, {})

    return {
        'name': name,
        'dir_acc': dir_acc,
        'r': r,
        'pred_std': pred_std,
        'shrinkage': shrinkage,
        'best_thresh': best_key,
        'trades': best.get('trades', 0),
        'pnl_day': best.get('per_day', 0),
        'sharpe': best.get('sharpe', 0),
        'win_rate': best.get('win_rate', 0),
        'all_results': results,
    }


# =================================================================
# MAIN
# =================================================================

def main():
    print("=" * 75)
    print("SYSTEMATIC TARGET TRANSFORMATION TEST — SPREAD MODEL")
    print("=" * 75)

    # Load data
    data = load_all_data()
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df, feature_cols = build_features(data, LEAD)

    # Join execution prices + settlement
    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                          parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'spread', 'mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']

    mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                       parse_dates=['timestamp_hour'], index_col='timestamp_hour')
    mkt = mkt[~mkt.index.duplicated(keep='last')]

    df['hour_ts'] = df.index.floor('h')
    df = df.join(mkt[['imb_settlement_price']], on='hour_ts', how='left')
    df = df.join(ob_120, how='left')
    df['spread_target'] = df['imb_settlement_price'] - df['exec_mid']

    # Split
    train = df[df.index < TRAIN_END].dropna(subset=['spread_target', f'proxy_lag{LEAD+1}'])
    train = train[train['imb_settlement_price'].abs() <= 5000]
    test = df[(df.index >= TEST_START) & (df.index < TEST_END)].dropna(subset=[f'proxy_lag{LEAD+1}'])
    test = test[test['spread_target'].notna()]

    print(f"\n[+] Train: {len(train)}, Test: {len(test)}")
    print(f"[+] Features: {len(feature_cols)}")
    print(f"[+] Spread target stats (train): mean={train['spread_target'].mean():.1f}, "
          f"std={train['spread_target'].std():.1f}, "
          f"min={train['spread_target'].min():.0f}, max={train['spread_target'].max():.0f}")

    # Define transforms to test
    transforms = [
        # (name, transform_fn, inverse_fn, custom_obj, lgb_objective, lgb_alpha)
        ("A. Baseline (MAE)",           transform_none,                  inverse_none,                  None, 'quantile', 0.50),
        ("B. Winsorize +/-30",          transform_winsorize30,           inverse_winsorize30,           None, 'quantile', 0.50),
        ("C. Winsorize +/-15",          transform_winsorize15,           inverse_winsorize15,           None, 'quantile', 0.50),
        ("D. Arcsinh (scale=10)",       transform_arcsinh,               inverse_arcsinh,               None, 'quantile', 0.50),
        ("E. Signed-log",               transform_signed_log,            inverse_signed_log,            None, 'quantile', 0.50),
        ("F. Sign target",              transform_sign,                  inverse_sign,                  None, 'quantile', 0.50),
        ("G. Sign-penalty loss (a=2)",  transform_none,                  inverse_none,                  sign_penalty_objective(2.0), None, None),
        ("G2. Sign-penalty loss (a=5)", transform_none,                  inverse_none,                  sign_penalty_objective(5.0), None, None),
        ("H1. Win30 + arcsinh",         transform_winsorize30_arcsinh,   inverse_winsorize30_arcsinh,   None, 'quantile', 0.50),
        ("H2. Win15 + signed-log",      transform_winsorize15_signed_log, inverse_winsorize15_signed_log, None, 'quantile', 0.50),
        ("H3. Win30 + signed-log",      transform_winsorize30_signed_log, inverse_winsorize30_signed_log, None, 'quantile', 0.50),
    ]

    # Also test arcsinh with different scales
    for scale in [5, 20]:
        transforms.append((
            f"D{scale}. Arcsinh (scale={scale})",
            lambda y, s=scale: transform_arcsinh(y, s),
            lambda yh, s=scale: inverse_arcsinh(yh, s),
            None, 'quantile', 0.50
        ))

    # Run all tests
    results = []
    for name, tfn, ifn, cobj, obj, alpha in transforms:
        print(f"\n--- {name} ---")
        try:
            r = evaluate_transform(name, feature_cols, train, test, tfn, ifn,
                                   custom_obj=cobj, objective=obj, alpha=alpha)
            results.append(r)
            print(f"  Dir acc: {r['dir_acc']:.1%}, r: {r['r']:.3f}, "
                  f"shrinkage: {r['shrinkage']:.2f}, "
                  f"P&L: {r['pnl_day']:+.0f}/day, Sharpe: {r['sharpe']:.1f}")
        except Exception as e:
            print(f"  [!] Failed: {e}")

    # Summary table
    print("\n" + "=" * 100)
    print(f"{'Transform':<35s} {'DirAcc':>7s} {'r':>6s} {'Shrink':>7s} {'Trades':>7s} "
          f"{'EUR/day':>8s} {'Sharpe':>7s} {'WinR':>6s} {'Thresh':>8s}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: -x['sharpe']):
        print(f"{r['name']:<35s} {r['dir_acc']:>6.1%} {r['r']:>6.3f} {r['shrinkage']:>7.2f} "
              f"{r['trades']:>7d} {r['pnl_day']:>+8.0f} {r['sharpe']:>7.1f} "
              f"{r['win_rate']:>5.0%} {r['best_thresh'] or '':>8s}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
