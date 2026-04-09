"""
BESS Weight Optimization Experiment
====================================
Tests whether adding DA price forecast improves BESS scheduling.

Candidate features (all ranked within-day, oriented so HIGH rank = EXPENSIVE hour):

  Group A (always available, 2024-2026):
    1. rank_load       - DAMAS load forecast
    2. rank_yest       - Yesterday same-hour DA price
    3. rank_7d         - 7-day ago same-hour DA price
    4. rank_2d         - 2-day ago same-hour DA price
    5. rank_load_grad  - Hour-to-hour load forecast gradient
    6. rank_solar_yest - Yesterday same-hour solar generation (DESCENDING: high solar=low rank=cheap)
    7. rank_30d_avg    - 30-day rolling average price by hour

  Group B (Sep 2025+ only):
    8. rank_da_fcst    - DA price forecast
    9. rank_da_resid   - DA forecast minus yesterday's price (incremental news)

Experiment design:
  Train:  Jan 2024 - day before first DA forecast  (Group A weight optimization)
  Test:   First DA forecast date - Jan 31, 2026    (all strategies compared)

Strategies compared on SAME test period:
  S1: Original 3-feature (fixed 40/35/25)           [baseline]
  S2: Optimized Group A  (weights from train period) [no DA forecast]
  S3: Augmented A+B      (time-series CV on test)    [with DA forecast]
  S4: DA forecast only   (standalone rank_da_fcst)   [pure forecast]

Usage:
  python bess_weight_experiment.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
import warnings
import sys

warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data')
OUT  = BASE / 'MarketPriceGap' / 'da_indicator'

sys.path.insert(0, str(OUT / 'scripts'))
from build_da_indicator import load_all_data, build_features
from bess_simulation import optimize_bess_dp, naive_schedule


# ============================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================

def load_da_forecasts():
    """Load DA price forecast data (hourly, Sep 2025+)."""
    print("[*] Loading DA price forecasts...")
    fpath = BASE / 'MarketPriceGap' / 'da_forecast_analysis' / 'data' / 'forecast_vs_actual_merged.csv'
    fcst = pd.read_csv(fpath)
    fcst['datetime'] = pd.to_datetime(fcst['datetime'])
    fcst['date'] = pd.to_datetime(fcst['date'])

    # Keep only rows with valid SK forecast1
    mask = fcst['sk_forecast1_spot'].notna()
    fcst = fcst[mask].copy()

    print(f"[+] DA forecasts: {len(fcst)} rows with valid SK forecast1")
    if len(fcst) > 0:
        print(f"    Range: {fcst['date'].min().date()} to {fcst['date'].max().date()}")
        print(f"    Dates: {fcst['date'].nunique()}")

    return fcst[['date', 'hour', 'sk_forecast1_spot']].copy()


def build_candidate_features(df, da_fcst):
    """Build all 9 candidate features ranked within-day."""
    print("[*] Building candidate features...")
    df = df.sort_values(['date', 'hour']).reset_index(drop=True)

    # ---- GROUP A: Always available ----

    # 1. rank_load: load forecast rank
    df['rank_load'] = df.groupby('date')['forecast_load_mw'].rank(
        method='first', na_option='bottom')

    # 2. rank_yest: yesterday same-hour price rank (already built by build_features)
    df['rank_yest'] = df.groupby('date')['da_price_yest_h'].rank(
        method='first', na_option='bottom')

    # 3. rank_7d: 7-day price rank (already built)
    df['rank_7d'] = df.groupby('date')['da_price_7d_h'].rank(
        method='first', na_option='bottom')

    # 4. rank_2d: 2-day ago price rank
    df = df.sort_values(['hour', 'date']).reset_index(drop=True)
    df['da_price_2d_h'] = df.groupby('hour')['da_price'].shift(2)
    df = df.sort_values(['date', 'hour']).reset_index(drop=True)
    df['rank_2d'] = df.groupby('date')['da_price_2d_h'].rank(
        method='first', na_option='bottom')

    # 5. rank_load_grad: hour-to-hour load forecast gradient
    df['load_grad'] = df.groupby('date')['forecast_load_mw'].diff()
    df['rank_load_grad'] = df.groupby('date')['load_grad'].rank(
        method='first', na_option='bottom')

    # 6. rank_solar_yest: yesterday same-hour solar (DESCENDING: high solar = low rank = cheap)
    if 'SolarActual' in df.columns:
        df = df.sort_values(['hour', 'date']).reset_index(drop=True)
        df['solar_yest_h'] = df.groupby('hour')['SolarActual'].shift(1)
        df = df.sort_values(['date', 'hour']).reset_index(drop=True)
        df['rank_solar_yest'] = df.groupby('date')['solar_yest_h'].rank(
            method='first', ascending=False, na_option='bottom')
    else:
        df['rank_solar_yest'] = np.nan

    # 7. rank_30d_avg: 30-day rolling average price by hour
    df = df.sort_values(['hour', 'date']).reset_index(drop=True)
    df['da_price_30d_h'] = df.groupby('hour')['da_price'].transform(
        lambda x: x.shift(1).rolling(30, min_periods=7).mean())
    df = df.sort_values(['date', 'hour']).reset_index(drop=True)
    df['rank_30d_avg'] = df.groupby('date')['da_price_30d_h'].rank(
        method='first', na_option='bottom')

    # ---- GROUP B: DA forecast period only ----

    # Merge DA forecasts
    df = df.merge(da_fcst, on=['date', 'hour'], how='left')

    # 8. rank_da_fcst: DA price forecast rank
    df['rank_da_fcst'] = df.groupby('date')['sk_forecast1_spot'].rank(
        method='first', na_option='bottom')

    # 9. rank_da_resid: forecast minus yesterday's price (incremental news)
    df['da_fcst_residual'] = df['sk_forecast1_spot'] - df['da_price_yest_h']
    df['rank_da_resid'] = df.groupby('date')['da_fcst_residual'].rank(
        method='first', na_option='bottom')

    # Target: actual price rank
    df['rank_actual'] = df.groupby('date')['da_price'].rank(
        method='first', na_option='bottom')

    # Availability flag
    df['has_da_fcst'] = df['sk_forecast1_spot'].notna()

    group_a = ['rank_load', 'rank_yest', 'rank_7d', 'rank_2d',
               'rank_load_grad', 'rank_solar_yest', 'rank_30d_avg']
    group_b = ['rank_da_fcst', 'rank_da_resid']

    # Report coverage
    for f in group_a + group_b:
        n = df[f].notna().sum()
        print(f"    {f:20s}: {n:>6d} valid ({n/len(df)*100:.1f}%)")

    return df, group_a, group_b


# ============================================================
# 2. WEIGHT OPTIMIZATION
# ============================================================

def optimize_weights(df_train, feature_cols, n_starts=25):
    """
    Find optimal feature weights minimizing rank MAE.
    Constraints: weights >= 0, sum(weights) = 1.
    """
    # Only complete days with valid price data
    day_lens = df_train.groupby('date').size()
    valid_dates = day_lens[day_lens >= 20].index
    days = df_train[df_train['date'].isin(valid_dates)].copy()

    X = days[feature_cols].fillna(12).values
    y = days['rank_actual'].values
    n_feats = len(feature_cols)

    def objective(w):
        pred = X @ w
        return np.mean(np.abs(pred - y))

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0, 1)] * n_feats

    best_result = None
    best_obj = np.inf

    for trial in range(n_starts):
        w0 = np.ones(n_feats) / n_feats if trial == 0 else np.random.dirichlet(np.ones(n_feats))
        try:
            result = minimize(objective, w0, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 500, 'ftol': 1e-10})
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return np.ones(n_feats) / n_feats, np.inf

    return best_result.x, best_result.fun


def cross_validate_weights(df_test, feature_cols, n_folds=3):
    """
    Time-series cross-validation (expanding window).
    Returns averaged weights and per-fold metrics.
    """
    dates = sorted(df_test['date'].unique())
    n_days = len(dates)
    fold_size = n_days // (n_folds + 1)

    print(f"[*] CV: {n_days} days, {n_folds} folds, ~{fold_size} days/fold")

    fold_results = []
    fold_weights = []

    for fold in range(n_folds):
        split_idx = fold_size * (fold + 1)
        test_end = min(split_idx + fold_size, n_days)

        if test_end <= split_idx:
            break

        train_dates = set(dates[:split_idx])
        test_dates = set(dates[split_idx:test_end])

        df_tr = df_test[df_test['date'].isin(train_dates)]
        df_te = df_test[df_test['date'].isin(test_dates)]

        if len(df_tr) < 200 or len(df_te) < 100:
            continue

        weights, train_obj = optimize_weights(df_tr, feature_cols)

        # Evaluate on test fold
        X_te = df_te[feature_cols].fillna(12).values
        y_te = df_te['rank_actual'].values
        test_mae = np.mean(np.abs(X_te @ weights - y_te))

        fold_weights.append(weights)
        fold_results.append({
            'fold': fold + 1,
            'train_days': len(train_dates),
            'test_days': len(test_dates),
            'train_mae': train_obj,
            'test_mae': test_mae,
        })
        print(f"    Fold {fold+1}: train={len(train_dates)}d, test={len(test_dates)}d, "
              f"train_MAE={train_obj:.3f}, test_MAE={test_mae:.3f}")

    if not fold_weights:
        print("[!] CV failed, using equal weights")
        return np.ones(len(feature_cols)) / len(feature_cols), []

    avg_weights = np.mean(fold_weights, axis=0)
    avg_weights = avg_weights / avg_weights.sum()
    return avg_weights, fold_results


# ============================================================
# 3. BESS SIMULATION
# ============================================================

def run_bess_sim(df, rank_col, terminal_penalty=20.0):
    """Run BESS simulation with persistent SoC using given rank column."""
    all_dates = sorted(df['date'].unique())

    results = []
    soc_ind = 1
    soc_pf = 1
    soc_naive = 1

    for date in all_dates:
        day = df[df['date'] == date].sort_values('hour')
        n_hours = len(day)
        if n_hours < 20:
            continue

        prices = day['da_price'].values[:n_hours]
        if np.any(np.isnan(prices)):
            continue

        pred_ranks = np.nan_to_num(day[rank_col].values[:n_hours], nan=12.0)

        # Indicator
        act_i, rev_i, soc_i = optimize_bess_dp(
            prices, pred_ranks, n_charge=4, n_discharge=4,
            start_soc=soc_ind, terminal_penalty=terminal_penalty)

        # Perfect foresight
        _, rev_pf, soc_pf_t = optimize_bess_dp(
            prices, prices, n_charge=4, n_discharge=4,
            start_soc=soc_pf, terminal_penalty=terminal_penalty)

        # Naive
        act_n, _ = naive_schedule(n_hours, soc_naive)
        soc_n = soc_naive
        feasible = True
        for h in range(n_hours):
            soc_n += int(act_n[h])
            if soc_n < 0 or soc_n > 2:
                feasible = False
                break
        rev_n = -np.sum(act_n * prices) if feasible else 0.0
        end_soc_n = soc_n if feasible else soc_naive

        results.append({
            'date': date,
            'rev_indicator': rev_i,
            'rev_perfect': rev_pf,
            'rev_naive': rev_n,
            'capture_rate': rev_i / rev_pf if rev_pf > 0 else np.nan,
            'start_soc': soc_ind,
            'end_soc': int(soc_i[-1]),
        })

        soc_ind = int(soc_i[-1])
        soc_pf = int(soc_pf_t[-1])
        soc_naive = end_soc_n

    res = pd.DataFrame(results)
    if len(res) > 0:
        res['date'] = pd.to_datetime(res['date'])
        res = res.sort_values('date').reset_index(drop=True)
    return res


# ============================================================
# 4. ABLATION: leave-one-out feature importance
# ============================================================

def ablation_study(df_train, df_test, feature_cols, full_weights):
    """Drop each feature and measure impact on test rank MAE."""
    X_test = df_test[feature_cols].fillna(12).values
    y_test = df_test['rank_actual'].values
    full_mae = np.mean(np.abs(X_test @ full_weights - y_test))

    results = []
    for i, feat in enumerate(feature_cols):
        reduced_feats = [f for j, f in enumerate(feature_cols) if j != i]
        w_reduced, mae_reduced_train = optimize_weights(df_train, reduced_feats, n_starts=10)

        X_te_red = df_test[reduced_feats].fillna(12).values
        mae_reduced = np.mean(np.abs(X_te_red @ w_reduced - y_test))

        results.append({
            'dropped': feat,
            'weight': full_weights[i],
            'full_mae': full_mae,
            'reduced_mae': mae_reduced,
            'mae_increase': mae_reduced - full_mae,
            'pct_increase': (mae_reduced - full_mae) / full_mae * 100,
        })

    return pd.DataFrame(results).sort_values('mae_increase', ascending=False)


# ============================================================
# 5. CHARTS
# ============================================================

def plot_results(strategies, all_feat_names, weight_dicts, test_start, test_end, n_test_days):
    """Generate comparison chart."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = {'S1': '#d32f2f', 'S2': '#1976d2', 'S3': '#4caf50', 'S4': '#ff9800'}
    styles = {'S1': '-', 'S2': '-', 'S3': '-', 'S4': '--'}

    # 1. Cumulative revenue
    ax = axes[0, 0]
    for key, name, res in strategies:
        if len(res) > 0:
            ax.plot(res['date'], res['rev_indicator'].cumsum(),
                    color=colors[key], ls=styles[key], lw=1.5, label=name)
    ref = strategies[0][2]
    if len(ref) > 0:
        ax.plot(ref['date'], ref['rev_perfect'].cumsum(), 'k--', lw=1, alpha=0.3, label='Perfect')
        ax.plot(ref['date'], ref['rev_naive'].cumsum(), 'gray', lw=1, alpha=0.2, label='Naive')
    ax.set_ylabel('Cumulative Revenue (EUR)')
    ax.set_title('BESS Revenue: Strategy Comparison')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. Rolling capture rate (14-day)
    ax = axes[0, 1]
    for key, name, res in strategies:
        if len(res) > 0:
            roll = res['capture_rate'].rolling(14, min_periods=3).mean()
            ax.plot(res['date'], roll, color=colors[key], lw=1.5, label=name)
    ax.set_ylabel('14-day Rolling Capture Rate')
    ax.set_title('Capture Rate Over Time')
    ax.set_ylim(0.3, 1.1)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Feature weights
    ax = axes[1, 0]
    x_pos = np.arange(len(all_feat_names))
    bar_w = 0.25
    for offset, (key, w_dict, color) in enumerate([
        ('S1', weight_dicts['S1'], colors['S1']),
        ('S2', weight_dicts['S2'], colors['S2']),
        ('S3', weight_dicts['S3'], colors['S3']),
    ]):
        vals = [w_dict.get(f, 0) for f in all_feat_names]
        ax.barh(x_pos + (offset - 1) * bar_w, vals, bar_w,
                label=key, color=color, edgecolor='white')
    ax.set_yticks(x_pos)
    ax.set_yticklabels([f.replace('rank_', '') for f in all_feat_names], fontsize=8)
    ax.set_xlabel('Weight')
    ax.set_title('Feature Weights by Strategy')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 4. Daily revenue scatter S1 vs S3
    ax = axes[1, 1]
    res_s1 = strategies[0][2]
    res_s3 = strategies[2][2]
    if len(res_s1) > 0 and len(res_s3) > 0:
        m = res_s1[['date', 'rev_indicator']].rename(columns={'rev_indicator': 'rev_s1'}).merge(
            res_s3[['date', 'rev_indicator']].rename(columns={'rev_indicator': 'rev_s3'}),
            on='date')
        ax.scatter(m['rev_s1'], m['rev_s3'], alpha=0.3, s=12, c='#1976d2')
        lim = max(m[['rev_s1', 'rev_s3']].abs().max().max(), 1)
        ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='Equal')
        s3_wins = (m['rev_s3'] > m['rev_s1']).sum()
        ax.annotate(f'Augmented wins: {s3_wins}/{len(m)} ({s3_wins/len(m):.0%})',
                    xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9,
                    va='top', fontweight='bold')
        ax.set_xlabel('S1: Original Revenue (EUR/day)')
        ax.set_ylabel('S3: Augmented Revenue (EUR/day)')
        ax.set_title('Daily Revenue: Original vs Augmented')
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'BESS Weight Experiment: DA Forecast Augmentation\n'
                 f'Test: {test_start} to {test_end} ({n_test_days} days)',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()

    path = OUT / 'plots' / '14_weight_experiment.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[+] Chart saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 75)
    print("  BESS WEIGHT OPTIMIZATION EXPERIMENT")
    print("  Testing DA forecast augmentation + broader feature set")
    print("=" * 75)

    # --- Load & prepare ---
    df = load_all_data()
    df, daily = build_features(df)
    da_fcst = load_da_forecasts()
    df, group_a, group_b = build_candidate_features(df, da_fcst)
    all_feats = group_a + group_b

    # --- Define train/test split based on DA forecast availability ---
    da_day_coverage = df[df['has_da_fcst']].groupby('date').size()
    da_full_dates = sorted(da_day_coverage[da_day_coverage >= 20].index)

    if len(da_full_dates) == 0:
        print("[!] No dates with sufficient DA forecast coverage. Aborting.")
        return

    test_start = da_full_dates[0]
    test_end = da_full_dates[-1]

    df_train = df[df['date'] < test_start].copy()
    df_test = df[df['date'].isin(da_full_dates)].copy()

    n_train = df_train['date'].nunique()
    n_test = len(da_full_dates)
    print(f"\n[*] Train: {df_train['date'].min().date()} to {df_train['date'].max().date()} ({n_train} days)")
    print(f"[*] Test:  {test_start.date()} to {test_end.date()} ({n_test} days)")

    # ===========================================================
    # S1: Original 3-feature (fixed 40/35/25)
    # ===========================================================
    print("\n" + "-" * 65)
    print("S1: Original 3-feature (fixed 40/35/25)")
    print("-" * 65)

    orig_feats = ['rank_load', 'rank_yest', 'rank_7d']
    orig_w = np.array([0.40, 0.35, 0.25])
    df_test['rank_s1'] = df_test[orig_feats].fillna(12).values @ orig_w

    res_s1 = run_bess_sim(df_test, 'rank_s1')
    _print_strat(res_s1)

    # ===========================================================
    # S2: Optimized Group A (trained pre-forecast, tested on forecast period)
    # ===========================================================
    print("\n" + "-" * 65)
    print("S2: Optimized Group A (trained on pre-forecast data)")
    print("-" * 65)

    # Filter to features with >50% coverage in training period
    valid_a = []
    for f in group_a:
        cov = df_train[f].notna().mean()
        status = "INCLUDED" if cov > 0.5 else "EXCLUDED"
        print(f"  {f:20s}: {cov:.1%} train coverage -> {status}")
        if cov > 0.5:
            valid_a.append(f)

    w_a, obj_a = optimize_weights(df_train, valid_a)
    print(f"\n  Optimized weights (train rank MAE = {obj_a:.3f}):")
    for feat, w in zip(valid_a, w_a):
        tag = " ***" if w > 0.05 else ""
        print(f"    {feat:20s}: {w:.3f}{tag}")

    df_test['rank_s2'] = df_test[valid_a].fillna(12).values @ w_a
    res_s2 = run_bess_sim(df_test, 'rank_s2')
    _print_strat(res_s2)

    # ===========================================================
    # S3: Augmented A+B (cross-validated on forecast period)
    # ===========================================================
    print("\n" + "-" * 65)
    print("S3: Augmented A+B (cross-validated on forecast period)")
    print("-" * 65)

    valid_all = []
    for f in all_feats:
        cov = df_test[f].notna().mean()
        status = "INCLUDED" if cov > 0.5 else "EXCLUDED"
        print(f"  {f:20s}: {cov:.1%} test coverage -> {status}")
        if cov > 0.5:
            valid_all.append(f)

    w_cv, cv_results = cross_validate_weights(df_test, valid_all, n_folds=3)
    print(f"\n  CV-averaged weights:")
    for feat, w in zip(valid_all, w_cv):
        tag = " ***" if w > 0.05 else ""
        print(f"    {feat:20s}: {w:.3f}{tag}")

    # Also full-period optimization (for reference, less rigorous)
    w_full, obj_full = optimize_weights(df_test, valid_all)
    print(f"\n  Full-period weights (rank MAE = {obj_full:.3f}):")
    for feat, w in zip(valid_all, w_full):
        tag = " ***" if w > 0.05 else ""
        print(f"    {feat:20s}: {w:.3f}{tag}")

    # Use CV weights for fair evaluation
    df_test['rank_s3'] = df_test[valid_all].fillna(12).values @ w_cv
    res_s3 = run_bess_sim(df_test, 'rank_s3')
    _print_strat(res_s3)

    # ===========================================================
    # S4: DA forecast only (standalone)
    # ===========================================================
    print("\n" + "-" * 65)
    print("S4: DA forecast only (standalone)")
    print("-" * 65)

    df_test['rank_s4'] = df_test['rank_da_fcst'].fillna(12)
    res_s4 = run_bess_sim(df_test, 'rank_s4')
    _print_strat(res_s4)

    # ===========================================================
    # COMPARISON TABLE
    # ===========================================================
    print("\n" + "=" * 75)
    print("  COMPARISON ON TEST PERIOD")
    print(f"  {test_start.date()} to {test_end.date()} ({n_test} days)")
    print("=" * 75)

    strats = [
        ('S1', 'Original 3-feat (40/35/25)', res_s1),
        ('S2', 'Optimized Group A', res_s2),
        ('S3', 'Augmented A+B (CV)', res_s3),
        ('S4', 'DA forecast only', res_s4),
    ]

    print(f"\n  {'Strategy':<35s} {'Revenue':>10s} {'Perfect':>10s} {'Capture':>8s} {'vs S1':>8s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    s1_rev = res_s1['rev_indicator'].sum() if len(res_s1) > 0 else 0
    for key, name, res in strats:
        if len(res) > 0:
            rev = res['rev_indicator'].sum()
            pf = res['rev_perfect'].sum()
            cap = res['capture_rate'].dropna().mean()
            delta = (rev - s1_rev) / abs(s1_rev) * 100 if s1_rev != 0 else 0
            sign = "+" if delta >= 0 else ""
            print(f"  {name:<35s} {rev:>10,.0f} {pf:>10,.0f} {cap:>7.1%} {sign}{delta:>6.1f}%")

    # Naive / Perfect reference
    if len(res_s1) > 0:
        naive_rev = res_s1['rev_naive'].sum()
        pf_rev = res_s1['rev_perfect'].sum()
        print(f"\n  {'Naive (fixed schedule)':<35s} {naive_rev:>10,.0f}")
        print(f"  {'Perfect foresight':<35s} {pf_rev:>10,.0f}")

    # ===========================================================
    # ABLATION STUDY (for augmented model)
    # ===========================================================
    print("\n" + "-" * 65)
    print("ABLATION: Feature importance (drop-one-out, augmented model)")
    print("-" * 65)

    # Use full-period weights for ablation (faster, still informative)
    abl = ablation_study(df_test, df_test, valid_all, w_full)
    print(f"\n  {'Dropped feature':<22s} {'Weight':>7s} {'MAE w/o':>8s} {'MAE full':>9s} {'Increase':>9s}")
    print(f"  {'-'*22} {'-'*7} {'-'*8} {'-'*9} {'-'*9}")
    for _, row in abl.iterrows():
        print(f"  {row['dropped']:<22s} {row['weight']:>6.3f} {row['reduced_mae']:>8.3f} "
              f"{row['full_mae']:>9.3f} {row['pct_increase']:>+8.1f}%")

    # ===========================================================
    # CHARTS
    # ===========================================================
    all_feat_names = sorted(set(orig_feats + valid_a + valid_all))
    weight_dicts = {
        'S1': dict(zip(orig_feats, orig_w)),
        'S2': dict(zip(valid_a, w_a)),
        'S3': dict(zip(valid_all, w_cv)),
    }
    plot_results(strats, all_feat_names, weight_dicts,
                 test_start.date(), test_end.date(), n_test)

    # ===========================================================
    # SAVE OUTPUTS
    # ===========================================================
    data_dir = OUT / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Weight summary
    rows = []
    for feat in all_feat_names:
        rows.append({
            'feature': feat,
            'w_s1_original': dict(zip(orig_feats, orig_w)).get(feat, 0),
            'w_s2_opt_a': dict(zip(valid_a, w_a)).get(feat, 0),
            'w_s3_cv': dict(zip(valid_all, w_cv)).get(feat, 0),
            'w_s3_full': dict(zip(valid_all, w_full)).get(feat, 0),
        })
    pd.DataFrame(rows).to_csv(data_dir / 'weight_experiment_weights.csv', index=False)

    # Daily comparison
    comp = res_s1[['date', 'rev_indicator', 'rev_perfect', 'rev_naive', 'capture_rate']].rename(
        columns={'rev_indicator': 'rev_s1', 'capture_rate': 'cap_s1'})
    for tag, res in [('s2', res_s2), ('s3', res_s3), ('s4', res_s4)]:
        if len(res) > 0:
            comp = comp.merge(
                res[['date', 'rev_indicator', 'capture_rate']].rename(
                    columns={'rev_indicator': f'rev_{tag}', 'capture_rate': f'cap_{tag}'}),
                on='date', how='outer')
    comp.to_csv(data_dir / 'weight_experiment_daily.csv', index=False)

    # CV results
    if cv_results:
        pd.DataFrame(cv_results).to_csv(data_dir / 'weight_experiment_cv.csv', index=False)

    # Ablation
    abl.to_csv(data_dir / 'weight_experiment_ablation.csv', index=False)

    print(f"\n[+] All outputs saved to {data_dir}")
    print("[+] Experiment complete")


def _print_strat(res):
    """Print single strategy summary."""
    if len(res) == 0:
        print("  [!] No results")
        return
    rev = res['rev_indicator'].sum()
    pf = res['rev_perfect'].sum()
    naive = res['rev_naive'].sum()
    cap = res['capture_rate'].dropna().mean()
    print(f"  Revenue:  {rev:>10,.0f} EUR")
    print(f"  Perfect:  {pf:>10,.0f} EUR")
    print(f"  Naive:    {naive:>10,.0f} EUR")
    print(f"  Capture:  {cap:.1%}")
    print(f"  Days:     {len(res)}")


if __name__ == '__main__':
    main()
