"""
Feature Selection for Spread Model (Trading-Metric-Driven)
===========================================================

Phase 1: Permutation importance on trading P&L
  - Train spread model on walk-forward folds
  - For each feature, shuffle in test data, re-predict, measure P&L drop
  - Repeat N times for stable estimates

Phase 2: Greedy backward elimination
  - Remove least important features in groups
  - Retrain walk-forward, check Sharpe/P&L degradation
  - Stop when Sharpe drops > 10% from baseline

Phase 3: Stability & correlation analysis
  - Per-fold feature importance consistency
  - Cross-correlation pruning (|r| > 0.95)

Output:
  - data/feature_selection/permutation_importance.csv
  - data/feature_selection/elimination_curve.csv
  - data/feature_selection/recommended_features.txt
  - data/feature_selection/correlation_clusters.csv
  - plots/feature_selection/01_permutation_importance.png
  - plots/feature_selection/02_elimination_curve.png
  - plots/feature_selection/03_fold_stability.png
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent  # ImbalanceForcastingProd/
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
REPO_ROOT = BASE_DIR.parent

sys.path.insert(0, str(BASE_DIR / "scripts" / "training"))
import train_multi_lead as tml
# Fix path resolution: train_multi_lead's REPO_ROOT points to ImbalanceForcastingProd/
# instead of the actual repo root after the restructure
tml.REPO_ROOT = REPO_ROOT
tml.DATA_DIR = DATA_DIR
tml.PLOT_DIR = PLOT_DIR
tml.MODEL_DIR = BASE_DIR / "models"
from train_multi_lead import load_all_data, build_features

# Output dirs
FS_DATA = DATA_DIR / "feature_selection"
FS_PLOT = PLOT_DIR / "feature_selection"
for d in [FS_DATA, FS_PLOT]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (16, 10), "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})

LEAD = 8
N_PERM_REPEATS = 10  # shuffles per feature for stable estimates

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                  reg_lambda=1.0, n_estimators=600, verbose=-1)

# Walk-forward folds: train on everything before test start
FOLDS = [
    {'name': 'Feb2026', 'train_end': '2026-02-01', 'test_start': '2026-02-01', 'test_end': '2026-03-01'},
    {'name': 'Mar2026', 'train_end': '2026-03-01', 'test_start': '2026-03-01', 'test_end': '2026-04-01'},
]

PRED_THRESHOLD = 3  # EUR/MWh — minimum |prediction| to trade


# ============================================================
# P&L CALCULATION
# ============================================================

def calc_trading_pnl(predictions, test_df, threshold=PRED_THRESHOLD):
    """Calculate trading P&L from spread predictions.

    Returns dict with: total_pnl, daily_pnl (Series), sharpe, n_trades, win_rate, max_dd
    """
    t = test_df.copy()
    t['pred'] = predictions

    surplus = t['pred'] <= -threshold
    deficit = t['pred'] >= threshold
    active = surplus | deficit
    sub = t[active].copy()

    if len(sub) < 10:
        return {'total_pnl': 0, 'daily_pnl': pd.Series(dtype=float),
                'sharpe': 0, 'n_trades': 0, 'win_rate': 0, 'max_dd': 0,
                'pnl_per_day': 0}

    sub['size'] = sub['pred'].abs().clip(upper=5)
    s = surplus.reindex(sub.index, fill_value=False)
    d = deficit.reindex(sub.index, fill_value=False)

    sub['pnl'] = 0.0
    sub.loc[s, 'pnl'] = (sub.loc[s, 'exec_bid'] - sub.loc[s, 'imb_settlement_price']) * sub.loc[s, 'size'] / 4
    sub.loc[d, 'pnl'] = (sub.loc[d, 'imb_settlement_price'] - sub.loc[d, 'exec_ask']) * sub.loc[d, 'size'] / 4

    daily = sub.groupby(sub.index.date)['pnl'].sum()
    n_days = max(len(daily), 1)
    total = sub['pnl'].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0

    # Max drawdown
    cumulative = daily.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = drawdown.min() if len(drawdown) > 0 else 0

    return {
        'total_pnl': total,
        'daily_pnl': daily,
        'sharpe': sharpe,
        'n_trades': len(sub),
        'win_rate': (sub['pnl'] > 0).mean(),
        'max_dd': max_dd,
        'pnl_per_day': total / n_days,
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data():
    """Load and prepare data for spread model feature selection."""
    print("[*] Loading all data sources...")
    data = load_all_data()

    # Build features with widest possible window
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, feature_cols = build_features(data, LEAD)

    # Join execution prices
    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv", parse_dates=['delivery_start'])
    ob_120 = ob_exec[ob_exec['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'spread', 'mid']]
    ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
    ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']

    # Join imbalance settlement price
    mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                       parse_dates=['timestamp_hour'], index_col='timestamp_hour')
    mkt = mkt[~mkt.index.duplicated(keep='last')]

    df_base = df_base.join(ob_120, how='left')
    df_base['hour_ts'] = df_base.index.floor('h')
    df_base = df_base.join(mkt[['imb_settlement_price']], on='hour_ts', how='left')
    df_base['spread_target'] = df_base['imb_settlement_price'] - df_base['exec_mid']

    print(f"[+] Base features: {len(feature_cols)} columns, {len(df_base)} rows")
    return df_base, feature_cols


def get_fold_data(df_base, feature_cols, fold):
    """Split data into train/test for a fold, filter to valid spread rows."""
    train_mask = df_base.index < fold['train_end']
    test_mask = (df_base.index >= fold['test_start']) & (df_base.index < fold['test_end'])

    scada_shift = LEAD + 1
    train = df_base[train_mask].dropna(subset=['spread_target', f'proxy_lag{scada_shift}'])
    train = train[train['imb_settlement_price'].abs() <= 5000]

    test = df_base[test_mask].dropna(subset=[f'proxy_lag{scada_shift}'])
    test = test[test['exec_spread'].notna() & (test['exec_spread'] <= 10)]

    return train, test


# ============================================================
# PHASE 1: PERMUTATION IMPORTANCE ON TRADING P&L
# ============================================================

def phase1_permutation_importance(df_base, feature_cols):
    """Shuffle each feature in test data, measure P&L degradation."""
    print("\n" + "=" * 70)
    print("PHASE 1: PERMUTATION IMPORTANCE (Trading P&L)")
    print("=" * 70)

    # Train models and get baseline P&L for each fold
    fold_models = []
    fold_tests = []
    baseline_pnls = []

    for fold in FOLDS:
        train, test = get_fold_data(df_base, feature_cols, fold)
        print(f"\n  Fold {fold['name']}: train={len(train)}, test={len(test)}")

        model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        model.fit(train[feature_cols].values, train['spread_target'].values)

        baseline_pred = model.predict(test[feature_cols].values)
        baseline = calc_trading_pnl(baseline_pred, test)
        baseline_pnls.append(baseline)

        print(f"    Baseline: {baseline['total_pnl']:+,.0f} EUR, "
              f"Sharpe={baseline['sharpe']:.1f}, {baseline['n_trades']} trades, "
              f"win={baseline['win_rate']:.0%}")

        fold_models.append(model)
        fold_tests.append(test)

    total_baseline_pnl = sum(b['total_pnl'] for b in baseline_pnls)
    print(f"\n  Combined baseline P&L: {total_baseline_pnl:+,.0f} EUR")

    # Permutation importance: shuffle each feature N times
    print(f"\n  Running permutation importance ({len(feature_cols)} features x "
          f"{N_PERM_REPEATS} repeats x {len(FOLDS)} folds)...")

    rng = np.random.RandomState(42)
    results = []

    for fi, feat in enumerate(feature_cols):
        feat_idx = feature_cols.index(feat)
        pnl_drops = []

        for repeat in range(N_PERM_REPEATS):
            total_shuffled_pnl = 0

            for fold_i, (model, test) in enumerate(zip(fold_models, fold_tests)):
                X_test = test[feature_cols].values.copy()
                # Shuffle this feature
                X_test[:, feat_idx] = rng.permutation(X_test[:, feat_idx])
                shuffled_pred = model.predict(X_test)
                shuffled = calc_trading_pnl(shuffled_pred, test)
                total_shuffled_pnl += shuffled['total_pnl']

            pnl_drops.append(total_baseline_pnl - total_shuffled_pnl)

        mean_drop = np.mean(pnl_drops)
        std_drop = np.std(pnl_drops)

        results.append({
            'feature': feat,
            'pnl_drop_mean': mean_drop,
            'pnl_drop_std': std_drop,
            'pnl_drop_pct': mean_drop / max(abs(total_baseline_pnl), 1) * 100,
            'is_positive': mean_drop > 0,  # True = feature helps trading
        })

        if (fi + 1) % 20 == 0:
            print(f"    ... {fi+1}/{len(feature_cols)} features done")

    imp_df = pd.DataFrame(results).sort_values('pnl_drop_mean', ascending=False)
    imp_df['rank'] = range(1, len(imp_df) + 1)
    imp_df['cumulative_pct'] = imp_df['pnl_drop_pct'].cumsum()

    # Print top/bottom features
    print(f"\n  --- Top 20 features (highest P&L drop when shuffled) ---")
    for _, row in imp_df.head(20).iterrows():
        print(f"    {row['rank']:3d}. {row['feature']:<35s}  "
              f"drop={row['pnl_drop_mean']:>+8.1f} EUR  ({row['pnl_drop_pct']:>+5.1f}%)")

    print(f"\n  --- Bottom 20 features (no P&L impact or negative) ---")
    for _, row in imp_df.tail(20).iterrows():
        sign = "HURTS" if row['pnl_drop_mean'] < -1 else "dead"
        print(f"    {row['rank']:3d}. {row['feature']:<35s}  "
              f"drop={row['pnl_drop_mean']:>+8.1f} EUR  ({row['pnl_drop_pct']:>+5.1f}%)  [{sign}]")

    n_positive = imp_df['is_positive'].sum()
    n_negative = (~imp_df['is_positive']).sum()
    print(f"\n  Summary: {n_positive} features help P&L, {n_negative} features are dead/harmful")

    # Save
    imp_df.to_csv(FS_DATA / "permutation_importance.csv", index=False)
    print(f"  [+] Saved: {FS_DATA / 'permutation_importance.csv'}")

    # Plot
    _plot_permutation_importance(imp_df)

    return imp_df, fold_models, fold_tests, baseline_pnls


def _plot_permutation_importance(imp_df):
    """Plot permutation importance results."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))

    # Top plot: bar chart of P&L drop per feature (top 40)
    ax = axes[0]
    top40 = imp_df.head(40)
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top40['pnl_drop_mean']]
    ax.barh(range(len(top40)), top40['pnl_drop_mean'].values, color=colors,
            xerr=top40['pnl_drop_std'].values, capsize=2, alpha=0.8)
    ax.set_yticks(range(len(top40)))
    ax.set_yticklabels(top40['feature'].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('P&L Drop When Shuffled (EUR)')
    ax.set_title('Phase 1: Permutation Importance — Top 40 Features by Trading P&L Impact')
    ax.axvline(x=0, color='black', linewidth=0.5)

    # Bottom plot: cumulative importance
    ax = axes[1]
    ax.plot(range(1, len(imp_df) + 1), imp_df['cumulative_pct'].values,
            'b-', linewidth=2)
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% of total')
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)

    # Mark where 90% is reached
    above_90 = imp_df[imp_df['cumulative_pct'] >= 90]
    if len(above_90) > 0:
        n_for_90 = above_90.iloc[0]['rank']
        ax.axvline(x=n_for_90, color='red', linestyle='--', alpha=0.5)
        ax.annotate(f'{int(n_for_90)} features for 90%',
                    xy=(n_for_90, 90), fontsize=10, color='red')

    ax.set_xlabel('Number of Features (ranked by importance)')
    ax.set_ylabel('Cumulative P&L Importance (%)')
    ax.set_title('Cumulative Feature Importance')
    ax.legend()

    plt.tight_layout()
    plt.savefig(FS_PLOT / "01_permutation_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved: {FS_PLOT / '01_permutation_importance.png'}")


# ============================================================
# PHASE 2: GREEDY BACKWARD ELIMINATION
# ============================================================

def phase2_backward_elimination(df_base, feature_cols, imp_df):
    """Remove features in groups, retrain, check trading metric degradation."""
    print("\n" + "=" * 70)
    print("PHASE 2: GREEDY BACKWARD ELIMINATION")
    print("=" * 70)

    # Sort features by importance (worst first for removal)
    ranked = imp_df.sort_values('pnl_drop_mean', ascending=True)['feature'].tolist()

    # Elimination schedule: remove in groups
    # Start aggressive (remove 20), then fine-grained (remove 5)
    steps = []
    n_total = len(feature_cols)
    # Remove harmful/dead features first (big chunks), then slow down
    remove_sizes = [20, 15, 10, 10, 5, 5, 5, 5, 5, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1]
    cumulative_removed = 0
    for rs in remove_sizes:
        cumulative_removed += rs
        if cumulative_removed >= n_total - 10:  # keep at least 10 features
            break
        steps.append(cumulative_removed)

    print(f"  Elimination steps: {len(steps)} rounds")
    print(f"  Features to remove at each step: {steps}")

    results = []

    # Baseline (all features)
    baseline = _evaluate_feature_set(df_base, feature_cols, "all")
    results.append({
        'n_features': len(feature_cols),
        'n_removed': 0,
        'removed_features': '',
        'total_pnl': baseline['total_pnl'],
        'sharpe': baseline['sharpe'],
        'n_trades': baseline['n_trades'],
        'win_rate': baseline['win_rate'],
        'max_dd': baseline['max_dd'],
        'pnl_per_day': baseline['pnl_per_day'],
    })
    baseline_sharpe = baseline['sharpe']
    baseline_pnl = baseline['total_pnl']

    print(f"\n  Baseline ({len(feature_cols)} features): "
          f"P&L={baseline_pnl:+,.0f}, Sharpe={baseline_sharpe:.1f}")

    best_sharpe = baseline_sharpe
    best_n_features = len(feature_cols)
    best_features = feature_cols.copy()

    for step_i, n_remove in enumerate(steps):
        removed = ranked[:n_remove]
        kept = [f for f in feature_cols if f not in removed]

        metrics = _evaluate_feature_set(df_base, kept, f"step_{step_i}")
        pnl_change = (metrics['total_pnl'] - baseline_pnl) / max(abs(baseline_pnl), 1) * 100
        sharpe_change = (metrics['sharpe'] - baseline_sharpe) / max(abs(baseline_sharpe), 1) * 100

        results.append({
            'n_features': len(kept),
            'n_removed': n_remove,
            'removed_features': '; '.join(removed[-5:]) + ('...' if n_remove > 5 else ''),
            'total_pnl': metrics['total_pnl'],
            'sharpe': metrics['sharpe'],
            'n_trades': metrics['n_trades'],
            'win_rate': metrics['win_rate'],
            'max_dd': metrics['max_dd'],
            'pnl_per_day': metrics['pnl_per_day'],
        })

        marker = ""
        if metrics['sharpe'] > best_sharpe:
            best_sharpe = metrics['sharpe']
            best_n_features = len(kept)
            best_features = kept.copy()
            marker = " ** NEW BEST **"

        print(f"  Step {step_i+1}: {len(kept):3d} features (-{n_remove:2d}) | "
              f"P&L={metrics['total_pnl']:>+8,.0f} ({pnl_change:>+5.1f}%) | "
              f"Sharpe={metrics['sharpe']:>5.1f} ({sharpe_change:>+5.1f}%) | "
              f"trades={metrics['n_trades']:4d} | win={metrics['win_rate']:.0%}{marker}")

        # Stop if Sharpe drops more than 15% below best
        if metrics['sharpe'] < best_sharpe * 0.85 and step_i > 2:
            print(f"\n  [!] Sharpe dropped >15% below best ({metrics['sharpe']:.1f} vs {best_sharpe:.1f}), stopping")
            break

    elim_df = pd.DataFrame(results)
    elim_df.to_csv(FS_DATA / "elimination_curve.csv", index=False)
    print(f"\n  [+] Saved: {FS_DATA / 'elimination_curve.csv'}")

    print(f"\n  Best configuration: {best_n_features} features, Sharpe={best_sharpe:.1f}")

    # Plot
    _plot_elimination_curve(elim_df, baseline_sharpe)

    return elim_df, best_features


def _evaluate_feature_set(df_base, features, label):
    """Train spread model with given features on walk-forward, return combined metrics."""
    all_daily = []
    total_pnl = 0
    total_trades = 0
    total_wins = 0
    worst_dd = 0

    for fold in FOLDS:
        train, test = get_fold_data(df_base, features, fold)
        if len(train) < 100 or len(test) < 50:
            continue

        model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        model.fit(train[features].values, train['spread_target'].values)
        pred = model.predict(test[features].values)
        metrics = calc_trading_pnl(pred, test)

        total_pnl += metrics['total_pnl']
        total_trades += metrics['n_trades']
        total_wins += metrics['n_trades'] * metrics['win_rate']
        worst_dd = min(worst_dd, metrics['max_dd'])
        if len(metrics['daily_pnl']) > 0:
            all_daily.append(metrics['daily_pnl'])

    if all_daily:
        combined_daily = pd.concat(all_daily)
        sharpe = combined_daily.mean() / combined_daily.std() * np.sqrt(252) if combined_daily.std() > 0 else 0
    else:
        combined_daily = pd.Series(dtype=float)
        sharpe = 0

    n_days = max(len(combined_daily), 1)
    return {
        'total_pnl': total_pnl,
        'sharpe': sharpe,
        'n_trades': total_trades,
        'win_rate': total_wins / max(total_trades, 1),
        'max_dd': worst_dd,
        'pnl_per_day': total_pnl / n_days,
    }


def _plot_elimination_curve(elim_df, baseline_sharpe):
    """Plot elimination curve."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    x = elim_df['n_features']

    ax = axes[0]
    ax.plot(x, elim_df['total_pnl'], 'b-o', linewidth=2, markersize=5)
    ax.axhline(y=elim_df.iloc[0]['total_pnl'], color='gray', linestyle='--', alpha=0.5, label='All features')
    best_idx = elim_df['sharpe'].idxmax()
    ax.axvline(x=elim_df.loc[best_idx, 'n_features'], color='green', linestyle='--', alpha=0.5, label='Best Sharpe')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Total P&L (EUR)')
    ax.set_title('Phase 2: Backward Elimination — P&L vs Feature Count')
    ax.legend()
    ax.invert_xaxis()

    ax = axes[1]
    ax.plot(x, elim_df['sharpe'], 'r-o', linewidth=2, markersize=5)
    ax.axhline(y=baseline_sharpe, color='gray', linestyle='--', alpha=0.5, label='All features')
    ax.axhline(y=baseline_sharpe * 0.9, color='orange', linestyle='--', alpha=0.3, label='90% threshold')
    ax.axvline(x=elim_df.loc[best_idx, 'n_features'], color='green', linestyle='--', alpha=0.5, label='Best Sharpe')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Sharpe Ratio (annualized)')
    ax.set_title('Backward Elimination — Sharpe Ratio vs Feature Count')
    ax.legend()
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(FS_PLOT / "02_elimination_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved: {FS_PLOT / '02_elimination_curve.png'}")


# ============================================================
# PHASE 3: STABILITY & CORRELATION ANALYSIS
# ============================================================

def phase3_stability_correlation(df_base, feature_cols, imp_df):
    """Check per-fold importance stability and cross-correlation."""
    print("\n" + "=" * 70)
    print("PHASE 3: STABILITY & CORRELATION ANALYSIS")
    print("=" * 70)

    # --- 3A: Per-fold split-based importance ---
    print("\n  --- 3A: Per-fold feature importance (split-based) ---")
    fold_importances = {}

    for fold in FOLDS:
        train, test = get_fold_data(df_base, feature_cols, fold)
        model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        model.fit(train[feature_cols].values, train['spread_target'].values)

        imp = pd.Series(model.feature_importances_, index=feature_cols)
        imp = imp / imp.sum() * 100  # normalize to %
        fold_importances[fold['name']] = imp

    fold_imp_df = pd.DataFrame(fold_importances)
    fold_imp_df['mean'] = fold_imp_df.mean(axis=1)
    fold_imp_df['std'] = fold_imp_df.std(axis=1)
    fold_imp_df['cv'] = fold_imp_df['std'] / fold_imp_df['mean'].clip(lower=0.01)
    fold_imp_df = fold_imp_df.sort_values('mean', ascending=False)

    # Flag unstable features (high CV + low importance)
    unstable = fold_imp_df[(fold_imp_df['cv'] > 1.0) & (fold_imp_df['mean'] < 1.0)]
    print(f"\n  Unstable features (CV > 1.0, importance < 1%): {len(unstable)}")
    if len(unstable) > 0:
        for feat, row in unstable.head(15).iterrows():
            vals = [f"{row[f['name']]:.2f}%" for f in FOLDS]
            print(f"    {feat:<35s}  mean={row['mean']:.2f}%  CV={row['cv']:.1f}  [{', '.join(vals)}]")

    fold_imp_df.to_csv(FS_DATA / "fold_stability.csv")
    print(f"  [+] Saved: {FS_DATA / 'fold_stability.csv'}")

    # --- 3B: Cross-correlation pruning ---
    print("\n  --- 3B: Cross-correlation analysis ---")

    # Use training data for correlation
    train_data = df_base[df_base.index < '2026-02-01']
    scada_shift = LEAD + 1
    train_valid = train_data.dropna(subset=[f'proxy_lag{scada_shift}'])

    corr_matrix = train_valid[feature_cols].corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    checked = set()
    for i, f1 in enumerate(feature_cols):
        for j, f2 in enumerate(feature_cols):
            if i >= j:
                continue
            pair_key = (min(f1, f2), max(f1, f2))
            if pair_key in checked:
                continue
            checked.add(pair_key)

            r = corr_matrix.loc[f1, f2]
            if abs(r) > 0.95:
                # Keep the one with higher permutation importance
                imp1 = imp_df[imp_df['feature'] == f1]['pnl_drop_mean'].values
                imp2 = imp_df[imp_df['feature'] == f2]['pnl_drop_mean'].values
                imp1 = imp1[0] if len(imp1) > 0 else 0
                imp2 = imp2[0] if len(imp2) > 0 else 0
                drop = f2 if imp1 >= imp2 else f1
                keep = f1 if imp1 >= imp2 else f2

                high_corr_pairs.append({
                    'feature_1': f1, 'feature_2': f2,
                    'correlation': r,
                    'keep': keep, 'drop': drop,
                    'keep_importance': max(imp1, imp2),
                    'drop_importance': min(imp1, imp2),
                })

    corr_df = pd.DataFrame(high_corr_pairs)
    if len(corr_df) > 0:
        corr_df = corr_df.sort_values('correlation', ascending=False, key=abs)
        print(f"\n  Highly correlated pairs (|r| > 0.95): {len(corr_df)}")
        for _, row in corr_df.head(20).iterrows():
            print(f"    r={row['correlation']:+.3f}  KEEP {row['keep']:<30s}  DROP {row['drop']}")

        corr_df.to_csv(FS_DATA / "correlation_clusters.csv", index=False)
        print(f"  [+] Saved: {FS_DATA / 'correlation_clusters.csv'}")
    else:
        print("  No highly correlated pairs found (|r| > 0.95)")

    # Plot fold stability
    _plot_fold_stability(fold_imp_df)

    return fold_imp_df, corr_df if len(high_corr_pairs) > 0 else pd.DataFrame()


def _plot_fold_stability(fold_imp_df):
    """Plot per-fold feature importance stability."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    top30 = fold_imp_df.head(30)
    fold_names = [f['name'] for f in FOLDS]
    x = np.arange(len(top30))
    width = 0.35

    for i, fn in enumerate(fold_names):
        offset = (i - len(fold_names) / 2 + 0.5) * width
        ax.barh(x + offset, top30[fn].values, width, label=fn, alpha=0.7)

    ax.set_yticks(x)
    ax.set_yticklabels(top30.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (%)')
    ax.set_title('Phase 3: Per-Fold Feature Importance Stability (Top 30)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(FS_PLOT / "03_fold_stability.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved: {FS_PLOT / '03_fold_stability.png'}")


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

def build_recommendation(imp_df, elim_df, best_features, fold_imp_df, corr_df):
    """Combine all phases into a final feature recommendation."""
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)

    # Start with features that have positive permutation importance
    positive_features = set(imp_df[imp_df['is_positive']]['feature'].tolist())
    print(f"\n  Phase 1 filter: {len(positive_features)} features with positive P&L contribution")

    # Remove features that are in correlated pairs (drop the less important one)
    corr_drops = set()
    if len(corr_df) > 0:
        corr_drops = set(corr_df['drop'].tolist())
    after_corr = positive_features - corr_drops
    print(f"  Phase 3B filter: removed {len(positive_features) - len(after_corr)} correlated duplicates")

    # Cross-reference with backward elimination best
    best_set = set(best_features)
    recommended = after_corr & best_set
    # Add back any features that are in best_features but not in positive
    # (they may have marginal positive impact that permutation missed)
    in_best_not_perm = best_set - positive_features
    if in_best_not_perm:
        print(f"  Features in best elimination set but not in permutation positive: {len(in_best_not_perm)}")
        # Add them back if they're not correlated
        recommended = recommended | (in_best_not_perm - corr_drops)

    # Final sort by permutation importance
    rec_imp = imp_df[imp_df['feature'].isin(recommended)].sort_values('pnl_drop_mean', ascending=False)
    recommended_list = rec_imp['feature'].tolist()

    print(f"\n  RECOMMENDED FEATURE SET: {len(recommended_list)} features")
    print(f"  (down from {len(imp_df)} original)")

    # Group by category
    groups = {
        'proxy': [f for f in recommended_list if f.startswith('proxy_') or f.startswith('reg_')],
        'load': [f for f in recommended_list if f.startswith('load_')],
        'weather': [f for f in recommended_list if any(f.startswith(p) for p in
                    ['temp', 'wind', 'radiation', 'pressure', 'cloud'])],
        'da_prices': [f for f in recommended_list if f.startswith('da_')],
        'damas': [f for f in recommended_list if f.startswith('damas_')],
        'market': [f for f in recommended_list if any(f.startswith(p) for p in
                   ['idm_', 'spread_da', 'imb_price'])],
        'solar': [f for f in recommended_list if f.startswith('solar_')],
        'nowcast': [f for f in recommended_list if f.startswith('nowcast_')],
        'production': [f for f in recommended_list if f.startswith('prod_')],
        'xborder': [f for f in recommended_list if f.startswith('xborder_')],
        'time': [f for f in recommended_list if any(f.startswith(p) for p in
                 ['hour_', 'qh_', 'is_', 'dow_', 'month_'])],
    }

    print(f"\n  By category:")
    for group, feats in sorted(groups.items(), key=lambda x: -len(x[1])):
        if feats:
            print(f"    {group:<15s}: {len(feats):2d} features")

    # Removed features
    all_features = set(imp_df['feature'].tolist())
    removed = all_features - set(recommended_list)
    removed_imp = imp_df[imp_df['feature'].isin(removed)].sort_values('pnl_drop_mean', ascending=True)
    print(f"\n  Removed {len(removed)} features:")
    for _, row in removed_imp.iterrows():
        reason = []
        if not row['is_positive']:
            reason.append("negative/zero P&L impact")
        if row['feature'] in corr_drops:
            reason.append("correlated duplicate")
        if row['feature'] not in best_set:
            reason.append("eliminated in backward step")
        print(f"    {row['feature']:<35s}  drop={row['pnl_drop_mean']:>+8.1f} EUR  [{', '.join(reason) or 'combined'}]")

    # Save recommended features
    with open(FS_DATA / "recommended_features.txt", 'w') as f:
        f.write(f"# Recommended features for spread model ({len(recommended_list)} features)\n")
        f.write(f"# Generated by feature_selection_spread.py\n")
        f.write(f"# Original: {len(all_features)} features\n\n")
        for feat in recommended_list:
            imp_val = imp_df[imp_df['feature'] == feat]['pnl_drop_mean'].values[0]
            f.write(f"{feat}  # P&L importance: {imp_val:+.1f} EUR\n")

    print(f"\n  [+] Saved: {FS_DATA / 'recommended_features.txt'}")

    # Save as Python list for easy import
    with open(FS_DATA / "recommended_features.py", 'w') as f:
        f.write(f"# Recommended features for spread model ({len(recommended_list)} features)\n")
        f.write(f"# Generated by feature_selection_spread.py\n\n")
        f.write("SELECTED_FEATURES = [\n")
        for feat in recommended_list:
            f.write(f"    '{feat}',\n")
        f.write("]\n")

    print(f"  [+] Saved: {FS_DATA / 'recommended_features.py'}")

    return recommended_list


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("FEATURE SELECTION FOR SPREAD MODEL")
    print("=" * 70)

    df_base, feature_cols = prepare_data()

    # Phase 1
    imp_df, fold_models, fold_tests, baseline_pnls = phase1_permutation_importance(df_base, feature_cols)

    # Phase 2
    elim_df, best_features = phase2_backward_elimination(df_base, feature_cols, imp_df)

    # Phase 3
    fold_imp_df, corr_df = phase3_stability_correlation(df_base, feature_cols, imp_df)

    # Final recommendation
    recommended = build_recommendation(imp_df, elim_df, best_features, fold_imp_df, corr_df)

    # Validate recommended set
    print("\n" + "=" * 70)
    print("VALIDATION: Recommended vs Full feature set")
    print("=" * 70)

    full_metrics = _evaluate_feature_set(df_base, feature_cols, "full")
    rec_metrics = _evaluate_feature_set(df_base, recommended, "recommended")

    n_full = len(feature_cols)
    n_rec = len(recommended)
    print(f"\n  {'Metric':<20s}  {'Full ('+str(n_full)+')':>15s}  {'Rec ('+str(n_rec)+')':>15s}  {'Change':>10s}")
    print(f"  {'':->20s}  {'':->15s}  {'':->15s}  {'':->10s}")
    for metric in ['total_pnl', 'sharpe', 'n_trades', 'win_rate', 'max_dd', 'pnl_per_day']:
        v1 = full_metrics[metric]
        v2 = rec_metrics[metric]
        if metric in ['win_rate']:
            fmt = lambda x: f"{x:.1%}"
        elif metric in ['sharpe']:
            fmt = lambda x: f"{x:.1f}"
        else:
            fmt = lambda x: f"{x:+,.0f}" if abs(x) >= 1 else f"{x:.2f}"

        change = ((v2 - v1) / abs(v1) * 100) if abs(v1) > 0 else 0
        print(f"  {metric:<20s}  {fmt(v1):>15s}  {fmt(v2):>15s}  {change:>+9.1f}%")

    print(f"\n  Features: {n_full} -> {n_rec} ({n_full - n_rec} removed, {(n_full-n_rec)/n_full*100:.0f}% reduction)")
    print("\n[+] Feature selection complete!")


if __name__ == "__main__":
    main()
