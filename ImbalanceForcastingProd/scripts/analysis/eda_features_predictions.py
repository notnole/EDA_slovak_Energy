"""
Detailed Exploratory Data Analysis: Features & Predictions
============================================================

Covers:
  1. Target distributions (imbalance + spread)
  2. Feature distributions & missingness
  3. Feature correlations & collinearity
  4. Feature importance deep-dive (permutation vs LightGBM split)
  5. Prediction quality (imbalance model + spread model)
  6. Temporal patterns (hourly, daily, monthly)
  7. Residual analysis
  8. Trading signal quality
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- Paths ---
BASE = Path(__file__).resolve().parent.parent.parent  # ImbalanceForcastingProd
PLOT_DIR = BASE / "plots" / "eda"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (18, 12), "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 130,
})


# =================================================================
# LOAD DATA
# =================================================================

def load_data():
    """Load all relevant data files."""
    # Stacked predictions (test period: Feb-Mar 2026)
    stacked = pd.read_csv(BASE / "data" / "predictions" / "stacked_test_predictions.csv",
                          parse_dates=['datetime'], index_col='datetime')

    # Lead-8 predictions (primary imbalance test set, Feb-Mar 2026)
    preds_v2 = pd.read_csv(BASE / "data" / "predictions" / "predictions_lead8.csv",
                           parse_dates=['datetime'], index_col='datetime')

    # Feature importance (LightGBM split-based)
    feat_imp = pd.read_csv(BASE / "data" / "features" / "feature_importance_v2.csv")

    # Permutation importance (P&L-based)
    perm_imp = pd.read_csv(BASE / "data" / "feature_selection" / "permutation_importance.csv")

    # Backtest trades
    trades = pd.read_csv(BASE / "data" / "backtests" / "backtest_realistic.csv",
                         parse_dates=['datetime'], index_col='datetime')

    # Stacked validation (includes selected-feature model)
    stk_val = pd.read_csv(BASE / "data" / "feature_selection" / "stacked_validation_predictions.csv",
                          parse_dates=['datetime'], index_col='datetime')

    # Elimination curve
    elim = pd.read_csv(BASE / "data" / "feature_selection" / "elimination_curve.csv")

    # Fold stability
    fold_stab = pd.read_csv(BASE / "data" / "feature_selection" / "fold_stability.csv", index_col=0)

    # Correlation clusters
    corr_clusters = pd.read_csv(BASE / "data" / "feature_selection" / "correlation_clusters.csv")

    # Multi-lead predictions
    lead_preds = {}
    for lead in [4, 5, 6, 7, 8]:
        path = BASE / "data" / "predictions" / f"predictions_lead{lead}.csv"
        if path.exists():
            lead_preds[lead] = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')

    print(f"[+] Stacked test: {len(stacked)} rows ({stacked.index.min()} to {stacked.index.max()})")
    print(f"[+] Imbalance test (v2): {len(preds_v2)} rows ({preds_v2.index.min()} to {preds_v2.index.max()})")
    print(f"[+] Trades: {len(trades)} rows")
    print(f"[+] Feature importance: {len(feat_imp)} features")
    print(f"[+] Permutation importance: {len(perm_imp)} features")
    print(f"[+] Multi-lead predictions: leads {list(lead_preds.keys())}")

    return {
        'stacked': stacked, 'preds_v2': preds_v2, 'feat_imp': feat_imp,
        'perm_imp': perm_imp, 'trades': trades, 'stk_val': stk_val,
        'elim': elim, 'fold_stab': fold_stab, 'corr_clusters': corr_clusters,
        'lead_preds': lead_preds,
    }


# =================================================================
# PLOT 1: TARGET DISTRIBUTIONS
# =================================================================

def plot_target_distributions(data):
    """Imbalance and spread target distributions + temporal patterns."""
    stk = data['stacked']
    pv2 = data['preds_v2']

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1a: Imbalance histogram
    ax1 = fig.add_subplot(gs[0, 0])
    target = pv2['target'].dropna()
    ax1.hist(target, bins=80, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.3)
    ax1.axvline(target.mean(), color='red', ls='--', label=f'Mean={target.mean():.1f}')
    ax1.axvline(target.median(), color='orange', ls='--', label=f'Median={target.median():.1f}')
    ax1.set_xlabel('System Imbalance (MWh)')
    ax1.set_ylabel('Count')
    ax1.set_title('Imbalance Distribution (Oct 2025 - Mar 2026)')
    ax1.legend(fontsize=9)

    # 1b: Spread histogram
    ax2 = fig.add_subplot(gs[0, 1])
    spread = stk['spread_target'].dropna()
    ax2.hist(spread, bins=80, alpha=0.7, color='coral', edgecolor='black', linewidth=0.3)
    ax2.axvline(spread.mean(), color='red', ls='--', label=f'Mean={spread.mean():.1f}')
    ax2.axvline(spread.median(), color='orange', ls='--', label=f'Median={spread.median():.1f}')
    ax2.set_xlabel('IDM-to-Settlement Spread (EUR/MWh)')
    ax2.set_ylabel('Count')
    ax2.set_title('Spread Target Distribution (Feb-Mar 2026)')
    ax2.legend(fontsize=9)

    # 1c: Imbalance vs spread scatter
    ax3 = fig.add_subplot(gs[0, 2])
    valid = stk.dropna(subset=['target', 'spread_target'])
    sc = ax3.scatter(valid['target'], valid['spread_target'], s=3, alpha=0.3, c='steelblue')
    ax3.set_xlabel('System Imbalance (MWh)')
    ax3.set_ylabel('Spread (EUR/MWh)')
    ax3.set_title('Imbalance vs Spread')
    r = valid['target'].corr(valid['spread_target'])
    ax3.text(0.05, 0.95, f'r = {r:.2f}', transform=ax3.transAxes, fontsize=12, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat'))

    # 1d: Hourly imbalance boxplot
    ax4 = fig.add_subplot(gs[1, 0])
    pv2_h = pv2.copy()
    pv2_h['hour'] = pv2_h.index.hour
    pv2_h.boxplot(column='target', by='hour', ax=ax4, showfliers=False,
                  boxprops=dict(color='steelblue'), medianprops=dict(color='red'))
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Imbalance (MWh)')
    ax4.set_title('Imbalance by Hour')
    plt.suptitle('')

    # 1e: Hourly spread boxplot
    ax5 = fig.add_subplot(gs[1, 1])
    stk_h = stk.copy()
    stk_h['hour'] = stk_h.index.hour
    stk_h.boxplot(column='spread_target', by='hour', ax=ax5, showfliers=False,
                  boxprops=dict(color='coral'), medianprops=dict(color='red'))
    ax5.set_xlabel('Hour of Day')
    ax5.set_ylabel('Spread (EUR/MWh)')
    ax5.set_title('Spread by Hour')
    plt.suptitle('')

    # 1f: Imbalance time series (test period)
    ax6 = fig.add_subplot(gs[1, 2])
    daily_imb = pv2['target'].resample('D').mean()
    ax6.plot(daily_imb.index, daily_imb.values, color='steelblue', alpha=0.8)
    ax6.axhline(0, color='gray', ls='--', alpha=0.5)
    ax6.set_ylabel('Daily Mean Imbalance (MWh)')
    ax6.set_title('Daily Imbalance (Oct 2025 - Mar 2026)')
    ax6.tick_params(axis='x', rotation=30)

    # 1g: Imbalance autocorrelation
    ax7 = fig.add_subplot(gs[2, 0])
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(target.iloc[:5000], ax=ax7, color='steelblue')
    ax7.set_xlim(0, 200)
    ax7.set_title('Imbalance Autocorrelation')

    # 1h: Surplus vs deficit fraction by hour
    ax8 = fig.add_subplot(gs[2, 1])
    pv2_h['is_deficit'] = pv2_h['target'] > 0
    deficit_rate = pv2_h.groupby('hour')['is_deficit'].mean()
    ax8.bar(deficit_rate.index, deficit_rate.values, color='coral', alpha=0.7, label='Deficit %')
    ax8.bar(deficit_rate.index, 1 - deficit_rate.values, bottom=deficit_rate.values,
            color='steelblue', alpha=0.7, label='Surplus %')
    ax8.set_xlabel('Hour')
    ax8.set_ylabel('Fraction')
    ax8.set_title('Deficit vs Surplus by Hour')
    ax8.axhline(0.5, color='gray', ls='--', alpha=0.5)
    ax8.legend(fontsize=9)

    # 1i: QQ plot of imbalance
    ax9 = fig.add_subplot(gs[2, 2])
    sorted_t = np.sort(target.values)
    n = len(sorted_t)
    theoretical = np.random.normal(target.mean(), target.std(), n)
    theoretical.sort()
    ax9.scatter(theoretical, sorted_t, s=1, alpha=0.3, c='steelblue')
    lims = [min(theoretical.min(), sorted_t.min()), max(theoretical.max(), sorted_t.max())]
    ax9.plot(lims, lims, 'r--', alpha=0.5)
    ax9.set_xlabel('Theoretical Quantiles')
    ax9.set_ylabel('Observed Quantiles')
    ax9.set_title('QQ Plot: Imbalance vs Normal')

    fig.savefig(PLOT_DIR / "01_target_distributions.png", bbox_inches='tight')
    plt.close(fig)
    print("[+] Saved 01_target_distributions.png")

    # Print stats
    print(f"\n--- Target Statistics ---")
    print(f"Imbalance: mean={target.mean():.2f}, std={target.std():.2f}, "
          f"skew={target.skew():.2f}, kurt={target.kurtosis():.2f}")
    print(f"  Deficit fraction: {(target > 0).mean():.1%}")
    print(f"  |target| > 10 MWh: {(target.abs() > 10).mean():.1%}")
    print(f"  |target| > 30 MWh: {(target.abs() > 30).mean():.1%}")
    print(f"Spread:  mean={spread.mean():.2f}, std={spread.std():.2f}, "
          f"skew={spread.skew():.2f}, kurt={spread.kurtosis():.2f}")
    print(f"  Positive spread (profit on deficit): {(spread > 0).mean():.1%}")
    print(f"  Imbalance-Spread correlation: {r:.3f}")


# =================================================================
# PLOT 2: FEATURE IMPORTANCE COMPARISON
# =================================================================

def plot_feature_importance(data):
    """Compare LightGBM split importance vs P&L-based permutation importance."""
    feat_imp = data['feat_imp'].copy()
    perm_imp = data['perm_imp'].copy()

    fig, axes = plt.subplots(2, 2, figsize=(22, 18))

    # 2a: Top 30 by LightGBM split importance
    ax = axes[0, 0]
    top30 = feat_imp.nlargest(30, 'importance')
    colors = []
    for f in top30['feature']:
        if 'proxy' in f or 'reg' in f: colors.append('#1f77b4')
        elif 'temp' in f or 'wind' in f or 'rad' in f or 'press' in f or 'cloud' in f: colors.append('#2ca02c')
        elif 'da_' in f or 'idm' in f or 'spread' in f or 'imb_price' in f: colors.append('#d62728')
        elif 'damas' in f or 'nowcast' in f: colors.append('#9467bd')
        elif 'load' in f: colors.append('#ff7f0e')
        elif 'solar' in f: colors.append('#bcbd22')
        elif 'hour' in f or 'dow' in f or 'month' in f or 'qh' in f or 'peak' in f or 'weekend' in f: colors.append('#8c564b')
        elif 'prod' in f or 'xborder' in f: colors.append('#e377c2')
        else: colors.append('#7f7f7f')
    ax.barh(range(len(top30)), top30['pct'].values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30['feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Split Importance (%)')
    ax.set_title('Top 30 Features: LightGBM Split Importance')

    # 2b: Top 30 by P&L permutation importance
    ax = axes[0, 1]
    perm_pos = perm_imp[perm_imp['is_positive']].nlargest(30, 'pnl_drop_mean')
    colors2 = []
    for f in perm_pos['feature']:
        if 'proxy' in f or 'reg' in f: colors2.append('#1f77b4')
        elif 'temp' in f or 'wind' in f or 'rad' in f or 'press' in f or 'cloud' in f: colors2.append('#2ca02c')
        elif 'da_' in f or 'idm' in f or 'spread' in f or 'imb_price' in f: colors2.append('#d62728')
        elif 'damas' in f or 'nowcast' in f: colors2.append('#9467bd')
        elif 'load' in f: colors2.append('#ff7f0e')
        elif 'solar' in f: colors2.append('#bcbd22')
        elif 'hour' in f or 'dow' in f or 'month' in f or 'qh' in f or 'peak' in f or 'weekend' in f: colors2.append('#8c564b')
        elif 'prod' in f or 'xborder' in f: colors2.append('#e377c2')
        else: colors2.append('#7f7f7f')
    ax.barh(range(len(perm_pos)), perm_pos['pnl_drop_mean'].values, color=colors2, alpha=0.8)
    ax.set_yticks(range(len(perm_pos)))
    ax.set_yticklabels(perm_pos['feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('P&L Drop When Shuffled (EUR)')
    ax.set_title('Top 30 Features: P&L Permutation Importance')

    # 2c: Split vs Permutation rank comparison
    ax = axes[1, 0]
    feat_imp['split_rank'] = feat_imp['importance'].rank(ascending=False)
    perm_imp['perm_rank'] = perm_imp['pnl_drop_mean'].rank(ascending=False)
    merged = feat_imp.merge(perm_imp[['feature', 'perm_rank', 'pnl_drop_mean']], on='feature', how='inner')
    ax.scatter(merged['split_rank'], merged['perm_rank'], s=30, alpha=0.6, c='steelblue')
    for _, row in merged.iterrows():
        if row['split_rank'] <= 10 or row['perm_rank'] <= 10:
            ax.annotate(row['feature'], (row['split_rank'], row['perm_rank']),
                       fontsize=7, alpha=0.8, rotation=15)
    ax.plot([0, 115], [0, 115], 'r--', alpha=0.3)
    ax.set_xlabel('Split Importance Rank')
    ax.set_ylabel('Permutation Importance Rank')
    ax.set_title('Split vs Permutation Importance Rank')
    r_rank = merged['split_rank'].corr(merged['perm_rank'])
    ax.text(0.05, 0.95, f'Rank corr = {r_rank:.2f}', transform=ax.transAxes, fontsize=12, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat'))

    # 2d: Feature group importance (P&L-based)
    ax = axes[1, 1]
    def categorize(f):
        if 'proxy' in f or 'reg' in f: return 'Regulation/Proxy'
        if 'temp' in f or 'wind' in f or 'rad' in f or 'press' in f or 'cloud' in f: return 'Weather'
        if 'da_' in f or 'idm' in f or 'spread_da' in f or 'imb_price' in f: return 'Prices/Market'
        if 'damas' in f or 'nowcast' in f: return 'Load Forecast'
        if 'load' in f: return 'Load SCADA'
        if 'solar' in f: return 'Solar'
        if any(t in f for t in ['hour', 'dow', 'month', 'qh', 'peak', 'weekend']): return 'Time'
        if 'prod' in f or 'xborder' in f: return 'Production/XBorder'
        return 'Other'
    perm_imp['group'] = perm_imp['feature'].apply(categorize)
    group_imp = perm_imp.groupby('group')['pnl_drop_mean'].sum().sort_values(ascending=True)
    colors_g = {'Regulation/Proxy': '#1f77b4', 'Weather': '#2ca02c', 'Prices/Market': '#d62728',
                'Load Forecast': '#9467bd', 'Load SCADA': '#ff7f0e', 'Solar': '#bcbd22',
                'Time': '#8c564b', 'Production/XBorder': '#e377c2', 'Other': '#7f7f7f'}
    bar_colors = [colors_g.get(g, '#7f7f7f') for g in group_imp.index]
    ax.barh(range(len(group_imp)), group_imp.values, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(group_imp)))
    ax.set_yticklabels(group_imp.index, fontsize=10)
    ax.set_xlabel('Total P&L Impact (EUR)')
    ax.set_title('Feature Group P&L Importance')
    ax.axvline(0, color='gray', ls='--', alpha=0.5)

    fig.savefig(PLOT_DIR / "02_feature_importance.png", bbox_inches='tight')
    plt.close(fig)
    print("[+] Saved 02_feature_importance.png")

    # Stats
    print(f"\n--- Feature Importance ---")
    print(f"Split-Perm rank correlation: {r_rank:.3f}")
    n_positive = (perm_imp['pnl_drop_mean'] > 0).sum()
    n_negative = (perm_imp['pnl_drop_mean'] < 0).sum()
    print(f"Permutation: {n_positive} features positive, {n_negative} negative (harmful)")
    top5_pnl = perm_imp.nlargest(5, 'pnl_drop_mean')
    print(f"Top 5 by P&L: {', '.join(top5_pnl['feature'].values)}")
    print(f"  Combined P&L impact: {top5_pnl['pnl_drop_mean'].sum():.0f} EUR")
    for g in group_imp.index:
        print(f"  {g:25s}: {group_imp[g]:+8.0f} EUR")


# =================================================================
# PLOT 3: PREDICTION QUALITY (IMBALANCE MODEL)
# =================================================================

def plot_imbalance_predictions(data):
    """Imbalance model prediction quality analysis."""
    pv2 = data['preds_v2']
    lead_preds = data['lead_preds']

    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 3a: Actual vs predicted scatter
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(pv2['target'], pv2['pred_median'], s=3, alpha=0.2, c='steelblue')
    lims = [-60, 60]
    ax.plot(lims, lims, 'r--', alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim([-30, 30])
    ax.set_xlabel('Actual Imbalance (MWh)')
    ax.set_ylabel('Predicted Imbalance (MWh)')
    r = pv2['target'].corr(pv2['pred_median'])
    ax.set_title(f'Actual vs Predicted (r={r:.3f})')

    # 3b: Residual distribution
    ax = fig.add_subplot(gs[0, 1])
    residual = pv2['target'] - pv2['pred_median']
    ax.hist(residual, bins=80, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.3)
    ax.axvline(residual.mean(), color='red', ls='--', label=f'Mean={residual.mean():.2f}')
    ax.set_xlabel('Residual (MWh)')
    ax.set_title('Residual Distribution')
    ax.legend()

    # 3c: Direction accuracy by hour
    ax = fig.add_subplot(gs[0, 2])
    pv2_h = pv2.copy()
    pv2_h['hour'] = pv2_h.index.hour
    nonzero = pv2_h['target'].abs() > 0.1
    pv2_h['correct_dir'] = (np.sign(pv2_h['pred_median']) == np.sign(pv2_h['target']))
    dir_by_hour = pv2_h[nonzero].groupby('hour')['correct_dir'].mean()
    ax.bar(dir_by_hour.index, dir_by_hour.values, color='steelblue', alpha=0.7)
    ax.axhline(0.5, color='red', ls='--', alpha=0.5, label='Random')
    overall_dir = pv2_h[nonzero]['correct_dir'].mean()
    ax.axhline(overall_dir, color='green', ls='--', alpha=0.7, label=f'Overall={overall_dir:.1%}')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Direction Accuracy')
    ax.set_title('Direction Accuracy by Hour')
    ax.legend(fontsize=9)
    ax.set_ylim(0.3, 0.8)

    # 3d: Confidence vs accuracy
    ax = fig.add_subplot(gs[1, 0])
    pv2_nz = pv2[nonzero].copy()
    pv2_nz['abs_pred'] = pv2_nz['pred_median'].abs()
    pv2_nz['correct_dir'] = (np.sign(pv2_nz['pred_median']) == np.sign(pv2_nz['target']))
    bins = pd.qcut(pv2_nz['abs_pred'], q=10, duplicates='drop')
    conf_acc = pv2_nz.groupby(bins)['correct_dir'].agg(['mean', 'count'])
    x_labels = [f'{b.left:.1f}-{b.right:.1f}' for b in conf_acc.index]
    ax.bar(range(len(conf_acc)), conf_acc['mean'], color='steelblue', alpha=0.7)
    ax2_twin = ax.twinx()
    ax2_twin.plot(range(len(conf_acc)), conf_acc['count'], 'ro-', alpha=0.6, markersize=4)
    ax2_twin.set_ylabel('Count', color='red')
    ax.set_xticks(range(len(conf_acc)))
    ax.set_xticklabels(x_labels, rotation=45, fontsize=8)
    ax.set_ylabel('Direction Accuracy')
    ax.set_title('Confidence vs Direction Accuracy')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.3)

    # 3e: Quantile coverage
    ax = fig.add_subplot(gs[1, 1])
    if 'q10' in pv2.columns and 'q90' in pv2.columns:
        in_band = ((pv2['target'] >= pv2['q10']) & (pv2['target'] <= pv2['q90']))
        coverage_overall = in_band.mean()
        pv2_hc = pv2.copy()
        pv2_hc['hour'] = pv2_hc.index.hour
        pv2_hc['in_band'] = in_band
        cov_by_hour = pv2_hc.groupby('hour')['in_band'].mean()
        ax.bar(cov_by_hour.index, cov_by_hour.values, color='steelblue', alpha=0.7)
        ax.axhline(0.8, color='red', ls='--', label=f'Target=80%')
        ax.axhline(coverage_overall, color='green', ls='--', label=f'Overall={coverage_overall:.1%}')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Coverage (Q10-Q90)')
        ax.set_title('Quantile Band Coverage by Hour')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Quantile columns not available', transform=ax.transAxes, ha='center')

    # 3f: Multi-lead accuracy comparison
    ax = fig.add_subplot(gs[1, 2])
    leads_acc = []
    leads_mae = []
    for lead in sorted(lead_preds.keys()):
        lp = lead_preds[lead]
        nz = lp['target'].abs() > 0.1
        dacc = (np.sign(lp['pred_median']) == np.sign(lp['target']))[nz].mean()
        mae = (lp['target'] - lp['pred_median']).abs().mean()
        leads_acc.append(dacc)
        leads_mae.append(mae)
    leads_sorted = sorted(lead_preds.keys())
    ax.bar([f'Lead {l}\n({l*15}m)' for l in leads_sorted], leads_acc, color='steelblue', alpha=0.7)
    ax.set_ylabel('Direction Accuracy', color='steelblue')
    ax.set_title('Multi-Lead Accuracy & MAE')
    ax2_twin = ax.twinx()
    ax2_twin.plot([f'Lead {l}\n({l*15}m)' for l in leads_sorted], leads_mae, 'ro-', markersize=6)
    ax2_twin.set_ylabel('MAE (MWh)', color='red')

    # 3g: Rolling direction accuracy
    ax = fig.add_subplot(gs[2, 0])
    pv2_roll = pv2.copy()
    pv2_roll['correct_dir'] = (np.sign(pv2_roll['pred_median']) == np.sign(pv2_roll['target']))
    rolling_acc = pv2_roll['correct_dir'].rolling(96 * 7).mean()  # 7-day rolling
    ax.plot(rolling_acc.index, rolling_acc.values, color='steelblue', alpha=0.8)
    ax.axhline(0.5, color='red', ls='--', alpha=0.5)
    ax.set_ylabel('7-Day Rolling Direction Accuracy')
    ax.set_title('Direction Accuracy Over Time')
    ax.tick_params(axis='x', rotation=30)

    # 3h: Prediction distribution vs actual
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(pv2['target'], bins=60, alpha=0.5, color='steelblue', label='Actual', density=True)
    ax.hist(pv2['pred_median'], bins=60, alpha=0.5, color='coral', label='Predicted', density=True)
    ax.set_xlabel('Imbalance (MWh)')
    ax.set_title('Actual vs Predicted Distribution')
    ax.legend()
    ax.text(0.05, 0.95, f'Actual std={pv2["target"].std():.1f}\nPred std={pv2["pred_median"].std():.1f}',
            transform=ax.transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    # 3i: Residual vs hour (bias check)
    ax = fig.add_subplot(gs[2, 2])
    pv2_res = pv2.copy()
    pv2_res['residual'] = pv2_res['target'] - pv2_res['pred_median']
    pv2_res['hour'] = pv2_res.index.hour
    res_by_hour = pv2_res.groupby('hour')['residual'].agg(['mean', 'std'])
    ax.bar(res_by_hour.index, res_by_hour['mean'], color='steelblue', alpha=0.7)
    ax.errorbar(res_by_hour.index, res_by_hour['mean'], yerr=res_by_hour['std'] / 5,
                color='red', fmt='none', capsize=3)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Mean Residual (MWh)')
    ax.set_title('Residual Bias by Hour')

    fig.savefig(PLOT_DIR / "03_imbalance_predictions.png", bbox_inches='tight')
    plt.close(fig)
    print("[+] Saved 03_imbalance_predictions.png")

    # Stats
    print(f"\n--- Imbalance Prediction Quality ---")
    mae = (pv2['target'] - pv2['pred_median']).abs().mean()
    rmse = np.sqrt(((pv2['target'] - pv2['pred_median'])**2).mean())
    print(f"MAE={mae:.2f} MWh, RMSE={rmse:.2f} MWh, r={r:.3f}")
    print(f"Direction accuracy (overall): {overall_dir:.1%}")
    print(f"Prediction shrinkage: actual_std={pv2['target'].std():.1f}, pred_std={pv2['pred_median'].std():.1f}")
    if 'q10' in pv2.columns:
        print(f"Q10-Q90 coverage: {coverage_overall:.1%} (target: 80%)")
    for lead in sorted(lead_preds.keys()):
        print(f"  Lead {lead} ({lead*15}m): dir_acc={leads_acc[leads_sorted.index(lead)]:.1%}, MAE={leads_mae[leads_sorted.index(lead)]:.2f}")


# =================================================================
# PLOT 4: SPREAD MODEL & STACKED PREDICTIONS
# =================================================================

def plot_spread_predictions(data):
    """Spread model prediction quality and stacking analysis."""
    stk = data['stacked']
    stk_val = data['stk_val']

    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 4a: Spread actual vs stacked prediction
    ax = fig.add_subplot(gs[0, 0])
    valid = stk.dropna(subset=['spread_target', 'stk_spread_pred'])
    ax.scatter(valid['spread_target'], valid['stk_spread_pred'], s=3, alpha=0.2, c='coral')
    lims = [-100, 150]
    ax.plot(lims, lims, 'r--', alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim([-40, 40])
    r_sp = valid['spread_target'].corr(valid['stk_spread_pred'])
    ax.set_xlabel('Actual Spread (EUR/MWh)')
    ax.set_ylabel('Predicted Spread (EUR/MWh)')
    ax.set_title(f'Spread: Actual vs Stacked Pred (r={r_sp:.3f})')

    # 4b: Spread actual vs standalone prediction
    ax = fig.add_subplot(gs[0, 1])
    valid2 = stk.dropna(subset=['spread_target', 'standalone_spread_pred'])
    ax.scatter(valid2['spread_target'], valid2['standalone_spread_pred'], s=3, alpha=0.2, c='steelblue')
    ax.plot(lims, lims, 'r--', alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim([-40, 40])
    r_sa = valid2['spread_target'].corr(valid2['standalone_spread_pred'])
    ax.set_xlabel('Actual Spread (EUR/MWh)')
    ax.set_ylabel('Predicted Spread (EUR/MWh)')
    ax.set_title(f'Spread: Actual vs Standalone Pred (r={r_sa:.3f})')

    # 4c: Stacked vs Standalone comparison
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(stk['standalone_spread_pred'], stk['stk_spread_pred'], s=3, alpha=0.2, c='gray')
    ax.plot([-40, 40], [-40, 40], 'r--', alpha=0.5)
    ax.set_xlabel('Standalone Pred')
    ax.set_ylabel('Stacked Pred')
    ax.set_title('Stacked vs Standalone Spread Predictions')
    r_ss = stk['standalone_spread_pred'].corr(stk['stk_spread_pred'])
    ax.text(0.05, 0.95, f'r = {r_ss:.3f}', transform=ax.transAxes, fontsize=12, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat'))

    # 4d: Spread direction accuracy by hour
    ax = fig.add_subplot(gs[1, 0])
    stk_h = valid.copy()
    stk_h['hour'] = stk_h.index.hour
    stk_h['correct_stk'] = (np.sign(stk_h['stk_spread_pred']) == np.sign(stk_h['spread_target']))
    stk_h['correct_sa'] = (np.sign(stk_h['standalone_spread_pred']) == np.sign(stk_h['spread_target']))
    dir_stk = stk_h.groupby('hour')['correct_stk'].mean()
    dir_sa = stk_h.groupby('hour')['correct_sa'].mean()
    x = np.arange(24)
    w = 0.35
    ax.bar(x - w/2, dir_stk.reindex(x, fill_value=0.5), w, label='Stacked', alpha=0.7, color='coral')
    ax.bar(x + w/2, dir_sa.reindex(x, fill_value=0.5), w, label='Standalone', alpha=0.7, color='steelblue')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Direction Accuracy')
    ax.set_title('Spread Direction Accuracy by Hour')
    ax.legend(fontsize=9)

    # 4e: Imbalance OOF vs actual
    ax = fig.add_subplot(gs[1, 1])
    valid_imb = stk.dropna(subset=['target', 'stk_imb_pred'])
    ax.scatter(valid_imb['target'], valid_imb['stk_imb_pred'], s=3, alpha=0.2, c='steelblue')
    ax.plot([-60, 60], [-60, 60], 'r--', alpha=0.5)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-15, 15)
    r_imb = valid_imb['target'].corr(valid_imb['stk_imb_pred'])
    ax.set_xlabel('Actual Imbalance (MWh)')
    ax.set_ylabel('Stage 2 OOF Imbalance Pred')
    ax.set_title(f'Stage 2 Imbalance OOF (r={r_imb:.3f})')

    # 4f: Spread prediction by QH within hour
    ax = fig.add_subplot(gs[1, 2])
    stk_q = valid.copy()
    stk_q['qh'] = stk_q.index.minute // 15
    for qh in range(4):
        sub = stk_q[stk_q['qh'] == qh]
        dir_acc = (np.sign(sub['stk_spread_pred']) == np.sign(sub['spread_target'])).mean()
        mae = (sub['spread_target'] - sub['stk_spread_pred']).abs().mean()
        ax.bar(qh, dir_acc, color='coral', alpha=0.7)
        ax.text(qh, dir_acc + 0.01, f'MAE={mae:.1f}', ha='center', fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Q1 (:00)', 'Q2 (:15)', 'Q3 (:30)', 'Q4 (:45)'])
    ax.set_ylabel('Direction Accuracy')
    ax.set_title('Spread Accuracy by Quarter-Hour')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5)

    # 4g: Execution spread impact
    ax = fig.add_subplot(gs[2, 0])
    exec_sp = stk['exec_spread'].dropna()
    ax.hist(exec_sp, bins=60, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.3)
    ax.axvline(exec_sp.median(), color='red', ls='--', label=f'Median={exec_sp.median():.2f}')
    ax.axvline(10, color='orange', ls='--', label='Filter=10')
    ax.set_xlabel('Bid-Ask Spread (EUR/MWh)')
    ax.set_title('Execution Spread Distribution')
    ax.legend()

    # 4h: Settlement price distribution
    ax = fig.add_subplot(gs[2, 1])
    settle = stk['imb_settlement_price'].dropna()
    ax.hist(settle, bins=60, alpha=0.7, color='coral', edgecolor='black', linewidth=0.3)
    ax.axvline(settle.mean(), color='red', ls='--', label=f'Mean={settle.mean():.0f}')
    ax.set_xlabel('Imbalance Settlement Price (EUR/MWh)')
    ax.set_title('Settlement Price Distribution')
    ax.legend()

    # 4i: Spread prediction calibration (predicted size vs actual)
    ax = fig.add_subplot(gs[2, 2])
    valid_cal = valid.copy()
    valid_cal['pred_abs'] = valid_cal['stk_spread_pred'].abs()
    bins_cal = pd.qcut(valid_cal['pred_abs'], q=8, duplicates='drop')
    cal = valid_cal.groupby(bins_cal).agg(
        pred_mean=('stk_spread_pred', lambda x: x.abs().mean()),
        actual_mean=('spread_target', lambda x: x.abs().mean()),
        count=('spread_target', 'count')
    )
    ax.scatter(cal['pred_mean'], cal['actual_mean'], s=cal['count']/5, alpha=0.7, c='coral')
    for idx, row in cal.iterrows():
        ax.annotate(f'n={int(row["count"])}', (row['pred_mean'], row['actual_mean']), fontsize=8)
    ax.plot([0, 30], [0, 30], 'r--', alpha=0.3)
    ax.set_xlabel('Mean |Predicted Spread|')
    ax.set_ylabel('Mean |Actual Spread|')
    ax.set_title('Spread Calibration (|pred| bins)')

    fig.savefig(PLOT_DIR / "04_spread_predictions.png", bbox_inches='tight')
    plt.close(fig)
    print("[+] Saved 04_spread_predictions.png")

    print(f"\n--- Spread Prediction Quality ---")
    print(f"Stacked: r={r_sp:.3f}, Standalone: r={r_sa:.3f}")
    stk_dir = (np.sign(valid['stk_spread_pred']) == np.sign(valid['spread_target'])).mean()
    sa_dir = (np.sign(valid2['standalone_spread_pred']) == np.sign(valid2['spread_target'])).mean()
    print(f"Direction acc: Stacked={stk_dir:.1%}, Standalone={sa_dir:.1%}")
    print(f"Execution spread: median={exec_sp.median():.2f}, mean={exec_sp.mean():.2f}, <=10: {(exec_sp <= 10).mean():.1%}")


# =================================================================
# PLOT 5: TRADING SIGNAL QUALITY
# =================================================================

def plot_trading_quality(data):
    """Trading backtest analysis and signal quality."""
    trades = data['trades']
    stk = data['stacked']

    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 5a: Cumulative P&L
    ax = fig.add_subplot(gs[0, 0])
    trades_sorted = trades.sort_index()
    cum_pnl = trades_sorted['pnl'].cumsum()
    ax.plot(cum_pnl.index, cum_pnl.values, color='steelblue', alpha=0.8)
    ax.fill_between(cum_pnl.index, 0, cum_pnl.values, alpha=0.1, color='steelblue')
    ax.set_ylabel('Cumulative P&L (EUR)')
    ax.set_title(f'Cumulative P&L: {cum_pnl.iloc[-1]:+,.0f} EUR')
    ax.tick_params(axis='x', rotation=30)

    # 5b: Daily P&L
    ax = fig.add_subplot(gs[0, 1])
    daily_pnl = trades_sorted.groupby(trades_sorted.index.date)['pnl'].sum()
    colors_d = ['green' if p > 0 else 'red' for p in daily_pnl.values]
    ax.bar(range(len(daily_pnl)), daily_pnl.values, color=colors_d, alpha=0.7)
    ax.axhline(daily_pnl.mean(), color='blue', ls='--', label=f'Mean={daily_pnl.mean():.0f}/day')
    ax.set_xlabel('Day')
    ax.set_ylabel('Daily P&L (EUR)')
    ax.set_title(f'Daily P&L (Win rate: {(daily_pnl > 0).mean():.0%})')
    ax.legend()

    # 5c: P&L by direction
    ax = fig.add_subplot(gs[0, 2])
    for direction, color in [('surplus', 'steelblue'), ('deficit', 'coral')]:
        sub = trades_sorted[trades_sorted['direction'] == direction]
        if len(sub) > 0:
            ax.hist(sub['pnl'], bins=40, alpha=0.5, color=color, label=f'{direction} (n={len(sub)})')
    ax.axvline(0, color='gray', ls='--')
    ax.set_xlabel('Trade P&L (EUR)')
    ax.set_title('P&L Distribution by Direction')
    ax.legend()

    # 5d: Trade P&L by hour
    ax = fig.add_subplot(gs[1, 0])
    trades_h = trades_sorted.copy()
    trades_h['hour'] = trades_h.index.hour
    hourly_pnl = trades_h.groupby('hour')['pnl'].agg(['sum', 'count', 'mean'])
    colors_h = ['green' if s > 0 else 'red' for s in hourly_pnl['mean'].values]
    ax.bar(hourly_pnl.index, hourly_pnl['mean'], color=colors_h, alpha=0.7)
    ax2_t = ax.twinx()
    ax2_t.plot(hourly_pnl.index, hourly_pnl['count'], 'ko-', alpha=0.4, markersize=3)
    ax2_t.set_ylabel('Trade Count', color='gray')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Mean Trade P&L (EUR)')
    ax.set_title('P&L by Hour')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)

    # 5e: Position size distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(trades_sorted['size'], bins=40, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Position Size (MWh)')
    ax.set_title(f'Position Sizes (mean={trades_sorted["size"].mean():.2f})')

    # 5f: P&L vs prediction size
    ax = fig.add_subplot(gs[1, 2])
    trades_pred = trades_sorted.copy()
    trades_pred['pred_abs'] = trades_pred['pred_median'].abs()
    bins_t = pd.qcut(trades_pred['pred_abs'], q=6, duplicates='drop')
    trade_quality = trades_pred.groupby(bins_t)['pnl'].agg(['mean', 'sum', 'count'])
    x_pos = range(len(trade_quality))
    colors_q = ['green' if m > 0 else 'red' for m in trade_quality['mean'].values]
    ax.bar(x_pos, trade_quality['mean'], color=colors_q, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{b.left:.1f}-{b.right:.1f}' for b in trade_quality.index], rotation=30, fontsize=8)
    ax.set_xlabel('|Prediction|')
    ax.set_ylabel('Mean P&L per Trade')
    ax.set_title('P&L vs Prediction Confidence')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)

    # 5g: Drawdown
    ax = fig.add_subplot(gs[2, 0])
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    ax.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown.values, color='red', alpha=0.7)
    ax.set_ylabel('Drawdown (EUR)')
    ax.set_title(f'Drawdown (Max: {drawdown.min():,.0f} EUR)')
    ax.tick_params(axis='x', rotation=30)

    # 5h: Win rate by weekday
    ax = fig.add_subplot(gs[2, 1])
    trades_dw = trades_sorted.copy()
    trades_dw['dow'] = trades_dw.index.dayofweek
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_pnl = trades_dw.groupby('dow')['pnl'].agg(['mean', 'sum', 'count'])
    dow_wr = trades_dw.groupby('dow').apply(lambda x: (x['pnl'] > 0).mean())
    ax.bar(range(7), dow_wr.values, color='steelblue', alpha=0.7)
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate by Day of Week')
    ax.axhline(0.5, color='red', ls='--', alpha=0.5)

    # 5i: Spread filter impact
    ax = fig.add_subplot(gs[2, 2])
    spread_thresholds = [2, 5, 8, 10, 15, 20, 50]
    filtered_results = []
    for thresh in spread_thresholds:
        sub = trades_sorted[trades_sorted['spread'] <= thresh]
        if len(sub) > 5:
            total = sub['pnl'].sum()
            avg = sub['pnl'].mean()
            wr = (sub['pnl'] > 0).mean()
            filtered_results.append({'thresh': thresh, 'total': total, 'avg': avg, 'wr': wr, 'n': len(sub)})
    fr_df = pd.DataFrame(filtered_results)
    ax.bar(range(len(fr_df)), fr_df['total'], color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(fr_df)))
    ax.set_xticklabels([f'<={t}' for t in fr_df['thresh']], fontsize=9)
    ax.set_xlabel('Max Bid-Ask Spread (EUR/MWh)')
    ax.set_ylabel('Total P&L (EUR)')
    ax.set_title('P&L by Spread Filter')

    fig.savefig(PLOT_DIR / "05_trading_quality.png", bbox_inches='tight')
    plt.close(fig)
    print("[+] Saved 05_trading_quality.png")

    print(f"\n--- Trading Quality ---")
    print(f"Total P&L: {cum_pnl.iloc[-1]:+,.0f} EUR over {len(daily_pnl)} days")
    print(f"Mean: {daily_pnl.mean():+.0f}/day, Median: {daily_pnl.median():+.0f}/day")
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
    print(f"Sharpe: {sharpe:.1f}")
    print(f"Daily win rate: {(daily_pnl > 0).mean():.0%}")
    print(f"Max drawdown: {drawdown.min():,.0f} EUR")
    print(f"Trade win rate: {(trades_sorted['pnl'] > 0).mean():.0%}")
    print(f"Trades: {len(trades)} total, {len(trades)/len(daily_pnl):.0f}/day avg")
    surplus_trades = trades_sorted[trades_sorted['direction'] == 'surplus']
    deficit_trades = trades_sorted[trades_sorted['direction'] == 'deficit']
    print(f"  Surplus: {len(surplus_trades)} trades, avg P&L={surplus_trades['pnl'].mean():+.1f}")
    print(f"  Deficit: {len(deficit_trades)} trades, avg P&L={deficit_trades['pnl'].mean():+.1f}")


# =================================================================
# PLOT 6: FEATURE SELECTION ANALYSIS
# =================================================================

def plot_feature_selection(data):
    """Elimination curve, fold stability, correlation clusters."""
    elim = data['elim']
    fold_stab = data['fold_stab']
    corr_cl = data['corr_clusters']
    perm_imp = data['perm_imp']

    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 6a: Elimination curve (P&L vs n_features)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(elim['n_features'], elim['total_pnl'], 'o-', color='steelblue', markersize=4, alpha=0.8)
    best_idx = elim['total_pnl'].idxmax()
    ax.scatter([elim.loc[best_idx, 'n_features']], [elim.loc[best_idx, 'total_pnl']],
               s=100, color='red', zorder=5, label=f'Best: {int(elim.loc[best_idx, "n_features"])} features')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Total P&L (EUR)')
    ax.set_title('Feature Elimination Curve')
    ax.legend()

    # 6b: Sharpe vs n_features
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(elim['n_features'], elim['sharpe'], 'o-', color='coral', markersize=4, alpha=0.8)
    best_sh = elim['sharpe'].idxmax()
    ax.scatter([elim.loc[best_sh, 'n_features']], [elim.loc[best_sh, 'sharpe']],
               s=100, color='red', zorder=5, label=f'Best Sharpe: {int(elim.loc[best_sh, "n_features"])} features')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe vs Number of Features')
    ax.legend()

    # 6c: Fold stability heatmap (top 30 features)
    ax = fig.add_subplot(gs[0, 2])
    # fold_stab has features as rows, months as columns
    top_feats = perm_imp.nlargest(25, 'pnl_drop_mean')['feature'].values
    valid_feats = [f for f in top_feats if f in fold_stab.index]
    if valid_feats:
        sub = fold_stab.loc[valid_feats]
        # Normalize for visualization
        if len(sub.columns) >= 2:
            im = ax.imshow(sub.values, aspect='auto', cmap='RdYlGn')
            ax.set_yticks(range(len(sub)))
            ax.set_yticklabels(sub.index, fontsize=8)
            ax.set_xticks(range(len(sub.columns)))
            ax.set_xticklabels(sub.columns, fontsize=9, rotation=30)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title('Top Feature Stability Across Folds')
    else:
        ax.text(0.5, 0.5, 'No matching features', transform=ax.transAxes, ha='center')

    # 6d: Positive vs negative P&L features
    ax = fig.add_subplot(gs[1, 0])
    perm_sorted = perm_imp.sort_values('pnl_drop_mean', ascending=True)
    n_show = 20
    bottom = perm_sorted.head(n_show)
    top = perm_sorted.tail(n_show)
    combined = pd.concat([bottom, top])
    colors_pn = ['red' if x < 0 else 'green' for x in combined['pnl_drop_mean'].values]
    ax.barh(range(len(combined)), combined['pnl_drop_mean'], color=colors_pn, alpha=0.7)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined['feature'].values, fontsize=7)
    ax.axvline(0, color='gray', ls='--')
    ax.set_xlabel('P&L Drop (EUR)')
    ax.set_title('Best & Worst Features by P&L Impact')

    # 6e: Correlation cluster network
    ax = fig.add_subplot(gs[1, 1])
    if len(corr_cl) > 0:
        # Show top correlated pairs
        top_corr = corr_cl.nlargest(20, 'correlation')
        y_pos = range(len(top_corr))
        ax.barh(y_pos, top_corr['correlation'].abs(), color='steelblue', alpha=0.7)
        labels = [f"{r['feature_1'][:15]} <> {r['feature_2'][:15]}" for _, r in top_corr.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('|Correlation|')
        ax.set_title('Top Correlated Feature Pairs')

    # 6f: Cumulative P&L importance
    ax = fig.add_subplot(gs[1, 2])
    perm_pos = perm_imp[perm_imp['is_positive']].sort_values('pnl_drop_mean', ascending=False)
    cum_pct = perm_pos['pnl_drop_mean'].cumsum() / perm_pos['pnl_drop_mean'].sum() * 100
    ax.plot(range(1, len(cum_pct)+1), cum_pct.values, 'o-', color='steelblue', markersize=3)
    ax.axhline(80, color='red', ls='--', alpha=0.5, label='80%')
    ax.axhline(95, color='orange', ls='--', alpha=0.5, label='95%')
    n_80 = (cum_pct <= 80).sum() + 1
    n_95 = (cum_pct <= 95).sum() + 1
    ax.axvline(n_80, color='red', ls=':', alpha=0.3)
    ax.axvline(n_95, color='orange', ls=':', alpha=0.3)
    ax.set_xlabel('Number of Features (ranked by importance)')
    ax.set_ylabel('Cumulative P&L Importance (%)')
    ax.set_title(f'Concentration: {n_80} features = 80%, {n_95} features = 95%')
    ax.legend()

    fig.savefig(PLOT_DIR / "06_feature_selection.png", bbox_inches='tight')
    plt.close(fig)
    print("[+] Saved 06_feature_selection.png")

    print(f"\n--- Feature Selection ---")
    print(f"Elimination curve: best P&L at {int(elim.loc[best_idx, 'n_features'])} features "
          f"({elim.loc[best_idx, 'total_pnl']:,.0f} EUR)")
    print(f"Best Sharpe at {int(elim.loc[best_sh, 'n_features'])} features ({elim.loc[best_sh, 'sharpe']:.2f})")
    print(f"P&L concentration: {n_80} features capture 80% of importance")
    n_harmful = (perm_imp['pnl_drop_mean'] < 0).sum()
    harmful_total = perm_imp[perm_imp['pnl_drop_mean'] < 0]['pnl_drop_mean'].sum()
    print(f"Harmful features: {n_harmful} features, total drag: {harmful_total:,.0f} EUR")


# =================================================================
# MAIN
# =================================================================

def main():
    print("=" * 70)
    print("DETAILED EDA: FEATURES & PREDICTIONS")
    print("ImbalanceForcastingProd")
    print("=" * 70)

    data = load_data()

    print("\n" + "=" * 70)
    print("SECTION 1: TARGET DISTRIBUTIONS")
    print("=" * 70)
    plot_target_distributions(data)

    print("\n" + "=" * 70)
    print("SECTION 2: FEATURE IMPORTANCE")
    print("=" * 70)
    plot_feature_importance(data)

    print("\n" + "=" * 70)
    print("SECTION 3: IMBALANCE PREDICTIONS")
    print("=" * 70)
    plot_imbalance_predictions(data)

    print("\n" + "=" * 70)
    print("SECTION 4: SPREAD PREDICTIONS")
    print("=" * 70)
    plot_spread_predictions(data)

    print("\n" + "=" * 70)
    print("SECTION 5: TRADING QUALITY")
    print("=" * 70)
    plot_trading_quality(data)

    print("\n" + "=" * 70)
    print("SECTION 6: FEATURE SELECTION")
    print("=" * 70)
    plot_feature_selection(data)

    print("\n" + "=" * 70)
    print(f"ALL PLOTS SAVED TO: {PLOT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
