"""
04 Temporal Stability
=====================
Analyzes how feature-label relationships change over time:
  01: Monthly correlation heatmap (top 25 features x months)
  02: Walk-forward feature importance drift (top 10 features)
  03: Direction accuracy evolution + mean |spread| + P&L/day
"""
import sys, os, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

# --- Data setup ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

LEAD = 8
OUT_DIR = Path(__file__).resolve().parent

YEAR_COLORS = {2024: 'steelblue', 2025: 'forestgreen', 2026: 'indianred'}

plt.rcParams.update({
    "figure.figsize": (18, 12), "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})

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

# ---- load data ----
print("[*] Loading data...")
data = load_all_data()
tml.TRAIN_END = '2026-04-15'
tml.TEST_START = '2026-04-15'
df_base, feature_cols = build_features(data, LEAD)

ob = pd.read_csv(
    Path(__file__).resolve().parents[2] / "data" / "features" / "orderbook_qh_features.csv",
    parse_dates=['delivery_start'],
)
ob_120 = ob[ob['lead_minutes'] == 120].set_index('delivery_start')[['bid', 'ask', 'mid', 'spread']]
ob_120 = ob_120[~ob_120.index.duplicated(keep='last')]
ob_120.columns = ['exec_bid', 'exec_ask', 'exec_spread', 'exec_mid']
df_base = df_base.join(ob_120, how='left')
df_base['imb_settlement_price'] = df_base['imb_settle_price']
df_base['spread_target'] = df_base['imb_settlement_price'] - df_base['exec_mid']

df = df_base.dropna(subset=['spread_target']).copy()

# Resolve selected features to actual columns
def resolve_col(name):
    if name in df.columns:
        return name
    for c in df.columns:
        if name in c:
            return c
    return None

sel_cols = []
sel_names = []
for f in SELECTED_50:
    col = resolve_col(f)
    if col is not None:
        sel_cols.append(col)
        sel_names.append(f)

print(f"[+] Dataset: {len(df)} rows, {len(sel_cols)}/{len(SELECTED_50)} features resolved")

# ================================================================
# PLOT 01: Monthly Correlation Heatmap
# ================================================================
print("\n[*] 01: Monthly correlation heatmap...")

df['ym'] = df.index.to_period('M')
months = sorted(df['ym'].unique())

# Compute correlations per month for top 25 features (by abs avg corr)
avg_corrs = {}
for f, col in zip(sel_names, sel_cols):
    sub = df[[col, 'spread_target']].dropna()
    if len(sub) > 100:
        avg_corrs[f] = abs(sub.corr().iloc[0, 1])
top25 = sorted(avg_corrs, key=avg_corrs.get, reverse=True)[:25]

corr_matrix = pd.DataFrame(index=top25, columns=[str(m) for m in months], dtype=float)

for m in months:
    m_data = df[df['ym'] == m]
    for f in top25:
        col = resolve_col(f)
        if col is None:
            continue
        sub = m_data[[col, 'spread_target']].dropna()
        if len(sub) > 30:
            corr_matrix.loc[f, str(m)] = sub.corr().iloc[0, 1]

corr_matrix = corr_matrix.astype(float)

fig, ax = plt.subplots(figsize=(20, 12))
im = ax.imshow(corr_matrix.values, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)

ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(corr_matrix.index)))
ax.set_yticklabels(corr_matrix.index, fontsize=9)

# Annotate cells
for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        val = corr_matrix.iloc[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.15 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

fig.colorbar(im, ax=ax, label='Pearson Correlation', shrink=0.8)
ax.set_title('Monthly Feature-Spread Correlations (Top 25 Features)', fontsize=14, fontweight='bold')

fig.tight_layout()
fig.savefig(OUT_DIR / '01_monthly_correlations.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[+] Saved 01_monthly_correlations.png")

# ================================================================
# PLOT 02: Feature Importance Drift (Walk-Forward)
# ================================================================
print("\n[*] 02: Feature importance drift (walk-forward)...")

MONTHLY_FOLDS = []
for year in [2024, 2025, 2026]:
    for month in range(1, 13):
        start = f'{year}-{month:02d}-01'
        if month == 12:
            end = f'{year+1}-01-01'
        else:
            end = f'{year}-{month+1:02d}-01'
        if pd.Timestamp(start) >= pd.Timestamp('2024-07-01') and pd.Timestamp(start) < pd.Timestamp('2026-04-13'):
            MONTHLY_FOLDS.append((start, start, end))

# Use only available selected features
use_feats = [c for c in sel_cols if c in df.columns]

# LightGBM handles NaN natively, so only require spread_target to be non-null
df_valid = df.dropna(subset=['spread_target']).copy()

importance_records = []
accuracy_records = []

for fold_train_end, test_start, test_end in MONTHLY_FOLDS:
    train = df_valid[df_valid.index < fold_train_end]
    test = df_valid[(df_valid.index >= test_start) & (df_valid.index < test_end)]

    if len(train) < 500 or len(test) < 50:
        print(f"  [-] Skipping {test_start}: train={len(train)}, test={len(test)}")
        continue

    X_train = train[use_feats].values
    y_train = train['spread_target'].values
    X_test = test[use_feats].values
    y_test = test['spread_target'].values

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train)

    # Feature importance
    imp = model.feature_importances_
    imp_norm = imp / imp.sum() if imp.sum() > 0 else imp
    for i, feat in enumerate(use_feats):
        # Map back to readable name
        feat_name = sel_names[sel_cols.index(feat)] if feat in sel_cols else feat
        importance_records.append({
            'month': test_start, 'feature': feat_name, 'importance': imp_norm[i]
        })

    # Predictions & accuracy
    preds = model.predict(X_test)
    dir_correct = np.sign(preds) == np.sign(y_test)
    dir_acc = dir_correct.mean()
    mean_abs_spread = np.abs(y_test).mean()

    # P&L: sign(pred) * actual spread, 1 MWh position
    pnl = np.sign(preds) * y_test
    days_in_month = (pd.Timestamp(test_end) - pd.Timestamp(test_start)).days
    pnl_per_day = pnl.sum() / max(days_in_month, 1)

    accuracy_records.append({
        'month': test_start,
        'direction_accuracy': dir_acc,
        'mean_abs_spread': mean_abs_spread,
        'pnl_per_day': pnl_per_day,
        'n_samples': len(test),
    })
    print(f"  [+] {test_start}: acc={dir_acc:.1%}, |spread|={mean_abs_spread:.1f}, PnL/day={pnl_per_day:.1f}")

imp_df = pd.DataFrame(importance_records)
acc_df = pd.DataFrame(accuracy_records)
acc_df['month'] = pd.to_datetime(acc_df['month'])

# Top 10 features by average importance
avg_imp = imp_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
top10 = avg_imp.head(10).index.tolist()

# Pivot importance for top 10
imp_pivot = imp_df[imp_df['feature'].isin(top10)].pivot_table(
    index='month', columns='feature', values='importance', aggfunc='mean'
)
imp_pivot.index = pd.to_datetime(imp_pivot.index)
imp_pivot = imp_pivot[top10]  # maintain order

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [3, 1]})

colors_10 = plt.cm.tab10(np.linspace(0, 1, 10))
for i, feat in enumerate(top10):
    if feat in imp_pivot.columns:
        ax1.plot(imp_pivot.index, imp_pivot[feat], color=colors_10[i],
                label=feat, linewidth=2, marker='o', markersize=4)
ax1.set_title('Feature Importance Drift (Walk-Forward Monthly Models)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Normalized Importance')
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.tick_params(axis='x', rotation=45)

# Bottom: importance change (last 3 months vs first 3 months)
if len(imp_pivot) >= 6:
    early = imp_pivot.iloc[:3].mean()
    late = imp_pivot.iloc[-3:].mean()
    change = late - early
    change = change.sort_values()
    colors_bar = ['indianred' if v < 0 else 'forestgreen' for v in change.values]
    ax2.barh(range(len(change)), change.values, color=colors_bar, alpha=0.7)
    ax2.set_yticks(range(len(change)))
    ax2.set_yticklabels(change.index, fontsize=9)
    ax2.set_xlabel('Importance Change (Recent 3mo - Early 3mo)')
    ax2.set_title('Features Gaining vs Losing Importance', fontsize=12)
    ax2.axvline(0, color='black', linewidth=0.5)

fig.tight_layout()
fig.savefig(OUT_DIR / '02_feature_importance_drift.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[+] Saved 02_feature_importance_drift.png")

# ================================================================
# PLOT 03: Direction Accuracy Evolution
# ================================================================
print("\n[*] 03: Direction accuracy evolution...")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

# Panel 1: Direction accuracy bars
bar_colors = [YEAR_COLORS.get(pd.Timestamp(m).year, 'grey') for m in acc_df['month']]
ax1.bar(acc_df['month'], acc_df['direction_accuracy'] * 100, width=20, color=bar_colors, alpha=0.8)
ax1.axhline(50, color='black', linewidth=1, linestyle='--', label='Random (50%)')
ax1.set_ylabel('Direction Accuracy (%)')
ax1.set_title('Monthly Walk-Forward Direction Accuracy + Spread Magnitude + P&L',
              fontsize=14, fontweight='bold')
ax1.legend()
ax1.set_ylim(40, 70)

# Overlay mean |spread| on secondary axis
ax1b = ax1.twinx()
ax1b.plot(acc_df['month'], acc_df['mean_abs_spread'], color='darkred',
          linewidth=2, marker='s', markersize=5, label='Mean |Spread|')
ax1b.set_ylabel('Mean |Spread| (EUR/MWh)', color='darkred')
ax1b.tick_params(axis='y', labelcolor='darkred')
ax1b.legend(loc='upper left')

# Panel 2: P&L per day
pnl_colors = ['forestgreen' if v > 0 else 'indianred' for v in acc_df['pnl_per_day']]
ax2.bar(acc_df['month'], acc_df['pnl_per_day'], width=20, color=pnl_colors, alpha=0.8)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_ylabel('P&L per Day (EUR/MWh)')
ax2.set_title('Monthly P&L per Day (1 MWh flat position, direction-based)', fontsize=12)
cumulative_pnl = acc_df['pnl_per_day'].cumsum()
ax2b = ax2.twinx()
ax2b.plot(acc_df['month'], cumulative_pnl, color='navy', linewidth=2, marker='o', markersize=4)
ax2b.set_ylabel('Cumulative P&L/day', color='navy')
ax2b.tick_params(axis='y', labelcolor='navy')

# Panel 3: Scatter accuracy vs |spread|
for yr, color in YEAR_COLORS.items():
    mask = acc_df['month'].dt.year == yr
    if mask.any():
        ax3.scatter(acc_df.loc[mask, 'mean_abs_spread'],
                   acc_df.loc[mask, 'direction_accuracy'] * 100,
                   color=color, s=80, label=str(yr), zorder=5, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('Mean |Spread| (EUR/MWh)')
ax3.set_ylabel('Direction Accuracy (%)')
ax3.set_title('Accuracy vs Spread Magnitude: Does Accuracy Drop When Spreads Compress?', fontsize=12)
ax3.axhline(50, color='black', linewidth=1, linestyle='--')
ax3.legend()

# Add correlation annotation
if len(acc_df) > 3:
    from scipy import stats
    r, p = stats.pearsonr(acc_df['mean_abs_spread'], acc_df['direction_accuracy'])
    ax3.annotate(f'r = {r:.2f} (p = {p:.3f})', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.tight_layout()
fig.savefig(OUT_DIR / '03_direction_accuracy_evolution.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("[+] Saved 03_direction_accuracy_evolution.png")

# ================================================================
# SUMMARY
# ================================================================
print("\n[*] Writing summary.md...")

summary_lines = ["# 04 Temporal Stability\n"]

# Feature importance drift summary
summary_lines.append("## Feature Importance Drift\n")
if len(imp_pivot) >= 6:
    early = imp_pivot.iloc[:3].mean()
    late = imp_pivot.iloc[-3:].mean()
    change = (late - early).sort_values(ascending=False)
    gainers = change[change > 0.005]
    losers = change[change < -0.005]
    summary_lines.append("### Features Gaining Importance (2026 vs 2024)")
    for f in gainers.index:
        summary_lines.append(f"- **{f}**: +{change[f]:.4f}")
    summary_lines.append("\n### Features Losing Importance")
    for f in losers.index:
        summary_lines.append(f"- **{f}**: {change[f]:.4f}")

# 2026 vs 2025 comparison
summary_lines.append("\n## Model Signal Shift: 2025 vs 2026\n")
imp_by_year = {}
for _, row in imp_df.iterrows():
    yr = pd.Timestamp(row['month']).year
    if yr not in imp_by_year:
        imp_by_year[yr] = {}
    if row['feature'] not in imp_by_year[yr]:
        imp_by_year[yr][row['feature']] = []
    imp_by_year[yr][row['feature']].append(row['importance'])

for yr in imp_by_year:
    imp_by_year[yr] = {f: np.mean(v) for f, v in imp_by_year[yr].items()}

if 2025 in imp_by_year and 2026 in imp_by_year:
    summary_lines.append("| Feature | 2025 Importance | 2026 Importance | Change |")
    summary_lines.append("|---------|----------------|----------------|--------|")
    all_feats_yr = set(list(imp_by_year[2025].keys()) + list(imp_by_year[2026].keys()))
    feat_changes = []
    for f in all_feats_yr:
        v25 = imp_by_year[2025].get(f, 0)
        v26 = imp_by_year[2026].get(f, 0)
        feat_changes.append((f, v25, v26, v26 - v25))
    feat_changes.sort(key=lambda x: abs(x[3]), reverse=True)
    for f, v25, v26, ch in feat_changes[:15]:
        summary_lines.append(f"| {f} | {v25:.4f} | {v26:.4f} | {ch:+.4f} |")

# Direction accuracy vs spread
summary_lines.append("\n## Direction Accuracy vs Spread Magnitude\n")
if len(acc_df) > 3:
    r, p = stats.pearsonr(acc_df['mean_abs_spread'], acc_df['direction_accuracy'])
    summary_lines.append(f"- Correlation between |spread| and accuracy: r = {r:.2f} (p = {p:.3f})")
    if r > 0.3:
        summary_lines.append("- **Accuracy tracks spread magnitude**: model performs better when spreads are wider")
    elif r < -0.3:
        summary_lines.append("- **Inverse relationship**: model struggles when spreads are wide (regime shifts?)")
    else:
        summary_lines.append("- **Weak relationship**: accuracy does not strongly depend on spread magnitude")

summary_lines.append("\n## Monthly Performance Summary\n")
summary_lines.append("| Month | Dir Accuracy | Mean |Spread| | P&L/Day |")
summary_lines.append("|-------|-------------|----------------|---------|")
for _, row in acc_df.iterrows():
    summary_lines.append(
        f"| {row['month'].strftime('%Y-%m')} | {row['direction_accuracy']:.1%} "
        f"| {row['mean_abs_spread']:.1f} | {row['pnl_per_day']:.1f} |"
    )

avg_acc = acc_df['direction_accuracy'].mean()
avg_pnl = acc_df['pnl_per_day'].mean()
summary_lines.append(f"\n**Average direction accuracy**: {avg_acc:.1%}")
summary_lines.append(f"**Average P&L/day**: {avg_pnl:.1f} EUR/MWh")

summary_text = "\n".join(summary_lines)
with open(OUT_DIR / "summary.md", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print("[+] Saved summary.md")
print("[+] 04_temporal_stability complete!")
