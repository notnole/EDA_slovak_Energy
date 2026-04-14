"""
03 Feature-vs-Label Relationships
=================================
For each feature group, creates a 2x2 panel:
  TL: scatter with binned means (20 quantiles)
  TR: monthly rolling correlation (30-day) colored by year
  BL: box plot of spread target by feature quintile
  BR: year-over-year comparison (separate regression lines per year)
"""
import sys, os, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

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
df['year'] = df.index.year

print(f"[+] Dataset ready: {len(df)} rows, spread_target mean={df['spread_target'].mean():.2f}")

# ---- feature groups ----
FEATURE_GROUPS = {
    '01_proxy_regulation': ['proxy_rmean16', 'proxy_dev_from_hour', 'reg_rmean8'],
    '02_weather': ['temp_forecast_da', 'temp_national_spread', 'cloudcover', 'radiation_national'],
    '03_da_prices': ['da_price', 'da_price_change24h', 'da_supply', 'da_flow_cz'],
    '04_load_nowcast': ['nowcast_pred_rmean4', 'nowcast_momentum_h2h3', 'nowcast_h5'],
    '05_market_features': ['idm_vwap_lag', 'load_rmean16', 'load_momentum'],
}

# Map short names that may differ in df_base
RENAME_MAP = {
    'radiation_national': 'radiation_national_mean',
    'temp_national_spread': 'temp_national_spread',
}


def resolve_col(name):
    """Find the actual column name in df."""
    if name in df.columns:
        return name
    alt = RENAME_MAP.get(name)
    if alt and alt in df.columns:
        return alt
    # try partial match
    for c in df.columns:
        if name in c:
            return c
    return None


def plot_feature_group(group_name, feature_list):
    """Create a 2x2 panel per feature, arranged in a grid."""
    available = []
    for f in feature_list:
        col = resolve_col(f)
        if col is not None:
            available.append((f, col))
        else:
            print(f"  [-] Feature '{f}' not found, skipping")

    if not available:
        print(f"  [!] No features available for {group_name}")
        return

    n_features = len(available)
    fig, axes = plt.subplots(n_features, 4, figsize=(22, 5 * n_features))
    if n_features == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Feature-Label Relationships: {group_name.replace("_", " ").title()}',
                 fontsize=16, fontweight='bold', y=0.98)

    for i, (feat_name, col_name) in enumerate(available):
        sub = df[[col_name, 'spread_target', 'year']].dropna()
        if len(sub) < 100:
            print(f"  [-] Too few rows for {feat_name}: {len(sub)}")
            continue

        ax_scatter = axes[i, 0]
        ax_rolling = axes[i, 1]
        ax_box = axes[i, 2]
        ax_yoy = axes[i, 3]

        # --- TL: Scatter with binned means ---
        try:
            sub['bin'] = pd.qcut(sub[col_name], 20, labels=False, duplicates='drop')
        except ValueError:
            sub['bin'] = pd.cut(sub[col_name], 20, labels=False)
        bin_means = sub.groupby('bin').agg(
            x_mean=(col_name, 'mean'),
            y_mean=('spread_target', 'mean'),
            y_std=('spread_target', 'std'),
            count=('spread_target', 'count'),
        )
        ax_scatter.scatter(sub[col_name], sub['spread_target'], alpha=0.02, s=3, c='grey')
        ax_scatter.errorbar(bin_means['x_mean'], bin_means['y_mean'],
                           yerr=bin_means['y_std'] / np.sqrt(bin_means['count']),
                           fmt='o-', color='crimson', linewidth=2, markersize=5, capsize=3, zorder=5)
        ax_scatter.axhline(0, color='black', linewidth=0.5, linestyle='--')
        r = sub[[col_name, 'spread_target']].corr().iloc[0, 1]
        ax_scatter.set_title(f'{feat_name}\nScatter + Binned Means (r={r:.3f})', fontsize=10)
        ax_scatter.set_xlabel(feat_name, fontsize=9)
        ax_scatter.set_ylabel('Spread Target', fontsize=9)

        # --- TR: Monthly rolling correlation ---
        corr_series = sub[[col_name, 'spread_target']].sort_index()
        rolling_corr = corr_series[col_name].rolling(window=2880, min_periods=500).corr(
            corr_series['spread_target'])  # ~30 days at 15min
        for yr, color in YEAR_COLORS.items():
            mask = rolling_corr.index.year == yr
            if mask.any():
                ax_rolling.plot(rolling_corr.index[mask], rolling_corr.values[mask],
                              color=color, label=str(yr), linewidth=1.2, alpha=0.8)
        ax_rolling.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax_rolling.set_title(f'{feat_name}\n30-Day Rolling Correlation', fontsize=10)
        ax_rolling.legend(fontsize=8)
        ax_rolling.set_ylabel('Correlation', fontsize=9)
        ax_rolling.set_ylim(-0.5, 0.5)

        # --- BL: Box plot by quintile ---
        try:
            sub['quintile'] = pd.qcut(sub[col_name], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                       duplicates='drop')
        except ValueError:
            sub['quintile'] = pd.cut(sub[col_name], 5, labels=False)
            sub['quintile'] = sub['quintile'].astype(str)

        quintile_data = [g['spread_target'].values for _, g in sub.groupby('quintile', sort=True)]
        quintile_labels = [str(k) for k in sorted(sub['quintile'].dropna().unique())]
        bp = ax_box.boxplot(quintile_data, labels=quintile_labels[:len(quintile_data)],
                           patch_artist=True, showfliers=False,
                           medianprops=dict(color='crimson', linewidth=2))
        colors_box = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(bp['boxes'])))
        for patch, c in zip(bp['boxes'], colors_box):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax_box.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax_box.set_title(f'{feat_name}\nSpread by Feature Quintile', fontsize=10)
        ax_box.set_xlabel('Quintile', fontsize=9)
        ax_box.set_ylabel('Spread Target', fontsize=9)

        # --- BR: Year-over-year regression ---
        for yr, color in YEAR_COLORS.items():
            yr_sub = sub[sub['year'] == yr]
            if len(yr_sub) < 50:
                continue
            x = yr_sub[col_name].values
            y = yr_sub['spread_target'].values
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() < 50:
                continue
            slope, intercept, r_val, _, _ = stats.linregress(x[finite], y[finite])
            x_range = np.linspace(np.percentile(x[finite], 5), np.percentile(x[finite], 95), 50)
            ax_yoy.plot(x_range, slope * x_range + intercept, color=color,
                       label=f'{yr} (r={r_val:.2f})', linewidth=2)
            ax_yoy.scatter(x[finite], y[finite], alpha=0.02, s=2, color=color)
        ax_yoy.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax_yoy.set_title(f'{feat_name}\nYear-over-Year Regression', fontsize=10)
        ax_yoy.legend(fontsize=8)
        ax_yoy.set_xlabel(feat_name, fontsize=9)
        ax_yoy.set_ylabel('Spread Target', fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / f'{group_name}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved {out_path.name}")


# ---- generate all plots ----
print("\n[*] Generating feature-vs-label plots...")
for group_name, features in FEATURE_GROUPS.items():
    print(f"\n--- {group_name} ---")
    plot_feature_group(group_name, features)

# ---- Compute summary statistics ----
print("\n[*] Computing summary statistics for summary.md...")

summary_lines = []
summary_lines.append("# 03 Feature-vs-Label Relationships\n")
summary_lines.append("## Overall Correlations with Spread Target\n")

all_corrs = {}
for group_name, features in FEATURE_GROUPS.items():
    for f in features:
        col = resolve_col(f)
        if col is not None:
            sub = df[[col, 'spread_target']].dropna()
            if len(sub) > 100:
                r = sub.corr().iloc[0, 1]
                all_corrs[f] = r

sorted_corrs = sorted(all_corrs.items(), key=lambda x: abs(x[1]), reverse=True)

summary_lines.append("| Feature | Correlation | Abs Corr |")
summary_lines.append("|---------|------------|----------|")
for feat, r in sorted_corrs:
    summary_lines.append(f"| {feat} | {r:+.4f} | {abs(r):.4f} |")

summary_lines.append("\n## Key Findings\n")

# Strongest
top3 = sorted_corrs[:3]
summary_lines.append("### Strongest Relationships")
for feat, r in top3:
    summary_lines.append(f"- **{feat}** (r={r:+.3f}): {'Positive' if r > 0 else 'Negative'} relationship")

# Nonlinearity check
summary_lines.append("\n### Linearity Assessment")
for feat, r in sorted_corrs:
    col = resolve_col(feat)
    if col is None:
        continue
    sub = df[[col, 'spread_target']].dropna()
    if len(sub) < 200:
        continue
    try:
        bins = pd.qcut(sub[col], 10, labels=False, duplicates='drop')
    except ValueError:
        continue
    bin_means = sub.groupby(bins)['spread_target'].mean()
    monotonic = bin_means.is_monotonic_increasing or bin_means.is_monotonic_decreasing
    if not monotonic:
        summary_lines.append(f"- **{feat}**: Non-monotonic pattern across deciles (nonlinear)")
    else:
        summary_lines.append(f"- **{feat}**: Approximately monotonic (linear)")

# Year stability
summary_lines.append("\n### Year-over-Year Stability")
for feat, _ in sorted_corrs:
    col = resolve_col(feat)
    if col is None:
        continue
    yr_corrs = {}
    for yr in [2024, 2025, 2026]:
        sub = df[df['year'] == yr][[col, 'spread_target']].dropna()
        if len(sub) > 100:
            yr_corrs[yr] = sub.corr().iloc[0, 1]
    if len(yr_corrs) >= 2:
        vals = list(yr_corrs.values())
        stability = "Stable" if (max(vals) - min(vals)) < 0.1 else "Unstable"
        yr_str = ", ".join(f"{y}: {r:+.3f}" for y, r in yr_corrs.items())
        summary_lines.append(f"- **{feat}** [{stability}]: {yr_str}")

summary_text = "\n".join(summary_lines)

with open(OUT_DIR / "summary.md", 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(f"\n[+] Saved summary.md")
print("[+] 03_features_vs_label complete!")
