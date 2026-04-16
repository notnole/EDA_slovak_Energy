"""
Solar Curtailment Optimization v2 - Full Period + Summer Simulation
===================================================================

Extends v1 by:
  1. Combining all available prediction data (Oct 2025 - Mar 2026)
  2. Running separately on 2025 summer (Mar-Aug) using actual imbalance
     as oracle + noised version to simulate realistic prediction quality

Rules:
  Rule 1: Curtail if pred > X                        (extreme surplus, always)
  Rule 2: Curtail if DA < A  AND  pred > Y            (moderate surplus + weak DA)
  Rule 3: Curtail if DA < B  AND  pred > Z             (very cheap DA, Z near 0)
  Combined: (pred > X) OR (DA < A AND pred > Y) OR (DA < B AND pred > Z)
  Where Z < Y < X and B < A
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parents[1]

plt.rcParams.update({
    'figure.figsize': (14, 7), 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.3
})

# ============================================================
# SOLAR MODEL
# ============================================================
PVGIS = {1:24.9, 2:41.0, 3:78.9, 4:107.8, 5:123.6, 6:131.1,
         7:133.3, 8:114.2, 9:87.0, 10:55.2, 11:27.4, 12:19.2}
DAYS  = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

def _shape(hour, month):
    dl = {1:8.5,2:10,3:11.5,4:13.5,5:15,6:16,7:15.5,8:14,9:12.5,10:11,11:9,12:8}[month]
    sr, ss = 12 - dl/2, 12 + dl/2
    if hour < sr or hour >= ss:
        return 0.0
    sp = (hour - sr) / (ss - sr)
    irr = np.sin(sp * np.pi)
    ef = np.exp(-((sp - 0.30)**2) / (2*0.12**2))
    wf = np.exp(-((sp - 0.70)**2) / (2*0.12**2))
    return 0.5*irr*(0.4 + 0.6*ef) + 0.5*irr*(0.4 + 0.6*wf)

def solar_mwh(hour, month):
    rd = sum(_shape(h, month) for h in range(24))
    if rd <= 0:
        return 0.0
    return _shape(hour, month) * PVGIS[month] / (rd * DAYS[month])


def run_grid_search(df, pred_col, label):
    """Run the three-rule grid search on a dataframe.

    Rule 1: pred > X                     (extreme surplus, always curtail)
    Rule 2: DA < A  AND pred > Y         (moderate surplus + weak DA)
    Floor:  pred <= F  -> ALWAYS PRODUCE  (override, safety net for deficit)
    Combined: IF pred <= F: produce
              ELSE: curtail if (pred > X) OR (DA < A AND pred > Y)

    Returns (grid_df, best_r1, best_r12, best_full, baseline_pm, current_pm, oracle_pm)
    """
    df = df.copy()
    total_solar = df['solar_mwh'].sum()
    n_periods = len(df)

    pred_vals = df[pred_col].values
    da_vals = df['da_price'].values
    settle_vals = df['imb_settlement_price'].values
    sol_vals = df['solar_mwh'].values

    # Baselines
    baseline_rev = (settle_vals * sol_vals).sum()
    baseline_pm = baseline_rev / total_solar

    current_mask = pred_vals > 15
    current_rev = (settle_vals[~current_mask] * sol_vals[~current_mask]).sum()
    current_pm = current_rev / total_solar

    oracle_mask = settle_vals < 0
    oracle_rev = (settle_vals[~oracle_mask] * sol_vals[~oracle_mask]).sum()
    oracle_pm = oracle_rev / total_solar

    print(f"\n[*] {label}")
    print(f"    Periods: {n_periods}, Solar: {total_solar:.1f} MWh")
    print(f"    Baseline (always produce): {baseline_pm:.1f} EUR/MWh")
    print(f"    Current (pred>15):         {current_pm:.1f} EUR/MWh")
    print(f"    Oracle (curtail neg):      {oracle_pm:.1f} EUR/MWh")

    # --- Phase 1: Find best Rule 1 + Rule 2 (no floor) ---
    print("    Phase 1: Rule 1 + Rule 2 grid...")
    rule1_grid = np.arange(5, 61, 3)
    da_a_grid = np.arange(0, 105, 5)
    rule2_y_grid = np.arange(-10, 51, 3)

    results = []

    for X in rule1_grid:
        curtail_r1 = pred_vals > X
        rev_r1 = (settle_vals[~curtail_r1] * sol_vals[~curtail_r1]).sum()

        results.append({
            'X': X, 'A': np.nan, 'Y': np.nan, 'F': np.nan,
            'rule': 'rule1_only',
            'total_rev': rev_r1,
            'rev_per_mwh': rev_r1 / total_solar,
            'n_curtailed': curtail_r1.sum(),
            'pct_curtailed': curtail_r1.mean() * 100,
        })

        for A in da_a_grid:
            for Y in rule2_y_grid:
                if Y >= X:
                    continue
                curtail = (pred_vals > X) | ((da_vals < A) & (pred_vals > Y))
                rev = (settle_vals[~curtail] * sol_vals[~curtail]).sum()
                results.append({
                    'X': X, 'A': A, 'Y': Y, 'F': np.nan,
                    'rule': 'rule12',
                    'total_rev': rev,
                    'rev_per_mwh': rev / total_solar,
                    'n_curtailed': curtail.sum(),
                    'pct_curtailed': curtail.mean() * 100,
                })

    grid_12 = pd.DataFrame(results)
    best_r1 = grid_12[grid_12['rule'] == 'rule1_only'].sort_values('rev_per_mwh', ascending=False).iloc[0]
    best_r12 = grid_12[grid_12['rule'] == 'rule12'].sort_values('rev_per_mwh', ascending=False).iloc[0]

    print(f"    Best R1: X={best_r1['X']:.0f} -> {best_r1['rev_per_mwh']:.1f} EUR/MWh")
    print(f"    Best R1+R2: X={best_r12['X']:.0f}, A={best_r12['A']:.0f}, Y={best_r12['Y']:.0f} "
          f"-> {best_r12['rev_per_mwh']:.1f} EUR/MWh")

    # --- Phase 2: Add produce floor around the best R1+R2 ---
    # Floor: if pred <= F, always produce (override curtailment)
    print("    Phase 2: Adding produce floor F...")
    X_best = best_r12['X']
    A_best = best_r12['A']
    Y_best = best_r12['Y']

    # Refine around best R1+R2
    x_refine = np.arange(max(5, X_best - 9), X_best + 10, 2)
    a_refine = np.arange(max(0, A_best - 15), A_best + 20, 5)
    y_refine = np.arange(max(-10, Y_best - 9), Y_best + 10, 2)
    # Floor: sweep from -15 to +10
    f_grid = np.arange(-15, 11, 1)

    results_f = []
    for X in x_refine:
        for A in a_refine:
            for Y in y_refine:
                if Y >= X:
                    continue
                raw_curtail = (pred_vals > X) | ((da_vals < A) & (pred_vals > Y))

                for F in f_grid:
                    # Floor overrides: if pred <= F, force produce
                    curtail = raw_curtail & (pred_vals > F)
                    rev = (settle_vals[~curtail] * sol_vals[~curtail]).sum()
                    n_curt = curtail.sum()
                    results_f.append({
                        'X': X, 'A': A, 'Y': Y, 'F': F,
                        'rule': 'full',
                        'total_rev': rev,
                        'rev_per_mwh': rev / total_solar,
                        'n_curtailed': n_curt,
                        'pct_curtailed': n_curt / n_periods * 100,
                    })

    grid_full = pd.DataFrame(results_f)
    best_full = grid_full.sort_values('rev_per_mwh', ascending=False).iloc[0]

    print(f"    Best Full: X={best_full['X']:.0f}, A={best_full['A']:.0f}, "
          f"Y={best_full['Y']:.0f}, F={best_full['F']:.0f} "
          f"-> {best_full['rev_per_mwh']:.1f} EUR/MWh (curtail {best_full['pct_curtailed']:.1f}%)")

    # Combine all results
    grid_all = pd.concat([grid_12, grid_full], ignore_index=True)

    print(f"\n  --- SUMMARY: {label} ---")
    print(f"  Baseline (no curtailment):  {baseline_pm:.1f} EUR/MWh")
    print(f"  Current (pred>15):          {current_pm:.1f} EUR/MWh")
    print(f"  Best Rule 1:                {best_r1['rev_per_mwh']:.1f} EUR/MWh  (X={best_r1['X']:.0f})")
    print(f"  Best Rule 1+2:              {best_r12['rev_per_mwh']:.1f} EUR/MWh  "
          f"(X={best_r12['X']:.0f}, A={best_r12['A']:.0f}, Y={best_r12['Y']:.0f})")
    print(f"  Best Full (R1+R2+Floor):    {best_full['rev_per_mwh']:.1f} EUR/MWh  "
          f"(X={best_full['X']:.0f}, A={best_full['A']:.0f}, Y={best_full['Y']:.0f}, "
          f"F={best_full['F']:.0f})")
    print(f"  Oracle:                     {oracle_pm:.1f} EUR/MWh")
    print(f"  Floor uplift vs R1+R2:      {best_full['rev_per_mwh'] - best_r12['rev_per_mwh']:+.1f} EUR/MWh")

    return grid_all, best_r1, best_r12, best_full, baseline_pm, current_pm, oracle_pm


# ============================================================
# LOAD DATA
# ============================================================
print("[*] Loading data...")

# Predictions v2 (Oct 2025 - Jan 2026)
pred_v2 = pd.read_csv(
    ROOT / "ImbalanceForcastingProd" / "data" / "predictions" / "predictions_test_v2.csv",
    parse_dates=['datetime'])
pred_v2 = pred_v2.rename(columns={'datetime': 'timestamp_qh'})
pred_v2 = pred_v2[['timestamp_qh', 'pred_median', 'target']].copy()

# Predictions lead8 (Feb-Mar 2026) - longest lead time
pred_lead = pd.read_csv(
    ROOT / "ImbalanceForcastingProd" / "data" / "predictions" / "predictions_lead8.csv",
    parse_dates=['datetime'])
pred_lead = pred_lead.rename(columns={'datetime': 'timestamp_qh'})
pred_lead = pred_lead[['timestamp_qh', 'pred_median', 'target']].copy()

# Combine, drop duplicates (prefer v2 where overlap)
pred_all = pd.concat([pred_v2, pred_lead], ignore_index=True).drop_duplicates(subset='timestamp_qh', keep='first')
pred_all = pred_all.sort_values('timestamp_qh').reset_index(drop=True)
print(f"  Combined predictions: {len(pred_all)} rows, "
      f"{pred_all['timestamp_qh'].min()} to {pred_all['timestamp_qh'].max()}")

# Market data (quarter-hourly)
mkt = pd.read_csv(
    ROOT / "MarketPriceGap" / "data" / "processed" / "qh_market_prices.csv",
    parse_dates=['timestamp_qh'])

# ============================================================
# DATASET 1: Real predictions (Oct 2025 - Mar 2026)
# ============================================================
df_pred = pred_all.merge(
    mkt[['timestamp_qh', 'da_price', 'imb_settlement_price', 'imbalance_mwh']],
    on='timestamp_qh', how='inner')
df_pred['month'] = df_pred['timestamp_qh'].dt.month
df_pred['hour'] = df_pred['timestamp_qh'].dt.hour
df_pred['solar_mwh'] = df_pred.apply(lambda r: solar_mwh(r['hour'], r['month']), axis=1)
df_pred = df_pred[(df_pred['solar_mwh'] > 0.001)].dropna(
    subset=['da_price', 'imb_settlement_price', 'pred_median'])

grid1, best_r1_1, best_r12_1, best_full_1, bl1, cur1, orc1 = run_grid_search(
    df_pred, 'pred_median', 'REAL PREDICTIONS (Oct 2025 - Mar 2026)')

# ============================================================
# DATASET 2: 2025 Summer (Mar-Aug) with actual imbalance as signal
# This represents what an ORACLE would achieve
# ============================================================
df_summer = mkt[
    (mkt['timestamp_qh'].dt.year == 2025) &
    (mkt['timestamp_qh'].dt.month.between(3, 8))
].copy()
df_summer['month'] = df_summer['timestamp_qh'].dt.month
df_summer['hour'] = df_summer['timestamp_qh'].dt.hour
df_summer['solar_mwh'] = df_summer.apply(lambda r: solar_mwh(r['hour'], r['month']), axis=1)
df_summer['pred_median'] = df_summer['imbalance_mwh']  # Oracle
df_summer = df_summer[(df_summer['solar_mwh'] > 0.001)].dropna(
    subset=['da_price', 'imb_settlement_price', 'pred_median'])

grid2, best_r1_2, best_r12_2, best_full_2, bl2, cur2, orc2 = run_grid_search(
    df_summer, 'pred_median', '2025 SUMMER ORACLE (actual imbalance as signal)')

# ============================================================
# DATASET 3: 2025 Summer with noised predictions
# Simulate prediction quality: pred = actual + noise(0, MAE)
# ============================================================
# Estimate prediction MAE from the real prediction data
pred_mae = (df_pred['pred_median'] - df_pred['target']).abs().mean()
print(f"\n[*] Prediction MAE from real data: {pred_mae:.1f} MWh")

np.random.seed(42)
df_noised = df_summer.copy()
noise = np.random.normal(0, pred_mae, len(df_noised))
df_noised['pred_noised'] = df_noised['imbalance_mwh'] + noise

grid3, best_r1_3, best_r12_3, best_full_3, bl3, cur3, orc3 = run_grid_search(
    df_noised, 'pred_noised',
    f'2025 SUMMER SIMULATED (actual + noise, MAE={pred_mae:.1f})')


# ============================================================
# FIGURES
# ============================================================

# --- FIGURE 1: Rule 1 sweep for all 3 datasets ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, grid, label, bl, cur, best_r1 in [
    (axes[0], grid1, 'Real Preds (Oct-Mar)', bl1, cur1, best_r1_1),
    (axes[1], grid2, 'Summer Oracle', bl2, cur2, best_r1_2),
    (axes[2], grid3, 'Summer Simulated', bl3, cur3, best_r1_3),
]:
    r1 = grid[grid['rule'] == 'rule1_only'].copy()
    ax.plot(r1['X'], r1['rev_per_mwh'], 'k-', linewidth=2, label='Rule 1: pred > X')
    ax.axhline(bl, color='gray', ls=':', lw=1.5, label=f'No curtail = {bl:.1f}')
    ax.axhline(cur, color='blue', ls='--', lw=1.5, label=f'Current (>15) = {cur:.1f}')
    ax.scatter([best_r1['X']], [best_r1['rev_per_mwh']], color='red', s=120, zorder=5, edgecolors='k')
    ax.set_xlabel('Threshold X (MWh)')
    ax.set_ylabel('Revenue (EUR/MWh)')
    ax.set_title(label)
    ax.legend(fontsize=8)

plt.suptitle('Rule 1 Sweep: Curtail if pred > X', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / '01_rule1_all_datasets.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 01_rule1_all_datasets.png")

# --- FIGURE 2: Strategy comparison across datasets ---
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

for idx, (grid, best_r1, best_r12, best_full, bl, cur, orc, label) in enumerate([
    (grid1, best_r1_1, best_r12_1, best_full_1, bl1, cur1, orc1, 'Real Preds\n(Oct-Mar)'),
    (grid2, best_r1_2, best_r12_2, best_full_2, bl2, cur2, orc2, 'Summer\nOracle'),
    (grid3, best_r1_3, best_r12_3, best_full_3, bl3, cur3, orc3, 'Summer\nSimulated'),
]):
    ax = axes[idx]
    strategies = {
        'No\ncurtail': bl,
        'Current\n(>15)': cur,
        f'R1\n(>{best_r1["X"]:.0f})': best_r1['rev_per_mwh'],
        f'R1+R2': best_r12['rev_per_mwh'],
        f'R1+R2\n+Floor': best_full['rev_per_mwh'],
        'Oracle': orc,
    }
    colors = ['#95a5a6', '#3498db', '#e67e22', '#e74c3c', '#8e44ad', '#27ae60']
    bars = ax.bar(strategies.keys(), strategies.values(), color=colors,
                  edgecolor='black', linewidth=0.5)
    for bar, rev in zip(bars, strategies.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{rev:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('EUR/MWh')
    ax.set_title(label, fontsize=12)
    ax.tick_params(axis='x', labelsize=7)

plt.suptitle('Strategy Comparison: R1 vs R1+R2 vs R1+R2+Floor', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / '02_strategy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 02_strategy_comparison.png")

# --- FIGURE 3: Decision space for summer simulated (R1+R2+Floor) ---
fig, ax = plt.subplots(figsize=(14, 9))
best = best_full_3
df_data = df_noised

sc = ax.scatter(df_data['pred_noised'], df_data['da_price'],
                c=df_data['imb_settlement_price'], cmap='RdYlGn',
                s=8, alpha=0.5, vmin=-100, vmax=200)
plt.colorbar(sc, ax=ax, label='Settlement Price (EUR/MWh)', shrink=0.8)

xlim_max = max(df_data['pred_noised'].max(), best['X'] + 10)
xlim_min = min(df_data['pred_noised'].min(), best['F'] - 5)
ylim_max = df_data['da_price'].max() + 10

# Rule 1: vertical line at X (curtail right of X)
ax.axvline(best['X'], color='red', lw=2.5, ls='--',
           label=f'R1: pred > {best["X"]:.0f} (always curtail)')
ax.axvspan(best['X'], xlim_max, alpha=0.15, color='red')

# Rule 2: DA < A, pred > Y (curtail in the box)
ax.fill_between([best['Y'], best['X']], [0, 0], [best['A'], best['A']],
                alpha=0.15, color='orange',
                label=f'R2: DA < {best["A"]:.0f} AND pred > {best["Y"]:.0f}')
ax.axhline(best['A'], color='orange', lw=1.5, ls=':', xmin=0, xmax=1)

# Floor: vertical line at F (always produce left of F)
ax.axvline(best['F'], color='green', lw=2.5, ls='--',
           label=f'Floor: pred <= {best["F"]:.0f} (always produce)')
ax.axvspan(xlim_min, best['F'], alpha=0.10, color='green')

ax.set_xlabel('Predicted Imbalance (MWh)', fontsize=12)
ax.set_ylabel('DA Price (EUR/MWh)', fontsize=12)
ax.set_title('Curtailment Decision Space (Summer Simulated)\n'
             'Green = always produce, Red/Orange = curtail zones', fontsize=13)
ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig(OUT_DIR / '03_decision_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 03_decision_space.png")

# --- FIGURE 4: Cumulative P&L comparison ---
fig, ax = plt.subplots(figsize=(14, 7))
df_sorted = df_noised.sort_values('timestamp_qh').copy()
s_vals = df_sorted['imb_settlement_price'].values
p_vals = df_sorted['pred_noised'].values
d_vals = df_sorted['da_price'].values
sol_s = df_sorted['solar_mwh'].values
dates = df_sorted['timestamp_qh'].values

bf = best_full_3
raw_r12 = (p_vals > best_r12_3['X']) | ((d_vals < best_r12_3['A']) & (p_vals > best_r12_3['Y']))
raw_full = (p_vals > bf['X']) | ((d_vals < bf['A']) & (p_vals > bf['Y']))
full_curtail = raw_full & (p_vals > bf['F'])  # floor override
strats = {
    'No curtailment': np.zeros(len(df_sorted), dtype=bool),
    'Current (pred>15)': p_vals > 15,
    f'R1+R2 (X={best_r12_3["X"]:.0f},A={best_r12_3["A"]:.0f},Y={best_r12_3["Y"]:.0f})': raw_r12,
    f'R1+R2+Floor (F={bf["F"]:.0f})': full_curtail,
    'Oracle': s_vals < 0,
}
colors_cum = ['#95a5a6', '#3498db', '#e74c3c', '#8e44ad', '#27ae60']
for (name, mask), color in zip(strats.items(), colors_cum):
    pnl = np.where(mask, 0, s_vals * sol_s)
    ax.plot(dates, np.cumsum(pnl), label=name, linewidth=1.8, color=color)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Revenue (EUR)', fontsize=12)
ax.set_title('Cumulative Revenue: 2025 Summer (Simulated Predictions)', fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / '04_cumulative_pnl.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] 04_cumulative_pnl.png")


# ============================================================
# SAVE RESULTS
# ============================================================
summary_rows = []
for label, best_r1, best_r12, best_full, bl, cur, orc in [
    ('Real Preds (Oct-Mar)', best_r1_1, best_r12_1, best_full_1, bl1, cur1, orc1),
    ('Summer Oracle', best_r1_2, best_r12_2, best_full_2, bl2, cur2, orc2),
    ('Summer Simulated', best_r1_3, best_r12_3, best_full_3, bl3, cur3, orc3),
]:
    summary_rows.append({
        'dataset': label,
        'baseline_eur_mwh': bl,
        'current_pred15_eur_mwh': cur,
        'oracle_eur_mwh': orc,
        'best_r1_X': best_r1['X'],
        'best_r1_eur_mwh': best_r1['rev_per_mwh'],
        'best_r12_X': best_r12['X'],
        'best_r12_A': best_r12['A'],
        'best_r12_Y': best_r12['Y'],
        'best_r12_eur_mwh': best_r12['rev_per_mwh'],
        'best_full_X': best_full['X'],
        'best_full_A': best_full['A'],
        'best_full_Y': best_full['Y'],
        'best_full_F': best_full['F'],
        'best_full_eur_mwh': best_full['rev_per_mwh'],
        'best_full_pct_curtailed': best_full['pct_curtailed'],
        'floor_uplift_vs_r12': best_full['rev_per_mwh'] - best_r12['rev_per_mwh'],
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / 'data' / 'summary_v2.csv', index=False)
print("\n[+] summary_v2.csv")

grid3.to_csv(OUT_DIR / 'data' / 'grid_summer_simulated.csv', index=False)
print("[+] Grid CSVs saved")

bf = best_full_3
print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print(f"\nBased on 2025 summer data (Mar-Aug) with simulated prediction noise:")
print(f"  Floor:  If pred <= {bf['F']:.0f} MWh -> ALWAYS PRODUCE")
print(f"  Rule 1: Curtail when pred > {bf['X']:.0f} MWh                    (always)")
print(f"  Rule 2: Curtail when DA < {bf['A']:.0f} EUR AND pred > {bf['Y']:.0f} MWh   (weak DA)")
print(f"\n  Logic: IF pred <= {bf['F']:.0f}: produce")
print(f"         ELSE: curtail if (pred>{bf['X']:.0f}) OR (DA<{bf['A']:.0f} AND pred>{bf['Y']:.0f})")
print(f"\n  Revenue:            {bf['rev_per_mwh']:.1f} EUR/MWh")
print(f"  vs current (>15):   {cur3:.1f} EUR/MWh  ({bf['rev_per_mwh']-cur3:+.1f})")
print(f"  vs R1+R2 (no floor):{best_r12_3['rev_per_mwh']:.1f} EUR/MWh  ({bf['rev_per_mwh']-best_r12_3['rev_per_mwh']:+.1f})")
print(f"  vs no curtailment:  {bl3:.1f} EUR/MWh  ({bf['rev_per_mwh']-bl3:+.1f})")
print(f"  Gap to oracle:      {orc3 - bf['rev_per_mwh']:.1f} EUR/MWh")
print(f"  Curtailed:          {bf['pct_curtailed']:.1f}%")
print(f"\n[+] Done.")
