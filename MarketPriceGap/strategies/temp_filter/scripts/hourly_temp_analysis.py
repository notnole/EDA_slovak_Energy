"""
Hourly breakdown of the temperature-filtered DA-Imbalance strategy.
Question: Does hour-of-day significantly matter for the cold/hot reversal?

Analysis:
1. Mean spread by hour, split by temperature regime (cold / normal / hot)
2. Hourly P&L contribution: which hours drive the reversal value?
3. Hour-specific reversal: would reversing only certain hours improve results?
4. Statistical significance: t-test of spread by hour in cold vs normal days
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data")
OUT_DIR = BASE / "MarketPriceGap" / "strategies" / "temp_filter"

# ============================================================================
# 1. LOAD DATA (reuse parsing from temp_reverse_backtest.py)
# ============================================================================
print("[*] Loading market data...")
mkt = pd.read_csv(BASE / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                   parse_dates=["timestamp_hour"])
mkt = mkt.dropna(subset=["spread_da_imb"])
mkt["spread_da_imb"] = mkt["spread_da_imb"].astype(float)
mkt["da_price"] = mkt["da_price"].astype(float)
mkt["imb_settlement_price"] = mkt["imb_settlement_price"].astype(float)
mkt["date"] = mkt["timestamp_hour"].dt.date
mkt["hour"] = mkt["timestamp_hour"].dt.hour
mkt = mkt.set_index("timestamp_hour").sort_index()
print(f"    {mkt.index.min()} to {mkt.index.max()}, {len(mkt)} hours")

print("[*] Loading actual temperature...")
def parse_actual_temp(filepath):
    rows = []
    with open(filepath, encoding='utf-8-sig') as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line or '(invalid)' in line or line == ',,':
            continue
        parts = line.rstrip(',').split(',')
        if len(parts) < 4:
            continue
        dt_str = parts[0].strip()
        int_part = parts[2].strip()
        dec_part = parts[3].strip()
        try:
            dt = pd.to_datetime(dt_str, format="%d/%m/%Y %H:%M:%S")
        except Exception:
            continue
        try:
            temp = float(int_part + '.' + dec_part)
        except Exception:
            continue
        rows.append({"datetime": dt, "temp_actual": temp})
    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    return df

actual_temp = parse_actual_temp(BASE / "RawData" / "ActualTemp2024-2026.csv")
daily_temp = actual_temp.resample('D').mean()
daily_temp.columns = ["daily_mean_temp"]
daily_temp.index = daily_temp.index.date

# Merge daily temp onto hourly market data
mkt["daily_temp"] = mkt["date"].map(daily_temp["daily_mean_temp"].to_dict())
mkt = mkt.dropna(subset=["daily_temp"])

# Temperature regime classification
COLD_THRESH = 0.0
HOT_THRESH = 25.0
mkt["regime"] = np.where(mkt["daily_temp"] < COLD_THRESH, "cold",
                np.where(mkt["daily_temp"] > HOT_THRESH, "hot", "normal"))

n_cold = (mkt["regime"] == "cold").sum()
n_hot = (mkt["regime"] == "hot").sum()
n_normal = (mkt["regime"] == "normal").sum()
print(f"    Regime counts: cold={n_cold}, normal={n_normal}, hot={n_hot}")

# ============================================================================
# 2. MEAN SPREAD BY HOUR AND REGIME
# ============================================================================
print("\n" + "=" * 90)
print("ANALYSIS 1: Mean DA-Imbalance Spread by Hour and Temperature Regime")
print("=" * 90)
print("(Positive spread = DA > Imbalance = baseline profitable)")
print("(Negative spread = Imbalance > DA = reversal profitable)\n")

hourly_regime = mkt.groupby(["hour", "regime"])["spread_da_imb"].agg(["mean", "std", "count"])
hourly_regime = hourly_regime.unstack("regime")

print(f"{'Hour':>4s}  |  {'Cold Mean':>10s} {'(N)':>5s}  |  {'Normal Mean':>11s} {'(N)':>6s}  |  {'Hot Mean':>10s} {'(N)':>5s}  |  {'Cold-Normal':>12s}")
print("-" * 90)
for h in range(24):
    cold_m = hourly_regime.loc[h, ("mean", "cold")] if ("mean", "cold") in hourly_regime.columns and h in hourly_regime.index else np.nan
    norm_m = hourly_regime.loc[h, ("mean", "normal")] if ("mean", "normal") in hourly_regime.columns and h in hourly_regime.index else np.nan
    hot_m = hourly_regime.loc[h, ("mean", "hot")] if ("mean", "hot") in hourly_regime.columns and h in hourly_regime.index else np.nan
    cold_n = hourly_regime.loc[h, ("count", "cold")] if ("count", "cold") in hourly_regime.columns and h in hourly_regime.index else 0
    norm_n = hourly_regime.loc[h, ("count", "normal")] if ("count", "normal") in hourly_regime.columns and h in hourly_regime.index else 0
    hot_n = hourly_regime.loc[h, ("count", "hot")] if ("count", "hot") in hourly_regime.columns and h in hourly_regime.index else 0
    diff = cold_m - norm_m if not (np.isnan(cold_m) or np.isnan(norm_m)) else np.nan

    cold_str = f"{cold_m:>10.2f}" if not np.isnan(cold_m) else f"{'N/A':>10s}"
    norm_str = f"{norm_m:>11.2f}" if not np.isnan(norm_m) else f"{'N/A':>11s}"
    hot_str = f"{hot_m:>10.2f}" if not np.isnan(hot_m) else f"{'N/A':>10s}"
    diff_str = f"{diff:>12.2f}" if not np.isnan(diff) else f"{'N/A':>12s}"

    print(f"{h:>4d}  |  {cold_str} {int(cold_n):>5d}  |  {norm_str} {int(norm_n):>6d}  |  {hot_str} {int(hot_n):>5d}  |  {diff_str}")

# ============================================================================
# 3. HOURLY P&L CONTRIBUTION: BASELINE vs REVERSAL
# ============================================================================
print("\n" + "=" * 90)
print("ANALYSIS 2: Hourly P&L Contribution (Baseline vs Reversal on Extreme Days)")
print("=" * 90)
print("Baseline P&L = +1 * spread; Reversal P&L = -1 * spread on extreme days\n")

# For each hour, compute:
# - Baseline P&L (position = +1 always)
# - Reversal P&L (position = -1 on cold/hot, +1 on normal)
# - Delta = Reversal - Baseline at that hour
hourly_pnl = pd.DataFrame(index=range(24))

for h in range(24):
    hdata = mkt[mkt["hour"] == h]

    # Baseline: always +1
    baseline_pnl = hdata["spread_da_imb"].sum()

    # Reversal: -1 on extreme, +1 on normal
    pos = np.where(hdata["regime"].isin(["cold", "hot"]), -1, 1)
    reversal_pnl = (pos * hdata["spread_da_imb"]).sum()

    hourly_pnl.loc[h, "baseline"] = baseline_pnl
    hourly_pnl.loc[h, "reversal"] = reversal_pnl
    hourly_pnl.loc[h, "delta"] = reversal_pnl - baseline_pnl

    # Cold-only contribution to the delta
    cold_data = hdata[hdata["regime"] == "cold"]
    if len(cold_data) > 0:
        # Baseline would earn: sum(spread), reversal earns: sum(-spread)
        # Delta from cold = -2 * sum(spread_cold)
        hourly_pnl.loc[h, "cold_baseline_pnl"] = cold_data["spread_da_imb"].sum()
        hourly_pnl.loc[h, "cold_reversal_pnl"] = -cold_data["spread_da_imb"].sum()
        hourly_pnl.loc[h, "cold_delta"] = -2 * cold_data["spread_da_imb"].sum()
        hourly_pnl.loc[h, "cold_mean_spread"] = cold_data["spread_da_imb"].mean()
        hourly_pnl.loc[h, "cold_count"] = len(cold_data)
    else:
        hourly_pnl.loc[h, "cold_baseline_pnl"] = 0
        hourly_pnl.loc[h, "cold_reversal_pnl"] = 0
        hourly_pnl.loc[h, "cold_delta"] = 0
        hourly_pnl.loc[h, "cold_mean_spread"] = np.nan
        hourly_pnl.loc[h, "cold_count"] = 0

print(f"{'Hour':>4s}  {'Baseline':>10s}  {'Reversal':>10s}  {'Delta':>10s}  |  {'Cold Base':>10s}  {'Cold Rev':>10s}  {'Cold Delta':>10s}  {'Cold Spread':>11s}")
print("-" * 105)
for h in range(24):
    r = hourly_pnl.loc[h]
    cs = f"{r['cold_mean_spread']:>11.2f}" if not np.isnan(r['cold_mean_spread']) else f"{'N/A':>11s}"
    print(f"{h:>4d}  {r['baseline']:>10,.0f}  {r['reversal']:>10,.0f}  {r['delta']:>+10,.0f}  |  "
          f"{r['cold_baseline_pnl']:>10,.0f}  {r['cold_reversal_pnl']:>10,.0f}  {r['cold_delta']:>+10,.0f}  {cs}")

print("-" * 105)
print(f"{'TOT':>4s}  {hourly_pnl['baseline'].sum():>10,.0f}  {hourly_pnl['reversal'].sum():>10,.0f}  {hourly_pnl['delta'].sum():>+10,.0f}  |  "
      f"{hourly_pnl['cold_baseline_pnl'].sum():>10,.0f}  {hourly_pnl['cold_reversal_pnl'].sum():>10,.0f}  {hourly_pnl['cold_delta'].sum():>+10,.0f}")

# ============================================================================
# 4. STATISTICAL SIGNIFICANCE: Cold vs Normal spread by hour
# ============================================================================
print("\n" + "=" * 90)
print("ANALYSIS 3: Statistical Significance (t-test: cold vs normal spread by hour)")
print("=" * 90)
print("H0: mean spread on cold days = mean spread on normal days\n")

print(f"{'Hour':>4s}  {'Cold Mean':>10s}  {'Normal Mean':>11s}  {'Diff':>8s}  {'t-stat':>8s}  {'p-value':>8s}  {'Sig?':>5s}  {'N_cold':>6s}  {'N_norm':>6s}")
print("-" * 85)

sig_hours = []
for h in range(24):
    cold_vals = mkt[(mkt["hour"] == h) & (mkt["regime"] == "cold")]["spread_da_imb"]
    norm_vals = mkt[(mkt["hour"] == h) & (mkt["regime"] == "normal")]["spread_da_imb"]

    if len(cold_vals) < 5 or len(norm_vals) < 5:
        print(f"{h:>4d}  {'insufficient data':>50s}")
        continue

    t_stat, p_val = stats.ttest_ind(cold_vals, norm_vals, equal_var=False)
    cold_mean = cold_vals.mean()
    norm_mean = norm_vals.mean()
    diff = cold_mean - norm_mean
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

    if p_val < 0.05:
        sig_hours.append(h)

    print(f"{h:>4d}  {cold_mean:>10.2f}  {norm_mean:>11.2f}  {diff:>8.2f}  {t_stat:>8.2f}  {p_val:>8.4f}  {sig:>5s}  {len(cold_vals):>6d}  {len(norm_vals):>6d}")

print(f"\n[+] Significant hours (p < 0.05): {sig_hours if sig_hours else 'None'}")

# ============================================================================
# 5. HOUR-SELECTIVE REVERSAL: What if we only reverse certain hours?
# ============================================================================
print("\n" + "=" * 90)
print("ANALYSIS 4: Hour-Selective Reversal (reverse cold only at specific hours)")
print("=" * 90)

# Strategy variants:
# A) Reverse ALL hours on cold days (current approach)
# B) Reverse only hours where cold mean spread < 0 (spread inverts)
# C) Reverse only hours where cold mean spread < -X (strongly inverts)
# D) Reverse only significant hours (p < 0.05)
# E) Reverse peak hours only (6-22)
# F) Reverse off-peak only (23-5)

# Identify hours where cold spread is negative
cold_negative_hours = []
cold_strongly_negative_hours = []
for h in range(24):
    cold_vals = mkt[(mkt["hour"] == h) & (mkt["regime"] == "cold")]["spread_da_imb"]
    if len(cold_vals) > 0:
        m = cold_vals.mean()
        if m < 0:
            cold_negative_hours.append(h)
        if m < -5:
            cold_strongly_negative_hours.append(h)

print(f"[*] Hours where cold mean spread < 0: {cold_negative_hours}")
print(f"[*] Hours where cold mean spread < -5: {cold_strongly_negative_hours}")
print(f"[*] Statistically significant hours: {sig_hours}")

variants = {
    "A_all_hours": list(range(24)),
    "B_negative_spread": cold_negative_hours,
    "C_strongly_neg": cold_strongly_negative_hours,
    "D_significant": sig_hours,
    "E_peak_6_22": list(range(6, 23)),
    "F_offpeak_23_5": list(range(23, 24)) + list(range(0, 6)),
}

print(f"\n{'Variant':<25s}  {'Rev Hours':>10s}  {'Total P&L':>10s}  {'Sharpe':>8s}  {'MaxDD':>10s}  {'vs Baseline':>12s}")
print("-" * 85)

# Baseline P&L (no reversal)
mkt["pnl_baseline"] = mkt["spread_da_imb"]  # position = +1
baseline_daily = mkt.groupby("date")["pnl_baseline"].sum()
baseline_total = baseline_daily.sum()
baseline_sharpe = (baseline_daily.mean() / baseline_daily.std()) * np.sqrt(252) if baseline_daily.std() > 0 else 0
baseline_dd = (baseline_daily.cumsum() - baseline_daily.cumsum().cummax()).min()
print(f"{'Baseline (no reversal)':<25s}  {'all':>10s}  {baseline_total:>10,.0f}  {baseline_sharpe:>8.2f}  {baseline_dd:>10,.0f}  {'---':>12s}")

for vname, hours in variants.items():
    # Reverse cold days only at specified hours, otherwise baseline
    pos = np.where(
        (mkt["regime"] == "cold") & (mkt["hour"].isin(hours)), -1,
        np.where(
            (mkt["regime"] == "hot") & (mkt["hour"].isin(hours)), -1,
            1
        )
    )
    pnl = pos * mkt["spread_da_imb"]
    daily = pnl.groupby(mkt["date"]).sum()
    total = daily.sum()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else 0
    cum = daily.cumsum()
    dd = (cum - cum.cummax()).min()

    print(f"{vname:<25s}  {len(hours):>10d}  {total:>10,.0f}  {sharpe:>8.2f}  {dd:>10,.0f}  {total - baseline_total:>+12,.0f}")

# ============================================================================
# 6. COLD-ONLY HOUR-BY-HOUR MARGINAL VALUE
# ============================================================================
print("\n" + "=" * 90)
print("ANALYSIS 5: Marginal Value of Reversing Each Hour on Cold Days")
print("=" * 90)
print("How much P&L does reversing EACH individual hour add vs baseline?\n")

print(f"{'Hour':>4s}  {'Baseline PnL':>12s}  {'If Reversed':>12s}  {'Marginal':>10s}  {'Cold Spread':>11s}  {'Direction':>10s}")
print("-" * 75)

marginal_values = []
for h in range(24):
    # Baseline: all hours position +1
    # Modified: reverse only hour h on cold days
    h_cold = mkt[(mkt["hour"] == h) & (mkt["regime"] == "cold")]

    if len(h_cold) == 0:
        continue

    # Baseline earns +spread, reversal earns -spread on cold hours at h
    base_earn = h_cold["spread_da_imb"].sum()
    rev_earn = -h_cold["spread_da_imb"].sum()
    marginal = rev_earn - base_earn  # = -2 * sum(spread)
    mean_sp = h_cold["spread_da_imb"].mean()
    direction = "REVERSE" if marginal > 0 else "KEEP"

    marginal_values.append({"hour": h, "marginal": marginal, "mean_spread": mean_sp})
    print(f"{h:>4d}  {base_earn:>12,.0f}  {rev_earn:>12,.0f}  {marginal:>+10,.0f}  {mean_sp:>11.2f}  {direction:>10s}")

marginal_df = pd.DataFrame(marginal_values)
total_marginal = marginal_df["marginal"].sum()
positive_hours = marginal_df[marginal_df["marginal"] > 0]["hour"].tolist()
print(f"\n[+] Hours where reversal adds value: {positive_hours}")
print(f"[+] Total marginal value (all hours): {total_marginal:,.0f} EUR")
print(f"[+] Marginal value (only positive hours): {marginal_df[marginal_df['marginal'] > 0]['marginal'].sum():,.0f} EUR")

# ============================================================================
# 7. OPTIMAL HOUR SUBSET (greedy forward selection)
# ============================================================================
print("\n" + "=" * 90)
print("ANALYSIS 6: Greedy Forward Selection of Reversal Hours")
print("=" * 90)

# Sort hours by marginal value descending
sorted_hours = marginal_df.sort_values("marginal", ascending=False)

selected = []
cumulative_gain = 0
print(f"\n{'Step':>4s}  {'Add Hour':>8s}  {'Marginal':>10s}  {'Cumulative':>12s}  {'Selected Hours'}")
print("-" * 80)

for _, row in sorted_hours.iterrows():
    h = int(row["hour"])
    m = row["marginal"]
    if m > 0:
        selected.append(h)
        cumulative_gain += m
        print(f"{len(selected):>4d}  {h:>8d}  {m:>+10,.0f}  {cumulative_gain:>+12,.0f}  {sorted(selected)}")

print(f"\n[+] Optimal reversal hours (cold days): {sorted(selected)}")
print(f"[+] Total gain from selective reversal: {cumulative_gain:,.0f} EUR")

# Compare: selective vs all-hours reversal
# All-hours reversal gain
all_hours_gain = hourly_pnl["cold_delta"].sum()
print(f"[*] All-hours reversal gain: {all_hours_gain:,.0f} EUR")
print(f"[*] Selective improvement over all-hours: {cumulative_gain - all_hours_gain:+,.0f} EUR")

# ============================================================================
# 8. VISUALIZATION
# ============================================================================
print("\n[*] Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Hour-of-Day Analysis: Temperature-Based DA-Imbalance Reversal",
             fontsize=14, fontweight='bold', y=0.98)

# Panel 1: Mean spread by hour and regime
ax1 = axes[0, 0]
hours = list(range(24))

cold_means = [mkt[(mkt["hour"] == h) & (mkt["regime"] == "cold")]["spread_da_imb"].mean() for h in hours]
normal_means = [mkt[(mkt["hour"] == h) & (mkt["regime"] == "normal")]["spread_da_imb"].mean() for h in hours]
hot_means = [mkt[(mkt["hour"] == h) & (mkt["regime"] == "hot")]["spread_da_imb"].mean()
             if len(mkt[(mkt["hour"] == h) & (mkt["regime"] == "hot")]) > 0 else np.nan for h in hours]

ax1.bar(np.array(hours) - 0.25, cold_means, 0.25, label="Cold (<0C)", color="blue", alpha=0.7)
ax1.bar(np.array(hours), normal_means, 0.25, label="Normal", color="gray", alpha=0.7)
ax1.bar(np.array(hours) + 0.25, hot_means, 0.25, label="Hot (>25C)", color="red", alpha=0.7)
ax1.axhline(0, color='black', linewidth=0.5)
ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("Mean Spread DA-Imb (EUR/MWh)")
ax1.set_title("Mean Spread by Hour and Temperature Regime")
ax1.legend(fontsize=9)
ax1.set_xticks(hours)
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: Marginal value of reversal by hour (cold days)
ax2 = axes[0, 1]
colors_bar = ['green' if m > 0 else 'red' for m in marginal_df["marginal"]]
ax2.bar(marginal_df["hour"], marginal_df["marginal"], color=colors_bar, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Marginal P&L Gain (EUR)")
ax2.set_title("Marginal Value of Reversing Each Hour (Cold Days)")
ax2.set_xticks(hours)
ax2.grid(True, alpha=0.3, axis='y')

# Annotate significant hours
for h in sig_hours:
    if h in marginal_df["hour"].values:
        val = marginal_df.loc[marginal_df["hour"] == h, "marginal"].values[0]
        ax2.annotate('*', (h, val), textcoords="offset points", xytext=(0, 5),
                    ha='center', fontsize=14, fontweight='bold', color='black')

# Panel 3: Strategy comparison bars
ax3 = axes[1, 0]
comparison_data = {
    "Baseline\n(no rev)": baseline_total,
    "All hours\nreverse": baseline_total + all_hours_gain,
    "Negative\nspread hrs": None,  # will fill
    "Significant\nhours": None,
    "Peak\n(6-22)": None,
}

# Recompute for the chart
for vname, hours_list in [("Negative\nspread hrs", cold_negative_hours),
                           ("Significant\nhours", sig_hours),
                           ("Peak\n(6-22)", list(range(6, 23)))]:
    pos = np.where(
        (mkt["regime"] == "cold") & (mkt["hour"].isin(hours_list)), -1,
        np.where(
            (mkt["regime"] == "hot") & (mkt["hour"].isin(hours_list)), -1, 1
        )
    )
    pnl_total = (pos * mkt["spread_da_imb"]).sum()
    comparison_data[vname] = pnl_total

names = list(comparison_data.keys())
values = list(comparison_data.values())
bar_colors = ['gray', 'steelblue', 'green', 'orange', 'purple']
ax3.bar(names, values, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax3.set_ylabel("Total P&L (EUR)")
ax3.set_title("Strategy Comparison: Hour-Selective Reversal")
ax3.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(values):
    ax3.text(i, v + 500, f"{v:,.0f}", ha='center', fontsize=9, fontweight='bold')

# Panel 4: Cumulative P&L by hour (cold days) - heatmap style
ax4 = axes[1, 1]
# Show cold-day spread distribution by hour (box plot)
cold_data_by_hour = [mkt[(mkt["hour"] == h) & (mkt["regime"] == "cold")]["spread_da_imb"].values for h in range(24)]
bp = ax4.boxplot(cold_data_by_hour, positions=range(24), widths=0.6,
                  patch_artist=True, showfliers=False, medianprops=dict(color='red', linewidth=1.5))
for i, patch in enumerate(bp['boxes']):
    if i in positive_hours:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.8)
    else:
        patch.set_facecolor('lightyellow')
        patch.set_alpha(0.6)

ax4.axhline(0, color='black', linewidth=1, linestyle='--')
ax4.set_xlabel("Hour of Day")
ax4.set_ylabel("Spread DA-Imb (EUR/MWh)")
ax4.set_title("Cold-Day Spread Distribution by Hour\n(blue = reversal profitable)")
ax4.set_xticks(range(24))
ax4.set_xticklabels(range(24))
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.96])
outpath = OUT_DIR / "hourly_temp_analysis.png"
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"[+] Saved: {outpath}")
plt.close()

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 90)
print("FINAL SUMMARY")
print("=" * 90)

print(f"\n1. DOES HOUR MATTER?")
print(f"   - {len(sig_hours)} of 24 hours show statistically significant (p<0.05) spread difference cold vs normal")
print(f"   - Significant hours: {sig_hours}")

print(f"\n2. REVERSAL VALUE BY HOUR:")
print(f"   - Hours where reversal adds value: {positive_hours}")
print(f"   - Hours where reversal destroys value: {[h for h in range(24) if h not in positive_hours and h in marginal_df['hour'].values]}")

print(f"\n3. STRATEGY COMPARISON:")
print(f"   - Baseline (no reversal):          {baseline_total:>10,.0f} EUR")
print(f"   - All-hours reversal:              {baseline_total + all_hours_gain:>10,.0f} EUR  (delta: {all_hours_gain:>+,.0f})")
selective_gain = marginal_df[marginal_df["marginal"] > 0]["marginal"].sum()
print(f"   - Selective reversal (best hours):  {baseline_total + selective_gain:>10,.0f} EUR  (delta: {selective_gain:>+,.0f})")
print(f"   - Selective improvement vs all-hrs: {selective_gain - all_hours_gain:>+,.0f} EUR")

pct_from_hours = (selective_gain / all_hours_gain * 100) if all_hours_gain != 0 else 0
print(f"\n4. CONCLUSION:")
if selective_gain > all_hours_gain * 1.1:
    print(f"   [+] Hour selection MATTERS: selective reversal captures {pct_from_hours:.0f}% more value")
    print(f"   [+] Recommendation: reverse only hours {sorted(positive_hours)} on cold days")
elif selective_gain > all_hours_gain * 0.9:
    print(f"   [*] Hour selection has MARGINAL impact ({pct_from_hours:.0f}% of all-hours value)")
    print(f"   [*] Full-day reversal is simpler and nearly as good")
else:
    print(f"   [-] Hour selection DOES NOT significantly improve results")
    print(f"   [-] Full-day reversal captures the effect broadly enough")

print("\n[+] Analysis complete.")
