"""
04_decision_rules.py - Extract decision rules and backtest them.

Based on analysis findings:
  - Spread (DA vs IDM): weekend/weekday + load deviation are dominant
  - Peak timing: nuclear output is the key driver (r=-0.87)
  - Volatility: yesterday's volatility persistence + HU flow

For each target, build simple threshold rules and measure accuracy.
No ML -- only 139 days of data.

Input:  da_daily_features.csv
Output: plots/05_decision_rules.png, backtest results printed
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "MarketPriceGap" / "da_decision_matrix" / "data"
PLOT_DIR = ROOT / "MarketPriceGap" / "da_decision_matrix" / "plots"


def load_data():
    df = pd.read_csv(DATA_DIR / "da_daily_features.csv", parse_dates=[0], index_col=0)
    print(f"[*] Loaded {len(df)} days, {len(df.columns)} columns")
    return df


# ==============================================================
# RULE 1: DA-IDM Spread Direction
# ==============================================================
def rule_spread_direction(df):
    """
    Predict: will DA be more expensive than IDM (spread > 0)?

    Key findings from analysis:
      - Weekend: spread = +28.8 (DA expensive)
      - Weekday + high load dev: spread = -28.0 (DA cheap)
      - Weekday + low load dev: spread = +26.5 (DA expensive)
      - spread_yesterday persistence: r = 0.41
    """
    print("\n" + "=" * 60)
    print("RULE 1: DA-IDM Spread Direction")
    print("=" * 60)

    valid = df[["spread_da_idm_mean", "spread_positive", "is_weekend",
                "load_fcst_vs_7d", "spread_yesterday", "day_of_week"]].dropna()
    n = len(valid)
    print(f"[*] {n} days with spread data")

    # Compute thresholds
    load_dev_median = valid["load_fcst_vs_7d"].median()

    # Rule: Weekend → DA > IDM; Weekday + high load deviation → DA < IDM
    predictions = []
    pred_spread = []
    for _, row in valid.iterrows():
        if row["is_weekend"] == 1:
            predictions.append(1)  # DA > IDM
            pred_spread.append(+25.0)
        elif row["load_fcst_vs_7d"] > load_dev_median:
            predictions.append(0)  # DA < IDM (high load pushes DA down)
            pred_spread.append(-15.0)
        else:
            # Low load deviation weekday — use yesterday's spread as tiebreaker
            if row["spread_yesterday"] > 0:
                predictions.append(1)
                pred_spread.append(+10.0)
            else:
                predictions.append(0)
                pred_spread.append(-5.0)

    valid = valid.copy()
    valid["pred_direction"] = predictions
    valid["pred_spread"] = pred_spread
    actual_dir = (valid["spread_da_idm_mean"] > 0).astype(int)

    # Accuracy
    correct = (valid["pred_direction"] == actual_dir).sum()
    accuracy = correct / n
    print(f"\n--- Rule-Based Results ---")
    print(f"  Direction accuracy: {correct}/{n} = {accuracy:.1%}")

    # Baseline: always predict majority class
    majority = actual_dir.mode()[0]
    baseline_acc = (actual_dir == majority).mean()
    print(f"  Baseline (always={majority}): {baseline_acc:.1%}")
    print(f"  Skill vs baseline: {accuracy - baseline_acc:+.1%}")

    # MAE of spread prediction
    mae = (valid["spread_da_idm_mean"] - valid["pred_spread"]).abs().mean()
    naive_mae = valid["spread_da_idm_mean"].abs().mean()
    print(f"  Spread MAE: {mae:.1f} EUR/MWh (naive MAE = {naive_mae:.1f})")

    # Breakdown by condition
    for cond, label in [
        (valid["is_weekend"] == 1, "Weekend"),
        ((valid["is_weekend"] == 0) & (valid["load_fcst_vs_7d"] > load_dev_median), "Weekday + High Load Dev"),
        ((valid["is_weekend"] == 0) & (valid["load_fcst_vs_7d"] <= load_dev_median), "Weekday + Low Load Dev"),
    ]:
        sub = valid[cond]
        if len(sub) > 0:
            acc = (sub["pred_direction"] == (sub["spread_da_idm_mean"] > 0).astype(int)).mean()
            avg_spread = sub["spread_da_idm_mean"].mean()
            print(f"    {label:30s}: acc={acc:.0%} (n={len(sub)}, avg spread={avg_spread:+.1f})")

    return valid


# ==============================================================
# RULE 2: Peak Price Timing
# ==============================================================
def rule_peak_timing(df):
    """
    Predict: when will the highest DA price occur?

    Key finding: nuclear output is the dominant driver (r=-0.87).
    Low nuclear → evening peak; High nuclear → morning peak.
    """
    print("\n" + "=" * 60)
    print("RULE 2: Peak Price Timing")
    print("=" * 60)

    valid = df[["da_price_max_hour", "peak_in_evening",
                "nuclear_yesterday", "temp_fcst_mean", "is_weekend"]].dropna()
    n = len(valid)
    print(f"[*] {n} days with peak timing data")

    # Use yesterday's nuclear (strictly DA-available)
    nuc_yesterday_median = valid["nuclear_yesterday"].median()

    # Rule: low nuclear yesterday → evening peak (H16+)
    # Nuclear < median AND warm temp → evening peak
    predictions_evening = []
    pred_hours = []
    for _, row in valid.iterrows():
        nuc = row["nuclear_yesterday"]
        temp = row["temp_fcst_mean"]

        if nuc < nuc_yesterday_median - 100:
            # Very low nuclear → evening peak
            predictions_evening.append(1)
            pred_hours.append(18)
        elif nuc < nuc_yesterday_median and temp > 5:
            # Below-median nuclear + warm → likely evening
            predictions_evening.append(1)
            pred_hours.append(17)
        else:
            # Normal/high nuclear → morning peak
            predictions_evening.append(0)
            if row["is_weekend"] == 1:
                pred_hours.append(8)  # Weekend morning peak tends later
            else:
                pred_hours.append(7)  # Weekday morning peak

    valid = valid.copy()
    valid["pred_evening"] = predictions_evening
    valid["pred_hour"] = pred_hours

    # Evening peak accuracy
    correct_evening = (valid["pred_evening"] == valid["peak_in_evening"]).sum()
    acc_evening = correct_evening / n
    baseline_evening = 1 - valid["peak_in_evening"].mean()  # always predict "not evening"
    print(f"\n--- Evening Peak Prediction ---")
    print(f"  Accuracy: {correct_evening}/{n} = {acc_evening:.1%}")
    print(f"  Baseline (always morning): {baseline_evening:.1%}")
    print(f"  Skill: {acc_evening - baseline_evening:+.1%}")

    # Hour prediction MAE
    hour_error = (valid["da_price_max_hour"] - valid["pred_hour"]).abs()
    mae_hours = hour_error.mean()
    within_2h = (hour_error <= 2).mean()
    within_4h = (hour_error <= 4).mean()
    print(f"\n--- Peak Hour Prediction ---")
    print(f"  MAE: {mae_hours:.1f} hours")
    print(f"  Within 2 hours: {within_2h:.1%}")
    print(f"  Within 4 hours: {within_4h:.1%}")

    # Breakdown
    evening_days = valid[valid["peak_in_evening"] == 1]
    morning_days = valid[valid["peak_in_evening"] == 0]
    if len(evening_days) > 0:
        print(f"\n  Evening peak days (n={len(evening_days)}):")
        print(f"    Nuclear yesterday: {evening_days['nuclear_yesterday'].mean():.0f} MW "
              f"(vs {morning_days['nuclear_yesterday'].mean():.0f} MW for morning)")
        print(f"    Actual peak hour: {evening_days['da_price_max_hour'].mean():.1f} "
              f"(vs {morning_days['da_price_max_hour'].mean():.1f})")

    return valid


# ==============================================================
# RULE 3: Volatility Regime
# ==============================================================
def rule_volatility(df):
    """
    Predict: will it be a high-volatility day?

    Key finding: yesterday's volatility is the best predictor (persistence).
    Also: HU flow, extreme price history.
    """
    print("\n" + "=" * 60)
    print("RULE 3: Volatility Regime")
    print("=" * 60)

    valid = df[["da_price_std", "high_volatility", "da_volatility_yesterday",
                "da_range_yesterday", "da_volatility_7d",
                "net_import_yesterday", "extreme_price_flag"]].dropna()
    n = len(valid)
    print(f"[*] {n} days with volatility data")

    # Threshold for high volatility (75th percentile of std)
    vol_threshold = valid["da_price_std"].quantile(0.75)
    print(f"[*] High volatility threshold: std > {vol_threshold:.1f} EUR/MWh")

    # Rule: yesterday volatile AND 7d trend volatile → today volatile
    vol_yesterday_q75 = valid["da_volatility_yesterday"].quantile(0.75)
    vol_7d_q75 = valid["da_volatility_7d"].quantile(0.75)

    predictions = []
    for _, row in valid.iterrows():
        if row["da_volatility_yesterday"] > vol_yesterday_q75:
            predictions.append(1)  # High vol yesterday → expect high today
        elif row["da_volatility_7d"] > vol_7d_q75:
            predictions.append(1)  # Elevated 7d trend
        else:
            predictions.append(0)

    valid = valid.copy()
    valid["pred_high_vol"] = predictions

    correct = (valid["pred_high_vol"] == valid["high_volatility"]).sum()
    accuracy = correct / n
    baseline = 1 - valid["high_volatility"].mean()  # always predict "not high"
    print(f"\n--- High Volatility Prediction ---")
    print(f"  Accuracy: {correct}/{n} = {accuracy:.1%}")
    print(f"  Baseline (always low): {baseline:.1%}")
    print(f"  Skill: {accuracy - baseline:+.1%}")

    # Precision and recall
    tp = ((valid["pred_high_vol"] == 1) & (valid["high_volatility"] == 1)).sum()
    fp = ((valid["pred_high_vol"] == 1) & (valid["high_volatility"] == 0)).sum()
    fn = ((valid["pred_high_vol"] == 0) & (valid["high_volatility"] == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  Precision: {precision:.1%} (of predicted high-vol days, how many were right)")
    print(f"  Recall:    {recall:.1%} (of actual high-vol days, how many we caught)")

    # Extreme price prediction
    print(f"\n--- Extreme Price Flag ---")
    ext_correct = ((valid["pred_high_vol"] == 1) == (valid["extreme_price_flag"] == 1)).sum()
    ext_acc = ext_correct / n
    ext_baseline = 1 - valid["extreme_price_flag"].mean()
    print(f"  Accuracy (using vol rule): {ext_acc:.1%}")
    print(f"  Baseline: {ext_baseline:.1%}")

    return valid


# ==============================================================
# SUMMARY PLOT
# ==============================================================
def plot_decision_summary(df, spread_results, peak_results, vol_results):
    print("\n[*] Creating summary plot...")

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))

    # 1a: Spread prediction time series
    ax = axes[0, 0]
    v = spread_results.copy()
    actual = v["spread_da_idm_mean"]
    ax.bar(v.index, actual, color=["green" if s > 0 else "red" for s in actual],
           alpha=0.4, width=0.8, label="Actual spread")
    # Mark correct/wrong predictions
    correct = v["pred_direction"] == (actual > 0).astype(int)
    ax.scatter(v.index[correct], actual[correct] * 0 + actual.max() * 1.05,
              marker="v", color="green", s=15, alpha=0.7, label="Correct")
    ax.scatter(v.index[~correct], actual[~correct] * 0 + actual.max() * 1.05,
              marker="x", color="red", s=15, alpha=0.7, label="Wrong")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("DA-IDM Spread (EUR/MWh)")
    acc = correct.mean()
    ax.set_title(f"Spread Direction: {acc:.0%} accuracy", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)

    # 1b: Spread direction by condition
    ax = axes[0, 1]
    conditions = {
        "Weekend": spread_results["is_weekend"] == 1,
        "WD + High\nLoad Dev": (spread_results["is_weekend"] == 0) & (spread_results["load_fcst_vs_7d"] > spread_results["load_fcst_vs_7d"].median()),
        "WD + Low\nLoad Dev": (spread_results["is_weekend"] == 0) & (spread_results["load_fcst_vs_7d"] <= spread_results["load_fcst_vs_7d"].median()),
    }
    x_pos = range(len(conditions))
    means = []
    stds = []
    labels = []
    for label, mask in conditions.items():
        sub = spread_results.loc[mask, "spread_da_idm_mean"]
        means.append(sub.mean())
        stds.append(sub.std())
        labels.append(label)
    colors = ["green" if m > 0 else "red" for m in means]
    ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Spread (EUR/MWh)")
    ax.set_title("Spread by Decision Condition", fontsize=10)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 2, f"{m:+.0f}", ha="center", fontsize=9, fontweight="bold")

    # 2a: Peak hour prediction
    ax = axes[1, 0]
    v = peak_results.copy()
    ax.scatter(v["da_price_max_hour"], v["pred_hour"], alpha=0.5, s=30,
              c=v["nuclear_yesterday"], cmap="RdYlGn")
    ax.plot([0, 23], [0, 23], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Actual Peak Hour")
    ax.set_ylabel("Predicted Peak Hour")
    mae = (v["da_price_max_hour"] - v["pred_hour"]).abs().mean()
    within_2 = ((v["da_price_max_hour"] - v["pred_hour"]).abs() <= 2).mean()
    ax.set_title(f"Peak Hour: MAE={mae:.1f}h, within 2h={within_2:.0%}", fontsize=10)

    # 2b: Nuclear vs peak hour
    ax = axes[1, 1]
    ax.scatter(v["nuclear_yesterday"], v["da_price_max_hour"], alpha=0.5, s=30,
              c=v["peak_in_evening"], cmap="RdYlBu")
    r = np.corrcoef(v["nuclear_yesterday"].dropna(), v.loc[v["nuclear_yesterday"].notna(), "da_price_max_hour"])[0, 1]
    ax.set_xlabel("Nuclear Yesterday (MW)")
    ax.set_ylabel("Actual Peak Hour")
    ax.set_title(f"Nuclear Output vs Peak Timing (r={r:.2f})", fontsize=10)
    ax.axhline(12, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.text(v["nuclear_yesterday"].max() * 0.95, 12.5, "Morning/Evening divide",
           ha="right", fontsize=8, color="gray")

    # 3a: Volatility prediction
    ax = axes[2, 0]
    v = vol_results.copy()
    x = v.index
    ax.plot(x, v["da_price_std"], "k-", alpha=0.5, linewidth=1, label="Actual Std")
    vol_thresh = v["da_price_std"].quantile(0.75)
    ax.axhline(vol_thresh, color="orange", linewidth=1, linestyle="--", label=f"High-vol threshold ({vol_thresh:.0f})")
    # Mark predictions
    pred_high = v["pred_high_vol"] == 1
    actual_high = v["high_volatility"] == 1
    tp = pred_high & actual_high
    fp = pred_high & ~actual_high
    fn = ~pred_high & actual_high
    ax.scatter(x[tp], v.loc[tp, "da_price_std"], marker="^", color="green", s=30, label="True positive", zorder=5)
    ax.scatter(x[fp], v.loc[fp, "da_price_std"], marker="^", color="orange", s=30, label="False positive", zorder=5)
    ax.scatter(x[fn], v.loc[fn, "da_price_std"], marker="v", color="red", s=30, label="Missed", zorder=5)
    ax.set_ylabel("DA Price Std (EUR/MWh)")
    acc = (v["pred_high_vol"] == v["high_volatility"]).mean()
    ax.set_title(f"Volatility Prediction: {acc:.0%} accuracy", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)

    # 3b: Backtest summary table
    ax = axes[2, 1]
    ax.axis("off")

    # Build summary table
    table_data = [
        ["Target", "Accuracy", "Baseline", "Skill", "Key Driver"],
        ["DA-IDM Spread\nDirection",
         f"{(spread_results['pred_direction'] == (spread_results['spread_da_idm_mean'] > 0).astype(int)).mean():.0%}",
         f"{(spread_results['spread_da_idm_mean'] > 0).astype(int).mode()[0]}={max((spread_results['spread_da_idm_mean'] > 0).mean(), 1-(spread_results['spread_da_idm_mean'] > 0).mean()):.0%}",
         f"{(spread_results['pred_direction'] == (spread_results['spread_da_idm_mean'] > 0).astype(int)).mean() - max((spread_results['spread_da_idm_mean'] > 0).mean(), 1-(spread_results['spread_da_idm_mean'] > 0).mean()):+.0%}",
         "Weekend + Load Dev"],
        ["Evening Peak",
         f"{(peak_results['pred_evening'] == peak_results['peak_in_evening']).mean():.0%}",
         f"{1 - peak_results['peak_in_evening'].mean():.0%}",
         f"{(peak_results['pred_evening'] == peak_results['peak_in_evening']).mean() - (1 - peak_results['peak_in_evening'].mean()):+.0%}",
         "Nuclear output"],
        ["Peak Hour\n(MAE)",
         f"{(peak_results['da_price_max_hour'] - peak_results['pred_hour']).abs().mean():.1f}h",
         f"{peak_results['da_price_max_hour'].std():.1f}h std",
         f"within 2h: {((peak_results['da_price_max_hour'] - peak_results['pred_hour']).abs() <= 2).mean():.0%}",
         "Nuclear + Temp"],
        ["High Volatility",
         f"{(vol_results['pred_high_vol'] == vol_results['high_volatility']).mean():.0%}",
         f"{1 - vol_results['high_volatility'].mean():.0%}",
         f"{(vol_results['pred_high_vol'] == vol_results['high_volatility']).mean() - (1 - vol_results['high_volatility'].mean()):+.0%}",
         "Yesterday vol"],
    ]

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    # Color header
    for j in range(len(table_data[0])):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Decision Matrix Backtest Summary", fontsize=11, fontweight="bold", pad=20)

    fig.suptitle("DA Market Decision Matrix - Rule Backtest\n(Sep 2025 - Jan 2026, 139 days)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_decision_rules.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[+]   Saved 05_decision_rules.png")


# ==============================================================
# DECISION MATRIX PRINTOUT
# ==============================================================
def print_decision_matrix(df, spread_results):
    print("\n" + "=" * 60)
    print("DECISION MATRIX SUMMARY")
    print("=" * 60)

    load_dev_median = spread_results["load_fcst_vs_7d"].median()
    nuc_median = df["nuclear_yesterday"].dropna().median()
    vol_q75 = df["da_volatility_yesterday"].quantile(0.75)

    print("""
+---------------------------------------------------------------+
|  CONDITION               |  SPREAD   |  PEAK     |  VOLATILITY|
+---------------------------------------------------------------+
|  WEEKEND                 |  DA > IDM |  Morning  |  Low       |
|    (any load/temp)       |  +25 EUR  |  H7-H9    |            |
+---------------------------------------------------------------+
|  WEEKDAY + HIGH LOAD     |  DA < IDM |  Morning  |  Check vol |
|    (load dev > median)   |  -15 EUR  |  H6-H8    |  history   |
+---------------------------------------------------------------+
|  WEEKDAY + LOW LOAD      |  DA > IDM |  Morning  |  Low       |
|    (load dev < median)   |  +10 EUR  |  H7-H9    |            |
+---------------------------------------------------------------+
|  LOW NUCLEAR             |  --       |  Evening  |  High      |
|    (< median - 100 MW)   |           |  H17-H19  |            |
+---------------------------------------------------------------+
|  HIGH VOL YESTERDAY      |  --       |  --       |  High      |
|    (std > Q75)           |           |           |            |
+---------------------------------------------------------------+

Thresholds (from data):
  Load deviation median: {load_dev:.0f} MW
  Nuclear median: {nuc:.0f} MW
  Volatility Q75: {vol:.1f} EUR/MWh
""".format(load_dev=load_dev_median, nuc=nuc_median, vol=vol_q75))


def main():
    print("=" * 70)
    print("DA Decision Matrix - Step 4: Decision Rules & Backtest")
    print("=" * 70)

    df = load_data()

    spread_results = rule_spread_direction(df)
    peak_results = rule_peak_timing(df)
    vol_results = rule_volatility(df)

    plot_decision_summary(df, spread_results, peak_results, vol_results)
    print_decision_matrix(df, spread_results)


if __name__ == "__main__":
    main()
