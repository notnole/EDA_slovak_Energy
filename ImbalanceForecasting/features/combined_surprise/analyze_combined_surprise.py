# -*- coding: utf-8 -*-
"""
Combined Load + Solar Surprise Effect on Imbalance (Summer)
===========================================================
Tests whether combining load forecast error and solar forecast error
gives a stronger imbalance signal than either alone.

Load surprise  = Actual Load  - DA Forecast Load   (demand-side)
Solar surprise = Actual Solar - DA Forecast Solar   (supply-side)

Net surprise   = Solar surprise - Load surprise
  (positive = system has more energy than expected -> LONG)

Hypothesis: load surprise captures demand errors, solar surprise captures
supply errors. Together they should explain more imbalance variance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from sklearn.linear_model import LinearRegression

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
SOLAR_PATH = BASE_DIR / "data" / "clean" / "solar" / "solar_hourly.csv"
LOAD_PATH = BASE_DIR / "features" / "DamasLoad" / "load_data.csv"
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"

SUMMER_MONTHS = [6, 7, 8]


def load_all_data():
    """Load and merge solar, load, and imbalance data."""
    # Solar
    print("[*] Loading solar data...")
    solar = pd.read_csv(SOLAR_PATH, parse_dates=["datetime"])
    solar = solar[["datetime", "solar_actual_mw", "solar_forecast_da_mw", "solar_surprise_mw"]].copy()
    solar.columns = ["datetime", "solar_actual", "solar_forecast", "solar_surprise"]
    print(f"    Solar: {len(solar)} records")

    # Load
    print("[*] Loading load forecast data...")
    load = pd.read_csv(LOAD_PATH, parse_dates=["datetime"])
    load = load[["datetime", "actual_load_mw", "forecast_load_mw", "forecast_error_mw"]].copy()
    load.columns = ["datetime", "load_actual", "load_forecast", "load_surprise"]
    print(f"    Load: {len(load)} records")

    # Imbalance (15-min -> hourly)
    print("[*] Loading imbalance data...")
    imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
    imb = imb[["datetime", "System Imbalance (MWh)"]].copy()
    imb.columns = ["datetime", "imbalance"]
    imb = imb[imb["imbalance"].abs() <= 300]
    imb["datetime_hour"] = imb["datetime"].dt.floor("h")
    imb_h = imb.groupby("datetime_hour").agg(
        imbalance=("imbalance", "mean"),
        n_qh=("imbalance", "count"),
    ).reset_index()
    imb_h.columns = ["datetime", "imbalance", "n_qh"]
    print(f"    Imbalance: {len(imb_h)} hourly records")

    # Merge all three
    print("[*] Merging...")
    df = solar.merge(load, on="datetime", how="inner")
    df = df.merge(imb_h, on="datetime", how="inner")
    df = df.dropna()

    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

    # Net surprise: positive = more energy than expected
    # Solar over-production adds energy, load under-forecast means less demand needed
    # Net = solar_surprise - load_surprise
    df["net_surprise"] = df["solar_surprise"] - df["load_surprise"]

    print(f"[+] Merged: {len(df)} records, {df['datetime'].min()} to {df['datetime'].max()}")
    return df


def filter_summer_daylight(df):
    """Filter to summer daylight hours."""
    summer = df[df["month"].isin(SUMMER_MONTHS)].copy()
    daylight = summer[
        (summer["solar_actual"] > 0) | (summer["solar_forecast"] > 0)
    ].copy()
    print(f"[*] Summer daylight: {len(daylight)} records")
    for year in sorted(daylight["year"].unique()):
        yr = daylight[daylight["year"] == year]
        print(f"    {year}: {len(yr)} records")
    return daylight


def compare_signals(df):
    """Compare individual and combined signals."""
    print("\n" + "=" * 60)
    print("SIGNAL COMPARISON: Individual vs Combined")
    print("=" * 60)

    results = []

    # Individual correlations
    for name, col in [("Load surprise", "load_surprise"),
                       ("Solar surprise", "solar_surprise"),
                       ("Net surprise", "net_surprise")]:
        r, p = stats.pearsonr(df[col], df["imbalance"])
        results.append({
            "signal": name,
            "correlation": r,
            "p_value": p,
            "r_squared": r ** 2,
            "std": df[col].std(),
        })
        print(f"  {name:20s}: r = {r:+.4f}, R2 = {r**2:.4f}, p = {p:.2e}")

    # Multiple regression: both as predictors
    X = df[["load_surprise", "solar_surprise"]].values
    y = df["imbalance"].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_combined = 1 - ss_res / ss_tot
    r_combined = np.sqrt(r2_combined) * np.sign(reg.coef_.sum())

    results.append({
        "signal": "Combined (regression)",
        "correlation": r_combined,
        "p_value": np.nan,
        "r_squared": r2_combined,
        "std": np.nan,
    })
    print(f"\n  Combined regression R2 = {r2_combined:.4f}")
    print(f"    Load coef:  {reg.coef_[0]:+.6f}")
    print(f"    Solar coef: {reg.coef_[1]:+.6f}")
    print(f"    Intercept:  {reg.intercept_:+.4f}")

    # Partial correlations
    # Partial r of solar | load
    from numpy.linalg import lstsq
    load_resid = df["solar_surprise"] - lstsq(
        df[["load_surprise"]].values, df["solar_surprise"].values, rcond=None
    )[0][0] * df["load_surprise"]
    imb_resid = df["imbalance"] - lstsq(
        df[["load_surprise"]].values, df["imbalance"].values, rcond=None
    )[0][0] * df["load_surprise"]
    partial_r_solar, _ = stats.pearsonr(load_resid, imb_resid)

    load_resid2 = df["load_surprise"] - lstsq(
        df[["solar_surprise"]].values, df["load_surprise"].values, rcond=None
    )[0][0] * df["solar_surprise"]
    imb_resid2 = df["imbalance"] - lstsq(
        df[["solar_surprise"]].values, df["imbalance"].values, rcond=None
    )[0][0] * df["solar_surprise"]
    partial_r_load, _ = stats.pearsonr(load_resid2, imb_resid2)

    print(f"\n  Partial correlations:")
    print(f"    Solar | controlling for Load:  r = {partial_r_solar:+.4f}")
    print(f"    Load  | controlling for Solar: r = {partial_r_load:+.4f}")

    results.append({"signal": "Partial: Solar|Load", "correlation": partial_r_solar,
                     "p_value": np.nan, "r_squared": partial_r_solar**2, "std": np.nan})
    results.append({"signal": "Partial: Load|Solar", "correlation": partial_r_load,
                     "p_value": np.nan, "r_squared": partial_r_load**2, "std": np.nan})

    # By year
    print("\n--- By Year ---")
    for year in sorted(df["year"].unique()):
        yr = df[df["year"] == year]
        r_load, _ = stats.pearsonr(yr["load_surprise"], yr["imbalance"])
        r_solar, _ = stats.pearsonr(yr["solar_surprise"], yr["imbalance"])
        r_net, _ = stats.pearsonr(yr["net_surprise"], yr["imbalance"])

        X_yr = yr[["load_surprise", "solar_surprise"]].values
        y_yr = yr["imbalance"].values
        reg_yr = LinearRegression().fit(X_yr, y_yr)
        y_pred_yr = reg_yr.predict(X_yr)
        r2_yr = 1 - np.sum((y_yr - y_pred_yr)**2) / np.sum((y_yr - y_yr.mean())**2)

        print(f"  {year}: Load r={r_load:+.3f}, Solar r={r_solar:+.3f}, "
              f"Net r={r_net:+.3f}, Combined R2={r2_yr:.4f}")

        results.append({"signal": f"Load {year}", "correlation": r_load,
                         "p_value": np.nan, "r_squared": r_load**2, "std": np.nan})
        results.append({"signal": f"Solar {year}", "correlation": r_solar,
                         "p_value": np.nan, "r_squared": r_solar**2, "std": np.nan})
        results.append({"signal": f"Combined {year}", "correlation": np.sqrt(r2_yr),
                         "p_value": np.nan, "r_squared": r2_yr, "std": np.nan})

    return pd.DataFrame(results), reg


def analyze_direction_combined(df):
    """Direction prediction using combined signals."""
    print("\n[*] Analyzing direction prediction with combined signals...")

    df = df.copy()

    # Create combined bins based on net surprise
    bins = [-np.inf, -150, -75, -25, 0, 25, 75, 150, np.inf]
    labels = ["<-150", "-150:-75", "-75:-25", "-25:0", "0:25", "25:75", "75:150", ">150"]
    df["net_bin"] = pd.cut(df["net_surprise"], bins=bins, labels=labels)

    results = []
    for bin_name in labels:
        bd = df[df["net_bin"] == bin_name]
        if len(bd) > 5:
            n_long = (bd["imbalance"] > 0).sum()
            results.append({
                "net_surprise_bin": bin_name,
                "n_samples": len(bd),
                "n_long": n_long,
                "n_short": len(bd) - n_long,
                "pct_long": 100 * n_long / len(bd),
                "mean_imbalance": bd["imbalance"].mean(),
            })

    return pd.DataFrame(results)


def create_visualizations(df, comparison_df, direction_df, reg):
    """Create comparison plots."""
    print("\n[*] Creating visualizations...")

    years = sorted(df["year"].unique())
    year_str = ", ".join(str(y) for y in years)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"Combined Load + Solar Surprise vs Imbalance - Summer ({year_str})",
                 fontsize=14, fontweight="bold", y=1.01)

    # 1. R-squared comparison bar chart
    ax1 = fig.add_subplot(3, 2, 1)
    main_signals = comparison_df[comparison_df["signal"].isin([
        "Load surprise", "Solar surprise", "Net surprise", "Combined (regression)"
    ])]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    ax1.bar(range(len(main_signals)), main_signals["r_squared"] * 100,
            color=colors, edgecolor="black")
    ax1.set_xticks(range(len(main_signals)))
    ax1.set_xticklabels(main_signals["signal"], rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("R-squared: Individual vs Combined")
    for i, (_, row) in enumerate(main_signals.iterrows()):
        ax1.text(i, row["r_squared"] * 100 + 0.2,
                 f"{row['r_squared']*100:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # 2. Scatter: load surprise vs imbalance
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.scatter(df["load_surprise"], df["imbalance"], alpha=0.1, s=4, c="#2196F3")
    z = np.polyfit(df["load_surprise"], df["imbalance"], 1)
    x_line = np.linspace(df["load_surprise"].min(), df["load_surprise"].max(), 100)
    r_load = comparison_df[comparison_df["signal"] == "Load surprise"]["correlation"].values[0]
    ax2.plot(x_line, np.poly1d(z)(x_line), "r-", lw=2, label=f"r = {r_load:+.3f}")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Load Surprise (MW)")
    ax2.set_ylabel("Imbalance (MWh)")
    ax2.set_title("Load Surprise vs Imbalance")
    ax2.legend()

    # 3. Scatter: solar surprise vs imbalance
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.scatter(df["solar_surprise"], df["imbalance"], alpha=0.1, s=4, c="#FF9800")
    z = np.polyfit(df["solar_surprise"], df["imbalance"], 1)
    x_line = np.linspace(df["solar_surprise"].min(), df["solar_surprise"].max(), 100)
    r_solar = comparison_df[comparison_df["signal"] == "Solar surprise"]["correlation"].values[0]
    ax3.plot(x_line, np.poly1d(z)(x_line), "r-", lw=2, label=f"r = {r_solar:+.3f}")
    ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax3.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Solar Surprise (MW)")
    ax3.set_ylabel("Imbalance (MWh)")
    ax3.set_title("Solar Surprise vs Imbalance")
    ax3.legend()

    # 4. Scatter: net surprise vs imbalance
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.scatter(df["net_surprise"], df["imbalance"], alpha=0.1, s=4, c="#4CAF50")
    z = np.polyfit(df["net_surprise"], df["imbalance"], 1)
    x_line = np.linspace(df["net_surprise"].min(), df["net_surprise"].max(), 100)
    r_net = comparison_df[comparison_df["signal"] == "Net surprise"]["correlation"].values[0]
    ax4.plot(x_line, np.poly1d(z)(x_line), "r-", lw=2, label=f"r = {r_net:+.3f}")
    ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax4.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Net Surprise: Solar - Load (MW)")
    ax4.set_ylabel("Imbalance (MWh)")
    ax4.set_title("Net Surprise vs Imbalance")
    ax4.legend()

    # 5. Direction prediction with net surprise
    ax5 = fig.add_subplot(3, 2, 5)
    if len(direction_df) > 0:
        x_pos = range(len(direction_df))
        colors_dir = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(direction_df)))
        ax5.bar(x_pos, direction_df["pct_long"], color=colors_dir, edgecolor="black")
        ax5.axhline(y=50, color="gray", linestyle="--", alpha=0.7, label="50% baseline")
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(direction_df["net_surprise_bin"], rotation=30, ha="right")
        ax5.set_xlabel("Net Surprise Bin (MW)")
        ax5.set_ylabel("% Hours System LONG")
        ax5.set_title("Imbalance Direction by Net Surprise")
        ax5.legend()
        ax5.set_ylim(0, 100)
        for i, (_, row) in enumerate(direction_df.iterrows()):
            ax5.text(i, row["pct_long"] + 2, f"n={row['n_samples']}",
                     ha="center", fontsize=7)

    # 6. Correlation between the two surprise signals
    ax6 = fig.add_subplot(3, 2, 6)
    r_cross, _ = stats.pearsonr(df["load_surprise"], df["solar_surprise"])
    ax6.scatter(df["load_surprise"], df["solar_surprise"], alpha=0.1, s=4, c="gray")
    ax6.set_xlabel("Load Surprise (MW)")
    ax6.set_ylabel("Solar Surprise (MW)")
    ax6.set_title(f"Load vs Solar Surprise (r = {r_cross:+.3f})")
    ax6.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax6.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_combined_surprise.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 01_combined_surprise.png")


def generate_summary(df, comparison_df, direction_df, reg):
    """Generate summary.md."""
    print("\n[*] Generating summary...")

    years = sorted(df["year"].unique())
    year_str = ", ".join(str(y) for y in years)

    main = comparison_df[comparison_df["signal"].isin([
        "Load surprise", "Solar surprise", "Net surprise", "Combined (regression)"
    ])]

    partial_solar = comparison_df[comparison_df["signal"] == "Partial: Solar|Load"]["correlation"].values[0]
    partial_load = comparison_df[comparison_df["signal"] == "Partial: Load|Solar"]["correlation"].values[0]

    r_cross, _ = stats.pearsonr(df["load_surprise"], df["solar_surprise"])

    summary = f"""# Combined Load + Solar Surprise vs Imbalance (Summer)

## Concept

**Net Surprise = Solar Surprise - Load Surprise**

- Solar surprise (actual - forecast solar): supply-side error
- Load surprise (actual - forecast load): demand-side error
- Net surprise: overall energy balance surprise (positive = system has more than expected)

## Data

- **Period**: Summer (Jun-Aug) {year_str}, daylight hours only
- **Observations**: {len(df):,}
- **Sources**: `data/clean/solar/solar_hourly.csv`, `features/DamasLoad/load_data.csv`, `data/master/master_imbalance_data.csv`

## Key Results

### Variance Explained (R-squared)

| Signal | r | R-squared | Improvement |
|--------|---|-----------|-------------|
"""
    baseline_r2 = main[main["signal"] == "Load surprise"]["r_squared"].values[0]
    for _, row in main.iterrows():
        improvement = ""
        if row["signal"] != "Load surprise":
            pct = (row["r_squared"] - baseline_r2) / baseline_r2 * 100
            improvement = f"{pct:+.0f}% vs load alone"
        summary += f"| {row['signal']} | {row['correlation']:+.4f} | {row['r_squared']:.4f} | {improvement} |\n"

    summary += f"""
### Partial Correlations

| Signal | Partial r | Interpretation |
|--------|-----------|----------------|
| Solar | controlling for Load | {partial_solar:+.4f} | Solar adds info beyond load |
| Load | controlling for Solar | {partial_load:+.4f} | Load adds info beyond solar |

### Cross-correlation

Load surprise vs Solar surprise: r = {r_cross:+.3f}
({"Weakly" if abs(r_cross) < 0.3 else "Moderately"} correlated - {"mostly independent signals" if abs(r_cross) < 0.3 else "some overlap"})

### Regression Coefficients

| Predictor | Coefficient | Interpretation |
|-----------|-------------|----------------|
| Load surprise | {reg.coef_[0]:+.6f} | +1 MW load surprise -> {reg.coef_[0]:+.4f} MWh imbalance |
| Solar surprise | {reg.coef_[1]:+.6f} | +1 MW solar surprise -> {reg.coef_[1]:+.4f} MWh imbalance |
| Intercept | {reg.intercept_:+.4f} | Baseline imbalance bias |

### Direction Prediction (Net Surprise)

| Net Surprise Bin | N | % System LONG | Mean Imbalance (MWh) |
|-----------------|---|---------------|----------------------|
"""
    for _, row in direction_df.iterrows():
        summary += f"| {row['net_surprise_bin']} | {row['n_samples']:,} | {row['pct_long']:.0f}% | {row['mean_imbalance']:+.2f} |\n"

    summary += f"""
## Conclusion

1. **Combined model explains {main[main['signal']=='Combined (regression)']['r_squared'].values[0]*100:.1f}% of summer imbalance variance**
   vs {baseline_r2*100:.1f}% for load surprise alone.

2. Solar surprise adds **independent information** (partial r = {partial_solar:+.3f})
   - The two signals are {"largely independent" if abs(r_cross) < 0.3 else "somewhat correlated"}
     (cross-correlation r = {r_cross:+.3f})

3. **Physical interpretation**: Load surprise direction is *opposite* to imbalance
   (higher demand -> system SHORT), while solar surprise is *same direction*
   (more sun -> system LONG). Net surprise captures both effects.

## Files

- `01_combined_surprise.png` - 6-panel comparison dashboard
- `data/signal_comparison.csv`
- `data/direction_net_surprise.csv`
"""

    with open(OUTPUT_DIR / "summary.md", "w", encoding="utf-8") as f:
        f.write(summary)
    print("[+] Saved summary.md")


def main():
    print("=" * 60)
    print("Combined Load + Solar Surprise Analysis (Summer)")
    print("=" * 60)

    df = load_all_data()
    df_summer = filter_summer_daylight(df)

    comparison_df, reg = compare_signals(df_summer)
    direction_df = analyze_direction_combined(df_summer)

    DATA_DIR.mkdir(exist_ok=True)
    comparison_df.to_csv(DATA_DIR / "signal_comparison.csv", index=False)
    direction_df.to_csv(DATA_DIR / "direction_net_surprise.csv", index=False)
    print("[+] Saved data/")

    create_visualizations(df_summer, comparison_df, direction_df, reg)
    generate_summary(df_summer, comparison_df, direction_df, reg)

    print("\n" + "=" * 60)
    print("[+] Combined analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
