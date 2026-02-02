"""
Load Surprise Effect on Imbalance Analysis
==========================================
Analyzes how DAMAS forecast error (actual - predicted load) correlates with
system imbalance, with focus on day hours (5:00-22:00).

Load Surprise = Actual Hourly Load - DAMAS Day-Ahead Forecast
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
DAMAS_PATH = BASE_DIR / "features" / "DamasLoad" / "load_data.csv"
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"


def load_data():
    """Load and merge imbalance and DAMAS load forecast data."""
    print("[*] Loading imbalance data (15-min)...")
    imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
    imb = imb[["datetime", "System Imbalance (MWh)"]].copy()
    imb.columns = ["datetime", "imbalance"]

    # Remove outliers
    imb = imb[imb["imbalance"].abs() <= 300]

    print("[*] Loading DAMAS forecast data (hourly)...")
    damas = pd.read_csv(DAMAS_PATH, parse_dates=["datetime"])
    damas = damas[["datetime", "actual_load_mw", "forecast_load_mw", "forecast_error_mw", "is_weekend"]].copy()
    damas.columns = ["datetime_hour", "actual_load", "forecast_load", "load_surprise", "is_weekend"]

    # Add hour column to imbalance for merging
    imb["datetime_hour"] = imb["datetime"].dt.floor("h")

    # Merge - each 15-min imbalance gets the hourly load surprise
    print("[*] Merging datasets...")
    df = pd.merge(imb, damas, on="datetime_hour", how="inner")

    # Add time features
    df["hour"] = df["datetime"].dt.hour
    df["day_type"] = df["is_weekend"].map({True: "Weekend", False: "Weekday"})
    df["period"] = df["hour"].apply(lambda h: "Day (5-22)" if 5 <= h < 22 else "Night (22-5)")

    # Drop NaN values
    n_before = len(df)
    df = df.dropna(subset=["load_surprise", "imbalance"])
    n_dropped = n_before - len(df)

    print(f"[+] Loaded {len(df):,} samples (dropped {n_dropped} with missing values)")
    print(f"    Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"    Load surprise range: {df['load_surprise'].min():.0f} to {df['load_surprise'].max():.0f} MW")
    return df


def analyze_day_vs_night(df):
    """Compare load surprise correlation day vs night."""
    print("\n[*] Analyzing day vs night correlation...")

    results = []
    for period in ["Day (5-22)", "Night (22-5)"]:
        subset = df[df["period"] == period]
        r, p = stats.pearsonr(subset["load_surprise"], subset["imbalance"])
        results.append({
            "period": period,
            "n_samples": len(subset),
            "correlation": r,
            "p_value": p,
            "r_squared": r**2,
            "load_surprise_mean": subset["load_surprise"].mean(),
            "load_surprise_std": subset["load_surprise"].std()
        })

    # Overall
    r, p = stats.pearsonr(df["load_surprise"], df["imbalance"])
    results.append({
        "period": "Overall",
        "n_samples": len(df),
        "correlation": r,
        "p_value": p,
        "r_squared": r**2,
        "load_surprise_mean": df["load_surprise"].mean(),
        "load_surprise_std": df["load_surprise"].std()
    })

    results_df = pd.DataFrame(results)
    print("\n--- Day vs Night Correlation ---")
    for _, row in results_df.iterrows():
        print(f"  {row['period']:15} r = {row['correlation']:+.4f}  R2 = {row['r_squared']:.4f}  n = {row['n_samples']:,}")
    return results_df


def analyze_by_hour(df):
    """Correlation by hour of day."""
    print("\n[*] Analyzing correlation by hour...")

    results = []
    for hour in range(24):
        subset = df[df["hour"] == hour]
        if len(subset) > 30:
            r, p = stats.pearsonr(subset["load_surprise"], subset["imbalance"])
            results.append({
                "hour": hour,
                "n_samples": len(subset),
                "correlation": r,
                "load_surprise_std": subset["load_surprise"].std(),
                "imbalance_std": subset["imbalance"].std(),
                "load_surprise_mean": subset["load_surprise"].mean()
            })

    return pd.DataFrame(results)


def analyze_by_magnitude(df, day_only=True):
    """Correlation by load surprise magnitude."""
    print("\n[*] Analyzing by load surprise magnitude...")

    if day_only:
        subset = df[df["period"] == "Day (5-22)"].copy()
    else:
        subset = df.copy()

    # Create magnitude bins based on absolute value
    subset["surprise_abs"] = subset["load_surprise"].abs()

    # Define bins manually for clearer interpretation
    bins = [0, 50, 100, 200, 500, 2000]
    labels = ["<50 MW", "50-100 MW", "100-200 MW", "200-500 MW", ">500 MW"]
    subset["magnitude_bin"] = pd.cut(subset["surprise_abs"], bins=bins, labels=labels)

    results = []
    for bin_name in labels:
        bin_data = subset[subset["magnitude_bin"] == bin_name]
        if len(bin_data) > 30:
            r, _ = stats.pearsonr(bin_data["load_surprise"], bin_data["imbalance"])
            results.append({
                "magnitude": bin_name,
                "n_samples": len(bin_data),
                "correlation": r,
                "mean_imbalance": bin_data["imbalance"].mean(),
                "std_imbalance": bin_data["imbalance"].std()
            })

    return pd.DataFrame(results)


def analyze_direction(df, day_only=True):
    """Analyze effect of load surprise direction (over vs under prediction)."""
    print("\n[*] Analyzing load surprise direction...")

    if day_only:
        subset = df[df["period"] == "Day (5-22)"].copy()
    else:
        subset = df.copy()

    # Positive surprise = actual > forecast (underestimated demand)
    # Negative surprise = actual < forecast (overestimated demand)
    subset["direction"] = subset["load_surprise"].apply(
        lambda x: "Under-forecast (+)" if x > 0 else "Over-forecast (-)"
    )

    results = []
    for direction in ["Under-forecast (+)", "Over-forecast (-)"]:
        dir_data = subset[subset["direction"] == direction]
        r, _ = stats.pearsonr(dir_data["load_surprise"], dir_data["imbalance"])
        results.append({
            "direction": direction,
            "n_samples": len(dir_data),
            "correlation": r,
            "mean_surprise": dir_data["load_surprise"].mean(),
            "mean_imbalance": dir_data["imbalance"].mean(),
            "std_imbalance": dir_data["imbalance"].std()
        })

    return pd.DataFrame(results)


def create_visualizations(df, hourly_df, magnitude_df, day_night_df, direction_df):
    """Create comprehensive visualizations."""
    print("\n[*] Creating visualizations...")

    fig = plt.figure(figsize=(16, 14))

    # 1. Scatter plot - Day hours
    ax1 = fig.add_subplot(3, 2, 1)
    day_data = df[df["period"] == "Day (5-22)"]
    ax1.scatter(day_data["load_surprise"], day_data["imbalance"],
                alpha=0.1, s=3, c="orange")

    # Fit line
    z = np.polyfit(day_data["load_surprise"], day_data["imbalance"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(-600, 600, 100)
    day_r = day_night_df[day_night_df["period"] == "Day (5-22)"]["correlation"].values[0]
    ax1.plot(x_line, p(x_line), "r-", linewidth=2, label=f"r = {day_r:.3f}")

    ax1.set_xlabel("Load Surprise: Actual - DAMAS Forecast (MW)")
    ax1.set_ylabel("Imbalance (MWh)")
    ax1.set_title("Day Hours (5:00-22:00): Load Surprise vs Imbalance")
    ax1.legend(loc="upper right")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlim(-600, 600)
    ax1.set_ylim(-60, 60)

    # 2. Scatter plot - Night hours
    ax2 = fig.add_subplot(3, 2, 2)
    night_data = df[df["period"] == "Night (22-5)"]
    ax2.scatter(night_data["load_surprise"], night_data["imbalance"],
                alpha=0.1, s=3, c="blue")

    z = np.polyfit(night_data["load_surprise"], night_data["imbalance"], 1)
    p = np.poly1d(z)
    night_r = day_night_df[day_night_df["period"] == "Night (22-5)"]["correlation"].values[0]
    ax2.plot(x_line, p(x_line), "b-", linewidth=2, label=f"r = {night_r:.3f}")

    ax2.set_xlabel("Load Surprise: Actual - DAMAS Forecast (MW)")
    ax2.set_ylabel("Imbalance (MWh)")
    ax2.set_title("Night Hours (22:00-5:00): Load Surprise vs Imbalance")
    ax2.legend(loc="upper right")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlim(-600, 600)
    ax2.set_ylim(-60, 60)

    # 3. Hourly correlation bar chart
    ax3 = fig.add_subplot(3, 2, 3)
    colors = ["orange" if 5 <= h < 22 else "blue" for h in hourly_df["hour"]]
    ax3.bar(hourly_df["hour"], hourly_df["correlation"], color=colors, edgecolor="black", alpha=0.7)
    ax3.axhline(y=0, color="black", linewidth=0.5)
    ax3.axhline(y=day_r, color="orange", linestyle="--", linewidth=2, label=f"Day avg: {day_r:.3f}")
    ax3.axhline(y=night_r, color="blue", linestyle="--", linewidth=2, label=f"Night avg: {night_r:.3f}")
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Correlation (r)")
    ax3.set_title("Load Surprise - Imbalance Correlation by Hour")
    ax3.set_xticks(range(24))
    ax3.legend()
    ax3.set_ylim(-0.35, 0.1)

    # 4. Correlation by magnitude
    ax4 = fig.add_subplot(3, 2, 4)
    x_pos = range(len(magnitude_df))
    colors_mag = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(magnitude_df)))
    bars = ax4.bar(x_pos, magnitude_df["correlation"], color=colors_mag, edgecolor="black")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(magnitude_df["magnitude"], rotation=15)
    ax4.set_xlabel("Load Surprise Magnitude")
    ax4.set_ylabel("Correlation (r)")
    ax4.set_title("Correlation by Load Surprise Magnitude (Day Hours)")
    ax4.axhline(y=0, color="black", linewidth=0.5)

    for i, (idx, row) in enumerate(magnitude_df.iterrows()):
        ax4.text(i, row["correlation"] - 0.03, f"n={row['n_samples']:,}",
                ha="center", va="top", fontsize=8)

    # 5. Direction analysis - box plot
    ax5 = fig.add_subplot(3, 2, 5)
    day_data = df[df["period"] == "Day (5-22)"].copy()

    # Create bins for load surprise
    bins = [-np.inf, -200, -100, -50, 0, 50, 100, 200, np.inf]
    labels = ["<-200", "-200:-100", "-100:-50", "-50:0", "0:50", "50:100", "100:200", ">200"]
    day_data["surprise_bin"] = pd.cut(day_data["load_surprise"], bins=bins, labels=labels)

    # Calculate mean imbalance for each bin
    bin_stats = day_data.groupby("surprise_bin", observed=True)["imbalance"].agg(["mean", "std", "count"]).reset_index()

    colors_box = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(bin_stats)))
    ax5.bar(range(len(bin_stats)), bin_stats["mean"], yerr=bin_stats["std"]/np.sqrt(bin_stats["count"]),
            color=colors_box, edgecolor="black", capsize=3, alpha=0.7)
    ax5.set_xticks(range(len(bin_stats)))
    ax5.set_xticklabels(bin_stats["surprise_bin"], rotation=30, ha="right")
    ax5.set_xlabel("Load Surprise Bin (MW)")
    ax5.set_ylabel("Mean Imbalance (MWh)")
    ax5.set_title("Mean Imbalance by Load Surprise Direction (Day Hours)")
    ax5.axhline(y=0, color="gray", linestyle="--", alpha=0.7)

    # 6. Hourly load surprise variability
    ax6 = fig.add_subplot(3, 2, 6)
    colors = ["orange" if 5 <= h < 22 else "blue" for h in hourly_df["hour"]]
    ax6.bar(hourly_df["hour"], hourly_df["load_surprise_std"], color=colors, edgecolor="black", alpha=0.7)
    ax6.set_xlabel("Hour of Day")
    ax6.set_ylabel("Load Surprise Std (MW)")
    ax6.set_title("DAMAS Forecast Error Variability by Hour")
    ax6.set_xticks(range(24))

    # Add mean line
    ax6_twin = ax6.twinx()
    ax6_twin.plot(hourly_df["hour"], hourly_df["load_surprise_mean"], "ko-", markersize=4, label="Mean bias")
    ax6_twin.set_ylabel("Mean Load Surprise (MW)")
    ax6_twin.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax6_twin.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_load_surprise_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 01_load_surprise_analysis.png")

    # Additional plot: 2D density
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (period, data, color) in zip(axes, [("Day (5-22)", day_data, "YlOrRd"),
                                                  ("Night (22-5)", night_data, "YlGnBu")]):
        hb = ax.hexbin(data["load_surprise"], data["imbalance"],
                       gridsize=40, cmap=color, mincnt=1)
        ax.set_xlabel("Load Surprise: Actual - DAMAS Forecast (MW)")
        ax.set_ylabel("Imbalance (MWh)")
        ax.set_title(f"{period} - Density Plot")
        ax.axhline(y=0, color="white", linestyle="--", alpha=0.7)
        ax.axvline(x=0, color="white", linestyle="--", alpha=0.7)
        plt.colorbar(hb, ax=ax, label="Count")
        ax.set_xlim(-500, 500)
        ax.set_ylim(-50, 50)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_density_day_night.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 02_density_day_night.png")

    # Time series sample
    fig3, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Sample 7 days
    sample_start = df[df["hour"] == 5].iloc[200]["datetime"]
    sample_end = sample_start + pd.Timedelta(days=7)
    sample = df[(df["datetime"] >= sample_start) & (df["datetime"] < sample_end)]

    axes[0].plot(sample["datetime"], sample["load_surprise"],
                 label="Load Surprise (MW)", color="blue", alpha=0.7)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Load Surprise (MW)")
    axes[0].set_title("DAMAS Forecast Error vs Imbalance - Sample Week")
    axes[0].legend(loc="upper right")
    axes[0].fill_between(sample["datetime"], 0, sample["load_surprise"],
                         where=sample["load_surprise"] > 0, alpha=0.3, color="red", label="Under-forecast")
    axes[0].fill_between(sample["datetime"], 0, sample["load_surprise"],
                         where=sample["load_surprise"] < 0, alpha=0.3, color="green", label="Over-forecast")

    axes[1].plot(sample["datetime"], sample["imbalance"],
                 label="Imbalance (MWh)", color="red", alpha=0.7)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Imbalance (MWh)")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_timeseries_sample.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 03_timeseries_sample.png")


def generate_summary(day_night_df, hourly_df, magnitude_df, direction_df):
    """Generate summary.md file."""
    print("\n[*] Generating summary...")

    day_r = day_night_df[day_night_df["period"] == "Day (5-22)"]["correlation"].values[0]
    night_r = day_night_df[day_night_df["period"] == "Night (22-5)"]["correlation"].values[0]
    overall_r = day_night_df[day_night_df["period"] == "Overall"]["correlation"].values[0]

    best_hour = hourly_df.loc[hourly_df["correlation"].idxmin()]
    worst_hour = hourly_df.loc[hourly_df["correlation"].idxmax()]

    summary = f"""# Load Surprise Effect on Imbalance

## Definition

**Load Surprise = Actual Hourly Load - DAMAS Day-Ahead Forecast**

- Positive surprise: Demand higher than predicted (under-forecast)
- Negative surprise: Demand lower than predicted (over-forecast)

## Key Findings

### Day vs Night Comparison

| Period | Correlation (r) | R-squared | N samples |
|--------|-----------------|-----------|-----------|
| **Day (5-22)** | **{day_r:.4f}** | **{day_r**2:.4f}** | {day_night_df[day_night_df['period']=='Day (5-22)']['n_samples'].values[0]:,} |
| Night (22-5) | {night_r:.4f} | {night_r**2:.4f} | {day_night_df[day_night_df['period']=='Night (22-5)']['n_samples'].values[0]:,} |
| Overall | {overall_r:.4f} | {overall_r**2:.4f} | {day_night_df[day_night_df['period']=='Overall']['n_samples'].values[0]:,} |

**Key Insight**: Day hours show {abs(day_r)/abs(night_r):.1f}x stronger correlation than night
({day_r:.3f} vs {night_r:.3f}).

### Hourly Analysis

| Metric | Hour | Value |
|--------|------|-------|
| Strongest correlation | {int(best_hour['hour']):02d}:00 | r = {best_hour['correlation']:.3f} |
| Weakest correlation | {int(worst_hour['hour']):02d}:00 | r = {worst_hour['correlation']:.3f} |
| Highest variability | {int(hourly_df.loc[hourly_df['load_surprise_std'].idxmax()]['hour']):02d}:00 | std = {hourly_df['load_surprise_std'].max():.0f} MW |

### By Load Surprise Magnitude (Day Hours)

| Magnitude | N Samples | Correlation | Imbalance Std |
|-----------|-----------|-------------|---------------|
"""

    for _, row in magnitude_df.iterrows():
        summary += f"| {row['magnitude']} | {row['n_samples']:,} | {row['correlation']:.3f} | {row['std_imbalance']:.1f} MWh |\n"

    summary += f"""
### Direction Effect (Day Hours)

| Direction | N Samples | Mean Imbalance |
|-----------|-----------|----------------|
"""

    for _, row in direction_df.iterrows():
        summary += f"| {row['direction']} | {row['n_samples']:,} | {row['mean_imbalance']:+.2f} MWh |\n"

    summary += f"""
## Physical Interpretation

1. **Negative correlation** (r = {day_r:.3f}):
   - Higher-than-expected load (positive surprise) --> More negative imbalance
   - When demand exceeds forecast, system tends to be SHORT (negative imbalance)

2. **Causality chain**:
   - DAMAS forecast sets day-ahead schedules
   - If actual load > forecast, generation is insufficient
   - TSO activates upward regulation --> negative imbalance

3. **Day vs Night**:
   - Day: More load variability, harder to forecast
   - Night: Stable baseload, smaller forecast errors

## Model Implications

1. **Load surprise explains ~{day_r**2*100:.1f}% of imbalance variance** (day hours)
   - Compare to regulation which explains ~45%
   - Useful as secondary feature, not primary predictor

2. **Strongest effect at large surprises**:
   - |surprise| > 200 MW: r = {magnitude_df[magnitude_df['magnitude']=='200-500 MW']['correlation'].values[0] if '200-500 MW' in magnitude_df['magnitude'].values else 'N/A'}
   - Consider threshold-based feature

3. **Hour interaction**:
   - Peak hours ({int(best_hour['hour'])}:00) show strongest effect
   - Evening ({int(worst_hour['hour'])}:00) shows weakest

## Files Generated

- `01_load_surprise_analysis.png` - Main analysis dashboard
- `02_density_day_night.png` - 2D density comparison
- `03_timeseries_sample.png` - Sample week time series
- `data/day_night_correlation.csv`
- `data/hourly_correlation.csv`
- `data/magnitude_correlation.csv`
- `data/direction_analysis.csv`
"""

    with open(OUTPUT_DIR / "summary.md", "w") as f:
        f.write(summary)

    print("[+] Saved summary.md")


def main():
    print("=" * 60)
    print("Load Surprise Effect on Imbalance Analysis")
    print("Using DAMAS Day-Ahead Forecast Error")
    print("=" * 60)

    # Load data
    df = load_data()

    # Analysis
    day_night_df = analyze_day_vs_night(df)
    hourly_df = analyze_by_hour(df)
    magnitude_df = analyze_by_magnitude(df, day_only=True)
    direction_df = analyze_direction(df, day_only=True)

    # Save data
    DATA_DIR.mkdir(exist_ok=True)
    day_night_df.to_csv(DATA_DIR / "day_night_correlation.csv", index=False)
    hourly_df.to_csv(DATA_DIR / "hourly_correlation.csv", index=False)
    magnitude_df.to_csv(DATA_DIR / "magnitude_correlation.csv", index=False)
    direction_df.to_csv(DATA_DIR / "direction_analysis.csv", index=False)
    print("[+] Saved analysis data to data/")

    # Visualizations
    create_visualizations(df, hourly_df, magnitude_df, day_night_df, direction_df)

    # Summary
    generate_summary(day_night_df, hourly_df, magnitude_df, direction_df)

    print("\n" + "=" * 60)
    print("[+] Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
