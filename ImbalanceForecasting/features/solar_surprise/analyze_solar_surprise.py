# -*- coding: utf-8 -*-
"""
Solar Surprise Effect on Imbalance Direction (Summer Only)
==========================================================
Analyzes how solar forecast error (actual - DA forecast solar production)
correlates with system imbalance direction during summer months (Jun-Aug).

Solar Surprise = Actual Solar - DA Forecast Solar (MW)
- Positive: more sun than expected -> excess generation -> system LONG?
- Negative: less sun than expected -> deficit generation -> system SHORT?

Data sources:
- Solar: data/clean/solar/solar_hourly.csv (DAMAS actual + DA forecast, 2024-2026)
- Imbalance: data/master/master_imbalance_data.csv (15-min, aggregated to hourly)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
SOLAR_PATH = BASE_DIR / "data" / "clean" / "solar" / "solar_hourly.csv"
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"

SUMMER_MONTHS = [6, 7, 8]  # June, July, August


def load_solar():
    """Load clean solar data with DA forecast."""
    print("[*] Loading solar data from data/clean/solar/solar_hourly.csv...")
    df = pd.read_csv(SOLAR_PATH, parse_dates=["datetime"])
    df = df.dropna(subset=["solar_actual_mw", "solar_forecast_da_mw"])
    # Recompute surprise for clarity
    df["solar_surprise"] = df["solar_actual_mw"] - df["solar_forecast_da_mw"]
    print(f"[+] Solar: {len(df)} records, {df['datetime'].min()} to {df['datetime'].max()}")
    return df


def load_imbalance():
    """Load 15-min imbalance data and aggregate to hourly."""
    print("[*] Loading imbalance data (15-min)...")
    imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
    imb = imb[["datetime", "System Imbalance (MWh)"]].copy()
    imb.columns = ["datetime", "imbalance"]

    # Remove outliers
    imb = imb[imb["imbalance"].abs() <= 300]

    # Aggregate to hourly (mean of 4 QH values)
    imb["datetime_hour"] = imb["datetime"].dt.floor("h")
    imb_hourly = imb.groupby("datetime_hour").agg(
        imbalance_mean=("imbalance", "mean"),
        imbalance_sum=("imbalance", "sum"),
        n_qh=("imbalance", "count"),
        imbalance_positive_frac=("imbalance", lambda x: (x > 0).mean()),
    ).reset_index()

    imb_hourly["imb_direction"] = np.where(
        imb_hourly["imbalance_positive_frac"] > 0.5, "LONG", "SHORT"
    )

    print(f"[+] Hourly imbalance: {len(imb_hourly)} records")
    return imb_hourly


def merge_and_filter(solar_df, imb_df):
    """Merge solar and imbalance, filter to summer daylight hours."""
    print("\n[*] Merging solar surprise with imbalance...")

    df = solar_df.merge(
        imb_df, left_on="datetime", right_on="datetime_hour", how="inner"
    )
    print(f"[*] Merged records: {len(df)}")

    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

    # Filter to summer
    df_summer = df[df["month"].isin(SUMMER_MONTHS)].copy()
    print(f"[*] Summer (Jun-Aug) records: {len(df_summer)}")

    # Filter to daylight only (actual or forecast > 0)
    df_daylight = df_summer[
        (df_summer["solar_actual_mw"] > 0) | (df_summer["solar_forecast_da_mw"] > 0)
    ].copy()
    print(f"[*] Summer daylight records: {len(df_daylight)}")

    for year in sorted(df_daylight["year"].unique()):
        yr = df_daylight[df_daylight["year"] == year]
        months = sorted(yr["month"].unique())
        print(f"    {year}: {len(yr)} records, months: {months}")

    return df_daylight, df


def analyze_correlation(df):
    """Core correlation analysis: solar surprise vs imbalance."""
    print("\n[*] Analyzing solar surprise vs imbalance correlation...")

    results = []

    # Overall
    r, p = stats.pearsonr(df["solar_surprise"], df["imbalance_mean"])
    results.append({
        "group": "All summer daylight",
        "n_samples": len(df),
        "correlation": r,
        "p_value": p,
        "r_squared": r ** 2,
        "surprise_mean": df["solar_surprise"].mean(),
        "surprise_std": df["solar_surprise"].std(),
    })
    print(f"  Overall: r = {r:+.4f}, p = {p:.2e}, n = {len(df)}")

    # By year
    for year in sorted(df["year"].unique()):
        yr = df[df["year"] == year]
        if len(yr) > 20:
            r, p = stats.pearsonr(yr["solar_surprise"], yr["imbalance_mean"])
            results.append({
                "group": f"Summer {year}",
                "n_samples": len(yr),
                "correlation": r,
                "p_value": p,
                "r_squared": r ** 2,
                "surprise_mean": yr["solar_surprise"].mean(),
                "surprise_std": yr["solar_surprise"].std(),
            })
            print(f"  {year}:    r = {r:+.4f}, p = {p:.2e}, n = {len(yr)}")

    # By month
    for month in SUMMER_MONTHS:
        mo = df[df["month"] == month]
        if len(mo) > 20:
            r, p = stats.pearsonr(mo["solar_surprise"], mo["imbalance_mean"])
            month_name = {6: "June", 7: "July", 8: "August"}[month]
            results.append({
                "group": month_name,
                "n_samples": len(mo),
                "correlation": r,
                "p_value": p,
                "r_squared": r ** 2,
                "surprise_mean": mo["solar_surprise"].mean(),
                "surprise_std": mo["solar_surprise"].std(),
            })
            print(f"  {month_name:8}: r = {r:+.4f}, p = {p:.2e}, n = {len(mo)}")

    return pd.DataFrame(results)


def analyze_by_hour(df):
    """Correlation by hour of day (daylight hours)."""
    print("\n[*] Analyzing correlation by hour...")

    results = []
    for hour in range(5, 21):
        subset = df[df["hour"] == hour]
        if len(subset) > 20:
            try:
                r, p = stats.pearsonr(subset["solar_surprise"], subset["imbalance_mean"])
            except Exception:
                continue
            results.append({
                "hour": hour,
                "n_samples": len(subset),
                "correlation": r,
                "p_value": p,
                "surprise_std": subset["solar_surprise"].std(),
                "surprise_mean": subset["solar_surprise"].mean(),
                "mean_actual_solar": subset["solar_actual_mw"].mean(),
            })

    print(f"    {len(results)} hours with sufficient data")
    return pd.DataFrame(results)


def analyze_by_magnitude(df):
    """Correlation by solar surprise magnitude."""
    print("\n[*] Analyzing by solar surprise magnitude...")

    df = df.copy()
    df["surprise_abs"] = df["solar_surprise"].abs()

    bins = [0, 10, 30, 60, 100, 2000]
    labels = ["<10 MW", "10-30 MW", "30-60 MW", "60-100 MW", ">100 MW"]
    df["magnitude_bin"] = pd.cut(df["surprise_abs"], bins=bins, labels=labels)

    results = []
    for bin_name in labels:
        bin_data = df[df["magnitude_bin"] == bin_name]
        if len(bin_data) > 20:
            try:
                r, p = stats.pearsonr(bin_data["solar_surprise"], bin_data["imbalance_mean"])
            except Exception:
                continue
            results.append({
                "magnitude": bin_name,
                "n_samples": len(bin_data),
                "correlation": r,
                "p_value": p,
                "mean_imbalance": bin_data["imbalance_mean"].mean(),
                "std_imbalance": bin_data["imbalance_mean"].std(),
            })

    return pd.DataFrame(results)


def analyze_direction(df):
    """Direction analysis: does solar surprise predict imbalance sign?"""
    print("\n[*] Analyzing direction prediction...")

    df = df.copy()

    bins = [-np.inf, -100, -50, -20, 0, 20, 50, 100, np.inf]
    labels = ["<-100", "-100:-50", "-50:-20", "-20:0", "0:20", "20:50", "50:100", ">100"]
    df["surprise_bin"] = pd.cut(df["solar_surprise"], bins=bins, labels=labels)

    results = []
    for bin_name in labels:
        bin_data = df[df["surprise_bin"] == bin_name]
        if len(bin_data) > 5:
            n_long = (bin_data["imbalance_mean"] > 0).sum()
            n_short = (bin_data["imbalance_mean"] <= 0).sum()
            results.append({
                "surprise_bin": bin_name,
                "n_samples": len(bin_data),
                "n_long": n_long,
                "n_short": n_short,
                "pct_long": 100 * n_long / len(bin_data),
                "mean_imbalance": bin_data["imbalance_mean"].mean(),
                "mean_surprise": bin_data["solar_surprise"].mean(),
            })

    return pd.DataFrame(results)


def create_visualizations(df, corr_df, hourly_df, magnitude_df, direction_df):
    """Create analysis plots."""
    print("\n[*] Creating visualizations...")

    years = sorted(df["year"].unique())
    year_str = ", ".join(str(y) for y in years)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"Solar Surprise vs Imbalance Direction - Summer ({year_str})",
                 fontsize=14, fontweight="bold", y=1.01)

    overall_r = corr_df[corr_df["group"] == "All summer daylight"]["correlation"].values[0]

    # 1. Scatter: solar surprise vs imbalance
    ax1 = fig.add_subplot(3, 2, 1)
    for year in years:
        yr = df[df["year"] == year]
        ax1.scatter(yr["solar_surprise"], yr["imbalance_mean"],
                    alpha=0.15, s=6, label=str(year), edgecolors="none")
    z = np.polyfit(df["solar_surprise"], df["imbalance_mean"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df["solar_surprise"].min(), df["solar_surprise"].max(), 100)
    ax1.plot(x_range, p(x_range), "r-", linewidth=2, label=f"r = {overall_r:+.3f}")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Solar Surprise: Actual - DA Forecast (MW)")
    ax1.set_ylabel("Hourly Mean Imbalance (MWh)")
    ax1.set_title("Summer Daylight: Solar Surprise vs Imbalance")
    ax1.legend(loc="upper left", fontsize=8)

    # 2. Hourly correlation
    ax2 = fig.add_subplot(3, 2, 2)
    if len(hourly_df) > 0:
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(hourly_df)))
        ax2.bar(hourly_df["hour"], hourly_df["correlation"], color=colors, edgecolor="black")
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.axhline(y=overall_r, color="red", linestyle="--", label=f"Overall: {overall_r:+.3f}")
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Correlation (r)")
        ax2.set_title("Solar Surprise - Imbalance Correlation by Hour")
        ax2.set_xticks(range(5, 21))
        ax2.legend()

    # 3. Magnitude analysis
    ax3 = fig.add_subplot(3, 2, 3)
    if len(magnitude_df) > 0:
        x_pos = range(len(magnitude_df))
        colors_mag = plt.cm.Oranges(np.linspace(0.3, 0.9, len(magnitude_df)))
        ax3.bar(x_pos, magnitude_df["correlation"], color=colors_mag, edgecolor="black")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(magnitude_df["magnitude"], rotation=15)
        ax3.set_ylabel("Correlation (r)")
        ax3.set_title("Correlation by Solar Surprise Magnitude")
        ax3.axhline(y=0, color="black", linewidth=0.5)
        for i, (_, row) in enumerate(magnitude_df.iterrows()):
            ax3.text(i, row["correlation"] + 0.01 * np.sign(row["correlation"]),
                     f"n={row['n_samples']}", ha="center", va="bottom", fontsize=8)

    # 4. Direction prediction - % long by surprise bin
    ax4 = fig.add_subplot(3, 2, 4)
    if len(direction_df) > 0:
        x_pos = range(len(direction_df))
        colors_dir = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(direction_df)))
        ax4.bar(x_pos, direction_df["pct_long"], color=colors_dir, edgecolor="black")
        ax4.axhline(y=50, color="gray", linestyle="--", alpha=0.7, label="50% baseline")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(direction_df["surprise_bin"], rotation=30, ha="right")
        ax4.set_xlabel("Solar Surprise Bin (MW)")
        ax4.set_ylabel("% Hours System LONG")
        ax4.set_title("Imbalance Direction by Solar Surprise")
        ax4.legend()
        ax4.set_ylim(0, 100)

    # 5. Mean imbalance by surprise bin
    ax5 = fig.add_subplot(3, 2, 5)
    if len(direction_df) > 0:
        ax5.bar(x_pos, direction_df["mean_imbalance"], color=colors_dir, edgecolor="black")
        ax5.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(direction_df["surprise_bin"], rotation=30, ha="right")
        ax5.set_xlabel("Solar Surprise Bin (MW)")
        ax5.set_ylabel("Mean Imbalance (MWh)")
        ax5.set_title("Mean Imbalance by Solar Surprise Bin")

    # 6. Year comparison
    ax6 = fig.add_subplot(3, 2, 6)
    for year in years:
        yr = df[df["year"] == year]
        ax6.hist(yr["solar_surprise"], bins=40, alpha=0.5, label=str(year), edgecolor="black", linewidth=0.5)
    ax6.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    ax6.set_xlabel("Solar Surprise (MW)")
    ax6.set_ylabel("Frequency")
    ax6.set_title("Solar Surprise Distribution by Year")
    ax6.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_solar_surprise_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 01_solar_surprise_analysis.png")

    # --- Plot 2: Time series sample (1 week from each summer) ---
    fig2, axes = plt.subplots(len(years), 1, figsize=(14, 5 * len(years)), sharex=False)
    if len(years) == 1:
        axes = [axes]

    for ax, year in zip(axes, years):
        yr = df[df["year"] == year]
        # Pick a July week
        july = yr[yr["month"] == 7]
        if len(july) > 0:
            sample_start = july["datetime"].iloc[0]
        else:
            sample_start = yr["datetime"].iloc[0]
        sample_end = sample_start + pd.Timedelta(days=10)
        sample = yr[(yr["datetime"] >= sample_start) & (yr["datetime"] < sample_end)]

        ax.fill_between(sample["datetime"], 0, sample["solar_surprise"],
                        where=sample["solar_surprise"] > 0, alpha=0.5, color="green",
                        label="Over-production")
        ax.fill_between(sample["datetime"], 0, sample["solar_surprise"],
                        where=sample["solar_surprise"] <= 0, alpha=0.5, color="red",
                        label="Under-production")
        ax2_twin = ax.twinx()
        ax2_twin.plot(sample["datetime"], sample["imbalance_mean"],
                      color="purple", linewidth=1, alpha=0.7, label="Imbalance")
        ax2_twin.set_ylabel("Imbalance (MWh)", color="purple")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Solar Surprise (MW)")
        ax.set_title(f"Summer {year} - Solar Surprise vs Imbalance (sample)")
        ax.legend(loc="upper left", fontsize=8)
        ax2_twin.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_timeseries_sample.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 02_timeseries_sample.png")


def generate_summary(df, corr_df, hourly_df, magnitude_df, direction_df):
    """Generate summary.md."""
    print("\n[*] Generating summary...")

    years = sorted(df["year"].unique())
    year_str = ", ".join(str(y) for y in years)
    date_min = df["datetime"].min().strftime("%Y-%m-%d")
    date_max = df["datetime"].max().strftime("%Y-%m-%d")

    overall = corr_df[corr_df["group"] == "All summer daylight"].iloc[0]

    summary = f"""# Solar Surprise Effect on Imbalance Direction (Summer)

## Definition

**Solar Surprise = Actual Solar Production - DA Forecast Solar Production (MW)**

- Positive surprise: more sun than expected -> excess generation
- Negative surprise: less sun than expected -> generation shortfall

## Data

- **Solar source**: `data/clean/solar/solar_hourly.csv` (DAMAS actual + DA forecast)
- **Imbalance source**: `data/master/master_imbalance_data.csv` (aggregated to hourly)
- **Filter**: Summer months only (June, July, August), daylight hours
- **Period**: {date_min} to {date_max}
- **Years**: {year_str}
- **Summer daylight observations**: {len(df):,}

## Key Findings

### Overall Correlation

| Group | N | Correlation (r) | R-squared | Surprise Std (MW) |
|-------|---|-----------------|-----------|-------------------|
"""
    for _, row in corr_df.iterrows():
        summary += f"| {row['group']} | {row['n_samples']:,} | {row['correlation']:+.4f} | {row['r_squared']:.4f} | {row['surprise_std']:.1f} |\n"

    summary += """
### Hourly Correlation Profile

| Hour | N | Correlation | Mean Solar (MW) | Surprise Std (MW) |
|------|---|-------------|-----------------|-------------------|
"""
    for _, row in hourly_df.iterrows():
        summary += f"| {int(row['hour']):02d}:00 | {row['n_samples']:,} | {row['correlation']:+.4f} | {row['mean_actual_solar']:.0f} | {row['surprise_std']:.1f} |\n"

    if len(magnitude_df) > 0:
        summary += """
### By Solar Surprise Magnitude

| Magnitude | N | Correlation | Mean Imbalance (MWh) |
|-----------|---|-------------|----------------------|
"""
        for _, row in magnitude_df.iterrows():
            summary += f"| {row['magnitude']} | {row['n_samples']:,} | {row['correlation']:+.4f} | {row['mean_imbalance']:+.2f} |\n"

    summary += """
### Direction Prediction

| Solar Surprise Bin | N | % System LONG | Mean Imbalance (MWh) |
|-------------------|---|---------------|----------------------|
"""
    for _, row in direction_df.iterrows():
        summary += f"| {row['surprise_bin']} | {row['n_samples']:,} | {row['pct_long']:.0f}% | {row['mean_imbalance']:+.2f} |\n"

    summary += f"""
## Physical Interpretation

1. **Expected mechanism**: Solar over-production (positive surprise) should push
   the system LONG (positive imbalance) as more unscheduled generation enters.
   Conversely, solar shortfall should push SHORT.

2. **Correlation strength**: r = {overall['correlation']:+.3f} explains
   {overall['r_squared']*100:.1f}% of imbalance variance in summer daylight hours.

3. **Comparison with load surprise**: Load surprise has r = -0.30 with imbalance
   (year-round). Solar surprise is complementary - active only in summer daylight.

## Files

- `01_solar_surprise_analysis.png` - Main 6-panel dashboard
- `02_timeseries_sample.png` - Sample time series per year
- `data/correlation_summary.csv`
- `data/hourly_correlation.csv`
- `data/magnitude_correlation.csv`
- `data/direction_analysis.csv`
"""

    with open(OUTPUT_DIR / "summary.md", "w", encoding="utf-8") as f:
        f.write(summary)
    print("[+] Saved summary.md")


def main():
    print("=" * 60)
    print("Solar Surprise vs Imbalance Direction (Summer)")
    print("=" * 60)

    solar_df = load_solar()
    imb_df = load_imbalance()
    df, df_all = merge_and_filter(solar_df, imb_df)

    if len(df) == 0:
        print("[!] No summer daylight records found.")
        return

    corr_df = analyze_correlation(df)
    hourly_df = analyze_by_hour(df)
    magnitude_df = analyze_by_magnitude(df)
    direction_df = analyze_direction(df)

    DATA_DIR.mkdir(exist_ok=True)
    corr_df.to_csv(DATA_DIR / "correlation_summary.csv", index=False)
    hourly_df.to_csv(DATA_DIR / "hourly_correlation.csv", index=False)
    magnitude_df.to_csv(DATA_DIR / "magnitude_correlation.csv", index=False)
    direction_df.to_csv(DATA_DIR / "direction_analysis.csv", index=False)
    print("[+] Saved analysis data to data/")

    create_visualizations(df, corr_df, hourly_df, magnitude_df, direction_df)
    generate_summary(df, corr_df, hourly_df, magnitude_df, direction_df)

    print("\n" + "=" * 60)
    print("[+] Solar surprise analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
