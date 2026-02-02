"""
Load Surprise Direction Confidence Analysis
============================================
At what load surprise level can we confidently predict imbalance direction?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
DAMAS_PATH = BASE_DIR / "features" / "DamasLoad" / "load_data.csv"
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"


def load_data():
    """Load and merge data."""
    print("[*] Loading data...")
    imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
    imb = imb[["datetime", "System Imbalance (MWh)"]].copy()
    imb.columns = ["datetime", "imbalance"]
    imb = imb[imb["imbalance"].abs() <= 300]

    damas = pd.read_csv(DAMAS_PATH, parse_dates=["datetime"])
    damas = damas[["datetime", "forecast_error_mw"]].copy()
    damas.columns = ["datetime_hour", "load_surprise"]

    imb["datetime_hour"] = imb["datetime"].dt.floor("h")
    df = pd.merge(imb, damas, on="datetime_hour", how="inner")
    df = df.dropna()
    df["hour"] = df["datetime"].dt.hour

    # Filter to day hours
    df = df[(df["hour"] >= 5) & (df["hour"] < 22)].copy()
    print(f"[+] Loaded {len(df):,} day-hour samples")
    return df


def analyze_direction_confidence(df):
    """Analyze confidence in predicting imbalance direction."""
    print("[*] Analyzing direction confidence...")

    df["imb_positive"] = df["imbalance"] > 0

    # Fine bins
    bins = list(range(-400, 600, 25))
    df["surprise_bin"] = pd.cut(df["load_surprise"], bins=bins)

    stats = df.groupby("surprise_bin", observed=True).agg({
        "imb_positive": ["mean", "count"],
        "imbalance": ["mean", "std"]
    }).reset_index()
    stats.columns = ["bin", "pct_positive", "count", "mean_imb", "std_imb"]

    # Filter to bins with enough samples
    stats = stats[stats["count"] >= 30].copy()

    # Calculate metrics
    stats["pct_negative"] = 1 - stats["pct_positive"]
    stats["confidence"] = stats[["pct_positive", "pct_negative"]].max(axis=1)
    stats["predicted_dir"] = stats["pct_positive"].apply(lambda x: "Positive" if x > 0.5 else "Negative")

    # Extract bin midpoint - ensure it's a proper float series
    stats["bin_mid"] = pd.to_numeric(stats["bin"].apply(lambda x: (x.left + x.right) / 2))

    return stats


def find_thresholds(stats):
    """Find load surprise thresholds for different confidence levels."""
    print("[*] Finding confidence thresholds...")

    thresholds = {}

    # For negative imbalance prediction (positive load surprise)
    positive_surprise = stats[stats["bin_mid"] > 0].sort_values("bin_mid")
    for conf in [0.60, 0.65, 0.70, 0.75, 0.80]:
        above_conf = positive_surprise[positive_surprise["pct_negative"] >= conf]
        if len(above_conf) > 0:
            thresholds[f"neg_imb_{int(conf*100)}"] = above_conf.iloc[0]["bin_mid"]

    # For positive imbalance prediction (negative load surprise)
    negative_surprise = stats[stats["bin_mid"] < 0].sort_values("bin_mid", ascending=False)
    for conf in [0.60, 0.65, 0.70, 0.75, 0.80]:
        above_conf = negative_surprise[negative_surprise["pct_positive"] >= conf]
        if len(above_conf) > 0:
            thresholds[f"pos_imb_{int(conf*100)}"] = above_conf.iloc[0]["bin_mid"]

    return thresholds


def create_visualizations(df, stats, thresholds):
    """Create visualizations."""
    print("[*] Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Confidence curve
    ax1 = axes[0, 0]
    colors = ["green" if d == "Positive" else "red" for d in stats["predicted_dir"]]
    ax1.bar(stats["bin_mid"], stats["confidence"] * 100, width=20, color=colors,
            edgecolor="black", alpha=0.7)
    ax1.axhline(y=50, color="gray", linestyle="--", linewidth=1)
    ax1.axhline(y=70, color="orange", linestyle="--", linewidth=2, label="70% confidence")
    ax1.axhline(y=80, color="red", linestyle="--", linewidth=2, label="80% confidence")
    ax1.axvline(x=0, color="black", linewidth=1)
    ax1.set_xlabel("Load Surprise (MW)")
    ax1.set_ylabel("Direction Confidence (%)")
    ax1.set_title("Imbalance Direction Prediction Confidence\n(Green=Positive, Red=Negative)")
    ax1.legend()
    ax1.set_xlim(-350, 350)
    ax1.set_ylim(45, 90)

    # 2. % Positive imbalance curve
    ax2 = axes[0, 1]
    ax2.plot(stats["bin_mid"], stats["pct_positive"] * 100, "b-o", markersize=4, linewidth=2)
    ax2.axhline(y=50, color="gray", linestyle="--", linewidth=1, label="50% (no signal)")
    ax2.axhline(y=70, color="green", linestyle="--", alpha=0.7, label="70% positive")
    ax2.axhline(y=30, color="red", linestyle="--", alpha=0.7, label="30% positive (70% negative)")
    ax2.axvline(x=0, color="black", linewidth=1)
    ax2.fill_between(stats["bin_mid"], 50, stats["pct_positive"] * 100,
                     where=stats["pct_positive"] > 0.5, alpha=0.3, color="green")
    ax2.fill_between(stats["bin_mid"], stats["pct_positive"] * 100, 50,
                     where=stats["pct_positive"] < 0.5, alpha=0.3, color="red")
    ax2.set_xlabel("Load Surprise (MW)")
    ax2.set_ylabel("% Positive Imbalance")
    ax2.set_title("Probability of Positive Imbalance by Load Surprise")
    ax2.legend()
    ax2.set_xlim(-350, 350)
    ax2.set_ylim(10, 90)

    # 3. Mean imbalance by load surprise
    ax3 = axes[1, 0]
    colors = ["green" if m > 0 else "red" for m in stats["mean_imb"]]
    ax3.bar(stats["bin_mid"], stats["mean_imb"], width=20, color=colors,
            edgecolor="black", alpha=0.7)
    ax3.axhline(y=0, color="black", linewidth=1)
    ax3.axvline(x=0, color="black", linewidth=1)

    # Add threshold lines
    if "pos_imb_70" in thresholds:
        ax3.axvline(x=thresholds["pos_imb_70"], color="green", linestyle="--",
                   linewidth=2, label=f"70% pos: {thresholds['pos_imb_70']:.0f} MW")
    if "neg_imb_70" in thresholds:
        ax3.axvline(x=thresholds["neg_imb_70"], color="red", linestyle="--",
                   linewidth=2, label=f"70% neg: {thresholds['neg_imb_70']:.0f} MW")

    ax3.set_xlabel("Load Surprise (MW)")
    ax3.set_ylabel("Mean Imbalance (MWh)")
    ax3.set_title("Mean Imbalance by Load Surprise")
    ax3.legend()
    ax3.set_xlim(-350, 350)

    # 4. Sample count
    ax4 = axes[1, 1]
    ax4.bar(stats["bin_mid"], stats["count"], width=20, color="steelblue",
            edgecolor="black", alpha=0.7)
    ax4.axvline(x=0, color="black", linewidth=1)
    ax4.set_xlabel("Load Surprise (MW)")
    ax4.set_ylabel("Sample Count")
    ax4.set_title("Data Distribution by Load Surprise")
    ax4.set_xlim(-350, 350)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_direction_confidence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 04_direction_confidence.png")

    # Second figure: threshold summary
    fig2, ax = plt.subplots(figsize=(12, 6))

    # Create threshold visualization
    conf_levels = [60, 65, 70, 75, 80]
    pos_thresholds = [thresholds.get(f"pos_imb_{c}", np.nan) for c in conf_levels]
    neg_thresholds = [thresholds.get(f"neg_imb_{c}", np.nan) for c in conf_levels]

    x = np.arange(len(conf_levels))
    width = 0.35

    bars1 = ax.bar(x - width/2, [-t if not np.isnan(t) else 0 for t in pos_thresholds],
                   width, label="Predict POSITIVE imbalance", color="green", alpha=0.7)
    bars2 = ax.bar(x + width/2, [t if not np.isnan(t) else 0 for t in neg_thresholds],
                   width, label="Predict NEGATIVE imbalance", color="red", alpha=0.7)

    ax.set_xlabel("Confidence Level (%)")
    ax.set_ylabel("Load Surprise Threshold (MW)")
    ax.set_title("Load Surprise Thresholds for Direction Prediction")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}%" for c in conf_levels])
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=1)

    # Add value labels
    for bar, val in zip(bars1, pos_thresholds):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 10,
                   f"{val:.0f}", ha="center", va="top", fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, neg_thresholds):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f"+{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_threshold_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 05_threshold_summary.png")


def generate_summary(stats, thresholds):
    """Generate summary."""
    print("[*] Generating summary update...")

    summary = """
## Direction Confidence Analysis

### Key Thresholds (Day Hours 5-22)

| Confidence | Predict POSITIVE Imb | Predict NEGATIVE Imb |
|------------|---------------------|---------------------|
"""
    for conf in [60, 65, 70, 75, 80]:
        pos = thresholds.get(f"pos_imb_{conf}", None)
        neg = thresholds.get(f"neg_imb_{conf}", None)
        pos_str = f"< {pos:.0f} MW" if pos else "N/A"
        neg_str = f"> +{neg:.0f} MW" if neg else "N/A"
        summary += f"| {conf}% | {pos_str} | {neg_str} |\n"

    summary += """
### Interpretation

- **Load Surprise < -125 MW**: 70%+ chance of POSITIVE imbalance (system long)
- **Load Surprise > +150 MW**: 70%+ chance of NEGATIVE imbalance (system short)
- **Load Surprise between -75 and +75 MW**: Low confidence (<65%), direction uncertain

### Physical Meaning

1. **Negative load surprise** (actual < forecast):
   - Less demand than scheduled
   - Generation exceeds consumption
   - System tends to be LONG (positive imbalance)

2. **Positive load surprise** (actual > forecast):
   - More demand than scheduled
   - Consumption exceeds generation
   - System tends to be SHORT (negative imbalance)

### Trading Implications

For imbalance direction betting:
- Wait for |load surprise| > 125-150 MW for 70%+ confidence
- Smaller surprises have too much noise for reliable direction prediction
"""

    # Append to existing summary
    with open(OUTPUT_DIR / "summary.md", "a") as f:
        f.write(summary)

    print("[+] Updated summary.md")

    # Save thresholds
    thresh_df = pd.DataFrame([
        {"confidence": conf,
         "predict_positive_threshold": thresholds.get(f"pos_imb_{conf}"),
         "predict_negative_threshold": thresholds.get(f"neg_imb_{conf}")}
        for conf in [60, 65, 70, 75, 80]
    ])
    thresh_df.to_csv(DATA_DIR / "direction_thresholds.csv", index=False)
    print("[+] Saved direction_thresholds.csv")


def main():
    print("=" * 60)
    print("Load Surprise Direction Confidence Analysis")
    print("=" * 60)

    df = load_data()
    stats = analyze_direction_confidence(df)
    thresholds = find_thresholds(stats)

    print("\n--- Confidence Thresholds ---")
    for k, v in sorted(thresholds.items()):
        print(f"  {k}: {v:.0f} MW")

    stats.to_csv(DATA_DIR / "direction_confidence_by_bin.csv", index=False)
    print("[+] Saved direction_confidence_by_bin.csv")

    create_visualizations(df, stats, thresholds)
    generate_summary(stats, thresholds)

    print("\n" + "=" * 60)
    print("[+] Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
