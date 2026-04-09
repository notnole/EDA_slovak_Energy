"""
Analyze relationship between Hub-to-Hub cross-border capacity and IDM prices.

Input: data/h2h_idm_hourly.csv (from extract_h2h_idm_data.py)
Output: 5 PNG plots + summary statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from scipy import stats

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_PATH = BASE_DIR / "data" / "h2h_idm_hourly.csv"

# --- Config ---
AREAS = ["ceps", "apg", "mavir", "pse", "de"]
AREA_LABELS = {
    "ceps": "CEPS (CZ)",
    "apg": "APG (AT)",
    "mavir": "MAVIR (HU)",
    "pse": "PSE (PL)",
    "de": "DE (avg)",
}

# Price outlier clipping for visual clarity
PRICE_CLIP = (-100, 500)


def load_data():
    """Load and prepare the merged dataset."""
    df = pd.read_csv(DATA_PATH, parse_dates=["tradeday"])
    df["tradeday_date"] = df["tradeday"].dt.date

    # Use idm_price (order book mid) as primary, fall back to last_price
    if "idm_price" not in df.columns:
        df["idm_price"] = df["last_price"]

    # Fill any remaining NaN from last_price
    if "last_price" in df.columns:
        df["idm_price"] = df["idm_price"].fillna(df["last_price"])

    valid_mask = df["idm_price"].notna()
    print(f"[*] Loaded {len(df)} rows, {valid_mask.sum()} with valid prices")
    print(f"    Date range: {df['tradeday'].min().date()} to {df['tradeday'].max().date()}")
    print(f"    Price range: {df.loc[valid_mask, 'idm_price'].min():.1f} to "
          f"{df.loc[valid_mask, 'idm_price'].max():.1f} EUR/MWh")

    return df


def plot_correlation_matrix(df):
    """Plot 1: Correlation heatmap between H2H features and IDM price."""
    print("[*] Plotting correlation matrix...")

    # Select relevant columns
    cap_cols = []
    for area in AREAS:
        cap_cols.extend([f"{area}_in", f"{area}_out"])
    agg_cols = ["total_import_cap", "total_export_cap", "net_capacity", "congested_in_count"]
    price_cols = ["idm_price"]
    if "idm_spread" in df.columns:
        price_cols.append("idm_spread")

    all_cols = cap_cols + agg_cols + price_cols
    available_cols = [c for c in all_cols if c in df.columns]

    corr = df[available_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Labels
    labels = [c.replace("_", " ").title() for c in available_cols]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate values
    for i in range(len(available_cols)):
        for j in range(len(available_cols)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=7)

    plt.colorbar(im, ax=ax, label="Pearson Correlation", shrink=0.8)
    ax.set_title("H2H Cross-Border Capacity vs IDM Price - Correlation Matrix", fontsize=12)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "01_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 01_correlation_matrix.png")


def plot_price_vs_capacity(df):
    """Plot 2: Scatter plots of import capacity vs IDM price per area."""
    print("[*] Plotting price vs capacity scatters...")

    valid = df[df["idm_price"].notna()].copy()
    valid["price_clipped"] = valid["idm_price"].clip(*PRICE_CLIP)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, area in enumerate(AREAS):
        ax = axes[idx]
        col = f"{area}_in"
        if col not in valid.columns:
            continue

        x = valid[col].values
        y = valid["price_clipped"].values
        mask = np.isfinite(x) & np.isfinite(y)

        # Scatter with hour-of-day coloring
        scatter = ax.scatter(x[mask], y[mask], c=valid.loc[mask, "hour"].values,
                             cmap="twilight", alpha=0.3, s=8, edgecolors="none")
        ax.set_xlabel(f"{AREA_LABELS[area]} Import Cap [MW]")
        ax.set_ylabel("IDM Price [EUR/MWh]")
        ax.set_title(AREA_LABELS[area])

        # Regression line
        r, p = stats.pearsonr(x[mask], y[mask])
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=1.5, alpha=0.8)
        ax.text(0.05, 0.95, f"r = {r:.3f}\np = {p:.1e}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Last subplot: total import capacity
    ax = axes[5]
    x = valid["total_import_cap"].values
    y = valid["price_clipped"].values
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], c=valid.loc[mask, "hour"].values,
               cmap="twilight", alpha=0.3, s=8, edgecolors="none")
    r, p = stats.pearsonr(x[mask], y[mask])
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=1.5, alpha=0.8)
    ax.text(0.05, 0.95, f"r = {r:.3f}\np = {p:.1e}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Total Import Capacity [MW]")
    ax.set_ylabel("IDM Price [EUR/MWh]")
    ax.set_title("Total (All Borders)")

    plt.suptitle("IDM Price vs Cross-Border Import Capacity", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "02_price_vs_capacity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 02_price_vs_capacity.png")


def plot_congestion_impact(df):
    """Plot 3: Box plots of IDM price when border is congested vs not."""
    print("[*] Plotting congestion impact...")

    valid = df[df["idm_price"].notna()].copy()
    valid["price_clipped"] = valid["idm_price"].clip(*PRICE_CLIP)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, area in enumerate(AREAS):
        ax = axes[idx]
        cong_col = f"{area}_congested_in"
        if cong_col not in valid.columns:
            continue

        cong = valid[valid[cong_col] == 1]["price_clipped"]
        not_cong = valid[valid[cong_col] == 0]["price_clipped"]

        data = [not_cong, cong]
        bp = ax.boxplot(data, tick_labels=["Open", "Congested"],
                        patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][1].set_facecolor("#F44336")

        premium = cong.median() - not_cong.median()
        ax.set_title(f"{AREA_LABELS[area]}\nPremium: {premium:+.1f} EUR/MWh")
        ax.set_ylabel("IDM Price [EUR/MWh]")

        # Annotate counts
        ax.text(0.5, -0.12, f"n={len(not_cong)}", transform=ax.transAxes,
                ha="left", fontsize=8, color="gray")
        ax.text(0.85, -0.12, f"n={len(cong)}", transform=ax.transAxes,
                ha="left", fontsize=8, color="gray")

    # Last subplot: congested count vs price
    ax = axes[5]
    if "congested_in_count" in valid.columns:
        groups = valid.groupby("congested_in_count")["price_clipped"]
        data = []
        labels = []
        for count, grp in groups:
            data.append(grp.values)
            labels.append(f"{int(count)}\n(n={len(grp)})")
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=2))
        cmap = plt.cm.RdYlGn_r
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(cmap(i / max(len(bp["boxes"]) - 1, 1)))
        ax.set_title("# Congested Borders")
        ax.set_ylabel("IDM Price [EUR/MWh]")
        ax.set_xlabel("Number of congested import borders")

    plt.suptitle("IDM Price by Border Congestion State", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "03_congestion_impact.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 03_congestion_impact.png")


def plot_timeseries_sample(df):
    """Plot 4: One-week time series of price + capacity overlay."""
    print("[*] Plotting time series sample...")

    valid = df[df["idm_price"].notna()].copy()
    valid["datetime"] = valid["tradeday"] + pd.to_timedelta(valid["hour"], unit="h")

    # Pick a representative week (mid-January 2026)
    week_start = pd.Timestamp("2026-01-12")
    week_end = pd.Timestamp("2026-01-19")
    week = valid[(valid["datetime"] >= week_start) & (valid["datetime"] < week_end)].copy()

    if len(week) < 24:
        # Fallback: use first full week
        for start in pd.date_range(valid["tradeday"].min(), valid["tradeday"].max() - pd.Timedelta(days=7)):
            end = start + pd.Timedelta(days=7)
            week = valid[(valid["datetime"] >= start) & (valid["datetime"] < end)]
            if len(week) >= 140:
                week_start, week_end = start, end
                break

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Top: IDM price
    ax1.plot(week["datetime"], week["idm_price"].clip(*PRICE_CLIP),
             color="#1976D2", linewidth=1, alpha=0.9)
    ax1.fill_between(week["datetime"], 0, week["idm_price"].clip(*PRICE_CLIP),
                     alpha=0.15, color="#1976D2")
    ax1.set_ylabel("IDM Price [EUR/MWh]")
    ax1.set_title(f"IDM Price & Cross-Border Import Capacity ({week_start.date()} to {week_end.date()})")
    ax1.grid(True, alpha=0.3)

    # Bottom: stacked area of import capacity per area
    colors = {"ceps": "#E53935", "apg": "#43A047", "mavir": "#FB8C00",
              "pse": "#1E88E5", "de": "#8E24AA"}
    bottom = np.zeros(len(week))
    for area in AREAS:
        col = f"{area}_in"
        if col in week.columns:
            vals = week[col].fillna(0).values
            ax2.fill_between(week["datetime"], bottom, bottom + vals,
                             alpha=0.7, label=AREA_LABELS[area], color=colors[area])
            bottom += vals

    ax2.set_ylabel("Import Capacity [MW]")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(BASE_DIR / "04_timeseries_sample.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 04_timeseries_sample.png")


def plot_total_capacity_effect(df):
    """Plot 5: Total import capacity effect and binned analysis."""
    print("[*] Plotting total capacity effect...")

    valid = df[df["idm_price"].notna()].copy()
    valid["price_clipped"] = valid["idm_price"].clip(*PRICE_CLIP)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: scatter of total import cap vs price
    ax = axes[0]
    x = valid["total_import_cap"]
    y = valid["price_clipped"]
    ax.scatter(x, y, alpha=0.2, s=10, c="#1976D2", edgecolors="none")

    # Binned average
    bins = pd.cut(x, bins=10)
    binned = valid.groupby(bins, observed=True)["price_clipped"].agg(["mean", "median", "count"])
    bin_centers = [(b.left + b.right) / 2 for b in binned.index]
    ax.plot(bin_centers, binned["mean"], "r-o", linewidth=2, markersize=6, label="Binned Mean")
    ax.plot(bin_centers, binned["median"], "g--s", linewidth=2, markersize=5, label="Binned Median")

    r, p = stats.pearsonr(x, y)
    ax.set_xlabel("Total Import Capacity [MW]")
    ax.set_ylabel("IDM Price [EUR/MWh]")
    ax.set_title(f"Total Import Capacity vs Price\nr={r:.3f}, p={p:.1e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle: net capacity
    ax = axes[1]
    x = valid["net_capacity"]
    y = valid["price_clipped"]
    ax.scatter(x, y, alpha=0.2, s=10, c="#E53935", edgecolors="none")

    bins_net = pd.cut(x, bins=10)
    binned_net = valid.groupby(bins_net, observed=True)["price_clipped"].agg(["mean", "median"])
    bin_centers_net = [(b.left + b.right) / 2 for b in binned_net.index]
    ax.plot(bin_centers_net, binned_net["mean"], "r-o", linewidth=2, markersize=6, label="Binned Mean")
    ax.plot(bin_centers_net, binned_net["median"], "g--s", linewidth=2, markersize=5, label="Binned Median")

    r_net, p_net = stats.pearsonr(x, y)
    ax.set_xlabel("Net Capacity (Import - Export) [MW]")
    ax.set_ylabel("IDM Price [EUR/MWh]")
    ax.set_title(f"Net Capacity vs Price\nr={r_net:.3f}, p={p_net:.1e}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: hour-of-day effect
    ax = axes[2]
    hourly_stats = valid.groupby("hour").agg(
        price_mean=("price_clipped", "mean"),
        price_median=("price_clipped", "median"),
        import_mean=("total_import_cap", "mean"),
    )
    ax2r = ax.twinx()
    ax.bar(hourly_stats.index, hourly_stats["price_mean"], alpha=0.5,
           color="#1976D2", label="Price (mean)")
    ax.plot(hourly_stats.index, hourly_stats["price_median"], "b-o",
            markersize=4, label="Price (median)")
    ax2r.plot(hourly_stats.index, hourly_stats["import_mean"], "r--s",
              markersize=4, label="Import Cap (mean)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("IDM Price [EUR/MWh]", color="#1976D2")
    ax2r.set_ylabel("Import Capacity [MW]", color="red")
    ax.set_title("Hourly Profiles: Price & Capacity")
    ax.set_xticks(range(0, 24, 2))

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Aggregate Capacity Effects on IDM Price", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "05_total_capacity_effect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 05_total_capacity_effect.png")


def compute_statistics(df):
    """Compute and print summary statistics."""
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    valid = df[df["idm_price"].notna()].copy()
    valid["price_clipped"] = valid["idm_price"].clip(*PRICE_CLIP)

    # --- Correlations ---
    print("\n--- Pearson Correlations with IDM Price ---")
    results = []
    for area in AREAS:
        col = f"{area}_in"
        if col in valid.columns:
            r, p = stats.pearsonr(valid[col].dropna(), valid.loc[valid[col].notna(), "idm_price"])
            rs, ps = stats.spearmanr(valid[col].dropna(), valid.loc[valid[col].notna(), "idm_price"])
            print(f"  {AREA_LABELS[area]:15s} import cap: Pearson r={r:+.3f} (p={p:.1e}), "
                  f"Spearman r={rs:+.3f} (p={ps:.1e})")
            results.append({"area": area, "direction": "import", "pearson_r": r,
                           "pearson_p": p, "spearman_r": rs, "spearman_p": ps})

    for col, label in [("total_import_cap", "Total Import Cap"),
                       ("total_export_cap", "Total Export Cap"),
                       ("net_capacity", "Net Capacity"),
                       ("congested_in_count", "# Congested Borders")]:
        if col in valid.columns:
            r, p = stats.pearsonr(valid[col].dropna(), valid.loc[valid[col].notna(), "idm_price"])
            rs, ps = stats.spearmanr(valid[col].dropna(), valid.loc[valid[col].notna(), "idm_price"])
            print(f"  {label:15s}:            Pearson r={r:+.3f} (p={p:.1e}), "
                  f"Spearman r={rs:+.3f} (p={ps:.1e})")

    # --- Congestion premium ---
    print("\n--- Congestion Price Premium (median) ---")
    congestion_stats = []
    for area in AREAS:
        cong_col = f"{area}_congested_in"
        if cong_col not in valid.columns:
            continue
        cong = valid[valid[cong_col] == 1]["idm_price"]
        not_cong = valid[valid[cong_col] == 0]["idm_price"]
        premium = cong.median() - not_cong.median()
        pct_congested = (valid[cong_col] == 1).mean() * 100
        print(f"  {AREA_LABELS[area]:15s}: Open median={not_cong.median():.1f}, "
              f"Congested median={cong.median():.1f}, "
              f"Premium={premium:+.1f} EUR/MWh, "
              f"Congested {pct_congested:.0f}% of time")
        congestion_stats.append({
            "area": AREA_LABELS[area],
            "open_median": not_cong.median(),
            "congested_median": cong.median(),
            "premium": premium,
            "pct_congested": pct_congested,
        })

    # --- Congested border count ---
    print("\n--- Price by Number of Congested Import Borders ---")
    for count, grp in valid.groupby("congested_in_count"):
        print(f"  {int(count)} borders congested: "
              f"median={grp['idm_price'].median():.1f}, "
              f"mean={grp['idm_price'].mean():.1f}, "
              f"n={len(grp)}")

    # --- OLS regression ---
    print("\n--- OLS Regression: price ~ h2h_features + hour dummies ---")
    try:
        import statsmodels.api as sm

        feature_cols = []
        for area in AREAS:
            feature_cols.extend([f"{area}_in", f"{area}_out"])
        feature_cols.extend(["total_import_cap", "congested_in_count"])

        available_features = [c for c in feature_cols if c in valid.columns]

        # Add hour dummies
        hour_dummies = pd.get_dummies(valid["hour"], prefix="h", drop_first=True, dtype=float)
        X = pd.concat([valid[available_features].reset_index(drop=True),
                        hour_dummies.reset_index(drop=True)], axis=1)
        X = sm.add_constant(X)
        X = X.astype(float)
        y = valid["idm_price"].reset_index(drop=True).astype(float)

        # Drop rows with NaN
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        model = sm.OLS(y, X).fit()
        print(f"  R-squared: {model.rsquared:.4f}")
        print(f"  Adj R-squared: {model.rsquared_adj:.4f}")
        print(f"  F-statistic: {model.fvalue:.1f} (p={model.f_pvalue:.1e})")
        print(f"  Observations: {model.nobs:.0f}")

        # Top coefficients (excluding hour dummies)
        coef_df = pd.DataFrame({
            "coef": model.params,
            "pval": model.pvalues,
            "se": model.bse,
        })
        h2h_coefs = coef_df[~coef_df.index.str.startswith("h_") & (coef_df.index != "const")]
        h2h_coefs = h2h_coefs.sort_values("pval")
        print("\n  H2H Feature Coefficients (sorted by significance):")
        for name, row in h2h_coefs.iterrows():
            sig = "***" if row["pval"] < 0.001 else "**" if row["pval"] < 0.01 else "*" if row["pval"] < 0.05 else ""
            print(f"    {name:25s}: coef={row['coef']:+.4f}, p={row['pval']:.3e} {sig}")

    except ImportError:
        print("  [!] statsmodels not installed, skipping OLS regression")

    return results, congestion_stats


def write_summary(df, corr_results, congestion_stats):
    """Write summary.md with key findings."""
    valid = df[df["idm_price"].notna()]

    lines = [
        "# Hub-to-Hub Cross-Border Capacity vs IDM Price Analysis",
        "",
        "## Overview",
        "",
        f"Analysis of how cross-border interconnection capacity affects Slovak IDM prices.",
        f"- **Period**: {df['tradeday'].min().date()} to {df['tradeday'].max().date()} "
        f"({df['tradeday'].nunique()} days)",
        f"- **Observations**: {len(valid)} hourly data points with valid prices",
        f"- **IDM Price**: mean {valid['idm_price'].mean():.1f}, "
        f"median {valid['idm_price'].median():.1f} EUR/MWh",
        f"- **Cross-border areas**: CEPS (CZ), APG (AT), MAVIR (HU), PSE (PL), DE (50HzT+TTG avg)",
        "",
        "## Key Findings",
        "",
    ]

    # Correlation findings
    lines.append("### Correlations (Import Capacity vs IDM Price)")
    lines.append("")
    lines.append("| Area | Pearson r | Spearman r |")
    lines.append("|------|-----------|------------|")
    for res in corr_results:
        lines.append(f"| {AREA_LABELS[res['area']]} | {res['pearson_r']:+.3f} | {res['spearman_r']:+.3f} |")

    # Total import
    r_total, _ = stats.pearsonr(
        valid["total_import_cap"].dropna(),
        valid.loc[valid["total_import_cap"].notna(), "idm_price"],
    )
    rs_total, _ = stats.spearmanr(
        valid["total_import_cap"].dropna(),
        valid.loc[valid["total_import_cap"].notna(), "idm_price"],
    )
    lines.append(f"| **Total Import** | **{r_total:+.3f}** | **{rs_total:+.3f}** |")
    lines.append("")

    # Congestion findings
    lines.append("### Congestion Premium")
    lines.append("")
    lines.append("| Border | Open Median | Congested Median | Premium | % Congested |")
    lines.append("|--------|-------------|-----------------|---------|-------------|")
    for cs in congestion_stats:
        lines.append(f"| {cs['area']} | {cs['open_median']:.1f} | "
                     f"{cs['congested_median']:.1f} | {cs['premium']:+.1f} EUR/MWh | "
                     f"{cs['pct_congested']:.0f}% |")
    lines.append("")

    # Congested count
    lines.append("### Price by Number of Congested Borders")
    lines.append("")
    lines.append("| # Congested | Median Price | Mean Price | Count |")
    lines.append("|-------------|-------------|------------|-------|")
    for count, grp in valid.groupby("congested_in_count"):
        lines.append(f"| {int(count)} | {grp['idm_price'].median():.1f} | "
                     f"{grp['idm_price'].mean():.1f} | {len(grp)} |")
    lines.append("")

    lines.extend([
        "## Plots",
        "",
        "1. `01_correlation_matrix.png` - Full correlation heatmap",
        "2. `02_price_vs_capacity.png` - Per-area scatter with regression",
        "3. `03_congestion_impact.png` - Box plots: congested vs open borders",
        "4. `04_timeseries_sample.png` - Weekly overlay of price + capacity",
        "5. `05_total_capacity_effect.png` - Aggregate capacity effects + hourly profile",
    ])

    summary_path = BASE_DIR / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[+] Saved {summary_path}")


def main():
    print("=" * 60)
    print("H2H Capacity vs IDM Price Analysis")
    print("=" * 60)

    df = load_data()

    # Generate all plots
    plot_correlation_matrix(df)
    plot_price_vs_capacity(df)
    plot_congestion_impact(df)
    plot_timeseries_sample(df)
    plot_total_capacity_effect(df)

    # Statistical analysis
    corr_results, congestion_stats = compute_statistics(df)

    # Write summary
    write_summary(df, corr_results, congestion_stats)

    print("\n--- Analysis complete ---")


if __name__ == "__main__":
    main()
