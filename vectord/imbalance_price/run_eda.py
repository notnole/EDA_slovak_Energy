"""EDA for Okte.Calc.ZuctovaciaCenaOdchylky_Comb (imbalance settlement price).

Vectors pulled:
  price : Okte.Calc.ZuctovaciaCenaOdchylky_Comb  (EUR/MWh, 15-min)
  imb   : Okte.Combine.Odchylka                  (MWh, 15-min) -- regime sign

Windows:
  full  : 2026-01-01 -> today  (stats, seasonality, decomposition)
  april : 2026-04-01 -> today  (high-resolution time-series plots)

Requires SSH tunnel:
    ssh -L8080:10.100.0.70:8080 noel@greenbat1.vps.wbsprt.com
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vectord import VectordClient

OUT = Path(__file__).resolve().parent
PLOT_DIR = OUT / "plots"
DATA_DIR = OUT / "data"
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

PRICE_VECTOR = "Okte.Calc.ZuctovaciaCenaOdchylky_Comb"
IMB_VECTOR   = "Okte.Combine.Odchylka"

# Regime thresholds (MWh) -- split at zero
SURPLUS_THRESH =  0.0
DEFICIT_THRESH =  0.0

FULL_START  = datetime(2026, 1,  1, tzinfo=timezone.utc)
APRIL_START = datetime(2026, 4,  1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(client: VectordClient, start: datetime, end: datetime) -> pd.DataFrame:
    print(f"[*] Fetching {PRICE_VECTOR} ...")
    price = client.read_df(PRICE_VECTOR, start, end).rename(columns={"value": "price"})
    print(f"    {len(price)} points")

    print(f"[*] Fetching {IMB_VECTOR} ...")
    imb = client.read_df(IMB_VECTOR, start, end).rename(columns={"value": "imb"})
    print(f"    {len(imb)} points")

    df = pd.concat([price["price"], imb["imb"]], axis=1)
    df["regime"] = "surplus"
    df.loc[df["imb"] <  DEFICIT_THRESH, "regime"] = "deficit"
    df["hour"]    = df.index.hour + df.index.minute / 60
    df["hour_of_day"] = df.index.hour
    df["weekday"] = df.index.day_of_week  # 0=Mon
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

REGIME_COLORS = {"surplus": "#2196F3", "deficit": "#F44336"}


def plot_timeseries_2026(df: pd.DataFrame, ylim: tuple) -> None:
    """Full-year 15-min price with 7-day rolling mean, regime background shading."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # --- top panel: raw 15-min scatter, coloured by regime ---
    ax = axes[0]
    for regime, color in REGIME_COLORS.items():
        mask = df["regime"] == regime
        ax.scatter(df.index[mask], df["price"][mask], s=1, alpha=0.25,
                   color=color, label=regime, rasterized=True)
    roll = df["price"].rolling("7D").mean()
    ax.plot(roll.index, roll.values, color="black", lw=1.2, label="7-day rolling mean")
    ax.set_ylim(*ylim)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Imbalance settlement price (15-min) -- 2026  [axis p1-p99]")
    ax.legend(markerscale=4, fontsize=8)
    ax.grid(alpha=0.3)

    # --- bottom panel: daily median price per regime ---
    ax2 = axes[1]
    for regime, color in REGIME_COLORS.items():
        sub = df.loc[df["regime"] == regime, "price"].resample("1D").median()
        ax2.plot(sub.index, sub.values, color=color, lw=0.9, alpha=0.8, label=regime)
    ax2.set_ylabel("EUR/MWh (daily median)")
    ax2.set_title("Daily median price by regime")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_timeseries_2026.png", dpi=110)
    plt.close(fig)
    print("[+] 01_timeseries_2026.png")


def plot_april_detail(df_apr: pd.DataFrame, ylim: tuple) -> None:
    """April 15-min detail: price coloured by regime, imbalance on second axis."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

    ax = axes[0]
    for regime, color in REGIME_COLORS.items():
        mask = df_apr["regime"] == regime
        ax.scatter(df_apr.index[mask], df_apr["price"][mask], s=3, alpha=0.5,
                   color=color, label=regime, rasterized=True)
    ax.set_ylim(*ylim)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Imbalance settlement price -- April 2026 (15-min detail)  [axis p1-p99]")
    ax.legend(markerscale=3, fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df_apr.index, df_apr["imb"], color="#555", lw=0.7, alpha=0.8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.fill_between(df_apr.index, df_apr["imb"], 0,
                     where=df_apr["imb"] > 0, alpha=0.25, color="#2196F3", label="surplus")
    ax2.fill_between(df_apr.index, df_apr["imb"], 0,
                     where=df_apr["imb"] < 0, alpha=0.25, color="#F44336", label="deficit")
    ax2.set_ylabel("System imbalance (MWh)")
    ax2.set_title("System imbalance (Okte.Combine.Odchylka)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_april_detail.png", dpi=110)
    plt.close(fig)
    print("[+] 02_april_detail.png")


def plot_price_vs_imbalance(df_apr: pd.DataFrame, ylim: tuple) -> None:
    """Scatter price vs imbalance (April), to see the pricing curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for regime, color in REGIME_COLORS.items():
        mask = df_apr["regime"] == regime
        ax.scatter(df_apr.loc[mask, "imb"], df_apr.loc[mask, "price"],
                   s=6, alpha=0.4, color=color, label=regime)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_ylim(*ylim)
    ax.set_xlabel("System imbalance (MWh)")
    ax.set_ylabel("Settlement price (EUR/MWh)")
    ax.set_title("Settlement price vs system imbalance -- April 2026  [axis p1-p99]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_price_vs_imbalance.png", dpi=110)
    plt.close(fig)
    print("[+] 03_price_vs_imbalance.png")


def plot_distribution(df: pd.DataFrame) -> None:
    """Histogram overall + per regime, and CDF."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # overall
    ax = axes[0]
    clipped = df["price"].clip(df["price"].quantile(0.01), df["price"].quantile(0.99))
    ax.hist(clipped, bins=80, color="steelblue", edgecolor="white")
    ax.axvline(df["price"].median(), color="k", lw=1.2, ls="--",
               label=f"median {df['price'].median():.0f}")
    ax.set_title("Overall (p1-p99 clipped)")
    ax.set_xlabel("EUR/MWh")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # per regime
    for ax, (regime, color) in zip(axes[1:], REGIME_COLORS.items()):
        sub = df.loc[df["regime"] == regime, "price"]
        if sub.empty:
            ax.set_title(f"{regime} (no data)")
            continue
        lo, hi = sub.quantile(0.01), sub.quantile(0.99)
        ax.hist(sub.clip(lo, hi), bins=60, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(sub.median(), color="k", lw=1.2, ls="--",
                   label=f"med={sub.median():.0f}")
        ax.set_title(f"{regime.capitalize()} (imb {'>=0' if regime == 'surplus' else '<0'})")
        ax.set_xlabel("EUR/MWh")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Settlement price distributions -- 2026", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_distributions.png", dpi=110)
    plt.close(fig)
    print("[+] 04_distributions.png")


def plot_hourly_seasonality(df: pd.DataFrame) -> None:
    """Median price and regime share by hour-of-day."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax = axes[0]
    overall = df.groupby("hour_of_day")["price"].median()
    ax.plot(overall.index, overall.values, color="black", lw=2, label="overall")
    for regime, color in REGIME_COLORS.items():
        sub = df.loc[df["regime"] == regime].groupby("hour_of_day")["price"].median()
        if sub.empty:
            continue
        ax.plot(sub.index, sub.values, color=color, lw=1.2, alpha=0.8, label=regime)
    ax.set_ylabel("Median EUR/MWh")
    ax.set_title("Median settlement price by hour of day (2026)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    regime_share = df.groupby("hour_of_day")["regime"].value_counts(normalize=True).unstack(fill_value=0)
    bottom = np.zeros(24)
    for regime, color in REGIME_COLORS.items():
        if regime not in regime_share.columns:
            continue
        vals = regime_share[regime].reindex(range(24), fill_value=0).values
        ax2.bar(range(24), vals, bottom=bottom, color=color, alpha=0.8, label=regime)
        bottom += vals
    ax2.set_xlabel("Hour of day (UTC)")
    ax2.set_ylabel("Fraction of periods")
    ax2.set_title("Regime share by hour of day")
    ax2.set_xticks(range(24))
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_hourly_seasonality.png", dpi=110)
    plt.close(fig)
    print("[+] 05_hourly_seasonality.png")


def plot_weekday_seasonality(df: pd.DataFrame, ylim: tuple) -> None:
    """Box of price by day-of-week per regime."""
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (regime, color) in zip(axes, REGIME_COLORS.items()):
        sub = df.loc[df["regime"] == regime]
        if sub.empty:
            ax.set_title(f"{regime} (no data)")
            continue
        data_by_day = [sub.loc[sub["weekday"] == d, "price"].dropna().values for d in range(7)]
        bp = ax.boxplot(data_by_day, patch_artist=True, medianprops={"color": "black"},
                        flierprops={"marker": ".", "alpha": 0.3, "ms": 3})
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(1, 8))
        ax.set_xticklabels(day_labels)
        ax.set_title(f"{regime.capitalize()} regime")
        ax.set_ylabel("EUR/MWh" if ax == axes[0] else "")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Settlement price by day-of-week and regime -- 2026  [axis p1-p99]", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_weekday_seasonality.png", dpi=110)
    plt.close(fig)
    print("[+] 06_weekday_seasonality.png")


def plot_decomposition(df: pd.DataFrame) -> None:
    """STL decomposition of daily average price. Falls back to rolling if statsmodels absent."""
    daily = df["price"].resample("1D").median().dropna()

    try:
        from statsmodels.tsa.seasonal import STL
        stl = STL(daily, period=7, robust=True).fit()
        trend    = stl.trend
        seasonal = stl.seasonal
        resid    = stl.resid
        method = "STL (weekly period)"
    except ImportError:
        trend    = daily.rolling(7, center=True).mean()
        seasonal = daily - trend
        resid    = daily - trend - seasonal
        method = "Rolling-window (fallback)"

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(daily.index, daily.values, lw=0.9, color="steelblue")
    axes[0].set_ylabel("EUR/MWh")
    axes[0].set_title(f"Daily median price -- {method} decomposition")
    axes[0].grid(alpha=0.3)

    axes[1].plot(trend.index, trend.values, lw=1.2, color="black")
    axes[1].set_ylabel("Trend")
    axes[1].grid(alpha=0.3)

    axes[2].plot(seasonal.index, seasonal.values, lw=0.9, color="darkorange")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel("Seasonal")
    axes[2].grid(alpha=0.3)

    axes[3].plot(resid.index, resid.values, lw=0.8, color="firebrick", alpha=0.8)
    axes[3].axhline(0, color="k", lw=0.5)
    axes[3].set_ylabel("Residual")
    axes[3].grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_decomposition.png", dpi=110)
    plt.close(fig)
    print("[+] 07_decomposition.png")
    return method


def plot_regime_boxplot(df: pd.DataFrame, ylim: tuple) -> None:
    """Side-by-side boxplot of price in each regime."""
    fig, ax = plt.subplots(figsize=(8, 6))
    groups = [df.loc[df["regime"] == r, "price"].dropna().values
              for r in ["surplus", "deficit"]]
    bp = ax.boxplot(groups, patch_artist=True,
                    medianprops={"color": "black", "lw": 2},
                    flierprops={"marker": ".", "alpha": 0.3, "ms": 3})
    colors = [REGIME_COLORS["surplus"], REGIME_COLORS["deficit"]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_ylim(*ylim)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Surplus (imb >= 0)", "Deficit (imb < 0)"])
    ax.set_ylabel("Settlement price (EUR/MWh)")
    ax.set_title("Settlement price by imbalance regime -- 2026  [axis p1-p99]")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "08_regime_boxplot.png", dpi=110)
    plt.close(fig)
    print("[+] 08_regime_boxplot.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for label, sub in [("all", df), ("surplus", df[df["regime"] == "surplus"]),
                        ("deficit", df[df["regime"] == "deficit"])]:
        p = sub["price"].dropna()
        stats[label] = dict(n=len(p), mean=p.mean(), median=p.median(),
                            std=p.std(), p5=p.quantile(0.05), p95=p.quantile(0.95),
                            min=p.min(), max=p.max())
    return stats


def write_summary(df: pd.DataFrame, df_apr: pd.DataFrame,
                  stats: dict, decomp_method: str,
                  start: datetime, end: datetime) -> None:
    regime_counts = df["regime"].value_counts()
    n_total = len(df.dropna(subset=["price"]))

    def fmt_row(label, s):
        return (f"| {label} | {s['n']} | {s['mean']:.1f} | {s['median']:.1f} | "
                f"{s['std']:.1f} | {s['p5']:.1f} | {s['p95']:.1f} | "
                f"{s['min']:.1f} | {s['max']:.1f} |")

    regime_share = {r: regime_counts.get(r, 0) / n_total * 100
                    for r in ["surplus", "deficit"]}

    lines = [
        "# Imbalance Settlement Price EDA\n",
        f"Vector: `{PRICE_VECTOR}`  \n",
        f"Regime split from: `{IMB_VECTOR}` (split at imb = 0)  \n",
        f"Full window: `{start.date()}` -> `{end.date()}`  \n",
        f"April detail: `{APRIL_START.date()}` -> `{end.date()}`  \n",
        f"N 15-min periods: {n_total}\n",
        "\n## Regime breakdown\n",
        "| Regime | N | Share |",
        "| --- | --- | --- |",
        *[f"| {r} | {regime_counts.get(r, 0)} | {regime_share[r]:.1f}% |"
          for r in ["surplus", "deficit"]],
        "\n## Descriptive statistics (EUR/MWh)\n",
        "| Regime | N | Mean | Median | Std | P5 | P95 | Min | Max |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *[fmt_row(lbl, stats[lbl]) for lbl in ["all", "surplus", "deficit"]],
        f"\n## Decomposition\n",
        f"Method: {decomp_method} on daily median price.\n",
        "\n## Plots\n",
        "- `01_timeseries_2026.png` — full-year 15-min scatter coloured by regime, 7-day rolling mean, daily median by regime",
        "- `02_april_detail.png` — April 15-min price + system imbalance on second axis",
        "- `03_price_vs_imbalance.png` — price vs imbalance scatter (April), reveals pricing curve",
        "- `04_distributions.png` — histograms overall, surplus (imb>=0), deficit (imb<0)",
        "- `05_hourly_seasonality.png` — median price by hour-of-day, regime share by hour",
        "- `06_weekday_seasonality.png` — boxplot by day-of-week per regime",
        "- `07_decomposition.png` — daily median price STL decomposition (trend / seasonal / residual)",
        "- `08_regime_boxplot.png` — price boxplot surplus vs deficit\n",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("[+] summary.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    client = VectordClient()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    print(f"[*] Full window : {FULL_START.date()} -> {end.date()}")
    print(f"[*] April window: {APRIL_START.date()} -> {end.date()}")

    df      = load(client, FULL_START, end)
    df_apr  = df[df.index >= APRIL_START]

    n_price = df["price"].notna().sum()
    n_imb   = df["imb"].notna().sum()
    print(f"[*] Price points: {n_price}  |  Imbalance points: {n_imb}")

    if n_price == 0:
        print("[!] No price data returned -- check vector name and tunnel.")
        return

    # save raw data
    df.to_csv(DATA_DIR / "full_2026.csv")
    df_apr.to_csv(DATA_DIR / "april_2026.csv")
    print("[+] Raw data saved to data/")

    # clip bounds for all raw-price axes (p1-p99 of full series)
    ylim = (df["price"].quantile(0.01), df["price"].quantile(0.99))
    print(f"[*] Plot y-axis clip: {ylim[0]:.1f} .. {ylim[1]:.1f} EUR/MWh  (p1-p99)")

    plot_timeseries_2026(df, ylim)
    plot_april_detail(df_apr, ylim)
    plot_price_vs_imbalance(df_apr, ylim)
    plot_distribution(df)
    plot_hourly_seasonality(df)
    plot_weekday_seasonality(df, ylim)
    decomp_method = plot_decomposition(df)
    plot_regime_boxplot(df, ylim)

    stats = compute_stats(df)
    write_summary(df, df_apr, stats, decomp_method, FULL_START, end)

    print("--- Done ---")


if __name__ == "__main__":
    main()
