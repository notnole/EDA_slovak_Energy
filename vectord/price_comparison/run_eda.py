"""Comparison: real imbalance price vs proxy, split by surplus/deficit regime.

Vectors:
  real  : Okte.Combine.CenaSysOdchylkySR        (EUR/MWh, 15-min) -- actual settlement price
  proxy : Okte.Calc.ZuctovaciaCenaOdchylky_Comb  (EUR/MWh, 15-min) -- proxy/calc price
  imb   : Okte.Combine.Odchylka                  (MWh, 15-min)     -- regime sign (split at 0)

Windows:
  full  : 2026-01-01 -> today  (stats, distributions, hourly patterns)
  april : 2026-04-01 -> today  (high-resolution detail)

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

OUT      = Path(__file__).resolve().parent
PLOT_DIR = OUT / "plots"
DATA_DIR = OUT / "data"
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

REAL_VECTOR  = "Okte.Combine.CenaSysOdchylkySR"
PROXY_VECTOR = "Okte.Calc.ZuctovaciaCenaOdchylky_Comb"
IMB_VECTOR   = "Okte.Combine.Odchylka"
ACE_VECTOR   = "DaE.OH.RE_WITH_GCC_SEP_3M_GCC"

FULL_START  = datetime(2026, 1,  1, tzinfo=timezone.utc)
APRIL_START = datetime(2026, 4,  1, tzinfo=timezone.utc)

REGIME_COLORS  = {"surplus": "#2196F3", "deficit": "#F44336"}
SERIES_COLORS  = {"real": "#1B5E20", "proxy": "#FF6F00"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(client: VectordClient, start: datetime, end: datetime) -> pd.DataFrame:
    print(f"[*] Fetching {REAL_VECTOR} ...")
    real  = client.read_df(REAL_VECTOR,  start, end).rename(columns={"value": "real"})
    print(f"    {len(real)} points")

    print(f"[*] Fetching {PROXY_VECTOR} ...")
    proxy = client.read_df(PROXY_VECTOR, start, end).rename(columns={"value": "proxy"})
    print(f"    {len(proxy)} points")

    print(f"[*] Fetching {IMB_VECTOR} ...")
    imb   = client.read_df(IMB_VECTOR,   start, end).rename(columns={"value": "imb"})
    print(f"    {len(imb)} points")

    df = pd.concat([real["real"], proxy["proxy"], imb["imb"]], axis=1)
    df["regime"]     = "surplus"
    df.loc[df["imb"] < 0, "regime"] = "deficit"
    df["spread"]     = df["proxy"] - df["real"]
    df["hour_of_day"] = df.index.hour
    df["weekday"]    = df.index.day_of_week
    return df


def clip_bounds(df: pd.DataFrame, col: str, q: float = 0.01) -> tuple:
    lo = min(df["real"].quantile(q),  df["proxy"].quantile(q))
    hi = max(df["real"].quantile(1-q), df["proxy"].quantile(1-q))
    return lo, hi


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_timeseries(df: pd.DataFrame, ylim: tuple) -> None:
    """Full-year daily medians of real vs proxy, per regime."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

    for ax, (regime, rc) in zip(axes, REGIME_COLORS.items()):
        sub = df[df["regime"] == regime]
        for col, color in SERIES_COLORS.items():
            daily = sub[col].resample("1D").median()
            ax.plot(daily.index, daily.values, color=color, lw=1.0,
                    alpha=0.85, label=col)
        ax.set_ylim(*ylim)
        ax.set_ylabel("EUR/MWh (daily median)")
        ax.set_title(f"{regime.capitalize()} regime -- real vs proxy")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_facecolor(rc + "08")  # faint regime tint

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    fig.suptitle("Real vs proxy imbalance price -- daily medians by regime (2026)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_timeseries_2026.png", dpi=110)
    plt.close(fig)
    print("[+] 01_timeseries_2026.png")


def plot_april_detail(df_apr: pd.DataFrame, ylim: tuple) -> None:
    """April 15-min detail: real vs proxy side-by-side per regime."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    for ax, (regime, rc) in zip(axes, REGIME_COLORS.items()):
        sub = df_apr[df_apr["regime"] == regime]
        for col, color in SERIES_COLORS.items():
            ax.scatter(sub.index, sub[col], s=3, alpha=0.45, color=color,
                       label=col, rasterized=True)
        ax.set_ylim(*ylim)
        ax.set_ylabel("EUR/MWh")
        ax.set_title(f"{regime.capitalize()} -- April 2026  [p1-p99 axis]")
        ax.legend(markerscale=3, fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    fig.autofmt_xdate()
    fig.suptitle("Real vs proxy imbalance price -- April 2026 (15-min)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_april_detail.png", dpi=110)
    plt.close(fig)
    print("[+] 02_april_detail.png")


def plot_scatter(df: pd.DataFrame, ylim: tuple) -> None:
    """Scatter proxy vs real, coloured by regime (overall + per-regime panels)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # overall
    ax = axes[0]
    for regime, color in REGIME_COLORS.items():
        mask = df["regime"] == regime
        sub  = df[mask].dropna(subset=["real", "proxy"])
        ax.scatter(sub["real"], sub["proxy"], s=2, alpha=0.2, color=color,
                   label=regime, rasterized=True)
    lo, hi = ylim
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Real (EUR/MWh)")
    ax.set_ylabel("Proxy (EUR/MWh)")
    ax.set_title("Overall  [p1-p99]")
    ax.legend(markerscale=4, fontsize=8)
    ax.grid(alpha=0.3)

    # per regime
    for ax, (regime, color) in zip(axes[1:], REGIME_COLORS.items()):
        sub = df[df["regime"] == regime].dropna(subset=["real", "proxy"])
        ax.scatter(sub["real"], sub["proxy"], s=2, alpha=0.25, color=color,
                   rasterized=True)
        lo_r = min(sub["real"].quantile(0.01), sub["proxy"].quantile(0.01))
        hi_r = max(sub["real"].quantile(0.99), sub["proxy"].quantile(0.99))
        ax.plot([lo_r, hi_r], [lo_r, hi_r], "k--", lw=1, label="y=x")
        ax.set_xlim(lo_r, hi_r)
        ax.set_ylim(lo_r, hi_r)
        corr = sub["real"].corr(sub["proxy"])
        mae  = (sub["proxy"] - sub["real"]).abs().mean()
        ax.set_xlabel("Real (EUR/MWh)")
        ax.set_ylabel("Proxy (EUR/MWh)")
        ax.set_title(f"{regime.capitalize()}  r={corr:.3f}  MAE={mae:.1f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Proxy vs real price scatter -- 2026", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_scatter.png", dpi=110)
    plt.close(fig)
    print("[+] 03_scatter.png")


def plot_distributions(df: pd.DataFrame) -> None:
    """Overlapping distributions of real vs proxy per regime."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (regime, rc) in zip(axes, REGIME_COLORS.items()):
        sub = df[df["regime"] == regime]
        for col, color in SERIES_COLORS.items():
            s = sub[col].dropna()
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            ax.hist(s.clip(lo, hi), bins=70, alpha=0.45, color=color,
                    edgecolor="white", label=f"{col}  med={s.median():.0f}")
        ax.set_title(f"{regime.capitalize()} regime")
        ax.set_xlabel("EUR/MWh")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Real vs proxy price distributions by regime -- 2026  [p1-p99 per series]",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_distributions.png", dpi=110)
    plt.close(fig)
    print("[+] 04_distributions.png")


def plot_spread(df: pd.DataFrame) -> None:
    """(Proxy - real) spread: timeseries + distribution, by regime."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    # timeseries per regime
    for ax, (regime, color) in zip(axes[:, 0], REGIME_COLORS.items()):
        sub   = df[df["regime"] == regime]["spread"].dropna()
        daily = sub.resample("1D").median()
        ax.plot(daily.index, daily.values, color=color, lw=1.0)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.fill_between(daily.index, daily.values, 0,
                        where=daily.values > 0, alpha=0.2, color=color)
        ax.fill_between(daily.index, daily.values, 0,
                        where=daily.values < 0, alpha=0.2, color="grey")
        bias = sub.mean()
        ax.set_title(f"{regime.capitalize()} -- spread (proxy-real)  bias={bias:+.1f}")
        ax.set_ylabel("EUR/MWh")
        ax.grid(alpha=0.3)

    # distribution per regime
    for ax, (regime, color) in zip(axes[:, 1], REGIME_COLORS.items()):
        sub = df[df["regime"] == regime]["spread"].dropna()
        lo, hi = sub.quantile(0.01), sub.quantile(0.99)
        ax.hist(sub.clip(lo, hi), bins=70, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="k", lw=1.0, ls="--")
        ax.axvline(sub.mean(), color="k", lw=1.2,
                   label=f"mean {sub.mean():+.1f}")
        ax.axvline(sub.median(), color="grey", lw=1.2, ls=":",
                   label=f"median {sub.median():+.1f}")
        ax.set_title(f"{regime.capitalize()} -- spread distribution  [p1-p99]")
        ax.set_xlabel("Proxy - Real (EUR/MWh)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[-1, :]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    fig.suptitle("Proxy - Real spread by regime -- 2026", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_spread.png", dpi=110)
    plt.close(fig)
    print("[+] 05_spread.png")


def plot_hourly(df: pd.DataFrame) -> None:
    """Median real vs proxy by hour-of-day, per regime."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (regime, rc) in zip(axes, REGIME_COLORS.items()):
        sub = df[df["regime"] == regime]
        for col, color in SERIES_COLORS.items():
            med = sub.groupby("hour_of_day")[col].median()
            ax.plot(med.index, med.values, color=color, lw=1.8, label=col)
        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel("Median EUR/MWh")
        ax.set_title(f"{regime.capitalize()} regime")
        ax.set_xticks(range(24))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Median real vs proxy by hour-of-day -- 2026", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_hourly.png", dpi=110)
    plt.close(fig)
    print("[+] 06_hourly.png")


def plot_spread_vs_imbalance(df_apr: pd.DataFrame) -> None:
    """Scatter: spread (proxy-real) vs imbalance size -- April (more data density)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (regime, color) in zip(axes, REGIME_COLORS.items()):
        sub = df_apr[df_apr["regime"] == regime].dropna(subset=["imb", "spread"])
        lo, hi = sub["spread"].quantile(0.01), sub["spread"].quantile(0.99)
        sub = sub[(sub["spread"] >= lo) & (sub["spread"] <= hi)]
        ax.scatter(sub["imb"].abs(), sub["spread"], s=5, alpha=0.35,
                   color=color, rasterized=True)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("|Imbalance| (MWh)")
        ax.set_ylabel("Proxy - Real (EUR/MWh)")
        ax.set_title(f"{regime.capitalize()}")
        ax.grid(alpha=0.3)

    fig.suptitle("Spread (proxy-real) vs |imbalance| -- April 2026  [spread p1-p99]",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_spread_vs_imbalance.png", dpi=110)
    plt.close(fig)
    print("[+] 07_spread_vs_imbalance.png")


# ---------------------------------------------------------------------------
# New plots
# ---------------------------------------------------------------------------

def plot_3day_step(df_apr: pd.DataFrame) -> None:
    """3-day step-function timeseries of real vs proxy, background shaded by regime."""
    # pick 3 days from mid-April for a representative mix of both regimes
    start3 = pd.Timestamp("2026-04-10", tz="UTC")
    end3   = pd.Timestamp("2026-04-13", tz="UTC")
    sub = df_apr.loc[start3:end3].dropna(subset=["real", "proxy"])

    if sub.empty:
        # fallback: first 3 days of available April data
        start3 = df_apr.index[0]
        end3   = start3 + pd.Timedelta(days=3)
        sub = df_apr.loc[start3:end3].dropna(subset=["real", "proxy"])

    fig, ax = plt.subplots(figsize=(15, 6))

    # shade background by regime
    regime_arr = sub["regime"]
    for i in range(len(sub) - 1):
        t0 = sub.index[i]
        t1 = sub.index[i + 1]
        color = REGIME_COLORS[regime_arr.iloc[i]]
        ax.axvspan(t0, t1, alpha=0.10, color=color, linewidth=0)

    # step lines
    ax.step(sub.index, sub["real"],  where="post", color=SERIES_COLORS["real"],
            lw=1.8, label="real")
    ax.step(sub.index, sub["proxy"], where="post", color=SERIES_COLORS["proxy"],
            lw=1.8, label="proxy", ls="--")

    # regime legend patches
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(color=REGIME_COLORS["surplus"], alpha=0.35, label="surplus bg"),
                Patch(color=REGIME_COLORS["deficit"], alpha=0.35, label="deficit bg")]
    ax.legend(handles=handles, fontsize=9)

    ax.set_ylabel("EUR/MWh")
    ax.set_title(
        f"Real vs proxy settlement price -- 3-day step view  "
        f"({start3.strftime('%d %b')} - {end3.strftime('%d %b %Y')})"
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "10_3day_step.png", dpi=110)
    plt.close(fig)
    print("[+] 10_3day_step.png")


def plot_cumulative_step(df: pd.DataFrame) -> None:
    """Cumulative sum of (proxy-real) spread over time, step style, per regime."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    for ax, (regime, color) in zip(axes, REGIME_COLORS.items()):
        sub = df[df["regime"] == regime]["spread"].dropna().sort_index()
        cumsum = sub.cumsum()
        ax.step(cumsum.index, cumsum.values, where="post", color=color, lw=1.2)
        ax.fill_between(cumsum.index, cumsum.values, step="post",
                        alpha=0.18, color=color)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        final = cumsum.iloc[-1]
        ax.set_ylabel("Cumulative proxy-real (EUR/MWh)")
        ax.set_title(
            f"{regime.capitalize()} -- cumulative spread  "
            f"(final: {final:+.0f} EUR/MWh over {len(sub)} periods)"
        )
        ax.grid(alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    fig.suptitle("Cumulative (proxy - real) spread by regime -- 2026", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "08_cumulative_step.png", dpi=110)
    plt.close(fig)
    print("[+] 08_cumulative_step.png")


def plot_regime_confusion(df: pd.DataFrame) -> dict:
    """Analyse periods where spread direction opposes the regime expectation.

    In surplus (imb>=0) the proxy tends to be below real (spread < 0).
    When spread > 0 in surplus  -> proxy acting like deficit pricing.

    In deficit (imb< 0) the proxy tends to be above real (spread > 0).
    When spread < 0 in deficit  -> proxy acting like surplus pricing.

    These 'wrong-direction' periods are the closest proxy for regime confusion.
    """
    clean = df.dropna(subset=["real", "proxy", "spread"])
    total_abs_error = clean["spread"].abs().sum()

    confusion = {}
    for regime, wrong_sign in [("surplus", ">"), ("deficit", "<")]:
        sub = clean[clean["regime"] == regime]
        if wrong_sign == ">":
            confused = sub[sub["spread"] > 0]
        else:
            confused = sub[sub["spread"] < 0]

        n_confused   = len(confused)
        n_total      = len(sub)
        abs_err_confused = confused["spread"].abs().sum()
        pct_periods  = n_confused / n_total * 100 if n_total else 0
        pct_error    = abs_err_confused / total_abs_error * 100 if total_abs_error else 0
        mae_confused = confused["spread"].abs().mean() if n_confused else 0
        mae_normal   = sub[~sub.index.isin(confused.index)]["spread"].abs().mean()

        confusion[regime] = dict(
            n_total=n_total, n_confused=n_confused,
            pct_periods=pct_periods, pct_error=pct_error,
            mae_confused=mae_confused, mae_normal=mae_normal,
        )

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (regime, color) in zip(axes, REGIME_COLORS.items()):
        sub = clean[clean["regime"] == regime]
        wrong_sign = ">" if regime == "surplus" else "<"
        if wrong_sign == ">":
            normal    = sub[sub["spread"] <= 0]["spread"]
            confused  = sub[sub["spread"] >  0]["spread"]
            wrong_label = "wrong-dir (spread > 0)"
        else:
            normal    = sub[sub["spread"] >= 0]["spread"]
            confused  = sub[sub["spread"] <  0]["spread"]
            wrong_label = "wrong-dir (spread < 0)"

        lo = sub["spread"].quantile(0.01)
        hi = sub["spread"].quantile(0.99)
        c  = confusion[regime]

        ax.hist(normal.clip(lo, hi),  bins=60, alpha=0.6, color=color,
                edgecolor="white", label=f"expected dir  n={len(normal)}")
        ax.hist(confused.clip(lo, hi), bins=40, alpha=0.7, color="black",
                edgecolor="white", label=f"{wrong_label}  n={len(confused)}")
        ax.axvline(0, color="k", lw=1.0, ls="--")

        info = (
            f"{c['pct_periods']:.1f}% of periods\n"
            f"{c['pct_error']:.1f}% of total abs error\n"
            f"MAE confused: {c['mae_confused']:.1f}\n"
            f"MAE normal:   {c['mae_normal']:.1f}"
        )
        ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=8,
                va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title(f"{regime.capitalize()} -- wrong-direction periods")
        ax.set_xlabel("Proxy - Real (EUR/MWh)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Regime confusion: periods where spread direction opposes expectation -- 2026",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "09_regime_confusion.png", dpi=110)
    plt.close(fig)
    print("[+] 09_regime_confusion.png")

    return confusion


# ---------------------------------------------------------------------------
# ACE signal confusion analysis
# ---------------------------------------------------------------------------

def load_ace(client: VectordClient, start: datetime, end: datetime) -> pd.Series:
    """Fetch 3-min ACE, resample to 15-min mean (mirrors Ipesoft %STEP_15Min)."""
    print(f"[*] Fetching {ACE_VECTOR} (3-min) ...")
    raw = client.read_df(ACE_VECTOR, start, end).rename(columns={"value": "ace"})
    print(f"    {len(raw)} 3-min points")
    ace_15 = raw["ace"].resample("15min").mean()
    print(f"    {len(ace_15)} 15-min periods after resample")
    return ace_15


def build_ace_df(df: pd.DataFrame, ace_15: pd.Series) -> pd.DataFrame:
    """Merge ACE into main df and label confusion.

    Proxy regime from ACE sign (formula logic):
      ace >= 0  ->  proxy applies REp  (proxy thinks: deficit)
      ace <  0  ->  proxy applies REm  (proxy thinks: surplus)

    Actual regime from settled imbalance (our convention):
      imb >= 0  ->  surplus
      imb <  0  ->  deficit

    Aligned:
      ace >= 0 AND imb < 0   (both deficit)
      ace <  0 AND imb >= 0  (both surplus)

    Confused:
      ace >= 0 AND imb >= 0  (proxy: deficit, actual: surplus)  -> proxy uses REp in surplus
      ace <  0 AND imb <  0  (proxy: surplus, actual: deficit)  -> proxy uses REm in deficit
    """
    merged = df.copy()
    merged["ace"] = ace_15.reindex(merged.index)

    # proxy's view of regime (from ACE sign)
    merged["ace_regime"] = "surplus"
    merged.loc[merged["ace"] >= 0, "ace_regime"] = "deficit"

    # confusion: proxy regime != actual regime
    merged["confused"] = merged["ace_regime"] != merged["regime"]

    return merged


def analyse_ace_confusion(adf: pd.DataFrame) -> dict:
    """Compute confusion stats: rate, error contribution, MAE split."""
    clean = adf.dropna(subset=["real", "proxy", "ace", "spread"])
    total_abs = clean["spread"].abs().sum()

    results = {}
    for regime in ["all", "surplus", "deficit"]:
        sub = clean if regime == "all" else clean[clean["regime"] == regime]
        aligned  = sub[~sub["confused"]]
        confused = sub[sub["confused"]]
        results[regime] = dict(
            n_total   = len(sub),
            n_confused= len(confused),
            pct_prd   = len(confused) / len(sub) * 100 if len(sub) else 0,
            pct_err   = confused["spread"].abs().sum() / total_abs * 100 if total_abs else 0,
            mae_all   = sub["spread"].abs().mean(),
            mae_conf  = confused["spread"].abs().mean() if len(confused) else float("nan"),
            mae_aln   = aligned["spread"].abs().mean()  if len(aligned)  else float("nan"),
            bias_conf = confused["spread"].mean()        if len(confused) else float("nan"),
            bias_aln  = aligned["spread"].mean()         if len(aligned)  else float("nan"),
        )
    return results


def plot_ace_confusion(adf: pd.DataFrame, ace_stats: dict) -> None:
    """4-panel plot: signals timeseries, confused periods marked, error breakdown."""
    clean = adf.dropna(subset=["real", "proxy", "ace", "spread"])

    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=False)

    # --- panel 1: daily signals + confusion rate ---
    ax = axes[0]
    daily_imb = clean["imb"].resample("1D").mean()
    daily_ace = clean["ace"].resample("1D").mean()
    ax.plot(daily_imb.index, daily_imb.values, color="#2196F3", lw=1.2,
            label="Imbalance (Odchylka, 15-min mean)", alpha=0.85)
    ax.plot(daily_ace.index, daily_ace.values, color="#FF6F00", lw=1.2,
            ls="--", label="ACE (3-min -> 15-min mean)", alpha=0.85)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("MWh / MW")
    ax.set_title("Daily mean: imbalance vs ACE signal -- sign agreement drives proxy regime")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # --- panel 2: confusion flag timeseries (daily %) ---
    ax2 = axes[1]
    daily_conf = clean["confused"].resample("1D").mean() * 100
    ax2.bar(daily_conf.index, daily_conf.values, width=1, color="crimson", alpha=0.7)
    ax2.set_ylabel("% confused periods")
    ax2.set_title("Daily confusion rate (ACE sign != imbalance sign)")
    ax2.grid(alpha=0.3, axis="y")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # --- panel 3: spread distribution aligned vs confused ---
    ax3 = axes[2]
    aligned  = clean[~clean["confused"]]["spread"]
    confused = clean[ clean["confused"]]["spread"]
    lo = clean["spread"].quantile(0.01)
    hi = clean["spread"].quantile(0.99)
    ax3.hist(aligned.clip(lo, hi),  bins=70, alpha=0.6, color="#1B5E20",
             edgecolor="white", label=f"aligned  n={len(aligned)}  MAE={aligned.abs().mean():.1f}")
    ax3.hist(confused.clip(lo, hi), bins=50, alpha=0.7, color="crimson",
             edgecolor="white", label=f"confused n={len(confused)}  MAE={confused.abs().mean():.1f}" if len(confused) else "confused n=0")
    ax3.axvline(0, color="k", lw=1.0, ls="--")
    ax3.set_xlabel("Proxy - Real (EUR/MWh)  [p1-p99]")
    ax3.set_ylabel("Periods")
    ax3.set_title("Spread distribution: aligned vs confused periods")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # --- panel 4: MAE breakdown table ---
    ax4 = axes[3]
    ax4.axis("off")
    rows = []
    for regime in ["all", "surplus", "deficit"]:
        s = ace_stats[regime]
        rows.append([
            regime,
            f"{s['n_confused']} / {s['n_total']}",
            f"{s['pct_prd']:.1f}%",
            f"{s['pct_err']:.1f}%",
            f"{s['mae_aln']:.1f}",
            f"{s['mae_conf']:.1f}" if not np.isnan(s['mae_conf']) else "n/a",
            f"{s['bias_aln']:+.1f}" if not np.isnan(s['bias_aln']) else "n/a",
            f"{s['bias_conf']:+.1f}" if not np.isnan(s['bias_conf']) else "n/a",
        ])
    cols = ["Regime", "Confused/Total", "% periods", "% abs error",
            "MAE aligned", "MAE confused", "Bias aligned", "Bias confused"]
    tbl = ax4.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax4.set_title("ACE vs imbalance confusion -- summary table", pad=12)

    fig.suptitle(
        "ACE signal confusion analysis: does the proxy pick the wrong price formula?",
        fontsize=12, y=1.002
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "11_ace_confusion.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[+] 11_ace_confusion.png")


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> dict:
    results = {}
    for regime in ["all", "surplus", "deficit"]:
        sub = df if regime == "all" else df[df["regime"] == regime]
        sub = sub.dropna(subset=["real", "proxy"])
        spread = sub["proxy"] - sub["real"]
        results[regime] = {
            "n":          len(sub),
            "corr":       sub["real"].corr(sub["proxy"]),
            "mae":        spread.abs().mean(),
            "rmse":       np.sqrt((spread**2).mean()),
            "bias":       spread.mean(),
            "med_real":   sub["real"].median(),
            "med_proxy":  sub["proxy"].median(),
        }
    return results


def write_summary(stats: dict, confusion: dict, start: datetime, end: datetime) -> None:
    def row(label, s):
        return (f"| {label} | {s['n']} | {s['corr']:.3f} | {s['mae']:.1f} | "
                f"{s['rmse']:.1f} | {s['bias']:+.1f} | "
                f"{s['med_real']:.1f} | {s['med_proxy']:.1f} |")

    def crow(regime, c):
        return (f"| {regime} | {c['n_confused']} / {c['n_total']} | "
                f"{c['pct_periods']:.1f}% | {c['pct_error']:.1f}% | "
                f"{c['mae_confused']:.1f} | {c['mae_normal']:.1f} |")

    lines = [
        "# Real vs Proxy Imbalance Price -- Comparison EDA\n",
        f"Real:  `{REAL_VECTOR}`  \n",
        f"Proxy: `{PROXY_VECTOR}`  \n",
        f"Regime from: `{IMB_VECTOR}` (split at imb = 0)  \n",
        f"Full window: `{start.date()}` -> `{end.date()}`  \n",
        f"April detail: `{APRIL_START.date()}` -> `{end.date()}`  \n",
        "\n## Comparison metrics\n",
        "| Regime | N | Corr | MAE | RMSE | Bias (proxy-real) | Med real | Med proxy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *[row(r, stats[r]) for r in ["all", "surplus", "deficit"]],
        "\n## Regime confusion (wrong-direction spread periods)\n",
        "Surplus: confused = spread > 0 (proxy above real, acting like deficit pricing)  \n",
        "Deficit: confused = spread < 0 (proxy below real, acting like surplus pricing)  \n",
        "| Regime | Confused / Total | % periods | % of total abs error | MAE confused | MAE normal |",
        "| --- | --- | --- | --- | --- | --- |",
        *[crow(r, confusion[r]) for r in ["surplus", "deficit"]],
        "\n## Plots\n",
        "- `01_timeseries_2026.png` — daily medians of real vs proxy per regime",
        "- `02_april_detail.png` — 15-min scatter real vs proxy per regime, April",
        "- `03_scatter.png` — proxy vs real scatter (overall + per regime with r and MAE)",
        "- `04_distributions.png` — overlapping histograms real vs proxy per regime",
        "- `05_spread.png` — (proxy-real) spread: daily timeseries + histogram per regime",
        "- `06_hourly.png` — median real vs proxy by hour-of-day per regime",
        "- `07_spread_vs_imbalance.png` — spread vs |imbalance| size (April)",
        "- `08_cumulative_step.png` — cumulative (proxy-real) error over time per regime",
        "- `09_regime_confusion.png` — wrong-direction spread periods per regime\n",
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

    df     = load(client, FULL_START, end)
    df_apr = df[df.index >= APRIL_START]

    n_real  = df["real"].notna().sum()
    n_proxy = df["proxy"].notna().sum()
    print(f"[*] Real points: {n_real}  |  Proxy points: {n_proxy}")

    if n_real == 0:
        print("[!] No real price data -- check vector name and tunnel.")
        return

    df.to_csv(DATA_DIR / "full_2026.csv")
    df_apr.to_csv(DATA_DIR / "april_2026.csv")
    print("[+] Raw data saved to data/")

    ylim = clip_bounds(df, "price")
    print(f"[*] Plot y-axis clip: {ylim[0]:.1f} .. {ylim[1]:.1f} EUR/MWh  (p1-p99)")

    plot_timeseries(df, ylim)
    plot_april_detail(df_apr, ylim)
    plot_scatter(df, ylim)
    plot_distributions(df)
    plot_spread(df)
    plot_hourly(df)
    plot_spread_vs_imbalance(df_apr)
    plot_3day_step(df_apr)
    plot_cumulative_step(df)
    confusion = plot_regime_confusion(df)

    # ACE signal confusion
    ace_15 = load_ace(client, FULL_START, end)
    adf    = build_ace_df(df, ace_15)
    adf.to_csv(DATA_DIR / "ace_confusion.csv")
    ace_stats = analyse_ace_confusion(adf)
    plot_ace_confusion(adf, ace_stats)

    stats = compute_stats(df)
    write_summary(stats, confusion, FULL_START, end)

    print("\n--- Comparison metrics ---")
    for regime, s in stats.items():
        print(f"  {regime:8s}  n={s['n']}  r={s['corr']:.3f}  "
              f"MAE={s['mae']:.1f}  bias={s['bias']:+.1f}")

    print("\n--- ACE confusion ---")
    for regime, s in ace_stats.items():
        mae_c = f"{s['mae_conf']:.1f}" if not np.isnan(s['mae_conf']) else "n/a"
        print(f"  {regime:8s}  confused={s['n_confused']}/{s['n_total']} "
              f"({s['pct_prd']:.1f}% periods, {s['pct_err']:.1f}% abs error)  "
              f"MAE aligned={s['mae_aln']:.1f}  MAE confused={mae_c}")
    print("--- Done ---")


if __name__ == "__main__":
    main()
