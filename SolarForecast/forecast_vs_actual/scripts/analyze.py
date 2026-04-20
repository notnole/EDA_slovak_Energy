"""Solar forecast vs actual for the past 30 days.

Pulls Sk.A.Solar (actual, 15-min) and Sk.F.Solar (forecast) from vectord,
aligns them, and reports error metrics + plots.

Requires SSH tunnel:
    ssh -L8080:10.100.0.70:8080 noel@greenbat1.vps.wbsprt.com
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from vectord import VectordClient

OUT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = OUT_DIR / "data"


def main():
    client = VectordClient()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=30)

    print(f"[*] Window: {start.isoformat()} -> {end.isoformat()}")

    print("[*] Fetching Sk.A.Solar (actual)...")
    actual = client.read_df("Sk.A.Solar", start, end).rename(columns={"value": "actual"})
    print(f"    {len(actual)} points")

    print("[*] Fetching Sk.F.Solar (forecast)...")
    forecast = client.read_df("Sk.F.Solar", start, end).rename(columns={"value": "forecast"})
    print(f"    {len(forecast)} points")

    if actual.empty or forecast.empty:
        print("[!] One of the series is empty, aborting.")
        return

    # Align on 15-min grid
    grid = pd.date_range(start, end, freq="15min", tz="UTC")
    actual_r = actual["actual"].reindex(grid).interpolate(limit=2)
    forecast_r = forecast["forecast"].reindex(grid).interpolate(limit=2)
    df = pd.concat([actual_r, forecast_r], axis=1).dropna()
    df["error"] = df["forecast"] - df["actual"]

    print(f"[*] Aligned {len(df)} joint 15-min observations")

    # --- Metrics ---
    mae = df["error"].abs().mean()
    rmse = np.sqrt((df["error"] ** 2).mean())
    bias = df["error"].mean()
    corr = df["actual"].corr(df["forecast"])
    mean_actual = df["actual"].mean()
    peak_actual = df["actual"].max()

    # Daytime-only (actual > 50 MW)
    day = df[df["actual"] > 50]
    day_mae = day["error"].abs().mean() if len(day) else np.nan
    day_bias = day["error"].mean() if len(day) else np.nan
    day_mape = (day["error"].abs() / day["actual"]).mean() * 100 if len(day) else np.nan

    print("\n--- Metrics (all hours) ---")
    print(f"MAE       : {mae:.2f} MW")
    print(f"RMSE      : {rmse:.2f} MW")
    print(f"Bias      : {bias:+.2f} MW  (forecast - actual)")
    print(f"Corr      : {corr:.3f}")
    print(f"Mean solar: {mean_actual:.1f} MW  (peak {peak_actual:.0f} MW)")

    print("\n--- Metrics (daytime, actual > 50 MW) ---")
    print(f"MAE       : {day_mae:.2f} MW")
    print(f"Bias      : {day_bias:+.2f} MW")
    print(f"MAPE      : {day_mape:.1f} %")
    print(f"Samples   : {len(day)}")

    # Save aligned data
    df.to_csv(DATA_DIR / "aligned_15min.csv")
    print(f"\n[+] Wrote {DATA_DIR / 'aligned_15min.csv'}")

    # --- Plots ---
    _plot_timeseries(df, OUT_DIR / "01_timeseries.png")
    _plot_scatter(df, OUT_DIR / "02_scatter.png", corr)
    _plot_error_by_hour(df, OUT_DIR / "03_error_by_hour.png")
    _plot_daily_mae(df, OUT_DIR / "04_daily_mae.png")

    # --- Summary ---
    _write_summary(
        OUT_DIR / "summary.md",
        start=start,
        end=end,
        n=len(df),
        mae=mae,
        rmse=rmse,
        bias=bias,
        corr=corr,
        mean_actual=mean_actual,
        peak_actual=peak_actual,
        day_mae=day_mae,
        day_bias=day_bias,
        day_mape=day_mape,
        n_day=len(day),
    )
    print(f"[+] Wrote {OUT_DIR / 'summary.md'}")


def _plot_timeseries(df: pd.DataFrame, path: Path):
    # Show last 7 days for readability
    tail = df[df.index >= df.index.max() - pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(tail.index, tail["actual"], label="Actual (Sk.A.Solar)", lw=1.2)
    ax.plot(tail.index, tail["forecast"], label="Forecast (Sk.F.Solar)", lw=1.2, alpha=0.8)
    ax.set_ylabel("Solar generation (MW)")
    ax.set_title("Solar: forecast vs actual (last 7 days)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[+] Wrote {path}")


def _plot_scatter(df: pd.DataFrame, path: Path, corr: float):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["actual"], df["forecast"], s=4, alpha=0.25)
    lim = max(df["actual"].max(), df["forecast"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("Actual (MW)")
    ax.set_ylabel("Forecast (MW)")
    ax.set_title(f"Forecast vs actual  (r={corr:.3f})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[+] Wrote {path}")


def _plot_error_by_hour(df: pd.DataFrame, path: Path):
    by_hour = df.groupby(df.index.hour).agg(
        mae=("error", lambda s: s.abs().mean()),
        bias=("error", "mean"),
        actual=("actual", "mean"),
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(by_hour.index, by_hour["mae"], "o-", label="MAE")
    ax.plot(by_hour.index, by_hour["bias"], "s-", label="Bias (F-A)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("MW")
    ax.set_title("Forecast error by hour of day")
    ax.legend()
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(by_hour.index, by_hour["actual"], "--", color="orange", alpha=0.6, label="Mean actual")
    ax2.set_ylabel("Mean actual solar (MW)", color="orange")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[+] Wrote {path}")


def _plot_daily_mae(df: pd.DataFrame, path: Path):
    daily = df.resample("1D").agg(
        mae=("error", lambda s: s.abs().mean()),
        bias=("error", "mean"),
        peak=("actual", "max"),
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(daily.index, daily["mae"], width=0.8, label="Daily MAE", color="steelblue")
    ax.plot(daily.index, daily["bias"], "o-", color="firebrick", label="Daily bias")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("MW")
    ax.set_title("Daily MAE and bias over the past 30 days")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[+] Wrote {path}")


def _write_summary(path: Path, **kw):
    md = f"""# Solar forecast vs actual — 30 days

Window: `{kw['start'].date()}` → `{kw['end'].date()}` ({kw['n']} aligned 15-min obs)

Vectors:
- Actual: `Sk.A.Solar`
- Forecast: `Sk.F.Solar`

## Overall metrics

| Metric | Value |
|--------|-------|
| MAE | {kw['mae']:.2f} MW |
| RMSE | {kw['rmse']:.2f} MW |
| Bias (F - A) | {kw['bias']:+.2f} MW |
| Correlation | {kw['corr']:.3f} |
| Mean actual | {kw['mean_actual']:.1f} MW |
| Peak actual | {kw['peak_actual']:.0f} MW |

## Daytime only (actual > 50 MW, n={kw['n_day']})

| Metric | Value |
|--------|-------|
| MAE | {kw['day_mae']:.2f} MW |
| Bias | {kw['day_bias']:+.2f} MW |
| MAPE | {kw['day_mape']:.1f} % |

## Plots

- `01_timeseries.png` — last 7 days, forecast vs actual
- `02_scatter.png` — scatter with y=x reference
- `03_error_by_hour.png` — MAE and bias by hour of day
- `04_daily_mae.png` — daily MAE and bias over the window
"""
    path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
