"""Compare Okte.Odchylka (initial) vs Okte.Combine.Odchylka (settled)
over the past 365 days.

Questions answered:
  - How much do they differ? (MAE, bias, % of periods with revision)
  - Does the sign of imbalance flip between initial and settled?
  - Where are the worst revisions?

Requires SSH tunnel to vectord.
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

OUT = Path(__file__).resolve().parent


def main():
    client = VectordClient()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=365)
    print(f"[*] Window: {start.isoformat()} -> {end.isoformat()}")

    print("[*] Fetching Okte.Odchylka (initial)...")
    init = client.read_df("Okte.Odchylka", start, end).rename(columns={"value": "initial"})
    print(f"    {len(init)} points")

    print("[*] Fetching Okte.Combine.Odchylka (settled-combine)...")
    comb = client.read_df("Okte.Combine.Odchylka", start, end).rename(columns={"value": "settled"})
    print(f"    {len(comb)} points")

    if init.empty or comb.empty:
        print("[!] one series is empty")
        return

    df = pd.concat([init["initial"], comb["settled"]], axis=1).dropna()
    df["diff"] = df["settled"] - df["initial"]
    df["revised"] = df["diff"].abs() > 0.01
    df["sign_flip"] = np.sign(df["initial"]) != np.sign(df["settled"])
    df["sign_flip"] &= (df["initial"].abs() > 1) & (df["settled"].abs() > 1)

    print(f"[*] Aligned {len(df)} joint 15-min periods")

    # --- Overall metrics ---
    mae = df["diff"].abs().mean()
    rmse = np.sqrt((df["diff"] ** 2).mean())
    bias = df["diff"].mean()
    p_revised = df["revised"].mean() * 100
    p_flip = df["sign_flip"].mean() * 100
    max_abs = df["diff"].abs().max()

    # Compared to signal scale
    sd_settled = df["settled"].std()
    mae_settled_mae = df["settled"].abs().mean()

    print("\n--- Initial vs Combined (settled) ---")
    print(f"N periods       : {len(df)}")
    print(f"MAE (revision)  : {mae:.3f} MWh")
    print(f"RMSE            : {rmse:.3f} MWh")
    print(f"Bias (S - I)    : {bias:+.3f} MWh")
    print(f"Max abs diff    : {max_abs:.2f} MWh")
    print(f"% revised       : {p_revised:.1f} %")
    print(f"% sign flips    : {p_flip:.2f} %  (when |either| > 1)")
    print(f"Signal MAE      : {mae_settled_mae:.2f} MWh (mean |settled|)")
    print(f"Revision/signal : {mae / mae_settled_mae * 100:.1f} %")

    # --- Monthly breakdown ---
    df["month"] = df.index.to_period("M")
    monthly = df.groupby("month").agg(
        n=("diff", "size"),
        mae=("diff", lambda s: s.abs().mean()),
        bias=("diff", "mean"),
        pct_revised=("revised", lambda s: s.mean() * 100),
        pct_flip=("sign_flip", lambda s: s.mean() * 100),
        max_abs=("diff", lambda s: s.abs().max()),
    ).round(3)
    print("\n--- Monthly ---")
    print(monthly.to_string())

    monthly.to_csv(OUT / "monthly_revisions.csv")
    df.drop(columns=["month"]).to_csv(OUT / "aligned.csv")

    # --- Plots ---
    _plot_timeseries(df, OUT / "01_timeseries_revisions.png")
    _plot_diff_hist(df, OUT / "02_diff_histogram.png")
    _plot_scatter(df, OUT / "03_scatter.png")
    _plot_monthly(monthly, OUT / "04_monthly.png")
    _plot_worst(df, OUT / "05_worst_revisions.png")

    # --- Summary ---
    _write_summary(OUT / "summary.md", df, monthly, mae, rmse, bias, p_revised,
                   p_flip, max_abs, mae_settled_mae, start, end)
    print(f"[+] Wrote {OUT / 'summary.md'}")


def _plot_timeseries(df, path):
    sample = df.resample("1D")[["initial", "settled"]].mean()
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(sample.index, sample["initial"], label="Initial (Okte.Odchylka)", lw=0.9, alpha=0.8)
    ax.plot(sample.index, sample["settled"], label="Settled (Okte.Combine.Odchylka)",
            lw=0.9, alpha=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Daily-averaged imbalance: initial vs settled (365 days)")
    ax.set_ylabel("MWh (per 15-min period, daily avg)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[+] {path.name}")


def _plot_diff_hist(df, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    clipped = df["diff"].clip(-20, 20)
    ax.hist(clipped, bins=80, color="steelblue", edgecolor="white")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Settled - Initial (MWh), clipped to +/-20")
    ax.set_ylabel("15-min periods")
    ax.set_title("Distribution of revisions (settled - initial)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[+] {path.name}")


def _plot_scatter(df, path):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(df["initial"], df["settled"], s=2, alpha=0.15)
    lim = max(df["initial"].abs().max(), df["settled"].abs().max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Initial (MWh)")
    ax.set_ylabel("Settled (MWh)")
    ax.set_title("Settled vs Initial imbalance")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[+] {path.name}")


def _plot_monthly(monthly, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    x = monthly.index.astype(str)
    axes[0].bar(x, monthly["mae"], color="steelblue", label="MAE")
    axes[0].plot(x, monthly["bias"], "o-", color="firebrick", label="Bias")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("MWh")
    axes[0].set_title("Monthly revision MAE and bias (settled - initial)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].bar(x, monthly["pct_revised"], color="darkgreen", alpha=0.6, label="% revised")
    axes[1].plot(x, monthly["pct_flip"], "o-", color="orange", label="% sign flips")
    axes[1].set_ylabel("% of 15-min periods")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    for ax in axes:
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[+] {path.name}")


def _plot_worst(df, path):
    worst = df.reindex(df["diff"].abs().nlargest(10).index).copy()
    worst = worst[["initial", "settled", "diff"]].round(2)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table = ax.table(
        cellText=worst.reset_index().values,
        colLabels=["time"] + list(worst.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("10 largest revisions |settled - initial|")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[+] {path.name}")


def _write_summary(path, df, monthly, mae, rmse, bias, p_revised, p_flip,
                   max_abs, mae_settled_mae, start, end):
    worst_rows = df.reindex(df["diff"].abs().nlargest(5).index)
    worst_md = _md_rows(worst_rows[["initial", "settled", "diff"]].round(2))

    lines = [
        "# Okte.Odchylka initial vs Okte.Combine.Odchylka (settled)\n",
        f"Window: `{start.date()}` -> `{end.date()}`   N periods: {len(df)}\n",
        "\n## Overall\n",
        "| Metric | Value |",
        "| --- | --- |",
        f"| MAE (|settled - initial|) | {mae:.3f} MWh |",
        f"| RMSE | {rmse:.3f} MWh |",
        f"| Bias (settled - initial) | {bias:+.3f} MWh |",
        f"| Max abs diff | {max_abs:.2f} MWh |",
        f"| Revised periods | {p_revised:.1f} % |",
        f"| Sign flips (when |either| > 1 MWh) | {p_flip:.2f} % |",
        f"| Mean |settled| (signal scale) | {mae_settled_mae:.2f} MWh |",
        f"| **Revision / signal** | **{mae / mae_settled_mae * 100:.1f} %** |",
        "\n## Monthly\n",
        "| month | n | mae | bias | % revised | % flip | max abs |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m, r in monthly.iterrows():
        lines.append(
            f"| {m} | {int(r.n)} | {r.mae:.3f} | {r.bias:+.3f} | "
            f"{r.pct_revised:.1f} | {r.pct_flip:.2f} | {r.max_abs:.2f} |"
        )

    lines += [
        "\n## 5 largest revisions\n",
        "| time | initial | settled | diff |",
        "| --- | --- | --- | --- |",
        *worst_md,
        "\n## Plots\n",
        "- `01_timeseries_revisions.png` — daily averages",
        "- `02_diff_histogram.png` — revision distribution",
        "- `03_scatter.png` — settled vs initial",
        "- `04_monthly.png` — MAE / bias / revision share by month",
        "- `05_worst_revisions.png` — table of 10 worst\n",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _md_rows(df):
    rows = []
    for idx, r in df.iterrows():
        rows.append(f"| {idx.isoformat()} | {r['initial']:+.2f} | {r['settled']:+.2f} | {r['diff']:+.2f} |")
    return rows


if __name__ == "__main__":
    main()
