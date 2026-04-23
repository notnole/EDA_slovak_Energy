"""Quick EDA across the most relevant vectord vectors.

For each vector, over the past 30 days:
  - point count, time coverage, observed cadence
  - value distribution (min, p05, median, p95, max, mean, std)
  - nan / zero fractions
  - simple time-series plot

Writes:
  - eda/summary.md with a compact table
  - eda/plots/<vector>.png per vector
  - eda/data/vector_stats.csv

Requires SSH tunnel to vectord.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vectord import VectordClient

OUT_DIR = Path(__file__).resolve().parent
PLOT_DIR = OUT_DIR / "plots"
DATA_DIR = OUT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class Target:
    name: str
    unit: str
    group: str
    note: str = ""


TARGETS = [
    # Imbalance
    Target("Okte.Odchylka", "MWh", "imbalance", "System imbalance actual, 15-min"),
    Target("F.B.Odchylka", "MWh", "imbalance", "BEAM imbalance prediction"),
    # Prices
    Target("Okte.MargCena", "EUR/MWh", "price", "DA clearing price, hourly"),
    Target("SK.F.M.Merged.Spot", "EUR/MWh", "price", "DA spot forecast, 15-min"),
    Target("PICASSO.MarginalPricess.SEPS_POS.Weighted", "EUR/MWh", "price",
           "Positive imbalance price (weighted)"),
    Target("PICASSO.MarginalPricess.SEPS_NEG.Weighted", "EUR/MWh", "price",
           "Negative imbalance price (weighted)"),
    # Load / generation
    Target("Sk.Final.Cons.SEPS", "MW", "load", "SK actual load (ENTSO-E), hourly"),
    Target("SK.F.Cons.M.1", "MW", "load", "SK consumption forecast (GFS)"),
    Target("Sk.A.Solar", "MW", "gen", "Actual solar"),
    Target("SK.A.Nuclear", "MW", "gen", "Actual nuclear (baseload)"),
    Target("Sk.A.HydroPump", "MW", "gen", "Pumped hydro generation"),
    # Weather
    Target("Sk.T.Actual60", "C", "weather", "Actual temperature, hourly"),
]


def main():
    client = VectordClient()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=30)
    print(f"[*] Window {start.isoformat()} -> {end.isoformat()}")

    rows = []
    for t in TARGETS:
        print(f"\n[*] {t.name}")
        try:
            df = client.read_df(t.name, start, end)
        except Exception as e:
            print(f"    [!] fetch failed: {e}")
            rows.append({"vector": t.name, "group": t.group, "status": f"error: {e}"})
            continue

        if df.empty:
            print("    [!] empty")
            rows.append({"vector": t.name, "group": t.group, "status": "empty"})
            continue

        stats = _describe(df["value"])
        cadence = _cadence(df.index)
        coverage_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
        expected = int(round(coverage_days * 86400 / cadence["median_s"])) if cadence["median_s"] else 0
        row = {
            "vector": t.name,
            "group": t.group,
            "unit": t.unit,
            "note": t.note,
            "status": "ok",
            "n": len(df),
            "first": df.index[0].isoformat(),
            "last": df.index[-1].isoformat(),
            "cadence_median_s": cadence["median_s"],
            "cadence_label": cadence["label"],
            "gap_regularity_pct": cadence["regularity_pct"],
            "coverage_days": round(coverage_days, 2),
            "expected_points": expected,
            "completeness_pct": round(100 * len(df) / expected, 1) if expected else None,
            **stats,
        }
        rows.append(row)
        print(f"    n={len(df)}  cadence={cadence['label']}  "
              f"range=[{stats['min']:.2f}, {stats['max']:.2f}]  "
              f"mean={stats['mean']:.2f}")

        _plot(df, t, PLOT_DIR / f"{_safe(t.name)}.png")

    out_csv = DATA_DIR / "vector_stats.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print(f"\n[+] Wrote {out_csv}")

    _write_summary(OUT_DIR / "summary.md", df_out, start, end)
    print(f"[+] Wrote {OUT_DIR / 'summary.md'}")


def _describe(s: pd.Series) -> dict:
    s = s.dropna()
    return {
        "min": float(s.min()),
        "p05": float(s.quantile(0.05)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
        "std": float(s.std()),
        "zero_frac": float((s == 0).mean()),
    }


def _cadence(idx: pd.DatetimeIndex) -> dict:
    diffs = idx.to_series().diff().dropna()
    if diffs.empty:
        return {"median_s": None, "label": "n/a", "regularity_pct": None}
    med = diffs.median()
    med_s = med.total_seconds()
    same = (diffs == med).sum()
    reg = round(100 * same / len(diffs), 1)
    label = _label_cadence(med_s)
    return {"median_s": int(med_s), "label": label, "regularity_pct": reg}


def _label_cadence(s: float) -> str:
    if s is None:
        return "n/a"
    if s < 180 + 30:
        return "3-min"
    if s < 900 + 60:
        return "15-min"
    if s < 3600 + 60:
        return "hourly"
    if s < 86400 + 60:
        return "daily"
    return f"{int(s)}s"


def _plot(df: pd.DataFrame, t: Target, path: Path):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(df.index, df["value"], lw=0.8)
    ax.set_title(f"{t.name}  ({t.note})")
    ax.set_ylabel(t.unit)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _safe(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:g}")
            else:
                vals.append("" if pd.isna(v) else str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows])


def _write_summary(path: Path, df: pd.DataFrame, start, end):
    ok = df[df["status"] == "ok"].copy()
    bad = df[df["status"] != "ok"]

    lines = []
    lines.append("# Vectord EDA — 30 day snapshot\n")
    lines.append(f"Window: `{start.isoformat()}` -> `{end.isoformat()}`\n")
    lines.append(f"Vectors probed: {len(df)}  ({len(ok)} ok, {len(bad)} empty/error)\n")

    for group in ["imbalance", "price", "load", "gen", "weather"]:
        g = ok[ok["group"] == group]
        if g.empty:
            continue
        lines.append(f"\n## {group.capitalize()}\n")
        cols = ["vector", "cadence_label", "n", "completeness_pct",
                "min", "median", "mean", "p95", "max", "unit"]
        lines.append(_md_table(g[cols].round(2)))
        lines.append("")

    if not bad.empty:
        lines.append("\n## Failed / empty\n")
        lines.append(_md_table(bad[["vector", "group", "status"]]))

    lines.append("\n## Plots\n")
    lines.append("Per-vector time-series plots are in `plots/`.\n")

    lines.append("\n## Observations\n")
    lines.extend(_observations(ok))

    path.write_text("\n".join(lines), encoding="utf-8")


def _observations(ok: pd.DataFrame) -> list[str]:
    out = []
    # Cadence surprises (anything claimed 15-min but hourly)
    hourly = ok[ok["cadence_label"] == "hourly"]["vector"].tolist()
    if hourly:
        out.append(f"- Hourly cadence observed for: {', '.join(hourly)}")
    # Completeness flags
    low = ok[ok["completeness_pct"].fillna(100) < 90]
    if not low.empty:
        items = ", ".join(f"{r.vector} ({r.completeness_pct:.0f}%)" for r in low.itertuples())
        out.append(f"- Sub-90% completeness: {items}")
    # Regularity flags
    irr = ok[ok["gap_regularity_pct"].fillna(100) < 95]
    if not irr.empty:
        items = ", ".join(f"{r.vector} ({r.gap_regularity_pct:.0f}%)" for r in irr.itertuples())
        out.append(f"- Irregular spacing (<95% same-gap): {items}")
    # Stale / sparse: fewer than 10 points in 30 days
    sparse = ok[ok["n"] < 10]
    if not sparse.empty:
        items = ", ".join(f"{r.vector} (n={r.n}, last {r.last})" for r in sparse.itertuples())
        out.append(f"- **Stale / sparse (n<10):** {items}")
    if not out:
        out.append("- No cadence, completeness, or regularity anomalies detected.")
    return out


if __name__ == "__main__":
    main()
