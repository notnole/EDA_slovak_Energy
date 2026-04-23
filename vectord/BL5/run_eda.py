"""EDA on BESS block 5 (locality GBAT_BAT_5) using the EDA locality norm.

Pulls five attributes over the past 30 days and produces combined views that
reveal battery behaviour:

  - Pdod        -> actual delivered power (+ discharge / - charge)
  - SoC.15M     -> state of charge, 15-min series (SoC.Actual is a live snapshot)
  - PP_Pdb      -> planned schedule (market dispatch baseline)
  - RBO_RE      -> RBO regulation energy (balancing contribution)
  - Adelta      -> Pdod - PP_Pdb (schedule deviation)

Combined views:
  1. Time-series overlay: Pdod vs PP_Pdb vs SoC
  2. Plan-adherence scatter: PP_Pdb vs Pdod (identity line = perfect tracking)
  3. SoC density + daily trajectories
  4. Adelta histogram + split into RBO_RE and residual
  5. Cumulative energy check: SoC change vs integral(Pdod)

Locality norm: the full vector name is
    EMS#UNT..#<LOCALITY_PADDED_TO_40>#<ATTRIBUTE>
Swap LOCALITY to point the same code at another block (GBAT_BAT_4, ...).

Requires SSH tunnel to vectord.
"""

from __future__ import annotations

import argparse
import re
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

DEFAULT_LOCALITY = "GBAT_BAT_5"
DEFAULT_LOOKBACK_DAYS = 30


def _short_name(locality: str) -> str:
    """GBAT_BAT_5 -> BL5, GBAT_BAT_4 -> BL4, fallback to locality itself."""
    m = re.search(r"_(\d+)$", locality)
    return f"BL{m.group(1)}" if m else locality


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA over one EDA locality (battery block).")
    p.add_argument("--locality", default=DEFAULT_LOCALITY,
                   help=f"Locality code, e.g. GBAT_BAT_5 (default: {DEFAULT_LOCALITY}).")
    p.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                   help=f"Lookback window in days (default: {DEFAULT_LOOKBACK_DAYS}).")
    p.add_argument("--out", default=None,
                   help="Output directory (default: vectord/<BLn> derived from locality).")
    return p.parse_args()


ARGS = _parse_args()
LOCALITY = ARGS.locality
LOOKBACK_DAYS = ARGS.days
VECTORD_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = Path(ARGS.out) if ARGS.out else VECTORD_DIR / _short_name(LOCALITY)
PLOT_DIR = OUT_DIR / "plots"
DATA_DIR = OUT_DIR / "data"
OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def locality_vector(locality: str, attribute: str) -> str:
    """Build a full EDA vector name per the locality norm."""
    padded = locality.ljust(40, ".")
    return f"EMS#UNT..#{padded}#{attribute}"


@dataclass
class Attr:
    code: str
    unit: str
    note: str


ATTRS = [
    Attr("Pdod",    "MW",  "Actual delivered power (+ discharge / - charge)"),
    Attr("SoC.15M", "%",   "State of charge, 15-min series"),
    Attr("PP_Pdb",  "MW",  "Planned Pdb from production plan"),
    Attr("RBO_RE",  "MWh", "RBO regulation energy (balancing)"),
    Attr("Adelta",  "MW",  "Pdod - PP_Pdb (schedule deviation)"),
]
SOC_KEY = "SoC.15M"


def fetch_all(client: VectordClient, start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for a in ATTRS:
        vec = locality_vector(LOCALITY, a.code)
        print(f"[*] reading {a.code:11s}  <- {vec}")
        try:
            df = client.read_df(vec, start, end)
        except Exception as exc:
            print(f"[!] failed {a.code}: {exc}")
            df = pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([], tz="UTC"))
        df = df.rename(columns={"value": a.code})
        data[a.code] = df
        print(f"    {len(df):>6d} points")
    return data


def summary_stats(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for a in ATTRS:
        s = data[a.code][a.code] if a.code in data[a.code].columns else data[a.code].iloc[:, 0]
        if len(s) == 0:
            rows.append({"attr": a.code, "unit": a.unit, "n": 0})
            continue
        rows.append({
            "attr": a.code,
            "unit": a.unit,
            "n": len(s),
            "first": s.index.min().isoformat(),
            "last": s.index.max().isoformat(),
            "min": float(s.min()),
            "p05": float(s.quantile(0.05)),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
            "std": float(s.std()),
            "note": a.note,
        })
    return pd.DataFrame(rows)


def plot_timeseries_overlay(data: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    ax = axes[0]
    pdod = data["Pdod"].iloc[:, 0]
    ppdb = data["PP_Pdb"].iloc[:, 0]
    if len(pdod):
        ax.plot(pdod.index, pdod.values, label="Pdod (actual)", lw=0.7, color="tab:blue")
    if len(ppdb):
        ax.plot(ppdb.index, ppdb.values, label="PP_Pdb (planned)", lw=0.7, color="tab:orange", alpha=0.8)
    ax.set_ylabel("Power (MW)")
    ax.set_title(f"{LOCALITY}: actual vs planned power")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(loc="upper right")

    ax = axes[1]
    ade = data["Adelta"].iloc[:, 0]
    rbo = data["RBO_RE"].iloc[:, 0]
    if len(ade):
        ax.plot(ade.index, ade.values, label="Adelta (Pdod-PP_Pdb)", lw=0.6, color="tab:red")
    if len(rbo):
        ax.plot(rbo.index, rbo.values, label="RBO_RE", lw=0.6, color="tab:green", alpha=0.7)
    ax.set_ylabel("Power/Energy")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(loc="upper right")

    ax = axes[2]
    soc = data[SOC_KEY].iloc[:, 0]
    if len(soc):
        ax.plot(soc.index, soc.values, lw=0.7, color="tab:purple")
    ax.set_ylabel("SoC (%)")
    ax.set_xlabel("time (UTC)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_plan_adherence(data: dict[str, pd.DataFrame], path: Path) -> None:
    pdod = data["Pdod"].iloc[:, 0]
    ppdb = data["PP_Pdb"].iloc[:, 0]
    if len(pdod) == 0 or len(ppdb) == 0:
        return
    joined = pd.concat([ppdb.rename("plan"), pdod.rename("act")], axis=1).dropna()
    if joined.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(joined["plan"], joined["act"], s=3, alpha=0.2)
    lim = max(abs(joined.values).max(), 1)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="perfect tracking")
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.3)
    ax.set_xlabel("PP_Pdb planned (MW)")
    ax.set_ylabel("Pdod actual (MW)")
    ax.set_title(f"{LOCALITY}: plan adherence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_soc_behaviour(data: dict[str, pd.DataFrame], path: Path) -> None:
    soc = data[SOC_KEY].iloc[:, 0]
    if len(soc) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(soc.values, bins=50, color="tab:purple", alpha=0.8)
    axes[0].set_xlabel("SoC (%)")
    axes[0].set_ylabel("count")
    axes[0].set_title("SoC distribution")

    # Daily SoC trajectories
    local = soc.copy()
    local.index = local.index.tz_convert("Europe/Bratislava")
    df = pd.DataFrame({"soc": local.values, "day": local.index.date,
                       "minute": local.index.hour * 60 + local.index.minute})
    for _, grp in df.groupby("day"):
        axes[1].plot(grp["minute"].values, grp["soc"].values, color="tab:purple", alpha=0.15, lw=0.7)
    axes[1].set_xlabel("minute of day (local)")
    axes[1].set_ylabel("SoC (%)")
    axes[1].set_title("daily SoC trajectories")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_adelta_split(data: dict[str, pd.DataFrame], path: Path) -> None:
    ade = data["Adelta"].iloc[:, 0]
    rbo = data["RBO_RE"].iloc[:, 0]
    if len(ade) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(ade.values, bins=60, color="tab:red", alpha=0.8)
    axes[0].set_xlabel("Adelta = Pdod - PP_Pdb")
    axes[0].set_ylabel("count")
    axes[0].set_title("schedule deviation distribution")
    axes[0].axvline(0, color="k", lw=0.5)

    if len(rbo):
        joined = pd.concat([ade.rename("adelta"), rbo.rename("rbo_re")], axis=1).dropna()
        if not joined.empty:
            axes[1].scatter(joined["rbo_re"], joined["adelta"], s=3, alpha=0.25)
            axes[1].axhline(0, color="k", lw=0.3)
            axes[1].axvline(0, color="k", lw=0.3)
            axes[1].set_xlabel("RBO_RE (balancing)")
            axes[1].set_ylabel("Adelta")
            axes[1].set_title("Adelta vs RBO_RE (share of deviation = balancing)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_energy_check(data: dict[str, pd.DataFrame], path: Path) -> None:
    """Cumulative integral of Pdod should track SoC change (up to efficiency)."""
    pdod = data["Pdod"].iloc[:, 0]
    soc = data[SOC_KEY].iloc[:, 0]
    if len(pdod) == 0 or len(soc) == 0:
        return
    # Assume cadence is regular-ish; use per-point dt.
    dt_h = pdod.index.to_series().diff().dt.total_seconds().fillna(0) / 3600.0
    cum_energy_mwh = -(pdod.values * dt_h.values).cumsum()  # discharge drains SoC -> negate
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(pdod.index, cum_energy_mwh, label="cumulative -integral(Pdod) [MWh]", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(soc.index, soc.values - soc.iloc[0], label="SoC change (%)", color="tab:purple", alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("cumulative energy (MWh, + = charged)")
    ax2.set_ylabel("SoC delta (%)")
    ax.set_title(f"{LOCALITY}: energy accounting (slope ratio = capacity)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _df_to_md(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return "" if v is None else str(v)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


def write_summary(stats: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# {_short_name(LOCALITY)} EDA ({LOCALITY})")
    lines.append("")
    lines.append(f"- Lookback: {LOOKBACK_DAYS} days")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Vector coverage")
    lines.append("")
    if len(stats):
        keep = ["attr", "unit", "n", "first", "last", "min", "p05", "median", "mean", "p95", "max", "std"]
        keep = [c for c in keep if c in stats.columns]
        lines.append(_df_to_md(stats[keep]))
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("- `plots/01_timeseries_overlay.png` - Pdod vs PP_Pdb, Adelta vs RBO_RE, SoC")
    lines.append("- `plots/02_plan_adherence.png` - scatter PP_Pdb vs Pdod (identity = perfect)")
    lines.append("- `plots/03_soc_behaviour.png` - SoC histogram + daily trajectories")
    lines.append("- `plots/04_adelta_split.png` - Adelta distribution and share explained by RBO_RE")
    lines.append("- `plots/05_energy_check.png` - cumulative integral(Pdod) vs SoC change")
    lines.append("")
    lines.append("## Next")
    lines.append("")
    lines.append("Re-run for another block with `python run_eda.py --locality GBAT_BAT_4`. "
                 "See `vectord/localities.md` for the naming norm.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    print(f"[*] {_short_name(LOCALITY)} EDA ({LOCALITY}): {start.isoformat()} -> {end.isoformat()}")
    print(f"[*] output: {OUT_DIR}")
    client = VectordClient()
    data = fetch_all(client, start, end)

    stats = summary_stats(data)
    stats.to_csv(DATA_DIR / "vector_stats.csv", index=False)
    print(stats.to_string(index=False))

    # Save raw frames for reuse.
    for code, df in data.items():
        if len(df):
            df.to_csv(DATA_DIR / f"{code.replace('.', '_')}.csv")

    print("[*] plotting")
    plot_timeseries_overlay(data, PLOT_DIR / "01_timeseries_overlay.png")
    plot_plan_adherence(data,    PLOT_DIR / "02_plan_adherence.png")
    plot_soc_behaviour(data,     PLOT_DIR / "03_soc_behaviour.png")
    plot_adelta_split(data,      PLOT_DIR / "04_adelta_split.png")
    plot_energy_check(data,      PLOT_DIR / "05_energy_check.png")

    write_summary(stats, OUT_DIR / "summary.md")
    print(f"[+] done. See {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
