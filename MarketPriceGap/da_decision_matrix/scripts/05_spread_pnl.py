"""
05_spread_pnl.py - Hourly P&L backtest for spread direction strategy.

Strategy: Buy 1 MWh each hour on the cheaper market based on daily rule.
  - Predict DA > IDM (weekend, low-load weekday) -> buy on IDM
  - Predict DA < IDM (high-load weekday)          -> buy on DA

P&L = saving vs always buying on DA.
  - When buying IDM:  saving = DA_price - IDM_price = spread
  - When buying DA:   saving = 0 (same as benchmark)

Input:  da_master_hourly.csv, da_daily_features.csv
Output: plots/06_spread_pnl.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "MarketPriceGap" / "da_decision_matrix" / "data"
PLOT_DIR = ROOT / "MarketPriceGap" / "da_decision_matrix" / "plots"


def load_data():
    print("[*] Loading hourly master and daily features...")
    hourly = pd.read_csv(DATA_DIR / "da_master_hourly.csv", parse_dates=[0], index_col=0)
    daily = pd.read_csv(DATA_DIR / "da_daily_features.csv", parse_dates=[0], index_col=0)
    print(f"[+]   Hourly: {len(hourly)} rows")
    print(f"[+]   Daily:  {len(daily)} rows")
    return hourly, daily


def apply_spread_rule(daily):
    """Apply the spread direction rule from 04_decision_rules.py."""
    required = ["is_weekend", "load_fcst_vs_7d", "spread_yesterday"]
    valid = daily[required].dropna()

    load_dev_median = valid["load_fcst_vs_7d"].median()
    print(f"[*] Load deviation median: {load_dev_median:.1f} MW")

    signals = {}  # date -> +1 (buy IDM) or -1 (buy DA)
    reasons = {}  # date -> rule label

    for date, row in valid.iterrows():
        d = date.date() if hasattr(date, 'date') else date
        if row["is_weekend"] == 1:
            signals[d] = +1  # predict DA > IDM -> buy IDM
            reasons[d] = "Weekend"
        elif row["load_fcst_vs_7d"] > load_dev_median:
            signals[d] = -1  # predict DA < IDM -> buy DA
            reasons[d] = "WD + High Load"
        else:
            if row["spread_yesterday"] > 0:
                signals[d] = +1
                reasons[d] = "WD + Low Load (spread+)"
            else:
                signals[d] = -1
                reasons[d] = "WD + Low Load (spread-)"

    print(f"[+]   Signals for {len(signals)} days")
    return signals, reasons


def compute_hourly_pnl(hourly, signals):
    """Compute hour-by-hour P&L for 1 MWh traded each hour."""
    print("[*] Computing hourly P&L...")

    # Filter to hours with both DA and IDM prices
    mask = hourly["da_price"].notna() & hourly["idm_vwap"].notna()
    h = hourly.loc[mask, ["da_price", "idm_vwap", "spread_da_idm"]].copy()
    h["date"] = h.index.date

    # Map daily signal to each hour
    h["signal"] = h["date"].map(signals)
    h = h.dropna(subset=["signal"])
    h["signal"] = h["signal"].astype(int)
    print(f"[+]   {len(h)} hours with signal")

    # --- Strategy costs ---
    # Buy 1 MWh each hour:
    #   signal = +1 (buy IDM): cost = idm_vwap
    #   signal = -1 (buy DA):  cost = da_price
    h["cost_strategy"] = np.where(h["signal"] == 1, h["idm_vwap"], h["da_price"])

    # Benchmark costs (1 MWh each hour)
    h["cost_always_da"] = h["da_price"]
    h["cost_always_idm"] = h["idm_vwap"]
    h["cost_perfect"] = np.minimum(h["da_price"], h["idm_vwap"])

    # P&L vs always-DA (positive = we saved money)
    h["pnl_vs_da"] = h["cost_always_da"] - h["cost_strategy"]
    # P&L of perfect strategy vs always-DA
    h["pnl_perfect_vs_da"] = h["cost_always_da"] - h["cost_perfect"]
    # P&L of always-IDM vs always-DA
    h["pnl_idm_vs_da"] = h["cost_always_da"] - h["cost_always_idm"]

    # Cumulative P&L (EUR, for 1 MWh/hour)
    h["cum_pnl_strategy"] = h["pnl_vs_da"].cumsum()
    h["cum_pnl_perfect"] = h["pnl_perfect_vs_da"].cumsum()
    h["cum_pnl_always_idm"] = h["pnl_idm_vs_da"].cumsum()

    return h


def print_stats(h):
    """Print P&L summary statistics."""
    n_hours = len(h)
    n_days = h["date"].nunique()

    print("\n" + "=" * 60)
    print("P&L SUMMARY (1 MWh per hour)")
    print("=" * 60)

    total_da = h["cost_always_da"].sum()
    total_idm = h["cost_always_idm"].sum()
    total_strategy = h["cost_strategy"].sum()
    total_perfect = h["cost_perfect"].sum()

    pnl_strategy = total_da - total_strategy
    pnl_always_idm = total_da - total_idm
    pnl_perfect = total_da - total_perfect

    print(f"\n  Period: {h.index.min().date()} to {h.index.max().date()}")
    print(f"  Hours traded: {n_hours} ({n_days} days)")
    print(f"  Volume: {n_hours} MWh total")

    print(f"\n  --- Total cost (EUR) ---")
    print(f"  Always buy DA:      {total_da:>12,.0f} EUR")
    print(f"  Always buy IDM:     {total_idm:>12,.0f} EUR  (P&L vs DA: {pnl_always_idm:>+10,.0f})")
    print(f"  Our strategy:       {total_strategy:>12,.0f} EUR  (P&L vs DA: {pnl_strategy:>+10,.0f})")
    print(f"  Perfect foresight:  {total_perfect:>12,.0f} EUR  (P&L vs DA: {pnl_perfect:>+10,.0f})")

    print(f"\n  --- Per MWh (EUR/MWh) ---")
    print(f"  Avg DA price:       {total_da / n_hours:>8.2f}")
    print(f"  Avg IDM price:      {total_idm / n_hours:>8.2f}")
    print(f"  Avg strategy cost:  {total_strategy / n_hours:>8.2f}")
    print(f"  Strategy saving:    {pnl_strategy / n_hours:>+8.2f} EUR/MWh")
    print(f"  Perfect saving:     {pnl_perfect / n_hours:>+8.2f} EUR/MWh")

    # Capture ratio: how much of perfect foresight do we capture?
    if pnl_perfect > 0:
        capture = pnl_strategy / pnl_perfect
        print(f"\n  Capture ratio: {capture:.1%} of perfect foresight P&L")

    # Win rate at hourly level
    h_win = (h["pnl_vs_da"] > 0).sum()
    h_lose = (h["pnl_vs_da"] < 0).sum()
    h_flat = (h["pnl_vs_da"] == 0).sum()
    print(f"\n  --- Hourly win/loss ---")
    print(f"  Win (saved money):  {h_win:>5d} hours ({100*h_win/n_hours:.1f}%)")
    print(f"  Lose (overpaid):    {h_lose:>5d} hours ({100*h_lose/n_hours:.1f}%)")
    print(f"  Flat (bought DA):   {h_flat:>5d} hours ({100*h_flat/n_hours:.1f}%)")

    # Average win/loss size
    wins = h.loc[h["pnl_vs_da"] > 0, "pnl_vs_da"]
    losses = h.loc[h["pnl_vs_da"] < 0, "pnl_vs_da"]
    if len(wins) > 0:
        print(f"  Avg win:   {wins.mean():>+8.2f} EUR/MWh")
    if len(losses) > 0:
        print(f"  Avg loss:  {losses.mean():>+8.2f} EUR/MWh")

    # Daily P&L stats
    daily_pnl = h.groupby("date")["pnl_vs_da"].sum()
    print(f"\n  --- Daily P&L ---")
    print(f"  Avg daily P&L:      {daily_pnl.mean():>+8.1f} EUR/day")
    print(f"  Std daily P&L:       {daily_pnl.std():>8.1f} EUR/day")
    print(f"  Best day:           {daily_pnl.max():>+8.1f} EUR")
    print(f"  Worst day:          {daily_pnl.min():>+8.1f} EUR")
    print(f"  Positive days:      {(daily_pnl > 0).sum()}/{len(daily_pnl)} ({100*(daily_pnl > 0).mean():.0f}%)")
    print(f"  Sharpe (daily):      {daily_pnl.mean() / daily_pnl.std():.2f}")

    # Monthly breakdown
    h_monthly = h.copy()
    h_monthly["month"] = h_monthly.index.to_period("M")
    monthly = h_monthly.groupby("month").agg(
        hours=("pnl_vs_da", "count"),
        pnl_total=("pnl_vs_da", "sum"),
        pnl_mean=("pnl_vs_da", "mean"),
    )
    print(f"\n  --- Monthly Breakdown ---")
    print(f"  {'Month':>10s}  {'Hours':>6s}  {'Total P&L':>10s}  {'Per MWh':>8s}")
    for idx, row in monthly.iterrows():
        print(f"  {str(idx):>10s}  {int(row['hours']):>6d}  {row['pnl_total']:>+10.0f}  {row['pnl_mean']:>+8.2f}")

    return daily_pnl


def plot_pnl(h, daily_pnl, signals, reasons):
    """Plot P&L analysis."""
    print("\n[*] Creating P&L plot...")

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))

    # 1a: Cumulative P&L over time
    ax = axes[0, 0]
    ax.plot(h.index, h["cum_pnl_strategy"], "b-", linewidth=1.5, label="Our strategy")
    ax.plot(h.index, h["cum_pnl_perfect"], "g--", linewidth=1, alpha=0.7, label="Perfect foresight")
    ax.plot(h.index, h["cum_pnl_always_idm"], "r:", linewidth=1, alpha=0.7, label="Always buy IDM")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative P&L vs Always-DA (EUR)")
    ax.set_title("Cumulative P&L (1 MWh/hour)", fontsize=11)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.grid(alpha=0.3)

    # 1b: Daily P&L bars
    ax = axes[0, 1]
    dates = pd.to_datetime(daily_pnl.index)
    colors = ["green" if p > 0 else "red" for p in daily_pnl.values]
    ax.bar(dates, daily_pnl.values, color=colors, alpha=0.7, width=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(daily_pnl.mean(), color="blue", linewidth=1, linestyle="--",
               label=f"Mean: {daily_pnl.mean():+.0f} EUR/day")
    ax.set_ylabel("Daily P&L (EUR)")
    pos_pct = 100 * (daily_pnl > 0).mean()
    ax.set_title(f"Daily P&L ({pos_pct:.0f}% positive days)", fontsize=11)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.grid(alpha=0.3)

    # 2a: P&L by signal type
    ax = axes[1, 0]
    h_copy = h.copy()
    h_copy["reason"] = h_copy["date"].map(reasons)
    reason_pnl = h_copy.groupby("reason")["pnl_vs_da"].agg(["sum", "mean", "count"])
    reason_pnl = reason_pnl.sort_values("sum", ascending=True)

    bars = ax.barh(range(len(reason_pnl)), reason_pnl["sum"],
                   color=["green" if s > 0 else "red" for s in reason_pnl["sum"]], alpha=0.7)
    ax.set_yticks(range(len(reason_pnl)))
    ax.set_yticklabels(reason_pnl.index, fontsize=9)
    ax.set_xlabel("Total P&L (EUR)")
    ax.set_title("P&L by Rule Condition", fontsize=11)
    ax.axvline(0, color="black", linewidth=0.5)
    for i, (total, mean, cnt) in enumerate(reason_pnl.values):
        ax.text(total + (50 if total >= 0 else -50), i,
                f"{total:+,.0f} ({int(cnt)}h, {mean:+.1f}/MWh)",
                va="center", ha="left" if total >= 0 else "right", fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    # 2b: P&L by hour of day
    ax = axes[1, 1]
    h_copy["hour"] = h_copy.index.hour
    hourly_pnl = h_copy.groupby("hour")["pnl_vs_da"].agg(["sum", "mean"])
    colors_h = ["green" if s > 0 else "red" for s in hourly_pnl["sum"]]
    ax.bar(hourly_pnl.index, hourly_pnl["sum"], color=colors_h, alpha=0.7)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Total P&L (EUR)")
    ax.set_title("P&L by Hour of Day", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)

    # 3a: P&L by day of week
    ax = axes[2, 0]
    h_copy["dow"] = h_copy.index.dayofweek
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_pnl = h_copy.groupby("dow")["pnl_vs_da"].agg(["sum", "mean", "count"])
    colors_d = ["green" if s > 0 else "red" for s in dow_pnl["sum"]]
    ax.bar(range(7), dow_pnl["sum"], color=colors_d, alpha=0.7)
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)
    ax.set_ylabel("Total P&L (EUR)")
    ax.set_title("P&L by Day of Week", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5)
    for i, (total, mean, cnt) in enumerate(dow_pnl.values):
        ax.text(i, total + (total/abs(total) * 100 if total != 0 else 50),
                f"{mean:+.1f}/MWh", ha="center", fontsize=8, fontweight="bold")
    ax.grid(alpha=0.3)

    # 3b: Summary stats box
    ax = axes[2, 1]
    ax.axis("off")

    total_pnl = h["pnl_vs_da"].sum()
    perfect_pnl = h["pnl_perfect_vs_da"].sum()
    n_hours = len(h)
    n_days = h["date"].nunique()
    capture = total_pnl / perfect_pnl if perfect_pnl > 0 else 0
    sharpe = daily_pnl.mean() / daily_pnl.std() if daily_pnl.std() > 0 else 0

    summary_text = (
        f"STRATEGY P&L SUMMARY\n"
        f"{'='*40}\n\n"
        f"Period:          {h.index.min().strftime('%Y-%m-%d')} to {h.index.max().strftime('%Y-%m-%d')}\n"
        f"Hours traded:    {n_hours:,d} ({n_days} days)\n"
        f"Volume:          {n_hours:,d} MWh\n\n"
        f"Total P&L:       {total_pnl:>+10,.0f} EUR\n"
        f"Per MWh:         {total_pnl/n_hours:>+10.2f} EUR/MWh\n"
        f"Per day:         {daily_pnl.mean():>+10.1f} EUR/day\n\n"
        f"Perfect P&L:     {perfect_pnl:>+10,.0f} EUR\n"
        f"Capture ratio:   {capture:>10.1%}\n\n"
        f"Positive days:   {(daily_pnl > 0).sum()}/{n_days} ({100*(daily_pnl>0).mean():.0f}%)\n"
        f"Daily Sharpe:    {sharpe:>10.2f}\n"
        f"Best day:        {daily_pnl.max():>+10.0f} EUR\n"
        f"Worst day:       {daily_pnl.min():>+10.0f} EUR\n"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontfamily="monospace", fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("DA-IDM Spread Strategy P&L Backtest\n(Buy 1 MWh/hour on predicted cheaper market)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_spread_pnl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+]   Saved 06_spread_pnl.png")


def main():
    print("=" * 70)
    print("DA Decision Matrix - Step 5: Spread Strategy P&L Backtest")
    print("=" * 70)

    hourly, daily = load_data()
    signals, reasons = apply_spread_rule(daily)
    h = compute_hourly_pnl(hourly, signals)
    daily_pnl = print_stats(h)
    plot_pnl(h, daily_pnl, signals, reasons)


if __name__ == "__main__":
    main()
