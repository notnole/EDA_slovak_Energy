"""
Temperature-filtered "Sell DA, Buy Imbalance" strategy with REVERSE on extreme temps.
Backtest: Feb 2024 - Jan 2026 (spread data availability).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data")
OUT_DIR = BASE / "MarketPriceGap" / "strategies" / "temp_filter"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. LOAD MARKET DATA
# =============================================================================
print("[*] Loading market data...")
mkt = pd.read_csv(BASE / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                   parse_dates=["timestamp_hour"])

# Keep only rows with valid spread
mkt = mkt.dropna(subset=["spread_da_imb"])
mkt = mkt[mkt["spread_da_imb"].astype(str).str.strip() != ""]
mkt["spread_da_imb"] = mkt["spread_da_imb"].astype(float)
mkt["da_price"] = mkt["da_price"].astype(float)
mkt["imb_settlement_price"] = mkt["imb_settlement_price"].astype(float)
mkt["date"] = mkt["timestamp_hour"].dt.date
mkt = mkt.set_index("timestamp_hour").sort_index()

print(f"    Market data: {mkt.index.min()} to {mkt.index.max()}, {len(mkt)} hours")


# =============================================================================
# 2. LOAD ACTUAL TEMPERATURE
# =============================================================================
print("[*] Loading actual temperature data...")

def parse_actual_temp(filepath):
    """Parse the European-format actual temperature CSV."""
    rows = []
    with open(filepath, encoding='utf-8-sig') as f:
        lines = f.readlines()

    for line in lines[1:]:  # skip header
        line = line.strip()
        if not line or '(invalid)' in line or line == ',,':
            continue
        parts = line.rstrip(',').split(',')
        # Format: DD/MM/YYYY HH:MM:SS,000,integer_part,decimal_part
        if len(parts) < 4:
            continue
        dt_str = parts[0].strip()
        int_part = parts[2].strip()
        dec_part = parts[3].strip()

        try:
            dt = pd.to_datetime(dt_str, format="%d/%m/%Y %H:%M:%S")
        except Exception:
            continue

        try:
            if int_part.startswith('-'):
                # Negative: e.g. "-0" and "29" -> -0.29, or "-1" and "05617356" -> -1.05617356
                temp = float(int_part + '.' + dec_part)
            else:
                temp = float(int_part + '.' + dec_part)
        except Exception:
            continue

        rows.append({"datetime": dt, "temp_actual": temp})

    df = pd.DataFrame(rows)
    df = df.set_index("datetime").sort_index()
    return df

actual_temp = parse_actual_temp(BASE / "RawData" / "ActualTemp2024-2026.csv")
print(f"    Actual temp: {actual_temp.index.min()} to {actual_temp.index.max()}, {len(actual_temp)} rows")

# Daily mean from actual temperature
daily_temp_actual = actual_temp.resample('D').mean()
daily_temp_actual.columns = ["daily_mean_temp_actual"]
daily_temp_actual.index = daily_temp_actual.index.date
print(f"    Daily actual temp range: {daily_temp_actual['daily_mean_temp_actual'].min():.1f} to {daily_temp_actual['daily_mean_temp_actual'].max():.1f} C")


# =============================================================================
# 3. LOAD GFS FORECAST TEMPERATURE
# =============================================================================
print("[*] Loading GFS forecast temperature...")

def parse_gfs_temp(filepath):
    """Parse the GFS forecast + actual temperature CSV (15-min resolution)."""
    rows = []
    with open(filepath, encoding='utf-8-sig') as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.rstrip(',').split(',')
        # Format: dt1,000,forecast_int,forecast_dec,dt2,000,actual_int,actual_dec
        if len(parts) < 4:
            continue

        dt_str = parts[0].strip()
        fc_int = parts[2].strip()
        fc_dec = parts[3].strip() if len(parts) > 3 else ''

        if '(invalid)' in fc_int or not dt_str:
            continue

        try:
            dt = pd.to_datetime(dt_str, format="%d/%m/%Y %H:%M:%S")
        except Exception:
            continue

        try:
            temp_fc = float(fc_int + '.' + fc_dec)
        except Exception:
            continue

        rows.append({"datetime": dt, "temp_gfs": temp_fc})

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = df.set_index("datetime").sort_index()
    return df

gfs_temp = parse_gfs_temp(BASE / "RawData" / "Tampretures.csv")
print(f"    GFS temp: {gfs_temp.index.min()} to {gfs_temp.index.max()}, {len(gfs_temp)} rows")

# Aggregate 15-min to hourly, then to daily
gfs_hourly = gfs_temp.resample('h').mean()
gfs_daily = gfs_hourly.resample('D').mean()
gfs_daily.columns = ["daily_mean_temp_gfs"]
gfs_daily.index = gfs_daily.index.date
print(f"    GFS daily temp range: {gfs_daily['daily_mean_temp_gfs'].min():.1f} to {gfs_daily['daily_mean_temp_gfs'].max():.1f} C")


# =============================================================================
# 4. STRATEGY FUNCTIONS
# =============================================================================

def compute_strategy(hourly_market, daily_temp, cold_thresh=0.0, hot_thresh=25.0,
                     mode="baseline"):
    """
    Compute strategy P&L.

    Modes:
        baseline: always position = +1
        skip_extreme: position = 0 on extreme days
        reverse_extreme: position = -1 on extreme days
        reverse_cold_only: reverse cold, skip hot
        reverse_hot_only: skip cold, reverse hot

    P&L per hour = position * spread_da_imb
    spread_da_imb = da_price - imb_settlement_price
    position +1 means "sell DA, buy imbalance" -> profit when DA > imbalance
    position -1 means reverse -> profit when imbalance > DA
    """
    df = hourly_market.copy()
    df["day"] = df.index.date

    # Map daily temp
    df["daily_temp"] = df["day"].map(
        daily_temp.iloc[:, 0].to_dict() if hasattr(daily_temp, 'iloc') else daily_temp
    )

    # Classify days
    df["is_cold"] = df["daily_temp"] < cold_thresh
    df["is_hot"] = df["daily_temp"] > hot_thresh
    df["is_extreme"] = df["is_cold"] | df["is_hot"]

    # Assign positions
    if mode == "baseline":
        df["position"] = 1
    elif mode == "skip_extreme":
        df["position"] = np.where(df["is_extreme"], 0, 1)
    elif mode == "reverse_extreme":
        df["position"] = np.where(df["is_cold"], -1,
                         np.where(df["is_hot"], -1, 1))
    elif mode == "reverse_cold_only":
        df["position"] = np.where(df["is_cold"], -1,
                         np.where(df["is_hot"], 0, 1))
    elif mode == "reverse_hot_only":
        df["position"] = np.where(df["is_hot"], -1,
                         np.where(df["is_cold"], 0, 1))
    else:
        df["position"] = 1

    # P&L
    df["hourly_pnl"] = df["position"] * df["spread_da_imb"]

    # Drop rows where temp is missing
    df = df.dropna(subset=["daily_temp"])

    return df


def compute_metrics(df):
    """Compute summary metrics from hourly strategy df."""
    daily_pnl = df.groupby("day")["hourly_pnl"].sum()
    total_pnl = daily_pnl.sum()

    # Sharpe (annualized)
    if daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    cum = daily_pnl.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = drawdown.min()

    # Day counts
    daily_pos = df.groupby("day")["position"].first()
    n_normal = (daily_pos == 1).sum()
    n_skip = (daily_pos == 0).sum()
    n_reverse = (daily_pos == -1).sum()

    # Mean spread by position type
    mean_spread_normal = df.loc[df["position"] == 1, "spread_da_imb"].mean() if (df["position"] == 1).any() else np.nan
    mean_spread_other = df.loc[df["position"] != 1, "spread_da_imb"].mean() if (df["position"] != 1).any() else np.nan

    return {
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_days": len(daily_pnl),
        "n_normal": n_normal,
        "n_skip": n_skip,
        "n_reverse": n_reverse,
        "mean_spread_normal": mean_spread_normal,
        "mean_spread_other": mean_spread_other,
        "daily_pnl": daily_pnl,
    }


# =============================================================================
# 5. PART 1: Full period results (Feb 2024 - Jan 2026, actual temps)
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: FULL PERIOD RESULTS (Feb 2024 - Jan 2026, Actual Temperature)")
print("=" * 80)

variants = {
    "A_baseline": {"mode": "baseline"},
    "B_skip_extreme": {"mode": "skip_extreme"},
    "C_reverse_extreme": {"mode": "reverse_extreme"},
    "D_reverse_cold_only": {"mode": "reverse_cold_only"},
    "E_reverse_hot_only": {"mode": "reverse_hot_only"},
}

# Default thresholds: 0 C and 25 C
results_default = {}
for name, params in variants.items():
    strat = compute_strategy(mkt, daily_temp_actual,
                             cold_thresh=0.0, hot_thresh=25.0,
                             mode=params["mode"])
    metrics = compute_metrics(strat)
    results_default[name] = metrics

print("\n--- Default Thresholds: cold < 0 C, hot > 25 C ---")
print(f"{'Variant':<25s} {'Total P&L':>12s} {'Sharpe':>8s} {'MaxDD':>12s} {'Days':>6s} {'Normal':>8s} {'Skip':>6s} {'Rev':>6s} {'Spread(N)':>10s} {'Spread(O)':>10s}")
print("-" * 115)
for name, m in results_default.items():
    print(f"{name:<25s} {m['total_pnl']:>12,.0f} {m['sharpe']:>8.2f} {m['max_dd']:>12,.0f} {m['n_days']:>6d} "
          f"{m['n_normal']:>8d} {m['n_skip']:>6d} {m['n_reverse']:>6d} "
          f"{m['mean_spread_normal']:>10.2f} {m['mean_spread_other']:>10.2f}" if not np.isnan(m.get('mean_spread_other', np.nan)) else
          f"{name:<25s} {m['total_pnl']:>12,.0f} {m['sharpe']:>8.2f} {m['max_dd']:>12,.0f} {m['n_days']:>6d} "
          f"{m['n_normal']:>8d} {m['n_skip']:>6d} {m['n_reverse']:>6d} "
          f"{m['mean_spread_normal']:>10.2f} {'N/A':>10s}")

# IS-optimal thresholds
print("\n--- IS-Optimal Thresholds: cold < 2 C, hot > 26 C ---")
results_optimal = {}
for name, params in variants.items():
    strat = compute_strategy(mkt, daily_temp_actual,
                             cold_thresh=2.0, hot_thresh=26.0,
                             mode=params["mode"])
    metrics = compute_metrics(strat)
    results_optimal[name] = metrics

print(f"{'Variant':<25s} {'Total P&L':>12s} {'Sharpe':>8s} {'MaxDD':>12s} {'Days':>6s} {'Normal':>8s} {'Skip':>6s} {'Rev':>6s}")
print("-" * 95)
for name, m in results_optimal.items():
    print(f"{name:<25s} {m['total_pnl']:>12,.0f} {m['sharpe']:>8.2f} {m['max_dd']:>12,.0f} {m['n_days']:>6d} "
          f"{m['n_normal']:>8d} {m['n_skip']:>6d} {m['n_reverse']:>6d}")


# =============================================================================
# 6. PART 2: January 2026 zoom-in
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: JANUARY 2026 ZOOM-IN")
print("=" * 80)

import datetime
jan26_start = pd.Timestamp("2026-01-01")
jan26_end = pd.Timestamp("2026-01-31 23:59:59")
mkt_jan26 = mkt.loc[jan26_start:jan26_end]
print(f"[*] Jan 2026 market data: {len(mkt_jan26)} hours, {mkt_jan26.index.min()} to {mkt_jan26.index.max()}")

# Run strategies with actual temp
jan_variants_actual = {}
for name, params in variants.items():
    strat = compute_strategy(mkt_jan26, daily_temp_actual,
                             cold_thresh=0.0, hot_thresh=25.0,
                             mode=params["mode"])
    jan_variants_actual[name] = strat

# Also with GFS forecast
jan_variants_gfs = {}
for name, params in variants.items():
    strat = compute_strategy(mkt_jan26, gfs_daily,
                             cold_thresh=0.0, hot_thresh=25.0,
                             mode=params["mode"])
    jan_variants_gfs[name] = strat

# Daily P&L tables
print("\n--- Jan 2026 Day-by-Day: Actual Temperature ---")
print(f"{'Date':<14s} {'Temp':>6s} {'Class':>8s} {'A(base)':>10s} {'B(skip)':>10s} {'C(rev)':>10s} {'D(r_cold)':>10s} {'E(r_hot)':>10s}")
print("-" * 90)

# Build daily summary
jan_daily_actual = {}
for name in variants:
    df = jan_variants_actual[name]
    jan_daily_actual[name] = df.groupby("day")["hourly_pnl"].sum()

all_days = sorted(set().union(*[set(v.index) for v in jan_daily_actual.values()]))
for day in all_days:
    temp_val = daily_temp_actual.loc[day, "daily_mean_temp_actual"] if day in daily_temp_actual.index else np.nan
    if temp_val < 0:
        cls = "COLD"
    elif temp_val > 25:
        cls = "HOT"
    else:
        cls = "normal"

    vals = []
    for name in ["A_baseline", "B_skip_extreme", "C_reverse_extreme", "D_reverse_cold_only", "E_reverse_hot_only"]:
        v = jan_daily_actual[name].get(day, 0)
        vals.append(v)

    print(f"{str(day):<14s} {temp_val:>6.1f} {cls:>8s} {vals[0]:>10.0f} {vals[1]:>10.0f} {vals[2]:>10.0f} {vals[3]:>10.0f} {vals[4]:>10.0f}")

# Totals
print("-" * 90)
totals_actual = []
for name in ["A_baseline", "B_skip_extreme", "C_reverse_extreme", "D_reverse_cold_only", "E_reverse_hot_only"]:
    totals_actual.append(jan_daily_actual[name].sum())
print(f"{'TOTAL':<14s} {'':>6s} {'':>8s} {totals_actual[0]:>10.0f} {totals_actual[1]:>10.0f} {totals_actual[2]:>10.0f} {totals_actual[3]:>10.0f} {totals_actual[4]:>10.0f}")

# GFS version
print("\n--- Jan 2026 Day-by-Day: GFS Forecast Temperature ---")
print(f"{'Date':<14s} {'GFS_T':>6s} {'Act_T':>6s} {'Class':>8s} {'A(base)':>10s} {'C(rev)':>10s}")
print("-" * 70)

jan_daily_gfs = {}
for name in variants:
    df = jan_variants_gfs[name]
    if len(df) > 0:
        jan_daily_gfs[name] = df.groupby("day")["hourly_pnl"].sum()
    else:
        jan_daily_gfs[name] = pd.Series(dtype=float)

all_days_gfs = sorted(set().union(*[set(v.index) for v in jan_daily_gfs.values() if len(v) > 0]))
for day in all_days_gfs:
    gfs_t = gfs_daily.loc[day, "daily_mean_temp_gfs"] if day in gfs_daily.index else np.nan
    act_t = daily_temp_actual.loc[day, "daily_mean_temp_actual"] if day in daily_temp_actual.index else np.nan
    if np.isnan(gfs_t):
        cls = "N/A"
    elif gfs_t < 0:
        cls = "COLD"
    elif gfs_t > 25:
        cls = "HOT"
    else:
        cls = "normal"

    v_a = jan_daily_gfs.get("A_baseline", pd.Series(dtype=float)).get(day, 0)
    v_c = jan_daily_gfs.get("C_reverse_extreme", pd.Series(dtype=float)).get(day, 0)

    gfs_str = f"{gfs_t:>6.1f}" if not np.isnan(gfs_t) else "   N/A"
    act_str = f"{act_t:>6.1f}" if not np.isnan(act_t) else "   N/A"
    print(f"{str(day):<14s} {gfs_str} {act_str} {cls:>8s} {v_a:>10.0f} {v_c:>10.0f}")

if all_days_gfs:
    print("-" * 70)
    tot_a = jan_daily_gfs.get("A_baseline", pd.Series(dtype=float)).sum()
    tot_c = jan_daily_gfs.get("C_reverse_extreme", pd.Series(dtype=float)).sum()
    print(f"{'TOTAL':<14s} {'':>6s} {'':>6s} {'':>8s} {tot_a:>10.0f} {tot_c:>10.0f}")


# =============================================================================
# 7. PART 3: Monthly Breakdown
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: MONTHLY P&L BREAKDOWN (A, B, C)")
print("=" * 80)

monthly_results = {}
for name in ["A_baseline", "B_skip_extreme", "C_reverse_extreme"]:
    params = variants[name]
    strat = compute_strategy(mkt, daily_temp_actual,
                             cold_thresh=0.0, hot_thresh=25.0,
                             mode=params["mode"])
    strat["month"] = strat.index.to_period('M')
    monthly = strat.groupby("month")["hourly_pnl"].sum()
    monthly_results[name] = monthly

# Combine
monthly_df = pd.DataFrame(monthly_results)
print(f"\n{'Month':<12s} {'A_baseline':>12s} {'B_skip':>12s} {'C_reverse':>12s} {'C - A':>12s}")
print("-" * 65)
for idx in monthly_df.index:
    a = monthly_df.loc[idx, "A_baseline"]
    b = monthly_df.loc[idx, "B_skip_extreme"]
    c = monthly_df.loc[idx, "C_reverse_extreme"]
    diff = c - a
    print(f"{str(idx):<12s} {a:>12,.0f} {b:>12,.0f} {c:>12,.0f} {diff:>12,.0f}")
print("-" * 65)
print(f"{'TOTAL':<12s} {monthly_df['A_baseline'].sum():>12,.0f} {monthly_df['B_skip_extreme'].sum():>12,.0f} {monthly_df['C_reverse_extreme'].sum():>12,.0f} {(monthly_df['C_reverse_extreme'].sum() - monthly_df['A_baseline'].sum()):>12,.0f}")


# =============================================================================
# 8. PART 4: Cold Threshold Sweep
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: COLD THRESHOLD SWEEP (Reverse below threshold, no hot filter)")
print("=" * 80)

cold_thresholds = np.arange(-3, 5.5, 0.5)
sweep_results = {"threshold": [], "pnl_2024": [], "pnl_2025plus": [], "pnl_total": []}

# Split market data
mkt_2024 = mkt[mkt.index < "2025-01-01"]
mkt_2025plus = mkt[mkt.index >= "2025-01-01"]

for ct in cold_thresholds:
    # In-sample: 2024
    strat_is = compute_strategy(mkt_2024, daily_temp_actual,
                                cold_thresh=ct, hot_thresh=999,  # no hot filter
                                mode="reverse_cold_only")
    pnl_is = strat_is["hourly_pnl"].sum()

    # Out-of-sample: 2025+
    strat_oos = compute_strategy(mkt_2025plus, daily_temp_actual,
                                 cold_thresh=ct, hot_thresh=999,
                                 mode="reverse_cold_only")
    pnl_oos = strat_oos["hourly_pnl"].sum()

    sweep_results["threshold"].append(ct)
    sweep_results["pnl_2024"].append(pnl_is)
    sweep_results["pnl_2025plus"].append(pnl_oos)
    sweep_results["pnl_total"].append(pnl_is + pnl_oos)

sweep_df = pd.DataFrame(sweep_results)

print(f"\n{'Threshold':>10s} {'IS (2024)':>12s} {'OOS (2025+)':>12s} {'Total':>12s}")
print("-" * 50)
for _, row in sweep_df.iterrows():
    print(f"{row['threshold']:>10.1f} {row['pnl_2024']:>12,.0f} {row['pnl_2025plus']:>12,.0f} {row['pnl_total']:>12,.0f}")

# Find optimal
best_is = sweep_df.loc[sweep_df["pnl_2024"].idxmax()]
best_oos = sweep_df.loc[sweep_df["pnl_2025plus"].idxmax()]
best_total = sweep_df.loc[sweep_df["pnl_total"].idxmax()]

print(f"\n[+] Best IS threshold: {best_is['threshold']:.1f} C (PnL: {best_is['pnl_2024']:,.0f})")
print(f"[+] Best OOS threshold: {best_oos['threshold']:.1f} C (PnL: {best_oos['pnl_2025plus']:,.0f})")
print(f"[+] Best Total threshold: {best_total['threshold']:.1f} C (PnL: {best_total['pnl_total']:,.0f})")


# =============================================================================
# 9. VISUALIZATION
# =============================================================================
print("\n[*] Creating visualization...")

fig, axes = plt.subplots(4, 1, figsize=(14, 14))
fig.suptitle("Temperature-Filtered Sell DA / Buy Imbalance Strategy\nwith Extreme-Day REVERSAL",
             fontsize=14, fontweight='bold', y=0.98)

# ---- Panel 1: Cumulative P&L ----
ax1 = axes[0]
for name, label, color, ls in [
    ("A_baseline", "A: Baseline (always trade)", "black", "-"),
    ("B_skip_extreme", "B: Skip extreme (<0C, >25C)", "blue", "--"),
    ("C_reverse_extreme", "C: Reverse extreme", "red", "-"),
]:
    strat = compute_strategy(mkt, daily_temp_actual, cold_thresh=0.0, hot_thresh=25.0,
                             mode=variants[name]["mode"])
    daily = strat.groupby("day")["hourly_pnl"].sum()
    cum = daily.cumsum()
    dates = pd.to_datetime(list(cum.index))
    ax1.plot(dates, cum.values, label=label, color=color, linestyle=ls, linewidth=1.5)

ax1.set_title("Panel 1: Cumulative P&L (Feb 2024 - Jan 2026)", fontsize=11)
ax1.set_ylabel("Cumulative P&L (EUR)")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='gray', linewidth=0.5)

# ---- Panel 2: Jan 2026 daily bars + temperature ----
ax2 = axes[1]
ax2_twin = ax2.twinx()

jan_daily_a = jan_daily_actual["A_baseline"]
jan_daily_c = jan_daily_actual["C_reverse_extreme"]

jan_dates = pd.to_datetime(list(jan_daily_a.index))
width = 0.35
x = np.arange(len(jan_dates))

bars_a = ax2.bar(x - width/2, jan_daily_a.values, width, label="A: Baseline", color="gray", alpha=0.7)
bars_c = ax2.bar(x + width/2, jan_daily_c.values, width, label="C: Reverse extreme", color="red", alpha=0.7)

# Temperature line
jan_temps = [daily_temp_actual.loc[d, "daily_mean_temp_actual"] if d in daily_temp_actual.index else np.nan
             for d in jan_daily_a.index]
ax2_twin.plot(x, jan_temps, color="blue", linewidth=1.5, marker='o', markersize=3, label="Daily Mean Temp")
ax2_twin.axhline(0, color='blue', linewidth=0.5, linestyle='--', alpha=0.5)
ax2_twin.set_ylabel("Temperature (C)", color='blue')
ax2_twin.tick_params(axis='y', labelcolor='blue')

ax2.set_title("Panel 2: January 2026 Daily P&L: Baseline vs Reverse", fontsize=11)
ax2.set_ylabel("Daily P&L (EUR)")
ax2.set_xticks(x[::2])
ax2.set_xticklabels([str(d) for d in list(jan_daily_a.index)[::2]], rotation=45, fontsize=7, ha='right')
ax2.legend(loc="upper left", fontsize=8)
ax2_twin.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='gray', linewidth=0.5)

# ---- Panel 3: Monthly P&L comparison ----
ax3 = axes[2]
months = monthly_df.index.astype(str)
x3 = np.arange(len(months))
w = 0.35

ax3.bar(x3 - w/2, monthly_df["A_baseline"].values, w, label="A: Baseline", color="gray", alpha=0.7)
ax3.bar(x3 + w/2, monthly_df["C_reverse_extreme"].values, w, label="C: Reverse extreme", color="red", alpha=0.7)
ax3.set_title("Panel 3: Monthly P&L Comparison (A vs C)", fontsize=11)
ax3.set_ylabel("Monthly P&L (EUR)")
ax3.set_xticks(x3)
ax3.set_xticklabels(months, rotation=45, fontsize=8, ha='right')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(0, color='gray', linewidth=0.5)

# ---- Panel 4: Cold threshold sweep ----
ax4 = axes[3]
ax4.plot(sweep_df["threshold"], sweep_df["pnl_2024"], 'b-o', markersize=4, label="IS: 2024", linewidth=1.5)
ax4.plot(sweep_df["threshold"], sweep_df["pnl_2025plus"], 'r-s', markersize=4, label="OOS: 2025+", linewidth=1.5)
ax4.plot(sweep_df["threshold"], sweep_df["pnl_total"], 'k--', linewidth=1, alpha=0.5, label="Total")
ax4.axvline(best_is["threshold"], color='blue', linestyle=':', alpha=0.5, label=f"Best IS: {best_is['threshold']:.1f}C")
ax4.axvline(best_oos["threshold"], color='red', linestyle=':', alpha=0.5, label=f"Best OOS: {best_oos['threshold']:.1f}C")
ax4.set_title("Panel 4: Cold Reversal Threshold Sweep", fontsize=11)
ax4.set_xlabel("Cold Threshold (C)")
ax4.set_ylabel("Total P&L (EUR)")
ax4.legend(fontsize=8, loc="best")
ax4.grid(True, alpha=0.3)
ax4.axhline(0, color='gray', linewidth=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
outpath = OUT_DIR / "temp_reverse_strategy.png"
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"[+] Saved: {outpath}")
plt.close()


# =============================================================================
# 10. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

baseline_pnl = results_default["A_baseline"]["total_pnl"]
reverse_pnl = results_default["C_reverse_extreme"]["total_pnl"]
skip_pnl = results_default["B_skip_extreme"]["total_pnl"]
rev_cold_pnl = results_default["D_reverse_cold_only"]["total_pnl"]

print(f"\n[+] Baseline P&L:           {baseline_pnl:>12,.0f} EUR")
print(f"[+] Skip extreme P&L:      {skip_pnl:>12,.0f} EUR  (delta: {skip_pnl - baseline_pnl:>+,.0f})")
print(f"[+] Reverse extreme P&L:   {reverse_pnl:>12,.0f} EUR  (delta: {reverse_pnl - baseline_pnl:>+,.0f})")
print(f"[+] Reverse cold only P&L: {rev_cold_pnl:>12,.0f} EUR  (delta: {rev_cold_pnl - baseline_pnl:>+,.0f})")

# Jan 2026 improvement
jan_base = totals_actual[0]
jan_rev = totals_actual[2]
print(f"\n[*] Jan 2026 Baseline:     {jan_base:>12,.0f} EUR")
print(f"[*] Jan 2026 Reverse:      {jan_rev:>12,.0f} EUR  (delta: {jan_rev - jan_base:>+,.0f})")

print(f"\n[+] Optimal cold reversal threshold (IS): {best_is['threshold']:.1f} C")
print(f"[+] Optimal cold reversal threshold (OOS): {best_oos['threshold']:.1f} C")
print(f"[+] OOS P&L at IS-optimal threshold ({best_is['threshold']:.1f} C): "
      f"{sweep_df.loc[sweep_df['threshold'] == best_is['threshold'], 'pnl_2025plus'].values[0]:,.0f} EUR")

print("\n[+] Analysis complete.")
