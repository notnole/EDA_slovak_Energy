"""
Bardejov CHP - Final Dispatch Optimization
============================================
Uses the ACTUAL cost model from Priprava prevadzky TEHO_3.xlsx.

Key parameters from operator Excel:
  - Fuel: 51 EUR/t wet woodchip, 42% moisture, 2.6 MWh/t LHV
  - Boiler eff: 85% -> fuel cost = 23.08 EUR/MWh_th
  - Ash: 1.6% content, 95 EUR/t disposal -> 0.688 EUR/MWh_th
  - Own consumption: 8% of EE
  - T/EE ratio (1-boiler): 5.3 @1MW -> 3.1 @4MW
  - T/EE ratio (2-boiler): 1.86 @5MW -> 1.16 @8MW
  - Boiler 2 startup: ~300 EUR net (4h warmup, but 5MW is CHEAPER than 4MW)

Critical finding: 2-boiler mode is far more electrically efficient.
8 MW breakeven = 49 EUR/MWh. 1 MW breakeven = 111 EUR/MWh.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PLOT_DIR = BASE / "dispatch_analysis" / "plots"

# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
FUEL_PER_MWH_TH = 23.08
ASH_PER_MWH_TH = 0.688
OWN_CONSUMPTION = 0.08

# T/EE ratios from Excel
RATIO = {
    0: 0,
    1: 5.3, 2: 4.0, 3: 3.5, 4: 3.1,       # 1-boiler
    5: 1.86, 6: 1.55, 7: 1.33, 8: 1.16,     # 2-boiler
}


def regime(step):
    if step == 0: return 0
    if step <= 4: return 1
    return 2


def hourly_cost(step, heat_demand):
    if step == 0:
        return 0.0
    ratio = RATIO[step]
    heat_produced = step * ratio
    dissipated = max(0, heat_produced - heat_demand)
    fuel = (step + dissipated) * FUEL_PER_MWH_TH
    ash = (step + dissipated) * ASH_PER_MWH_TH
    return fuel + ash


def hourly_profit(step, price, heat_demand):
    if step == 0:
        return 0.0
    cost = hourly_cost(step, heat_demand)
    revenue = step * price * (1 - OWN_CONSUMPTION)
    return revenue - cost


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("[*] Loading data...")
plant = pd.read_csv(BASE / "plant_timeseries.csv", parse_dates=["datetime"])
prices = pd.read_csv(BASE / "da_prices_hourly.csv", parse_dates=["datetime"])

plant = plant.set_index("datetime").resample("h").mean().reset_index()
plant["ee_MW"] = plant["electricity_kW"].fillna(0) / 1000
plant["heat_MW"] = plant["heat_load_kW"].fillna(0) / 1000
plant["step"] = plant["ee_MW"].round().clip(0, 8).astype(int)

df = plant.merge(prices, on="datetime", how="inner")
df = df[df["datetime"].dt.year == 2025].copy().reset_index(drop=True)
n = len(df)
print(f"[+] {n} hours in 2025")

# ---------------------------------------------------------------------------
# Actual profit
# ---------------------------------------------------------------------------
df["actual_profit"] = df.apply(
    lambda r: hourly_profit(r["step"], r["da_price_eur"], r["heat_MW"]), axis=1
)

# ---------------------------------------------------------------------------
# Greedy optimal (no boiler constraints)
# ---------------------------------------------------------------------------
def best_step_greedy(price, heat_demand):
    best_s, best_p = 0, 0.0
    for s in range(1, 9):
        p = hourly_profit(s, price, heat_demand)
        if p > best_p:
            best_s, best_p = s, p
    return best_s, best_p

greedy = df.apply(lambda r: best_step_greedy(r["da_price_eur"], r["heat_MW"]), axis=1)
df["greedy_step"] = [r[0] for r in greedy]
df["greedy_profit"] = [r[1] for r in greedy]

# ---------------------------------------------------------------------------
# Boiler-constrained DP
# ---------------------------------------------------------------------------
print("[*] Running boiler-constrained DP...")

# States: (regime, warmup)
# regime 0=off, 1=1-boiler, 2=2-boiler
# warmup: hours remaining before free dispatch
COLD_START_WARMUP = 6
BOILER2_WARMUP = 4
COLD_START_MAINT = 500
BOILER2_MAINT = 300  # net startup cost (fuel savings offset most of it)
MAX_WARMUP = 6

STATES = []
STATE_IDX = {}
for r in range(3):
    for w in range(MAX_WARMUP + 1):
        STATE_IDX[(r, w)] = len(STATES)
        STATES.append((r, w))
N_STATES = len(STATES)

prices_arr = df["da_price_eur"].values
heat_arr = df["heat_MW"].values


def best_in_regime(reg, warmup, price, heat):
    if reg == 0:
        return 0, 0.0
    elif reg == 1:
        if warmup > 0:
            s = 1
            return s, hourly_profit(s, price, heat)
        best_s, best_p = 1, hourly_profit(1, price, heat)
        for s in range(2, 5):
            p = hourly_profit(s, price, heat)
            if p > best_p:
                best_s, best_p = s, p
        return best_s, best_p
    else:
        if warmup > 0:
            s = 5
            return s, hourly_profit(s, price, heat)
        best_s, best_p = 5, hourly_profit(5, price, heat)
        for s in range(6, 9):
            p = hourly_profit(s, price, heat)
            if p > best_p:
                best_s, best_p = s, p
        return best_s, best_p


# DP backward pass
profit_table = np.full((n, N_STATES), -np.inf)
choice_table = np.zeros((n, N_STATES), dtype=int)
step_table = np.zeros((n, N_STATES), dtype=int)

for si, (r, w) in enumerate(STATES):
    s, p = best_in_regime(r, w, prices_arr[n-1], heat_arr[n-1])
    profit_table[n-1][si] = p
    step_table[n-1][si] = s

for t in range(n-2, -1, -1):
    price = prices_arr[t]
    heat = heat_arr[t]
    for si, (r, w) in enumerate(STATES):
        s, immediate = best_in_regime(r, w, price, heat)
        best_future = -np.inf
        best_next_si = si

        # Stay
        if r == 0:
            nsi = STATE_IDX[(0, 0)]
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi
        else:
            nsi = STATE_IDX[(r, max(0, w-1))]
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi

        # Switch
        if r == 0:
            nsi = STATE_IDX[(1, COLD_START_WARMUP)]
            f = profit_table[t+1][nsi] - COLD_START_MAINT
            if f > best_future:
                best_future = f
                best_next_si = nsi
        elif r == 1:
            nsi = STATE_IDX[(0, 0)]
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi
            if w == 0:
                nsi = STATE_IDX[(2, BOILER2_WARMUP)]
                f = profit_table[t+1][nsi] - BOILER2_MAINT
                if f > best_future:
                    best_future = f
                    best_next_si = nsi
        elif r == 2:
            nsi = STATE_IDX[(1, 0)]
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi
            nsi = STATE_IDX[(0, 0)]
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi

        profit_table[t][si] = immediate + best_future
        choice_table[t][si] = best_next_si
        step_table[t][si] = s

# Forward pass
opt_states = np.zeros(n, dtype=int)
opt_steps = np.zeros(n, dtype=int)
opt_states[0] = np.argmax(profit_table[0])
opt_steps[0] = step_table[0][opt_states[0]]
for t in range(1, n):
    opt_states[t] = choice_table[t-1][opt_states[t-1]]
    opt_steps[t] = step_table[t][opt_states[t]]

df["dp_step"] = opt_steps
df["dp_regime"] = [STATES[s][0] for s in opt_states]
df["dp_profit"] = df.apply(
    lambda r: hourly_profit(r["dp_step"], r["da_price_eur"], r["heat_MW"]), axis=1
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2025 FINAL RESULTS - REAL COST MODEL FROM OPERATOR EXCEL")
print("=" * 70)

for label, col in [("Actual", "actual_profit"),
                   ("Greedy (no constraints)", "greedy_profit"),
                   ("DP (boiler constrained)", "dp_profit")]:
    total = df[col].sum() / 1000
    print(f"  {label:35s}: {total:8.1f} kEUR")

gap_dp = (df["dp_profit"].sum() - df["actual_profit"].sum()) / 1000
gap_greedy = (df["greedy_profit"].sum() - df["actual_profit"].sum()) / 1000
print(f"\n  Gap (DP):     {gap_dp:.0f} kEUR")
print(f"  Gap (greedy): {gap_greedy:.0f} kEUR")
print(f"  Boiler constraint cost: {gap_greedy - gap_dp:.0f} kEUR")

# Monthly
print("\n" + "=" * 70)
print("MONTHLY (kEUR)")
print("=" * 70)
df["month"] = df["datetime"].dt.month
monthly = df.groupby("month").agg(
    actual=("actual_profit", "sum"),
    dp=("dp_profit", "sum"),
    greedy=("greedy_profit", "sum"),
    mean_price=("da_price_eur", "mean"),
    mean_heat=("heat_MW", "mean"),
)
monthly[["actual", "dp", "greedy"]] /= 1000
monthly["gap"] = monthly["dp"] - monthly["actual"]
monthly["gap_pct"] = (monthly["gap"] / monthly["dp"].abs().clip(lower=1) * 100).round(0)
print(monthly[["actual", "dp", "gap", "gap_pct", "mean_price", "mean_heat"]].round(1).to_string())

# Step distribution
print("\n" + "=" * 70)
print("STEP DISTRIBUTION")
print("=" * 70)
for label, col in [("Actual", "step"), ("DP", "dp_step"), ("Greedy", "greedy_step")]:
    dist = df[col].value_counts(normalize=True).sort_index() * 100
    line = ", ".join(f"{s}:{dist.get(s,0):.0f}%" for s in range(0, 9))
    print(f"  {label:8s}: {line}")

# Regime distribution
print("\n" + "=" * 70)
print("REGIME DISTRIBUTION")
print("=" * 70)
df["actual_regime"] = df["step"].apply(regime)
for label, col in [("Actual", "actual_regime"), ("DP", "dp_regime")]:
    dist = df[col].value_counts(normalize=True).sort_index() * 100
    names = {0: "OFF", 1: "1-boiler", 2: "2-boiler"}
    line = ", ".join(f"{names[r]}:{dist.get(r,0):.0f}%" for r in range(3))
    print(f"  {label:8s}: {line}")

# Regime transitions
for label, col in [("Actual", "actual_regime"), ("DP", "dp_regime")]:
    b2_starts = ((df[col].shift(1) == 1) & (df[col] == 2)).sum()
    cold_starts = ((df[col].shift(1) == 0) & (df[col] > 0)).sum()
    print(f"  {label}: {cold_starts} cold starts, {b2_starts} boiler-2 starts")

# Top mistakes
print("\n" + "=" * 70)
print("TOP MISTAKES")
print("=" * 70)
df["gap"] = df["dp_profit"] - df["actual_profit"]
wrong = df[df["step"] != df["dp_step"]].copy()
wrong["pattern"] = wrong["step"].astype(str) + "MW->" + wrong["dp_step"].astype(str) + "MW"
patterns = wrong.groupby("pattern").agg(
    count=("gap", "count"),
    total_gap=("gap", "sum"),
    avg_price=("da_price_eur", "mean"),
    avg_heat=("heat_MW", "mean"),
).sort_values("total_gap", ascending=False)
patterns["total_gap"] /= 1000
print(f"Wrong-step hours: {len(wrong)} ({len(wrong)/len(df)*100:.0f}%)")
print(f"Total gap: {wrong['gap'].sum()/1000:.0f} kEUR")
print("\nPattern       Hours   Gap(kEUR)  AvgPrice  AvgHeat")
for pat, row in patterns.head(12).iterrows():
    print(f"  {pat:12s} {row['count']:5.0f}   {row['total_gap']:7.1f}    {row['avg_price']:6.0f}    {row['avg_heat']:5.1f}")

# Seasonal
print("\n" + "=" * 70)
print("SEASONAL")
print("=" * 70)
df["season"] = df["month"].apply(lambda m: "winter" if m >= 10 or m <= 3 else "summer")
for season in ["winter", "summer"]:
    sdf = df[df["season"] == season]
    a = sdf["actual_profit"].sum() / 1000
    d = sdf["dp_profit"].sum() / 1000
    print(f"  {season:8s}: actual={a:.0f}, optimal={d:.0f}, gap={d-a:.0f} kEUR")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
print("\n[*] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Bardejov CHP - Final Dispatch Analysis (Real Cost Model, 2025)",
             fontsize=14, fontweight="bold")

# 1. Breakeven vs heat demand
ax = axes[0, 0]
heat_range = range(0, 18)
for step, color, ls in [(1, "red", "-"), (3, "orange", "-"), (4, "brown", "--"),
                         (5, "green", "-"), (8, "blue", "-")]:
    breakevens = []
    for h in heat_range:
        ratio = RATIO[step]
        dissip = max(0, step * ratio - h)
        cost = (step + dissip) * (FUEL_PER_MWH_TH + ASH_PER_MWH_TH)
        be = cost / (step * (1 - OWN_CONSUMPTION))
        breakevens.append(be)
    boiler = "1B" if step <= 4 else "2B"
    ax.plot(heat_range, breakevens, color=color, linestyle=ls,
            linewidth=2, label=f"{step}MW ({boiler})")
ax.set_xlabel("Heat Demand (MW)")
ax.set_ylabel("Breakeven Price (EUR/MWh)")
ax.set_title("Breakeven vs Heat Demand")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 150)

# 2. Step distribution
ax = axes[0, 1]
steps_range = range(0, 9)
width = 0.25
for i, (label, col, color) in enumerate([
    ("Actual", "step", "red"),
    ("DP", "dp_step", "green"),
    ("Greedy", "greedy_step", "blue"),
]):
    dist = df[col].value_counts(normalize=True).reindex(steps_range, fill_value=0)
    ax.bar([s + i*width - width for s in steps_range], dist * 100, width,
           label=label, color=color, alpha=0.7)
ax.set_xlabel("MW Step")
ax.set_ylabel("% of Hours")
ax.set_title("Step Distribution")
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(4.5, color="gray", linestyle=":", alpha=0.5)
ax.text(4.5, ax.get_ylim()[1]*0.9, "1B|2B", ha="center", fontsize=9, color="gray")

# 3. Monthly gap
ax = axes[1, 0]
x = monthly.index
w = 0.35
ax.bar(x - w/2, monthly["actual"], w, label="Actual", color="red", alpha=0.7)
ax.bar(x + w/2, monthly["dp"], w, label="Optimal (DP)", color="green", alpha=0.7)
ax.set_xlabel("Month")
ax.set_ylabel("Profit (kEUR)")
ax.set_title("Monthly Profit: Actual vs Optimal")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# 4. Weekly timeseries
ax = axes[1, 1]
weekly = df.set_index("datetime").resample("W").agg({
    "actual_profit": "sum",
    "dp_profit": "sum",
}) / 1000
ax.fill_between(weekly.index, weekly["actual_profit"], weekly["dp_profit"],
                where=weekly["dp_profit"] > weekly["actual_profit"],
                alpha=0.3, color="green")
ax.plot(weekly.index, weekly["actual_profit"], "r-", linewidth=1, label="Actual")
ax.plot(weekly.index, weekly["dp_profit"], "g-", linewidth=1, label="DP Optimal")
ax.set_ylabel("Weekly Profit (kEUR)")
ax.set_title("Weekly Profit Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "09_final_real_model.png", dpi=150)
plt.close()
print(f"[+] Saved {PLOT_DIR / '09_final_real_model.png'}")

print("\n[+] Done!")
