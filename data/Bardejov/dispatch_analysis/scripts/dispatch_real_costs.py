"""
Bardejov CHP - Dispatch Optimization with Real Cost Model
==========================================================
Uses the ACTUAL cost model from the operator's Kalkulacie Excel sheets.

Cost model (from formulas in kalkulacia leto / predikcia sheets):
  - Fuel: 1.1 ATT per MWh_ee, at 110 EUR/ATT = 121 EUR/MWh_ee GROSS
  - Heat credit: 110/5/0.787 = 27.95 EUR per MWh_th of useful heat
  - Net EE cost = (fuel - heat_credit) / EE_MWh

BUT: the operator model is simplified. It assumes fuel ~ EE output.
Reality: fuel ~ total boiler thermal output, and dissipated heat = wasted fuel.

This script uses two approaches:
  A) Operator model (121 EUR/MWh breakeven, simple)
  B) Physics model (fuel ~ total thermal, dissipation matters)
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
# Constants
# ---------------------------------------------------------------------------
CHIP_PRICE = 110        # EUR/ATT
ATT_PER_MWH_EE = 1.1   # operator's assumption
FUEL_GROSS = CHIP_PRICE * ATT_PER_MWH_EE  # 121 EUR/MWh_ee
HEAT_CREDIT = CHIP_PRICE / 5 / 0.787      # 27.95 EUR/MWh_th

# Physics model
BOILER_EFF = 0.85
MWH_PER_ATT = 3.5  # woodchip energy content

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("[*] Loading data...")
plant = pd.read_csv(BASE / "plant_timeseries.csv", parse_dates=["datetime"])
prices = pd.read_csv(BASE / "da_prices_hourly.csv", parse_dates=["datetime"])

plant = plant.set_index("datetime").resample("h").mean().reset_index()
plant["ee_MW"] = plant["electricity_kW"].fillna(0) / 1000
plant["heat_MW"] = plant["heat_load_kW"].fillna(0) / 1000
plant["cool_MW"] = plant["cooling_kW"].fillna(0) / 1000
plant["step"] = plant["ee_MW"].round().clip(0, 8).astype(int)

df = plant.merge(prices, on="datetime", how="inner")
df = df[df["datetime"].dt.year == 2025].copy().reset_index(drop=True)
print(f"[+] {len(df)} hours in 2025")

# ---------------------------------------------------------------------------
# Build thermal profile lookup from actual data
# At each step, what's the TYPICAL thermal balance?
# We need this for counterfactual: "if they had run at step X this hour,
# what would the thermal balance have been?"
#
# Key insight: heat demand is exogenous (weather/network driven).
# Dissipation = total_boiler_thermal - EE - useful_heat
# Total boiler thermal scales with step (more EE = more boiler load)
# ---------------------------------------------------------------------------

# Observed total thermal output at each step (boiler_output = EE + heat + cooling)
# Group by step to get average boiler thermal profile
print("[*] Building thermal profiles per step...")

# For the physics model we need: at each step, what's the total boiler output?
# This determines fuel consumption regardless of how much heat is useful
thermal_by_step = df[df["step"] > 0].groupby("step").agg(
    mean_ee=("ee_MW", "mean"),
).copy()
thermal_by_step["mean_total_thermal"] = 0.0

# Total thermal = ee + heat + max(0, cooling)
for step in thermal_by_step.index:
    mask = df["step"] == step
    thermal_by_step.loc[step, "mean_total_thermal"] = (
        df.loc[mask, "ee_MW"] + df.loc[mask, "heat_MW"] + df.loc[mask, "cool_MW"].clip(lower=0)
    ).mean()

# Boiler thermal output at each step (this is what burns fuel)
BOILER_THERMAL = {0: 0}
for step in range(1, 9):
    if step in thermal_by_step.index:
        BOILER_THERMAL[step] = thermal_by_step.loc[step, "mean_total_thermal"]
    else:
        # interpolate
        BOILER_THERMAL[step] = step * 3.2  # rough approximation

print("Boiler thermal output by step:")
for s in range(0, 9):
    print(f"  {s} MW EE -> {BOILER_THERMAL[s]:.1f} MW thermal")


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def fuel_cost_physics(step):
    """Fuel cost per hour based on total boiler thermal output."""
    thermal = BOILER_THERMAL[step]
    fuel_mw = thermal / BOILER_EFF
    att_per_h = fuel_mw / MWH_PER_ATT
    return att_per_h * CHIP_PRICE

def profit_physics(step, price, useful_heat_mw):
    """Hourly profit with physics-based fuel cost."""
    if step == 0:
        return 0.0
    ee_revenue = step * price
    heat_revenue = useful_heat_mw * HEAT_CREDIT
    fuel = fuel_cost_physics(step)
    return ee_revenue + heat_revenue - fuel

def profit_operator(step, price, useful_heat_mw):
    """Hourly profit with operator's simplified model."""
    if step == 0:
        return 0.0
    ee_revenue = step * price
    heat_revenue = useful_heat_mw * HEAT_CREDIT
    fuel = step * FUEL_GROSS
    return ee_revenue + heat_revenue - fuel


# ---------------------------------------------------------------------------
# For counterfactual dispatch: at a given step, how much useful heat?
# Useful heat = min(heat_demand, boiler_thermal - EE)
# Heat demand is exogenous (from the network). We observe it as heat_MW.
# If we change step, heat demand stays the same, but boiler output changes.
# If boiler output > heat demand: excess goes to dissipator
# If boiler output < heat demand: can't serve all heat (shouldn't happen with biomass)
# ---------------------------------------------------------------------------

def counterfactual_useful_heat(step, heat_demand_mw):
    """If running at this step, how much useful heat can be delivered?"""
    if step == 0:
        return 0.0
    thermal = BOILER_THERMAL[step]
    # Available heat = total thermal - electricity (rest is heat/cooling)
    available_heat = thermal - step  # rough: EE extraction reduces available heat
    # But actually: total thermal = EE + heat + cooling, so available = thermal - EE
    # available for heat/cooling
    available_for_heat = max(0, thermal - step)
    # Useful heat = min of what's available and what's demanded
    return min(available_for_heat, heat_demand_mw)


# ---------------------------------------------------------------------------
# Compute actual profits
# ---------------------------------------------------------------------------
print("[*] Computing actual profits...")

df["actual_profit_phys"] = df.apply(
    lambda r: profit_physics(r["step"], r["da_price_eur"], r["heat_MW"]), axis=1
)
df["actual_profit_oper"] = df.apply(
    lambda r: profit_operator(r["step"], r["da_price_eur"], r["heat_MW"]), axis=1
)

# ---------------------------------------------------------------------------
# Greedy optimal: best step each hour (no constraints)
# ---------------------------------------------------------------------------
print("[*] Computing greedy optimal (physics model)...")

def best_step_physics(price, heat_demand):
    best_s, best_p = 0, 0.0
    for s in range(1, 9):
        useful_heat = counterfactual_useful_heat(s, heat_demand)
        p = profit_physics(s, price, useful_heat)
        if p > best_p:
            best_s, best_p = s, p
    return best_s, best_p

greedy_results = df.apply(
    lambda r: best_step_physics(r["da_price_eur"], r["heat_MW"]), axis=1
)
df["greedy_step_phys"] = [r[0] for r in greedy_results]
df["greedy_profit_phys"] = [r[1] for r in greedy_results]

# Same for operator model
def best_step_operator(price, heat_demand):
    best_s, best_p = 0, 0.0
    for s in range(1, 9):
        p = profit_operator(s, price, heat_demand)
        if p > best_p:
            best_s, best_p = s, p
    return best_s, best_p

greedy_results_oper = df.apply(
    lambda r: best_step_operator(r["da_price_eur"], r["heat_MW"]), axis=1
)
df["greedy_step_oper"] = [r[0] for r in greedy_results_oper]
df["greedy_profit_oper"] = [r[1] for r in greedy_results_oper]

# ---------------------------------------------------------------------------
# Boiler-constrained DP (physics model)
# ---------------------------------------------------------------------------
print("[*] Running boiler-constrained DP (physics model)...")

n = len(df)
prices_arr = df["da_price_eur"].values
heat_arr = df["heat_MW"].values

# States: (regime, warmup) where regime in {0,1,2}, warmup in {0..6}
MAX_WARMUP = 6
COLD_START_WARMUP = 6
BOILER2_WARMUP = 4
COLD_START_MAINT = 500
BOILER2_START_MAINT = 200

STATES = []
STATE_IDX = {}
for r in range(3):
    for w in range(MAX_WARMUP + 1):
        STATE_IDX[(r, w)] = len(STATES)
        STATES.append((r, w))
N_STATES = len(STATES)


def best_in_regime_phys(regime, warmup, price, heat_demand):
    """Best step and profit within a regime."""
    if regime == 0:
        return 0, 0.0
    elif regime == 1:
        if warmup > 0:
            s = 1
            uh = counterfactual_useful_heat(s, heat_demand)
            return s, profit_physics(s, price, uh)
        else:
            best_s, best_p = 0, 0.0
            for s in range(1, 5):
                uh = counterfactual_useful_heat(s, heat_demand)
                p = profit_physics(s, price, uh)
                if p > best_p:
                    best_s, best_p = s, p
            # Also consider: not running (0) even if in 1-boiler regime
            # This represents hot standby cost vs shutdown
            if best_p <= 0:
                return 0, 0.0
            return best_s, best_p
    else:  # regime 2
        if warmup > 0:
            s = 5
            uh = counterfactual_useful_heat(s, heat_demand)
            return s, profit_physics(s, price, uh)
        else:
            best_s, best_p = 0, 0.0
            for s in range(5, 9):
                uh = counterfactual_useful_heat(s, heat_demand)
                p = profit_physics(s, price, uh)
                if p > best_p:
                    best_s, best_p = s, p
            if best_p <= 0:
                return 0, 0.0
            return best_s, best_p


# DP backward pass
profit_table = np.full((n, N_STATES), -np.inf)
choice_table = np.zeros((n, N_STATES), dtype=int)
step_table = np.zeros((n, N_STATES), dtype=int)

for si, (r, w) in enumerate(STATES):
    s, p = best_in_regime_phys(r, w, prices_arr[n-1], heat_arr[n-1])
    profit_table[n-1][si] = p
    step_table[n-1][si] = s

for t in range(n-2, -1, -1):
    price = prices_arr[t]
    heat = heat_arr[t]
    for si, (r, w) in enumerate(STATES):
        s, immediate = best_in_regime_phys(r, w, price, heat)
        best_future = -np.inf
        best_next_si = si

        # Stay in same regime
        if r == 0:
            nsi = STATE_IDX[(0, 0)]
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi
        elif r == 1:
            nsi = STATE_IDX[(1, max(0, w-1))]
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi
        elif r == 2:
            nsi = STATE_IDX[(2, max(0, w-1))]
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi

        # Switch regime
        if r == 0:
            nsi = STATE_IDX[(1, COLD_START_WARMUP)]
            future = profit_table[t+1][nsi] - COLD_START_MAINT
            if future > best_future:
                best_future = future
                best_next_si = nsi
        elif r == 1:
            # -> OFF
            nsi = STATE_IDX[(0, 0)]
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi
            # -> 2-boiler
            if w == 0:
                nsi = STATE_IDX[(2, BOILER2_WARMUP)]
                future = profit_table[t+1][nsi] - BOILER2_START_MAINT
                if future > best_future:
                    best_future = future
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
df["dp_profit_phys"] = df.apply(
    lambda r: profit_physics(
        r["dp_step"], r["da_price_eur"],
        counterfactual_useful_heat(r["dp_step"], r["heat_MW"])
    ), axis=1
)

# ---------------------------------------------------------------------------
# Heat-must-run constraint: need minimum output to serve heat
# ---------------------------------------------------------------------------
# What's the minimum step needed to satisfy heat demand?
# If heat demand > available heat at step 0... they must run
df["heat_constrained"] = df["heat_MW"] > 1.0  # need at least some output for heat

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2025 DISPATCH RESULTS - REAL COST MODEL")
print("=" * 70)

# Full year
for label, profit_col in [
    ("Actual", "actual_profit_phys"),
    ("Greedy optimal (physics)", "greedy_profit_phys"),
    ("DP optimal (boiler constrained)", "dp_profit_phys"),
    ("Actual (operator model)", "actual_profit_oper"),
    ("Greedy (operator model)", "greedy_profit_oper"),
]:
    total = df[profit_col].sum() / 1000
    print(f"  {label:40s}: {total:8.1f} kEUR")

gap_phys = (df["dp_profit_phys"].sum() - df["actual_profit_phys"].sum()) / 1000
gap_oper = (df["greedy_profit_oper"].sum() - df["actual_profit_oper"].sum()) / 1000
gap_greedy = (df["greedy_profit_phys"].sum() - df["actual_profit_phys"].sum()) / 1000
print(f"\n  Gap (physics DP):     {gap_phys:.0f} kEUR")
print(f"  Gap (physics greedy): {gap_greedy:.0f} kEUR")
print(f"  Gap (operator model): {gap_oper:.0f} kEUR")

# Monthly
print("\n" + "=" * 70)
print("MONTHLY BREAKDOWN (physics model, kEUR)")
print("=" * 70)

df["month"] = df["datetime"].dt.month
monthly = df.groupby("month").agg(
    actual=("actual_profit_phys", "sum"),
    dp=("dp_profit_phys", "sum"),
    greedy=("greedy_profit_phys", "sum"),
    mean_price=("da_price_eur", "mean"),
    mean_heat=("heat_MW", "mean"),
)
monthly[["actual", "dp", "greedy"]] /= 1000
monthly["gap_dp"] = monthly["dp"] - monthly["actual"]
monthly["gap_pct"] = (monthly["gap_dp"] / monthly["dp"].clip(lower=1) * 100).round(0)
print(monthly[["actual", "dp", "greedy", "gap_dp", "gap_pct", "mean_price", "mean_heat"]].round(1).to_string())

# Step distributions
print("\n" + "=" * 70)
print("STEP DISTRIBUTION")
print("=" * 70)

for label, col in [("Actual", "step"), ("DP optimal", "dp_step"), ("Greedy", "greedy_step_phys")]:
    dist = df[col].value_counts(normalize=True).sort_index() * 100
    print(f"\n{label}:")
    for step in range(0, 9):
        pct = dist.get(step, 0)
        bar = "#" * int(pct / 2)
        print(f"  {step} MW: {pct:5.1f}% {bar}")

# Mistake analysis
print("\n" + "=" * 70)
print("MISTAKE ANALYSIS (where actual step != DP optimal)")
print("=" * 70)

df["gap_phys"] = df["dp_profit_phys"] - df["actual_profit_phys"]
df["wrong"] = df["step"] != df["dp_step"]
wrong = df[df["wrong"]]
print(f"Wrong-step hours: {len(wrong):,} ({len(wrong)/len(df)*100:.1f}%)")
print(f"Total gap from mistakes: {wrong['gap_phys'].sum()/1000:.1f} kEUR")

# Top mistake patterns
wrong_copy = wrong.copy()
wrong_copy["pattern"] = wrong_copy["step"].astype(str) + "MW->" + wrong_copy["dp_step"].astype(str) + "MW"
patterns = wrong_copy.groupby("pattern").agg(
    count=("gap_phys", "count"),
    total_gap=("gap_phys", "sum"),
    avg_price=("da_price_eur", "mean"),
    avg_heat=("heat_MW", "mean"),
).sort_values("total_gap", ascending=False)
patterns["total_gap"] /= 1000

print("\nTop mistakes (actual -> optimal, count, gap, avg price, avg heat):")
for pat, row in patterns.head(15).iterrows():
    print(f"  {pat:12s}: {row['count']:4.0f}h, {row['total_gap']:6.1f} kEUR, "
          f"price={row['avg_price']:.0f}, heat={row['avg_heat']:.1f} MW")

# Seasonal gap
print("\n" + "=" * 70)
print("SEASONAL GAP")
print("=" * 70)

df["season"] = df["month"].apply(lambda m: "winter" if m >= 10 or m <= 3 else "summer")
seasonal = df.groupby("season").agg(
    actual=("actual_profit_phys", "sum"),
    dp=("dp_profit_phys", "sum"),
    hours=("step", "count"),
)
seasonal[["actual", "dp"]] /= 1000
seasonal["gap"] = seasonal["dp"] - seasonal["actual"]
print(seasonal.to_string())

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
print("\n[*] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Bardejov CHP - Real Cost Model Dispatch Optimization (2025)",
             fontsize=14, fontweight="bold")

# 1. Breakeven by step and season
ax = axes[0, 0]
for season, color, marker in [("winter", "blue", "s"), ("summer", "red", "o")]:
    sdf = df[(df["season"] == season) & (df["step"] > 0)]
    breakevens = []
    for step in range(1, 9):
        sdf_s = sdf[sdf["step"] == step]
        if len(sdf_s) > 10:
            total_th = (sdf_s["ee_MW"] + sdf_s["heat_MW"] + sdf_s["cool_MW"].clip(lower=0)).mean()
            fuel = (total_th / BOILER_EFF / MWH_PER_ATT) * CHIP_PRICE
            heat_rev = sdf_s["heat_MW"].mean() * HEAT_CREDIT
            be = (fuel - heat_rev) / step
            breakevens.append((step, be))
    if breakevens:
        steps, bes = zip(*breakevens)
        ax.plot(steps, bes, f"{color[0]}-{marker}", label=season, markersize=8)
ax.set_xlabel("MW Step")
ax.set_ylabel("Breakeven Price (EUR/MWh)")
ax.set_title("Electricity Breakeven by Step & Season")
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(100, color="gray", linestyle=":", alpha=0.5, label="100 EUR ref")

# 2. Step distribution
ax = axes[0, 1]
steps_range = range(0, 9)
width = 0.35
actual_dist = df["step"].value_counts(normalize=True).reindex(steps_range, fill_value=0)
dp_dist = df["dp_step"].value_counts(normalize=True).reindex(steps_range, fill_value=0)
ax.bar([s - width/2 for s in steps_range], actual_dist * 100, width,
       label="Actual", color="red", alpha=0.7)
ax.bar([s + width/2 for s in steps_range], dp_dist * 100, width,
       label="DP Optimal", color="green", alpha=0.7)
ax.set_xlabel("MW Step")
ax.set_ylabel("% of Hours")
ax.set_title("Step Distribution: Actual vs Optimal")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Monthly profit comparison
ax = axes[1, 0]
m_plot = monthly[["actual", "dp"]].copy()
m_plot.plot(kind="bar", ax=ax, color=["red", "green"], alpha=0.7)
ax.set_xlabel("Month")
ax.set_ylabel("Profit (kEUR)")
ax.set_title("Monthly Profit: Actual vs Boiler-Constrained DP")
ax.legend(["Actual", "Optimal (DP)"])
ax.grid(True, alpha=0.3, axis="y")

# 4. Gap decomposition
ax = axes[1, 1]
top_patterns = patterns.head(8)
colors_bar = plt.cm.Set2(range(len(top_patterns)))
bars = ax.barh(range(len(top_patterns)), top_patterns["total_gap"], color=colors_bar)
ax.set_yticks(range(len(top_patterns)))
ax.set_yticklabels(top_patterns.index)
ax.set_xlabel("Gap (kEUR)")
ax.set_title("Top Dispatch Mistakes by Gap")
ax.grid(True, alpha=0.3, axis="x")
for i, (_, row) in enumerate(top_patterns.iterrows()):
    ax.text(row["total_gap"] + 1, i, f"{row['total_gap']:.0f}k, p={row['avg_price']:.0f}",
            va="center", fontsize=8)

plt.tight_layout()
plt.savefig(PLOT_DIR / "06_real_cost_model.png", dpi=150)
plt.close()
print(f"[+] Saved {PLOT_DIR / '06_real_cost_model.png'}")

# Weekly gap
fig, ax = plt.subplots(figsize=(14, 5))
weekly = df.set_index("datetime").resample("W").agg({
    "actual_profit_phys": "sum",
    "dp_profit_phys": "sum",
}) / 1000
ax.fill_between(weekly.index, weekly["actual_profit_phys"], weekly["dp_profit_phys"],
                where=weekly["dp_profit_phys"] > weekly["actual_profit_phys"],
                alpha=0.3, color="green", label="Could have earned more")
ax.fill_between(weekly.index, weekly["actual_profit_phys"], weekly["dp_profit_phys"],
                where=weekly["dp_profit_phys"] <= weekly["actual_profit_phys"],
                alpha=0.3, color="blue", label="Beat optimal (noise)")
ax.plot(weekly.index, weekly["actual_profit_phys"], "r-", linewidth=1, label="Actual")
ax.plot(weekly.index, weekly["dp_profit_phys"], "g-", linewidth=1, label="Optimal (DP)")
ax.set_ylabel("Weekly Profit (kEUR)")
ax.set_title("Weekly Profit: Actual vs Boiler-Constrained Optimal (Physics Model)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "07_weekly_real_costs.png", dpi=150)
plt.close()
print(f"[+] Saved {PLOT_DIR / '07_weekly_real_costs.png'}")

# Dissipation analysis
fig, ax = plt.subplots(figsize=(14, 5))
monthly_diss = df.set_index("datetime").resample("ME").agg({
    "cool_MW": lambda x: x.clip(lower=0).sum(),
    "heat_MW": "sum",
})
monthly_diss.columns = ["Dissipated (MWh)", "Useful Heat (MWh)"]
monthly_diss.plot(kind="bar", stacked=True, ax=ax, color=["red", "green"], alpha=0.7)
ax.set_ylabel("MWh")
ax.set_title("Monthly Heat Balance: Useful vs Dissipated (2025)")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PLOT_DIR / "08_heat_dissipation.png", dpi=150)
plt.close()
print(f"[+] Saved {PLOT_DIR / '08_heat_dissipation.png'}")

print("\n[+] Done!")
