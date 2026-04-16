"""
Bardejov CHP Dispatch Optimization Analysis
=============================================
Compares actual plant dispatch against price-optimal dispatch to quantify
the revenue gap and identify actionable improvements.

Plant specs (from memory/operator data):
- 2x ~18 t/h steam boilers (wood chips, 42% moisture, 78 bar/520C)
- Steam turbine (new May 2024), 1-8 MW electrical
- 1 boiler max = 4 MW EE; 2 boilers needed for 5-8 MW
- Ramp: +0.5 MW/15min up, -1.0 MW/15min down

Cost curve (from Kalkulacie sheets):
- Fuel: 23.08 EUR/MWh_th, boiler eff: 85%, own consumption: 8%, ash: 0.69 EUR/MWh
- Net cost per MW step varies nonlinearly due to fixed/variable split
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]  # data/Bardejov
PLOT_DIR = BASE / "dispatch_analysis" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("[*] Loading data...")

plant = pd.read_csv(BASE / "plant_timeseries.csv", parse_dates=["datetime"])
prices = pd.read_csv(BASE / "da_prices_hourly.csv", parse_dates=["datetime"])

# Resample plant to hourly (it has mixed 15min/hourly)
plant = plant.set_index("datetime").resample("h").mean().reset_index()
plant["electricity_MW"] = plant["electricity_kW"] / 1000
plant["heat_load_MW"] = plant["heat_load_kW"] / 1000
plant["cooling_MW"] = plant["cooling_kW"] / 1000

# Merge on datetime
df = plant.merge(prices, on="datetime", how="inner")
print(f"[+] Merged dataset: {len(df)} hours, {df.datetime.min()} to {df.datetime.max()}")

# ---------------------------------------------------------------------------
# 2. Define cost model
# ---------------------------------------------------------------------------
# Cost curve: marginal cost of electricity at each MW step
# From operator Kalkulacie data:
#   Fuel cost = 23.08 EUR/MWh_th
#   Boiler efficiency = 85%
#   Own consumption = 8%
#   Ash disposal = 0.69 EUR/MWh_ee
#
# Heat rate varies by load (partial load penalty)
# At low load: more heat per MWh_ee, but also more useful heat coproduced
# Key: the ELECTRIC cost depends on how much fuel goes to electricity vs heat

# Empirical cost curve from operator data (EUR/MWh_ee at each step)
# These include fuel, ash, own consumption, but NOT heat credit
COST_CURVE = {
    1: 37.7,
    2: 42.1,
    3: 48.3,
    4: 56.8,   # 1-boiler max -- expensive per MWh
    5: 62.4,   # 2 boilers needed
    6: 70.1,
    7: 78.2,
    8: 84.6,
}

# Hot standby cost (keeping 1 boiler warm at minimum load ~1 MW)
HOT_STANDBY_COST_EUR_H = 37.7  # cost of running at 1 MW with no revenue

# Ramp constraints
RAMP_UP_MW_PER_H = 2.0    # 0.5 MW/15min * 4
RAMP_DOWN_MW_PER_H = 4.0  # 1.0 MW/15min * 4


def electricity_revenue(mw, price):
    """Revenue from selling electricity at DA price."""
    return mw * price


def electricity_cost(mw):
    """Cost of producing electricity at given MW step."""
    if mw <= 0:
        return 0
    step = int(round(mw))
    step = max(1, min(8, step))
    return COST_CURVE[step] * mw


def net_profit(mw, price):
    """Net profit per hour at given MW and price."""
    return electricity_revenue(mw, price) - electricity_cost(mw)


# ---------------------------------------------------------------------------
# 3. Map actual generation to discrete steps
# ---------------------------------------------------------------------------
print("[*] Classifying actual dispatch steps...")

# The plant operates at discrete steps (1-8 MW)
# Map continuous readings to nearest step
df["electricity_MW"] = df["electricity_MW"].fillna(0)
df["heat_load_MW"] = df["heat_load_MW"].fillna(0)
df["cooling_MW"] = df["cooling_MW"].fillna(0)
df["actual_step"] = df["electricity_MW"].round().clip(0, 8).astype(int)

# ---------------------------------------------------------------------------
# 4. Compute actual revenue & cost
# ---------------------------------------------------------------------------
print("[*] Computing actual economics...")

df["actual_revenue"] = df["electricity_MW"] * df["da_price_eur"]
df["actual_cost"] = df["actual_step"].map(COST_CURVE).fillna(0) * df["electricity_MW"]
# For hours where plant is off (step 0), cost = 0
df.loc[df["actual_step"] == 0, "actual_cost"] = 0
df["actual_profit"] = df["actual_revenue"] - df["actual_cost"]

# ---------------------------------------------------------------------------
# 5. Optimal dispatch (no ramp constraints) -- upper bound
# ---------------------------------------------------------------------------
print("[*] Computing unconstrained optimal dispatch...")

def optimal_step_no_ramp(price):
    """Find the MW step that maximizes profit given price."""
    best_profit = 0  # shutting down = 0 profit
    best_step = 0
    for step in range(1, 9):
        p = net_profit(step, price)
        if p > best_profit:
            best_profit = p
            best_step = step
    return best_step

df["optimal_step_unconstrained"] = df["da_price_eur"].apply(optimal_step_no_ramp)
df["optimal_profit_unconstrained"] = df.apply(
    lambda r: net_profit(r["optimal_step_unconstrained"], r["da_price_eur"]), axis=1
)

# ---------------------------------------------------------------------------
# 6. Optimal dispatch WITH ramp constraints -- dynamic programming
# ---------------------------------------------------------------------------
print("[*] Computing ramp-constrained optimal dispatch (DP)...")

n = len(df)
prices_arr = df["da_price_eur"].values
steps = list(range(0, 9))  # 0 = off, 1-8 = running

# profit[t][s] = max cumulative profit from t to end, being at step s at time t
# We solve backwards
profit_table = np.full((n, 9), -np.inf)
choice_table = np.zeros((n, 9), dtype=int)

# Initialize last hour
for s in steps:
    profit_table[n-1][s] = net_profit(s, prices_arr[n-1])

# Backward pass
for t in range(n-2, -1, -1):
    for s in steps:
        immediate = net_profit(s, prices_arr[t])
        best_future = -np.inf
        best_next = s
        for s_next in steps:
            # Check ramp feasibility
            if s_next > s and (s_next - s) > RAMP_UP_MW_PER_H:
                continue
            if s_next < s and (s - s_next) > RAMP_DOWN_MW_PER_H:
                continue
            if profit_table[t+1][s_next] > best_future:
                best_future = profit_table[t+1][s_next]
                best_next = s_next
        profit_table[t][s] = immediate + best_future
        choice_table[t][s] = best_next

# Forward pass: extract optimal path
optimal_ramped = np.zeros(n, dtype=int)
# Start from best initial state
optimal_ramped[0] = np.argmax(profit_table[0])
for t in range(1, n):
    optimal_ramped[t] = choice_table[t-1][optimal_ramped[t-1]]

df["optimal_step_ramped"] = optimal_ramped
df["optimal_profit_ramped"] = df.apply(
    lambda r: net_profit(r["optimal_step_ramped"], r["da_price_eur"]), axis=1
)

# ---------------------------------------------------------------------------
# 7. Simple threshold rules
# ---------------------------------------------------------------------------
print("[*] Computing threshold-based dispatch rules...")

# 3-threshold rule (from memory of previous analysis)
# Winter (Oct-Mar): <41=1MW, 96-104=5MW, >104=8MW
# Summer (Apr-Sep): <83=1MW, >83=4MW
def threshold_rule(row):
    price = row["da_price_eur"]
    month = row["datetime"].month
    is_winter = month >= 10 or month <= 3

    if is_winter:
        if price < 41:
            return 1  # hot standby
        elif price < 96:
            return 4  # hmm, let's test various thresholds
        elif price < 104:
            return 5
        else:
            return 8
    else:
        if price < 41:
            return 0  # summer off or 1MW standby
        elif price < 83:
            return 1
        else:
            return 4

df["threshold_step"] = df.apply(threshold_rule, axis=1)
df["threshold_profit"] = df.apply(
    lambda r: net_profit(r["threshold_step"], r["da_price_eur"]), axis=1
)

# Also test: simple breakeven rule (run only when price > marginal cost at that step)
# Pick the highest profitable step
def greedy_step(price):
    """Run at highest step where price > marginal cost."""
    best_step = 0
    best_profit = 0
    for s in range(1, 9):
        p = net_profit(s, price)
        if p > best_profit:
            best_profit = p
            best_step = s
    return best_step

df["greedy_step"] = df["da_price_eur"].apply(greedy_step)
df["greedy_profit"] = df.apply(
    lambda r: net_profit(r["greedy_step"], r["da_price_eur"]), axis=1
)

# ---------------------------------------------------------------------------
# 8. Aggregate results
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DISPATCH OPTIMIZATION RESULTS")
print("=" * 70)

# Annual summaries
df["year"] = df["datetime"].dt.year
for year in sorted(df["year"].unique()):
    ydf = df[df["year"] == year]
    hours = len(ydf)
    actual_rev = ydf["actual_profit"].sum() / 1000
    optimal_unc = ydf["optimal_profit_unconstrained"].sum() / 1000
    optimal_ramp = ydf["optimal_profit_ramped"].sum() / 1000
    threshold = ydf["threshold_profit"].sum() / 1000
    greedy = ydf["greedy_profit"].sum() / 1000

    print(f"\n--- {year} ({hours} hours) ---")
    print(f"  Actual dispatch profit:        {actual_rev:8.1f} kEUR")
    print(f"  Optimal (no ramps):            {optimal_unc:8.1f} kEUR")
    print(f"  Optimal (with ramps):          {optimal_ramp:8.1f} kEUR")
    print(f"  Threshold rule:                {threshold:8.1f} kEUR")
    print(f"  Greedy (best step each hour):  {greedy:8.1f} kEUR")
    print(f"  Gap (actual vs optimal-ramp):  {optimal_ramp - actual_rev:8.1f} kEUR")

# Full period
print(f"\n--- FULL PERIOD ---")
total_actual = df["actual_profit"].sum() / 1000
total_opt_unc = df["optimal_profit_unconstrained"].sum() / 1000
total_opt_ramp = df["optimal_profit_ramped"].sum() / 1000
total_threshold = df["threshold_profit"].sum() / 1000
total_greedy = df["greedy_profit"].sum() / 1000
print(f"  Actual dispatch profit:        {total_actual:8.1f} kEUR")
print(f"  Optimal (no ramps):            {total_opt_unc:8.1f} kEUR")
print(f"  Optimal (with ramps):          {total_opt_ramp:8.1f} kEUR")
print(f"  Threshold rule:                {total_threshold:8.1f} kEUR")
print(f"  Greedy (best step each hour):  {total_greedy:8.1f} kEUR")
print(f"  Total gap (actual vs ramp):    {total_opt_ramp - total_actual:8.1f} kEUR")

# ---------------------------------------------------------------------------
# 9. Step distribution analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP DISTRIBUTION (% of hours)")
print("=" * 70)

for label, col in [("Actual", "actual_step"),
                   ("Optimal (ramped)", "optimal_step_ramped"),
                   ("Greedy", "greedy_step")]:
    dist = df[col].value_counts(normalize=True).sort_index() * 100
    print(f"\n{label}:")
    for step in range(0, 9):
        pct = dist.get(step, 0)
        bar = "#" * int(pct / 2)
        print(f"  {step} MW: {pct:5.1f}% {bar}")

# ---------------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------------
print("\n[*] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Bardejov CHP Dispatch Optimization", fontsize=14, fontweight="bold")

# Plot 1: Price vs optimal step
ax = axes[0, 0]
price_range = np.arange(0, 200, 1)
opt_steps = [optimal_step_no_ramp(p) for p in price_range]
ax.plot(price_range, opt_steps, "b-", linewidth=2)
ax.set_xlabel("DA Price (EUR/MWh)")
ax.set_ylabel("Optimal MW Step")
ax.set_title("Optimal Step vs DA Price")
ax.grid(True, alpha=0.3)
# Add breakeven lines
for step, cost in COST_CURVE.items():
    ax.axvline(cost, color="gray", linestyle="--", alpha=0.3)
    ax.text(cost + 0.5, step, f"{cost:.0f}", fontsize=7, alpha=0.5)

# Plot 2: Step distribution comparison
ax = axes[0, 1]
steps_range = range(0, 9)
width = 0.25
actual_dist = df["actual_step"].value_counts(normalize=True).reindex(steps_range, fill_value=0)
optimal_dist = df["optimal_step_ramped"].value_counts(normalize=True).reindex(steps_range, fill_value=0)
greedy_dist = df["greedy_step"].value_counts(normalize=True).reindex(steps_range, fill_value=0)

ax.bar([s - width for s in steps_range], actual_dist * 100, width, label="Actual", color="red", alpha=0.7)
ax.bar([s for s in steps_range], optimal_dist * 100, width, label="Optimal (DP)", color="green", alpha=0.7)
ax.bar([s + width for s in steps_range], greedy_dist * 100, width, label="Greedy", color="blue", alpha=0.7)
ax.set_xlabel("MW Step")
ax.set_ylabel("% of Hours")
ax.set_title("Step Distribution: Actual vs Optimal")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Monthly profit comparison
ax = axes[1, 0]
monthly = df.set_index("datetime").resample("ME").agg({
    "actual_profit": "sum",
    "optimal_profit_ramped": "sum",
    "greedy_profit": "sum",
}) / 1000
monthly.plot(ax=ax, linewidth=1.5)
ax.set_ylabel("Profit (kEUR)")
ax.set_title("Monthly Profit: Actual vs Optimal vs Greedy")
ax.legend(["Actual", "Optimal (DP)", "Greedy"], fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 4: Cost curve with price distribution
ax = axes[1, 1]
ax2 = ax.twinx()
# Cost curve
steps_list = sorted(COST_CURVE.keys())
costs_list = [COST_CURVE[s] for s in steps_list]
ax.bar(steps_list, costs_list, color="coral", alpha=0.7, label="Marginal cost")
ax.set_xlabel("MW Step")
ax.set_ylabel("Cost (EUR/MWh)", color="coral")
ax.set_title("Cost Curve vs Price Distribution")
# Price histogram on twin axis
ax2.hist(df["da_price_eur"].clip(-20, 200), bins=80, color="steelblue", alpha=0.4, density=True)
ax2.set_ylabel("Price density", color="steelblue")
ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig(PLOT_DIR / "01_dispatch_overview.png", dpi=150)
plt.close()
print(f"[+] Saved {PLOT_DIR / '01_dispatch_overview.png'}")

# Plot 5: Profit heatmap by hour and month
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Profit Gap by Hour and Month (kEUR)", fontsize=13, fontweight="bold")

df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour
df["profit_gap"] = df["optimal_profit_ramped"] - df["actual_profit"]

for idx, (label, col) in enumerate([
    ("Actual Profit", "actual_profit"),
    ("Optimal Profit (DP)", "optimal_profit_ramped"),
    ("Missed Profit (Gap)", "profit_gap"),
]):
    pivot = df.pivot_table(values=col, index="hour", columns="month", aggfunc="sum") / 1000
    im = axes[idx].imshow(pivot.values, aspect="auto", cmap="RdYlGn" if "Gap" not in label else "Reds")
    axes[idx].set_title(label)
    axes[idx].set_xlabel("Month")
    axes[idx].set_ylabel("Hour")
    axes[idx].set_xticks(range(12))
    axes[idx].set_xticklabels(range(1, 13))
    axes[idx].set_yticks(range(0, 24, 3))
    plt.colorbar(im, ax=axes[idx], shrink=0.8)

plt.tight_layout()
plt.savefig(PLOT_DIR / "02_profit_heatmap.png", dpi=150)
plt.close()
print(f"[+] Saved {PLOT_DIR / '02_profit_heatmap.png'}")

# Plot 6: Scatter of actual vs optimal by price band
fig, ax = plt.subplots(figsize=(12, 6))

# Weekly rolling profit
weekly = df.set_index("datetime").resample("W").agg({
    "actual_profit": "sum",
    "optimal_profit_ramped": "sum",
}) / 1000
ax.fill_between(weekly.index, weekly["actual_profit"], weekly["optimal_profit_ramped"],
                alpha=0.3, color="red", label="Gap")
ax.plot(weekly.index, weekly["actual_profit"], "r-", linewidth=1, label="Actual")
ax.plot(weekly.index, weekly["optimal_profit_ramped"], "g-", linewidth=1, label="Optimal (DP)")
ax.set_ylabel("Weekly Profit (kEUR)")
ax.set_title("Weekly Profit: Actual vs Optimal Dispatch")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_weekly_gap.png", dpi=150)
plt.close()
print(f"[+] Saved {PLOT_DIR / '03_weekly_gap.png'}")

# ---------------------------------------------------------------------------
# 11. Breakeven analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BREAKEVEN PRICES (EUR/MWh)")
print("=" * 70)
print("Step | Cost/MWh | Breakeven | Profit @80 | Profit @120")
print("-----|----------|-----------|------------|------------")
for step in range(1, 9):
    cost = COST_CURVE[step]
    p80 = net_profit(step, 80)
    p120 = net_profit(step, 120)
    print(f"  {step}  |  {cost:5.1f}   |  {cost:5.1f}    | {p80:7.1f}    | {p120:7.1f}")

# ---------------------------------------------------------------------------
# 12. The "4 MW dead zone" analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4 MW DEAD ZONE ANALYSIS")
print("=" * 70)

at_4 = df[df["actual_step"] == 4]
opt_at_4_hours = df[df["optimal_step_ramped"] == 4]
print(f"Hours plant ran at 4 MW:          {len(at_4):,} ({len(at_4)/len(df)*100:.1f}%)")
print(f"Hours optimal dispatch uses 4 MW: {len(opt_at_4_hours):,} ({len(opt_at_4_hours)/len(df)*100:.1f}%)")

if len(at_4) > 0:
    # What should they have done instead?
    better = df[df["actual_step"] == 4]["optimal_step_ramped"].value_counts().sort_index()
    print("\nWhen actually at 4 MW, optimal says:")
    for step, count in better.items():
        print(f"  -> {step} MW: {count:,} hours ({count/len(at_4)*100:.1f}%)")

    # Cost of the mistake
    at_4_actual_profit = at_4["actual_profit"].sum()
    at_4_optimal_profit = at_4["optimal_profit_ramped"].sum()
    print(f"\nProfit at 4 MW hours:")
    print(f"  Actual:  {at_4_actual_profit/1000:.1f} kEUR")
    print(f"  Optimal: {at_4_optimal_profit/1000:.1f} kEUR")
    print(f"  Gap:     {(at_4_optimal_profit - at_4_actual_profit)/1000:.1f} kEUR")

# ---------------------------------------------------------------------------
# 13. Dissipator (cooling) analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DISSIPATOR / COOLING ANALYSIS")
print("=" * 70)

dissipating = df[df["cooling_MW"] > 0.5]
print(f"Hours with significant cooling (>0.5 MW): {len(dissipating):,} ({len(dissipating)/len(df)*100:.1f}%)")
if len(dissipating) > 0:
    print(f"Mean cooling when active: {dissipating['cooling_MW'].mean():.1f} MW")
    print(f"Total heat dissipated: {dissipating['cooling_MW'].sum():.0f} MWh")
    # Cooling by month
    monthly_cool = dissipating.set_index("datetime").resample("ME")["cooling_MW"].sum()
    print("\nMonthly dissipated heat (MWh):")
    for dt, val in monthly_cool.items():
        if val > 100:
            print(f"  {dt.strftime('%Y-%m')}: {val:,.0f} MWh")

print("\n[+] Analysis complete!")
print(f"[+] Plots saved to: {PLOT_DIR}")
