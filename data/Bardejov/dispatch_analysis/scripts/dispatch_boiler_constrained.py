"""
Bardejov CHP - Boiler-Constrained Dispatch Optimization
=========================================================
Proper DP that models the REAL constraints:
  - 3 regimes: OFF, 1-boiler (1-4 MW), 2-boiler (5-8 MW)
  - Cold start (OFF -> 1-boiler): 6h warmup at 1 MW before free dispatch
  - Second boiler start (1-boiler -> 2-boiler): 4h at 5 MW before free dispatch
  - Shutdown: instant (can ramp down within 1h)
  - Within-regime step changes: instant (ramp rates fast enough within regime)
  - Startup cost: extra fuel during warmup, maintenance wear
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
# Cost model (same as before)
# ---------------------------------------------------------------------------
COST_CURVE = {
    0: 0,
    1: 37.7, 2: 42.1, 3: 48.3, 4: 56.8,
    5: 62.4, 6: 70.1, 7: 78.2, 8: 84.6,
}

# Startup costs (fuel burned during warmup with minimal/no useful output)
COLD_START_COST = 6 * 37.7   # ~226 EUR: 6h at 1 MW cost with no revenue
BOILER2_START_COST = 4 * 62.4  # ~250 EUR: 4h warming second boiler at ~5 MW cost

# Cycling maintenance penalty (boiler wear per start)
COLD_START_MAINT = 500    # EUR per cold start (conservative)
BOILER2_START_MAINT = 200  # EUR per second boiler start


def net_profit(step, price):
    if step == 0:
        return 0
    return step * price - COST_CURVE[step] * step


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("[*] Loading data...")
plant = pd.read_csv(BASE / "plant_timeseries.csv", parse_dates=["datetime"])
prices = pd.read_csv(BASE / "da_prices_hourly.csv", parse_dates=["datetime"])

plant = plant.set_index("datetime").resample("h").mean().reset_index()
plant["MW"] = plant["electricity_kW"].fillna(0) / 1000
plant["step"] = plant["MW"].round().clip(0, 8).astype(int)

df = plant.merge(prices, on="datetime", how="inner")
print(f"[+] {len(df)} hours, {df.datetime.min()} to {df.datetime.max()}")

# Actual regime
def to_regime(step):
    if step == 0: return 0
    elif step <= 4: return 1
    else: return 2

df["actual_regime"] = df["step"].apply(to_regime)

# ---------------------------------------------------------------------------
# Boiler-constrained DP
# ---------------------------------------------------------------------------
# State: (regime, warmup_remaining)
#   regime 0: OFF
#   regime 1: 1-boiler, warmup_remaining = hours until free dispatch (0 = warm)
#   regime 2: 2-boiler, warmup_remaining = hours until free dispatch (0 = warm)
#
# Simplification: discretize warmup into integer hours
# State space: (regime, warmup) where warmup in {0, 1, 2, ..., 6}

print("[*] Running boiler-constrained DP...")

n = len(df)
prices_arr = df["da_price_eur"].values

MAX_WARMUP = 6  # max warmup hours to track
COLD_START_WARMUP = 6   # hours at minimum load before free dispatch
BOILER2_WARMUP = 4      # hours at 5 MW before free dispatch

# States: (regime, warmup) -- regime in {0,1,2}, warmup in {0..6}
# Total states: 3 * 7 = 21
STATES = []
STATE_IDX = {}
for r in range(3):
    for w in range(MAX_WARMUP + 1):
        idx = len(STATES)
        STATES.append((r, w))
        STATE_IDX[(r, w)] = idx

N_STATES = len(STATES)

def best_step_in_regime(regime, warmup, price):
    """Best MW step and its profit given regime and warmup status."""
    if regime == 0:
        return 0, 0.0
    elif regime == 1:
        if warmup > 0:
            # Warming up: stuck at 1 MW
            return 1, net_profit(1, price)
        else:
            # Free to choose 1-4 MW
            best_s, best_p = 1, net_profit(1, price)
            for s in range(2, 5):
                p = net_profit(s, price)
                if p > best_p:
                    best_s, best_p = s, p
            return best_s, best_p
    else:  # regime 2
        if warmup > 0:
            # Warming up second boiler: stuck at 5 MW
            return 5, net_profit(5, price)
        else:
            # Free to choose 5-8 MW (or drop to 1-boiler steps)
            # Actually if 2 boilers running, can also run at 1-4 (just 1 boiler active)
            # But keeping 2nd boiler hot has a cost -- model as: stay in regime 2 at 5+ MW
            best_s, best_p = 5, net_profit(5, price)
            for s in range(6, 9):
                p = net_profit(s, price)
                if p > best_p:
                    best_s, best_p = s, p
            return best_s, best_p


# DP: profit_table[t][state_idx] = max cumulative profit from t to end
profit_table = np.full((n, N_STATES), -np.inf)
choice_table = np.zeros((n, N_STATES), dtype=int)  # next state index
step_table = np.zeros((n, N_STATES), dtype=int)     # MW step chosen

# Last hour
for si, (r, w) in enumerate(STATES):
    s, p = best_step_in_regime(r, w, prices_arr[n-1])
    profit_table[n-1][si] = p
    step_table[n-1][si] = s

# Backward pass
for t in range(n-2, -1, -1):
    price = prices_arr[t]
    for si, (r, w) in enumerate(STATES):
        s, immediate = best_step_in_regime(r, w, price)

        # Possible next states
        best_future = -np.inf
        best_next_si = si

        # Option 1: Stay in same regime
        if r == 0:
            # Stay off
            nsi = STATE_IDX[(0, 0)]
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi
        elif r == 1:
            next_w = max(0, w - 1)
            nsi = STATE_IDX[(1, next_w)]
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi
        elif r == 2:
            next_w = max(0, w - 1)
            nsi = STATE_IDX[(2, next_w)]
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi

        # Option 2: Switch regime
        if r == 0:
            # OFF -> 1-boiler (cold start)
            nsi = STATE_IDX[(1, COLD_START_WARMUP)]
            transition_cost = COLD_START_MAINT
            future = profit_table[t+1][nsi] - transition_cost
            if future > best_future:
                best_future = future
                best_next_si = nsi

        elif r == 1:
            # 1-boiler -> OFF (shutdown)
            nsi = STATE_IDX[(0, 0)]
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi

            # 1-boiler -> 2-boiler (start second boiler)
            if w == 0:  # only if first boiler is warm
                nsi = STATE_IDX[(2, BOILER2_WARMUP)]
                transition_cost = BOILER2_START_MAINT
                future = profit_table[t+1][nsi] - transition_cost
                if future > best_future:
                    best_future = future
                    best_next_si = nsi

        elif r == 2:
            # 2-boiler -> 1-boiler (shut down second boiler, instant)
            nsi = STATE_IDX[(1, 0)]  # first boiler stays warm
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi

            # 2-boiler -> OFF (full shutdown)
            nsi = STATE_IDX[(0, 0)]
            if profit_table[t+1][nsi] > best_future:
                best_future = profit_table[t+1][nsi]
                best_next_si = nsi

        profit_table[t][si] = immediate + best_future
        choice_table[t][si] = best_next_si
        step_table[t][si] = s

# Forward pass
# Start from best initial state (assume cold start = OFF)
optimal_states = np.zeros(n, dtype=int)
optimal_steps = np.zeros(n, dtype=int)

optimal_states[0] = STATE_IDX[(0, 0)]  # start OFF
# Actually, pick best starting state
best_start = np.argmax(profit_table[0])
optimal_states[0] = best_start
optimal_steps[0] = step_table[0][best_start]

for t in range(1, n):
    optimal_states[t] = choice_table[t-1][optimal_states[t-1]]
    optimal_steps[t] = step_table[t][optimal_states[t]]

df["opt_step"] = optimal_steps
df["opt_regime"] = [STATES[s][0] for s in optimal_states]
df["opt_warmup"] = [STATES[s][1] for s in optimal_states]
df["opt_profit"] = df.apply(lambda r: net_profit(r["opt_step"], r["da_price_eur"]), axis=1)

# Actual profit
df["actual_profit"] = df.apply(
    lambda r: net_profit(r["step"], r["da_price_eur"]), axis=1
)

# Unconstrained (for reference)
def greedy_step(price):
    best_s, best_p = 0, 0
    for s in range(1, 9):
        p = net_profit(s, price)
        if p > best_p:
            best_s, best_p = s, p
    return best_s

df["greedy_step"] = df["da_price_eur"].apply(greedy_step)
df["greedy_profit"] = df.apply(lambda r: net_profit(r["greedy_step"], r["da_price_eur"]), axis=1)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BOILER-CONSTRAINED DISPATCH RESULTS")
print("=" * 70)

df["year"] = df["datetime"].dt.year
for year in sorted(df["year"].unique()):
    ydf = df[df["year"] == year]
    h = len(ydf)
    actual = ydf["actual_profit"].sum() / 1000
    opt = ydf["opt_profit"].sum() / 1000
    greedy = ydf["greedy_profit"].sum() / 1000

    print(f"\n--- {year} ({h} hours) ---")
    print(f"  Actual:                  {actual:8.1f} kEUR")
    print(f"  Optimal (boiler DP):     {opt:8.1f} kEUR")
    print(f"  Greedy (no constraints): {greedy:8.1f} kEUR")
    print(f"  Gap (actual vs DP):      {opt - actual:8.1f} kEUR")
    print(f"  Ramp/boiler cost:        {greedy - opt:8.1f} kEUR  (what constraints cost)")

total_actual = df["actual_profit"].sum() / 1000
total_opt = df["opt_profit"].sum() / 1000
total_greedy = df["greedy_profit"].sum() / 1000
print(f"\n--- FULL PERIOD ---")
print(f"  Actual:                  {total_actual:8.1f} kEUR")
print(f"  Optimal (boiler DP):     {total_opt:8.1f} kEUR")
print(f"  Greedy (no constraints): {total_greedy:8.1f} kEUR")
print(f"  Achievable gap:          {total_opt - total_actual:8.1f} kEUR")
print(f"  Constraint cost:         {total_greedy - total_opt:8.1f} kEUR")

# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("REGIME DISTRIBUTION")
print("=" * 70)

for label, rcol in [("Actual", "actual_regime"), ("Optimal", "opt_regime")]:
    dist = df[rcol].value_counts(normalize=True).sort_index() * 100
    print(f"\n{label}:")
    for r in range(3):
        pct = dist.get(r, 0)
        name = ["OFF", "1-boiler (1-4 MW)", "2-boiler (5-8 MW)"][r]
        print(f"  {name}: {pct:.1f}%")

# Count regime transitions
for label, rcol in [("Actual", "actual_regime"), ("Optimal", "opt_regime")]:
    changes = (df[rcol] != df[rcol].shift(1)).sum()
    cold_starts = ((df[rcol].shift(1) == 0) & (df[rcol] > 0)).sum()
    b2_starts = ((df[rcol].shift(1) == 1) & (df[rcol] == 2)).sum()
    print(f"\n{label} transitions:")
    print(f"  Total regime changes: {changes}")
    print(f"  Cold starts (OFF->running): {cold_starts}")
    print(f"  2nd boiler starts (1->2): {b2_starts}")

# ---------------------------------------------------------------------------
# Step distribution
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP DISTRIBUTION")
print("=" * 70)

for label, col in [("Actual", "step"), ("Optimal (boiler DP)", "opt_step")]:
    dist = df[col].value_counts(normalize=True).sort_index() * 100
    print(f"\n{label}:")
    for step in range(0, 9):
        pct = dist.get(step, 0)
        bar = "#" * int(pct / 2)
        print(f"  {step} MW: {pct:5.1f}% {bar}")

# ---------------------------------------------------------------------------
# Where is the remaining gap?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("GAP DECOMPOSITION")
print("=" * 70)

# When actual was suboptimal, what went wrong?
df["gap"] = df["opt_profit"] - df["actual_profit"]
df["month"] = df["datetime"].dt.month
df["season"] = df["month"].apply(lambda m: "winter" if m >= 10 or m <= 3 else "summer")

seasonal_gap = df.groupby(["year", "season"])["gap"].sum() / 1000
print("\nSeasonal gap breakdown (kEUR):")
print(seasonal_gap.to_string())

# Gap by actual regime
print("\nGap by what they were actually doing:")
for r in [0, 1, 2]:
    rdf = df[df["actual_regime"] == r]
    name = ["OFF", "1-boiler", "2-boiler"][r]
    gap = rdf["gap"].sum() / 1000
    hours = len(rdf)
    print(f"  When {name} ({hours:,}h): {gap:.1f} kEUR gap")

# Gap by what they SHOULD have been doing
print("\nGap by what optimal says:")
for r in [0, 1, 2]:
    rdf = df[df["opt_regime"] == r]
    name = ["OFF", "1-boiler", "2-boiler"][r]
    gap = rdf["gap"].sum() / 1000
    hours = len(rdf)
    print(f"  When should be {name} ({hours:,}h): {gap:.1f} kEUR gap")

# Wrong-regime hours
df["wrong_regime"] = df["actual_regime"] != df["opt_regime"]
wrong = df[df["wrong_regime"]]
print(f"\nWrong regime hours: {len(wrong):,} ({len(wrong)/len(df)*100:.1f}%)")
regime_confusion = pd.crosstab(
    df["actual_regime"].map({0: "OFF", 1: "1-boiler", 2: "2-boiler"}),
    df["opt_regime"].map({0: "OFF", 1: "1-boiler", 2: "2-boiler"}),
)
print("\nConfusion matrix (rows=actual, cols=optimal):")
print(regime_confusion)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
print("\n[*] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Boiler-Constrained Dispatch Optimization", fontsize=14, fontweight="bold")

# Plot 1: Step distribution
ax = axes[0, 0]
steps_range = range(0, 9)
width = 0.35
actual_dist = df["step"].value_counts(normalize=True).reindex(steps_range, fill_value=0)
opt_dist = df["opt_step"].value_counts(normalize=True).reindex(steps_range, fill_value=0)
ax.bar([s - width/2 for s in steps_range], actual_dist * 100, width, label="Actual", color="red", alpha=0.7)
ax.bar([s + width/2 for s in steps_range], opt_dist * 100, width, label="Optimal (boiler DP)", color="green", alpha=0.7)
ax.set_xlabel("MW Step")
ax.set_ylabel("% of Hours")
ax.set_title("Step Distribution: Actual vs Boiler-Constrained Optimal")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Regime over time (sample 2 months)
ax = axes[0, 1]
sample = df[(df["datetime"] >= "2025-01-01") & (df["datetime"] < "2025-03-01")].copy()
ax.step(sample["datetime"], sample["actual_regime"], "r-", alpha=0.7, label="Actual", where="post")
ax.step(sample["datetime"], sample["opt_regime"], "g-", alpha=0.7, label="Optimal", where="post")
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["OFF", "1-boiler", "2-boiler"])
ax.set_title("Regime Timeline: Jan-Feb 2025")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Monthly gap
ax = axes[1, 0]
monthly = df.set_index("datetime").resample("ME").agg({
    "actual_profit": "sum",
    "opt_profit": "sum",
    "greedy_profit": "sum",
}) / 1000
monthly.plot(ax=ax, linewidth=1.5)
ax.set_ylabel("Profit (kEUR)")
ax.set_title("Monthly Profit Comparison")
ax.legend(["Actual", "Boiler DP", "Greedy"], fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 4: Gap decomposition by regime mismatch
ax = axes[1, 1]
labels = ["Should be OFF\n(was running)", "Should be 1-boiler\n(was not)",
          "Should be 2-boiler\n(was not)", "Right regime\n(wrong step)"]
# Decompose
gap_should_off = df[(df["opt_regime"] == 0) & (df["actual_regime"] != 0)]["gap"].sum() / 1000
gap_should_1 = df[(df["opt_regime"] == 1) & (df["actual_regime"] != 1)]["gap"].sum() / 1000
gap_should_2 = df[(df["opt_regime"] == 2) & (df["actual_regime"] != 2)]["gap"].sum() / 1000
gap_right_regime = df[df["actual_regime"] == df["opt_regime"]]["gap"].sum() / 1000
values = [gap_should_off, gap_should_1, gap_should_2, gap_right_regime]
colors = ["#d32f2f", "#f57c00", "#1976d2", "#388e3c"]
bars = ax.bar(labels, values, color=colors, alpha=0.8)
ax.set_ylabel("Gap (kEUR)")
ax.set_title("Where Does the Gap Come From?")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(PLOT_DIR / "04_boiler_constrained.png", dpi=150)
plt.close()
print(f"[+] Saved {PLOT_DIR / '04_boiler_constrained.png'}")

# Weekly gap plot
fig, ax = plt.subplots(figsize=(14, 5))
weekly = df.set_index("datetime").resample("W").agg({
    "actual_profit": "sum",
    "opt_profit": "sum",
}) / 1000
ax.fill_between(weekly.index, weekly["actual_profit"], weekly["opt_profit"],
                alpha=0.3, color="red", label="Achievable gap")
ax.plot(weekly.index, weekly["actual_profit"], "r-", linewidth=1, label="Actual")
ax.plot(weekly.index, weekly["opt_profit"], "g-", linewidth=1, label="Optimal (boiler DP)")
ax.set_ylabel("Weekly Profit (kEUR)")
ax.set_title("Weekly Profit: Actual vs Boiler-Constrained Optimal")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "05_weekly_boiler_constrained.png", dpi=150)
plt.close()
print(f"[+] Saved {PLOT_DIR / '05_weekly_boiler_constrained.png'}")

print("\n[+] Done!")
