"""
BESS Persistent SoC Simulation
==============================
1 MW / 2 MWh battery, 2 cycles/day (4 charge + 4 discharge hours).
SoC persists across days: end-of-day SoC -> next day's start SoC.
Soft terminal SoC penalty to prevent pathological drift.

Baselines: indicator-guided, perfect foresight, naive fixed hours.
All use same SoC persistence rules.

Usage:
  python bess_simulation.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

BASE = Path(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data')
OUT  = BASE / 'MarketPriceGap' / 'da_indicator'

sys.path.insert(0, str(OUT / 'scripts'))
from build_da_indicator import load_all_data, build_features, generate_signals


# ============================================================
# DP OPTIMIZER (variable-length days, terminal SoC penalty)
# ============================================================

def optimize_bess_dp(prices, predicted_ranks, n_charge=4, n_discharge=4,
                     start_soc=1, terminal_penalty=5.0, efficiency=1.0):
    """
    Optimal BESS scheduling via dynamic programming.

    Parameters
    ----------
    prices : array-like
        Actual prices for revenue calculation. Length = number of hours in day.
    predicted_ranks : array-like
        Predicted ranks (or prices for perfect foresight). Same length as prices.
    n_charge : int
        Max charge actions per day.
    n_discharge : int
        Max discharge actions per day.
    start_soc : int
        Starting SoC in MWh (0, 1, or 2).
    terminal_penalty : float
        EUR penalty for ending at SoC 0 or SoC 2 (encourages SoC=1 at end of day).
    efficiency : float
        Round-trip efficiency (default 1.0, i.e. no losses).

    Returns
    -------
    actions : np.array
        Action per hour: 1=charge, -1=discharge, 0=idle.
    revenue : float
        Net revenue in EUR (based on actual prices).
    soc_trace : np.array
        SoC at each hour boundary (length = n_hours + 1).
    """
    n_hours = len(prices)
    if n_hours == 0 or np.any(np.isnan(prices)):
        return np.zeros(n_hours), 0.0, np.ones(n_hours + 1) * start_soc

    p = np.array(predicted_ranks[:n_hours], dtype=float)
    actual = np.array(prices[:n_hours], dtype=float)

    NC = n_charge + 1
    ND = n_discharge + 1
    INF = -1e18

    # dp[h][soc][cr][dr] = best "predicted revenue" from hour h onwards
    dp = np.full((n_hours + 1, 3, NC, ND), INF)
    choice = np.zeros((n_hours, 3, NC, ND), dtype=int)

    # Terminal: penalize SoC != 1
    for s in range(3):
        penalty = 0.0
        if s == 0 or s == 2:
            penalty = -terminal_penalty
        for cr in range(NC):
            for dr in range(ND):
                dp[n_hours][s][cr][dr] = penalty

    # Backward pass
    for h in range(n_hours - 1, -1, -1):
        for s in range(3):
            for cr in range(NC):
                for dr in range(ND):
                    best = INF
                    best_a = 0

                    # Option 1: idle
                    val = dp[h + 1][s][cr][dr]
                    if val > best:
                        best = val
                        best_a = 0

                    # Option 2: charge (buy power)
                    if cr > 0 and s + 1 <= 2:
                        val = -p[h] + dp[h + 1][s + 1][cr - 1][dr]
                        if val > best:
                            best = val
                            best_a = 1

                    # Option 3: discharge (sell power)
                    if dr > 0 and s - 1 >= 0:
                        val = p[h] * efficiency + dp[h + 1][s - 1][cr][dr - 1]
                        if val > best:
                            best = val
                            best_a = -1

                    dp[h][s][cr][dr] = best
                    choice[h][s][cr][dr] = best_a

    # Forward pass: reconstruct actions
    actions = np.zeros(n_hours)
    soc_trace = np.zeros(n_hours + 1)
    soc_trace[0] = start_soc
    soc = start_soc
    cr = n_charge
    dr = n_discharge

    for h in range(n_hours):
        a = choice[h][soc][cr][dr]
        actions[h] = a
        if a == 1:
            soc += 1
            cr -= 1
        elif a == -1:
            soc -= 1
            dr -= 1
        soc_trace[h + 1] = soc

    # Revenue based on ACTUAL prices
    revenue = 0.0
    for h in range(n_hours):
        if actions[h] == 1:
            revenue -= actual[h]  # pay to charge
        elif actions[h] == -1:
            revenue += actual[h] * efficiency  # earn from discharge

    return actions, revenue, soc_trace


def naive_schedule(n_hours, start_soc):
    """
    Naive 2-cycle fixed schedule.
    Cycle 1: charge h2-3, discharge h10-11
    Cycle 2: charge h14-15, discharge h18-19
    Respects SoC constraints with given start_soc.
    """
    actions = np.zeros(n_hours)
    if n_hours < 20:
        return actions, start_soc

    # Attempt the fixed schedule
    schedule = {2: 1, 3: 1, 10: -1, 11: -1, 14: 1, 15: 1, 18: -1, 19: -1}

    soc = start_soc
    for h in range(n_hours):
        a = schedule.get(h, 0)
        new_soc = soc + a
        if new_soc < 0 or new_soc > 2:
            a = 0  # skip action if SoC violated
        actions[h] = a
        soc += a

    return actions, soc


# ============================================================
# MAIN SIMULATION
# ============================================================

def run_simulation():
    print("[*] Loading and preparing data...")
    df = load_all_data()
    df, daily = build_features(df)
    df = generate_signals(df)

    # Get all dates and sort them
    day_counts = df.groupby('date').size()
    all_dates = sorted(day_counts.index.tolist())

    print(f"[+] Total dates: {len(all_dates)}")
    print(f"[+] Date range: {min(all_dates)} to {max(all_dates)}")

    # DST info: count hours per day
    hours_per_day = day_counts.to_dict()
    dst_days = {d: n for d, n in hours_per_day.items() if n != 24}
    if dst_days:
        print(f"[*] Non-24-hour days (DST): {len(dst_days)}")
        for d, n in sorted(dst_days.items()):
            print(f"    {d}: {n} hours")

    # ============================================================
    # PERSISTENT SOC SIMULATION
    # ============================================================
    print("\n[*] Running persistent SoC simulation (2 cycles/day)...")

    results = []
    soc_timeseries = []

    # State carries across days
    soc_indicator = 1  # start at SoC=1 on first day
    soc_perfect = 1
    soc_naive = 1

    skipped = 0
    for date in all_dates:
        day = df[df['date'] == date].sort_values('hour')
        n_hours = len(day)

        if n_hours < 20:  # skip very short days
            skipped += 1
            continue

        prices = day['da_price'].values[:n_hours]
        pred_ranks = day['rank_predicted'].values[:n_hours]

        if np.any(np.isnan(prices)):
            skipped += 1
            continue

        pred_ranks = np.nan_to_num(pred_ranks, nan=12.0)

        # --- Indicator-guided (2 cycles) ---
        act_ind, rev_ind, soc_ind_trace = optimize_bess_dp(
            prices, pred_ranks, n_charge=4, n_discharge=4,
            start_soc=soc_indicator, terminal_penalty=20.0
        )
        end_soc_ind = int(soc_ind_trace[-1])

        # --- Perfect foresight (2 cycles) ---
        act_pf, rev_pf, soc_pf_trace = optimize_bess_dp(
            prices, prices, n_charge=4, n_discharge=4,
            start_soc=soc_perfect, terminal_penalty=20.0
        )
        end_soc_pf = int(soc_pf_trace[-1])

        # --- Naive fixed schedule (2 cycles) ---
        act_naive, end_soc_naive_raw = naive_schedule(n_hours, soc_naive)
        rev_naive = 0.0
        soc_n = soc_naive
        naive_soc_trace = [soc_n]
        feasible = True
        for h in range(n_hours):
            soc_n += int(act_naive[h])
            if soc_n < 0 or soc_n > 2:
                feasible = False
                break
            naive_soc_trace.append(soc_n)
        if feasible:
            rev_naive = -np.sum(act_naive * prices)
            end_soc_naive = soc_n
        else:
            # Fallback: idle all day
            act_naive = np.zeros(n_hours)
            rev_naive = 0.0
            end_soc_naive = soc_naive
            naive_soc_trace = [soc_naive] * (n_hours + 1)

        # Collect results
        charge_h_ind = list(np.where(act_ind > 0.5)[0])
        discharge_h_ind = list(np.where(act_ind < -0.5)[0])

        results.append({
            'date': date,
            'n_hours': n_hours,
            # Indicator
            'start_soc_ind': soc_indicator,
            'end_soc_ind': end_soc_ind,
            'rev_indicator': rev_ind,
            'charge_hours_ind': str(charge_h_ind),
            'discharge_hours_ind': str(discharge_h_ind),
            # Perfect
            'start_soc_pf': soc_perfect,
            'end_soc_pf': end_soc_pf,
            'rev_perfect': rev_pf,
            # Naive
            'start_soc_naive': soc_naive,
            'end_soc_naive': end_soc_naive,
            'rev_naive': rev_naive,
            # Metrics
            'capture_rate': rev_ind / rev_pf if rev_pf > 0 else np.nan,
            'price_range': prices.max() - prices.min(),
            'price_std': prices.std(),
            'price_mean': prices.mean(),
        })

        # Store SoC timeseries for detailed analysis
        for h in range(n_hours + 1):
            soc_timeseries.append({
                'date': date,
                'hour': h,
                'soc_indicator': soc_ind_trace[h],
                'soc_perfect': soc_pf_trace[h],
                'soc_naive': naive_soc_trace[h] if h < len(naive_soc_trace) else end_soc_naive,
            })

        # Carry SoC to next day
        soc_indicator = end_soc_ind
        soc_perfect = end_soc_pf
        soc_naive = end_soc_naive

    if skipped > 0:
        print(f"[-] Skipped {skipped} days (insufficient data)")

    res = pd.DataFrame(results)
    res['date'] = pd.to_datetime(res['date'])
    res = res.sort_values('date').reset_index(drop=True)
    res['year'] = res['date'].dt.year
    res['month'] = res['date'].dt.to_period('M')

    soc_ts = pd.DataFrame(soc_timeseries)

    # ============================================================
    # PRINT RESULTS
    # ============================================================
    print("\n" + "=" * 75)
    print("  BESS PERSISTENT SOC SIMULATION (2 CYCLES/DAY)")
    print("=" * 75)

    print(f"\n  Days simulated: {len(res)}")
    print(f"  Period: {res['date'].min().date()} to {res['date'].max().date()}")

    tot_ind = res['rev_indicator'].sum()
    tot_pf = res['rev_perfect'].sum()
    tot_naive = res['rev_naive'].sum()
    avg_cap = res['capture_rate'].dropna().mean()

    print(f"\n--- OVERALL ---")
    print(f"  {'Metric':<35s} {'Value':>15s}")
    print(f"  {'-'*35} {'-'*15}")
    print(f"  {'Total Revenue (Indicator)':<35s} {tot_ind:>12,.0f} EUR")
    print(f"  {'Total Revenue (Perfect)':<35s} {tot_pf:>12,.0f} EUR")
    print(f"  {'Total Revenue (Naive)':<35s} {tot_naive:>12,.0f} EUR")
    print(f"  {'Avg Daily Revenue (Indicator)':<35s} {res['rev_indicator'].mean():>12.1f} EUR")
    print(f"  {'Avg Capture Rate':<35s} {avg_cap:>12.1%}")
    print(f"  {'Indicator / Naive':<35s} {tot_ind / tot_naive:>12.1f}x" if tot_naive > 0 else "")

    # By year
    print(f"\n--- BY YEAR ---")
    for yr in sorted(res['year'].unique()):
        sub = res[res['year'] == yr]
        r_ind = sub['rev_indicator'].sum()
        r_pf = sub['rev_perfect'].sum()
        r_naive = sub['rev_naive'].sum()
        cap = sub['capture_rate'].dropna().mean()
        print(f"  {yr}: Ind={r_ind:>8,.0f} EUR  Pf={r_pf:>8,.0f} EUR  "
              f"Naive={r_naive:>8,.0f} EUR  Cap={cap:.1%}  n={len(sub)}")

    # Monthly
    print(f"\n--- MONTHLY ---")
    monthly = res.groupby('month').agg(
        rev_ind=('rev_indicator', 'sum'),
        rev_pf=('rev_perfect', 'sum'),
        rev_naive=('rev_naive', 'sum'),
        cap=('capture_rate', 'mean'),
        n=('date', 'count'),
    ).reset_index()
    for _, row in monthly.iterrows():
        print(f"  {str(row['month']):>8s}: Ind={row['rev_ind']:>8,.0f}  "
              f"Pf={row['rev_pf']:>8,.0f}  Naive={row['rev_naive']:>8,.0f}  "
              f"Cap={row['cap']:.1%}  n={row['n']:.0f}")

    # SoC persistence verification
    print(f"\n--- SOC PERSISTENCE VERIFICATION ---")
    soc_breaks = 0
    for i in range(1, len(res)):
        if res.iloc[i]['start_soc_ind'] != res.iloc[i-1]['end_soc_ind']:
            soc_breaks += 1
            print(f"  [!] SoC break at {res.iloc[i]['date'].date()}: "
                  f"prev end={res.iloc[i-1]['end_soc_ind']}, "
                  f"curr start={res.iloc[i]['start_soc_ind']}")
    if soc_breaks == 0:
        print(f"  [+] SoC chain verified: no breaks in {len(res)} days")

    # SoC distribution at end of day
    print(f"\n--- END-OF-DAY SOC DISTRIBUTION ---")
    for soc_val in [0, 1, 2]:
        n = (res['end_soc_ind'] == soc_val).sum()
        pct = n / len(res) * 100
        print(f"  SoC={soc_val}: {n:4d} days ({pct:.1f}%)")

    # ============================================================
    # SAVE OUTPUTS
    # ============================================================
    out_dir = OUT / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)

    res.to_csv(out_dir / 'bess_persistent_daily.csv', index=False)
    print(f"\n[+] Daily results: {out_dir / 'bess_persistent_daily.csv'}")

    # Monthly summary
    monthly.to_csv(out_dir / 'bess_persistent_monthly.csv', index=False)
    print(f"[+] Monthly summary: {out_dir / 'bess_persistent_monthly.csv'}")

    # SoC timeseries (for charts)
    soc_ts.to_csv(out_dir / 'bess_soc_timeseries.csv', index=False)
    print(f"[+] SoC timeseries: {out_dir / 'bess_soc_timeseries.csv'}")

    # ============================================================
    # SUMMARY CHART
    # ============================================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Cumulative revenue
    ax = axes[0, 0]
    ax.plot(res['date'], res['rev_indicator'].cumsum(), 'b-', lw=1.5, label='Indicator (2 cycles)')
    ax.plot(res['date'], res['rev_perfect'].cumsum(), 'g--', lw=1, alpha=0.6, label='Perfect (2 cycles)')
    ax.plot(res['date'], res['rev_naive'].cumsum(), 'r:', lw=1, alpha=0.6, label='Naive (fixed hours)')
    jan26 = pd.Timestamp('2026-01-01')
    if res['date'].max() > jan26:
        ax.axvline(jan26, color='gray', ls='--', alpha=0.3)
    ax.set_ylabel('Cumulative Revenue (EUR)')
    ax.set_title('BESS Persistent SoC: Cumulative Revenue (2 Cycles/Day)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Monthly comparison
    ax = axes[0, 1]
    x = range(len(monthly))
    w = 0.25
    ax.bar([i - w for i in x], monthly['rev_ind'], w, label='Indicator', color='#1976d2')
    ax.bar(x, monthly['rev_pf'], w, label='Perfect', color='#4caf50', alpha=0.7)
    ax.bar([i + w for i in x], monthly['rev_naive'], w, label='Naive', color='#ff9800')
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(m) for m in monthly['month']], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Monthly Revenue (EUR)')
    ax.set_title('Monthly Revenue (Persistent SoC)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. End-of-day SoC over time
    ax = axes[1, 0]
    ax.plot(res['date'], res['end_soc_ind'], 'b-', lw=0.5, alpha=0.5)
    ax.plot(res['date'], res['end_soc_ind'].rolling(30, min_periods=5).mean(),
            'b-', lw=2, label='30-day avg SoC (indicator)')
    ax.axhline(1, color='gray', ls='--', alpha=0.5, label='Target SoC=1')
    ax.set_ylabel('End-of-Day SoC (MWh)')
    ax.set_title('SoC Persistence: End-of-Day State')
    ax.set_ylim(-0.2, 2.3)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Capture rate over time
    ax = axes[1, 1]
    cap_roll = res['capture_rate'].rolling(30, min_periods=5).mean()
    ax.plot(res['date'], cap_roll, 'b-', lw=1.5, label='30-day rolling capture rate')
    ax.axhline(avg_cap, color='red', ls='--', alpha=0.5, label=f'Overall mean: {avg_cap:.1%}')
    ax.set_ylabel('Capture Rate')
    ax.set_title('Capture Rate Over Time')
    ax.set_ylim(0.4, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('BESS 1 MW/2 MWh -- 2 Cycles/Day, Persistent SoC', fontweight='bold', fontsize=13)
    fig.tight_layout()

    plot_dir = OUT / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / '13_bess_persistent_soc.png', dpi=150)
    plt.close(fig)
    print(f"\n[+] Summary chart: 13_bess_persistent_soc.png")

    print("\n" + "=" * 75)
    return res, soc_ts


if __name__ == '__main__':
    run_simulation()
