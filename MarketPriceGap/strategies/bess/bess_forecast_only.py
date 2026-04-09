"""
BESS Forecast-Only Simulation
==============================
1 MW / 2 MWh battery, 2 cycles/day, persistent SoC.
Uses ONLY the DA price forecast as the DP input signal.

The DP optimizer receives forecast prices (EUR/MWh) and finds the
optimal charge/discharge schedule. Revenue is calculated on actual prices.

Compared against:
  - Perfect foresight (DP on actual prices)
  - Naive fixed schedule
  - Legacy 3-feature rank (for reference)

Usage:
  python bess_forecast_only.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data')
OUT = BASE / 'MarketPriceGap' / 'strategies' / 'bess'


# ============================================================
# DP OPTIMIZER
# ============================================================

def optimize_bess_dp(prices, predicted_prices, n_charge=4, n_discharge=4,
                     start_soc=1, terminal_penalty=20.0):
    """
    Optimal BESS scheduling via dynamic programming.

    State space:
      - hour h (0..n_hours-1)
      - SoC s (0, 1, or 2 MWh)
      - charges remaining cr (0..n_charge)
      - discharges remaining dr (0..n_discharge)

    At each hour, three actions:
      0 = idle
      1 = charge (buy 1 MWh, SoC += 1)
     -1 = discharge (sell 1 MWh, SoC -= 1)

    The DP maximizes SUM(predicted_prices * action) + terminal_penalty.
    Revenue is then computed on ACTUAL prices using the chosen schedule.

    Parameters
    ----------
    prices : array
        Actual DA prices (EUR/MWh), used only for revenue calculation.
    predicted_prices : array
        Forecast prices fed to the optimizer. Can be actual prices
        (perfect foresight), forecast prices, or rank values.
    n_charge : int
        Max charge actions per day (2 per cycle).
    n_discharge : int
        Max discharge actions per day (2 per cycle).
    start_soc : int
        Starting SoC: 0, 1, or 2 MWh.
    terminal_penalty : float
        EUR penalty for ending at SoC != 1 (prevents drift).

    Returns
    -------
    actions : array of {-1, 0, 1} per hour
    revenue : float (EUR, based on actual prices)
    soc_trace : array of SoC at each hour boundary
    """
    n_hours = len(prices)
    if n_hours == 0 or np.any(np.isnan(prices)):
        return np.zeros(n_hours), 0.0, np.ones(n_hours + 1) * start_soc

    p = np.array(predicted_prices[:n_hours], dtype=float)
    actual = np.array(prices[:n_hours], dtype=float)

    NC = n_charge + 1      # possible values: 0..n_charge
    ND = n_discharge + 1
    INF = -1e18

    # dp[h][s][cr][dr] = best predicted revenue from hour h to end
    dp = np.full((n_hours + 1, 3, NC, ND), INF)
    choice = np.zeros((n_hours, 3, NC, ND), dtype=int)

    # Terminal condition: penalize SoC != 1
    for s in range(3):
        penalty = -terminal_penalty if s != 1 else 0.0
        for cr in range(NC):
            for dr in range(ND):
                dp[n_hours][s][cr][dr] = penalty

    # BACKWARD PASS: fill dp table from last hour to first
    for h in range(n_hours - 1, -1, -1):
        for s in range(3):
            for cr in range(NC):
                for dr in range(ND):
                    best = INF
                    best_a = 0

                    # Action 0: idle
                    val = dp[h + 1][s][cr][dr]
                    if val > best:
                        best = val
                        best_a = 0

                    # Action 1: charge (buy power at predicted price)
                    if cr > 0 and s + 1 <= 2:
                        val = -p[h] + dp[h + 1][s + 1][cr - 1][dr]
                        if val > best:
                            best = val
                            best_a = 1

                    # Action -1: discharge (sell power at predicted price)
                    if dr > 0 and s - 1 >= 0:
                        val = p[h] + dp[h + 1][s - 1][cr][dr - 1]
                        if val > best:
                            best = val
                            best_a = -1

                    dp[h][s][cr][dr] = best
                    choice[h][s][cr][dr] = best_a

    # FORWARD PASS: reconstruct optimal actions
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

    # REVENUE: based on ACTUAL prices (not predictions)
    revenue = 0.0
    for h in range(n_hours):
        if actions[h] == 1:
            revenue -= actual[h]   # pay to charge
        elif actions[h] == -1:
            revenue += actual[h]   # earn from discharge

    return actions, revenue, soc_trace


def naive_schedule(n_hours, start_soc):
    """Naive 2-cycle: charge h2-3, discharge h10-11, charge h14-15, discharge h18-19."""
    actions = np.zeros(n_hours)
    if n_hours < 20:
        return actions, start_soc
    schedule = {2: 1, 3: 1, 10: -1, 11: -1, 14: 1, 15: 1, 18: -1, 19: -1}
    soc = start_soc
    for h in range(n_hours):
        a = schedule.get(h, 0)
        if soc + a < 0 or soc + a > 2:
            a = 0
        actions[h] = a
        soc += a
    return actions, soc


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load market prices and DA price forecasts, merge to hourly."""
    print("[*] Loading market prices...")
    mkt = pd.read_csv(BASE / 'MarketPriceGap' / 'data' / 'processed' / 'hourly_market_prices.csv')
    mkt['timestamp_hour'] = pd.to_datetime(mkt['timestamp_hour'])
    mkt['date'] = pd.to_datetime(mkt['date'])
    print(f"[+] Market: {len(mkt)} rows, {mkt['date'].min().date()} to {mkt['date'].max().date()}")

    # Load DA price forecasts (15-min -> hourly)
    print("[*] Loading DA price forecasts...")
    fcst_path = BASE / 'data' / 'clean' / 'market' / 'da_forecasts' / 'da_price_forecasts_15min.csv'
    fcst = pd.read_csv(fcst_path)
    fcst['datetime'] = pd.to_datetime(fcst['datetime'])
    fcst = fcst[fcst['sk_forecast1_spot'].notna()].copy()
    fcst['date'] = fcst['datetime'].dt.normalize()
    fcst['hour'] = fcst['datetime'].dt.hour

    # Average 15-min forecasts to hourly
    fcst_h = fcst.groupby(['date', 'hour']).agg(
        da_forecast=('sk_forecast1_spot', 'mean'),
    ).reset_index()
    fcst_h['date'] = pd.to_datetime(fcst_h['date'])
    print(f"[+] Forecasts: {fcst_h['date'].nunique()} days, "
          f"{fcst_h['date'].min().date()} to {fcst_h['date'].max().date()}")

    # Merge
    df = mkt.merge(fcst_h, on=['date', 'hour'], how='inner')
    df = df.sort_values(['date', 'hour']).reset_index(drop=True)

    # Only keep days with full forecast coverage (>=20 hours)
    day_counts = df.groupby('date').size()
    full_days = day_counts[day_counts >= 20].index
    df = df[df['date'].isin(full_days)].copy()

    print(f"[+] Merged: {len(df)} rows, {df['date'].nunique()} full days")
    return df


# ============================================================
# LEGACY RANK (for comparison)
# ============================================================

def add_legacy_rank(df):
    """Add legacy 3-feature rank for comparison."""
    # Load forecast rank
    df = df.sort_values(['hour', 'date']).reset_index(drop=True)
    df['da_price_yest_h'] = df.groupby('hour')['da_price'].shift(1)
    df['da_price_7d_h'] = df.groupby('hour')['da_price'].shift(7)
    df = df.sort_values(['date', 'hour']).reset_index(drop=True)

    # We don't have load forecasts here, so use yesterday price as proxy
    # for a fair comparison, use just yesterday + 7d price ranks
    df['rank_yest'] = df.groupby('date')['da_price_yest_h'].rank(method='first', na_option='bottom')
    df['rank_7d'] = df.groupby('date')['da_price_7d_h'].rank(method='first', na_option='bottom')

    # Simple legacy rank: 60% yesterday, 40% weekly
    df['rank_legacy'] = 0.60 * df['rank_yest'].fillna(12) + 0.40 * df['rank_7d'].fillna(12)
    return df


# ============================================================
# SIMULATION
# ============================================================

def run_simulation(df):
    print("\n[*] Running BESS simulation...")
    print("[*] Battery: 1 MW / 2 MWh, 2 cycles/day, persistent SoC")
    print("[*] Signal: DA price forecast (EUR/MWh) fed directly to DP\n")

    all_dates = sorted(df['date'].unique())

    results = []
    soc_fcst = 1      # forecast-driven
    soc_perfect = 1   # perfect foresight
    soc_naive = 1     # fixed schedule
    soc_legacy = 1    # legacy rank

    for date in all_dates:
        day = df[df['date'] == date].sort_values('hour')
        n_hours = len(day)
        if n_hours < 20:
            continue

        prices = day['da_price'].values[:n_hours]
        forecasts = day['da_forecast'].values[:n_hours]
        legacy_ranks = day['rank_legacy'].values[:n_hours]

        if np.any(np.isnan(prices)) or np.any(np.isnan(forecasts)):
            continue

        legacy_ranks = np.nan_to_num(legacy_ranks, nan=12.0)

        # --- Forecast-driven ---
        act_f, rev_f, soc_f = optimize_bess_dp(
            prices, forecasts, n_charge=4, n_discharge=4,
            start_soc=soc_fcst, terminal_penalty=20.0)

        # --- Perfect foresight ---
        act_p, rev_p, soc_p = optimize_bess_dp(
            prices, prices, n_charge=4, n_discharge=4,
            start_soc=soc_perfect, terminal_penalty=20.0)

        # --- Legacy rank ---
        act_l, rev_l, soc_l = optimize_bess_dp(
            prices, legacy_ranks, n_charge=4, n_discharge=4,
            start_soc=soc_legacy, terminal_penalty=20.0)

        # --- Naive ---
        act_n, end_soc_n = naive_schedule(n_hours, soc_naive)
        rev_n = -np.sum(act_n * prices)

        # Collect charge/discharge hours for forecast strategy
        charge_hours = list(np.where(act_f > 0.5)[0])
        discharge_hours = list(np.where(act_f < -0.5)[0])

        # Price info
        avg_charge_price = prices[act_f > 0.5].mean() if (act_f > 0.5).any() else np.nan
        avg_discharge_price = prices[act_f < -0.5].mean() if (act_f < -0.5).any() else np.nan

        results.append({
            'date': date,
            'n_hours': n_hours,
            # Forecast
            'rev_forecast': rev_f,
            'charge_hours': str(charge_hours),
            'discharge_hours': str(discharge_hours),
            'avg_charge_price': avg_charge_price,
            'avg_discharge_price': avg_discharge_price,
            'start_soc_f': soc_fcst,
            'end_soc_f': int(soc_f[-1]),
            # Perfect
            'rev_perfect': rev_p,
            'start_soc_p': soc_perfect,
            'end_soc_p': int(soc_p[-1]),
            # Legacy
            'rev_legacy': rev_l,
            'start_soc_l': soc_legacy,
            'end_soc_l': int(soc_l[-1]),
            # Naive
            'rev_naive': rev_n,
            'start_soc_n': soc_naive,
            'end_soc_n': end_soc_n,
            # Metrics
            'capture_forecast': rev_f / rev_p if rev_p > 0 else np.nan,
            'capture_legacy': rev_l / rev_p if rev_p > 0 else np.nan,
            'forecast_vs_legacy': rev_f - rev_l,
            'price_range': prices.max() - prices.min(),
            'price_mean': prices.mean(),
            'forecast_mae': np.mean(np.abs(forecasts - prices)),
        })

        # Carry SoC
        soc_fcst = int(soc_f[-1])
        soc_perfect = int(soc_p[-1])
        soc_legacy = int(soc_l[-1])
        soc_naive = end_soc_n

    res = pd.DataFrame(results)
    res['date'] = pd.to_datetime(res['date'])
    res = res.sort_values('date').reset_index(drop=True)
    return res


# ============================================================
# REPORTING
# ============================================================

def print_results(res):
    print("=" * 70)
    print("  BESS FORECAST-ONLY SIMULATION RESULTS")
    print("=" * 70)

    print(f"\n  Days: {len(res)}")
    print(f"  Period: {res['date'].min().date()} to {res['date'].max().date()}")

    tot_f = res['rev_forecast'].sum()
    tot_p = res['rev_perfect'].sum()
    tot_l = res['rev_legacy'].sum()
    tot_n = res['rev_naive'].sum()
    cap_f = res['capture_forecast'].dropna().mean()
    cap_l = res['capture_legacy'].dropna().mean()

    print(f"\n--- REVENUE COMPARISON ---")
    print(f"  {'Strategy':<30s} {'Revenue (EUR)':>14s} {'Capture':>10s}")
    print(f"  {'-'*30} {'-'*14} {'-'*10}")
    print(f"  {'Forecast (DA fcst -> DP)':<30s} {tot_f:>12,.0f}   {cap_f:>8.1%}")
    print(f"  {'Legacy (rank -> DP)':<30s} {tot_l:>12,.0f}   {cap_l:>8.1%}")
    print(f"  {'Perfect foresight':<30s} {tot_p:>12,.0f}   {'100.0%':>8s}")
    print(f"  {'Naive fixed schedule':<30s} {tot_n:>12,.0f}   {'---':>8s}")

    print(f"\n  Forecast vs Legacy:    {tot_f - tot_l:+,.0f} EUR ({(tot_f-tot_l)/abs(tot_l)*100:+.1f}%)")
    print(f"  Forecast vs Naive:     {tot_f - tot_n:+,.0f} EUR")
    print(f"  Avg forecast MAE:      {res['forecast_mae'].mean():.1f} EUR/MWh")
    print(f"  Avg charge price:      {res['avg_charge_price'].mean():.1f} EUR/MWh")
    print(f"  Avg discharge price:   {res['avg_discharge_price'].mean():.1f} EUR/MWh")
    print(f"  Avg daily spread:      {(res['avg_discharge_price'] - res['avg_charge_price']).mean():.1f} EUR/MWh")

    # Monthly
    res['month'] = res['date'].dt.to_period('M')
    print(f"\n--- MONTHLY ---")
    monthly = res.groupby('month').agg(
        rev_f=('rev_forecast', 'sum'),
        rev_l=('rev_legacy', 'sum'),
        rev_p=('rev_perfect', 'sum'),
        rev_n=('rev_naive', 'sum'),
        cap_f=('capture_forecast', 'mean'),
        cap_l=('capture_legacy', 'mean'),
        mae=('forecast_mae', 'mean'),
        n=('date', 'count'),
    ).reset_index()

    for _, row in monthly.iterrows():
        diff = row['rev_f'] - row['rev_l']
        print(f"  {str(row['month']):>8s}: Fcst={row['rev_f']:>8,.0f}  "
              f"Legacy={row['rev_l']:>8,.0f}  Pf={row['rev_p']:>8,.0f}  "
              f"Diff={diff:>+6,.0f}  Cap={row['cap_f']:.1%}  "
              f"MAE={row['mae']:.1f}  n={row['n']:.0f}")

    # Days where forecast beats legacy
    wins = (res['forecast_vs_legacy'] > 0).sum()
    print(f"\n  Forecast beats legacy: {wins}/{len(res)} days ({wins/len(res):.0%})")

    # SoC verification
    soc_breaks = 0
    for i in range(1, len(res)):
        if res.iloc[i]['start_soc_f'] != res.iloc[i-1]['end_soc_f']:
            soc_breaks += 1
    if soc_breaks == 0:
        print(f"  [+] SoC chain verified: no breaks")
    else:
        print(f"  [!] SoC breaks: {soc_breaks}")

    print("=" * 70)
    return monthly


# ============================================================
# CHART
# ============================================================

def plot_results(res, monthly):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Cumulative revenue
    ax = axes[0, 0]
    ax.plot(res['date'], res['rev_forecast'].cumsum(), 'b-', lw=2, label='DA Forecast')
    ax.plot(res['date'], res['rev_legacy'].cumsum(), 'r-', lw=1.5, alpha=0.7, label='Legacy Rank')
    ax.plot(res['date'], res['rev_perfect'].cumsum(), 'g--', lw=1, alpha=0.5, label='Perfect')
    ax.plot(res['date'], res['rev_naive'].cumsum(), 'gray', lw=1, alpha=0.3, label='Naive')
    ax.set_ylabel('Cumulative Revenue (EUR)')
    ax.set_title('Cumulative Revenue')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Daily forecast vs legacy scatter
    ax = axes[0, 1]
    ax.scatter(res['rev_legacy'], res['rev_forecast'], alpha=0.4, s=15, c='#1976d2')
    lim = max(res[['rev_legacy', 'rev_forecast']].abs().max().max(), 1) * 1.05
    ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5)
    ax.set_xlabel('Legacy Revenue (EUR/day)')
    ax.set_ylabel('Forecast Revenue (EUR/day)')
    wins = (res['rev_forecast'] > res['rev_legacy']).sum()
    ax.set_title(f'Daily: Forecast wins {wins}/{len(res)} ({wins/len(res):.0%})')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.3)

    # 3. Capture rate over time
    ax = axes[1, 0]
    roll = res['capture_forecast'].rolling(14, min_periods=3).mean()
    roll_l = res['capture_legacy'].rolling(14, min_periods=3).mean()
    ax.plot(res['date'], roll, 'b-', lw=1.5, label='Forecast (14d)')
    ax.plot(res['date'], roll_l, 'r-', lw=1.5, alpha=0.7, label='Legacy (14d)')
    ax.axhline(res['capture_forecast'].dropna().mean(), color='b', ls='--', alpha=0.3)
    ax.set_ylabel('Capture Rate')
    ax.set_title('Rolling Capture Rate')
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Monthly revenue bars
    ax = axes[1, 1]
    x = range(len(monthly))
    w = 0.25
    ax.bar([i - w for i in x], monthly['rev_f'], w, label='Forecast', color='#1976d2')
    ax.bar(x, monthly['rev_l'], w, label='Legacy', color='#d32f2f', alpha=0.7)
    ax.bar([i + w for i in x], monthly['rev_p'], w, label='Perfect', color='#4caf50', alpha=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(m) for m in monthly['month']], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Monthly Revenue (EUR)')
    ax.set_title('Monthly Revenue')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    tot_f = res['rev_forecast'].sum()
    tot_l = res['rev_legacy'].sum()
    cap = res['capture_forecast'].dropna().mean()
    fig.suptitle(f'BESS 1 MW/2 MWh -- Forecast-Only DP\n'
                 f'Forecast: {tot_f:,.0f} EUR | Legacy: {tot_l:,.0f} EUR | '
                 f'Capture: {cap:.1%}',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()

    path = OUT / 'bess_forecast_only.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[+] Chart saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = add_legacy_rank(df)
    res = run_simulation(df)
    monthly = print_results(res)
    plot_results(res, monthly)

    # Save
    res.to_csv(OUT / 'bess_forecast_only_daily.csv', index=False)
    print(f"[+] Results saved: {OUT / 'bess_forecast_only_daily.csv'}")


if __name__ == '__main__':
    main()
