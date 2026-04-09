"""
Portfolio Analysis - Combined BESS + Speculative Strategy
=========================================================
Reads existing baseline results, computes combined metrics,
and runs comprehensive simulation analysis.

Output saved to MarketPriceGap/strategies/simulation/results/

Usage:
  python portfolio_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from backtest_engine import (
    BatteryDegradationModel, TransactionCostModel,
    compute_sharpe, compute_sortino, compute_max_drawdown,
    compute_calmar, compute_profit_factor,
    rolling_metrics, identify_worst_days,
    monte_carlo_annual_pnl, clustered_stress_test,
    load_bess_daily, load_speculative_daily, load_enhanced_bess_daily,
)

BASE = Path(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data')
OUT_DIR = BASE / 'MarketPriceGap' / 'strategies' / 'simulation' / 'results'


def setup():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

def load_and_prepare():
    """Load all strategies' daily results and align dates."""
    print("[*] Loading strategy results...")

    bess = load_bess_daily()
    spec = load_speculative_daily()
    enh_bess = load_enhanced_bess_daily()

    print(f"[+] BESS baseline: {len(bess)} days ({bess['date'].min().date()} to {bess['date'].max().date()})")
    print(f"[+] Speculative: {len(spec)} days ({spec['date'].min().date()} to {spec['date'].max().date()})")
    if enh_bess is not None:
        print(f"[+] Enhanced BESS: {len(enh_bess)} days ({enh_bess['date'].min().date()} to {enh_bess['date'].max().date()})")
    else:
        print("[-] Enhanced BESS data not found, using baseline only")

    # Find overlapping date range
    common_start = max(bess['date'].min(), spec['date'].min())
    common_end = min(bess['date'].max(), spec['date'].max())
    print(f"[+] Overlapping period: {common_start.date()} to {common_end.date()}")

    # Merge on date -- start with baseline BESS
    combined = bess[['date', 'rev_indicator', 'rev_perfect', 'rev_naive',
                      'capture_rate', 'price_range', 'price_std', 'price_mean']].copy()
    combined = combined.rename(columns={
        'rev_indicator': 'bess_pnl',
        'rev_perfect': 'bess_perfect',
        'rev_naive': 'bess_naive',
    })

    # Add enhanced BESS if available
    # Note: rev_enhanced includes degradation; rev_enh_nodeg is gross revenue
    if enh_bess is not None:
        rev_col = 'rev_enh_nodeg' if 'rev_enh_nodeg' in enh_bess.columns else 'rev_enhanced'
        cap_col = 'capture_enh_nodeg' if 'capture_enh_nodeg' in enh_bess.columns else 'capture_enhanced'
        enh_cols = enh_bess[['date', rev_col]].copy()
        enh_cols = enh_cols.rename(columns={rev_col: 'bess_enhanced_pnl'})
        if cap_col in enh_bess.columns:
            enh_cols['capture_enhanced'] = enh_bess[cap_col].values
        combined = combined.merge(enh_cols, on='date', how='left')

    spec_cols = spec[['date', 'pnl', 'pnl_raw', 'n_correct', 'n_traded', 'accuracy']].copy()
    spec_cols = spec_cols.rename(columns={
        'pnl': 'spec_pnl',
        'pnl_raw': 'spec_pnl_raw',
        'n_correct': 'spec_n_correct',
        'n_traded': 'spec_n_traded',
        'accuracy': 'spec_accuracy',
    })

    combined = combined.merge(spec_cols, on='date', how='inner')
    combined = combined.sort_values('date').reset_index(drop=True)

    print(f"[+] Combined dataset: {len(combined)} days")
    return bess, spec, combined, enh_bess


# ============================================================
# 2. BATTERY DEGRADATION ANALYSIS
# ============================================================

def analyze_degradation(bess: pd.DataFrame, enh_bess: pd.DataFrame = None):
    """Comprehensive battery degradation analysis."""
    print("\n" + "=" * 70)
    print("  BATTERY DEGRADATION ANALYSIS")
    print("=" * 70)

    # Test multiple CAPEX scenarios
    scenarios = [
        ("Conservative (300 EUR/kWh, 4000 cycles)", 300.0, 4000),
        ("Mid-range  (250 EUR/kWh, 5000 cycles)", 250.0, 5000),
        ("Optimistic (200 EUR/kWh, 6000 cycles)", 200.0, 6000),
    ]

    gross_total = bess['rev_indicator'].sum()
    n_days = len(bess)
    n_years = n_days / 365.25

    enh_total = None
    if enh_bess is not None:
        enh_rev_col = 'rev_enh_nodeg' if 'rev_enh_nodeg' in enh_bess.columns else 'rev_enhanced'
        enh_total = enh_bess[enh_rev_col].sum()
        print(f"\n  Gross BESS baseline revenue: {gross_total:>12,.0f} EUR over {n_days} days ({n_years:.1f} years)")
        print(f"  Gross BESS enhanced revenue: {enh_total:>12,.0f} EUR (+{enh_total - gross_total:,.0f})")
    else:
        print(f"\n  Gross BESS revenue: {gross_total:>12,.0f} EUR over {n_days} days ({n_years:.1f} years)")
    print(f"  Assumed: 2 cycles/day, 1 MW / 2 MWh battery\n")

    print(f"  {'Scenario':<45s} {'Capex':>10s} {'Deg.Cost':>10s} {'Net Rev':>10s} {'Net/yr':>10s}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    results = []
    for name, capex, cycles in scenarios:
        model = BatteryDegradationModel(capex_per_kwh=capex, cycle_life=cycles)
        df = model.compute_degradation_costs(bess)
        total_deg = df['degradation_cost_eur'].sum()
        net_total = df['net_revenue_eur'].sum()
        net_annual = net_total / n_years

        print(f"  {name:<45s} {model.total_capex:>10,.0f} {total_deg:>10,.0f} "
              f"{net_total:>10,.0f} {net_annual:>10,.0f}")

        results.append({
            'scenario': name,
            'capex_per_kwh': capex,
            'cycle_life': cycles,
            'total_capex': model.total_capex,
            'total_degradation_cost': total_deg,
            'gross_revenue': gross_total,
            'net_revenue': net_total,
            'net_annual': net_annual,
            'payback_years': model.total_capex / (net_annual if net_annual > 0 else 1),
        })

    # Use mid-range for further analysis
    mid = BatteryDegradationModel(capex_per_kwh=250.0, cycle_life=5000)
    bess_deg = mid.compute_degradation_costs(bess)

    print(f"\n  Mid-range scenario details:")
    print(f"    Cost per cycle: {mid.amortized_cost_per_cycle:.1f} EUR")
    print(f"    Calendar aging cost/year: {mid.total_capex * mid.calendar_aging_per_year:,.0f} EUR")
    print(f"    Payback period: {mid.total_capex / (bess_deg['net_revenue_eur'].sum() / n_years):.1f} years")

    # Enhanced BESS degradation (mid-range only)
    if enh_total is not None:
        mid = BatteryDegradationModel(capex_per_kwh=250.0, cycle_life=5000)
        enh_deg_cost = mid.amortized_cost_per_cycle * 2 * n_days + \
                       mid.total_capex * mid.calendar_aging_per_year * n_years
        enh_net = enh_total - enh_deg_cost
        print(f"\n  Enhanced BESS (mid-range degradation):")
        print(f"    Gross: {enh_total:>10,.0f} EUR vs Baseline: {gross_total:>10,.0f} EUR")
        print(f"    Net:   {enh_net:>10,.0f} EUR vs Baseline: {gross_total - enh_deg_cost:>10,.0f} EUR")
        print(f"    Enhancement adds {enh_total - gross_total:>+,.0f} EUR gross, same degradation cost")

    pd.DataFrame(results).to_csv(OUT_DIR / 'degradation_scenarios.csv', index=False)
    return bess_deg


# ============================================================
# 3. TRANSACTION COST ANALYSIS
# ============================================================

def analyze_transaction_costs(spec: pd.DataFrame):
    """Analyze impact of transaction costs on speculative strategy."""
    print("\n" + "=" * 70)
    print("  TRANSACTION COST ANALYSIS (SPECULATIVE)")
    print("=" * 70)

    spreads = [0.5, 1.0, 2.0, 3.0, 5.0]
    gross_total = spec['pnl'].sum()

    print(f"\n  Gross speculative P&L: {gross_total:>12,.0f} EUR")
    print(f"  Total traded hours: {spec['n_traded'].sum():,.0f}\n")

    print(f"  {'IDM Spread (EUR/MWh)':<25s} {'Txn Cost':>10s} {'Net P&L':>10s} {'Impact':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")

    results = []
    for spread in spreads:
        model = TransactionCostModel(idm_spread_eur=spread)
        df = model.compute_speculative_costs(spec)
        total_cost = df['transaction_cost_eur'].sum()
        net_pnl = df['net_pnl'].sum()
        impact_pct = (1 - net_pnl / gross_total) * 100 if gross_total != 0 else 0

        print(f"  {spread:<25.1f} {total_cost:>10,.0f} {net_pnl:>10,.0f} {impact_pct:>7.1f}%")

        results.append({
            'idm_spread': spread,
            'transaction_cost': total_cost,
            'gross_pnl': gross_total,
            'net_pnl': net_pnl,
            'impact_pct': impact_pct,
        })

    pd.DataFrame(results).to_csv(OUT_DIR / 'transaction_cost_scenarios.csv', index=False)

    # Use 2.0 EUR spread as baseline
    tc_model = TransactionCostModel(idm_spread_eur=2.0)
    return tc_model.compute_speculative_costs(spec)


# ============================================================
# 4. COMBINED PORTFOLIO ANALYSIS
# ============================================================

def analyze_portfolio(combined: pd.DataFrame, bess_deg: pd.DataFrame,
                      spec_tc: pd.DataFrame):
    """Combined portfolio analysis."""
    print("\n" + "=" * 70)
    print("  COMBINED PORTFOLIO ANALYSIS")
    print("=" * 70)

    # Get net columns aligned with combined dates
    bess_net = bess_deg.set_index('date')['net_revenue_eur']
    spec_net = spec_tc.set_index('date')['net_pnl']

    # Build portfolio dataframe on common dates
    port = combined[['date']].copy()
    port = port.merge(
        bess_net.reset_index().rename(columns={'net_revenue_eur': 'bess_net'}),
        on='date', how='left'
    )
    port = port.merge(
        spec_net.reset_index().rename(columns={'net_pnl': 'spec_net'}),
        on='date', how='left'
    )
    port['bess_gross'] = combined['bess_pnl'].values
    port['spec_gross'] = combined['spec_pnl'].values

    # Enhanced BESS if available
    has_enhanced = 'bess_enhanced_pnl' in combined.columns
    if has_enhanced:
        port['bess_enh_gross'] = combined['bess_enhanced_pnl'].values
        # Compute net for enhanced BESS using same degradation model
        enh_net = bess_deg.copy()
        # Replace revenue with enhanced values on matching dates
        enh_map = combined.set_index('date')['bess_enhanced_pnl']
        enh_net_vals = port['bess_enh_gross'] - bess_deg.set_index('date').reindex(
            port['date'])['degradation_cost_eur'].values
        port['bess_enh_net'] = enh_net_vals

    # Combined portfolio (baseline)
    port['portfolio_gross'] = port['bess_gross'] + port['spec_gross']
    port['portfolio_net'] = port['bess_net'].fillna(port['bess_gross']) + \
                            port['spec_net'].fillna(port['spec_gross'])

    # Enhanced portfolio
    if has_enhanced:
        port['portfolio_enh_gross'] = port['bess_enh_gross'] + port['spec_gross']
        port['portfolio_enh_net'] = port['bess_enh_net'] + \
                                    port['spec_net'].fillna(port['spec_gross'])

    # Cumulative P&L
    port['cum_bess_gross'] = port['bess_gross'].cumsum()
    port['cum_spec_gross'] = port['spec_gross'].cumsum()
    port['cum_portfolio_gross'] = port['portfolio_gross'].cumsum()
    port['cum_bess_net'] = port['bess_net'].fillna(port['bess_gross']).cumsum()
    port['cum_spec_net'] = port['spec_net'].fillna(port['spec_gross']).cumsum()
    port['cum_portfolio_net'] = port['portfolio_net'].cumsum()
    if has_enhanced:
        port['cum_bess_enh_gross'] = port['bess_enh_gross'].cumsum()
        port['cum_portfolio_enh_gross'] = port['portfolio_enh_gross'].cumsum()

    n_days = len(port)
    n_years = n_days / 365.25

    # --- Individual Strategy Metrics ---
    strategies = {
        'BESS baseline (gross)': port['bess_gross'],
        'BESS baseline (net)': port['bess_net'].fillna(port['bess_gross']),
        'Speculative (gross)': port['spec_gross'],
        'Speculative (net of txn costs)': port['spec_net'].fillna(port['spec_gross']),
        'Portfolio baseline (gross)': port['portfolio_gross'],
        'Portfolio baseline (net)': port['portfolio_net'],
    }
    if has_enhanced:
        strategies['BESS enhanced (gross)'] = port['bess_enh_gross']
        strategies['BESS enhanced (net)'] = port['bess_enh_net']
        strategies['Portfolio enhanced (gross)'] = port['portfolio_enh_gross']
        strategies['Portfolio enhanced (net)'] = port['portfolio_enh_net']

    print(f"\n  Period: {port['date'].min().date()} to {port['date'].max().date()} ({n_days} days, {n_years:.1f} years)")
    print(f"\n  {'Strategy':<35s} {'Total':>10s} {'Ann.':>10s} {'Sharpe':>7s} {'Sortino':>8s} "
          f"{'MaxDD':>10s} {'WinRate':>8s} {'PF':>6s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*7} {'-'*8} {'-'*10} {'-'*8} {'-'*6}")

    metrics_list = []
    for name, returns in strategies.items():
        total = returns.sum()
        annual = total / n_years
        sharpe = compute_sharpe(returns)
        sortino = compute_sortino(returns)
        cum = returns.cumsum()
        max_dd, _, _ = compute_max_drawdown(cum)
        win_rate = (returns > 0).mean()
        pf = compute_profit_factor(returns)

        print(f"  {name:<35s} {total:>10,.0f} {annual:>10,.0f} {sharpe:>7.2f} {sortino:>8.2f} "
              f"{max_dd:>10,.0f} {win_rate:>8.1%} {pf:>6.2f}")

        metrics_list.append({
            'strategy': name,
            'total_pnl': total,
            'annual_pnl': annual,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': pf,
            'avg_daily_pnl': returns.mean(),
            'std_daily_pnl': returns.std(),
            'n_days': n_days,
        })

    # --- Correlation Analysis ---
    print(f"\n--- CORRELATION BETWEEN STRATEGIES ---")
    corr = port['bess_gross'].corr(port['spec_gross'])
    print(f"  Daily P&L correlation (BESS vs Speculative): {corr:.3f}")

    # Monthly correlation
    port['month'] = port['date'].dt.to_period('M')
    monthly = port.groupby('month').agg(
        bess=('bess_gross', 'sum'),
        spec=('spec_gross', 'sum'),
    )
    monthly_corr = monthly['bess'].corr(monthly['spec'])
    print(f"  Monthly P&L correlation: {monthly_corr:.3f}")

    # --- Diversification Benefit ---
    print(f"\n--- DIVERSIFICATION BENEFIT ---")
    bess_vol = port['bess_gross'].std() * np.sqrt(252)
    spec_vol = port['spec_gross'].std() * np.sqrt(252)
    port_vol = port['portfolio_gross'].std() * np.sqrt(252)
    undiv_vol = bess_vol + spec_vol

    div_benefit = 1 - port_vol / undiv_vol if undiv_vol > 0 else 0
    print(f"  BESS annual volatility:      {bess_vol:>10,.0f} EUR")
    print(f"  Speculative annual vol:      {spec_vol:>10,.0f} EUR")
    print(f"  Undiversified portfolio vol: {undiv_vol:>10,.0f} EUR")
    print(f"  Actual portfolio vol:        {port_vol:>10,.0f} EUR")
    print(f"  Diversification benefit:     {div_benefit:>10.1%}")

    # Save metrics
    pd.DataFrame(metrics_list).to_csv(OUT_DIR / 'strategy_metrics.csv', index=False)
    port.to_csv(OUT_DIR / 'portfolio_daily.csv', index=False)

    return port, metrics_list


# ============================================================
# 5. WALK-FORWARD ANALYSIS
# ============================================================

def analyze_walk_forward(port: pd.DataFrame):
    """Rolling performance analysis."""
    print("\n" + "=" * 70)
    print("  WALK-FORWARD ROLLING ANALYSIS")
    print("=" * 70)

    windows = [30, 90, 180]

    for name, col in [('BESS', 'bess_gross'), ('Speculative', 'spec_gross'),
                       ('Portfolio', 'portfolio_gross')]:
        roll = rolling_metrics(port[col], port['date'], windows)

        print(f"\n  --- {name} ---")
        for w in windows:
            latest = roll[f'sharpe_{w}d'].dropna()
            if len(latest) > 0:
                current = latest.iloc[-1]
                avg = latest.mean()
                print(f"    {w}d rolling Sharpe: current={current:.2f}, avg={avg:.2f}, "
                      f"min={latest.min():.2f}, max={latest.max():.2f}")

        # Save
        roll.to_csv(OUT_DIR / f'rolling_{name.lower()}.csv', index=False)


# ============================================================
# 6. STRESS TESTING
# ============================================================

def analyze_stress(port: pd.DataFrame):
    """Stress testing and Monte Carlo analysis."""
    print("\n" + "=" * 70)
    print("  STRESS TESTING")
    print("=" * 70)

    for name, col in [('BESS', 'bess_gross'), ('Speculative', 'spec_gross'),
                       ('Portfolio', 'portfolio_gross')]:
        print(f"\n  --- {name}: Worst Days ---")
        worst = identify_worst_days(port[col], port['date'], n=10)
        for _, row in worst.iterrows():
            date = row['date']
            pnl = row['pnl']
            # Find context for that day
            day_data = port[port['date'] == date].iloc[0]
            price_info = ""
            if 'price_range' in day_data.index and not pd.isna(day_data.get('price_range', np.nan)):
                price_info = f"  range={day_data['price_range']:.0f}"
            print(f"    {pd.Timestamp(date).date()}: {pnl:>10,.0f} EUR{price_info}")

        # Clustered stress
        cluster = clustered_stress_test(port[col])
        print(f"  Worst 10 days combined: {cluster['worst_10_days_total']:>10,.0f} EUR")
        print(f"  Worst 5-day cluster:    {cluster['worst_5_days_cluster']:>10,.0f} EUR")

    # --- Monte Carlo ---
    print(f"\n  --- MONTE CARLO SIMULATION (10,000 annual scenarios) ---")
    mc_results = {}
    for name, col in [('BESS', 'bess_gross'), ('Speculative', 'spec_gross'),
                       ('Portfolio', 'portfolio_gross')]:
        mc = monte_carlo_annual_pnl(port[col], n_simulations=10000, trading_days=365)
        mc_results[name] = mc

        print(f"\n  {name}:")
        print(f"    Mean annual P&L:    {mc['mean']:>12,.0f} EUR")
        print(f"    Median annual P&L:  {mc['median']:>12,.0f} EUR")
        print(f"    Std dev:            {mc['std']:>12,.0f} EUR")
        print(f"    95% VaR:            {mc['var_95']:>12,.0f} EUR")
        print(f"    95% CVaR (ES):      {mc['cvar_95']:>12,.0f} EUR")
        print(f"    P(profit > 0):      {mc['prob_positive']:>12.1%}")
        print(f"    5th percentile:     {mc['p5']:>12,.0f} EUR")
        print(f"    95th percentile:    {mc['p95']:>12,.0f} EUR")
        print(f"    Best year:          {mc['best']:>12,.0f} EUR")
        print(f"    Worst year:         {mc['worst']:>12,.0f} EUR")

    # Save MC results
    mc_df = pd.DataFrame(mc_results).T
    mc_df.index.name = 'strategy'
    mc_df.to_csv(OUT_DIR / 'monte_carlo_results.csv')

    return mc_results


# ============================================================
# 7. BREAK-EVEN ANALYSIS
# ============================================================

def analyze_break_even(port: pd.DataFrame, bess: pd.DataFrame, spec: pd.DataFrame):
    """How bad can conditions get before we lose money?"""
    print("\n" + "=" * 70)
    print("  BREAK-EVEN ANALYSIS")
    print("=" * 70)

    n_years = len(port) / 365.25

    # BESS break-even
    bess_total = port['bess_gross'].sum()
    bess_annual = bess_total / n_years
    bess_daily_avg = port['bess_gross'].mean()

    # Mid-range degradation cost
    deg = BatteryDegradationModel(capex_per_kwh=250.0, cycle_life=5000)
    annual_deg_cost = deg.amortized_cost_per_cycle * 2 * 365.25 + \
                      deg.total_capex * deg.calendar_aging_per_year
    bess_net_annual = bess_annual - annual_deg_cost

    print(f"\n  BESS Break-Even:")
    print(f"    Current annual gross revenue:    {bess_annual:>10,.0f} EUR")
    print(f"    Annual degradation cost:         {annual_deg_cost:>10,.0f} EUR")
    print(f"    Current annual net revenue:      {bess_net_annual:>10,.0f} EUR")
    print(f"    Revenue can drop by:             {bess_net_annual/bess_annual*100:>10.1f}% before loss")

    # Price volatility sensitivity
    # Merge price_range from bess data if not already in port
    if 'price_range' not in port.columns:
        bess_pr = bess[['date', 'price_range']].copy()
        bess_pr['date'] = pd.to_datetime(bess_pr['date'])
        port = port.merge(bess_pr, on='date', how='left')

    if 'price_range' in port.columns and port['price_range'].notna().sum() > 10:
        print(f"\n    Price range sensitivity:")
        port['price_range_q'] = pd.qcut(
            port['price_range'].dropna(), 4,
            labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)']
        ).reindex(port.index)
        for q in ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']:
            sub = port[port['price_range_q'] == q]
            if len(sub) > 0:
                avg_rev = sub['bess_gross'].mean()
                avg_range = sub['price_range'].mean()
                print(f"      {q}: avg_range={avg_range:.0f} EUR/MWh, avg_daily_rev={avg_rev:.0f} EUR")

    # Speculative break-even
    spec_total = port['spec_gross'].sum()
    spec_annual = spec_total / n_years
    print(f"\n  Speculative Break-Even:")
    print(f"    Current annual gross P&L:        {spec_annual:>10,.0f} EUR")
    print(f"    With 2 EUR/MWh txn cost:")
    # Estimate traded hours per year from spec data
    traded_per_year = spec['n_traded'].sum() / (len(spec) / 365.25) if len(spec) > 0 else 0
    annual_txn_cost = traded_per_year * 1.0  # half spread of 2 EUR
    spec_net_annual = spec_annual - annual_txn_cost
    print(f"    Annual transaction costs:         {annual_txn_cost:>10,.0f} EUR")
    print(f"    Net annual:                       {spec_net_annual:>10,.0f} EUR")
    print(f"    Accuracy can drop to:             ", end="")
    # Rough estimate: at what accuracy does mean(signal * spread) = 0?
    # P&L = accuracy * avg_win - (1-accuracy) * avg_loss
    wins = port.loc[port['spec_gross'] > 0, 'spec_gross']
    losses = port.loc[port['spec_gross'] < 0, 'spec_gross']
    if len(wins) > 0 and len(losses) > 0:
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        breakeven_acc = avg_loss / (avg_win + avg_loss)
        print(f"{breakeven_acc:.1%} (currently {(port['spec_gross'] > 0).mean():.1%})")
    else:
        print("N/A")

    # Portfolio break-even
    print(f"\n  Portfolio Break-Even:")
    port_annual = port['portfolio_gross'].sum() / n_years
    total_costs = annual_deg_cost + annual_txn_cost
    print(f"    Combined annual gross:           {port_annual:>10,.0f} EUR")
    print(f"    Combined annual costs:           {total_costs:>10,.0f} EUR")
    print(f"    Net annual:                      {port_annual - total_costs:>10,.0f} EUR")
    print(f"    Margin of safety:                {(port_annual - total_costs)/port_annual*100:>10.1f}%")


# ============================================================
# 8. REGIME ANALYSIS
# ============================================================

def analyze_regimes(port: pd.DataFrame, bess: pd.DataFrame):
    """Identify performance regimes and structural breaks."""
    print("\n" + "=" * 70)
    print("  REGIME ANALYSIS")
    print("=" * 70)

    # Ensure we have price_range and price_std from bess data
    if 'price_range' not in port.columns:
        bess_cols = bess[['date', 'price_range', 'price_std']].copy()
        bess_cols['date'] = pd.to_datetime(bess_cols['date'])
        port = port.merge(bess_cols, on='date', how='left')

    port['month'] = port['date'].dt.to_period('M')
    port['year'] = port['date'].dt.year
    port['quarter'] = port['date'].dt.to_period('Q')

    # Quarterly performance
    agg_dict = {
        'bess_gross': ('bess_gross', 'sum'),
        'spec_gross': ('spec_gross', 'sum'),
        'portfolio_gross': ('portfolio_gross', 'sum'),
        'n_days': ('date', 'count'),
    }
    if 'price_range' in port.columns:
        agg_dict['avg_price_range'] = ('price_range', 'mean')

    quarterly = port.groupby('quarter').agg(**agg_dict).reset_index()

    print(f"\n  --- QUARTERLY PERFORMANCE ---")
    has_range = 'avg_price_range' in quarterly.columns
    header = f"  {'Quarter':<10s} {'BESS':>10s} {'Spec':>10s} {'Portfolio':>10s}"
    if has_range:
        header += f" {'AvgRange':>10s}"
    header += f" {'Days':>5s}"
    print(header)
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}" +
          (f" {'-'*10}" if has_range else "") + f" {'-'*5}")
    for _, row in quarterly.iterrows():
        line = (f"  {str(row['quarter']):<10s} {row['bess_gross']:>10,.0f} {row['spec_gross']:>10,.0f} "
                f"{row['portfolio_gross']:>10,.0f}")
        if has_range:
            line += f" {row['avg_price_range']:>10.1f}"
        line += f" {row['n_days']:>5.0f}"
        print(line)

    quarterly.to_csv(OUT_DIR / 'quarterly_performance.csv', index=False)

    # Performance by volatility regime
    if 'price_std' in port.columns and port['price_std'].notna().sum() > 10:
        print(f"\n  --- PERFORMANCE BY PRICE VOLATILITY REGIME ---")
        port['vol_regime'] = pd.qcut(port['price_std'].dropna(), 3,
                                      labels=['Low', 'Medium', 'High']).reindex(port.index)
        vol_perf = port.dropna(subset=['vol_regime']).groupby('vol_regime').agg(
            bess_avg=('bess_gross', 'mean'),
            spec_avg=('spec_gross', 'mean'),
            port_avg=('portfolio_gross', 'mean'),
            bess_std=('bess_gross', 'std'),
            n=('date', 'count'),
        ).reset_index()

        for _, row in vol_perf.iterrows():
            print(f"  {row['vol_regime']:<8s}: BESS={row['bess_avg']:>7.0f} +/- {row['bess_std']:.0f}, "
                  f"Spec={row['spec_avg']:>7.0f}, Portfolio={row['port_avg']:>7.0f} (n={row['n']:.0f})")


# ============================================================
# 9. GENERATE CHARTS
# ============================================================

def generate_charts(port: pd.DataFrame, mc_results: dict):
    """Generate analysis charts."""
    print("\n[*] Generating portfolio analysis charts...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Chart 1: Cumulative P&L Comparison ---
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(port['date'], port['cum_bess_gross'], 'b-', lw=1.5, label='BESS baseline (gross)')
    if 'cum_bess_enh_gross' in port.columns:
        ax.plot(port['date'], port['cum_bess_enh_gross'], 'b--', lw=1.5, label='BESS enhanced (gross)')
    ax.plot(port['date'], port['cum_spec_gross'], 'r-', lw=1.5, label='Speculative (gross)')
    ax.plot(port['date'], port['cum_portfolio_gross'], 'k-', lw=2, label='Portfolio baseline')
    if 'cum_portfolio_enh_gross' in port.columns:
        ax.plot(port['date'], port['cum_portfolio_enh_gross'], 'k--', lw=2.5, label='Portfolio enhanced')
    ax.fill_between(port['date'], 0, port['cum_portfolio_gross'], alpha=0.05, color='green')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative P&L (EUR)')
    ax.set_title('Combined Portfolio: Cumulative P&L (Baseline vs Enhanced)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'portfolio_cumulative_pnl.png', dpi=150)
    plt.close(fig)

    # --- Chart 2: Rolling Sharpe ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for idx, (name, col) in enumerate([('BESS', 'bess_gross'),
                                        ('Speculative', 'spec_gross'),
                                        ('Portfolio', 'portfolio_gross')]):
        ax = axes[idx]
        for w, color in [(30, 'lightblue'), (90, 'blue'), (180, 'darkblue')]:
            roll = port[col].rolling(w, min_periods=max(w//2, 10))
            sharpe = roll.mean() / roll.std() * np.sqrt(252)
            ax.plot(port['date'], sharpe, color=color, lw=1, label=f'{w}d', alpha=0.8)
        ax.axhline(0, color='red', ls='--', alpha=0.5)
        ax.set_ylabel('Rolling Sharpe')
        ax.set_title(f'{name}')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_ylim(-5, 10)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Rolling Sharpe Ratios', fontweight='bold', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'rolling_sharpe.png', dpi=150)
    plt.close(fig)

    # --- Chart 3: Daily P&L Scatter (BESS vs Spec) ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(port['bess_gross'], port['spec_gross'], alpha=0.2, s=10, c='blue')
    corr = port['bess_gross'].corr(port['spec_gross'])
    ax.set_xlabel('BESS Daily P&L (EUR)')
    ax.set_ylabel('Speculative Daily P&L (EUR)')
    ax.set_title(f'Daily P&L Correlation: r = {corr:.3f}')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.axvline(0, color='gray', ls='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'pnl_correlation.png', dpi=150)
    plt.close(fig)

    # --- Chart 4: Drawdown Analysis ---
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, col, color in [('BESS', 'cum_bess_gross', 'blue'),
                               ('Speculative', 'cum_spec_gross', 'red'),
                               ('Portfolio', 'cum_portfolio_gross', 'black')]:
        cum = port[col]
        running_max = cum.cummax()
        drawdown = cum - running_max
        ax.fill_between(port['date'], drawdown, 0, alpha=0.2, color=color, label=name)
        ax.plot(port['date'], drawdown, color=color, lw=0.5, alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (EUR)')
    ax.set_title('Drawdown Analysis')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'drawdown_analysis.png', dpi=150)
    plt.close(fig)

    # --- Chart 5: Monte Carlo Distribution ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, name in enumerate(['BESS', 'Speculative', 'Portfolio']):
        ax = axes[idx]
        mc = mc_results[name]
        # Regenerate distribution for plotting
        rng = np.random.default_rng(42)
        daily_col = {'BESS': 'bess_gross', 'Speculative': 'spec_gross',
                     'Portfolio': 'portfolio_gross'}[name]
        annual_pnls = [rng.choice(port[daily_col].values, size=365, replace=True).sum()
                       for _ in range(10000)]
        ax.hist(annual_pnls, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(mc['mean'], color='red', ls='--', label=f"Mean: {mc['mean']:,.0f}")
        ax.axvline(mc['p5'], color='orange', ls='--', label=f"5th pct: {mc['p5']:,.0f}")
        ax.axvline(0, color='black', ls='-', lw=2)
        ax.set_title(f'{name}\nP(profit)={mc["prob_positive"]:.1%}')
        ax.set_xlabel('Annual P&L (EUR)')
        ax.legend(fontsize=7)
    fig.suptitle('Monte Carlo: Annual P&L Distribution (10,000 simulations)', fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'monte_carlo_distributions.png', dpi=150)
    plt.close(fig)

    print(f"[+] Charts saved to {OUT_DIR}")


# ============================================================
# 10. EXECUTIVE SUMMARY
# ============================================================

def print_executive_summary(port: pd.DataFrame, mc_results: dict):
    """Print actionable executive summary."""
    print("\n" + "=" * 70)
    print("  EXECUTIVE SUMMARY")
    print("=" * 70)

    n_days = len(port)
    n_years = n_days / 365.25

    bess_total = port['bess_gross'].sum()
    spec_total = port['spec_gross'].sum()
    port_total = port['portfolio_gross'].sum()

    bess_sharpe = compute_sharpe(port['bess_gross'])
    spec_sharpe = compute_sharpe(port['spec_gross'])
    port_sharpe = compute_sharpe(port['portfolio_gross'])

    bess_dd, _, _ = compute_max_drawdown(port['cum_bess_gross'])
    spec_dd, _, _ = compute_max_drawdown(port['cum_spec_gross'])
    port_dd, _, _ = compute_max_drawdown(port['cum_portfolio_gross'])

    corr = port['bess_gross'].corr(port['spec_gross'])

    has_enh = 'bess_enh_gross' in port.columns
    if has_enh:
        enh_total = port['bess_enh_gross'].sum()
        enh_port_total = port['portfolio_enh_gross'].sum()
        enh_sharpe = compute_sharpe(port['bess_enh_gross'])
        enh_port_sharpe = compute_sharpe(port['portfolio_enh_gross'])
        enh_dd, _, _ = compute_max_drawdown(port['cum_bess_enh_gross'])
        enh_port_dd, _, _ = compute_max_drawdown(port['cum_portfolio_enh_gross'])
        enh_improvement = enh_total - bess_total

    if has_enh:
        print(f"\n  COMBINED PORTFOLIO ({n_days} days, {n_years:.1f} years)")
        print(f"  ====================================================\n")
        print(f"  Strategy Performance (Gross, overlapping period only):")
        print(f"  -------------------------------------------------------")
        print(f"                        BESS base  BESS enh   Speculative  Port(base)  Port(enh)")
        print(f"  Total P&L (EUR):   {bess_total:>10,.0f} {enh_total:>10,.0f}  {spec_total:>10,.0f}  {port_total:>10,.0f}  {enh_port_total:>10,.0f}")
        print(f"  Annual P&L (EUR):  {bess_total/n_years:>10,.0f} {enh_total/n_years:>10,.0f}  {spec_total/n_years:>10,.0f}  {port_total/n_years:>10,.0f}  {enh_port_total/n_years:>10,.0f}")
        print(f"  Sharpe Ratio:      {bess_sharpe:>10.2f} {enh_sharpe:>10.2f}  {spec_sharpe:>10.2f}  {port_sharpe:>10.2f}  {enh_port_sharpe:>10.2f}")
        print(f"  Max Drawdown:      {bess_dd:>10,.0f} {enh_dd:>10,.0f}  {spec_dd:>10,.0f}  {port_dd:>10,.0f}  {enh_port_dd:>10,.0f}")
        print(f"\n  Enhanced BESS improvement: +{enh_improvement:,.0f} EUR ({enh_improvement/bess_total*100:.1f}%)")
        print(f"  Correlation (base): {corr:.3f} (low = good diversification)")
    else:
        print(f"\n  COMBINED PORTFOLIO ({n_days} days, {n_years:.1f} years)")
        print(f"  ====================================================\n")
        print(f"  Strategy Performance (Gross, overlapping period only):")
        print(f"  -------------------------------------------------------")
        print(f"                        BESS      Speculative  Portfolio")
        print(f"  Total P&L (EUR):   {bess_total:>10,.0f}    {spec_total:>10,.0f}  {port_total:>10,.0f}")
        print(f"  Annual P&L (EUR):  {bess_total/n_years:>10,.0f}    {spec_total/n_years:>10,.0f}  {port_total/n_years:>10,.0f}")
        print(f"  Sharpe Ratio:      {bess_sharpe:>10.2f}    {spec_sharpe:>10.2f}  {port_sharpe:>10.2f}")
        print(f"  Max Drawdown:      {bess_dd:>10,.0f}    {spec_dd:>10,.0f}  {port_dd:>10,.0f}")
        print(f"\n  Correlation: {corr:.3f} (low correlation = good diversification)")

    # Monte Carlo summary
    mc_bess = mc_results['BESS']
    mc_spec = mc_results['Speculative']
    mc_port = mc_results['Portfolio']
    print(f"\n  Monte Carlo (annual projection):")
    print(f"  ---------------------------------")
    print(f"  BESS:        Mean={mc_bess['mean']:>10,.0f}, 5th pct={mc_bess['p5']:>10,.0f}, P(+)={mc_bess['prob_positive']:.1%}")
    print(f"  Speculative: Mean={mc_spec['mean']:>10,.0f}, 5th pct={mc_spec['p5']:>10,.0f}, P(+)={mc_spec['prob_positive']:.1%}")
    print(f"  Portfolio:   Mean={mc_port['mean']:>10,.0f}, 5th pct={mc_port['p5']:>10,.0f}, P(+)={mc_port['prob_positive']:.1%}")

    print(f"\n  KEY FINDINGS:")
    print(f"  1. BESS provides stable base revenue with {bess_sharpe:.1f} Sharpe ratio")
    if has_enh:
        print(f"  2. Enhanced BESS adds +{enh_improvement:,.0f} EUR over baseline ({enh_improvement/bess_total*100:.1f}% improvement)")
        print(f"  3. Speculative adds alpha but with higher volatility")
        print(f"  4. Low correlation ({corr:.2f}) means strong diversification benefit")
        print(f"  5. Combined enhanced portfolio: {enh_port_total/n_years:,.0f} EUR/year (Sharpe {enh_port_sharpe:.2f})")
        print(f"  6. Monte Carlo: {mc_port['prob_positive']:.0%} probability of positive annual return")
    else:
        print(f"  2. Speculative adds alpha but with higher volatility")
        print(f"  3. Low correlation ({corr:.2f}) means strong diversification benefit")
        print(f"  4. Combined portfolio has better risk-adjusted returns (Sharpe {port_sharpe:.2f})")
        print(f"  5. Monte Carlo: {mc_port['prob_positive']:.0%} probability of positive annual return")


# ============================================================
# MAIN
# ============================================================

def main():
    setup()

    # Load data
    bess, spec, combined, enh_bess = load_and_prepare()

    # Battery degradation
    bess_deg = analyze_degradation(bess, enh_bess)

    # Transaction costs
    spec_tc = analyze_transaction_costs(spec)

    # Combined portfolio
    port, metrics = analyze_portfolio(combined, bess_deg, spec_tc)

    # Walk-forward
    analyze_walk_forward(port)

    # Stress testing & Monte Carlo
    mc_results = analyze_stress(port)

    # Break-even
    analyze_break_even(port, bess, spec)

    # Regime analysis
    analyze_regimes(port, bess)

    # Charts
    generate_charts(port, mc_results)

    # Executive summary
    print_executive_summary(port, mc_results)

    print("\n" + "=" * 70)
    print(f"  [+] All results saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
