"""
Backtest Engine - Unified simulation and backtesting framework
==============================================================
Provides:
  1. Battery degradation model (cycle counting, capacity fade, calendar aging)
  2. Transaction cost model
  3. Walk-forward rolling analysis
  4. Stress testing and Monte Carlo
  5. Combined portfolio analysis

Usage:
  python backtest_engine.py
  (or import individual classes/functions)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


BASE = Path(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data')
DATA_DIR = BASE / 'MarketPriceGap' / 'da_indicator' / 'data'


# ============================================================
# 1. BATTERY DEGRADATION MODEL
# ============================================================

@dataclass
class BatteryDegradationModel:
    """
    Models battery degradation for a grid-scale BESS.

    Parameters
    ----------
    capacity_kwh : float
        Nameplate energy capacity in kWh (2000 for 1MW/2MWh).
    capex_per_kwh : float
        Total installed cost per kWh (EUR).
    cycle_life : int
        Number of equivalent full cycles until end-of-life (~80% capacity).
    calendar_life_years : float
        Calendar aging: years until end-of-life regardless of cycling.
    capacity_fade_per_cycle : float
        Fractional capacity loss per equivalent full cycle.
    """
    capacity_kwh: float = 2000.0
    capex_per_kwh: float = 250.0     # EUR/kWh, mid-range 2024-2026
    cycle_life: int = 5000
    calendar_life_years: float = 15.0
    capacity_fade_per_cycle: float = 0.00004  # 0.004% per cycle -> ~20% at 5000 cycles

    @property
    def total_capex(self) -> float:
        """Total battery system cost in EUR."""
        return self.capacity_kwh * self.capex_per_kwh

    @property
    def amortized_cost_per_cycle(self) -> float:
        """EUR cost per equivalent full cycle (linear amortization)."""
        return self.total_capex / self.cycle_life

    @property
    def calendar_aging_per_year(self) -> float:
        """Fraction of capacity lost per year from calendar aging."""
        return 0.20 / self.calendar_life_years  # ~1.33% per year

    def compute_degradation_costs(self, daily_results: pd.DataFrame) -> pd.DataFrame:
        """
        Add degradation cost columns to daily BESS results.

        Expects columns: date, rev_indicator
        Computes: equivalent_cycles, cycle_cost, net_revenue
        """
        df = daily_results.copy()

        # Each day with 2 charge-discharge cycles = 2 equivalent full cycles
        # (charge 2 MWh + discharge 2 MWh = 1 cycle of 2 MWh capacity)
        # With 4 charge + 4 discharge hours at 1 MW: 4 MWh in, 4 MWh out
        # = 2 full cycles per day
        df['equiv_full_cycles'] = 2.0  # 2 cycles/day by design

        # Days with zero or very low revenue likely had constraints -> fewer cycles
        low_rev_mask = df['rev_indicator'].abs() < 1.0
        df.loc[low_rev_mask, 'equiv_full_cycles'] = 0.5

        # Cycle degradation cost
        df['cycle_cost_eur'] = df['equiv_full_cycles'] * self.amortized_cost_per_cycle

        # Calendar aging cost (spread daily)
        annual_calendar_cost = self.total_capex * self.calendar_aging_per_year
        df['calendar_cost_eur'] = annual_calendar_cost / 365.25

        # Total degradation cost
        df['degradation_cost_eur'] = df['cycle_cost_eur'] + df['calendar_cost_eur']

        # Net revenue after degradation
        df['net_revenue_eur'] = df['rev_indicator'] - df['degradation_cost_eur']

        return df


# ============================================================
# 2. TRANSACTION COST MODEL
# ============================================================

@dataclass
class TransactionCostModel:
    """
    Models transaction costs for electricity trading.

    Parameters
    ----------
    da_fee_pct : float
        DA market fee as fraction of traded value (typically 0 for clearing).
    idm_spread_eur : float
        Estimated IDM bid-ask spread in EUR/MWh.
    imbalance_risk_premium : float
        Additional cost for imbalance exposure in EUR/MWh.
    """
    da_fee_pct: float = 0.0
    idm_spread_eur: float = 2.0   # conservative estimate
    imbalance_risk_premium: float = 5.0

    def compute_bess_costs(self, daily_results: pd.DataFrame) -> pd.DataFrame:
        """Add transaction costs for BESS (DA market only, minimal costs)."""
        df = daily_results.copy()
        # BESS trades in DA market: clearing price, no spread
        # Only cost: exchange fee (negligible for most markets)
        df['transaction_cost_eur'] = 0.0
        return df

    def compute_speculative_costs(self, daily_results: pd.DataFrame) -> pd.DataFrame:
        """Add transaction costs for speculative trades."""
        df = daily_results.copy()
        # Each traded hour incurs half the spread (we cross the spread)
        df['transaction_cost_eur'] = df['n_traded'] * (self.idm_spread_eur / 2.0)
        df['net_pnl'] = df['pnl'] - df['transaction_cost_eur']
        return df


# ============================================================
# 3. RISK METRICS
# ============================================================

def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.0,
                   annualization: float = 252.0) -> float:
    """Annualized Sharpe ratio from daily returns."""
    excess = returns - risk_free_rate / annualization
    if excess.std() == 0 or len(excess) < 2:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(annualization))


def compute_sortino(returns: pd.Series, risk_free_rate: float = 0.0,
                    annualization: float = 252.0) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    excess = returns - risk_free_rate / annualization
    downside = excess[excess < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(annualization))


def compute_max_drawdown(cumulative_pnl: pd.Series) -> tuple:
    """
    Maximum drawdown from cumulative P&L.
    Returns (max_drawdown_eur, drawdown_start, drawdown_end).
    """
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_dd = drawdown.min()
    if max_dd == 0:
        return 0.0, None, None
    dd_end_idx = drawdown.idxmin()
    dd_start_idx = cumulative_pnl.loc[:dd_end_idx].idxmax()
    return float(max_dd), dd_start_idx, dd_end_idx


def compute_calmar(total_return: float, max_drawdown: float,
                   n_years: float) -> float:
    """Calmar ratio = annualized return / max drawdown."""
    if max_drawdown == 0 or n_years == 0:
        return 0.0
    ann_return = total_return / n_years
    return ann_return / abs(max_drawdown)


def compute_profit_factor(returns: pd.Series) -> float:
    """Profit factor = gross profits / gross losses."""
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


# ============================================================
# 4. WALK-FORWARD ROLLING ANALYSIS
# ============================================================

def rolling_metrics(daily_pnl: pd.Series, dates: pd.Series,
                    windows: list = None) -> pd.DataFrame:
    """
    Compute rolling performance metrics over multiple windows.

    Parameters
    ----------
    daily_pnl : pd.Series
        Daily P&L values.
    dates : pd.Series
        Corresponding dates.
    windows : list
        Rolling window sizes in days.

    Returns
    -------
    pd.DataFrame with rolling metrics for each window.
    """
    if windows is None:
        windows = [30, 90, 180]

    result = pd.DataFrame({'date': dates, 'daily_pnl': daily_pnl.values})

    for w in windows:
        roll = daily_pnl.rolling(w, min_periods=max(w // 2, 10))
        result[f'mean_{w}d'] = roll.mean().values
        result[f'std_{w}d'] = roll.std().values
        result[f'sharpe_{w}d'] = (
            roll.mean().values / np.where(roll.std().values > 0, roll.std().values, np.nan)
            * np.sqrt(252)
        )
        result[f'total_{w}d'] = roll.sum().values
        result[f'win_rate_{w}d'] = roll.apply(lambda x: (x > 0).mean()).values
        # Rolling max drawdown
        cum = daily_pnl.cumsum()
        dd_vals = []
        for i in range(len(daily_pnl)):
            start = max(0, i - w + 1)
            window_cum = cum.iloc[start:i+1] - cum.iloc[start:i+1].cummax()
            dd_vals.append(window_cum.min() if len(window_cum) > 0 else 0.0)
        result[f'max_dd_{w}d'] = dd_vals

    return result


# ============================================================
# 5. STRESS TESTING
# ============================================================

def identify_worst_days(daily_pnl: pd.Series, dates: pd.Series,
                        n: int = 10) -> pd.DataFrame:
    """Identify the N worst days."""
    df = pd.DataFrame({'date': dates, 'pnl': daily_pnl.values})
    return df.nsmallest(n, 'pnl').reset_index(drop=True)


def monte_carlo_annual_pnl(daily_pnl: pd.Series, n_simulations: int = 10000,
                           trading_days: int = 365) -> dict:
    """
    Monte Carlo simulation of annual P&L by resampling daily returns.

    Returns distribution statistics.
    """
    rng = np.random.default_rng(42)
    daily_vals = daily_pnl.values
    annual_pnls = []

    for _ in range(n_simulations):
        sampled = rng.choice(daily_vals, size=trading_days, replace=True)
        annual_pnls.append(sampled.sum())

    annual_pnls = np.array(annual_pnls)
    return {
        'mean': float(np.mean(annual_pnls)),
        'median': float(np.median(annual_pnls)),
        'std': float(np.std(annual_pnls)),
        'p5': float(np.percentile(annual_pnls, 5)),
        'p10': float(np.percentile(annual_pnls, 10)),
        'p25': float(np.percentile(annual_pnls, 25)),
        'p75': float(np.percentile(annual_pnls, 75)),
        'p90': float(np.percentile(annual_pnls, 90)),
        'p95': float(np.percentile(annual_pnls, 95)),
        'prob_positive': float((annual_pnls > 0).mean()),
        'prob_loss_gt_50k': float((annual_pnls < -50000).mean()),
        'worst': float(np.min(annual_pnls)),
        'best': float(np.max(annual_pnls)),
        'var_95': float(np.percentile(annual_pnls, 5)),  # 95% VaR
        'cvar_95': float(np.mean(annual_pnls[annual_pnls <= np.percentile(annual_pnls, 5)])),
    }


def clustered_stress_test(daily_pnl: pd.Series, n_worst: int = 10,
                          cluster_days: int = 5) -> dict:
    """
    What if the N worst days happened consecutively?
    Simulate clustered worst-case scenario.
    """
    worst = daily_pnl.nsmallest(n_worst).values
    # Sort worst days (most negative first)
    worst_sorted = np.sort(worst)

    # Scenario 1: all N worst days consecutively
    total_loss_all = worst_sorted.sum()

    # Scenario 2: worst cluster_days in a row
    total_loss_cluster = worst_sorted[:cluster_days].sum()

    return {
        f'worst_{n_worst}_days_total': float(total_loss_all),
        f'worst_{cluster_days}_days_cluster': float(total_loss_cluster),
        'worst_single_day': float(worst_sorted[0]),
        'avg_worst_day': float(np.mean(worst_sorted)),
    }


# ============================================================
# 6. DATA LOADING
# ============================================================

def load_bess_daily() -> pd.DataFrame:
    """Load BESS persistent SoC daily results."""
    path = DATA_DIR / 'bess_persistent_daily.csv'
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_speculative_daily() -> pd.DataFrame:
    """Load speculative daily results."""
    path = DATA_DIR / 'speculative_daily.csv'
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_enhanced_bess_daily() -> pd.DataFrame:
    """Load enhanced BESS daily results (from Task #3)."""
    path = BASE / 'MarketPriceGap' / 'strategies' / 'bess' / 'enhanced_bess_daily.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


# ============================================================
# MAIN (standalone test)
# ============================================================

if __name__ == '__main__':
    print("[*] Backtest engine loaded successfully")
    print("[*] Loading data to verify...")

    bess = load_bess_daily()
    spec = load_speculative_daily()

    print(f"[+] BESS daily: {len(bess)} days, {bess['date'].min().date()} to {bess['date'].max().date()}")
    print(f"[+] Speculative daily: {len(spec)} days, {spec['date'].min().date()} to {spec['date'].max().date()}")

    # Quick degradation test
    deg = BatteryDegradationModel()
    print(f"\n[*] Battery degradation model:")
    print(f"    Total CAPEX: {deg.total_capex:,.0f} EUR")
    print(f"    Cost per cycle: {deg.amortized_cost_per_cycle:.1f} EUR")
    print(f"    Calendar aging: {deg.calendar_aging_per_year:.2%}/year")

    bess_deg = deg.compute_degradation_costs(bess)
    total_deg = bess_deg['degradation_cost_eur'].sum()
    total_net = bess_deg['net_revenue_eur'].sum()
    print(f"    Total degradation cost: {total_deg:,.0f} EUR")
    print(f"    Net BESS revenue after degradation: {total_net:,.0f} EUR")
    print(f"    Gross BESS revenue: {bess['rev_indicator'].sum():,.0f} EUR")
