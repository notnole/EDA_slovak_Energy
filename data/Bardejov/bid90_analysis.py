"""Bid-at-90 strategy backtest across 2025 months.

Strategy: always bid to produce 8 MW when DA price >= 90 EUR/MWh.
If price clears below 90, bid not filled -> stay at heat-driven plan.
No forecast needed -- it's a price-limited bid.

For each month, compare (observed) vs (counterfactual: had they bid 90):
  - extra MWh billable
  - extra revenue (using Porovnanie formula: billable = 0.99*MW - 1.9)
  - extra fuel cost (linear model: 121 EUR/MWh_EE gross)
  - extra heat revenue already captured (no change, since heat is heat-driven)
  - NET incremental profit
"""

import pandas as pd
import numpy as np
import glob, os

# Constants (from Jan sheet model)
BID_PRICE     = 115       # EUR/MWh threshold
EE_OWN_SLOPE  = 0.99      # billable = 0.99 * gross - 1.9
EE_OWN_INT    = 1.9
FUEL_GROSS    = 121       # EUR/MWh_EE (linear model: 1.1 ATT * 110)
FUEL_MARGINAL = 85        # EUR/MWh_EE (engineering model: ~85 EUR for 5->8 MW upgrade)
HEAT_PRICE    = 27.95     # EUR/MWh heat (regulated tariff)

def billable(mw):
    return np.maximum(0, EE_OWN_SLOPE*mw - EE_OWN_INT)

def load_month(path, sheet_pred):
    """Return df with cols: ts, actual_MW, heat_MWh_avg, DA_price, dt_h."""
    df = pd.read_excel(path, sheet_name=sheet_pred, header=None, skiprows=2)
    # Columns: [date, time, y1..y4, avg, MW_EE, Maric, sum, PLAN, lbl, val, _, _?, actual_heat_kWh, actual_maric_kWh, actual_EE_kW, DA_price, revenue, ...]
    # The first 19 cols are consistent; some months have extra cols after
    df = df.iloc[:, :19].copy()
    df.columns = ['date','time','y1','y2','y3','y4','avg_heat','MW_EE','Maric','sum',
                  'PLAN','lbl','val','_','actual_heat_kWh','actual_maric_kWh',
                  'actual_EE_kW','DA_price','revenue']
    # Drop rows with no date
    df = df.dropna(subset=['date']).reset_index(drop=True)
    for c in ['actual_heat_kWh','actual_EE_kW','DA_price','PLAN']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['actual_EE_kW','DA_price']).reset_index(drop=True)
    # Filter outlier heat readings (sensor rollovers -- |kWh| > 100,000 is impossible on a 15-min row)
    df = df[df['actual_heat_kWh'].abs() < 100000].reset_index(drop=True)
    # Detect interval: check first two timestamps
    t0 = pd.to_datetime(str(df.iloc[0]['time']), errors='coerce')
    t1 = pd.to_datetime(str(df.iloc[1]['time']), errors='coerce') if len(df) > 1 else t0
    dt_min = (t1 - t0).total_seconds() / 60 if pd.notna(t0) and pd.notna(t1) else 60
    if dt_min <= 0:
        dt_min = 60
    dt_h = dt_min / 60
    df['dt_h'] = dt_h
    df['actual_MW']      = df['actual_EE_kW'] / 1000        # avg MW during interval
    # Heat is kWh *during the interval*, convert to MW average
    df['heat_MW_avg']    = (df['actual_heat_kWh'] / 1000 / dt_h).clip(0, 32)   # cap at plant thermal ceiling
    return df, dt_h

def simulate(df):
    """Apply bid-at-90 rule and compute incremental profit.
    When DA >= 90 and actual < 7.5 MW, counterfactual = produce 8 MW.
    Otherwise counterfactual = actual.
    Assume ramp is not binding (observed +3 MW/h achievable)."""
    cf_MW = np.where((df['DA_price'] >= BID_PRICE) & (df['actual_MW'] < 7.5), 8.0, df['actual_MW'])
    dt = df['dt_h'].iloc[0]

    # Extra billable EE (MWh) per interval
    d_billable_MWh = (billable(cf_MW) - billable(df['actual_MW'].values)) * dt
    # Extra EE revenue
    d_ee_rev = d_billable_MWh * df['DA_price'].values
    # Extra fuel: use both models to bound the answer
    d_prod_MWh = (cf_MW - df['actual_MW'].values) * dt
    d_fuel_lin = d_prod_MWh * FUEL_GROSS       # linear Jan-sheet model
    d_fuel_eng = d_prod_MWh * FUEL_MARGINAL    # engineering marginal model
    # No change to heat (they stick to heat plan for the base case)
    # Net incremental under each model
    d_net_lin = d_ee_rev - d_fuel_lin
    d_net_eng = d_ee_rev - d_fuel_eng

    # Totals
    hrs_bid_hit = ((df['DA_price'] >= BID_PRICE) & (df['actual_MW'] < 7.5)).sum() * dt
    hrs_total_above90 = (df['DA_price'] >= BID_PRICE).sum() * dt
    hrs_already_at_8 = ((df['DA_price'] >= BID_PRICE) & (df['actual_MW'] >= 7.5)).sum() * dt
    capture_frac = hrs_already_at_8 / max(hrs_already_at_8 + hrs_bid_hit, 1e-9)

    return {
        'intervals'       : len(df),
        'hrs'             : len(df) * dt,
        'hrs_DA_ge_90'    : hrs_total_above90,
        'hrs_already_8MW' : hrs_already_at_8,
        'hrs_to_add_8MW'  : hrs_bid_hit,
        'capture_frac'    : capture_frac,
        'actual_MW_mean'  : df['actual_MW'].mean(),
        'heat_MW_mean'    : df['heat_MW_avg'].mean(),
        'DA_mean'         : df['DA_price'].mean(),
        'd_prod_MWh'      : d_prod_MWh.sum(),
        'd_ee_rev_EUR'    : d_ee_rev.sum(),
        'd_fuel_lin_EUR'  : d_fuel_lin.sum(),
        'd_fuel_eng_EUR'  : d_fuel_eng.sum(),
        'd_net_lin_EUR'   : d_net_lin.sum(),
        'd_net_eng_EUR'   : d_net_eng.sum(),
    }

# All 12 months
months = [
    ('01', 'data/bardejov/kalkulácie TeHo BJ 2025 01.XLSX', 'predikcia'),
    ('02', 'data/bardejov/Kalkulácie TeHo BJ 2025 02.xlsx', 'Predikcia'),
    ('03', 'data/bardejov/Kalkulácie TeHo BJ 2025 03.xlsx', 'Predikcia 15min.'),
    ('04', 'data/bardejov/Kalkulácie TeHo BJ 2025 04.xlsx', 'Predikcia 15min'),
    ('05', 'data/bardejov/Kalkulácie TeHo BJ 2025 05.xlsx', 'Predikcia 15min.'),
    ('06', 'data/bardejov/Kalkulácie TeHo BJ 2025 06.xlsx', 'Predikcia 15min.'),
    ('07', 'data/bardejov/Kalkulácie TeHo BJ 2025 07.xlsx', 'Predikcia 15min.'),
    ('08', 'data/bardejov/Kalkulácie TeHo BJ 2025 08.xlsx', 'Predikcia 15min.'),
    ('09', 'data/bardejov/Kalkulácie TeHo BJ 2025 09.xlsx', 'Predikcia 15min.'),
    ('10', 'data/bardejov/Kalkulácie TeHo BJ 2025 10.xlsx', 'Predikcia 15min.'),
    ('11', 'data/bardejov/Kalkulácie TeHo BJ 2025 11.xlsx', 'Predikcia 15min.'),
    ('12', 'data/bardejov/Kalkulácie TeHo BJ 2025 12.xlsx', 'Predikcia 15min.'),
]

results = []
for mo, path, sheet in months:
    try:
        df, dt = load_month(path, sheet)
        r = simulate(df)
        r['month'] = mo
        r['dt_min'] = int(dt * 60)
        results.append(r)
        print(f'[+] {mo} ({r["dt_min"]:>2d}min grid, {r["hrs"]:.0f} h): '
              f'avg actual {r["actual_MW_mean"]:4.1f} MW, heat {r["heat_MW_mean"]:4.1f} MW, '
              f'DA mean {r["DA_mean"]:5.1f}, '
              f'net_lin {r["d_net_lin_EUR"]:+7,.0f} | net_eng {r["d_net_eng_EUR"]:+7,.0f} EUR')
    except Exception as e:
        print(f'[!] {mo}: {e}')

print('\n' + '='*118)
print(f'{"Month":>6s} {"grid":>5s} {"hrs":>5s} {"avg MW":>7s} {"heat MW":>8s} {"DA avg":>7s} '
      f'{"hrs>=90":>8s} {"hrs@8":>6s} {"to add":>7s} {"capture":>8s} '
      f'{"+MWh":>7s} {"+rev":>9s} {"+NET lin":>10s} {"+NET eng":>10s}')
for r in results:
    print(f'{r["month"]:>6s} {r["dt_min"]:>3d}min {r["hrs"]:>5.0f} {r["actual_MW_mean"]:>7.2f} '
          f'{r["heat_MW_mean"]:>8.2f} {r["DA_mean"]:>7.1f} {r["hrs_DA_ge_90"]:>8.0f} '
          f'{r["hrs_already_8MW"]:>6.0f} {r["hrs_to_add_8MW"]:>7.0f} {100*r["capture_frac"]:>7.0f}% '
          f'{r["d_prod_MWh"]:>7.0f} {r["d_ee_rev_EUR"]:>9,.0f} '
          f'{r["d_net_lin_EUR"]:>+10,.0f} {r["d_net_eng_EUR"]:>+10,.0f}')

# Totals
total_net_lin = sum(r['d_net_lin_EUR']  for r in results)
total_net_eng = sum(r['d_net_eng_EUR']  for r in results)
total_hrs     = sum(r['hrs']            for r in results)
total_mwh     = sum(r['d_prod_MWh']     for r in results)
total_rev     = sum(r['d_ee_rev_EUR']   for r in results)
total_hrs_ge  = sum(r['hrs_DA_ge_90']   for r in results)
total_hrs_add = sum(r['hrs_to_add_8MW'] for r in results)
print('-'*118)
print(f'12-month total ({total_hrs:.0f} h): +{total_mwh:.0f} MWh, +{total_rev:,.0f} EUR gross rev')
print(f'  hours >= threshold    : {total_hrs_ge:.0f}')
print(f'  hours to add 8 MW     : {total_hrs_add:.0f}')
print(f'  net LINEAR (121/MWh)  : {total_net_lin:+,.0f} EUR/yr')
print(f'  net ENGINEERING (85/MWh): {total_net_eng:+,.0f} EUR/yr')
