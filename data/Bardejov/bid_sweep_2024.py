"""Replay bid-price sweep on 2024 data (new turbine installed May 2024, so we
 limit to Jun-Dec 2024)."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EE_OWN_SLOPE  = 0.99
EE_OWN_INT    = 1.9
FUEL_MARGINAL = 85

def billable(mw):
    return np.maximum(0, EE_OWN_SLOPE*mw - EE_OWN_INT)

# Load plant data and DA prices
plant = pd.read_csv('data/bardejov/plant_timeseries.csv', parse_dates=['datetime'])
prices = pd.read_csv('data/bardejov/da_prices_hourly.csv', parse_dates=['datetime'])

# Aggregate to hourly (plant may be mixed 15-min / hourly)
plant['hour'] = plant['datetime'].dt.floor('h')
plant_h = plant.groupby('hour').agg(
    heat_kW  = ('heat_load_kW', 'mean'),
    cool_kW  = ('cooling_kW', 'mean'),
    ee_kW    = ('electricity_kW', 'mean'),
).reset_index()
plant_h['actual_MW'] = plant_h['ee_kW'] / 1000
plant_h['heat_MW']   = plant_h['heat_kW'] / 1000

# Merge with prices
prices = prices.rename(columns={'datetime':'hour'})
df = plant_h.merge(prices, on='hour', how='inner')

# 2024 only, new turbine installed May 2024 -> use Jun-Dec 2024
df_2024 = df[(df['hour'] >= '2024-06-01') & (df['hour'] < '2025-01-01')].copy()
df_2025 = df[(df['hour'] >= '2025-01-01') & (df['hour'] < '2026-01-01')].copy()

print(f'[*] 2024 (Jun-Dec): {len(df_2024)} hours')
print(f'    DA price mean: {df_2024["da_price_eur"].mean():.1f} EUR/MWh')
print(f'    Actual MW mean: {df_2024["actual_MW"].mean():.2f}')
print(f'    Heat MW mean  : {df_2024["heat_MW"].mean():.2f}')
print(f'[*] 2025 full: {len(df_2025)} hours (for comparison)')

# Price distribution vs 2025
print('\n=== DA price distribution comparison ===')
for p in [50, 75, 90, 100, 115, 130, 150, 200]:
    h24 = (df_2024['da_price_eur'] >= p).sum()
    h25 = (df_2025['da_price_eur'] >= p).sum()
    pct24 = 100*h24/len(df_2024)
    pct25 = 100*h25/len(df_2025)
    print(f'  hrs DA>={p:>4d}:  2024(Jun-Dec) = {h24:>5d} ({pct24:4.1f}%)  '
          f'|  2025 = {h25:>5d} ({pct25:4.1f}%)')

# Sweep for both years
bids = list(range(50, 201, 5))
def sweep(df, label):
    out = []
    for bid in bids:
        cf = np.where(
            (df['da_price_eur'] >= bid) & (df['actual_MW'] < 7.5),
            8.0, df['actual_MW']
        )
        d_bill = billable(cf) - billable(df['actual_MW'].values)
        d_prod = cf - df['actual_MW'].values
        d_rev  = d_bill * df['da_price_eur'].values
        d_fuel = d_prod * FUEL_MARGINAL
        out.append({
            'bid': bid,
            'hrs_add': ((df['da_price_eur'] >= bid) & (df['actual_MW'] < 7.5)).sum(),
            'extra_MWh': d_prod.sum(),
            'extra_rev': d_rev.sum(),
            'net_eng': (d_rev - d_fuel).sum(),
        })
    r = pd.DataFrame(out)
    r['label'] = label
    return r

r24 = sweep(df_2024, '2024 (Jun-Dec)')
r25 = sweep(df_2025, '2025 full')

# Annualize 2024 (7 months) to full year equivalent
r24['net_eng_annualized'] = r24['net_eng'] * 12/7

print('\n' + '='*90)
print(f'{"bid":>4s} {"2024 net":>12s} {"2024 annualized":>18s} {"2025 net":>12s}')
for i in range(len(r24)):
    print(f'{r24["bid"].iloc[i]:>4.0f} {r24["net_eng"].iloc[i]:>+12,.0f} '
          f'{r24["net_eng_annualized"].iloc[i]:>+18,.0f} {r25["net_eng"].iloc[i]:>+12,.0f}')

# Find optima
print('\nOptimal bid:')
i24 = r24['net_eng'].idxmax()
i25 = r25['net_eng'].idxmax()
print(f'  2024(Jun-Dec) annualized: bid={r24.iloc[i24]["bid"]:.0f} -> {r24.iloc[i24]["net_eng_annualized"]:+,.0f} EUR/yr')
print(f'  2025 full              : bid={r25.iloc[i25]["bid"]:.0f} -> {r25.iloc[i25]["net_eng"]:+,.0f} EUR/yr')

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(r24['bid'], r24['net_eng_annualized']/1000, 's-', color='C4',
             label='2024 (Jun-Dec, annualized)')
axes[0].plot(r25['bid'], r25['net_eng']/1000, 'o-', color='C1', label='2025 full')
axes[0].axhline(0, color='k', linewidth=0.5)
axes[0].axvline(115, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Bid price (EUR/MWh)')
axes[0].set_ylabel('Annual net gain (engineering model, k EUR)')
axes[0].set_title('Bid-price sweep: 2024 vs 2025')
axes[0].legend()
axes[0].grid(alpha=0.3)

# DA price distribution comparison
axes[1].hist(df_2024['da_price_eur'].clip(-50,400), bins=60, alpha=0.5,
             label=f'2024 Jun-Dec (avg {df_2024["da_price_eur"].mean():.0f})', color='C4')
axes[1].hist(df_2025['da_price_eur'].clip(-50,400), bins=60, alpha=0.5,
             label=f'2025 full (avg {df_2025["da_price_eur"].mean():.0f})', color='C1')
axes[1].axvline(90, color='orange', linestyle=':', label='90')
axes[1].axvline(115, color='red', linestyle='--', label='115')
axes[1].set_xlabel('DA price (EUR/MWh)')
axes[1].set_ylabel('Hours')
axes[1].set_title('DA price distribution')
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig('data/bardejov/bid_sweep_2024_vs_2025.png', dpi=130, bbox_inches='tight')
print('\n[+] Saved bid_sweep_2024_vs_2025.png')
