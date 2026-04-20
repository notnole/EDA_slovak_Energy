"""Sweep bid prices 50-250 EUR/MWh, all 12 months of 2025, both cost models."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EE_OWN_SLOPE  = 0.99
EE_OWN_INT    = 1.9
FUEL_GROSS    = 121       # linear Jan-sheet fuel model
FUEL_MARGINAL = 85        # engineering Priprava marginal fuel model

def billable(mw):
    return np.maximum(0, EE_OWN_SLOPE*mw - EE_OWN_INT)

def load_month(path, sheet_pred):
    df = pd.read_excel(path, sheet_name=sheet_pred, header=None, skiprows=2)
    df = df.iloc[:, :19].copy()
    df.columns = ['date','time','y1','y2','y3','y4','avg_heat','MW_EE','Maric','sum',
                  'PLAN','lbl','val','_','actual_heat_kWh','actual_maric_kWh',
                  'actual_EE_kW','DA_price','revenue']
    df = df.dropna(subset=['date']).reset_index(drop=True)
    for c in ['actual_heat_kWh','actual_EE_kW','DA_price']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['actual_EE_kW','DA_price']).reset_index(drop=True)
    df = df[df['actual_heat_kWh'].abs() < 100000].reset_index(drop=True)
    t0 = pd.to_datetime(str(df.iloc[0]['time']), errors='coerce')
    t1 = pd.to_datetime(str(df.iloc[1]['time']), errors='coerce') if len(df) > 1 else t0
    dt_min = (t1 - t0).total_seconds() / 60 if pd.notna(t0) and pd.notna(t1) else 60
    if dt_min <= 0: dt_min = 60
    df['dt_h']        = dt_min / 60
    df['actual_MW']   = df['actual_EE_kW'] / 1000
    return df

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

# Load once, concatenate
print('[*] Loading all months...')
parts = []
for mo, path, sheet in months:
    try:
        df = load_month(path, sheet)
        df['month'] = mo
        parts.append(df)
        print(f'  {mo}: {len(df)} rows, dt={df["dt_h"].iloc[0]*60:.0f} min')
    except Exception as e:
        print(f'  {mo}: FAIL {e}')

all_df = pd.concat(parts, ignore_index=True)
print(f'[+] Total: {len(all_df)} rows ({all_df["dt_h"].sum():.0f} hours)')

# Sweep
bid_prices = list(range(50, 201, 5))
out_rows = []

for bid in bid_prices:
    cf_MW = np.where(
        (all_df['DA_price'] >= bid) & (all_df['actual_MW'] < 7.5),
        8.0,
        all_df['actual_MW']
    )
    dt = all_df['dt_h'].values
    d_bill = (billable(cf_MW) - billable(all_df['actual_MW'].values)) * dt
    d_prod = (cf_MW - all_df['actual_MW'].values) * dt
    d_rev  = d_bill * all_df['DA_price'].values
    d_fuel_lin = d_prod * FUEL_GROSS
    d_fuel_eng = d_prod * FUEL_MARGINAL
    net_lin = (d_rev - d_fuel_lin).sum()
    net_eng = (d_rev - d_fuel_eng).sum()
    hrs_hit = ((all_df['DA_price'] >= bid) & (all_df['actual_MW'] < 7.5)).astype(float).mul(dt).sum()
    out_rows.append({
        'bid': bid,
        'hrs_to_add_8MW': hrs_hit,
        'extra_MWh': d_prod.sum(),
        'extra_rev': d_rev.sum(),
        'net_linear': net_lin,
        'net_engineering': net_eng,
    })

res = pd.DataFrame(out_rows)

# Print table
print('\n' + '='*80)
print(f'{"bid":>5s} {"hrs_add":>8s} {"+MWh":>8s} {"+rev":>10s} {"net LIN":>11s} {"net ENG":>11s}')
for _, r in res.iterrows():
    star_lin = '  *' if r['net_linear'] == res['net_linear'].max() else '   '
    star_eng = '  *' if r['net_engineering'] == res['net_engineering'].max() else '   '
    print(f'{r["bid"]:>5.0f} {r["hrs_to_add_8MW"]:>8.0f} {r["extra_MWh"]:>8.0f} '
          f'{r["extra_rev"]:>10,.0f} {r["net_linear"]:>+11,.0f}{star_lin}'
          f'{r["net_engineering"]:>+11,.0f}{star_eng}')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(res['bid'], res['net_linear']/1000, 'o-', label='Linear (121 EUR/MWh fuel)', color='C0')
axes[0].plot(res['bid'], res['net_engineering']/1000, 's-', label='Engineering (85 EUR/MWh margin)', color='C1')
axes[0].axhline(0, color='k', linewidth=0.5)
axes[0].axvline(90, color='gray', linestyle=':', alpha=0.6, label='bid 90')
axes[0].axvline(115, color='gray', linestyle='--', alpha=0.6, label='bid 115')
# Mark optima
opt_lin = res.loc[res['net_linear'].idxmax()]
opt_eng = res.loc[res['net_engineering'].idxmax()]
axes[0].scatter(opt_lin['bid'], opt_lin['net_linear']/1000, marker='*', s=200, color='C0', zorder=5)
axes[0].scatter(opt_eng['bid'], opt_eng['net_engineering']/1000, marker='*', s=200, color='C1', zorder=5)
axes[0].annotate(f'opt lin: {opt_lin["bid"]:.0f} -> {opt_lin["net_linear"]/1000:.0f}k',
                  xy=(opt_lin['bid'], opt_lin['net_linear']/1000),
                  xytext=(5, -25), textcoords='offset points', color='C0')
axes[0].annotate(f'opt eng: {opt_eng["bid"]:.0f} -> {opt_eng["net_engineering"]/1000:.0f}k',
                  xy=(opt_eng['bid'], opt_eng['net_engineering']/1000),
                  xytext=(5, 10), textcoords='offset points', color='C1')
axes[0].set_xlabel('Bid price (EUR/MWh)')
axes[0].set_ylabel('Annual net gain (k EUR)')
axes[0].set_title('TEHO Bardejov -- bid-price sweep, annual 2025 net')
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

axes[1].plot(res['bid'], res['hrs_to_add_8MW'], 'o-', color='C2', label='hours needing ramp to 8 MW')
axes[1].plot(res['bid'], res['extra_MWh'], 's-', color='C3', label='extra MWh billable')
axes[1].set_xlabel('Bid price (EUR/MWh)')
axes[1].set_ylabel('Count / MWh')
axes[1].set_title('Opportunity sizing vs bid threshold')
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.savefig('data/bardejov/bid_sweep.png', dpi=130, bbox_inches='tight')
print('\n[+] Saved data/bardejov/bid_sweep.png')
