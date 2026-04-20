"""Find 10 clear "bad choice" examples across 5 months in 2025.

Criteria for a bad choice:
  - Contiguous cluster of DA >= 115 EUR/MWh, length >= 3h
  - Plant was online (actual_MW > 1 MW, not a trip/outage)
  - Actual average MW during cluster <= 5 MW (well below 8 cap)
  - Prior hour showed actual_MW >= 4 MW (ramp to 8 was feasible)

For each chosen example, plot a 36-hour window showing:
  - DA price (top axis)
  - Actual MW (what they did)
  - Proposed MW (what they could have done: 8 during cluster)
  - Missed profit EUR in title
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

EE_OWN_SLOPE  = 0.99
EE_OWN_INT    = 1.9
FUEL_MARGINAL = 85

def billable(mw):
    return np.maximum(0, EE_OWN_SLOPE*mw - EE_OWN_INT)

def load_month(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=2)
    df = df.iloc[:, :19].copy()
    df.columns = ['date','time','y1','y2','y3','y4','avg_heat','MW_EE','Maric','sum',
                  'PLAN','lbl','val','_','actual_heat_kWh','actual_maric_kWh',
                  'actual_EE_kW','DA_price','revenue']
    df = df.dropna(subset=['date']).reset_index(drop=True)
    for c in ['actual_heat_kWh','actual_EE_kW','DA_price']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['actual_EE_kW','DA_price']).reset_index(drop=True)
    t0 = pd.to_datetime(str(df.iloc[0]['time']), errors='coerce')
    t1 = pd.to_datetime(str(df.iloc[1]['time']), errors='coerce') if len(df) > 1 else t0
    dt_min = (t1 - t0).total_seconds() / 60 if pd.notna(t0) and pd.notna(t1) else 60
    if dt_min <= 0: dt_min = 60
    df['dt_h'] = dt_min / 60
    df['actual_MW'] = df['actual_EE_kW'] / 1000
    def to_td(v):
        t = pd.to_datetime(str(v), errors='coerce')
        if pd.isna(t): return pd.Timedelta(0)
        return pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    df['ts'] = pd.to_datetime(df['date'].astype(str)) + df['time'].apply(to_td)
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

print('[*] Loading all months...')
parts = []
for mo, path, sheet in months:
    try:
        df = load_month(path, sheet)
        df['month'] = mo
        parts.append(df)
    except Exception as e:
        print(f'  {mo}: {e}')
all_df = pd.concat(parts, ignore_index=True).sort_values('ts').reset_index(drop=True)
all_df['hour'] = all_df['ts'].dt.floor('h')
hourly = all_df.groupby('hour').agg(
    DA_price   = ('DA_price', 'mean'),
    actual_MW  = ('actual_MW', 'mean'),
).reset_index().sort_values('hour').reset_index(drop=True)
print(f'[+] {len(hourly)} hourly rows')

# --- Find bad-choice clusters ----
THR = 115
clusters = []
above = (hourly['DA_price'] >= THR).values
i = 0
while i < len(above):
    if above[i]:
        j = i
        while j < len(above) and above[j]:
            j += 1
        clusters.append((i, j-1, j-i))
        i = j
    else:
        i += 1
print(f'[*] {len(clusters)} clusters with DA>={THR}')

# Score each cluster
rows = []
for s,e,L in clusters:
    if L < 3: continue
    sl = hourly.iloc[s:e+1]
    if (sl['actual_MW'] < 1).any(): continue   # plant offline during cluster
    if sl['actual_MW'].mean() > 5: continue    # already running high
    prev_MW = hourly.iloc[max(s-1,0)]['actual_MW'] if s > 0 else 0
    if prev_MW < 4: continue                   # ramp constrained
    cf = np.where(sl['actual_MW'] < 7.5, 8.0, sl['actual_MW'])
    d_bill = billable(cf) - billable(sl['actual_MW'].values)
    d_prod = cf - sl['actual_MW'].values
    d_rev  = d_bill * sl['DA_price'].values
    d_fuel = d_prod * FUEL_MARGINAL
    miss = (d_rev - d_fuel).sum()
    if miss < 50: continue
    rows.append({
        'start': sl['hour'].iloc[0],
        'end'  : sl['hour'].iloc[-1],
        'length_h': L,
        'mean_DA': sl['DA_price'].mean(),
        'max_DA' : sl['DA_price'].max(),
        'mean_actual_MW': sl['actual_MW'].mean(),
        'prev_MW': prev_MW,
        'missed_EUR': miss,
        's_idx': s, 'e_idx': e,
    })
cands = pd.DataFrame(rows).sort_values('missed_EUR', ascending=False)
cands['month'] = cands['start'].dt.month
print(f'[*] {len(cands)} qualifying clusters after filters')

# Pick up to 2 top per month across 5 months, prefer months with high missed value
picked = []
seen_months = set()
# Order months by total missed value
month_totals = cands.groupby('month')['missed_EUR'].sum().sort_values(ascending=False)
print('\nMissed value by month (bid-115 filter):')
print(month_totals)

# Pick 2 top per month, expanding through the ranked months until we have 10
for m in month_totals.index:
    picks = cands[cands['month']==m].head(2)
    for _, row in picks.iterrows():
        picked.append(row)
        seen_months.add(m)
        if len(picked) >= 10: break
    if len(picked) >= 10: break
# trim or pad
picked = picked[:10]
print(f'\n[*] Months covered: {sorted(seen_months)}')

print(f'\n[+] Selected {len(picked)} examples:')
for i, r in enumerate(picked, 1):
    print(f'  {i:>2d}. {r["start"]} ({r["length_h"]}h): '
          f'actual={r["mean_actual_MW"]:.1f} MW, DA mean={r["mean_DA"]:.0f}, '
          f'max DA={r["max_DA"]:.0f}, miss={r["missed_EUR"]:,.0f} EUR')

# --- Plot ---
fig = plt.figure(figsize=(18, 16))
gs = GridSpec(5, 2, figure=fig, hspace=0.7, wspace=0.25)

for i, r in enumerate(picked):
    ax = fig.add_subplot(gs[i//2, i%2])
    # ±12h window around cluster
    s_idx = r['s_idx']; e_idx = r['e_idx']
    lo = max(0, s_idx - 12)
    hi = min(len(hourly), e_idx + 13)
    win = hourly.iloc[lo:hi].copy()
    # "could-have-done" = 8 during cluster, actual outside
    cf = win['actual_MW'].values.copy().astype(float)
    for k in range(lo, hi):
        if s_idx <= k <= e_idx and win['actual_MW'].iloc[k-lo] < 7.5:
            cf[k-lo] = 8.0

    # Plot actual + proposed on left axis, DA on right axis
    ax.fill_between(win['hour'], 0, win['actual_MW'], color='C0', alpha=0.3,
                     label='What they did (MW)')
    ax.plot(win['hour'], win['actual_MW'], color='C0', linewidth=1.3)
    ax.plot(win['hour'], cf, color='C2', linewidth=1.8, linestyle='--',
            label='Proposed (8 MW when DA>=115)')
    ax.set_ylabel('MW', color='C0')
    ax.set_ylim(0, 9)
    ax.axhspan(0, 9, xmin=0, xmax=0, alpha=0)  # dummy to set limits
    # Highlight cluster span
    ax.axvspan(win['hour'].iloc[s_idx-lo], win['hour'].iloc[e_idx-lo],
                color='red', alpha=0.08)

    ax2 = ax.twinx()
    ax2.plot(win['hour'], win['DA_price'], color='C3', linewidth=1.2,
             label='DA price')
    ax2.axhline(115, color='C3', linestyle=':', alpha=0.5)
    ax2.set_ylabel('DA EUR/MWh', color='C3')
    ax2.set_ylim(0, max(250, win['DA_price'].max() * 1.1))

    ax.set_title(f'{r["start"].strftime("%Y-%m-%d %H:%M")} - {r["end"].strftime("%H:%M")}  '
                 f'({r["length_h"]}h)  DA avg {r["mean_DA"]:.0f} EUR, missed {r["missed_EUR"]:,.0f} EUR',
                 fontsize=10)
    ax.tick_params(axis='x', labelsize=8, rotation=15)
    if i == 0:
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle(f'10 missed-profit clusters (bid=115 EUR/MWh threshold, engineering cost model)\n'
             f'Total missed in these 10 examples: '
             f'{sum(r["missed_EUR"] for r in picked):,.0f} EUR',
             fontsize=13, y=0.995)
fig.savefig('data/bardejov/bad_choices.png', dpi=130, bbox_inches='tight')
print('\n[+] Saved bad_choices.png')
