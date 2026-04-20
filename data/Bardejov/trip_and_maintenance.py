"""Turbine trip risk + maintenance deferral analysis.

PART 1 -- TRIP RISK
  Detect hours where the turbine dropped to ~0 MW unexpectedly (trips / shutdowns).
  Under a bid-at-X strategy with DA commitment, a trip means paying imbalance
  penalty on 8 MWh of undelivered energy. Quantify the expected annual cost.

PART 2 -- MAINTENANCE DEFERRAL
  Currently one boiler is down for maintenance somewhere in the year (typically
  summer). Running 2 boilers warm May-Sep means that maintenance MUST happen
  either (a) rotated one-at-a-time, or (b) in shoulder season.
  Estimate the value lost if capacity is capped at 4 MW EE (one boiler) during a
  4-6 week maintenance window, placed in different parts of the year.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Loader (same as bid_sweep) ------------------------------------------
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
        if pd.isna(t):
            return pd.Timedelta(0)
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

# Collapse to hourly
all_df['hour'] = all_df['ts'].dt.floor('h')
hourly = all_df.groupby('hour').agg(
    DA_price   = ('DA_price', 'mean'),
    actual_MW  = ('actual_MW', 'mean'),
).reset_index().sort_values('hour').reset_index(drop=True)
print(f'[+] {len(hourly)} hourly rows ({hourly["hour"].min()} to {hourly["hour"].max()})')

# ====================== PART 1 - TRIP RISK ======================
print('\n' + '='*80)
print('PART 1 -- TRIP / DOWNTIME RISK')
print('='*80)

# Trip = actual_MW drops to ~0 after a run of producing
hourly['prev_MW'] = hourly['actual_MW'].shift(1)
hourly['is_down'] = hourly['actual_MW'] < 0.3              # under 0.3 MW ~ offline
hourly['was_up']  = hourly['prev_MW']  > 2.0               # was producing >2 MW last hour
hourly['trip_start'] = hourly['is_down'] & hourly['was_up']  # transition into downtime

# Count downtime runs
down = hourly['is_down'].values
runs = []  # (start_idx, end_idx, len_h)
i = 0
while i < len(down):
    if down[i]:
        j = i
        while j < len(down) and down[j]:
            j += 1
        runs.append((i, j-1, j-i))
        i = j
    else:
        i += 1

print(f'  Downtime runs (actual_MW < 0.3): {len(runs)}')
total_down_h = sum(r[2] for r in runs)
print(f'  Total downtime hours           : {total_down_h}  ({100*total_down_h/len(hourly):.1f}% of available hours)')
print(f'  Longest downtime run           : {max(r[2] for r in runs)} h  ({max(r[2] for r in runs)/24:.1f} days)')

# Split: "scheduled" (long, likely maintenance) vs "trip" (short, unexpected)
scheduled = [r for r in runs if r[2] >= 48]    # >= 2 days = probably scheduled
trips     = [r for r in runs if r[2] <  48]
print(f'\n  >=48h runs (scheduled outages) : {len(scheduled)}  total {sum(r[2] for r in scheduled)} h')
for r in scheduled[:5]:
    t = hourly["hour"].iloc[r[0]]
    print(f'    {t.date()} to +{r[2]} h')

print(f'\n  <48h runs (trips / short outages): {len(trips)}')
trip_hrs = sum(r[2] for r in trips)
print(f'    Total trip hours              : {trip_hrs}')
print(f'    Median trip length            : {np.median([r[2] for r in trips]) if trips else 0:.1f} h')
print(f'    Max trip length               : {max([r[2] for r in trips]) if trips else 0} h')

# Imbalance exposure: if we'd bid 8 MW and tripped, penalty on undelivered
# Conservative: imbalance cost = 1.5 * DA_price (typical SK deficit multiplier)
IMBAL_MULT = 1.5

# For each trip hour, check if DA_price was above threshold (=> we'd have committed)
for bid in [90, 115, 130]:
    committed = hourly['DA_price'] >= bid
    trip_mask = np.zeros(len(hourly), dtype=bool)
    for s,e,L in trips:
        trip_mask[s:e+1] = True
    risk_hrs = (committed & trip_mask).sum()
    # each such hour = 8 MWh short * IMBAL_MULT * DA_price
    risk_cost = (8 * IMBAL_MULT * hourly.loc[committed & trip_mask, 'DA_price']).sum()
    print(f'  bid {bid}: committed-and-tripped hours = {risk_hrs}  expected imbalance cost = {risk_cost:,.0f} EUR/yr')

# ====================== PART 2 - MAINTENANCE DEFERRAL ======================
print('\n' + '='*80)
print('PART 2 -- MAINTENANCE DEFERRAL COST')
print('='*80)
# Under the proposed strategy, one boiler must still get maintenance.
# Assumption: during a 4- or 6-week window, the plant can only run ONE boiler,
# so max EE = 4 MW. Compute the value captured during each candidate window
# if capped at 4 MW vs 8 MW.

def sim_bid(hourly, bid, mw_cap=8.0):
    cf = np.where(
        (hourly['DA_price'] >= bid) & (hourly['actual_MW'] < min(mw_cap, 7.5)),
        mw_cap,
        hourly['actual_MW']
    )
    d_bill = billable(cf) - billable(hourly['actual_MW'].values)
    d_prod = cf - hourly['actual_MW'].values
    d_rev  = d_bill * hourly['DA_price'].values
    d_fuel = d_prod * FUEL_MARGINAL
    return d_rev - d_fuel                 # per-hour incremental

for bid in [90, 115, 130]:
    full_cap = sim_bid(hourly, bid, 8.0)
    half_cap = sim_bid(hourly, bid, 4.0)
    print(f'\n  bid {bid}  -- full-year at 8 MW cap: {full_cap.sum():+,.0f} EUR  '
          f'| at 4 MW cap: {half_cap.sum():+,.0f} EUR')

    # 4-week window: slide across year, compute value LOST when capping at 4 MW
    # for that window only.
    hw_h = 4 * 7 * 24                        # 4 weeks
    # For each starting week, compute full-year - (4-week period at 4 MW) - (rest at 8 MW)
    # Equivalent: loss = sum(full_cap[window] - half_cap[window]) during the window
    loss_at_start = np.zeros(len(hourly))
    for i in range(len(hourly) - hw_h):
        loss_at_start[i] = full_cap[i:i+hw_h].sum() - half_cap[i:i+hw_h].sum()
    # Find best/worst placements
    best_start_idx = int(np.argmin(loss_at_start[:len(hourly)-hw_h]))
    worst_start_idx = int(np.argmax(loss_at_start[:len(hourly)-hw_h]))
    best_start = hourly['hour'].iloc[best_start_idx]
    worst_start = hourly['hour'].iloc[worst_start_idx]
    print(f'    4-week maintenance window -- best placement (least loss):  '
          f'start {best_start.date()}  loss = {loss_at_start[best_start_idx]:,.0f} EUR')
    print(f'    4-week maintenance window -- worst placement (most loss): '
          f'start {worst_start.date()}  loss = {loss_at_start[worst_start_idx]:,.0f} EUR')

    # Monthly placement
    print(f'    Loss by month of window-start:')
    month_losses = {m:[] for m in range(1,13)}
    for i in range(len(hourly) - hw_h):
        m = hourly['hour'].iloc[i].month
        month_losses[m].append(loss_at_start[i])
    for m in range(1,13):
        if month_losses[m]:
            print(f'      M{m:02d}: median loss = {np.median(month_losses[m]):>+8,.0f} EUR')

# ----- Plot ---------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

ax = axes[0]
ax.plot(hourly['hour'], hourly['actual_MW'], linewidth=0.3, color='C0', label='actual MW')
for s,e,L in scheduled:
    ax.axvspan(hourly['hour'].iloc[s], hourly['hour'].iloc[e], color='red', alpha=0.3,
               label='scheduled outage' if s==scheduled[0][0] else None)
for s,e,L in trips:
    ax.axvspan(hourly['hour'].iloc[s], hourly['hour'].iloc[e], color='orange', alpha=0.5)
ax.set_title('2025 actual EE output with downtime (red=scheduled outage, orange=trip/short)')
ax.set_ylabel('Actual EE (MW)')
ax.legend()
ax.grid(alpha=0.3)

# Loss-by-maintenance-placement curve for bid=115
ax = axes[1]
full_cap = sim_bid(hourly, 115, 8.0)
half_cap = sim_bid(hourly, 115, 4.0)
hw_h = 4*7*24
loss_series = np.zeros(len(hourly))
for i in range(len(hourly) - hw_h):
    loss_series[i] = full_cap[i:i+hw_h].sum() - half_cap[i:i+hw_h].sum()
loss_series[len(hourly)-hw_h:] = np.nan
ax.plot(hourly['hour'], loss_series, color='C3')
ax.set_title('Bid-115: value LOST if 4-week maintenance starts at this date (1-boiler cap)')
ax.set_ylabel('Value lost over 4 weeks (EUR)')
ax.set_xlabel('Start date of 4-week maintenance window')
ax.axhline(0, color='k', linewidth=0.5)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig('data/bardejov/trip_maintenance.png', dpi=130, bbox_inches='tight')
print('\n[+] Saved data/bardejov/trip_maintenance.png')
