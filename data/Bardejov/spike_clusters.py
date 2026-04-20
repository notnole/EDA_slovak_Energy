"""Price-spike cluster analysis for TEHO Bardejov 2025.

For the bid-at-90 (or 115/130) strategy to actually capture value, the plant
needs 2 boilers hot. This script quantifies:

  1. Contiguous runs of DA price >= threshold (spike clusters)
  2. Length distribution of clusters
  3. Gap distribution between clusters
  4. Per-cluster value at 8 MW
  5. Which gaps are short enough to justify hot standby
  6. Which clusters are long enough to cover a cold start
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Same loader as before -----------------------------------------------
EE_OWN_SLOPE  = 0.99
EE_OWN_INT    = 1.9
FUEL_MARGINAL = 85              # engineering model (user-accepted)
STANDBY_EUR_H = 32_000 / (365*24)  # ~3.65 EUR/h for hot-standby 2nd boiler
COLDSTART_H   = 6                   # memory: 4-8 h cold start, use midpoint

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
    df['dt_h']      = dt_min / 60
    df['actual_MW'] = df['actual_EE_kW'] / 1000
    # Time column may be "HH:MM:SS" string OR a spurious datetime like "1900-01-02 06:30:00".
    # Extract time-of-day robustly.
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

# Collapse to HOURLY grid (clusters are a boiler/ramp concept, hourly is right)
all_df['hour'] = all_df['ts'].dt.floor('h')
hourly = all_df.groupby('hour').agg(
    DA_price   = ('DA_price', 'mean'),
    actual_MW  = ('actual_MW', 'mean'),
).reset_index().sort_values('hour').reset_index(drop=True)
print(f'[+] {len(hourly)} hourly rows')

# --- Cluster detection ---------------------------------------------------
def find_clusters(prices, thr):
    """Return list of (start_idx, end_idx, length_h) for contiguous runs >= thr."""
    above = (prices >= thr).values
    clusters = []
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
    return clusters

def cluster_value(hourly, start, end):
    """Incremental value (eng model) if we ran 8 MW through this cluster."""
    sl = hourly.iloc[start:end+1]
    cf = np.where(sl['actual_MW'] < 7.5, 8.0, sl['actual_MW'])
    d_bill = billable(cf) - billable(sl['actual_MW'].values)
    d_prod = cf - sl['actual_MW'].values
    d_rev  = d_bill * sl['DA_price'].values
    d_fuel = d_prod * FUEL_MARGINAL
    return (d_rev - d_fuel).sum()

for threshold in [90, 115, 130]:
    print('\n' + '='*80)
    print(f'THRESHOLD = {threshold} EUR/MWh')
    print('='*80)
    cls = find_clusters(hourly['DA_price'], threshold)
    lengths = np.array([c[2] for c in cls])
    values  = np.array([cluster_value(hourly, s, e) for s,e,_ in cls])

    print(f'  Clusters                 : {len(cls)}')
    print(f'  Total hours in clusters  : {lengths.sum()}')
    print(f'  Total value (eng model)  : {values.sum():+,.0f} EUR')
    print(f'  Avg value per cluster    : {values.mean():+,.0f} EUR')
    print(f'  Median cluster length    : {np.median(lengths):.0f} h')
    print(f'  p75 cluster length       : {np.percentile(lengths,75):.0f} h')
    print(f'  p90 cluster length       : {np.percentile(lengths,90):.0f} h')
    print(f'  Max cluster length       : {lengths.max()} h')

    # Cluster length bins
    print(f'\n  Cluster length distribution:')
    bins = [1,2,3,4,6,9,12,24,48,1001]
    labels = ['1h','2h','3h','4-5h','6-8h','9-11h','12-23h','24-47h','>=48h']
    cat = pd.cut(lengths, bins=bins, labels=labels, right=False)
    hist = cat.value_counts().sort_index()
    value_by_bin = pd.Series(values).groupby(cat, observed=False).sum().fillna(0)
    for lbl in labels:
        print(f'    {lbl:>7s}: {hist.get(lbl,0):>4d} clusters, {value_by_bin.get(lbl,0):>+10,.0f} EUR total')

    # Coldstart-feasible: cluster longer than startup time
    cold_ok = lengths >= COLDSTART_H
    print(f'\n  Clusters >= {COLDSTART_H}h (cold-start feasible): '
          f'{cold_ok.sum()}  value: {values[cold_ok].sum():+,.0f} EUR')
    print(f'  Clusters <  {COLDSTART_H}h (need hot standby)  : '
          f'{(~cold_ok).sum()}  value: {values[~cold_ok].sum():+,.0f} EUR')

    # Gap distribution (hours between clusters)
    gaps = []
    for i in range(1, len(cls)):
        gap_h = cls[i][0] - cls[i-1][1] - 1   # hours strictly between clusters
        gaps.append(gap_h)
    gaps = np.array(gaps)
    print(f'\n  Gap distribution between consecutive clusters:')
    print(f'    median gap : {np.median(gaps):.0f} h ({np.median(gaps)/24:.1f} days)')
    print(f'    p75 gap    : {np.percentile(gaps,75):.0f} h')
    print(f'    p90 gap    : {np.percentile(gaps,90):.0f} h')
    print(f'    max gap    : {gaps.max()} h')

    # For each gap, compute: should we keep boiler warm OR stand down?
    # Keep warm if standby_cost(gap) < next_cluster_value AND we didn't already have capacity
    # Simplification: sum standby cost if we kept it warm whole summer, vs sum of all cluster value
    # reported above.

# ----- Plots --------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: price with clusters (>=90)
ax = axes[0,0]
ax.plot(hourly['hour'], hourly['DA_price'], linewidth=0.3, color='gray', alpha=0.8)
ax.axhline(90,  color='orange', linestyle='--', alpha=0.6, label='bid 90')
ax.axhline(115, color='red',    linestyle='--', alpha=0.6, label='bid 115')
ax.axhline(130, color='darkred',linestyle='--', alpha=0.6, label='bid 130')
for s,e,L in find_clusters(hourly['DA_price'], 115):
    ax.axvspan(hourly['hour'].iloc[s], hourly['hour'].iloc[e], color='red', alpha=0.1)
ax.set_title('DA price 2025 (red shading = cluster with DA>=115)')
ax.set_ylabel('EUR/MWh')
ax.legend(loc='upper right')
ax.set_ylim(-50, 400)
ax.grid(alpha=0.3)

# Panel 2: cluster length histograms for the 3 thresholds
ax = axes[0,1]
for thr, color in zip([90, 115, 130], ['C0','C1','C2']):
    cls = find_clusters(hourly['DA_price'], thr)
    lengths = [c[2] for c in cls]
    ax.hist(lengths, bins=np.arange(1, 60, 1), alpha=0.5, label=f'bid {thr}', color=color)
ax.axvline(COLDSTART_H, color='red', linestyle=':', label=f'cold start ({COLDSTART_H}h)')
ax.set_xlabel('Cluster length (hours)')
ax.set_ylabel('Count')
ax.set_title('Cluster length distribution')
ax.legend()
ax.grid(alpha=0.3)

# Panel 3: gaps (hours between clusters)
ax = axes[1,0]
for thr, color in zip([90, 115, 130], ['C0','C1','C2']):
    cls = find_clusters(hourly['DA_price'], thr)
    gaps = []
    for i in range(1, len(cls)):
        gap_h = cls[i][0] - cls[i-1][1] - 1
        gaps.append(gap_h)
    ax.hist(gaps, bins=np.arange(0, 300, 10), alpha=0.5, label=f'bid {thr}', color=color)
standby_breakeven = 200 / STANDBY_EUR_H    # ~50 EUR average cluster value / 3.65 EUR/h = 13 hours, but let's use 200 EUR
ax.axvline(standby_breakeven, color='red', linestyle=':',
            label=f'standby break-even ({standby_breakeven:.0f}h @ 200 EUR/cluster)')
ax.set_xlabel('Gap between clusters (hours)')
ax.set_ylabel('Count')
ax.set_title('Gap distribution between clusters')
ax.legend()
ax.grid(alpha=0.3)

# Panel 4: monthly cluster value at bid=115
ax = axes[1,1]
hourly['m'] = hourly['hour'].dt.month
monthly_val = []
monthly_hrs = []
cls_115 = find_clusters(hourly['DA_price'], 115)
monthly_cluster_val = {m:0.0 for m in range(1,13)}
monthly_cluster_hrs = {m:0 for m in range(1,13)}
for s,e,L in cls_115:
    v = cluster_value(hourly, s, e)
    m = hourly['hour'].iloc[s].month
    monthly_cluster_val[m] += v
    monthly_cluster_hrs[m] += L
months_x = list(range(1,13))
vals = [monthly_cluster_val[m]/1000 for m in months_x]
hrs  = [monthly_cluster_hrs[m] for m in months_x]
bars = ax.bar(months_x, vals, color=['C0' if 4 < m < 10 else 'C3' for m in months_x])
for i, (v,h) in enumerate(zip(vals, hrs)):
    ax.text(i+1, v, f'{h}h', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Month')
ax.set_ylabel('Cluster value at bid 115 (k EUR)')
ax.set_title('Per-month value of bid-115 clusters (blue=Summer May-Sep)')
ax.grid(alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig('data/bardejov/spike_clusters.png', dpi=130, bbox_inches='tight')
print('\n[+] Saved data/bardejov/spike_clusters.png')
